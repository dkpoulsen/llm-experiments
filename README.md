# llm-experiments

Planning, running and recording LLM training experiments on a single **5× NVIDIA CMP 170HX** box.

The rig is unusual: A100-class matmul rates and 320 GB of VRAM, but a PCIe Gen2 x4 fabric with no
NVLink and no P2P. What works here is not what works on a normal multi-GPU node, so every experiment
is chosen against the hardware's real limits rather than against a paper's assumptions.

## The hardware

| | |
|---|---|
| GPUs | 5 × CMP 170HX (GA100 die, 64 GB HBM2e each after unlock) |
| VRAM | 320 GB aggregate, 64 GB per card |
| Compute | ~162 TFLOPS bf16 GEMM peak per card; **90–100 TFLOPS sustained** |
| Fabric | PCIe **Gen2 x4**, ~1.6 GB/s link, no NVLink, no P2P |
| Power | 250 W stock cap; `PWRBRK#` on edge pin B30 silently caps a card at ~88 W |

## The question

Four interlocking claims in the planning docs, each of which decides what the box can attempt:

1. Is DDP the only viable topology at 1M-token global batches?
2. Are expert and tensor parallelism ruled out by the fabric?
3. Do frozen Engram memory tables make a 7B-total / ~1B-trainable model fit DDP on 64 GB cards?
4. Does CSA2 compressed sparse attention pay off at 32k training context?

## The finding

**The recommendation survives. The reasoning offered for it does not.**

Data-parallel training, 1M-token global batches, frozen Engram tables, no EP/TP, and a
sparse-attention substitution arm are the right program — each defended by a number that is
unmeasured, mislabelled, or computed against the wrong denominator. The two largest errors point in
opposite directions and nearly cancel, which is why the conclusion looked sound.

What binds the box is **not the fabric**. It is **wall-clock**: 90–100 TFLOPS/card sustained, not the
162 TFLOPS GEMM peak the plan divides by.

| Claim | Plan says | Evidence says | Tier |
|---|---|---|---|
| DDP overhead at 1M tokens | 36% | **5.3–7.1%** at Gen2 x4 | T4 |
| Throughput at 32k | 80–120k tok/s | **14–39k** → 9B tokens = **3–7 days**, not 25 h | T4 |
| 32k vs 4k wall-clock | 4.39× | **2.19×** measured | T3 |
| Seed-variance floor | 3.5% | **≤0.004 nats** (≤0.14% relative) | T3 |
| Frozen Engram table | 6B params / 12 GB | **12.29B / 24.58 GB** (two layers) | T3/T4 |
| EP / TP cost | ~70× | **1.4× / 2.8×** — ruled out on *availability*, not bandwidth | T3/T4 |
| NSA payoff at 32k | 6.3× | **≈1.71×** end-to-end; sign depends on GQA group size | T3 |
| Sustained compute | 162 TFLOPS/card | **90–100** sustained, 162 is GEMM peak | T1/T2 |

Evidence carries a tier throughout: **T1** measured on this box (user-reported only — the corpus
contains no on-box training measurement), **T2** community report or adjacent-hardware receipt,
**T3** paper-measured, **T4** arithmetic from the tiers above.

## The plan

`experiments/02-128k-engram.md` is the current capability target: a 128k-context model in **4.5 days**.

| Leg | Context | Tokens | Memory | Wall-clock |
|---|---|---|---|---|
| Pretrain | 8k | 9B | 40.7 GB | 2.40 days |
| Adapt (FSA + YaRN) | 128k | 1B | 52.8 GB | 2.13 days |
| **Total** | | | | **4.53 days** |

Against 5.9 days for from-scratch 32k — 22% faster and 4× the context, because the quadratic attention
penalty applies only to the short adaptation leg.

**Architecture:** dense 24L/d=2048 backbone (not MoE — at 128k attention is 93% of FLOPs, so MoE's 7%
saving does not pay for its topology cost), GQA 16 Q / 4 KV heads, ~0.92B trainable, 12.29B frozen
Engram params, FSA sparse attention, DDP across 5 ranks.

Three decision gates decide whether it works, and each is cheap: the GQA group size (a config read),
an FSA smoke test at 128k (hours), and a 1B-token Engram A/B in nats with two seeds.

## Repository layout

```
goal-and-requirements.md   What the project is for, and what it must produce
research.md                The original deep-research pass over the hardware and architectures
data-pipeline.md           Token supply, mix, packing, determinism
experiments/
  01-dense-baseline.md     The 500M dense control run (32k)
  02-128k-engram.md        The 128k Engram plan (8k pretrain + 128k adaptation)
research/
  notes/                   226 source notes + the final feasibility report
  runs/                    Pipeline artifacts: critic findings, patch logs, evidence digest
.claude/                   Agent and skill definitions used to produce the research
```

The final report is `research/notes/final_report_cmp170hx-training-feasibility-a0d4fc.md`.

## How the research was done

The corpus is the output of a 16-step deep-research pipeline, not a literature skim:

- **225 sources** fetched and summarised, **1,371 claims** extracted with verbatim support
- **13 contradiction clusters** and **11 consensus claims** mapped across the corpus
- **6 depth investigations** into the questions where the evidence actually forks
- **4 adversarial critics** (dialectic, depth, width, instruction) producing 31 findings
- **111 citation-sentence pairs** verified, 0 dangling
- **11 quote-integrity failures** repaired — every quote now resolves verbatim to a vault note

Raw PDFs and the local search index are gitignored; both are regenerable from the `source:` URL
recorded in each note.

## Measure before you train

Four checks, each cheap, each able to invalidate a run:

1. **`lspci -vv` per card** — "Gen2 x4" is two independent axes, not one fact. The capacitor mod
   changes width only, the unlocker changes generation only, and a card with neither runs Gen1 x4.
   The four reachable states span ~8× in one-way bandwidth.
2. **5-rank `all_reduce_perf`, both columns labelled** — no five-rank receipt exists anywhere in the
   literature. algbw and busbw differ by 1.6× at N=5, and that column choice moves the DDP overhead
   between 5.3% and 7.1%.
3. **Sustained BF16 burn at 250 W** — the decision-grade benchmark in the corpus is 74.6–78.5
   TFLOP/s *at 180 W enforced*. Every wall-clock estimate carries a 1.3× error bar from this axis.
4. **ECC query + `memtest_vulkan` on all 5 cards** (~50 min) — the field report is ~50% failure above
   8 GiB, and a bad card invalidates every comparability claim.

Plus `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"` before anything else: an asserted `PWRBRK#`
takes a card from 155.7 to 39.3 TFLOPS and looks exactly like a slow card.

## Status

Planning and research complete; **no training run has been executed**. The 128k plan is an arithmetic
projection from measured adjacent hardware, not a measurement on this box. The three decision gates
above are the cheapest way to convert it into one.
