# Experiment 02 — 128k Engram: 8k pretrain + 128k adaptation

**Status:** planned, not run
**Axis:** capability target (this is the run that produces a 128k-context model)
**Tokens:** 9B pretrain @ 8k + ~1B adaptation @ 128k
**Context:** 8k → 128k
**Evidence base:** `research/notes/final_report_cmp170hx-training-feasibility-a0d4fc.md`

## Hypothesis

A ~0.92B-trainable dense backbone with frozen Engram tables, pretrained at 8k and adapted to 128k,
trains end-to-end on this box in **under 5 days** and yields a 128k-context model — faster than the
repo's from-scratch 32k plan (5.9 days) and at four times the context.

The mechanism is that the quadratic attention penalty applies only to the short adaptation leg rather
than to the whole run. At 128k attention is 93% of step FLOPs, so pretraining at 8k (where it is 47%)
removes most of the quadratic bill.

If it does not, the binding constraint is one of three unmeasured variables in the decision gates
below, not the hardware.

## Why this architecture — four decisions

Each of these is a correction to the plan in `research.md`, and each is load-bearing.

### 1. Dense, not MoE

At 128k, attention is **93% of step FLOPs**. MoE reduces only the linear term (`6N`), which is 7% of
the step at this context — so MoE buys almost nothing here while costing the topology freedom it
needs. A 7B-total/1B-active MoE does not fit DDP (112 GB/card) and forces either ZeRO-2 (worst case
for CPU-side per-parameter hooks on many small tensors) or PP=5 (static shapes conflict with routing).
MoE is a short-context lever, not a long-context one.

### 2. GQA with group size ≥ 4

This is the decisive variable for whether sparse attention pays. FSA reports that when one GQA group
contains one query head at 32K, NSA consistently falls short of full attention; at group size 4 or
more it wins. **16 query heads / 4 KV heads** gives group size 4. Read this off the config before
anything else — it is one line and it decides the sign of the whole sparse arm.

### 3. NSA/FSA, not CSA2

CSA2 has no sm_80 training kernel: every training-capable DSA kernel in existence is Hopper-or-newer,
and the only sm_80 path is an inference-only Triton backend with no backward. Even if it ran, its
token-granular gather would be a **+10–15% step-time penalty** at 32k.

FSA is the only sm_80 training receipt in the corpus: tested on A100 SXM, backprops through
65,536 + 32,768 tokens, requires `flash-attn==2.6.3`. fla-org's Triton NSA ships training support via
`flame` as the secondary option.

### 4. Frozen Engram tables, replicated not sharded

Replication eliminates the forward gather collective; freezing eliminates the gradient collective.
Both are needed — a frozen-but-sharded table would reintroduce the forward all-to-all at one-fifth the
volume, and this fabric cannot run it.

The table is **12.29B params / 24.58 GB for two grafted layers**, not the 6B/12 GB the plan budgets.
That figure holds only on the paper's 3M entries per layer; the repo's own §3 figure of 33M entries
per layer implies 135 GB/layer and 270 GB for two, which fits neither 64 GB nor the box's 320 GB
aggregate. **Reconcile §3 against §5 before trusting the budget.**

## Step 0 — Pre-flight (do this before spending GPU time)

Six checks. The first five are cheap; the sixth is the one that can invalidate the whole design.

1. **Negotiated link state, per card.**
   ```
   for d in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
     lspci -vv -s "${d#00000000:}" | grep -E 'LnkCap|LnkSta'
   done
   ```
   "Gen2 x4" is two independent axes, not one fact: the capacitor mod changes width only, the
   unlocker changes generation only, and a card with neither runs Gen1 x4. The four reachable states
   span roughly 8× in one-way bandwidth. Record which state each card is in.

2. **Five-rank all-reduce, both columns labelled.**
   ```
   all_reduce_perf -b 1M -e 1G -f 2 -g 5
   ```
   Record **algbw and busbw separately**. At N=5, busbw = 1.6 × algbw by construction, and the
   plan's 0.75 GiB/s is plausible as busbw and implausible as algbw. No five-rank receipt exists in
   any accessible source, so this measurement is the only decisive artifact for that question. It
   moves the DDP overhead between 5.3% and 7.1%.

   Also check the odd-rank alignment penalty: NCCL's maintainer attributes the 5-GPU and 7-GPU
   slowdown to offset alignment (4/8 bytes vs 16). Padding the bucket to a multiple of 16 × 5 bytes
   removes it.

3. **Sustained BF16 burn at 250 W.**
   ```
   nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
   ```
   `Active` on a card that is not thermally or power limited means the board is asserting `PWRBRK#`
   (edge pin B30): ~88 W, ~1140 MHz, 155.7 → 39.3 TFLOPS. Fix with Kapton tape over B30 or a riser
   that does not route it.

   Then run a ten-minute `gpu-burn -tc` at the stock 250 W cap and record sustained TFLOPS. The
   decision-grade benchmark in the corpus is 74.6–78.5 TFLOP/s **at 180 W enforced**, so 250 W should
   be higher — but nobody has measured it. Every wall-clock number below carries a 1.3× error bar
   from this axis alone.

4. **ECC and VRAM integrity on all 5 cards (~50 min).**
   ```
   nvidia-smi -q | grep -A3 "ECC Mode"
   memtest_vulkan --device <n>
   ```
   The corpus holds an unresolved contradiction on ECC status (one source reports Working, three
   report not restored, and the unlocker's v0.5 release notes report it Enabled at ≤0.3% cost). The
   field report is ~50% failure above 8 GiB. A bad card invalidates every comparability claim.

5. **GQA group size at 128k.** One line in the model config. See decision gate 1.

6. **FSA smoke test at 128k.** 200M parameters, 100 steps, single GPU, `flash-attn==2.6.3`. FSA's
   published receipt is 65,536 + 32,768 tokens — 128k is 2× beyond it. Confirm the kernel holds
   before committing the adaptation leg.

## Decision gates

Three unknowns decide whether this plan works. Each is cheap to resolve and each can flip the design.

| Gate | Question | If yes | If no |
|---|---|---|---|
| **1. GQA group size** | Is group size ≥ 4 at 128k? | Sparse attention pays; adaptation leg is a −20% to −30% win | Run full attention; adaptation leg costs ~3 days instead of 2.1 |
| **2. FSA at 128k** | Does the kernel hold quality at 128k? | Proceed with FSA | Fall back to fla-org Triton NSA, or cap at 64k |
| **3. Engram at 128k** | Does the frozen table still help at long context? | Proceed with the Engram design | Fall back to dense 3B (see Fallback) |

Gate 3 is the biggest unmeasured assumption in the design. Engram is n-gram memory — parametric
knowledge, not context retrieval — and the paper's receipts are at 50B/100B tokens and short context.
At 128k the model has far more context to draw on, so the table's marginal value could shrink.

**Resolve gate 3 cheaply before the full run:** a 1B-token A/B at 8k with and without the frozen
table, reporting the delta **in nats with two seeds**. The measured noise floor is ≤0.004 nats, so a
0.02-nat delta is resolvable and a null is an informative bound. Do not spend 100B tokens per arm on
this — the floor is ~25× tighter than the 3.5% restart spread the pessimistic reading assumed.

## Config

| | |
|---|---|
| Architecture | Dense decoder, 24 layers, d=2048, GQA 16 Q / 4 KV heads |
| Trainable params | ~0.92B |
| Frozen params | 12.29B (two grafted Engram layers, 3M entries/layer × 2048 dims × bf16) |
| Pretrain context | 8k, dense attention (FlashAttention-2) |
| Adaptation context | 128k, FSA sparse attention + YaRN RoPE scaling |
| Pretrain tokens | 9B |
| Adaptation tokens | ~1B |
| Data | `data-pipeline.md` stream: 75% FineWeb-Edu / 15% Stack-Edu / 10% FineMath4+, Qwen3 tokenizer |
| **Boundary masking** | **ON** — see below |
| Precision | bf16 autocast, fp32 master weights |
| Optimizer | AdamW, cosine schedule, warmup ~1% of steps |
| Loss | fused cross-entropy (0.62 GB, not 9.94 GB) |
| Checkpointing | full activation checkpointing |
| Grad compression | `bf16_compress_hook` on |

### Data pipeline — the one thing the plan gets wrong

The repo's default is random interleaving across the three sources with **document-boundary masking
off**. That is MIXChunk, the worst of four measured configurations: intra-document masking beats it by
0.762 PPL average at 2K and 0.986 at 8K, with the largest gain on code (−1.28 PPL at 2K, −2.49 at 8K
on GitHub). Code is 15% of this mix.

The repo already stores `cu_seqlens` per shard, so the fix is a kernel flag costing a measured **4.0%
of step time** — not a re-tokenisation. **Turn masking on before the first run.**

Also: the determinism manifest is not optional. Floating-point arithmetic order — induced by sharding
topology, gradient accumulation depth or precision — produces functional divergence within ~17% of
changing the data seed.

## Memory budget

| Component | Pretrain @ 8k | Adaptation @ 128k |
|---|---:|---:|
| Frozen table (12.29B params, bf16) | 24.58 GB | 24.58 GB |
| Trainable recipient (0.92B × 16 B) | 14.72 GB | 14.72 GB |
| Checkpointed activations | 0.81 GB | 12.88 GB |
| LM-head logits (fused CE) | 0.62 GB | 0.62 GB |
| **Total** | **40.73 GB** | **52.80 GB** |
| Headroom on 64 GB | 23.27 GB | **11.20 GB** |

Activations are `L × d × S × 2 bytes`. At 128k this is the term that grows, and it is why the
adaptation leg is limited to **one sequence per GPU**: two sequences would need 25.77 GB of
activations and push the total to 65.69 GB, over the card.

## Parallelism and batch plan

- **Topology:** DDP across 5 ranks. No FSDP, no EP, no TP, no SP.
- **Why no sequence parallelism:** at 128k the standard move is to shard the sequence across GPUs.
  Ulysses needs all-to-all; Ring Attention needs P2P ring passes. Neither exists on this fabric. So
  one sequence per GPU, and the activation bill is per-card and never amortizes. This is the real
  wall at 128k, not the byte count.
- **`no_sync()`:** wrap all but the last micro-step in `model.no_sync()`. Without it DDP all-reduces
  every micro-step and multiplies communication by the accumulation depth.

| Leg | Global batch | Per-GPU micro-batch | Accumulation | DDP overhead |
|---|---:|---:|---:|---:|
| Pretrain @ 8k | 983,040 tokens (120 seq) | 1 × 8,192 | 24 micro-steps | **15.4%** |
| Adaptation @ 128k | 655,360 tokens (5 seq) | 1 × 131,072 | 1 micro-step | **2.9%** |

Two things worth noting:

- **The 8k leg pays 15.4% DDP overhead, the 128k leg 2.9%** — because the compute window is 5× longer
  at 128k (22.6 s vs 120.6 s) while the all-reduce payload is fixed at 1.84 GB. Long context makes
  DDP cheaper, which inverts the usual intuition. If 15.4% proves painful, doubling the 8k global
  batch to ~2M tokens halves it to ~7.5%; 2M is still ~8× below the measured plateau critical batch
  size of ~16.8M.
- **1M tokens is not reachable at 128k.** The global batch granularity is 655k (1 seq/GPU) or 1.31M
  (2 seq/GPU), and 2 seq/GPU does not fit. The adaptation leg runs at 655k.

- **Batch-size warmup is required.** The repo plans a fixed 1M batch from step 0. CBS is ~16.8M at
  plateau but near zero at initialisation, so a fixed large batch from step 0 loses to the
  small-batch control despite using 75% fewer steps. Start small, double as CBS doubles, scale LR by
  square root.

## Wall-clock budget

At 450 TFLOPS sustained aggregate (5 × 90):

| Leg | FLOPs/token | tok/s | Tokens | Wall-clock |
|---|---:|---:|---:|---:|
| Pretrain @ 8k | 1.035e10 | 43,471 | 9B | **2.40 days** |
| Adaptation @ 128k | 8.283e10 | 5,433 | 1B | **2.13 days** |
| **Total** | | | | **4.53 days** |

Against **5.9 days** for from-scratch 32k. The saving is 22% of wall-clock plus 4× the context.

**Error bars:** ±1.3× on the throughput axis alone (the sustained rate is 75–100 TFLOPS/card with the
upper end unmeasured). If the real rate is 100 TFLOPS/card the total is ~4.1 days; if 75, ~5.4 days.

## Acceptance criteria

- All 5 cards report the same negotiated link state, and no card is power-braked.
- `memtest_vulkan` passes on all 5 cards.
- GQA group size ≥ 4 confirmed in the config.
- FSA smoke test at 128k completes 100 steps without OOM or NaN.
- Pretrain leg completes 9B tokens at 8k without OOM or dataloader stalls.
- Adaptation leg completes 1B tokens at 128k without OOM.
- VRAM high-water is recorded for both legs and matches the budget above within 10%.
- Loss is recorded at 1B / 3B / 6B / 9B pretrain tokens and at 250M / 500M / 1B adaptation tokens,
  overall and per domain (web, code, math).
- **Context probe:** needle-in-a-haystack and long-document QA at 32k / 64k / 128k, to confirm the
  adaptation actually extended usable context rather than merely extending the window.

## Fallback

If decision gate 3 fails (the frozen table does not help at long context), fall back to a **dense 3B
under ZeRO-2**. It fits comfortably at 128k and carries no unmeasured architectural assumptions:

| Leg | Memory @ 128k | tok/s | Wall-clock |
|---|---:|---:|---:|
| Pretrain @ 8k, 9B tokens | 16.4 GB | 17,272 | 6.03 days |
| Adaptation @ 128k, 1B tokens | 36.5 GB | 3,064 | 3.78 days |
| **Total** | | | **9.81 days** |

That is 2.2× slower than the Engram design, but it removes the single largest unmeasured assumption.
ZeRO-2's CPU-hook risk is milder here than for MoE because a dense model has fewer, larger tensors.

## Results

Not run yet. Results go to `results.md` in this shape:

| Run | Date | Leg | Tokens | Context | Topology | tok/s | TFLOPS/GPU | VRAM high-water | Val Loss (Overall / Web / Code / Math) | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## What this unblocks

This is the run that produces the 128k model. It supersedes `01-dense-baseline.md` as the capability
target — the 500M dense baseline remains the control, but it is a control for a 32k plan that this
experiment replaces.

The three decision gates are also the three cheapest experiments in the whole program: a config read,
a kernel smoke test, and a 1B-token A/B. Run them first, and the 4.5-day estimate becomes a
measurement rather than an extrapolation.
