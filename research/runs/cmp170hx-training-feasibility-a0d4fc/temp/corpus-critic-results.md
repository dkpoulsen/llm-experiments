# Corpus critic results — step 8 gap-fill

Vault grew from 177 to **225 notes** during this step. Six gaps were named; three were overturned, one strengthened, one confirmed with a mechanism finding, one partially overturned. `comparisons.md` carries a "Post-critic revisions" section (A–F) that the draft must use instead of the pre-critic numbers.

## Gap outcomes

| id | target position | outcome | decisive source |
|---|---|---|---|
| cc-1 | attention is 88% of step FLOPs → 8.5× 6ND → 9–22k tok/s | **OVERTURNED** | `readmemd-faircslmtrainbench-at-main` (measured 2.19× per-token ratio, not 4.39×) |
| cc-2 | DDP overhead ~4%, fabric not binding | **STRENGTHENS** | `i-tested-the-cmp170hx` (1.6 GB/s at Gen2 x4 on this exact hardware); falsifier arithmetically unreachable |
| cc-3 | ZeRO-2 / PP=5 are the two viable routes | **NO OVERTURN** + mechanism finding | `pcie-110-gpu-스케일링` (ZeRO-2 beats DDP on CNS fabric); `perf-29-...-issue-177` (overhead is CPU-side hooks, not bandwidth) |
| cc-4 | CSA2 unportable → run NSA (6.3×/3.4×) | **PARTIAL OVERTURN** | `published-as-a-conference-paper-at-iclr-2026` (NSA ≈ 1.71× end-to-end, and *slower* than full attention at 32K when g=1) |
| cc-5 | 9B null is only a bound; 3.5% noise floor | **OVERTURNED** | `quantifying-pretraining-variance-...` (seed SD ≤0.004 nats = 0.14% relative, ~25× below the floor) |
| cc-6 | 0.75 GiB/s is busbw; P2P is latency-only | **PARTIAL OVERTURN** | `2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb` (NCCL over BAR1 P2P at 4.83 GB/s busbw, N=2) |

## What changed in the committed positions

**cc-1 — throughput band moves up.** SLMTrainBench measures 32k/4k per-token cost at **2.19×** on A100 80GB PCIe for a 505.5M model, using the same `6N + 12LHS` MFU formula. A 1.60× MFU uplift absorbs 40% of the quadratic penalty. Systematic across sizes (150M 2.21×, 250M 2.27×, 350M 2.23×, 500M 2.19×; 2.31–2.36× on H100). Cross-checked against SkyLadder (1B: 2.20× FLOPs, time tracks FLOPs within 5%), Ulysses (7B TFLOPs/GPU flat across 65k→262k), and Seq1F1B (7B/TP8: MFU *rises* 110.6→136.7 from 32k→128k).
- Throughput band: **14–39k tok/s** (centre ~20–27k), not 9–22k.
- 9B tokens: **~3–7 days**, not 5–12.
- "Attention is 88% of FLOPs" survives in the FLOPs convention but **not** as 88% of wall-clock — the 32k step runs at 63.88% MFU.
- The committed config was arithmetically impossible: 494M × 28L × d=2048 = 1.41B params. A consistent 494M/28L model has d≈1216 → 3.51× FLOPs, 2.19× measured wall-clock. Even granting 4.39× FLOPs, the MFU uplift calibrates it to ~2.97×.

**cc-5 — the 9B experiment is not hopelessly underpowered.** 23 runs of a 184M transformer (C4, 3.7B tokens, bf16, bitwise determinism verified) span [2.941, 2.953] — max deviation **0.4%**, within-group SD **≤0.004 nats**. That is ~25× below the 3.5% floor. `effective-learning-rate-governs-loss-dynamics-...` gives Llama-124M seed-only SD of 1.11e-2 / 1.62e-2 nats and a practical detection threshold ≈**0.01 nats**. The memory axis's 0.014–0.082 nat deltas sit **at or above** the floor, so a 9B A/B can resolve the upper half of the range and a null is an informative bound (~0.02 nats). **The ladder recommendation is downgraded from necessary to nice-to-have.** Caveat: floors measured at 124M–184M, not 500M/9B; seed variance typically shrinks with scale, which favours the overturn but is not directly measured at 500M.

**cc-6 — P2P is a bandwidth lever at N=2 and a pessimization at N=3 and at x4.** NCCL over BAR1 P2P on 4× CMP 170HX (64 GiB BAR1, Gen2 x16, two PLX, four patches): N=2 3.69→**4.83 GB/s busbw (+31%)**; N=3 4.24→**3.86 GB/s (−9%)**. Real workload: TP4 decode 19.4→26.1 ms/step (−25 to −44%). At Gen2 x4 with P2P working, "no row shows a strict win"; PP4 −19.1% aggregate. Counter-evidence at PCIe 4.0 x16: 8× RTX 3090 NCCL busbw 24.7 vs 2.7 GB/s SHM (9×). The literal claim must be restated as "P2P is a bandwidth lever only at N=2 on wide links; on a 5-rank Gen2 x4 fabric it is at best neutral, at worst negative." The "64 MB BAR1 ⇒ NCCL hangs" reading is **superseded** — the unlock supplies 64 GiB BAR1, the `kbusIsPcieBar1P2PMappingSupported_GH100` blocker disappears, and collectives run.

**cc-2 — confirmed and sharpened.** No 5-rank receipt exists anywhere accessible (club has 1/2/3/4-card results only). The falsifier is arithmetically unreachable: measured Gen2 x4 link on this hardware is **1.6 GB/s** (`i-tested-the-cmp170hx`: "800MB/s at Gen 1 and 1.6GB/s at Gen 2"); NCCL drives 53% of raw link at Gen2 x16, so at x4 ~0.85 GB/s busbw → ~0.53 GB/s algbw at N=5 → 3.8 s over 56.8 s = **6.6%** (13.2% with fp32 buckets). Reaching 0.1 GB/s would need NCCL at <6% of a link it drives at 53%. NCCL maintainer sjeaugey confirms 5- and 7-GPU runs are "much lower than others" — mechanism is offset alignment (4/8 bytes vs 16), **payload-shape-dependent and paddable away in DDP buckets**.

**cc-3 — routes hold, with a new mechanism risk.** ZeRO-2 beats DDP at every GPU count on the closest fabric analogue (10×3090, CNS, ~3 GB/s shared: +27%/+50%/+10%). But on 8×A100 NVSwitch, ZeRO-2's 2.90× slowdown is **not bandwidth** — every overlap knob gave zero change (3,071→3,093 ms); the delta is **CPU-side per-parameter Python hooks**, fabric-independent, and would bite on this box worst on many-small-tensor models (MoE experts). Also: a 4×RTX PRO 6000 no-NVLink 500M MoE at ~1M-token effective batch shows gradient comm at **50% of per-microbatch step time**, with `no_sync()` every 10th step cutting 18 h → 13.2 h.

**cc-4 — CSA2 verdict stands; NSA payoff halves.** No sm_80 DSA training kernel exists (FlashMLA Ampere port is decode-only with "poor" performance due to register overflow; upstream tracker says forward-only; flash-attention sparse-MLA bwd is SM100-only; TileLang profiled on H200; SGLang says "no end-to-end CUDA path for DSA on A100/SM80"). But FSA (ICLR 2026) measures **NSA ≈ 1.71× full attention end-to-end**, and states directly: "when the sequence length is 32K, one GQA group contains one query head, NSA consistently falls short of full attention." fla-org's Triton NSA: 2.66×/1.76× at 16,384, 1.35×/1.06× at 8,192, **slower than flash at 4,096**. So sparse attention's step-time lever is ~−20 to −30%, not −39 to −54%, and **its sign depends on the repo's GQA group size at 32k — the decisive unmeasured variable**. The substitution arm IS runnable on sm_80 (FSA tested on A100 SXM, backprops through 65,536+32,768; fla-org ships full NSA training).

## Confidence updates

| position | before | after | reason |
|---|---|---|---|
| throughput 9–22k tok/s | medium-high | **revised to 14–39k** | measured 2.19× ratio replaces 4.39× arithmetic |
| DDP overhead ~4% | medium | **high** | falsifier unreachable; 6.6% bridge from measured link rate |
| 9B underpowered | medium | **overturned** | measured seed SD 25× below the assumed floor |
| P2P latency-only | medium (~72%) | **restated** | N=2 counter-example at 4.83 GB/s; substance holds at N≥3 and x4 |
| CSA2 unportable | high | **high** | five independent confirmations, none changed |
| NSA payoff 6.3×/3.4× | medium | **revised to ~1.71×** | FSA end-to-end measurement; sign depends on GQA group size |
| ZeRO-2 route | medium | **medium** (unchanged) | routes hold; new CPU-hook risk is fabric-independent |

## Open items the draft must state honestly

1. **No 5-rank all-reduce receipt exists** in any accessible source. The user's own `all_reduce_perf` output is the only decisive artifact for the algbw-vs-busbw question, and no fetch can obtain it.
2. **The repo's GQA group size at 32k is not in the corpus** — and it decides whether sparse attention is a win or a loss.
3. **Both measured noise floors are at 124M–184M**, not 500M/9B. The extrapolation favours the overturn but is not directly measured at the target scale.
4. **The 4-card BAR1-P2P README references `receipts/nccl/` logs that are not committed** to the repo; if they surface they are the strongest possible cc-6 receipt.
5. **Fetch tooling note:** crawl4ai crashed once and returned wrong content for a raw.githubusercontent URL, creating one note with false provenance (`deepspeed-zero-vs-pytorch-fsdp2-measured-18x-on-nvlink-barely-finishing-on-pcie`), which was deprecated. `--provider builtin` worked for every subsequent fetch.
