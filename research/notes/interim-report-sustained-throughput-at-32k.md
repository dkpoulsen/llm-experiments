---
title: Interim report — sustained-throughput-at-32k
id: interim-report-sustained-throughput-at-32k
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-sustained-throughput-at-32k
created: '2026-10-27T13:47:58.657323Z'
updated: '2026-10-27T19:02:13.183549Z'
status: review
type: interim
tier: practitioner
content_type: review
deprecated: false
summary: 'The six per-card TFLOPS figures are a ladder across lock state, power cap
  and duration, not a dispute: defensible sustained training denominator is ~90-100
  TFLOPS/card (450-500 aggregate, 45-50% MFU), not 162/810; 6ND is invalid at 32k
  where attention is 88% of step FLOPs and total FLOPs are 8.5x 6ND, so the repo''s
  80-120k tok/s is 2-3x above the 39.6k hardware ceiling and realistic is ~9-22k tok/s
  (5-12 days for 9B tokens, not 25 h).'
---

# Interim report: sustained-throughput-at-32k

**Locus question:** What is the sustained bf16 training throughput per card — 42, 49, 74.6-78.5, 91-100, 155.7, 162, or 164.4-192.7 TFLOPS — and is the repo's 6-FLOPs/param/token model even valid at 32k, where attention's quadratic term is 52-72% of step FLOPs?
**Flavor:** dialectical

## What the corpus already said

The corpus holds at least six per-card figures and no training-shaped measurement at all. [[performancemd]] is the strongest measured-compute source: BF16 tensor measures **164.4-192.7 TFLOPS** across seven tools and a dozen cards, with a **202.1 TFLOPS** ceiling that is "arithmetic: 2048 x 70 SM x 1410 MHz, verified exact", against **6.41 TFLOPS** locked. [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] reports a "tensor-core gate 74.6–78.5 TFLOP/s per card" and 74 SMs with cmpunlocker `6c442ee`. [[lessonsmd]] gives the PWRBRK# A/B: "measured fp16 throughput 39.3 vs 155.7 TFLOPS healthy". [[exploration-of-cryptocurrency-mining-specific-gpus-in-ai]] gives "Boost FP16 Performance 50.53 TFLOPS" as a *theoretical* number and says "The FP16 performance reported by PyTorch and GPU-Burn is only around 6.3 TFLOPS". [[all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]] measures "~42 TFLOPS (41869 GFLOPS)" in mixbench on the stock card and concludes "the Tensor Cores probably aren't working or at least not useful". [[full-specifications-170th-street]] and [[specifications-cmp-170hx-documentation]] both print **49.05 TFLOPS/s** in *both* the "Original" and "Unlocked" columns.

The repo side is `research.md` §2 and `experiments/01-dense-baseline.md`: **810 TFLOPS bf16 aggregate** (5 × 162), **6 FLOPs/param/token**, 24 layers, hidden 2048, 1B active, and a predicted **80-120k tok/s at 32k**. The repo also states "At realistic ~55% achieved MFU (405 TFLOPS aggregate, matching the measured 91–100 TFLOPS/GPU from the brief)". The 494M baseline is "~21 h at ~140k tok/s on 5 GPUs (91–100 TFLOPS/GPU)" with "context length unconfirmed, inferred ~4k".

Two things the corpus does *not* contain: any sustained bf16 tensor measurement at 250 W, and any training-shaped measurement at 32k. Both [[performancemd]] and [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] say so explicitly — every number is a GEMM, a microbenchmark, or an inference row.

## What the new sources say

### The health gate is a sustained 10-minute burn at 180 W, not a peak

[[2026-09-02-cmp170hx-health-gateipynb]] is the receipt behind the 74.6-78.5 figure, and it settles what that number *is*. The notebook's header table reads "gpu-burn -tc range | 74.6 - 78.5 TFLOP/s", and the TL;DR describes "10-minute `gpu-burn -tc`, a deterministic BF16/FP16/TF32/INT8 matmul cross-check, a full-VRAM `memtest_vulkan` pass, and a PCIe replay/AER snapshot". The hardware section is explicit: "4x CMP 170HX, 64 GiB each, forced airflow, **180 W power cap per card**". And the interpretation is a floor, not a peak: "All four cards clear 74 TFLOP/s and stay well under the 80 C core / 85 C memory stop conditions."

[[readmemd-2]] is the sanitized receipt chain and gives the per-card numbers plus the conditions that matter:

| GPU | gpu-burn -tc (10 min) | Core / memory temp (peak) |
|---|---|---:|
| 0 | 77.7 TFLOP/s | 62 °C / 64 °C |
| 1 | 76.4 TFLOP/s | 56 °C / 64 °C |
| 2 | 74.6 TFLOP/s | 63 °C / 71 °C |
| 3 | 78.5 TFLOP/s | 58 °C / 65 °C |

"Power limit: 180 W cap, enforced per card before its stage started"; "Cards run one at a time; no two cards under load simultaneously"; and the gate's own caveat: "this gate does not exercise sustained multi-hour operation". Peak core temperature is 63 °C against an 80 °C stop rule, so **the shortfall from 202 TFLOPS is not thermal throttling**. It is either the 180 W power cap pulling the clock down or `gpu-burn`'s own kernel efficiency — and the receipt carries no clock telemetry, so the corpus cannot distinguish the two.

### PaLM supplies the canonical FLOPs formula and the A100 MFU anchor

[[palm-scaling-language-modeling-with-pathways]] Appendix B is the authoritative statement of what "training FLOPs" means, and it is *not* 6ND:

> "Excluding self-attention, an N-parameter decoder-only model requires 6N matmul FLOPs per token seen (2N for forward and 4N for backward) ... The matmuls in dense self-attention add 6LH(2QT) FLOPs per token ... the model FLOPs utilization is the ratio of the achieved throughput in tokens per second to the theoretical peak throughput **R = P (6N + 12LHQT)**."

So the canonical formula is `6N + 12·L·H·Q·T` per token, where `H·Q` is the attention inner width. The repo's 6ND is the *non-attention* half of that formula with the attention half dropped.

PaLM's Table 3 is the MFU anchor set, and the A100 row is the one that matters here:

| Model | Params | Chips | MFU |
|---|---:|---|---:|
| GPT-3 | 175B | V100 | 21.3% |
| Gopher | 280B | 4096 TPU v3 | 32.5% |
| Megatron-Turing NLG | 530B | 2240 **A100** | **30.2%** |
| PaLM | 540B | 6144 TPU v4 | 46.2% |

The A100 arithmetic is spelled out: "it was trained on A100 GPUs with 312 peak matmul TFLOP/s, and achieved a training throughput of 65.43K tokens/second ... on 2240 GPUs ..., resulting in MFU of 29.7% without attention ... or 30.2% with it." And the rematerialization cost is quantified: PaLM 540B is "3.28 [TFLOPs per token] (non-attn+attn)" versus "4.10 (non-attn+attn+remat)" — **+25% hardware FLOPs** for a scheme that recomputes "75% of non-attention forward pass FLOPs ... while all attention forward pass FLOPs are recomputed". The corresponding hardware FLOPs utilization is 57.8%.

### FlashAttention-2 says attention runs at 50-73% of peak, not at GEMM efficiency

[[flashattention-2]] answers whether the quadratic term converts into wall-clock. It does, but at a discount:

> "FlashAttention-2 ... reaching **50-73% of the theoretical maximum FLOPs/s on A100** and getting close to the efficiency of GEMM operations ... up to 73% of the theoretical max throughput in the forward pass, and up to 63% of the theoretical max throughput in the backward pass ... In contrast, optimized GEMM can reach up to **80-90%** of the theoretical maximum device throughput."

And the end-to-end ceiling: "FlashAttention-2 reaches training speed of up to **225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)**". The non-matmul penalty is stated as a hardware fact: "the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32 ... each non-matmul FLOP is 16× more expensive than a matmul FLOP."

So attention FLOPs are real FLOPs that run at roughly 60-73% of peak on A100 silicon — lower than GEMM's 80-90%, but not a rounding error. The quadratic term at 32k is a genuine throughput penalty, not a paper exercise.

## Evidence synthesis

### The six figures are a ladder, not six measurements of one quantity

The dialectic dissolves once the figures are sorted by *what state of the card each one measures*. They differ along three axes — lock state, power cap, and duration — and on each axis they are mutually consistent:

| Figure | Card state | Duration / cap | Tier |
|---|---|---|---|
| 39.3 TFLOPS fp16 | PWRBRK# asserted (B30) | ~88 W, burst | measured (club A/B) |
| 42 TFLOPS fp16 | **locked**, mixbench SYCL, non-tensor path | burst | measured (2023 review) |
| 49.05 / 50.53 TFLOPS fp16 | **locked**, tensor path = peak/4 | theoretical / measured | vendor spec + paper |
| 6.41 TFLOPS bf16 | **locked**, tensor MMA | burst | measured (performancemd) |
| 74.6-78.5 TFLOPS | unlocked, 180 W cap | **10-minute sustained burn** | measured (club health gate) |
| 91-100 TFLOPS/GPU | unlocked, 250 W | end-to-end training, ~1-4k ctx | derived from repo baseline |
| 155.7 TFLOPS fp16 | unlocked, healthy | burst | measured (club A/B) |
| 162 TFLOPS bf16 | unlocked, 250 W box | burst GEMM (short benchmark) | measured (user) |
| 164.4-192.7 TFLOPS bf16 | unlocked, stock clocks | burst GEMM, 7 tools | measured (performancemd) |
| 202.1 / 213.6 TFLOPS | unlocked | arithmetic ceiling, 70 / 74 SM | arithmetic |

Two reconciliations fall out immediately. First, **50.53 = 202.1 / 4 exactly**: the crypto-mining paper's "theoretical FP16" and the 170th Street spec sheet's 49.05 are the *locked* tensor rate at a 1/4 issue divisor, not the silicon's peak. The "Unlocked" column on those spec sheets refers to the *memory* unlock (8 GB → 64 GB), which predates the SS0/SS1 compute unlock by years. Second, the 42 TFLOPS mixbench number is a non-tensor FP16 path on the same locked card — the review itself says "the Tensor Cores probably aren't working or at least not useful" and reports gpu_burn with `CUBLAS_TENSOR_OP_MATH` at 6.2 TFLOPS. So position B's four sources are not four measurements of the unlocked card; they are four measurements of a card state that no longer exists once cmpunlocker is installed.

### Position A: the unlocked card really delivers 162-193 TFLOPS bf16

This side has the best measurement quality in the corpus. [[performancemd]] reports BF16 at 164.4 TFLOPS (`mma_bf16bf16f32`, both tile shapes), 171.4 (mean of eight rented cards), 180.09 (torch GEMM 4096²), 183.75 (4096³ modded sweep), and 188.1-192.7 (`gemm_probe.cu` n=8192, fp32 accumulate). Eight unlocked 64 GB cards from one rental showed "under 2.5% per-card spread" and "8/8 passes" on a full byte-compare VRAM integrity test. The unlock is "a register write, not a clock change" and "Nothing in the shipping unlocker touches core clock, memory clock, power limit or PCIe link speed". The control is decisive: in the same-card A/B where FP32 moves 30.7x, memory bandwidth moves 1592 → 1599 GB/s (1.0x). And the ceiling is arithmetic, not a guess: 2048 FLOPs/SM/cycle × 70 SM × 1410 MHz = 202.1 TFLOPS, "verified exact". With the 74-SM unlock the ceiling rises to 213.6 TFLOPS, which makes the repo's 162 look *conservative* rather than optimistic. On the peak question, A wins outright.

### Position B: the card delivers only 42-49 TFLOPS fp16

This side is not lying; it is measuring a different card. [[exploration-of-cryptocurrency-mining-specific-gpus-in-ai]] is a 2023 study of the *stock* card, and its own conclusion is that tensor cores are unusable there: "This discrepancy likely stems from the inability to utilize Tensor Cores for acceleration", with llama.cpp prefill reaching "only 14-45% of theoretical limits". [[full-specifications-170th-street]]'s 49.05 appears in both columns because the page carries the compute unlock for FP32/FP64/INT8 (0.41 → 12.99 TFLOPS) but leaves FP16 at the vendor's locked-mode figure — it has no unlocked FP16 measurement to report. That is exactly the signature of a page whose FP16 row was never re-measured after the SS0/SS1 register writes were discovered. The strongest version of B is therefore: *on a stock card, before cmpunlocker, the tensor path is capped at 1/4 of peak and the practical FP16 rate is 42-49 TFLOPS*. That is true and it is irrelevant to a box that has already run a 494M model at 140k tok/s. The weakest version of B — "49.05 is what the card does even unlocked" — is refuted by a 3.4-3.9x measured gap that no measurement error can explain.

### Position C: real sustained throughput is ~75-100 TFLOPS/card

This is the position the new evidence most strengthens, and it is the only one that answers the locus's actual question. The health gate's 74.6-78.5 is a **10-minute sustained tensor burn at 180 W** — the closest thing in the entire corpus to a training-shaped sustained load — and it is 37-39% of the 202.1 ceiling. The repo's own baseline independently lands at 91-100 TFLOPS/GPU at 250 W, which is 45-50% MFU. And PaLM's Table 3 brackets it from above and below: MT-NLG on A100 achieves **30.2% MFU** (61 TFLOPS/card at 202.1 peak) and PaLM on TPU v4 achieves **46.2%** (93 TFLOPS/card). Three independent lines converge on **~60-110 TFLOPS/card sustained at 250 W**, i.e. **300-550 TFLOPS aggregate**, with a defensible centre at ~90-100 TFLOPS/card. The power scaling is coherent: 74.6-78.5 at 180 W → 91-100 at 250 W is a 1.16-1.34x throughput gain for a 1.39x power gain, exactly the sub-linear scaling you expect when P ∝ f^2.5-3. Position C is right about the *training* number; position A is right about the *peak* number; they are not in conflict.

### Position D: 6 FLOPs/param/token is the right denominator

The best case for D is that it is the standard, citation-backed convention: Chinchilla ([[training-compute-optimal-large-language-models]]) uses C ≈ 6ND, and for large models at short context the attention term genuinely is small — PaLM 540B at 2k context has attention at 1.6% of FLOPs (MFU 45.7% without attention vs 46.2% with it). The repo's 24-layer/2048-hidden/1B-active design at 1M-token batches is exactly the regime where 6ND is conventional. D is correct *as a convention for large models at short context*. It is not correct as a description of this box's 32k workload.

### Position E: 6ND omits the quadratic term, which dominates at 32k

The strongest version of E is not an estimate — it is PaLM's own published formula, `R = P(6N + 12LHQT)`, plus Megatron-Core's measured statement that "at 64K tokens, SDPA consumes **69% of FLOPs**, compared to only 10–15% in short-sequence scenarios" ([[scalable-training-of-mixture-of-experts-models-with]]). Applying the formula to the repo's own baseline model (494-500M, 28 layers, 16 heads × head_dim 128 → H·Q = 2048, T = 32,768):

- Linear: 6N = 3.0e9 FLOPs/token.
- Attention: 12·L·H·Q·T = 12 × 28 × 2048 × 32768 = **2.254e10 FLOPs/token**.
- Total: **2.554e10 FLOPs/token**; attention is **88.3%** of the step; total is **8.5x** the 6ND estimate.

The locus's own arithmetic used `4·L·s·d` for attention — the *forward-only* term — against a *training-total* linear term. That is an apples-to-oranges comparison that understates the attention term by exactly 3x. Corrected, Qwen3-0.6B's 596M params give **86% attention and 7.3x 6ND**, not 68% and 3.09x; the repo's 500M baseline gives 88% and 8.5x. At 4k the same model is 48% attention and 1.94x 6ND; at 1k, 19% and 1.24x. For the 1B-active/24L/2048 MoE at 32k the corrected numbers are 76% attention and 4.2x 6ND (not 52% and 2.07x). E wins, and wins by more than the locus claimed.

### The compounding consequence: the repo's 32k prediction is above the hardware ceiling

The two errors compound. The repo takes its measured 140k tok/s baseline and applies a 32k haircut of only 1.2-1.75x (140k → 80-120k). The FLOPs ratio from 4k to 32k is **4.39x**, and from 1k to 32k it is **6.89x**. That single mismatch is the whole error.

Worse, the prediction violates a hard bound. At 100% MFU of the measured ceiling, 5 cards deliver 5 × 202.1 = 1010.5 TFLOPS (1068 TFLOPS at 74 SMs). A 500M model at 32k needs 2.554e10 FLOPs/token, so the **absolute maximum is 39.6k tok/s (41.8k at 74 SMs)**. The repo's 80k tok/s requires 2043 TFLOPS aggregate = **409 TFLOPS/card = 2.02x the 202.1 ceiling**; 120k requires 613 TFLOPS/card = 3.03x. The prediction is not merely optimistic — it is physically impossible by 2-3x, before any MFU discount.

With a realistic denominator the picture is worse still. At 30.2% MFU (MT-NLG/A100) the box does 305 TFLOPS → 11.9k tok/s; at 46.2% (PaLM) 467 TFLOPS → 18.3k; at FA2's best-case 72% 727 TFLOPS → 28.5k. Adding PaLM's +25% rematerialization cost (the repo plans *full* checkpointing, which is +33%) and the ~4% DDP overhead gives **~9-22k tok/s, centre ~15k**. The 9B-token run therefore takes **~5-12 days, centre ~7 days, not 25 h**.

The quadratic term also flips the DDP overhead in the repo's favour. At 1M tokens and 32k context, a 1B-active/24L model needs 2.532e16 FLOPs per step; at 450 TFLOPS that is 56 s of compute against a 2.48 s all-reduce (2 GB at 0.805 GB/s) — **4.4% overhead**, or 3.5% with rematerialization, not the repo's 36%. The decomposition is clean: the 6ND correction alone (same 810 TFLOPS denominator) cuts the overhead from 36% to **7.9%**, and the peak-vs-sustained correction cuts it again to 4.4%. The repo's 36% is only correct at 6ND *and* 810 TFLOPS, i.e. at short context and burst peak. So the "DDP is the only viable topology" conclusion survives and strengthens, while the throughput headline collapses.

### PWRBRK# is a precondition, not a footnote

[[lessonsmd]]: "Power-brake (`PWRBRK#`, PCIe sideband pin B30) is a separate, board-level hazard: some workstation boards assert it on x16 slots, permanently capping a card near 88 W of a 250 W budget and dropping clocks to about 1140 MHz (measured fp16 throughput **39.3 vs 155.7 TFLOPS** healthy)." A braked card is 4.0x below a healthy one and looks exactly like a slow card. Every number in this note assumes `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"` reads inactive on all five cards. The repo already carries this as a pre-flight check; it should be treated as a hard gate on the throughput denominator, not a diagnostic.

## Committed position

The six per-card figures are not six incompatible measurements of one quantity — they are a coherent ladder across three axes (lock state, power cap, duration), and reading them as a dispute is the actual error. The defensible training denominator is **~90-100 TFLOPS/card sustained at 250 W (≈450-500 TFLOPS aggregate, ~45-50% MFU)**, not 162 and certainly not 810: the 162-193 band is a real GEMM peak measured at stock clocks (whether it survives a sustained 250 W burn is unmeasured), the 42-49 band is the locked card, and the 74.6-78.5 band is a 10-minute sustained burn at a 180 W cap that lands between what MT-NLG's 30.2% A100 MFU and PaLM's 46.2% TPU-v4 MFU predict. The 810 headline overstates sustained training throughput by **1.6-1.8x**, and every overhead estimate built on it is wrong by that factor. Separately, 6 FLOPs/param/token is invalid at 32k — PaLM's own published formula is `6N + 12LHQT`, and for the repo's 494-500M/28L/2048 model at 32k attention is **88% of the step** and total FLOPs are **8.5x** the 6ND estimate (the locus's 68%/3.09x used forward-only attention and understated the term by 3x). The consequence is decisive: the repo's **80-120k tok/s at 32k is 2-3x above the hardware's theoretical maximum of 39.6k tok/s at 100% MFU**, and ~4-13x above a realistic **~9-22k tok/s (centre ~15k)**; the 9B-token baseline takes **~5-12 days, centre ~7 days, not 25 h**. The one place the correction helps the repo is the topology argument: at 32k the quadratic term inflates compute 8.5x against a fixed gradient payload, so DDP overhead at 1M-token batches falls from 36% to **~4%**, which makes "DDP only" more true, not less.

- **Position:** The defensible sustained training denominator is ~90-100 TFLOPS/card (450-500 TFLOPS aggregate, 45-50% MFU), not 162/810; 6ND is invalid at 32k where attention is 88% of step FLOPs and total FLOPs are 8.5x the 6ND estimate; therefore the repo's 80-120k tok/s prediction at 32k is 2-3x above the hardware ceiling and the realistic rate is ~9-22k tok/s.
- **Confidence:** high on the 6ND invalidity and on 80-120k being unreachable (both are arithmetic bounds anchored in PaLM's published formula, the measured 202.1 TFLOPS ceiling, and Megatron-Core's 69%-at-64K measurement). Medium on the exact sustained denominator (~90-100 TFLOPS/card): no training-shaped measurement exists on this box, and the health gate's 74.6-78.5 is confounded by `gpu-burn`'s kernel efficiency versus the 180 W cap.
- **Boundary conditions:** This applies to 0.5-1B-active dense/MoE models at 32k context on 5 cards at 250 W with full activation checkpointing. At 4k the 6ND error is only 1.94x and the repo's prediction is within ~2x of feasible; at 1k it is 1.24x and 6ND is nearly right. For a 7B/32L/4096 model at 32k attention is 55% of FLOPs and the 6ND error is 2.2x — still wrong, but not catastrophic. Outside 32k-and-small-model, the 6ND convention is defensible.
- **What would change this position:** A training-shaped benchmark at 32k on one unlocked card — a 100-step 500M-model smoke test reported as sustained TFLOPS/GPU and MFU, with and without activation checkpointing, plus a measured 32k-vs-4k tok/s ratio. If a measured 32k run hit **>25k tok/s** on 5 cards (>640 TFLOPS aggregate model-FLOPs, >63% MFU), my centre estimate would be too pessimistic by ~1.7x; if it hit **>39.6k tok/s**, the FLOPs model itself would be wrong, which would require attention FLOPs ~3x below `12LHQT` (e.g. a 24-layer model with head_dim 64). Conversely, if a sustained BF16 GEMM burn at 250 W held only ~100 TFLOPS (i.e. the 164-193 band is burst-only), my throughput estimate is optimistic rather than pessimistic and the denominator should drop to ~60-70 TFLOPS/card. A clock trace during the health gate's burn would settle whether 74.6-78.5 is power-capped or kernel-limited.
- **Evidence weight:** 3 measured sources on the ladder ([[performancemd]] burst GEMM, [[2026-09-02-cmp170hx-health-gateipynb]] + [[readmemd-2]] sustained burn, [[lessonsmd]] PWRBRK#), 1 canonical formula source ([[palm-scaling-language-modeling-with-pathways]]), 1 kernel-efficiency source ([[flashattention-2]]), 1 measured attention-fraction source ([[scalable-training-of-mixture-of-experts-models-with]]), 1 internally inconsistent derived baseline (repo: 140k tok/s + 494M + ~4k ctx implies 163 TFLOPS/GPU if attention is counted, or 83 if only 6ND is counted — neither is the stated 91-100), and **0 training-shaped measurements on this box**.

## Open questions

- **Is the 164.4-192.7 BF16 GEMM a burst or a sustained number at 250 W?** [[performancemd]] states "Sustained SM clock is 1410 MHz" but the burn-in it describes is FP32 at 12.2 TFLOPS, not BF16 tensor at 190. A BF16 tensor GEMM at 190 TFLOPS draws far more power than FP32 at 12 TFLOPS, so the 250 W cap may pull the clock below 1410 under a real training step. Nobody has measured it.
- **Is 74.6-78.5 power-limited or `gpu-burn`-limited?** `gpu-burn -tc` typically reaches only ~40-50% of peak on A100-class silicon, so 78.5/202 = 39% could be either. The receipt has no clock telemetry; a 10-minute burn with `nvidia-smi --query-gpu=clocks.sm` sampled at 1 Hz would settle it and give the true 180 W sustained ceiling.
- **What MFU does a 500M model at 32k actually achieve on 70 SMs?** FA2's 50-73% attention efficiency is measured on 108-SM A100s; 70 SMs changes occupancy and wave quantization. The blended efficiency at 88% attention could be anywhere from 35% to 65%.
- **What context did the 494M baseline actually run at?** The repo's triple (140k tok/s, 494M, "inferred ~4k") is internally inconsistent with its own 91-100 TFLOPS/GPU claim: at 4k with attention counted the baseline implies 163 TFLOPS/GPU (80% MFU, implausible); at 91-100 TFLOPS/GPU it implies ~400-880 tokens of context. The run log would resolve which of the three numbers is wrong.
- **Does CSA2 actually reduce the attention FLOPs at 32k?** If the indexer is a second attention running in parallel, the FLOP win depends on the Full/Reindex/Reuse layer mix — cross-locus with the CSA2 investigator, but it bears directly on whether the 88% attention share can be bought down.
- **74 SMs vs 70 SMs.** The club measures 74 with cmpunlocker `6c442ee`, which raises the ceiling to 213.6 TFLOPS and the 32k hard bound to 41.8k tok/s. Whether the user's box is on that patch is unconfirmed; it changes the bound by 5.6%.

## Sources

1. [[2026-09-02-cmp170hx-health-gateipynb]] — club-170hx health-gate notebook: 74.6-78.5 TFLOP/s from 10-minute `gpu-burn -tc` at 180 W per card.
2. [[readmemd-2]] — health-gate receipt chain: per-card 77.7/76.4/74.6/78.5 TFLOP/s, 180 W cap, peak 63 °C core, no thermal throttle.
3. [[palm-scaling-language-modeling-with-pathways]] — canonical MFU formula `R = P(6N + 12LHQT)`; Table 3 MFU (A100 30.2%, TPU v4 46.2%); rematerialization +25%.
4. [[flashattention-2]] — attention reaches 50-73% of A100 peak (fwd 73%, bwd 63%) vs GEMM 80-90%; end-to-end 225 TFLOPS/GPU = 72% MFU.
5. [[performancemd]] — BF16 164.4-192.7 TFLOPS burst GEMM across 7 tools and 8 cards; 202.1 TFLOPS ceiling; 6.41 locked.
6. [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] — 74 SMs with `6c442ee`; tensor-core gate 74.6-78.5; power/clock ladder.
7. [[lessonsmd]] — PWRBRK# 39.3 vs 155.7 TFLOPS fp16; ~88 W cap and ~1140 MHz clocks when braked.
8. [[exploration-of-cryptocurrency-mining-specific-gpus-in-ai]] — locked-card theoretical FP16 50.53 TFLOPS; PyTorch/gpu-burn 6.3 TFLOPS; tensor cores unusable on stock card.
9. [[all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]] — locked-card mixbench FP16 ~42 TFLOPS; gpu_burn with tensor cores 6.2 TFLOPS.
10. [[full-specifications-170th-street]] — 49.05 TFLOPS in both Original and Unlocked columns; FP32 moves 0.41 → 12.99 but FP16 is never re-measured.
11. [[specifications-cmp-170hx-documentation]] — same 49.05 figure; FP32 0.41 → 12.99 unlocked.
12. [[scalable-training-of-mixture-of-experts-models-with]] — "at 64K tokens, SDPA consumes 69% of FLOPs"; long-context MFU holds at 88-129% of short-context MFU with CP/TP.
13. [[training-compute-optimal-large-language-models]] — Chinchilla's C ≈ 6ND convention, the source the repo's 6ND model rests on.
14. `research.md` (repo, not a vault note) — §2: 810 TFLOPS aggregate, 6 FLOPs/param/token, 36% DDP overhead at 1M tokens.
15. `experiments/01-dense-baseline.md` (repo, not a vault note) — the 80-120k tok/s prediction at 32k and the 25 h estimate for 9B tokens.
