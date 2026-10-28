# Synthesis outline

**Opening (no H2):** the direct answer in two sentences — the repo's recommendation survives, its reasoning does not, and the constraint it designed around is not the one that binds.

## Hardware Reality
Establishes that "Gen2 x4" is a compound of two independently-fixable axes and that the TFLOPS denominator is a ladder, not a number. Anchored on the measured 1.6 GB/s Gen2 x4 link rate, the 3.69–4.24 GB/s Gen2 x16 baseline, and the 42/49 → 74.6–78.5 → 91–100 → 162–193 TFLOPS ladder across lock state × power cap × duration. Beat: four link states and four TFLOPS figures are all defensible; the repo picked one of each without saying why.

## Training Topologies
Reframes "DDP only" as "data-parallel only" and names the two routes: ZeRO-2 at five ranks (33.6 GB, DDP-identical 2Ψ) and PP=5 (25.8 GB/card, 4–14% bubble). Anchored on the FSDP 3Ψ-vs-2Ψ derivation, the PP hand-off volume of 10,240 B/token/hop, and the measured PP-over-FSDP wins on consumer PCIe. Beat: the repo's topology conclusion is right and its table is wrong in three places.

## DDP Overhead Model
The joint re-derivation. Numerator grows 1.48× (0.75 GiB/s is busbw, so algbw is 0.503 GB/s); denominator grows ~7.6× (`6N + 12LHQT` plus sustained-not-peak TFLOPS); net ~6.6%. Then the second consequence: the same correction makes 80–120k tok/s at 32k impossible and stretches 9B tokens to ~3–7 days. Anchored on the `2(n−1)/n` busbw formula, PaLM's MFU formula, the measured 2.19× 32k/4k ratio, and the measured 1.7% on a no-NVLink PCIe 4 box at 1M-token batches. Beat: the repo got the right number by multiplying a wrong numerator by a wrong denominator.

## MoE & EP
The bandwidth case collapses: EP ≈ 1.4× and TP ≈ 2.8× at 32k, not 70× and not 28×. What remains is availability (no SM80 training-grade EP implementation) and critical-path serialization (TP's collectives sit inside the layer; EP's can overlap). Then the unstated reason to avoid MoE here: it spends the scarce resource (memory) to save the abundant one (FLOPs), and multiplies DDP's payload by the total-to-active ratio. Anchored on Megatron-Core's parameter-compute asymmetry, the closed list of EP libraries requiring NVLink/P2P/RDMA, and the 60%-of-training-time figure on NVLink-rich hardware.

## Engram & Frozen Memory
The reframe: freezing is a collective-elimination design, not a memory trick — Engram's training recipe uses All-to-All to gather rows and dispatch gradients, the one collective this fabric cannot run. Then the arithmetic: 24 GB of frozen tables (two layers), 9.94 GB of naive logits, ~53 GB/card for 1B trainable — marginal, not comfortable. Then the alternative: ZeRO-2, Adafactor and 8-bit Adam all fit 7B fully trainable, so freezing is an architecture choice. Then the measurement problem: the paper's own numbers show trained Engram loses to MoE at 0.92B, so frozen grafting is the only variant that wins — and the win is in score points, a unit with no established noise floor. Anchored on the paper's table sizes and score deltas, the All-to-All recipe, and the measured seed-variance floor of 0.004 nats.

## CSA2 at 32k
Go/no-go, not judgment call. The memory objection is soft (64–129 GB score tensor, 32× reducible to 0.40 GB); the architecture objection is hard (five independent confirmations that no SM80 training path exists). Substitute NSA or FSA — both runnable on Ampere — but with the payoff corrected: ~1.71× end-to-end, not 6.3×, and the sign depends on GQA group size at 32k. Then the regime point: every published 32k sparse-attention win is an adaptation-stage result, so the repo's from-scratch 32k plan is outside the published regime, and 8k-then-32k also cuts the quadratic penalty for most of the run. Anchored on StreamIndex's measured peak, the FlashMLA/TileLang/SGLang scope statements, FSA's end-to-end table, and NSA's 270B-at-8k protocol.

## Verdict
The inverted framing, stated in one actionable sentence, then the ranked levers (sparse attention > capacitor mod > optimizer swap), then the four pre-flight measurements (`lspci -vv`, 5-rank `all_reduce_perf` with both columns, sustained BF16 burn at 250 W, GQA group size), then the ECC/memtest_vulkan requirement, then the honest open items. Beat: the recommendation is robust across all four branches of the measurement tree even though the reasoning survives none of them.

## Source Tensions
All six disputes, both sides named, each resolved: width-is-not-speed; the club's evidence standard vs the repo's; incompatible units; adaptation vs from-scratch; P2P's two incompatible figures; 180 W vs 250 W.

## Sources
Omitted — `citation_style` is `wikilink`, so the `[[note-id]]` markers in the body are self-resolving.
