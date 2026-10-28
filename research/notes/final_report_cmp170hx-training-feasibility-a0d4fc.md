The recommendation in the repo survives. The reasoning offered for it does not. Data-parallel training, million-token global batches, frozen Engram tables, no expert or tensor parallelism, and a sparse-attention substitution arm are the right program for a five-card NVIDIA CMP 170HX box — and each of those choices is defended in the documentation by a number that is unmeasured, mislabelled, or computed against the wrong denominator. The two largest errors point in opposite directions: the all-reduce is slower than the model assumes once the right column is read, and the step is far longer than the model assumes once attention is counted. The second correction dominates — the ratio falls 5.4× — which is why the conclusion looks sound, and why the program cannot tell when it has stopped being sound.

What binds this box is not the PCIe Gen2 x4 fabric. At million-token global batches, DDP's gradient all-reduce costs single-digit percent of step time — 5.3–7.1% at Gen2 x4 depending on which column the plan's 0.75 GiB/s is read as, 11.4% at Gen1 x4, 1.5% at Gen2 x16 (T4) — against the documentation's 36%. What binds is wall-clock: 90–100 TFLOPS per card sustained (T1/T2), not the 162 TFLOPS GEMM peak the plan divides by, which turns a planned 25-hour run into 3–7 days and makes sparse attention, not the capacitor mod, the only lever worth pulling.

Evidence carries a tier throughout. **T1** is measured on this box and is user-reported only: the corpus contains no on-box training, all-reduce, NCCL or DDP measurement at all, and both [[performancemd]] and [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] say so explicitly about their own pages. **T2** is a community report or a measurement on adjacent hardware with a receipt. **T3** is paper-measured. **T4** is arithmetic from the tiers above.

## 1. Hardware Reality: Link Width, Power Brake, and What the Numbers Assume

A PCIe link's usable bandwidth is the product of two numbers set in different places. Generation fixes the signalling rate per lane — Gen1 is 2.5 GT/s, Gen2 is 5 GT/s — and width fixes how many lanes carry data. A card can be fast-and-narrow or slow-and-wide, and the two properties are fixed by different mechanisms: generation by firmware and fuses, width by whether the board has the capacitors that let the extra lanes carry a signal.

### Two axes, four states

The documentation states "PCIe Gen2 x4" as one measured fact and derives every bandwidth number from it. It is not one fact. Width is a board-level omission: twelve of sixteen lanes ship with their AC-coupling capacitors depopulated, the lane fuses are all clear (`OPT_PCIE_LANE_DISABLE` = `0x00000000`, "No lanes fused off"), and the registers show a physical-layer discontinuity — `LnkCap: Width x16` against `LnkSta: Width x4 (downgraded)` [[physical-modsmd]]. Soldering 24 × 0402 220 nF X7R capacitors restores x16 with no software patch, and the same page is explicit that the mod "changes link width only. It never changes PCIe generation" [[physical-modsmd]]. Generation is firmware: stock trains at Gen1, the cmpunlocker link patch reaches Gen2, and Gen3 is blocked by `OPT_DISABLE_GEN3_SPEED`, which reads 1 on 4/4 cards checked [[pcie-subsystemmd]], [[club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github]].

| State | Reached by | One-way rate | Tier |
|---|---|---:|---|
| Gen1 x4 | stock | ~0.8–1.0 GB/s | T2 |
| Gen2 x4 | unlocker only | 1.6–1.68 GB/s | T2 |
| Gen1 x16 | capacitor mod only | ~3.2 GB/s | T4 |
| Gen2 x16 | unlocker + capacitor mod | 6.21–6.68 GB/s | T2 |

The anchors are measured. A hands-on review of four of these cards states "the transfer rate is 800MB/s at Gen 1 and 1.6GB/s at Gen 2" [[i-tested-the-cmp170hx]] (T2), and a sanitized NCCL receipt on three cards at Gen2 x16 behind PLX switches measures 6.21–6.30 GB/s host-staged copy against a 6.68 GB/s single-card D2H reference [[readmemd]] (T2). The modal state is Gen2 x4 — a four-card rig reports "All four slots negotiate Gen2 x4" across cold power cycles [[readme_enmd]] (T2) — because the capacitor mod is 24 solder joints nobody performs by accident.

### What the link delivers to a collective

The rate that matters to a training step is the all-reduce rate, not the link rate, and the two differ by a large, measurable factor. At Gen2 x16, NCCL reaches 3.69–4.24 GB/s busbw against a 6.21–6.68 GB/s one-way rate — 55–64% of the link [[readmemd]] (T2/T4). Scaling by the measured width ratio 1.68/6.68 = 0.252 puts the x4 busbw band at 0.93–1.07 GB/s and the algbw band at 0.58–0.67 GB/s (T4), because a ring all-reduce's busbw is 2(n−1)/n times its algbw, which at five ranks is exactly 1.6 [[all_reducecu]], [[whats-the-difference-in-allreduce-for-factor-between-the-nccl-and-nccl-test-issu]].

The 1.6 factor is the ideal ring load and it assumes no alignment penalty. Five is an odd rank count, and both the canonical PCIe interconnect study and an NCCL maintainer flag it: Tartan warns "One should avoid adopting 5 GPUs in their application setup" because ring formation congests under NUMA [[evaluating-modern-gpu-interconnect-pcie]] (T3). The maintainer attributes the 5- and 7-GPU shortfall to offset alignment — NCCL "cannot use optimized 16 bytes loads and stores" unless the size is a multiple of 16×5 [[why-is-the-bandwidth-utilization-of-7-gpus-much-lower-than-that-of-6-8-gpus-for]] (T2). Every overhead figure below is therefore a floor, and padding DDP buckets to 80-byte multiples belongs in the pre-flight, not in the optimiser.

Against that band, the plan's 0.75 GiB/s (0.805 GB/s) is either a busbw reading or an algbw reading, and only one is consistent with the best measured efficiency. Read as busbw it sits at 75–86% of the calibrated band — a slightly conservative measurement. Read as algbw it implies a busbw of 1.29 GB/s, which is 77% of the one-way rate on a fabric where every byte crosses PCIe twice; the best efficiency measured on this hardware class is 64% [[readmemd]] (T2). The busbw reading is the one the evidence supports — at ~72% confidence, not as a measurement.

The repo's 494M/140k tok/s datapoint weakly favours algbw, and if the box's "no P2P" is a driver-configuration statement rather than a hardware one — `cudaDeviceCanAccessPeer` returns 0 even when `cuMemcpyPeer` succeeds via host staging — the algbw reading becomes plausible and the overhead doubles. The pre-flight should establish P2P status with `cuMemcpyPeer`, not `cudaDeviceCanAccessPeer` alone.

### The compute ladder

The corpus holds at least nine per-card compute figures, and reading them as a dispute is the error. They are a ladder across three axes — lock state, power cap, duration.

| Figure | State | Duration | Tier |
|---|---|---:|---|
| 39.3 TFLOPS | PWRBRK# asserted, ~88 W | short | T2 |
| 74.6–78.5 TFLOPS | unlocked, 180 W enforced | 10-minute burn | T2 |
| 91–100 TFLOPS | unlocked, 250 W | end-to-end training, ~4k ctx | T1 |
| 155.7 TFLOPS | healthy unlocked, 250 W | short | T2 |
| 162 TFLOPS | the plan's figure | burst GEMM | T1 |
| 164.4–192.7 TFLOPS | unlocked, stock clocks | burst GEMM, 7 tools | T2 |
| 202.1 TFLOPS | arithmetic ceiling, 2048 × 70 SM × 1410 MHz | — | T4 |

The 162–193 band is real and it is a GEMM peak under a 202.1 TFLOPS arithmetic ceiling [[performancemd]] (T4). The only figure with a full receipt chain is the 180 W one: 77.7 / 76.4 / 74.6 / 78.5 TFLOP/s per card over a ten-minute `gpu-burn -tc`, peak core 63 °C against an 80 °C stop rule, so the shortfall from 202 TFLOPS is not thermal throttling [[readmemd-2]] (T2).

Three independent lines converge on 75–100 TFLOPS/card sustained at 250 W, centre ~90, aggregate 375–500 (T4). The 180 W burn gives 74.6–78.5; scaling to 250 W at the sub-linear rate expected when power goes as clock cubed gives 83–88; and PaLM's Table 3 brackets the range, with Megatron-Turing NLG on A100 at 30.2% MFU and PaLM on TPU v4 at 46.2% [[palm-scaling-language-modeling-with-pathways]] (T3). A 90-TFLOP centre is 44.5% of this card's ceiling, and the plan's 810 TFLOPS aggregate implies 80% MFU for a five-rank DDP job with full activation checkpointing, which no published training run achieves.

### The power brake, and the measurement nobody has made

The PWRBRK# power brake is the hardware hazard that most threatens every number here. Asserted on PCIe sideband pin B30, it silently caps a card near 88 W of a 250 W budget, drops clocks to ~1140 MHz, and takes measured fp16 throughput from 155.7 to 39.3 TFLOPS — a 4× collapse that presents as a bad GPU rather than a bad pin [[lessonsmd]] (T2). It is detected with `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"` and fixed by masking B30 or using a riser that does not route it.

The measurement gap is narrower and more consequential: every receipt is one, two, three or four ranks [[readmemd]], [[2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]] (T2), and the three-card receipt concedes "One probe run per pair; no cross-run error bars" [[readmemd]].

## 2. Training Topologies on a No-P2P PCIe Fabric

Six ways to split a training job across five cards, and they differ in what their communication scales with.

- **Data parallelism (DDP)** replicates the model and all-reduces gradients once per optimizer step, so its traffic scales with parameter count and not with sequence length.
- **Fully sharded data parallelism (FSDP, also called ZeRO-3)** shards parameters, gradients and optimizer state, all-gathering each layer before use and reduce-scattering it after — the same parameter scaling, paid per layer.
- **Pipeline parallelism (PP)** splits the model by layer into stages and passes activations between them, so its traffic scales with tokens × hidden × stages.
- **Tensor parallelism (TP)** splits each layer's matrices across ranks and all-reduces inside every layer, so its traffic scales with tokens × hidden × layers and sits on the critical path.
- **Expert parallelism (EP)** distributes mixture-of-experts (MoE) experts across ranks and all-to-alls each token's activations to the ranks holding its experts.
- **Sequence parallelism (SP)** shards the sequence dimension across ranks and exchanges attention state per layer — Ulysses by all-to-all on the head dimension, Ring Attention by P2P ring passes of K/V blocks — and at 32k it is the standard alternative to activation checkpointing.

### DDP: one collective per step

DDP's only per-step traffic is the gradient all-reduce, and PyTorch's reducer registers one autograd hook per parameter, so a parameter with `requires_grad=False` has no gradient accumulator, no hook fires, and no bucket includes it [[pytorch-ddp-comm-hooks-default_hookspy-bf16fp16-compress-hooks]] (T3). That is the mechanism behind the frozen-table design and it is sound. The one rule the plan gets exactly right is `no_sync()`.

The closest published analogue confirms the shape. Four RTX PRO 6000 cards on PCIe 4 with no NVLink, training a 500M-total/~330M-active MoE at an effective global batch of ~1M tokens — the same regime — measured "gradient communication accounted for 50% of the single-step time". Enabling `no_sync()` so the all-reduce fires only on the tenth micro-step cut wall-clock from under 18 hours to 13.2 hours, leaving ~1.7% in sync [[a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m]] (T2).

### FSDP: 1.5×, not 2×, and the comparison is confounded

The plan's "FSDP costs ~2× DDP" is wrong on the multiplier and wrong on the comparison. ZeRO's arithmetic is explicit — "The total volume is therefore 3Ψ which is 1.5x compared to the baseline" [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] (T3). The 2× only reproduces if you force fp32 gradient reduction via `MixedPrecision(reduce_dtype=torch.float32)`; FSDP's default `reduce_dtype=None` "still running gradient reduction in low precision" [[fullyshardeddataparallel-pytorch-213-documentation]] (T3). With bf16 reduce, FSDP moves 6 B/param against DDP's 4 = 1.5×; forcing fp32 gives 8 B/param = 2.0×. The plan's 2× is the pessimistic variant, not the default.

### PP and ZeRO-2: the two routes the plan never considers

PP's traffic is O(tokens × hidden × stages). For five stages at 32k tokens and hidden 2048, one hand-off is 32,768 × 2048 × 2 B = 134 MB (T4), and 1F1B overlaps each hand-off with the previous micro-batch's compute rather than serializing it. At 1.68 GB/s the hand-off takes ~80 ms inside a per-micro-batch compute window of 240–440 ms, so it hides with room to spare. What remains is the bubble: (5−1)/32 = 12.5% for 1F1B at 32 micro-batches, 11.1% on the exact (S−1)/(M+S−1) formula, ~4.2% for zero-bubble schedules — a third of 1F1B's [[zero-bubble-pipeline-parallelism]] — and under 4.5% measured for RoundPipe with asynchronous optimizer updates [[efficient-training-on-multiple-consumer-gpus-with]] (T2/T3). 

PP=5 puts 7B/5 = 1.4B parameters per stage × 16 B/param = 22.4 GB of weights and optimizer state, plus ~3.4 GB of in-flight activations, at ~25.8 GB/card (T4) — 64 GB with 38 GB of headroom, and it trains the experts without freezing anything. The caveat is static shapes — `torch.distributed.pipelining` requires them, a genuine constraint for a model with MoE routing, Engram lookup and CSA2 top-k.

ZeRO-2 at five ranks is the other route. It shards optimizer state and gradients but not parameters, so it costs 4.8 B/param × 7B = 33.6 GB of model state (T4) and its communication is "exactly 2Ψ" — the same byte volume as DDP's all-reduce [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] (T3) — while preserving Adam, so comparability with the control survives. The evidence on whether it is faster than DDP is genuinely split. On the closest published fabric analogue — ten RTX 3090s sharing ~3 GB/s of PCIe with P2P disabled — ZeRO-2 beats DDP at every GPU count, by +27% at four and +50% at eight [[pcie-110-gpu-스케일링]] (T2). On 8×A100 with NVSwitch it loses 2.90×, because the ~2.4 s delta was CPU-side Python execution of per-parameter hooks [[perf-29-throughput-by-switching-from-deepspeed-to-ddp-issue-177-tensorautoopen-2]] (T2). The recommendation is ZeRO-2 for the 7B arm with a measured step-time A/B against PP=5 before committing.

### The corrected comparison

The communication-cost comparison table below is the whole argument in one place.

| Topology | Collectives per step | Traffic scales with | Per-GPU volume at 1M tokens | Overhead vs 56.3 s | Fits 7B on 64 GB | Verdict |
|---|---|---|---:|---:|---|---|
| DDP | 1 all-reduce | O(params) | 2.0 GB bf16 grads | 5.3–7.1% | No (112 GB/card) | Viable ≤1B trainable |
| ZeRO-2 | reduce-scatter + all-gather | 2Ψ (DDP-identical) | 2.0 GB | 5.3–7.1% + CPU hooks | Yes (33.6 GB) | Best memory/comm trade |
| FSDP / ZeRO-3 | all-gather + reduce-scatter per layer | 3Ψ | 3.0 GB | 1.5× DDP bytes, per-layer on critical path | Yes | Viable but hook-bound |
| PP=5 | 2 P2P per stage boundary | O(tokens × hidden × stages) | 134 MB per hand-off | 4–14% bubble | Yes (25.8 GB) | Viable, underrated |
| TP=5 | 4 all-reduces per layer | O(tokens × hidden × layers) | 126 GB | ~2.8× | Yes | Feasible, dominated |
| EP=5 | 2 all-to-alls per MoE layer | O(tokens × top-k × hidden × EP⁻¹) | 62.9 GB | ~1.4× | Yes | Ruled out on availability |
| SP=5 | 2 all-to-alls per attention layer | O(tokens × hidden × layers) | 62.9 GB | ~1.4× | Yes | Ruled out on availability |

The table's shape is the argument. DDP and ZeRO-2 are cheap; PP is cheap and solves memory; TP is expensive but feasible; EP and SP are unavailable rather than expensive. "DDP only" should be restated as data-parallel only: ZeRO-2 is the memory fix that costs nothing in communication, and PP=5 is the fallback that costs nothing in memory.

## 3. The DDP Overhead Model at 1M-Token Batches

An overhead ratio is communication time divided by compute time. The communication half is a payload divided by the achieved collective bandwidth; the compute half is FLOPs divided by sustained throughput. Two conventions decide both halves. The first is the bandwidth column: nccl-tests reports algbw, the payload rate, and busbw, which multiplies algbw by 2(n−1)/n to describe the per-rank link rate, so at five ranks the two differ by 1.6×. The second is the FLOPs convention: the shorthand "6ND" counts only the matmuls outside attention, six FLOPs per parameter per token, while the canonical formula adds the attention term and grows with context length. Model FLOPs utilization (MFU) is achieved tokens per second divided by the peak the formula implies, so a peak used as a sustained rate inflates MFU by the gap.

### The joint re-derivation

The plan's model is a single ratio with a single constant: overhead = 3.6 × 10⁵ / global_batch_tokens, giving 36% at 1M tokens and 18% at 2M. It comes from dividing a 2.0 GB bf16 gradient payload by 0.75 GiB/s to get 2.7 s, and dividing 6 FLOPs/param/token × 10⁹ params × 10⁶ tokens by 810 TFLOPS to get 7.4 s. Both halves are wrong, and they are wrong in opposite directions.

One premise is unstated and load-bearing: 2.0 GB is 2 bytes per parameter, which is only reachable with `bf16_compress_hook` registered. Under standard AMP the parameters and gradients stay fp32, so the bucket payload is 4.0 GB and the overhead doubles [[pytorch-ddp-comm-hooks-default_hookspy-bf16fp16-compress-hooks]], [[does-nccl-allreduce-use-fp16-mixed-precision-pytorch-forums]] (T3); with pure-bf16 parameters the hook is a no-op cast.

It is measured on this fabric class — on four RTX A6000s over PCIe with no NVLink, bf16 gradients moved plain DDP from 2.06× to 2.65× at 124M and 1.91× to 2.72× at 355M [[ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compre]] (T2). At the corrected window it is worth only ~3% of step time: halving a 4.0 GB payload saves 1.85 s against 56.3 s of compute, and the ~18% it buys belongs to the plan's own 36% premise. The measured analogue says the same thing: the +33% corresponds to a 131% overhead regime, 170 ms of communication against a 130 ms compute step. 

The cluster's consensus runs against intuition — compression pays more, not less, on slow fabrics: AGoQ's 8-bit all-reduce is 3.7× at 10 Gbps against 3.4× at 200 Gbps [[agoq-activation-and-gradient-quantization-for]], and CERAR cuts Pythia-1.4B by 17.3% on PCIe while running 33% slower on NVLink [[communication-efficient-distributed-training]] (T3). So the hook is insurance against the Gen1-x4 and short-context cases, not a lever at 32k.

The numerator grows 1.48×. 0.75 GiB/s is 0.805 GB/s. At N=5, busbw = 1.6 × algbw by construction, so if 0.805 GB/s is a busbw reading then algbw is 0.503 GB/s and the payload takes 3.98 s rather than 2.7 s. Physics corroborates the reading: as algbw, 0.805 GB/s would require NCCL to hit 77% of the one-way link rate on a fabric where every byte crosses PCIe twice, and the best efficiency measured on this hardware class is 64%. The measured anchor — 1.6 GB/s link, ~53% NCCL efficiency, ~0.85 GB/s busbw, ~0.53 GB/s algbw — lands within 5% of the busbw reading (T2/T4), about 10% below the floor of the width-scaled band above because the two derivations start from different inputs.

The denominator grows 7.6×. First, 6 FLOPs/param/token is the non-attention half of the canonical formula. PaLM states it explicitly: "an N-parameter decoder-only model requires 6N matmul FLOPs per token seen (2N for forward and 4N for backward)" and separately that "The matmuls in dense self-attention add 6LH(2QT) FLOPs per token", so the theoretical peak is R = P(6N + 12LHQT) [[palm-scaling-language-modeling-with-pathways]] (T3). For a 1B-active, 24-layer, hidden-2048 model at 32k, the attention term is 12 × 24 × 2048 × 32,768 = 1.933 × 10¹⁰ FLOPs/token against a linear term of 6.0 × 10⁹ — attention is 76% of the step and total FLOPs are 4.2× the 6ND estimate (T4). Megatron-Core corroborates from measurement: "at 64K tokens, SDPA consumes 69% of FLOPs, compared to only 10–15% in short-sequence scenarios" [[scalable-training-of-mixture-of-experts-models-with]] (T3). 

Second, 810 TFLOPS is a peak, not a sustained rate; the defensible figure is 450–500. Together: 2.533 × 10¹⁶ FLOPs per 1M-token step ÷ 450 TFLOPS = 56.3 s of compute, not 7.4 s.

The net effect is **36% → 6.6%**: the ratio falls 5.4×.

| Configuration | Compute per 1M-token step | All-reduce time | Overhead |
|---|---:|---:|---:|
| Plan: 6ND, 810 TFLOPS, 0.805 GB/s as algbw | 7.4 s | 2.5 s | 34% |
| Sustained denominator only: 6ND, 450 TFLOPS | 13.3 s | 2.5 s | 19% |
| 32k, 450 TFLOPS, 0.805 GB/s as algbw | 56.3 s | 2.5 s | 4.4% |
| 32k, 450 TFLOPS, busbw reading (3.2 GB wire at 0.805 GB/s) | 56.3 s | 4.0 s | 7.1% |
| 32k, 450 TFLOPS, Gen1 x4 busbw (~0.5 GB/s) | 56.3 s | 6.4 s | 11.4% |
| 32k, 450 TFLOPS, Gen2 x16 busbw (3.7 GB/s) | 56.3 s | 0.86 s | 1.5% |
| 2M tokens, 32k, 450 TFLOPS, busbw reading | 112.6 s | 4.0 s | 3.5% |

Doubling the global batch to 2M doubles the compute window and leaves the 2.0 GB payload unchanged, so the overhead halves to 3.5% — 7.1% if the buckets stay fp32 — and the invariance claim holds at both batch sizes.

The 36% is reproducible from the plan's own inputs only under the algbw reading the evidence rejects: it is the overhead of a short-context run at burst peak, and under the busbw reading it is 54–58% at the plan's denominator. The 32k arms sit at 1.5–11.4%, centre 5–7%. The algbw-versus-busbw question, which looked like the decisive dispute, moves the answer by about 2.7 percentage points once the compute window is corrected — a narrative question, not a quantitative one. The 6.6% itself is a T4 extrapolation, not a measurement: it scales a three-card Gen2 x16 receipt by width and NCCL efficiency, and the corpus holds no five-rank all-reduce at any width.

The calibrated algbw band gives 5.3–6.1%, the busbw row of the table above gives 7.1%, and Gen1 x4 gives 11.4% — so the defensible statement is 5.3–7.1% at Gen2 x4, before the unquantified N=5 alignment penalty. It sits within a factor of four of the adjacent-hardware measurement at the same batch regime, 1.7% amortized after `no_sync()` [[a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m]] (T2).

### Where the invariance claim fails

The plan's headline — "DDP's overhead depends only on batch size, not model size" — is an identity for dense models. Communication time is 2·N·b/B_wire; compute time is 6·N·T/F_agg; the ratio is 2·b·F_agg/(6·T·B_wire), independent of N. At 1M tokens and the corrected denominators that is 6.6%, and it holds whether the model is 500M or 7B, until it stops fitting.

It fails for MoE, and the failure is mechanical rather than a matter of degree. Every expert weight is a parameter with `requires_grad=True`, so DDP's reducer buckets and all-reduces the full parameter set regardless of which experts saw tokens; an expert that received zero tokens contributes a zero gradient but still occupies its slice of the all-reduce buffer. Megatron-Core states both halves: "parameters, gradients, and optimizer states must reside in memory during training" while "This is not a matter of degree; it is a qualitative difference from dense models" [[scalable-training-of-mixture-of-experts-models-with]] (T3). The corrected formula is overhead ≈ 6.6% × (N_total/N_active), which for the plan's own 7B/1B axis is 47% (T4). The invariance claim is true only for the frozen-table design where trainable equals active. Stated properly: freezing the tables is what makes DDP's overhead invariant, not a property of DDP.

### The same correction destroys the throughput prediction

The plan predicts 80–120k tok/s for a ~500M dense baseline at 32k. At 100% MFU of the measured 202.1 TFLOPS ceiling, five cards deliver 1010.5 TFLOPS; a 500M model at 32k needs 2.554 × 10¹⁰ FLOPs per token, so the absolute maximum is 39.6k tok/s (T4). The prediction requires 2.0–3.0 PFLOPS aggregate, or 400–600 TFLOPS per card — 2–3× the hardware ceiling, before any MFU discount. At 30.2% MFU the box does 305 TFLOPS and 11.9k tok/s; at 46.2% it does 467 TFLOPS and 18.3k tok/s; adding PaLM's +25% rematerialization cost and the 4–7% DDP overhead gives 14–39k tok/s, centre 20–27k (T4). Nine billion tokens therefore takes 3–7 days, centre 4–5, not 25 hours.

The calibration that matters is that the FLOPs ratio is not the wall-clock ratio. SLMTrainBench measures its 500M-class model at 4k = 26,297 tok/s (39.88% MFU) and at 32k = 11,990 tok/s (63.88% MFU) on an A100 80GB PCIe, computing MFU with the same 6N + 12LHS formula: a measured per-token cost ratio of 2.19×, with a 1.60× MFU uplift absorbing 40% of the quadratic penalty [[readmemd-faircslmtrainbench-at-main]] (T3). The pattern is systematic — 2.19–2.27× across 150M–500M — and SkyLadder's 1B model cross-checks it with a 2.20× FLOPs ratio from 8K to 32K [[skyladder-better-and-faster-pretraining-via]] (T3). The FLOPs ratio at this configuration is 4.2×; the wall-clock ratio is 2.19×.

One further correction: the plan's baseline triple is arithmetically impossible. 494M parameters at 28 layers and hidden 2048 is ~1.41B parameters, not 494M (T4). A physically consistent 494M/28-layer model has hidden ≈1216, which gives a 32k/4k FLOPs ratio of 3.51× and the measured wall-clock ratio of 2.19×.

## 4. MoE and Expert Parallelism: Ruled Out or Misjudged?

A mixture-of-experts layer replaces one dense feed-forward block with many expert blocks and a router that sends each token to a few of them, so total parameters can be large while per-token FLOPs stay small — only the selected experts compute. Expert parallelism distributes those experts across ranks, which means each token's hidden state must be shipped to whichever ranks hold its experts (dispatch) and the results shipped back (combine): two all-to-alls per MoE layer. That is the only collective in the training stack whose payload scales with tokens rather than parameters, and it is the one the plan rules out.

### The 70× is an aggregate-versus-per-GPU error

The plan's estimate — two all-to-alls per MoE layer, ~70× the compute time at 1M-token batches — divides the aggregate dispatch-plus-combine volume across all ranks (~400 GB) by the per-GPU collective rate. Each GPU sends only its share. Megatron-Core's per-GPU send volume in one all-to-all is T·K·h·(EP−1)/EP, where T is the tokens on that GPU, K is top-k, h is hidden width and EP is the expert-parallel size [[scalable-training-of-mixture-of-experts-models-with]] (T3). At a 1M-token global batch over five data-parallel ranks, T_local = 200,000; with top-k 2, hidden 2048 and EP=5, dispatch is 200,000 × 2 × 2048 × 0.8 = 6.55 × 10⁸ elements = 1.31 GB in bf16; dispatch plus combine is 2.62 GB per layer; across 24 MoE layers that is 62.9 GB per GPU per step (T4). At 0.805 GB/s that is 78 s, or 1.39× the corrected 56.3 s window.

TP at the same configuration moves four all-reduces per layer over forward and backward, each carrying T_local × h × 2 B = 819 MB per rank, which on a five-way ring is 1.6× that on the wire = 1.31 GB, so 4 × 1.31 × 24 = 126 GB per GPU per step, 156 s, or 2.8× against the corrected window (T4).

### The bandwidth case evaporates; availability and placement remain

Correcting the number strengthens the conclusion rather than weakening it, because bandwidth arithmetic was never the right argument. What rules EP out is software. Every MoE-specialised EP library in existence — DeepEP, pplx-kernels, NIXL-EP, NCCL EP, UCCL-EP — assumes GPUs can directly access one another through NVLink, GPUDirect P2P or RDMA, and "existing EP communication libraries cannot operate on these systems" [[efficient-expert-parallel-communication]] (T3). 

The one prototype that does work on PCIe without P2P, ThunderEP, reaches 2.00× dispatch and 1.53× combine over NCCL on 6×4090 and 6×5090 systems — but it is integrated into vLLM and evaluated on inference, with no gradient collectives [[efficient-expert-parallel-communication]] (T3). The inference-only siblings reach 85–90% of an NVLink A800 configuration at a fraction of the hardware cost, but none of them ships a backward pass [[democratizing-moe-inference-on-commodity-gpus]] (T3).

So the corpus holds no evidence of a training-grade EP implementation for a no-P2P, no-NVLink, no-RDMA fabric — the closest are VarioPath, a PCIe-native all-to-all scheduler evaluated on inference [[variopath-workload-aware-all-to-all-communication-for-pcie-gpu-clusters]], and ThunderEP, a no-P2P dispatch/combine with no backward. Both are systems projects to port, not hardware impossibilities.

TP is dominated rather than ruled out. Its all-reduces sit inside the layer, serialized with the layer's compute, so equal volume does not mean equal cost: EP's all-to-all can overlap with expert computation, which is why communication-computation overlap is the standard EP optimization. DeepSeek-V3's DualPipe hides both all-to-all and pipeline communication behind compute, reaching "near-zero all-to-all communication overhead" as long as the compute-to-communication ratio holds [[deepseek-v3-technical-report]] (T3). 

TP has no equivalent hiding place at this scale — though model-parallel communication can be overlapped with compute given an overlap-friendly schedule [[on-optimizing-the-communication-of-model-parallelism]] (T3), and on PCIe TP overtakes DP above ~20B parameters by avoiding full-parameter synchronisation [[efficient-training-on-multiple-consumer-gpus-with]] (T2). At 1B trainable, TP=5 fits and costs ~2.8×; it is dominated by ZeRO-2 and PP=5, not excluded by the fabric.

### MoE is the wrong axis for this box

MoE's purpose is to decouple parameter count from per-token FLOPs, which lets you buy capacity with memory. On a five-card box with 320 GB of HBM, memory is the scarce resource and FLOPs are comparatively abundant at 375–500 TFLOPS aggregate, so MoE spends the scarce resource to save the abundant one. Worse, it multiplies DDP's payload by the total-to-active ratio derived above: a 7B-total/1B-active model all-reduces 7B parameters while computing only 1B, so its DDP overhead is 47% rather than 6.6% (T4).

The plan's own MoE arm is the biggest correction here: a 7B-total/1B-active MoE has 7B trainable parameters, because the experts are learned, and under plain DDP with Adam that is 112 GB per card (T4) — it does not fit 64 GB.

## 5. Engram and Frozen Memory: Making 7B Fit on 64 GB

Engram is conditional memory: hashed n-gram lookup tables that return learned hidden vectors and splice them into the residual stream, so the model can store knowledge in a table rather than in weights. The table costs VRAM but almost no per-token FLOPs, because a lookup is not a matmul. Memory Grafting takes a donor model's hidden states for the most frequent n-grams, freezes them as a table, and grafts them into a smaller recipient model that looks rows up by longest-match suffix. The question is whether freezing those tables is what makes a 7B-total / ~1B-trainable model fit DDP on 64 GB cards — and whether the design can be measured at all.

### Freezing is a collective-elimination design, not a memory trick

The plan justifies freezing on memory: 7B fully trainable needs 112 GB/card, so freeze the tables. That arithmetic is correct for the one configuration it tests and it is not the strongest available argument. Engram's own training recipe shards the tables across GPUs and uses All-to-All to gather active rows in the forward pass and dispatch gradients in the backward pass [[conditional-memory-via-scalable-lookup]] (T3). That is precisely the collective this fabric cannot run — the same all-to-all that rules out expert parallelism. Freezing removes the gradient collective; replicating the table removes the forward gather collective, and the design needs both. A frozen-but-sharded table would cost ~4.9 GB/card and leave ~16 GB more headroom, but it would reintroduce the forward all-to-all at one-fifth the volume. Replication is what buys the collective elimination, not freezing alone.

### The table is 24 GB, not 12 GB

The paper is unambiguous: "For each n-gram order, we select the top 1M most frequent n-grams and compute their frozen memory representations offline, yielding 3M entries per grafted layer. Each entry stores a hidden-state vector of the grafting model in bfloat16; for Qwen3.5-35B-A3B, this corresponds to roughly 12 GB of frozen memory per layer" [[memory-grafting-scaling-language-model]] (T3). The Figure 4 caption confirms the total: "0.92B trainable parameters with approximately 12B frozen grafting memory offloaded". Three million entries × 2,048 dimensions × 2 B = 12.29 GB per layer; two grafted layers (donor layers 8 and 24) = 24.58 GB = 12.29B parameters (T4).

That figure holds only on the paper's 3M entries per layer, and the repo's own §3 gives a different number: "33M entries per grafted layer (top 11M n-grams per order)" alongside the same "roughly 12 GB of frozen memory per layer". Thirty-three million entries × 2,048 × 2 B is 135 GB per layer and 270 GB for two — the design would fit neither 64 GB nor the box's 320 GB aggregate. The verdict below therefore rests on adopting the paper's 3M, not the repo's 33M; the repo has to reconcile §3 against §5 before the memory budget is trusted.

### Does it fit DDP on 64 GB?

The memory-budget table below is the arithmetic component by component.

| Component | Plan's budget | Corrected | Tier |
|---|---:|---:|---|
| Frozen table (12.29B params, bf16) | 12 GB (6B) | 24.58 GB | T3/T4 |
| Trainable recipient (0.92B × 16 B) | 16 GB (1B) | 14.72 GB | T4 |
| Checkpointed activations at 32k, 24L, d=2048 | 3.2 GB | 3.22 GB | T4 |
| LM-head logits (fused CE) | omitted | 0.62 GB (9.94 GB naive) | T3/T4 |
| **Total** | **~31 GB** | **~43.1 GB** | T4 |
| Headroom on 64 GB | ~33 GB | **~21 GB** | T4 |

Yes, the frozen-table design fits DDP on 64 GB — at ~43.1 GB with fused cross-entropy and 21 GB of headroom, or ~53.1 GB and 11 GB of headroom without it. The conclusion survives; the arithmetic is wrong by ~12 GB in one direction (the table is twice as big) and ~10 GB in the other (the logits term is omitted). DDP's all-reduce payload is only the 0.92B trainable gradient, 1.84 GB, which at 0.53 GB/s algbw is 3.5 s against the 56.3 s compute window — 6.2% (T4).

### Freezing is an architecture choice, not a memory necessity

The plan's headline tests only one configuration. Three alternatives fit:

| Route | Bytes per trainable param | Model state | Total with activations | Fits 64 GB? |
|---|---:|---:|---:|---|
| DDP + Adam (fp32 master + m + v) | 16 | 112 GB | 115.8 GB | No |
| DDP + 8-bit Adam (block-wise quantized) | 6 | 42 GB | 45.8 GB | Yes |
| DDP + Adafactor (bf16 w+g, factored state) | 4 | 28 GB | 31.8 GB | Yes |
| ZeRO-2 at 5 ranks + Adam | 4.8 | 33.6 GB | 37.4 GB | Yes |

Adafactor keeps only the per-row and per-column sums of Adam's second moment, giving state proportional to n + m rather than nm [[adafactor-adaptive-learning-rates-with-sublinear-memory-cost]] (T3), and 8-bit Adam's block-wise quantization matches 32-bit performance at 1.5B [[published-as-a-conference-paper-at-iclr-2022]] (T3). The plan's headline is false as a general claim.

### Why freeze anyway: the sign flips, and training the table costs step time

Freezing is nonetheless the right choice here, for two reasons the plan does not state.

The first is that at this scale the trained-table baseline is negative. The paper's 0.92B/50B-token table reads MoE 45.62, vanilla Engram 45.03, MG-GLM 45.83, MG-DeepSeek 46.42, MG-Qwen3.5 46.98 [[memory-grafting-scaling-language-model]] (T3). Vanilla Engram loses to MoE by 0.59 points, which the paper attributes to hash collisions at the limited table size. At 2.8B/100B tokens the sign flips: MoE 51.95, vanilla Engram 52.43, MG 53.86. On a ≤1B-active box, freezing plus grafting is the only configuration in the paper with a positive sign.

The second is fabric cost. Training the table rather than freezing it means its gradients must move. Under DDP the gradient tensor is dense, 12 GB at bf16, so the all-reduce moves 1.6 × 12 = 19.2 GB per rank; at 0.805 GB/s that is ~24 s against 56.3 s of compute, a ~42% step-time penalty (T4). Under the Engram recipe of sharded tables plus all-to-all the traffic is ~20 GB per step, ~25 s, a ~44% penalty (T4). Freezing removes both, and the penalty is a function of the link: at Gen2 x16 with 3.7 GB/s busbw it is 9%, while at Gen1 x4 with ~0.5 GB/s it is 68%.

### The measurement problem is a unit problem

The deltas above are score points on a ten-task average, not nats of training loss, and no published Engram or CSA2 delta is reported in a unit in which a noise floor has been established. That is a stronger claim than calling it underpowered: it is falsifiable by a single experiment that reports the delta in nats with two seeds.

The floor itself has been measured. A 23-run variance decomposition at 184M parameters and 3.7B tokens in bf16, with bitwise determinism verified across data centres, reports "Within-group standard deviations are ≤0.004 for every category; FP arithmetic is even tighter at 0.001" [[quantifying-pretraining-variance-data-initialization-and-floating-point-arithmet]] (T3). That is ≤0.004 nats, or ≤0.14% relative — roughly 25× below the 3.5% restart spread the pessimistic reading borrowed from the scaling-law literature.

Against a 0.004-nat floor, the memory axis's deltas are at or above it. Engram's allocation law buys 0.0139 nats (1.7248 → 1.7109) at 993M active and ~100B tokens [[conditional-memory-via-scalable-lookup]] (T3), 3.5× the tighter floor; OverEncoding's headline at 500B tokens is 0.082 nats on OLMoE-1.3B [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] (T3), 20× above it. So a 9B-token A/B at 500M can resolve the upper half of the expected effect range, and a null would be an informative bound (~0.02 nats) rather than a foregone conclusion. The caveat is honest: the floor is measured at 124M–184M, not at 500M/9B, and seed variance typically shrinks with scale, so the extrapolation favours detectability but is unmeasured at the target scale. The right response is not 100B tokens per arm but the 9B A/B plus a second seed if the delta lands within 2× of the floor.

## 6. CSA2 Sparse Attention at 32k Context

Sparse attention attends to a selected subset of the key-value cache instead of all of it. An indexer scores candidate positions, a top-k selects which ones each query will read, and the attention kernel then computes only over those. The selection can be token-granular or block-granular, and that choice decides whether the kernel is fast: fewer FLOPs do not automatically mean less latency, because a sparse kernel's inner loop is a memory gather. CSA2 is DeepSeek's compressed sparse attention, second generation: Full, Reindex and Reuse layer modes, a hierarchical sparse indexer with a 16,384-candidate cap, and a KV footprint of 890 bytes per token at inference.

### The memory objection is soft; the architecture objection is hard

The indexer's score tensor is a `[B, S, H_I, T]` FP32 intermediate materialized before the head-sum, which at S=32,768 with V4-Flash dimensions is 64 GiB, with a measured peak of 129.00 GB and 317 ms on a 140 GB H200 [[streamindex-memory-bounded-compressed-sparse]] (T3). The pure-PyTorch cuDNN reference has the same defect, materializing `(B, S_q, H_q, S_k)` FP32 at 68.7 GB for 32k/H_q=16 [[dsa_referencepy]] (T3/T4). But the fundamental requirement is only the head-summed `(B, S_q, S_k)` FP32 tensor: 4.29 GB at 32k (T4). StreamIndex demonstrates the fix on real hardware: 129.00 GB → 0.40 GB at S=32,768, and 10.3× faster (317 ms → 31 ms) [[streamindex-memory-bounded-compressed-sparse]] (T3). The indexer's score-tensor memory is a solved engineering problem.

The architecture objection is hard, and harder than the plan assumes. Every training-capable DSA kernel in existence is Hopper-or-newer:

| Stack | Forward | Backward | Architectures |
|---|---|---|---|
| cuDNN DSA | SM90/SM100 | SM90/SM100 | Hopper, Blackwell |
| FlashMLA | sm_100a / sm_103a + Ascend 950 | — | Blackwell, Ascend (Hopper dropped) |
| TileLang `deepseek_v32` | H800/H200 tuned, wgmma | H800/H200 tuned | Hopper |
| StreamIndex | Triton, tested sm_90a | Triton, tested sm_90a | H100 / H200 |
| vLLM `TRITON_MLA_SPARSE` | Triton, tested 8×A800 | none | SM80/SM121, inference only |

The mechanism is architectural. The tuned stacks are built on Hopper's `wgmma.mma_async`; Ampere has only `mma.sync`, and TileLang's own `mma.sync` fallback path runs 1.81× slower on H800 SXM — a layout-induced fallback inside an SM90 kernel, not an sm_80 measurement — so an sm_80 build would be slower still [[tilelangexamplesdeepseek_v32-at-main-tile-aitilelang-github]] (T2). The one arch-portable Triton stack is worse than slow: StreamIndex's attention forward reaches only 8% of H200 BF16 peak at V4-Pro dimensions against TileLang's 40% on the same shape [[streamindex-memory-bounded-compressed-sparse]] (T3), and at 8% of peak even a 64× FLOP reduction does not produce a usable training kernel.

The indexer, not the sparse MLA, is the hard half — HISA measures 1.6 ms for the Sparse MLA operator against 5.6 ms for the indexer at 64K on A100, and concludes "the main performance bottleneck in DSA lies in the indexer rather than in Sparse MLA itself" [[published-as-a-conference-paper-at-colm-2026]] (T3). The layer mix bounds how much of that matters: V4.1-Flash ships 2 SWA + 4 Full + 4 Reindex + 30 Reuse over 40 layers, so 32 of 40 run no indexer and the 16,384-candidate cap binds on only the 4 decoder Reindex layers [[deepseek-v41-flash]] (T3). Attention density — top-k 512 of 32,768, 1.56%, a 64× FLOP cut — is the right quantity; the cap is not the lever the plan thinks it is.

The FlashMLA row is decisive. Its current release lists sm_100a / sm_103a and Ascend 950, DeepSeek-V4.1 only, and it has dropped Hopper entirely [[github-deepseek-aiflashmla-flashmla-efficient-multi-head-latent-attention-kernel]] (T2). The production sparse-attention forward for CSA2 does not run on Hopper, let alone Ampere. flash-attention's sparse-MLA forward and backward kernels are SM100-only files (`flash_fwd_mla_sm100.py`, `flash_bwd_mla_sm100.py`) [[cutebwdsm100-add-sparse-mla-deepseek-v4-backward-kernels-2621-dao-ailabflash-att]] (T2). 

The Ampere port of FlashMLA exposes only paged-KV decode, has no backward, and reports that "the performance of Ampere is currently poor due to register overflow" — 464 GB/s and 59 TFLOPS on A100 SXM against 2,039 GB/s and 312 TFLOPS peak [[github-pzhao-engflashmla-github]] (T2). SGLang states "SGLang currently has no end-to-end CUDA path for running DeepSeek Sparse Attention (DSA), including DeepSeek-V3.2, on NVIDIA A100 / SM80" [[feature-add-an-opt-in-sm80-fallback-for-deepseek-sparse-attention-dsa-on-nvidia]] (T2).

The only sm_80 path is vLLM's `TRITON_MLA_SPARSE`, which adds a Triton-based sparse MLA backend "so DSA sparse-MLA models (DeepSeek-V3.2, GLM-5.x) can run on GPUs where DeepGEMM and FlashMLA-Sparse are unavailable" [[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]] (T2). It passes its kernel tests on 8×A800-80GB PCIe, but it is a serving backend: no backward pass, no training loop, no optimizer step. So CSA2 at 32k on this box is not an experiment, it is a kernel-porting project — write an sm_80 indexer backward and an sm_80 sparse-MLA backward, tune both to usable MFU, then validate the distillation loss, before learning anything about whether the architecture helps.

### The payoff is half the headline, and its sign is not guaranteed

The plan cites NSA's 6.3× forward and 3.4× backward at 32k as evidence of a large sparse-attention win. Those are kernel-level ratios measured on 8×A100 with Triton kernels [[native-sparse-attention-hardware-aligned-and-natively]] (T3). FSA (ICLR 2026) is the measured end-to-end comparison the corpus needed: Llama3-8B, Qwen3-14B and Qwen2.5-32B at 32K and 64K in BF16, on H20 and H200 with NVLink. It reports up to 1.25× and 1.09× on average end-to-end training speedup over the Triton NSA kernel, and up to 2.47× and 1.86× on average over full attention [[published-as-a-conference-paper-at-iclr-2026]] (T3). Dividing the two averages implies NSA ≈ 1.71× full attention end-to-end, not 6.3×. The fla-org Triton implementation reproduces a shrinking advantage: 2.66× forward at 16,384 tokens, 1.35× at 8,192, and 0.72× at 4,096 — NSA is slower than flash at short context [[github-fla-orgnative-sparse-attention-efficient-triton-implementations-for-nativ]] (T2).

Worse, the sign is not guaranteed. FSA reports directly that vanilla NSA lags behind full attention in many tested cases, even with its sparse attention mechanism: when the sequence length is 32K and one GQA group contains one query head, NSA consistently falls short of full attention [[published-as-a-conference-paper-at-iclr-2026]] (T3). GQA is grouped-query attention, where several query heads share one key-value head; the group size is the number of query heads per KV head. If the 500M baseline has a group size of 1 at 32k, NSA is a net loss; at 4 or more it wins. That configuration value is the decisive unmeasured variable and it is not in the corpus. It is also free to measure — one line in the model config.

The corrected payoff is therefore −20% to −30% of step time at a 76–88% attention FLOP share, not the −39% to −54% implied by the 6.3×/3.4× kernel ratios (T4). That is still the largest single lever available — but it is not net of the indexer. Top-k is non-differentiable, so the indexer is trained by a clipped-log KL distillation against the main attention distribution rather than end-to-end [[dsa_referencepy]] (T3), which adds a second score-tensor forward and backward and a separate loss term to every step. The indexer's cost is additive and measured at 3.5× the kernel it feeds [[published-as-a-conference-paper-at-colm-2026]] (T3). Netting that against a 76–88% attention share is what turns −20% to −30% into a much thinner margin, and it is the term the plan's estimate omits.

### Gather granularity decides the sign, not the candidate cap

The determinant of the 32k payoff is gather granularity, and LongCat measures it directly. LSA versus dense MLA per attention layer at 32K: 1.08× forward, 0.79× backward, 0.83× total, with crossover only at 64K (1.39×) [[longcat-sparse-attention-taming-the-lightning-via]] (T2). Because 0.83× is a speedup ratio, LSA takes 1/0.83 = 1.20× dense MLA's latency — a net loss. The mechanism: each selected token is fetched as an independent gather of a single 1,152 B latent KV vector spanning three cachelines, and "the resulting net effective bandwidth thus plummets to a mere ~4.5% (~1/22) of peak" [[longcat-sparse-attention-taming-the-lightning-via]] (T2). That is why 64× fewer FLOPs produces a 20% slowdown. CSA2, token-granular and DSA-lineage, would be a +10.7% to +14.7% step-time penalty at 32k even if it ran (T4) — a 50–69 percentage-point swing against the NSA arm, decided by gather granularity rather than by the candidate cap.

### Run NSA or FSA instead

The substitution is not itself a porting project.

- **FSA** — the official implementation is "well tested with NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)", supports "Training and inference (prefill)", and calls `loss.backward()` through a 65,536 + 32,768-token batch; it requires `flash-attn==2.6.3` [[github-relaxed-system-labflash-sparse-attention-efficient-implementations-of-nat]] (T2).
- **fla-org Triton NSA** — ships full training support via `flame` [[github-fla-orgnative-sparse-attention-efficient-triton-implementations-for-nativ]] (T2).
- **MoBA** — the fallback, also pinned to `flash-attn==2.6.3` [[github-moonshotaimoba-moba-mixture-of-block-attention-for-long-context-llms-gith]] (T2), but it is compute-neutral at 32k and lossy alone: its fitted 32K trailing-loss curve sits above full attention's in all five runs (1.546·C^−0.108 against 1.464·C^−0.097) [[moba-mixture-of-block-attention-for]] (T3).

The regime point is the last piece. Every published 32k sparse-attention win is an adaptation-stage result. NSA pretrained on "270B tokens of 8k-length texts, followed by continued training and supervised fine-tuning on 32k-length texts with YaRN" [[native-sparse-attention-hardware-aligned-and-natively]] (T3). DeepSeek-V4.1-Flash trains sparse attention from scratch at 64K across 45T tokens [[deepseek-v41-flash]] (T3) — a capability receipt, not a speedup receipt, and at 64K rather than 32k. DeepSeek-V4 warms up with dense attention for the first 1T tokens and introduces sparse attention only at 64K [[deepseek-v4]] (T3). The plan's from-scratch 32k training sits outside the regime every published result was measured in, and 8k pretraining followed by 32k adaptation also cuts the quadratic penalty for most of the run, roughly halving wall-clock (T4).

## Source Tensions

Five disputes in the source base bear directly on the argument above. Each is stated with both sides, and each resolves to a position that changes a number rather than a conclusion.

**Width is not speed.** One side holds that PCIe width and generation are independent axes that must never be conflated: the capacitor mod changes width only, the unlocker changes generation only, and a card with neither runs Gen1 x4 [[physical-modsmd]], [[pcie-subsystemmd]] (T2). The other side states "PCIe Gen2 x4" as a single fact and derives every bandwidth figure from it. The premise is under-specified rather than wrong: the four states span roughly 8× in one-way bandwidth, and the plan's 0.75 GiB/s is plausible as a busbw reading and implausible as an algbw reading.

**Effect sizes are reported in units that have no established noise floor.** One side holds that the memory-axis deltas are large: Memory Grafting moves the 0.92B average from 45.62 and 45.03 to 46.98, and the 2.8B average from 51.95/52.43 to 53.86 [[memory-grafting-scaling-language-model]] (T3). The other holds that published architecture effects are 0.014–0.082 nats against a 3.5%-relative single-run floor, so a 9B-token null is the expected result [[conditional-memory-via-scalable-lookup]], [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]], [[a-hitchhikers-guide-to-scaling-law-estimation]] (T3). The two sides are not commensurable: score points on a ten-task average are not nats of training loss. The resolution is that no published Engram or CSA2 delta is reported in a unit with an established noise floor, and the floor is now measured at ≤0.004 nats — so the falsifier is one experiment reporting the delta in nats with two seeds.

**Every sparse-attention 32k win is an adaptation-stage result.** One side holds that sparse attention delivers large measured 32k wins: NSA reports 6.3× forward and 3.4× backward at 32k on A100 with a quality win, and LongCat's LSA claims parity with full attention and native training to 1M tokens [[native-sparse-attention-hardware-aligned-and-natively]], [[longcat-sparse-attention-taming-the-lightning-via]] (T3/T2). The other side plans from-scratch 32k training as the primary axis. The plan sits outside the regime every published result was measured in: NSA spent 270B tokens at 8k before touching 32k, and LongCat's 0.83×-at-32K figure is a latency measurement, not a from-scratch efficiency claim.

**P2P has two incompatible data-movement figures and the larger one is unverified.** One side holds that BAR1 P2P works and moves data at 1.68 GB/s each way at Gen2 x4 on 12/12 pairs [[github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2]] (T2). The other holds that P2P is absent by default and that the one patch that could enable it half-works — data movement at 6.25 GB/s with synchronisation broken, unverified, on one rig [[p2pmd]] (T2). All three figures are consistent once the link state is separated: 1.68 GB/s is the one-way Gen2 x4 rate, so P2P at x4 removes a traversal but cannot exceed the link; 6.25 GB/s is at x16 and unverified. The decisive fact is the one nobody cites: on a four-card PLX rig at Gen2 x16, host-staged copy already moves large buffers at 6.21–6.30 GB/s, so P2P is 8% slower above 16 MiB and costs 25–44% on TP4 decode [[2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]] (T2).

**The most trustworthy sustained numbers are at 180 W, not the plan's 250 W.** One side holds that the decision-grade benchmarks run against an enforced 180.00 W limit, and that 74.6–78.5 TFLOP/s is a ten-minute `gpu-burn -tc` at that cap with no thermal throttle [[readmemd-2]] (T2). The other holds that the card's stock cap is 250 W and it delivers ~162 TFLOPS bf16, the basis of the ~810 aggregate [[performancemd]], [[lessonsmd]] (T2/T1). The resolution is that 74.6–78.5 is a floor, not a ceiling: it is measured at 180 W enforced, so at 250 W the sustained number should be higher, but nobody has measured it, and the 162 and 164–192 figures are GEMM peaks, not sustained BF16 training at 250 W. The honest denominator is 75–100 TFLOPS/card with the upper end unmeasured, so every wall-clock estimate here carries a 1.3× error bar from that axis alone.

## 7. Verdict: Which Experiments Are Worth Running

Feasibility here means three things at once: the experiment **fits in memory**, it **fits in wall-clock**, and its result **can be measured**.

**(a) DDP is the only viable topology at 1M-token global batches.** True for the frozen-table design, false for the stated 7B/1B MoE axis. At 32k with ≤1B trainable, DDP's overhead is single-digit percent at Gen2 x4 and no alternative beats it by more than the width of its own unmeasured premise. At 7B trainable, DDP needs 112 GB/card and does not fit; ZeRO-2 fits at 33.6 GB with DDP-identical communication, and PP=5 fits at 25.8 GB/card with a 4–14% bubble.

**(b) Expert and tensor parallelism are ruled out by the fabric.** True for EP; for TP the fabric does not rule it out so much as make it lose. EP is ≈1.4× at 32k, not 70×, and is blocked because no training-grade EP library runs without NVLink, GPUDirect P2P or RDMA. TP is ≈2.8× and is dominated rather than blocked: its collectives sit inside the layer where little can hide them, and at 1B trainable it loses to ZeRO-2 and PP=5.

**(c) Frozen Engram tables make a 7B-total / ~1B-trainable model fit DDP on 64 GB cards.** True, at ~43.1 GB/card with fused cross-entropy and ~53.1 GB without — but only on the paper's 3M entries per layer; on the repo's §3 figure of 33M the table is 270 GB and the claim is false. The plan's own table arithmetic also breaks its own claim: 12 GB is one layer, not the total, and the recipient carries two. Freezing is also not a memory necessity — ZeRO-2, Adafactor and 8-bit Adam all fit a fully-trainable 7B on 64 GB. Freezing is correct here because the trained variant loses to MoE at 0.92B, and because training the table costs 9–68% of step time depending on the link.

**(d) CSA2 compressed sparse attention pays off at 32k training context.** **False as stated.** CSA2 is not runnable on sm_80 — every training-capable kernel in the stack is Hopper-or-newer, and the only sm_80 path is an inference-only Triton backend with no backward. The underlying question has a positive answer through fine-block sparse attention, at a measured ~1.71× end-to-end rather than 6.3×, and its sign is decidable before any kernel work by checking the GQA group size.

The claim-by-claim verdict table below is the same four answers with the evidence tier attached to each.

| Claim | Plan's figure | Best available evidence | Tier | Verdict |
|---|---|---|---|---|
| Fabric is "PCIe Gen2 x4" | one fact | four reachable states spanning ~8× | T2 | Under-specified: two independent axes |
| All-reduce 0.75 GiB/s | measured | no receipt; 75–86% of the x4 busbw band; 1.20–1.39× the algbw band | T1 | Plausible as busbw, unreachable as algbw |
| ~810 TFLOPS bf16 aggregate | 162/card | 164.4–192.7 GEMM peak, 202.1 ceiling, 74.6–78.5 sustained at 180 W | T2 | Peak, not a sustained rate |
| DDP overhead 36% at 1M tokens | 36% | 5.3–7.1% at Gen2 x4, 11.4% at Gen1 x4 | T4 | Wrong by ~5×, in the safe direction |
| EP ruled out at ~70× | ~70× | ~1.4× arithmetic; zero training-grade EP libraries | T4/T3 | Right conclusion, wrong arithmetic |
| TP ruled out at ~70× | ~70× | ~2.8× arithmetic (126 GB/GPU/step, 156 s); no training-grade TP path on a no-P2P link | T4/T3 | Wrong by ~25×; dominated, not excluded |
| Frozen tables fit 7B/1B on 64 GB | yes | 43.1 GB with fused CE, 53.1 GB without | T4 | True, but the plan's own figure breaks it |
| CSA2 pays off at 32k | yes | NSA ≈1.71× end-to-end; sign depends on GQA group size | T3 | Not runnable; payoff ~half; sign unmeasured |
| 80–120k tok/s at 32k | yes | 39.6k theoretical max at 100% MFU; 14–39k realistic | T4 | Wrong by 2–3× |
| 9B tokens is a sufficient budget | yes | 0.004-nat measured floor; deltas 0.014–0.082 nats | T3 | Adequate for the upper half, not for a verdict |

The lever ranking inverts the plan's framing:

| Lever | Wall-clock payoff | Tier | Cost |
|---|---:|---|---|
| Sparse attention (NSA/FSA at 32k) | −20% to −30% step time | T3/T4 | kernel integration; GQA-dependent sign |
| 8k pretraining + 32k adaptation | ~−50% wall-clock | T3/T4 | changes the control's regime |
| 24-capacitor x16 mod | ~−6% to −7% wall-clock | T2/T4 | 24 × 0402 solder joints, ~20 min/card |
| ZeRO-2 / PP=5 for the MoE arm | 0% (enables the arm) | T3/T4 | systems work; CPU-hook risk |
| Optimizer swap (Adafactor / 8-bit Adam) | 0% | T3 | memory only; comparability risk |

Four measurements would settle the premises, and none is expensive:

1. **`lspci -vv` per card**, reading `LnkCap` against `LnkSta` for speed, width and the downgraded flag. If the box is Gen2 x16 rather than x4, every communication number improves ~4× and the capacitor mod becomes irrelevant; if it is Gen1 x4, they get ~1.6× worse.
2. **A five-rank `all_reduce_perf`** at 1 MB / 100 MB / 1 GB with both columns printed, which settles the algbw-versus-busbw question in an afternoon.
3. **A sustained BF16 GEMM burn at 250 W** with 1 Hz clock telemetry, which distinguishes whether 74.6–78.5 at 180 W is power-capped or kernel-limited and pins the sustained denominator between 75 and 100 TFLOPS/card.
4. **A GQA group-size check** in the planned config, which decides whether the sparse-attention arm is a win or a loss before any kernel work begins.

A fifth is not about performance at all. ECC is stale by one release: it read as off or N/A through cmpunlocker v0.4, and v0.5 (2026-10-06) reports ECC N/A → Enabled at a measured ≤0.3% integration-level cost, verified only by counters moving and a copy pattern bit-verifying because no fault-injection tooling exists [[readme_enmd]] (T2). Amoghmunikote lists it Working [[github-amoghmunikotecmpunlocker-a-tool-to-unlobotomize-your-nvidia-card-github]], while bayley, dl4rce and the 170th Street spec sheet say it is not restored [[full-specifications-170th-street]], [[cmp-170hx-inference-labdocsunlock-and-modsmd-at-main-dl4rcecmp-170hx-inference-l]] (T2). 

Enable it before the first long run, deploy the five-minute counter snapshot [[readme_enmd]], and run `memtest_vulkan` over the full 64 GB rather than the 8 GiB a mining rig would touch — independent reports put field failure around 50% among buyers, with faults appearing only above that region [[lessonsmd]] (T2).

Those checks gate a program, and the ordering matters more than the runs. If the GQA group size is 1 at 32k, drop the sparse-attention arm before any kernel work — FSA's own result says it loses in that configuration. 

The architecture comparison should run at 8K, not 32k: a four-point ladder of our own design — 50M / 125M / 250M / 500M (T4) — × two architectures, each trained to ~1/3 of budget with the first 10% of checkpoints discarded, fitting L(C) and comparing exponents rather than single losses [[a-hitchhikers-guide-to-scaling-law-estimation]] (T3). At 8K the per-token FLOPs are ~3× lower, so the ladder costs ~12–35 days — the same envelope already planned, and the only design in the evidence that resolves deltas below the noise floor. CSA2/NSA then runs alone at 32k on a small budget with a phase-stratified metric. Matching the literature is not an option: at 20–27k tok/s, 100B tokens per arm is 43–58 days and five arms is 215–290 (T4).

The batch plan carries an unpriced convergence risk. At 1M tokens the repo sits ~17× below the measured critical batch size at plateau (~16.8M tokens), so token efficiency is safe for most of the run — but CBS starts near zero at initialisation, so a 1M-token batch at step 0 is far above it. A fixed 4,096-document batch from step 0 ends with worse loss than the small-batch control despite using 75% fewer steps, while batch-size warmup — start small, double as CBS doubles, scale LR by square root — beats that control by 0.0166 loss with 43% fewer steps [[critical-batch-size-revisited-a-simple-empirical]] (T3). The repo has LR warmup but no batch-size warmup, and the linear LR-scaling rule it may be leaning on is an SGD-on-vision result, not an Adam-on-LLM one [[accurate-large-minibatch-sgd]] (T3).

The data pipeline is the part of the plan that is wrong. Token supply is fine — nine billion tokens at a 75/15/10 mix draws 6.75B from FineWeb-Edu's 1.3T tokens [[huggingfacefwfineweb-edu-datasets-at-hugging-face]], 1.35B from Stack-Edu's ~125B [[huggingfacetbstack-edu-datasets-at-hugging-face]] and 0.9B from FineMath-4+'s 9.6B [[huggingfacetbfinemath-datasets-at-hugging-face-2]] (T2) — but the packing is the worst of four measured configurations. The repo's default is random interleaving across the three sources with document-boundary masking off; that is MIXChunk. Intra-document masking beats it by 0.762 PPL average at 2K (8.410 against 9.172) and 0.986 at 8K (8.079 against 9.065), with the largest gain on code — −1.28 PPL at 2K and −2.49 at 8K on GitHub, the domain the paper finds most damaged by cross-document distraction [[analysing-the-impact-of-sequence-composition]] (T3). Code is 15% of this mix. 

The repo already stores `cu_seqlens` per shard, so the fix is a kernel flag costing a measured 4.0% of step time, not a re-tokenisation. Turn masking on before the first run.

The determinism manifest is the part that must not be skipped either: floating-point arithmetic order — induced by sharding topology, gradient accumulation depth or precision — produces functional divergence within ~17% of changing the data seed (and ~10% of changing the initialization seed) [[quantifying-pretraining-variance-data-initialization-and-floating-point-arithmet]] (T3).

The fabric decides what this box can attempt, not what it costs to attempt it — so design for availability, measure for cost, and stop treating 0.75 GiB/s as the number that governs the program.
