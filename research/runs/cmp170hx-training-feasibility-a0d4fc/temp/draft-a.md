**The answer first.** Four experiments are on the table and three are worth running, but not in the shape the plan gives them. A ~500M dense control at 32k is feasible in roughly **4–5 days of wall-clock, not the 25 hours predicted**. A 7B-total/~1B-active MoE is feasible but **not under plain DDP** — it needs ZeRO-2 at five ranks or a 5-stage pipeline. A frozen-Engram Memory-Grafting arm **does fit DDP on 64 GB cards**, with ~21 GB of headroom, but the frozen table is **twice the size the plan budgets** and the effect it chases **flips sign** between the paper's 0.92B and 2.8B settings. CSA2 at 32k is **not runnable at all** on sm_80 and would be a net step-time *loss* if it were; run NSA or FSA instead. And the premise the whole program is designed around — that the PCIe fabric is the binding constraint — is **wrong**: after correction, DDP overhead at 1M-token batches is **~7%, not 36%**, and the binding constraint is wall-clock at **90–100 TFLOPS/card**.

**The thesis.** The repo's program is directionally right and its quantitative spine is wrong. It reached a defensible conclusion — data-parallel, large batches, frozen tables — while both premises fail and one comparison is confounded. "PCIe Gen2 x4" is a compound of two independently-fixable axes the repo never resolved, and the measured link rate on this exact silicon at this exact width is **1.6 GB/s** (T2), not the 0.75 GiB/s the model divides by. "162 TFLOPS bf16" is a *GEMM peak*, not a sustained training rate; the defensible denominator is **90–100 TFLOPS/card** (T1 user-reported end-to-end, floored by a 74.6–78.5 TFLOPS sustained burn at 180 W, capped by a 202.1 TFLOPS arithmetic ceiling). The FSDP comparison is **1.5×, not 2×**, and it pits FSDP at 7B trainable against DDP at 1B trainable. Every correction is a *measured* number rather than arithmetic, and they move in **opposite directions** — the link correction makes communication 1.6× slower, the compute correction makes the step 7.6× longer, the 32k/4k correction makes the context penalty 2× smaller, and the noise-floor correction makes the experiment 25× better powered. That is why the conclusion survives its own arithmetic being wrong: the errors do not point the same way, so they do not compound into a wrong decision.

**Evidence tiers.** **T1** = measured on this box, *user-reported*, no receipt chain — the corpus contains no on-box training, all-reduce, NCCL or DDP measurement at all, and both [[performancemd]] and [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] say so explicitly. **T2** = community-measured on CMP 170HX silicon (a different rig) or adjacent hardware. **T3** = paper-measured elsewhere. **T4** = arithmetic from stated assumptions. Hierarchy: T1 > T2 > T4, with T3 alongside T2 as external measurement.

## Hardware Reality

### Width and generation are two axes, not one

The repo states "PCIe Gen2 x4, no NVLink, no P2P" as one measured fact and builds every communication number on it [[research.md]] — but "Gen2 x4" is a compound of two physically independent restrictions, and the repo never says which of four link states its measurement was taken under.

The width half is a board-level omission, not a fuse. Twelve of sixteen lanes ship with their AC-coupling capacitors depopulated; the lane fuses are all clear (`OPT_PCIE_LANE_DISABLE` = `0x00000000`, "No lanes fused off"), and the registers show a physical-layer discontinuity — `LnkCap: Width x16` against `LnkSta: Width x4 (downgraded)` [[physical-modsmd]]. Soldering 24 × 0402 220 nF X7R capacitors restores x16 with no software patch, and the page warns: "The capacitor mod changes **link width only**. It never changes PCIe generation. A cap-modded card with no unlocker installed reports **x16 at PCIe 1.0**" [[physical-modsmd]].

The speed half is firmware. Stock trains at Gen1 2.5 GT/s, enforced by OTP fuses plus a signed DevInit table; the cmpunlocker link patch reaches Gen2 (5 GT/s); Gen3 is blocked by `OPT_DISABLE_GEN3_SPEED`, which reads 1 on 4/4 cards checked [[pcie-subsystemmd]] [[club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github]]. The independence holds both ways: "A Gen2-patched unmodded card is Gen2 x4" and "A fully modded unpatched card is Gen1 x16" [[pcie-subsystemmd]].

So the premise is **under-specified rather than wrong**, and that under-specification is the largest source of error in the repo's communication model. The four reachable states are Gen1 x4 (~1.0 GB/s), Gen2 x4 (~1.6 GB/s), Gen1 x16 (~4 GB/s) and Gen2 x16 (~6.6 GB/s). The repo's 0.75 GiB/s could be any of them, and its own headline concedes it: "If the cards are actually Gen2 **x16** rather than x4, every communication cost below drops ~4× and the 'large-batch-only' conclusion weakens a lot. One `lspci -vv` per card settles it" [[research.md]]. That is a conditional premise dressed as a measurement.

### The measured link rate: 1.6 GB/s at Gen2 x4

The number to anchor to is measured on this exact card model at this exact width. A four-card CMP 170HX serving report on PCIe Gen2 x4 states plainly: "the transfer rate is 800MB/s at Gen 1 and 1.6GB/s at Gen 2" [[i-tested-the-cmp170hx]] (T2 — community-measured on this silicon at this width, not on the user's five-card box). That is the hard ceiling any 5-rank ring all-reduce must live under.

Two things follow that the repo's model lacks. First, **NCCL drives about 53% of the raw link** at Gen2 x16 (4.24 GB/s busbw against an 8 GB/s line rate) [[readmemd]] (T2), and width scales linearly — a card retrained at x2 "ran exactly half speed everywhere" [[p2pmd]] (T2). At x4 that implies ~0.85 GB/s busbw and — since busbw = 1.6 × algbw at N=5 by construction — **~0.53 GB/s algbw**. Second, the host-staged path **saturates**: a Gen3 x16 PCIe-only box delivers 4.3 GB/s busbw, only 1.01–1.16× the CMP box's Gen2 x16 figure despite roughly double the line rate [[behind-the-scenes-of-distributed-training-and-why-your-gpu-wiring-matters-as-muc]] (T2). So the capacitor mod captures essentially all available gain, and a hypothetical Gen3 unlock would buy almost nothing — convenient, because Gen3 is fuse-blocked anyway.

### The TFLOPS ladder: one defensible denominator

The corpus holds at least nine per-card compute figures, and reading them as a dispute is the actual error. They are a coherent ladder across three axes — lock state, power cap, duration. The four lowest (39.3, 42, 49.05/50.53 and 6.41 bf16) all measure a locked or power-braked card, and the reconciliation is exact: **50.53 = 202.1 / 4**, so the "theoretical FP16" spec-sheet figures are the *locked* tensor rate, not the silicon's peak [[performancemd]] (T2). The five that matter for a training denominator:

| Figure | Card state | Duration / cap | Tier |
|---|---|---|---|
| **74.6–78.5 TFLOPS** | unlocked, **180 W enforced** | **10-minute sustained burn** | T2 |
| **91–100 TFLOPS/GPU** | unlocked, 250 W | **end-to-end training, ~4k ctx** | **T1 (user-reported)** |
| 162 TFLOPS bf16 | unlocked, 250 W | burst GEMM | T1 (user-reported) |
| 164.4–192.7 TFLOPS bf16 | unlocked, stock clocks | burst GEMM, 7 tools | T2 |
| 202.1 / 213.6 TFLOPS | unlocked | arithmetic ceiling, 70 / 74 SM | T4 |

The 162–193 band is real and it is a **GEMM peak**, at stock clocks under a 202.1 TFLOPS ceiling that is "arithmetic: 2048 x 70 SM x 1410 MHz, verified exact" [[performancemd]] (T2/T4). The repo's 162 is therefore ~80% of ceiling and *conservative as a peak* — which is exactly why it is the wrong number to divide by. The only figure with a full receipt chain is the 180 W one: 77.7 / 76.4 / 74.6 / 78.5 TFLOP/s per card over a 10-minute `gpu-burn -tc`, peak core 63 °C against an 80 °C stop rule, so the shortfall from 202 TFLOPS is **not thermal throttling** [[readmemd-2]] [[2026-09-02-cmp170hx-health-gateipynb]] (T2). The only figure with a training-shaped workload is the repo's own 91–100 TFLOPS/GPU end-to-end (T1). PaLM's Table 3 brackets both: Megatron-Turing NLG on A100 reaches **30.2% MFU**, PaLM on TPU v4 **46.2%** [[palm-scaling-language-modeling-with-pathways]] (T3).

Three independent lines therefore converge on **~90–100 TFLOPS/card sustained at 250 W** — **450–500 TFLOPS aggregate, 45–50% MFU** — not 162 and certainly not 810. The repo's 810 implies 80% MFU for a five-rank DDP job with full activation checkpointing, which no published training run on any hardware achieves. The correction is not that 162 is wrong; it is that **162 is a peak and 90–100 is a sustained rate**, and the repo used the peak as the denominator.

### The measurement nobody has made

The corpus contains **no 5-rank all-reduce at any width**. Every receipt is 2, 3 or 4 ranks [[readmemd]] [[2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]] (T2), and the 3-card receipt admits it: "True BAR1 P2P (community fork patches) was **untested** here… One probe run per pair; no cross-run error bars" [[readmemd]]. Worse, the PCIe literature warns five ranks are pathological: "the NUMA effects appear quite significant with 5 GPUs… One should avoid adopting 5 GPUs in their application setup" [[evaluating-modern-gpu-interconnect-pcie]] (T3). An NCCL maintainer confirms the mechanism — 5- and 7-GPU runs are "much lower than others" because of offset alignment (4/8 bytes instead of 16) [[why-is-the-bandwidth-utilization-of-7-gpus-much-lower-than-that-of-6-8-gpus-for]] (T2) — and it is payload-shape-dependent and paddable away in DDP buckets. So the N=5 penalty is a **CPU-flow/alignment artifact, not a link-bandwidth collapse**, and it is fixable. A 5-rank `all_reduce_perf` at 1 MB / 100 MB / 1 GB with both columns printed is the highest-value missing measurement in the program — one afternoon.

## Training Topologies

### What the repo says, and what the fabric actually forbids

The repo's position is compact: "Parallelism: DDP or FSDP only. EP and TP cost ~70× the compute time on this fabric (two all-to-alls per MoE layer); FSDP costs ~2× DDP. DDP overhead is set by batch size alone" [[goal-and-requirements.md]], with a ranking table putting DDP at 36%, FSDP at 72%, PP at ~2.9× and EP at ~72× [[research.md]].

The conclusion — drop EP and TP — is right. The reasons are wrong in three ways, and the framing hides a topology the repo never considers.

The fabric's real constraint is **per-collective latency, not bandwidth**. On a host-staged link every collective is a GPU→host→GPU round trip, and the round trip costs the same whether the tensor is large or small [[club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]] (T2). That is why TP dies here and DDP does not: TP performs one all-reduce per layer per token — "86 collectives per forward pass on a 43-layer model" [[lessonsmd]] (T2) — while DDP performs **one** all-reduce per optimizer step. The measured consequence is stark: when one card's link retrained from x8 to x1, TP4 fell from 70.5 to 14.4 tok/s under an unchanged protocol while PP4 held 60.8 tok/s on the same box — "about 4.2x more link-tolerant" [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] (T2). On the same box PP4 reached 87.55 tok/s c=1 against TP4's 60.4 [[club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]] (T2).

The fabric **punishes collectives-per-step, not bytes-per-step.** DDP is the one topology that amortizes a fixed latency across a large payload; TP and EP are the two that cannot.

### DDP: the topology the fabric does not punish

DDP's only per-step traffic is the gradient all-reduce, and PyTorch's reducer registers one autograd hook per parameter, so a parameter with `requires_grad=False` has no gradient accumulator, no hook fires, and no bucket includes it [[distributed-data-parallel-pytorch-214-documentation]] (T3). That is the mechanism behind the frozen-table design and it is sound.

The one rule the repo gets exactly right is `no_sync()`. DDP all-reduces once per optimizer step; without `no_sync()` it all-reduces on every micro-step, multiplying communication by the accumulation depth. The repo wraps the first five micro-steps in `model.no_sync()` and all-reduces only on the sixth [[experiments/01-dense-baseline.md]], which is correct and load-bearing.

The strongest independent confirmation comes from the closest analogue in the corpus: four RTX PRO 6000 Blackwell cards on **PCIe 4 with no NVLink and no InfiniBand**, training a Qwen3-style MoE with 500M total / ~330M active at ~100k tokens per micro-batch and `gradient_accumulation=10` — an effective global batch of ~1M tokens, the same regime. Measured there: "gradient communication accounted for 50% of the single-step time, while matrix multiplication took up only about 20%"; ~101,000 micro-batches accumulated **~2.26 hours of gradient sync**; and `no_sync()` so the all-reduce fires only on the 10th step cut wall-clock from under 18 hours to **13.2 hours**, leaving ~0.23 hours of sync — **1.7%** [[a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m]] (T2). On a no-NVLink PCIe box at the same batch regime, DDP's amortized all-reduce cost after `no_sync()` is **under 2% of wall-clock**. The repo's 36% is an order of magnitude off, and there is a measurement on adjacent hardware that says so.

### FSDP: 1.5× not 2×, and the comparison is confounded

The repo's "FSDP costs ~2× DDP" is wrong on the multiplier and wrong on the comparison.

On the multiplier: ZeRO's arithmetic is explicit — "The total volume is therefore 3Ψ which is 1.5x compared to the baseline" [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] (T3) — and FSDP's PVLDB paper states "full sharding has 1.5x communication overhead and volume over DDP if using bandwidth optimal ring algorithm" [[pytorch-fsdp-experiences-on-scaling-fully-sharded-data-parallel-pvldb-16]] (T3). The 2× only reproduces if you force fp32 gradient reduction via `MixedPrecision(reduce_dtype=torch.float32)`; FSDP's default `reduce_dtype=None` "takes on the param_dtype value, still running gradient reduction in low precision" [[fullyshardeddataparallel-pytorch-213-documentation]] (T3). With bf16 reduce, FSDP moves 6 B/param against DDP's 4 = **1.5×**; forcing fp32 gives 8 B/param = 2.0×. The repo's 2× is the *pessimistic variant*, not the default.

On the comparison: the repo's table pits FSDP at **7B trainable** (28 GB, 37 s, 72%) against DDP at **1B trainable** (2 GB, 2.7 s, 36%). At equal 7B trainable, DDP is 28 GB/18.7 s and FSDP is 42 GB/28.0 s — 1.5×, not 2×. The FSDP row also uses DDP's 2Ψ volume at the algorithmic collective rate, overstating the time by 1.33×. The correct FSDP overhead at 7B dense is **54%, not 72%**.

### PP: the topology whose cost does not scale with parameters

This is the topology the repo's framing hides, and it is the one that makes the goal document's own MoE axis feasible.

PP's communication volume is `O(tokens × hidden × stages)`, not `O(parameters)`. DDP and FSDP move bytes proportional to the parameter count; PP moves one hidden-state hand-off per stage boundary. For a 5-stage pipeline at 32k tokens and hidden 2048, that is 32,768 × 2048 × 2 B = **134 MB per hand-off** (T4), and 1F1B overlaps each hand-off with the previous micro-batch's compute rather than serializing it. Seq1F1B's sequence-level schedule removes the per-micro-batch bubble entirely by splitting the sequence rather than the batch [[seq1f1b-efficient-sequence-level-pipeline-parallelism-for]] (T3). The repo models the traffic as *additive* — precisely what 1F1B exists to avoid — and gets ~2.9×; the correct model is `max(compute, comm)` per stage, and the residual cost is the pipeline bubble, `(p−1)/m` = (5−1)/32 = **12.5%** for 1F1B, ~4.2% for zero-bubble schedules, and under 4.5% measured for RoundPipe with asynchronous optimizer updates [[efficient-training-on-multiple-consumer-gpus-with]] (T2).

PP is also the one topology with a published no-P2P demonstration: RoundPipe "achieves highly competitive performance without utilizing any GPU peer-to-peer communication (NVLink), relying entirely on PCIe host-to-device transfers" and still reaches "over 76% of the throughput of existing SOTA frameworks running on the A800" [[efficient-training-on-multiple-consumer-gpus-with]] (T2). Its memory profile is what makes the 7B-trainable MoE arm possible at all: 7B/5 stages = 1.4B parameters per stage × 16 B/param = **22.4 GB** of weights+optimizer, plus ~3.4 GB of in-flight activations ≈ **25.8 GB/card** (T4) — 64 GB with 38 GB of headroom.

The caveat is real. At 1.6 GB/s per direction the 134 MB hand-off takes ~84 ms inside a ~9.2 s compute window (T4) — ~1% of it, comfortably hidden; below ~1.1 GB/s it stops being hidden. And PP requires static shapes: `torch.distributed.pipelining` "needs to know the input and output shapes for the stage model… The shapes must be static" [[pipeline-parallelism-pytorch-214-documentation]] (T3), a genuine constraint for a model with MoE routing, Engram lookup and CSA2 top-k.

### TP and EP: out, but for latency and library reasons, not the repo's 70×

**TP** is out because its traffic is 2 all-reduces per layer × 24 layers × 1M tokens × 2048 hidden × 2 B = 196.6 GB of tensor volume, which on a 5-way ring is 1.6× that on the wire = **314.6 GB per GPU per step** (T4). At 0.53 GB/s algbw that is ~594 s against a 56 s compute window — an order of magnitude. Out.

**EP** is out for a stronger reason than arithmetic, and the repo's 72× is an order of magnitude too high. The error is mundane: it divides the **aggregate** dispatch+combine volume (~400 GB) by the **per-GPU** collective rate (0.75 GB/s). The correct per-GPU volume, from Megatron-Core's `T·K·h·(EP−1)/EP` per all-to-all [[scalable-training-of-mixture-of-experts-models-with]] (T3), is ~2.6 GB per layer × 24 = **62.9 GB per GPU** (T4) — ~6× compute at the repo's denominator, not 72×, and ~2× at the corrected one.

But bandwidth arithmetic is the wrong model for EP anyway, because its traffic is **48 small collectives per step** (two per MoE layer × 24 layers), and small collectives here are latency-bound. The decisive barrier is software: "Communication libraries specialized for MoE EP, including DeepEP, pplx-kernels, NIXL-EP, NCCL EP, and UCCL-EP, similarly rely on NVLink, GPUDirect P2P, or RDMA for their optimized GPU-initiated paths" and "existing EP communication libraries cannot operate on these systems" [[efficient-expert-parallel-communication]] (T3). The one prototype that works on PCIe without P2P — ThunderEP, at 2.00× dispatch and 1.53× combine over NCCL on 6×4090 and 6×5090 — is **inference-only, with no gradient collectives** [[efficient-expert-parallel-communication]] (T3). So EP is ruled out because **no EP library runs on this fabric at all**, and the one prototype that does is not a training system. The 72× should be relabelled an estimate at ~6×, and the argument rebuilt on the library barrier — which is stronger, not weaker.

### Communication-cost comparison, corrected

All rows use the repo's geometry (7B total / 1B active MoE, 24 layers, hidden 2048, top-k 2, 1M-token global batch, 5 GPUs) and one measured wire rate: 1.6 GB/s link, ~0.53 GB/s algbw at N=5 (T2/T4). Compute window at 32k is 56.3 s at 450 TFLOPS (T4).

| Topology | Per-GPU wire volume per step | Time @ measured rate | Overhead vs 56.3 s | Repo's figure |
|---|---:|---:|---:|---:|
| DDP, dense 1B trainable | 2Ψb = 2.0 GB | 3.8 s | **6.7%** (13.4% fp32 buckets) | 36% ✗ |
| DDP, MoE 7B trainable / 1B active | 2Ψb = 14 GB | 26.4 s | **47%** | 36% ✗ |
| FSDP, dense 7B trainable | 3Ψb = 42 GB | 79 s | **140%** | 72% ✗ |
| PP, 5 stages, 7B/1B | 32.8 GB per intermediate GPU, concurrent | hidden in compute | **4–14%** (bubble-bound) | ~290% ✗ |

The DDP row is the one that matters, and it is the subject of the next section.

## DDP Overhead Model

### The joint re-derivation

The repo's model is a single ratio: communication time over compute time. Its 36% comes from dividing a 2.0 GB gradient payload by 0.75 GiB/s to get 2.7 s, and dividing 6 FLOPs/param/token × 1B params × 1M tokens by 810 TFLOPS to get 7.4 s [[research.md]]. Both halves are wrong, in **opposite directions** — which is why the conclusion survives.

**The numerator correction (communication is slower than assumed).** 0.75 GiB/s is 0.805 GB/s. At N=5, busbw = 1.6 × algbw by construction, and busbw is *defined* as the per-rank link rate. If 0.805 GB/s is a busbw reading — the column `all_reduce_perf` exists to produce, and what a user quoting "all-reduce bandwidth" almost always quotes — then algbw is **0.503 GB/s**, and the payload takes 3.98 s, not 2.7 s: a **1.48× increase** in the numerator. Physics corroborates the reading: as algbw, 0.805 GB/s would require NCCL to hit 77% of the one-way link rate on a fabric where every byte crosses PCIe twice, and the best efficiency ever measured on this hardware class is 64% [[readmemd]] (T2). The measured anchor — 1.6 GB/s link, ~53% NCCL efficiency, ~0.85 GB/s busbw, ~0.53 GB/s algbw — lands within 5% of the busbw reading (T2/T4).

**The denominator correction (compute is much longer than assumed).** Two errors compound. First, 6 FLOPs/param/token is the *non-attention* half of the canonical formula. PaLM states it explicitly: "an N-parameter decoder-only model requires 6N matmul FLOPs per token seen… The matmuls in dense self-attention add 6LH(2QT) FLOPs per token… the model FLOPs utilization is the ratio of the achieved throughput in tokens per second to the theoretical peak **R = P (6N + 12LHQT)**" [[palm-scaling-language-modeling-with-pathways]] (T3). For a 1B-active / 24-layer / hidden-2048 model at 32k, the attention term is 12 × 24 × 2048 × 32,768 = **1.933e10 FLOPs/token** against a linear term of 6.0e9 — attention is **76% of the step** and total FLOPs are **4.2×** the 6ND estimate (T4). For the repo's own 500M baseline at 28 layers and hidden 2048, attention is **88%** and total FLOPs are **8.5×** (T4). Megatron-Core corroborates the magnitude from measurement: "at 64K tokens, SDPA consumes **69% of FLOPs**, compared to only 10–15% in short-sequence scenarios" [[scalable-training-of-mixture-of-experts-models-with]] (T3). Second, the denominator 810 TFLOPS is a peak, not a sustained rate; the defensible figure is 450–500 TFLOPS.

Put together: 2.533e16 FLOPs per 1M-token step ÷ 450 TFLOPS = **56.3 s** of compute, not 7.4 s — a **7.6× increase**.

**Net effect: 36% → 6.7%.** The numerator grows 1.48×, the denominator grows 7.6×, and the ratio falls **5.4×**. With fp32 gradient buckets (4.0 GB payload) the overhead is 13.4%. Both are single-digit-to-low-teens.

### The corrected number is independently corroborated

The adjacent-hardware measurement at the same batch regime gives **1.7%** amortized after `no_sync()` [[a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m]] (T2). My derived 6.7% is the conservative figure — it uses the measured link rate, the full attention FLOPs, and a 45% MFU — and it sits within a factor of four of a real measurement on a real no-NVLink PCIe box. The repo's 36% sits a factor of five above the derived number and a factor of twenty above the measured one.

### Where the invariance claim fails: MoE, by exactly the total/active ratio

The repo's headline finding #3 — "DDP's overhead depends only on batch size, not model size" [[research.md]] — is an **identity for dense models**. Communication time is `2·N·b / B_wire`; compute time is `6·N·T / F_agg`; the ratio is `2·b·F_agg / (6·T·B_wire)`, independent of `N`. At 1M tokens and the corrected denominators that is 6.7%, and it stays 6.7% whether the model is 500M or 7B — until it stops fitting. The regime is also far below the critical batch size, so the large-batch choice costs no sample efficiency: the empirical critical-batch-size literature puts the knee well above 1M tokens for models at this scale [[critical-batch-size-revisited-a-simple-empirical]] (T3).

It **fails for MoE**, and the failure is exact and mechanical rather than a matter of degree. Every expert weight is an `nn.Parameter` with `requires_grad=True`, so DDP's reducer buckets and all-reduces the **full** parameter set regardless of which experts saw tokens; an expert that received zero tokens contributes a zero gradient but still occupies its slice of the all-reduce buffer. Megatron-Core states both halves in one paragraph: "All E experts' parameters, gradients, and optimizer states must reside in memory" and "Per-token FLOPs scale only with N_active (∝K), not N_total (∝E), leaving insufficient computation to overlap with the growing communication… **This is not a matter of degree; it is a qualitative difference from dense models**" [[scalable-training-of-mixture-of-experts-models-with]] (T3).

The corrected formula is `overhead ≈ 6.7% × (N_total / N_active)`. For the repo's own 7B/1B axis that is **47%** (T4); at the repo's own numbers it was 252%. So the invariance claim is true only for the **frozen-table design where trainable = active**, and false for the goal document's "sparse MoE up to 7B total / ~1B active", whose 7B experts are trainable by construction because routing is learned.

That distinction is the sharpest thing the repo gets right by accident. Its headline says "DDP's overhead depends only on batch size" and then its design freezes the tables so that trainable equals active — precisely the special case where the invariance holds. Stated properly: **freezing the tables is what makes DDP's overhead invariant, not a property of DDP.**

### What the overhead model was never the constraint on

The repo's framing is that the fabric is the thing to design around, so the design goal is "make the trainable set fit on one card" [[research.md]]. After correction the fabric constraint on DDP is **nearly gone** — 6.7% at 1M tokens, 1.7% measured on adjacent hardware — and the binding constraint is **sustained compute throughput × token budget**.

The consequence is a lever inversion, and it is the most actionable finding here. The 24-capacitor x16 mod is the only order-of-magnitude fabric lever: it takes busbw from ~0.93–1.07 to 3.69–4.24 GB/s, a 3.9–4.0× step [[physical-modsmd]] [[readmemd]] (T2). Under the repo's framing that takes DDP overhead from 54% to 11%. Under the corrected framing it takes **6.7% to 1.7%** — a 5-percentage-point saving on a 56-second step, i.e. ~2.8 s per step, or roughly **6–7% of wall-clock** over a 9,000-step run (T4). That is not nothing, but it is a small lever: sparse attention, at −20% to −30% of step time, is **three to four times larger**. The repo's central premise — that the fabric is the thing to design around — is **inverted**: the thing to design around is 90–100 TFLOPS/card and the token budget.

## MoE & EP

### The 70× error, and why correcting it strengthens the case

The repo's "EP does two all-to-alls per MoE layer; at 1M-token batches that is ~70× the compute time" [[research.md]] is an aggregate-vs-per-GPU error. It divides the aggregate dispatch+combine volume (~400 GB across all ranks) by the per-GPU collective rate (0.75 GB/s). The per-GPU volume is ~62.9 GB (T4), which at the repo's denominator is ~6× and at the corrected denominator ~2×.

Correcting the number **strengthens** the conclusion rather than weakening it, because the bandwidth arithmetic was never the right argument. EP's traffic is 48 small collectives per step, and on a host-staged fabric each one pays a fixed round trip regardless of size [[club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]] (T2). The real barrier is that **no EP library runs without NVLink, GPUDirect P2P or RDMA** [[efficient-expert-parallel-communication]] (T3), and the one prototype that does work on PCIe is inference-only. So EP is out — but the repo should say "no library exists" rather than "70× the compute", because the first is a hard blocker and the second is an arithmetic estimate a reader can check and find wrong.

### The MoE arm is not DDP-viable, and that is the biggest correction to the plan

The repo's goal document asks for "sparse MoE up to 7B total / ~1B active" while its research notes conclude "the design goal is *make the trainable set fit on one card*" [[research.md]]. Those statements are inconsistent, and the inconsistency is the single largest correction this draft makes to the plan.

A 7B-total / 1B-active MoE has **7B trainable parameters** — the experts are learned. Under plain DDP + Adam that is 7B × 16 B/param = **112 GB per card** (T4), which does not fit 64 GB. So DDP is not merely suboptimal for the MoE arm; it is **not viable at all**. The repo's headline #4 says exactly this about a fully-trainable 7B model — "7B fully trainable with Adam needs ~112 GB per card — it does not fit DDP on 64 GB" [[research.md]] — but then applies the fix (freezing) to the *Engram* tables, which is a different model. The goal document's MoE arm has no frozen tables to freeze.

The MoE arm therefore needs one of two things:

- **ZeRO-2 at five ranks.** 4.8 B/param × 7B = **33.6 GB** of model state, plus ~3.2 GB of checkpointed activations and ~0.6 GB of fused-CE logits ≈ **37.4 GB/card** (T4). ZeRO-2's communication is "exactly 2Ψ" — the same byte volume as DDP's all-reduce [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] (T3) — so the 6.7% overhead figure carries over unchanged, and Adam is preserved, so comparability with the control is preserved.
- **PP=5.** 22.4 GB of weights+optimizer per stage plus ~3.4 GB of in-flight activations ≈ **25.8 GB/card** (T4), with a 4–14% bubble.

The evidence is genuinely split. On the closest published no-P2P analogue — 10× RTX 3090 with `topo -p2p r` = CNS and ~3 GB/s of shared PCIe — ZeRO-2 beats DDP at **every** GPU count: 4 GPUs 36.3 vs 28.6 samples/s (+27%), 8 GPUs 59.4 vs 39.6 (+50%), 10 GPUs 52.6 vs 47.9 (+10%) [[pcie-110-gpu-스케일링]] (T2). But on 8×A100 with NVSwitch, ZeRO-2 runs at 28.7 against DDP's 83.1 — a 2.90× loss — and the mechanism is **not bandwidth**: every communication-overlap knob produced zero improvement (3,071 → 3,093 ms), and the ~2.4 s delta is CPU-side Python execution of per-parameter hooks [[perf-29-throughput-by-switching-from-deepspeed-to-ddp-issue-177-tensorautoopen-2]] (T2). That overhead is fabric-independent, would bite here too, and is worst on many-small-tensor models — exactly MoE experts.

So the recommendation is **ZeRO-2 for the MoE arm, with a measured step-time A/B against PP=5 before committing**: ZeRO-2's byte volume is DDP-identical, but its CPU-side per-parameter overhead is unmeasured on this stack and is the known failure mode for many-small-tensor models.

## Engram & Frozen Memory

### The 112 GB arithmetic is right and irrelevant

The repo's memory claim is arithmetically correct for the one configuration it tests: mixed-precision Adam costs 16 B per trainable parameter (2 bf16 weight + 2 bf16 grad + 12 fp32 master and two moments), so 7B × 16 B = 112 GB, which does not fit 64 GB [[research.md]] (T4).

But it tests only one configuration, and **7B fully trainable fits 64 GB by four independent routes**:

| Route | Bytes/trainable param | 7B model state | Total (FLCE) | Fits 64 GB? |
|---|---:|---:|---:|---|
| DDP + Adam (fp32 master + m + v) | 16 | 112 GB | 115.8 GB | **No** |
| DDP + 8-bit Adam (bf16 w+g, 2 B state) | 6 | 42 GB | 45.8 GB | Yes |
| DDP + Adafactor (bf16 w+g, factored state) | 4 | 28 GB | 31.8 GB | Yes |
| ZeRO-2, 5 ranks (Adam, fp32 master) | 4.8 | 33.6 GB | 37.4 GB | Yes |

Both optimizer routes are validated. Adafactor replaces Adam's full per-parameter second moment with "only the per-row and per-column sums of these moving averages", giving state proportional to `n + m` rather than `nm` [[adafactor-adaptive-learning-rates-with-sublinear-memory-cost]] (T3), and Adafactor-with-momentum is "comparable to Adam" in both optimal loss and learning-rate stability across 150M–1.2B [[published-as-a-conference-paper-at-iclr-2025]] (T3). 8-bit Adam uses block-wise dynamic quantization and matches 32-bit performance at 1.5B and 355M with no hyperparameter changes [[published-as-a-conference-paper-at-iclr-2022]] (T3); naive linear quantization of the second moment "failed to converge smoothly throughout training" [[exploring-quantization-for-efficient-pre-training-of-transformer]] (T3), so block-wise quantization is load-bearing. So **freezing is an architecture choice, not a memory necessity**, and the repo's headline #4 ("~112 GB/card … freezing is required") is false as a general claim. The memory-axis literature agrees that the binding constraint is retrieval cost, not parameter count: UltraMemV2 scales memory networks to 120B parameters by making the lookup sparse rather than by shrinking the table [[ultramemv2-memory-networks-scaling-to-120b]] (T3).

### The frozen table is 12.29B params / 24.58 GB, not 6B / 12 GB

The repo's §3 says the frozen Memory-Grafting table is "33M entries per grafted layer (top 11M n-grams per order)" at "roughly 12 GB of frozen memory per layer", and its §5 budgets "6B frozen" = 12 GB total [[research.md]]. Both numbers are wrong, and they are wrong in opposite directions.

The paper is unambiguous: "For each n-gram order, we select the top 1M most frequent n-grams and compute their frozen memory representations offline, yielding **3M entries per grafted layer**. Each entry stores a hidden-state vector of the grafting model in bfloat16; for Qwen3.5-35B-A3B, this corresponds to roughly **12 GB of frozen memory per layer**" [[memory-grafting-scaling-language-model]] (T3). The Figure 4 caption confirms the total: "0.92B trainable parameters with approximately **12B frozen grafting memory** offloaded."

The arithmetic: 3M × 2,048 dims × 2 B = **12.29 GB/layer**; two grafted layers (donor layers 8 and 24) = **24.58 GB = 12.29B parameters** (T4). The repo's "33M entries per layer" would imply 33M × 2,048 × 2 B = **135.3 GB/layer** — an 11× inconsistency with its own 12 GB figure. And its "6B frozen = 12 GB total" is exactly **one** grafted layer of the paper's table, i.e. half the paper's configuration.

### The logits term the repo omits

The repo's activation estimate — "32,768 tokens × 2048 hidden × 2 B × 24 layers ≈ 3.2 GB" [[research.md]] — counts only checkpointed layer inputs and silently assumes a fused cross-entropy. At 32k × 151,669 vocabulary × 2 B, the LM head produces **9.94 GB bf16** (19.88 GB fp32) if materialized (T4). Liger Kernel states the same defect at a smaller scale: "single GPU training with a batch size of 8 and sequence length of 4096, the 256k vocabulary size will result in a 16.8 GB logit tensor of precision bfloat16" [[liger-kernel-efficient-triton-kernels-for-llm-training]] (T3). Its fix, FusedLinearCrossEntropy, "never materializes the full [B*T, V] logits tensor" and cuts the term to ~0.62 GB at this shape (T4). So the repo's 3.2 GB is valid **only with fused CE** — standard in 2024–25 stacks but an unstated assumption. Without it the budget grows by 10 GB, the difference between a comfortable fit and a tight one.

### Does it fit DDP on 64 GB? Yes — with the corrected arithmetic

| Component | Repo's budget | Corrected | Tier |
|---|---:|---:|---|
| Frozen table (12.29B params, bf16) | 12 GB (6B) | **24.58 GB** | T3/T4 |
| Trainable recipient (0.92B × 16 B) | 16 GB (1B) | **14.72 GB** | T4 |
| Checkpointed activations at 32k, 24L, d=2048 | 3.2 GB | 3.22 GB | T4 |
| LM-head logits (FLCE) | omitted | **0.62 GB** (9.94 GB naive) | T3/T4 |
| **Total** | **~31 GB** | **~43.1 GB** | T4 |
| Headroom on 64 GB | ~33 GB | **~21 GB** | T4 |

So **yes, the frozen-table design fits DDP on 64 GB** — with ~21 GB of headroom, not the ~33 GB the repo claims. The conclusion survives; the arithmetic is wrong by ~12 GB in one direction (the table is twice as big) and ~10 GB in the other (the logits term is omitted). And DDP's all-reduce payload is only the 0.92B trainable gradient = 1.84 GB, which at 0.53 GB/s algbw is 3.5 s against a 56 s compute window — **6.2%** (T4). The frozen parameters contribute zero bytes because they have no gradient accumulator and therefore no reducer hook [[distributed-data-parallel-pytorch-214-documentation]] (T3).

### The real reason to freeze is bandwidth and FLOPs, not VRAM

If the 6B table is trained rather than frozen, its gradients must move. Under DDP the gradient tensor is dense (12 GB at bf16), so the all-reduce is 12 GB/step; at ~0.53 GB/s that is ~23 s against a 56 s compute window — a **41% step-time penalty** (T4). Under the Engram recipe the traffic is worse: "sharding the tables across available GPUs. An All-to-All communication primitive is used to gather active rows in the forward pass and dispatch gradients in the backward pass" [[conditional-memory-via-scalable-lookup]] (T3), which at ~10 KB/token × 1M tokens is 20 GB/step, ~38 s, a **67% penalty** (T4). Freezing removes both.

The other half is FLOPs. A fully-trainable 7B dense model at 450 TFLOPS runs 6 × 7e9 = 4.2e10 FLOPs/token → 10.7k tok/s aggregate; 9B tokens takes ~234 h (~10 days) before overhead (T4). A 7B-total / 1B-active MoE runs 7× faster (~75k tok/s → ~33 h) (T4). So the design driver is **tokens/hour**, not GB/card. The repo's headline #4 ("a memory problem, not a bandwidth problem") has it exactly backwards: the memory fits either way; the bandwidth and FLOPs are what force the architecture.

### The quality premise: the sign flips between 0.92B and 2.8B

The repo's §3 says "0.92B scale: all grafting variants beat both baselines" [[research.md]]. That is technically true and omits the negative result that matters. The paper's own table: at 0.92B trainable / 0.29B active, 50B tokens — MoE 45.62, **vanilla Engram 45.03**, MG-GLM 45.83, MG-DeepSeek 46.42, MG-Qwen3.5 46.98 [[memory-grafting-scaling-language-model]] (T3). Vanilla Engram **loses** to MoE at this scale. At 2.8B / 0.55B active, 100B tokens — MoE 51.95, **vanilla Engram 52.43**, MG 53.86. The sign flips.

So the same architecture axis is −0.59 points at 0.92B and +0.48 points at 2.8B. That is the strongest evidence that a 9B-token run at one size cannot tell you the direction of the memory-axis effect — and it is the reason the frozen variant is defensible: at ≤1B the *trained* Engram loses to MoE while frozen grafting wins, so freezing is not just cheaper, it is the only variant that wins at this box's scale. The repo's allocation is also outside both published laws: Engram's U-shaped optimum puts "roughly 20%–25% of the sparse parameter budget" into Engram [[conditional-memory-via-scalable-lookup]] (T3), and "training efficiency began to decline once the memory module accounts for more than 70% of total parameters" [[memory-grafting-scaling-language-model]] (T3) — the repo's 86%-frozen allocation is outside that range threefold.

### The experiment is better powered than the pessimistic reading suggests

The earlier reading was that a 9B-token A/B is hopelessly underpowered — every published delta is 0.014–0.082 nats against a 3.5%-relative single-run noise floor. That reading is wrong, and the correction is a measured number.

A 23-run variance decomposition at 184M parameters / 3.7B tokens (bf16, bitwise determinism verified across data centres) reports: "At the final step, the 23 runs span the range [2.941, 2.953]—a maximum deviation of **0.4%** from the overall mean of 2.946. Within-group standard deviations are **≤0.004** for every category; FP arithmetic is even tighter at 0.001" [[quantifying-pretraining-variance-data-initialization-and-floating-point-arithmet]] (T3). That is **≤0.004 nats = ≤0.14% relative** — roughly **25× below** the 3.5% floor the pessimistic reading borrowed from the scaling-law literature. A second measurement lands in the same range: Llama-124M on FineWeb, changing only the initialization or data-order seed, varies by 0.011–0.016 nats [[effective-learning-rate-governs-loss-dynamics-in-language-model-pretraining]] (T3).

Against a 0.004-nat floor, the memory axis's deltas are **at or above** it: Engram's allocation law buys 0.0139 nats (1.7248 → 1.7109) at 993M active / ~100B tokens [[conditional-memory-via-scalable-lookup]] (T3), 3.5× the tighter floor; OverEncoding's headline at 500B tokens is 0.082 nats on OLMoE-1.3B [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] (T3), 20× above it. So a 9B-token A/B at 500M **can resolve the upper half of the expected effect range**, and a null would be an informative bound (~0.02 nats), not a foregone conclusion.

The caveat is honest: the floor is measured at 184M, not 500M/9B, and seed variance typically shrinks with scale, so the extrapolation favours detectability but is unmeasured at the target scale. The right response is not "run 100B tokens per arm" — at 20–27k tok/s that is 43–58 days per arm (T4) — but "run the 9B A/B, and add a second seed if the delta lands within 2× of the floor."

## CSA2 at 32k

### The memory objection is soft; the architecture objection is hard

The repo's CSA2 verdict is "worth one experiment, but it is the axis most likely to disappoint at 32k on this box" [[research.md]], resting on three claims: the indexer is a second attention in every layer, the 16,384 candidate cap barely binds at 32k, and the 890 B/token figure is an inference number. The verdict is right in direction and wrong in mechanism, understating one blocker while overstating another.

**The memory objection is soft.** The indexer's score tensor is a `[B, S, H_I, T]` FP32 intermediate materialized before the head-sum, which at S=32,768 with V4-Flash dimensions is 64 GiB, with a measured peak of 129.00 GB on a 140 GB H200 [[streamindex-memory-bounded-compressed-sparse]] (T3). The pure-PyTorch cuDNN reference has the same defect with no chunked variant, materializing `(B, S_q, H_q, S_k)` FP32 at 68.7 GB for 32k/H_q=16 [[dsa_referencepy]] (T3/T4). But the *fundamental* requirement is only the head-summed `(B, S_q, S_k)` FP32 tensor: **4.29 GB** at 32k/m=1 (T4). StreamIndex demonstrates the fix on real hardware: 129.00 GB → **0.40 GB** at S=32,768, and 10.3× faster (317 ms → 31 ms) [[streamindex-memory-bounded-compressed-sparse]] (T3). So the indexer's score-tensor memory is a **solved engineering problem**, not a hard blocker. Anyone arguing CSA2 is ruled out by memory is arguing against an implementation detail already fixed and measured.

**The architecture objection is hard, and harder than the repo assumes.** Every *training-capable* DSA kernel in existence is Hopper-or-newer:

| Stack | Forward | Backward | Architectures |
|---|---|---|---|
| cuDNN DSA | SM90/SM100 | SM90/SM100 | Hopper, Blackwell |
| FlashMLA | sm_100a/sm_103a + Ascend 950 | — | Blackwell, Ascend (**Hopper dropped**) |
| TileLang `deepseek_v32` | H800/H200 tuned, wgmma | H800/H200 tuned | Hopper |
| StreamIndex | Triton, tested sm_90a | Triton, tested sm_90a | "Tested on H100 / H200 (Hopper, `sm_90a`)" |
| vLLM `TRITON_MLA_SPARSE` | Triton, tested 8×A800 | **none** | SM80/SM121, **inference only** |

The FlashMLA row is decisive. Its current release states its scope verbatim: "Supported scope: sm_100a / sm_103a and Ascend 950 (dav-3510), DeepSeek-V4.1 only… **Drop support for NVIDIA Hopper GPUs**" [[github-deepseek-aiflashmla-flashmla-efficient-multi-head-latent-attention-kernel]] (T2). So the production sparse-attention forward for CSA2 does not run on Hopper, let alone Ampere. StreamIndex — the one arch-portable full stack — says plainly "Tested on H100 / H200 (Hopper, `sm_90a`)" and defers A100 validation to future work [[github-rightnow-aistreamindex-memory-bounded-compressed-sparse-attention-via-str]] (T2); its Triton attention forward "reaches about 8% of H200 BF16 peak at V4-Pro dimensions" (T3), and at 8% of peak even a 64× FLOP reduction does not produce a usable training kernel.

The only sm_80 path is vLLM's `TRITON_MLA_SPARSE`, which "adds a `TRITON_MLA_SPARSE` backend so DSA sparse-MLA models… can run on GPUs where DeepGEMM and FlashMLA-Sparse are unavailable — SM80 (A100/A800) and SM121" [[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]] (T2). It was tested on 8×A800-80GB PCIe and passes its kernel tests — but it is a **serving backend**: no backward pass, no training loop, no optimizer step, and the measured numbers are decode tok/s.

So the honest formulation is: **CSA2 at 32k on this box is not an experiment, it is a kernel-porting project.** You must write an sm_80 indexer backward *and* an sm_80 sparse-MLA backward, tune both to usable MFU, then validate the distillation loss — before learning anything about whether CSA2 helps at 32k.

### The payoff is half the headline, and its sign is not guaranteed

The repo cites NSA's 6.3× forward / 3.4× backward at 32k as evidence of a large sparse-attention win. Those are **kernel-level** ratios measured on 8×A100 with Triton kernels [[native-sparse-attention-hardware-aligned-and-natively]] (T3), and the dense baseline they are measured against is itself a moving target — FlashAttention-2 already delivers ~2× over FlashAttention-1 on A100, so a sparse kernel's advantage over "full attention" depends on which dense kernel it is compared to [[flashattention-2]] (T3). The end-to-end training payoff is roughly half the headline.

FSA (ICLR 2026) is the measured end-to-end comparison the corpus needed: Llama3-8B, Qwen3-14B and Qwen2.5-32B at 32K and 64K, BF16, on H20 and H200 with NVLink at 450 GB/s. It reports "up to 1.25× and 1.09× on average end-to-end training speedup" over the Triton NSA kernel, and "up to 2.47× and an average of 1.86×" over full attention [[published-as-a-conference-paper-at-iclr-2026]] (T3). Dividing the two averages implies **NSA ≈ 1.71× full attention end-to-end** — not 6.3×. And the fla-org Triton implementation, the reference the NSA paper's efficiency section is built on, reproduces a shrinking advantage: **2.66× forward / 1.76× backward** at T=16,384, 1.35× / 1.06× at T=8,192, and **0.72× / 0.61× at T=4,096 — NSA is slower than flash** [[github-fla-orgnative-sparse-attention-efficient-triton-implementations-for-nativ]] (T2).

Worse, the sign is not guaranteed. FSA reports directly: "vanilla NSA lags behind full attention in many tested cases… when the sequence length is 32K, one GQA group contains one query head, NSA consistently falls short of full attention" [[published-as-a-conference-paper-at-iclr-2026]] (T3). So if the 500M baseline has a GQA group size of 1 at 32k, NSA is a **net loss**; at g≥4 it wins. **That GQA configuration is the decisive unmeasured variable and it is not in the corpus.** It is also free to measure — one line in the model config.

The corrected payoff is therefore **−20% to −30% of step time**, not the −39% to −54% implied by the 6.3×/3.4× kernel ratios (T4, at a 76–88% attention share). That is still the largest single lever available, three to four times the capacitor mod's ~6–7% of wall-clock.

### The repo's causal story is wrong in both directions

Both the repo and the pessimistic reading fixate on the 16,384 candidate cap, and neither is measuring the right quantity. The cap governs the **indexer's** search cost, not the **attention's** density. Attention density at 32k is set by top-k 512 against 32,768 KV = **1.56% → 64× FLOP reduction** (T4), and that is true whether or not the cap binds.

The actual determinant of the 32k payoff is **gather granularity**, and LongCat measures it directly. LSA vs dense MLA per attention layer at 32K: **1.08× forward, 0.79× backward, 0.83× total**, with crossover only at 64K (1.39×) [[longcat-sparse-attention-taming-the-lightning-via]] (T2). Note that 0.83× is a *speedup ratio*, so LSA takes **1/0.83 = 1.20× dense MLA's latency** — a net loss. The mechanism is quantified: each selected token is fetched "via an independent gather of a single latent KV vector (1,152 B in BF16), which spans only 3 cachelines… this gather occupies merely 3 of the ~50 outstanding memory slots… The resulting net effective bandwidth thus plummets to a mere **~4.5% (~1/22) of peak**" [[longcat-sparse-attention-taming-the-lightning-via]] (T2). That is why 64× fewer FLOPs produces a 20% *slowdown*.

So CSA2 — token-granular, DSA-lineage — would be a **+10.7% to +14.7% step-time penalty** at 32k even if it ran (T4, at a 76–88% attention share). That is a 50–69 percentage-point swing between the CSA2 arm and the NSA arm, and it is decided by gather granularity, not by the candidate cap.

### Run NSA or FSA instead

The substitution is not itself a porting project. FSA's official implementation is "well tested with NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)", supports "Training and inference (prefill)", and calls `loss.backward()` through a 65,536+32,768-token batch; it requires `flash-attn==2.6.3` [[github-relaxed-system-labflash-sparse-attention-efficient-implementations-of-nat]] (T2). The fla-org Triton NSA ships full training support via `flame` [[github-fla-orgnative-sparse-attention-efficient-triton-implementations-for-nativ]] (T2). MoBA is the fallback — "current kernel implementations rely on `flash-attn==2.6.3`" [[github-moonshotaimoba-moba-mixture-of-block-attention-for-long-context-llms-gith]] (T2) — but it is compute-neutral at 32k and lossy alone: trailing LM loss at 32K is **1.546 vs 1.464 for full attention**, recovered only by the 90/10 hybrid [[moba-mixture-of-block-attention-for]] (T3).

The recommendation is therefore: **run NSA (or FSA) at 32k with a 90/10 or 95/5 dense-attention hybrid, keep MoBA as the flash-attn-2.6.3 fallback, and drop CSA2 from the program until someone publishes an sm_80 sparse-MLA backward.** And measure the GQA group size first, because it decides the sign.

## Source Tensions

Five disagreements in the source base are load-bearing for the verdict. Each is named with both sides, and each resolves to a position the plan can act on.

**1. Width is not speed.** One side holds that PCIe link width and generation are independent axes that must never be conflated: the capacitor mod changes width only, the unlocker changes generation only, and the fuse table proves the lanes are not fused off (`OPT_PCIE_LANE_DISABLE` = `0x00000000`, `LnkCap` x16 against `LnkSta` x4 downgraded) [[physical-modsmd]] [[pcie-subsystemmd]] (T2). The other side is the repo, which states "PCIe Gen2 x4" as a single fact and derives every bandwidth figure from it [[research.md]]. **Resolution:** the premise is under-specified rather than wrong, and that under-specification is the largest source of error in the communication model. The report must state the link as a measured triple — generation, width, measured GB/s — and require `lspci -vv` as a pre-flight rather than accepting "Gen2 x4" as a premise.

**2. Effect sizes are reported in units that have no established noise floor.** One side: the memory-axis deltas are large — Memory Grafting moves the 0.92B average benchmark score from 45.62 (MoE) and 45.03 (vanilla Engram) to 46.98, and the 2.8B score from 51.95/52.43 to 53.86 [[memory-grafting-scaling-language-model]] (T3). The other side: published architecture effects are 0.014–0.082 nats against a 3.5%-relative single-run floor, so a 9B-token null is the expected result [[moba-mixture-of-block-attention-for]] [[conditional-memory-via-scalable-lookup]] (T3). **Resolution:** the two sides are not commensurable — score points on a ten-task average are not nats of training loss — so the sharper finding is that **no published Engram or CSA2 delta is reported in a unit in which a noise floor has been established**. But the floor itself has now been measured at **≤0.004 nats (0.14% relative)** [[quantifying-pretraining-variance-data-initialization-and-floating-point-arithmet]] (T3), 25× below the 3.5% figure, so the published deltas are at or above the real floor and a 9B A/B can resolve the upper half. The falsifier is one experiment: report the delta in nats, two seeds.

**3. Every sparse-attention 32k win is an adaptation-stage result.** One side: NSA reports 6.3× forward / 3.4× backward at 32k on A100 with a quality win (LongBench 0.469 vs 0.437), and LongCat's LSA claims parity with full attention and native training to 1M tokens [[native-sparse-attention-hardware-aligned-and-natively]] [[longcat-sparse-attention-taming-the-lightning-via]] (T3/T2). The other side: the repo plans **from-scratch 32k training** as the primary axis [[experiments/01-dense-baseline.md]]. **Resolution:** the repo's plan sits outside the regime every published result was measured in. NSA pretrained on "270B tokens of 8k-length texts, followed by continued training and supervised fine-tuning on 32k-length texts with YaRN" [[native-sparse-attention-hardware-aligned-and-natively]] (T3); LongCat's 0.83× at 32K is a latency measurement, not a from-scratch efficiency claim. So the 6.3×/3.4× does not transfer as a from-scratch number, and the recommendation is **8k pretraining followed by 32k adaptation** — which also cuts the quadratic penalty for most of the run and roughly halves wall-clock (T4). The long-context training literature reaches the same conclusion from the other direction: long-context ability is best acquired by staged extension, not by training at final length from initialization [[how-to-train-long-context-language-models-effectively]] (T3).

**4. P2P has two incompatible data-movement figures and the larger one is unverified.** One side: BAR1 P2P works and moves data at 1.68 GB/s each way at Gen2 x4 on 12/12 pairs [[p2pmd]] (T2). The other side: "peer-to-peer is absent on this card by default… One builder has since reported that the patch half-works on GA100: peer data movement runs at 6.25 GB/s while peer synchronisation does not work at all… That report is **unverified**, from one rig with no independent reproduction" [[p2pmd]] (T2), and `torch.cuda.can_device_access_peer(i,j)` returns False for all 56 pairs. **Resolution:** all three figures are consistent once the link state is separated. 1.68 GB/s is the one-way Gen2 x4 link rate, so P2P at x4 removes a traversal but cannot exceed the link — it is not a bandwidth fix. 6.25 GB/s is at x16 and unverified. And the measured NCCL result is more nuanced than either side: BAR1 P2P is **+31% at N=2** (3.69 → 4.83 GB/s busbw) but **−9% at N=3** (4.24 → 3.86 GB/s), and on a real workload GLM-5.3-Flash TP4 decode runs 293 tok/s with P2P on against 396 with it off — a 26% loss [[2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]] (T2). So P2P is a conditional +30% for same-switch small-rank collectives, a pessimization at three or more ranks behind PLX, and irrelevant to DDP, which is one large all-reduce per step.

**5. The most trustworthy sustained numbers are at 180 W, not the repo's 250 W.** One side: the decision-grade benchmarks run against a 180.00 W enforced limit, and the 74.6–78.5 TFLOP/s figure is a 10-minute `gpu-burn -tc` at that cap with peak core 63 °C and no thermal throttle [[readmemd-2]] [[2026-09-02-cmp170hx-health-gateipynb]] (T2). The other side: the card's stock cap is 250 W and it delivers ~162 TFLOPS bf16, which is the basis of the ~810 TFLOPS aggregate [[performancemd]] [[lessonsmd]] [[research.md]] (T2/T1). **Resolution:** the 74.6–78.5 figure is a **floor**, not a ceiling — it is measured at 180 W enforced, so at 250 W the sustained number should be higher, but nobody has measured it, and the 162/164–192 figures are GEMM peaks or FP32 burn-in, not sustained BF16 training at 250 W. The honest training denominator is therefore a range, **75–100 TFLOPS/card**, with the upper end unmeasured.

## Verdict

### Which experiments are feasible, ranked

**1. The dense ~500M control at 32k — feasible, run it, but budget 4–5 days not 25 hours.** DDP across 5 ranks, 1M-token global batches via 6 micro-steps of one 32k sequence per GPU, `no_sync()` on the first five, `bf16_compress_hook` on. Corrected DDP overhead: **3.3% bf16 / 6.6% fp32** (T4). Corrected throughput: **14–39k tok/s, centre ~20–27k** (T4, anchored in the measured 2.19× 32k/4k ratio and the 90–100 TFLOPS/card sustained rate). 9B tokens: **3–7 days, centre ~4–5 days** — the repo's 25 h is 4–7× optimistic. That budget is also far past the compute-optimal frontier (~20 tokens/param), so the run is deliberately over-trained relative to Chinchilla and the extra tokens buy loss, not efficiency [[training-compute-optimal-large-language-models]] (T3). The FLOPs model that produces this band is not 6ND; it is PaLM's `6N + 12LHQT`, and at 32k attention is 88% of the step for this model (T4). The cross-check is that measured wall-clock tracks that formula at roughly constant MFU: SkyLadder's 1B model shows a 2.20× FLOPs ratio from 8K to 32K and a time ratio within ~5% of it [[skyladder-better-and-faster-pretraining-via]] (T3), and DeepSpeed-Ulysses holds per-GPU TFLOPs nearly flat (161.4 → 157.4 → 147.4) as sequence length doubles twice [[deepspeed-ulysses-system-optimizations-for-enabling]] (T3).

There is one more correction the plan needs: **the stated model is arithmetically impossible.** "494M params, 28 layers, hidden 2048, vocab 151,669" is ~1.4–1.7B parameters, not 494M (T4). A physically consistent 494M/28-layer model has hidden ≈1216, which gives a 32k/4k FLOPs ratio of **3.51×** and a measured wall-clock ratio of **2.19×** — the SLMTrainBench figure [[readmemd-faircslmtrainbench-at-main]] (T3). So either the model is 1.4B (and throughput is at the low end of the band) or it is 494M with hidden 1216 (and the FLOPs ratio is 3.51×, not 4.39×). Either way the 80–120k tok/s prediction fails.

**2. The 7B-total / 1B-active MoE — feasible, but not under DDP.** 7B trainable × 16 B/param = 112 GB/card under DDP + Adam (T4): it does not fit. Run it under **ZeRO-2 at five ranks** (33.6 GB state, DDP-identical 2Ψ communication, Adam preserved) or **PP=5** (25.8 GB/card, 4–14% bubble). Do not run it under FSDP: 3Ψ = 42 GB/step against 1B-active compute is a 140% overhead (T4), and FSDP's per-layer all-gather sits on the forward critical path, the worst possible shape for a latency-bound fabric. EP is out — not because of the repo's 72× (which is ~6× once corrected) but because no EP library runs without NVLink, GPUDirect P2P or RDMA [[efficient-expert-parallel-communication]] (T3).

**3. The frozen-Engram Memory-Grafting arm — feasible, and the design is right for reasons the repo doesn't state.** 43.1 GB/card with ~21 GB headroom (T4), DDP overhead 6.2% on the 0.92B trainable gradient (T4). But budget the table at **12.29B params / 24.58 GB**, not 6B/12 GB, and budget the LM-head logits at 0.62 GB with fused CE (9.94 GB without). Freeze because training the table costs 41–67% step time in gradient or all-to-all traffic (T4) and because the trained variant *loses* to MoE at 0.92B while frozen grafting wins [[memory-grafting-scaling-language-model]] (T3) — not because 7B cannot fit, which it can, four different ways.

**4. CSA2 at 32k — not feasible, and the wrong family anyway.** No sm_80 training kernel exists; the only sm_80 path is an inference-only Triton backend with no backward [[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]] (T2). And token-granular sparse attention is a **1.20× slowdown** at 32K on the vendor's own measurement [[longcat-sparse-attention-taming-the-lightning-via]] (T2). Substitute **NSA or FSA**, runnable on Ampere, and expect **−20% to −30% of step time**, not −39% to −54%, with the sign set by the GQA group size at 32k [[published-as-a-conference-paper-at-iclr-2026]] (T3).

### The lever ranking, inverted from the repo's framing

| Lever | Wall-clock payoff | Tier | Cost |
|---|---:|---|---|
| Sparse attention (NSA/FSA at 32k) | **−20% to −30%** step time | T3/T4 | kernel integration; GQA-dependent sign |
| 8k pretraining + 32k adaptation | **~−50%** wall-clock | T3/T4 | changes the control's regime |
| 24-capacitor x16 mod | **~−6% to −7%** wall-clock | T2/T4 | 24 × 0402 solder joints, ~20 min/card |
| ZeRO-2 / PP=5 for the MoE arm | 0% (enables the arm) | T3/T4 | systems work; CPU-hook risk |
| Optimizer swap (Adafactor / 8-bit Adam) | 0% | T3 | memory only; comparability risk |

The ordering is the argument: the repo's framing puts the fabric first, making the capacitor mod the headline lever. After correction the fabric is worth ~6–7% of wall-clock and sparse attention is worth three to four times that. **The thing to design around is 90–100 TFLOPS/card and the token budget, not the link.**

### The four measurements that would settle the premises

None is expensive, and each invalidates a load-bearing number if it comes back the other way.

1. **`lspci -vv` per card.** Records negotiated generation and width. If the box is Gen2 x16 rather than x4, every communication number here improves ~4× and the capacitor mod becomes irrelevant. If it is Gen1 x4, they get ~1.6× worse.
2. **A 5-rank `all_reduce_perf` at 1 MB / 100 MB / 1 GB, both columns printed.** This is the single missing measurement in the program: no 5-rank all-reduce exists at any width in the corpus [[readmemd]] (T2), and the PCIe literature warns five ranks are pathological [[evaluating-modern-gpu-interconnect-pcie]] (T3). It settles the 0.75 GiB/s question in one afternoon.
3. **A sustained BF16 GEMM burn at 250 W with 1 Hz clock telemetry.** This distinguishes whether the 74.6–78.5 at 180 W is power-capped or kernel-limited, and pins the sustained training denominator between 75 and 100 TFLOPS/card [[readmemd-2]] (T2).
4. **A 100-step, 32k, single-GPU smoke test reported as sustained TFLOPS/GPU and MFU, with and without activation checkpointing, plus a measured 32k-vs-4k tok/s ratio.** This is the one number that would confirm or refute the 2.19× per-token cost ratio on this silicon rather than on an A100.

Add a fifth that is not about performance at all: **ECC and `memtest_vulkan`**. ECC is off on these cards and the unlock path is unsupported [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] (T2). Silent corruption would invalidate the run-comparability premise the entire program rests on, and the club's own health gate shows what a fleet-wide OOM cascade does to a box — Xid 31 on two cards, Xid 154 on all four, a required reboot [[readmemd-2]] (T2).

### What to change in the plan

Five changes, in priority order:

1. **Reframe the program around wall-clock, not the fabric.** The overhead model was never the binding constraint. Rewrite the design goal from "make the trainable set fit on one card" to "maximize tokens per second at 90–100 TFLOPS/card", and re-rank the levers accordingly.
2. **Split the MoE arm off DDP.** The 7B-total/1B-active MoE is not DDP-viable. Commit to ZeRO-2 or PP=5, and run a measured step-time A/B between them before committing — ZeRO-2's byte volume is DDP-identical but its CPU-side per-parameter hook overhead is the known failure mode for many-small-tensor models [[perf-29-throughput-by-switching-from-deepspeed-to-ddp-issue-177-tensorautoopen-2]] (T2).
3. **Fix the memory arithmetic in both directions.** The frozen table is 12.29B/24.58 GB, not 6B/12 GB; the logits term is 0.62 GB with FLCE and 9.94 GB without. The design still fits — with 21 GB of headroom, not 33.
4. **Drop CSA2, run NSA/FSA, and measure the GQA group size first.** CSA2 is a porting project with a negative expected payoff; NSA/FSA are runnable on Ampere with a ~1.71× end-to-end payoff whose sign depends on a config value nobody has recorded.
5. **Change the control's regime to 8k-pretrain + 32k-adapt.** Every published sparse-attention result was measured in an adaptation regime, and the measured 2.19× per-token cost ratio means pretraining at 8k for most of the run roughly halves wall-clock (T4). The control should match the regime the treatments are measured in, not the one the plan assumed.

The box is a wall-clock-limited single-node pretraining rig, not a fabric-limited one. The corrections above do not change what it is good for — architecture-axis questions at 0.1–1B activated parameters — but they change what to measure first, and they move the design goal from the link to tokens per second.
