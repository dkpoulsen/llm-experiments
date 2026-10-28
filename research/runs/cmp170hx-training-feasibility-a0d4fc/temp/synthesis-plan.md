# Synthesis plan

## Core thesis (1-2 sentences)

The repo's experiment program is directionally right — data-parallel, 1M-token global batches, frozen Engram tables, no EP or TP — but its quantitative spine rests on two unmeasured premises (Gen2 **x4** and **162 TFLOPS bf16**) and one confounded comparison (FSDP 2× vs DDP). After correction the PCIe fabric is **not** the binding constraint — DDP overhead at 1M-token batches is single-digit percent — and the real constraint is wall-clock at ~90–100 TFLOPS/card. The recommendation survives; the reasoning offered for it does not, and the two errors point in opposite directions, which is why the repo could not tell when it had stopped being right.

## The 7 strongest argumentative beats

1. **The denominator war** — sourced from Draft A. The repo's 36% DDP overhead is wrong in both halves and in opposite directions: the numerator grows 1.48× (0.75 GiB/s is a *busbw* reading) while the denominator grows ~7.6× (PaLM's `6N + 12LHQT` plus a sustained-not-peak TFLOPS figure). Net: ~6.6%. This is the report's spine and must be presented as one joint re-derivation, not two errata.
2. **"DDP only" is really "data-parallel only"** — sourced from Draft A and B. DDP cannot hold a 7B model (112 GB/card), but ZeRO-2 at five ranks needs 33.6 GB of sharded state at DDP-identical 2Ψ communication, and PP=5 fits at ~25.8 GB/card with a 4–14% bubble. Two routes, not one.
3. **The frozen-table design is a collective-elimination design, not a memory trick** — sourced from Draft B, the single best reframe in the three drafts. Engram's own training recipe shards tables and uses All-to-All to gather rows and dispatch gradients — the one collective this fabric cannot run. Freezing removes the collective entirely. That is a fabric argument, and it is the strongest available form of the repo's claim.
4. **The bandwidth case against EP and TP evaporates at 32k** — resolved in `synthesis-conflicts.md`. EP ≈ 1.4× and TP ≈ 2.8× under the corrected compute window, not 70× and not 28×. What remains is availability (no SM80 training-grade EP implementation) and critical-path serialization (TP's collectives sit inside the layer; EP's all-to-all can overlap). The report must argue EP/TP out on availability and scheduling, never on arithmetic.
5. **Sparse attention is the only lever that moves wall-clock, and CSA2 is the wrong vehicle** — sourced from Draft A and C. Attention is 76–88% of step *FLOPs* at 32k, so a 4× attention reduction is worth −39 to −54% of step time — three to four times more than the capacitor mod, which attacks an overhead that is already only ~6.6%. But CSA2 has no SM80 training path (five independent confirmations), and NSA's measured end-to-end payoff is ~1.71×, not 6.3× — and its **sign** depends on the GQA group size at 32k, which nobody has measured.
6. **The measurement problem is a unit problem, not a power problem** — sourced from Draft B and C. Memory Grafting's deltas are score points on a ten-task average; MoBA's are nats of loss; the only measured noise floor is relative loss/spread. No published Engram or CSA2 delta is reported in a unit in which a noise floor has been established, so a 9B-token run cannot detect the effect it is designed to detect — not because the effect is small, but because it has never been expressed in the unit the run will measure.
7. **The four measurements that would settle everything** — sourced from Draft C. `lspci -vv` per card (generation × width × measured GB/s), a 5-rank `all_reduce_perf` with both columns labelled, a sustained BF16 burn at 250 W, and a GQA-group-size check. None exists in the corpus. The report's contribution is the decision tree, not a verdict that pretends to certainty.

## Section structure

Seven literal unnumbered H2 headings, in this order, plus `## Source Tensions`:

1. `## Hardware Reality`
2. `## Training Topologies`
3. `## DDP Overhead Model`
4. `## MoE & EP`
5. `## Engram & Frozen Memory`
6. `## CSA2 at 32k`
7. `## Verdict`
8. `## Source Tensions` (placed after Verdict, before any closing note)

## Per-section commitments

### Hardware Reality
- **Evidence:** `physical-modsmd` (width and generation are independent axes; the 24-capacitor BOM), `pcie-subsystemmd`, `i-tested-the-cmp170hx` (1.6 GB/s at Gen2 x4 on this exact hardware), `readmemd` (3.69–4.24 GB/s at Gen2 x16), `lessonsmd` and `performancemd` (the TFLOPS ladder), `2026-09-02-cmp170hx-health-gateipynb` and `readmemd-2` (74.6–78.5 at 180 W), `club-170hxdocsbenchmarksmd` (the evidence-tier rule).
- **Beat:** the repo states "Gen2 x4" as one fact when it is a compound of two independently-fixable axes. Four link states are possible (Gen1 x4 ~1.0 GB/s, Gen2 x4 1.6 GB/s, Gen2 x16 ~6.7 GB/s, Gen1 x16), and the repo never says which its 0.75 GiB/s came from. Same for TFLOPS: 42/49 stock, 74.6–78.5 at 180 W, 91–100 end-to-end, 162–193 GEMM peak — the defensible training denominator is 75–100, not 162.
- **Tension to engage:** source-tension 1 ("Width is not speed") and source-tension 6 (180 W vs 250 W).

### Training Topologies
- **Evidence:** `zero-memory-optimizations-toward-training-trillion-parameter-models-sc20`, `fullyshardeddataparallel-pytorch-213-documentation`, `pytorch-fsdp-experiences-on-scaling-fully-sharded-data-parallel-pvldb-16`, `physical-modsmd` (PP hand-off 10,240 B/token/hop), `zeropp-unleashing-exceptional-parallelism-efficiency`, `efficient-training-on-multiple-consumer-gpus-with`, `club-170hxdocstopology-and-parallelismmd`.
- **Beat:** "DDP only" → "data-parallel only". FSDP is 1.5× not 2× (3Ψ vs 2Ψ), and the repo's table also confounds 7B-trainable FSDP against 1B-trainable DDP. ZeRO-2 at five ranks fits 7B at 33.6 GB with DDP-identical communication; PP=5 fits at 25.8 GB/card with a 4–14% bubble. TP is the worst option but for a scheduling reason, not a bandwidth one.
- **Tension to engage:** comparisons Tension 2 (the convergence of the two memory loci) and source-tension 5 (P2P's two incompatible figures).

### DDP Overhead Model
- **Evidence:** `all_reducecu` and `performancemd-2` (the busbw formula `2(n−1)/n`), `whats-the-difference-in-allreduce-for-factor-between-the-nccl-and-nccl-test-issu` (Jeaugey: busbw is "theoretical, on a flat hardware topology"), `palm-scaling-language-modeling-with-pathways` (`R = P(6N + 12LHQT)`), `readmemd-faircslmtrainbench-at-main` (2.19× measured), `scalable-training-of-mixture-of-experts-models-with` (Megatron-Core: SDPA is 69% of FLOPs at 64K), `a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m` (1.7% measured on a no-NVLink PCIe 4 box at 1M-token batches), `critical-batch-size-revisited-a-simple-empirical`.
- **Beat:** the joint re-derivation (beat 1). Present as one ratio corrected in two directions. Then the consequence: DDP overhead ~6.6%, so the fabric is not binding. Then the *other* consequence: the same correction destroys the throughput prediction — 80–120k tok/s at 32k is 2–3× above the theoretical maximum, and 9B tokens takes ~3–7 days, not 25 hours.
- **Tension to engage:** comparisons Tension 1 (the denominator war) and source-tension 3 (incompatible units).

### MoE & EP
- **Evidence:** `scalable-training-of-mixture-of-experts-models-with` (the parameter-compute asymmetry; "unoptimized all-to-all can consume up to 60% of total training time" on NVLink-rich DeepSeek-V3), `efficient-expert-parallel-communication` (the closed list of EP libraries requiring NVLink/P2P/RDMA), `variopath-workload-aware-all-to-all-communication-for-pcie-gpu-clusters`, `democratizing-moe-inference-on-commodity-gpus`, `deepseek-v3-technical-report`, `megablocks-efficient-sparse-training-with-mixture-of-experts`.
- **Beat:** beat 4. EP ≈ 1.4× at 32k, not 70×. The bandwidth case collapses; the availability case stands. Then the second, unstated reason to avoid MoE here: MoE spends the scarce resource (memory) to save the abundant one (FLOPs), and it multiplies DDP's communication by the total-to-active ratio (7× for 7B/1B). MoE is the wrong axis for this box.
- **Tension to engage:** the repo's ~70× vs the corrected ~1.4×; and the MoE-vs-frozen-table axis choice.

### Engram & Frozen Memory
- **Evidence:** `memory-grafting-scaling-language-model` (3M entries/layer, ~12 GB/layer, ~12B frozen params, 45.62/45.03→46.98 at 0.92B, 51.95/52.43→53.86 at 2.8B, "training efficiency began to decline once the memory module accounts for more than 70% of total parameters"), `dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis` (the All-to-All training recipe), `conditional-memory-via-scalable-lookup`, `ultramemv2-memory-networks-scaling-to-120b`, `over-tokenized-transformer-vocabulary-is-generally-worth-scaling`, `published-as-a-conference-paper-at-iclr-2022`, `adafactor-adaptive-learning-rates-with-sublinear-memory-cost`, `liger-kernel-efficient-triton-kernels-for-llm-training`.
- **Beat:** beat 3 (collective elimination) plus the memory arithmetic. Frozen table is 24 GB (two layers), not 12 GB; logits are 9.94 GB naive (0.6 GB with fused CE); 1B-trainable lands at ~53 GB, marginal not comfortable. And freezing is an architecture choice, not a necessity — ZeRO-2/Adafactor/8-bit Adam all fit 7B. Then beat 6: the paper's own numbers show trained Engram *loses* to MoE at 0.92B, so frozen grafting is the only variant that wins — and the win is in the wrong unit to be measured at 9B.
- **Tension to engage:** comparisons Tension 3 (right design, undetectable experiment) and source-tension 3 (units).

### CSA2 at 32k
- **Evidence:** `streamindex-memory-bounded-compressed-sparse` (64 GB score tensor at 32k, 129 GB peak on H200, 0.40 GB chunked), `dsa_referencepy` (68.7 GB at 32k/H_q=16), `deepseek-sparse-attention-dsa-nvidia-cudnn` (SM90/SM100), `github-pzhao-engflashmla-github` (Ampere port is decode-only, "poor" performance), `missing-sparse-backward-kernel-issue-107-deepseek-aiflashmla-github`, `cutebwdsm100-add-sparse-mla-deepseek-v4-backward-kernels-2621-dao-ailabflash-att`, `feature-add-an-opt-in-sm80-fallback-for-deepseek-sparse-attention-dsa-on-nvidia`, `published-as-a-conference-paper-at-iclr-2026` (NSA ≈ 1.71×, and slower than full attention at 32K when g=1), `github-fla-orgnative-sparse-attention` (2.66×/1.76× at 16k, slower than flash at 4k), `github-relaxed-system-labflash-sparse-attention` (FSA runs on A100 SXM), `longcat-sparse-attention-taming-the-lightning-via` (0.83× at 32K; indexer is 90% of per-layer latency at 1024K), `native-sparse-attention-hardware-aligned-and-natively` (270B tokens at 8k then 32k adaptation), `moba-mixture-of-block-attention-for`.
- **Beat:** beat 5. CSA2 is a porting project, not an experiment — the memory objection is soft (32× reducible), the architecture objection is hard (no SM80 training path). Run NSA or FSA instead, but with eyes open: the measured payoff is ~1.71× end-to-end, and the sign depends on GQA group size at 32k. Also: every published 32k sparse-attention win is an *adaptation*-stage result (NSA spent 270B tokens at 8k first), so the repo's from-scratch 32k plan is outside the published regime — and 8k-pretrain-then-32k-adapt also cuts the quadratic penalty for most of the run.
- **Tension to engage:** comparisons Tension 4 (sparse attention is the biggest lever, CSA2 is the wrong vehicle) and source-tension 4 (adaptation vs from-scratch).

### Verdict
- **Evidence:** everything above, plus `club-170hxdocsbenchmarksmd` (the evidence-tier rule) and `lessonsmd` (PWRBRK# detection).
- **Beat:** beat 7 plus the inverted framing. Rank the levers explicitly: sparse attention (−20 to −30% step time at the measured payoff) > capacitor mod (attacks an overhead already at ~6.6%) > optimizer swap. Name the four pre-flight measurements. Carry the ECC/memtest_vulkan requirement (dropped from the locus set but binding: ~50% field failure above 8 GiB would invalidate the comparability premise). State the honest open items: no 5-rank all-reduce receipt exists anywhere; the GQA group size at 32k is not in the corpus; both measured noise floors are at 124M–184M, not 500M/9B.
- **Tension to engage:** comparisons Tension 5 (the framing is inverted).

### Source Tensions
- Engage all six from `source-tensions.json`, both sides named, each resolved: width-is-not-speed, the club's evidence standard vs the repo's, incompatible units, adaptation vs from-scratch, P2P's two figures, 180 W vs 250 W.

## Where drafts disagreed

- **EP per-GPU volume:** Draft A 62.9 GB; Draft C 314 GB; Draft B 79 GB. **Commit to 62.9 GB** — Draft C used the global batch where the formula wants per-GPU tokens, which is the exact error it criticises the repo for. Draft B's time arithmetic is internally inconsistent.
- **EP/TP ratios:** Draft C ~7×; Draft B ~6×; Draft A ~6×/~2×. **Commit to EP ≈ 1.4× and TP ≈ 2.8×** under the corrected 32k window. The other readings use the repo's uncorrected 6ND denominator.
- **Frozen-table bytes:** Draft B 12 GB total; Drafts A and C 24 GB. **Commit to 24 GB** — the paper says ~12 GB *per layer* and ~12B frozen params for the 0.92B recipient.
- **The 32k wall-clock penalty:** Draft A 8.5× FLOPs; Draft C 4.2×; measured 2.19×. **Commit to 2.19× as wall-clock and state the FLOPs ratio separately with its convention.** This is the most important calibration in the report.
- **Draft B's structure:** numbered headings and `[[interim-report-*]]` citations. **Use the seven literal unnumbered headings and cite only source notes.**

## Length target

- response_format: `argumentative` (5,000–10,000 words; hard ceiling 10,000, ship gate fails >20% over)
- Pass 1 target: ~8,500 words
- Pass 2 final target: **7,500–8,500 words** (pass 2 must be shorter than pass 1)
