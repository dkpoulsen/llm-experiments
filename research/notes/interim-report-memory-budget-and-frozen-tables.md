---
title: Interim report — memory-budget-and-frozen-tables
id: interim-report-memory-budget-and-frozen-tables
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-memory-budget-and-frozen-tables
created: '2026-10-27T13:45:26.915494Z'
updated: '2026-10-27T19:02:12.658746Z'
status: review
type: interim
tier: practitioner
content_type: review
deprecated: false
summary: Frozen table is 12.29B/24.58 GB not 6B/12 GB (repo §3 has an 11x transcription
  error); 32k logits are 9.94 GB naive but ~0.6 GB with FLCE; 7B fully trainable fits
  64 GB via ZeRO-2 (33.6 GB), Adafactor (28 GB) or 8-bit Adam (42 GB), so freezing
  is a choice justified by scale-specific quality and 1.4-2.2x step-time fabric cost,
  not VRAM; the locus's 80%-of-link offload finding is a 5x aggregate-vs-per-link
  error (~13-16%) and the table fits resident at 43.1 GB.
---

# Interim report: memory-budget-and-frozen-tables

**Locus question:** Re-derive the 64 GB budget from the cited papers: is the frozen table 12B params (24 GB) or 6B (12 GB), do the 32k LM-head logits add 10-20 GB the repo omits, and does 7B fit fully trainable via 8-bit Adam / Adafactor / ZeRO-1-2 — making freezing an architecture choice rather than a necessity?
**Flavor:** dialectical

**Evidence tiers used below:** T1 = measured on this box (the user's own numbers, unverifiable externally); T2 = community-measured on CMP 170HX-class hardware; T3 = measured in a published paper on other hardware; T4 = arithmetic from stated assumptions. The run's binding hierarchy is T1 > T2 > T4; T3 sits alongside T2 as external measurement.

## What the corpus already said

The repo's `research.md` makes three claims this locus tests. §3: the frozen Memory-Grafting table is "33M entries per grafted layer (top 11M n-grams per order)" at "roughly 12 GB of frozen memory per layer". §5: a 7B-total / 1B-trainable design lands at "~28 GB/card" plus "~3.2 GB" of checkpointed activations, i.e. ~31 GB. Headline finding #4: "7B fully trainable with Adam needs ~112 GB per card — it does not fit DDP on 64 GB. The fix is to freeze the Engram memory tables ... so 7B total with ~1B trainable fits DDP at ~28 GB/card." §5 adds: "Anything above ~2B trainable needs FSDP and pays the doubled link overhead."

The corpus already contained the primary sources that contradict the first and third claims. [[memory-grafting-scaling-language-model]] reports "the top 1M most frequent n-grams per order ... yielding 3M entries per grafted layer" and "approximately 12B frozen grafting memory offloaded" for a 0.92B-trainable recipient. [[conditional-memory-via-scalable-lookup]] shows Engram tables are trained via model-parallel sharding plus all-to-all, with host offload an inference-only mechanism. [[published-as-a-conference-paper-at-iclr-2022]], [[adafactor-adaptive-learning-rates-with-sublinear-memory-cost]] and [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] each break the 112 GB arithmetic. [[scaling-embeddings-outperforms-scaling-experts-in-language]], [[ultramemv2-memory-networks-scaling-to-120b]] and [[factorengram-factorized-n-gram-memory-with-basis-level]] show production memory models train their tables. [[delta-n-gram-forward-only-continual-writing]] and [[dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]] supply the counter-case on quality and on training-time offload.

What the corpus did not have was (a) a source on the LM-head logits term, (b) a source on 8-bit optimizer state *during pretraining* rather than fine-tuning, (c) a source on Adafactor-vs-Adam comparability at 1B scale, or (d) a source on whether DDP synchronizes frozen parameters. Those are the four new fetches below.

## What the new sources say

### 1. Liger Kernel (arXiv 2410.10989) — the logits term is real but avoidable

[[liger-kernel-efficient-triton-kernels-for-llm-training]] is the canonical engineering source for the LM-head memory problem. It states the naive cost explicitly: "single GPU training with a batch size of 8 and sequence length of 4096, the 256k vocabulary size will result in a 16.8 GB logit tensor of precision bfloat16, causing a huge spike in the peak memory usage." Scaled to this box's 32k × 151,669, that is 9.94 GB bf16 / 19.88 GB fp32 — exactly the term the repo's 3.2 GB activation estimate omits.

But the paper's fix, FusedLinearCrossEntropy (FLCE), "never materializes the full [B*T, V] logits tensor": it flattens batch×sequence, applies the head "sequentially on the chunked hidden states", and stores logit gradients in place. With Liger's chunk-size heuristic (≈2,048 rows for 32k × 151,669 at hidden 2,048), the materialized logits fall to ~0.62 GB. So the logits term is 9.94 GB only if the repo uses naive `F.cross_entropy`; with FLCE it is ~0.6 GB. Tier: T3 (measured on A100-80GB; the 16.8 GB figure is arithmetic from the stated shape).

### 2. PyTorch DDP docs (2.14) — frozen params contribute zero bytes to the all-reduce

[[distributed-data-parallel-pytorch-214-documentation]] and [[distributeddataparallel-pytorch-214-documentation]] describe the reducer: "The `Reducer` also registers autograd hooks during construction, one hook per parameter. These hooks will be triggered during the backward pass when the gradient becomes ready." A parameter with `requires_grad=False` has no gradient accumulator, so no hook can fire and no bucket can include it; DDP's only per-step traffic is the gradient all-reduce. This is the mechanism behind Position A's arithmetic. It is a docs-level inference rather than a source-level confirmation — I could not fetch `distributed.py` (two `FETCH_ERROR`s on raw.githubusercontent.com) — but the autograd semantics are unambiguous. Tier: T3 (documentation) + T4 (inference).

### 3. Chitsaz et al. (arXiv 2407.11722) — 8-bit optimizer state survives pretraining, but only with the right scheme

[[exploring-quantization-for-efficient-pre-training-of-transformer]] is the only source in the vault that quantizes Adam state *during pretraining* rather than fine-tuning. For the first moment: "per-channel quantization to 8 bits works well, achieving performance similar to that of the baseline model" (WikiText103 ppl 39.84 vs 39.94 baseline; WikiText2 33.78 vs 34.32; PTB 35.67 vs 35.13; 1BW 44.29 vs 44.03). For the second moment: "the quantized model failed to converge smoothly throughout training, even when applying per-channel quantization with 8 bits", because "a linear symmetric quantization function around zero ... causes all small values to be set to zero after quantization", and since v sits in Adam's denominator, "such clustering to zero leads to excessively large weight updates, causing training to diverge".

That failure is an artifact of naive linear quantization; Dettmers' block-wise dynamic quantization ([[published-as-a-conference-paper-at-iclr-2022]]) is designed to avoid exactly it, and validates parity at 1.5B. The combined reading: 8-bit Adam is validated to 1.5B with block-wise quantization, and is fragile with naive quantization. Tier: T3 (measured, 4×A100-80G, GPT-2-class model on OpenWebText, 300k steps).

### 4. Zhao et al. (ICLR 2025, arXiv 2407.07972) — Adafactor-with-momentum is comparable to Adam at 1.2B

[[published-as-a-conference-paper-at-iclr-2025]] sweeps Adam, Adafactor, Lion, Signum and SGD across 150M, 300M, 600M and 1.2B parameter models. Its finding: "multiple optimizers introduced in the literature after Adam—such as Lion and Adafactor (with momentum)—demonstrate robustness comparable to Adam", and "Figure 1 illustrates the remarkable similarity in performance and robustness of these optimizers across different learning rates and across multiple model scales (150m, 300m, 600m, and 1.2b parameters)."

The parenthetical matters: the authors "implement ourselves a modified version of Adafactor which maintains the factored estimates of second moments but has momentum i.e. it is equivalent to Adam with factored second moment estimates." Plain Adafactor (momentum dropped, as in the original paper) is not what they validated. Tier: T3 (measured, short runs: 3.3B tokens at 150M, 200k steps at 1.2B).

## Evidence synthesis

### The frozen table is 12B params / 24.6 GB, not 6B / 12 GB — and the repo's §3 has an 11× transcription error

The paper is unambiguous: "For each n-gram order, we select the top 1M most frequent n-grams and compute their frozen memory representations offline, yielding 3M entries per grafted layer. Each entry stores a hidden-state vector of the grafting model in bfloat16; for Qwen3.5-35B-A3B, this corresponds to roughly 12 GB of frozen memory per layer." 3M × 2,048 dims × 2 B = 12.29 GB/layer; two grafted layers (donor layers 8 and 24) = 24.58 GB = 12.29B params. The Figure 4 caption confirms: "0.92B trainable parameters with approximately 12B frozen grafting memory offloaded." [T3]

The repo's numbers are internally inconsistent three ways. §3 says "33M entries per grafted layer (top 11M n-grams per order)" while keeping "roughly 12 GB per layer" — but 33M × 2,048 × 2 B = 135.3 GB/layer, an 11× inconsistency with its own 12 GB figure. §5 then budgets "6B frozen" = 12 GB *total*, which is exactly one grafted layer of the paper's table (3M × 2,048 × 2 B = 12.29 GB = 6.14B params). So the repo's frozen table is half the paper's: either half the entries per layer (if two layers) or half the layers (if one). The correct figure for the paper's configuration is 12.29B params / 24.58 GB. [T4]

The quality consequence is smaller than the 2× gap suggests. The paper's capacity ablation (30k → 300k → 3M entries/layer, 5–10B tokens) is monotone: "Increasing the memory size from 30k to 300k and further to 3M improves the loss curve throughout training." The repo's 6B table is 2.93M entries total — at the paper's 3M point if concentrated in one layer, or 1.46M/layer if split across two. The single-layer ablation (Table 5, 0.92B/50B tokens, 56-expert MoE baseline) gives MoE 45.37 → MG 46.58 (+1.21), versus the two-layer MG-Qwen 46.98 against a 64-expert MoE baseline of 45.62 (+1.36). So a one-layer 3M-entry graft captures ~89% of the two-layer gain; a two-layer 1.46M-entry graft would land between the 300k and 3M curves, perhaps 0.1–0.3 points lower. The repo's table is undersized, but not catastrophically. [T3/T4]

### The logits term is 9.94 GB naive, ~0.6 GB fused — the repo's 3.2 GB is only valid with FLCE

At 32k × 151,669 × 2 B, the LM head produces 9.94 GB bf16 (19.88 GB fp32) if materialized. The repo's activation estimate — "32,768 tokens × 2048 hidden × 2 B × 24 layers ≈ 3.2 GB" — counts only checkpointed layer inputs and silently assumes a fused/chunked cross-entropy. That assumption is defensible (FLCE is standard in 2024–25 training stacks) but it is not stated, and without it the budget grows by 10 GB. The repo's own numbers then become: 7B/1B design 31.8 GB with FLCE, 41.8 GB naive; the paper's 13.2B design 43.1 GB with FLCE, 53.1 GB naive. Both fit 64 GB either way, but the naive-CE variant of the paper's design leaves only 11 GB of headroom. [T3/T4]

Note also that the paper's own efficiency numbers were measured at sequence length 2,048 and vocabulary 128,256 — logits 0.53 GB, not 9.94 GB. Its "0.92× training throughput" therefore says nothing about the 32k regime the repo plans to run. [T3]

### 7B fully trainable fits on 64 GB by four independent routes — freezing is not a memory necessity

The repo's 112 GB figure is correct for plain DDP + Adam (7B × 16 B/param = 112 GB), but it tests only one configuration. The alternatives, all with 3.2 GB checkpointed activations and ~0.6 GB FLCE logits:

| Route | Bytes/trainable param | 7B model state | Total (FLCE) | Fits 64 GB? |
|---|---:|---:|---:|---|
| DDP + Adam (fp32 master + m + v) | 16 | 112 GB | 115.8 GB | No |
| DDP + 8-bit Adam (bf16 w+g, 2 B state) | 6 | 42 GB | 45.8 GB | Yes |
| DDP + Adafactor (bf16 w+g, factored state) | 4 | 28 GB | 31.9 GB | Yes |
| ZeRO-1, 5 ranks (Adam, fp32 master) | 6.4 | 44.8 GB | 48.6 GB | Yes |
| ZeRO-2, 5 ranks (Adam, fp32 master) | 4.8 | 33.6 GB | 37.4 GB | Yes |

The ZeRO arithmetic is the paper's own: [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] gives 7.5B at 120 GB under standard DP, 31.4 GB under Pos (ZeRO-1) and 16.6 GB under Pos+g (ZeRO-2) at Nd=64; at Nd=5 the same formulas give 44.8 GB and 33.6 GB. Crucially, ZeRO-1/2 communication is "exactly 2Psi" — the same byte volume as DDP's all-reduce — so the fabric cost is unchanged, and the optimizer stays Adam, so comparability with the baseline is preserved. [T3/T4]

The repo's "anything above ~2B trainable needs FSDP" is also too conservative. With a realistic 4 GB reserve (activations + logits + workspace), 60 GB of model state supports ~3.75B trainable under plain DDP + Adam, ~9.4B under ZeRO-1, ~12.5B under ZeRO-2, and ~18.75B under ZeRO-3/FSDP. The repo's 2B ceiling comes from reserving 25 GB of headroom, which is a policy choice, not a hardware limit. [T4]

### The real reason to freeze is bandwidth and FLOPs, not VRAM

If the 6B table is trained rather than frozen, its gradients must move. Under DDP the gradient tensor is dense (12 GB at bf16), so the all-reduce is 12 GB/step; at the measured 0.8 GB/s that is ~15 s against 13.5 s of compute at 1B active params and 445 TFLOPS (55% MFU), i.e. ~1.4× step time even with perfect overlap of the backward. Under the Engram recipe ([[conditional-memory-via-scalable-lookup]]: "sharding the tables across available GPUs. An All-to-All communication primitive is used to gather active rows in the forward pass and dispatch gradients in the backward pass"), the traffic is ~10 KB/token × 1M tokens = 10 GB forward plus 10 GB backward = 20 GB/step, i.e. ~25 s, ~1.8–2.2× step time. Freezing removes both. [T4]

The other half is FLOPs. A fully-trainable 7B dense model at 445 TFLOPS runs 6 × 7e9 = 4.2e10 FLOPs/token → 10.6k tok/s aggregate; 9B tokens takes ~236 h (~10 days) before overhead. A 7B-total / 1B-active MoE runs 7× faster (~74k tok/s → ~34 h). So the design driver is tokens/hour, not GB/card. The repo's headline #4 ("a memory problem, not a bandwidth problem") has it exactly backwards: the memory fits either way; the bandwidth and FLOPs are what force the architecture. [T4]

### The offload-traffic finding does not hold at 80% — it is ~13–16% of one link, and the table fits resident anyway

The locus's arithmetic is `5 KB × 2 layers × 135k tok/s = 1.35 GB/s = 80% of the 1.68 GB/s link`. That divides *aggregate* throughput by *one card's* bandwidth. The 140k tok/s baseline is aggregate across 5 GPUs (9B tokens in 21 h = 119k tok/s aggregate), so per-GPU token rate is ~28k tok/s and per-GPU lookup traffic is 28k × 8 KB = 224 MB/s = 13% of the measured 1.7 GB/s H2D link ([[github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-lab-vllm-scripts-me]]: "PCIe ×16 vs ×4: H2D 6.63 vs 1.7 GB/s"; [[github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2]]: "Measured 1.68 GB/s each way on PCIe Gen2 x4"). At a more realistic 60k tok/s aggregate for a 1B-active model, it is 96 MB/s = 5.6%. The correct aggregate figure is ~1.1 GB/s against ~8.5 GB/s of aggregate link capacity, not 80% of one link. [T2/T4]

Two further corrections. First, the lookup traffic is comparable to, not dominant over, the DDP all-reduce. At 140k tok/s a 1M-token step takes 7.1 s; a ring all-reduce of a 2 GB payload sends 2S(N−1)/N = 3.2 GB per rank, i.e. ~450 MB/s averaged over the step, and ~1.29 GB/s during its ~2.5 s execution window (76% of the link). The lookup volume is 1M tokens × 8 KB / 5 GPUs = 1.6 GB per GPU per step — half the all-reduce's 3.2 GB — which fits in the ~3.5 s forward window at ~470 MB/s (27%). The two can be scheduled apart; they cannot overlap without contention. Second, the lookup keys depend only on input token IDs, so retrieval can be prefetched a layer ahead — the mechanism [[pooling-engram-conditional-memory-in-large]] describes ("Since the hash function relies on N-gram token IDs rather than intermediate hidden states, retrieval can be initiated at the start of the decoding step") and the reason [[conditional-memory-via-scalable-lookup]] measures only 2.8% inference penalty for a 100B table on H800. At 32k-token micro-batches the per-layer compute window (~1.2 s at 1B active / 162 TFLOPS per GPU) dwarfs the lookup time. [T3/T4]

Third, and most important: the table fits resident. The paper's 13.2B design lands at 43.1 GB with FLCE (53.1 GB naive), leaving 11–21 GB of headroom on a 64 GB card. The repo's 7B design lands at 31.8 GB. There is no VRAM pressure forcing offload, so the rational choice on this box is to keep the table resident and pay zero PCIe traffic. The paper's single word "offloaded" (in a figure caption, with no mechanism, no host/CPU mention anywhere in the text, and no measured traffic) does not establish that host offload is required — and the Engram paper's offload path is explicitly inference-only ("(b) Inference Phase: Engram tables are offloaded to host memory"). [T3/T4]

### The quality premise is weaker than the repo's §3 implies

The paper's 0.92B/50B-token table is: MoE 45.62, vanilla Engram 45.03, MG-GLM 45.83, MG-DeepSeek 46.42, MG-Qwen3.5 46.98. Two things follow. First, vanilla Engram *loses* to MoE at this scale ("vanilla Engram underperforms the MoE baseline in this 1B-scale setting, with 45.03 versus 45.62") — the repo's §3 line "all grafting variants beat both baselines" is technically true but omits the negative result that matters. Second, the grafting gain over MoE is +0.21 to +1.36 points, at 50B tokens. The repo's planned run is 9B tokens, 5.5× shorter; the paper's capacity ablation runs 5–10B tokens, so a 9B run can detect the loss-curve effect but not the benchmark delta. [T3]

The paper's own configuration also differs from the repo's in every dimension that matters: 12 layers, hidden 768, vocab 128,256, sequence length 2,048, 50B tokens, AdamW, 32×A100 — versus the repo's 24 layers, hidden 2,048, vocab 151,669, 32k context, 9B tokens. The 0.92× training-throughput figure was measured in the paper's regime, not the repo's. [T3]

## Steelmanning the five positions

### Position A: Frozen tables are pure VRAM — 7B/1B fits at ~28 GB and is the cheapest capacity available

Best case: a frozen parameter has no gradient accumulator and no optimizer state, so it costs exactly its bf16 rows and nothing else. [[delta-n-gram-forward-only-continual-writing]] makes this concrete: its frozen table "carries no optimizer state", and the only trained component is a 4.08M-parameter writer against a 48-layer, 512-expert backbone. DDP's reducer registers hooks on gradient accumulators, so a frozen table contributes zero bytes to the all-reduce. The arithmetic is clean: 7B × 2 B (weights) + 1B × 2 B (grads) + 1B × 12 B (Adam) = 28 GB, plus 3.2 GB checkpointed activations = 31.2 GB, leaving 33 GB of headroom. Per-token FLOPs are constant in table size, so capacity is free at the margin. This is the strongest version of the repo's position, and its arithmetic is correct for the repo's own 6B-frozen design.

### Position B: Production-scale memory models train their tables, not freeze them

Best case: every production memory model in the corpus trains its memory. Engram-27B carries 5.7B Engram parameters trained with model-parallel sharding and all-to-all ([[conditional-memory-via-scalable-lookup]]); LongCat-Flash-Lite allocates "31.4B of its 68.5B parameters (46%) to N-gram Embedding and trains them FROM SCRATCH jointly with the backbone" ([[scaling-embeddings-outperforms-scaling-experts-in-language]]); UltraMemV2's "120B sparse memory parameters are trained from scratch jointly with the backbone, not frozen" ([[ultramemv2-memory-networks-scaling-to-120b]]); FactorEngram trains 1–2B lookup-table parameters at 340M–1B scale ([[factorengram-factorized-n-gram-memory-with-basis-level]]). The only frozen-table evidence is one paper at 0.92B/2.8B and a 4M-param writer. On the face of it, freezing is a niche.

### Position C: 7B fully trainable needs ~112 GB/card, so freezing is required

Best case: 16 B/param × 7B = 112 GB is the correct mixed-precision Adam accounting (bf16 weight 2 + bf16 grad 2 + fp32 master 4 + fp32 m 4 + fp32 v 4), and it is confirmed by [[zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum]] and [[memory-efficient-mixed-precision-optimizers]] ("standard AMP keeps an fp32 weight plus fp16 copy (6 bytes), an fp32 gradient (4 bytes) and Adam state (8 bytes) per parameter"). On a 64 GB card, 112 GB does not fit, and the alternatives each carry a cost: 8-bit Adam changes the optimizer and its quality evidence tops out at 1.5B; Adafactor changes the optimizer and needs momentum added back to match Adam; ZeRO-1/2 change the parallelism and require reduce-scatter, which is unmeasured on this fabric. So the conservative reading — freeze and keep Adam + DDP — is defensible.

### Position D: 7B fits without freezing via 8-bit Adam, Adafactor, or ZeRO-1/2

Best case: the arithmetic is decisive. ZeRO-2 at 5 ranks gives 4.8 B/param = 33.6 GB of model state with "exactly 2Psi" communication — the same byte volume as DDP — and keeps Adam, so comparability is preserved. Adafactor gives 4 B/param = 28 GB, and [[published-as-a-conference-paper-at-iclr-2025]] shows Adafactor-with-momentum is "comparable to Adam" in both optimal loss and learning-rate stability across 150M–1.2B. 8-bit Adam gives 6 B/param = 42 GB, and [[published-as-a-conference-paper-at-iclr-2022]] validates parity at 1.5B with block-wise quantization. All three fit 64 GB with 18–32 GB of headroom. The repo's "does not fit" conclusion is true only for the one configuration it tested.

### Position E: Frozen Engram at ~1B is not a free win

Best case: the paper's own numbers undercut the premise. At 0.92B, vanilla Engram loses to MoE (45.03 vs 45.62), and grafting recovers only +0.21 to +1.36 after 50B tokens. The repo's run is 9B tokens, so it cannot reproduce the benchmark delta. The repo's 6B table is half the paper's 12B, and its allocation (86% of total parameters frozen) is far outside both published allocation laws — Engram's U-shaped optimum ("reallocating roughly 20%–25% of the sparse parameter budget to Engram yields the best performance ... stable across regimes (ρ≈75%–80%)") and LongCat's "no more than 50% of the total parameter budget to N-gram Embedding". [[delta-n-gram-forward-only-continual-writing]] adds that frozen memory is not free capability: net acquisition on unseen tasks is negative (−12/448), recall of written examples is 23–25%, and general-input big-K drift is 32.8%. So the experiment may well show nothing.

## Open questions

- **Reduce-scatter vs all-reduce on this box.** The ZeRO-2 route's 2Ψ advantage is a byte-volume claim; no source measures reduce-scatter throughput on a no-P2P Gen2 x4 fabric. If reduce-scatter is much slower than the measured 0.75 GiB/s all-reduce, the 7B-fully-trainable route gets more expensive. This is the single measurement that would most change the position.
- **Does the paper actually offload to host?** The word "offloaded" appears once, in a figure caption, with no mechanism and no host/CPU/PCIe mention anywhere in the paper. I could not resolve whether it means host DRAM or merely "outside the trainable parameter set". A v2 or author clarification would settle it.
- **Training-time lookup latency at 3M entries/layer, resident vs offloaded.** [[tf-engram-a-train-free-engram-with-ssd-backed-memory-for]] gives an inference curve (2.108 ms/token at 1M entries → 2.175 ms/token at 100M on an RTX 5090) and [[pooling-engram-conditional-memory-in-large]] gives a latency analysis, but neither measures a training step at 32k-token micro-batches.
- **8-bit Adam at 7B pretraining.** No source validates it above 1.5B. The repo's own 1B-trainable design is inside the validated range, but the fully-trainable 7B alternative is not.
- **DDP frozen-parameter behavior at source level.** Two `FETCH_ERROR`s on the PyTorch source; the docs-level inference is sound but unconfirmed.
- **The paper's 2.8B frozen-table size.** The paper reports 12B frozen for the 0.92B recipient; it does not state the 2.8B recipient's table size, so the scaling of frozen capacity with recipient size is unknown.

## Sources

1. [[memory-grafting-scaling-language-model]] — Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory (arXiv 2605.20948)
2. [[conditional-memory-via-scalable-lookup]] — Conditional Memory via Scalable Lookup (Engram, arXiv 2601.07372)
3. [[liger-kernel-efficient-triton-kernels-for-llm-training]] — Liger Kernel: Efficient Triton Kernels for LLM Training (arXiv 2410.10989) **[new]**
4. [[published-as-a-conference-paper-at-iclr-2022]] — 8-Bit Optimizers via Block-Wise Quantization (arXiv 2110.02861)
5. [[exploring-quantization-for-efficient-pre-training-of-transformer]] — Exploring Quantization for Efficient Pre-Training of Transformer LMs (arXiv 2407.11722) **[new]**
6. [[adafactor-adaptive-learning-rates-with-sublinear-memory-cost]] — Adafactor (arXiv 1804.04235)
7. [[published-as-a-conference-paper-at-iclr-2025]] — Deconstructing What Makes a Good Optimizer for Autoregressive LMs (arXiv 2407.07972) **[new]**
8. [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] — ZeRO (SC20, arXiv 1910.02054)
9. [[zero-ofﬂoad-democratizing-billion-scale-model-training]] — ZeRO-Offload (arXiv 2101.06840)
10. [[fullyshardeddataparallel-pytorch-213-documentation]] — PyTorch FSDP docs
11. [[distributed-data-parallel-pytorch-214-documentation]] — PyTorch DDP notes **[new]**
12. [[distributeddataparallel-pytorch-214-documentation]] — PyTorch DDP class reference **[new]**
13. [[memory-efficient-mixed-precision-optimizers]] — Memory Efficient Mixed Precision Optimizers (arXiv 2309.12381)
14. [[zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum]] — ZeRO-3 vs FSDP (vendor explainer)
15. [[scaling-embeddings-outperforms-scaling-experts-in-language]] — LongCat-Flash-Lite (arXiv 2601.21204)
16. [[ultramemv2-memory-networks-scaling-to-120b]] — UltraMemV2 (arXiv 2508.18756)
17. [[factorengram-factorized-n-gram-memory-with-basis-level]] — FactorEngram (arXiv 2609.35578)
18. [[delta-n-gram-forward-only-continual-writing]] — Delta N-gram (arXiv 2610.31125)
19. [[dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]] — Dr. Robert Li critique
20. [[tf-engram-a-train-free-engram-with-ssd-backed-memory-for]] — TF-Engram (arXiv 2607.07388)
21. [[pooling-engram-conditional-memory-in-large]] — Pooling Engram Conditional Memory (arXiv 2603.10087)
22. [[github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2]] — bayley/cmpunlocker (1.68 GB/s BAR1 P2P)
23. [[github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-lab-vllm-scripts-me]] — dl4rce/cmp-170hx-inference-lab (H2D 1.7 GB/s at x4)
24. [[qwen3-technical-report]] — Qwen3 Technical Report (vocab 151,669)

## Committed position

The dialectical question resolves in favor of Position D on the memory question and Position A on the architecture question, and against the repo's stated reasoning on both. **7B fully trainable does fit on 64 GB**: ZeRO-2 at 5 ranks needs 33.6 GB of model state at DDP-identical 2Ψ communication and keeps Adam; Adafactor-with-momentum needs 28 GB and is loss-comparable to Adam at 1.2B; 8-bit Adam needs 42 GB and is validated to 1.5B. So freezing is an architecture choice, not a memory necessity, and the repo's headline #4 ("~112 GB/card ... freezing is required") is false as a general claim. But the choice is still correct on this box, for reasons the repo does not state: at ≤1B scale the paper's own trained-Engram baseline *loses* to MoE (45.03 vs 45.62) while frozen grafting wins (46.98), and training a 6B table costs ~1.4–2.2× step time on this fabric (12–20 GB/step of gradient or all-to-all traffic against a measured 0.8 GB/s), whereas a fully-trainable 7B dense model runs 7× slower per token than a 1B-active MoE. The repo's memory table is wrong in both directions: it understates the paper's frozen table by 2× (12.29B params / 24.58 GB, not 6B / 12 GB — its §3 "33M entries/layer" is an 11× transcription error that would imply 135 GB/layer), and it omits the 9.94 GB naive LM-head logits term (which FLCE reduces to ~0.6 GB, so the 3.2 GB activation figure is valid only with fused CE). The locus's load-bearing offload finding does not survive checking: 1.35 GB/s is aggregate lookup traffic, not per-link, so it is ~13–16% of one card's 1.7 GB/s link (not 80%), it is comparable to rather than dominant over the all-reduce, it is prefetchable because keys depend only on token IDs, and the table fits resident at 43.1 GB anyway. The repo's experiment as designed (9B tokens, 6B frozen) can test the loss-curve effect at 5–10B tokens but cannot reproduce the paper's +0.2 to +1.4 benchmark delta at 50B tokens; matching the paper's configuration (12.29B frozen, 13.2B total) fits at 43.1 GB and would make the result comparable.

- **Position:** Freezing the Engram tables is an architecture choice, not a memory necessity — 7B fully trainable fits 64 GB via ZeRO-2 (33.6 GB), Adafactor (28 GB) or 8-bit Adam (42 GB) — but it is the right choice on this box for scale-specific quality (trained Engram loses to MoE at 0.92B) and fabric cost (~1.4–2.2× step time to train a 6B table), not for VRAM.
- **Confidence:** high on the arithmetic and the source readings (>80%); medium (50–80%) on the prescriptive claim that the repo should keep the frozen design rather than test a fully-trainable 7B arm, because the reduce-scatter measurement on this fabric is missing.
- **Boundary conditions:** This holds for a 5× CMP 170HX box at Gen2 x4 with 64 GB/card, a 1M-token global batch, and a ≤1B-active model. It does not hold if the link is actually Gen2 x16 (all communication costs drop ~4× and the fully-trainable route gets cheaper still), if the frozen table grows beyond ~4 grafted layers × 12 GB (48 GB of frozen weights plus ~40 GB of state would exceed 64 GB and make freezing a genuine memory necessity), or at production scale (≥27B, ≥262B tokens), where every published memory model trains its tables and the Engram recipe's all-to-all is affordable.
- **What would change this position:** A measured reduce-scatter throughput on this box below ~0.4 GB/s would make ZeRO-2's 2Ψ advantage illusory and push the 7B-fully-trainable route from "fits" to "fits but impractical", restoring freezing as a near-necessity. A demonstrated >0.02-nat loss gap for Adafactor-with-momentum or 8-bit Adam at 1B–7B pretraining would break the optimizer routes and do the same. Conversely, a measured training-time lookup latency at 3M entries/layer that exceeds the per-layer compute window would make even the resident-table design's lookup path a bottleneck, and a v2 of the Memory Grafting paper clarifying that "offloaded" means host DRAM with measured traffic would revive the offload concern at the ~13–16% level rather than the 80% level.
- **Evidence weight:** 6 primary papers support the arithmetic (Memory Grafting, Engram, ZeRO, 8-bit Adam, Adafactor, Zhao et al.); 2 papers (Chitsaz et al., Liger) supply the pretraining-quantization and logits corrections; 3 production-model papers (LongCat, UltraMemV2, FactorEngram) support Position B but are outside this box's scale; 2 community measurements (bayley, dl4rce) anchor the 1.68–1.7 GB/s link; 1 adversarial analysis (Dr. Robert Li) and 1 counter-datapoint (Delta N-gram) support Position E; the repo's own docs supply the claims under test and are the only source for the 0.75 GiB/s all-reduce and the 140k tok/s baseline (T1, unverifiable externally).
