---
title: Interim report — csa2-at-32k-sm80
id: interim-report-csa2-at-32k-sm80
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-csa2-at-32k-sm80
created: '2026-10-27T13:58:54.800810Z'
updated: '2026-10-27T19:02:13.700052Z'
status: review
type: interim
tier: practitioner
content_type: review
deprecated: false
summary: 'CSA2 at 32k on sm_80 is a kernel-porting project, not an experiment: the
  indexer''s 64-128 GiB score tensor is a 32x-reducible artifact (StreamIndex measures
  0.40 GB at 32k), but every training-capable DSA kernel is Hopper-or-newer (cuDNN
  SM90/SM100, FlashMLA now sm_100a/103a+Ascend with Hopper dropped, TileLang H800/H200,
  StreamIndex tested sm_90a only) and the only sm_80 path is vLLM''s inference-only
  TRITON_MLA_SPARSE with no backward. LongCat''s 0.83x speedup ratio at 32K means
  CSA2 would be a +10.7-14.7% step-time penalty even if it ran; NSA''s measured 6.3x
  fwd / 3.4x bwd at 32k on 8xA100 implies a 4.02x attention-time reduction = -39.1%
  to -54.1% step time. Run NSA (MoBA as flash-attn-2.6.3 fallback), drop CSA2.'
---

# Interim report: csa2-at-32k-sm80

**Locus question:** Is CSA2 trainable at 32k on GA100 within 64 GB — or is the indexer's score-tensor memory and SM90-only kernel path a hard blocker, and is a block-sparse family (MoBA/NSA) the better experiment?
**Flavor:** dialectical

**Evidence tiers used throughout** (per run directive): **[T1]** measured on this box · **[T2]** community/vendor measurement on comparable hardware, or vendor documentation of a hard constraint · **[T3]** arithmetic estimate from stated assumptions. There is **no [T1] evidence anywhere in this locus** — nobody has run any sparse-attention kernel on a CMP 170HX and recorded a number. That absence is itself the finding.

---

## What the corpus already said

The corpus had already assembled the pieces of a feasibility blocker but had not joined them. [[streamindex-memory-bounded-compressed-sparse]] establishes that the CSA indexer — not the attention kernel — gates sequence length: the materialize path allocates a `[B, S, H_I, T]` FP32 intermediate before the head-sum, and at S=32,768 with V4-Flash dimensions that is 64 GiB, with a measured peak of 129.00 GB on a 140 GB H200. Its chunked partition-merge top-k driver runs the same indexer at 0.40 GB and is 10.3× faster at 32k. [[dsa_referencepy]] shows the pure-PyTorch cuDNN reference has the same defect with no chunked variant: `ref_indexer_forward` materializes `(B, S_q, H_q, S_k)` FP32, which is 68.7 GB at 32k/H_q=16, while `ref_sparse_attention_forward` gathers `(t, topk, d)` FP32 — 154.6 GB at t=32768, topk=2048, d=576 — and only the `query_chunk_size=64` wrapper (~302 MB) is usable. [[deepseek-sparse-attention-dsa-nvidia-cudnn]] documents that every cuDNN DSA kernel (Sparse Attention Backward, Indexer Forward, Indexer Top-K, Indexer Backward, Score Recompute) is SM90/SM100-only, and that the production sparse-attention forward is FlashMLA in C++, not integrated. [[deepseek-v41-flash]] supplies the layer mix and the candidate-cap arithmetic; [[longcat-sparse-attention-taming-the-lightning-via]] supplies the 32K payoff measurement; [[native-sparse-attention-hardware-aligned-and-natively]], [[moba-mixture-of-block-attention-for]], [[hysparse2-hybrid-sparse-attention]], [[published-as-a-conference-paper-at-colm-2026]] and [[indexcache-accelerating-sparse-attention-via-cross-layer]] supply the block-sparse comparison set.

The repo's own position (research.md §4) is: *"worth one experiment, but it is the axis most likely to disappoint at 32k on this box"*, resting on three claims — the indexer is "a second, smaller attention running in parallel" in every layer, the 16,384 candidate cap "barely binds" at 32k, and the 890 B/token figure is an inference number. The locus rationale argues the verdict is right for the wrong reasons and that the corpus contains a *hard* feasibility blocker.

---

## What the new sources say

### 1. FlashMLA has dropped Hopper entirely — the production CSA2 forward is Blackwell/Ascend only — [[github-deepseek-aiflashmla-flashmla-efficient-multi-head-latent-attention-kernel]]

This is the single most decisive new fact, and it is stronger than the locus assumed. The locus said production DSA kernels are "SM90/SM100-only". The current FlashMLA release (2026-09-30, commit `2e5429f`) states its supported scope verbatim:

> "Supported scope: sm_100a / sm_103a and Ascend 950 (dav-3510), DeepSeek-V4.1 only, with the fp8 and fp4 KV-cache layouts. ... **Drop support for NVIDIA Hopper GPUs**, and drop support for DeepSeek V3 / V3.2 / V4.0 models"

**[T2 — vendor documentation]** So the production sparse-attention forward kernel for CSA2 does not run on Hopper, let alone Ampere. The only NVIDIA targets are Blackwell `sm_100a`/`sm_103a`. Combined with the cuDNN DSA limitation (SM90/SM100), the *entire* NVIDIA production DSA stack — forward and backward — is Hopper-or-newer, and the forward has now moved to Blackwell-only. A GA100 box is two architecture generations below the floor.

### 2. TileLang's DSA reference kernels are SM90-tuned and their throughput numbers are H800/H200 — [[tilelangexamplesdeepseek_v32-at-main-tile-aitilelang-github]]

TileLang ships a complete DSA stack — `fp8_lighting_indexer.py`, `topk_selector.py`, `sparse_mla_fwd.py`, `sparse_mla_fwd_pipelined.py`, `sparse_mla_fwd_fp8.py`, `sparse_mla_bwd.py` — which is the most complete open training-capable DSA implementation in existence. Its performance claims are all Hopper:

> "The pipelined version (`sparse_mla_fwd_pipelined.py`) is a manual pipeline implementation designed to match the schedule of FlashMLA ... It achieves close to **600 TFlops on H800 SXM**"
> "**Performance**: The sparse MLA backward achieves excellent performance: **H800 SXM**: ~100 TFlops; **H200 SXM**: ~115 TFlops"

**[T2 — vendor measurement on Hopper]** The kernels are explicitly built around Hopper primitives: the FP8 section explains that "On SM90 the fp8 form of `wgmma.mma_async` has no `imm-trans-a` / `imm-trans-b` operands", and the pipelined kernel uses `T.cp_async_barrier_noinc` + `T.alloc_barrier` producer/consumer warp specialization. `wgmma` is Hopper-only; Ampere has only `mma.sync`. TileLang *can* emit `mma.sync` (the note records a fallback: "Without a K-major V, `T.gemm` falls back to `mma.sync` and the whole kernel runs 1.81x slower"), so an sm_80 build is not architecturally impossible — but nothing in this stack has been compiled or measured for sm_80, and the tuned schedule is Hopper's.

### 3. An sm_80 sparse-MLA path *does* exist — but it is inference-only — [[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]]

This is the source that most constrains the locus's "no sm_80 path" claim, and it cuts both ways. vLLM PR #47629 adds a `TRITON_MLA_SPARSE` backend "so DSA sparse-MLA models (DeepSeek-V3.2, GLM-5.x) can run on GPUs where DeepGEMM and FlashMLA-Sparse are unavailable — SM80 (A100/A800) and SM121 (GB10/DGX Spark)", plus "a Triton `fp8_mqa_logits` for the indexer. `mqa_logits_triton.py` reproduces DeepGEMM's prefill" path. **[T2 — community measurement]** It was tested on 8× A800-80GB PCIe (SM80):

> "`pytest tests/kernels/attention/test_mqa_logits_triton.py` — **41 passed** (24.8s)"
> "`pytest tests/kernels/attention/test_triton_mla_sparse_kernel.py` — **53 passed** (100.4s)"
> "Throughput: 32.3 tok/s single-stream decode, 100.3 tok/s aggregate at 4-way concurrency"

So the locus's claim that "`dsa_reference.py` is the only SM80-runnable path" is **false as of now**. But the correction does not rescue CSA2: this is a *serving* backend. vLLM is inference-only — there is no backward pass, no training loop, no optimizer step. The measured numbers are decode tok/s, not training throughput. And the Ampere forward is acknowledged to be slow: the parallel `pzhao-eng/FlashMLA` Ampere port states "the performance of Ampere is currently poor due to register overflow" **[T2]**, and the SM80 DeepSeek-V4 branch is labelled "Status: UNTESTED on hardware" **[T2]**. The PR also records "sporadic NaN logits under sustained high load" and two Triton-path kernel bugs under spec decode.

The correct statement is therefore: **an sm_80 DSA *forward* exists as an unvalidated, low-performance Triton fallback; no sm_80 DSA *backward* exists anywhere.**

### 4. MoBA is the one block-sparse alternative with a shipped, Ampere-compatible implementation — [[github-moonshotaimoba-moba-mixture-of-block-attention-for-long-context-llms-gith]]

> "**Note that current kernel implementations rely on `flash-attn==2.6.3` and `torch >= 2.1.0`**"
> "**moba_efficient**: Our production-ready implementation optimized for performance. It achieves up to 40x speedup compared to moba_naive (tested with 32K sequence length, 1 attention head, MoBA Block 2048 and MoBA Topk 3)."

**[T2 — repo documentation]** `flash-attn==2.6.3` supports Ampere (sm_80), and the repo is 100% Python over flash-attn's kernels, so MoBA is *runnable* on a CMP 170HX without a port. But the 40× figure is vs MoBA's own naive masked implementation, not vs dense attention — it says nothing about MoBA-vs-FlashAttention at 32k. The paper's own efficiency section says MoBA's advantage at short context is nil: "the inset graph ... focusing on shorter sequences (32K to 512K), shows that **even though both methods perform comparably at smaller scales**, MoBA's computational advantage becomes increasingly evident as sequences grow longer" **[T2]**. And its 32K quality is worse: fitted trailing LM loss at 32K is **1.546 for MoBA vs 1.464 for full attention** — a 0.082 gap — recovered only by the 90/10 MoBA→full hybrid, which "reaches a loss nearly identical to that of full attention" **[T2]**.

---

## Evidence synthesis

### The memory objection is real but 32× reducible — it is not the hard blocker

The locus's headline arithmetic is right and the units are consistent once reconciled: `[B,S,H_I,T]` FP32 at S=32,768 with V4-Flash dimensions (H_I=64, m=4) is 68.7 GB = **64 GiB**, which is exactly the paper's "64 GB intermediate" and the reference implementation's 68.7 GB at H_q=16. **[T3]** But with V4.1-Flash's *actual* shipped dimensions the number is worse than the locus states: indexer query heads = 32, head dim = 128, compression m=2 in the encoder and **m=1 in the decoder** ([[deepseek-v41-flash]] §4.2.1). At S=32,768 that gives **64 GiB for an encoder Full layer (T=16,384) and 128 GiB for the decoder Full layer (T=32,768)** — one to two full 64 GB cards. **[T3]**

Against that, the *fundamental* requirement is only the head-summed `(B, S_q, S_k)` FP32 score tensor: **4.29 GB** at 32k/m=1, **2.15 GB** at 32k/m=2. **[T3]** The 64–128 GiB figure is an artifact of materializing the per-head axis. StreamIndex demonstrates the fix on real hardware: 129.00 GB → **0.40 GB** at S=32,768, and 10.3× faster (317 ms → 31 ms) **[T2]**. The reference path's 154.6 GB unchunked attention gather already has a chunked wrapper in the same file (~302 MB per 64-query chunk) **[T2]**. And the sparse indexer backward consumes only `(B,S_q,topk)` = 32768×512×4 = **67 MB** **[T3]**.

So: **the indexer's score-tensor memory is a solved engineering problem, not a hard blocker.** Anyone arguing CSA2 is ruled out by memory is arguing against an implementation detail that has already been fixed and measured.

### The architecture objection is the hard blocker, and it is harder than the locus stated

Every *training-capable* DSA kernel in existence is Hopper-or-newer:

| Stack | Forward | Backward | Architectures |
|---|---|---|---|
| cuDNN DSA ([[deepseek-sparse-attention-dsa-nvidia-cudnn]]) | SM90/SM100 | SM90/SM100 | Hopper, Blackwell |
| FlashMLA ([[github-deepseek-aiflashmla-flashmla-efficient-multi-head-latent-attention-kernel]]) | sm_100a/sm_103a + Ascend 950 | — | Blackwell, Ascend (**Hopper dropped**) |
| TileLang `deepseek_v32` ([[tilelangexamplesdeepseek_v32-at-main-tile-aitilelang-github]]) | H800/H200 tuned, wgmma | H800/H200 tuned | Hopper |
| StreamIndex ([[github-rightnow-aistreamindex-memory-bounded-compressed-sparse-attention-via-str]]) | Triton, tested sm_90a | Triton `sparse_attn_bwd.py`, tested sm_90a | **"Tested on H100 / H200 (Hopper, `sm_90a`)"** |
| vLLM `TRITON_MLA_SPARSE` ([[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]]) | Triton, tested on 8×A800 | **none** | SM80/SM121, **inference only** |
| `dsa_reference.py` ([[dsa_referencepy]]) | pure PyTorch | pure PyTorch | any, but OOMs at 32k |

**[T2 for all rows]** The StreamIndex row is the crucial nuance: it *does* ship a Triton sparse-attention backward (`sparse_attn_bwd.py`, "atomic-scatter dKV") and a Triton chunked indexer — an arch-portable full training stack. But the repo says plainly "Tested on H100 / H200 (Hopper, `sm_90a`)", and the paper lists "multi-hardware validation on H100, **A100**, and MI300X to confirm the regime extension is hardware-agnostic" as *future work*. Its own Triton attention forward "reaches about 8% of H200 BF16 peak at V4-Pro dimensions (`d_h = 512`)" **[T2]** — against TileLang's 40% on the same shape. At 8% of peak, even a 64× FLOP reduction does not produce a usable training kernel.

So the honest formulation is: **CSA2 at 32k on this box is not an experiment, it is a kernel-porting project.** The pieces exist in Triton, but none has been compiled, validated, or benchmarked on sm_80, and the one sm_80 forward that does exist is an inference fallback with acknowledged register-overflow performance problems.

### The repo's reasoning is wrong in a way the locus also gets wrong

Both the repo and the locus commit the same category error: treating the 16,384 candidate cap as the source of attention sparsity.

The repo says the cap "barely binds" at 32k, therefore "the sparsity win is smaller than the published numbers suggest". The locus says the cap binds 2× at 32k, therefore "the sparsity is real". Neither is the right quantity. The candidate cap governs the **indexer's** search cost, not the **attention's** density. Attention density at 32k is set by top-k=512 against 32,768 KV = **1.56% → 64× FLOP reduction** **[T3]**, and that is true whether or not the cap binds.

And the cap binds on far fewer layers than either side assumes. V4.1-Flash's shipped 40-layer mix is **2 SWA + 4 Full + 4 Reindex + 30 Reuse** (3 encoder Full + 1 decoder Full; 4 decoder Reindex; 15 encoder + 15 decoder Reuse) **[T2, [[deepseek-v41-flash]] §4.2.1]**. That is **32 of 40 layers (80%) running no indexer at all**, not the locus's 27/40 = 67.5%. A per-layer indexer cost model overstates the indexer's aggregate cost by **5×** (40/8), not 10×. Worse for the locus's argument: the Hierarchical Sparse Indexer is "used **only in the decoder**", and "The first Full Mode layer **still scans the entire causally visible range**" **[T2]**. So the cap binds on exactly the 4 decoder Reindex layers — **10% of layers**, halving their indexer work. That is marginal, and the repo's "barely binds" is directionally correct about the *cap*.

The repo's conclusion (small net win at 32k) is right; its causal story is wrong. The locus's correction is wrong in the opposite direction. The actual determinant of the 32k payoff is **gather granularity**, which LongCat measures directly.

### Position A: CSA2 is worth one experiment at 32k — the mechanism is real and the memory objection is engineering

Steelman. CSA2 is not a research prototype: DeepSeek-V4.1-Flash is a 552B-total MoE pretrained on **45T tokens at 1M context** with CSA2 in 38 of 40 layers, delivering **890 bytes of KV cache per token** and near-flat decode FLOPs **[T2, [[deepseek-v41-flash]]]**. The mechanism is published, ablated, and shipped. At 32k the attention sparsity is genuinely 64× (top-k 512 of 32,768) **[T3]**, which is more than NSA's 21.3× (16 blocks × 64 + 512 window = 1,536 of 32,768) or MoBA's 21.3× (3 × 512) **[T3]**. The indexer memory that the locus calls a "hard blocker" has been measured down to **0.40 GB at 32k, 10.3× faster than the materialize path** on an H200 **[T2]**. The layer mix means only 8 of 40 layers run an indexer, so the indexer overhead the repo worries about applies to 20% of layers. And the payoff ceiling is not small: at 32k, attention is 52–72% of step FLOPs for 0.5–1B models, so a 2× attention win is worth **26–36% of step time** **[T3]**. On this reading, CSA2 is a legitimate arm — the only reason to skip it is tooling, and tooling is fixable.

### Position B: CSA2 is not runnable at 32k on this box — the kernel path is the blocker, not the memory

Steelman. The memory argument is a strawman; the architecture argument is not. cuDNN's DSA kernels are SM90/SM100 **[T2]**. FlashMLA's current release supports only `sm_100a`/`sm_103a` and Ascend 950, having explicitly "Drop[ped] support for NVIDIA Hopper GPUs" **[T2]**. TileLang's DSA stack is wgmma-based and benchmarked on H800/H200 at ~600 TFLOPs forward and ~100 TFLOPs backward **[T2]**. StreamIndex — the one arch-portable full stack — is "Tested on H100 / H200 (Hopper, `sm_90a`)" with A100 validation deferred to future work, and its Triton attention forward reaches 8% of peak at `d_h=512` **[T2]**. The only sm_80 path is vLLM's inference-only `TRITON_MLA_SPARSE`, which has no backward, is described elsewhere as "poor ... due to register overflow", and whose DeepSeek-V4 variant is "UNTESTED on hardware" **[T2]**. The pure-PyTorch reference OOMs at 32k: 68.7 GB for the indexer at H_q=16, 137.4 GB at V4.1's H_q=32/m=1, 154.6 GB for the unchunked attention gather **[T3]**. To run a CSA2 training arm you must write an sm_80 indexer backward *and* an sm_80 sparse-MLA backward, then tune both to usable MFU, then validate the distillation loss — before you learn anything about whether CSA2 helps at 32k. That is not "one experiment".

### Position C: sparse attention at 32k is granularity-dependent — token-granular loses, fine-block wins, coarse-block is neutral

Steelman. This is the position with the best *measured* evidence, and it is measured on sm_80.

**Token-granular (DSA/LSA lineage) loses at 32K.** LongCat measures LSA vs dense MLA per attention layer: **1.08× forward, 0.79× backward, 0.83× total at 32K**, with crossover only at 64K (1.39×) and 128K under variable-length packing **[T2, [[longcat-sparse-attention-taming-the-lightning-via]] Fig. 10]**. Note that 0.83× is a *speedup ratio*, so LSA takes **1/0.83 = 1.20× dense MLA's latency** — a net loss. The mechanism is quantified: "each selected token is fetched via an independent gather of a single latent KV vector (1,152 B in BF16), which spans only 3 cachelines ... this gather occupies merely 3 of the ~50 outstanding memory slots (yielding ~6% memory-level parallelism); ... data packing efficiency is only ~75%. The resulting net effective bandwidth thus plummets to a mere **~4.5% (~1/22) of peak**" **[T2]**. That is why 64× fewer FLOPs produces a 20% *slowdown*.

**Fine-block (NSA) wins at 32k, measured on A100.** NSA's efficiency analysis is run on an **8-GPU A100 system** with Triton kernels: **6.3× forward and 3.4× backward at 32k** (2.1×/1.1× at 8k, 3.8×/2.0× at 16k, 9.0×/6.0× at 64k) **[T2, [[native-sparse-attention-hardware-aligned-and-natively]] Fig. 6]**. Its config is fine-grained — compression block 32, selection block 64, 16 selected blocks, sliding window 512 — i.e. 1,536 of 32,768 tokens = 4.69% density, 21.3× theoretical **[T3]**, achieving 6.3× forward (≈30% of theoretical, consistent with memory-bound sparse attention). And it *improves* quality: average benchmark 0.456 vs 0.443 for full attention at 27B/260B tokens **[T2]**.

**Coarse-block (MoBA) is compute-neutral at 32k and lossy alone.** "Even though both methods perform comparably at smaller scales" **[T2]**; trailing LM loss at 32K is 1.546 vs 1.464 **[T2]**; the 90/10 MoBA→full hybrid recovers full-attention loss **[T2]**. MoBA is the runnable fallback (flash-attn 2.6.3, sm_80-compatible) but it buys no step time at 32k.

**Training-native block-sparse exists at scale.** HySparse2 trains 80B-A3B for **~500B tokens at 32k** with token-level selection (128 forced local + 1,024 global tokens) and — critically — "supports native end-to-end training **without a separate indexer or an auxiliary distillation loss**" **[T2, [[hysparse2-hybrid-sparse-attention]]]**. That is the structural advantage over CSA2/DSA, which requires the clipped-log KL distillation against main attention ([[dsa_referencepy]] `ref_indexer_backward`, [[deepseek-sparse-attention-dsa-nvidia-cudnn]] §8).

**The indexer is the bottleneck, not the sparse MLA, and this is measured on A100.** HISA: "With 2048 selected tokens, the Sparse MLA operator consistently costs about **1.6 ms**, while the indexer reaches **5.6 ms** at 64K context length. This suggests that the main performance bottleneck in DSA lies in the indexer rather than in Sparse MLA itself." All comparisons "conducted on an **NVIDIA A100 GPU**" with TileLang kernels **[T2, [[published-as-a-conference-paper-at-colm-2026]] §5.1]**. This is the closest thing in the corpus to a sm_80 measurement of a DSA indexer — and it is inference-shaped (query_len=1024, forward only).

---

## Committed position

**CSA2 at 32k on this box is not an experiment — it is a kernel-porting project, and the repo's "worth one experiment, most likely to disappoint" verdict should be replaced with "not runnable; run NSA instead."** The load-bearing reason is architecture support, not memory: the indexer's 64–128 GiB score tensor is a 32×-reducible implementation artifact (measured down to 0.40 GB on H200), so the memory objection the locus calls "hard" is soft; but every training-capable DSA kernel in existence is Hopper-or-newer — cuDNN DSA SM90/SM100, FlashMLA now `sm_100a`/`sm_103a` + Ascend 950 with Hopper explicitly dropped, TileLang's wgmma-based stack benchmarked on H800/H200, and StreamIndex's Triton fwd+bwd stack "Tested on H100 / H200 (Hopper, `sm_90a`)" with A100 validation deferred to future work — while the only sm_80 path that exists is vLLM's inference-only `TRITON_MLA_SPARSE`, which has no backward and is acknowledged to be register-overflow-slow. Meanwhile the payoff arithmetic says the CSA2 arm would be *negative-value even if it ran*: LongCat's 0.83× speedup ratio at 32K means CSA2 makes the attention layer **1.20× slower** than dense MLA, which at a 52–72% attention share is a **+10.7% to +14.7% step-time penalty**; NSA's measured 6.3×/3.4× at 32k on A100 implies a **4.02× attention-time reduction**, which is a **−39.1% to −54.1% step-time saving**. That is a 50–69 percentage-point swing between the two arms, and it is decided by gather granularity (token-granular = 4.5% of peak HBM bandwidth; 64-token blocks = coalesced), not by the candidate cap that both the repo and the locus fixate on. **Run NSA at 32k with a 90/10 or 95/5 dense-attention hybrid; keep MoBA as the flash-attn-2.6.3 fallback if NSA's Triton build fails on sm_80; drop CSA2 from the experiment program entirely until someone publishes an sm_80 sparse-MLA backward.**

- **Position:** CSA2/DSA is not runnable for training on sm_80 at any context length, and at 32k it would be a net step-time loss even if it were; NSA is the correct sparse-attention arm because it is the only family with a measured-on-sm_80 (8×A100) forward+backward speedup at 32k (6.3×/3.4×) and a quality win over full attention.
- **Confidence:** **high** (>80%) on the substitution and on "no sm_80 DSA *training* path exists today" — five independent stacks all confirm Hopper-or-newer for training, and the one sm_80 path is inference-only by construction. **Medium** (50–80%) on "CSA2 could not be made to run on sm_80 within a few weeks" — StreamIndex's Triton indexer + `sparse_attn_bwd.py` are arch-portable in principle, and Triton does target sm_80, so a determined port might work; what is *not* uncertain is that its attention forward reaches only 8% of peak at `d_h=512`, which would make the ported kernel unusable for training regardless.
- **Boundary conditions:** This holds for a 5× CMP 170HX box (GA100, sm_80, 70 SMs, 250 W, 64 GB, Gen2 x4, no P2P) running a 32k-context training arm at 1M-token global batches. It does *not* generalise to Hopper/Blackwell hosts, where cuDNN DSA and FlashMLA are available and the CSA2 arm becomes a genuine experiment. It also does not generalise to *inference*: vLLM's `TRITON_MLA_SPARSE` runs DSA on sm_80 today, so CSA2 serving on this box is feasible even though CSA2 training is not — a distinction the repo's research.md §4 blurs by citing the 890 B/token inference KV figure as evidence for a training-side win.
- **What would change this position:** (1) A StreamIndex (or equivalent Triton) run at S=32,768 on a CMP 170HX or A100 reporting peak HBM ≤ 8 GB **and** indexer+attention fwd+bwd wall-clock within 2× of dense FlashAttention-2 — that would make CSA2 runnable and flip the recommendation to "run both arms". (2) TileLang's `sparse_mla_bwd.py` compiling and running on sm_80 at ≥30% of peak. (3) A measured NSA kernel on a CMP 170HX showing <1.5× speedup at 32k, which would collapse the 39–54% payoff estimate and make the whole sparse-attention axis marginal. (4) A published CSA2-at-32k training-loss win over dense MLA at matched tokens — LongCat's 0.83× is a latency measurement, not a quality measurement, and if CSA2's quality advantage at 32k were large enough it could justify the porting cost even at a step-time penalty.
- **Evidence weight:** 5 independent stacks document Hopper-or-newer for training DSA **[T2]**; 1 stack (StreamIndex) documents a 32× memory reduction at 32k **[T2]**; 1 stack (vLLM) documents an sm_80 inference forward with no backward **[T2]**; 1 measured-on-sm_80 kernel comparison (NSA on 8×A100) supports the block-sparse substitution **[T2]**; 1 measured-on-vendor-accelerator comparison (LongCat) supports the token-granular loss at 32K **[T2]**; 1 measured-on-A100 indexer-vs-MLA breakdown (HISA) supports "indexer is the bottleneck" **[T2]**; 0 measurements on this box **[T1 absent]**. The 26–36% / 39–54% / +10.7–14.7% payoff figures are **[T3]** arithmetic layered on **[T2]** kernel ratios and **[T3]** attention-share estimates from locus 4.

---

## Open questions

- **Does StreamIndex's Triton stack actually compile and run on sm_80?** The repo says "Tested on H100 / H200 (Hopper, `sm_90a`)" and the paper defers A100 validation to future work. A single `pytest tests/` run on one CMP 170HX would settle it — this is the highest-value cheap measurement in the locus, and it is the one thing that could flip the committed position.
- **What is the CMP 170HX's actual sparse-attention throughput?** No [T1] number exists. The relevant measurement is NSA's Triton kernel at S=32768 on one card, reported as wall-clock ms for fwd and bwd alongside dense FlashAttention-2 on the same card — not TFLOPS, because sparse attention is memory-bound.
- **Does the 70-SM / 250 W part change the NSA ratio?** NSA's 6.3×/3.4× is on 108-SM A100s at 400 W. Both kernels scale with SM count and power, so the *ratio* should hold, but the absolute throughput will be ~0.65× A100 per card. Unverified.
- **Is there a quality cost to NSA at 32k that the 27B/260B-token result hides?** NSA's 0.456-vs-0.443 average is at 27B params and 260B tokens. The repo's program is ~1B active at 9B tokens (locus 6's problem), which is 3–39× short of every published ablation. A 32k NSA arm at 9B tokens may not resolve the delta.
- **Does the 1M-token global batch interact with 32k sequences in a way that changes the memory picture?** At 32k sequences and 1M tokens, each of 5 GPUs holds ~6 sequences. With the chunked indexer at 0.40 GB/sequence that is 2.4 GB — fine. With the materialize path at 128 GiB/sequence it is impossible. Worth stating explicitly in any experiment plan.
- **What does the fabric contribute?** No P2P and Gen2 x4 rule out context parallelism, which is how LongCat reaches 32K on 4 ranks. So the 32k arm must run single-GPU per sequence — exactly the regime where the materialize indexer's memory wall is worst. This is a real fabric↔CSA2 interaction the repo does not name, though it is neutralised by chunking.
- **Is HySparse2's training-native block-sparse a better third arm than MoBA?** It trains 80B-A3B at 32k for 500B tokens with no indexer and no distillation loss — structurally the cleanest fit for a small-budget program. But no kernel availability was established in the corpus, so it is unverified for sm_80.

---

## Sources

**Corpus (read in full):**
1. [[streamindex-memory-bounded-compressed-sparse]] — StreamIndex: Memory-Bounded Compressed Sparse Attention via Streaming Top-k (arXiv 2605.02568)
2. [[github-rightnow-aistreamindex-memory-bounded-compressed-sparse-attention-via-str]] — RightNow-AI/StreamIndex repo README ("Tested on H100 / H200 (Hopper, `sm_90a`)")
3. [[dsa_referencepy]] — cuDNN Frontend DSA reference implementation (`dsa_reference.py`)
4. [[deepseek-sparse-attention-dsa-nvidia-cudnn]] — DeepSeek Sparse Attention (DSA), NVIDIA cuDNN v1.28.0 docs
5. [[deepseek-v41-flash]] — DeepSeek-V4.1-Flash (arXiv 2609.19969)
6. [[longcat-sparse-attention-taming-the-lightning-via]] — LongCat Sparse Attention (LSA)
7. [[native-sparse-attention-hardware-aligned-and-natively]] — Native Sparse Attention (NSA, arXiv 2502.11089)
8. [[moba-mixture-of-block-attention-for]] — MoBA: Mixture of Block Attention (arXiv 2502.13189)
9. [[hysparse2-hybrid-sparse-attention]] — HySparse2: Hybrid Sparse Attention
10. [[published-as-a-conference-paper-at-colm-2026]] — HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention (COLM 2026)
11. [[indexcache-accelerating-sparse-attention-via-cross-layer]] — IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse

**Newly fetched (4 of 4 budget):**
12. [[github-deepseek-aiflashmla-flashmla-efficient-multi-head-latent-attention-kernel]] — deepseek-ai/FlashMLA (sm_100a/sm_103a + Ascend 950; Hopper dropped)
13. [[tilelangexamplesdeepseek_v32-at-main-tile-aitilelang-github]] — tile-ai/tilelang `examples/deepseek_v32` (sparse MLA fwd/bwd, H800/H200)
14. [[github-moonshotaimoba-moba-mixture-of-block-attention-for-long-context-llms-gith]] — MoonshotAI/MoBA (flash-attn 2.6.3, 32K tested)
15. [[attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of]] — vLLM PR #47629, `TRITON_MLA_SPARSE` for SM80/SM121 (inference-only, tested on 8×A800)
