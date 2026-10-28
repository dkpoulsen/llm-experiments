---
title: "Tag: fp8"
id: "_tag-fp8"
type: index
created: 2026-10-27T19:00:34.890399+00:00
updated: 2026-10-27T19:00:34.890399+00:00
---

# Tag: fp8

**3** notes.

- [[coat-compressing-optimizer-states-and-activation-for-memory-efficient-fp8-traini]] — COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training — COAT (Xi, Cai, Zhu, Lu, Keutzer, Chen, Han; UC Berkeley / NVIDIA / MIT / Tsinghua; arXiv 2410.19313, NVlabs) is the FP8 training framework AGoQ benchmarks against. Two innovations: Dynamic Range Expansion, which shifts optimizer-state distributions to better fill the FP8 representable range and so cuts quantisation error; and Mixed-Granularity Activation Quantization, which combines per-tensor and per-group strategies for activation memory.

Results: 1.54x end-to-end training-memory reduction versus BF16 with nearly lossless quality across LLM pretraining, fine-tuning and VLM training; 1.43x end-to-end training speedup versus BF16, on par with or better than Transformer Engine. It enables full-parameter training of large models on fewer GPUs and "facilitates doubling the batch size in distributed training settings". The worked example is Llama-2-13B on 8x80GB H100 under FSDP.

Relevance here: COAT is the memory-side counterpart to the communication-side FP8 work, and its 1.54x memory reduction is the strongest published lever for fitting a larger trainable parameter count per card. But like FP8-LM it is Hopper-native - the paper explicitly ties FP8 to "the advent of Nvidia's H100 GPU" - so on GA100/sm_80 the memory savings would require emulation. Its "double the batch size" result is directly relevant to the repo's large-global-batch DDP strategy, since batch size is the only lever the repo has for amortising communication. `review`
- [[fp8-lm-training-fp8-large-language-models]] — FP8-LM: Training FP8 Large Language Models — FP8-LM (Peng et al., Microsoft Azure/Research; arXiv 2310.18313) is the primary FP8 training framework paper and the baseline GeoFP8/AGoQ build on. It pushes FP8 into weights, gradients, optimizer states and collective communication in three incremental levels, and reports on GPT-175B over Azure NDv5 H100 nodes: 39% reduction in real memory use and 75% faster than the BF16 Megatron-LM baseline, 37% faster than NVIDIA Transformer Engine, with 42% less memory than TE.

The communication result is the load-bearing one for this run: 8-bit gradients for all-reduce give a theoretical 75% reduction in communication cost versus the mainstream 32-bit scheme (BF16 mixed precision computes gradients in 16-bit but still all-reduces in 32-bit), and the observed practical reduction during GPT training is 63% to 65%. Memory reduction versus BF16 is 28% to 39% (29% for GPT-7B, 39% for GPT-175B).

The technical obstacle they solve is overflow/underflow in low-bit all-reduce: pre-scaling divides before summation (underflow), post-scaling sums first then divides (keeps values near the FP8 max). They add distributed and automated scaling with a single shared scalar per gradient tensor, computed as a global minimum across GPUs, so that standard NCCL all-reduce can still be used - NCCL itself "lacks the capability of performing all-reduce operation considering the additional tensor-wise scaling factors".

HARDWARE CAVEAT that decides feasibility on the CMP 170HX: FP8 here is Hopper-native. The paper frames FP8 as arriving "with the release of Nvidia H100 GPU", and the only prior usable framework (NVIDIA Transformer Engine) applies FP8 solely to GEMM. GA100/sm_80 has no FP8 tensor-core path, so the 63-65% communication saving is not directly available on a CMP 170HX without emulation. Training config: 100B tokens, batch size 4M tokens, sequence length 2048, Adam with beta1=0.9/beta2=0.95, weight decay 0.1. `review`
- [[geofp8-geometry-aware-fp8-gradient]] — GeoFP8: Geometry-Aware FP8 Gradient — GeoFP8 (Wang et al., Rutgers/NCSU; arXiv 2607.07494) attacks low-precision gradient communication from the geometry side rather than the format side. Diagnosis: gradients are highly anisotropic, so axis-aligned (Euclidean) FP8 scaling incurs direction-dependent distortion; transforming gradients into a near-isotropic space before quantisation - using a block-diagonal K-FAC local metric, applied selectively and approximated with a rank-32 input-side low-rank factor - makes the FP8 representation substantially more faithful. GeoFP8 deliberately does not change the optimizer, training recipe, collective, or low-precision format; only the coordinate system used for the low-precision communication.

Numbers: GeoFP8 reduces the error between FP8-averaged and FP32-averaged gradients by up to 67.4% over the Euclidean baseline; it cuts gradient communication volume by 75.0% relative to FP32, yielding a 7.6% end-to-end pretraining-time reduction for Llama-600M on 64 NVIDIA GH200 Superchips. Direct Euclidean FP8 at the same scale gives a 10.79% end-to-end reduction, i.e. GeoFP8 deliberately trades ~3 points of speed for downstream-task preservation. Experiments use 32 GPUs, global batch 512, micro-batch 4, Muon optimizer, lr 5e-4, on TACC Vista GH200 nodes (96 GB HBM3 per GPU, NVLink-coupled Grace-Hopper).

Two caveats matter for the CMP 170HX box. (1) The 7.6% end-to-end figure is measured on a fast NVLink fabric where gradient communication is a minority of step time; on a ~1 GB/s PCIe fabric the same 75% payload cut would translate into a much larger fraction of step time, so the paper's end-to-end number understates the potential gain here. (2) FP8 is not natively supported on GA100/sm_80 - the CMP 170HX is Ampere-generation silicon without FP8 tensor-core paths, so an FP8 gradient path would need software emulation whose cost is not accounted for in the GH200 measurements. Their BF16-vs-FP32 comparison is also relevant to the repo's bf16 gradient plan: BF16 gradient communication "matches FP32 on 7 out of 14 downstream tasks" but with slightly lower absolute values, which is why they kept FP32 as the reference. `review`

---
*Auto-generated by hyperresearch. Do not edit manually.*
