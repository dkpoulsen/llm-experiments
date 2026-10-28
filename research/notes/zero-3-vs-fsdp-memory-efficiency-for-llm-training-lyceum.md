---
title: 'ZeRO-3 vs FSDP: Memory Efficiency for LLM Training | Lyceum'
id: zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum
tags:
- cmp170hx-training-feasibility-a0d4fc
- fsdp
- zero
- memory-accounting
- communication-volume
- vendor-blog
created: '2026-10-27T11:27:06.786717Z'
updated: '2026-10-27T19:00:31.518224Z'
source: https://lyceum.technology/magazine/zero-3-vs-fsdp-memory-efficiency/
source_domain: lyceum.technology
fetched_at: '2026-10-27T11:27:06.168121Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: blog
deprecated: false
summary: 'Vendor explainer (Lyceum, a European GPU cloud) comparing DeepSpeed ZeRO-3
  and PyTorch FSDP on memory and communication. The usable content is the standard
  mixed-precision accounting it repeats: a 7B model holds ~14 GB of bf16/fp16 weights,
  Adam adds 12 bytes/param (4 fp32 master copy + 4 momentum + 4 variance), fp16 gradients
  add 2 bytes/param, so model states alone are ~16 bytes/param and a 7B model needs
  >100 GB of VRAM — more than an A100 40GB. DDP replicates all of that on every GPU
  (7/8 of aggregate memory wasted at 8 GPUs); ZeRO-3 and FSDP both shard params+grads+optimizer
  states to O(M/N) and both raise total communication volume by roughly 1.5x versus
  plain DDP, because they add an all-gather before each forward/backward and a reduce-scatter
  of gradients. Qualitative claims: ZeRO-3 wins in over-subscription and has richer
  CPU/NVMe offload (ZeRO-Infinity), FSDP tends to higher TFLOPS/GPU when the model
  already fits, FSDP2 (torch.distributed.fsdp.fully_shard) shards each parameter on
  dim 0 as a DTensor and relaxes constraints around frozen parameters and sharded
  state dicts, and CPU offload costs 2-10x throughput. Relevance: this is the arithmetic
  behind the repo''s ''7B fully trainable needs ~112 GB/card'' and ''FSDP costs ~1.5x
  more communication than DDP'' claims. Tier caveat: it is a marketing-adjacent blog
  with zero measurements of its own — the 16 bytes/param figure traces to the ZeRO
  paper (arXiv 1910.02054) and the 1.5x figure to the FSDP PVLDB paper, both already
  in the vault; cite those, not this.'
utility_score: 11.0
---

[Skip to content](https://lyceum.technology/magazine/zero-3-vs-fsdp-memory-efficiency/#main-content)
[Download brand assets ZIP](https://lyceum.technology/images/brand/lyceum-brand-assets.zip)[Download logo SVG](https://lyceum.technology/images/brand/lyceum-wordmark.svg)[Download white logo SVG](https://lyceum.technology/images/brand/lyceum-wordmark-white.svg)[Download logo PNG](https://lyceum.technology/images/brand/lyceum-wordmark.png)[Download white logo PNG](https://lyceum.technology/images/brand/lyceum-wordmark-white.png)[Press and company facts ↗](https://lyceum.technology/company/press/)
Contents
Training modern large language models (LLMs) has fundamentally changed the requirements for GPU memory management. Traditional Distributed Data Parallel (DDP) methods replicate the entire model state on every GPU, which quickly leads to Out-of-Memory (OOM) errors as model parameters exceed the capacity of a single card. To solve this, engineers turn to sharding techniques like Microsoft's DeepSpeed ZeRO-3 and PyTorch's Fully Sharded Data Parallel (FSDP). Both frameworks aim to reduce the memory footprint by distributing parameters, gradients, and optimizer states across the cluster. However, choosing between them requires a nuanced understanding of their communication patterns, offloading capabilities, and integration complexity. This article explores the technical trade-offs to help you maximize efficiency on sovereign GPU infrastructure. Lyceum publishes this article and competes in this market.
## The Memory Wall and the Need for Sharding
The primary bottleneck in training large-scale models is the memory footprint of the model states. For a model with 7 billion parameters, the weights alone occupy approximately 14 GB in half-precision (FP16/BF16). However, the [total memory required during training](https://lyceum.technology/magazine/gradient-checkpointing-memory-savings/) is significantly higher. Optimizer states, such as those used in Adam, require an additional 12 bytes per parameter under the [ZeRO paper's mixed-precision accounting](https://arxiv.org/abs/1910.02054) (4 bytes for the fp32 master copy of the weights, 4 bytes for the momentum, and 4 bytes for the variance). Gradients in fp16 add another 2 bytes per parameter, putting the model states alone at 16 bytes per parameter. When combined with activations and temporary buffers, a 7B model can require more than 100 GB of VRAM, exceeding the capacity of common GPUs like the A100 40GB.
### Why Standard DDP Falls Short
Standard Distributed Data Parallel (DDP) is inefficient because it replicates these states across every GPU in the cluster. If you have 8 GPUs, you are essentially wasting 7/8ths of your total aggregate memory on redundant data. This redundancy prevents researchers from scaling to larger models without resorting to complex model parallelism or pipeline parallelism. ZeRO (Zero Redundancy Optimizer) was introduced to eliminate this redundancy by sharding the states. By distributing the data, the memory requirement per GPU scales as 1/N, where N is the number of GPUs. This allows for training models that are orders of magnitude larger than what a single device could hold. Understanding how ZeRO-3 and FSDP implement this sharding is critical for any team moving beyond hyperscaler credits and into production-grade AI development on platforms like Lyceum, where hardware selection is automated for cost and performance.
## DeepSpeed ZeRO-3: Architecture and Mechanics
DeepSpeed ZeRO-3 is the most advanced stage of the Zero Redundancy Optimizer. It builds upon Stage 1 (sharding optimizer states) and Stage 2 (sharding gradients) by also sharding the model parameters themselves. In a ZeRO-3 configuration, no single GPU holds the full set of weights for any given layer for the entire training step; the parameters are collected during the forward and backward passes and re-partitioned afterwards. Instead, parameters are fetched from other GPUs just-in-time for the forward and backward passes and then discarded (released) immediately after use. This process is managed through a series of collective communication calls, specifically all-gather and reduce-scatter operations.
One of the defining features of ZeRO-3 is its integration with DeepSpeed's ecosystem, which includes advanced features like ZeRO-Infinity. This allows for offloading model states not just to CPU memory, but also to NVMe storage, effectively enabling the training of trillion-parameter models on limited GPU resources. The framework uses a sophisticated prefetching mechanism to hide the latency of these data transfers. By predicting which parameters will be needed next, ZeRO-3 can begin the all-gather operation while the current layer is still computing. This overlapping of communication and computation is vital for maintaining high hardware utilization. For teams operating in EU-sovereign environments like Lyceum's European data centers in Paris and Finland, ZeRO-3 provides a robust, battle-tested way to handle massive datasets while ensuring that the orchestration layer handles the underlying hardware complexity smoothly.
## PyTorch FSDP: The Native Sharding Alternative
PyTorch FSDP reaches the same place from inside the framework. The original `FullyShardedDataParallel` wrapper groups a module's parameters into flat units and shards each unit across the data-parallel workers. [PyTorch's reference](https://docs.pytorch.org/docs/2.13/fsdp.html) sets out the default `FULL_SHARD` lifecycle plainly: parameters, gradients and optimizer states are all sharded, parameters are unsharded by an all-gather before the forward pass and resharded after it, all-gathered again for the backward pass, and gradients are reduce-scattered so each rank keeps only its own slice. Milder strategies sit alongside it, such as `SHARD_GRAD_OP`, which shards gradients and optimizer states but leaves parameters gathered between passes.
The newer FSDP2 entry point, [`torch.distributed.fsdp.fully_shard`](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html), changes how that sharding is represented rather than what it achieves. Where FSDP1 flattens, concatenates and chunks a group of tensors together, FSDP2 chunks each parameter on dimension 0 across the workers and holds it as a DTensor. PyTorch documents the result as a more intuitive account of which slice of which tensor a rank owns, with relaxed constraints around frozen parameters and sharded state dicts that save without extra communication. In both APIs the unit of sharding comes from the wrapping policy, and that choice is where most of the tuning happens.
## Direct Memory Efficiency Comparison
When comparing the memory efficiency of ZeRO-3 and FSDP, the theoretical savings are identical as both follow the same mathematical principles of sharding. The memory footprint for model states on each GPU is reduced from $O(M)$ to $O(M/N)$, where $M$ is the model size and $N$ is the number of GPUs. However, practical differences arise in how each framework handles auxiliary memory, such as buffers, temporary tensors, and fragmentation. DeepSpeed ZeRO-3 often has a slight edge in extremely constrained environments due to its highly optimized memory allocator and its ability to aggressively offload almost everything to the CPU or NVMe.
FSDP, on the other hand, can sometimes suffer from higher memory fragmentation if the `auto_wrap_policy` is not configured correctly. If the wrapped units are too large, the temporary memory required to hold the gathered parameters during computation can cause OOM errors. Conversely, if the units are too small, the communication overhead becomes the dominant factor, leading to poor GPU utilization. Lyceum's platform helps here by predicting the memory footprint of single-node jobs before they run, which narrows the search for a workable FSDP wrapping policy or ZeRO-3 configuration; multi-node runs still need a profiling pass. In many benchmarks, FSDP shows slightly better performance in terms of TFLOPS per GPU when the model fits comfortably within the aggregate VRAM of the cluster, while ZeRO-3 remains the king of 'over-subscription' where the model size vastly exceeds the total VRAM.
## Communication Overhead and Scaling Laws
The cost of memory efficiency is increased communication. Both ZeRO-3 and FSDP increase the total volume of data moved across the network by approximately 50% compared to standard DDP. Specifically, they require an all-gather operation to collect parameters before computation and a reduce-scatter operation to synchronize and shard gradients after computation. This makes the performance of these frameworks highly dependent on the interconnect bandwidth of the cluster. In a multi-node setup, the bottleneck is almost always the inter-node network (e.g., InfiniBand or 100GbE+ Ethernet).
ZeRO-3 manages this overhead through its 'communication overlap' features, which are highly tunable via the `deepspeed_config.json`. Parameters like `overlap_comm` and `allgather_bucket_size` allow engineers to balance the size of communication chunks. FSDP uses a similar concept with its `limit_all_gathers` and `backward_prefetch` settings. In practice, FSDP's integration with the PyTorch autograd engine can sometimes lead to more efficient overlapping of the backward pass. However, DeepSpeed's long history of optimization for massive clusters often gives it the advantage in complex, multi-node environments. For [European scaleups using Lyceum](https://lyceum.technology/magazine/runpod-alternatives-europe/), the absence of egress fees and high-performance networking in its European data centers mitigate some of these communication costs, making both frameworks viable for large-scale training.
## CPU Offloading and NVMe Integration
When GPU memory is not enough, offloading to system RAM (CPU) or disk (NVMe) becomes necessary. DeepSpeed ZeRO-3 is the pioneer in this space. Its ZeRO-Offload and ZeRO-Infinity modules allow for the optimizer states and parameters to reside in CPU memory, only moving to the GPU for the actual computation. This can increase the effective capacity of a single GPU by 10x or more. The trade-off is a significant hit to training speed, as the PCIe bus becomes the bottleneck. However, for fine-tuning large models on a budget, this is an invaluable feature.
PyTorch FSDP also supports CPU offloading through the `cpu_offload` parameter in the `FullyShardedDataParallel` constructor, though this FSDP1 wrapper API is deprecated in favour of FSDP2 (`torch.distributed.fsdp.fully_shard`). While functional, it is generally less feature-rich than DeepSpeed's implementation. For example, FSDP's offloading is primarily focused on the parameters and gradients, whereas DeepSpeed can offload the entire optimizer logic to the CPU (using highly optimized CPU kernels for Adam). This means that with DeepSpeed, the CPU not only stores the data but also performs the weight updates, freeing up the GPU to focus entirely on the forward and backward passes. For teams using Lyceum's auto hardware selection, the platform can help determine if a high-VRAM GPU (like an H100) is more cost-effective than a lower-spec GPU paired with aggressive DeepSpeed offloading, based on the total cost of compute (TCC).
## Practical Implementation: Code and Configuration
Implementing ZeRO-3 typically involves creating a configuration file and initializing the model using the `deepspeed.initialize()` wrapper. This abstracts much of the complexity but requires the user to adhere to the DeepSpeed API. A typical config might look like this:

```
{
 "zero_optimization": {
 "stage": 3,
 "offload_optimizer": { "device": "cpu" },
 "overlap_comm": true,
 "contiguous_gradients": true
 }
 }
```

FSDP is implemented directly in the training script using PyTorch wrappers. This is often preferred by engineers who want to keep their code 'pure' PyTorch. The implementation involves wrapping the model: `model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)`. While this looks simpler, the complexity lies in defining the `auto_wrap_policy` to ensure that the model is sharded at the correct granularity. If you wrap the entire model as one block, you get no memory savings; if you wrap every individual linear layer, the communication overhead will destroy performance. Finding the 'Goldilocks' zone is the key challenge. Lyceum's CLI and VS Code extension keep that iteration loop local: you edit the wrapping policy in your own editor and submit the run with a command such as `lyceum python run train.py --machine gpu.h100`, so a change in sharding granularity is measured rather than guessed at.
## Choosing the Right Strategy on Lyceum
The choice between ZeRO-3 and FSDP often comes down to the specific requirements of the project and the existing codebase. If you are starting a new project and want the tightest integration with the PyTorch ecosystem, FSDP is an excellent choice. It is increasingly becoming the standard for PyTorch-based LLM training. However, if you are working with extremely large models that require advanced offloading or if you are using a framework like Hugging Face Accelerate that has deep DeepSpeed integration, ZeRO-3 remains a powerful contender. 
Across the industry, average [GPU utilization](https://lyceum.technology/magazine/gpu-utilization-too-low-how-to-fix/) is widely reported to be low, often well under half of available capacity. This waste is often due to sub-optimal sharding configurations or poor hardware-software alignment. Our platform addresses this by offering workload-aware pricing and runtime and memory-footprint predictions for single-node jobs before they start. Whether you choose ZeRO-3 or FSDP, Lyceum's European cloud keeps processing in its European data centers, supporting the compliance and security requirements of European enterprises. By automating the hardware selection process, Lyceum ensures that your chosen sharding strategy runs on the most cost-effective and performance-optimized instances, whether that is a cluster of A100s or H100s in its European data centers. This allows ML engineers to focus on model architecture rather than infrastructure plumbing.
## Sources
[1] [DeepSpeed ZeRO: New System Optimizations](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/); [2] [Hugging Face: Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many); [3] [arXiv: PyTorch FSDP, Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
## Frequently asked questions
### What is the main difference between ZeRO-2 and ZeRO-3?
ZeRO-2 shards optimizer states and gradients but keeps a full copy of the model parameters on each GPU. ZeRO-3 goes a step further by sharding the parameters as well. This means ZeRO-3 provides significantly higher memory savings but introduces more communication overhead because parameters must be gathered before every forward and backward pass.
### When should I choose FSDP over ZeRO-3?
Choose FSDP if you want a native PyTorch experience, are using the latest PyTorch features like torch.compile, or need a highly customizable sharding policy. FSDP is also generally easier to use if you are already comfortable with PyTorch's internal mechanics and want to avoid adding an external dependency like DeepSpeed.
### How does communication overhead impact scaling in these frameworks?
Both frameworks increase the total communication volume by roughly 1.5x compared to standard DDP. This means that as you scale to more GPUs, the speed of your network interconnect (e.g., NVLink, InfiniBand) becomes a major limiting factor on training throughput. Without high-speed networking, the time spent waiting for data transfers can outweigh the benefits of larger batch sizes.
### What role does NVLink play in ZeRO-3 and FSDP performance?
NVLink provides high-bandwidth, low-latency communication between GPUs on the same node. Since ZeRO-3 and FSDP rely heavily on all-gather and reduce-scatter operations, NVLink significantly reduces the time these operations take. On platforms like Lyceum, utilizing hardware with NVLink is essential for maintaining high GPU utilization when using full sharding strategies.
### How do I handle OOM errors even when using ZeRO-3 or FSDP?
If you still hit OOM errors, consider: 1) Increasing the number of GPUs to further shard the states. 2) Enabling activation checkpointing to trade compute for memory. 3) Using CPU offloading for optimizer states. 4) Reducing the micro-batch size. 5) Ensuring your wrapping policy (for FSDP) or bucket sizes (for ZeRO-3) are not creating excessively large temporary buffers.
### Is there a performance penalty for using CPU offloading?
Yes, there is a significant performance penalty because the PCIe bus is much slower than GPU VRAM or NVLink. Training speed can drop by 2x to 10x depending on the model size and the amount of data being offloaded. However, it allows you to train models that would otherwise be impossible to fit on your available GPU hardware.
## Related resources
  * [How Many GPUs for Model Training? A Practical Scaling Guide→](https://lyceum.technology/magazine/how-many-gpus-for-model-training/)
  * [Solving OOM Errors in 70B Model Fine-Tuning→](https://lyceum.technology/magazine/oom-error-70b-model-fine-tuning/)
  * [Deploy DeepSeek R1 on European GPU Cloud: VRAM, Costs, and Compliance→](https://lyceum.technology/magazine/deploy-deepseek-r1-gpu-cloud-europe/)


GPU pricing
See GPU rates and reserved capacity options.
[View GPU pricing→](https://lyceum.technology/resources/pricing/#compute)
Lyceum Technology
## Related articles
[All articles→](https://lyceum.technology/magazine/)
Keep GPU jobs running · Distributed training
### [Training MoE Models: Preventing Expert Collapse & Balancing Load](https://lyceum.technology/magazine/training-moe-models-preventing-expert-collapse-balancing/)
Training a Mixture-of-Experts model introduces a critical bottleneck: if the router favors a small subset of experts, the starved parameters waste memory while overloaded ones halt the distributed run. Here is how to prevent routing collapse and balance MoE loads effectively.
Maximilian Niroomand24 September 2026 · 15 min read
Keep GPU jobs running · Orchestration
### [Optimize Slurm GPU Allocation for High Performance AI Workloads](https://lyceum.technology/magazine/optimize-slurm-gpu-allocation/)
GPU scarcity and high operational costs make inefficient scheduling a terminal risk for AI startups. We break down how to tune Slurm for maximum throughput while maintaining the data sovereignty your enterprise clients demand.
Caspar Lehmkühler16 January 2026 · 7 min read
Lower GPU cost
### [Migrating from AWS to dedicated GPUs: a performance and cost guide](https://lyceum.technology/magazine/migrate-from-aws-to-dedicated-gpu/)
Migrating from AWS to dedicated GPUs means moving your training and inference to GPUs that another provider reserves for you. Do it in five steps: package the code, copy the data, test on one GPU, move training, then switch inference. Before you move any data, ask AWS to cancel its fee for data leaving AWS. It does so only if you move all your data, or all of one service, off AWS. Without that, moving 50 TB out of AWS's Stockholm region costs about $4,394 in our example.
Justus Amen13 February 2026 · 8 min read
## Your next workload starts here
[Talk to an expert](https://lyceum.technology/talk-to-an-expert/)
cookies.txt
We use cookies to improve your experience and for marketing. Read our [cookie policy](https://lyceum.technology/legal/privacy/).
Reject allAccept all
