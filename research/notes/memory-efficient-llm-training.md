---
title: Memory Efficient LLM Training
id: memory-efficient-llm-training
tags:
- cmp170hx-training-feasibility-a0d4fc
- memory-budget
- zero-stages
- fsdp
- survey
created: '2026-10-27T11:24:42.390132Z'
updated: '2026-10-27T19:01:52.122550Z'
source: https://archiesengupta.com/blog/memory-efficient-training-deep-dive
source_domain: archiesengupta.com
fetched_at: '2026-10-27T11:24:41.773214Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'Archie Sengupta blog, Memory Efficient LLM Training (~5,100 words) - practitioner
  survey of the training-memory stack, useful mainly as an arithmetic reference. Per-parameter
  accounting: fp32 weights 4P, fp32 grads 4P, Adam first plus second moment 8P, bf16
  copy 2P; activations grow quadratically with sequence length and linearly with batch.
  Gradient checkpointing: full checkpointing costs ~30-40% compute overhead, every-K
  checkpointing gives ~Kx memory for 20-30% overhead. Gradient accumulation: memory
  stays at micro-batch size while AllReduce frequency falls with accumulation. ZeRO
  stages: per-GPU memory 2P + 8P/N (stage 1 and 2) and 12P/N (stage 3), with stage
  3 costing O(N) AllGathers per layer per forward/backward pass, which is why ZeRO-3
  is normally paired with tensor parallelism inside a node. ZeRO-Offload and FSDP
  CPU offload cost 10-20% to 30-50% slowdown. Its topology table assumes NVLink at
  900 GB/s for TP and PP and InfiniBand at 50 GB/s for DP - there is no PCIe-Gen2-only
  case anywhere in the piece, and its only fabric-relevant statement is that TP must
  stay within a node (NVLink required). Internal inconsistency worth flagging: it
  states Adam needs 8 bytes per parameter but its total equation is P x (4+4+4+2)
  = 14 bytes, i.e. only 4 bytes of optimizer state; the repo''s 112 GB / 7B figure
  uses the correct 12 bytes (fp32 m + fp32 v + fp32 master).'
utility_score: 12.0
---

[← Back to Writing](https://archiesengupta.com/blog)
Training large language models at scale is fundamentally a memory management problem. A single 70B parameter model in FP32 requires 280 GB just to store the weights. Add gradients, optimizer states, activations, and the KV cache, and you're looking at 1 TB or more of memory per replica. Modern GPUs like the H100 offer only 80 GB of HBM. The gap between what's required and what's available has driven remarkable innovation in memory-efficient training techniques.
This deep dive examines every significant approach to reducing training memory footprint, from fundamental numerical optimizations to sophisticated distributed training strategies. We'll build understanding through the memory breakdown, then systematically explore each technique with architectural depth and practical trade-offs.
## The Memory Problem in LLM Training
### Where Does Memory Go?
When training a large language model, GPU memory serves multiple concurrent purposes. Understanding what consumes each byte is the first step toward optimization.
**Model Weights** : The parameters of the neural network itself. For a model with P parameters:
  * FP32 (full precision): 4P bytes
  * FP16/BF16 (half precision): 2P bytes
  * INT8: 1P bytes
  * INT4: 0.5P bytes


For a 70B parameter model, weights alone consume 140 GB in BF16.
**Gradients** : The partial derivatives of the loss with respect to each parameter, computed during backpropagation. These must be stored for the optimizer update:
  * FP32 gradients: 4P bytes (even when training in FP16/BF16)
  * Gradient memory typically matches weight memory in precision


**Optimizer States** : The Adam optimizer stores two additional values per parameter:
  * First moment estimate (momentum): 4P bytes
  * Second moment estimate (variance): 4P bytes


This is why Adam is memory-intensive: 8 bytes per parameter just for optimizer state, on top of weights and gradients.
**Activations** : The intermediate tensor values computed during the forward pass. These must be stored to compute gradients during backpropagation. For a transformer:
  * Attention scores: batch_size × num_heads × seq_len × seq_len
  * MLP intermediates: batch_size × seq_len × hidden_dim × expansion_factor
  * Activation memory grows quadratically with sequence length and linearly with batch size


For a 70B model with batch_size=1, seq_len=2048, activations can exceed 100 GB.
**KV Cache** : During training, the attention mechanism caches key and value tensors:
  * Per layer: 2 × batch_size × num_heads × seq_len × head_dim
  * For 80 layers: roughly 2 × batch_size × seq_len × hidden_dim × bytes_per_param


### The Total Memory Equation
For standard DDP (DistributedDataParallel) training with Adam optimizer:

```
Total Memory = P × (4 + 4 + 4 + 2) + Activations + KV Cache
            = P × 14 bytes + Activations + KV Cache

```

For a 70B model in BF16:

```
Weights:        70B × 2 = 140 GB
Gradients:     70B × 4 = 280 GB
Optimizer:     70B × 8 = 560 GB
Activations:   ~100-200 GB
Total:         ~1 TB per replica

```

This is why single-GPU training of large models is impossible without massive optimization.
## Mixed Precision Training
### The Precision Landscape
Mixed precision training exploits the insight that not all operations require full 32-bit precision. The key formats:
**FP32 (Single Precision)** : 32 bits - 1 sign, 8 exponent, 23 mantissa. The gold standard for training, providing the widest dynamic range.
**FP16 (IEEE Half Precision)** : 16 bits - 1 sign, 5 exponent, 10 mantissa. Range: ±65,504. Precision: ~3 decimal digits.
The critical problem: FP16's limited exponent range causes gradient underflow during backpropagation. Gradients can become smaller than FP16's minimum representable value (~10^-7), resulting in zeros and training failure.
**BF16 (Brain Float)** : 16 bits - 1 sign, 8 exponent, 7 mantissa. Range: ±3.4×10^38 (same as FP32!). Precision: ~2 decimal digits.
The 8-bit exponent is transformative: it matches FP32's dynamic range, eliminating gradient underflow concerns. BF16 has become the default for LLM pre-training at every major lab.
**TF32 (Tensor Float-32)** : NVIDIA's format - 19 bits. Uses FP32 mantissa with FP16 exponent range. Automatically applied on Ampere+ GPUs without code changes.
**FP8 (8-bit Float)** : Emerging format with 1 sign, 5 exponent, 2 mantissa (E4M3) or 1 sign, 4 exponent, 3 mantissa (E5M2). Used in Hopper GPUs for transformer GEMMs. Provides 2x memory savings over BF16.
### Automatic Mixed Precision (AMP)
AMP automatically casts operations to FP16/BF16 while maintaining FP32 master weights:

```
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

```

The key components:
**Forward Pass** : Operations run in FP16/BF16 automatically. Memory: reduced by 2x.
**Backward Pass** : Gradients computed in FP16/BF16, then scaled up.
**Master Weights** : Model weights are maintained in FP32 for stable updates. However, many implementations now use FP16 weights with separate FP32 accumulators with negligible accuracy difference.
### Gradient Scaling
FP16 gradients are too small to use directly. The GradScaler addresses this:
  1. **Forward/Backward** : Compute loss in FP16
  2. **Scale** : Multiply loss by a large factor (e.g., 65536)
  3. **Backward** : Gradient values are now in a representable range
  4. **Unscale** : Divide gradients by the scale factor
  5. **Clip** : Clip gradients to prevent overflow in FP16 optimizer update
  6. **Update** : Apply optimizer step in FP16


The scaler dynamically adjusts the scale factor based on whether gradients overflowed:

```
if overflow_detected:
    scale_factor *= 0.5
else:
    scale_factor *= 1.01  # Gradually increase

```

With BF16, gradient scaling is typically unnecessary since the dynamic range accommodates gradients directly.
## Gradient Checkpointing (Activation Checkpointing)
### The Compute-Memory Trade-off
Gradient checkpointing (also called activation checkpointing or recomputation) trades compute for memory. Instead of storing all activations from the forward pass, we store only a subset and recompute the rest during backpropagation.
**Without Checkpointing** :
  * Forward: Store all activations
  * Backward: Use stored activations directly
  * Memory: O(activations)
  * Compute: 1× forward pass


**With Checkpointing** :
  * Forward: Store checkpointed activations only
  * Backward: Recompute non-checkpointed activations on-demand
  * Memory: O(activations / checkpoint_interval)
  * Compute: 1.3-1.5× forward pass


### Selective Checkpointing Strategies
**Full Checkpointing** : Checkpoint every layer. Maximum memory savings (~linear in number of layers), maximum compute cost (~30-40% overhead).
**Sparse Checkpointing** : Checkpoint every K layers (typically K=2-4). A practical balance:

```
Memory reduction: ~K×
Compute overhead: ~20-30%

```

**Layer-wise Scaling** : Apply different checkpoint intervals to different parts of the model. Attention layers have higher activation memory (O(n²)) than MLPs, so checkpointing them more frequently saves more memory per recompute cost.
**Gradient-Free Checkpointing (PyTorch)** : PyTorch's `torch.utils.checkpoint.checkpoint` automatically handles the backward pass recomputation:

```
from torch.utils.checkpoint import checkpoint_sequential

# Checkpoint every 4 layers
model = checkpoint_sequential(module_list, chunks=4, input_args)

```

### Memory Estimation with Checkpointing
For a transformer with checkpoint_interval = K:

```
Activation memory ≈ (Activations per layer) × (Layers / K) × checkpoint_overhead

```

For a 70B model (80 layers), checkpointing every 4 layers:

```
Activation memory ≈ 100 GB / 4 = 25 GB
Compute overhead: ~20-25%

```

### When to Use Checkpointing
Checkpointing is essential when:
  * Training with long sequences (activation memory dominates)
  * Batch size must be large (activations scale with batch)
  * GPU memory is the bottleneck, not compute


Checkpointing may not help when:
  * Model weights dominate memory (very large models)
  * Compute is the bottleneck (already compute-bound)
  * Training speed is critical over memory efficiency


## Gradient Accumulation
### Concept and Memory Benefits
Gradient accumulation simulates larger batch sizes without requiring more memory. Instead of updating weights after each micro-batch, gradients accumulate across multiple micro-batches:

```
# Without accumulation (batch_size = 32)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# With accumulation (effective_batch = 256)
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()  # Accumulates gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

```

### Memory Analysis
Gradient accumulation provides memory benefits because:
  1. **No additional model copies** : Unlike increasing batch size directly, gradient accumulation uses the same model weights
  2. **Gradient storage unchanged** : Gradients accumulate in-place, no additional memory for larger batches
  3. **Activation memory per step** : Stays constant regardless of effective batch size


**The key insight** : Effective batch size = micro_batch_size × accumulation_steps, but memory ≈ micro_batch_size
### Trade-offs  
| Aspect  | Gradient Accumulation  |  
| --- | --- |  
| Memory  | Constant regardless of effective batch  |  
| Throughput  | Slight overhead from optimizer steps  |  
| Convergence  | Identical to true large batch (given proper learning rate scaling)  |  
| Communication  | AllReduce frequency decreases with accumulation  |  
For large models where memory is the constraint, gradient accumulation is essential. Combined with other techniques (ZeRO, checkpointing), it enables effective batch sizes that would otherwise be impossible.
## DeepSpeed ZeRO: Zero Redundancy Optimizer
ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy across data parallel ranks by partitioning (sharding) optimizer states, gradients, and parameters. Originally from Microsoft Research, ZeRO has become fundamental to large-scale LLM training.
### ZeRO Stage 1: Optimizer State Sharding
Each GPU stores:
  * Full model weights (P)
  * Full gradients (P)
  * 1/N of optimizer states (where N = data parallel size)


Memory per GPU:

```
2P (weights + gradients) + 8P/N (optimizer)

```

For 8 GPUs training a 70B model:

```
2 × 140 GB + 560/8 GB = 280 + 70 = 350 GB per GPU

```

vs. standard DDP: 980 GB per GPU
**Communication** : AllReduce for gradients only (same as DDP). Optimizer states don't need communication because each rank has full gradients.
### ZeRO Stage 2: Gradient Sharding
Each GPU stores:
  * Full model weights (P)
  * 1/N of gradients (P/N)
  * 1/N of optimizer states (8P/N)


Memory per GPU:

```
2P + 8P/N

```

For 8 GPUs:

```
280 + 70 = 350 GB (same as Stage 1 for small N)

```

For 64 GPUs:

```
280 + 8.75 = 288.75 GB

```

**Communication** : Additional broadcast of partitioned gradients before optimizer step. Each rank broadcasts its gradient shard, receives others, then discards.
### ZeRO Stage 3: Parameter Sharding
Each GPU stores:
  * 1/N of model weights (P/N)
  * 1/N of gradients (P/N)
  * 1/N of optimizer states (8P/N)


Memory per GPU:

```
12P / N

```

For 8 GPUs:

```
12 × 140 / 8 = 210 GB per GPU

```

For 64 GPUs:

```
12 × 140 / 64 = 26.25 GB per GPU (fits on single H100!)

```

**Communication Pattern** :
  1. **Forward Pass** : AllGather parameter shards → forward → discard non-local parameters
  2. **Backward Pass** : AllGather parameter shards → compute local gradients → AllGather for gradient reduction → reduce-scatter to partition gradients


The communication cost is significant: O(N) AllGathers per layer per forward/backward pass. This is why ZeRO-3 typically combines with tensor parallelism within nodes.
### ZeRO-Offload
ZeRO-Offload extends ZeRO by offloading optimizer states and gradients to CPU memory when not needed:
**ZeRO-1 Offload** :
  * Optimizer states → CPU
  * Compute optimizer step on CPU, then copy updated states to GPU
  * Benefits: Frees GPU memory for larger models or batches


**ZeRO-2 Offload** :
  * Gradients → CPU (after backward pass)
  * Optimizer step on CPU
  * Benefits: Massive memory reduction at 10-20% performance cost


**ZeRO-3 Offload** :
  * Parameters + Gradients + Optimizer states → CPU
  * Only gather what needed, when needed
  * Benefits: Enables huge models on limited GPUs, but slower


### ZeRO++
ZeRO++ builds on ZeRO with three key improvements:
  1. **Quantized Gradients** : Store gradients in INT8 instead of FP16/BF16, reducing gradient communication by 2x
  2. **Hierarchical Parameter Sharding** : First shard within node (fast), then across nodes (slower), reducing cross-node bandwidth
  3. **Weight Quantization in Transit** : Quantize parameters during AllGather communication, reducing bandwidth requirements


### When to Use Each ZeRO Stage  
| Stage  | Memory Reduction  | Communication Increase  | Best For  |  
| --- | --- | --- | --- |  
| ZeRO-1  | 4× (optimizer)  | Minimal  | Memory-bound, many GPUs  |  
| ZeRO-2  | 8× (optimizer + gradients)  | Low  | Moderate memory constraints  |  
| ZeRO-3  | N× (full sharding)  | High  | Very large models, few GPUs per replica  |  
Practical guideline: Start with ZeRO-1, then ZeRO-2 if memory still insufficient. ZeRO-3 when model won't fit otherwise.
## FSDP: Fully Sharded Data Parallel
FSDP (Fully Sharded Data Parallel), introduced by PyTorch, shards model parameters, gradients, and optimizer states across data parallel ranks. While conceptually similar to ZeRO, FSDP offers different trade-offs and integration with PyTorch ecosystems.
### Sharding Strategies
**FULL_SHARD** (formerly Sharded):
  * Forward: Gather parameters → forward → discard
  * Backward: Gather parameters → backward → reduce-scatter gradients
  * Memory: Minimum per rank, maximum communication


**SHARD_GRAD_OP** (formerly ShardedGradScalable):
  * Forward: Keep all parameters (no gather in forward)
  * Backward: Gather parameters → backward → reduce-scatter
  * Memory: Higher than FULL_SHARD, lower communication


**HYBRID_SHARD** :
  * Combines intra-node FSDP with inter-node data parallelism
  * Parameters gathered within node (NVLink), reducing cross-node traffic
  * Best for multi-node clusters


**HYBRID_SHARD_ZERO2** :
  * Intra-node gradient sharding, inter-node full gradient
  * Compromise between HYBRID_SHARD and standard zero2


### CPU Offloading Options
FSDP supports offloading to CPU:

```
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

policy = transformer_auto_wrap_policy

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(
        param_offload=True,  # Parameters to CPU
        pin_memory=True,    # Faster CPU-GPU transfer
    ),
    auto_wrap_policy=policy,
)

```

Full CPU offloading provides maximum memory savings at significant performance cost (typically 30-50% slower).
### FSDP vs. ZeRO: Comparison  
| Aspect  | FSDP  | ZeRO  |  
| --- | --- | --- |  
| Framework  | PyTorch native  | DeepSpeed (PyTorch/MyTorch)  |  
| Flexibility  | More granular control  | Opinionated defaults  |  
| Mixed Precision  | Native BF16/FP16  | Native BF16/FP16  |  
| Checkpointing  | Built-in  | Built-in  |  
| Optimizer Offload  | Via CPU offload  | ZeRO-Offload  |  
| Communication Overlap  | Better overlap support  | Good overlap support  |  
| Multi-node  | Requires setup  | Better multi-node defaults  |  
In practice, ZeRO-3 and FSDP FULL_SHARD provide similar memory characteristics. The choice often comes down to framework preference and ecosystem integration.
## Parameter and Optimizer Offloading
### Optimizer Offloading
Optimizer offloading moves optimizer states (momentum, variance for Adam) to CPU memory, keeping only the optimizer update computation on GPU:

```
Standard: Optimizer states in GPU memory
Offloaded: Optimizer states in CPU memory, fetched when needed

```

The workflow:
  1. **Forward/Backward** : Compute gradients on GPU
  2. **Gradient Transfer** : Move gradients to CPU
  3. **Optimizer Step** : Compute parameter update on CPU
  4. **Parameter Transfer** : Move updated parameters to GPU


**Memory Savings** : 8 bytes per parameter (Adam states)
**Performance Cost** : CPU-GPU transfer bandwidth becomes bottleneck. Typically 20-40% slower.
### Parameter Offloading
Parameter offloading extends the concept to model parameters themselves. Only parameters needed for current computation reside on GPU:

```
Standard: Full model on GPU
Offloaded: Parameters fetched layer-by-layer during forward/backward

```

This enables training models that exceed GPU memory:

```
Training 175B model on 8× A100-40GB (320 GB total)
Standard: 175B × 12 bytes = 2100 GB (impossible)
With full offloading: Fits in 320 GB with careful management

```

**Communication Overlap** : Critical for performance. Overlap parameter fetch with computation of previous layer.
### Fused Optimizer
Fused optimizers combine optimizer computation with gradient reduction, reducing memory pressure:
**Standard Adam** :

```
for param in parameters:
    m = optimizer_state[param]['m']      # Read from memory
    v = optimizer_state[param]['v']      # Read from memory
    # compute update
    optimizer_state[param]['m'] = m      # Write to memory
    optimizer_state[param]['v'] = v      # Write to memory

```

**Fused Adam** :

```
# Single kernel: read gradients, compute and update in registers, write back

```

Benefits:
  * Fewer memory accesses
  * Better register utilization
  * 10-20% faster optimizer step
  * Reduced memory pressure


PyTorch's `torch.optim.AdamW` with `fused=True` provides fused implementation.
## LoRA and QLoRA: Low-Rank Adaptation
### LoRA: Low-Rank Adaptation
LoRA freezes pretrained model weights and injects trainable low-rank matrices. For a weight matrix W ∈ R^(d×k), LoRA adds a low-rank update:

```
W' = W + BA

```

Where:
  * B ∈ R^(d×r)
  * A ∈ R^(r×k)
  * r << min(d, k) is the rank (typically 8-64)


Instead of updating d×k parameters, we train (d + k)×r parameters:

```
Parameter reduction = (d×k) / ((d+k)×r)

```

For a 70B model with r=16:

```
Traditional fine-tuning: 70B parameters
LoRA: ~1B parameters (98.6% reduction)

```

### LoRA Layers
LoRA applies to specific weight matrices in transformers:
  * Attention: Q, K, V projections
  * Output projection
  * MLP gates and up/down projections


Common configurations (denoted as LoRA rank):
  * LoRA-8: rank = 8
  * LoRA-16: rank = 16 (more expressive)
  * LoRA-64: rank = 64 (for larger models)


### QLoRA: Quantized LoRA
QLoRA adds quantization to LoRA, enabling fine-tuning of quantized base models:
**NF4 (Normalized Float 4)** : A quantization format optimized for pretrained weight distributions:

```
NF4: 16 levels distributed to match weight quantiles

```

The quantization mapping:

```
FP32 → NF4 (4-bit) → BF16 (for computation) → NF4 (for storage)

```

**Double Quantization** : Quantize the already-quantized values:

```
First quantization: FP16 → INT4 (16× compression)
Second quantization: INT4 → 2-bit (additional 2× compression)

```

Combined compression: 32× from FP32 to double-quantized NF4
**Paged Optimizers** : Use CPU memory for optimizer states when GPU memory is constrained:

```
Optimizer states in CPU → Compute on GPU when needed → Return to CPU

```

### Memory Comparison  
| Method  | Memory for 70B Model  | GPU Requirements  |  
| --- | --- | --- |  
| Full Fine-tuning  | ~1.4 TB  | 8× A100-80GB  |  
| LoRA  | ~160 GB  | 2× A100-80GB  |  
| QLoRA (INT4)  | ~40 GB  | 1× A100-80GB  |  
| QLoRA (NF4 + Double)  | ~22 GB  | Consumer GPU  |  
### When to Use LoRA/QLoRA
**Use LoRA when** :
  * Fine-tuning for downstream tasks
  * Limited GPU memory (2-4 GPUs)
  * Fast experimentation with multiple tasks
  * Model weights shouldn't be modified


**Use QLoRA when** :
  * Extreme memory constraints
  * Fine-tuning on consumer hardware
  * Running many parallel experiments
  * Accuracy loss is acceptable


**Use Full Fine-tuning when** :
  * Maximum accuracy required
  * Sufficient GPU memory available
  * Training from scratch or major domain shift


## 3D Parallelism
Real-world large-scale training combines three parallelism strategies to maximize both scale and efficiency.
### Data Parallelism (DP)
Each GPU (or GPU group) holds a complete model replica, processing different data:

```
GPU 0: Model → Different Batch → Gradients
GPU 1: Model → Different Batch → Gradients
...
AllReduce: Average gradients → Identical update

```

**Strengths** :
  * Simple implementation
  * Linear scaling with GPU count
  * Good for models fitting in single GPU memory


**Weaknesses** :
  * Memory scales with model size
  * Communication per step (gradients)


### Tensor Parallelism (TP)
Each layer's weights are split across GPUs. For a linear layer Y = X × W:

```
GPU 0: Y₀ = X × W₀ (column partition)
GPU 1: Y₁ = X × W₁ (column partition)
AllReduce: Y = Y₀ + Y₁

```

**Attention parallelization** :
  * Split heads across GPUs (for num_heads % tp_size == 0)
  * Each GPU computes attention subset
  * AllReduce to combine


**Communication** :
  * Forward: AllReduce per layer
  * Backward: AllReduce per layer
  * Must stay within node (NVLink required)


### Pipeline Parallelism (PP)
Layers are distributed across GPUs (or nodes):

```
GPU 0: Layers 0-19
GPU 1: Layers 20-39
GPU 2: Layers 40-59
GPU 3: Layers 60-79

```

**Pipeline Bubble** : The time when GPUs are idle waiting for activations:

```
Time →
GPU 0: [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 1:    [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 2:       [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 3:          [F1][F2][F3][F4][B4][B3][B2][B1]
              └─bubble─►

```

**Interleaved Schedule** : Reduce bubble by forwarding multiple micro-batches:

```
Forward 1, Forward 2, Forward 3, Forward 4 → Backward 4, Backward 3, Backward 2, Backward 1

```

Modern implementations (Megatron-LM, DeepSpeed) achieve <5% bubble overhead.
### Combining 3D Parallelism
A typical configuration for 175B model training:

```
Configuration:
- 512 GPUs (64 nodes × 8 GPUs)
- TP = 8 (within node)
- PP = 8 (across nodes)
- DP = 8 (model replicas)

Memory per GPU:
- 175B × 2 bytes (BF16) = 350 GB total
- 350 GB / 8 (TP) / 8 (PP) = 5.5 GB per GPU ✓

```

The formula for GPU count:

```
Total GPUs = DP × TP × PP

```

And effective batch size:

```
Batch = DP × micro_batch_size × pipeline_stages

```

### Communication Topology Mapping
The mapping between parallelism strategies and hardware topology:  
| Strategy  | Communication Pattern  | Hardware  | Bandwidth  |  
| --- | --- | --- | --- |  
| TP  | AllReduce per layer  | NVLink  | 900 GB/s  |  
| PP  | P2P activations  | NVLink/InfiniBand  | 900/50 GB/s  |  
| DP  | AllReduce per step  | InfiniBand  | 50 GB/s  |  
**Rule** : Put TP within a node, PP across fast network, DP across any network.
## FlashAttention and Memory-Efficient Attention
### The Attention Memory Problem
Standard attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V

```

For sequence length n:
  * QK^T: n² values
  * softmax(QK^T): n² values
  * Intermediate storage: 2 × n² × batch × heads


For n=2048, batch=1, heads=32:

```
2 × 2048² × 32 × 2 bytes = 512 MB per layer
For 80 layers: 40 GB just for attention intermediates

```

### FlashAttention Algorithm
FlashAttention computes attention in blocks, keeping most data in SRAM:

```
Algorithm:
1. Divide Q, K, V into T blocks
2. For each block:
   a. Load block from HBM to SRAM
   b. Compute partial attention
   c. Update output accumulator
   d. Scale correctly
3. Write final output to HBM

```

The key insight: instead of computing full softmax, maintain running maximum and sum for numerical stability:

```
m(x) = max_i exp(x_i)           # running maximum
s(x) = Σ_i exp(x_i - m(x))       # sum of exponentials
softmax(x)_j = exp(x_j - m(x)) / s(x)

```

### Memory and Speed Analysis  
| Method  | Memory  | Speed  |  
| --- | --- | --- |  
| Standard Attention  | O(n²)  | Baseline  |  
| FlashAttention  | O(n)  | 2-4× faster  |  
| FlashAttention-2  | O(n)  | 2-3× faster than FA  |  
Memory savings:

```
Standard: 40 GB (80 layers)
FlashAttention: ~1 GB (80 layers)

```

The reduction is dramatic: attention becomes memory-bound → compute-bound.
### FlashAttention-2
FlashAttention-2 improves on original FlashAttention:
  1. **Better Kernel Fusion** : Fuses more operations
  2. **Sequence Parallelism** : Split sequence across GPUs within a node
  3. **Improved Swide/SiLU** : Faster non-linearities
  4. **Backward Pass Optimization** : Better memory during backprop


For training, FlashAttention-2 provides:
  * 5-20× less activation memory
  * 1.5-2× faster forward + backward


### Integration with Training Frameworks
FlashAttention integrates with major frameworks:

```
# PyTorch
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None)

# xFormers
from xformers.ops import memory_efficient_attention

output = memory_efficient_attention(q, k, v, attn_bias=None)

```

HuggingFace transformers enables FlashAttention:

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    attn_implementation="flash_attention_2",
)

```

## Gradient Communication Overlap
### The Communication Bottleneck
In data-parallel training, after each backward pass, gradients must be synchronized across all GPUs:

```
Compute: 10 seconds
AllReduce: 7 seconds ← GPU idle during this
Optimizer: 1 second
Total: 18 seconds per step (39% idle)

```

### Overlapping Strategies
**Layer-wise Overlap** :

```
Time →
Layer N:   [Forward][Backward][Gradient Ready]──────────[AllReduce]
Layer N-1:                [Forward][Backwd][Grad Ready]────[AllReduce]
Layer N-2:                           [Forward][Backwd][Grad Ready]─[AllReduce]

```

As soon as layer N's gradients are ready, launch AllReduce while computing layer N-1.
**Bucket Management** :
Gradients accumulate in "buckets". AllReduce launches when bucket is full, not when entire backward pass completes:

```
Small buckets: Earlier overlap, more communication overhead
Large buckets: Less overhead, later overlap
Optimal: 25-50 MB typically

```

**Backward Computation Overlap** :

```
Standard:
  Backward Pass → AllReduce → Optimizer Step

With Overlap:
  Layer N backward → Layer N AllReduce
  Layer N-1 backward → Layer N-1 AllReduce
  ...
  Optimizer step (all reduced)

```

### PyTorch DDP Implementation
PyTorch DDP implements overlap automatically:

```
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    bucket_cap_mb=25,  # Bucket size for overlap
    gradient_as_bucket_view=True,  # Avoid memory copy
)

```

Key parameters:
  * `bucket_cap_mb`: Size of gradient buckets (default 25 MB)
  * `gradient_as_bucket_view`: Memory-efficient gradient storage
  * `broadcast_buffers`: Whether to broadcast buffers at init (default True)


### DeepSpeed Communication Overlap
DeepSpeed provides more sophisticated overlap:
  1. **Partitioned AllReduce** : Reduce within node first, then across nodes
  2. **Hierarchical Gradient Synchronization** : Matches hardware topology
  3. **Tensor Fusion** : Fuse small gradients into larger transfers
  4. **Overlap Schedule** : Asymmetric timing for forward/backward passes


```
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config={
        "train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,  # Enable overlap
            "reduce_bucket_size": 100_000_000,
        },
    },
)

```

### Measuring Overlap Effectiveness
Monitor overlap efficiency:

```
# PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
) as prof:
    for step, batch in enumerate(dataloader):
        # Training step
        prof.step()

# Check for communication gaps
print(prof.key_averages().table(sort_by="cuda_time_total"))

```

Good overlap achieves 80-95% compute utilization. Poor overlap shows large gaps in CUDA activity.
## Training Memory Estimation
### Quick Estimation Formulas
For standard DDP with Adam optimizer:

```
Total Memory = Model_Weights + Gradients + Optimizer_States + Activations
             = 2P + 4P + 8P + Activation_Memory
             = 14P + Activation_Memory

```

For transformer activations:

```
Activation_Memory ≈
  batch × seq_len × hidden × (attention_heads × 4 + mlp_expansion × 2 + layer_norm × 2) × layers × bytes_per_param

```

### Memory Estimation Examples
**70B Model, BF16, seq=2048, batch=1** :

```
Weights:        70B × 2 = 140 GB
Gradients:      70B × 4 = 280 GB
Optimizer:      70B × 8 = 560 GB
Activations:    ~100 GB
Total:          ~1080 GB

```

**With ZeRO-3 (N=8)** :

```
Per GPU: (2 + 4 + 8) × 140 / 8 + 100/8 ≈ 280 GB + 12.5 GB ≈ 293 GB
Still doesn't fit in 80 GB!

```

**With ZeRO-3 + Activation Checkpointing + FlashAttention** :

```
Activations reduced: 100 GB → ~15 GB
Per GPU: 280 + 15 = 295 GB (still tight)

```

**With ZeRO-3 + Checkpointing + Quantization (INT8 optimizer)** :

```
Optimizer: 70B × 4 = 280 GB
Per GPU: (2 + 4 + 4) × 140 / 8 + 15 ≈ 175 + 15 = 190 GB
Fits with 80 GB? No, needs optimization.

```

**With ZeRO-2 + Checkpointing + FSDP hybrid** :

```
Memory ≈ 400 GB → Still needs multiple GPUs

```

### Practical Guidelines  
| Model Size  | GPU Memory  | Recommended Setup  |  
| --- | --- | --- |  
| 7B  | 16 GB  | BF16, DP, no checkpointing  |  
| 7B  | 24 GB  | BF16, ZeRO-2, gradient checkpointing  |  
| 13B  | 40 GB  | BF16, ZeRO-2, checkpointing, LoRA  |  
| 70B  | 80 GB  | ZeRO-3, checkpointing, TP within node  |  
| 70B  | 8×80 GB  | ZeRO-3, checkpointing, PP, DP  |  
| 175B  | 8×80 GB  | 3D Parallel (TP×PP×DP)  |  
### Memory Optimization Checklist
When fitting a model:
  1. Enable BF16/FP16 mixed precision
  2. Enable activation checkpointing (start with every 4 layers)
  3. Enable FlashAttention
  4. Use gradient accumulation for larger effective batch
  5. Apply ZeRO-1/2/3 based on memory pressure
  6. Consider LoRA/QLoRA if fine-tuning
  7. Enable CPU offloading if still insufficient
  8. Reduce sequence length if possible
  9. Reduce batch size and compensate with accumulation
  10. Use optimized kernels (FusedAdam, xFormers)


## Latest Trends and Tools in 2025-2026
### Emerging Techniques
**FP8 Training (Hopper/Blackwell)** : Native FP8 computation with minimal accuracy loss. Provides 2× memory and compute efficiency over BF16. NVIDIA H100/H200 and newer GPUs support FP8 Tensor Cores.
**Mixture-of-Experts (MoE) Activation Sparsity** : Models like Mixtral use 8 experts with 2 active per token, achieving 12B active parameters with 46B total. Massive memory reduction for inference, training optimization active.
**Lazy Attention** : Compute attention incrementally rather than full sequence. Reduces activation memory from O(n²) to O(n) for long sequences. Active research area.
**Streaming Lightweight Checkpoints** : Save only optimizer states (largest component) to CPU, keep parameters/gradients on GPU. Enables faster recovery.
**Gradient sparsification** : Only communicate top-k% of gradients. Reduces communication at cost of convergence. Useful for multi-node training.
### Key Frameworks and Libraries  
| Framework  | Focus  | Key Features  |  
| --- | --- | --- |  
| DeepSpeed  | Microsoft training  | ZeRO, 3D parallelism, inference  |  
| FSDP  | PyTorch native  | Flexible sharding, HYBRID_SHARD  |  
| Megatron-LM  | NVIDIA research  | TP/PP optimization, expert parallelism  |  
| PyTorch Lightning  | High-level training  | Automated mixed precision, checkpointing  |  
| Composer  | MosaicML  | Algorithm composition, efficiency metrics  |  
| vLLM  | Inference  | PagedAttention, continuous batching  |  
### Hardware Trends
**H100/H200** : 80-141 GB HBM3e, FP8 support, 900 GB/s NVLink
**B100/B200 (Blackwell)** : 192 GB HBM3e, FP4 support expected, 1.8 TB/s NVLink
**GH200** : GPU + CPU combined memory (144 GB + 4 TB), enables massive model training without sharding
**Multi-Node Training** : Focus on reducing cross-node bandwidth requirements through better sharding strategies
### Recommended Stack for 2025-2026
For large-scale LLM training:
  1. **Framework** : DeepSpeed or PyTorch FSDP
  2. **Precision** : BF16 with FP8 for supported operations
  3. **Parallelism** : 3D (TP within node, PP across nodes, DP across replicas)
  4. **Memory Optimization** : ZeRO-3 + gradient checkpointing + FlashAttention
  5. **Communication** : NCCL with topology-aware scheduling
  6. **Checkpointing** : Periodic to distributed filesystem, streaming for large models


For fine-tuning:
  1. **Framework** : HuggingFace PEFT or DeepSpeed with LoRA
  2. **Quantization** : QLoRA with NF4 + double quantization
  3. **Memory** : Fits on single A100-80GB or consumer GPU with optimization


## Conclusion
Memory efficient training combines multiple orthogonal techniques. The key insights:
  1. **Memory Budget** : Training 70B+ models requires 10×+ memory reduction from naive FP32
  2. **Layered Optimization** : Mixed precision → checkpointing → ZeRO/FSDP → 3D parallelism
  3. **Trade-offs** : Every memory reduction incurs compute or communication cost
  4. **Hardware Matching** : Choose techniques based on GPU count, interconnect topology, and model size


The field continues to evolve rapidly. FP8 training, MoE architectures, and new hardware (Blackwell) will further shift the boundaries of what's possible. But the fundamental principle remains: understand your memory bottlenecks, apply targeted optimizations, and measure the trade-offs.
Training large models is ultimately about managing the flow of data across compute, memory, and network hierarchies. Master these techniques, and terabyte-scale training becomes accessible.
## References
### Papers
  * [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
  * [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
  * [ZeRO++: Scaling Giants with Communication Efficiency](https://arxiv.org/abs/2306.10209)
  * [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
  * [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
  * [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  * [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
  * [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
  * [DeepSpeed: System Optimizations for Enabling Training at Scale](https://arxiv.org/abs/2207.03288)
  * [PyTorch FSDP: Fully Sharded Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
  * [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/2104.04473)


### Tools & Libraries
  * [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
  * [HuggingFace PEFT](https://huggingface.co/blog/peft)


### Further Reading
  * [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)
  * [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  * [Lightning AI Memory Optimization](https://lightning.ai/docs/pytorch/stable/advanced/memory_optimization.html)
  * [Transformer Memory Calculations](https://kipp.ly/transformer-memory-calculations/)


Human
