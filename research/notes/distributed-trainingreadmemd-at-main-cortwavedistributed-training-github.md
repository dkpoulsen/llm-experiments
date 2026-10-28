---
title: distributed-training/README.md at main · cortwave/distributed-training · GitHub
id: distributed-trainingreadmemd-at-main-cortwavedistributed-training-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-3
created: '2026-10-27T14:33:01.166219Z'
updated: '2026-10-27T19:00:31.740617Z'
source: https://github.com/cortwave/distributed-training/blob/main/README.md
source_domain: github.com
fetched_at: '2026-10-27T14:33:00.544616Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Reproducible DDP/FSDP/DeepSpeed benchmark harness run on a no-P2P PCIe fabric:
  4x RTX 4090 24GB on Vast.ai, PCIe 4.0 x16, topology PHB/NODE, and `check_p2p.py`
  reports all 12 GPU pairs P2P DISABLED (GPU0->GPU1 247 ms/32 Gbps, GPU1->GPU0 76
  ms/105 Gbps asymmetric, GPU0->GPU2 233 ms/34 Gbps cross-NUMA). The FSDP comparison
  at batch 1/GPU is the closest thing found to a ZeRO-2-vs-DDP A/B on a host-staged
  fabric: DDP 6.90 samples/s (fwd 0.033 s, bwd 0.423 s, opt 0.128 s; 21.1 GB/GPU),
  FSDP SHARD_GRAD_OP ~= ZeRO-2 4.13 samples/s (fwd 0.323 s, bwd 0.643 s, opt 0.009
  s; 15.8 GB), FULL_SHARD ~= ZeRO-3 2.88, NO_SHARD 3.88. Step-time ratio DDP -> SHARD_GRAD_OP
  is 1.67x (+67% overhead), which exceeds cc-3''s 50% overturn threshold. But the
  bar is not met: model and sequence length are NOT stated (21.1 GB DDP residency
  implies ~1.3B params at 16 bytes/param, but that is inference, not a stated figure),
  batch is 1/GPU so DDP''s 0.033 s forward is launch-bound, and SHARD_GRAD_OP''s forward
  is 10x DDP''s (0.323 s) — the signature of a small-model all-gather-dominated regime,
  not a 32k/1B one. Also measures a separate ''bad topology'' DDP run where 2 GPUs
  give 0.29x and 4 GPUs 0.91x of single-GPU throughput.'
---

[Skip to content](https://github.com/cortwave/distributed-training/blob/main/README.md#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/cortwave/distributed-training/blob/main/README.md) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/cortwave/distributed-training/blob/main/README.md) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/cortwave/distributed-training/blob/main/README.md) to refresh your session. Dismiss alert
## Collapse file tree
## Files
Search this repository(forward slash)` forward slash/`
/
# README.md
Copy path
More file actions
More file actions
## Latest commit
Dec 22, 2025
[f39834e](https://github.com/cortwave/distributed-training/commit/f39834edd4b290cd5ae4b08678cab1fac09cd220) · Dec 22, 2025
## History
[History](https://github.com/cortwave/distributed-training/commits/main/README.md)
Open commit details
History
145 lines (111 loc) · 3.73 KB
/
# README.md
Copy path
## File metadata and controls
  * Preview
  * 

145 lines (111 loc) · 3.73 KB
Copy raw file
Download raw file
Outline
Edit and raw actions
# Distributed Training Benchmarks: P2P Investigation
Reproducible benchmarks investigating multi-GPU training performance across different cloud providers and GPU configurations.
## Quick Start
Install dependencies:

```
uv sync
uv pip install flash-attn --no-build-isolation
```

### Check P2P Status

```
uv run python3 check_p2p.py
```

### Run Benchmarks

```
# DDP benchmark (1, 2, 4 GPUs)
PYTHONPATH=./ uv run python3 ddp/run.sh

# FSDP benchmark (different sharding strategies)
PYTHONPATH=./ uv run python3 fsdp/run.sh

# DeepSpeed ZeRO benchmark
PYTHONPATH=./ uv run python3 deepspeed/run.sh
```

## Results
### RTX 4090 Performance (Vast.ai)  
| GPUs  | Batch  | Attn  | Samples/sec  | Speedup  | Efficiency  | Step Time  | Optimizer Time  |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| 1  | 2  | sdpa  | 15.38  | 1.00×  | 100%  | 0.130s  | 0.019s  |  
| 2  | 2  | sdpa  | 4.50  | 0.29×  | 15%  | 0.526s  | 0.400s  |  
| 4  | 2  | sdpa  | 14.01  | 0.91×  | 23%  | 0.571s  | 0.354s  |  
**Raw data** : See [ddp/rtx_4090_bad_topology.csv](https://github.com/cortwave/distributed-training/blob/main/ddp/rtx_4090_bad_topology.csv)
### P2P Status
All tested providers showed P2P disabled:

```
GPU 0 -> GPU 1: ✗ DISABLED
GPU 0 -> GPU 2: ✗ DISABLED
[... all 12 pairs disabled]

```

### Bandwidth Measurements (RunPod, RTX 4090)

```
GPU 0 -> GPU 1:  247ms,  32 Gbps  (expected: ~200 Gbps with P2P)
GPU 1 -> GPU 0:   76ms, 105 Gbps  (asymmetric!)
GPU 0 -> GPU 2:  233ms,  34 Gbps  (cross-NUMA)

```

### FSDP Comparison  
| Strategy  | Samples/sec  | Memory/GPU  | Forward  | Backward  | Optimizer  |  
| --- | --- | --- | --- | --- | --- |  
| DDP  | 6.90  | 21.1 GB  | 0.033s  | 0.423s  | 0.128s  |  
| FULL_SHARD  | 2.88  | 12.5 GB  | 0.474s  | 0.910s  | 0.009s  |  
| SHARD_GRAD_OP  | 4.13  | 15.8 GB  | 0.323s  | 0.643s  | 0.009s  |  
| NO_SHARD  | 3.88  | 20.8 GB  | 0.045s  | 0.420s  | 0.582s  |  
**Raw data** : See [fsdp/rtx_4090.csv](https://github.com/cortwave/distributed-training/blob/main/fsdp/rtx_4090.csv)
## Hardware Tested
### Vast.ai - RTX 4090
  * CPUs: AMD Threadripper PRO 3955WX (16-core)
  * GPUs: 4× RTX 4090 (24GB)
  * PCIe: 4.0 x16
  * Topology: PHB/NODE connections
  * P2P: ✗ Disabled


### RunPod - RTX 4090
  * CPUs: Dual-socket (NUMA)
  * GPUs: 4× RTX 4090 (24GB)
  * PCIe: 4.0
  * Topology: NODE/SYS (cross-NUMA)
  * P2P: ✗ Disabled


### Also tested (historical data)
  * RTX 3090 (4×, NVLink inactive): [rtx_3090_inactive_nvlink_benchmark_results.csv](https://github.com/cortwave/distributed-training/blob/main/ddp/rtx_3090_inactive_nvlink_benchmark_results.csv)
  * RTX 2080 Ti (4×): [rtx2080ti_benchmark_results.csv](https://github.com/cortwave/distributed-training/blob/main/ddp/rtx2080ti_benchmark_results.csv)


## GPU Topology Examples
### Vast.ai (PHB/NODE)

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NODE    NODE    NODE
GPU1    NODE     X      PHB     NODE
GPU2    NODE    PHB      X      NODE
GPU3    NODE    NODE    NODE     X

```

### RunPod (NODE/SYS - worse)

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NODE    SYS     SYS
GPU1    NODE     X      SYS     SYS
GPU2    SYS     SYS      X      NODE
GPU3    SYS     SYS     NODE     X

```

### Configuration
**Model** : Qwen/Qwen3-1.7B (~1.7B parameters) **Dataset** : Guilherme34/anthropic-Awareness-interview **Precision** : bfloat16 (auto) **Sequence length** : 512 tokens **Batch sizes tested** : 1, 2, 4, 8 **Max steps** : 100 (for benchmarking)
### Environment

```
# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL debugging (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Viewing Results
### TensorBoard

```
uv run tensorboard --logdir runs/benchmarks --port 6006
```

## Key Findings
Summary:
  * P2P disabled on budget clouds → 3-4× slower multi-GPU training
  * FSDP worse than DDP on broken P2P (more communication)
  * Single-GPU optimal for models <10B on budget infrastructure


## License
MIT
You can’t perform that action at this time. 
