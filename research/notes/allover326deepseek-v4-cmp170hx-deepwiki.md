---
title: allover326/deepseek-v4-cmp170hx | DeepWiki
id: allover326deepseek-v4-cmp170hx-deepwiki
tags:
- cmp170hx-training-feasibility-a0d4fc
- cmp-170hx
- deepwiki-index
- pipeline-parallel
- pwrbrk-power-brake
- vram-unlock-64gb
created: '2026-10-27T10:54:42.921574Z'
updated: '2026-10-27T19:02:22.585604Z'
source: https://deepwiki.com/allover326/deepseek-v4-cmp170hx
source_domain: deepwiki.com
fetched_at: '2026-10-27T10:54:42.298223Z'
fetch_provider: crawl4ai
status: review
type: note
tier: commentary
content_type: article
deprecated: false
summary: 'Auto-generated DeepWiki index of allover326/deepseek-v4-cmp170hx, last indexed
  2026-08-10 at commit 317dac. Confirms the deployment shape: 4x CMP 170HX with VRAM
  unlocked to 64 GB per card, PCIe Gen2 x4, 1,047,736-token verified context ceiling,
  PP4+DSpark decode 98.1 tok/s versus 50.8 tok/s for plain PP4, prefill 5,272 tok/s
  at 77k context, decode at 100k context 90.0 versus 38.8 tok/s. Names PWRBRK# (pin
  B30) as a ~4x performance trap on certain motherboards. Thin (661 words) and derivative
  - it is a navigation index over the underlying repo, so cite the repo''s README/RESULTS/SETTINGS
  files rather than this page.'
utility_score: 14.0
---

Index your code with Devin
[DeepWiki](https://deepwiki.com/)
[DeepWiki](https://deepwiki.com/)
[allover326/deepseek-v4-cmp170hx ](https://github.com/allover326/deepseek-v4-cmp170hx "Open repository")
Index your code with
Devin
Edit WikiShare
Last indexed: 10 August 2026 ([317dac](https://github.com/allover326/deepseek-v4-cmp170hx/commits/317dac12))
  * [Performance Results at a Glance](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/1.2-performance-results-at-a-glance)
  * [Pipeline Parallelism vs Tensor Parallelism](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/2.1-pipeline-parallelism-vs-tensor-parallelism)
  * [DSpark Speculative Decoding on Pipeline Parallel](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/2.2-dspark-speculative-decoding-on-pipeline-parallel)
  * [Sparse Attention Indexer and Long-Context Scaling](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/2.3-sparse-attention-indexer-and-long-context-scaling)
  * [Patches 0001–0005: DSpark-on-PP Enablement](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/3.1-patches-0001-0005:-dspark-on-pp-enablement)
  * [Patches 0005a and 0006: Sparse Indexer Fixes](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/3.2-patches-0005a-and-0006:-sparse-indexer-fixes)
  * [Launch Scripts and Runtime Configuration](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/4.2-launch-scripts-and-runtime-configuration)
  * [Configuration Reference (SETTINGS.md)](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/4.3-configuration-reference-\(settings.md\))
  * [Correctness and Stability Benchmarks](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/5.2-correctness-and-stability-benchmarks)
  * [Multi-turn Accumulation and Analysis](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/5.3-multi-turn-accumulation-and-analysis)
  * [Retrieval Accuracy vs Context Depth](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/6.1-retrieval-accuracy-vs-context-depth)
  * [Degeneration Detection and Analysis](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/6.2-degeneration-detection-and-analysis)


# Overview
Relevant source files


This project provides a production-grade deployment of **DeepSeek-V4-Flash-0731** on **4× NVIDIA CMP 170HX** mining cards. By leveraging the GA100 silicon (sm_80) and unlocking the VRAM to 64 GB per card, the system achieves the model's full **1,047,736-token context ceiling** and a **1.93× decode speedup** through the enablement of DSpark speculative decoding under Pipeline Parallelism (PP).
The core contribution of this repository is a series of patches and configurations that allow vLLM to run DSpark speculative decoding alongside Pipeline Parallelism—a combination previously restricted by the engine. On this specific hardware, Pipeline Parallelism provides a **6.6× prefill advantage** over Tensor Parallelism due to the severe PCIe Gen2 x4 bandwidth constraints of the CMP 170HX.
### System Architecture Association
The following diagram illustrates how the natural language components of the system map to specific code entities and files within the repository.
**Diagram: System Component Mapping**

```



Code Entity Space
Natural Language Space
Patched in
Defined in
Logic in
Speculative Decoding (DSpark)
Pipeline Parallelism (PP4)
Sparse Attention Indexing
Hardware Management
vllm/model_executor/layers/spec_decode/dspark/
broadcast_draft()
run-pp-dspark.sh
VLLM_PP_LAYER_PARTITION
sparse_attn_indexer.py
DSV4_LOGITS_ROW_CHUNK
nvidia-smi -q
PWRBRK# (Pin B30)
0001-0005-dspark-pp.patch
vllm/executor/ray_gpu_executor.py
vllm/model_executor/layers/attention/backends/dsv4.py




```

Sources: [README.md30-43](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L30-L43) [README.md107-110](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L107-L110) [RESULTS.md12-20](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/RESULTS.md?plain=1#L12-L20)
## Hardware: NVIDIA CMP 170HX
The **NVIDIA CMP 170HX** is a specialized mining card featuring GA100 silicon. While it lacks display outputs and is limited to PCIe Gen2 x4, it offers 64 GB of HBM2e memory once the VRAM is unlocked. A critical deployment factor is the **PWRBRK# (pin B30)** diagnostic; on certain motherboards, this signal can force the cards into a low-power state, reducing performance by ~4×.
For detailed coverage of GA100 silicon, VRAM unlocking, and hardware-specific troubleshooting, see [Hardware: NVIDIA CMP 170HX](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/1.1-hardware:-nvidia-cmp-170hx).
Sources: [README.md3-6](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L3-L6) [README.md19-22](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L19-L22) [README.md116-159](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L116-L159)
## Performance Results at a Glance
The system delivers high-throughput inference even at extreme context lengths. The integration of DSpark speculative decoding allows the system to maintain a decode rate of **~98 tok/s** for single streams, while the Pipeline Parallel (PP4) configuration ensures prefill remains efficient across the 1M token window.  
| Metric  | Plain PP4  | PP4 + DSpark  |  
| --- | --- | --- |  
| **Decode (Single Stream)**  | 50.8 tok/s  | **98.1 tok/s**  |  
| **Prefill @ 77k Context**  | 5,272 tok/s  | 5,207 tok/s  |  
| **Max Verified Context**  | 1,047,736  | 1,047,736  |  
| **Decode @ 100k Context**  | 38.8 tok/s  | **90.0 tok/s**  |  
For a full breakdown of throughput, TTFT, and DSpark acceptance rates, see [Performance Results at a Glance](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/1.2-performance-results-at-a-glance).
Sources: [RESULTS.md12-20](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/RESULTS.md?plain=1#L12-L20) [RESULTS.md33](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/RESULTS.md?plain=1#L33-L33) [RESULTS.md63-67](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/RESULTS.md?plain=1#L63-L67)
## Repository Layout and Workflow
The project is structured to facilitate rapid deployment via Docker and automated patching of the `haosdent/vllm` source tree.
**Diagram: Deployment and Patching Workflow**

```



Build Process
Repository Structure
0001-0006
Dockerfile.devel
dsv4-a100:devel
/
patches/
docker/
launch/
bench/
haosdent/vllm (f8ea5bb)
patch -p1
docker build
run-pp-dspark.sh




```

Sources: [README.md62-81](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L62-L81) [README.md30-43](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L30-L43)
### Key Components
  * **Patches** : Seven patches applied to `haosdent/vllm@f8ea5bb` to enable DSpark on PP and fix context ceiling issues [README.md74](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L74-L74)
  * **Launch Scripts** : `run-pp-dspark.sh` handles the complex environment variable setup required for PP4 and DSpark [README.md80](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/README.md?plain=1#L80-L80)
  * **Benchmarking** : A comprehensive suite of 16 harnesses in `bench/` for measuring performance and retrieval accuracy [RESULTS.md7-8](https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/RESULTS.md?plain=1#L7-L8)


For details on the deployment process, see [Docker Build](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/4.1-docker-build) and [Launch Scripts and Runtime Configuration](https://deepwiki.com/allover326/deepseek-v4-cmp170hx/4.2-launch-scripts-and-runtime-configuration).
Dismiss
Refresh this wiki
Enter email to refresh
### On this page
  * [System Architecture Association](https://deepwiki.com/allover326/deepseek-v4-cmp170hx#system-architecture-association)
  * [Hardware: NVIDIA CMP 170HX](https://deepwiki.com/allover326/deepseek-v4-cmp170hx#hardware-nvidia-cmp-170hx)
  * [Performance Results at a Glance](https://deepwiki.com/allover326/deepseek-v4-cmp170hx#performance-results-at-a-glance)
  * [Repository Layout and Workflow](https://deepwiki.com/allover326/deepseek-v4-cmp170hx#repository-layout-and-workflow)


