---
title: 'GitHub - promisezackr/glm53-flash-170hx-pp8: GLM-5.3-Flash (320B MoE) served
  with vLLM on 8x CMP 170HX (SM80, 64 GB, PCIe Gen2 x4): pipeline-parallel 8, NVFP4
  weights, DFlash2 speculative decoding, 1M context. Patches, launch config and benchmarks.
  · GitHub'
id: github-promisezackrglm53-flash-170hx-pp8-glm-53-flash-320b-moe-served-with-vllm
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- topology
- pipeline-parallel
- tensor-parallel
- pcie-gen2-x4
- community-reported
- primary-source
created: '2026-10-27T11:10:00.978084Z'
updated: '2026-10-27T19:00:31.420866Z'
source: https://github.com/promisezackr/glm53-flash-170hx-pp8
source_domain: github.com
fetched_at: '2026-10-27T11:10:00.356846Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'promisezackr/glm53-flash-170hx-pp8 (9 stars, 1 commit 90ec72e 2026-08-31,
  Apache-2.0 derivative of vLLM) — the community reference for serving a 320B MoE
  on the exact fabric the query names: 8x CMP 170HX (GA100, SM80, 64 GB unlocked),
  PCIe Gen2 x4 each, no P2P, 2x Xeon E5-2690 v4 host with 96 GB DDR4. Directly load-bearing
  for the topology question because it states the conclusion explicitly: ''TP is not
  an option on this box (no P2P); PP6 saves only ~6%.'' So on a Gen2 x4 no-P2P box,
  tensor parallelism is ruled out by the fabric in practice, and pipeline parallelism
  is the topology that works — corroborating the repo''s TP claim while refuting the
  stronger claim that DDP is the only viable topology. Production config: --pipeline-parallel-size
  8 with VLLM_PP_LAYER_PARTITION=7,5,7,5,7,5,7,2 (partitioned around the 11 sparse-MLA
  layers at indices 3,7,...,43 so stages holding two of them get fewer layers and
  a larger KV budget), --max-model-len 1048576, --gpu-memory-utilization 0.90, --max-num-seqs
  8, --max-num-batched-tokens 4096, no prefix caching, DFlash2 drafter with num_speculative_tokens
  7, VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2. Benchmarks (single stream, temp 0, including
  speculation): code 123 tok/s, json 120, counting 114-118, math 59, prose 37 under
  NVFP4 + DFlash2 k=7; FP8 eager no-spec code was only 6.2 tok/s, so the patch set
  is worth ~20x. Steady-state aggregate: 201-203 tok/s at 4 streams, 259-267 at 8,
  276 at 16. Long context: a 192k-token prompt completes in 53 s and a 960k-token
  prompt in 208 s end-to-end with the needle retrieved, KV pool 6.62M tokens (6.3
  concurrent 1M requests) at utilization 0.90 — i.e. roughly 4,600 tok/s aggregate
  prefill across 8 cards at 960k, which is the closest thing in the corpus to a measured
  1M-token-batch prefill cost. Step-time model from per-stage CUDA events: ~32 ms
  fixed + 2.1 ms x verified tokens per step. Startup: ~5 min weight load from SATA
  SSD, ~15 min first warmup, ~12 min later boots. 24 patches on top of vllm/vllm-openai:glm53-flash
  (65 files, +4.8k/-0.2k), covering SM8x port, glm5next PP intermediate-tensor factories,
  MTP-under-PP weight loading, DFlash2 under PP8, decode micro-batch cap, int64 KV
  offsets, drafter sidecar KV re-blocking 16->256, and env-gated stage instrumentation.'
---

*Suggested by [[club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]] — Cited as the community PP8 cost model and benchmark source measured on an 8-card PCIe Gen2 x4 box — the exact fabric in the research query*

[Skip to content](https://github.com/promisezackr/glm53-flash-170hx-pp8#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/promisezackr/glm53-flash-170hx-pp8) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/promisezackr/glm53-flash-170hx-pp8) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/promisezackr/glm53-flash-170hx-pp8) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/promisezackr/glm53-flash-170hx-pp8/branches)[](https://github.com/promisezackr/glm53-flash-170hx-pp8/tags)
Go to file
Open more actions menu
## Latest commit
[GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307)
Open commit details
Aug 31, 2026
[90ec72e](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307) · Aug 31, 2026
## History
[1 Commit](https://github.com/promisezackr/glm53-flash-170hx-pp8/commits/main/)
Open commit details
1 Commit
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
|   |   | [GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, ben…](https://github.com/promisezackr/glm53-flash-170hx-pp8/commit/90ec72e9525e90be701e742c70a20c4154418307 "GLM-5.3-Flash on 8x CMP 170HX: vLLM PP8 patch set, launch config, benchmarks")  | Aug 31, 2026  |  
| View all files |  
## Repository files navigation
# GLM-5.3-Flash on 8x CMP 170HX — vLLM PP8 / NVFP4 / DFlash2
在 8 张 CMP 170HX(SM80, 64 GB, PCIe Gen2 x4, 无 P2P)上跑通 GLM-5.3-Flash(320B MoE, 18B active, 1M context)的 vLLM 移植:流水线并行 8、NVFP4 权重、DFlash2 投机解码。本仓库公开全部补丁、启动参数和实测数据。
Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on hardware vLLM does not support (SM80, no NVLink/P2P, Gen2 x4 links). Everything here is a patch set on top of the official `vllm/vllm-openai:glm53-flash` image (`v0.1.dev20051`).
## Hardware / software  
| GPUs  | 8x NVIDIA CMP 170HX (GA100, SM80, 64 GB unlocked), PCIe Gen2 x4 each, no P2P  |  
| --- | --- |  
| Host  | 2x Xeon E5-2690 v4, 96 GB DDR4, SATA SSD for weights  |  
| Image  |  `vllm/vllm-openai:glm53-flash` (vLLM `v0.1.dev20051+g487ecf187`) + `patches/`  |  
| Weights  |  [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (ModelOpt NVFP4, routed experts only; 181 GiB)  |  
| Draft  | GLM-5.3-Flash DFlash2 drafter (5 sliding-window layers) — [z-lab/dflash](https://github.com/z-lab/dflash), [inco.ai blog](https://inco.ai/blog/dflash2/)  |  
| Kernels  | Marlin W4A16 for NVFP4 MoE, Triton sparse MLA / indexer fallbacks for SM80, FA2 for the drafter  |  
## Production configuration (`scripts/run_prod.sh`)

```
--pipeline-parallel-size 8            VLLM_PP_LAYER_PARTITION=7,5,7,5,7,5,7,2
--max-model-len 1048576               --gpu-memory-utilization 0.90
--max-num-seqs 8                      --max-num-batched-tokens 4096
--no-enable-prefix-caching
--speculative-config '{"method":"dflash","model":"/dflash2","num_speculative_tokens":7}'
VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2   # spread decodes over PP in-flight micro-batches
VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256     # drafter KV block size (see patch 0024)
VLLM_WORKER_MULTIPROC_METHOD=spawn    CUDA_DEVICE_ORDER=PCI_BUS_ID

```

Layer partition is chosen around the 11 sparse-MLA layers (idx 3,7,…,43): stages holding two of them get fewer layers so their KV budget is larger. Host tuning: CPU governor `performance` + `nvidia-smi -lgc 1695` (`scripts/glm-perf.service`, ~-10% step time).
## Benchmarks
Single stream, `scripts/bench.py` (temperature 0, tokens/s including speculative decoding):  
| workload  | FP8 eager, no spec  | FP8 + MTP k=3  | NVFP4 + MTP k=3  | **NVFP4 + DFlash2 k=7 (prod)**  |  
| --- | --- | --- | --- | --- |  
| code  | 6.2  | 51–61  | 71  | **123**  |  
| json  | –  | –  | 71  | **120**  |  
| counting  | –  | 77  | 65  | **114–118**  |  
| math  | –  | –  | 59  |  
| prose  | –  | –  | 43  | 37  |  
Steady-state concurrency, `scripts/conc2.py N 40` (chat, 200-token completions, aggregate tokens/s):  
| streams  | aggregate t/s  |  
| --- | --- |  
| 4  | 201–203  |  
| 8  | 259–267  |  
| 16  | 276  |  
Long context (`scripts/needle.py`, prod config): 192k-token prompt in 53 s, **960k-token prompt in 208 s** end-to-end, needle retrieved; KV pool 6.62M tokens (6.3 concurrent 1M requests) at utilization 0.90. Quality probe `scripts/qual.py`: 8/8 checkable items correct on NVFP4.
Step time model measured with per-stage CUDA events (patch 0018/0019): `≈ 32 ms fixed + 2.1 ms × verified tokens per step`. The fixed part is the 8-stage serial pipeline (attention/KDA/dense per stage + hops + host prep/sampling); the variable part is active-expert weight bandwidth (bytes ∝ tokens × top-k). TP is not an option on this box (no P2P); PP6 saves only ~6%.
Startup: weight load ~5 min from SATA SSD, first warmup ~15 min (Triton cache persisted under `triton-cache/`), later boots ~12 min to READY.
## What the patches do
`patches/NNNN-*.patch` are the 24 commits on top of the image's `vllm/` package, in order; `patches/all-changes-vs-baseline.diff` is the same as one diff (65 files, +4.8k/-0.2k).
  1. SM8x port of the glm53-flash tree (Marlin FP4 / Triton fallbacks, from [344303947/vllm170hx](https://github.com/344303947/vllm170hx)); kpool fp8 software encode, NoPE sparse-MLA(512), backend registration.
  2. `glm5next` pipeline parallelism: intermediate-tensor factories, mHC `hc_post` materialization at stage boundaries, dynamic top-k width, draft built `pp_size=1` on the last rank.
  3. MTP under PP: draft `embed_tokens` / `shared_head.head` loaded from the checkpoint (target→draft sharing is skipped under PP upstream, leaving them random → 0% acceptance); sync-free draft-token scatter (2x single stream); unquantized shared head for quantized checkpoints.
  4. DFlash2 (backport of vLLM PR #52816) under PP8: aux hidden-state relay across stages, async-safe staging out of CUDA-graph buffers, sidecar KV groups for the drafter, and the fix for zero draft embeddings under PP (root cause of low acceptance).
  5. Scheduler: `VLLM_PP_MAX_DECODE_REQS_PER_BATCH` decode micro-batch cap (8-stream aggregate 117 → 260 t/s).
  6. Sparse MLA: int64 KV offsets (Xid-31 MMU fault once the KV pool exceeded 2^31 bytes per layer).
  7. KV: re-block the drafter's sidecar specs 16 → 256 tokens. Block ids come from one shared pool and every id is charged the full MLA page on every rank; at block 16 a 1M request needed 2433 sidecar ids vs 277 core ids and 90% of the pool sat empty (1M concurrency 1.13x → 6.31x at less memory).
  8. Env-gated instrumentation: `VLLM_STAGE_TIMING`, `VLLM_STAGE_TRACE`, `VLLM_DFLASH_TRACE`, plus `preflight.py` (seconds-level import/protocol check before a 12-minute boot).


## Reproduce

```
# 1. take the vllm package out of the official image
docker create --name tmp vllm/vllm-openai:glm53-flash
docker cp tmp:/usr/local/lib/python3.12/dist-packages/vllm ./glm53/vllm  docker rm tmp
# 2. apply the patch set (paths are rooted at the parent of vllm/)
cd glm53  git init -q  git add -A  git commit -qm baseline
git apply --whitespace=nowarn ../patches/all-changes-vs-baseline.diff      # or ./apply.sh
# 3. bind-mount it over the image's package and launch
bash ../scripts/run_prod.sh   # edit model/draft paths and the container name first
```

Model weights go under `/data/models/GLM-5.3-Flash-NVFP4` and `/data/models/GLM-5.3-Flash-DFlash2` (ModelScope is far faster than HF from mainland China).
## Credits
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) · [344303947/vllm170hx](https://github.com/344303947/vllm170hx) SM8x patches · [bayley/vllm-170hx-glm5](https://github.com/bayley) PP-on-170HX validation · vLLM PR #52816 (DFlash2) and #30906 (Marlin NVFP4) · [LibertAIDAI](https://huggingface.co/LibertAIDAI) NVFP4 checkpoint · [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) DGX Spark recipe used as a reference.
License: Apache-2.0 (derivative of vLLM).
## About
GLM-5.3-Flash (320B MoE) served with vLLM on 8x CMP 170HX (SM80, 64 GB, PCIe Gen2 x4): pipeline-parallel 8, NVFP4 weights, DFlash2 speculative decoding, 1M context. Patches, launch config and benchmarks.
### Topics
[cmp-170hx](https://github.com/topics/cmp-170hx)[dflash](https://github.com/topics/dflash)[glm-5](https://github.com/topics/glm-5)[glm-5-3-flash](https://github.com/topics/glm-5-3-flash)[llm-inference](https://github.com/topics/llm-inference)[moe](https://github.com/topics/moe)[nvfp4](https://github.com/topics/nvfp4)[pipeline-parallelism](https://github.com/topics/pipeline-parallelism)[sm80](https://github.com/topics/sm80)[speculative-decoding](https://github.com/topics/speculative-decoding)[vllm](https://github.com/topics/vllm)
### Resources
### Stars
**9** stars
### Watchers
**0** watching
### Forks
[ fork](https://github.com/promisezackr/glm53-flash-170hx-pp8/forks)
## 
No releases published
## 
  * [**promisezackr** promisezackr](https://github.com/promisezackr)


## Languages
  * [Python85.5%](https://github.com/promisezackr/glm53-flash-170hx-pp8/search?l=python)
  * [Shell14.5%](https://github.com/promisezackr/glm53-flash-170hx-pp8/search?l=shell)


You can’t perform that action at this time. 
