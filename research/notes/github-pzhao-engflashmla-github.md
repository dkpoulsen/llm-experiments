---
title: GitHub - pzhao-eng/FlashMLA · GitHub
id: github-pzhao-engflashmla-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:40:17.088113Z'
updated: '2026-10-27T19:00:31.812366Z'
source: https://github.com/pzhao-eng/FlashMLA
source_domain: github.com
fetched_at: '2026-10-27T14:40:17.087204Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Community Ampere port of FlashMLA (branch `a100_replace`). Adapts the Hopper
  FlashMLA to Ampere with a warp-specialized SM80 kernel (`warp_spec=True`, 8 warps
  split 4 consumer QK^T+softmax / 4 producer loads, double-buffered K/V, raw PTX barriers
  for SM80). Scope is decode only: the only entry point is `flash_mla_with_kvcache(q,
  kvcache, block_table, cache_seqlens, dv, tile_scheduler_metadata, num_splits, causal=True)`,
  i.e. paged-KV decode, with no training/backward API. Performance is poor by the
  author''s own admission (''performance of Ampere is currently poor due to register
  overflow''): up to 464 GB/s memory-bound and 59 TFLOPS compute-bound on A100 SXM
  vs 2,039 GB/s and 312 TFLOPS peak. This is the strongest candidate for an sm_80
  sparse-MLA path and it is inference-only — it does not overturn cc-4(a).'
---

GitHub - pzhao-eng/FlashMLA · GitHub
Skip to content
You signed in with another tab or window.
Reload
to refresh your session.
You signed out in another tab or window.
Reload
to refresh your session.
You switched accounts on another tab or window.
Reload
to refresh your session.
Dismiss alert
a100_replace
Branches
Tags
Go to file
Code
Open more actions menu
Latest commit
History
28 Commits
28 Commits
Folders and files
Name
Name
Last commit message
Last commit date
csrc
csrc
flash_mla
flash_mla
tests
tests
.gitignore
.gitignore
.gitmodules
.gitmodules
CMakeLists.txt
CMakeLists.txt
LICENSE
LICENSE
Makefile
Makefile
README.md
README.md
setup.py
setup.py
View all files
Repository files navigation
FlashMLA
Adapted from：
https://github.com/deepseek-ai/FlashMLA/
FlashMLA was initially developed based on Hopper(can refer to:
https://github.com/deepseek-ai/FlashMLA/
), and I adapted it to Ampere GPUs. Due to the different architectures, the performance of Ampere is currently poor due to register overflow. Welcome to add good optimization ideas.
Currently released:
BF16
Paged kvcache with block size of 32
Warp-specialized SM80 kernel (
`warp_spec=True`
): splits 8 warps into 4 consumer (QK^T GEMM + softmax) and 4 producer (global memory loads), with all 8 warps cooperating on PV GEMM. Uses double-buffered K/V loads and raw PTX barriers for SM80 compatibility.
Quick start
Install
```
python setup.py install
```
Benchmark
```
# amphere gpus
python tests/test_flash_mla_sm80.py

# hopper gpus
python tests/test_flash_mla_sm90.py
```
It is able up to 464 GB/s in memory-bound configuration and 59 TFLOPS in computation-bound configuration on A100 SXM, using CUDA 12.8.
For
reference
, the peak bandwidth and fp16 FLOPS of A100 SXM are 2039 GB/s and 312 TFLOPS respectively. More efforts are needed to optimize the performance.
Usage
```
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    # Cooperative kernel (default)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    # Warp-specialized kernel (SM80 only)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True, warp_spec=True,
    )
    ...
```
Requirements
Ampere GPUs
CUDA 12.3 and above
PyTorch 2.0 and above
Acknowledgement
FlashMLA is inspired by
FlashAttention 2&3
and
cutlass
projects.
Citation
```
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
About
No description, website, or topics provided.
Resources
Readme
MIT license
Activity
Stars
67
stars
Watchers
1
watching
Forks
12
forks
Report repository
Releases
Packages
Used by
Contributors
Languages
You can’t perform that action at this time.