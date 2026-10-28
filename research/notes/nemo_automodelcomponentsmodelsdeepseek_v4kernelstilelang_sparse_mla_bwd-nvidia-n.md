---
title: nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd
  | NVIDIA NeMo AutoModel
id: nemo_automodelcomponentsmodelsdeepseek_v4kernelstilelang_sparse_mla_bwd-nvidia-n
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:36:29.190620Z'
updated: '2026-10-27T19:00:31.769421Z'
source: https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_bwd
source_domain: docs.nvidia.com
fetched_at: '2026-10-27T14:36:29.189552Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: NVIDIA NeMo AutoModel API docs for `tilelang_sparse_mla_bwd` — the DeepSeek-V4
  sparse MLA backward, implemented in TileLang, with `preprocess` (row-wise delta),
  `bwd` (B, S, S_kv, H, D, topk, block_size=32), `postprocess`, and a `sparse_mqa_bwd_interface(q,
  kv, attn_sink, o, do, topk_idxs, lse, ...)` taking int32 top-k indices and fp32
  LSE from the forward. Confirms a production-grade DSA-family training backward exists
  in NVIDIA's stack, but it is TileLang (wgmma/SM90-class) and carries no sm_80 claim.
  Relevant to cc-4 as a 'the backward exists, but not on Ampere' datapoint.
---

nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd | NVIDIA NeMo AutoModel
For AI agents: a documentation index is available at the root level at /llms.txt. Append /llms.txt to any URL for a page-level index, or .md for the markdown version of any page.
Skip to navigation
This page is for version Nightly (default).
For other versions, use one of these documentation indexes:
- Nightly (default): https://docs.nvidia.com/nemo/automodel/nightly/llms.txt
- Latest: https://docs.nvidia.com/nemo/automodel/latest/llms.txt
- 0.7.0 · 26.10: https://docs.nvidia.com/nemo/automodel/v0.7/llms.txt
- 0.5.0 · 26.06: https://docs.nvidia.com/nemo/automodel/v0.5/llms.txt
- 0.4.0 · 26.04: https://docs.nvidia.com/nemo/automodel/v0.4/llms.txt
Module Contents
Functions
Name
Description
`_bwd_threads_for_heads`
Threads per backward kernel: 128 for up to 16 heads, 256 for up to 32, 512 beyond.
`bwd`
-
`postprocess`
-
`preprocess`
-
`sparse_mqa_bwd_interface`
Backward interface for V4 sparse MQA attention.
API
```
nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd._bwd_threads_for_heads(    num_heads: int) -> int
```
Threads per backward kernel: 128 for up to 16 heads, 256 for up to 32, 512 beyond.
```
nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd.bwd(    B,    S,    S_kv,    H,    D,    topk,    sm_scale = None,    block_size = 32,    num_stages = 0,    threads = 128,    indices_dtype = T.int32,    dtype = T.bfloat16,    accum_dtype = T.float32)
```
```
nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd.postprocess(    B,    S_kv,    D,    block_N = 64,    threads = 128,    dtype = T.bfloat16,    accum_dtype = T.float32)
```
```
nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd.preprocess(    B,    S,    H,    D,    block_ND = 32,    num_stages = 5,    dtype = T.bfloat16,    accum_dtype = T.float32)
```
```
nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd.sparse_mqa_bwd_interface(    q,    kv,    attn_sink,    o,    do,    topk_idxs,    lse,    sm_scale = None,    return_dkv_accum_dtype = False)
```
Backward interface for V4 sparse MQA attention.
Parameters:
q
[B, S, H, D] bf16
kv
[B, S_kv, D] bf16
attn_sink
[H] fp32
o
[B, S, H, D] bf16 (forward output)
do
[B, S, H, D] bf16 (grad of output)
topk_idxs
[B, S, topk] int32
lse
[B, S, H] fp32 (log-sum-exp from forward)
sm_scale
Defaults to
None
float or None
Returns:
[B, S, H, D] bf16