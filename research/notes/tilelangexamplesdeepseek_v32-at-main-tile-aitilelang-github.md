---
title: tilelang/examples/deepseek_v32 at main · tile-ai/tilelang · GitHub
id: tilelangexamplesdeepseek_v32-at-main-tile-aitilelang-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-csa2-at-32k-sm80
- measured
- pipeline-parallel
- sparse-attention
- all-to-all
- data-pipeline
created: '2026-10-27T13:42:06.178191Z'
updated: '2026-10-27T19:01:45.837798Z'
source: https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32
source_domain: github.com
fetched_at: '2026-10-27T13:42:05.547626Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: docs
deprecated: false
summary: You signed in with another tab or window. Reload to refresh your session.
  You signed out in another tab or window. Re...
---

[Skip to content](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) to refresh your session. Dismiss alert
## Collapse file tree
## Files
Search this repository(forward slash)` forward slash/`
/
# deepseek_v32
/
Copy path
## Directory actions
## More options
More options
## Directory actions
## More options
More options
## Latest commit
[[Bugfix] Fix candidate buffer overflow in deepseek_v32 topk selector (](https://github.com/tile-ai/tilelang/commit/4da574c01326eab5da5345af0aec1d19ce473c6a)[#…](https://github.com/tile-ai/tilelang/pull/3280)
Open commit detailssuccess
Oct 22, 2026
[4da574c](https://github.com/tile-ai/tilelang/commit/4da574c01326eab5da5345af0aec1d19ce473c6a) · Oct 22, 2026
## History
[History](https://github.com/tile-ai/tilelang/commits/main/examples/deepseek_v32)
Open commit details
History
/
# deepseek_v32
/
Copy path
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
| 
### parent directory
 |  
|   |   |  [[Example] Add topk into sparse mla example and append some docs (](https://github.com/tile-ai/tilelang/commit/6021ef32c80387a589f4142360f562a612f3cab5 "\[Example\] Add topk into sparse mla example and append some docs \(#901\)

* Remove unused `fp8_mqa_logits.py` file and update README.md to reflect new directory structure and file descriptions for deepseek_v32 example. Added sections for architecture overview, Lightning Indexer, Top-k Selector, and Sparse MLA Forward implementations.

* Update linting configurations and improve code formatting in deepseek_v32 example scripts

- Added per-file ignores for the inference directory in `pyproject.toml`.
- Refactored code in `topk_selector.py`, `convert.py`, `generate.py`, `kernel.py`, and `model.py` to enhance readability by adjusting spacing and line breaks.
- Ensured consistent formatting across function definitions and assertions for better clarity.

* Refactor test functions in deepseek_v32 example scripts for improved clarity and consistency

- Updated `fp8_lighting_indexer.py` to define a dedicated test function for the lighting indexer.
- Refactored `sparse_mla_fwd_pipelined.py` and `sparse_mla_fwd.py` to standardize test function parameters and improve readability.
- Enhanced `topk_selector.py` by introducing a test function with parameters for batch size and sequence length.
- Ensured all test functions are invoked correctly in the main execution block.

* Enhance test functions in deepseek_v32 example scripts with CUDA requirements and parameterization

- Added CUDA requirements decorators to `test_example_sparse_mla_fwd` and `test_example_sparse_mla_fwd_pipelined`.
- Parameterized test functions to use specific small shapes for testing, improving test coverage and clarity.

* lint fix

* Update README.md to correct image path for DeepSeek V3.2 architecture diagram")[#901](https://github.com/tile-ai/tilelang/pull/901)[)](https://github.com/tile-ai/tilelang/commit/6021ef32c80387a589f4142360f562a612f3cab5 "\[Example\] Add topk into sparse mla example and append some docs \(#901\)

* Remove unused `fp8_mqa_logits.py` file and update README.md to reflect new directory structure and file descriptions for deepseek_v32 example. Added sections for architecture overview, Lightning Indexer, Top-k Selector, and Sparse MLA Forward implementations.

* Update linting configurations and improve code formatting in deepseek_v32 example scripts

- Added per-file ignores for the inference directory in `pyproject.toml`.
- Refactored code in `topk_selector.py`, `convert.py`, `generate.py`, `kernel.py`, and `model.py` to enhance readability by adjusting spacing and line breaks.
- Ensured consistent formatting across function definitions and assertions for better clarity.

* Refactor test functions in deepseek_v32 example scripts for improved clarity and consistency

- Updated `fp8_lighting_indexer.py` to define a dedicated test function for the lighting indexer.
- Refactored `sparse_mla_fwd_pipelined.py` and `sparse_mla_fwd.py` to standardize test function parameters and improve readability.
- Enhanced `topk_selector.py` by introducing a test function with parameters for batch size and sequence length.
- Ensured all test functions are invoked correctly in the main execution block.

* Enhance test functions in deepseek_v32 example scripts with CUDA requirements and parameterization

- Added CUDA requirements decorators to `test_example_sparse_mla_fwd` and `test_example_sparse_mla_fwd_pipelined`.
- Parameterized test functions to use specific small shapes for testing, improving test coverage and clarity.

* lint fix

* Update README.md to correct image path for DeepSeek V3.2 architecture diagram")  | Sep 29, 2025  |  
|   |   |  [[Cleanup] Remove obsolete disable-fast-math pass config (](https://github.com/tile-ai/tilelang/commit/5899e626c833bb6b93f87a5bf4a33ebcaeec3fe2 "\[Cleanup\] Remove obsolete disable-fast-math pass config \(#3098\)

\[Cleanup\] Remove obsolete disable-fast-math config")[#3098](https://github.com/tile-ai/tilelang/pull/3098)[)](https://github.com/tile-ai/tilelang/commit/5899e626c833bb6b93f87a5bf4a33ebcaeec3fe2 "\[Cleanup\] Remove obsolete disable-fast-math pass config \(#3098\)

\[Cleanup\] Remove obsolete disable-fast-math config")  | Aug 27, 2026  |  
|   |   |  [[Example] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper (](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")[#3224](https://github.com/tile-ai/tilelang/pull/3224)[)](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")  | Sep 16, 2026  |  
|   |   |  [[Example] Migrate eligible examples to eager style (](https://github.com/tile-ai/tilelang/commit/b9a5907a2025fcef91aa0489614c78895e13e052 "\[Example\] Migrate eligible examples to eager style \(#2010\)

* \[Example\] Migrate eligible examples to eager style

* fix wrong shape annotation in gemm_split_k examples

* fix incorrect dtype in gemm_streamk example

* fix

* revert some changes

* Refactor attention functions to use new parameter structures and improve readability. Updated `example_amd_flash_attn_bwd.py` and `example_amd_flash_attn_fwd.py` to define input parameters more clearly. Enhanced shared memory usage in `flashattn_fwd` and `flashattn_sink` functions across various examples, ensuring compatibility with new configurations. Adjusted kernel definitions and added necessary checks for GPU capabilities.

* lint fix

* FIX

---------

Co-authored-by: LeiWang1999 <leiwang1999@outlook.com>")[#2010](https://github.com/tile-ai/tilelang/pull/2010)[)](https://github.com/tile-ai/tilelang/commit/b9a5907a2025fcef91aa0489614c78895e13e052 "\[Example\] Migrate eligible examples to eager style \(#2010\)

* \[Example\] Migrate eligible examples to eager style

* fix wrong shape annotation in gemm_split_k examples

* fix incorrect dtype in gemm_streamk example

* fix

* revert some changes

* Refactor attention functions to use new parameter structures and improve readability. Updated `example_amd_flash_attn_bwd.py` and `example_amd_flash_attn_fwd.py` to define input parameters more clearly. Enhanced shared memory usage in `flashattn_fwd` and `flashattn_sink` functions across various examples, ensuring compatibility with new configurations. Adjusted kernel definitions and added necessary checks for GPU capabilities.

* lint fix

* FIX

---------

Co-authored-by: LeiWang1999 <leiwang1999@outlook.com>")  | May 21, 2026  |  
| [regression_tilelang_example_deepseek_v32.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/regression_tilelang_example_deepseek_v32.py "regression_tilelang_example_deepseek_v32.py")  | [regression_tilelang_example_deepseek_v32.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/regression_tilelang_example_deepseek_v32.py "regression_tilelang_example_deepseek_v32.py")  |  [[Example] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper (](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")[#3224](https://github.com/tile-ai/tilelang/pull/3224)[)](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")  | Sep 16, 2026  |  
|   |   |  [[CUDA] Adopt multi-staged buffers in examples (](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")[#2836](https://github.com/tile-ai/tilelang/pull/2836)[)](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")  | Aug 4, 2026  |  
|   |   |  [[ROCm] Run portable example validation in CI (](https://github.com/tile-ai/tilelang/commit/48cd23e18168643becb2328f159bba422e533914 "\[ROCm\] Run portable example validation in CI \(#3165\)

* \[ROCm\] Run portable example validation in CI

Signed-off-by: andyluo7 <andy.luo@amd.com>

* \[ROCm\] Make portable example checks fail reliably

Signed-off-by: andyluo7 <andy.luo@amd.com>

---------

Signed-off-by: andyluo7 <andy.luo@amd.com>")[#3165](https://github.com/tile-ai/tilelang/pull/3165)[)](https://github.com/tile-ai/tilelang/commit/48cd23e18168643becb2328f159bba422e533914 "\[ROCm\] Run portable example validation in CI \(#3165\)

* \[ROCm\] Run portable example validation in CI

Signed-off-by: andyluo7 <andy.luo@amd.com>

* \[ROCm\] Make portable example checks fail reliably

Signed-off-by: andyluo7 <andy.luo@amd.com>

---------

Signed-off-by: andyluo7 <andy.luo@amd.com>")  | Sep 18, 2026  |  
|   |   |  [[Example] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper (](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")[#3224](https://github.com/tile-ai/tilelang/pull/3224)[)](https://github.com/tile-ai/tilelang/commit/e350730319443156431efd3de4216fce8ec77c02 "\[Example\] FP8 sparse MLA forward for DeepSeek V3.2 on Hopper \(#3224\)")  | Sep 16, 2026  |  
| [sparse_mla_fwd_pipelined.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/sparse_mla_fwd_pipelined.py "sparse_mla_fwd_pipelined.py")  | [sparse_mla_fwd_pipelined.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/sparse_mla_fwd_pipelined.py "sparse_mla_fwd_pipelined.py")  |  [[CUDA] Adopt multi-staged buffers in examples (](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")[#2836](https://github.com/tile-ai/tilelang/pull/2836)[)](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")  | Aug 4, 2026  |  
|   |   |  [[CUDA] Adopt multi-staged buffers in examples (](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")[#2836](https://github.com/tile-ai/tilelang/pull/2836)[)](https://github.com/tile-ai/tilelang/commit/caf694446d3ef0c7f3dd88c3070f502552f643d3 "\[CUDA\] Adopt multi-staged buffers in examples \(#2836\)")  | Aug 4, 2026  |  
| [test_tilelang_example_deepseek_v32.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/test_tilelang_example_deepseek_v32.py "test_tilelang_example_deepseek_v32.py")  | [test_tilelang_example_deepseek_v32.py](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/test_tilelang_example_deepseek_v32.py "test_tilelang_example_deepseek_v32.py")  |  [[ROCm] Run portable example validation in CI (](https://github.com/tile-ai/tilelang/commit/48cd23e18168643becb2328f159bba422e533914 "\[ROCm\] Run portable example validation in CI \(#3165\)

* \[ROCm\] Run portable example validation in CI

Signed-off-by: andyluo7 <andy.luo@amd.com>

* \[ROCm\] Make portable example checks fail reliably

Signed-off-by: andyluo7 <andy.luo@amd.com>

---------

Signed-off-by: andyluo7 <andy.luo@amd.com>")[#3165](https://github.com/tile-ai/tilelang/pull/3165)[)](https://github.com/tile-ai/tilelang/commit/48cd23e18168643becb2328f159bba422e533914 "\[ROCm\] Run portable example validation in CI \(#3165\)

* \[ROCm\] Run portable example validation in CI

Signed-off-by: andyluo7 <andy.luo@amd.com>

* \[ROCm\] Make portable example checks fail reliably

Signed-off-by: andyluo7 <andy.luo@amd.com>

---------

Signed-off-by: andyluo7 <andy.luo@amd.com>")  | Sep 18, 2026  |  
|   |   |  [[Bugfix] Fix candidate buffer overflow in deepseek_v32 topk selector (](https://github.com/tile-ai/tilelang/commit/4da574c01326eab5da5345af0aec1d19ce473c6a "\[Bugfix\] Fix candidate buffer overflow in deepseek_v32 topk selector \(#3280\)

* Fix candidate buffer overflow in deepseek_v32 topk selector

tl_topk_impl staged radix candidates in a fixed \[2, 4096\] shared
buffer, but nothing bounds the threshold bucket selected by the first
radix pass to 4096 elements. Inputs where more candidates share that
bucket \(e.g. FP32 scores with common low mantissa bits, or all-equal
inputs\) made the atomic appends write past the buffer end and crash
with CUDA_ERROR_ILLEGAL_ADDRESS.

Allocate a \(batch, 2, seq_len\) global-memory workspace in the tl_topk
wrapper instead, bounding both radix buffers by seq_len. The public
tl_topk\(\) signature is unchanged.

Fixes #1351

Signed-off-by: liuyun7345 <liuyun7345@sina.com>

* Document detection scope of the topk regression matrix

In a plain run the low-mantissa cases catch the candidate overflow on
the unfixed kernel, but right past the 4096 boundary \(seq_len 4097\)
the stray write can go unobserved and the case only fails under
compute-sanitizer memcheck. Note this in the test docstring so the
matrix's detection capability is not misread.

Signed-off-by: liuyun7345 <liuyun7345@sina.com>

---------

Signed-off-by: liuyun7345 <liuyun7345@sina.com>")[#…](https://github.com/tile-ai/tilelang/pull/3280)  | Oct 22, 2026  |  
|   |   |  [[Lint] Phaseout Yapf format and embrace ruff format (](https://github.com/tile-ai/tilelang/commit/29051439dbed90583bfad1d16dfca88a95e78709 "\[Lint\] Phaseout Yapf format and embrace ruff format \(#1417\)")[#1417](https://github.com/tile-ai/tilelang/pull/1417)[)](https://github.com/tile-ai/tilelang/commit/29051439dbed90583bfad1d16dfca88a95e78709 "\[Lint\] Phaseout Yapf format and embrace ruff format \(#1417\)")  | Dec 12, 2025  |  
| View all files |  
## [README.md](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32#readme)
Outline
## Directory Structure

```
deepseek_v32/
├── README.md                           # This file
├── figures/                            # Figures and diagrams
├── inference/                          # Inference implementation folder
├── fp8_lighting_indexer.py             # FP8 lighting indexer
├── sparse_mla_bwd.py                   # Sparse MLA backward implementation
├── sparse_mla_fwd.py                   # Sparse MLA forward implementation
├── sparse_mla_fwd_fp8.py               # FP8 (e4m3) sparse MLA forward pass
├── sparse_mla_fwd_pipelined.py         # Pipelined implementation of sparse MLA forward pass
├── topk_selector.py                    # Top-k selector implementation

```

## File Descriptions
### Architecture Overview
The architecture diagram above highlights three key components (shown in green) that correspond to our kernel implementations:
  1. **Lightning Indexer** (`fp8_lighting_indexer.py`) - Efficiently indexes and processes sparse attention patterns using FP8 precision
  2. **Top-k Selector** (`topk_selector.py`) - Selects the top-k most relevant tokens for sparse attention computation
  3. **Multi-Query Attention** (`sparse_mla_fwd.py`, `sparse_mla_fwd_fp8.py`, `sparse_mla_fwd_pipelined.py`, and `sparse_mla_bwd.py`) - Core attention mechanism implementation with sparse MLA (Multi-Latent Attention) forward and backward passes


### Lightning Indexer
Looking at the architecture diagram, the Lightning Indexer sits at the bottom right. It takes the input hidden states and produces compressed representations `{q^A_{t,i}}`, `{k^R_t}`, and `{w^I_{t,j}}`. These FP8-quantized index vectors are what feed into the top-k selector.
The main kernel `mqa_attn_return_logits_kernel` computes similarity scores between query and key indices:

```
T.gemm(
    index_k_shared,
    index_q_shared,
    s,
    transpose_B=True,
    clear_accum=True,
    policy=T.GemmWarpPolicy.FullCol,
)
```

After the matmul, we apply ReLU and aggregate across heads with learned weights:

```
for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
    s_reshaped[bn_i, bq_i, h_i] = (T.max(s[bn_i, bq_i * heads + h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[bn_i]

T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)
```

The result is a `[seq_len, seq_len_kv]` logits matrix. For long sequences, the kernel uses per-token bounds (`CuSeqLenKS`, `CuSeqLenKE`) to skip irrelevant KV positions:

```
for bq_i in T.serial(block_Q):
    cu_k_s_min[0] = T.min(cu_k_s_min[0], T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
for bq_i in T.serial(block_Q):
    cu_k_e_max[0] = T.max(cu_k_e_max[0], T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))
```

The pipelined loop then only processes keys in the `[cu_k_s_min, cu_k_e_max)` range, which is crucial for handling variable-length sequences in distributed training.
### Top-k Selector
The Top-k Selector takes the logits matrix from the indexer and picks the top-k indices for each query. In the architecture diagram, this sits between the Lightning Indexer and the Multi-Query Attention block. The output indices tell the attention layer which KV tokens to actually load and process.
The implementation uses a radix-sort-based approach that processes floats as unsigned integers. Stage 1 does a quick 8-bit pass over the whole sequence:

```
for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    input_idx = s * BLOCK_SIZE + tx
    if input_idx  l_end_idx and input_idx >= l_start_idx and input_idx  seq_len:
        inval_int16 = convert_to_uint16(input[bx, input_idx])
        T.atomic_add(s_histogram[inval_int16], 1)
```

The `convert_to_uint16` function maps floats to uint16 such that larger floats map to larger integers. After building a histogram and doing a cumulative sum, we find the threshold bin:

```
if s_histogram[tx]  l_new_topk and s_histogram[tx + 1] <= l_new_topk:
    s_threshold_bin_id[0] = tx
```

Elements above the threshold go directly to the output. Elements in the threshold bin get collected for further processing:

```
if l_bin_id32  l_threshold_bin_id:
    pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
    index[bx, pos] = input_idx
elif l_bin_id32 == l_threshold_bin_id and l_new_topk  0:
    pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
    s_input_idx[0, pos] = input_idx
```

Stage 2 refines the threshold bin with up to 4 rounds of 8-bit radix sort, processing progressively higher bits. This gives exact top-k selection without sorting the entire sequence.
### Sparse MLA Forward
The Sparse MLA kernel is where the actual attention computation happens. In the architecture diagram, this is the large "Multi-Query Attention (Core Attention)" block at the top. It takes the selected top-k indices and computes attention only over those tokens.
Turning dense MLA into sparse MLA requires surprisingly few changes - essentially just modifying how we iterate and load KV tokens. The key difference from dense MLA (see `../deepseek_mla/example_mla_decode.py`) is the iteration pattern. Dense MLA iterates over all KV positions:

```
# Dense MLA: iterate over full sequence
loop_range = T.ceildiv(seqlen_kv, block_N)
for k in T.Pipelined(loop_range, num_stages=2):
    T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared)
    # ... compute attention over this block
```

Sparse MLA only loads KV positions selected by the top-k selector:

```
# Sparse MLA: iterate over selected indices only
for i_i in T.Pipelined(NI, num_stages=num_stages):
    for bi_i, d_i in T.Parallel(BI, D):
        KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
    # ... compute attention over selected tokens
```

This reduces compute from O(seq_len _seq_len_kv) to O(seq_len_ topk). The causal mask is enforced by checking whether each index position is valid:

```
for bi_i in T.Parallel(BI):
    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i
```

Beyond this sparse indexing, the rest of the attention computation (online softmax, output accumulation) follows the same pattern as dense MLA.
### Sparse MLA Forward (Pipelined)
The pipelined version (`sparse_mla_fwd_pipelined.py`) is a manual pipeline implementation designed to match the schedule of [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm90/prefill/sparse/fwd.cu). It achieves close to 600 TFlops on H800 SXM by carefully orchestrating memory and compute pipelines.
The key difference is splitting the warp groups into specialized roles:

```
if tx  128:
    # Consumer 0: computes left half of output (D//2 dimensions)
    # Handles QK matmul, softmax, and PV for left half

elif tx >= 128 and tx  256:
    # Consumer 1: computes right half of output (D//2 dimensions)
    # Only does PV matmul for right half

elif tx >= 256:
    # Producer: loads KV data from global memory
    # Uses async copy with barriers to feed consumers
```

The producer thread group (tx >= 256) uses double buffering with barriers to keep consumers fed:

```
num_stages = 2
KV_shared = T.alloc_shared([num_stages, BI, D], dtype)
bar_k_ready = T.alloc_barrier(arrive_count=[128] * num_stages)
bar_k_free = T.alloc_barrier(arrive_count=[256] * num_stages)

# Producer alternates between slices of one multi-version buffer
for i_i in T.serial(T.ceildiv(NI, num_stages)):
    for stage in T.unroll(num_stages):
        T.barrier_wait(bar_k_free[stage], ((i_i  1) ^ 1))
        # ... load KV into KV_shared[stage, :, :]
        T.cp_async_barrier_noinc(bar_k_ready[stage])
```

Consumer threads wait on barriers and process buffers as they become ready. This manual orchestration hides memory latency behind compute, which is why it outperforms the simpler auto-pipelined version. The output dimension is also split in half so that the two consumer groups can work in parallel on different parts of the matmul.
### Sparse MLA Forward (FP8)
`sparse_mla_fwd_fp8.py` takes an fp8 e4m3 query and KV cache and returns a bf16 output. FP8 halves the KV cache footprint, which is why DeepSeek introduced an fp8 KV cache for V3.2.
Switching `dtype` to `T.float8_e4m3` in the bf16 kernel is not enough. On SM90 the fp8 form of `wgmma.mma_async` has no `imm-trans-a` / `imm-trans-b` operands, unlike the bf16 form, so both GEMM operands must be K-major:

```
wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {...}, %a, %b, p, 1, 1, 0, 1;  // ok
wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {...}, %a, %b, p, 1, 1, 0, 1;
// ptxas: error : Arguments mismatch for instruction 'wgmma.mma_async with FP8 types'

```

The QK product is fine -- `Q` and `KV` are both contiguous along `dim`. The PV product is not: it contracts over `block_I`, while `KV_shared` is laid out `[block_I, dim]`. Without a K-major V, `T.gemm` falls back to `mma.sync` and the whole kernel runs 1.81x slower.
Storing V column-major in global memory (what FlashAttention-3 does for dense fp8 attention) does not work here, because the top-k gather reads one contiguous row per selected token; a column-major cache would turn it into per-element scattered loads. So the kernel keeps the coalesced gather and transposes V in shared memory:

```
# 16x16 tiles: read 128-bit vectors along dim, swap in registers, write back
for bi_o, d_o in T.Parallel(BI // TM, D // TN):
    for i in T.serial(TM):
        for j in T.vectorized(TN):
            kv_t_frag[i, j] = KV_shared[bi_o * TM + i, d_o * TN + j]
    for j in T.serial(TN):
        for i in T.serial(TM):
            KV_T_shared[d_o * TN + j, bi_o * TM + i] = kv_t_frag[i, j]
...
T.gemm(S_shared, KV_T_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

The transpose only depends on `KV_shared`, so it is issued before the QK GEMM and overlaps with the asynchronous wgmma that follows. Measured on H800 SXM at S=8192, H=128, topk=2048: 25.79 ms with the `mma.sync` fallback, 14.24 ms with a K-major V, and 13.77 ms for the bf16 kernel of the same structure.
Two notes on the numbers. FP8 does not beat bf16 here -- the kernel is limited by softmax and shared-memory traffic rather than Tensor Core throughput, so the gain from FP8 arithmetic has nothing to absorb it; the win is the smaller KV cache. And the manually pipelined bf16 kernel above is considerably faster than both; applying the same K-major V change there is future work.
### Sparse MLA Backward
The Sparse MLA backward kernel (`sparse_mla_bwd.py`) computes gradients with respect to queries (dQ) and key-values (dKV) for the sparse attention mechanism. Like the forward pass, it processes only the selected top-k indices, maintaining O(seq_len * topk) complexity.
The backward pass consists of three main stages:
**1. Preprocessing** : Computes delta values (row-wise dot products of output and output gradient):

```
for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
    T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
    T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
    for i, j in T.Parallel(block_ND, block_ND):
        acc[i, j] += o[i, j] * do[i, j]
T.reduce_sum(acc, delta, 1)
```

**2. Main Backward Computation** : Computes gradients through sparse attention:

```
# Sparse MLA backward: iterate over selected indices only
for i_i in T.Pipelined(NI, num_stages=num_stages):
    # Load KV data for selected indices
    for bi_i, d_i in T.Parallel(BI, D):
        KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz, i_i * BI + bi_i], bz, d_i]

    # Recompute attention scores for backward
    T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

    # Apply softmax gradient: dP = P * (dP_raw - Delta)
    for h_i, bi_i in T.Parallel(padded_H, BI):
        acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * padded_H + h_i]) * sm_scale
```

The key gradient computations are:
  * **dQ = dP @ K** (query gradients)
  * **dK = dP^T @ Q** (key gradients)
  * **dV = P^T @ dO** (value gradients)


**3. Atomic Sparse Updates** : Uses atomic operations for dKV accumulation:

```
# Atomically update dKV at selected indices
for bi_i, d_i in T.Parallel(BI // split_store, D // 4):
    T.atomic_addx4(dKV[by, Indices[by, s_i, bz, i_i * BI + bi_i + s * (BI // split_store)], bz, d_i * 4], acc_dkv_shared[bi_i, d_i * 4])
```

**Performance** : The sparse MLA backward achieves excellent performance:
  * **H800 SXM** : ~100 TFlops
  * **H200 SXM** : ~115 TFlops


The implementation efficiently handles the irregular memory access patterns inherent in sparse attention while maintaining high compute utilization through careful memory management and atomic update strategies. Note that this is a relatively naive implementation that requires further optimization.
You can’t perform that action at this time. 
