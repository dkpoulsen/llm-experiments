---
title: 'GitHub - fla-org/native-sparse-attention: 🐳 Efficient Triton implementations
  for "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
  · GitHub'
id: github-fla-orgnative-sparse-attention-efficient-triton-implementations-for-nativ
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:34:41.118256Z'
updated: '2026-10-27T19:00:31.757800Z'
source: https://github.com/fla-org/native-sparse-attention
source_domain: github.com
fetched_at: '2026-10-27T14:34:41.117268Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'fla-org''s Triton implementation of NSA (the reference implementation the
  NSA paper''s own efficiency section is built on), with full NSA training support
  via `flame` (2025-03-15), an online top-k selection kernel that avoids materializing
  the attention matrix, a fused selected+sliding Triton kernel, and variable-block-count
  support. Ships `parallel_nsa` with autograd (q/k/v require_grad) and `benchmarks/benchmark_nsa.py`.
  Its published benchmark table is the independent check on the paper''s 6.3x/3.4x
  at 32k: at T=16,384, nsa 20.165 ms vs flash 53.634 ms = 2.66x forward, nsa_bwd 116.298
  ms vs flash_bwd 204.354 ms = 1.76x backward; at T=8,192, 9.954 vs 13.465 = 1.35x
  fwd and 49.575 vs 52.567 = 1.06x bwd; at T=4,096, 4.870 vs 3.520 = 0.72x fwd (NSA
  SLOWER) and 23.083 vs 14.193 = 0.61x bwd (NSA SLOWER). So this reimplementation
  reproduces a shrinking advantage well below the paper''s headline ratios at every
  length it reports, and is a net loss below ~8k. Triton, so arch-portable to sm_80
  in principle; the repo does not state a tested A100 receipt.'
---

GitHub - fla-org/native-sparse-attention: 🐳 Efficient Triton implementations for "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" · GitHub
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
main
Branches
Tags
Go to file
Code
Open more actions menu
Latest commit
History
90 Commits
90 Commits
Folders and files
Name
Name
Last commit message
Last commit date
.github
.github
3rdparty
3rdparty
benchmarks
benchmarks
configs
configs
native_sparse_attention
native_sparse_attention
tests
tests
.flake8
.flake8
.gitignore
.gitignore
.gitmodules
.gitmodules
.pre-commit-config.yaml
.pre-commit-config.yaml
LICENSE
LICENSE
README.md
README.md
fla
fla
flame
flame
pyproject.toml
pyproject.toml
setup.py
setup.py
train.py
train.py
train.sh
train.sh
View all files
Repository files navigation
🐳 Native Sparse Attention
Efficient Triton implementations for
Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
.
News
[2025-03-15] We now add full support for NSA training with
`flame`
.
[2025-02-25] Introduce an online top‑k selection kernel that avoids materializing the attention matrix during selection.
[2025-02-24] Add support for a fused Triton kernel combining selected attention with sliding attention.
[2025-02-21] Enable handling of a variable number of selected blocks for queries across different positions and batches.
Setup
To get started, clone the
`native-sparse-attention`
repository and install the required dependencies:
```
git clone https://github.com/fla-org/native-sparse-attention.git
cd native-sparse-attention
git submodule update --init --recursive
pip install .
```
Usage
To test the correctness of NSA:
```
pytest tests/test_nsa.py
```
To validate the correctness of NSA with top‑k selection, run the command below. Please note that the initial trial may take some time as the kernel compiles, but subsequent runs will be faster.
```
pytest tests/test_nsa_with_compression.py
```
To measure the efficiency of NSA:
```
python benchmarks/benchmark_nsa.py
```
To direct use our NSA kernel:
```
from native_sparse_attention.ops.parallel import parallel_nsa

B, T, H, HQ, D = 4, 2048, 4, 64, 64
block_size = 64
window_size = 64

q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)

# randomly generated block indices
block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[b, t, h, :len(i_i)] = i_i
block_indices = block_indices.sort(-1)[0]
block_counts = torch.randint(1, S + 1, (B, T, H), device=device)

parallel_nsa(
    q=q,
    k=k,
    v=v,
    g_slc=g_slc,
    g_swa=g_swa,
    block_indices=block_indices,
    block_counts=block_counts,
    block_size=block_size,
    window_size=window_size,
)

# variable-length inputs are supported as well
# randomly split the sequence into N segments
N, T = 4, 2048
offsets = torch.cat([
    torch.tensor([0], dtype=torch.long),
    torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
    torch.tensor([T], dtype=torch.long)
], 0).cuda().sort()[0]
# seq-first required for inputs with variable lengths
q = torch.rand((1, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
k = torch.rand((1, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
v = torch.rand((1, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)

# randomly generated block indices
block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[b, t, h, :len(i_i)] = i_i
block_indices = block_indices.sort(-1)[0]
block_counts = torch.randint(1, S + 1, (B, T, H), device=device)

parallel_nsa(
    q=q,
    k=k,
    v=v,
    g_slc=g_slc,
    g_swa=g_swa,
    block_indices=block_indices,
    block_counts=block_counts,
    block_size=block_size,
    window_size=window_size,
    cu_seqlens=offsets
)
```
Benchmarks
```
Performance:
         T        nsa     nsa_bwd      flash   flash_bwd
0    128.0   0.091168    0.672992   0.020128    0.161504
1    256.0   0.189408    1.222848   0.045024    0.225056
2    512.0   0.435616    2.363264   0.105664    0.503264
3   1024.0   1.043200    5.091552   0.296944    1.323456
4   2048.0   2.322016   11.124559   0.970208    4.076928
5   4096.0   4.869712   23.082577   3.520352   14.193248
6   8192.0   9.953824   49.575199  13.464992   52.566914
7  16384.0  20.164879  116.297920  53.633568  204.353607
```
Citations
```
@inproceedings{Yuan2025NativeSA,
    title   = {Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
    author  = {Jingyang Yuan and Huazuo Gao and Damai Dai and Junyu Luo and Liang Zhao and Zhengyan Zhang and Zhenda Xie and Y. X. Wei and Lean Wang and Zhiping Xiao and Yuqing Wang and Chong Ruan and Ming Zhang and Wenfeng Liang and Wangding Zeng},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276408911}
}
```
About
🐳 Efficient Triton implementations for "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
arxiv.org/abs/2502.11089
Resources
Readme
MIT license
Activity
Custom properties
Stars
1.0k
stars
Watchers
7
watching
Forks
54
forks
Report repository
Releases
Packages
Used by
Contributors
Languages
You can’t perform that action at this time.