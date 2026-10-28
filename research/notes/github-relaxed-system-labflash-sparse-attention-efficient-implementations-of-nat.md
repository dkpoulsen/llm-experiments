---
title: 'GitHub - Relaxed-System-Lab/Flash-Sparse-Attention: 🚀🚀 Efficient implementations
  of Native Sparse Attention · GitHub'
id: github-relaxed-system-labflash-sparse-attention-efficient-implementations-of-nat
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:34:41.278361Z'
updated: '2026-10-27T19:00:31.760566Z'
source: https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention
source_domain: github.com
fetched_at: '2026-10-27T14:34:41.277213Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Flash-Sparse-Attention (FSA), the official implementation of arXiv 2508.18224
  (ICLR 2026), an optimized Triton NSA kernel. Explicitly tested on ''NVIDIA Ampere
  or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)'',
  fp16/bf16, head dim <=256, GQA group sizes 1-16, and ''Training and inference (prefill)''
  — the README''s usage example calls `loss.backward()` through a 65,536+32,768-token
  variable-length batch, so this is a runnable sm_80 NSA-family TRAINING path (requires
  flash-attn==2.6.3). This answers cc-4''s third falsifier: ''run NSA instead'' is
  NOT itself a Hopper-only porting project. Its kernel claim is that NSA''s selected-attention
  kernel wastes work padding GQA groups below 8, so FSA reorders the loop nest (KV
  blocks outer, queries inner) into main+reduction+online-softmax kernels.'
---

GitHub - Relaxed-System-Lab/Flash-Sparse-Attention: 🚀🚀 Efficient implementations of Native Sparse Attention · GitHub
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
47 Commits
47 Commits
Folders and files
Name
Name
Last commit message
Last commit date
benchmarks
benchmarks
fsa
fsa
fsa_preview
fsa_preview
nsa_ref
nsa_ref
scripts
scripts
test
test
.gitignore
.gitignore
LICENSE
LICENSE
README.md
README.md
requirements-dev.txt
requirements-dev.txt
requirements.txt
requirements.txt
utils.py
utils.py
View all files
Repository files navigation
This repository provides the official implementation of
F
lash
S
parse
A
ttention (FSA)
, which includes a novel kernel design that enables efficient Native Sparse Attention (NSA) across a wide range of popular LLMs on modern GPUs.
News
Method
Advantages
Features
Installation
Usage
Instantiate FSA Module
Train with FSA
Evaluation
Benchmark FSA Module
Benchmark FSA Selected Attention Module
Performance
Kernel Performance
End-to-end Performance
Citation
Acknowledgments
News
$\texttt{[2025-09, upcoming]}$
:
🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
$\texttt{[2025-08]}$
:
💥 Our
Arxiv paper
is released.
$\texttt{[2025-08]}$
:
🎈 Beta version of one-step decoding is released, check the code residing in
`fsa_preview`
.
$\texttt{[2025-08]}$
:
🎉 Open sourced
`Flash-Sparse-Attention`
, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.
Method
For NSA selected attention module, the major system bottleneck, NSA loops over query tokens in the outer loop and loops over KV blocks in the inner loop. To optimize performance, NSA batches query heads that share the same key-value head for more efficient computation. However, when GQA group size is not sufficiently large, NSA selected attention kernel must pad data to satisfy hardware requirements on the matrix dimensions of matrix multiplication. Specifically, for NVIDIA GPUs, the
warp-level matrix multiply-accumulate instructions
require that each dimension of a matrix tile executed on a warp must be larger than the specified value (at least 8 on Hopper GPUs). In
Triton
, NSA selected attention kernel must also satisfy that each dimension of a matrix tile executed on a thread block must be at least 16.
In contrast, FSA exchanges the kernel loop order of original NSA kernel design, i.e., FSA loops over KV blocks in the outer loop and loops over query tokens in the inner loop. To optimize performance, FSA decouples the computation into three major kernels: (i) the main kernel batches query tokens that attend to the same KV block and stores the partial results to a buffer, (ii) the reduction kernel accumulates attention results for each query token, and (iii) the online softmax kernel that handles online softmax statistics computation. The key insight behind this arrangement is to effectively reduce unnecessary memory access and computations for the padded data, while avoiding
`atomic`
additions for accumulating attention results for each query token across KV blocks.
The concrete computation process comparison between NSA (left) and FSA main kernel (right) can be visualized as follows:
Advantages
🚀 The speedup of FSA originates from significantly lowered kernel-level memory access volume and computations.
Under varied GQA group sizes, NSA hyperparameters block size
$B_K=64$
and topk-k value
$T=16$
, 64K sequence length, 4 KV heads, the execution latency comparisons between NSA and our method are as follows (execution latency of our method is normalized to 1):
Features
FSA provides an optimized kernel implementation for NSA selected attention module. Without modifying NSA algorithm, FSA provides an efficient Triton-based implementation for GQA group sizes smaller than 8, which is more popular on state-of-the-art large language models (LLMs), on modern high-performance NVIDIA GPUs. For GQA group sizes larger than or equal to 8, FSA usually chooses to fall back to the original NSA implementation for better performance.
FSA is currently well tested with:
NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM);
Datatype of fp16 and bf16;
The same head dimension (less than or equal to 256) across query, key, and value;
Varied GQA group sizes, ranging from 1 to 16;
Training and inference (prefill).
Installation
The following requirements should be satisfied:
PyTorch
>= 2.4
Triton
>=3.0
transformers
>=4.45.0
datasets
>=3.3.0
accelerate
>= 1.9.0
flash-attn
==2.6.3
You can install dependencies for FSA with:
```
pip install -r requirements.txt
```
Usage
Instantiate FSA Module
We provide
`FlashSparseAttention`
class for you to use, it can be used as the following example:
```
import torch
from fsa.module.fsa import FlashSparseAttention, RopeConfig

FSA = (
    FlashSparseAttention(
        hidden_size=4096,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=128,
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        topk=16,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
        rope_config=RopeConfig(
            max_position_embeddings=131072,
            head_dim=128,
            rope_theta=500000,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        ),
    )
    .cuda()
    .to(torch.bfloat16)
)
# random input
seqlens = torch.LongTensor([65536, 32768]).int().cuda()

cu_seqlens = torch.cat(
    [
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seqlens, dim=0),
    ],
    dim=0,
).to(torch.int32)
x = torch.randn(cu_seqlens[-1], 4096, device="cuda", dtype=torch.bfloat16)

y = FSA(x, cu_seqlens)
loss = (y * torch.randn_like(y)).sum(-1).mean()
loss.backward()
```
Under the hood, the
`FSATopkSparseAttention`
class is called, provding the optimized kernels that accelerate the NSA selected attention module.
Train with FSA
Training with FSA can be esaily achieved by replacing the attention module. The only thing you may need to handle is to instantiate the FSA module, and compute the
`cu_seqlens`
for FSA. We provide an example on how to insert FSA into a LLM in
`SparseLlamaAttention`
.
Evaluation
Benchmark FSA Module
We provide detailed commands in
`scripts/run_unit_test.sh`
for convenient benchmarking of FSA module. The benchmarking provides correctness comparison of forward and backward outputs, performance comparison, and memory usage comparison.
Benchmark FSA Selected Attention Module
The optimized NSA selected attention module, which is the major system bottleneck, can be benchmarked through the commands in
`scripts/run_unit_test_sel_attn.sh`
.
Tip
Try varied
`gqa`
,
`seqlen`
,
`block_size`
,
`topk`
argument in the provided scripts for more comprehensive benchmarking on your machine! Compared to benchmarking the FSA attention module, benchmarking the FSA selected attention module usually provides a higher speedup.
Performance
Kernel Performance
Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple (
$64$
,
$16$
) / (
$128$
,
$8$
) represents the block size
$BK$
and top-k value
$Topk$
, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.
End-to-end Performance
End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.
Citation
Please cite our
arXiv paper
:
```
@misc{yan2026fsaalternativeefficientimplementation,
  title={{FSA}: An Alternative Efficient Implementation of Native Sparse Attention Kernel},
  author={Ran Yan and Youhe Jiang and Zhuoming Chen and Haohui Mai and Beidi Chen and Binhang Yuan},
  year={2026},
  eprint={2508.18224},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2508.18224},
}
```
Acknowledgments
NSA paper:
Native Sparse Attention
NSA reference implementation:
Native Sparse Attention Triton
About
🚀🚀 Efficient implementations of Native Sparse Attention
github.com/Relaxed-System-Lab/Flash-Sparse-Attention
Topics
kernels
large-language-models
machine-learning-systems
Resources
Readme
Apache-2.0 license
Activity
Custom properties
Stars
591
stars
Watchers
1
watching
Forks
16
forks
Report repository
Releases
Packages
Contributors
Languages
You can’t perform that action at this time.