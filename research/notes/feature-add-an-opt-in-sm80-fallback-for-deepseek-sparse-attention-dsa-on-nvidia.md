---
title: '[Feature] Add an opt-in SM80 fallback for DeepSeek Sparse Attention (DSA)
  on NVIDIA A100 · Issue #35420 · sgl-project/sglang · GitHub'
id: feature-add-an-opt-in-sm80-fallback-for-deepseek-sparse-attention-dsa-on-nvidia
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:35:27.285564Z'
updated: '2026-10-27T19:00:31.763136Z'
source: https://github.com/sgl-project/sglang/issues/35420
source_domain: github.com
fetched_at: '2026-10-27T14:35:27.284623Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'SGLang issue #35420 (opened 2026-08-19, closed inactive) requesting an opt-in
  SM80 fallback for DeepSeek Sparse Attention on A100. Establishes that ''SGLang currently
  has no end-to-end CUDA path for running DeepSeek Sparse Attention (DSA), including
  DeepSeek-V3.2, on NVIDIA A100 / SM80'' with two independent blockers: (1) the DSA
  indexer''s `DSAPagedMQALogitsBackend.AUTO` selects DeepGEMM on CUDA, and the paged-MQA
  logits implementation is unavailable on SM80; (2) the BF16 DSA decode path selects
  FA3 on pre-Blackwell GPUs, and SM80 FA3 does not support DSA''s asymmetric head
  dims (QK 576 = 512 nope + 64 RoPE, V 512), failing with ''RuntimeError: Only Hopper
  supports different V headdim''. The proposed fix is explicitly inference-only —
  a chunked Torch sparse-MLA reference for correctness, then SM80 Triton paged-MQA
  logits and a BF16 Triton sparse-MLA DECODE kernel with online softmax. No backward,
  no training path. Confirms cc-4''s ''no sm_80 DSA training path'' as of Aug 2026.'
---

[Feature] Add an opt-in SM80 fallback for DeepSeek Sparse Attention (DSA) on NVIDIA A100 · Issue #35420 · sgl-project/sglang · GitHub
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
[Feature] Add an opt-in SM80 fallback for DeepSeek Sparse Attention (DSA) on NVIDIA A100
#35420
New issue
Copy link
New issue
Copy link
Closed
Closed
[Feature] Add an opt-in SM80 fallback for DeepSeek Sparse Attention (DSA) on NVIDIA A100
#35420
Copy link
Labels
inactive
Description
xiangui33423
opened
on Aug 19, 2026
Issue body actions
Checklist
If this is not a feature request but a general question, please start a discussion at
https://github.com/sgl-project/sglang/discussions
. Otherwise, it will be closed.
Please use English. Otherwise, it will be closed.
Motivation
Motivation
SGLang currently has no end-to-end CUDA path for running DeepSeek Sparse
Attention (DSA), including DeepSeek-V3.2, on NVIDIA A100 / SM80.
I am testing with:
GPU:  × NVIDIA A100 <40GB/80GB>
Compute capability: SM80
Model: <MODEL_ID>
SGLang commit/version: <COMMIT_OR_VERSION>
CUDA: <CUDA_VERSION>
PyTorch: <PYTORCH_VERSION>
Tensor parallel size: <TP_SIZE>
KV cache dtype: BF16
The current DSA execution path has two independent SM80 blockers:
DSA indexer
`DSAPagedMQALogitsBackend.AUTO`
selects DeepGEMM on CUDA.
The paged-MQA logits implementation used by the DSA indexer is not
available on SM80.
Relevant path:
`python/sglang/srt/layers/attention/dsa/dsa_indexer.py::_get_topk_paged`
Sparse-MLA attention
The BF16 DSA decode path selects FA3 on pre-Blackwell NVIDIA GPUs.
DSA uses asymmetric dimensions:
QK head dimension: 576 (512 nope + 64 RoPE)
V head dimension: 512
The SM80 FA3 implementation does not support this different V head
dimension and can fail with:
`RuntimeError: Only Hopper supports different V headdim`
Relevant path:
`python/sglang/srt/layers/attention/dsa_backend.py::_forward_fa3`
A separate issue may also occur if the SM90 thread-block-cluster Top-K v2
kernel is selected on SM80.
This feature request is not proposing to port the whole DeepGEMM or FA3
projects to Ampere. Instead, I would like to add opt-in, SGLang-native
fallback backends for SM80.
Proposed implementation
I am willing to implement this incrementally.
Phase 1: correctness fallbacks
Add explicit Torch backends:
`--dsa-paged-mqa-logits-backend torch`
`--dsa-prefill-backend torch`
`--dsa-decode-backend torch`
The first implementation would:
support BF16 main KV cache;
implement generic DSA paged-MQA logits without DeepGEMM;
implement a chunked Torch sparse-MLA reference without FA3;
skip all DeepGEMM schedule-metadata creation and replay refresh when the
selected indexer backend is not DeepGEMM;
support eager execution first;
remain opt-in and leave all existing defaults unchanged.
This phase is intended as a correctness reference and functional fallback,
not as the final performance solution.
Phase 2: SM80 Triton kernels
After validating the Torch path:
add a Triton paged-MQA logits kernel for SM80;
pass FP8 E4M3FN query/index-cache data as
`uint8`
;
decode E4M3FN values inside the kernel because A100 has no native FP8
Tensor Core path;
use FP32 accumulation and the existing DSA Top-K transform;
add a BF16 Triton sparse-MLA decode kernel with online softmax;
keep both kernels explicitly opt-in.
Proposed backend names:
`--dsa-paged-mqa-logits-backend triton`
`--dsa-prefill-backend triton_sm80`
`--dsa-decode-backend triton_sm80`
Phase 3: runtime integration
Once the kernels are correct:
prefill/extend;
target verification and draft-extend-v2;
CUDA Graph capture and replay;
overlap scheduling;
long-context memory chunking;
A100 kernel and end-to-end tests.
The SM80 backend would not become an automatic default until all these modes
are validated.
Initial scope
The initial scope would be:
NVIDIA A100 / SM80;
generic DSA used by DeepSeek-V3.2;
BF16 main KV cache;
page size 64 for the DSA index cache;
index head dimension 128;
eager prefill and decode.
Initially out of scope:
DeepSeek-V4's dedicated DSV4 backend;
FP8 main sparse-MLA KV cache;
SM86/SM89;
context parallelism, PD disaggregation and HiCache;
unrelated model-weight, quantization or MoE compatibility issues on A100.
Compatibility
The implementation would be guarded behind explicit backend values.
It would not change the existing backend selection or behavior for:
Hopper / SM90;
Blackwell / SM100 and SM12x;
AMD ROCm;
users not selecting the new backend.
Questions for maintainers
Before starting the implementation, I would like guidance on:
Is an opt-in A100/SM80 DSA fallback within SGLang's accepted hardware
support scope?
Would maintainers accept a correctness-first Torch fallback as the first PR?
Are
`torch`
,
`triton`
, and
`triton_sm80`
acceptable backend names?
Should the Torch paged-MQA implementation be shared with the existing DSV4
reference or kept in the generic DSA directory?
Is an A100 CI runner available for registered kernel tests?
Should the work be submitted as multiple small PRs following the phases
above?
Reproduction
Launch command:
```
  <PASTE_THE_COMPLETE_LAUNCH_COMMAND>
```
First relevant error:
```
  <PASTE_THE_EXACT_DEEPGEMM_OR_FA3_ERROR>
```
Environment:
```
  python3 -m sglang.check_env

  <PASTE_CHECK_ENV_OUTPUT>
```
Related resources
[Feature] Support fa3 backend for Ampire GPUs again
#5536
— FA3 on Ampere and the different V head-dimension failure:
[Feature] Support fa3 backend for Ampire GPUs again
#5536
[Feature] FA3 support sm80
#5911
— general FA3 SM80 request:
[Feature] FA3 support sm80
#5911
[DSA] Add an arch-independent torch paged-MQA-logits backend with a fused Triton fast path
#31480
— architecture-independent Torch/Triton paged-MQA proposal:
[DSA] Add an arch-independent torch paged-MQA-logits backend with a fused Triton fast path
#31480
[DSA] Enable sparse MLA decode+prefill on SM120/SM121 (consumer Blackwell) via flashinfer's packed sparse backend
#31481
— DSA attention support for SM120/SM121, not SM80:
[DSA] Enable sparse MLA decode+prefill on SM120/SM121 (consumer Blackwell) via flashinfer's packed sparse backend
#31481
Fix DeepSeek DSA top-k v2 defaulting on below SM90
#33266
— pre-SM90 DSA Top-K v2 gating:
Fix DeepSeek DSA top-k v2 defaulting on below SM90
#33266
Related resources
No response
Reactions are currently unavailable
Activity
github-actions
closed this
as
completed
on Oct 19, 2026
github-actions
added
inactive
on Oct 19, 2026
github-actions
commented
on
Oct 19, 2026
github-actions
Bot
on Oct 19, 2026
– with
GitHub Actions
Contributor
More actions
This issue has been automatically closed due to inactivity. Please feel free to reopen it if needed.
Sign up for free
to join this conversation on GitHub.
Already have an account?
Sign in to comment
Metadata
Metadata
Assignees
No one assigned
Labels
inactive
Type
No type
Projects
No projects
Milestone
No milestone
Relationships
None yet
Development
No branches or pull requests
Issue actions
Open in GitHub Copilot app
You can’t perform that action at this time.