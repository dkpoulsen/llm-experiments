---
title: '[Example][DeepSeek-V3.2] Adaptive threads for sparse MLA backward by Butterfingrz
  · Pull Request #2592 · tile-ai/tilelang · GitHub'
id: exampledeepseek-v32-adaptive-threads-for-sparse-mla-backward-by-butterfingrz-pul
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:36:29.239457Z'
updated: '2026-10-27T19:00:31.772302Z'
source: https://github.com/tile-ai/tilelang/pull/2592
source_domain: github.com
fetched_at: '2026-10-27T14:36:29.238197Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'TileLang PR #2592 (merged 2026-07-22), adaptive thread count for the sparse
  MLA backward kernel. Identifies this kernel as ''the sole hotspot of the GLM-5 DSA
  (DeepSeek Sparse Attention) training backward path (`SparseMLA.autograd.Function`),
  computing sparse MLA gradients for the latent geometry (DQKV=576, DV=512, topk=2048)''.
  Profiling is on H200 (sm_90) — `ncu` shows L1/TEX ~=81-83% SOL, DRAM <1%, occupancy
  ~=12.2%, memory-pipe bound; the fix raises in-flight cp.async bytes via 8 warps
  and yields ~3.6x kernel-only bwd speedup on sm_90 Hopper. No sm_80/A100 path, no
  sm_80 benchmark. Confirms the only open-source DSA training backward is TileLang-on-Hopper.'
---

[Example][DeepSeek-V3.2] Adaptive threads for sparse MLA backward by Butterfingrz · Pull Request #2592 · tile-ai/tilelang · GitHub
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
Conversation
Butterfingrz
commented
Jul 10, 2026
•
edited by coderabbitai
Bot
Loading
Uh oh!
There was an error while loading.
Please reload this page
.
Copy link
Copy Markdown
Contributor
Summary
Make the thread count of the sparse MLA backward kernel adaptive instead of hard-coded.
`bwd(..., threads=...)`
now defaults to
`None`
and derives the launch width from the head-block size:
`256`
threads (8 warps) when
`block_H >= 64`
, otherwise
`128`
. This kernel is memory-pipe bound, and raising in-flight
`cp.async`
bytes with 8 warps is the main performance lever. For the default
`deepseek_v32`
shape (
`H=64`
,
`block_H=64`
) the gate resolves to
`256`
, so behavior is unchanged.
Motivation
This operator is the sole hotspot of the GLM-5 DSA (DeepSeek Sparse Attention) training backward path (
`SparseMLA.autograd.Function`
), computing sparse MLA gradients for the latent geometry (
`DQKV=576`
,
`DV=512`
,
`topk=2048`
).
`ncu`
profiling on H200 (sm_90) shows it is severely
memory-pipe bound
: L1/TEX ≈ 81–83% SOL, DRAM < 1%, achieved occupancy ≈ 12.2% (structurally pinned by registers + shared memory). Occupancy cannot be raised and DRAM is not the bottleneck, so the only legal lever is
increasing parallelism to hide KV-gather latency
.
Changes
`bwd(..., threads=...)`
default changed from
`256`
to
`None`
.
After
`block_H`
is computed, select the launch width from the head-block size:
```
padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
block_H = min(64, padded_H)
# adaptive: this kernel is memory-pipe bound. When the head count is not split across
# blocks (block_H>=64), 256 threads (8 warps) issue many more cp.async copies at once
# than 128 (4 warps), greatly raising in-flight bytes -- the main perf lever here.
# Smaller head blocks lack the GEMM warp-tiling work to fill 256 threads, so fall back
# to 128 and still build.
if threads is None:
    threads = 256 if block_H >= 64 else 128
```
8 lines changed (+8/−1); no numerical logic is touched.
Mechanism
The bottleneck is waiting for KV-gather to land. Moving a block from 4 warps (128 threads) to 8 warps (256 threads) doubles the concurrent in-flight
`cp.async`
/ LDGSTS transfers, i.e. it greatly raises in-flight bytes. By Little's law, effective memory-pipe throughput ≈ in-flight bytes / gather latency: the gather latency is fixed, so maximizing in-flight bytes overlaps that fixed latency more fully, and the long latency is hidden behind more in-flight warps.
Constraint: an 8-warp GEMM warp-tiling needs
`block_H >= 64`
to fill the M dimension; smaller head blocks cannot fill it and fall back to 128 threads so tiny shapes still build.
Applicability (Tensor Parallelism)
TP1 (no head split) benefits most
: with
`num_attention_heads=64`
and no attention-TP split, the kernel sees the full
`H=64`
,
`block_H=64`
lands exactly in the 256-thread tier, and in-flight bytes are maximized.
TP ≥ 2
:
`H`
shrinks,
`block_H < 64`
forces the 128-thread fallback, so this lever no longer applies — but the gate keeps such shapes compiling and correct.
Benchmark
Kernel-only
`bwd`
, CUPTI backend,
`H=64, HKV=1, DQKV=576, DV=512, topk=2048, bf16`
(this shape resolves to 256 threads). Comparing the adaptive
`256`
against the
`128`
fallback, the relative speedup is a stable
~3.6×
on sm_90 Hopper, consistent with the reference path (~3.6×) in the H200
`ncu`
analysis:
S
SKV
adaptive (→256)
128 (fallback)
speedup
2048
4096
7.436 ms / 198.7 TFlops
27.204 ms / 54.3 TFlops
3.66×
4096
8192
15.757 ms / 187.5 TFlops
56.113 ms / 52.7 TFlops
3.56×
8192
16384
31.786 ms / 185.9 TFlops
114.525 ms / 51.6 TFlops
3.60×
16384
32768
66.402 ms / 178.0 TFlops
237.057 ms / 49.9 TFlops
3.57×
Absolute throughput was measured locally on sm_90 Hopper; the ~3.6× relative speedup matches the H200
`ncu`
analysis. H200 absolute numbers can be swapped in from a re-run.
Environment
tilelang 0.1.11, tvm-ffi (apache-tvm-ffi) 0.1.11
torch 2.11.0+cu129, CUDA 12.9, H200 (sm_90)
Compatibility
For the default
`deepseek_v32`
shape (
`H=64`
,
`block_H=64`
) the gate resolves to
`256`
,
identical to the previous hard-coded value — no behavior change and no regression
. The ~3.6× in the table is relative to the
`128`
fallback, and explains why
`256`
is chosen when
`block_H >= 64`
.
Only adds a
`128`
fallback for small-head shapes (
`block_H < 64`
) so tiny shapes still compile.
Tests
Correctness:
`test_sparse_mla_bwd`
(
`assert_tensors_similar`
,
`dq`
/
`dkv`
vs the autograd reference,
`eps=1e-4`
) passes.
Summary
Updated
`sparse_mla_bwd.bwd`
to select thread count adaptively when unspecified:
`256`
threads for
`block_H >= 64`
`128`
threads otherwise
Preserves the default
`deepseek_v32`
configuration and numerical behavior.
Improves KV-gather latency hiding for larger head blocks.
`test_sparse_mla_bwd`
passes; reported H200 kernel-only benchmarking shows approximately 3.6× speedup with 256 versus 128 threads.
Sorry, something went wrong.
Uh oh!
There was an error while loading.
Please reload this page
.
All reactions
`
[Example] Add adaptive thread selection to sparse MLA backward
`
…
`
6f75a92
`
```
Default `bwd(..., threads=...)` to None and derive the launch width from
the head-block size instead of hard-coding 256.

This kernel is memory-pipe bound, so its throughput is set by how many
cp.async copies stay in flight, not by raw DRAM bandwidth. When the head
count is not split across blocks (block_H>=64), 256 threads (8 warps)
issue far more concurrent cp.async transfers than 128 (4 warps), greatly
raising in-flight bytes -- the main perf lever for this shape. Smaller
head blocks lack enough GEMM warp-tiling work to fill 256 threads, so
they fall back to 128 and still build.

For the deepseek_v32 shape (H=64, block_H=64) this resolves to 256,
matching the previous behavior.

Signed-off-by: Butterfingrz <13524387014@163.com>
```
coderabbitai
Bot
commented
Jul 10, 2026
•
edited
Loading
Uh oh!
There was an error while loading.
Please reload this page
.
Copy link
Copy Markdown
Contributor
No actionable comments were generated in the recent review. 🎉
ℹ️ Recent review info
⚙️ Run configuration
Configuration used
: Path: .coderabbit.yaml
Review profile
: CHILL
Plan
: Pro
Run ID
:
`36c0f89c-3695-48ef-b258-786c08adafe6`
📥 Commits
Reviewing files that changed from the base of the PR and between
3b37333
and
6f75a92
.
📒 Files selected for processing (1)
`examples/deepseek_v32/sparse_mla_bwd.py`
📝 Walkthrough
Walkthrough
The
`bwd`
kernel now defaults
`threads`
to
`None`
and selects 256 threads for
`block_H >= 64`
, otherwise 128 threads.
Changes
Adaptive thread configuration
Layer / File(s)
Summary
Select threads from
`block_H`
`examples/deepseek_v32/sparse_mla_bwd.py`
The
`bwd`
kernel uses an optional
`threads`
parameter and selects 256 or 128 threads based on
`block_H`
when no value is provided.
Estimated code review effort:
2 (Simple) | ~5 minutes
🚥 Pre-merge checks | ✅ 4 | ❌ 1
❌ Failed checks (1 warning)
Check name
Status
Explanation
Resolution
Docstring Coverage
⚠️
Warning
Docstring coverage is 0.00% which is insufficient. The required threshold is 80.00%.
Write docstrings for the functions missing them to satisfy the coverage threshold.
✅ Passed checks (4 passed)
Check name
Status
Explanation
Description Check
✅ Passed
Check skipped - CodeRabbit’s high-level summary is enabled.
Title check
✅ Passed
The title clearly summarizes the main change: adaptive thread selection for the sparse MLA backward kernel.
Linked Issues check
✅ Passed
Check skipped because no linked issues were found for this pull request.
Out of Scope Changes check
✅ Passed
Check skipped because no linked issues were found for this pull request.
✨ Finishing Touches
🧪 Generate unit tests (beta)
Create PR with unit tests
Thanks for using
CodeRabbit
! It's free for OSS, and your support helps us grow. If you like it, consider giving us a shout-out.
❤️ Share
X
Mastodon
Reddit
LinkedIn
Comment
`@coderabbitai help`
to get the list of available commands.
All reactions
Sorry, something went wrong.
Uh oh!
There was an error while loading.
Please reload this page
.
Butterfingrz
commented
Jul 10, 2026
•
edited
Loading
Uh oh!
There was an error while loading.
Please reload this page
.
Copy link
Copy Markdown
Contributor
Author
Hi , could you please take a look at this PR when you have a chance, Thanks!
@LeiWang1999
@chengyupku
@Rachmanino
All reactions
Sorry, something went wrong.
Uh oh!
There was an error while loading.
Please reload this page
.
LeiWang1999
approved these changes
Jul 22, 2026
View reviewed changes
Hide details
View details
LeiWang1999
merged commit
`e4e110e`
into
tile-ai
:
main
Jul 22, 2026
1 check passed
Uh oh!
There was an error while loading.
Please reload this page
.
Butterfingrz
deleted the
example/sparse-mla-bwd-adaptive-threads
branch
July 22, 2026 18:14
Butterfingrz
mentioned this pull request
Jul 25, 2026
perf(glm): adapt sparse MLA backward threads by block size
NVIDIA-NeMo/Automodel#3220
Open
3 tasks done
pengdurice
mentioned this pull request
Oct 16, 2026
Add deepspeed.models.glm_moe_dsa: TileLang sparse attention for GLM-5.2
deepspeedai/DeepSpeed#8659
Open
Sign up for free
to join this conversation on GitHub
.
    Already have an account?
Sign in to comment
Reviewers
LeiWang1999
LeiWang1999 approved these changes
Uh oh!
There was an error while loading.
Please reload this page
.
Assignees
No one assigned
Labels
None yet
Projects
None yet
Milestone
No milestone
Development
Successfully merging this pull request may close these issues.
Uh oh!
There was an error while loading.
Please reload this page
.
2 participants
Add this suggestion to a batch that can be applied as a single commit.
This suggestion is invalid because no changes were made to the code.
Suggestions cannot be applied while the pull request is closed.
Suggestions cannot be applied while viewing a subset of changes.
Only one suggestion per line can be applied in a batch.
Add this suggestion to a batch that can be applied as a single commit.
Applying suggestions on deleted lines is not supported.
You must change the existing code in this line in order to create a valid suggestion.
Outdated suggestions cannot be applied.
This suggestion has been applied or marked resolved.
Suggestions cannot be applied from pending reviews.
Suggestions cannot be applied on multi-line comments.
Suggestions cannot be applied while the pull request is queued to merge.
Suggestion cannot be applied right now. Please check back later.
You can’t perform that action at this time.