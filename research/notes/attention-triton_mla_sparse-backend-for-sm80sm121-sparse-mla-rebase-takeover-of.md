---
title: '[Attention] TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA (rebase &
  takeover of #38476) by thomaslwang · Pull Request #47629 · vllm-project/vllm'
id: attention-triton_mla_sparse-backend-for-sm80sm121-sparse-mla-rebase-takeover-of
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-csa2-at-32k-sm80
created: '2026-10-27T13:44:45.957461Z'
updated: '2026-10-27T19:00:31.712985Z'
source: https://github.com/vllm-project/vllm/pull/47629
source_domain: github.com
fetched_at: '2026-10-27T13:44:45.332252Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: Suggested by deepseek-sparse-attention-dsa-nvidia-cudnn — cuDNN DSA docs
  establish SM90/SM100-only kernels; this PR i...
---

*Suggested by [[deepseek-sparse-attention-dsa-nvidia-cudnn]] — cuDNN DSA docs establish SM90/SM100-only kernels; this PR is the community SM80 sparse-MLA path that would falsify the 'no sm_80 DSA path' claim*

[Skip to content](https://github.com/vllm-project/vllm/pull/47629#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/vllm-project/vllm/pull/47629) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/vllm-project/vllm/pull/47629) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/vllm-project/vllm/pull/47629) to refresh your session. Dismiss alert
## Conversation
### 
commented [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#issue-4811759553) •
edited 
Loading
###  Uh oh! 
There was an error while loading. [Please reload this page](https://github.com/vllm-project/vllm/pull/47629).
Copy link  Copy Markdown 
# [Attention] TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA (rebase & takeover of [#38476](https://github.com/vllm-project/vllm/pull/38476))
## Purpose
Rebase and continuation of [#38476](https://github.com/vllm-project/vllm/pull/38476) by [@haosdent](https://github.com/haosdent), which adds a Triton-based sparse MLA attention backend (`TRITON_MLA_SPARSE`) so DSA sparse-MLA models (DeepSeek-V3.2, GLM-5.x) can run on GPUs where DeepGEMM and FlashMLA-Sparse are unavailable — SM80 (A100/A800) and SM121 (GB10/DGX Spark).
Per the discussion in [#38476](https://github.com/vllm-project/vllm/pull/38476) (see comments from 2026-06-29 onward): [#43477](https://github.com/vllm-project/vllm/pull/43477) has merged, maintainers indicated openness to merging this backend after a rebase, and the original author has been inactive since mid-May. This PR takes over that work as invited in the thread, crediting the original author (the main commit retains [@haosdent](https://github.com/haosdent)'s authorship).
## Changes vs the original [#38476](https://github.com/vllm-project/vllm/pull/38476)
  * **Rebased onto current main.** The indexer dispatch in `sparse_attn_indexer.py` is now a three-way `is_xpu() → DeepGEMM → Triton fallback` chain, preserving main's XPU path, `skip_topk_buffer_clear` optimization, and DCP handling.
  * **Backend priority (maintainer request):** `TRITON_MLA_SPARSE` is appended _after_ `FLASH_ATTN_MLA_SPARSE` and `FLASHMLA_SPARSE` in the SM8x/9x candidate list, so SM90+ keeps native sparse backends and only SM80/SM121 fall through to Triton.
  * **New fix:** gate `use_fused_indexer_q` (from [[GLM5.2 Perf] `fused_indexer_q_rope_quant` triton kernel, 1.9% ~ 3.3% E2E Throughput improvement. #46862](https://github.com/vllm-project/vllm/pull/46862)) on SM89+ — its Triton kernel stores fp8e4nv, which does not compile on SM80; older archs use the pre-existing unfused rope + `per_token_group_quant_fp8` path.


## Relationship to other PRs
  * [[bugfix]: support deepseek sparse attention on unsupported targets #39594](https://github.com/vllm-project/vllm/pull/39594) forces _dense_ attention on unsupported targets via an env var (a correctness workaround that gives up sparse attention); this PR keeps sparse attention with Triton kernels. The two are complementary.
  * [[WIP]Support DeepSeek V4 flash on SM120 with Triton fallback #40929](https://github.com/vllm-project/vllm/pull/40929) / [[New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash with essential fixes #41834](https://github.com/vllm-project/vllm/pull/41834) target DeepSeek-V4 on SM120 specifically; this PR is about the DSA sparse-MLA indexer/attention path for V3.2/GLM-5.x on SM80/SM121.


## Test Plan / Test Result
All on 8× NVIDIA A800-80GB PCIe (SM80), driver 580.95.05, this branch installed with `VLLM_USE_PRECOMPILED=1`:
  * `pytest tests/kernels/attention/test_mqa_logits_triton.py` — **41 passed** (24.8s)
  * `pytest tests/kernels/attention/test_triton_mla_sparse_kernel.py` — **53 passed** (100.4s)
  * End-to-end serve of GLM-5.2 NVFP4 (`nvidia/GLM-5.2-NVFP4`), TP=8, EP, bf16 KV cache, `--max-model-len 131072`: 
    * Backend selection: `Using TRITON_MLA_SPARSE attention backend out of potential backends: ['TRITON_MLA_SPARSE']` + DeepGEMM-fallback warning as expected
    * CUDA graph capture clean under default `FULL_AND_PIECEWISE` (mixed PIECEWISE 4/4 + decode FULL 3/3, 10s) — no Triton compile-during-capture issues; the backend's built-in autotune warmup covers it
    * GPU KV cache size: 202,688 tokens; default block size works (no `--block-size` override needed)
    * Correctness spot checks: math prompt correct; tool calling via `glm47` parser produces well-formed `tool_calls` with `finish_reason=tool_calls`
    * Throughput: 32.3 tok/s single-stream decode, 100.3 tok/s aggregate at 4-way concurrency (256-token completions)
  * Regression guard: SM90+ candidate order keeps `FLASH_ATTN_MLA_SPARSE` and `FLASHMLA_SPARSE` ahead of `TRITON_MLA_SPARSE` (Triton is last-resort only).
  * Pipeline parallelism (single node, TP=2 × PP=4, `--block-size 128`): 
    * Tool/reasoning parsers work under PP (`glm47`/`glm45`): correct `tool_calls` after long contexts; needle retrieval accurate at 79k prompt tokens; two concurrent 53k-token tool-call requests both correct
    * bf16 KV capacity: 524,288 boots at TP=2×PP=4; **full 1M (`max-model-len 1048576`) boots at TP=1×PP=8** with `VLLM_PP_LAYER_PARTITION="11,10,10,10,10,10,10,7"` (the default even split leaves the last stage — LM head + sampler activations — short) and `--gpu-memory-utilization 0.96`: KV cache 1,062,656 tokens, needle retrieval exact at 250k prompt tokens (with [[Bugfix][Core] Sync reused pinned input buffers under PP batch queue (fixes IMA with sparse MLA + PP) #47644](https://github.com/vllm-project/vllm/pull/47644) applied; without it, a single long chunked prefill hits the pinned-buffer race)
    * The `No common block size for 16.` PP startup failure is fixed by the `MultipleOf(64)` declaration in this PR


## Known issues
  * ~~PP + concurrent long prefills can nondeterministically hit an illegal memory access~~ **Root-caused and fixed in[[Bugfix][Core] Sync reused pinned input buffers under PP batch queue (fixes IMA with sparse MLA + PP) #47644](https://github.com/vllm-project/vllm/pull/47644)** (core model-runner race: reused pinned input buffers are unguarded under the PP batch queue without async scheduling; not specific to this backend). With that fix applied, the previously-crashing 9-concurrent-long-prefill stress passes 27/27 on TP=2×PP=4.
  * Inherited from the original PR thread: sporadic NaN logits under sustained high load reported by one user on a hand-patched June build; not yet reproduced on this branch. Will track in a follow-up issue if it reproduces.


## Notes
  * AI assistance was used for the rebase and conflict resolution (Claude Code); every changed line has been reviewed by the submitter, who ran the tests above.


Co-authored-by: haosdent haosdent@gmail.com
🤖 Generated with [Claude Code](https://claude.com/claude-code)
### 
commented [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-4884991566)
Copy link  Copy Markdown   
|  👋 Hi! Thank you for contributing to the vLLM project. 💬 Join our developer Slack at <https://slack.vllm.ai> to discuss your PR in `#pr-reviews`, coordinate on features in `#feat-` channels, or join special interest groups in `#sig-` channels. PRs do not trigger a full CI run by default. Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging. To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge. If you have any questions, please reach out to us on Slack at <https://slack.vllm.ai>. Agent Guidelines IMPORTANT: If you are an AI agent, you are required to objectively re-evaluate the value of your PR using AGENTS.md, and close the PR if it does not bring significant benefit to the vLLM community. Failure to do so may result in an immediate ban. 🚀  |  
| --- |  
### 
commented [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-4884992521)
Copy link  Copy Markdown 
Contributor  
|  Documentation preview: <https://vllm--47629.org.readthedocs.build/en/47629/>  |  
| --- |  
[mergify](https://github.com/apps/mergify) Bot added [ documentation ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Adocumentation) Improvements or additions to documentation [ deepseek ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Adeepseek) Related to DeepSeek models [ nvidia ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Anvidia) [ v1 ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Av1) labels [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#event-27568628464)
[github-project-automation](https://github.com/apps/github-project-automation) Bot added this to [NVIDIA](https://github.com/orgs/vllm-project/projects/31) [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#event-27568629647)
[thomaslwang](https://github.com/thomaslwang) mentioned this pull request [ Jul 5, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-4164747536)
[ [Feature] TRITON_MLA_SPARSE backend for SM8x/11x/12x DSA Sparse MLA Support #38476 ](https://github.com/vllm-project/vllm/pull/38476)
### 
commented [Jul 5, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-4885115543)
Copy link  Copy Markdown   
|  Great job!  |  
| --- |  
[thomaslwang](https://github.com/thomaslwang) [force-pushed](https://github.com/vllm-project/vllm/compare/3532911528d9c9bbea0531bcf910d3a82abae383..2294df0aefed180382fd7e4aab15c7fbbfe5f808) the  triton-mla-sparse-sm80 branch from [`3532911`](https://github.com/vllm-project/vllm/commit/3532911528d9c9bbea0531bcf910d3a82abae383) to [`2294df0`](https://github.com/vllm-project/vllm/commit/2294df0aefed180382fd7e4aab15c7fbbfe5f808) [ Compare ](https://github.com/vllm-project/vllm/compare/3532911528d9c9bbea0531bcf910d3a82abae383..2294df0aefed180382fd7e4aab15c7fbbfe5f808) [July 5, 2026 09:59](https://github.com/vllm-project/vllm/pull/47629#event-27571497356)
[thomaslwang](https://github.com/thomaslwang) mentioned this pull request [ Jul 5, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-4812560353)
[ [Bugfix][Core] Sync reused pinned input buffers under PP batch queue (fixes IMA with sparse MLA + PP) #47644 ](https://github.com/vllm-project/vllm/pull/47644)
[thomaslwang](https://github.com/thomaslwang) marked this pull request as ready for review [July 5, 2026 11:12](https://github.com/vllm-project/vllm/pull/47629#event-27572333046)
[thomaslwang](https://github.com/thomaslwang) requested review from [AndreasKaratzas](https://github.com/AndreasKaratzas), [LucasWilkinson](https://github.com/LucasWilkinson), [MatthewBonanni](https://github.com/MatthewBonanni), [WoosukKwon](https://github.com/WoosukKwon), [mgoin](https://github.com/mgoin), [pavanimajety](https://github.com/pavanimajety), [tlrmchlsmth](https://github.com/tlrmchlsmth), [yewentao256](https://github.com/yewentao256) and [zyongye](https://github.com/zyongye) as [code owners](https://github.com/vllm-project/vllm/blob/fa4321de3d894c50c5ca0766dffa352d3fb07423/.github/CODEOWNERS#L99) [July 5, 2026 11:12](https://github.com/vllm-project/vllm/pull/47629#event-27572333201)
reviewed  [ Jul 5, 2026 ](https://github.com/vllm-project/vllm/pull/47629#pullrequestreview-4631188148)
[ View reviewed changes ](https://github.com/vllm-project/vllm/pull/47629/files)
### 
left a comment 
Copy link  Copy Markdown 
There was a problem hiding this comment.
### Choose a reason for hiding this comment
The reason will be displayed to describe this comment to others. [Learn more](https://docs.github.com/articles/managing-disruptive-comments/#hiding-a-comment). 
## Claude Code Review
This pull request is from a fork — automated review is disabled. A repository maintainer can comment `@claude review` to run a one-time review.
### 
commented [Jul 7, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-4899276754)
Copy link  Copy Markdown   
|  Thanks for this! I spent the last couple days trying to get it all running on my own heterogenous hardware. I worked off this PR to patch bits and pieces until it landed at a decent spot. I used claude Fable to assist heavily. I'm a long time dev, but never contributed to open source or AI related work, so I hope this is helpful... **Field report: TRITON_MLA_SPARSE running GLM-5.2 744B (AWQ INT4) on a mixed SM86+SM120 fleet, TP2×PP6 — including working MTP speculative decoding under pipeline parallelism** Thanks for this backend — it resurrected DSA models on hardware the DeepGEMM path can't serve. We have GLM-5.2 (AWQ INT4, compressed-tensors) in production on 12 mixed GPUs (2× RTX PRO 6000 Blackwell, 2× RTX 5090, 8× RTX 3090), TP2×PP6, with `--attention-backend TRITON_MLA_SPARSE`: 904 tok/s prefill, 14.8 tok/s decode with MTP spec decode (~80% acceptance, k=1), 40–62k context, output-validated to 25k+ context. Sharing the bugs we hit — the first two are one-line kernel fixes in this PR's Triton path that affect ANY spec-decode user, the rest are generic vLLM issues this backend newly exposes.
### Two Triton-path kernel bugs (spec decode only, one-line fixes)
Both are invisible at `next_n == 1` and below `index_topk` (2048) context — which is why plain decode looks perfect and why they survive short tests. With MTP (`next_n == 2`) output corrupts progressively beyond ~2048 total context (word doubling → repetition loops → garbage). Empirically isolated and fixed; both validated together (3400-token generations flawless, 25k prompts clean): **A. 2D`seq_lens` fed to a kernel that indexes it 1D.** The decode caller passes the raw `(B, next_n)` seq_lens to `fp8_paged_mqa_logits_triton`, but `_fp8_paged_mqa_logits_kernel` loads `context_lens_ptr + batch_id` (1D, full length expected — it derives per-row causal offsets itself via `q_offset = L - next_n + next_n_id`). Every request reads a wrong length (and at B>1, _other requests'_ lengths). The sibling XPU branch already flattens with `seq_lens[:, -1]`; the Triton branch misses it. Fix: same flatten at the Triton call site in `sparse_attn_indexer.py`. **B. Unmasked OOB store spills`-inf` into the next logits row.** The kernel's last block stores `k_offset` up to `128*ceil(L/128)-1` masked only by `mask_n`, while the logits row stride is the ACTIVE batch max_seq_len (not block-aligned). Up to 127 columns of `-inf` land in the next row's oldest positions — with spec decode, the base row erases the bonus row's oldest context (including the prompt), and past `index_topk` the value-based top-k then drops those positions. (Below 2048 the trivial top-k branch selects by position and hides it.) Also writes past the allocation on the last row. Fix: `mask=mask_n & (k_offset < context_len)` on the store. Happy to PR both (they're one line each plus comments).
### Portability fixes needed for mixed-capability fleets
  1. **Device-0-scoped capability gates.** `use_fused_indexer_q` gates on `current_platform.has_device_capability(89)`, which checks device 0. On a heterogeneous fleet, SM86 workers then compile the fp8e4nv Triton indexer kernel and die. Fixed locally by gating on `torch.cuda.get_device_capability()` (the worker's own device). Backend auto-selection has the same device-0 scoping (we force the backend explicitly as a workaround).
  2. **safetensors lazy load + dispatch modes.** In worker context, `safe_open.get_tensor` intermittently fails with "could not determine the shape of object type 'torch.storage.UntypedStorage'" (a leaked torch function/dispatch mode routes it into `torch._refs`). Guarded retry under `DisableTorchFunction` + `_disable_current_modes` fixes it. (Will file separately — unrelated to this PR.)


### MTP speculative decoding: draft-config inheritance bug family
We got MTP working with a separate draft checkpoint (`method:"mtp"`, `num_speculative_tokens:1`). Every failure below traces to the same root pattern: **the draft model's config is derived from the target's and is only correct for vanilla single-node TP.** Together with [local-inference-lab#72](https://github.com/local-inference-lab/vllm/pull/72) (draft config doesn't inherit `decode_context_parallel_size` under DCP), that's four confirmed members:
  1. **`SpeculativeConfig`doesn't inherit`attention_backend`.** The draft auto-selects independently (device-0-scoped again) and can pick a backend with an incompatible KV spec: we got `fp8_ds_mla` (656B/token) allocation vs a bf16 (1152B/token) reshape — `shape '[730, 64, 576]' invalid for input of size 15324160`. Workaround: `"attention_backend": "TRITON_MLA_SPARSE"` inside `--speculative-config`.
  2. **Draft inherits the target's`quant_config`.** `_create_draft_vllm_config` keeps `base.quant_config`. When the draft checkpoint's compressed-tensors ignore-list differs from the target's (ours: draft quantizes attention, target doesn't; draft ignores layer-scoped indexer/gate projections the target's list doesn't name), layer-78 modules are built with the wrong scheme. Fix: derive `quant_config` from the draft's own model config when the draft is a separate checkpoint.
  3. **MTP embedding sharing is gated`pp_group().world_size == 1`** (`_maybe_share_embeddings`), while `_maybe_share_lm_head` has no PP guard. Under PP the drafter's `embed_tokens` is silently left at random init (DeepSeek-style MTP checkpoints intentionally omit it, and `DeepSeekMTP.load_weights` drops any tensor without a spec-layer index, so nothing loads it). Result: the drafter reproduces the target's own prediction — exactly-0% acceptance that looks like an echo bug. Workaround: re-export the draft checkpoint with the target's embedding spliced in **as`model.layers.<n>.embed_tokens.weight`** (top-level naming is silently dropped by the loader). Proper fix: share/load the embedding on the drafter's rank under PP.


### Sync-PP scheduler races (spec decode + batch queue)
With PP>1 and `--no-async-scheduling`, three interlocking races made spec decode produce 0% acceptance and corrupted output streams. All three are in core vLLM, not this PR, but they're only reachable once a backend like this one makes PP spec decode viable on such fleets:
  1. **Batch queue schedules a spec request's next step before`update_from_output` processes its verification** — stale `num_computed_tokens`, empty payloads, worker-side `IndexError: list index out of range` in `_update_states`, silent KV desync. Notably the prefill→first-verify boundary: the first verify can be scheduled before the prefill's sampled token is appended, which puts the draft token at the base KV position (one-position verify shift; a token vanishes from the emitted stream).
  2. **Drafts produced by verify steps are dropped.** `post_step` gates `take_draft_token_ids` on the _schedule phase's_ `model_executed`; when the spec request is (correctly) not re-scheduled until its verification is processed, that phase is empty and the freshly proposed drafts are never collected — the scheduler then schedules stale/zero draft ids (the rejection sampler reads draft ids positionally from `input_ids`, so this also manifests as `draft_token_ids == 0` worker-side).
  3. **Accepted draft tokens are never delivered to non-last PP ranks.** They reach the last rank as spec inputs, but the next step's payload (`_make_cached_request_data`) starts at `num_computed_tokens`, so ranks 0..N-2 never see them — their token streams shift one position per acceptance and verify verdicts drift progressively (accumulating wrongful rejections, output corruption).

Our fixes for 6–8 (scheduler-side serialization of spec-request steps under sync-PP, forced draft collection after processing spec-carrying batches, and payload over-delivery of accepted tokens — the worker's `num_computed + len(new) - req_state.num_tokens` reconciliation dedups the overlap) are small and validated: 83% acceptance, byte-correct output over long multi-turn/streaming/concurrent sessions. Happy to open PRs / share diffs if there's interest — or if the newer runner path already restructures this, pointers appreciated. Environment: vLLM @ [`bbe2ab4`](https://github.com/vllm-project/vllm/commit/bbe2ab4d6cc01a634640d0b502ac67055d114219) (this PR's head, pinned) + PR [#47644](https://github.com/vllm-project/vllm/pull/47644) cherry-picked, `VLLM_USE_PRECOMPILED=1`, CUDA graphs off, `--disable-custom-all-reduce`, blocks=128.  |  
| --- |  
This was referenced Jul 11, 2026
[ [Bug]: xpu_mla_sparse NaN-poisons attention output when a row's leading topk index chunk is fully masked #48364 ](https://github.com/vllm-project/vllm/issues/48364)
Closed 
[ [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks #48366 ](https://github.com/vllm-project/vllm/pull/48366)
Merged 
[ [RFC]: fp8 KV cache for the Ampere sparse-MLA path (TRITON_MLA_SPARSE) via software dequant #48374 ](https://github.com/vllm-project/vllm/issues/48374)
### 
commented [Jul 19, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-5013986232)
Copy link  Copy Markdown 
Contributor  
|  This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, [@thomaslwang](https://github.com/thomaslwang). <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork>  |  
| --- |  
[mergify](https://github.com/apps/mergify) Bot added the [ needs-rebase ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Aneeds-rebase) label [Jul 19, 2026](https://github.com/vllm-project/vllm/pull/47629#event-28171708436)
[haosdent](https://github.com/haosdent) and others added 2 commits [July 19, 2026 03:52](https://github.com/vllm-project/vllm/pull/47629#commits-pushed-fb50482)
` Add TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA support[](https://github.com/vllm-project/vllm/pull/47629/commits/fb5048234b81f6283e628555bd893459a1d5a8e8 "Add TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA support

Closes #38006. See PR description for full details.

Signed-off-by: haosdent <haosdent@gmail.com>
Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>") `

```
Closes vllm-project#38006[](https://github.com/vllm-project/vllm/issues/38006). See PR description for full details.

Signed-off-by: haosdent <haosdent@gmail.com>
Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>
```

` Gate fused_indexer_q_rope_quant on SM89+ (Triton fp8e4nv store)[](https://github.com/vllm-project/vllm/pull/47629/commits/8c8877b19ef4f4c4062ffa85bb025c35fa746f7b "Gate fused_indexer_q_rope_quant on SM89+ \(Triton fp8e4nv store\)

The fused indexer-Q rope+quant Triton kernel stores fp8e4nv, which
Triton only supports on SM89+. On SM80 \(A100/A800\) the kernel fails to
compile at startup. Fall back to the unfused rope +
per_token_group_quant_fp8 path on older archs.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>") `

```
The fused indexer-Q rope+quant Triton kernel stores fp8e4nv, which
Triton only supports on SM89+. On SM80 (A100/A800) the kernel fails to
compile at startup. Fall back to the unfused rope +
per_token_group_quant_fp8 path on older archs.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>
```

### 
commented [Aug 28, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-5447499354) •
edited 
Loading
###  Uh oh! 
There was an error while loading. [Please reload this page](https://github.com/vllm-project/vllm/pull/47629).
Copy link  Copy Markdown   
|  Hi [@thomaslwang](https://github.com/thomaslwang) — thanks for pushing the TRITON_MLA_SPARSE work forward. Writing because we just ran into exactly the gap this PR covers: GLM-5.3-Flash (`glm5_next`, via [#53906](https://github.com/vllm-project/vllm/pull/53906)) fails to start on our 8x RTX 4090 (Ada, sm_89) box since no sparse-MLA backend accepts its geometry — details in [#54059](https://github.com/vllm-project/vllm/issues/54059). We have a real need for this model on this hardware, so SM8x coverage is directly relevant to us. One note on top of the current scope: GLM-5.3 is rope-free NoPE with `dim_qk=512` (`qk_rope_head_dim=0`), so it would additionally need the 512-wide NoPE handling that was sketched in the now-closed [#54031](https://github.com/vllm-project/vllm/pull/54031) (`get_supported_head_sizes -> [512, 576]`, plus the Triton MQA-logits fallback for archs without DeepGEMM). We'll build and try the branch once the NoPE geometry is covered and report back in [#54059](https://github.com/vllm-project/vllm/issues/54059) with results.  |  
| --- |  
[mikekg](https://github.com/mikekg) mentioned this pull request [ Sep 3, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-5339022591)
[ Fix/dsv4 sparse mla portability #55177 ](https://github.com/vllm-project/vllm/pull/55177)
Draft 
4 tasks done
[mikekg](https://github.com/mikekg) mentioned this pull request [ Sep 3, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-5339347430)
[ Fix/dsv4 pre-sm89/sm90 (software fp8) sparse mla omnibus #55184 ](https://github.com/vllm-project/vllm/pull/55184)
4 tasks done
[mikekg](https://github.com/mikekg) added a commit to mikekg/vllm that referenced this pull request [ Sep 13, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-dab25fd)
` [Bugfix][DeepSeek V4] Support sparse MLA on SM8x[](https://github.com/mikekg/vllm/commit/dab25fd0d77691ff0a93f5e9556aa85185322d68 "\[Bugfix\]\[DeepSeek V4\] Support sparse MLA on SM8x

DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from #47629 and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `\[batch\]` contract and DeepSeek V4's `\[batch, next_n\]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses #47629's generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on #47629 and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing #47629 paged-MQA test was extended with nonuniform two-dimensional context lengths `\[128, 129, 129, 130\]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \\
      -v

    TRITON_OVERRIDE_ARCH=sm80 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    TRITON_OVERRIDE_ARCH=sm86 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: \[2560, 2561, 2562, 2563\]
    expected: \[-1, -1, -1, -1\]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$\(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1\)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":\[1,2,4,8,16,32,64\],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=\(\)
      if \(\( SLURM_NODEID > 0 \)\); then
        headless=\(--headless\)
      fi

      exec vllm serve "$MODEL" \\
        --revision "$REVISION" \\
        --served-model-name deepseek-v4-pro \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --trust-remote-code \\
        --tokenizer-mode deepseek_v4 \\
        --load-format fastsafetensors \\
        --safetensors-load-strategy lazy \\
        --distributed-executor-backend mp \\
        --master-addr "$MASTER_ADDR" \\
        --master-port "$MASTER_PORT" \\
        --nnodes "$SLURM_NNODES" \\
        --node-rank "$SLURM_NODEID" \\
        --tensor-parallel-size 16 \\
        --pipeline-parallel-size 2 \\
        --enable-expert-parallel \\
        --kv-cache-dtype fp8 \\
        --block-size 256 \\
        --max-model-len 10240 \\
        --max-num-seqs 64 \\
        --max-num-batched-tokens 16384 \\
        --gpu-memory-utilization 0.9 \\
        --linear-backend auto \\
        --moe-backend auto \\
        --no-enable-prefix-caching \\
        --distributed-timeout-seconds 7200 \\
        --cpu-distributed-timeout-seconds 7200 \\
        --compilation-config "$COMPILATION_CONFIG" \\
        "${headless\[@\]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 16 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 16 \\
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 1319 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 64 \\
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with #47629 and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>") `

```
DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `[batch]` contract and DeepSeek V4's `[batch, next_n]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629)'s generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) paged-MQA test was extended with nonuniform two-dimensional context lengths `[128, 129, 129, 130]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \
      -v

    TRITON_OVERRIDE_ARCH=sm80 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    TRITON_OVERRIDE_ARCH=sm86 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: [2560, 2561, 2562, 2563]
    expected: [-1, -1, -1, -1]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=()
      if (( SLURM_NODEID > 0 )); then
        headless=(--headless)
      fi

      exec vllm serve "$MODEL" \
        --revision "$REVISION" \
        --served-model-name deepseek-v4-pro \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --load-format fastsafetensors \
        --safetensors-load-strategy lazy \
        --distributed-executor-backend mp \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --nnodes "$SLURM_NNODES" \
        --node-rank "$SLURM_NODEID" \
        --tensor-parallel-size 16 \
        --pipeline-parallel-size 2 \
        --enable-expert-parallel \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --max-model-len 10240 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9 \
        --linear-backend auto \
        --moe-backend auto \
        --no-enable-prefix-caching \
        --distributed-timeout-seconds 7200 \
        --cpu-distributed-timeout-seconds 7200 \
        --compilation-config "$COMPILATION_CONFIG" \
        "${headless[@]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 16 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 16 \
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 1319 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 64 \
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>
```

[hnt2601](https://github.com/hnt2601) mentioned this pull request [ Sep 15, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-5457103493)
[ [Bugfix][Spec Decode] Don't swap the Gemma4 MTP draft into the target's VllmConfig #56936 ](https://github.com/vllm-project/vllm/pull/56936)
[thomaslwang](https://github.com/thomaslwang) mentioned this pull request [ Sep 16, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-issue-5472579135)
[ [Feature]: SM8x (Ampere A100/A800) support for DeepSeek-V4.1-Flash #57144 ](https://github.com/vllm-project/vllm/issues/57144)
1 task done
[positive666](https://github.com/positive666) mentioned this pull request [ Sep 19, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-5508616319)
[ [Perf][DSv4.1] Add SM90 native-head sparse attention backend #57673 ](https://github.com/vllm-project/vllm/pull/57673)
Draft 
6 tasks done
[ming616](https://github.com/ming616) mentioned this pull request [ Sep 28, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-issue-5613691506)
[ [RFC]: Software-dequant fp8 KV cache for MLA on Ampere (sm80/sm86) — consolidate the existing pieces, a 1M-ctx field case, and a validation offer #59016 ](https://github.com/vllm-project/vllm/issues/59016)
### 
commented [Sep 29, 2026](https://github.com/vllm-project/vllm/pull/47629#issuecomment-5891802832)
Copy link  Copy Markdown   
|  Context from a production deployment that this PR directly unblocks (posted originally on [#48374](https://github.com/vllm-project/vllm/issues/48374), consolidated in [#59016](https://github.com/vllm-project/vllm/issues/59016)): We run **GLM-5.3-Flash** (Glm5NextForConditionalGeneration — hybrid KDA + sparse-MLA indexer) on **4×A800 (sm80)** in production at **1M context** on the bf16 path today (930K-context needle recall 5/5, full-length concurrency 2.15×, KV ~11.6 KiB/token). `TRITON_MLA_SPARSE` for sm80 is the only viable MLA backend for this hardware class in our testing, which makes this PR the prerequisite for the whole Ampere fp8-KV line ([#48374](https://github.com/vllm-project/vllm/issues/48374) → [#55184](https://github.com/vllm-project/vllm/pull/55184)/[#55173](https://github.com/vllm-project/vllm/pull/55173)/[#55177](https://github.com/vllm-project/vllm/pull/55177), consolidated proposal in [#59016](https://github.com/vllm-project/vllm/issues/59016)). Two concrete motivations for prioritizing the rebase/merge:
  1. **Capacity math is the binding constraint for us.** The 1.76× realized figure from the sm86 validation would take our KV pool from 2.25M tokens to ~4M — turning full-length (1M) concurrency from 2-way into 3-4 way. This is currently physically impossible for us by any other lever (fp8-KV is the only one that scales the pool; util/workspace/headroom are exhausted).
  2. **We can validate the moment this lands.** We have a standing 4×A800 rig running this exact model family at 1M: we can run capacity benchmarks, bf16-vs-fp8 needle-recall parity, and decode-throughput regression on production-shaped traffic (long agent contexts, 100K-962K prefills) and post numbers here. If rebase effort is the blocker for the author, that validation obligation is one less thing to worry about — and we're glad to help test intermediate states too.

No rush implied — just adding a standing data point that there is a real sm80 + 1M-context deployment waiting on this path, so the maintainers can weigh it against the conflict-resolution cost.  |  
| --- |  
[andylolu2](https://github.com/andylolu2) mentioned this pull request [ Oct 5, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-pullrequest-5712691580)
[ [Attention][MLA] Add DCP support to Triton sparse MLA #60057 ](https://github.com/vllm-project/vllm/pull/60057)
Closed 
[mikekg](https://github.com/mikekg) added a commit to mikekg/vllm that referenced this pull request [ Oct 8, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-149d8d2)
` [Bugfix][DeepSeek V4] Support sparse MLA on SM8x[](https://github.com/mikekg/vllm/commit/149d8d2a44f00188976b0e5e8181808ed8999325 "\[Bugfix\]\[DeepSeek V4\] Support sparse MLA on SM8x

DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from #47629 and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `\[batch\]` contract and DeepSeek V4's `\[batch, next_n\]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses #47629's generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on #47629 and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing #47629 paged-MQA test was extended with nonuniform two-dimensional context lengths `\[128, 129, 129, 130\]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \\
      -v

    TRITON_OVERRIDE_ARCH=sm80 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    TRITON_OVERRIDE_ARCH=sm86 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: \[2560, 2561, 2562, 2563\]
    expected: \[-1, -1, -1, -1\]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$\(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1\)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":\[1,2,4,8,16,32,64\],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=\(\)
      if \(\( SLURM_NODEID > 0 \)\); then
        headless=\(--headless\)
      fi

      exec vllm serve "$MODEL" \\
        --revision "$REVISION" \\
        --served-model-name deepseek-v4-pro \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --trust-remote-code \\
        --tokenizer-mode deepseek_v4 \\
        --load-format fastsafetensors \\
        --safetensors-load-strategy lazy \\
        --distributed-executor-backend mp \\
        --master-addr "$MASTER_ADDR" \\
        --master-port "$MASTER_PORT" \\
        --nnodes "$SLURM_NNODES" \\
        --node-rank "$SLURM_NODEID" \\
        --tensor-parallel-size 16 \\
        --pipeline-parallel-size 2 \\
        --enable-expert-parallel \\
        --kv-cache-dtype fp8 \\
        --block-size 256 \\
        --max-model-len 10240 \\
        --max-num-seqs 64 \\
        --max-num-batched-tokens 16384 \\
        --gpu-memory-utilization 0.9 \\
        --linear-backend auto \\
        --moe-backend auto \\
        --no-enable-prefix-caching \\
        --distributed-timeout-seconds 7200 \\
        --cpu-distributed-timeout-seconds 7200 \\
        --compilation-config "$COMPILATION_CONFIG" \\
        "${headless\[@\]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 16 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 16 \\
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 1319 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 64 \\
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with #47629 and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>") `

```
DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `[batch]` contract and DeepSeek V4's `[batch, next_n]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629)'s generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) paged-MQA test was extended with nonuniform two-dimensional context lengths `[128, 129, 129, 130]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \
      -v

    TRITON_OVERRIDE_ARCH=sm80 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    TRITON_OVERRIDE_ARCH=sm86 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: [2560, 2561, 2562, 2563]
    expected: [-1, -1, -1, -1]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=()
      if (( SLURM_NODEID > 0 )); then
        headless=(--headless)
      fi

      exec vllm serve "$MODEL" \
        --revision "$REVISION" \
        --served-model-name deepseek-v4-pro \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --load-format fastsafetensors \
        --safetensors-load-strategy lazy \
        --distributed-executor-backend mp \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --nnodes "$SLURM_NNODES" \
        --node-rank "$SLURM_NODEID" \
        --tensor-parallel-size 16 \
        --pipeline-parallel-size 2 \
        --enable-expert-parallel \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --max-model-len 10240 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9 \
        --linear-backend auto \
        --moe-backend auto \
        --no-enable-prefix-caching \
        --distributed-timeout-seconds 7200 \
        --cpu-distributed-timeout-seconds 7200 \
        --compilation-config "$COMPILATION_CONFIG" \
        "${headless[@]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 16 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 16 \
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 1319 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 64 \
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>
```

[mikekg](https://github.com/mikekg) added a commit to mikekg/vllm that referenced this pull request [ Oct 9, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-5fa70db)
` [Bugfix][DeepSeek V4] Support sparse MLA on SM8x[](https://github.com/mikekg/vllm/commit/5fa70dbc7697bf98d1cbcd37e5ece34cf81bad78 "\[Bugfix\]\[DeepSeek V4\] Support sparse MLA on SM8x

DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from #47629 and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `\[batch\]` contract and DeepSeek V4's `\[batch, next_n\]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses #47629's generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on #47629 and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing #47629 paged-MQA test was extended with nonuniform two-dimensional context lengths `\[128, 129, 129, 130\]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \\
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \\
      -v

    TRITON_OVERRIDE_ARCH=sm80 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    TRITON_OVERRIDE_ARCH=sm86 \\
      .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

    .venv/bin/python -m pytest \\
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \\
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: \[2560, 2561, 2562, 2563\]
    expected: \[-1, -1, -1, -1\]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$\(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1\)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":\[1,2,4,8,16,32,64\],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=\(\)
      if \(\( SLURM_NODEID > 0 \)\); then
        headless=\(--headless\)
      fi

      exec vllm serve "$MODEL" \\
        --revision "$REVISION" \\
        --served-model-name deepseek-v4-pro \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --trust-remote-code \\
        --tokenizer-mode deepseek_v4 \\
        --load-format fastsafetensors \\
        --safetensors-load-strategy lazy \\
        --distributed-executor-backend mp \\
        --master-addr "$MASTER_ADDR" \\
        --master-port "$MASTER_PORT" \\
        --nnodes "$SLURM_NNODES" \\
        --node-rank "$SLURM_NODEID" \\
        --tensor-parallel-size 16 \\
        --pipeline-parallel-size 2 \\
        --enable-expert-parallel \\
        --kv-cache-dtype fp8 \\
        --block-size 256 \\
        --max-model-len 10240 \\
        --max-num-seqs 64 \\
        --max-num-batched-tokens 16384 \\
        --gpu-memory-utilization 0.9 \\
        --linear-backend auto \\
        --moe-backend auto \\
        --no-enable-prefix-caching \\
        --distributed-timeout-seconds 7200 \\
        --cpu-distributed-timeout-seconds 7200 \\
        --compilation-config "$COMPILATION_CONFIG" \\
        "${headless\[@\]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 16 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 16 \\
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \\
      --host http://127.0.0.1 \\
      --port 8000 \\
      --num-questions 1319 \\
      --num-shots 5 \\
      --max-tokens 1024 \\
      --temperature 0 \\
      --seed 42 \\
      --max-concurrency 64 \\
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with #47629 and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>") `

```
DeepSeek V4's compressed sparse-MLA path selects CUDA implementations that require DeepGEMM, CuTeDSL SM90 code generation, or native Triton `fp8e4nv` conversion. On A100/SM80, the model therefore fails before generation. The same selection assumptions also exclude SM86 and SM89.

Observed A100 model failures included:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed

The root causes are connected parts of the same sparse-MLA path. CuTeDSL selection checks only whether CuTeDSL is installed instead of whether the GPU supports its SM90 code generation. The compressor, indexer-query, cache-gather, and sparse-decode Triton kernels use native `fp8e4nv` conversion where Triton cannot lower it. The paged-MQA kernel assumes one final context length per request, while DeepSeek V4 supplies a context length for every compressed token. CUDA-graph padding rows can also retain a derived compressed length and publish physical cache indices even when their slot mapping marks them invalid.

Route pre-SM90 CUDA systems through the Triton sparse-MLA implementation from vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and restrict CuTeDSL selection to SM90 and newer. Use the shared FP8 E4M3 conversion helper on SM80 and SM86 while retaining native conversion on SM89 and the existing optimized paths on SM90 and newer. Pass the context tensor's actual strides to paged MQA and support both its existing `[batch]` contract and DeepSeek V4's `[batch, next_n]` per-token contract. Force invalid CUDA-graph rows to zero compressed length before constructing cache indices.

The implementation preserves the existing ROCm FNUZ conversion branches. It reuses vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629)'s generic sparse-MLA and paged-MQA kernels instead of introducing a second CUDA backend, and it reuses the shared FP8 helper instead of duplicating conversion logic at each DeepSeek V4 call site. This commit is intentionally stacked on vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and contains only the DeepSeek V4 integration and model-specific regressions.

The existing FlashMLA sparse test was extended with an invalid CUDA-graph row whose slot mapping is `-1`. Before the fix, that row retained a nonzero decode length and physical cache indices. After the fix, its decode length is zero and every index is `-1`.

The architecture-selection case is CPU-only and explicitly checks SM80, SM86, SM89, SM90, and SM100. The CUDA graph-padding regression ran on a physical RTX 4090 with native SM89 execution. Together, the two focused FlashMLA tests pass all seven parameterized cases.

The existing vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) paged-MQA test was extended with nonuniform two-dimensional context lengths `[128, 129, 129, 130]` for `next_n > 1`. These CUDA tests ran on a physical RTX 4090: once with native SM89 code generation and once each with Triton forced to target SM80 and SM86. Before the fix, the kernel read the matrix as a one-dimensional vector and produced mismatched logits. After the fix, all 20 parameterized cases pass for each target.

Focused tests:

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_sparse_mla_supports_cuda_architectures \
      tests/kernels/attention/test_flashmla_sparse.py::test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride \
      -v

    TRITON_OVERRIDE_ARCH=sm80 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    TRITON_OVERRIDE_ARCH=sm86 \
      .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

    .venv/bin/python -m pytest \
      tests/kernels/attention/test_mqa_logits_triton.py::test_fp8_paged_mqa_logits_triton_matches_torch \
      -v

Focused results:

    Before, physical RTX 4090 / native SM89 graph-padding test:
    invalid row indices: [2560, 2561, 2562, 2563]
    expected: [-1, -1, -1, -1]

    Before, physical RTX 4090 / paged-MQA test:
    SM80 target: two-dimensional context cases failed
    SM86 target: two-dimensional context cases failed
    native SM89: two-dimensional context cases failed

    After:
    FlashMLA sparse tests: 7 passed
    Paged-MQA, physical RTX 4090 / SM80 target: 20 passed
    Paged-MQA, physical RTX 4090 / SM86 target: 20 passed
    Paged-MQA, physical RTX 4090 / native SM89: 20 passed

The complete compiled path was exercised on four eight-GPU A100 nodes using Slurm and vLLM native multiprocessing:

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=8
    #SBATCH --time=02:00:00

    set -euo pipefail

    export MODEL=nvidia/DeepSeek-V4-Pro-0813-NVFP4
    export REVISION=2ff4ec53ee664e54571f670276d2c89d0fcc7b82
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=29501
    export COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"inductor_compile_config":{"enable_auto_functionalized_v2":true}}'

    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc '
      headless=()
      if (( SLURM_NODEID > 0 )); then
        headless=(--headless)
      fi

      exec vllm serve "$MODEL" \
        --revision "$REVISION" \
        --served-model-name deepseek-v4-pro \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --load-format fastsafetensors \
        --safetensors-load-strategy lazy \
        --distributed-executor-backend mp \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        --nnodes "$SLURM_NNODES" \
        --node-rank "$SLURM_NODEID" \
        --tensor-parallel-size 16 \
        --pipeline-parallel-size 2 \
        --enable-expert-parallel \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --max-model-len 10240 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9 \
        --linear-backend auto \
        --moe-backend auto \
        --no-enable-prefix-caching \
        --distributed-timeout-seconds 7200 \
        --cpu-distributed-timeout-seconds 7200 \
        --compilation-config "$COMPILATION_CONFIG" \
        "${headless[@]}"
    '

After the A100 server becomes ready, run a 16-question concurrency smoke and the full GSM8K evaluation from its head node:

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 16 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 16 \
      --save-results gsm8k-c16.json

    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
      --host http://127.0.0.1[](http://127.0.0.1) \
      --port 8000 \
      --num-questions 1319 \
      --num-shots 5 \
      --max-tokens 1024 \
      --temperature 0 \
      --seed 42 \
      --max-concurrency 64 \
      --save-results gsm8k-full.json

End-to-end A100 result before the fix:

    ValueError: type fp8e4nv not supported in this architecture
    NVVM backend compilation failed
    The server did not reach generation.

Together with vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) and the co-pending FP8-conversion-helper and SM8x output-projection PRs, this commit lets the compiled A100 server capture CUDA graphs and produces the following successful end-to-end results:

    DeepSeek V4 Pro c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Pro full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.965125
    strict accuracy: 0.953753
    invalid responses: 0
    output tokens: 121802
    evaluation time: 429.349 seconds

    DeepSeek V4 Flash c16: 16/16 requests completed, 0 invalid responses
    DeepSeek V4 Flash full GSM8K: 1319/1319 requests completed
    flexible accuracy: 0.962
    CUDA illegal-memory-access errors: 0

All applicable pre-commit hooks pass.

Co-authored-by: OpenAI Codex <noreply@openai.com>
Signed-off-by: Michael Gschwind <mgschwind@nvidia.com>
```

[Malav-P](https://github.com/Malav-P) pushed a commit to Malav-P/vllm that referenced this pull request [ Oct 13, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-1f70fb7)
` Merge glm53-a100 snapshot into all-models-a100[](https://github.com/Malav-P/vllm/commit/1f70fb73f8543c4e60fc7f62368728708eaca18b "Merge glm53-a100 snapshot into all-models-a100

Consolidates the last per-model branch not represented in
all-models-a100: the 2026-10-02 GLM-5.3-W4A16 serving snapshot
\(492f2d779c\), which captured the PR #47629 TRITON_MLA_SPARSE SM80/
SM121 working-tree edits on the b6ff8a2f5 base.

Conflict resolution: both sides carry generations of the same PR
#47629 fileset. Resolved all 7 conflicted files \(mla_attention.py,
sparse_attn_indexer.py, cuda.py, mla/indexer.py, triton_mla_sparse.py,
mqa_logits_triton.py, triton_mla_sparse_kernel.py\) to the
all-models-a100 evolution, which strictly supersedes the frozen
snapshot generation \(adds the int64 KV-row overflow fix, NoPE /
glm5_next geometry support via derived block_dpe, kpool-aware indexer
alignment, and head_size-aware sparse backend ordering\).

Surviving deltas from the glm53 snapshot \(the only content not already
in all-models-a100\):
- offloading scheduler: clamp num_chunks to the group's available
  block_ids // blocks_per_chunk
- xpu_mla_sparse: decode/prefill split metadata fields +
  mha_prefill_supported=False \(kernel only handles compressed MLA\)

Co-Authored-By: Claude Code <noreply@anthropic.com>") `

```
Consolidates the last per-model branch not represented in
all-models-a100: the 2026-10-02 GLM-5.3-W4A16 serving snapshot
([`492f2d7`](https://github.com/Malav-P/vllm/commit/492f2d779c20bc873aee95a682dd9fb7802e2757)), which captured the PR vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) TRITON_MLA_SPARSE SM80/
SM121 working-tree edits on the [`b6ff8a2`](https://github.com/Malav-P/vllm/commit/b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd) base.

Conflict resolution: both sides carry generations of the same PR
vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629) fileset. Resolved all 7 conflicted files (mla_attention.py,
sparse_attn_indexer.py, cuda.py, mla/indexer.py, triton_mla_sparse.py,
mqa_logits_triton.py, triton_mla_sparse_kernel.py) to the
all-models-a100 evolution, which strictly supersedes the frozen
snapshot generation (adds the int64 KV-row overflow fix, NoPE /
glm5_next geometry support via derived block_dpe, kpool-aware indexer
alignment, and head_size-aware sparse backend ordering).

Surviving deltas from the glm53 snapshot (the only content not already
in all-models-a100):
- offloading scheduler: clamp num_chunks to the group's available
  block_ids // blocks_per_chunk
- xpu_mla_sparse: decode/prefill split metadata fields +
  mha_prefill_supported=False (kernel only handles compressed MLA)

Co-Authored-By: Claude Code <noreply@anthropic.com>
```

[michael-gpu](https://github.com/michael-gpu) pushed a commit to practical-inference/vllm-ampere-extended that referenced this pull request [ Oct 20, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-70425a4)
` Add TRITON_MLA_SPARSE as BF16-KV sparse fallback on SM12x[](https://github.com/practical-inference/vllm-ampere-extended/commit/70425a468c146f00535ac05c682e732fc75ccaf9 "Add TRITON_MLA_SPARSE as BF16-KV sparse fallback on SM12x

FLASHINFER_MLA_SPARSE_SM120 \(#43477\) requires an FP8 KV cache; with BF16
KV it is rejected by supports_combination and SM12x had no sparse-MLA
candidate left after the rebase. Append TRITON_MLA_SPARSE after it so
FP8 deployments keep the native backend and BF16 falls through to
Triton, restoring the original PR's SM121 coverage.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>

\(cherry picked from commit b2abb0b0a3, thomaslwang:vllm PR #47629; applied cleanly\)

Co-authored-by: opencode <noreply@opencode.ai>
Signed-off-by: mike-columbo <columbo-analytics@pm.me>
\(cherry picked from commit 94ee37b0780ebedaba97218cb5303fb3079fb4f8\)") `

```
FLASHINFER_MLA_SPARSE_SM120 (vllm-project#43477[](https://github.com/vllm-project/vllm/pull/43477)) requires an FP8 KV cache; with BF16
KV it is rejected by supports_combination and SM12x had no sparse-MLA
candidate left after the rebase. Append TRITON_MLA_SPARSE after it so
FP8 deployments keep the native backend and BF16 falls through to
Triton, restoring the original PR's SM121 coverage.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>

(cherry picked from commit b2abb0b0a3, thomaslwang:vllm PR vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629); applied cleanly)

Co-authored-by: opencode <noreply@opencode.ai>
Signed-off-by: mike-columbo <columbo-analytics@pm.me>
(cherry picked from commit 94ee37b0780ebedaba97218cb5303fb3079fb4f8)
```

[michael-gpu](https://github.com/michael-gpu) added a commit to practical-inference/vllm-ampere-extended that referenced this pull request [ Oct 20, 2026 ](https://github.com/vllm-project/vllm/pull/47629#ref-commit-d8dd326)
` Add TRITON_MLA_SPARSE as BF16-KV sparse fallback on SM12x[](https://github.com/practical-inference/vllm-ampere-extended/commit/d8dd3262126d2633b0553aaf42ed44b082f8a727 "Add TRITON_MLA_SPARSE as BF16-KV sparse fallback on SM12x

FLASHINFER_MLA_SPARSE_SM120 \(#43477\) requires an FP8 KV cache; with BF16
KV it is rejected by supports_combination and SM12x had no sparse-MLA
candidate left after the rebase. Append TRITON_MLA_SPARSE after it so
FP8 deployments keep the native backend and BF16 falls through to
Triton, restoring the original PR's SM121 coverage.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>

\(cherry picked from commit b2abb0b0a3, thomaslwang:vllm PR #47629; applied cleanly\)

Co-authored-by: opencode <noreply@opencode.ai>
Signed-off-by: michael-gpu <18089103+michael-gpu@users.noreply.github.com>
\(cherry picked from commit 94ee37b0780ebedaba97218cb5303fb3079fb4f8\)") `

```
FLASHINFER_MLA_SPARSE_SM120 (vllm-project#43477[](https://github.com/vllm-project/vllm/pull/43477)) requires an FP8 KV cache; with BF16
KV it is rejected by supports_combination and SM12x had no sparse-MLA
candidate left after the rebase. Append TRITON_MLA_SPARSE after it so
FP8 deployments keep the native backend and BF16 falls through to
Triton, restoring the original PR's SM121 coverage.

Signed-off-by: Thomas Wang <thomas.l.wang@gmail.com>

(cherry picked from commit b2abb0b0a3, thomaslwang:vllm PR vllm-project#47629[](https://github.com/vllm-project/vllm/pull/47629); applied cleanly)

Co-authored-by: opencode <noreply@opencode.ai>
Signed-off-by: michael-gpu <18089103+michael-gpu@users.noreply.github.com>
(cherry picked from commit 94ee37b0780ebedaba97218cb5303fb3079fb4f8)
```

### This branch has not been deployed
No deployments 
[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**. Already have an account? [Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm%2Fpull%2F47629)
###  Uh oh! 
There was an error while loading. [Please reload this page](https://github.com/vllm-project/vllm/pull/47629).
###  Labels 
[ deepseek ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Adeepseek) Related to DeepSeek models [ documentation ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Adocumentation) Improvements or additions to documentation [ needs-rebase ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Aneeds-rebase) [ nvidia ](https://github.com/vllm-project/vllm/issues?q=state%3Aopen%20label%3Anvidia)
###  11 participants 
Add this suggestion to a batch that can be applied as a single commit.This suggestion is invalid because no changes were made to the code.Suggestions cannot be applied while the pull request is closed.Suggestions cannot be applied while viewing a subset of changes.Only one suggestion per line can be applied in a batch.Add this suggestion to a batch that can be applied as a single commit.Applying suggestions on deleted lines is not supported.You must change the existing code in this line in order to create a valid suggestion.Outdated suggestions cannot be applied.This suggestion has been applied or marked resolved.Suggestions cannot be applied from pending reviews.Suggestions cannot be applied on multi-line comments.Suggestions cannot be applied while the pull request is queued to merge.Suggestion cannot be applied right now. Please check back later.
You can’t perform that action at this time. 
