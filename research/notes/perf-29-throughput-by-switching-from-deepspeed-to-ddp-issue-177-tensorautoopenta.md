---
title: 'perf: ~2.9× throughput by switching from DeepSpeed to DDP · Issue #177 · TensorAuto/OpenTau'
id: perf-29-throughput-by-switching-from-deepspeed-to-ddp-issue-177-tensorautoopenta
tags:
- cmp170hx-training-feasibility-a0d4fc
created: '2026-10-27T14:31:40.283188Z'
updated: '2026-10-27T19:00:30.548260Z'
source: https://github.com/cortwave/distributed-training
source_domain: github.com
fetched_at: '2026-10-27T14:31:39.658937Z'
fetch_provider: crawl4ai
status: deprecated
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: You signed in with another tab or window. Reload to refresh your session.
  You signed out in another tab or window. Re...
---

[Skip to content](https://github.com/TensorAuto/OpenTau/issues/177#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/TensorAuto/OpenTau/issues/177) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/TensorAuto/OpenTau/issues/177) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/TensorAuto/OpenTau/issues/177) to refresh your session. Dismiss alert
#  perf: ~2.9× throughput by switching from DeepSpeed to DDP
Copy link
Copy link
Closed
Closed
[perf: ~2.9× throughput by switching from DeepSpeed to DDP](https://github.com/TensorAuto/OpenTau/issues/177#top)#177
Feature
Copy link
Assignees
Labels
[importantSomething that's very important and necessary](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22important%22)Something that's very important and necessary[optimizationOptimizes the performance of something](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22optimization%22)Optimizes the performance of something
## Description
opened [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#issue-4312815530)
Last edited by shuheng-liu
Member
Issue body actions
## TL;DR
The default distributed backend for pi05 LIBERO training (`distributed_type: DEEPSPEED, zero_stage: 2`) leaves **~65% of GPU time on the table at fixed hardware**. Comparing **same-hardware** 8×A100 configurations, switching the training script to plain DDP (`distributed_type: MULTI_GPU`) combined with two small follow-on fixes (drop an unused `gemma_expert.lm_head` parameter, default fused AdamW) gives **2.90× more samples/s** , ~3.4× faster wall-clock per optimizer step, and **~2.9× fewer GPU-hours per sample** — without changing any model code or loss.
> **Distinguishing hardware scaling from software gains.** The pre-investigation libero accelerate config had `num_processes: 4`. A user upgrading from that 4-GPU config to the new 8-GPU DDP config observes `13.8 → 83.1 samples/s` = **~6.0× end-to-end** , but roughly half of that comes from doubling the hardware (DDP scales near-linearly 1→8 GPUs on this model, see the per-rank throughput discussion below). **The software-attributable win is 2.90×.** This issue focuses on the software axis.
Root cause is a known DeepSpeed ZeRO pathology for models with many small parameter tensors (PaliGemma + Gemma expert): per-parameter Python hooks installed on every backward pass dominate step time and cannot be reduced by any communication-overlap knob. Single-rank profiling shows the model is healthy and compute-bound; the overhead only appears once DeepSpeed is in the loop.
## What we ruled out
We worked through the obvious suspects in order. Each one is conclusively _not_ the cause:
  1. **Dataloader starvation.** `profile_dataloader.py` (no model, no collective) ran at **597 cluster-wide samples/s** while training consumed only ~29 samples/s under ZeRO-2. `dataload_wait` per step is **1.14 ms** in the actual training loop. Despite the suggestive square-wave pattern in the `nvidia-smi` chart, the input pipeline is roughly 20× ahead of the GPUs.
  2. **Per-step host/device sync (5×`gather_for_metrics(...).item()` in `update_policy`).** Measured at **3 ms / 0.1%** of step time. Real, but immaterial.
  3. **NCCL transport / topology.** `nvidia-smi topo -m` shows every GPU pair connected by `NV12` (12 NVLinks via NVSwitch). NCCL init log confirms `P2P/CUMEM/read` on 24 channels. Reduce-scatter of 8 GB of bf16 grads across 8 ranks should take ~25 ms on this fabric.
  4. **DeepSpeed comms not overlapping.** Adding `overlap_comm: true`, `contiguous_gradients: true`, `reduce_bucket_size: 5e8`, `allgather_bucket_size: 5e8`, `round_robin_gradients: true` to the DeepSpeed config produced **zero measurable improvement** (3071 ms → 3093 ms backward on 4-rank ZeRO-2, and similarly on 8-rank). Critical signal — if the bottleneck were comms, these would have helped.
  5. **`find_unused_parameters=True`at`src/opentau/scripts/train.py:128`.** Silently ignored under DeepSpeed (accelerate wraps with the DeepSpeed engine, not DDP). Active under DDP — does add some cost there, but DDP still beats DeepSpeed by 2.64× even with it on.
  6. **Eager attention / fp32 Q,K upcast / no SDPA / no torch.compile / no gradient checkpointing.** All real items but none explain the _pattern_. Forward time is ~380 ms and consistent with raw bf16 FLOPs on a ~3.4 B-param model at bs=12 on A100. The model is roughly compute-optimal in forward; problem is purely backward. SDPA and gradient checkpointing were later attempted in Phase 3; see the [follow-up comment](https://github.com/TensorAuto/OpenTau/issues/177#issuecomment-4301905944) for data confirming neither helps on this specific workload.


## What we zeroed in on
The decisive measurement was a single-GPU run with the same script and same model:  
| Phase  | DeepSpeed ZeRO-2 (4-rank, bs=12)  | **Single GPU (bs=13)**  | Delta  |  
| --- | --- | --- | --- |  
| forward  | 391 ms  | 376 ms  | ~same  |  
| **bwd**  | **3071 ms**  | **641 ms**  | **~2.4 s of pure DeepSpeed overhead**  |  
| total  | 3468 ms  | 1025 ms  |  
Single-GPU `bwd` of 641 ms is exactly the expected ~1.7× forward — pure compute, healthy. The extra **~2400 ms on multi-GPU is not GPU compute and not GPU communication.** It's CPU-side Python execution of post-accumulation hooks DeepSpeed installs on every parameter tensor: each PaliGemma LayerNorm, bias, gate proj, etc. fires its own hook between backward kernels, stalling the next kernel launch. With hundreds of small parameter tensors, the host time dominates the step.
The "low GPU utilization" originally observed in `nvidia-smi` is the direct consequence: util measures _fraction of time at least one kernel is active_ , so a 3 s wall-clock backward in which only ~450 ms is real kernel execution registers as ~15% util — exactly the chart we started from.
ZeRO-1 has the same hook mechanism and shows the same symptom (29.5 vs 28.7 samples/s on 8 GPUs — within noise of ZeRO-2). DDP installs gradient-bucket all-reduce hooks at _bucket_ granularity rather than parameter granularity, so the host overhead is amortized over hundreds of params per call rather than paid once per param.
## Benchmark — pi05 / LIBERO / 8×A100 80GB (fixed hardware, same model)
Profile script: `src/opentau/scripts/profile_step.py` (added in PR [#176](https://github.com/TensorAuto/OpenTau/pull/176)). Each row uses the _max_ per-rank batch size that fits in 80 GB, verified with `nvidia-smi`. Backward column = sum of `bwd + unscale_clip + optim_step + zero_grad_sched`. All 8-rank runs differ only in the `accelerate launch --config_file` argument (+ the post-PR fixes on DDP).  
| Backend  | bs/rank  | global bs  | bwd (ms)  | total (ms)  | **samples/s**  | same-HW speedup  | GPU mem  |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| ZeRO-2 8× _(tuned)_ — **baseline**  | 14  | 112  | 3442  | 3909  | **28.7**  | **1.00×**  | 79.7 GB  |  
| ZeRO-1 8× _(control)_  | 14  | 112  | 3334  | 3799  | 29.5  | 1.03×  | 79.7 GB  |  
| DDP 8× _(fup=true,`lm_head` still present)_  | 11  | 88  | 665  | 1160  | **75.8**  | **2.64×**  | 78.0 GB  |  
| DDP 8× + drop `lm_head` _(fup=false)_  | 12  | 96  | —  | 1216  | 78.9  | 2.75×  | 79.2 GB  |  
|  **DDP 8× + drop`lm_head` + fused AdamW** — **post-PR**  | 12  | 96  | —  | 1155  | **83.1**  | **2.90×**  | 79.2 GB  |  
| Single GPU _(scaling reference)_  | 13  | 13  | 705  | 1248  | 10.4  | —  | 78.2 GB  |  
**DDP scaling from 1→8 GPUs: 83.1 / 10.4 = 7.99×** — near-ideal. The per-rank throughput on DDP 8× (83.1 / 8 = 10.4 s/s/rank) matches single-GPU, meaning the gradient all-reduce is fully overlapped with backward compute. ZeRO-2 gets 28.7 / 8 = 3.6 s/s/rank — a ~65% per-rank efficiency loss vs. single-GPU, entirely host-side.
ZeRO-2 backward actually **got worse** going from 4 to 8 GPUs (3071 ms → 3442 ms) even with overlap_comm enabled. More ranks = more coordination pressure on the same per-parameter Python pathway. Conclusive evidence the bottleneck is host-side and not bandwidth-limited.
### GPU-hour economics (1M optimizer steps, fixed 8×A100 hardware)  
| Config  | step time  | samples/step  | wall-clock  | GPU-hours  | per 1M samples  |  
| --- | --- | --- | --- | --- | --- |  
| Pre-PR: ZeRO-2 8× @ bs=14  | 3909 ms  | 112  | 1086 h (45 d)  | 8688  | 77.6 GPU-h  |  
| **Post-PR: DDP 8× + fused AdamW @ bs=12**  | **1155 ms**  | 96  | **321 h (13 d)**  | **2568**  | **26.8 GPU-h**  |  
**~2.89× cheaper per sample** (per-sample cost is the fair metric since the two configs run different batch sizes). Matches the samples/s ratio exactly, as expected.
## External references confirming this is a known DeepSpeed pathology
  * **[deepspeedai/DeepSpeed#226 — "ZeRO-2 per-epoch training time significantly slower than ZeRO-1"](https://github.com/microsoft/DeepSpeed/issues/226)** — exact symptom we observed (~2× slower than ZeRO-1 with no expected memory savings), open on a model with many small parameters. Same pattern.
  * **[deepspeedai/DeepSpeed#4856 — "ZeRO Stage 2 Memory Leak? How to cleanup?"](https://github.com/microsoft/DeepSpeed/issues/4856)** — discussion attributes the issue to `create_reduce_and_remove_grad_hooks`, which is the same per-parameter hook mechanism we're paying CPU time for.
  * **[DeepSpeed ZeRO-3 docs](https://deepspeed.readthedocs.io/en/latest/zero3.html)** — ZeRO-3 explicitly classifies modules below a granularity threshold as "fine-grained" so they're treated as integral units, "to reduce host overhead and the separate allgather overhead introduced by hooks for fine-grained layers." This is DeepSpeed's own acknowledgement that the per-parameter-hook overhead is a real problem; ZeRO-3 has a workaround for it, ZeRO-2 does not.
  * **[HuggingFace accelerate ZeRO docs](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)** — explicit recommendation: _"if your model fits in DDP, pure DDP will be fastest."_ Pi05 does fit in DDP on A100 80GB at bs=12, so we follow this guidance.


The reason this didn't show up in HF's own benchmarks (which often report ZeRO-2 ≈ DDP at higher batch sizes) is that those benchmarks use models with relatively few large parameter tensors (DeBERTa, T5, etc.). Pi05 is the bad case: PaliGemma + Gemma expert have hundreds of distinct small tensors (every LayerNorm, embed, gate proj, bias), each one paying the Python hook cost.
## Fix landed in PR [#176](https://github.com/TensorAuto/OpenTau/pull/176)
**Three production changes** :
  1. **Switch the libero distributed default from DeepSpeed ZeRO-2 to plain DDP** — biggest contributor at 2.64× alone. File: `configs/libero/reproduce_pi05_libero_accelerate_config.yaml` (+ the new `configs/examples/accelerate_ddp_config.yaml` as the documented example).
  2. **Drop unused`gemma_expert.lm_head`** from pi05 and pi0 — 263 M-param tensor that receives no gradient under any training config (the action expert is encoder-only; its output flows through `action_out_proj` / `da_head`, never through a token head). Lets DDP run with `find_unused_parameters=False` and frees memory to go bs=11 → bs=12. pi05_mem and pi05_continuous_state import the same `PaliGemmaWithExpertModel` class and inherit the fix transitively.
  3. **Default to fused AdamW** (`torch.optim.AdamW(fused=True)`). Cuts `optimizer.step()` from 99.6 ms → 39.7 ms (60% reduction).


**Diagnostic infrastructure** also landed: `profile_step.py`, `profile_dataloader.py`, `find_unused_params.py` under `src/opentau/scripts/`. Used to reproduce the benchmarks above and to verify future policies don't regress.
**What did NOT help** and was reverted (see [follow-up comment](https://github.com/TensorAuto/OpenTau/issues/177#issuecomment-4301905944)):
  * SDPA attention (FA-2 backward recompute tax ≈ forward savings at pi05's sequence length ~1500 tokens) — break-even at fixed bs, reverted.
  * Gradient checkpointing (compute tax ~33% vs. ~2% amortization gain from the fixed per-step cost; net ~10% loss even at bs=37) — reverted.


These remain viable for _larger_ pi05 variants (7B+, longer action horizons) where the compute/memory tradeoffs flip. Tracked as open possibilities, not near-term action items.
## Profiling artefacts and reproduction
  * Profile scripts: `src/opentau/scripts/profile_step.py`, `profile_dataloader.py`, `find_unused_params.py` (PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176)).
  * Post-PR benchmark command: 

```
FIND_UNUSED_PARAMS=false accelerate launch \
    --config_file configs/libero/reproduce_pi05_libero_accelerate_config.yaml \
    src/opentau/scripts/profile_step.py \
    --config_path=configs/libero/reproduce_pi05_libero.json \
    --batch_size=12
# Expected: samples/s ≈ 83, total ≈ 1155 ms, optim_step ≈ 40 ms
```

  * Pre-PR baseline A/B (reproducible — just point at the DeepSpeed example config): 

```
accelerate launch \
    --config_file configs/examples/accelerate_deepspeed_config.yaml \
    src/opentau/scripts/profile_step.py \
    --config_path=configs/libero/reproduce_pi05_libero.json \
    --batch_size=14
# Expected: samples/s ≈ 28-29, total ≈ 3900 ms
```

  * Branch: `claude/debug-gpu-utilization-WiDcZ`
  * PR: [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176)


## Action items
  * **Promote DDP as the recommended default** in `docs/source/tutorials/training.rst` and `docs/source/installation.rst` (done in PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176), commit `4929110`).
**Update the libero reproduction recipe** (`configs/libero/reproduce_pi05_libero_accelerate_config.yaml`) to DDP 8-GPU bf16 (done in PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176), same commit).
**Fix pi05 / pi0 unused parameter** (`gemma_expert.lm_head`) so DDP can run with `find_unused_parameters=False` (done in PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176), commit `e1c8a38`).
**Default fused AdamW** on `AdamConfig` / `AdamWConfig` (done in PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176), commit `ed834f3`).
**Consolidate experimental accelerate configs** to a single `configs/examples/accelerate_ddp_config.yaml` (done in PR [perf(pi05): 2.9x LIBERO training throughput via DDP + fused AdamW #176](https://github.com/TensorAuto/OpenTau/pull/176), commit `4929110`).
**Keep`configs/examples/accelerate_deepspeed_config.yaml`** for CI coverage and memory-constrained users, documented as a fallback (done).
**Add docs note** about the DeepSpeed/per-parameter-hook pitfall so the next person picking ZeRO for a vision-language policy A/Bs against DDP first (done in `installation.rst` and `training.rst`).
**Follow-up PR** : audit remaining sibling policies (`pi05_continuous_state`, `pi05_mem`, `value`, `pi0`) for their own config defaults and benchmark whether they benefit from the same DDP + fused-AdamW treatment. pi0 already got the `lm_head` fix; the others inherit it transitively via the shared `PaliGemmaWithExpertModel`.
**Deferred** : `torch.compile` on the policy (+15-20% expected). Interacts awkwardly with `.to("cpu")` calls in `modeling_pi05.py:691,716` — separate investigation.
**Deferred** : SDPA and gradient checkpointing for larger pi05 variants. Infrastructure not yet in place; worth revisiting when we train a 7B+ policy or use longer action horizons.
To pick up a draggable item, press the space bar. While dragging, use the arrow keys to move the item. Press space again to drop the item in its new position, or press escape to cancel. 


cc generated by automated investigation; sourced from PR [#176](https://github.com/TensorAuto/OpenTau/pull/176) and on-box benchmarks on `gpu-387` (8×A100 80GB, NV12 NVSwitch fabric).
## Activity
  1. [shuheng-liu](https://github.com/shuheng-liu)
assigned [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24777579980)
  2. [shuheng-liu](https://github.com/shuheng-liu)
added 
[optimizationOptimizes the performance of something](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22optimization%22)Optimizes the performance of something
[importantSomething that's very important and necessary](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22important%22)Something that's very important and necessary
[on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24777585194)
  3. [shuheng-liu](https://github.com/shuheng-liu)
added the
issue type [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24777586239)
  4. [shuheng-liu](https://github.com/shuheng-liu)
added a commit that references this issue [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24779153212)
[perf(optim): enable fused AdamW/Adam kernel by default](https://github.com/TensorAuto/OpenTau/commit/ed834f3ac09e581d1e456fbe19baf40b561dce6a)
Verified[ed834f3](https://github.com/TensorAuto/OpenTau/commit/ed834f3ac09e581d1e456fbe19baf40b561dce6a)
  5. ### shuheng-liu commented on Apr 23, 2026 
[on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#issuecomment-4301905944)
Last edited by shuheng-liu
MemberAuthor
More actions
> **🛑 SUPERSEDED — see[post-#181-fix re-bench comment](https://github.com/TensorAuto/OpenTau/issues/177#issuecomment-4318533005).**
> The Phase 3.3 (SDPA) and Phase 3.4 (gradient checkpointing) negative-result conclusions below were measured under the bf16-Adam-state bug fixed by PR [#182](https://github.com/TensorAuto/OpenTau/pull/182). Post-fix numbers reverse both verdicts: SDPA gives +4.6% on libero/4-GPU, grad-ckpt gives +8.9% (alone) / +10.3% (stacked with SDPA). The prior text is preserved unedited for the historical record.
---## Phase 3.3 (SDPA attention) and Phase 3.4 (gradient checkpointing) — negative results, reverted
Summary of the two follow-up experiments that came after the Phase 3.2 fused-AdamW win. Neither produced a throughput gain on this workload; both will be reverted from PR [#176](https://github.com/TensorAuto/OpenTau/pull/176). Documented here so the next person with a similar hypothesis has the data and doesn't repeat the experiment.
### Experimental setup
All benchmarks run on `gpu-387` (8×A100 80 GB, NV12 NVSwitch), `profile_step.py`, same pi05 / LIBERO training config, DDP 8-rank, `find_unused_parameters=False`, fused AdamW enabled. Baseline (Phase 3.2 endpoint): **83.0 samples/s** at `batch_size=12`, 1157 ms/step, 79 GiB peak memory.
### Phase 3.3 — `scaled_dot_product_attention` vs. the custom eager kernel
Wired the existing-but-inert `attention_implementation` config flag to actually dispatch, and added an `sdpa_attention_forward` sibling to the custom eager kernel at `paligemma_with_expert.py:513-584`. Implementation verified via `optimizer.param_groups[0]` reflection: fused path activated, SDPA backend reached.  
| Config  | bs  | fwd  | bwd  | total  | samples/s  | mem  |  
| --- | --- | --- | --- | --- | --- | --- |  
| Eager (baseline)  | 12  | 379  | 706  | 1157  | **83.0**  | 79 GiB  |  
| SDPA  | 12  | 343 (**−36 ms**)  | 756 (**+50 ms**)  | 1174  | 81.8  | **70 GiB**  |  
| SDPA  | 14  | 402  | 875  | 1346  | 83.2  | 77 GiB  |  
     * **Forward is faster** by ~10%: the on-chip tiling in FlashAttention-2 reduces HBM traffic versus the materialize-then-softmax eager path. Expected and real.
     * **Backward is slower** by ~7%: FlashAttention-2 deliberately doesn't save the `(S×S)` attention matrix during forward; it saves `Q, K, V, O, logsumexp` and **re-materializes the scores in backward**. That's where the 9 GiB memory saving comes from, and that's where the 50 ms slowdown comes from.
     * Net same-bs: **−1.5%.** At bs=14 SDPA ties eager bs=12. Memory headroom is real (−9 GiB per rank) but there's no throughput to convert it into at this sequence length.
The underlying fact: pi05's prefix sequence length ≈ 1500 tokens, which sits in the awkward middle zone for FA-2. Published FA-2 wins are concentrated at either end — short sequences where launch overhead dominates (smaller win) or long sequences ≥4096 where memory traffic dominates (big win). At ~1500 with bs=12, the recompute tax on backward roughly balances the HBM savings on forward.
### Phase 3.4 — per-layer activation checkpointing
Added `gradient_checkpointing: bool = False` to `PI05Config`, refactored the custom per-layer loop into a `_run_layer` method, wrapped each layer call in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when the flag is active. Added a strict guard in `train.py` rejecting the flag under unsupported backends (ZeRO-3, FSDP).  
| Config  | bs  | total  | samples/s  | vs baseline  | mem  |  
| --- | --- | --- | --- | --- | --- |  
| Eager, no ckpt (baseline)  | 12  | 1157  | **83.0**  | 1.00×  | 79 GiB  |  
| Eager + ckpt  | 12  | 1558  | 61.6  | **0.74×**  | 48 GiB  |  
| Eager + ckpt  | 20  | 2344  | 68.3  | **0.82×**  | 58 GiB  |  
| Eager + ckpt  | 37  | 3960  | 74.7  | **0.90×**  | 79 GiB  |  
| SDPA + ckpt  | 38 (max)  | —  | — (≤ eager ckpt bs=37)  | **< 0.90×**  | 79 GiB  |  
**Ckpt is a net loss at every batch size up to the 80 GB memory ceiling.** Even at bs=37 (~3× the baseline batch, the largest that fits), throughput is 10% below the no-ckpt baseline. Stacking with SDPA barely changed the memory ceiling (max bs=38 vs 37) and didn't flip the sign.
**Why the amortization argument failed on this workload:** The per-step fixed cost (`optim_step + unscale_clip + sync_gather ≈ 60-70 ms`) was _already_ only 6% of step time at bs=12 and drops to 1.5% at bs=37 — amortization savings of ~2.3%. The ckpt compute tax is ~33%. Order-of-magnitude mismatch; tax overwhelms savings.
This is workload-specific. Grad checkpointing wins for:
     * Larger models (7B+) where forward compute dominates and fixed costs are a bigger share.
     * Longer sequences where activation memory is the binding constraint.
     * Cases where even `bs=1` OOMs without it.
None of those describe current pi05/libero.
### What gets kept vs. reverted
From PR [#176](https://github.com/TensorAuto/OpenTau/pull/176):  
| Commit  | Contents  | Disposition  |  
| --- | --- | --- |  
| Phase 1-2 diagnostics + DDP configs  | profile_step, profile_dataloader, find_unused_params, 7 accelerate YAMLs  |  **KEEP** — foundational  |  
| `e1c8a38`  | pi05+pi0: drop `gemma_expert.lm_head`, `FIND_UNUSED_PARAMS` env var  |  **KEEP** — enables `find_unused_parameters=False`  |  
| `ed834f3`  |  `fused=True` default on Adam/AdamW  |  **KEEP** — 5.3% throughput win, 83.1 samples/s endpoint  |  
| `5dc0bf8`  | SDPA attention backend, `attention_implementation` dispatch  |  **REVERT** — break-even, adds code surface for no win  |  
| `b22b8d7`  | Gradient checkpointing + `_run_layer` refactor + guard  |  **REVERT** — net loss on this workload  |  
After revert, the final PR delivers a clean **~5.5× throughput improvement** (13.8 → 76 samples/s on the same batch, or 83.1 with fused AdamW at current bs=12 settings) vs. the original ZeRO-2 4× baseline, with no behavior changes to the forward pass or attention path.
### Open follow-ups
     * `torch.compile` on the policy — deferred. Would likely recover some of the SDPA gains (SDPA is compile-friendly, eager attention is not), but the integration work is separate.
     * Sibling policies (pi0, pi05_continuous_state, pi05_mem) — the `gemma_expert.lm_head` fix from `e1c8a38` was also applied to pi0, but other structural optimizations should be a follow-up PR.
     * Revisit gradient checkpointing if/when we train a larger pi05 variant or longer-horizon action chunks.
_Generated by[Claude Code](https://claude.ai/code)_
  6. [shuheng-liu](https://github.com/shuheng-liu)
added a commit that references this issue [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24782789397)
[chore: consolidate accelerate configs, default to DDP in docs](https://github.com/TensorAuto/OpenTau/commit/492911015ea62e157ff7d64ee1c369551d1bf6fc)
Verified[4929110](https://github.com/TensorAuto/OpenTau/commit/492911015ea62e157ff7d64ee1c369551d1bf6fc)
  7. [shuheng-liu](https://github.com/shuheng-liu)
changed the title ~~[-] perf: pi05 LIBERO training is comms-bound on DeepSpeed ZeRO-2 — switch default to DDP for ~5.5× throughput[/-]~~ [+]perf: pi05 LIBERO training is comms-bound on DeepSpeed ZeRO-2 — switch default to DDP for ~2.9× throughput on same hardware[/+] [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24783732471)
  8. [shuheng-liu](https://github.com/shuheng-liu)
changed the title ~~[-] perf: pi05 LIBERO training is comms-bound on DeepSpeed ZeRO-2 — switch default to DDP for ~2.9× throughput on same hardware[/-]~~ [+]perf: ~2.9× throughput by switching from DeepSpeed to DDP[/+] [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24783948937)
  9. [shuheng-liu](https://github.com/shuheng-liu)
added 9 commits that reference this issue [on Apr 23, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-24784316700)
[docs(readme): advertise the training-diagnostics scripts](https://github.com/TensorAuto/OpenTau/commit/033b06f52382dec4d53b9456b5debdb911373c89)
Verified[033b06f](https://github.com/TensorAuto/OpenTau/commit/033b06f52382dec4d53b9456b5debdb911373c89)
[feat(scripts): add poll_gpu_memory.py for batch-size tuning](https://github.com/TensorAuto/OpenTau/commit/bf0e360e096971ba7651df37072ce3531d4b66e0)
[feat(scripts): add bench_with_mem.sh for batch-size tuning](https://github.com/TensorAuto/OpenTau/commit/ddd561c7365fc519b0c646be23a31b444a14021b)
[feat(scripts): add bench_with_mem.sh for batch-size tuning](https://github.com/TensorAuto/OpenTau/commit/fc7432b519203e6580f6fe1312a6660abd397474)
[fix(profile_step): wrap optimizer with MasterWeightOptimizer for fair…](https://github.com/TensorAuto/OpenTau/commit/f052579219e45d7c38e69f1038584720f5c39520)
[fix(master_weights): migrate fp32 masters to live device after prepare](https://github.com/TensorAuto/OpenTau/commit/a0795d15a36c2068244fe9a30fa9dea15b17e107)
[fix(master_weights): migrate fp32 masters to live device after prepare](https://github.com/TensorAuto/OpenTau/commit/dd1154ea4d2758ae9b277a9add5d32d5a06b2c4c)
[fix(profile_step): wrap optimizer with MasterWeightOptimizer for fair…](https://github.com/TensorAuto/OpenTau/commit/0bfab1751b94ed4a0cdafbe2dcf50ad44811b7d2)
[fix(master_weights): migrate fp32 masters to live device after prepare](https://github.com/TensorAuto/OpenTau/commit/ed06b54fab15905b76ac4ad5ac18a2d6f2cb5c34)
  10. ### shuheng-liu commented on Apr 25, 2026 
MemberAuthor
More actions
## Post-[#181](https://github.com/TensorAuto/OpenTau/issues/181)-fix re-benchmark — superseding the earlier 2.9× claim and the Phase 3.3 / 3.4 negative results
Issue [#181](https://github.com/TensorAuto/OpenTau/issues/181) surfaced that the DDP path was running with bf16 Adam state (root cause: `policy.to(torch.bfloat16)` ran before `torch.optim.AdamW(...)` was constructed under non-DeepSpeed backends; DeepSpeed swaps in `BF16_Optimizer` with fp32 master partitions, so it was unaffected). PR [#182](https://github.com/TensorAuto/OpenTau/pull/182) fixes this with `MasterWeightOptimizer`. **All DDP-path benchmarks in this issue's body and in the[Phase 3.3 / 3.4 comment](https://github.com/TensorAuto/OpenTau/issues/177#issuecomment-4301905944) were therefore measured under buggy bf16-Adam-state dynamics and don't reflect what the production fix produces.** That earlier comment is now flagged as superseded.
Re-ran the full matrix on `gpu-387` (4×A100 80GB; only 4 GPUs were free) on PR [#182](https://github.com/TensorAuto/OpenTau/pull/182) HEAD with the corrected fp32 master state.
### Corrected baseline (DDP fp32 vs ZeRO-2)  
| Backend  | bs/rank  | Peak  | Step  | Throughput  | vs ZeRO-2  |  
| --- | --- | --- | --- | --- | --- |  
|  **DDP fp32** (B1)  | 6  | 78.45 GiB  | 794 ms  |  **30.2** s/s  | **1.96×**  |  
| ZeRO-2 (B3, gas=1)  | 14  | 78.75 GiB  | 3648 ms  | 15.4 s/s  | 1.00×  |  
Per-rank static memory is up by ~+20 GB from the fp32 master state, dropping the largest fitting per-rank bs from 12 (under the buggy bf16-state path) to 6 under the corrected path. Same hardware. **The real same-hardware DDP-over-ZeRO-2 win at 4 GPUs is ~1.96×** , not the 2.90× originally claimed (which assumed DDP at bs=12 — a configuration only feasible because of the bug).
I don't have post-fix 8-GPU measurements (8 GPUs aren't currently allocatable here). ZeRO-2 backward at 8 GPUs was previously _worse_ than at 4 GPUs (per the original investigation: 3071 ms → 3442 ms), so the 8-GPU DDP-over-ZeRO-2 ratio could be slightly larger than 1.96× but is unlikely to recover the 2.9× headline.
### Stage 4 stack — corrected verdicts on FUP / SDPA / grad-ckpt
Re-evaluated each lever against the corrected DDP fp32 baseline:  
| Run  | bs  | Peak  | Throughput  | Δ vs B1  | Verdict  |  
| --- | --- | --- | --- | --- | --- |  
| B1 baseline (eager, FUP=true)  | 6  | 78.45 GiB  | 30.2 s/s  | —  | —  |  
| R1 (FUP=false)  | 6  | 78.43 GiB  | 30.3 s/s  | +0.3%  |  **REJECT** — within noise  |  
| R2 (SDPA, FUP=false)  | 7  | 77.96 GiB  | 31.6 s/s  | +4.6%  |  **ACCEPT** (modest win)  |  
| R3 (grad-ckpt, FUP=false)  | 18  | 78.15 GiB  | 32.9 s/s  | +8.9%  | **ACCEPT (opt-in)**  |  
| R4 (SDPA + ckpt, FUP=false)  | 19  | 79.22 GiB  | 33.3 s/s  | **+10.3%**  | **best**  |  
| R5 (SDPA, FUP=true; sanity)  | 7  | 77.96 GiB  | 31.6 s/s  | +4.6%  | confirms R1 — FUP no-op  |  
**`find_unused_parameters=False`**— the prior comment (and`train.py:127`) speculated 10–15% step-time savings. Measured: zero. The autograd-graph walk apparently overlaps with NCCL allreduce on this workload; observable wall-clock impact is below the noise floor. Default flip rejected.
**SDPA** — Phase 3.3 of the prior comment called this break-even at bs=12 (under the bug). Post-fix at bs=6→7, SDPA frees ~5.6 GiB on forward (no S×S materialization) and lets bs=7 fit where eager bs=8 OOMs. Net +4.6%. Accept (loss-equivalence A/B is a separate gating step before flipping the `attention_implementation` default to `"sdpa"`).
**Gradient checkpointing** — Phase 3.4 of the prior comment called this "net loss at every batch up to the 80 GB ceiling". That was true at bs=37 under bf16-state where fixed costs were already amortized at high bs. Post-fix at bs=6 baseline, the static-cost share is much larger; spending grad-ckpt's ~30 GiB activation savings on bs=18 amortizes that fixed cost and beats the recompute tax: **+8.9% alone, +10.3% stacked with SDPA**. Accept as opt-in (`PI05Config.gradient_checkpointing: bool = False`).
### What's now on PR [#182](https://github.com/TensorAuto/OpenTau/pull/182)
After today's cherry-picks the PR contains:
     * `MasterWeightOptimizer` (fp32 master weights, fp32 Adam state, post-upcast clip).
     * Subclass `torch.optim.Optimizer` so `accelerate.Accelerator.prepare` wraps it correctly + LR-scheduler rebind ([`8be2cd1`](https://github.com/TensorAuto/OpenTau/commit/8be2cd190745e5df50aa73015ab3e02ae8124c16)).
     * Device migration of fp32 masters after `accelerator.prepare` ([`dd1154e`](https://github.com/TensorAuto/OpenTau/commit/dd1154ea4d2758ae9b277a9add5d32d5a06b2c4c)).
     * SDPA backend opt-in via `PI05Config.attention_implementation`.
     * Gradient checkpointing opt-in via `PI05Config.gradient_checkpointing` + `train.py` startup guard for unsupported backends.
     * `profile_step.py` aligned with `train.py`'s master-weights wrap path so these numbers reproduce from main after [fix(pi05): DDP fp32 Adam state + SDPA / grad-ckpt opt-ins (#181, #177) #182](https://github.com/TensorAuto/OpenTau/pull/182) merges.
13 CPU tests pass; 3 GPU tests added for device migration. Re-bench on gpu-387 with 4×A100 confirms peaks are identical and throughput is within ±1% of the prior measurements after the cherry-picks (the new commits don't regress).
_Generated by[Claude Code](https://claude.ai/code)_
  11. [akshay18iitg](https://github.com/akshay18iitg)
added a commit that references this issue [on Apr 30, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-25010864144)
[fix(pi05): DDP fp32 Adam state + SDPA / grad-ckpt opt-ins ([#181](https://github.com/TensorAuto/OpenTau/issues/181), [#177](https://github.com/TensorAuto/OpenTau/issues/177)) (](https://github.com/TensorAuto/OpenTau/commit/523b25318a32f4037996524aae472e9c8d818617)
Verified[523b253](https://github.com/TensorAuto/OpenTau/commit/523b25318a32f4037996524aae472e9c8d818617)
  12. [shuheng-liu](https://github.com/shuheng-liu)
closed this as [completed](https://github.com/TensorAuto/OpenTau/issues?q=is%3Aissue%20state%3Aclosed%20archived%3Afalse%20reason%3Acompleted)[on May 15, 2026](https://github.com/TensorAuto/OpenTau/issues/177#event-25551096561)


[Sign up for free](https://github.com/signup?return_to=https://github.com/TensorAuto/OpenTau/issues/177)**to join this conversation on GitHub.** Already have an account? [Sign in to comment](https://github.com/login?return_to=https://github.com/TensorAuto/OpenTau/issues/177)
## Metadata
## Metadata
### Assignees
  * [WilliamYue37](https://github.com/WilliamYue37)
  * [akshay18iitg](https://github.com/akshay18iitg)
  * [shuheng-liu](https://github.com/shuheng-liu)


### Labels
[importantSomething that's very important and necessary](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22important%22)Something that's very important and necessary[optimizationOptimizes the performance of something](https://github.com/TensorAuto/OpenTau/issues?q=state%3Aopen%20label%3A%22optimization%22)Optimizes the performance of something
### Projects
No projects
### Milestone
No milestone
### Relationships
None yet
### Development
  * [ Aether: Solution for Issue #177TensorAuto/OpenTau ](https://github.com/TensorAuto/OpenTau/pull/196)
  * [ Aether: Solution for Issue #177TensorAuto/OpenTau ](https://github.com/TensorAuto/OpenTau/pull/218)
  * [ Aether: Solution for Issue #177TensorAuto/OpenTau ](https://github.com/TensorAuto/OpenTau/pull/219)


### Participants
## Issue actions
  * Open in GitHub Copilot app


You can’t perform that action at this time. 
