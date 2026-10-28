---
title: 'DeepSpeed ZeRO vs PyTorch FSDP2, Measured: 1.8x on NVLink, Barely Finishing
  on PCIe (Plus How to Actually Get GPUs on GKE) | Implicit None'
id: deepspeed-zero-vs-pytorch-fsdp2-measured-18x-on-nvlink-barely-finishing-on-pci-2
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-3
created: '2026-10-27T14:30:29.578843Z'
updated: '2026-10-27T19:01:56.675025Z'
source: https://implicit-none.com/en/deepspeed-zero-vs-fsdp2-benchmark/
source_domain: implicit-none.com
fetched_at: '2026-10-27T14:30:28.957593Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'Measured single-harness A/B of DeepSpeed ZeRO stages vs PyTorch FSDP2 on
  two interconnects — 4x A100 40GB (NVLink) and 4x L4 24GB (PCIe, no NVLink) — Qwen2.5-3B
  bf16, micro-batch 1, seq 2048, 4 ranks, 15 steps median. The PCIe run is the closest
  published analogue to a host-staged fabric: NCCL''s share of kernel time rises from
  17.3% to 50.2% for ZeRO-2, 26.1% to 46.7% for FSDP2 no-reshard, and 27.7% to 51.1%
  for FSDP2 reshard — communication becomes roughly half of kernel time once the wire
  is PCIe. Throughput on L4: ZeRO-2 1,923 tok/s (step 4,260.5 ms), FSDP2 no-reshard
  1,761 (4,651.6), FSDP2 reshard 1,218 (6,723.9), ZeRO-3 2,290 (3,577.2), ZeRO-3+offload
  1,405; on A100 ZeRO-2 17,124 and FSDP2 no-reshard 17,449. The ranking flips: on
  PCIe ZeRO-3 beats ZeRO-2 by ~19%, the reverse of NVLink. There is NO DDP baseline
  in the matrix (ZeRO-0 OOMs even on 40 GB at 30.01 GB resident before activations),
  so this cannot yield a ZeRO-2-vs-DDP step-time ratio. Caveats: L4 is PCIe Gen4 x16
  (~25 GB/s), ~25x faster than a Gen2 x4 CMP 170HX link; the author states the NCCL
  kernel-duration share ''does not map one-to-one onto pure communication-wait time'';
  n=1; defaults-vs-defaults.'
---

# DeepSpeed ZeRO vs PyTorch FSDP2, Measured: 1.8x on NVLink, Barely Finishing on PCIe (Plus How to Actually Get GPUs on GKE)
September 16, 2026 11 min read
## Introduction
DeepSpeed ZeRO and PyTorch FSDP are the two standard answers to “my model doesn’t fit”. “Which is faster?” gets asked constantly, yet numbers measured on **one shared harness, one model, the same step count — across two interconnects** — are surprisingly rare. So I wrote a benchmark that shares the measurement code across both engines ([distributed-training-bench](https://github.com/tsho/tsho-lab/tree/main/distributed-training-bench), Apache-2.0) and ran it on GKE in two configurations:
  * **4× A100 40GB (NVLink)** — fat pipes
  * **4× L4 24GB (PCIe)** — thin pipes


The results turned out to be about more than speed: **on PCIe, ZeRO-2/3 didn’t even finish** , and dissecting that failure (NCCL watchdog timeouts) and the one-env-var resurrection became the heart of this post. As a bonus, there’s a field log of the prerequisite problem — **having quota does not mean you can get GPUs** — and the design that fixes it.
Up front: this is an **out-of-the-box comparison — one model (3B), single node, n=1** (step times aggregated by median). DeepSpeed has tuning headroom (bucket sizes etc.) I did not explore; read this as a defaults-vs-defaults match.
## Results summary
Global throughput, **Qwen2.5-3B / bf16 / micro-batch 1 / seq 2048 / 4 GPUs** :  
| A100 ×4 (NVLink)  | L4 ×4 (PCIe)  |  
| --- | --- |  
| FSDP2 no-reshard (≈ZeRO-2)  | **17,449 tok/s**  | 1,761 tok/s  |  
| FSDP2 reshard (≈ZeRO-3)  | 14,936  | 1,218  |  
| FSDP2 reshard + CPU offload  | 1,687  | — (not collected)  |  
| ZeRO-1  | 16,642  | **OOM**  |  
| ZeRO-2  | 17,124  | 1,923 †  |  
| ZeRO-3  | 8,151  |  **2,290** †  |  
| ZeRO-3 + offload  | 2,615  | 1,405  |  
| ZeRO-0 (no sharding)  | **OOM**  | — (excluded; OOMs even at 40GB)  |  
† L4’s ZeRO-2/3 only finish with `PYTORCH_ALLOC_CONF=expandable_segments:True` (see below).
Four findings:
  1. **At equivalent sharding, FSDP2 reshard is 1.83x faster than ZeRO-3 and 27% lighter** (A100: 14,936 vs 8,151 tok/s, 13.34 vs 18.18 GB)
  2. **On PCIe, the NCCL share rises sharply for most configs** (FSDP2 no-reshard 26.1%→46.7%, ZeRO-2 17.3%→50.2%; ZeRO-3 was already at 42.7% on A100) — and on L4, **ZeRO-3 flips ahead of ZeRO-2** (the opposite of A100)
  3. **Even a 3B model doesn’t fit in 40GB without sharding** (ZeRO-0: 30GB resident before activations)
  4. **Near the memory boundary, OOM shows up wearing the absurd mask of “a 1-element allreduce that takes 10 minutes”** — the failure taxonomy is the fourth chapter


## Benchmark design
  * Model: Qwen/Qwen2.5-3B, bf16, micro-batch 1, seq 2048, 15 steps (median step time)
  * Configs: DeepSpeed ZeRO stages 0/1/2/3 (+ stage-3 offload) and FSDP2 reshard / no-reshard / reshard+offload — **measurement, model, and dataloader code shared across all of them**
  * Execution: **1 config = 1 Kubernetes Job** , all submitted at once. The node has 4 GPUs and each Job requests 4, so the scheduler serializes them automatically
  * Results upload to GCS from inside each job — completely decoupled from the session (Cloud Shell’s disconnect limit becomes irrelevant)
  * Jobs self-clean via `ttlSecondsAfterFinished`; **GPU nodes self-delete via autoscaler min-nodes 0** (structurally eliminating forgotten-node billing — more below)


This fire-and-forget design with fine-grained job splitting paid off twice: two Spot preemptions during the runs each killed only **the one job that was executing**. Completed results were already in GCS; Pending jobs evacuated automatically.
## Chapter 1: Getting the GPUs — quota is a borrowing right, not stock
Before benchmarking anything, I couldn’t get the GPUs. The process earned its own chapter.
### Six straight L4 failures, then an A100 in 60 seconds
L4×4 (g2-standard-48) failed **six times across Spot in 3 zones and on-demand in 3 zones** — with quota of 8 pre-confirmed in seven regions. **Quota is the right to borrow; it guarantees zero physical stock.** Most failures return this unhelpful shape (35 minutes of retrying, then an empty error field):

```
Deploy error: Not all instances running in IGM after 35m22.654468947s.
Expected 1, running 0, transitioning 1. Current errors: .
```

Only one zone was honest about the reason:

```
[GCE_STOCKOUT]: Instance 'gke-...-l4x4-...' creation failed: The zone
'.../zones/us-west4-a' does not have enough resources available to
fulfill the request. '(state:STOCKOUT, sub-state:STOCKOUT, resource type:compute)'.
```

Right after the sixth failure, **a 4× A100 40GB Spot pool in the very same zone (us-central1-a) provisioned in 60 seconds**. It wasn’t “Spot is unavailable” or “this zone is dry” — **a2 (A100) and g2 (L4) are separate physical pools**. When one GPU type is stuck, switching GPU type can beat switching zones. Also note the shape problem: in this experiment, getting g2-standard-48 (four L4s on one host) proved genuinely hard — it ultimately took about 14 hours of staking out. Single-node multi-GPU benchmarks hit this head-on.
### The fix: `--num-nodes 0` and letting the autoscaler play the lottery
A conventional pool creation retries for 35 minutes on stockout, **cannot be cancelled, and locks the cluster against other operations** (`incompatible operation`). Pulling that lever by hand every 35 minutes breaks a human by the third attempt.
The structural fix is an empty pool plus autoscaling:

```
gcloud container node-pools create l4x4 ... --spot --num-nodes 0 \
  --enable-autoscaling --min-nodes 0 --max-nodes 1
```

  * Pool creation succeeds in **seconds regardless of stock** (measured: the conventional create burned 35m6s into a STOCKOUT ERROR; the num-nodes 0 version succeeded immediately afterward)
  * The cluster autoscaler sees Pending pods and **retries the 0→1 scale-up indefinitely with backoff** — a machine staking out inventory on your behalf (the event log accumulates `in backoff after failed scale-up` x40, x44, x49 — the stakeout’s receipts)
  * Empty pools in multiple zones give the autoscaler more capacity candidates to try — GPU stock differs by zone, so in practice keeping several candidates open beat waiting on one zone; with `nodeSelector: gke-accelerator: nvidia-l4`, whichever zone wins runs the experiment automatically
  * When the pods finish, **the node self-deletes in ~10 minutes — forgotten-node billing is structurally gone**


The L4 stakeout eventually won (the ~14 hours above). Spot inventory is a tide — it shifts by hour, GPU type, and machine shape. Let the autoscaler do the staking out.
## Chapter 2: A100 (NVLink) — engine choice only gets expensive at stage 3  
| config  | tok/s  | step ms  | peak alloc GB  | NCCL share  |  
| --- | --- | --- | --- | --- |  
| fsdp2-noreshard  | **17,449**  | 469.5  | 13.99  | 26.1%  |  
| fsdp2-reshard  | 14,936  | 548.5  | **13.34**  | 27.7%  |  
| fsdp2-reshard-offload  | 1,687  | 4855.7  | 7.68  | 16.4%  |  
| zero-1  | 16,642  | 492.2  | 22.33  | 16.1%  |  
| zero-2  | 17,124  | 478.4  | 20.14  | 17.3%  |  
| zero-3  | 8,151  | 1005.0  | 18.18  | 42.7%  |  
| zero-3-offload  | 2,615  | 3133.1  | 6.69  | 35.5%  |  
| zero-0  | —  | —  | **OOM**  | —  |  
Note: the NCCL share is the ratio of NCCL kernel duration reported by the profiler; communication overlaps with compute, so it does not map one-to-one onto pure communication-wait time in wall-clock terms.
What to read from it:
  * **ZeRO-1/2 are nearly free** (16.6k/17.1k tok/s ≈ FSDP2 no-reshard’s 17.4k). Optimizer and gradient sharding cost almost nothing over NVLink
  * Only **full parameter sharding (stage-3 territory) is expensive — and its price depends on the engine** : FSDP2 reshard 14,936 vs ZeRO-3 8,151 tok/s (1.83x), with NCCL share 27.7% vs 42.7%. The implementation of the communication schedule is the difference
  * **The memory staircase is flatter than theory** (zero-1→2→3: 22.3→20.1→18.2 GB vs the ~21/17/12 you’d compute). Activations, allocator overhead, and gather buffers add a platform heel — capacity-plan from measurements, not formulas
  * **Offload buys GPU memory with speed.** FSDP2 reshard + CPU offload cut peak allocation from 13.34 to 7.68 GB (~5.7 GB back) while throughput fell 14,936→1,687 tok/s (~8.9x). ZeRO-3 + offload reclaimed more — 18.18→6.69 GB (~11.5 GB) — for a smaller 8,151→2,615 tok/s (~3.1x) hit
  * And **ZeRO-0 (pure replication) doesn’t fit a 3B model in 40GB** — params + grads + Adam fp32 state replicate to 30GB resident per rank before activations:


```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.50 GiB.
GPU 3 has a total capacity of 39.49 GiB of which 9.48 GiB is free.
... this process has 30.01 GiB memory in use.
```

## Chapter 3: L4 (PCIe) — communication takes over, and the ranking flips  
| config  | tok/s  | step ms  | peak alloc GB  | NCCL share  |  
| --- | --- | --- | --- | --- |  
| fsdp2-noreshard  | 1,761  | 4651.6  | 13.99  | 46.7%  |  
| fsdp2-reshard  | 1,218  | 6723.9  | 13.34  | **51.1%**  |  
| zero-1  | —  | —  | 21.35  | **OOM**  |  
| zero-2 †  | 1,923  | 4260.5  | 20.14  | 50.2%  |  
| zero-3 †  | **2,290**  | 3577.2  | 18.17  | 47.5%  |  
| zero-3-offload  | 1,405  | 5829.7  | 6.69  | 39.1%  |  
† Only finish with `expandable_segments:True` (next chapter).
  * **On the PCIe L4, the NCCL share rises sharply for most configs** : FSDP2 no-reshard goes from 26.1% on A100 to 46.7%, FSDP2 reshard from 27.7% to **51.1%** , ZeRO-2 from 17.3% to 50.2%. ZeRO-3, already communication-heavy at 42.7% on A100, sits at 47.5%. Resharding’s relative cost roughly doubles (reshard slowdown: -14% on A100 → **-31%** on L4). “Paying communication to buy memory” gets pricier as the pipes get thinner
  * **On L4, ZeRO-3 (2,290 tok/s) comes out ~19% ahead of ZeRO-2 (1,923 tok/s)** — the reverse of the A100 ranking. One strong hypothesis is memory pressure: at 24GB, ZeRO-2 runs closer to the allocator ceiling, which may favor ZeRO-3’s deeper sharding. This benchmark does not separate memory pressure from collective-scheduling and buffering effects, though, so the cause is not settled
  * **ZeRO-1 doesn’t fit 24GB** (peak 21.35GB sounds like it fits ~22GB effective; it OOMs immediately at initialization). GPU memory dictates your sharding strategy
  * Caveat: raw L4 compute is ~1/2.6 of A100, so the >10x gap is not all interconnect. **The NCCL time share is the instrument that separates the two** — that’s the methodological backbone of this benchmark


## Chapter 4: A taxonomy of how distributed training dies — OOM wearing a wedge’s mask
Vanilla ZeRO-2/3 on L4 **did not finish** , and how they died is the most interesting part of this post.
### A 1-element allreduce cannot take 10 minutes

```
[Rank 2] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=4016, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
Timeout(ms)=600000) ran for 600059 milliseconds before timing out.  → exit -6
```

**A one-element ALLREDUCE timing out after 10 minutes.** One element is never slow; what actually happened is: one rank went silent first → the other three waited forever inside the collective (a wedge) → the NCCL watchdog (default 10 min) killed everyone. The job had completed 4,016 collectives before freezing — this is deadlock, not bandwidth.
### The same disease, two faces
On the rerun, **zero-3 died differently: a clean CUDA OOM whose error message confessed everything** :

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 926.00 MiB.
GPU 1 has a total capacity of 22.03 GiB of which 439.00 MiB is free.
... 15.46 GiB is allocated by PyTorch, and 5.77 GiB is reserved by PyTorch
but unallocated. If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

In this reproduction, **allocator fragmentation is very likely the main factor** : the second run hit a clean CUDA OOM with 5.77 GiB reserved but unallocated, PyTorch itself suggested `expandable_segments:True`, and enabling it let ZeRO-2/ZeRO-3 finish (next section).
In distributed execution, when one rank fails to reach a collective (an OOM, a CUDA error), the other ranks can see it as an NCCL watchdog timeout — and here, the later OOM reproduction plus the expandable_segments recovery strongly suggest memory fragmentation was involved. The practical lesson: what would be a plain OOM on a single GPU can surface as “a 1-element allreduce took 10 minutes” in the distributed setting.
Peak memory correlated cleanly with the manner of death (peaks from the A100 runs; L4 effective ~22GB):  
| config  | peak alloc  | fate on L4  | collectives before death  |  
| --- | --- | --- | --- |  
| zero3-offload  | 6.7 GB  | finished  | —  |  
| fsdp2-reshard  | 13.3 GB  | finished  | —  |  
| fsdp2-noreshard  | 14.0 GB  | finished  | —  |  
| zero3  | 18.2 GB  | wedge → watchdog  | SeqNum 4016  |  
| zero2  | 20.1 GB  | wedge → watchdog  | SeqNum 456  |  
| zero1  | 21.4 GB  | immediate clean OOM  | 0  |  
**The survival line sits between 14GB and 18.2GB, and the closer to the boundary, the sooner the death.**
### One env var: dead → alive
Following the error message’s own prescription, I reran with `PYTORCH_ALLOC_CONF=expandable_segments:True` — and **ZeRO-2, ZeRO-3, and ZeRO-3-offload all finished cleanly** (the † values in Chapter 3). The `PYTORCH_CUDA_ALLOC_CONF` spelling in the error message still works as a backward-compatible alias; this post uses the current name `PYTORCH_ALLOC_CONF`. Note that `expandable_segments` is an experimental allocator option, not a universal fix for OOMs or fragmentation. A job with 20.14 GB peak alloc standing up on a 22GB GPU is a number fragmentation would never allow. The only change was one environment variable.
The practical takeaway — distinguish at least three kinds of “Failed” in distributed training:  
| death  | symptom  | how to tell  |  
| --- | --- | --- |  
| **OOM**  | records its own error politely  | exit 1, `torch.OutOfMemoryError` in logs  |  
| **preemption**  | logs go dark (kubelet dies too, so `kubectl logs` times out)  | node NotReady; autopsy via Cloud Logging (30-day retention)  |  
| **collective wedge**  | watchdog timeout (“N-element allreduce took 10 min”)  | exit -6, `Watchdog caught collective operation timeout`  |  
Recognizing which one you’re looking at is what determines your diagnosis speed.
## Bonus trap: the deepspeed launcher’s `--local_rank` injection
Right after the GPUs finally arrived, **all five ZeRO jobs died instantly with exit code 2** :

```
bench_zero_stages.py: error: unrecognized arguments: --local_rank=0
```

The `deepspeed` launcher passes **`--local_rank=N`to the training script by default** (it also sets the LOCAL_RANK env var). There are two ways out: have the training script accept `--local_rank`, or pass `--no_local_rank` to the launcher. I chose the argparse route (`p.add_argument("--local_rank", type=int, default=-1)` — accept and ignore) to keep the shared harness change minimal. The lesson, though, is the **asymmetric death** : the torchrun-launched (FSDP2) jobs on the identical harness were untouched, because torchrun is env-only. If you ever see “deepspeed jobs all dead, torchrun jobs fine”, suspect launcher argument injection first. It’s also a class of bug that CPU-side validation can never catch.
## When to use what (within this benchmark’s scope)  
| situation  | pick  | why  |  
| --- | --- | --- |  
| NVLink, memory comfortable  | ZeRO-1/2 or FSDP2 no-reshard  | sharding is nearly free (17k tok/s class)  |  
| NVLink, memory tight  | **FSDP2 reshard**  | 1.83x faster and 27% lighter than ZeRO-3  |  
| PCIe (thin pipes)  |  **FSDP2 reshard or ZeRO-3** + `expandable_segments:True`  | ~50% communication tax is unavoidable; near the boundary, fragmentation is life-or-death  |  
| 24GB-class GPUs, 3B+ models  | ZeRO-3 (+offload) or FSDP2 reshard  | ZeRO-1 physically doesn’t fit  |  
| memory above all  | offload variants  | ~5.7 GB (FSDP2) to ~11.5 GB (ZeRO-3) back, for a ~3–9x throughput hit  |  
## Summary
  * FSDP2 and ZeRO only diverge in **full-parameter-sharding territory** — and there FSDP2 is 1.8x faster (measured, A100)
  * The interconnect rewrites the leaderboard: on PCIe the NCCL share rises sharply for most configs and **ZeRO-3 flips ahead of ZeRO-2**
  * Near the memory boundary, **OOM can masquerade as a wedge**. `expandable_segments:True` is the env var worth taking home
  * GKE GPUs are a stock game, not a quota game. **num-nodes 0 + autoscaler min 0** makes a machine play the lottery for you and structurally removes forgotten-node billing
  * Harness, K8s manifests, and raw data are on [GitHub](https://github.com/tsho/tsho-lab/tree/main/distributed-training-bench)


## References
  * [PyTorch FSDP2 (fully_shard)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
  * [PyTorch CUDA memory management (expandable_segments)](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  * [GKE cluster autoscaler](https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler)


Share:
## Related articles
## [ Benchmarking TorchTPU 0.2.0: SingleTraceTrainer Is 3.4x Faster than Eager — Why Isn't torch.compile Faster?  Hands-on TPU v6e-1 benchmark of TorchTPU 0.2.0 across 7 configurations on a 0.87B Transformer. SingleTraceTrainer reaches 3.4x over Eager while torch.compile(model) ties DeferAndFuse. Includes 5 implementation notes and fixes. October 26, 2026 ](https://implicit-none.com/en/torch-tpu-execution-modes/) ## [ JAX on TPU v6e: DP vs FSDP-like Benchmark at 0.8B and 3.2B  Benchmarking JAX Data Parallel vs FSDP-like parameter sharding on 4 TPU v6e chips. FSDP-like is 12% faster at 0.87B, while DP is 34% faster at 3.27B. Includes memory use, stack-upgrade regressions, and TPU provisioning notes. September 17, 2026 ](https://implicit-none.com/en/jax-tpu-v6e-dp-fsdp-benchmark/) ## [ PyTorch 2.14: clamp's Boundary Gradient Silently Changes, bfloat16 Complex Promotion Now Raises, cholesky/qr Removed  What changed in PyTorch 2.14.0: clamp's boundary subgradient silently changes from 1.0 to 0.0, bfloat16-complex promotion moves to bcomplex32 and breaks previously working code with NotImplementedError, torch.cholesky/torch.qr finally removed, TorchScript starts emitting visible FutureWarnings, Python 3.15 (no-GIL) wheels. Every behavior verified by running both 2.13.0 and 2.14.0. September 6, 2026 ](https://implicit-none.com/en/pytorch-214-release-notes/)
