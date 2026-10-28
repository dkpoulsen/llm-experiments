---
title: 'DDP on a PCIe-only 4x RTX A6000 (no NVLink): plain DDP gave 2x, bf16 gradient
  compression plus accumulation gave 3.6x - distributed - PyTorch Forums'
id: ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compre
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:16:57.160196Z'
updated: '2026-10-27T19:01:55.156864Z'
source: https://discuss.pytorch.org/t/ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compression-plus-accumulation-gave-3-6x/225650
source_domain: discuss.pytorch.org
fetched_at: '2026-10-27T13:16:57.159438Z'
fetch_provider: builtin
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'DDP on a PCIe-only 4x RTX A6000 (no NVLink): plain DDP gave 2x, bf16 gradient
  compression plus accumulation gave 3.6x...'
utility_score: 16.0
---

DDP on a PCIe-only 4x RTX A6000 (no NVLink): plain DDP gave 2x, bf16 gradient compression plus accumulation gave 3.6x - distributed - PyTorch Forums
DDP on a PCIe-only 4x RTX A6000 (no NVLink): plain DDP gave 2x, bf16 gradient compression plus accumulation gave 3.6x
distributed
heydryft
(Nirupam Bhowmick)
October 26, 2026, 12:42pm
1
Disclosure up front: I work on Runcrate, where these ran.
Four GPUs should train about four times faster than one. On a rented 4x RTX A6000 machine this morning (2026-10-26), plain PyTorch DDP gave me 2x. This is the notebook of finding out why, and the two flags that took it to 3.6x.
Setup
Machine:
one Runcrate 4x RTX A6000 (48 GB each) VM, US Midwest, $2.64/hour for the whole machine at launch time. Two separate machines of the same type over the morning: the first for the baseline, the second for the fixes.
Topology
(
`nvidia-smi topo -m`
): every GPU pair is
`PHB`
, i.e. GPU to GPU traffic crosses the CPU’s PCIe host bridge. No NVLink. PCIe gen 3 x16 per card.
Model:
a GPT-2 style decoder written in plain PyTorch, two sizes: 124M parameters (12 layers, 768 wide) and 355M (24 layers, 1024 wide). Random tokens, sequence 1024, micro-batch 8 per GPU, bf16 autocast, fused AdamW.
Software:
torch 2.8.0+cu128 (first machine) and 2.11.0+cu128 (second), NCCL as shipped in the wheel.
Metric:
tokens/s from the median step time over 40 steps after 10 warm-up steps. Weak scaling: each GPU always gets its own micro-batch of 8.
The commands
```
# baseline: 1, 2 and 4 processes on the same machine
for N in 1 2 4; do torchrun --standalone --nproc_per_node $N train_ddp.py --size small; done

# all-reduce bandwidth, nccl-tests style (bus bandwidth = algbw x 2(n-1)/n)
torchrun --standalone --nproc_per_node 4 allreduce.py

# the fixes
torchrun --standalone --nproc_per_node 4 train_ddp.py --size small --hook bf16            # bf16 gradient compression
torchrun --standalone --nproc_per_node 4 train_ddp.py --size small --accum 4              # all-reduce every 4 micro-batches
torchrun --standalone --nproc_per_node 4 train_ddp.py --size small --accum 4 --hook bf16  # both
```
The two fixes are a few lines each in the training loop:
```
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
ddp = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)   # send gradients as bf16

for k in range(accum):                                            # gradient accumulation
    ctx = ddp.no_sync() if k < accum - 1 else contextlib.nullcontext()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model_loss(ddp, next_batch()) / accum
        loss.backward()
opt.step(); opt.zero_grad(set_to_none=True)
```
Baseline: 2x from 4 GPUs
Model
GPUs
Tokens/s
Speed-up vs 1 GPU
124M
1
61,907
1.00x
124M
2
77,287
1.25x
124M
4
127,682
2.06x
355M
1
25,762
1.00x
355M
2
30,431
1.18x
355M
4
48,611
1.89x
The all-reduce explains it. NCCL bus bandwidth between the four cards was
4.3 GB/s
(2 GPUs: 3.9 GB/s), flat from 8 MB to 1 GB messages. DDP all-reduces fp32 gradients: about 500 MB per step for the 124M model. At 2.9 GB/s of algorithm bandwidth that is roughly 170 ms of communication against a 130 ms compute step, and 4 GPUs measured about 255 ms per step. The GPUs spend more time talking than computing.
The fixes, on 4 GPUs
Model
Variant
Tokens/s
Speed-up vs 1 GPU (same variant)
$ per 1B training tokens
124M
1 GPU
64,912
1.00x
2.82 (one GPU’s share, $0.66/h)
124M
4 GPUs, plain DDP
129,032
1.99x
5.68
124M
4 GPUs, bf16 gradients
172,144
2.65x
4.26
124M
4 GPUs, accumulate 4
211,834
3.20x
3.46
124M
4 GPUs, both
234,175
3.54x
3.13
355M
1 GPU
27,109
1.00x
6.76
355M
4 GPUs, plain DDP
51,711
1.91x
14.18
355M
4 GPUs, bf16 gradients
73,653
2.72x
9.96
355M
4 GPUs, accumulate 4
86,507
3.10x
8.48
355M
4 GPUs, both
99,152
3.66x
7.40
(1-GPU tokens/s with accumulation 4: 66,189 and 27,893, so the “both” rows are 88% and 89% scaling efficiency against the same recipe on one GPU.)
What surprised me
The interconnect, not the GPU, set the speed.
4.3 GB/s of all-reduce bandwidth is a fraction of what PCIe gen 3 x16 can carry. With every pair going through the host bridge, plain DDP on four cards only doubled throughput.
Forcing peer-to-peer made it worse.
`NCCL_P2P_LEVEL=SYS`
(use P2P even across the host bridge) halved the all-reduce bandwidth to 2.3 GB/s and dropped 124M training from 129k to 80k tokens/s. NCCL’s default choice was right; don’t override it on a PHB machine.
Two cheap changes recovered most of it.
bf16 gradient compression halves the bytes; accumulating 4 micro-batches per all-reduce cuts the number of all-reduces by 4. Together: 3.5x to 3.7x on four GPUs. Accumulation changes your effective batch size (32 per GPU here), so retune the learning rate if you adopt it.
Per token, one GPU still won.
Even with both fixes, a token cost about 10% more on four GPUs than on one, because four GPUs at 88% efficiency is still four GPUs. Rent the 4x machine when wall-clock time matters (a 3.5x shorter run), not to save money.
The bigger model scaled slightly better,
as expected: more compute per byte of gradient.
Caveats
Random tokens and a synthetic model: this measures throughput, not convergence. bf16 gradient compression slightly changes numerics.
Two machines of the same type, one run per configuration. Topology varies between machines; run
`nvidia-smi topo -m`
and a 30-second all-reduce test before a long job.
Machines with NVLink behave completely differently; this is a PCIe-only result.
Prices are on-demand at launch time on 2026-10-26.
Reproduce it
Launch a 4x GPU instance through
`POST /api/v1/instances`
with
`max_runtime_minutes`
set.
`uv venv /opt/env && . /opt/env/bin/activate && uv pip install torch numpy --torch-backend=cu128`
.
Copy
`train_ddp.py`
and
`allreduce.py`
(about 100 lines together; the core is the snippet above) and run the torchrun commands.
Delete the instance.
Both 4x machines together, including setup and every configuration above, were billed $0.92.
Written with AI assistance; every number comes from the runs described above. Happy to answer questions about the method or rerun a configuration.
Original write-up:
4 GPUs, 2x speed: why PyTorch DDP stalled on a PCIe-only 4x RTX A6000, and two flags that took it to 3.6x - DEV Community
aicost_tools
(aicost.tools team)
October 27, 2026, 10:12am
2
Really useful writeup, and thanks for posting the topology output, PHB on every pair explains most of the 2x on its own. One more knob I’d be curious about on PCIe gen3: did you try a larger bucket_cap_mb (default is 25) together with gradient_as_bucket_view=True? With fewer, bigger all-reduces the overlap with backward sometimes gets noticeably better when there’s no NVLink. I work on aicost.tools, so the part I like most is that 2x vs 3.6x at the same $2.64/hour is almost half the cost per token.