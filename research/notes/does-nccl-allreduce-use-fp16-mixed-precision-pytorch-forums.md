---
title: Does NCCL allreduce use fp16? - mixed-precision - PyTorch Forums
id: does-nccl-allreduce-use-fp16-mixed-precision-pytorch-forums
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:16:55.760121Z'
updated: '2026-10-27T19:01:54.642137Z'
source: https://discuss.pytorch.org/t/does-nccl-allreduce-use-fp16/141461
source_domain: discuss.pytorch.org
fetched_at: '2026-10-27T13:16:55.759378Z'
fetch_provider: builtin
status: review
type: note
tier: practitioner
content_type: forum
deprecated: false
summary: Does NCCL allreduce use fp16? - mixed-precision - PyTorch Forums
utility_score: 16.0
---

Does NCCL allreduce use fp16? - mixed-precision - PyTorch Forums
Does NCCL allreduce use fp16?
mixed-precision
zzibc
January 12, 2022,  9:22am
1
Hello，
I tried to reduce communication bottleneck by using auto mixed precision training. I also used
`torch.profiler`
to check the training performance.
In the timeline, I noticed that the all-reduce kernel is
`ncclAllReduceRingLLKernel_sum_f32`
.
Does it mean the fp16 gradients are casted to fp32 to be all-reduced?  Does something like
`ncclAllReduceRingLLKernel_sum_f16`
exist?
My GPU is P100. PyTorch version is 1.9.1+cu111.
ptrblck
January 13, 2022,  5:32am
2
The parameters are stored in
`float32`
using the automatic mixed-precision util. and thus also the gradients. NCCL thus communicates them in
`float32`
, too. If you are calling
`.half()`
on the model directly and thus apply a pure
`float16`
training, NCCL should communicate in
`float16`
.
zzibc
January 13, 2022,  5:46am
3
Thanks ptrblck! I’ll try
`half()`
.
ptrblck
January 13, 2022,  5:47am
4
Sure, but be careful about the training stability, as
`amp`
makes sure that
`float32`
is used in layers where it’s needed.
zzibc
January 13, 2022,  6:20am
5
If I use
`torch.cuda.amp.autocast`
for mixed precision training, and I call
`model.half()`
to ensure that NCCL communicate in
`float16`
, do I need to call
`float()`
manually at some layers(like
`BatchNorm`
)?
ptrblck
January 13, 2022,  6:30am
6
`autocast`
doesn’t expect a manual
`.half()`
call so you should either use
`amp`
or your manual mixed-precision recipe.
epoch
January 14, 2022,  7:14am
7
Hi all,
I am confused here FP32 parameters are stored indeed as the master copy, but why not the gradients appear as FP16 to store and transfer to reduce the load as
Nvidia Apex Philosophy
details? Is there any tradeoff between the implementation and design of torch autocast?
Any feedback will be appreciated!
ptrblck
January 14, 2022,  5:34pm
8
`torch.cuda.amp`
is comparable to the deprecated
`O1`
`opt_level`
in apex, not
`O2`
, as the former was considered to be more stable.