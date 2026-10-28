---
title: 'half precision reduction accumulation in fp32? · Issue #1026 · NVIDIA/nccl
  · GitHub'
id: half-precision-reduction-accumulation-in-fp32-issue-1026-nvidianccl-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:18:33.895689Z'
updated: '2026-10-27T19:00:31.669952Z'
source: https://github.com/NVIDIA/nccl/issues/1026
source_domain: github.com
fetched_at: '2026-10-27T13:18:33.894332Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'half precision reduction accumulation in fp32? · Issue #1026 · NVIDIA/nccl
  · GitHub'
utility_score: 15.0
---

half precision reduction accumulation in fp32? · Issue #1026 · NVIDIA/nccl · GitHub
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
half precision reduction accumulation in fp32?
#1026
New issue
Copy link
New issue
Copy link
Closed as not planned
Closed as not planned
half precision reduction accumulation in fp32?
#1026
Copy link
Description
stas00
opened
on Oct 14, 2023
Issue body actions
Are there plans to fix NCCL to perform reductions on BFLOAT16 operands with fp32 accumulation? Otherwise we can't reduce grads without a large loss and have to use fp32 comms which is both expensive in terms of slower overall comms and up/downscaling bf16 operands compute overhead before/after the collective is performed.
Apologies if this has already been discussed or even implemented recently - last I checked some months back it wasn't the case.
Thank you.
Reactions are currently unavailable
Activity
sjeaugey
commented
on
Oct 16, 2023
sjeaugey
on Oct 16, 2023
Member
More actions
Well it's not really a "bug"; communication libraries have to minimize the amount of data they exchange between GPUs and that imposes constraints on how we perform reductions. And accumulating in fp32 would change nothing in most cases, as we're just adding two values at each step.
But we're certainly trying to improve the stability and precision of floating point reductions, by using algorithms which are more stable than others.
The least stable algorithm is rings, which unfortunately is also the one which gives the best peak bandwidth on most platforms. The tree algorithm (along with NVLSTree) is much more stable because it adds values of equal weights, as each value represents the sum of as many GPUs on each side of the tree -- rather than adding one value with the sum of N with N increasing as we progress along the ring. Some algorithms, based on direct alltoall are adding e.g. 8 values intra-node before reducing inter-node. In that case, we could accumulate in fp32 to gain some precision but we'll have to convert back to 16 bit precision before communicating inter-node. But those alltoall-based algorithms work only on systems with full NVLink connectivity between GPUs within the node; they can't be used everywhere.
Overall, we can't really accumulate in fp32 because it would mean communicating fp32 between GPUs, which would be the same as running the allreduce operation on fp32 to start with, which is always an option users have if the precision isn't good enough for their needs. Some algorithms can produce better numerical results, but it may be at the expense of performance. Right now there is no way to select a specific algorithm -- we could think about adding an API config so that users can ask for better performance or better precision.
❤️
React with ❤️
5
Reacted by
Stas Bekman, holmes313, Joonhyung Lee/이준형, Bilal Khan and Zheng Cai
stas00
commented
on
Oct 16, 2023
stas00
on Oct 16, 2023
Last edited by
stas00
Author
More actions
Thank you for this super-detailed reply, Sylvain.
I was thinking in terms of "fixing" from remembering how pytorch had to fix this for all operations that had their operands as bf16 and needed to do the accumulation in fp32.
e.g. norm.
I didn't realize that it was happening 2 numbers at a time! Then indeed it'd be pointless to do it in fp32.
I haven't thought if perhaps some other collective should be used instead so that the user could control the accumulation precision of the reduction, but typically those are much more expensive than having NCCL do the work.
What about Kahan summation which is used for this kind of use case, but it'd still require an additional payload to transmute along with the normal data.
And while at it as fp8 is emerging - and I'd imagine you'd support that format, if it's not the case already - what would happen there with respect to preserving precision during reductions?
sjeaugey
commented
on
Oct 17, 2023
sjeaugey
on Oct 17, 2023
Member
More actions
What about Kahan summation which is used for this kind of use case, but it'd still require an additional payload to transmute along with the normal data.
And while at it as fp8 is emerging - and I'd imagine you'd support that format, if it's not the case already - what would happen there with respect to preserving precision during reductions?
We do not support fp8 yet, for that exact precision reason. We could have added support for fp8 with little effort, but it would be useless as we would always overflow. So to add fp8 support we need to first figure out a way to make it numerically useful.
stas00
commented
on
Oct 17, 2023
stas00
on Oct 17, 2023
Last edited by
stas00
Author
More actions
Right, Kahan summation helps with lost precision, but small dynamic range requires a scaler to avoid overflow - like fp16 mixed precision - which is how fp8 mixed precision training is implemented.
Except in the case of a mixed precision fwd/bwd/update cycle while getting the scaling factor right one could simply skip the update, but what would one do with a comm overflow...
Do the creators of fp8 @ NVIDIA have any suggestions here?
👍
React with 👍
3
Reacted by
muhtasham, naveen marri and Joonhyung Lee/이준형
harishankar-gopalan
commented
on
Oct 24, 2023
harishankar-gopalan
on Oct 24, 2023
More actions
What about Kahan summation which is used for this kind of use case, but it'd still require an additional payload to transmute along with the normal data.
And while at it as fp8 is emerging - and I'd imagine you'd support that format, if it's not the case already - what would happen there with respect to preserving precision during reductions?
We do not support fp8 yet, for that exact precision reason. We could have added support for fp8 with little effort, but it would be useless
as we would always overflow
. So to add fp8 support we need to first figure out a way to make it numerically useful.
Hi
@sjeaugey
thanks for the detailed reply. I have been looking into FP8 and DelayedScaling for training LLMs. In this regard this specific comment caught my attention where the claim is that FP8 would always overflow. I have two questions (may be naive) regarding the same:
Is this claim solely because FP8 has smaller range to not overflow during a reduction operation or is there some other practical/experimental opinion where this already occurs ?
Also currently when we enable FP8 for training, how does it work currently ? I suppose as the model
dtype
is anyway going to be either FP16 or FP32 the accumulation during AllReduce ops happen over either FP32 or FP64 respectively.
Please help me better understand the current working of FP8 in NCCL.
dfyz
commented
on
Jan 17, 2024
dfyz
on Jan 17, 2024
More actions
Sorry for bumping this old issue, but I have an update which I hope some might find interesting.
When the discussion started,
@stas00
said that fp32 comms were problematic because of 1) "slower overall comms" and 2) "up/downscaling bf16 operands compute overhead before/after the collective is performed". To which
@sjeaugey
replied that (my paraphrase, sorry if got this wrong) when accumulating in fp32, the comms have to remain in fp32 anyways, so you don't lose anything from just upcasting the input to fp32 and reducing over that.
I think that this is not strictly true because of 2): when your real input (e.g., gradients) is sufficiently large and is in bfloat16, the additional upcasts/downcasts can hurt, mostly because of additional memory needed for temporary buffers. E.g., for one of our models, we have 2 GiB worth of gradients per layer we want to reduce over. This is 4 GiB in fp32, which is not insignificant.
The good news is that the ring algorithm combined with the SIMPLE protocol (this pair I think is used most of the time for large reductions) doesn't really need the whole tensor to be stored in fp32 upfront. As a proof of concept, I forked NCCL to
implement a mixed-precision ReduceScatter
collective that can reduce directly over bfloat16 inputs/outputs with fp32 comms/accumulators.
API-wise, there's a
ncclMixedPrecisionReduceScatter()
function that is identical to
ncclReduceScatter()
, except it has two
`ncclDataType_t`
's instead of one (for the input/output and for the comms/accumulations). I also have some preliminary PyTorch bindings, but they are too ugly to publish right now.
This appears to work well: for 2 GiB tensors and 256 GPUs, the new reduction kernel works exactly as fast as before, but the temporary 4 GiB buffer and the conversion kernels disappear.
Implementation-wise, we just pretend we're doing regular ring-based fp32 reduce-scatter (i.e., the communication plan for the proxy remains the same) that reduces the data over the ring using a series of steps operating on small-ish chunks. Whenever we want to load from the user input (once for all steps except the last one) or store to the user output (at the last step), we use a byte pack that is twice as short (i.e., bfloat16), converting the data in registers on-the-fly. The load/stores to network accumulation buffers still use the full-size byte packs (i.e., fp32).
I tried to integrate this directly into
`prims_simple.h`
and
`reduce_scatter.h`
, but unfortunately, it is really hard to integrate multiple datatypes into the existing NCCL code preserving full generality. For example, the
`reduceCopyPacks()`
function that is used for actually reducing data takes an unknown number of source pointers and destination pointers, which are all assumed to have the same type; it's hard to come up with a generic way to assign different types to some of them.
So, I came up with
specialized primitives
, which are basically the primitives for the SIMPLE protocol, hardcoded for ring-based reduce-scatter and stripped-down as much as possible (e.g., as we only have Ampere GPUs, I took away everything related to multimem loads). As a result, this is obviously not upstreamable as is, but at least I hope it's easy to read the code.
I don't know how easy it would be to get something like this in upstream NCCL, but IMO allowing to choose different types for the input/output and the comms is something worth having in a communication library.
❤️
React with ❤️
2
Reacted by
Gaurav Kaul and Jangwoong Kim
stas00
commented
on
Jan 18, 2024
stas00
on Jan 18, 2024
Author
More actions
that's very cool, Ivan. I hope the owners of nccl will be able to make this part of the core!
also CC'ing
@tjruwase
- perhaps Ivan's version could be used in Deepspeed directly to solve the current bf16->fp32 comms overhead in reduce-scatter.
sjeaugey
commented
on
Jan 18, 2024
sjeaugey
on Jan 18, 2024
Member
More actions
So the allreduce time is that of an fp32 (2x slower than a bf16 allreduce) but the goal here is only to save intermediate buffers and conversion kernels. Makes sense.
I thought it would be easier to handle that in PyTorch so that the communication buffers would be defined as fp32 and when they are being written to, the conversion occurs without an extra kernel.
Arguably though it's simply fusing the conversion with the last kernel in PyTorch instead of the NCCL kernel, so it's not better or worse, just moving the problem elsewhere.
Thanks for the feedback and the rationale.
dfyz
commented
on
Jan 18, 2024
dfyz
on Jan 18, 2024
Last edited by
dfyz
More actions
@sjeaugey
If this can be done in PyTorch, it should definitely be done in PyTorch instead of NCCL, but I don't really see how it's possible without wasting memory. To be more specific: the scenario I'm thinking of is that you have a lot of large bfloat16 GEMMs (bfloat16 input, bfloat16 output), and you want to reduce their outputs
in-place
with fp32 precision. You can force the GEMMs to store their output directly in fp32 (so indeed, as you said, no extra conversion kernel is needed), but then you effectively double the memory you use for GEMM outputs, which is non-negligible.
It is primarily this doubling that I want to avoid. As far as I can tell, this can only be done in NCCL, because only the underlying NCCL kernel can load/store directly from the bfloat16 tensor directly without materializing the whole tensor in fp32 precision elsewhere. Is there any other way that would move this logic to PyTorch?
@stas00
Just to set the expectations straight -- unfortunately, my code doesn't really make communications any faster (as
@sjeaugey
mentioned in the previous comment). :/ The root problem is still there: you have to communicate in fp32 if you want fp32 precision for partial sums. My NCCL kernel only saves memory and compute for intermediate conversions, so it will only help Deepspeed if they are bottlenecked on that.
👍
React with 👍
2
Reacted by
Vadim Kantorov and Vadim Markovtsev
stas00
commented
on
Jan 18, 2024
stas00
on Jan 18, 2024
Author
More actions
Using less memory and compute is always wanted - as both are limited resources, especially memory. This is one of the main reasons flash attention is so popular - it saves memory and compute.
dfyz
commented
on
Feb 2, 2024
dfyz
on Feb 2, 2024
More actions
@sjeaugey
Is there a chance that something
like this
can be merged in upstream NCCL? The rough proof-of-concept I linked actually works surprisingly well for us, and I still think that an optimization like this can't be implemented in PyTorch (but please correct me if I'm wrong). Furthermore, the linked issue in NVIDIA/TransformerEngine indicates that an all-gather/reduce-scatter on FP8 data with FP16 comms/accumulators is also a needed feature, and I think it is trivial to add to my proof-of-concept as well.
I also think that conceptually, this change is quite simple: we only need minor changes to the code that loads/stores the user data. The current proof-of-concept is messy just because that NCCL currently assumes every collective uses just a single data type (which is definitely a reasonable assumption), so I had to write lots of new boilerplate for mixed-precision collectives, and also duplicate some existing code. If there is a more elegant way to do this, I don't mind spending more time to polish my proof-of-concept PR to make it upstreamable.
At the very least (if you think this feature is not worth the additional code complexity), I would like to make the proof-of-concept code available in an unofficial fork, so that other people can use it.
sjeaugey
commented
on
Feb 5, 2024
sjeaugey
on Feb 5, 2024
Member
More actions
@dfyz
it makes sense, and the major hurdle, as you mentioned, is the amount of code complexity it adds to NCCL. Having different datatypes across the code is going to make the code messier, and also cause higher maintenance costs. It could also cause more metadata to be transmitted all over the place which can have a performance cost.
If we could find an elegant way to have this impact only a small portion of the code, it would definitely make it more appealing.
CC
@jbachan
in case you see a way to integrate that the same way as pre-scaling somehow.
stas00
commented
on
Feb 22, 2024
stas00
on Feb 22, 2024
Author
More actions
@sjeaugey
, I was just reading about SHARP which performs reductions in-network and it supports FP32, FP16 and BF16 dtypes - do you know if it can be told to accumulate in higher precision or perhaps it does it already?
I'm aware that SHARP is not available on all servers.
Thank you!
stephenmsachs
commented
on
Jul 7, 2025
stephenmsachs
on Jul 7, 2025
Contributor
More actions
Issue Cleanup: Helping Us Focus on Current Challenges
We're
reviewing
older issues to ensure we prioritize the most relevant and active ones. Since this issue hasn't seen updates in over 6 months, we'll be closing it for now.
,
This change helps us focus our efforts on addressing any current issues our users are facing.
If this issue still affects you, please don't hesitate to reopen it with a quick update (e.g., "Still relevant on [version=X]").
Thanks for your understanding and for contributing to NCCL.
stephenmsachs
closed this
as
not planned
on Jul 7, 2025
jeffhammond
commented
on
Sep 18, 2026
jeffhammond
on Sep 18, 2026
Last edited by
jeffhammond
Member
More actions
@stas00
from
https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
:
This release introduces symmetric memory support ... Reductions are computed using FP32 accumulators (or FP16 for FP8 on NVLink Switch (NVLS) Systems), improving both accuracy and determinism in operations like AllReduce, AllGather, and ReduceScatter.
I am working on something along the lines of mixed-precision accumulation right now, which will hopefully land in
https://github.com/NVIDIA/nccl/tree/master/contrib
.
stas00
commented
on
Sep 18, 2026
stas00
on Sep 18, 2026
Author
More actions
that's fantastic, Jeff. Thank you for the heads up!
Sign up for free
to join this conversation on GitHub.
Already have an account?
Sign in to comment
Metadata
Metadata
Assignees
No one assigned
Labels
No labels
No labels
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