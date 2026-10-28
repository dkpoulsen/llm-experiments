---
title: 'Why is the bandwidth utilization of 7 GPUs much lower than that of 6, 8 GPUs
  for tasks with a specific total parameter amount? · Issue #97 · NVIDIA/nccl-tests
  · GitHub'
id: why-is-the-bandwidth-utilization-of-7-gpus-much-lower-than-that-of-6-8-gpus-for
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-2
- nccl
- 5-gpu-penalty
- alignment
- expert-opinion
- community-reported
created: '2026-10-27T14:36:29.452970Z'
updated: '2026-10-27T19:00:31.781519Z'
source: https://github.com/NVIDIA/nccl-tests/issues/97
source_domain: github.com
fetched_at: '2026-10-27T14:36:29.452091Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'NVIDIA/nccl-tests issue #97 (Nov 2021) — the only authoritative statement
  found that odd GPU counts are pathological, and the mechanism behind it. [T2 expert-opinion
  from an NCCL maintainer.] Reporter measured all-gather at a fixed total parameter
  count (10,660,224) across 4, 5, 6, 7 and 8 GPUs and found ''the bandwidth utilization
  of 5, 7 GPUs will be much lower than others''. NCCL maintainer sjeaugey: ''With
  5 or 7 GPUs the alignment of the offsets is probably not great (e.g. 4 or 8 bytes
  instead of 16 bytes, or even down to 1 byte if using an int8 datatype).'' This corroborates
  the Tartan/TPDS warning that 5-GPU PCIe rings are pathological, but supplies NO
  GB/s figure — it is a mechanism confirmation, not a measurement, and it does not
  distinguish a 5-rank penalty of 1.6x (ring-load factor) from the 3-5x the cc-2 falsifier
  requires. It also implies the penalty is payload-shape dependent: a 1M-token global
  batch''s gradient bucket sizes are chosen by the framework, so alignment can be
  engineered away (pad buckets to 16-byte multiples) in a way that a fixed-total-parameter
  all-gather cannot.'
utility_score: 16.0
---

Why is the bandwidth utilization of 7 GPUs much lower than that of 6, 8 GPUs for tasks with a specific total parameter amount? · Issue #97 · NVIDIA/nccl-tests · GitHub
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
Why is the bandwidth utilization of 7 GPUs much lower than that of 6, 8 GPUs for tasks with a specific total parameter amount?
#97
New issue
Copy link
New issue
Copy link
Open
Open
Why is the bandwidth utilization of 7 GPUs much lower than that of 6, 8 GPUs for tasks with a specific total parameter amount?
#97
Copy link
Description
Youhe-Jiang
opened
on Nov 23, 2021
Issue body actions
Hello, I’m Jiang Youhe, a graduate student from Cornell University. I’m still in the learning stage, so I’m here to ask a question. I recently tested the performance of the allgather operator and found a strange phenomenon. Given a total parameter value of 10660224, respectively using 4, 5, 6, 7, 8 GPUs for experiments, the bandwidth utilization of 5, 7 GPUs will be much lower than others. Is there any academic explanation? Thanks so much for any answers!
Reactions are currently unavailable
Activity
sjeaugey
commented
on
Nov 23, 2021
sjeaugey
on Nov 23, 2021
Member
More actions
With 5 or 7 GPUs the alignment of the offsets is probably not great (e.g. 4 or 8 bytes instead of 16 bytes, or even down to 1 byte if using a int8 datatype).
That means NCCL cannot use optimized 16 bytes loads and stores and performance is lower.
If you run again on a size which is a multiple of 16×5 or 16×7 you should see good performance.
Youhe-Jiang
commented
on
Nov 23, 2021
Youhe-Jiang
on Nov 23, 2021
Author
More actions
sjeaugey
Thank you for your quick and clear reply! It seems that this is the second time you have answered my question, thanks again!
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