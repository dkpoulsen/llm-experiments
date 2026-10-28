---
title: 'What''s the difference in allreduce for factor between the NCCL and NCCL-test?
  · Issue #274 · NVIDIA/nccl-tests · GitHub'
id: whats-the-difference-in-allreduce-for-factor-between-the-nccl-and-nccl-test-issu
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:18:31.948905Z'
updated: '2026-10-27T19:00:31.668114Z'
source: https://github.com/NVIDIA/nccl-tests/issues/274
source_domain: github.com
fetched_at: '2026-10-27T13:18:31.948165Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'What''s the difference in allreduce for factor between the NCCL and NCCL-test?
  · Issue #274 · NVIDIA/nccl-tests · GitHub'
utility_score: 15.0
---

What's the difference in allreduce for factor between the NCCL and NCCL-test? · Issue #274 · NVIDIA/nccl-tests · GitHub
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
What's the difference in allreduce for factor between the NCCL and NCCL-test?
#274
New issue
Copy link
New issue
Copy link
Open
Open
What's the difference in allreduce for factor between the NCCL and NCCL-test?
#274
Copy link
Description
networkResearcher
opened
on Dec 9, 2024
Issue body actions
I find there is difference for this change factor between the NCCL and NCCL-test.
For allreduce in NCCL:
If the algorithm is ring,
`busBw = 2(n-1)/n * algBw`
, which is same as NCCL-test.
If the algorithm is tree,
`busBw = 2 * algBw`
, which is different with NCCL-test.
How can I understand this question?
Reactions are currently unavailable
Activity
sjeaugey
commented
on
Dec 9, 2024
sjeaugey
on Dec 9, 2024
Member
More actions
The AlgBw->BusBW is theoretical, on a flat hardware topology, using point-to-point transfers to execute the operation. NCCL perf tests is a benchmark, it cannot see inside NCCL and it does not detect the topology nor the hardware. So the BusBW is a simple re-calibration of the algorithm bandwidth to reflect the fact that we need to transmit a different amount of data depending on the number of ranks.
If BusBW peaks at say 50GB/s, it's telling you that your system is equivalent to a system where each GPU would be connected to a flat fabric at 50GB/s.
Tree is not bandwidth optimal so the real bus bandwidth may be higher in some places. Same if your hardware is hierarchical, if you're using NVLink SHARP or IB SHARP (as those are not based on point-to-point communication but on HW-offloaded collectives instead).
But you can still use the BusBW to compare between algorithms and see which one is faster.
networkResearcher
commented
on
Dec 9, 2024
networkResearcher
on Dec 9, 2024
Author
More actions
Thank you very much !
We are trying to think which algorithm to choose for running AllReduce within a single machine. It seems that using the conversion formula between algorithm bandwidth and bus bandwidth is not correct for making this decision. Could you tell us which algorithm to choose for running AllReduce within a single machine, both with and without Nvlink?
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