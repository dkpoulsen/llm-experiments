---
title: 'Missing sparse backward kernel. · Issue #107 · deepseek-ai/FlashMLA · GitHub'
id: missing-sparse-backward-kernel-issue-107-deepseek-aiflashmla-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:40:17.071603Z'
updated: '2026-10-27T19:00:31.810567Z'
source: https://github.com/deepseek-ai/FlashMLA/issues/107
source_domain: github.com
fetched_at: '2026-10-27T14:40:17.070632Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'deepseek-ai/FlashMLA issue #107 ''Missing sparse backward kernel'' (opened
  2025-10-09, still open). The reporter states the library ''provides a complete forward
  and backward pass for dense attention, but the sparse attention implementation appears
  to be forward-only'' and asks whether a sparse backward will be released. No maintainer
  answer; a community reply argues a sparse backward ''would take MUCH MUCH more amount
  of time to implement than using dense backwards'' because it needs atomic accumulation
  over misaligned K/V. Confirms from the upstream project''s own tracker that no sparse-MLA
  backward exists in FlashMLA — the only sm_80-adjacent DSA forward has no training
  counterpart.'
---

Missing sparse backward kernel. · Issue #107 · deepseek-ai/FlashMLA · GitHub
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
Missing sparse backward kernel.
#107
New issue
Copy link
New issue
Copy link
Open
Open
Missing sparse backward kernel.
#107
Copy link
Description
EricLina
opened
on Oct 9, 2025
Issue body actions
Thank you for the excellent work on the highly optimized attention kernels in FlashMLA. The performance of the sparse forward kernel
`flash_mla_sparse_fwd`
for the prefill stage is particularly impressive.
I've noticed that while the library provides a complete forward and backward pass for dense attention, but the sparse attention implementation appears to be forward-only.
Will the sparse attention backward kernel be released
?
Thanks.
Reactions are currently unavailable
Activity
sunying2018
commented
on
Oct 21, 2025
sunying2018
on Oct 21, 2025
More actions
same question, thanks!
IISuperluminaLII
commented
on
Oct 26, 2025
IISuperluminaLII
on Oct 26, 2025
More actions
This would take MUCH MUCH more amount of time to implement than using dense backwards. Why? the main point is to use CUTLASS gemm optimizations. Additionally sparse backwards needs atomic accumulation which is quite difficult to impediment (Thus CUTLASS issue)
If the Qs and Ks are aligned then sure, not the case for SPARSE backwards right... so since we are dealing with "misaligned" Ks and Vs the wall-time is exponentially if not MUCH larger(STRICTLY SPEAKING ON SPARSE COMPUTATIONS)
IISuperluminaLII
commented
on
Oct 26, 2025
IISuperluminaLII
on Oct 26, 2025
Last edited by
IISuperluminaLII
More actions
I mean CUTLASS is the reason they probably don't ship windows whl or source with windows links...
If I had a PhD and free time (2-6 months + 2 years of cuda classes) then maybe I would consider implementing it, but I have neither. I was just curious if anyone had tried it lol
😄
React with 😄
1
Reacted by
Elinx
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