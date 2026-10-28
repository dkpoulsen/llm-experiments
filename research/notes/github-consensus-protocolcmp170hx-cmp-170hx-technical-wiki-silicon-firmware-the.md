---
title: 'GitHub - Consensus-Protocol/cmp170hx: CMP 170HX technical wiki: silicon, firmware,
  the community unlock, procedures and open problems. 55 pages. · GitHub'
id: github-consensus-protocolcmp170hx-cmp-170hx-technical-wiki-silicon-firmware-the
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- cmp170hx
- source-hub
- pcie-link
- unlock
- community-reported
created: '2026-10-27T10:52:17.452908Z'
updated: '2026-10-27T19:00:31.342456Z'
source: https://github.com/Consensus-Protocol/cmp170hx
source_domain: github.com
fetched_at: '2026-10-27T10:52:16.828711Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Consensus-Protocol/cmp170hx (74 stars, 5 commits, current as of 2026-07-31)
  — README of a 55-page, ~278,000-word CMP 170HX technical wiki; a SOURCE HUB rather
  than a citable source, whose docs/ tree is the source of truth (sections: start,
  hardware, unlock, procedures, operations, frontier, history, appendix). Built from
  two public Discord servers (26,123 messages, 2026-06-22 to 2026-07-28), 1,121 archived
  attachments, 17 branches of the unlock tool, adjudicated against shipping source
  code and checked by four independent audit methods; no individuals named, confidence
  marked throughout. Two claims the wiki insists on, both directly relevant to the
  query''s premises: (1) capacity is per SKU and not interchangeable — the 8 GB card
  (10de:20c2) unlocks to 64 GB, the 10 GB card (10de:2082) unlocks to 40 GB, and an
  80 GB configuration for 10 GB cards was built, tested and rejected as unstable;
  (2) PCIe link speed and link width are separate problems — Gen1 to Gen2 is a software
  unlock shipped in cmpunlocker master since 2026-07-29 so any unlocked card runs
  Gen2, but going beyond x4 width requires hand-soldering 24 AC coupling capacitors,
  and neither achieves the other. That is the cleanest statement in the corpus of
  why ''Gen2 x4'' is the default unlocked state and why width is a hardware ceiling,
  not a firmware one. The frontier/ section holds the unsolved problems (PCIe Gen3/Gen4,
  NVLink, ECC, 80 GB, P2P) — the highest-value pages for this run are docs/hardware/pcie.md,
  docs/frontier/*, docs/operations/performance.md and docs/appendix/methodology.md,
  none of which were fetched in this batch.'
utility_score: 15.0
---

[Skip to content](https://github.com/Consensus-Protocol/cmp170hx#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/Consensus-Protocol/cmp170hx) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/Consensus-Protocol/cmp170hx) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/Consensus-Protocol/cmp170hx) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/Consensus-Protocol/cmp170hx/branches)[](https://github.com/Consensus-Protocol/cmp170hx/tags)
Go to file
Open more actions menu
## Latest commit
[Merge pull request](https://github.com/Consensus-Protocol/cmp170hx/commit/d923a25fb2ecb6ee3d62654983f453194b66485f) [#1](https://github.com/Consensus-Protocol/cmp170hx/pull/1) [from everyun/zh-translation](https://github.com/Consensus-Protocol/cmp170hx/commit/d923a25fb2ecb6ee3d62654983f453194b66485f)
Open commit details
Aug 22, 2026
[d923a25](https://github.com/Consensus-Protocol/cmp170hx/commit/d923a25fb2ecb6ee3d62654983f453194b66485f) · Aug 22, 2026
## History
[5 Commits](https://github.com/Consensus-Protocol/cmp170hx/commits/main/)
Open commit details
5 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
| [Add full Simplified Chinese translation (docs/zh/)](https://github.com/Consensus-Protocol/cmp170hx/commit/44678053d76c720438fdc756b52fdd57d32488ac "Add full Simplified Chinese translation \(docs/zh/\)

- 55-page Chinese mirror of the wiki under docs/zh/, same paths and
  relative links as the English docs/, per translation guide at
  docs/zh/translation-guide.md
- mkdocs.zh.yml builds the Chinese site \(mkdocs build -f mkdocs.zh.yml\)
- README.zh-CN.md with build/reading instructions; README links to it

All pages preserve Markdown structure, code blocks, callouts, register
values and link targets; terminology unified via translation guide.")  | Aug 7, 2026  |  
|   |   | [CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open …](https://github.com/Consensus-Protocol/cmp170hx/commit/b5f6f56a2526e76a52341948e586db747da644fe "CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open problems

A technical reference for the NVIDIA CMP 170HX \(GA100\), a mining card built
on the same die as the A100 with compute and memory restricted in firmware.

Covers the hardware, the fuse and OTP survey across 15 Ampere cards, the
Falcon/Booter exploit chain that opens the privilege level masks, the memory
geometry and compute unlocks, PCIe Gen2, install and troubleshooting
procedures, cooling and power, LLM inference results, the unsolved problems,
and a long record of the dead ends so they are not repeated.

Sources: two public Discord servers \(26,123 messages, 2026-06-22 to
2026-07-28\), 1,121 archived attachments, 17 branches of the unlock tool, and
external references. Claims were extracted, adjudicated against shipping
source code, and checked by four independent audit methods.

No individual is named anywhere; findings are attributed to dates and
channels. Confidence is marked throughout, and unresolved questions are
labelled rather than guessed. See docs/appendix/methodology.md for how this
was built and what it cannot guarantee.

Co-Authored-By: Claude Opus 5 \(1M context\) <noreply@anthropic.com>")  | Jul 28, 2026  |  
|   |   | [CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open …](https://github.com/Consensus-Protocol/cmp170hx/commit/b5f6f56a2526e76a52341948e586db747da644fe "CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open problems

A technical reference for the NVIDIA CMP 170HX \(GA100\), a mining card built
on the same die as the A100 with compute and memory restricted in firmware.

Covers the hardware, the fuse and OTP survey across 15 Ampere cards, the
Falcon/Booter exploit chain that opens the privilege level masks, the memory
geometry and compute unlocks, PCIe Gen2, install and troubleshooting
procedures, cooling and power, LLM inference results, the unsolved problems,
and a long record of the dead ends so they are not repeated.

Sources: two public Discord servers \(26,123 messages, 2026-06-22 to
2026-07-28\), 1,121 archived attachments, 17 branches of the unlock tool, and
external references. Claims were extracted, adjudicated against shipping
source code, and checked by four independent audit methods.

No individual is named anywhere; findings are attributed to dates and
channels. Confidence is marked throughout, and unresolved questions are
labelled rather than guessed. See docs/appendix/methodology.md for how this
was built and what it cannot guarantee.

Co-Authored-By: Claude Opus 5 \(1M context\) <noreply@anthropic.com>")  | Jul 28, 2026  |  
|   |   | [Add full Simplified Chinese translation (docs/zh/)](https://github.com/Consensus-Protocol/cmp170hx/commit/44678053d76c720438fdc756b52fdd57d32488ac "Add full Simplified Chinese translation \(docs/zh/\)

- 55-page Chinese mirror of the wiki under docs/zh/, same paths and
  relative links as the English docs/, per translation guide at
  docs/zh/translation-guide.md
- mkdocs.zh.yml builds the Chinese site \(mkdocs build -f mkdocs.zh.yml\)
- README.zh-CN.md with build/reading instructions; README links to it

All pages preserve Markdown structure, code blocks, callouts, register
values and link targets; terminology unified via translation guide.")  | Aug 7, 2026  |  
|   |   | [Add full Simplified Chinese translation (docs/zh/)](https://github.com/Consensus-Protocol/cmp170hx/commit/44678053d76c720438fdc756b52fdd57d32488ac "Add full Simplified Chinese translation \(docs/zh/\)

- 55-page Chinese mirror of the wiki under docs/zh/, same paths and
  relative links as the English docs/, per translation guide at
  docs/zh/translation-guide.md
- mkdocs.zh.yml builds the Chinese site \(mkdocs build -f mkdocs.zh.yml\)
- README.zh-CN.md with build/reading instructions; README links to it

All pages preserve Markdown structure, code blocks, callouts, register
values and link targets; terminology unified via translation guide.")  | Aug 7, 2026  |  
|   |   | [CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open …](https://github.com/Consensus-Protocol/cmp170hx/commit/b5f6f56a2526e76a52341948e586db747da644fe "CMP 170HX wiki: 55 pages covering silicon, firmware, unlock and open problems

A technical reference for the NVIDIA CMP 170HX \(GA100\), a mining card built
on the same die as the A100 with compute and memory restricted in firmware.

Covers the hardware, the fuse and OTP survey across 15 Ampere cards, the
Falcon/Booter exploit chain that opens the privilege level masks, the memory
geometry and compute unlocks, PCIe Gen2, install and troubleshooting
procedures, cooling and power, LLM inference results, the unsolved problems,
and a long record of the dead ends so they are not repeated.

Sources: two public Discord servers \(26,123 messages, 2026-06-22 to
2026-07-28\), 1,121 archived attachments, 17 branches of the unlock tool, and
external references. Claims were extracted, adjudicated against shipping
source code, and checked by four independent audit methods.

No individual is named anywhere; findings are attributed to dates and
channels. Confidence is marked throughout, and unresolved questions are
labelled rather than guessed. See docs/appendix/methodology.md for how this
was built and what it cannot guarantee.

Co-Authored-By: Claude Opus 5 \(1M context\) <noreply@anthropic.com>")  | Jul 28, 2026  |  
|   |   | [Add full Simplified Chinese translation (docs/zh/)](https://github.com/Consensus-Protocol/cmp170hx/commit/44678053d76c720438fdc756b52fdd57d32488ac "Add full Simplified Chinese translation \(docs/zh/\)

- 55-page Chinese mirror of the wiki under docs/zh/, same paths and
  relative links as the English docs/, per translation guide at
  docs/zh/translation-guide.md
- mkdocs.zh.yml builds the Chinese site \(mkdocs build -f mkdocs.zh.yml\)
- README.zh-CN.md with build/reading instructions; README links to it

All pages preserve Markdown structure, code blocks, callouts, register
values and link targets; terminology unified via translation guide.")  | Aug 7, 2026  |  
| View all files |  
## Repository files navigation
# The CMP 170HX Wiki
**中文版：** [README.zh-CN.md](https://github.com/Consensus-Protocol/cmp170hx/blob/main/README.zh-CN.md) · [中文站点](https://github.com/Consensus-Protocol/cmp170hx/tree/main/docs/zh)
A comprehensive technical reference for the NVIDIA CMP 170HX (GA100): silicon, firmware, the community unlock, operating procedures, and the open frontier.
55 pages, roughly 278,000 words. Current as of **2026-07-31**.
## Two ways to read this
  * for browsing, with a sidebar. Pages there are generated from `docs/` and are identical apart from link style.
  * **`docs/`in this repository** is the source of truth. It is reviewable, takes pull requests, and builds the full themed site with search via MkDocs.


Keep the two in step. `to_github_wiki.py` publishes `docs/` to the wiki; if edits are made on the wiki instead, sync them back before continuing so the trees never diverge.
## Reading it
The pages are plain Markdown and read fine directly on GitHub or in any editor. Callouts are written as GitHub alert blockquotes (`> [!NOTE]`), which GitHub styles natively; MkDocs renders them as plain blockquotes unless a callout extension is enabled. To get search and navigation:

```
pip install mkdocs mkdocs-material
mkdocs serve          # http://127.0.0.1:8000
mkdocs build          # static site into ../site
```

## What is covered  
| Section  | Contents  |  
| --- | --- |  
| `start/`  | Orientation, card identification, quick start, risks, glossary  |  
| `hardware/`  | GA100 silicon, board variants, memory subsystem, fuses and OTP, PCIe, NVLink, power, thermals, VBIOS  |  
| `unlock/`  | The mechanism end to end: Falcon and Booter, the ROP chain, privilege level masks, memory geometry, compute throttle, driver patches, PCIe Gen2, full register reference  |  
| `procedures/`  | Install, verify, troubleshoot, recover, multi-GPU, driver versions, uninstall  |  
| `operations/`  | Cooling, power and PSUs, physical mods, performance, LLM inference, tuning  |  
| `frontier/`  | Status board and the unsolved problems: PCIe Gen3/Gen4, NVLink, ECC, 80 GB, P2P  |  
| `history/`  | Timeline, the clean-room and provenance question, dead ends, tool lineage  |  
| `appendix/`  | Register index, preserved artifacts, external sources, methodology  |  
## Two things this wiki insists on
**Capacity is per SKU and is not interchangeable.** The 8 GB card (`10de:20c2`) unlocks to **64 GB**. The 10 GB card (`10de:2082`) unlocks to **40 GB**. The 80 GB configuration for 10 GB cards was built, tested, and rejected as unstable.
**PCIe link speed and link width are separate problems.** Gen1 to Gen2 is a software unlock, shipped in cmpunlocker `master` since 2026-07-29, so any card with the unlock installed runs Gen2. Going beyond x4 width requires hand soldering 24 AC coupling capacitors. Neither one achieves the other.
## Conventions
Plain prose is confirmed fact. Experimental, dangerous and unsolved material is marked with alerts. Claims resting on a single observation say so in the sentence. Where evidence genuinely conflicts and nothing settles it, the wiki says so rather than choosing quietly.
No individual is named anywhere. Findings are attributed to dates and channels rather than to people. See `docs/appendix/methodology.md` for how the underlying claims were gathered, adjudicated and verified, and for an honest account of the limitations.
## About
CMP 170HX technical wiki: silicon, firmware, the community unlock, procedures and open problems. 55 pages.
### Resources
### Stars
**74** stars
### Watchers
**3** watching
### Forks
[ forks](https://github.com/Consensus-Protocol/cmp170hx/forks)
## 
No releases published
## 
  * 

You can’t perform that action at this time. 
