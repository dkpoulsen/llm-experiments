---
title: 'GitHub - deepseek-ai/Engram: Conditional Memory via Scalable Lookup: A New
  Axis of Sparsity for Large Language Models · GitHub'
id: github-deepseek-aiengram-conditional-memory-via-scalable-lookup-a-new-axis-of-sp
tags:
- cmp170hx-training-feasibility-a0d4fc
- engram
- code
- reproducibility
- no-distributed-training
created: '2026-10-27T10:31:33.889157Z'
updated: '2026-10-27T19:00:31.203981Z'
source: https://github.com/deepseek-ai/Engram
source_domain: github.com
fetched_at: '2026-10-27T10:31:33.266567Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Official deepseek-ai/Engram repository (4 commits, latest Jan 14 2026, 4.8k
  stars). Contains the paper PDF, the architecture drawio, and a single standalone
  demo script; the README explicitly labels the code a demonstration that ''mocks
  standard components (like Attention/MoE/mHC) to focus on the Engram module''. CRITICAL
  NEGATIVE FINDING for training-feasibility work: the shipped implementation has no
  distributed training support at all, so the paper''s All-to-All table sharding and
  gradient dispatch path cannot be inspected or reproduced from public code. The repo
  confirms the system-efficiency claim is scoped to inference (''enabling the offloading
  of massive embedding tables to host memory with minimal inference overhead'').'
---

*Suggested by [[conditional-memory-via-scalable-lookup]] — Official Engram code release cited in the paper abstract; primary source for whether memory tables are frozen and how training sharding works*

[Skip to content](https://github.com/deepseek-ai/Engram#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/deepseek-ai/Engram) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/deepseek-ai/Engram) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/deepseek-ai/Engram) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/deepseek-ai/Engram/branches)[](https://github.com/deepseek-ai/Engram/tags)
Go to file
Open more actions menu
## Latest commit
Open commit details
Jan 14, 2026
[fb7f84a](https://github.com/deepseek-ai/Engram/commit/fb7f84a21f91223715394a33a1dc24bbfb7f788e) · Jan 14, 2026
## History
[4 Commits](https://github.com/deepseek-ai/Engram/commits/main/)
Open commit details
4 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   | Jan 12, 2026  |  
|   |   |   | Jan 12, 2026  |  
|   |   | Jan 12, 2026  |  
|   |   | Jan 12, 2026  |  
|   |   | Jan 12, 2026  |  
|   |   | [fix: 修复注释拼写错误 (third-class -> third-party)](https://github.com/deepseek-ai/Engram/commit/a31cd4d08cd293612d7803a2ec247af924a38fac "fix: 修复注释拼写错误 \(third-class -> third-party\)")  | Jan 13, 2026  |  
| View all files |  
## Repository files navigation
## 1. Introduction
This repository contains the official implementation for the paper: **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf)**.
> **Abstract:** While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup. To address this, we explore **conditional memory** as a complementary sparsity axis, instantiated via **Engram** , a module that modernizes classic -gram embeddings for  ( 1 ) lookup.
**Key Contributions:**
  * **Sparsity Allocation:** We formulate the trade-off between neural computation (MoE) and static memory (Engram), identifying a U-shaped scaling law that guides optimal capacity allocation.
  * **Empirical Verification:** Under strict iso-parameter and iso-FLOPs constraints, the Engram-27B model demonstrates consistent improvements over MoE baselines across knowledge, reasoning, code and math domains.
  * **Mechanistic Analysis:** Our analysis suggests that Engram relieves early layers from static pattern reconstruction, potentially preserving effective depth for complex reasoning.
  * **System Efficiency:** The module employs deterministic addressing, enabling the offloading of massive embedding tables to host memory with minimal inference overhead.


## 2. Architecture
The Engram module augments the backbone by retrieving static -gram memory and fusing it with dynamic hidden states. The architecture is shown below ([drawio provided](https://github.com/deepseek-ai/Engram/blob/main/drawio/Engram.drawio)):
## 3. Evaluation
### Scaling Law
### Large Scale Pre-training
### Long-context Training
## 4. Case Study of Engram
## 5. Quick Start
We recommend using Python 3.8+ and PyTorch.

```
pip install torch numpy transformers sympy
```

We provide a standalone implementation to demonstrate the core logic of the Engram module:

```
python engram_demo_v1.py
```

> ⚠️ **Note:** The provided code is a demonstration version intended to illustrate the data flow. It mocks standard components (like Attention/MoE/mHC) to focus on the Engram module.
## 6. License
The use of Engram models is subject to [the Model License](https://github.com/deepseek-ai/Engram/blob/main/LICENSE).
## 7. Contact
If you have any questions, please raise an issue or contact us at service@deepseek.com.
## About
Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
### Resources
### Stars
**4.8k** stars
### Watchers
**42** watching
### Forks
[ forks](https://github.com/deepseek-ai/Engram/forks)
## 
No releases published
## 
  *   * [**study8677** JingWen Fan](https://github.com/study8677)


## Languages
  * [Python100%](https://github.com/deepseek-ai/Engram/search?l=python)


You can’t perform that action at this time. 
