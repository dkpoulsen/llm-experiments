---
title: 'GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context
  LLMs · GitHub'
id: github-moonshotaimoba-moba-mixture-of-block-attention-for-long-context-llms-gith
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-csa2-at-32k-sm80
- long-context
- sparse-attention
- all-to-all
- 32k-training
- code
created: '2026-10-27T13:42:06.182800Z'
updated: '2026-10-27T19:01:46.336447Z'
source: https://github.com/MoonshotAI/MoBA
source_domain: github.com
fetched_at: '2026-10-27T13:42:05.547948Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: docs
deprecated: false
summary: You signed in with another tab or window. Reload to refresh your session.
  You signed out in another tab or window. Re...
---

[Skip to content](https://github.com/MoonshotAI/MoBA#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/MoonshotAI/MoBA) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/MoonshotAI/MoBA) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/MoonshotAI/MoBA) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/MoonshotAI/MoBA/branches)[](https://github.com/MoonshotAI/MoBA/tags)
Go to file
Open more actions menu
## Latest commit
luenzhe
Apr 3, 2025
[b5d5836](https://github.com/MoonshotAI/MoBA/commit/b5d58363311d3ca946f1ec444182727c15e338b5) · Apr 3, 2025
## History
[13 Commits](https://github.com/MoonshotAI/MoBA/commits/master/)
Open commit details
13 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   |   | Feb 18, 2025  |  
|   |   | [fix figures and update readme](https://github.com/MoonshotAI/MoBA/commit/857393dc5ba8e3417c9c337bb34e1354738f9d13 "fix figures and update readme")  | Feb 22, 2025  |  
| [remove duplicate filtered_chunk_indices](https://github.com/MoonshotAI/MoBA/commit/61e456bc956c5a25fd9c84e5496b661329cb1b72 "remove duplicate filtered_chunk_indices")  | Mar 7, 2025  |  
|   |   |   | Mar 7, 2025  |  
|   |   |   | Feb 18, 2025  |  
|   |   |   | Feb 18, 2025  |  
|   |   |   | Apr 3, 2025  |  
|   |   |   | Feb 22, 2025  |  
|   |   |   | Feb 18, 2025  |  
|   |   | [add torch to requirements](https://github.com/MoonshotAI/MoBA/commit/13e6cf37c6ec1042c497bc0bedf4f8dc0ab0efbd "add torch to requirements")  | Feb 19, 2025  |  
| View all files |  
## Repository files navigation
# MoBA: Mixture of Block Attention for Long-Context LLMs
[**Full Report**](https://github.com/MoonshotAI/MoBA/blob/master/MoBA_Tech_Report.pdf)
🚀 Introducing **MoBA --- Mixture of Block Attention**
  * **Trainable Block Sparse Attention** : The full context is divided into blocks, where each query token learns to attend to the most relevant KV blocks, enabling efficient processing of long sequences.
  * **Parameter-less Gating Mechanism** : A novel Parameter-less top-k gating mechanism is introduced to selects the most relevant blocks for each query token, ensuring that the model focuses only on the most informative blocks.
  * **Seamlessly Transition between Full and Sparse Attention** : MoBA is designed to be a flexible substitute for full attention, allowing seamless transitions between full and sparse attention modes.


> **Note** : MoBA requires continue training of existing models to achieve its acceleration benefits. It is not a drop-in sparse attention solution that can be directly applied to pretrained models without additional training.
## Abstract
Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored.
In this work, we propose a solution that adheres to the **“less structure”** principle, allowing the model to autonomously determine where to attend, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi’s long-context requests and demonstrates significant advancements in efficient attention computation for LLMs.
Our code is available at [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA).
### Evaluation with 1M context length
## Environment Setup
**Note that current kernel implementations rely on`flash-attn==2.6.3` and `torch >= 2.1.0`**

```
conda create -n moba python=3.10
conda activate moba
pip install .
```

## Quick Start
We provide a transformers-friendly implementation for MoBA.
Feel free to choose attention backends by `--attn` between `moba` and `moba_naive`.

```
python3 examples/llama.py --model meta-llama/Llama-3.1-8B --attn moba
```

### Implementation Details
  * **moba_naive** : A naive implementation based on attention masks. It's designed to help understand how MoBA selects corresponding chunks. You may save and visualize the attention masks to see the block selection process.
  * **moba_efficient** : Our production-ready implementation optimized for performance. It achieves up to 40x speedup compared to moba_naive (tested with 32K sequence length, 1 attention head, MoBA Block 2048 and MoBA Topk 3). We recommend using this version for practical applications.


## Unit Tests

```
pytest tests/test_moba_attn.py
```

## References
  * Llama Implementation: [huggingface/transformers](https://github.com/huggingface/transformers)
  * Flash Attention: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)


## Citation
If you find MoBA is useful or want to use in your projects, please kindly cite our paper:

```
@article{lu2025mobamixtureblockattention,
  author = {Enzhe Lu and Zhejun Jiang and Jingyuan Liu and Yulun Du and Tao Jiang and Chao Hong and Shaowei Liu and Weiran He and Enming Yuan and Yuzhi Wang and Zhiqi Huang and Huan Yuan and Suting Xu and Xinran Xu and Guokun Lai and Yanru Chen and Huabin Zheng and Junjie Yan and Jianlin Su and Yuxin Wu and Yutao Zhang and Zhilin Yang and Xinyu Zhou and Mingxing Zhang and Jiezhong Qiu},
  title = {MoBA: Mixture of Block Attention for Long-Context LLMs},
  journal={arXiv preprint arXiv:2502.13189},
  year={2025}
}

```

## About
MoBA: Mixture of Block Attention for Long-Context LLMs
### Topics
[flash-attention](https://github.com/topics/flash-attention)[llm](https://github.com/topics/llm)[llm-serving](https://github.com/topics/llm-serving)[llm-training](https://github.com/topics/llm-training)[moe](https://github.com/topics/moe)[pytorch](https://github.com/topics/pytorch)[transformer](https://github.com/topics/transformer)
### Resources
### Stars
**2.2k** stars
### Watchers
**30** watching
### Forks
[ forks](https://github.com/MoonshotAI/MoBA/forks)
## 
No releases published
## 
  * [**hewr2010** Wayne-Ho](https://github.com/hewr2010)
  * [**eltociear** Ikko Eltociear Ashimine](https://github.com/eltociear)


## Languages
  * [Python100%](https://github.com/MoonshotAI/MoBA/search?l=python)


You can’t perform that action at this time.

## Related

- [[moba-mixture-of-block-attention-for]]
