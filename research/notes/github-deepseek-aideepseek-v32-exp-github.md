---
title: GitHub - deepseek-ai/DeepSeek-V3.2-Exp · GitHub
id: github-deepseek-aideepseek-v32-exp-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- dsa
- github
- rope-gotcha
- implementation
created: '2026-10-27T10:42:16.844702Z'
updated: '2026-10-27T19:00:31.289169Z'
source: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
source_domain: github.com
fetched_at: '2026-10-27T10:42:16.223968Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'README for DeepSeek''s official DeepSeek-V3.2-Exp release (github.com/deepseek-ai/DeepSeek-V3.2-Exp,
  18 commits, last Nov 2025). Introduces DSA as an experimental intermediate step
  toward a next-generation architecture, framed as exploring and validating training
  and inference efficiency optimizations for long-context scenarios. States that DSA
  delivers substantial long-context training and inference efficiency gains ''while
  maintaining virtually identical model output quality'', and that the training configuration
  was deliberately aligned with V3.1-Terminus so the effect of sparse attention could
  be isolated; the benchmark table shows near-parity (MMLU-Pro 85.0 vs 85.0, GPQA-Diamond
  80.7 vs 79.9, AIME 2025 88.4 vs 89.3, SWE Verified 68.4 vs 67.8). The most useful
  item for a reproduction attempt is the 2025.11.17 update: previous inference demo
  code had a RoPE implementation discrepancy in the indexer module — the indexer''s
  RoPE input requires a non-interleaved layout while MLA''s RoPE expects an interleaved
  layout — which was fixed. Thin source (770 words); the substantive training recipe
  lives in the V3.2 technical report, not here.'
---

*Suggested by [[longcat-sparse-attention-taming-the-lightning-via]] — cited as the DSA technical report with the dense-warmup plus sparse-training recipe*

[Skip to content](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/branches)[](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tags)
Go to file
Open more actions menu
## Latest commit
Nov 18, 2025
[87e509a](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/commit/87e509a2e5a100d221c97df52c6e8be7835f0057) · Nov 18, 2025
## History
[18 Commits](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/commits/main/)
Open commit details
18 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   |   | Nov 18, 2025  |  
|   |   |   | Sep 29, 2025  |  
|   |   |   | Sep 29, 2025  |  
|   |   |   | Nov 18, 2025  |  
|   |   |   | Sep 29, 2025  |  
| View all files |  
## Repository files navigation
# DeepSeek-V3.2-Exp
## Introduction
We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.
This experimental release represents our ongoing research into more efficient transformer architectures, particularly focusing on improving computational efficiency when processing extended text sequences.
  * DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time, delivering substantial improvements in long-context training and inference efficiency while maintaining virtually identical model output quality.
  * To rigorously evaluate the impact of introducing sparse attention, we deliberately aligned the training configurations of DeepSeek-V3.2-Exp with V3.1-Terminus. Across public benchmarks in various domains, DeepSeek-V3.2-Exp demonstrates performance on par with V3.1-Terminus.

  
| Benchmark  | DeepSeek-V3.1-Terminus  | DeepSeek-V3.2-Exp  |  
| --- | --- | --- |  
| **Reasoning Mode w/o Tool Use**  |  
| MMLU-Pro  | 85.0  | 85.0  |  
| GPQA-Diamond  | 80.7  | 79.9  |  
| Humanity's Last Exam  | 21.7  | 19.8  |  
| LiveCodeBench  | 74.9  | 74.1  |  
| AIME 2025  | 88.4  | 89.3  |  
| HMMT 2025  | 86.1  | 83.6  |  
| Codeforces  | 2046  | 2121  |  
| Aider-Polyglot  | 76.1  | 74.5  |  
| **Agentic Tool Use**  |  
| BrowseComp  | 38.5  | 40.1  |  
| BrowseComp-zh  | 45.0  | 47.9  |  
| SimpleQA  | 96.8  | 97.1  |  
| SWE Verified  | 68.4  | 67.8  |  
| SWE-bench Multilingual  | 57.8  | 57.9  |  
| Terminal-bench  | 36.7  | 37.7  |  
## Update
  * 2025.11.17: **We have identified that previous versions of the inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded model performance.** Specifically, the input tensor to RoPE in the indexer module requires a non-interleaved layout, whereas RoPE in the MLA module expects an interleaved layout. This issue has now been resolved. Please refer to the updated version of the inference demo code and take note of this implementation detail.


## Open-Source Kernels
For TileLang kernels with **better readability and research-purpose design** , please refer to [TileLang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32).
For **high-performance CUDA kernels** , indexer logit kernels (including paged versions) are available in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/200). Sparse attention kernels are released in [FlashMLA](https://github.com/deepseek-ai/FlashMLA/pull/98).
## How to Run Locally
### HuggingFace
We provide an updated inference demo code in the [inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) folder to help the community quickly get started with our model and understand its architectural details.
First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:

```
cd inference
export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:

```
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

### SGLang
#### Installation with Docker

```
# H200
docker pull lmsysorg/sglang:dsv32

# MI350
docker pull lmsysorg/sglang:dsv32-rocm

# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3

```

#### Launch Command

```
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

vLLM provides day-0 support of DeepSeek-V3.2-Exp. See the [recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html) for up-to-date details.
## License
This repository and the model weights are licensed under the [MIT License](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/LICENSE).
## Citation

```
@misc{deepseekai2024deepseekv32,
      title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
      author={DeepSeek-AI},
      year={2025},
}

```

## Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/service@deepseek.com).
## About
No description, website, or topics provided.
### Resources
### Stars
**1.7k** stars
### Watchers
**10** watching
### Forks
[ forks](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/forks)
## 
No releases published
## 
## Languages
  * [Python100%](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/search?l=python)


You can’t perform that action at this time. 
