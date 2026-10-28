---
title: 'LISA: Linear-Indexed Sparse Attention for Efficient'
id: lisa-linear-indexed-sparse-attention-for-efficient
tags:
- cmp170hx-training-feasibility-a0d4fc
- sparse-attention
- retrofit
- distillation
- linear-attention
- training-cost
created: '2026-10-27T10:12:14.338968Z'
updated: '2026-10-27T19:00:31.122574Z'
source: https://arxiv.org/abs/2607.19358
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:14.337915Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'LISA (Zhao et al., Alibaba International Digital Commerce + Dalian University
  of Technology + Amazon, arXiv 2607.19358) is a plug-and-play attention replacement
  that requires NO pretraining from scratch: a Linear Attention branch (O(n) time,
  long-range memory) runs in parallel with a Lightning Indexer that selects the top-M
  tokens for a Sparse Self-Attention, the two branches fused by a gating mechanism,
  reducing inference complexity from O(n^2) to O(nM) for generating n tokens.


  Two-stage training: Stage 1 cold-starts the linear attention plus a sliding-window
  attention mechanism, optimized by knowledge distillation to approximate the full
  self-attention distribution of a frozen teacher; Stage 2 introduces the Indexer
  to replace the static sliding window, trained with a per-head KL divergence loss
  aligning its selection behaviour with the teacher''s attention patterns. All original
  model parameters are frozen — only the newly added linear-attention and Indexer
  modules train. Because top-M is argmax, gradients cannot flow back through the selection;
  early in training they select top-1.2M candidates and randomly mask out 20%, gradually
  reducing the dropout ratio, and apply z-loss to the Indexer.


  Budget analysis: MASS (attention mass captured) reaches 80% at M=256, only 6% below
  the theoretical oracle upper bound; 70% at M=128 and 88% at M=512, so M=256 is the
  recommended balance. Results on DeepSeek-distilled Qwen2.5-1.5B/7B: 50% inference
  speedup at 16K context with +5.6% average reasoning accuracy; on Qwen2.5-7B, 70.6%
  average accuracy vs 263.3 -> 226.6 relative latency measured on a single A100 (batch
  1, BF16, eager mode). Training cost: each of the two stages takes approximately
  240 GPU-hours on an NVIDIA H100 (~480 GPU-hours total) for the 7B retrofit, with
  Stage 1 at LR 2e-5, batch 128, 2 epochs, 50-step warmup. 10,747 words.'
raw_file: raw/lisa-linear-indexed-sparse-attention-for-efficient.pdf
doi: arXiv:2607.19358
utility_score: 12.0
citation_count: 0
venue: arXiv.org
---

LISA: Linear-Indexed Sparse Attention for Efficient
Long-Context Reasoning
Yu Zhao1∗, Zekun Zhang2∗, Fan Jiang3, Bo Zeng1, Linlong Xu1, Shimin Shan2,
Yu Liu2†, Longyue Wang1†, Weihua Luo1
1 Alibaba International Digital Commerce
2 School of Software Technology, Dalian University of Technology
3 Amazon
Abstract
Recent advances in long chain-of-thought reasoning models such as DeepSeek-
R1 have led to increasingly longer inference context lengths under the test-time
scaling paradigm. However, the O(n2) computational complexity of standard
self-attention causes inference costs to grow sharply with long sequences, limiting
the deployment of long-CoT reasoning in production settings. To address this,
we propose LISA (Linear-Indexed Sparse Attention), a plug-and-play attention
replacement module that requires no pretraining from scratch. LISA integrates
two lightweight components in parallel within the original model: (1) a Linear
Attention module that provides long-range memory with O(n) time complexity; (2)
a Lightning Indexer that selects the top-M important tokens from the full context
to feed into a Sparse Self-Attention. The two branches are fused via a gating
mechanism, reducing inference complexity from O(n2) to O(nM) (M ≪n) for
generating n tokens. We design a two-stage training pipeline: Stage 1 initializes
the model by integrating the linear attention to capture long-range dependencies,
complemented by a sliding-window attention mechanism that is optimized via
knowledge distillation to approximate the full self-attention distribution of a frozen
teacher model. In Stage 2, we further introduce the Indexer to replace the static
sliding-window mechanism, enabling dynamic token selection from broader con-
texts. The Indexer is trained using a novel per-head KL divergence loss, which
aligns its selection behavior with the attention patterns of the teacher model. Exper-
iments on DeepSeek-distilled-Qwen models demonstrate that LISA achieves a 50%
inference speedup under 16K-token context, while improving average performance
by 5.6% on reasoning benchmarks including AIME and MATH-500.
1
Introduction
Chain-of-Thought (CoT) [Wei et al., 2022, Kojima et al., 2022] has become a core technique for
enhancing the reasoning ability of large language models (LLMs) on complex tasks. By prompting
step-by-step reasoning, CoT enables LLMs to decompose complex problems into simpler subtasks,
thus improving their problem-solving capabilities [Yao et al., 2023, Wang et al., 2023, Zhou et al.,
2023]. Recent studies, including OpenAI o1 [OpenAI et al., 2026],DeepSeek-R1 [DeepSeek et al.,
2025a] and other reasoning model [Qwen Team, 2024, Zhao et al., 2024], and demonstrate that
scaling up CoT length can further enhance the reasoning abilities of LLMs.
However, since most current LLMs are built upon Transformer architecture [Vaswani et al., 2017],
the computational complexity of their attention grows quadratically with context length, and the
∗Equal contribution.
†Corresponding authors.
Preprint.
arXiv:2607.19358v1  [cs.AI]  29 May 2026


---

memory overhead of their KV-cache increases linearly. Hence, generating long CoT substantially
increases the computational and memory cost of LLMs, limiting their practical efficiency on complex
reasoning tasks.
To improve the reasoning efficiency of LLMs, prior work employs prompting [Han et al., 2024, Ma
et al., 2025a], supervised fine-tuning (SFT) [Liu et al., 2024, Munkhbat et al., 2025], or reinforcement
learning (RL) [Aggarwal and Welleck, 2025, Shen et al., 2025] to encourage LLMs to generate
shorter CoT sequences. However, these methods often impair reasoning ability [Jin et al., 2024,
Merrill and Sabharwal, 2024], since CoT shortening conflicts with test-time scaling [OpenAI et al.,
2026]. To preserve reasoning capability, some studies [Ma et al., 2025b, Kang et al., 2025, Xia et al.,
2025] express CoT in more concise text (e.g., by removing less important tokens) to reduce its length.
However, they risk losing critical reasoning information when simplifying long CoT [Wang et al.,
2025]. Additionally, Yan et al. [2025] proposed a state compression method based on linear attention,
but it requires complex data preprocessing and the inherently weak contextual indexing of linear
attention significantly degrades performance in context-sensitive tasks.
In this paper, we propose an efficient hybrid inference framework LISA (Linear-Indexed Sparse
Attention). We replace the standard self-attention with a hybrid architecture: a global linear attention
vector maintains long-range context state, while a fixed-size self-attention module preserves precise
contextual retrieval capability. This design makes the per-step computation cost constant (independent
of sequence length) while retaining precise access to critical context. Our framework requires no
complex data preprocessing, does not alter the original model parameters or output format, and
achieves plug-and-play improvement in both inference efficiency and quality.
Our key contributions are as follows:
1. We propose a plug-and-play hybrid attention module (LISA) that combines global linear
attention with fixed-size self-attention, maintaining constant inference cost while preserving
precise contextual retrieval.
2. We design a two-stage training strategy: the first stage cold-starts linear attention via pure
cross-entropy loss to avoid overfitting to the teacher distribution; the second stage introduces
an indexer module with top-M dynamic selection, augmented with KL loss to ensure stable
important-token selection.
3. We conduct extensive experiments on five mathematical reasoning benchmarks, demon-
strating that LISA not only reduces inference cost by up to 50% at 16K context but also
improves average accuracy by 5.6% over the full self-attention baseline.
2
Related Work
Efficient Reasoning Methods.
A growing body of work seeks to improve the efficiency of CoT
reasoning. One line of research uses prompt engineering, SFT, or RL to encourage models to generate
shorter CoT sequences [Nayab et al., 2024, Aggarwal and Welleck, 2025, Munkhbat et al., 2025].
However, these methods often compromise reasoning capability, as shortening CoT conflicts with
test-time scaling principles. Alternative approaches aim to express CoT more compactly without
losing information. LightThinker [Zhang et al., 2025] proposes a “thinking step-by-step compression”
paradigm that inserts learnable special tokens into hidden state sequences and compresses the
KV states of completed reasoning steps before each generation step. INFTYTHINK [Yan et al.,
2025] compresses the processed lengthy context and prior reasoning trajectories into compact
summary representations at each step, freeing up context window capacity for subsequent reasoning.
Additionally, TokenSkip [Xia et al., 2025] achieves controllable CoT compression through a learnable
token-skipping mechanism, and R1-Compress [Wang et al., 2025] explores long CoT compression
via chunk-level compression and search-based optimization. Zhang et al. [2026] maintain a global
state matrix via linear attention to record historical information, enabling the model to retrieve
relevant information without explicitly attending to prior reasoning steps. Our work differs from
these methods in that we do not compress the output CoT itself but rather accelerate the internal
inference process while preserving the full reasoning capability.
KV-Cache Reduction Methods.
Reducing KV-cache size is another major direction for efficient
LLM inference. H2O [Zhang et al., 2023] proposes a layer-wise eviction strategy that retains only KV
2


---

vectors of Heavy-Hitter tokens and recently generated tokens, maintaining a constant-size KV-cache
while preserving generation quality. StreamingLLM [Xiao et al., 2023] discovers the “attention sink”
phenomenon and proposes retaining only initial and recent tokens’ KV vectors to enable efficient
streaming generation. SnapKV [Li et al., 2024] observes that the model can identify important KV
pairs before generation and proposes a query-driven KV cache compression method. PyramidKV [Cai
et al., 2025] further reveals a pyramidal information funneling structure across layers and proposes
layer-wise adaptive KV cache budgets. SapLLM [Chen et al., 2025a] compresses each segment
into a single separator token and stores only the KV vectors of initial, neighboring, and separator
tokens. These methods focus on reducing memory footprint during decoding; our work instead targets
reducing both computation cost and memory consumption simultaneously for long-reasoning tasks.
Linear Attention.
To address the quadratic computational cost of softmax attention, many linear
attention methods have been proposed to improve training and inference efficiency. Early studies
primarily investigated kernel functions for linear attention, such as random feature attention [Peng
et al., 2021] and RNN-izing Transformers [Kasai et al., 2021]. Subsequently, several works introduced
gating mechanisms to more precisely control information flow in the model state: GLA [Yang et al.,
2024a] achieves hardware-efficient training with gated linear attention, HGRN2 [Qin et al., 2024]
introduces state-expanded gated linear RNNs, and Liger [Lan et al., 2025] linearizes large language
models into gated recurrent structures. Recent studies have provided theoretical evidence for the
Test-Time Training (TTT) property of linear attention [Yang et al., 2024b, Sun et al., 2025, Liu et al.,
2025, Chen et al., 2025b], offering theoretical foundations for designing online learning objectives to
enhance sequence modeling capability.
3
Method
In this section, we first briefly introduce Linear Attention and Test-Time Training, which serve as the
foundation of our framework. We then present the detailed design of LISA.
3.1
Preliminaries
Softmax Attention.
Modern decoder-only LLMs rely on multi-head softmax attention [Vaswani
et al., 2017]. Given an input sequence, the output is computed as a softmax-weighted sum over
values, with computational complexity O(t2) and KV-cache that grows linearly with sequence length,
creating a significant memory bottleneck under long-context settings.
Linear Attention.
Linear attention [Katharopoulos et al., 2020] replaces the exponential kernel in
softmax with a decomposable feature map ϕ(·), leveraging the associativity of matrix multiplication
to aggregate KV interactions into a single state matrix. While various kernel functions have been
explored [Kasai et al., 2021, Peng et al., 2021], recent studies [Sun et al., 2023, Yang et al., 2024a] have
shown that setting ϕ(·) to the identity function and removing the normalization term also achieves
good empirical results: ot = qtSt, where St = P
i k⊤
i vi is the state matrix that accumulates
historical information. This recurrent formulation enables O(1) per-step computation and constant
memory overhead.
Test-Time Training (TTT).
Recent studies [Yang et al., 2024b, Sun et al., 2025, Liu et al., 2025,
Chen et al., 2025b] interpret the state matrix St in linear attention as a fast-adapting parameter updated
at every token via lightweight gradient descent. These works introduce an online learning objective
for linear attention and derive a closed-form SGD update rule: the objective L(S) = −⟨Skt, vt⟩,
with the SGD update St = St−1 −β∇L(St−1) = St−1 + βvtk⊤
t . When the learning rate β = 1,
the linear attention state update is exactly equivalent to training St with the above objective. While
there is currently no direct empirical evidence that linear attention can enhance model reasoning
capability, its theoretical properties suggest considerable potential.
3.2
LISA: Linear-Indexed Sparse Attention
We design LISA to replace the original attention module. Specifically, our method combines
a linear attention stream with a fixed-size self-attention stream, where the former records long-
range contextual information and the latter preserves precise detail retrieval. A gating mechanism
3


---

× L
LISA
Add & Norm
FFN
Add & Norm
⋮
⋮
Linear Attention
𝑊!
𝑊"
𝑊#
𝑊$
𝑆!"#
𝑆!
𝑂%
ℎ′$
Self-Attention
𝑊!
𝑊"
𝑊#
SoftMax
ℎ′′$
𝑊%
ℎ$
Stage 1 — Sliding-Window SA
query
Fixed sliding
window of size M
+ initial query
(attention sink)
Stage 2 — Indexer-Selected Sparse SA
query
Top-M tokens 
picked
by the Indexer
from the full 
context
Indexer  (top-M)
query / sink tokens
sliding-window region (Stage 1)
Indexer-selected tokens (Stage 2)
thinking-step tokens (KV cache)
⊕
⊙∗
Selected
  topM
Figure 1: Overview of the LISA architecture.
controls the information flow ratio between the two streams. Following related works on linear
attention [MiniMax et al., 2025], we replace the standard attention in the teacher model with our
linear attention variant while retaining the original standard attention every 6 layers to preserve exact
retrieval capability.
Specifically, we divide the training into two stages.
Stage 1: Linear Attention Cold-Start.
In this stage, we convert the standard teacher self-attention
to a fixed-budget sliding window of size M, while introducing a linear attention as in Figure 1.
During training, we freeze all original model parameters and only train the newly added linear
attention parameters. We construct an attention block that fuses linear attention and self-attention
through a learnable gating parameter:
oLISA = g · oLA + (1 −g) · oSA_M,
(1)
where g is the learnable gate, oLA is the linear attention output, and oSA_M is the sliding-window
self-attention output with window size M.
We aim for the LISA output to match the teacher’s full self-attention as closely as possible. However,
rather than explicitly using a distillation loss (e.g., KL loss) in training LISA, we employ a standard
cross-entropy loss. This is because we observed severe overfitting when using KL loss during early
training. We hypothesize that, similar to findings in LoRA, the attention distribution is inherently
low-rank; fitting the attention state distribution is prone to overfitting, causing the model to overlook
fine-grained details beneficial for subsequent generation and leading to performance degradation.
Stage 2: Indexer-Guided Token Selection.
In this stage, we further introduce a lightweight
token selector called the Indexer. Its design is inspired by the Lightning Indexer in DeepSeek
V3.2 [DeepSeek et al., 2025b] and GLM-5 [Zhipu AI, 2026]: it contains only query/key projections
and a ReLU dot product, used to select important tokens based on the current context.
Specifically, we maintain the KV cache for the full context. At each generation step t, the Indexer
performs an attention computation over the preceding t −1 tokens, selecting M important tokens to
fill into the self-attention, which together with the linear attention state at step t produces the t-th
token. The resulting state is used to update both the linear attention state and the KV cache.
The Indexer’s role is to select important tokens from the full context for subsequent token generation,
which is similar to DeepSeek V3.2 [DeepSeek et al., 2025b] and GLM-5 [Zhipu AI, 2026]. The key
difference is that, due to the top-M (argmax) strategy, training gradients cannot flow back to the
Indexer module. Their approach maintains a distillation loss to ensure Indexer updates. However,
4


---

since they pretrain from scratch, the main model and Indexer attention are updated synchronously,
resulting in relatively lower overfitting risk. Building on this, we introduce dropout: specifically,
during early training, we select top-1.2M candidates and randomly mask out 20%, then reassemble
M hidden states to fill the self-attention stream, forcing the model to make effective selections. As
training progresses, we gradually reduce the dropout ratio. We also apply z-loss to the Indexer to
prevent collapse onto early tokens.
However, this still limits the model’s performance to some extent. Therefore, we introduce a multi-
head mechanism for the Indexer, with the loss applied directly on each head. We design a K-head
Indexer that simultaneously fits the output of P heads of the teacher attention. Specifically, each
Indexer head fits the attention distribution over the full context from P/K teacher full-attention
heads, with dynamic temperature scaling to prevent gradients from high-proportion tokens from
overwhelming those of secondary tokens.
Concretely, we sum the scores of K Indexer heads over all tokens and select the M positions with the
largest values as important tokens. Based on the Indexer’s selection, we fill the cached hidden states
at the corresponding positions into SA_M for subsequent computation. This approach mitigates the
loss of precise contextual information caused by linear attention state compression.
The linear attention continues to be updated via cross-entropy loss. The Indexer loss is defined as:
LIndexer =
1
KL
K
X
j=1
L
X
t=1
KL

T (j)
t,:



 softmax(S(j)
t,: )

,
(2)
where S(j)
t,:
∈RL is the ReLU dot-product score vector of the j-th Indexer head over all keys,
computed as ReLU
 ⟨q(j)
t , k(j)
·
⟩

, with q(j)
t
= RoPE(W Q
j ht) and k(j)
·
= RoPE(W K
j h·). The
target distribution T (j)
t,: is the normalized teacher target distribution for the j-th Indexer head, obtained
by summing the attention distributions of the corresponding P/K teacher full-attention heads and
re-normalizing (with stop-gradient applied). Both softmax(·) and the normalization in T (j)
t,: are
restricted to the causal triangle for row-wise normalization.
3.3
Inference State Transfer
During inference, leveraging the test-time training (TTT) properties inherent in linear attention, we
introduce a suite of state-transfer operations to optimize the model’s reasoning trajectory. We develop
a k-step corrective reasoning strategy, wherein the input sequence is partitioned into segments of size
k. We employ a dual-update mechanism that synchronizes linear attention (LA) state transitions with
segment-level state updates.
LA State Update.
In the l-th layer LA submodule, after completing the first t −1 reasoning
segments, the model state is updated to S(l)
t−1. This serves as the initial state S(l)
t,0 = S(l)
t−1 for the
current reasoning segment and is updated token-by-token:
S(l)
t,i−1 = S(l)
t,0 +
i−1
X
j=1
k(l)
j
⊤v(l)
j .
(3)
We then use the query vector q(l)
i−1 of input token xi−1 to extract relevant historical information from
this state:
o(l)
i−1 = q(l)
i−1S(l)
t,i−1.
(4)
We obtain the submodule output via a gating mechanism:
ˇh(l)
t,i−1 = σ

Wgh(l−1)
t,i−1

⊙o(l)
i−1,
(5)
where σ is the sigmoid function and Wg is a learnable weight. The LA submodule is implemented
via a LoRA strategy to control its parameter size.
5


---

Segment State Update.
During inference, LLMs often produce noisy reasoning segments that
mislead subsequent generation, leading to “overthinking”. In our framework, noise can cause state
transitions to deviate from the correct reasoning trajectory. To address this, we propose a correction
strategy that guides the model using a global reasoning direction, reducing noise-induced deviation.
We partition the generated token sequence into segments of fixed size k. The state change (reasoning
direction) at the t-th segment is defined as the difference between the segment’s start and end states:
∇l
t = S(l)
k·t −S(l)
k·(t−1).
(6)
Noisy steps typically produce ∇t that deviates from the global trend. Therefore, at the end of the t-th
segment, we perform a correction. First, we aggregate all prior reasoning directions via momentum
accumulation to obtain the global reasoning direction:
¯∇l
t−1 =
1
t −1
t−1
X
i=1
∇l
i,
(7)
with ¯∇l
0 initialized as a zero matrix. We then correct the current segment’s direction using the global
direction:
ˆ∇l
t = (1 −α)∇l
t + α ¯∇l
t−1,
(8)
where α is a fixed correction coefficient hyperparameter.
Finally, the state is updated using the corrected direction:
Sl
t = Sl
t−1 + ˆ∇l
t.
(9)
This mechanism enables the model to leverage historical reasoning trends to rectify the current step,
preventing local noise accumulation from causing overall directional drift.
4
Experiments
4.1
Implementation Details
Our experiments utilize the Qwen-2.5 model series [Qwen et al., 2025]. To construct the training set,
we randomly sample 100K samples from the OpenR1-Math-220K dataset. The model is initialized
using the weights of its corresponding DeepSeek-R1 distilled version [DeepSeek et al., 2025a].
During Stage 1 of training, we employ a learning rate of 2 × 10−5 and a batch size of 128 for 2
epochs and M is 256. To initialize the linear attention layers, we evaluated both random initialization
and Singular Value Decomposition (SVD) of the base model’s self-attention weights. Since SVD
initialization yielded no measurable performance gains, all reported experiments utilize random
initialization accompanied by a 50-step linear warmup. This stage takes approximately 240 GPU
hours on an NVIDIA H100.
For Stage 2, the model is trained for 1 epoch using a learning rate of 2 × 10−5 and a batch size
of 64 and M is 256. We initialize the Indexer randomly with a 50-step warmup. To mitigate the
impact of initial noisy Indexer representations, the linear attention module is freezing for the first 200
steps. Subsequently, the Indexer and linear attention are updated simultaneously. Although this stage
runs for fewer training steps, the computation is significantly increased due to the teacher attention
computation, taking approximately 240 GPU hours on an NVIDIA H100.
4.2
Baselines
We compare our framework with two types of baselines: (1) Efficient reasoning methods: Light-
Thinker [Zhang et al., 2025] uses learnable special tokens to compress the reasoning information
of completed reasoning steps; INFTYTHINK [Yan et al., 2025] compresses prior reasoning steps
into a concise summary; Zhang et al. [2026] proposes a state-transition framework that uses linear
attention to maintain a global state matrix for recording historical information, enabling the model
to retrieve relevant information without explicitly attending to prior reasoning steps. (2) KV-cache
reduction methods: H2O [Zhang et al., 2023] retains only the KV vectors of heavy-hitter tokens
and recent tokens; SapLLM [Chen et al., 2025a] only stores the KV vectors of initial, neighboring,
and separator tokens. All baselines are trained using the LoRA strategy with a comparable number of
trainable parameters to our model.
6


---

Method
GSM8K
MATH-500
AMC23
AIME24
AIME25
AVG.
Acc↑
Tok↓
ReL↓
Acc↑
Tok↓
ReL↓
Acc↑
Tok↓
ReL↓
Acc↑
Tok↓
ReL↓
Acc↑
Tok↓
ReL↓
Acc↑
ReL↓
Qwen2.5-1.5B Series
Ours (Base)
80.1
2086
76.2
78.8
3958
139.4
62.5
6392
242.9
20.0
13765
536.8
16.7
12684
504.7
51.5
300.0
+ H2O
76.1
2116
71.7
75.4
3835
128.5
55.0
6230
209.7
13.3
13921
462.2
10.0
12802
428.6
46.0
260.1
+ SepLLM
77.3
2230
75.8
76.0
3890
130.3
57.5
6573
220.2
13.3
13690
452.6
13.3
12690
423.9
47.5
260.6
LightThinker
78.8
2109
69.6
77.6
4060
134.8
60.5
6312
214.6
16.7
13827
461.6
13.3
12934
434.6
50.1
263.0
INFTYTHINK
79.5
2503
89.3
78.0
4750
170.2
62.5
7605
263.5
16.7
16318
566.2
16.7
15020
501.0
50.7
318.0
Zhang et al. [2026]
80.1
2116
69.7
81.2
3812
125.8
67.5
6460
213.2
26.7
13610
440.1
23.3
12910
416.0
56.2
253.0
LISA
80.2
1983
61.5
82.0
3636
102.5
70.0
6157
187.7
26.7
12966
392.3
26.7
12532
398.1
57.1
228.4
Qwen2.5-7B Series
Ours (Base)
89.4
2649
96.4
87.4
4253
161.6
82.5
6704
242.9
40.0
12901
522.1
30.0
13204
548.1
65.9
314.2
+ H2O
83.5
2730
92.0
82.6
4389
147.1
77.5
6843
209.7
30.0
13274
448.9
23.3
13204
443.7
59.4
268.3
+ SepLLM
84.9
2582
92.3
84.2
4244
144.3
80.0
6511
220.2
33.3
13100
484.7
23.3
12972
435.9
61.1
275.5
LightThinker
85.6
2779
90.0
84.8
4321
145.2
80.0
6790
214.6
33.3
13307
449.0
26.7
13250
443.9
62.1
268.5
INFTYTHINK
87.9
3237
117.0
86.2
5050
184.8
82.5
8061
254.3
36.7
15094
508.7
30.0
15448
530.5
64.7
319.0
Zhang et al. [2026]
90.9
2603
87.2
90.0
4196
140.6
85.0
6665
213.2
43.3
13017
434.8
33.3
13191
440.6
68.3
263.3
LISA
90.9
2462
82.1
91.2
4253
117.9
87.5
6841
156.3
43.3
12587
385.1
40.0
13642
398.1
70.6
226.6
Table 1: Results of our model and baselines on mathematical reasoning benchmarks.
4.3
Datasets & Metrics
We evaluate our framework on five mathematical reasoning benchmarks (GSM8K [Cobbe et al.,
2021], MATH-500 [Hendrycks et al., 2021], AMC23 [AMC, 2023], AIME24 [AIME, 2024],
AIME25 [AIME, 2025]). We report three metrics: (1) Accuracy (Acc) denotes the percentage
of correct answers; (2) Token Number (Tok) refers to the average length of the CoT sequence; (3)
Reasoning Latency (ReL) is defined as the average inference time per sample. We measure reasoning
latency on a single NVIDIA A100 GPU with a batch size of 1, using eager mode with BF16 precision.
4.4
Main Results
We conduct a comprehensive evaluation of LISA across five mathematical reasoning benchmarks.
The results, summarized in Table 1, compare LISA against state-of-the-art efficiency-driven baselines
using both Qwen2.5-1.5B and Qwen2.5-7B backbones.
LISA achieves superior accuracy and efficiency.
LISA consistently achieves superior perfor-
mance across all benchmarks while maintaining the lowest relative latency (ReL ↓) among all tested
methods. On the Qwen2.5-7B series, LISA reaches an average accuracy of 70.6%, outperforming
the strongest baseline [Zhang et al., 2026] by 2.3% absolute, while simultaneously reducing the
relative latency from 263.3 to 226.6. This demonstrates that LISA effectively optimizes self-attention
computation without sacrificing the model’s reasoning capabilities.
Impact of sequence length and task complexity.
A key observation from Table 1 is the correlation
between task complexity (and associated output length) and the performance margin of our method.
Specifically, on benchmarks with short outputs like GSM8K, the performance gap between LISA
and the dense baseline is minimal (e.g., 80.2% vs 80.1% for the 1.5B model). The primary reason
is that our method still introduces a fixed budget for self-Attention, yielding marginal gains on
short sequences. By contrast, on tasks requiring long-horizon reasoning, the advantages of LISA
become more pronounced. For instance, on AIME25 with the 7B model, LISA achieves 40.0%
accuracy, significantly higher than the 33.3% reached by Zhang et al. [2026]. This suggests that our
sparse attention mechanism is particularly adept at maintaining coherence and reasoning accuracy in
long-sequence generation where standard KV-cache or attention-reduction methods often struggle.
4.5
Analysis
Analyzing the Synergy between Linear and Sparse Attention.
We conduct a granular analysis
of the hidden states in both linear attention and sparse self-attention to verify whether the linear
component effectively provides the intended global context. To quantify this relationship, we compute
the cosine similarity between the outputs of these two mechanisms, with results illustrated in Figure 2.
Our analysis reveals that the cosine similarity between linear and sparse attention outputs is consis-
tently negative across all layers, suggesting that the two representations are non-trivial and mutually
complementary. This anti-correlation aligns with our architectural expectations, where each com-
ponent specializes in distinct information subspaces. Notably, in Stage 2, this similarity shifts from
7


---

1
2
3
4
5
7
8
9
10
11
13
14
15
16
17
19
20
21
22
23
25
26
Layer index
−0.5
−0.4
−0.3
−0.2
−0.1
0.0
0.1
cos(oLA, oSA)
S2 mean=-0.270
S1 mean=-0.297
LA and SA outputs are persistently anti-correlated across all layers
Random vs SA (S1, ≈+0.0001)
Random vs SA (S2, ≈-0.0001)
Stage 1 (LA only)
Stage 2 (full LISA)
Figure 2: Cosine similarity between linear attention and sparse self-attention outputs across all layers.
−0.297 to −0.270.3 We hypothesize that this incremental increase reflects the functional evolution
of the sparse component: while the initial sliding-window mechanism is strictly localized, the Stage 2
Indexer learns to capture salient tokens from the broader context. By incorporating these long-range
dependencies, the Indexer-based representation partially converges toward the global context captured
by the linear attention component, thereby reducing the divergence between the two signals.
Furthermore, the introduction of the Indexer mechanism in Stage 2 provides sparse self-attention
with an extended context window, reducing the magnitude of the corrective signal required from the
linear attention component. This validates our experimental design and supports our core hypothesis:
linear attention is not redundant, but rather serves as a persistent corrective signal for long-range
dependencies. In summary, LISA operates through a synergistic division of labor: sparse self-
attention generates a local estimate based on an M-step context, while linear attention contributes a
long-range refinement, with the gating mechanism learning the optimal integration ratio.
0
200
400
600
800
1000
1200
1
2
3
4
5
7
8
9
10
11
13
14
15
16
17
19
20
21
22
23
25
26
Layer
Case 1
user prompt boundary
query position
0
100
200
300
400
500
600
700
800
Key position in sequence
1
2
3
4
5
7
8
9
10
11
13
14
15
16
17
19
20
21
22
23
25
26
Layer
Case 2
0.0
0.2
0.4
0.6
0.8
1.0
Selection frequency
(row-normalized)
Figure 3: Visualization of the top-256 tokens selected by the Indexer across layers.
The Indexer Recovers Full Self-Attention Dynamics.
We conduct an analysis of the Indexer’s
outputs to verify its functional alignment with our design objective. By visualizing the top-256
3While these values are small, they are statistically significant; for comparison, the cosine similarity between random
vectors and sparse attention outputs is approximately 10−4, which is three orders of magnitude lower.
8


---

tokens selected by the Indexer for two randomly sampled examples (Figure 3), we observe that the
mechanism effectively learns to identify salient information.
Specifically, the distribution of selected tokens is heavily concentrated in the local context of the
current output token, a behavior consistent with established observations regarding recency bias in
transformer models [Peysakhovich and Lerer, 2023, Wang et al., 2024, Xiao et al., 2023]. Beyond the
local window, the Indexer selectively attends to global features, most notably punctuation tokens (see
Appendix for details), which aligns with existing research on sparse attention patterns [Lou et al.,
2024, Bondarenko et al., 2023]. Furthermore, our results confirm the emergence of attention sink
behavior Xiao et al. [2023]. Collectively, these findings demonstrate that the Indexer successfully
recovers the distribution of the original attention mechanism, validating its role in capturing both
local and long-range dependencies.
Optimal Window Size M and Attention Coverage.
We further investigate the impact of the
hyperparameter M for both sliding window size and the Indexer capacity. The objective is to select
a value for M that captures the majority of the attention distribution without incurring prohibitive
computational overhead. To quantify this trade-off, we introduce a metric termed MASS, defined as:
MASS(q, M) =
P
k∈top-M(q) Attn(q, k)
P
all k≤q Attn(q, k)
(10)
where Attn(q, k) denotes the oracle attention weights. This metric serves as a proxy for the repre-
sentational fidelity of the sparse subset relative to the full self-attention distribution. Moreover, it
provides a benchmark to evaluate the precision of the Indexer’s selection mechanism. Our empirical
results demonstrate that at M = 256, the MASS reaches 80%, trailing the theoretical upper bound
(defined by the oracle’s top-M weights) by only 6%. Performance scales predictably with the budget:
the metric decreases to 70% at M = 128 and improves to 88% at M = 512. These results suggest
that M = 256 offers an optimal balance between distributional coverage and efficiency.
5
Conclusion
In this work, we proposed LISA (Linear-Indexed Sparse Attention), a modular and plug-and-play
attention mechanism designed to accelerate long-context chain-of-thought reasoning without requiring
pre-training from scratch. LISA adopt two parallel streams: a global linear attention component for
long-range contextual modeling and a fixed-size sparse self-attention component for high-fidelity
local and salient reasoning. Through a coordinated two-stage training strategy, initially cold-starting
the linear attention for long-range dependencies followed by the introduction of a indexer for dynamic
token selection, LISA reduces inference complexity from O(n2) to O(nM) (M ≪n) for generating
n tokens. Empirical evaluations on mathematical reasoning benchmarks demonstrate that LISA
significantly reduces inference latency while maintaining or even surpassing the accuracy compared
to competitive baselines. Future work will explore adaptive token budget strategies and extend LISA
to broader applications such as code generation and multi-turn dialogue.
6
Limitations
Despite the efficiency and quality advantages demonstrated by LISA in long reasoning tasks, this
work still has several limitations:
Limited scope of experiments. Our experiments primarily focus on math reasoning tasks. While
LISA has been validated on math reasoning benchmarks, its effectiveness in broader application
scenarios (e.g., code generation, agent-based tasks) remains to be further explored.
Two-stage training complexity. Compared to standard SFT, the two-stage training pipeline of LISA
is more complex. Stage 1 requires cold-starting the linear attention for long-range fitting, and Stage 2
gradually introduces the Indexer with a freezing strategy. This multi-stage process increases the
difficulty of hyperparameter tuning and computational resource requirements.
Frozen original parameters. The training strategy of LISA freezes all original parameters of the
main model, only training the newly added linear attention and Indexer components. While this
design ensures the plug-and-play property and avoids catastrophic forgetting, it also means that the
9


---

synergy between the main model and the new components cannot be further optimized through
fine-tuning. In certain scenarios, joint fine-tuning may yield additional performance gains.
Retention of original attention layers. LISA currently retains the original standard attention
layers every 6 layers, which contribute approximately 70% of the total attention FLOPs. Due to
computational resource constraints, we have not yet verified the feasibility of a fully LISA-replaced
architecture (i.e., replacing all layers). We leave the exploration of removing the remaining standard
attention layers for future work to further reduce computational overhead.
10


---

References
Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with
reinforcement learning. arXiv:2503.04697, 2025.
AIME. American invitational mathematics examination (aime) 2024-i & ii, 2024. URL https:
//huggingface.co/datasets/Maxwell-Jia/AIME_2024.
AIME. American invitational mathematics examination (aime) 2025-i & ii, 2025. URL https:
//huggingface.co/datasets/opencompass/AIME2025.
AMC. American mathematics competitions, 2023. URL https://artofproblemsolving.com/
wiki/index.php/AMC_Problems_and_Solutions.
Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Quantizable transformers: Removing
outliers by helping attention heads do nothing. arXiv:2306.12929, 2023.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong,
Yue Dong, Junjie Hu, and Wen Xiao. Pyramidkv: Dynamic kv cache compression based on
pyramidal information funneling, 2025. URL https://arxiv.org/abs/2406.02069.
Guoxuan Chen, Han Shi, Jiawei Li, Yihang Gao, Xiaozhe Ren, Yimeng Chen, Xin Jiang, Zhenguo
Li, Weiyang Liu, and Chao Huang. Sepllm: Accelerate large language models by compressing one
segment into one separator. In Proceedings of ICML, 2025a.
Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu,
Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+3=? on the overthinking of
o1-like llms. In Proceedings of ICML, 2025b.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems. arXiv:2110.14168, 2021.
DeepSeek, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao
Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv:2501.12948, 2025a.
DeepSeek et al.
Deepseek-v3.2:
Pushing the frontier of open large language models.
arXiv:2512.02556, 2025b.
Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-
budget-aware llm reasoning. arXiv:2412.18547, 2024.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn
Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In
Proceedings of NeurIPS, 2021.
Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and
Mengnan Du. The impact of reasoning step length on large language models. In Proceedings of
ACL, 2024.
Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-of-thought
without compromising effectiveness. In Proceedings of AAAI, 2025.
Jungo Kasai, Hao Peng, Yizhe Zhang, Dani Yogatama, Gabriel Ilharco, Nikolaos Pappas, Yi Mao,
Weizhu Chen, and Noah A Smith. Finetuning pretrained transformers into RNNs. In Proceedings
of EMNLP, 2021.
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are
RNNs: Fast autoregressive transformers with linear attention. In Proceedings of ICML, 2020.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. In Proceedings of NeurIPS, 2022.
11


---

Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, and Yu Cheng. Liger: Linearizing large language models
to gated recurrent structures. In Proceedings of ICML, 2025.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation,
2024. URL https://arxiv.org/abs/2404.14469.
Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, and Qiang Liu. Longhorn: State space
models are amortized online learners. In Proceedings of ICLR, 2025.
Tengxiao Liu, Qipeng Guo, Xiangkun Hu, Cheng Jiayang, Yue Zhang, Xipeng Qiu, and Zheng Zhang.
Can language models learn to skip steps? In Proceedings of NeurIPS, 2024.
Chao Lou, Zixia Jia, Zilong Zheng, et al. Sparser is faster and less is more: Efficient sparse attention
for long-range transformers. arXiv:2406.16747, 2024.
Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning
models can be effective without thinking. arXiv:2504.09858, 2025a.
Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Length-
compressible chain-of-thought tuning. arXiv:2502.09601, 2025b.
William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought.
In Proceedings of ICLR, 2024.
MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang,
Congchao Guo, Da Chen, et al. Minimax-01: Scaling foundation models with lightning attention.
arXiv:2501.08313, 2025.
Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young Yun.
Self-training elicits concise reasoning in large language models. arXiv:2502.20122, 2025.
Sania Nayab, Giulio Rossolini, Marco Simoni, Andrea Saracino, Giorgio Buttazzo, Nicolamaria
Manes, and Fabrizio Giacomelli. Concise thoughts: Impact of output length on llm reasoning and
cost. arXiv:2407.19825, 2024.
OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden
Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko,
Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally
Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich,
Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghor-
bani, Bohan Zhang, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev,
Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo
Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen,
Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan
Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely,
David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund
Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan
Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song,
Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace
Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin,
Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian
O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever,
Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng,
Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish,
Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward,
Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe,
Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone,
Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian
Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz
Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor,
Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y.
12


---

Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek,
Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang,
Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowd-
hury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia
Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora,
Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm
Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman,
Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha
Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray
Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev,
Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan,
Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi,
Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie
Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wenting Zhan, Wes McCabe,
Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He,
Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2026. URL
https://arxiv.org/abs/2412.16720.
Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A Smith, and Lingpeng Kong.
Random feature attention. In Proceedings of ICLR, 2021.
Alexander Peysakhovich and Adam Lerer. Attention sorting combats recency bias in long context
language models. arXiv:2310.01427, 2023.
Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong.
Hgrn2: Gated linear RNNs with state expansion. In Proceedings of COLM, 2024.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin
Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi
Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL
https://arxiv.org/abs/2412.15115.
Qwen
Team.
Qwq:
Reflect
deeply
on
the
boundaries
of
the
unknown.
https://qwenlm.github.io/blog/qwq-32b-preview/, 2024.
Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai
Wang, Zhaoxiang Liu, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning
models. arXiv:2503.04472, 2025.
Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen,
Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test
time): Rnns with expressive hidden states, 2025. URL https://arxiv.org/abs/2407.04620.
Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu
Wei. Retentive network: A successor to transformer for large language models. arXiv:2307.08621,
2023.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of NeurIPS, 2017.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models.
In Proceedings of ICLR, 2023.
Yibo Wang, Li Shen, Huanjin Yao, Tiansheng Huang, Rui Liu, Naiqiang Tan, Jiaxing Huang,
Kai Zhang, and Dacheng Tao. R1-compress: Long chain-of-thought compression via chunk
compression and search. arXiv:2505.16838, 2025.
Ziqi Wang, Hanlin Zhang, Xiner Li, et al. Eliminating position bias of language models: A mechanis-
tic approach. arXiv:2407.01100, 2024.
13


---

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings
of NeurIPS, 2022.
Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable
chain-of-thought compression in llms. arXiv:2502.12067, 2025.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. arXiv:2309.17453, 2023.
Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Mengdi Zhang, Jian Shao, and Yueting Zhuang.
Inftythink: Breaking the length limits of long-context reasoning in large language models.
arXiv:2503.06692, 2025.
Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention
transformers with hardware-efficient training. In Proceedings of ICML, 2024a.
Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers
with the delta rule over sequence length. In Proceedings of NeurIPS, 2024b.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
Tree of thoughts: Deliberate problem solving with large language models. In Proceedings of
NeurIPS, 2023.
Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen,
and Ningyu Zhang. Lightthinker: Thinking step-by-step compression. arXiv:2502.15589, 2025.
Liang Zhang, Yu Zhao, Longyue Wang, Tianqi Shi, Weihua Luo, Kaifu Zhang, and Jinsong Su. A
state-transition framework for efficient llm reasoning, 2026. URL https://arxiv.org/abs/
2602.01198.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient
generative inference of large language models. In Proceedings of NeurIPS, 2023.
Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua
Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for open-ended solutions, 2024.
URL https://arxiv.org/abs/2411.14405.
Zhipu AI. Glm-5: From vibe coding to agentic engineering. arXiv:2602.15763, 2026.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning
in large language models. In Proceedings of ICLR, 2023.
14


---

A
Indexer Output Distribution: Case Studies
We visualise the top-256 keys selected by the LISA Indexer for two representative reasoning examples
at different network depths. The background colour intensity (white →dark red) reflects the attention
score assigned to each candidate key token.
A.1
Layer 1 Indexer Output Distribution
Question:
<|begin of sentence|> <|User|> The center of a circle touching the legs $A C$
and $B C$ of a right triangle $A B C$ lies on the hypotenuse $A B$.
Find the radius of
the circle, if it is six times smaller than the sum of the legs, and the area of triangle
$A B C$ is 27.<|Assistant|> <think>
Response:
<think>
Okay, so I
have this problem about a right triangle ABC where the circle touches the legs
AC and BC, and its center is on the hypotenuse AB. I need to find the radius of the
circle, which is six times smaller than the sum of the legs.
Also, the area of the
triangle is 27.
Hmm, let me break this down step by step.
First, let me visualize the triangle.
Since it’s a right triangle, let’s assume angle C
is the right angle.
So, legs AC and BC form the right angle, and AB is the hypotenuse.
The circle touches both legs AC and BC, so it must be tangent to both.
The center of the
circle lies on AB. That makes me think it’s an incircle or maybe an excircle, but since
it’s touching both legs and the center is on the hypotenuse, it’s probably not the
standard incircle because the incenter of a right triangle is located inside the triangle,
not on the hypotenuse.
Wait, but the problem says the center lies on the hypotenuse.
So
this must be a different circle, not the incircle.
Let me recall that if a circle is tangent to both legs of a right triangle and its center
is on the hypotenuse, there might be a specific formula or relation here.
Let me denote
the legs as AC = b and BC = a, and hypotenuse AB = c.
The area is (a*b)/2 = 27, so a*b =
54.
Also, the radius r is six times smaller than the sum of the legs, so r = (a + b)/6.
My goal is to find r.
Wait, so I need to relate r to a and b.
Let me think about the coordinates.
Maybe
coordinate geometry can help here.
Let me place point C at the origin (0,0), point B at
(a,0), point A at (0,b).
Then hypotenuse AB goes from (0,b) to (a,0).
The center of the
circle lies on AB and is equidistant from both legs AC and BC. Since the circle is tangent
to both legs, which are the y-axis and x-axis respectively, the distance from the center
to each axis must be equal to the radius.
Wait, if the center is (r, r) for the incircle,
but in this case, the center is on AB. So the center (h,k) must satisfy h = k = r?
No,
because the center is on AB. Wait, maybe the coordinates of the center can be
param etr ized
along the hyp otenuse.
Let me param etrize
AB. The hypotenuse AB can be represented
by the line from (0, b) to
(a, 0).
The equation of AB is y = (- b /a )x +
b. Any point on
AB can
be written
as ( x , (- b /a )x + b ).
The center of the circle is at some point (x, (-b/a)x + b)
on AB, and this point must be at a distance r from both legs AC and BC. The legs AC and BC
are the y-axis and x-axis respectively, so
the distance from the center to AC (y-axis) is
x, and the distance to BC (x-axis) is y.
Since the circle is tangent to both legs, these
distances must be equal to the radius.
So x = r and y = r. Therefore , the center is (r,
r), but also
lies on AB. Wait , but if the center is (r, r), then substit uting
into the
equation of AB gives r = (-b/a)r + b. Let me check that .
The equation of AB is y = (-b/a )x +
b. So if the center
is at (r ,
r ), then
substituting x = r into
the equation gives y
= (-b/a)r + b.
But the y-coordinate of the
center is also r, so we have:
r = (-b/a)r + b
Let me solve for
r:
r + (b/a)r = b
r(1 + b/a) = b
r = b / (1 + b/a) = (ab) / (a + b)
15


---

But from the problem statement, the radius r is (a + b)/6.
So according to this, (ab)/(a
+ b) = (a + b)/6
Multiply
both sides
by (a + b):
ab = (a + b)^2 / 6
Multiply both sides by 6:
6ab = (a + b)^2
But we also know that ab =
54 , so substitute :
6 * 5 4 = ( a + b )^ 2
3 2 4 = ( a +
b )^ 2
Therefore ,
a + b = sqrt ( 3 2 4 ) =
1 8
So then
r =
(a + b )/ 6 =
1 8 /
6 =
3
Wait , so the radius is
3 ? Let me check if that makes sense .
Wait , so starting from the center being ( r , r ) because it ’s distance r from
both axes , and since
it ’s on AB , substit uting into the equation gives r = (- b /a )r + b .
Then solving that gives r = ( ab )/( a + b ). Then according to the problem
, r is ( a + b )/ 6 . So equ ate them :
(ab )/( a + b ) = ( a + b )/ 6
Cross multiply :
6 ab = ( a + b )^ 2
But since ab =
5 4 ,
6 * 5
Question:
<|begin of sentence|> <|User|> The solution
set
of $\{} log_{\{}frac{1}{2}}|x- \{}frac{\{}pi}{3}| \{}geq \{}log_{\{}frac{1}{2}}
\{}frac{\{}pi}{2}$ is ( )
A: $\{}{x|-\{}frac{\{}pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6}\{}}$
B: $\{}{x|x \{}leq -\{}frac{\{}pi}{6}, \{}text{ or } x \{}geq \{}frac{5\{}pi}{6}\{}}$
C: $\{}{x|-\{}frac{\{}pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6} \{}text{ and } x \{}neq
\{}frac{\{}pi}{3}\{}}$
D: $\{}{x|-\{}frac{5\{}pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6} \{}text{ and } x \{}neq
\{}frac{\{}pi}{3}\{}}$<|Assistant|><think>
Response:
<think>
Okay, so I
need to solve this logarithmic inequality:
log base 1/2 of |x - π/3| is
greater than or equal to log base 1/2 of π/2.
Then choose the correct solution set from
the options A to D. Let me think step by step.
First, I remember that logarithmic functions have certain properties, especially when
dealing with inequalities.
Since the base here is 1/2, which is between 0 and 1, the
logarithmic function is decreasing.
That means when I apply it to both sides of an
inequality, the direction of the inequality will reverse.
So, if log_b(a) ≥log_b(c),
then for 0 < b < 1, a ≤c.
Right?
Let me confirm that.
Yes, because as the input to a
decreasing function increases, the output decreases.
So, the original inequality
log_{1/2}(A) ≥log_{1/2}(B) implies A ≤B.
Therefore, applying this property here, the given inequality log_{1/2}|x - π/3| ≥
log_{1/2}(π/2) implies that |x - π/3| ≤π/2.
But wait, also, we need to consider the
domain of the logarithm.
The argument of the logarithm must be positive.
So, |x - π/3| >
0.
Which means x̸ = π/3.
Because absolute value is always non-negative, but logarithm is
undefined at 0.
So x cannot be π/3.
So, putting this together, the solution set is the set of all x such that |x - π/3| ≤π/2
and x̸ = π/3.
Now, let’s solve |x - π/3| ≤π/2.
The absolute value inequality |A | ≤
16


---

B ( where
B >
0) is equivalent to -B ≤A ≤B. So here , - π / 2 ≤x - π/3 ≤π/2.
Let’s solve for
x.
First , add π/3 to all parts:
-π/2 + π/3 ≤x ≤π/2 + π/3
Compute
-π/2 + π/3.
To add these, find a common denominator.
6 is the least common
multiple of
2 and 3.
So:
-π/2 = -3π/6
π/3 = 2π/6
So, -3π/6 + 2π/6 = (-3π + 2π)/6 = -π/6
Similarly , π/2 + π/3:
π/2 = 3π/6
π/3 = 2π/6
3π/6 + 2π/6 =
5 π / 6
Therefore , the inequality | x -
π/3| ≤
π / 2 simpl ifies
to -π/6 ≤x ≤5π/6.
But
remember x̸ =
π/3 because |x - π/3| > 0.
So, the solution set
is all x in [- π/6, 5π/6] except x = π/3.
Now, looking at
the
options:
A: {x | -π/6 ≤x ≤5π/6} →Doesn’t exclude π/3, so incorrect .
B: {x | x ≤-π/6 or x ≥5π/6} →This
is the complement of
the interval , which would
be if the inequality was reversed .
But since our inequality
was |x - π/3| ≤π/2, this
is the outside , which is wrong .
C : { x | - π / 6 ≤x ≤
5 π / 6 and x̸ = π / 3 } →This matches our
solution . Includes the interval and excludes π / 3 .
D : { x | - 5 π / 6 ≤x ≤
5 π / 6 and x̸ = π / 3 } →The interval here
is from - 5 π / 6 to
5 π/ 6 , which is different . Let me check if I made a
mistake in solving the inequality .
Wait , when solving | x - π / 3 | ≤π / 2 , adding π / 3 to - π / 2 gives
- π / 6 , and adding π / 3 to π / 2 gives
5 π / 6 . That ’s correct .
So D is incorrect because the interval is from - 5 π / 6 , which is more
negative . So that ’s not matching . So the answer must be C .
But let me just verify once more . The original inequality is log _{ 1 /
A.2
Layer 26 Indexer Output Distribution
Question:
<|begin of sentence|> <|User|> The
center of a circle touching the legs $A
C$ and $B C$ of a right triangle $A B C$ lies on the
hypotenuse $A B$.
Find the radius
of the circle, if it is six times smaller than the sum of the legs, and the area of
triangle $A B C$ is 27. <|Assistant|> <think>
Response:
<think>
Okay ,
so I have this problem about a right triangle ABC where the circle touches the
legs AC and BC, and its center is on the hypotenuse AB .
I need to find the
radius of the
17


---

circle ,
which is six times smaller than the
sum of the legs .
Also, the area of the
triangle is 27.
Hmm ,
let me break this down step by
step .
First, let me visualize the triangle.
Since it’s a right triangle, let’s assume angle C
is the right angle.
So, legs AC and BC form the right angle, and AB is the hypotenuse.
The circle touches both legs AC and BC, so it must be tangent to both.
The center of the
circle lies on AB .
That makes me think it’s an incircle or maybe an excircle ,
but since
it’s touching both legs and the center is on the hypotenuse, it’s probably not the
standard incircle because the incenter of a right triangle is located inside the
triangle ,
not on the hypotenuse .
Wait, but the problem says the center lies on the
hypotenuse.
So this must be a different circle, not the incircle .
Let me recall that if a circle is tangent to both legs of a right triangle and its center
is on the hypotenuse, there might be a specific formula or relation here .
Let me denote
the legs as AC = b and BC = a, and hypotenuse AB = c . The
area is (a*b)/2 = 27 ,
so a*b
= 54 .
Also, the radius r is six times smaller than the sum of the legs ,
so r = (a +
b)/6 .
My goal is to find r .
Wait ,
so I need to relate r to a and b .
Let me think about the coordinates.
Maybe
coordinate geometry can help here .
Let me place point C at the origin (0,0 ),
point B at
(a,0), point A at (0,b).
Then hypotenuse AB goes from (0,b) to (a,0).
The center of the
circle lies on AB and is equidistant from both legs AC and BC .
Since the circle is
tangent to both legs, which are the y-axis and x-axis respectively ,
the distance from
the center to each axis must be equal to the radius .
Wait ,
if the center is (r, r )
for the
incircle ,
but in this case, the center is on AB .
So the
center (h,k) must
satisfy h = k = r ?
No ,
because the
center is on AB .
Wait ,
maybe the coordinates of
the center can be parametrized along
the hypotenuse .
Let me
parametrize AB . The
hypotenuse AB can be represented by the line from (0, b )
to
(a, 0 ). The
equation of
AB is
y = (- b/a )x + b .
Any point on AB can be written as
(x , (- b/a)x +
b ).
The center of the circle is at some point (x, (-b/a)x + b ) on
AB ,
and this point must be at a distance r
from both legs AC and BC . The
legs AC and BC are
the y-axis and
x-axis respectively ,
so the distance from the center to AC (y-axis) is
x , and the
distance to BC (x-axis) is y .
Since the circle is tangent to both legs ,
these distances must be equal to the
radius .
So x = r and
y = r .
Therefore , the
center is ( r ,
r ), but also
lies on AB . Wait , but if the
center is (r, r ), then
substit uting
into the
equation of AB gives r = (- b/a)r + b .
Let me
check that .
The
equation of AB is y = (-b/a)x + b .
So if the center is at (r, r ), then
substituting x = r into the
equation gives y = (-b/a)r + b .
But the
y-coordinate of the
center is also r ,
so we have :
r = (-b/a)r + b
Let me
solve for
r :
r + (b/a)r =
b
r(1 + b/a )
= b
r =
b / (1 + b/a )
= ( ab ) /
(a + b )
18


---

But
from the problem statement ,
the radius r is (a + b)/6 .
So according to this ,
(ab)/(a + b )
= (a + b)/6
Multiply both sides by (a + b ):
ab =
(a + b)^ 2
/ 6
Multiply both sides by 6:
6ab =
(a + b)^ 2
But we
also know that
ab = 54 , so substitute :
6 * 54 =
(a +
b )^ 2
32 4 = ( a + b)^ 2
Therefore ,
a +
b = sqrt ( 32 4 ) =
18
So then r
= (a + b)/6 =
18 /
6 =
3
Wait ,
so the
radius is
3 ?
Let me
check if that makes sense .
Wait , so
starting from the center being (r, r ) because it’s distance r from both
axes , and
since it’s on AB ,
substit uting
into the equation gives r = (-b/a)r +
b . Then
solving that
gives r = ( ab )/( a + b ). Then according to the
problem ,
r is
(a + b)/6 . So
equ ate them :
(ab )/( a + b ) =
(a + b )/ 6
Cross multiply :
6 ab = ( a + b )^ 2
But since ab =
5 4 ,
6 * 5
Question:
<|begin of sentence|> <|User|> The
solution set of
$\{}log_{ \{} frac { 1}{2}}|x- \{} frac {\{} pi}{ 3 }| \{}geq \{}log_{ \{} frac {1}{2}}
\{}frac{\{}pi}{2}$ is (
)
A :
$\{}{x| -\{} frac {\{} pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6 }\{} }$
B :
$\{}{x|x \{} leq -\{}frac {\{} pi}{6}, \{} text{ or } x \{}geq
\{}frac{5\{}pi}{6 }\{} }$
C: $\{}{x|-\{}frac{\{}pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6} \{}text{ and } x \{}neq
\{}frac{\{}pi}{3}\{}}$
D :
$\{}{x|-\{}frac{5\{}pi}{6} \{}leq x \{}leq \{}frac{5\{}pi}{6} \{}text{ and } x \{}neq
\{}frac{\{}pi}{3}\{}}$ <|Assistant|> <think>
Response:
<think>
Okay ,
so I need to solve this logarithmic inequality:
log base 1/2 of |x - π / 3 |
is
greater than or equal to log base 1/2 of π/2 .
Then choose the correct solution set from
the options A to D .
Let me think step by
step .
First, I remember that logarithmic functions have certain properties, especially when
dealing with inequalities.
Since the base here is 1/2, which is between 0 and
1, the
logarithmic function is decreasing.
That means when I apply it to both sides of an
inequality, the direction of the inequality will reverse .
So, if log_b(a) ≥log_b(c ),
then for 0 < b < 1 ,
a ≤
c .
Right ?
Let me confirm that.
Yes, because as the input to
19


---

a decreasing function increases , the
output decreases.
So, the original inequality
log_{1/2}(A) ≥log_{1/2 }( B) implies A ≤B .
Therefore, applying this property here, the given inequality log_{1/2 }| x -
π / 3 |
≥
log_{1/2 }( π /2 ) implies that |x - π/3| ≤π/ 2 .
But wait, also, we need to consider
the domain of the logarithm.
The argument of the logarithm must be positive .
So, |x -
π / 3 | >
0 .
Which means x̸ = π / 3 .
Because absolute value is always non-negative ,
but logarithm is undefined
at 0 .
So x cannot be π / 3 .
So, putting this together, the solution set is the set of all x such that
|x - π / 3 | ≤
π / 2 and
x̸ =
π/3 .
Now, let’s solve |x - π / 3 |
≤π/2 . The
absolute value inequality | A | ≤
B ( where
B >
0 ) is
equivalent to
-B ≤
A ≤
B .
So here, - π / 2
≤x - π / 3 ≤
π/ 2 .
Let ’s
solve for x .
First ,
add π/3 to all parts :
-π/2 +
π/3 ≤
x ≤
π/2 +
π/ 3
Compute -π/2 + π/3 .
To add these, find a common denominator .
6 is the
least common
multiple of 2 and
3 .
So :
-π/2 =
-3 π / 6
π/3 =
2 π / 6
So ,
-3π/6 + 2π/6 = (-3 π +
2π)/6 =
-π/6
Similarly ,
π/2 + π / 3 :
π/2 =
3π/ 6
π/3 =
2π/6
3π/6 + 2π/6 = 5π/ 6
Therefore, the inequality |x - π/3| ≤π/2 simpl ifies
to -π/6 ≤x ≤
5π/6 .
But
remember x̸ =
π/3 because
|x - π/3| > 0 .
So, the solution set is all x in [- π/6 ,
5π/ 6 ] except
x =
π/3 .
Now, looking at the
options:
A: {x | -π/6 ≤
x ≤5π/6 } →
Doesn’t exclude
π/3 ,
so incorrect .
B: {x |
x ≤
-π/6 or x ≥5π/6} →
This is the complement of the interval ,
which would
be if the
inequality was reversed .
But since our inequality was |x - π/3| ≤π/2 ,
this
is the outside ,
which is wrong .
C: {x |
-π/6 ≤
x ≤5π/6 and
x̸ =
π/ 3 } →
This matches our solution . Includes the
interval and excludes
π/3 .
D: {x |
-5π / 6 ≤
x ≤5π/6 and
x̸ = π / 3 } →The
interval here is
from -5π/6 to
5π/6 ,
which is different .
Let me check if I made a mistake in
solving the
inequality .
Wait ,
when solving |x - π/3| ≤π/2 ,
adding π/3 to -π/2 gives -π/6 , and
adding π/3
to π/2 gives 5π/6 .
That ’s
correct .
So D is
incorrect because the
interval is
from
-5π/6 ,
which is
more negative .
So that’s not
matching .
So the
answer must be
C .
But
let me
just verify
once more . The
original inequality is log _{ 1 /
20
