---
title: 'Memory Grafting: Scaling Language Model'
id: memory-grafting-scaling-language-model
tags:
- cmp170hx-training-feasibility-a0d4fc
- memory-grafting
- frozen-memory
- engram
- trainable-parameter-budget
created: '2026-10-27T10:11:55.371545Z'
updated: '2026-10-27T19:00:31.077216Z'
source: https://arxiv.org/abs/2605.20948
source_domain: arxiv.org
fetched_at: '2026-10-27T10:11:55.370081Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Memory Grafting (Tsinghua + Microsoft Research Asia, arXiv 2605.20948v1,
  20 May 2026) - the paper that actually introduces FROZEN Engram memory tables. Instead
  of learning the n-gram table from scratch, it runs a donor ''grafting model'' (Qwen3.5-35B-A3B,
  DeepSeek-V2-Lite, or GLM-4.7-Flash) offline over the top-1M most frequent n-grams
  per order (2-, 3-, 4-gram => 3M entries per grafted layer), stores the donor''s
  final-token hidden state in bf16 as a frozen table W_mem, and retrieves it by exact
  longest-match suffix lookup; the table is explicitly NOT updated during recipient
  training - only the recipient-side projections and gates are learned. A hash-based
  Engram table is retained as a TRAINABLE fallback for miss positions, so the memory
  subsystem is only partially frozen. Scale of frozen memory: ~12 GB per grafted layer
  for Qwen3.5-35B-A3B, and the 0.92B-trainable recipient carries approximately 12B
  frozen grafting memory, offloaded. Results: 2.8B-total/0.55B-activated recipient,
  100B tokens - average benchmark 51.95 (MoE) / 52.43 (vanilla Engram) -> 53.86 (MG-Qwen3.5),
  with LAMBADA 42.98->48.19, BoolQ 57.92->62.54, WinoGrande 58.98->60.93. 0.92B-trainable/0.29B-activated
  recipient, 50B tokens: MoE 45.62, vanilla Engram 45.03 (underperforms at this scale),
  MG-DeepSeek 46.42, MG-Qwen3.5 46.98, MG-GLM 45.83. Efficiency: 0.92x training throughput,
  0.95x inference throughput, 0.97x inference memory versus the MoE baseline. Motivation
  stated explicitly: Engram training efficiency declines once the memory module exceeds
  ~70% of total parameters, so frozen grafted memory decouples capacity growth from
  recipient training cost.'
raw_file: raw/memory-grafting-scaling-language-model.pdf
doi: arXiv:2605.20948
utility_score: 18.0
citation_count: 5
venue: arXiv.org
---

Memory Grafting: Scaling Language Model
Pre-training via Offline Conditional Memory
Runxi Cheng1
Yuchen Guan1
Yongxian Wei1
Qianpu Sun1
Qixiu Li1
Sinan Du1
Feng Xiong1
Chun Yuan1†
Yan Lu2†
Yeyun Gong2†
1Tsinghua University,
2Microsoft Reasearch Asia
† Corresponding Authors
crx23@mails.tsinghua.edu.cn;
yegong@microsoft.com
Scaling conditional memory offers a promising way to increase language-model capacity, but existing methods
such as Engram learn large memory tables from scratch during pre-training, making memory scaling expensive
and sometimes ineffective. We propose Memory Grafting, a conditional memory scaling method that utilize
frozen hidden states from a grafting model as conditional n-gram memory. Given frequent local n-grams,
we run the grafting model offline, store final-token hidden representations as memory values, and let the
recipient model retrieve them through exact longest-match suffix lookup. Retrieved memories are adapted
by lightweight projections and gates, while a hash-based Engram fallback preserves coverage for unmatched
contexts. Since the grafting model is only run offline and exact lookup has expected O(1) complexity with
respect to memory-bank size, Memory Grafting expands external latent capacity with limited training and
inference overhead. Experiments under matched recipient architectures and pre-training budgets show that
Memory Grafting improves over both MoE and vanilla Engram baselines. In the 2.8B-scale setting, it improves
the average benchmark score from 51.95 for MoE and 52.43 for vanilla Engram to 53.86. In the 0.92B–scale
setting, all grafting-model variants improve over the baselines, with Qwen3.5-35B-A3B giving the strongest
gains. These results suggest that pretrained models can serve as reusable constructors of external latent
memory, providing a practical step toward scaling future language models beyond trainable parameters alone.
1. Introduction
The rapid progress of large language models [1, 15, 31, 46, 47] over the past few years has been driven in large part
by joint scaling of model parameters and pre-training tokens [20, 25]. As frontier models have grown to hundreds of
billions to trillions of parameters trained on trillions of tokens, the engineering cost of each pre-training run has risen
substantially. In response, recent work has increasingly focused on architectural sparsity, where only a fraction of a
model’s total capacity participates in any individual forward pass. Mixture-of-Experts (MoE) architectures [8, 12, 42]
are a representative instance and underlie a number of recent open-weight systems. Another direction [5, 21, 28]
scales capacity through sparse memory lookup rather than sparse computation, attaching a large external table that is
queried at each token. We refer to this line of work as conditional memory.
Conditional memory offers a scalable way to expand model capacity while adding only limited inference overhead. In
particular, Engram [5] uses deterministic hash-based addressing to support O(1) retrieval per token, and reports that
offloading a 100B-parameter memory only costs around 2% end-to-end inference throughput on a 4B-dense backbone.
However, the training side is less favorable. The memory table is learned from scratch during pre-training, so scaling
it increases the parameters need to be optimized. Empirically, training efficiency began to decline once the memory
module accounts for more than 70% of total parameters. Thus, scaling Engram still demands substantial training-time
computation.
To solve this problem, we revisit what Engram memory stores [5, 26, 41]: an n-gram-indexed table of static embeddings.
Such entries naturally correspond to recurring multi-token constructions – named entities, formulaic phrases, library
calls, and mathematical notation [32] – whose meaning is largely fixed by surface form. A sufficiently pre-trained
model has already learned strong representations for these patterns. If these encodings can be used as conditional
arXiv:2605.20948v1  [cs.CL]  20 May 2026


---

Offline Construction
High Frequency
N-gram Corpus
Forward
Grafting Model
Grafted Memory
(Frozen)
Engram Memory
(Trainable)
Input Token IDs
…
N-gram Hash
Matching
𝒪(1)
Online Forward
Retrieve from
Engram Memory
Retrieve from
Grafted Memory
Hidden State
(Query)
Engram Block
(Adaptive Gating
+ Short Conv.)
Output
Miss
(Engram Fallback)
Match
Random Initialized
Figure 1: Computation pipeline for Memory Grafting
memory, the memory capacity can scale without increasing the recipient’s training cost, while the O(1) lookup that
makes conditional memory efficient at inference remains intact.
Motivated by this observation, we propose Memory Grafting, which constructs conditional memory from a grafting
model rather than learning the memory table entirely from scratch. We run the grafting model once offline over
frequent local n-grams and store the resulting hidden states as frozen memory values. During recipient training and
inference, the model retrieves these values through longest-match exact lookup, falls back to hash-based Engram
memory on misses, and injects the selected memory through lightweight projection and gating. Since the grafting-model
computation is paid once offline lightly, the input sequence length is unchanged and the per-token lookup remains
O(1), the conditional memory can be scaled while keeping both training and inference efficient. Our contributions are
summarized as follows:
• We introduce Memory Grafting, a representation-level knowledge-transfer paradigm for scaling conditional
memory. It stores frozen grafting-model hidden states in an exact n-gram-indexed table, retains a hash-based
Engram fallback for unmatched contexts, and grafts retrieved memories through a gated residual update. The
grafting-model cost is paid once offline lightly, and per-token lookup remains O(1).
• We evaluate Memory Grafting under matched recipient architectures and pre-training budgets. The results show
gains over MoE and vanilla Engram baselines, increasing exact-memory coverage as the table grows, practical
inference efficiency, and probing evidence that the recipient actually uses the grafted memory.
2. Related Work
2.1. Conditional Memory and Modular Sparse Scaling
Efficiently scaling model capacity is crucial for advancing the performance of large language models. Sparse scaling and
conditional memory represent two prominent directions in this pursuit. Mixture-of-Experts architectures [8, 12, 42],
which activate only a fraction of the total capacity during any single forward pass, have become a widely adopted
sparse paradigm in recent open-weight systems. Parallel to sparse computation, another line of research focuses on
scaling capacity through sparse memory lookups rather than computation, a mechanism collectively termed conditional
2


---

memory [5, 44, 45]. For instance, Product-Key Memory [28] replaces standard feed-forward layers with a massive
key-value store, enabling sparse nearest-neighbor search via Cartesian product sets. Building on this, UltraMem
[21] resolves sparse retrieval bottlenecks and scaling inefficiencies through tucker decomposed query-key retrieval
and implicit value expansion. STEM [39] replaces the FFN up-projection in Transformer layers with token-indexed
embedding modules, reducing per-token computation while increasing parameter capacity through input-dependent
embedding lookup. MoLE [24] improves MoE inference efficiency by training experts as FFNs and re-parameterizing
them into input-id-based lookup tables before inference, thereby reducing expert computation and offloading overhead
while preserving MoE-like capacity. More recently, Engram [5] learns an n-gram vocabulary as its memory module,
employing hash-based addressing to achieve O(1) sparse retrieval during the forward pass. Such conditional memory
offers a clear inference-side advantage, but scaling it still requires substantial computation when the memory itself is
learned from scratch during pre-training.
2.2. Retrieval-Augmented Generation and Knowledge Distillation
Beyond conditional memory, another line of work improves language models by leveraging external memory sources
in different ways. Retrieval-Augmented Generation [29] queries external non-parametric memories or documents
during training or inference, with pioneering works such as REALM [19] casting both pre-training and fine-tuning as an
end-to-end retrieve-then-predict process, and recent work [22, 30] further refining retrieval reasoning and extraction
quality. In contrast, Knowledge Distillation transfers the parametric memory of a larger teacher into a recipient model,
improving its capability without explicit retrieval at inference time [16, 35].
Although effective, RAG and KD introduce different efficiency bottlenecks. RAG relies on retrieval and longer augmented
contexts at inference, shifting substantial overhead to deployment and increasing latency [14, 23, 57]. KD instead
internalizes external knowledge into the student’s parameters, but logits-based distillation during from-scratch pre-
training requires running the teacher across the entire corpus and consuming large soft-label distributions, incurring
substantial compute and memory cost. In contrast, Memory Grafting needs only a lightweight one-time offline pass to
build the grafted memory table, preserving high efficiency in both training and inference and showing strong potential
for model scaling.
3. Method
3.1. Preliminaries on Engram
Engram [5] is a conditional memory module for language models. Unlike MoE, which sparsely activates computation
experts, Engram sparsely retrieves memory indexed by local token patterns. Given a token sequence X = (x1, . . . , xT)
and hidden states H(ℓ) ∈RT×D at layer ℓ, Engram performs two steps: hashed N-gram retrieval and context-aware
fusion.
For each position t, Engram forms suffix n-grams from the local history. Following the original design, token ids may
first be mapped through a vocabulary compression function P : V →V′, so that textually equivalent tokens share a
canonical id. The compressed suffix n-gram is:
gt,n =
(︀
P(xt−n+1), . . . , P(xt)
)︀
,
n ∈{2, . . . , N}.
(1)
Because enumerating all possible N-grams is infeasible, Engram uses multi-head hashing. For each order n and hash
head k, a deterministic hash function ϕn,k maps gt,n to an embedding-table row:
zt,n,k = ϕn,k(gt,n),
et,n,k = En,k[zt,n,k].
(2)
The retrieved embeddings across all n-gram orders and hash heads are concatenated into a static memory vector:
et =
N
∥
n=2
K
∥
k=1
et,n,k.
(3)
Since et is retrieved only from the local token pattern, it is not by itself aware of the global context. Engram therefore
uses the current hidden state as a query to gate the retrieved memory. For a branch c of the residual stream, let
3


---

Algorithm 1: Memory Grafting with Engram Fallback
1: Input: Hidden states H ∈RB×T×C×D, tokens X, frozen memory (M, Wmem), trainable Engram table Whash,
projections Wmem
k,v
, Weng
k,v
2: Output: Grafted hidden states̃︀H
3: Mmem ←0, m ←0
// Initialize memory features and hit mask
4: for each token position (b, t) do
5:
j ←ExactLookup(Xb,≤t, M)
// Select the largest matched n-gram
6:
if j exists then
7:
Mmem[b, t] ←Wmem[j], m[b, t] ←1
8:
end if
9: end for
10: for each token position (b, t) do
11:
if m[b, t] = 1 then
12:
eb,t ←Mmem[b, t],
(Kb,t, Vb,t) ←(Wmem
k
eb,t, Wmem
v
eb,t)
// Memory grafting path
13:
else
14:
zb,t ←NgramHashMapping(Xb,≤t)
// Generate deterministic hash ids
15:
Eb,t ←MultiHeadEmbedding(zb,t)
// Lookup trainable Engram table
16:
eb,t ←Concat(Eb,t)
17:
(Kb,t, Vb,t) ←(Weng
k
eb,t, Weng
v
eb,t)
// Engram fallback path
18:
end if
19: end for
20: Q ←RMSNormq(H),
¯K ←RMSNormk(K)
21: α ←σ(⟨¯K, Q⟩/
√
D)
// Query-key gate
22: U ←α ⊙V,
∆H ←U + ShortConv(U)
23:̃︀H ←H + ∆H
24: returñ︀H
h(c)
t
∈RD be the hidden state. The memory vector is projected into key and value vectors:
k(c)
t
= W(c)
K et,
vt = WVet.
(4)
The gate is computed by comparing the normalized key with the normalized query [5, 11, 56]:
α(c)
t
= σ
⎛
⎝
⟨
RMSNorm(k(c)
t ), RMSNorm(h(c)
t )
⟩
√
D
⎞
⎠
(5)
The gated value is then processed by a lightweight depthwise causal convolution:
u(c)
t
= α(c)
t vt,
∆h(c)
t
= u(c)
t
+ ShortConv(U(c))t.
(6)
3.2. Memory Grafting
Original Engram learns a hash-indexed memory table from scratch. Memory Grafting instead constructs an external
memory bank from latent representations of a grafting model and grafts the retrieved features into a recipient model.
This changes the role of conditional memory from purely trainable storage to reusable representation-level knowledge
transfer.
Frozen Latent Memory Construction.
We first build a set of frequent n-gram keys M = {mi}M
i=1, where each mi is
a token-id tuple such as a 2-, 3-, or 4-gram. For each key mi, we run a grafting model FG offline and extract a hidden
4


---

representation of its original text from a selected grafting layer. In our implementation, we use the hidden state at the
final token of the n-gram as the memory embedding. This produces a frozen table
Wmem[i] = FG(text(mi))(r)
last ∈RDmem,
(7)
where r is the grafting layer used for extraction. The mapping from mi to row index i is stored as an exact integer
lookup structure. The table Wmem is not updated during recipient-model training; only the recipient-side projection
and gating parameters are learned.
Exact Retrieval with Longest-Match Priority.
At a grafting layer of the recipient model, let H ∈RB×T×C×D denote
the hidden states, where B is the batch size, T is the sequence length, C is the number of residual branches, and D
is the branch dimension. For each position (b, t), Memory Grafting queries the frozen memory using suffix n-grams
ending at xb,t. If multiple keys match the same position, we keep the longest matched n-gram, e.g., a 4-gram match
overrides a 3-gram match. This yields a memory feature Emem,b,t and a binary hit mask mb,t ∈{0, 1}:
(jb,t, mb,t) = ExactLookup(Xb,≤t, M),
Emem,b,t = Wmem[jb,t]
if mb,t = 1.
(8)
The exact lookup is independent of the total memory size in expectation under hash-map or trie-style indexing, and
costs only a constant number of suffix queries per token because the maximum n-gram order is fixed.
Memory Grafting with Engram Fallback.
The frozen memory covers only the selected frequent n-grams. To preserve
full coverage over arbitrary inputs, we retain the original hash-based Engram path as a fallback. For every position, we
compute a trainable hash Engram feature:
Ehash,b,t = Concat
(︀MultiHeadEmbedding(NgramHashMapping(Xb,≤t))
)︀
.
(9)
Memory-hit positions use the frozen grafted memory, whereas memory-miss positions use the trainable Engram fallback.
We use separate projections for the two sources:
Kc
mem,b,t = Wmem,c
k
Emem,b,t,
Vmem,b,t = Wmem
v
Emem,b,t,
(10)
Kc
eng,b,t = Weng,c
k
Ehash,b,t,
Veng,b,t = Weng
v
Ehash,b,t.
(11)
The branch-specific key and shared value used by the gate are selected by the hit mask:
Kc
b,t = mb,tKc
mem,b,t + (1 −mb,t)Kc
eng,b,t,
Vb,t = mb,tVmem,b,t + (1 −mb,t)Veng,b,t.
(12)
Gated Grafting.
The selected memory value is grafted into the decoder using the same context-aware gating principle
as Engram. For each branch c, we compute
αc
b,t = σ
⎛
⎝
⟨
¯Kc
b,t, Qc
b,t
⟩
√
D
⎞
⎠, where Qc
b,t = RMSNormq(Hc
b,t), ¯Kc
b,t = RMSNormk(Kc
b,t)
(13)
The output can be calculated as:
Uc
b,t = αc
b,tVb,t,
∆H = U + ShortConv(U),̃︀
H = H + ∆H.
(14)
4. Experiments
4.1. Experimental Setup
Model Architectures. We implement two recipient model scales [48], with total parameters around 0.9B and 2.8B. For
both scales, the MoE layers contain 64 routed experts, and following DeepSeekMoE [8], we use one shared expert in
5


---

Table 1: Benchmark performance (%) in the 2.8B-trainable, 100B-token setting.
Benchmark (Metric)
# Shots
MoE Baseline
Vanilla Engram
Memory Grafting
# Trainable Params
2.8B
2.8B
2.8B
# Activated (w/o token embed)
0.55B
0.55B
0.55B
# Trained Tokens
100B
100B
100B
# Experts (shared + routed, top-k)
1 + 64 (top-4)
1 + 48 (top-4)
1 + 47 (top-4)
ARC-Challenge
0-shot
36.12
36.09
37.03
ARC-Easy
0-shot
72.39
71.80
73.40
BoolQ
0-shot
57.92
60.24
62.54
Social IQA
0-shot
41.61
41.76
42.94
RACE
0-shot
34.74
34.55
35.98
LAMBADA
5-shot
42.98
45.22
48.19
WinoGrande
5-shot
58.98
59.66
60.93
PIQA
5-shot
75.41
75.63
76.17
HellaSwag
10-shot
47.39
46.94
47.54
Average
51.95
52.43
53.86
Table 2: Result in the 0.92B-trainable, 50B-token setting with different grafting models. MG-X denotes Memory Grafting
using model X as the grafting model: MG-GLM refers to GLM-4.7-Flash, MG-Deepseek refers to DeepSeek-V2-Lite, and
MG-Qwen3.5 refers to Qwen3.5-35B-A3B.
Benchmark (Metric)
# Shots
MoE Baseline
Vanilla Engram
MG-GLM
MG-Deepseek
MG-Qwen3.5
# Trainable Params
0.92B
0.92B
0.92B
0.92B
0.92B
# Activated (w/o token embed)
0.29B
0.29B
0.29B
0.29B
0.29B
# Trained Tokens
50B
50B
50B
50B
50B
# Experts (shared + routed, top-k)
1 + 64 (top-4)
1 + 48 (top-4)
1 + 46 (top-4) 1 + 46 (top-4) 1 + 46 (top-4)
ARC-Challenge
0-shot
27.56
28.92
28.07
28.92
29.35
ARC-Easy
0-shot
64.31
64.31
62.96
64.39
64.06
BoolQ
0-shot
55.93
46.97
53.03
56.64
58.10
Social IQA
0-shot
39.36
40.28
40.12
40.48
40.84
RACE
0-shot
31.77
31.39
30.05
30.62
31.58
LAMBADA
5-shot
31.88
32.27
34.37
32.52
33.73
WinoGrande
5-shot
51.07
51.85
53.51
53.51
55.01
PIQA
5-shot
70.67
70.95
71.87
71.93
71.44
HellaSwag
10-shot
38.02
38.34
38.48
38.80
38.74
Average
45.62
45.03
45.83
46.42
46.98
each MoE layer. For the Engram module, we follow the best parameter allocation reported in the Engram paper and
set the parameter ratio between Engram memory and MoE computation to 1:3. Since Memory Grafting introduces
additional lightweight projection parameters for adapting the frozen memory features to the recipient hidden space,
we reduce the corresponding number of experts to keep the fair comparison. Across all experiments, we ensure that
the number of trainable parameters follows Memory Grafting ≤vanilla Engram ≤MoE baseline.
Data & Tokenizer. We trained the 925M parameter model on a 50B-token subset of the Nemotron-CC dataset [43]
and the 2.81B parameter model on a 100B-token subset of the Nemotron-CC dataset. All text was tokenized with the
6


---

10
20
Training Tokens (B)
8000
10000
12000
14000
16000
18000
20000
Training Step
-0.01
-0.01
-0.01
0.00
0.01
0.01
0.02
0.02
Δ Loss vs. Engram
(a) Δ Loss across different donor layers
MoE
Layer 8
Layer 16
Layer 24
Layer 32
Layer 40
Random
10
20
30
40
50
Training Tokens (B)
10000
20000
30000
40000
50000
Training Step
-0.02
-0.01
0.00
0.01
0.02
Δ Loss vs. Engram
(b) Δ Loss across different donor models
MoE
Qwen3.5 Layer 8+24
DeepSeek Layer 4+18
GLM Layer 4+28
Figure 2: Loss differences relative to vanilla Engram when varying the grafted memory source. (a) Comparison across
different Qwen3.5-35B-A3B source layers (b) Comparison across different grafting models.
LLaMA-3-8B tokenizer [17].
Grafting Setting. We use Qwen3.5-35B-A3B [36], DeepSeek-V2-Lite [10], and GLM-4.7-Flash [55] as grafting models.
For each n-gram order, we select the top 1M most frequent n-grams and compute their frozen memory representations
offline, yielding 3M entries per grafted layer. Each entry stores a hidden-state vector of the grafting model in bfloat16;
for Qwen3.5-35B-A3B , this corresponds to roughly 12 GB of frozen memory per layer. To cover different representation
levels of the grafting model, we extract hidden states from an early-to-middle layer and a deeper layer, approximately
corresponding to 10% and 60% of the model depth. In particular, we use Layers 8, 24 for Qwen3.5-35B-A3B, Layers 4,
18 for DeepSeek-V2-Lite, and Layers 4, 28 for GLM-4.7-Flash.
Hyper-Parameters. The hyper-parameters are selected based on the common practice for dense language models. We
replace all FFN layers with MoE layer in the transformer. Please refer Section A for detailed training hyper-parameters.
Benchmarks. We use the lm-evaluation-harness [13] for evaluation. The benchmarks used include ARC-C [7],ARC-E[7],
BoolQ [6], HellaSwag [54], LAMBADA [34], PIQA [2], RACE [27], WinoGrande [40], .
4.2. Main Results
We report two main sets of experimental results. Table 1 uses a 2.8B total / 0.55B activated MoE backbone trained
for 100B tokens and compares the MoE baseline, vanilla Engram, and Memory Grafting with a Qwen3.5-35B-A3B
grafting model. Memory Grafting is best on all nine benchmarks: the average improves from 51.95 for MoE and 52.43
for vanilla Engram to 53.86. The largest gains appear on LAMBADA, from 42.98 to 48.19; BoolQ, from 57.92 to 62.54;
and WinoGrande, from 58.98 to 60.93. Since the grafting memory is frozen and produced offline, and the recipient
keeps matched trainable parameters and activated parameters, the result demonstrate the effectiveness of memory
grafting.
Table 3 fixes a 0.92B trainable / 0.29B activated recipient trained for 50B tokens and varies only the grafting model.
All grafting variants improve average accuracy over the baselines, but the gains are uneven: MG-GLM gives only a
small improvement, from 45.62 for MoE to 45.83, whereas MG-Deepseek and MG-Qwen3.5 reach 46.42 and 46.98,
respectively. We also observe that vanilla Engram underperforms the MoE baseline in this 1B-scale setting, with 45.03
versus 45.62. These may caused by the limited Engram embedding table in 0.9B model, which lead to more hash
7


---

0
1
2
3
4
5
6
7
8
9
10
11
12
MoE Baseline Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MoE Baseline
0
1
2
3
4
5
6
7
8
9
10
11
12
Vallina Engram Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs Vallina Engram
0
1
2
3
4
5
6
7
8
9
10
11
12
MG-Qwen Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MG-Qwen
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
Figure 3: CKA comparison between Qwen3.5-35B-A3B and the MoE baseline, vanilla Engram, and Memory Grafting
model. The white trajectory marks the best-matching recipient layer for each Qwen3.5-35B-A3B layer.
collisions. This suggests that Memory Grafting benefits from external latent memory, but the quality of the grafting
model’s n-gram representations matters. Appendix B.1 analyzes the memory geometry and shows that GLM-4.7-Flash
memory is much more anisotropic and less discriminative, which helps explain why its downstream gain is limited.
4.3. Further Analysis
4.3.1. Impact of the Grafting-Model Source Layer
Beyond the choice of grafting model, we find that the grafting-model layer used to construct the frozen memory is
also important. In this analysis, we keep the recipient model, training data, memory keys, and recipient-side grafting
module fixed, and vary only the source layer from which the grafting-model n-gram embeddings are extracted. The loss
trend shows that memory built from an intermediate grafting-model layer, especially around Layer 6 in the compared
setting, gives the strongest benefit. When the memory embeddings are extracted from deeper grafting-model layers,
the loss can instead increase.
This result suggests that Memory Grafting requires a compatible match between the representation level of the external
memory and the recipient model’s own hidden states. Earlier or middle-layer grafting-model embeddings are more
likely to encode local lexical and phrase-level n-gram information that a small recipient model can align with. Deeper
grafting-model embeddings may be more abstract and more tightly coupled to the internal computation of the large
grafting model. For a smaller recipient, these high-level n-gram representations can be harder to interpret through
a lightweight projection and gate, making the injected signal less useful or even harmful. Therefore, the benefit of
Memory Grafting depends not only on whether external memory is available, but also on choosing a grafting-model
source layer whose representations the recipient can effectively understand.
4.3.2. CKA Analysis of Memory Grafting
We use centered kernel alignment (CKA) [9] to analyze how Memory Grafting changes the recipient representation
space, as shown in Figure 3. The heatmaps compare Qwen3.5-35B-A3B layers with the layers of the MoE baseline,
vanilla Engram, and Memory Grafting model, and the white trajectory marks the best-matching recipient layer for
each grafting-model layer. Compared with the two baselines, Memory Grafting shows a clearer alignment with deeper
Qwen3.5-35B-A3B layers at shallower recipient layers. In other words, after memory grafting, the recipient begins to
exhibit functions similar to the deep Qwen representations earlier in its own layer stack, whereas the MoE baseline
and vanilla Engram show weaker or later alignment. This indicates that the grafted memory leaves a measurable
signature in the recipient hidden states, which supports the intended mechanism of Memory Grafting. Specifically, the
recipient-side projection and gate allow the recipient model to effectively incorporate feature patterns injected from
the grafting model, thereby aligning its internal representations with the grafted memory.
4.3.3. Ablation of Grafting Mechanism
8


---

10
15
20
Training Tokens (B)
10000
15000
20000
Training Step
-0.03
-0.02
-0.01
Δ Loss vs. Engram
Gating-mechanism ablation
w/o Gate
w/ Gate v1
w/ Gate v2
w/ Gate v2 + Engram fallback
5
10
Training Tokens (B)
6000
8000
10000
Training Step
2.94
2.96
2.98
3.00
3.02
3.04
3.06
3.08
3.10
Training Loss
Effect of n-gram memory capacity on training loss
|| = 30k
|| = 300k
|| = 3M
Figure 5: Ablation of the Memory Grafting design and the grafted n-gram memory capacity.
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
1.1
Relative to MoE Baseline
↑ higher is better
Training Throughput
1.00×
0.94×
0.92×
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
1.1
Relative to MoE Baseline
↑ higher is better
Inference Throughput
1.00×
0.96×
0.95×
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
1.1
Relative to MoE Baseline
↓ lower is better
Inference Memory
1.00×
0.99×
0.97×
MoE Baseline
Vanilla Engram
Memory Grafting
Figure 4: Throughput and memory usage comparison.
The model of Memory Grafting has 0.92B trainable pa-
rameters with approximately 12B frozen grafting memory
offloaded.
We compare four memory grafting mechanisms in Figure 5:
directly grafting multiple matched n-gram embeddings
through attention, adding a gate after this attention aggre-
gation, grafting only the longest matched n-gram through
a gate, and further adding the Engram fallback to the
gated longest-match mechanism. The two attention-based
variants improve over vanilla Engram, but their gains are
limited. Selecting only the longest matched n-gram and
grafting it through a gate is more effective because the
selected memory is more specific and easier to regulate.
The best result is obtained by further adding the Engram
fallback to this gated longest-match grafting mechanism.
This variant consistently achieves the largest loss reduction
relative to vanilla Engram. Since the frozen grafting mem-
ory is built only for high-frequency n-grams, it cannot cover all tokens through exact matching. Using only this memory
may therefore introduce a hit-token bias during training: tokens with matched high-frequency contexts receive grafting
signals, while unmatched or low-frequency contexts receive no external memory. The Engram fallback mitigates this
coverage bias by providing a trainable memory path for missed tokens and by retaining additional low-frequency
n-gram memories. Therefore, the final Memory Grafting mechanism combines adaptive gating, efficient longest-match
grafting, and engram fallback mechanism.
4.3.4. Impact of Grafted Memory Capacity
For exploring the scaling potential of memory grafting, We compare the training loss across different grafted memory
size. As shown in the right of Figure 5, larger memory capacity consistently leads to lower training loss. Increasing the
memory size from 30k to 300k and further to 3M improves the loss curve throughout training, indicating a clear scaling
trend: grafting more memory provides more useful exact-match contexts and improves the recipient model more
strongly. The result demonstrate the benefit of Memory Grafting can be scaled with the size of the grafted memory.
4.3.5. Efficiency Analysis
We report the efficiency comparison in Figure 4. Memory Grafting keeps the runtime cost close to the MoE baseline,
reaching 0.92× training throughput and 0.95× inference throughput, while using 0.97× inference memory. Although
9


---

Memory Grafting introduces an external memory bank that is several times larger than the recipient model, the resulting
slowdown is limited. The main reason is that exact n-gram retrieval is implemented as a hash lookup, whose expected
complexity is O(1) with respect to the memory-bank size. The expensive grafting-model computation is also performed
offline, so the online path only adds lookup, projection, gating, and the memory update. The offline construction cost
is also modest in our setting: extracting a 3M n-gram memory from Qwen3.5-35B-A3B requires only about 3 A100
GPU-hours. Therefore, the result suggests that Memory Grafting can scale to substantially larger memory banks while
adding only modest offline construction cost and runtime overhead.
5. Conclusion
In this paper, we introduced Memory Grafting, a conditional memory scaling method that turns frozen grafting-model
hidden states into conditional n-gram memory for a smaller recipient model. The key idea is to move the expensive
grafting-model computation offline: the recipient performs exact longest-match retrieval over the frozen memory bank,
projects and gates the retrieved latent representation, and falls back to trainable Engram memory when no exact
match is available. Under matched pre-training settings, this design improves over both MoE and vanilla Engram
baselines, and the gains are stronger when the grafted memory comes from more discriminative grafting models such
as Qwen3.5-35B-A3B and DeepSeek-V2-Lite. The analysis further shows that the retrieved memories are actively used
by the recipient and that increasing memory capacity improves training loss. Since exact memory retrieval has expected
O(1) complexity with respect to memory-bank size, Memory Grafting mainly expands external latent capacity rather
than per-token activated computation, enabling efficient training and inference. We therefore view it as a practical step
toward scaling future language models through reusable external latent memory beyond trainable parameters alone.
10


---

References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.
[2] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in
natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439,
2020.
[3] Rongyi Chen, Jingjia Xiao, Zilu Wang, Menghan Yin, Xianzhe Fan, Zihe Ran, and Qing Xiao. Exploring influencers’
and users’ experiences in douyin’s virtual reality live-streaming. In Proceedings of the 30th ACM Symposium on
Virtual Reality Software and Technology, pages 1–2, 2024.
[4] Runxi Cheng, Feng Xiong, Yongxian Wei, Wanyun Zhu, and Chun Yuan. Whoever started the interference should
end it: Guiding data-free model merging via task vectors. arXiv preprint arXiv:2503.08099, 2025.
[5] Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu,
Zhewen Hao, Yukun Li, et al. Conditional memory via scalable lookup: A new axis of sparsity for large language
models. arXiv preprint arXiv:2601.07372, 2026.
[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova.
Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 conference
of the north American chapter of the association for computational linguistics: Human language technologies,
volume 1 (long and short papers), pages 2924–2936, 2019.
[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,
2018.
[8] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai
Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 1280–1297, 2024.
[9] MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. Reliability
of cka as a similarity measure in deep learning. arXiv preprint arXiv:2210.16156, 2022.
[10] DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
[11] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approxi-
mation in reinforcement learning. Neural networks, 107:3–11, 2018.
[12] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with
simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
[13] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang,
Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin
Wang, and Andy Zou. The language model evaluation harness, 07 2024.
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
Haofen Wang, et al. Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997, 2(1):32, 2023.
[15] Team Glm, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng,
Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint
arXiv:2406.12793, 2024.
[16] Sachin Goyal, David Lopez-Paz, and Kartik Ahuja. Distilled pretraining: A modern lens of data, in-context learning
and test-time scaling. arXiv preprint arXiv:2509.01649, 2025.
[17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint
arXiv:2407.21783, 2024.
[18] Yuchen Guan, Runxi Cheng, Kang Liu, and Chun Yuan. Enhancing logits distillation with plug&play kendall’s τ
ranking loss, 2025.
[19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language
model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.
[20] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, DDL
Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models.
11


---

arXiv preprint arXiv:2203.15556, 10, 2022.
[21] Zihao Huang, Qiyang Min, Hongzhi Huang, Defa Zhu, Yutao Zeng, Ran Guo, and Xun Zhou. Ultra-sparse memory
network. arXiv preprint arXiv:2411.12364, 2024.
[22] Shayekh Bin Islam, Md Asib Rahman, KSM Tozammel Hossain, Enamul Hoque, Shafiq Joty, and Md Rizwan
Parvez. Open-rag: Enhanced retrieval augmented reasoning with open-source large language models. In Findings
of the Association for Computational Linguistics: EMNLP 2024, pages 14231–14244, 2024.
[23] Wenqi Jiang, Marco Zeller, Roger Waleffe, Torsten Hoefler, and Gustavo Alonso. Chameleon: a heterogeneous and
disaggregated accelerator system for retrieval-augmented language models. arXiv preprint arXiv:2310.09949,
2023.
[24] Shibo Jie, Yehui Tang, Kai Han, Yitong Li, Duyu Tang, Zhi-Hong Deng, and Yunhe Wang. Mixture of lookup
experts. arXiv preprint arXiv:2503.15798, 2025.
[25] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Rad-
ford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,
2020.
[26] Reinhard Kneser and Hermann Ney. Improved backing-off for m-gram language modeling. In 1995 international
conference on acoustics, speech, and signal processing, volume 1, pages 181–184. IEEE, 1995.
[27] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension
dataset from examinations. In Proceedings of the 2017 conference on empirical methods in natural language
processing, pages 785–794, 2017.
[28] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Large
memory layers with product keys. Advances in Neural Information Processing Systems, 32, 2019.
[29] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in neural information processing systems, 33:9459–9474, 2020.
[30] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and
Weinan Zhang. Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension
in recommendation. In Proceedings of the ACM Web Conference 2024, pages 3497–3508, 2024.
[31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng,
Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
[32] Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, and Hannaneh Hajishirzi. Infini-gram: Scaling unbounded
n-gram language models to a trillion tokens. arXiv preprint arXiv:2401.17377, 2024.
[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101,
2017.
[34] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle,
Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad
discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics
(volume 1: Long papers), pages 1525–1534, 2016.
[35] Hao Peng, Xin Lv, Yushi Bai, Zijun Yao, Jiajie Zhang, Lei Hou, and Juanzi Li. Pre-training distillation for large
language models: A design space exploration. In Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 3603–3618, 2025.
[36] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.
[37] Zihe Ran, Xiyu Li, Qing Xiao, Xianzhe Fan, Franklin Mingzhe Li, Yanyun Wang, and Zhicong Lu. How users who
are blind or low vision play mobile games: Perceptions, challenges, and strategies. In Proceedings of the 2025
CHI Conference on Human Factors in Computing Systems, pages 1–18, 2025.
[38] Zihe Ran, Xiyu Li, Qing Xiao, Yanyun Wang, Franklin Mingzhe Li, and Zhicong Lu. Understanding how visually
impaired players socialize in mobile games. In Proceedings of the 27th International ACM SIGACCESS Conference
on Computers and Accessibility, pages 1–16, 2025.
[39] Ranajoy Sadhukhan, Sheng Cao, Harry Dong, Changsheng Zhao, Attiano Purpura-Pontoniere, Yuandong
Tian, Zechun Liu, and Beidi Chen.
Stem: Scaling transformers with embedding modules.
arXiv preprint
arXiv:2601.10639, 2026.
[40] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd
schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.
[41] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–
423, 1948.
[42] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Out-
12


---

rageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538,
2017.
[43] Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad
Shoeybi, and Bryan Catanzaro. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining
dataset. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pages 2459–2475, 2025.
[44] Gemma Team. Gemma 3n. 2025.
[45] Gemma Team. Gemma 4. 2026.
[46] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv
preprint arXiv:2312.11805, 2023.
[47] Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao
Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
[48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
[49] Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao,
and Dacheng Tao. Unifying multimodal large language model capabilities and modalities via model merging.
arXiv preprint arXiv:2505.19892, 2025.
[50] Yongxian Wei, Anke Tang, Li Shen, Zixuan Hu, Chun Yuan, and Xiaochun Cao. Modeling multi-task model
merging as adaptive projective gradient descent. arXiv preprint arXiv:2501.01230, 2025.
[51] Yongxian Wei, Yilin Zhao, Li Shen, Xinrui Chen, Runxi Cheng, Sinan Du, Hao Yu, Xiaohan Wang, Gang Liu,
Jiahong Yan, et al. Learning to pose problems: Reasoning-driven and solver-adaptive data synthesis for large
reasoning models. arXiv preprint arXiv:2511.09907, 2025.
[52] Feng Xiong, Runxi Cheng, Wang Chen, Zhanqiu Zhang, Yiwen Guo, Chun Yuan, and Ruifeng Xu. Multi-task
model merging via adaptive weight disentanglement. arXiv preprint arXiv:2411.18729, 2024.
[53] Feng Xiong, Hongling Xu, Yifei Wang, Runxi Cheng, Yong Wang, and Xiangxiang Chu. Hs-star: Hierarchical sam-
pling for self-taught reasoners via difficulty estimation and budget reallocation. arXiv preprint arXiv:2505.19866,
2025.
[54] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish
your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pages
4791–4800, 2019.
[55] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao
Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint
arXiv:2508.06471, 2025.
[56] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing
systems, 32, 2019.
[57] Zhihao Zhang, Alan Zhu, Lijie Yang, Yihua Xu, Lanting Li, Phitchaya Mangpo Phothilimthana, and Zhihao Jia.
Accelerating retrieval-augmented language model serving with speculation. arXiv preprint arXiv:2401.14021,
2024.
13


---

Appendix
Contents
A
Experiment Details
15
A.1
Model Architecture and Hyper Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.2 Construction of Grafting-Model Memory
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A.3 Calculate resources and environment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A.4 Appendix: Details of the Grafting Gate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B
Additional Experiment
19
B.1
Additional Analysis of Grafting-Model Memory Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.2
Loss Curves for Memory Grafting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.3
More CKA Analysis Results of Memory Grafting
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
B.4
Top-k N-gram Hit Rate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
B.5
Single-Layer Memory Grafting Results
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
B.6
Probe Analysis
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
C
Limitations & Discussion
25
C.1
Limitations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
C.2
Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
D LLM Usage
25
14


---

A. Experiment Details
A.1. Model Architecture and Hyper Parameters
Table 3: Architecture and training hyper-parameters for the 0.92B-class models.
MoE-1B
Engram-1B
MG-1B
Total Params
0.92B
0.92B
0.92B
Active Params
0.29B
0.29B
0.29B
Total Tokens
50B
50B
50B
Layers
12
12
12
Dimension
768
768
768
MoE Inter. Size
384
384
384
Routed Experts
64
48
46
Active Experts
4
4
4
Shared Experts
1
1
1
Heads
16
16
16
Sequence Length
2048
2048
2048
Vocab Size
128256
128256
128256
Training Steps
50000
50000
50000
Optimizer
AdamW
AdamW
AdamW
Base LR
5e-4
5e-4
5e-4
LR Scheduler
Cosine w/ warmup
Cosine w/ warmup
Cosine w/ warmup
Weight Decay
0.1
0.1
0.1
Engram Dim dmem
-
384
384
Engram N-gram
-
[2,3]
[2,3] †
Engram Num Head
-
8
8
Engram Vocab Size
-
112865
118265
Engram Layer (target)
-
[1, 6]
[1, 6]
Engram ShortConv Kernel
-
4
4
Engram Weight Decay
-
0.1
0.1
Engram Optimizer
-
AdamW [33]
AdamW
Memory Grafting Model
-
-
Qwen3.5-35B-A3B
Memory Grafting Source Layers
-
-
[8, 24]
Memory Grafting Recipient Layers
-
-
[1, 12]
Memory Grafting Entries per Layer
-
-
3,000,000
Memory Grafting Hidden Dim
-
-
2048
Memory Grafting Dtype
-
-
bfloat16
Memory Grafting Size per Layer
-
-
≈12 GB
Memory Grafting Total Frozen Memory
-
-
≈24 GB
Memory Grafting Trainable
-
-
False (frozen)
Memory Grafting Retrieval
-
-
Exact longest-match (2/3/4-gram)
Memory Grafting Fallback
-
-
Hash Engram (2/3-gram)
15


---

Table 4: Architecture and training hyper-parameters for the 2.8B-class models.
MoE-2.8B
Engram-2.8B
MG-2.8B
Total Params
2.8B
2.8B
2.8B
Active Params
0.55B
0.55B
0.55B
Total Tokens
100B
100B
100B
Layers
24
24
24
Dimension
1024
1024
1024
MoE Inter. Size
512
512
512
Routed Experts
64
48
47
Active Experts
4
4
4
Shared Experts
1
1
1
Heads
16
16
16
RoPE θ
10000
10000
10000
Sequence Length
2048
2048
2048
Vocab Size
128256
128256
128256
Training Steps
50000
50000
50000
Optimizer
AdamW
AdamW
AdamW
Base LR
5e-4
5e-4
5e-4
LR Scheduler
Cosine w/ warmup
Cosine w/ warmup
Cosine w/ warmup
Weight Decay
0.1
0.1
0.1
Engram Dim dmem
-
512
512
Engram N-gram
-
[2,3]
[2,3]
Engram Num Head
-
8
8
Engram Vocab Size
-
292680
292680
Engram Layer (target)
-
[1, 12]
[1, 12]
Engram ShortConv Kernel
-
4
4
Engram Weight Decay
-
0.1
0.1
Engram Optimizer
-
AdamW
AdamW
Memory Grafting Model
-
-
Qwen3.5-35B-A3B
Memory Grafting Source Layers
-
-
[8, 24]
Memory Grafting Recipient Layers
-
-
[1, 12]
Memory Grafting Entries per Layer
-
-
3,000,000
Memory Grafting Hidden Dim
-
-
2048
Memory Grafting Dtype
-
-
bfloat16
Memory Grafting Size per Layer
-
-
≈12 GB (≈6.1B params)
Memory Grafting Total Frozen Memory
-
-
≈24 GB (≈12.3B params)
Memory Grafting Trainable
-
-
False (frozen)
Memory Grafting Retrieval
-
-
Exact longest-match (2/3/4-gram)
Memory Grafting Fallback
-
-
Hash Engram (2/3-gram)
16


---

A.2. Construction of Grafting-Model Memory
We construct the frozen grafting-model memory from frequent local n-grams before recipient training. The input table
stores the surface text of each selected n-gram, together with the recipient-side lookup key used to index the memory
table. Following the notation in Section 3.2, the selected memory keys are
M = {mi}M
i=1,
mi = (xi,1, . . . , xi,ni),
ni ∈{2, 3, 4},
(15)
where each mi is the exact lookup key used by the recipient model. The implementation stores a map from this key to
the memory row index i.
To build the memory values, we run the grafting model offline on the surface text of each n-gram. In the Qwen3.5-
35B-A3B implementation, we tokenize the n-gram text with the Qwen tokenizer, run the text backbone, and extract
the hidden state at the last non-padding token from selected grafting-model layers. These vectors are written to disk
and then packed into frozen nn.Embedding tables in bfloat16 format. The same construction also saves a map from
the recipient-side n-gram id to the row index of the stored vector. At runtime, the recipient only performs exact lookup
over M, obtains the corresponding frozen vector, and uses learned projections and gates to adapt it to the recipient
hidden space. Equivalently, for a selected grafting-model layer r, the stored value is
Wmem[i] = FG(text(mi))(r)
last ∈RDmem,
(16)
which is the implementation-level form of FG(mi)(r)
last in Section 3.2. During recipient training and inference, the
recipient queries this frozen table with the current suffix context:
(jb,t, mb,t) = ExactLookup(Xb,≤t, M),
emem,b,t = Wmem[jb,t]
if mb,t = 1.
(17)
This design avoids conflicts when the recipient model and the grafting model use different tokenizers. Online retrieval
is always performed over the recipient-side key set M, while the grafting-model tokenizer is used only offline to encode
the same surface text and produce the stored latent value. Even if the grafting model splits the text differently, this
only changes the latent vector stored in Wmem; it does not change the online lookup key. To make this explicit, let τG
denote the grafting-model tokenizer and let si = text(mi). Then
Wmem[i] = FG(τG(si))(r)
last,
(18)
but online retrieval only queries M through mi. Thus, τG(si) affects only the stored latent value, not the exact lookup
key.
A.3. Calculate resources and environment
We use deepspeed as the training framework. For the 0.9B model, We conduct training on a cluster with 4 nodes and
32 A100 GPUs. For the 2.8B model, We conduct training on a cluster with 8 nodes and 64 A100 GPUs.
17


---

A.4. Appendix: Details of the Grafting Gate
We provide the detailed formulation of the grafting mechanisms used in the ablation using the notation in Section 3.
At recipient position (b, t) and branch c, let Hc
b,t denote the recipient hidden state. For an exact matched n-gram order
n, let e(n)
mem,b,t denote the corresponding frozen grafting-model memory feature, and let Nb,t denote the set of matched
orders. The hash-based Engram fallback feature is ehash,b,t. Given any candidate key-value pair (Kc
b,t, Vb,t), the Method
gate is
Qc
b,t = RMSNormq(Hc
b,t),
¯Kc
b,t = RMSNormk(Kc
b,t),
αc
b,t = σ
⎛
⎝
⟨
¯Kc
b,t, Qc
b,t
⟩
√
D
⎞
⎠,
Uc
b,t = αc
b,tVb,t,
∆Hc
b,t = Uc
b,t + ShortConv(U)c
b,t,̃︀
Hc
b,t = Hc
b,t + ∆Hc
b,t.
1
The attention-only grafting variant first aggregates all matched n-gram embeddings and adds the resulting value
without the query-key gate:
Uc
attn,b,t = eattn,b,t = Attn
(︁
Hc
b,t, {Uc
attn,b,t = e(n)
mem,b,t : n ∈Nb,t}
)︁
∆Hc
b,t = Uc
attn,b,t + ShortConv(Uattn)c
b,t.
2
The attention-gated variant uses the same attention-based memory but modulates it with the gate:
Kc
attn,b,t = Wmem,c
k
eattn,b,t,
Vattn,b,t = Wmem
v
eattn,b,t,
followed by the query-key gate above with (Kc
b,t, Vb,t) = (Kc
attn,b,t, Vattn,b,t).
3
The longest-match gated variant removes the attention aggregation and uses only the longest exact match. For this
ablation, missed positions do not use Engram fallback; if no exact match is found, the lookup returns zero memory:
(jb,t, mb,t) = ExactLookup(Xb,≤t, M),
mb,t = 0
if no exact memory hit is found.
When mb,t = 1, let n∗
b,t = max Nb,t. The candidate memory feature is then
elong,b,t =
{︃
e
(n∗
b,t)
mem,b,t,
mb,t = 1,
0,
mb,t = 0,
and the projected key and value are
Kc
long,b,t = Wmem,c
k
elong,b,t,
Vlong,b,t = Wmem
v
elong,b,t,
followed by the same query-key gate with (Kc
b,t, Vb,t) = (Kc
long,b,t, Vlong,b,t).
4
The final mechanism adds Engram fallback to the gated longest-match mechanism:
Kc
eng,b,t = Weng,c
k
ehash,b,t,
Veng,b,t = Weng
v
ehash,b,t,
Kc
b,t = mb,tKc
mem,b,t + (1 −mb,t)Kc
eng,b,t,
Vb,t = mb,tVmem,b,t + (1 −mb,t)Veng,b,t,
where mb,t is the exact-hit mask defined by ExactLookup(Xb,≤t, M). The selected (Kc
b,t, Vb,t) is then passed to the
Method gate. This formulation makes the role of the gate explicit: it does not decide which memory is retrieved, but
controls how strongly the selected grafting signal is written into the recipient hidden state through ∆H.
18


---

B. Additional Experiment
B.1. Additional Analysis of Grafting-Model Memory Geometry
Qwen3.5
Layer 8
Qwen3.5
Layer 24
DeepSeek
Layer 4
DeepSeek
Layer 18
GLM-4.7
Layer 4
GLM-4.7
Layer 28
100
101
102
Effective rank (log scale)
615.9
397.2
547.4
556.0
3.9
1.1
(a) Representation dimensionality
Qwen3.5
Layer 8
Qwen3.5
Layer 24
DeepSeek
Layer 4
DeepSeek
Layer 18
GLM-4.7
Layer 4
GLM-4.7
Layer 28
0
20
40
60
80
100
PC1 explained variance (%)
3.3
5.2
5.4
4.2
85.5
99.5
(b) Dominance of the first principal direction
Qwen3.5
Layer 8
Qwen3.5
Layer 24
DeepSeek
Layer 4
DeepSeek
Layer 18
GLM-4.7
Layer 4
GLM-4.7
Layer 28
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Norm coefficient of variation
0.13
0.10
0.12
0.08
1.23
2.91
(c) Memory-vector norm instability
Qwen3.5
Layer 8
Qwen3.5
Layer 24
DeepSeek
Layer 4
DeepSeek
Layer 18
GLM-4.7
Layer 4
GLM-4.7
Layer 28
0
5
10
15
20
25
30
Top-10 nearest-neighbor overlap (%)
(d) Local n-gram neighborhood structure
Token Jaccard
Same last token
Grafting-model memory geometry
Figure 6: Geometry of frozen grafting-model memory tables. We compare effective rank, first-principal-component
dominance, norm instability, and local nearest-neighbor structure across memory tables constructed from Qwen3.5-
35B-A3B, DeepSeek-V2-Lite, and GLM-4.7-Flash.
Figure 6 compares frozen memory tables constructed from the same n-gram key set but different grafting models:
Qwen3.5-35B-A3B Layers 8/24, DeepSeek-V2-Lite Layers 4/18, and GLM-4.7-Flash Layers 4/28. For each table, we
sample 10,000 memory rows for norm statistics, 2,048 rows for effective rank and first-principal-component variance,
and 1,024 rows for cosine nearest-neighbor structure. All diagnostics are computed directly on the stored memory
vectors before recipient training.
Qwen3.5-35B-A3B and DeepSeek-V2-Lite produce high-dimensional and stable memory geometry: their effective ranks
remain in the hundreds, PC1 explains only a small fraction of the variance, and norm variation is low. In contrast,
GLM-4.7-Flash is strongly anisotropic: Layer 4 already has low effective rank, and Layer 28 is nearly dominated by a
single direction with unstable norms. Thus, many GLM n-grams retrieve nearly collinear vectors, making the memory
less discriminative. This suggests that Memory Grafting requires not only a capable grafting model, but also structured
local n-gram representations; when this structure is weak, the improvement from grafting is limited.
19


---

B.2. Loss Curves for Memory Grafting
10
20
30
40
50
Training Tokens (B)
10000 15000 20000 25000 30000 35000 40000 45000 50000
Training Step
2.75
2.80
2.85
2.90
2.95
3.00
3.05
Training Loss
50 B tokens
(a) Single-layer Memory Grafting
Vallina MoE
Vallina Engram Layer 8
Memory Grafting Layer 8
10
20
30
40
50
Training Tokens (B)
10000 15000 20000 25000 30000 35000 40000 45000 50000
Training Step
2.70
2.75
2.80
2.85
2.90
2.95
3.00
Training Loss
50 B tokens
(b) Multi-layer Memory Grafting
Vallina MoE
Vallina Engram Layer 8+24
Memory Grafting Layer 8+24
Figure 7: Training loss curves for memory graftings.
We further report the training loss curves of single-layer and multi-layer grafting settings in Figure 7. The results show
that Memory Grafting effectively reduces the training loss in both cases.
20


---

B.3. More CKA Analysis Results of Memory Grafting
0
1
2
3
4
5
6
7
8
9
10
11
12
MoE Baseline Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MoE Baseline
0
1
2
3
4
5
6
7
8
9
10
11
12
Vallina Engram Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs Vallina Engram
0
1
2
3
4
5
6
7
8
9
10
11
12
MG-Qwen Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MG-Qwen
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
Figure 8: CKA comparison for single-layer Memory Grafting.
0
3
6
9
12
15
18
21
24
MoE Baseline Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MoE Baseline
0
3
6
9
12
15
18
21
24
Vallina Engram Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs Vallina Engram
0
3
6
9
12
15
18
21
24
MG-Qwen Layer
0
4
8
12
16
20
24
28
32
36
40
Qwen3.5-35B-A3B Layer
Qwen3.5-35B-A3B vs MG-Qwen
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
0.2
0.4
0.6
0.8
1.0
CKA
Figure 9: CKA comparison for the 2.8B model.
We provide additional CKA results in Figures 8 and 9. Both the single-layer setting and the 2.8B Memory Grafting
model show stronger alignment with deeper Qwen3.5-35B-A3B representations than the corresponding baselines.
This further indicates that the grafted memory changes the recipient hidden states rather than remaining an unused
external table.
21


---

B.4. Top-k N-gram Hit Rate
0
1
2
3
4
5
6
Top-k (Million)
0
10
20
30
40
50
Hit rate (%)
2-gram
0
1
2
3
4
5
6
Top-k (Million)
0
5
10
15
20
25
3-gram
0
1
2
3
4
5
6
Top-k (Million)
0
2
4
6
8
4-gram
Figure 10: Hit rate of top-k n-gram memory entries as the memory size increases.
We further analyze how the hit rate changes as more top-k n-gram entries are included in the grafting memory. As
shown in Figure 10, the hit rate increases with the number of stored n-grams, but the growth gradually slows down.
This trend indicates that frequent n-grams provide the largest coverage gain early on, while adding more entries further
improves coverage by capturing less frequent contexts.
22


---

B.5. Single-Layer Memory Grafting Results
Table 5: Benchmark performance (%) in the 0.8B-trainable, 50B-token setting. Only the first layer was grafted by the
layer 6 of Qwen3.5-35B-A3B
Benchmark (Metric)
# Shots
MoE Baseline
Vanilla Engram
Memory Grafting
# Trainable Params
0.92B
0.92B
0.92B
# Activated (w/o token embed)
0.29B
0.29B
0.29B
# Trained Tokens
50B
50B
50B
# Experts (shared + routed, top-k)
1 + 56 (top-4)
1 + 48 (top-4)
1 + 46 (top-4)
ARC-Challenge
0-shot
28.24
28.50
29.27
ARC-Easy
0-shot
63.30
63.30
63.51
BoolQ
0-shot
56.21
48.50
60.73
Social IQA
0-shot
40.53
39.92
40.69
RACE
0-shot
30.53
29.95
32.06
LAMBADA
5-shot
29.50
29.11
30.37
WinoGrande
5-shot
51.78
52.09
52.72
PIQA
5-shot
70.24
71.22
71.49
HellaSwag
10-shot
38.03
38.25
38.40
Average
45.37
44.54
46.58
We also evaluate Memory Grafting under the single-layer grafting setting. With the same model size and activated
computation, Memory Grafting achieves the best average performance among the compared models, improving over
both the MoE baseline and vanilla Engram. The gains are consistent across the evaluated benchmarks, showing that
even a single grafting layer can effectively transfer useful n-gram memory into the recipient model.
23


---

B.6. Probe Analysis
Layer 1
Layer 6
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Gate Value
(a) Memory Grafting Gate at Retrieval-Hit Positions
Layer 1
Layer 6
0
1
2
3
4
5
6
7
8
L2 Norm (Mean; P10-P90)
(b) Engram Output Magnitude at Retrieval-Hit Positions
Memory Grafting Output
Vallina Engram Output
Memory Grafting Minus Vallina Engram
Memory Grafting
vs. Vallina Engram
Memory Grafting
Disable Engram Fallback Vallina Engram
vs.
Disable Engram Fallback
0.0
0.2
0.4
0.6
0.8
1.0
Rate
(c) Prediction-Set Agreement
Top-1 Prediction Changed
Top-5 Overlap
Figure 11: Overview of gate values, output magnitudes, and prediction-set agreement at retrieval-hit positions.
−1.5
−1.0
−0.5
0.0
0.5
1.0
1.5
2.0
Δlog p(yt + 1)
Ground-Truth Next-Token Δlog p
0.0
0.2
0.4
0.6
0.8
1.0
KL(Treatment || Reference)
Predictive Distribution Shift
0
1
2
3
4
5
6
7
max
v
|Δzv |
Maximum Vocabulary-Logit Change
Memory Grafting vs. Vallina Engram
Memory Grafting vs. Disable Engram Fallback
Vallina Engram vs. Disable Engram Fallback
Figure 12: Logit-level impact of grafted memory at retrieval-hit positions.
We separately probe the behavior of the grafted memory at retrieval-hit positions, where an exact n-gram match is
found in the external memory. Figure 11 shows that the gate is non-zero and the memory output has a clear magnitude
at these positions, meaning that the retrieved memory is actually written into the recipient hidden states. Figure 12
further shows that this hidden-state change propagates to the output distribution: the next-token logits, target token
log-probability, and prediction set all change after grafting. Therefore, the grafted memory is not only retrieved, but
also participates in the model’s computation and directly affects its predictions.
24


---

C. Limitations & Discussion
C.1. Limitations
We do not include a full comparison with knowledge distillation. A from-scratch distillation baseline at the same
pre-training scale would require running a large teacher model over the full training corpus and repeating this process
under comparable recipient settings, which would require substantial additional computation and is difficult to complete
under our compute budget. Moreover, forcing distillation into the same compute budget would also be difficult to
interpret, because the result may reflect insufficient distillation tuning or an under-resourced distillation setup rather
than a clean comparison of the transfer mechanism. Such a comparison would therefore not provide a clean evaluation
of the effectiveness of Memory Grafting.
To make this gap concrete, For example, distilling a 1B-A0.2B recipient from a 30B-A3B teacher requires one extra
teacher forward pass for every training token. Using standard transformer compute accounting[20, 25], the recipient’s
own training cost is roughly 6 × 0.2B activated computation per token, while the teacher forward adds 2 × 3B. Thus,
online distillation would add about 5× the recipient’s own training compute, making the total cost about 6× solo
recipient pre-training.
Therefore, our experiments focus on comparisons under nearly matched recipient architectures, token budgets, and
activated computation. Under these controlled conditions, Memory Grafting serves as a knowledge-transfer method
independent of knowledge distillation, showing that reusable latent memory can improve pre-training without running
a teacher model online.
C.2. Future Work
Scaling the grafted memory. The capacity sweep in Figure 5 shows that loss keeps decreasing from 30k to 3M entries,
and the hit-rate curves are still rising. Pushing the bank to larger n-gram counts and higher orders, and sweeping it
together with recipient size and token budget, is a direct way to extend Memory Grafting to frontier scales.
Memory Grafting for post-training and continual learning. Beyond from-scratch pre-training, the same construction
can serve as a post-training and continual-learning strategy [3, 4, 18, 37, 38, 49–53]..Because the bank is built offline
at modest cost (about 3 A100-hours for a 3M-entry table in our setting) and the recipient consumes it only through
frozen retrieval plus lightweight projections and gates, a freshly built bank can be attached to an already-trained
recipient to inject domain- or task-specific knowledge during SFT, instruction tuning, or domain adaptation, without
re-running large-scale pre-training. In a continual-learning setting, the bank can be incrementally extended or rebuilt
as new corpora and stronger grafting models become available, letting the recipient absorb new knowledge through its
external memory rather than through parameter updates that risk catastrophic forgetting.
Grafting mechanism and recipient-side structure. The current interface—longest-match retrieval, linear projections,
a query-key gate, a short convolution, and a residual update at two layers—is already effective, and the ablation in
Figure ?? shows that each component matters. Richer aggregation over multiple matched n-grams, gates conditioned on
retrieval properties, alternatives to the additive residual update, and adaptive recipient-layer selection are all plausible
next steps on top of this design.
D. LLM Usage
This study utilizes Large Language Models (LLMs) to refine content, adjust formatting, construct tables, and provide
writing suggestions for specific chapters.
25
