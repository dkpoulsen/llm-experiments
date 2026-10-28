---
title: Eigenvectors of Experts are Training-free Non-collapsing Routers
id: eigenvectors-of-experts-are-training-free-non-collapsing-routers
tags:
- cmp170hx-training-feasibility-a0d4fc
- moe-routing
- training-free-router
- expert-collapse
- frozen-router
created: '2026-10-27T11:25:13.534229Z'
updated: '2026-10-27T19:00:31.490614Z'
source: https://arxiv.org/abs/2605.30992
source_domain: arxiv.org
fetched_at: '2026-10-27T11:25:13.533706Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Do, Le and Tran (Deakin University A2I2, ICML 2026, arXiv 2605.30992) -
  SSMoE, a training-free router for sparse MoE. Instead of a learned gating matrix
  it takes the top singular vectors of each expert''s weight matrices (SVD), builds
  a spectral embedding per expert, and routes each token by similarity to those directions.
  The motivating claim is that expert collapse persists even in well-pretrained SMoE
  models and that existing router fixes require training from scratch or fine-tuning,
  which is expensive in compute and data. Reports roughly 13% average gain over baselines
  across eight datasets and about 6% over the original GPT-OSS models while cutting
  memory roughly 23%, evaluated on GPT-OSS-20B (38.96 GB), GPT-OSS-120B (217.61 GB),
  Qwen3-30B, Qwen3-Next-80B, DeepSeek-16B, Qwen1.5-MoE and OLMoE-7B. For this run:
  a frozen, weight-derived router removes router parameters and auxiliary load-balancing
  losses from the trainable budget and removes router collapse as a failure mode,
  which matters for short-budget MoE runs on a 5-GPU box where a collapsed router
  wastes the whole run.'
raw_file: raw/eigenvectors-of-experts-are-training-free-non-collapsing-routers.pdf
doi: arXiv:2605.30992
utility_score: 11.0
citation_count: 1
venue: arXiv.org
---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Giang Do 1 Hung Le 1 Truyen Tran 1
Abstract
Sparse Mixture of Experts (SMoE) architectures
improve the training efficiency of Large Language
Models (LLMs) by routing input tokens to a se-
lected subset of specialized experts. Despite their
remarkable success, both training and inference
in SMoE models suffer from the expert collapse
issue (Chi et al., 2022), which degrades model
performance. Prior studies primarily focus on im-
proving the router; however, such methods rely
on training from scratch or fine-tuning, which re-
quires high computational and data-processing
costs. Furthermore, we demonstrate that, despite
these efforts, the issue persists when advancing
well-pretrained SMoE models, as evidenced by
both theoretical and empirical results. To fill that
gap, we analyze the advanced SMoE models and
observe that the eigenvectors of expert weight ma-
trices encode rich semantic information, pointing
to an effective alternative to conventional rout-
ing strategies. Building on this insight, we pro-
pose Singular Value Decomposition SMoE (SS-
MoE), a novel and training-free framework that
leverages spectral properties of the expert weights
to address the collapse issue and enhance model
performance. Extensive experiments across di-
verse language and vision tasks, under both clean
and corrupt data settings, demonstrate the strong
generalization and robustness of SSMoE. Our
findings highlight how a deeper understanding
of model internals can guide the development
of more effective SMoE architectures. Our im-
plementation is publicly available at https:
//github.com/giangdip2410/SSMoE.
1Applied Artificial Intelligence Intiative (A2I2), Deakin Uni-
versity, Victoria, Australia.
Correspondence to:
Giang Do
<truong.do@deakin.edu.au>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
1. Introduction
Sparse Mixture of Experts (SMoE) models, which of-
fer strong scalability, have achieved remarkable success
across various domains, including Natural Language Pro-
cessing (NLP) (Fedus et al., 2022; Du et al., 2022; Team,
2024; Muennighoff et al., 2025), visual representation learn-
ing (Riquelme et al., 2021; Shen et al., 2023), and multi-
modal tasks (Li et al., 2025a; Shen et al., 2024). However,
understanding the underlying reasons for SMoE’s success
remains a challenge, which limits its adoption in domains
that demand both high accuracy and full explainability, such
as medicine, economics, and law. To address this issue,
Chen et al. (Chen et al., 2022) provides a theoretical expla-
nation of SMoEs in a simplified two-layer non-linear setting.
Meanwhile, XMoE (Chi et al., 2022) highlights the problem
of representation collapse in Sparse Mixture models from
a theoretical standpoint, and SimSMoE (Do et al., 2025a)
further investigates this collapse by estimating it through
similarity metrics. Although most researchers regard the
router as a crucial component and have proposed numerous
enhancements (Dai et al., 2022; Do et al., 2023; Pham et al.,
2024), VQMoE (Do et al., 2025b) adopts a different strategy
employing discrete representation learning to guide token
routing.
Research Question
Is it possible to perform routing in Sparse Mixture of
Experts (SMoEs) without a router?
Finding 1: We observe that the eigenvectors of
expert weight matrices capture rich semantic infor-
mation.
We investigate routing behavior in several advanced, well-
trained SMoE models (Muennighoff et al., 2025; Dai et al.,
2024; Team, 2024) and observe that even pretrained routers
continue to exhibit collapse issues—a finding consistent
with prior work (Pham et al., 2024). To validate this ob-
servation, we evaluate router behavior across ten state-of-
the-art MoE-based models ranging from 4B to 120B pa-
rameters. Our experimental suite encompasses models of
varying scales and architectures, including PhiMoE-Tiny (Li
et al., 2025b), OLMoE (Muennighoff et al., 2025), Qwen1.5-
MoE (Team, 2024), and DeepSeek-MoE (Dai et al., 2024).
1
arXiv:2605.30992v1  [cs.LG]  29 May 2026


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
101
102
Total Model Parameters (Billions)
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Average Router Collapse Level
GPT-OSS-120B
Qwen3-30B
(Think & Inst)
ERNIE-4.5-21B
Qwen3-Next-80B
GPT-OSS-20B
DeepSeek-16B
Qwen1.5-MoE
OLMoE-7B
Phi-MoE
Model Type
Reasoning
Non-Reasoning
Bubble = total size
10B
50B
120B
Figure 1. Average router collapse levels across layers for ten state-
of-the-art MoE-based LLMs. The results demonstrate that while
all models exhibit router collapse, the intensity varies between
reasoning and non-reasoning models. Best viewed in color.
We also evaluate recent large-scale models such as GPT-
OSS-20B, GPT-OSS-120B (OpenAI et al., 2025), ERNIE-
4.5 (Baidu-ERNIE-Team, 2025), and the Qwen3-MoE series
(Instruct, Think, and Next variants) (Team, 2025). Notably,
all models exhibited router collapse to varying degrees as
Figure 1. This motivates our research question: Is it possible
to perform routing in Sparse Mixture of Experts (SMoEs)
without router? To address this question, we conduct an
in-depth analysis of expert weights, typically implemented
as two or three Feedforward Neural Network (FFN) layers,
through the lens of Singular Value Decomposition (SVD).
Our observations reveal that the eigenvectors of these weight
matrices encode rich semantic information. Building on this
insight, we propose Singular Value Decomposition SMoE
(SSMoE), a novel and efficient framework that leverages
the eigenvectors of expert weight matrices. These eigen-
vectors not only serve as effective semantic representations
but also contribute to improved token routing strategies.
To validate the generalization ability of our approach, we
evaluate SSMoE in a training-free setting across multiple
scenarios—including Large Reasoning Models, Large Lan-
guage Models, and Vision-Language Models, and with both
clean and corrupted data. SSMoE consistently outperforms
baseline methods across all three domains, with particularly
notable improvements on tasks requiring deep semantic un-
derstanding. Notably, SSMoE surpasses the state-of-the-art
open-source LLMs - the GPT-OSS models (OpenAI et al.,
2025), with an average improvement of approximately 6%,
while simultaneously reducing memory consumption by
roughly 23% as Figure 2. In summary, this paper makes the
following key contributions: (1) We investigate the expert
embedding collapse issue in advanced SMoE models; (2)
We observe that the eigenvectors of expert weight matrices
encode rich semantic information; (3) We introduce SS-
MoE, a novel and efficient framework that exploits these
spectral properties to enhance both expert representation and
token routing; (4) We propose an expert dropping method
based on SSMoE that improves Sparse Mixture of Experts
performance while reducing memory usage. Our findings
underscore the importance of understanding the internal
structure of SMoEs, and demonstrate how spectral insights
can lead to more effective and efficient architectures.
2. Related Work
Sparse Mixture of Experts (SMoE). Sparse Mixture of
Experts (SMoE), a scalable extension of the original Mix-
ture of Experts framework (Jacobs et al., 1991; Jordan &
Jacobs, 1994), has demonstrated strong empirical perfor-
mance in both Natural Language Processing (Fedus et al.,
2022; Du et al., 2022; Team, 2024; Muennighoff et al.,
2025) and computer vision (Riquelme et al., 2021; Shen
et al., 2023). Despite these advances, SMoE models are
frequently affected by representation collapse (Chi et al.,
2022), a phenomenon where different experts converge to
produce similar or redundant outputs, thereby reducing the
effective capacity of the model.
Several approaches have been proposed to address this is-
sue. XMoE introduces low-dimensional routing to promote
expert diversity (Chi et al., 2022), while methods such as Hy-
perRouter (Do et al., 2023) and StableMoE (Dai et al., 2022)
focus on improving routing stability and reducing variance
in expert selection. However, representation collapse re-
mains a persistent challenge, as recent studies continue to
report its detrimental impact on model expressiveness and
performance (Pham et al., 2024; Do et al., 2025a). In ad-
dition, theoretical analyses (Chen et al., 2022) and discrete
routing strategies (Do et al., 2025b) have further contributed
to understanding the underlying causes of collapse and de-
veloping potential mitigation strategies.
Singular Value Decomposition (SVD). Singular Value De-
composition (SVD) has proven to be an effective tool for
analyzing and fine-tuning Large Language Models (Wang
et al., 2025b;a). Prior work by Thamm et al. (Thamm et al.,
2022) highlights that critical information in neural network
weights is often concentrated in the largest singular values
and their corresponding eigenvectors. In contrast, Staats et
al. (Staats et al., 2026) demonstrate that although small sin-
gular values may seem negligible during pretraining, their
removal after fine-tuning can substantially impair model per-
formance. Building on this line of research, Sun et al. (Sun
et al., 2025) propose adjusting the scale of singular values in
weight matrices to construct expert vectors, each of which
specializes in a distinct task type. Recent works such as
SMILE (Tang et al., 2026) and MASS (Crisostomi et al.,
2026) use eigenvector-based subspace analysis for model
merging by decomposing fine-tuning updates ∆W in task-
specific image models. In contrast, SSMoE targets router
collapse in a single pretrained MoE by decomposing ex-
2


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
ARC-C
ARC-E
BoolQ
GSM8K
HellaSwag
OBQA
PIQA
WinoGrande
Avg.
0
10
20
30
40
50
60
70
80
90
37.6
68.7
65.8
29.5
31.0
21.4
60.3
58.5
46.6
46.8
80.4
75.6
24.9
37.9
24.0
71.0
63.5
53.0
50.4
82.0
73.0
30.0
38.8
24.2
72.5
64.9
54.5
Method (Memory Usage)
GPT-OSS-20B Original (38.96 GB)
RandDrop (30.05 GB)
Router (30.05 GB)
SSMOE (30.05 GB)
47.5
80.2
64.3
37.5
29.7
22.2
65.0
58.7
50.6
(a) GPT-OSS-20B 5-shot Evaluation
ARC-C
ARC-E
BoolQ
GSM8K
HellaSwag
OBQA
PIQA
WinoGrande
Avg.
0
10
20
30
40
50
60
70
80
90
44.3
76.0
73.6
45.7
34.4
19.8
59.7
56.9
51.3
46.1
78.7
76.9
50.0
37.6
22.6
67.5
59.4
54.9
48.5
78.6
78.5
56.0
39.4
23.0
64.1
61.0
56.1
Method (Memory Usage)
GPT-OSS-120B (217.61 GB)
RandomDrop (164.20 GB)
Router (164.20 GB)
SSMOE (164.20 GB)
55.3
84.0
64.2
44.4
32.4
25.0
67.4
58.7
53.9
(b) GPT-OSS-120B 5-shot Evaluation
Figure 2. We report performance comparisons across benchmarks for different GPT-OSS model scales under the 5-shot evaluation setting.
The proposed SSMoE consistently outperforms all baseline methods across the eight datasets, achieving an average improvement of
approximately 13%. Notably, SSMoE surpasses the original GPT-OSS models with an average gain of about 6%, while simultaneously
reducing memory consumption by roughly 23%.
pert weights W to capture intrinsic expert specialization in
general-purpose LLMs and VLMs.
To the best of our knowledge, this is the first study to inves-
tigate the eigenvectors of expert weight matrices in SMoE
models and propose their use as an alternative to conven-
tional routers. Our approach not only offers robust and
semantically rich representations but also enhances expert
selection effectiveness.
3. Methodology
3.1. Preliminaries
Sparse Mixture of Experts. The Sparse Mixture of Ex-
perts (SMoE) architecture is a scalable extension of the
standard Transformer model that introduces conditional
computation by replacing dense multi-layer perceptrons
(MLPs) with a collection of specialized subnetworks known
as experts (Shazeer et al., 2017). Rather than activating
all experts simultaneously, SMoE employs a sparse gating
mechanism that selects a small subset of the most relevant
experts—typically the top-k—based on the input. Given an
input x ∈Rb×d, where b is the batch size and d is the input
dimensionality, the SMoE layer computes its output as a
weighted combination of the outputs from the top-k experts:
fSMoE(x) =
X
i∈TopK(S(x),k)
Si(x) · Ei(x),
(1)
Where each expert i is defined by:
E(i)(x) = FFN(i)
1 · σ

FFN(i)
2 (x)

,
(2)
where FFN(i)
1
∈Rd×h and FFN(i)
2
∈Rh×d and S(x) =
softmax(xW ⊤
e ) denotes the gating function, which pro-
duces a probability distribution over the N available experts
using a learned projection matrix We ∈RN×d. Each expert
Ei is an independent feedforward neural network, and only
the k experts with the highest gating scores Si(x) are acti-
vated for a given input. This sparsity reduces computational
cost and memory footprint while maintaining or improving
model capacity and performance.
3.2. Singular Value Decomposition SMoE (SSMoE)
Eigenvectors Representation. We illustrate the concept
of eigenvector representations in Figure 3. Unlike SMoE,
which relies on learnable embeddings for expert selection,
our approach selects experts based on the eigenvectors of
their weight matrices, which capture rich semantic infor-
mation. The computation of eigenvector representations
involves the following four steps:
Step 1: Eigenvector Extraction. For each expert i, compute:
A(i) = FFN(i)
1 FFN(i)⊤
1
∈Rd×d,
B(i) = FFN(i)⊤
2
FFN(i)
2
∈Rd×d.
(3)
Let {v(i,1)
j
} and {v(i,2)
j
} denote the eigenvectors of A(i)
and B(i), respectively.
Step 2: Top-c Eigenvector Selection. Given the router em-
bedding r(i) ∈Rd, select the top-c most similar eigenvec-
tors from both sets:
C(i)
1
= TopC
 {v(i,1)
j
}, r(i), c

,
C(i)
2
= TopC
 {v(i,2)
j
}, r(i), c

.
(4)
Then compute the average to obtain the spectral embedding
for expert i:
C(i) = 1
2

mean(C(i)
1 ) + mean(C(i)
2 )

∈Rd
3


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Input
Output
Router
Expert 2
Expert 1
Expert 3
Expert 4
U
V
U
U
U
V
V
V
Figure 3. An illustration of our Eigenvectors Representation, which
leverages enriched information from the eigenvectors of expert
weights. In contrast, SMoE employs a learnable expert embedding
(router) to select the top-k experts for each token. SSMoE provides
an efficient and robust representation, as demonstrated in Section 4.
Best viewed in color.
Step 3: Spectral Routing Matrix. Concatenate the expert
spectral embeddings to form the routing weight matrix:
WEV =
h
C(1) C(2) . . . C(N)i
∈Rd×N
Step 4: Eigenvectors Representation.
fEV(x) = xWEV ∈Rb×N
SSMoE. We investigate the eigenvector representations of
expert weights, which encode rich semantic information.
Prior work (Li & Zhou, 2025) has shown that SMoE routers
capture contextually rich features. Building on these in-
sights, we propose a novel architecture that combines both
semantic and contextual information by linearly integrat-
ing the eigenvector representations with the SMoE router
outputs. The detailed algorithm is provided in Algorithm 1.
The output of the SSMoE layer for an input x is defined as:
fSSMoE(x) =
X
i∈TopK(V(x),k)
Vi(x) · Ei(x),
(5)
where V(x) = softmax(fEV(x)) · α + S(x) · (1 −α), and
α ∈[0, 1] is a balancing factor that interpolates between
the EV Router and the standard SMoE Router. The hyper-
parameter α is tuned based on the downstream task data,
indicating whether token inputs should rely more on seman-
tic information or contextual information. We empirically
observe that α values between 0.5 and 0.9 tend to produce
optimal results.
3.3. Theoretical Analysis
Lemma 3.1 (SSMoE router reduces pairwise logit cor-
relations:
router-independent case). Let fEV(x)
=
(fEV,1(x), . . . , fEV,N(x)) be the pre-softmax logits pro-
duced by the eigenvector (EV) router and let s(x) =
(s1(x), . . . , sN(x)) be the learned SMoE router logits for
input x. Form the convex combination
V(x) = αfEV(x) + (1 −α)s(x),
α ∈[0, 1].
Assume the following conditions:
1. (EV
orthogonality)
For
every
pair
i̸
=
j,
E

fEV,i(x) fEV,j(x)

= 0.
2. (EV–learned independence) For every i, j, the EV
logits are uncorrelated with the learned logits:
E

fEV,i(x) sj(x)

= 0.
Then for every i̸ = j the pairwise covariance of the com-
bined logits satisfies
E

Vi(x) Vj(x)

= (1 −α)2 E

si(x) sj(x)

.
In particular, when the learned router is collapsed so that
|E[sisj]| is large, any choice α ∈(0, 1] strictly reduces the
magnitude of pairwise logit covariance relative to using s
alone:
E[ViVj]
 = (1 −α)2 E[sisj]
 <
E[sisj]
.
Thus SSMoE router mitigates collapse (pairwise correlation)
of the learned router; the larger α the greater the reduction.
Lemma 3.1 demonstrates that SSMoE alleviates the repre-
sentation collapse problem commonly observed in conven-
tional SMoE models (Chi et al., 2022). Lemma 3.1 char-
acterizes the router-independent case of SSMoE, including
the simple-average EV variant used in our ablation. In this
setting, EV descriptors are constructed from expert weights
rather than from the learned router decisions, making the
decorrelation assumption a plausible analytical model.
Extension to selective SSMoE.
Our main SSMoE variant
further uses the learned router to select a subset of eigenvec-
tor directions. This router-guided selection improves em-
pirical performance by filtering misaligned EV directions,
but it also introduces mild dependence between fEV(x) and
s(x). Therefore, for selective SSMoE, the exact decorre-
lation condition in Lemma 3.1 should be interpreted as an
approximate decorrelation condition:
E

fEV,i(x)sj(x)
 ≤ε,
∀i, j,
for a small ε ≥0. Under this relaxed condition, for every
i̸ = j,
E

Vi(x)Vj(x)
 ≤(1−α)2 E

si(x)sj(x)
+2α(1−α)ε.
4


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Thus, selective SSMoE preserves the decorrelation effect
of Lemma 3.1 up to a small cross-correlation error term.
This distinction explains why the router-free average EV
variant already improves over the original model, while the
router-guided variant further improves performance by se-
lecting more relevant eigenvector directions. Empirically,
we also observe low overlap between SSMoE and the origi-
nal SMoE router, supporting that the EV signal is comple-
mentary rather than a trivial copy of the learned router. The
formal proof is presented in the Appendix B.2.
4. Experiments
4.1. Large Reasoning Models
Settings. We evaluate our method using state-of-the-art
open-source MoE LLMs from the GPT-OSS family (Ope-
nAI et al., 2025), specifically two variants: GPT-OSS-20B
with 24 layers and 32 experts, and GPT-OSS-120B with
36 layers and 128 experts. Both models activate 4 experts
per token. Due to resource constraints, we adopt an expert
pruning strategy inspired by expert dropping (Zhou et al.,
2025). We propose dropping 25% of experts and compare
three dropping strategies: (1) random expert dropping as a
baseline; (2) dropping based on average similarity scores
between tokens and the router; and (3) dropping based on
SSMoE (our method). This approach reduces memory con-
sumption by approximately 23% compared to the original
models.
Moreover, we test SSMoE performance across eight es-
tablished reasoning benchmarks.
The ARC-Challenge
and ARC-Easy benchmarks (Clark et al., 2018) assess
scientific question answering at varying difficulty levels.
BoolQ (Clark et al., 2019) evaluates binary reading compre-
hension through yes/no questions. GSM8K (Cobbe et al.,
2021) provides a high-quality dataset of linguistically di-
verse grade school math word problems. HellaSwag (Zellers
et al., 2019) tests commonsense natural language inference
(NLI), while OpenBookQA (OBQA) (Mihaylov et al., 2018)
probes deeper understanding through question answering
that requires reasoning over provided background facts.
PIQA (Bisk et al., 2020) measures physical commonsense
reasoning by selecting the most plausible outcomes in ev-
eryday scenarios, and WinoGrande (Sakaguchi et al., 2021)
targets coreference resolution through challenging pronoun
disambiguation tasks.
Reasoning Evaluation. As shown in Figure 2, SSMoE out-
performs the Random Drop approach across all datasets in
the 5-shot evaluation setting, achieving an average improve-
ment of 17% for GPT-OSS-20B and 10% for GPT-OSS-
120B. Notably, compared to the original models, SSMoE
surpasses the baseline GPT-OSS by an average of 8% for the
20B version and 4% for the 120B version, while utilizing
only 75% of the experts (24 out of 32 experts for GPT-
Classif
Clustering
PairClassif
Reranking
STS
Summ
Avg.
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Accuracy
0.46
0.14
0.59
0.41
0.55
0.30
0.41
0.43
0.15
0.39
0.37
0.24
0.21
0.34
0.41
0.14
0.45
0.38
0.40
0.28
0.30
+37% Acc.
EV
Router
SMoE
160
180
200
220
240
260
280
GFLOPS (Lower is Better)
263
275
-5% GFLOPS
Figure 4. We compare the performance of the Eigenvector represen-
tation (EV) with traditional routers and OLMoE-7B (Muennighoff
et al., 2025) on the Massive Text Embedding Benchmark (MTEB).
The results show that EV outperforms OLMoE-7B on 37% of the
tasks across six benchmarks while reducing computational cost
by 5% in GFLOPs. These findings suggest that EV captures rich
semantic information. Best viewed in color.
OSS-20B and 96 out of 128 experts for GPT-OSS-120B).
This leads to Finding 2: Not all experts are necessary for
reasoning tasks.
Following the 5-shot evaluation, we test SSMoE under the
10-shot setting without any additional training and report the
results in Table 1. The results show that SSMoE outperforms
the baseline by an average of 21% across the eight datasets
for GPT-OSS-20B and 10% for GPT-OSS-120B. Compared
to the original GPT-OSS models, SSMoE surpasses the
20B version by 11% and the 120B version by 4%, while
reducing memory usage by 23% and 25%, respectively.
Notably, SSMoE outperforms the original GPT-OSS-120B
model by 24% on the BoolQ task and 17% on the GSM8K
task. These results demonstrate that SSMoE excels not only
in accuracy but also in memory efficiency.
To disentangle the effect of routing from expert pruning, we
also evaluate SSMoE-FULL in the 10-shot setting, where all
experts are retained. SSMoE-FULL improves over the orig-
inal model by 6.4% on average in Table 1, confirming that
the performance gains primarily arise from the proposed
SSMoE routing mechanism rather than from reducing the
number of experts. Moreover, SSMoE-FULL only slightly
outperforms the 75%-expert SSMoE variant by 1.1%, in-
dicating that expert dropping is not the main source of the
improvement. In contrast, naive expert removal through
RandomDrop substantially degrades performance. Together,
these results demonstrate that SSMoE improves expert uti-
lization through better routing, rather than relying on expert
pruning alone.
Finding 2: Certain experts can be omitted from rea-
soning tasks without compromising performance.
5


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Model
Method
Memory (GB)
ARC-C
ARC-E
BoolQ
GSM8K
HellaSwag
OBQA
PIQA
WinoGrande
Avg.
GPT-OSS-20B
10-shot
Original
38.96
44.2±1.5
77.9±0.9
62.8±0.8
36.8±1.3
29.0±0.5
19.8±1.8
62.5±1.1
57.5±1.4
48.8±1.2
SSMoE-FULL (Ours)
38.96
50.2±1.5
79.3±0.8
71.7±0.8
43.4±1.4
40.9±0.5
33.8±2.1
63.9±1.1
58.3±1.4
55.2±1.1
RandDrop (Baseline)
30.05
33.5±1.4
65.8±1.0
66.5±0.8
33.1±1.3
29.7±0.5
20.6±1.8
57.2±1.2
53.3±1.4
44.9±1.2
Router
30.05
48.6±1.5
80.1±0.8
76.6±0.7
22.3±1.1
35.6±0.5
24.2±1.9
71.3±1.1
60.3±1.4
52.4±1.1
SSMoE (Ours)
30.05
52.1±1.5
82.4±0.8
72.4±0.8
30.8±1.3
37.5±0.5
23.0±1.9
72.2±1.0
62.2±1.4
54.1±1.1
vs. Baseline
0.0%
+55.5%
+25.2%
+8.9%
-6.9%
+26.3%
+11.7%
+26.2%
+16.7%
+20.5%
vs. Original
-22.9%
+17.9%
+5.8%
+15.3%
-16.3%
+29.3%
+16.2%
+15.5%
+8.2%
+10.9%
GPT-OSS-120B
10-shot
Original
217.61
52.4±1.5
83.0±0.8
64.0±0.8
49.3±1.4
30.5±0.5
22.0±1.9
62.9±1.1
56.7±1.4
52.6±1.2
RandDrop (Baseline)
164.20
43.3±1.4
74.1±0.9
74.2±0.8
46.7±1.4
33.2±0.5
21.4±1.8
56.0±1.2
51.8±1.4
50.1±1.2
Router
164.20
42.4±1.4
76.1±0.9
77.6±0.7
49.2±1.4
36.2±0.5
20.6±1.8
65.1±1.1
58.2±1.4
53.2±1.2
SSMoE (Ours)
164.20
45.6±1.5
79.1±0.8
79.3±0.7
57.6±1.4
37.1±0.5
21.2±1.8
63.3±1.1
56.4±1.4
54.9±1.1
vs. Baseline
0.0%
+5.3%
+6.8%
+6.9%
+23.3%
+11.7%
-0.9%
+13.0%
+8.9%
+9.6%
vs. Original
-24.5%
-13.0%
-4.7%
+23.9%
+16.8%
-3.6%
+0.6%
-0.5%
+4.4%
+4.4%
Table 1. Performance comparison of different methods on GPT-OSS models with 10-shot evaluation. Results show mean ± standard
deviation across eight benchmarks. SSMoE-FULL denotes our SSMoE variant using 100% of experts, enabling a memory-matched
comparison with the original model. Best results in each model size are shown in bold. Improvements are calculated relative to baseline
methods.
4.2. Large Language Models
Settings.
We evaluate the efficiency and robustness of
our method compared to baseline approaches under two
scenarios: (1) clean and (2) corrupt.
In the clean setting, inspired by (Li & Zhou, 2025), we ap-
ply our method as a plug-in module to three state-of-the-art
SMoE models (OLMoE-1B/7B (Muennighoff et al., 2025),
Qwen-MoE-7B (Team, 2024), and DeepSeekMoE-16B (Dai
et al., 2024)) without any fine-tuning. These models vary
in size (from 7B to 16B parameters) and architecture (rang-
ing from 16 to 28 layers and 60 to 64 experts). We assess
their performance on a subset of tasks from the Massive
Text Embedding Benchmark (MTEB) (Muennighoff et al.,
2023), which covers a wide range of downstream sentence
embedding applications, including classification, clustering,
pair classification, re-ranking, retrieval, semantic textual
similarity (STS), and summarization. Following the MTEB
evaluation protocol, we use Accuracy for classification, V-
Measure for clustering, Average Precision for pair classifi-
cation, Mean Average Precision (mAP) for re-ranking, Nor-
malized Discounted Cumulative Gain (nDCG) for retrieval,
and Spearman’s correlation for STS and summarization.
In the corrupt setting, following the protocol in (Nielsen
et al., 2025), we evaluate model robustness under adversarial
perturbations. Specifically, we inject random “AAA” token
sequences into the input to simulate noisy or corrupted data
(denoted as Corrupt). Further implementation details and
extended results are provided in the Appendix section.
We compare our method against an approach that uses the
similarity scores between the router and expert embeddings
as hidden representations, referred to as Router Embed-
ding (or simply Router). Additionally, we evaluate against
MoEE (Li & Zhou, 2025), which combines Router Embed-
ding with the hidden states of the SMoE model to form its
representations. We also include comparisons with prompt-
based methods, such as PromptEOL (Jiang et al., 2024), to
assess in-context learning capabilities.
Semantic Evaluation. We evaluate the semantic quality
of our method by extracting embeddings and testing them
on MTEB tasks, as shown in Table 2. Overall, SSMoE
(ours) consistently outperforms baseline methods across
advanced SMoE models, OLMoE-7B, and DeepSeekMoE-
16B, achieving performance improvements of 25% to 30%.
SSMoE demonstrates especially strong gains in tasks such
as Classification, Clustering, and STS, highlighting its abil-
ity to extract high-quality embeddings without the need for
fine-tuning or prompts. Interestingly, the Eigenvector-based
representation captures rich semantic information, making
the method both simple and highly effective—particularly
for OLMoE, where it achieves a top score of 40.9%.
Robust Evaluation.
To evaluate the robustness of our
method, we conduct experiments under the corrupt setting,
as shown on the right side of Table 2. We observe a con-
sistent trend: complex representations such as those used
in SMoE and MoEE experience a significant performance
drop when transitioning from clean to corrupt settings.
In contrast, lighter representations, including Router and
Eigenvector-based (EV) embeddings, demonstrate greater
resilience under corrupt conditions. This observation aligns
with the “No Free Lunch” theorem (Wolpert & Macready,
1997), which suggests trade-offs between model complexity
and generalization across varying data distributions. No-
tably, our proposed SSMoE achieves state-of-the-art per-
formance on 15 out of 18 tasks in the clean setting and 16
out of 18 tasks in the corrupt setting, highlighting both its
efficiency and robustness.
Finding 3: Eigenvectors of expert weights provide
an efficient and robust representation.
6


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Clean
Corrupt
Model
Task
Router
SMoE
MoEE
EV
SSMoE
Router
SMoE
MoEE
EV
SSMoE
OLMoE-7B
Classification
41.2
43.4
41.8
45.9
50.6
36.0
36.9
36.1
36.3
39.7
Clustering
13.7
14.7
14.5
14.2
16.4
6.7
6.8
6.8
6.9
7.1
PairClassification
45.3
39.1
45.7
59.2
51.2
28.1
26.7
27.6
29.1
30.7
Reranking
37.5
37.4
39.5
41.2
42.4
27.8
26.8
27.7
28.2
29.3
STS
39.9
24.1
39.9
55.3
47.0
5.1
1.9
3.6
7.5
13.4
Summarization
28.4
20.9
29.8
29.6
30.6
24.2
24.5
25.3
26.4
28.5
Avg.
34.3
29.9
35.2
40.9
39.7
21.3
20.6
21.2
22.4
24.8
Qwen-MoE-7B
Classification
43.8
50.3
47.7
43.4
51.6
34.4
38.3
37.0
36.9
39.8
Clustering
13.6
27.4
25.2
13.7
31.1
5.8
8.2
7.2
6.1
8.6
PairClassification
45.9
46.9
51.5
47.0
55.4
26.6
27.4
30.1
31.7
30.7
Reranking
39.6
45.3
48.5
40.3
49.1
26.2
28.9
29.4
26.7
29.8
STS
38.8
38.0
51.8
41.6
57.9
0.7
0.7
2.0
1.7
6.8
Summarization
28.3
13.4
31.2
30.3
29.6
26.7
23.0
25.5
27.9
28.5
Avg.
35.0
36.9
42.6
36.0
45.8
20.1
21.1
21.9
21.8
24.0
DeepSeekMoE-16B
Classification
43.4
46.6
44.4
44.0
48.6
36.2
36.7
36.0
40.0
39.7
Clustering
13.4
18.1
17.8
14.3
21.6
6.6
6.8
6.8
7.2
8.3
PairClassification
45.5
40.9
46.1
46.7
51.2
26.9
25.9
27.3
28.0
31.5
Reranking
38.5
38.9
42.2
39.2
44.9
38.8
38.3
39.0
39.5
41.0
STS
37.7
26.3
40.2
40.0
50.1
5.1
3.7
5.7
9.1
14.3
Summarization
24.9
22.0
24.4
28.9
29.9
27.8
25.2
28.3
28.3
28.4
Avg.
33.9
32.1
35.9
35.5
41.1
23.6
22.8
23.8
25.4
27.2
Table 2. Performance comparison across MTEB tasks under clean and corrupt settings. Results are shown side-by-side. The best score in
each row (clean or corrupt) is highlighted in bold.
Classif
Clustering
PairClassif
Reranking
STS
Summ
0
10
20
30
40
50
60
70
Performance
Router
SMoE
MoEE
EV
SSMoE
Figure 5. Average performance on MTEB tasks across three ad-
vanced LLMs (OLMoE-7B, Qwen-MoE-7B, and DeepSeekMoE-
16B), comparing SSMoE, SMoE, Eigenvectors (EV), and MoEE
using PromptEOL (Jiang et al., 2023) for in-context learning eval-
uation. Best viewed in color.
In-context Learning Evaluation. Figure 5 presents the
performance of various methods on MTEB tasks using
PromptEOL (Jiang et al., 2024) for in-context learning eval-
uation. We compare the average performance and variance
of SSMoE against baseline methods across three advanced
LLMs (OLMoE-7B, Qwen-MoE-7B, and DeepSeekMoE-
16B). The results show that SSMoE consistently achieves
the highest average scores and the lowest variance among all
methods. These findings indicate that SSMoE is not only ef-
fective at capturing semantic representations but also excels
in in-context learning, underscoring its potential to enhance
SMoE-based LLMs on complex tasks such as reasoning.
4.3. Vision Language Models
Settings.
To demonstrate the generalization capability
of our proposed method, we evaluate the performance
of SSMoE on two tasks: zero-shot image-text retrieval
and zero-shot image classification. These tasks assess the
model’s ability to capture fine-grained visual and seman-
tic information, in comparison to the baseline CLIP-MoE
model. CLIP-MoE (Zhang et al., 2025) is a variant of
CLIP (Radford et al., 2021) that incorporates a Sparse
Mixture of Experts (SMoE) architecture, trained on the
ShareGPT4V (ShareGPT) dataset (Chen et al., 2024) to en-
hance multimodal understanding. For image-text retrieval,
we conduct evaluations on the COCO (Lin et al., 2014) and
Flickr30k (Plummer et al., 2015) datasets. Following the
evaluation protocols for the LLMs section, we test both the
efficiency and robustness of SSMoE under two settings: (1)
clean images and (2) corrupted images. In the corrupted set-
ting, we introduce random noise to input images, following
the procedure described in (Ilyas et al., 2019).
Zero-Shot Image-Text Retrieval Evaluation.
Table 3
reports results for Image-to-Text (I2T) and Text-to-Image
(T2I) retrieval at Recall@1, Recall@5, and Recall@10. Un-
der the clean setting, SSMoE consistently outperforms both
the original CLIP model and the CLIP-MoE baseline across
all retrieval tasks. Specifically, SSMoE achieves improve-
ments of up to +1.1% at Recall@1 on COCO I2T, and up
to +1.0% on Flickr I2T, demonstrating its enhanced ability
7


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Setting
Model
COCO I2T
COCO T2I
Flickr I2T
Flickr T2I
@1
@5
@10
@1
@5
@10
@1
@5
@10
@1
@5
@10
CLIP (OpenAI)
56.1
79.5
86.8
35.4
60.1
70.2
48.5
72.6
80.8
28.0
49.3
58.7
Clean
CLIP-MoE
65.0
86.0
92.0
46.8
71.7
80.4
60.5
82.3
88.8
42.1
64.7
73.2
SSMoE (Ours)
65.7
86.5
92.3
47.1
72.0
80.7
61.1
82.6
89.0
42.5
65.2
73.6
% Improv.
+1.1%
+0.6%
+0.3%
+0.6%
+0.4%
+0.4%
+1.0%
+0.4%
+0.2%
+1.0%
+0.8%
+0.5%
Corrupt
CLIP-MoE
36.2
60.2
69.9
29.4
51.7
61.7
31.2
51.8
60.5
23.3
41.4
49.6
SSMoE (Ours)
37.2
61.0
70.3
30.0
52.1
62.2
31.6
52.1
60.7
23.6
41.9
50.1
% Improv.
+2.8%
+1.3%
+0.6%
+2.0%
+0.8%
+0.8%
+1.3%
+0.6%
+0.3%
+1.3%
+1.2%
+1.0%
Table 3. Performance comparison across different models and settings on COCO and Flickr benchmarks for Image-to-Text (I2T) and
Text-to-Image (T2I) retrieval tasks. Best results per column are in bold. Percentage improvements of SSMoE over CLIP-MoE baseline
are shown with blue color indicating gains.
to capture fine-grained cross-modal relationships.
In the corrupted setting—where random noise is added to
the input images—SSMoE maintains its superiority, show-
ing improved robustness compared to CLIP-MoE. Notably,
SSMoE yields gains of +2.8% at Recall@1 for COCO I2T
and consistent improvements across all other metrics and
datasets. These findings confirm the effectiveness of the SS-
MoE design in improving both accuracy and robustness for
zero-shot retrieval tasks, without requiring any additional
training.
Statistical Significance Test.
We observe that SSMoE
consistently improves over the original CLIP-MoE baseline
across datasets, retrieval metrics, and retrieval directions.
While the absolute gains on clean benchmarks are modest,
ranging from approximately 0.2 to 1.1 points, they are mean-
ingful because SSMoE is training-free and introduces no
additional optimization cost. To verify that these improve-
ments are statistically reliable rather than due to evaluation
noise, we conduct paired bootstrap resampling with 1000
samples. Table 4 reports the original baseline score, SSMoE
score, absolute improvement, 95% confidence interval, and
paired bootstrap p-value. Across both COCO and Flickr30k,
all improvements are positive, with narrow confidence in-
tervals and statistically significant p-values. On COCO,
SSMoE improves R@1 by +0.8 points with a 95% confi-
dence interval of [0.20, 1.58] and p = 0.002. Similarly, on
Flickr30k, SSMoE improves I2T@5 by +0.6 points with
a confidence interval of [0.40, 0.88] and p < 0.001. These
results indicate that the gains of SSMoE are consistent and
statistically significant despite their small absolute magni-
tude. Moreover, as shown in the corruption experiments,
the improvements become larger under distribution shift,
suggesting that the main benefit of SSMoE lies not only in
improving saturated clean retrieval performance, but also in
enhancing robustness through better expert utilization.
4.4. Ablation Studies
We investigate the effectiveness and robustness of SSMoE
to the different hyper-parameter settings.
Metric
CLIP-MoE
SSMoE (Ours)
∆
95% CI
p-value
COCO I2T
@1
65.0
65.9
+0.8
[0.20, 1.58]
0.002
@5
86.0
86.5
+0.5
[0.00, 1.00]
0.024
@10
92.0
92.2
+0.2
[0.05, 0.40]
0.018
Flickr30k I2T
@1
61.8
62.3
+0.5
[0.15, 0.80]
0.001
@5
83.4
84.1
+0.6
[0.40, 0.88]
< 0.001
@10
89.6
89.9
+0.3
[0.06, 0.44]
0.008
Table 4. Paired bootstrap significance testing on the clean setting
using 1000 bootstrap samples. ∆denotes the absolute improve-
ment of SSMoE over the CLIP-MoE baseline.
Selective vs. averaged eigenvectors.
We next ablate
whether SSMoE depends on the learned router for selecting
eigenvector directions. In addition to the router-guided vari-
ant, we evaluate a router-free variant that simply averages
the eigenvector descriptors of each expert. As shown in Ta-
ble 5, this simple averaging strategy already improves over
the original model by +3.9 points on average, indicating that
the eigenvector representations themselves encode useful
semantic information about expert specialization. Router-
guided selection further improves the average gain to +5.3
points by filtering less relevant or misaligned directions.
These results show that the router acts as a lightweight
selection mechanism rather than a strict dependency: eigen-
vectors provide the core routing signal, while the router
improves its precision.
Number of Eigenvectors. We assess the effectiveness and
stability of SSMoE under varying hyper-parameter configu-
rations. In particular, we examine the impact of changing the
number of top eigenvectors (TOPC) used in the model. As
shown in Table 6, performance across datasets demonstrates
that SSMoE remains robust to different values of TOPC,
with optimal results often achieved at higher settings. In
practice, we observe that setting TOPC ≈50 yields the best
performance.
Balancing factor α. We analyze the effect of the balancing
factor α on the performance of OLMoE-1B across three
8


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Task
Orig.
SSMoE
SSMoE w/ Avg. EV
ARC-C
44.2
52.1
49.4
ARC-E
77.9
82.4
80.7
BoolQ
62.8
72.4
69.8
GSM8K
36.8
30.8
33.3
HellaSwag
29.0
37.5
36.5
OBQA
19.8
23.0
22.8
PIQA
62.5
72.2
69.5
WinoGrande
57.5
62.2
59.6
Avg.
48.8
54.1
52.7
Table 5. Ablation of router-guided eigenvector selection. The
router-free variant, which simply averages eigenvector descriptors,
still outperforms the original model, while router-guided selection
provides further gains.
classification tasks. As shown in Table 6, performance
generally improves with increasing α, with optimal results
consistently observed around α = 0.9. Empirically, we
find that setting α in the range of approximately 0.5 to 0.9
provides a robust trade-off between the two, yields strong
and stable performance.
TOPC
α
Dataset
5
10
20
50
0.0
0.3
0.5
0.7
0.9
1.0
Emotion
36.5
35.4
35.3
37.4
26.1
27.5
29.9
30.2
32.7
28.9
Toxic
57.4
57.6
60.0
60.7
60.2
61.1
59.8
62.2
62.6
59.3
Tweet
51.3
51.7
53.2
53.4
48.4
49.8
53.3
52.5
54.3
48.9
Table 6. Ablation studies on OLMoE-1B classification tasks. We
vary the number of selected eigenvectors TOPC and the balancing
factor α. Higher values are better, with the best results highlighted
in bold.
4.5. In-depth Analysis
Finding 4: SMoE routers exhibit increased collapse
in deeper layers, whereas EV routers maintain or-
thogonality.
Collapse Problems. In practice, we observe that the SMoE
router exhibits increasing levels of collapse across layers,
as shown in Figure 6a. In contrast, the EV router main-
tains consistent orthogonality layer by layer, as illustrated in
Figure 6b, which is theoretically supported by Lemma B.1.
Latent Structure Discovery.
Figure 7 presents t-SNE
projections of two learned representations, colored by
KMeans cluster assignments (k = 64). Visually, Rep-
resentation 1 exhibits more coherent and well-separated
clusters, while SMoE appears more dispersed with overlap-
ping regions. This observation is quantitatively supported
by clustering metrics: SSMoE achieves a higher Silhou-
ette score (Rousseeuw, 1987) and a lower Davies-Bouldin
score (Davies & Bouldin, 1979), indicating stronger intrin-
sic cluster structure. These results suggest that SSMoE
(a) Correlation between expert embeddings within the SMoE router
of the OLMoE model.
(b) Correlation between expert weight eigenvectors (EV router).
Figure 6. A comparison of collapse behavior shows that the OL-
MoE router exhibits increased collapse at deeper layers, while the
EV router does not encounter this issue.
better preserves semantically meaningful groupings in the
latent space.
SSMoE Representation (Sil=0.59, DB=1.34)
SMoE Representation (Sil=0.19, DB=1.69)
Figure 7. t-SNE visualizations of two learned representations, col-
ored by KMeans cluster assignments with k = 64. SSMoE (left)
displays more compact and well-separated clusters than SMoE
(right), indicating stronger underlying cluster structure. Clustering
quality is further supported by quantitative metrics: higher Silhou-
ette and lower Davies-Bouldin index for SSMoE.
5. Conclusion
In this research, we investigate the behavior of routers in
advanced Sparse Mixture of Experts (SMoE) models and
highlight a common issue known as representation collapse.
Building on this analysis, we propose SVD-SMoE (SSMoE),
a novel and efficient framework that enhances SMoE by
leveraging Eigenvalue Representations of Expert Weights.
SSMoE learns more robust representations and effectively
addresses the representation collapse problem often encoun-
tered in traditional SMoEs. Experimental results across
both training-free and fine-tuning settings show that SSMoE
achieves more efficient and effective fine-tuning and infer-
ence than existing advanced SMoE models. We hope this
work opens new directions for designing effective SMoE
architectures without relying on conventional routers.
9


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Impact Statement
Our work aims to advance Machine Learning through trans-
parent, reproducible research using publicly available re-
sources. While the immediate scope of this study does
not involve human participants, we remain mindful of the
broader implications of LLM deployment. Specifically,
models trained on uncurated web data are susceptible to
systemic biases that require rigorous, continuous mitiga-
tion efforts. Additionally, we acknowledge that the high
computational demands of LLM training pose sustainability
challenges. This paper serves as a technical contribution
while advocating for more resource-efficient and ethically-
aware AI development.
References
Baidu-ERNIE-Team. Ernie 4.5 technical report. https:
//ernie.baidu.com/blog/publication/E
RNIE_Technical_Report.pdf, 2025.
Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y.
PIQA: reasoning about physical commonsense in natural
language. In The Thirty-Fourth AAAI Conference on Arti-
ficial Intelligence, AAAI 2020, The Thirty-Second Inno-
vative Applications of Artificial Intelligence Conference,
IAAI 2020, The Tenth AAAI Symposium on Educational
Advances in Artificial Intelligence, EAAI 2020, New York,
NY, USA, February 7-12, 2020, pp. 7432–7439. AAAI
Press, 2020. doi: 10.1609/AAAI.V34I05.6239. URL
https://doi.org/10.1609/aaai.v34i05.
6239.
Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao,
F., and Lin, D. Sharegpt4v: Improving large multi-modal
models with better captions. In European Conference on
Computer Vision, pp. 370–387. Springer, 2024.
Chen, Z., Deng, Y., Wu, Y., Gu, Q., and Li, Y. Towards
understanding the mixture-of-experts layer in deep learn-
ing. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave,
D., Cho, K., and Oh, A. (eds.), Advances in Neural In-
formation Processing Systems, volume 35, pp. 23049–
23062. Curran Associates, Inc., 2022. URL https:
//proceedings.neurips.cc/paper_files
/paper/2022/file/91edff07232fb1b55a5
05a9e9f6c0ff3-Paper-Conference.pdf.
Chi, Z., Dong, L., Huang, S., Dai, D., Ma, S., Patra, B.,
Singhal, S., Bajaj, P., SONG, X., Mao, X.-L., Huang,
H., and Wei, F. On the representation collapse of sparse
mixture of experts. In Koyejo, S., Mohamed, S., Agarwal,
A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances
in Neural Information Processing Systems, volume 35,
pp. 34600–34613. Curran Associates, Inc., 2022. URL
https://proceedings.neurips.cc/paper
_files/paper/2022/file/df4f371f1f89e
c8ba5014b3310578048-Paper-Conference.
pdf.
Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins,
M., and Toutanova, K. BoolQ: Exploring the surpris-
ing difficulty of natural yes/no questions. In Burstein,
J., Doran, C., and Solorio, T. (eds.), Proceedings of
the 2019 Conference of the North American Chapter
of the Association for Computational Linguistics: Hu-
man Language Technologies, Volume 1 (Long and Short
Papers), pp. 2924–2936, Minneapolis, Minnesota, June
2019. Association for Computational Linguistics. doi:
10.18653/v1/N19-1300. URL https://aclantho
logy.org/N19-1300/.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge,
2018. URL https://arxiv.org/abs/1803.0
5457.
Coates, A., Ng, A., and Lee, H. An analysis of single-
layer networks in unsupervised feature learning. In Gor-
don, G., Dunson, D., and Dud´ık, M. (eds.), Proceed-
ings of the Fourteenth International Conference on Arti-
ficial Intelligence and Statistics, volume 15 of Proceed-
ings of Machine Learning Research, pp. 215–223, Fort
Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL
https://proceedings.mlr.press/v15/co
ates11a.html.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., Hesse, C., and Schulman, J. Training verifiers to solve
math word problems, 2021. URL https://arxiv.
org/abs/2110.14168.
Crisostomi, D., Zirilli, A., Gargiulo, A. A., Bucarelli, M. S.,
Scardapane, S., Silvestri, F., Masi, I., and Rodol`a, E.
Mass: Moerging through adaptive subspace selection,
2026. URL https://arxiv.org/abs/2504.0
5342.
Dai, D., Dong, L., Ma, S., Zheng, B., Sui, Z., Chang, B., and
Wei, F. StableMoE: Stable routing strategy for mixture
of experts. In Muresan, S., Nakov, P., and Villavicencio,
A. (eds.), Proceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 7085–7095, Dublin, Ireland, May
2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.acl-long.489. URL https://acla
nthology.org/2022.acl-long.489/.
Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J.,
Zeng, W., Yu, X., Wu, Y., Xie, Z., Li, Y., Huang, P., Luo,
10


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
F., Ruan, C., Sui, Z., and Liang, W. DeepSeekMoE: To-
wards ultimate expert specialization in mixture-of-experts
language models. In Ku, L.-W., Martins, A., and Sriku-
mar, V. (eds.), Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 1280–1297, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.acl-long.70. URL https:
//aclanthology.org/2024.acl-long.70/.
Davies, D. L. and Bouldin, D. W. A cluster separation
measure. IEEE Transactions on Pattern Analysis and
Machine Intelligence, PAMI-1(2):224–227, 1979. doi:
10.1109/TPAMI.1979.4766909.
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and
Fei-Fei, L. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE Conference on Computer Vi-
sion and Pattern Recognition, pp. 248–255, 2009. doi:
10.1109/CVPR.2009.5206848.
Do, G., Le, H., and Tran, T.
SimSMoE: Toward effi-
cient training mixture of experts via solving represen-
tational collapse. In Chiruzzo, L., Ritter, A., and Wang,
L. (eds.), Findings of the Association for Computational
Linguistics: NAACL 2025, pp. 2012–2025, Albuquerque,
New Mexico, April 2025a. Association for Computa-
tional Linguistics.
ISBN 979-8-89176-195-7.
URL
https://aclanthology.org/2025.findin
gs-naacl.107/.
Do, G., Pham, K., Le, H., and Tran, T. On the role of discrete
representation in sparse mixture of experts. Transactions
on Machine Learning Research, 2025b. ISSN 2835-8856.
URL https://openreview.net/forum?id=
GTWKmojpI7.
Do, T. G., Khiem, L., Pham, Q., Nguyen, T., Doan, T.-N.,
Nguyen, B., Liu, C., Ramasamy, S., Li, X., and Hoi, S.
HyperRouter: Towards efficient training and inference of
sparse mixture of experts. In Bouamor, H., Pino, J., and
Bali, K. (eds.), Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pp.
5754–5765, Singapore, December 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.emn
lp-main.351. URL https://aclanthology.org
/2023.emnlp-main.351/.
Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu,
Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., Zoph, B.,
Fedus, L., Bosma, M. P., Zhou, Z., Wang, T., Wang, E.,
Webster, K., Pellat, M., Robinson, K., Meier-Hellstern,
K., Duke, T., Dixon, L., Zhang, K., Le, Q., Wu, Y., Chen,
Z., and Cui, C. GLaM: Efficient scaling of language mod-
els with mixture-of-experts. In Chaudhuri, K., Jegelka, S.,
Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.),
Proceedings of the 39th International Conference on Ma-
chine Learning, volume 162 of Proceedings of Machine
Learning Research, pp. 5547–5569. PMLR, 17–23 Jul
2022. URL https://proceedings.mlr.pres
s/v162/du22c.html.
Dutta, A., Krishnan, S., Kwatra, N., and Ramjee, R. Accu-
racy is not all you need. In Globerson, A., Mackey, L.,
Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang,
C. (eds.), Advances in Neural Information Processing Sys-
tems, volume 37, pp. 124347–124390. Curran Associates,
Inc., 2024. doi: 10.52202/079017-3950. URL https:
//proceedings.neurips.cc/paper_files
/paper/2024/file/e0e956681b04ac12667
9e8c7dd706b2e-Paper-Conference.pdf.
Fedus, W., Zoph, B., and Shazeer, N. Switch transformers:
scaling to trillion parameter models with simple and effi-
cient sparsity. J. Mach. Learn. Res., 23(1), January 2022.
ISSN 1532-4435.
Fei-Fei, L., Fergus, R., and Perona, P. Learning genera-
tive visual models from few training examples: An in-
cremental bayesian approach tested on 101 object cate-
gories. Computer Vision and Pattern Recognition Work-
shop, 2004.
Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,
A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
https://zenodo.org/records/12608602.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multi-
task language understanding. In ICLR. OpenReview.net,
2021a. URL http://dblp.uni-trier.de/db
/conf/iclr/iclr2021.html#HendrycksBB
ZMSS21.
Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and
Song, D. Natural adversarial examples, 2021b. URL
https://arxiv.org/abs/1907.07174.
Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B.,
and Madry, A. Adversarial examples are not bugs, they
are features. In Wallach, H., Larochelle, H., Beygelz-
imer, A., d'Alch´e-Buc, F., Fox, E., and Garnett, R.
(eds.), Advances in Neural Information Processing Sys-
tems, volume 32. Curran Associates, Inc., 2019. URL
https://proceedings.neurips.cc/paper
_files/paper/2019/file/e2c420d928d4b
f8ce0ff2ec19b371514-Paper.pdf.
11


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E.
Adaptive mixtures of local experts. Neural Computation,
3(1):79–87, 1991. doi: 10.1162/neco.1991.3.1.79.
Jiang, T., Huang, S., Luan, Z., Wang, D., and Zhuang, F.
Scaling sentence embeddings with large language models,
2023. URL https://arxiv.org/abs/2307.1
6645.
Jiang, T., Huang, S., Luan, Z., Wang, D., and Zhuang, F.
Scaling sentence embeddings with large language models.
In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.),
Findings of the Association for Computational Linguis-
tics: EMNLP 2024, pp. 3182–3196, Miami, Florida, USA,
November 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.findings-emnlp.181. URL
https://aclanthology.org/2024.findin
gs-emnlp.181/.
Jordan, M. and Jacobs, R. Hierarchical mixtures of experts
and the. Neural computation, 6:181–, 01 1994.
Krizhevsky, A. Learning multiple layers of features from
tiny images. pp. 32–33, 2009. URL https://www.
cs.toronto.edu/˜kriz/learning-feature
s-2009-TR.pdf.
Langley, P. Crafting papers on machine learning. In Langley,
P. (ed.), Proceedings of the 17th International Conference
on Machine Learning (ICML 2000), pp. 1207–1216, Stan-
ford, CA, 2000. Morgan Kaufmann.
Li, Y., Jiang, S., Hu, B., Wang, L., Zhong, W., Luo, W., Ma,
L., and Zhang, M. Uni-moe: Scaling unified multimodal
llms with mixture of experts. IEEE Trans. Pattern Anal.
Mach. Intell., 47(5):3424–3439, May 2025a. ISSN 0162-
8828.
doi: 10.1109/TPAMI.2025.3532688.
URL
https://doi.org/10.1109/TPAMI.2025.3
532688.
Li, Z. and Zhou, T. Your mixture-of-experts LLM is se-
cretly an embedding model for free. In The Thirteenth
International Conference on Learning Representations,
2025. URL https://openreview.net/forum
?id=eFGQ97z5Cd.
Li, Z., Liang, C., Zhang, Z., Hong, I., Kim, Y. J., Chen,
W., and Zhao, T. Slimmoe: Structured compression of
large moe models via expert slimming and distillation. In
Second Conference on Language Modeling, 2025b. URL
https://openreview.net/forum?id=oaCU
sn391F.
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft
coco: Common objects in context. In Fleet, D., Pajdla,
T., Schiele, B., and Tuytelaars, T. (eds.), Computer Vi-
sion – ECCV 2014, pp. 740–755, Cham, 2014. Springer
International Publishing. ISBN 978-3-319-10602-1.
Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can
a suit of armor conduct electricity? a new dataset for
open book question answering. In Riloff, E., Chiang,
D., Hockenmaier, J., and Tsujii, J. (eds.), Proceedings of
the 2018 Conference on Empirical Methods in Natural
Language Processing, pp. 2381–2391, Brussels, Belgium,
October-November 2018. Association for Computational
Linguistics. doi: 10.18653/v1/D18-1260. URL https:
//aclanthology.org/D18-1260/.
Muennighoff, N., Tazi, N., Magne, L., and Reimers, N.
MTEB: Massive text embedding benchmark. In Vlachos,
A. and Augenstein, I. (eds.), Proceedings of the 17th Con-
ference of the European Chapter of the Association for
Computational Linguistics, pp. 2014–2037, Dubrovnik,
Croatia, May 2023. Association for Computational Lin-
guistics. doi: 10.18653/v1/2023.eacl-main.148. URL
https://aclanthology.org/2023.eacl-m
ain.148/.
Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Mor-
rison, J., Min, S., Shi, W., Walsh, E. P., Tafjord, O.,
Lambert, N., Gu, Y., Arora, S., Bhagia, A., Schwenk,
D., Wadden, D., Wettig, A., Hui, B., Dettmers, T., Kiela,
D., Farhadi, A., Smith, N. A., Koh, P. W., Singh, A.,
and Hajishirzi, H. OLMoe: Open mixture-of-experts
language models. In The Thirteenth International Confer-
ence on Learning Representations, 2025. URL https:
//openreview.net/forum?id=xXTkbTBmqq.
Nielsen, S., Teo, R., Abdullaev, L., and Nguyen, T. M.
Tight clusters make specialized experts. In The Thirteenth
International Conference on Learning Representations,
2025. URL https://openreview.net/forum
?id=Pu3c0209cx.
OpenAI, :, Agarwal, S., Ahmad, L., Ai, J., Altman, S., Ap-
plebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker,
B., Bao, H., Barak, B., Bennett, A., Bertao, T., Brett,
N., Brevdo, E., Brockman, G., Bubeck, S., Chang, C.,
Chen, K., Chen, M., Cheung, E., Clark, A., Cook, D.,
Dukhan, M., Dvorak, C., Fives, K., Fomenko, V., Garipov,
T., Georgiev, K., Glaese, M., Gogineni, T., Goucher, A.,
Gross, L., Guzman, K. G., Hallman, J., Hehir, J., Hei-
decke, J., Helyar, A., Hu, H., Huet, R., Huh, J., Jain, S.,
Johnson, Z., Koch, C., Kofman, I., Kundel, D., Kwon,
J., Kyrylov, V., Le, E. Y., Leclerc, G., Lennon, J. P.,
Lessans, S., Lezcano-Casado, M., Li, Y., Li, Z., Lin, J.,
Liss, J., Lily, Liu, Liu, J., Lu, K., Lu, C., Martinovic, Z.,
McCallum, L., McGrath, J., McKinney, S., McLaughlin,
A., Mei, S., Mostovoy, S., Mu, T., Myles, G., Neitz, A.,
12


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Nichol, A., Pachocki, J., Paino, A., Palmie, D., Pantu-
liano, A., Parascandolo, G., Park, J., Pathak, L., Paz, C.,
Peran, L., Pimenov, D., Pokrass, M., Proehl, E., Qiu, H.,
Raila, G., Raso, F., Ren, H., Richardson, K., Robinson,
D., Rotsted, B., Salman, H., Sanjeev, S., Schwarzer, M.,
Sculley, D., Sikchi, H., Simon, K., Singhal, K., Song, Y.,
Stuckey, D., Sun, Z., Tillet, P., Toizer, S., Tsimpourlas, F.,
Vyas, N., Wallace, E., Wang, X., Wang, M., Watkins, O.,
Weil, K., Wendling, A., Whinnery, K., Whitney, C., Wong,
H., Yang, L., Yang, Y., Yasunaga, M., Ying, K., Zaremba,
W., Zhan, W., Zhang, C., Zhang, B., Zhang, E., and Zhao,
S. gpt-oss-120b & gpt-oss-20b model card, 2025. URL
https://arxiv.org/abs/2508.10925.
Pham, Q., Do, G., Nguyen, H., Nguyen, T., Liu, C., Sartipi,
M., Nguyen, B. T., Ramasamy, S., Li, X., Hoi, S., and Ho,
N. Competesmoe – effective training of sparse mixture
of experts via competition, 2024.
Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C.,
Hockenmaier, J., and Lazebnik, S. Flickr30k entities:
Collecting region-to-phrase correspondences for richer
image-to-sentence models. In 2015 IEEE International
Conference on Computer Vision (ICCV), pp. 2641–2649,
2015. doi: 10.1109/ICCV.2015.303.
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,
J., Krueger, G., and Sutskever, I. Learning transferable
visual models from natural language supervision.
In
Meila, M. and Zhang, T. (eds.), Proceedings of the 38th
International Conference on Machine Learning, volume
139 of Proceedings of Machine Learning Research, pp.
8748–8763. PMLR, 18–24 Jul 2021. URL https://
proceedings.mlr.press/v139/radford21a.
html.
Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M.,
Jenatton, R., Susano Pinto, A., Keysers, D., and Houlsby,
N. Scaling vision with sparse mixture of experts. In
Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and
Vaughan, J. W. (eds.), Advances in Neural Information
Processing Systems, volume 34, pp. 8583–8595. Curran
Associates, Inc., 2021. URL https://proceeding
s.neurips.cc/paper_files/paper/2021/
file/48237d9f2dea8c74c2a72126cf63d93
3-Paper.pdf.
Rousseeuw, P. J. Silhouettes: A graphical aid to the in-
terpretation and validation of cluster analysis.
Jour-
nal of Computational and Applied Mathematics, 20:
53–65, 1987. ISSN 0377-0427. doi: https://doi.or
g/10.1016/0377-0427(87)90125-7.
URL https:
//www.sciencedirect.com/science/arti
cle/pii/0377042787901257.
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.
Winogrande: an adversarial winograd schema challenge
at scale. Commun. ACM, 64(9):99–106, August 2021.
ISSN 0001-0782. doi: 10.1145/3474381. URL https:
//doi.org/10.1145/3474381.
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
Q., Hinton, G., and Dean, J. Outrageously large neural
networks: The sparsely-gated mixture-of-experts layer.
In International Conference on Learning Representations,
2017. URL https://openreview.net/forum
?id=B1ckMDqlg.
Shen, L., Chen, G., Shao, R., Guan, W., and Nie, L. Mome:
Mixture of multimodal experts for generalist multimodal
large language models. In Globerson, A., Mackey, L., Bel-
grave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C.
(eds.), Advances in Neural Information Processing Sys-
tems, volume 37, pp. 42048–42070. Curran Associates,
Inc., 2024. doi: 10.52202/079017-1330. URL https:
//proceedings.neurips.cc/paper_files
/paper/2024/file/4a3a14b9536806a0522
930007c5512f7-Paper-Conference.pdf.
Shen, S., Yao, Z., Li, C., Darrell, T., Keutzer, K., and He,
Y. Scaling vision-language models with sparse mixture
of experts. In Bouamor, H., Pino, J., and Bali, K. (eds.),
Findings of the Association for Computational Linguis-
tics: EMNLP 2023, pp. 11329–11344, Singapore, De-
cember 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.findings-emnlp.758. URL
https://aclanthology.org/2023.findin
gs-emnlp.758.
Staats, M., Thamm, M., and Rosenow, B. Small singular
values matter: A random matrix analysis of transformer
models. In The Thirty-ninth Annual Conference on Neural
Information Processing Systems, 2026. URL https:
//openreview.net/forum?id=YwMHgqHYyT.
Sun, Q., Cetin, E., and Tang, Y. Transformer-squared: Self-
adaptive LLMs. In The Thirteenth International Confer-
ence on Learning Representations, 2025. URL https:
//openreview.net/forum?id=dh4t9qmcvK.
Tang, A., Shen, L., Luo, Y., Xie, S., Hu, H., Zhang, L.,
Du, B., and Tao, D. Zero-shot sparse mixture of low-
rank experts construction from pre-trained foundation
models.
IEEE Transactions on Pattern Analysis and
Machine Intelligence, 48(2):1145–1157, 2026. doi: 10.1
109/TPAMI.2025.3612480.
Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X.,
Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford al-
paca: An instruction-following llama model. https://
github.com/tatsu-lab/stanford_alpaca,
2023.
13


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Team, Q. Qwen1.5-moe: Matching 7b model performance
with 1/3 activated parameters”, February 2024. URL
https://qwenlm.github.io/blog/qwen-m
oe/.
Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Thamm, M., Staats, M., and Rosenow, B. Random matrix
analysis of deep neural network weight matrices. Physical
Review E, 106(5), November 2022. ISSN 2470-0053. doi:
10.1103/physreve.106.054124. URL http://dx.doi
.org/10.1103/PhysRevE.106.054124.
Wang, X., Alam, S., Wan, Z., Shen, H., and Zhang, M.
SVD-LLM v2: Optimizing singular value truncation for
large language model compression. In Chiruzzo, L., Rit-
ter, A., and Wang, L. (eds.), Proceedings of the 2025
Conference of the Nations of the Americas Chapter of
the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp.
4287–4296, Albuquerque, New Mexico, April 2025a. As-
sociation for Computational Linguistics. ISBN 979-8-
89176-189-6. doi: 10.18653/v1/2025.naacl-long.217.
URL https://aclanthology.org/2025.na
acl-long.217/.
Wang, X., Zheng, Y., Wan, Z., and Zhang, M. SVD-LLM:
Truncation-aware singular value decomposition for large
language model compression. In The Thirteenth Interna-
tional Conference on Learning Representations, 2025b.
URL https://openreview.net/forum?id=
LNYIUouhdt.
Wolpert, D. and Macready, W. No free lunch theorems
for optimization. IEEE Transactions on Evolutionary
Computation, 1(1):67–82, 1997. doi: 10.1109/4235.585
893.
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
Y. HellaSwag: Can a machine really finish your sen-
tence?
In Korhonen, A., Traum, D., and M`arquez,
L. (eds.), Proceedings of the 57th Annual Meeting of
the Association for Computational Linguistics, pp. 4791–
4800, Florence, Italy, July 2019. Association for Compu-
tational Linguistics. doi: 10.18653/v1/P19-1472. URL
https://aclanthology.org/P19-1472/.
Zhang, J., Qu, X., Zhu, T., and Cheng, Y. CLIP-MoE:
Towards building mixture of experts for CLIP with di-
versified multiplet upcycling.
In Christodoulopoulos,
C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Pro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 5406–5419,
Suzhou, China, November 2025. Association for Com-
putational Linguistics. ISBN 979-8-89176-332-6. doi:
10.18653/v1/2025.emnlp-main.275. URL https://ac
lanthology.org/2025.emnlp-main.275/.
Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A.,
and Tian, Y. GaLore: Memory-efficient LLM training
by gradient low-rank projection. In Salakhutdinov, R.,
Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett,
J., and Berkenkamp, F. (eds.), Proceedings of the 41st
International Conference on Machine Learning, volume
235 of Proceedings of Machine Learning Research, pp.
61121–61143. PMLR, 21–27 Jul 2024. URL https:
//proceedings.mlr.press/v235/zhao24s
.html.
Zhou, Y., Zhao, Z., Cheng, D., Wu, Z., Gui, J., Yang, Y., Wu,
F., Cheng, Y., and Fan, H. Dropping experts, recombining
neurons: Retraining-free pruning for sparse mixture-of-
experts LLMs. In Christodoulopoulos, C., Chakraborty,
T., Rose, C., and Peng, V. (eds.), Findings of the Associ-
ation for Computational Linguistics: EMNLP 2025, pp.
15169–15186, Suzhou, China, November 2025. Associa-
tion for Computational Linguistics. ISBN 979-8-89176-
335-7. doi: 10.18653/v1/2025.findings-emnlp.820. URL
https://aclanthology.org/2025.findin
gs-emnlp.820/.
14


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
A. Supplementary Material for “Eigenvectors of Experts are Training-free Non-collapsing
Routers”
The appendix is organized as follows: Section B presents the additional theoretical analysis, while Section C provides
algorithm of the SSMoE. Supplementary experimental results are reported in Section D, and full implementation details are
described in Section E.
B. Additional Theoretical Analysis
Lemma B.1 (Approximate orthogonality via Gaussian representation). Let eC(1), . . . , eC(N) ∈Rd be spectral embeddings of
N experts. Assume that for each i the embedding eC(i) is distributed approximately as a random unit vector (Thamm et al.,
2022; Staats et al., 2026) drawn uniformly from the sphere Sd−1, and that different experts’ embeddings are independent.
Then for any fixed distinct indices i̸ = j and for every ε > 0,
P
 | eC(i)⊤eC(j)| ≥ε

−→0
as d →∞.
In particular, eC(i)⊤eC(j)
p−→0, so different expert embeddings are (asymptotically) nearly orthogonal.
Lemma B.1 establishes that the eigenvectors of expert weight matrices provide more effective routing signals than standard
MoE routers, owing to their orthogonality properties. This theoretical insight is corroborated by the empirical results in
Figure 6, which show that eigenvector-based routers exhibit near-orthogonality, whereas traditional routers suffer from high
correlations.
B.1. Proof for Lemma B.1
Proof. Represent independent standard normal vectors X, Y ∼N(0, Id). A uniformly random unit vector on Sd−1 has the
same distribution as U := X/∥X∥; similarly set V := Y/∥Y ∥. Thus for two independent uniform unit vectors we may
write their inner product as
⟨U, V ⟩=
⟨X, Y ⟩
∥X∥∥Y ∥.
By the strong law of large numbers,
∥X∥
√
d
=
v
u
u
t1
d
d
X
k=1
X2
k
a.s.
−−→1,
∥Y ∥
√
d
a.s.
−−→1.
Moreover, ⟨X, Y ⟩= Pd
k=1 XkYk is a sum of i.i.d. mean-zero terms with variance 1, so by the central limit theorem
⟨X, Y ⟩
√
d
d
−→N(0, 1).
Combining these two facts yields
√
d ⟨U, V ⟩=
⟨X, Y ⟩/
√
d
(∥X∥/
√
d)(∥Y ∥/
√
d)
d
−→N(0, 1),
hence for every fixed ε > 0,
P
 |⟨U, V ⟩| ≥ε

= P
√
d |⟨U, V ⟩| ≥ε
√
d

≤P

|N(0, 1)| ≥ε
√
d/2

+ o(1) −→0.
Therefore ⟨U, V ⟩
p−→0 as d →∞.
The lemma follows by taking U = eC(i) and V = eC(j) under the stated Random matrix theory (RMT) assumption that
the spectral embeddings behave like independent uniform unit vectors. The same argument extends to any fixed finite
collection of embeddings: all pairwise inner products vanish in probability as d →∞, so the collection is asymptotically
orthogonal.
15


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
B.2. Proof for Lemma 3.1
Proof. Expand the product for i̸ = j and take expectation:
E[ViVj] = α2E[fEV,ifEV,j] + (1 −α)2E[sisj] + α(1 −α)
 E[fEV,isj] + E[fEV,jsi]

.
By assumption (1) the first term is zero, and by assumption (2) the cross terms are zero. The remaining term is (1−α)2E[sisj],
which proves the identity. The inequality and the monotonic dependence on α follow immediately because (1 −α)2 < 1 for
α ∈(0, 1].
Extension to selective SSMoE.
The proof above corresponds to the router-independent case, such as the simple-average
EV variant. For the router-guided selective variant, the learned router is used to select a subset of eigenvector directions. This
selection can introduce mild dependence between fEV(x) and s(x), so the exact decorrelation condition E[fEV,isj] = 0
may not hold. We therefore consider the relaxed condition
|E[fEV,isj]| ≤ε,
∀i, j,
for some small ε ≥0.
Under this relaxed condition, using the same expansion as above gives
E[ViVj] = α2E[fEV,ifEV,j] + (1 −α)2E[sisj] + α(1 −α)
 E[fEV,isj] + E[fEV,jsi]

.
By EV orthogonality,
E[fEV,ifEV,j] = 0,
i̸ = j.
Taking absolute values and applying the triangle inequality, we obtain
|E[ViVj]| ≤(1 −α)2 |E[sisj]| + α(1 −α) (|E[fEV,isj]| + |E[fEV,jsi]|)
≤(1 −α)2 |E[sisj]| + 2α(1 −α)ε.
Thus, selective SSMoE reduces pairwise router-logit correlation by the same (1 −α)2 factor as in Lemma 2, up to an
additional cross-correlation error term 2α(1 −α)ε. When ε is small, the selective variant preserves the decorrelation effect
while allowing the learned router to filter more relevant eigenvector directions.
C. SSMoE Algorithm
Algorithm 1 summarizes the SSMoE layer. Importantly, eigenvectors are extracted from the expert Gram matrices rather
than from the rectangular FFN weights directly, ensuring that the resulting spectral descriptors lie in the token representation
space RD and can therefore be used as routing projections.
D. Additional Experiments
D.1. Vision Tasks
For image-text retrieval, we conduct evaluations on the COCO (Lin et al., 2014) and Flickr30k (Plummer et al., 2015)
datasets. For zero-shot image classification, we assess performance on six widely used benchmarks: CIFAR-10, CIFAR-
100 (Krizhevsky, 2009), STL-10 (Coates et al., 2011), Caltech101 (Fei-Fei et al., 2004), ImageNet-1K (Deng et al., 2009),
and ImageNet-O (Hendrycks et al., 2021b).
Zero-Shot Image Classification Evaluation. Table 7 presents the zero-shot image classification accuracy of CLIP-MoE
and our proposed SSMoE across six widely used benchmarks under both clean and corrupted image conditions. Under the
clean setting, SSMoE consistently outperforms CLIP-MoE on all datasets, achieving an average accuracy improvement of
+0.4%. Notably, the largest gain is observed on Caltech101, where SSMoE improves accuracy by +1.1%.
In the corrupted setting, where input images are perturbed with random noise, SSMoE maintains its advantage, again
outperforming CLIP-MoE across all benchmarks with an average improvement of +0.4%. The most significant gain is on
ImageNet-O: +0.7%, which is known for its challenging out-of-distribution samples. These results demonstrate that SSMoE
not only enhances performance in standard scenarios but also exhibits greater robustness under distributional shifts and
noise.
16


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Algorithm 1 SSMoE Layer with Eigenvector-based Routing
Input: Tokens X ∈RB×L×D; experts {Ei}N
i=1; learned router logits S(X) ∈RB×L×N; FFN weights W (1)
i
, W (2)
i
;
router embeddings {ri}N
i=1; Top-c, Top-k, mixing factor α ∈[0, 1].
Output: Y ∈RB×L×D.
Pre-compute eigenvector descriptors.
for i = 1, . . . , N do
Ai ←W (1)
i
(W (1)
i
)⊤;
Bi ←(W (2)
i
)⊤W (2)
i
{v(i,1)
j
} ←EigVec(Ai);
{v(i,2)
j
} ←EigVec(Bi)
C(i)
1
←TopC({v(i,1)
j
}, ri, c)
C(i)
2
←TopC({v(i,2)
j
}, ri, c)
Ci ←1
2
h
mean(C(i)
1 ) + mean(C(i)
2 )
i
end for
Build eigenvector router.
WEV ←[C1, . . . , CN] ∈RD×N
Compute routing probabilities.
FEV(X) ←XWEV
PEV(X) ←softmax(FEV(X))
PR(X) ←softmax(S(X))
P(X) ←αPEV(X) + (1 −α)PR(X)
Select experts and compute output.
for all tokens xb,ℓ∈X do
Ib,ℓ←TopK(P(xb,ℓ), k)
yb,ℓ←P
i∈Ib,ℓPi(xb,ℓ)Ei(xb,ℓ)
end for
return Y = {yb,ℓ}
Clean
Dataset
CLIP-MoE
SSMoE
∆
CIFAR-10
95.7
96.0
+0.3
CIFAR-100
79.6
79.9
+0.3
STL-10
99.2
99.3
+0.1
Caltech101
83.6
84.7
+1.1
ImageNet-1K
74.6
74.9
+0.3
ImageNet-O
33.5
33.7
+0.2
Avg.
77.7
78.1
+0.4
Corrupt
Dataset
CLIP-MoE
SSMoE
∆
CIFAR-10
66.7
66.9
+0.2
CIFAR-100
34.4
34.9
+0.5
STL-10
85.8
86.1
+0.3
Caltech101
76.0
76.2
+0.2
ImageNet-1K
46.6
46.9
+0.3
ImageNet-O
31.0
31.7
+0.7
Avg.
56.8
57.1
+0.4
Table 7. Classification accuracy (%) of CLIP-MoE and SSMoE on multiple datasets. The ”+/-” row indicates the improvement of SSMoE
over CLIP-MoE. Best results per row are in bold.
17


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
D.2. LLMs Finetuning Tasks
We showcase the effectiveness of our method through supervised fine-tuning on OLMoE-7B (Muennighoff et al., 2025).For
training, we adopt Supervised Fine-Tuning (SFT) using GaLore (Zhao et al., 2024), which enables full-parameter optimiza-
tion with improved memory efficiency compared to typical low-rank adaptation methods such as LoRA. The fine-tuning is
conducted on the Alpaca dataset (Taori et al., 2023) for 2,000 steps, with evaluation performed every 10 steps.
Reasoning Evaluation. We evaluate the performance of OLMoE on six key reasoning benchmarks. The ARC-C and ARC-
E benchmarks (Clark et al., 2018) assess scientific question answering at varying difficulty levels. BoolQ (Clark et al., 2019)
evaluates reading comprehension through yes/no questions. PIQA (Bisk et al., 2020) measures a model’s ability to perform
physical commonsense reasoning by selecting the most plausible outcome in everyday situations. MMLU (Hendrycks
et al., 2021a) tests broad general knowledge and reasoning capabilities, while WinoGrande (Sakaguchi et al., 2021) targets
coreference resolution through challenging pronoun disambiguation tasks.
Tasks
Method
ARC-C
ARC-E
BoolQ
Avg.
OLMoE-7B
SMoE
52.7
78.2
76.4
69.1
SSMoE
52.3
80.9
77.5
70.2
∆
-0.4
+2.7
+1.1
+1.1
Tasks
Method
PiQA
MMLU
WinoGrande
Avg.
OLMoE-7B
SMoE
81.2
50.8
69.9
67.3
SSMoE
82.5
51.0
71.2
68.2
∆
+1.3
+0.2
+1.3
+0.9
Table 8. Performance comparison between SSMoE and the SMoE baseline on reasoning tasks. Higher is better, best results per row are in
bold.
Overall, SSMoE outperforms the SMoE baseline on six benchmarks for the OLMoE model as Table 8. Notably, on ARC-E,
PIQA, and WinoGrande, SSMoE achieves gains of 1–3% over the baselines. These results suggest that SSMoE is not only
effective in training-free settings for downstream tasks, but also excels at handling complex reasoning tasks.
D.3. LLMs Embeding Evaluation
We present a comprehensive evaluation of three state-of-the-art SMoE models: OLMoE-7B (Table 9), Qwen-MoE-7B
(Table 10), and DeepSeekMoE-16B (Table 11). Our results highlight the effectiveness of our method across different
models and prompting conditions, demonstrating consistent improvements over baseline approaches such as the SMoE.
D.4. Comparison with Training-free Routing
We compare SSMoE with two training-free routing baselines on GPT-OSS-20B under the 10-shot setting: hash-based
routing and random routing. Hash-based routing assigns tokens to experts using a fixed token-dependent hash function,
while random routing samples experts without using token–expert compatibility. As shown in Table 12, SSMoE consistently
improves over both baselines on most tasks and achieves the best average performance. In particular, SSMoE improves the
average score from 36.1 to 54.1 over hash-based routing, corresponding to an absolute gain of +18.0 points. Compared
with random routing, SSMoE improves the average score from 45.0 to 54.1, yielding an absolute gain of +9.1 points. These
results suggest that the proposed eigenvector-based routing captures meaningful expert specialization beyond input-agnostic
or heuristic expert assignment strategies.
D.5. Comparison on Distribution-Sensitive Metrics
To further assess whether the gains of SSMoE arise from meaningful improvements rather than uncontrolled changes in
model behavior (Dutta et al., 2024), we report additional distribution-sensitive metrics, including answer flips and perplexity.
These metrics are less directly optimizable than task accuracy and therefore provide complementary evidence about the
stability and reliability of the proposed routing mechanism.
Answer flips.
We first analyze prediction flips between the original model and SSMoE. In particular, we distinguish
harmful flips, where a previously correct prediction becomes incorrect (C→W), from beneficial flips, where a previously
18


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
incorrect prediction becomes correct (W→C). As shown in Table 13, SSMoE produces more beneficial flips than harmful
ones on average. Across eight benchmarks, SSMoE achieves an average W→C rate of 13.5%, compared to a C→W rate
of 9.2%, yielding a net accuracy gain of +5.3%. This indicates that the observed improvements are primarily driven by
correcting errors rather than by random prediction changes.
Perplexity.
We also evaluate perplexity on WikiText-2 using GPT-OSS-20B. Perplexity provides a direct measure of
whether routing changes introduce harmful distribution shift in language modeling behavior. As reported in Table 14,
SSMoE achieves lower perplexity than both the original model and the Router baseline. This suggests that SSMoE improves
compatibility with the data distribution rather than degrading general language modeling quality. We note that the absolute
perplexity values are higher than those commonly reported for standard language-modeling benchmarks because GPT-OSS-
20B is an instruction-tuned LLM. Therefore, the main comparison is the relative difference under the same evaluation setup.
The lower perplexity of SSMoE compared with the Router baseline further indicates that the improvement is not merely
caused by modifying the routing behavior, but by selecting experts more effectively.
D.6. Correlation of Expert Selection between SSMoE and SMoE Routers
We further analyze whether SSMoE simply reproduces the expert choices of the original SMoE router or provides a
complementary routing signal. To this end, we measure the mean overlap between the experts selected by the eigenvector-
based routing mechanism and those selected by the original learned router. A high overlap would suggest that SSMoE
mostly mimics the baseline router, whereas a low overlap indicates that the eigenvector descriptors induce substantially
different expert selection patterns.
Table 15 reports the mean overlap across eight benchmarks. SSMoE exhibits consistently low overlap with the original
router, with an average overlap of only 0.1531. This shows that the eigenvector-based routing mechanism does not trivially
replicate the original router’s decisions. Instead, it selects a different subset of experts, suggesting that the eigenvector
descriptors capture complementary information about expert specialization. This complementary selection signal helps
explain why SSMoE can improve performance over the original routing mechanism.
D.7. Effect of Expert Capacity on Mathematical Reasoning
We further investigate the performance drop observed on GSM8K under the pruned expert setting. Our analysis suggests that
this drop is mainly caused by reducing the available expert pool, which can remove task-critical experts for reasoning-heavy
domains such as mathematics. Unlike some knowledge or commonsense benchmarks, mathematical reasoning often relies
on a broader set of specialized computational and symbolic capabilities. Therefore, pruning experts may disproportionately
affect tasks such as GSM8K by excluding experts that are important for multi-step arithmetic reasoning.
To verify this, we compare SSMoE under two settings: using the full expert pool and using only 75% of the experts. As
shown in Table 16, SSMoE with full expert capacity substantially outperforms both the original model and the pruned
SSMoE variant. Specifically, full-capacity SSMoE improves over the original model by +6.6 points and over 75% of the
experts by +12.6 points. These results indicate that the degradation on GSM8K is not due to the eigenvector-based routing
mechanism itself, but rather to the loss of important experts caused by pruning. This finding suggests that expert capacity is
particularly important for mathematical reasoning tasks. While SSMoE can improve expert selection, aggressive expert
pruning may remove experts that are necessary for solving reasoning-intensive problems. Thus, for mathematics-heavy
benchmarks, preserving the full expert pool is preferable when the goal is to maximize accuracy.
E. Implementation Details
For the Large Reasoning Model experiments, we evaluate our method using the lm-evaluation-harness library (Gao et al.,
2024). All experiments with the GPT-OSS models (OpenAI et al., 2025) are performed on two H200 GPUs.
For the Large Language Models experiments, we implement our method based on the publicly available MoEE imple-
mentation (Li & Zhou, 2025)1. Due to resource limitations, both our method and the baselines are evaluated using 4-bit
quantization with a batch size of 128. Experiments with the OLMoE-7B model are conducted on a single H100 GPU,
whereas the Qwen-MoE-7B and DeepSeekMoE-16B models are tested using two H100 GPUs.
1https://github.com/tianyi-lab/MoE-Embedding
19


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
For Vision Language models, we implement SSMoE as a modular extension to the publicly available CLIP-MoE frame-
work (Zhang et al., 2025)2. All experiments involving vision-language tasks are conducted using a single NVIDIA H100
GPU.
2https://github.com/OpenSparseLLMs/CLIP-MoE
20


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Category
Model
Dataset
Setting
Router
SMoE
MoEE
EV
SSMoE
Classification
OLMoE-7B
Emotion
None
24.1
24.5
25.1
32.3
37.4
Prompt
27.6
49.9
44.5
29.9
53.5
Toxic
None
51.9
58.9
51.9
52.5
60.8
Prompt
52.3
65.2
53.4
52.1
67.4
Tweet
None
47.7
46.8
48.4
53.0
53.5
Prompt
49.5
58.0
57.2
48.7
60.9
Clustering
OLMoE-7B
Medrxiv
None
15.0
17.6
17.4
15.5
19.0
Prompt
15.8
23.9
22.0
14.7
27.5
20Groups
None
12.4
11.8
11.5
12.9
13.7
Prompt
16.7
25.7
24.4
12.5
32.7
Pair Classification
OLMoE-7B
SemEval
None
43.6
35.8
43.6
50.7
44.8
Prompt
45.7
46.7
53.8
47.2
55.7
URLCorpus
None
47.0
42.4
47.8
67.7
57.5
Prompt
61.4
77.4
78.2
72.4
80.5
Reranking
OLMoE-7B
Ask
None
41.3
41.0
41.4
43.4
42.2
Prompt
43.4
51.9
50.2
44.0
51.5
SciDocs
None
45.5
46.3
50.8
49.6
57.4
Prompt
53.6
69.6
75.1
50.5
76.8
StackOver
None
25.8
24.8
26.4
30.5
27.5
Prompt
28.1
32.5
34.3
30.9
34.5
STS
OLMoE-7B
Biosses
None
39.3
13.6
29.7
65.2
36.1
Prompt
51.2
61.8
70.2
53.8
75.2
SickR
None
50.3
46.3
53.0
55.7
55.6
Prompt
51.9
65.7
66.1
49.0
66.8
STS12
None
40.1
8.6
37.8
57.2
46.7
Prompt
51.3
53.8
63.6
49.7
65.7
STS13
None
40.5
21.1
43.4
56.1
52.8
Prompt
52.5
66.5
72.7
53.9
75.5
STS14
None
29.5
13.4
31.7
46.4
40.3
Prompt
41.1
56.8
64.2
42.9
66.6
STS15
None
30.8
27.8
33.3
51.6
43.1
Prompt
46.4
69.3
66.4
47.4
67.1
STS16
None
46.5
38.9
45.8
57.5
52.1
Prompt
52.4
70.1
68.3
51.0
69.5
STSBen
None
42.2
23.4
44.5
52.3
49.5
Prompt
48.6
63.6
70.7
41.8
71.8
Summarization
OLMoE-7B
Medrxiv
None
28.4
20.9
29.8
30.0
31.0
Prompt
25.6
28.9
30.4
27.0
30.0
Table 9. Performance comparison of Router, SMoE, MoEE, SSMoE, EV, and SSMoE across MTEB Tasks with OLMoE-7B models. The
best result for each row is highlighted in bold.
21


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Category
Model
Dataset
Setting
Router
SMoE
MoEE
EV
SSMoE
Classification
Qwen-MoE-7B
Emotion
None
27.2
33.9
34.3
24.4
35.3
Prompt
37.0
48.5
47.2
38.6
51.0
Toxic
None
53.0
61.1
52.9
52.9
61.9
Prompt
53.4
64.5
54.1
53.8
65.3
Tweet
None
51.1
55.9
55.9
52.9
57.6
Prompt
56.1
61.1
60.7
58.8
62.3
Clustering
Qwen-MoE-7B
Medrxiv
None
15.3
23.3
23.0
15.2
26.0
Prompt
14.2
24.6
21.8
15.3
30.0
20Groups
None
12.0
31.5
27.4
12.2
36.1
Prompt
14.4
43.8
38.4
17.3
49.1
Pair Classification
Qwen-MoE-7B
SemEval
None
42.0
38.8
42.5
43.1
46.6
Prompt
47.0
52.4
52.4
49.9
56.2
URLCorpus
None
49.8
54.9
60.6
50.8
64.1
Prompt
56.7
68.7
68.2
61.2
74.3
Reranking
Qwen-MoE-7B
Ask
None
43.1
45.8
47.3
43.1
48.8
Prompt
43.3
48.3
49.5
44.4
51.4
SciDocs
None
49.6
60.6
67.0
50.9
66.0
Prompt
50.9
60.1
68.7
52.2
65.8
StackOver
None
26.2
29.5
31.1
26.9
32.6
Prompt
28.8
31.3
35.2
29.6
35.4
STS
Qwen-MoE-7B
Biosses
None
33.8
32.5
49.6
40.0
65.6
Prompt
55.1
55.8
68.4
58.2
67.7
SickR
None
51.0
55.5
61.0
51.2
62.1
Prompt
50.2
59.7
64.3
52.0
67.2
STS12
None
40.2
16.9
46.3
41.9
48.6
Prompt
49.3
25.0
59.2
51.8
61.8
STS13
None
38.1
42.9
56.7
41.1
64.2
Prompt
53.3
57.5
73.4
57.2
75.3
STS14
None
28.1
26.5
45.4
30.8
51.3
Prompt
40.4
38.8
60.0
44.4
64.4
STS15
None
34.8
40.5
46.1
37.0
51.7
Prompt
40.7
52.3
58.8
46.5
64.3
STS16
None
47.6
51.0
58.1
49.8
62.0
Prompt
51.6
64.2
65.7
53.3
67.6
STSBen
None
37.0
37.7
50.9
40.6
57.7
Prompt
45.6
47.8
64.5
50.1
69.3
Summarization
Qwen-MoE-7B
Medrxiv
None
28.3
13.4
31.2
30.3
29.6
Prompt
27.0
23.0
27.3
26.4
30.7
Table 10. Performance comparison of Router, SMoE, MoEE, SSMoE, EV, and SSMoE across MTEB Tasks with Qwen-MoE-7B models.
The best result for each row is highlighted in bold.
22


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Category
Model
Dataset
Setting
Router
SMoE
MoEE
EV
SSMoE
Classification
DeepSeekMoE-16B
Emotion
None
26.1
27.4
27.6
27.1
31.6
Prompt
37.9
48.3
46.4
38.1
48.9
Toxic
None
53.3
60.4
53.1
52.7
59.0
Prompt
53.1
62.4
53.6
54.3
67.0
Tweet
None
51.0
51.9
52.6
52.1
55.2
Prompt
54.9
58.4
58.9
57.1
60.9
Clustering
DeepSeekMoE-16B
Medrxiv
None
15.1
23.0
22.0
15.8
24.3
Prompt
17.0
25.7
24.0
17.0
27.4
20Groups
None
11.7
13.2
13.7
12.8
18.9
Prompt
18.6
32.3
33.0
20.2
35.1
Pair Classification
DeepSeekMoE-16B
SemEval
None
44.6
40.2
43.5
44.3
41.8
Prompt
48.4
47.2
51.3
48.8
53.4
URLCorpus
None
46.4
41.7
48.6
49.1
60.6
Prompt
66.5
72.4
75.4
69.7
75.6
Reranking
DeepSeekMoE-16B
Ask
None
41.7
41.1
42.3
41.4
43.1
Prompt
43.5
43.8
46.9
44.9
50.4
SciDocs
None
48.2
50.6
57.1
49.7
61.7
Prompt
58.3
65.6
72.6
60.1
72.6
StackOver
None
25.7
24.9
27.3
26.5
30.1
Prompt
29.7
27.6
32.3
30.6
33.5
STS
DeepSeekMoE-16B
Biosses
None
29.5
31.7
26.8
30.7
54.4
Prompt
47.0
40.1
57.6
48.8
66.0
SickR
None
50.4
47.4
53.1
52.4
57.8
Prompt
56.0
61.9
65.8
58.2
66.8
STS12
None
44.0
4.3
45.0
45.6
46.7
Prompt
57.8
31.0
64.0
60.1
64.7
STS13
None
36.0
28.4
41.1
38.0
50.0
Prompt
55.3
56.0
70.9
60.8
74.3
STS14
None
25.4
12.0
28.2
28.3
40.3
Prompt
44.9
41.0
58.6
49.8
64.0
STS15
None
34.8
33.9
38.7
35.8
44.9
Prompt
49.7
46.5
58.5
54.3
64.1
STS16
None
44.9
34.4
46.9
49.1
55.1
Prompt
56.7
58.0
64.5
61.9
67.4
STSBen
None
36.6
18.3
42.1
40.0
51.8
Prompt
54.9
57.7
67.8
60.0
71.0
Summarization
DeepSeekMoE-16B
Medrxiv
None
24.9
22.0
24.4
28.9
29.9
Prompt
29.1
24.4
29.2
28.8
29.6
Table 11. Performance comparison of Router, SMoE, MoEE, SSMoE, EV, and SSMoE across MTEB Tasks with DeepSeekMoE-16B
models. The best result for each row is highlighted in bold.
23


---

Eigenvectors of Experts are Training-free Non-collapsing Routers
Task
Hash Router
Random Router
SSMoE (Ours)
ARC-C
23.3
33.5
52.1
ARC-E
36.7
65.8
82.4
BoolQ
59.3
66.5
72.4
GSM8K
14.4
33.1
30.8
HellaSwag
29.2
29.7
37.5
OBQA
26.2
20.6
23.0
PIQA
53.2
57.2
72.2
WinoGrande
46.9
53.3
62.2
Average
36.1
45.0
54.1
Table 12. Comparison with hash-based and random routing on GPT-OSS-20B under the 10-shot setting. SSMoE achieves the best average
performance and substantially outperforms hash-based routing.
Task
Original
SSMoE
Gap
FlipRate
C→W
W→C
ARC-C
44.2
52.1
+7.9
15.7
7.3
8.5
ARC-E
77.9
82.4
+4.5
8.7
3.4
5.3
BoolQ
62.8
72.4
+9.6
39.2
12.6
26.7
GSM8K
36.8
30.8
-6.0
39.6
25.6
14.0
HellaSwag
29.0
37.5
+8.5
12.1
2.4
9.7
OBQA
19.8
23.0
+3.2
12.0
3.4
8.6
PIQA
62.5
72.2
+9.7
21.6
5.1
16.4
WinoGrande
57.5
62.2
+4.7
32.9
13.8
19.1
Avg.
48.8
54.1
+5.3
22.7
9.2
13.5
Table 13. Flip analysis between the original model and SSMoE. C→W denotes harmful flips from correct to wrong predictions, while
W→C denotes beneficial flips from wrong to correct predictions. SSMoE produces more beneficial than harmful flips on average.
Data
Model
Method
PPL ↓
Original
82.12
WikiText-2
GPT-OSS-20B
Router
75.91
SSMoE
73.23
Table 14. Perplexity comparison on WikiText-2 using GPT-OSS-20B. Lower perplexity is better. SSMoE reduces perplexity compared
with both the original model and the Router baseline.
Task
Mean Overlap
ARC-C
0.1581
ARC-E
0.1586
BoolQ
0.1526
GSM8K
0.1581
HellaSwag
0.1488
OBQA
0.1462
PIQA
0.1540
WinoGrande
0.1486
Mean
0.1531
Table 15. Mean overlap between experts selected by SSMoE’s eigenvector-based routing and the original SMoE router. Lower overlap
indicates that SSMoE selects substantially different experts rather than simply mimicking the learned router.
Task
Orig.
SSMoE (Full)
SSMoE (75% Experts)
GSM8K
36.8
43.4
30.8
Table 16. Effect of expert capacity on GSM8K using GPT-OSS-20B. SSMoE performs strongly when the full expert pool is preserved,
while pruning to 75% experts causes a large degradation on this reasoning-heavy task.
24
