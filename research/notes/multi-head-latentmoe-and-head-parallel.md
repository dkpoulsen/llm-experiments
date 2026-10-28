---
title: 'Multi-Head LatentMoE and Head Parallel:'
id: multi-head-latentmoe-and-head-parallel
tags:
- cmp170hx-training-feasibility-a0d4fc
- moe-training
- head-parallel
- communication-efficiency
- expert-parallelism
- architecture-design
created: '2026-10-27T10:12:36.804502Z'
updated: '2026-10-27T19:00:31.162241Z'
source: https://arxiv.org/abs/2602.04870
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:36.800363Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Multi-Head LatentMoE and Head Parallel (Cui et al., Arizona State University,
  arXiv 2602.04870): an architecture + parallelism co-design that attacks the property
  that makes EP expensive on a slow fabric. EP''s communication volume and all-to-all
  latency are O(k) in activated experts, traffic is load-imbalanced, and routing-dependent
  dispatch needs a metadata exchange to stay deterministic. Multi-Head LatentMoE splits
  each token into Nh sub-tokens, each with its own router and expert set, and Head
  Parallel distributes sub-tokens to GPUs BEFORE any routing decision, so each GPU
  routes and computes entirely locally. Result: O(1) communication volume regardless
  of k, perfectly balanced traffic, deterministic communication — and still composable
  with EP. At k=4 the communication volume is 25% of EP''s. Trained on a 10B-token
  FineWeb-EDU subset on H100 80GB (NVLink): 1.11x faster at 2B total params (31.68
  vs 35.36 h) and 1.61x faster at 4B (34.41 vs 55.34 h) at comparable quality, with
  IO-aware routing (online top-k in SRAM) and a pure-PyTorch dropless expert computation
  to avoid custom kernels. Relevance: this is the strongest architectural counterexample
  to ''expert parallelism is ruled out by the fabric'' — MoE training can be made
  fabric-insensitive by moving the communication before routing, at the cost of a
  fixed per-layer all-to-all that is still host-staged on a no-P2P box.'
raw_file: raw/multi-head-latentmoe-and-head-parallel.pdf
doi: arXiv:2602.04870
utility_score: 13.0
---

Multi-Head LatentMoE and Head Parallel:
Communication-Efficient and Deterministic MoE Parallelism
Chenwei Cui * 1 Rockwell Jackson * 1 Benjamin Joseph Herrera * 1 Ana Mar´ıa T´arano 1 Hannah Kerner 1
Abstract
Large language models have transformed many
applications but remain expensive to train. Sparse
Mixture of Experts (MoE) addresses this through
conditional computation, with Expert Parallel
(EP) as the standard distributed training method.
However, EP has three limitations: communica-
tion cost grows linearly with the number of ac-
tivated experts k, load imbalance affects latency
and memory usage, and data-dependent commu-
nication requires metadata exchange. We propose
Multi-Head LatentMoE and Head Parallel (HP),
a new architecture and parallelism achieving
O(1) communication cost regardless of k, com-
pletely balanced traffic, and deterministic com-
munication, all while remaining compatible with
EP. To accelerate Multi-Head LatentMoE, we
propose IO-aware routing and expert computa-
tion. Compared to MoE with EP, Multi-Head
LatentMoE with HP trains up to 1.61× faster
while having identical performance. With dou-
bled granularity, it achieves higher overall perfor-
mance while still being 1.11× faster. Our method
makes multi-billion-parameter foundation model
research more accessible.
 https://github.com/kerner-lab/
Sparse-GPT-Pretraining
1. Introduction
Large language models have transformed applications such
as code generation (Achiam et al., 2023), but training them
remains expensive (Abnar et al., 2025). Pretraining a large
model consumes substantial compute and electricity (Be-
siroglu et al., 2024). These costs limit who can conduct
foundation model research. Reducing training costs while
maintaining model quality has become a key interest in
machine learning research.
*Equal contribution
1School of Computing and Augmented
Intelligence, Arizona State University, Tempe, USA. Correspon-
dence to: Hannah Kerner <hkerner@asu.edu>.
Preprint. February 5, 2026.
The sparse Mixture of Experts (MoE) architecture addresses
this challenge through conditional computation (Jacobs
et al., 1991). For each input token, a router activates only
a small subset of expert networks. This enables models
to scale capacity without proportionally increasing com-
pute (Shazeer et al., 2017; Fedus et al., 2022).
Expert Parallel (EP) is the standard distributed training
method for MoEs (Lepikhin et al., 2020; Fedus et al., 2022).
Each token is routed to k experts, replicated k times, and
sent to corresponding GPUs via all-to-all communication.
Results are processed by experts and returned via another
all-to-all operation (Shoeybi et al., 2019).
EP has three limitations. First, both communication vol-
ume and all-to-all latency are O(k). Second, expert load
imbalance further degrades all-to-all performance. Third,
non-deterministic communication necessitates an additional
all-to-all to exchange metadata (Shoeybi et al., 2019).
Recently, LatentMoE (Elango et al., 2026) addressed the
communication bottleneck by projecting tokens from hid-
den dimension d into a smaller latent dimension dh before
expert routing and computation. This reduced both per-
expert parameter loads and all-to-all traffic by a factor of
d
dh ,
enabling higher k values at similar communication costs.
However, LatentMoE still relied on Expert Parallel; com-
munication happened after routing, so load imbalance and
non-deterministic patterns persisted.
We propose Multi-Head LatentMoE and Head Parallel (HP)
to address these limitations. Multi-Head LatentMoE decom-
poses a single MoE into multiple independent smaller mod-
ules, splitting each input token into Nh sub-tokens. Each
sub-token is processed by an independent MoE module
with its own router and experts. HP exploits this structure.
The sub-tokens are distributed to GPUs before any rout-
ing decision. Each GPU completes all routing and expert
computation locally. This has three advantages over EP:
1. Communication volume is constant because each token
is sent exactly once before routing.
2. Traffic is balanced because sub-tokens are evenly dis-
tributed regardless of routing decisions.
3. The inter-GPU communication is deterministic because
it does not depend on routing decisions.
1
arXiv:2602.04870v1  [cs.LG]  4 Feb 2026


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
A naive implementation of Multi-Head LatentMoE would
multiply High Bandwidth Memory (HBM) usage and IO
cost by Nh, because it materializes the full routing scores
and expert activations for each head. We develop IO-aware
routing by doing online top-k directly in SRAM, reducing
HBM access from O(T · Ne + Ne) to O(T + Ne), where T
is the number of tokens and Ne is the number of experts. We
also develop IO-aware expert computation by formulating
expert computation as block-sparse attention and leveraging
FlexAttention (Dong et al., 2024), reducing HBM access
from O(T · de + de) to O(T + de), where de is the number
of neurons per expert. Both algorithms are exact.
We evaluate Multi-Head LatentMoE with HP on language
modeling using 10B tokens from FineWebEdu (Penedo
et al., 2024). Our method trains up to 1.61x faster than
the standard MoE with EP while having identical perfor-
mance. The inter-GPU communication volume is reduced
to 25% when k = 4. Under the same number of active pa-
rameters, our method can reach 6.9 percentage points higher
overall accuracy than MLP. When doubling the granular-
ity, our method achieves higher overall model performance
while still being 1.11× faster.
Our contributions are:
1. We propose Multi-Head LatentMoE and Head Paral-
lel, achieving O(1) communication volume for any k,
perfect load balance, and deterministic communication
patterns for distributed MoE training.
2. We develop exact IO-aware routing and expert com-
putation operations, making Multi-Head LatentMoE
practical and efficient.
3. We demonstrate that Multi-Head LatentMoE with HP
trains 1.61x faster than standard MoE with EP. With
two times the granularity, it further achieves higher
overall accuracy while still being 1.11× faster.
4. Our work accelerates ultra-sparse MoE pre-training,
making multi-billion-parameter foundation model re-
search more accessible to the academic research com-
munity.
2. Prerequisites
2.1. Notation
We use italic letters for scalars, bold lowercase for vectors,
and bold uppercase for matrices or tensors. B denotes batch
size, T denotes the number of tokens in a sequence, and t
indexes token position. The model hidden dimension is d,
while Nh and dh denote the number of heads and per-head
dimension, respectively. For mixture-of-experts layers, Ne
is the number of experts, k is the number of activated experts
per token, and de is the expert hidden dimension. The input
and output tokens at position t are xt ∈Rd and ot ∈Rd,
with projection matrices Win, Wout ∈Rd×d.
2.2. Mixture of Experts
Mixture of Experts (MoE) scales model capacity without
proportionally increasing computational cost (Shazeer et al.,
2017). This definition follows Mixtral (Jiang et al., 2024).
We re-use this for our baseline, as well as to help define
Multi-Head LatentMoE.
An MoE has Ne neural network modules called experts.
Among them, k experts are activated for each token (Shazeer
et al., 2017). For each input token xt, the corresponding
output ot is:
ot =
Ne
X
i=1
gi,tEi(xt),
(1)
gi,t =
exp(g′
i,t)
PNe
j=1 exp(g′
j,t)
,
(2)
g′
i,t =
(
si,t,
si,t ∈top-k({sj,t|1 ≤j ≤Ne})
−∞,
otherwise
,
(3)
si,t = r(xt)i.
(4)
where gi,t is the normalized gating value for the i-th expert;
Ei(·) denotes the i-th expert; si,t and g′
i,t are intermediate
values; top-k(·) selects the top k values from the set; and
r(·) is a linear mapping called the router.
To ensure router stability, our load balancing strategy
is global (Qiu et al., 2025) and auxiliary-loss-free (aux-
free) (Wang et al., 2024a). Following (Liu et al., 2024), we
replace the first two MoE layers with Multilayer Perceptrons
(MLPs) to mitigate router imbalance in early layers. The
router computation is performed in FP32.
2.3. Expert Parallel
Expert Parallel (EP) distributes expert weights across GPUs,
enabling training of large MoE models whose parameters
exceed what a single GPU can hold (Lepikhin et al., 2020).
The typical execution first duplicates tokens and clusters
them by target experts (Shoeybi et al., 2019). Metadata
about queue lengths are exchanged via all-to-all. Afterwards,
tokens are dispatched to destination GPUs through token
all-to-all. Received tokens are sorted by expert index, 1
for Grouped GEMM computation. Finally, outputs are re-
sorted and returned via another all-to-all, then sorted again
for weighted aggregation.
EP has three limitations. First, communication volume
increases with k due to token duplication. Second, if expert
queues have uneven lengths, all-to-all must wait for the
longest queue. Third, non-deterministic communication
patterns require exchanging metadata via all-to-all.
1For example, using moe sort chunks by index(.)
from NVIDIA Transformer Engine (NVIDIA, 2022).
2


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Figure 1. Comparison of feedforward architectures. (a) Standard MLP applies a single feedforward network to each token. (b) Standard
MoE uses a router to dynamically select experts from a single set. (c) LatentMoE performs routing first, then applies linear down-projection
before expert computation and linear up-projection afterward. (d) Multi-Head LatentMoE projects each token into multiple sub-tokens,
each processed by an independent MoE module with its own separately-trained router and expert set. Orange blocks denote activated
experts; gray blocks denote inactive experts. Black lines indicate data flow.
2.4. Hardware-Aware Design
Modern GPUs feature hierarchical memory with high-
bandwidth low-capacity on-chip SRAM and low-bandwidth
large-capacity off-chip HBM. The standard practice in ar-
chitecture research is to exploit this structure.
FlashAttention (Dao, 2023) uses tile streaming and recom-
putation to keep intermediate results in SRAM without
materializing the full attention matrix in HBM. FlexAtten-
tion (Dong et al., 2024) extends FlashAttention with block
sparse attention and addresses the software lottery problem,
where developers must write and maintain custom CUDA
programs for novel designs.
We apply the same principle to implement IO-aware rout-
ing (see Section 3.3), keeping the intermediates in SRAM.
For dropless (Gale et al., 2023) expert computation (see
Section 3.4), we reuse FlexAttention to achieve highly opti-
mized exact computation.
3. Method
3.1. Multi-Head LatentMoE
The three limitations of Expert Parallel share a common root
cause: the all-to-all traffic depends on routing decisions. We
need an architecture that decouples all-to-all from routing.
We propose Multi-Head LatentMoE (Figure 1). The key idea
is to project a token into multiple sub-tokens, each processed
by an independent MoE instance. This builds upon Latent-
MoE (Elango et al., 2026) and Multi-Head MoE (Huang
et al., 2024), but different from either.
Specifically, for each input token xt, Multi-Head Latent-
MoE first projects and splits it into a list of sub-tokens:
[xt,1, . . . , xt,Nh] = split(Winxt)
(5)
where Win ∈Rd×d is a learned linear projection into
the latent space, and split(·) divides the projected vector
into Nh sub-vectors of dimension dh. The common prac-
tice (Vaswani et al., 2017) is to set dh · Nh = d, though they
do not have to match.
The output of Multi-Head LatentMoE is:
ot = Wout · concat(f1(xt,1), . . . , fNh(xt,Nh))
(6)
where Wout ∈Rd×d projects the concatenated outputs back
to the original space; each fi is an independent MoE in-
stance as defined in Section 2.2, with its own router and
expert set, sharing no parameters.
Multi-Head LatentMoE has a total FLOPs equaling that of
an MoE, discounting the linear projections. However, a
naive implementation would store Nh sets of routing scores
and expert activations, multiplying the HBM and IO cost
by Nh. We address this challenge in Section 3.3 and Sec-
tion 3.4.
3.2. Head Parallel
Multi-Head LatentMoE creates independent sub-tokens and
MoE instances. This creates natural boundaries that divide
expert weights and tokens.
We propose Head Parallel, the distributed training method
for Multi-Head LatentMoE. The key idea is to move all-to-
all to before routing.
Specifically, let P denote the number of GPUs, where
P ≤Nh and Nh is divisible by P. Initially, each GPU
3


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Algorithm 1 IO-Aware Routing: Forward Pass
Require: Sub-tokens X ∈RB×T ×Nh×dh, router weights
Wr ∈RNh×dh×Ne, load-balancing bias b ∈RNh×Ne,
number of active experts k, block sizes N (tokens), M
(experts).
Ensure: Top-k scores Stop ∈RB×T ×Nh×k, top-k indices
Itop ∈NB×T ×Nh×k.
1: Divide X into ⌈T/N⌉blocks of N tokens each.
2: for each (b, tblock, h) in parallel do
3:
Load Xblock ∈RN×dh from HBM to SRAM.
4:
Initialize accumulator A ∈NN×k with 0.
5:
for eblock = 0 to ⌈Ne/M⌉−1 do
6:
Load Wr,block ∈Rdh×M and bblock ∈R1×M
from HBM to SRAM.
7:
On chip, compute Sblock
= XblockWr,block +
bblock ∈RN×M.
8:
On chip, pack scores and indices into 64-bit un-
signed integers.
9:
On chip, find top-k of Sblock along expert dimen-
sion.
10:
On chip, merge with A and retain the best k.
11:
end for
12:
On chip, unpack A to scores S′
top and indices Itop.
13:
On chip, compute Stop = S′
top−b[Itop] (remove bias).
14:
Write Stop and Itop to HBM.
15: end for
holds a tensor of sub-tokens with shape (B, T, Nh, dh).
All-to-all redistributes this tensor so that each GPU re-
ceives all sub-tokens for its assigned heads, yielding shape
(B, T, P, Nh/P, dh). After MoE computation, a reverse
all-to-all sends the outputs back to the original GPUs.
Head Parallel is not fundamentally bounded by Nh. It com-
poses naturally with other parallelism strategies, such as
Expert Parallel, to scale beyond Nh GPUs. The key insight
is that HP provides a communication-efficient highway for
distributing sub-tokens, which can serve as a building block
within larger distributed training systems.
Head Parallel streamlines token communication. First, each
token is sent exactly once without duplication, so commu-
nication volume is O(1) relative to k. Second, each GPU
sends and receives the same amount of data deterministi-
cally, so there is no need to communicate any metadata
beforehand. This also avoids edge cases such as waiting
for the slowest node or out-of-memory, when too many to-
kens route to a single GPU. We verify these advantages in
Section 4.2.
3.3. IO-Aware Routing
Multi-Head LatentMoE projects each token into Nh sub-
tokens. However, a naive router materializes the routing
Algorithm 2 IO-Aware Routing: Backward Pass
Require: Sub-tokens X ∈RB×T ×Nh×dh, router weights
Wr ∈RNh×dh×Ne, top-k indices Itop ∈NB×T ×Nh×k,
gradient of top-k scores dStop ∈RB×T ×Nh×k, block
size N (tokens).
Ensure: Input gradient dX
∈
RB×T ×Nh×dh, router
weight gradient dWr ∈RNh×dh×Ne.
1: Initialize dWr = 0 in HBM.
2: Divide X, Itop, dStop into ⌈T/N⌉blocks of N tokens
each.
3: for each (b, tblock, h) in parallel do
4:
Load Xblock ∈RN×dh from HBM to SRAM.
5:
Initialize dXaccum = 0 ∈RN×dh on chip.
6:
for i = 1 to k do
7:
Load ei = Itop[:, :, :, i] ∈NN and dsi = dStop[:, :
, :, i] ∈RN from HBM to SRAM.
8:
Load Wr[:, ei] ∈RN×dh from HBM to SRAM
(gather by expert index).
9:
On chip, compute dXaccum += dsi · Wr[:, ei].
10:
Atomic add X⊤
blockdsi to dWr[:, ei] in HBM.
11:
end for
12:
Write dXaccum to dX in HBM.
13: end for
scores in HBM before selecting the top-k experts. With
Nh = 8, the HBM and IO costs are 8 times higher.
We propose IO-aware routing, inspired by FlashAtten-
tion (Dao et al., 2022). The key observation is that a router
only outputs the top-k indices and scores. Therefore, we
can stream the top-k results in SRAM without materializing
in HBM.
The forward pass is given in Algorithm 1. For each block of
experts, we compute scores in SRAM and find the block’s
local top-k results. The local results are merged with an
accumulator to track the running top-k indices and scores,
which are written to HBM by the end.
It is worth mentioning two computational tricks. First, to
support aux-free load balancing (Wang et al., 2024a), we
add bias before top-k and subtract it afterward. As such,
bias influences routing without affecting the final scores.
Second, following Tillet et al. (2019), we pack the score and
index into a 64-bit unsigned integer to achieve arg-top-k in
Triton.
The backward pass is given in Algorithm 2. Given S =
XW, the input gradient is dX = dS · W⊤and the weight
gradient is dW = X⊤·dS. Gradients are computed only for
the k selected experts per token, reducing time complexity
from O(Ne) to O(k), given that k << Ne.
IO-aware routing is efficient and exact. For both algorithms,
the HBM access reduces from O(Ne) to O(k). The forward
4


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Figure 2. Token clustering for expert computation expressed as
block-sparse attention. Q represents input tokens and K trans-
pose represents expert weights. Colors encode expert assignments.
Clustering reorders Q so that tokens assigned to the same expert
become contiguous, yielding a block-diagonal sparsity pattern.
pass has the same number of FLOPs as a standard router.
The backward pass exploits sparsity, reducing its time com-
plexity from O(Ne) to O(k). This makes our router more
efficient than a standard router. We verify these claims in
Section 4.2.
3.4. IO-Aware Expert Computation
Multi-Head LatentMoE projects each token into Nh sub-
tokens. However, Grouped GEMM (NVIDIA, 2022) materi-
alizes the expert activations, making memory grow linearly
with Nh. We need an operator that is dropless (Gale et al.,
2023) like Grouped GEMM, but also IO aware.
We propose IO-aware expert computation. Observing the
well-known duality between FFN and Attention (Vaswani
et al., 2017), we rewrite sparse expert computation into
block sparse attention. This allows us to leverage FlexAt-
tention (Dong et al., 2024), reusing their highly-optimized
IO-aware kernels.
FlexAttention (Dong et al., 2024) extends FlashAttention
with a score modification function and a block sparse mask:
O = softmax(score mod(QK⊤⊙M)) · V
(7)
where score mod(·) is a user-defined function applied
element-wise, and M is a binary block sparse mask. Flex-
Attention also returns log(ℓ), where ℓis the softmax denom-
inators. We leverage ℓto change the activation function.
Expert computation can be written as block sparse atten-
tion. Specifically, a single expert computes σ(XW⊤
in)Wout,
which mirrors the blockwise softmax(QK⊤)V.
To switch from softmax to gelu, consider
score mod(s) = log(gelu(s) + 1),
where offsetting by 1 makes the logarithm well-defined. 2
2For activation functions that are unbounded below, one can
The softmax output then becomes
exp(log(gelu(s) + 1))
P
j exp(log(gelu(sj) + 1)) = gelu(s) + 1
ℓ
.
(8)
We note that ℓis numerically stable.
FlexAttention returns both log(ℓ) and
O′ = (gelu(XK⊤) + 1)V/ℓ.
Since ℓis numerically stable, we can materialize it. Multi-
plying O′ by ℓyields
(gelu(XK⊤) + 1)V = gelu(XK⊤)V + 1V.
Finally, subtracting the bias term 1V = P
j Vj for each
token’s assigned expert recovers the exact result. This can
be implemented with torch.gather.
Similar to Grouped GEMM, our expert computation re-
quires tokens assigned to the same expert placed contigu-
ously.
This is usually through token clustering (See Figure 2).
The implemenation is pure-PyTorch (Paszke et al., 2019).
Practitioners with a limited engineering budget can avoid
writing custom CUDA kernels or expanding support for new
hardwares.
IO-aware expert computation is dropless (Gale et al., 2023)
and numerically correct like Grouped GEMM, but also IO-
aware. We verify these claims in Section 4.2.
4. Experiments
4.1. Language Modeling
Multi-Head LatentMoE aims to match or exceed traditional
MoE in model quality while providing faster training. In this
section, we evaluate both model performance and training
efficiency under controlled computation budgets.
We conduct language modeling experiments on a 10-billion-
token subset of FineWeb-EDU (Penedo et al., 2024). For
detailed experimental setup including model hyperparame-
ters, see Appendix A and Appendix B.
Table 1 shows the results.
Multi-Head LatentMoE HP
achieves comparable model performance to baselines while
training significantly faster: 1.11x faster at 2B total param-
eters (31.68 vs 35.36 hours) and 1.61x faster at 4B total
parameters (34.41 vs 55.34 hours). This speedup comes
from two sources. First, Head Parallel achieves O(1) com-
munication volume compared to O(k) for Expert Parallel;
with k=4, our communication volume is 25% of Expert Par-
allel. Second, our FlexAttention-based expert computation
compute the offset online, but this is beyond our scope.
5


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Table 1. Model performance comparison across feedforward designs. Parameters are reported as activated-total. FineWebEDU reports
validation perplexity (ppl.). HellaSwag (Hella.), PiQA, LAMBADA (LMB.), ARC-Easy (Arc E.), and ARC-Challenge (Arc C.) report
zero-shot accuracy (acc.). G doubles the granularity. Bold indicates best and underline indicates second best. LatentMoE is replicated
based on Elango et al. (2026). All accuracies are normalized based on length, except for LAMBADA.
Parameters
Model
Training Cost
FineWebEDU
Hella.
PiQA
LMB.
Arc E.
Arc C.
Avg.
active - all
hours (rel.) ↓
ppl. ↓
acc. ↑
acc. ↑
acc. ↑
acc. ↑
acc. ↑
acc. ↑
0.2B-0.2B
MLP
14.53 (1.00×)
20.16
31.8
62.1
26.0
47.6
25.3
38.53
0.2B-2.2B
MoE EP
35.36 (1.00×)
15.56
39.7
67.1
32.4
53.1
27.6
43.95
LatentMoE EP
32.71 (0.93×)
15.31
40.6
67.7
33.1
53.5
29.3
44.83
MH LatentMoE HP
31.68 (0.90×)
15.61
39.5
67.0
31.2
53.7
28.2
43.93
MH LatentMoE HP G
43.92 (1.24×)
15.52
39.6
67.2
32.0
55.2
29.7
44.76
0.2B-4.2B
MoE EP
55.34 (1.00×)
15.01
41.2
67.6
33.3
54.6
29.8
45.30
LatentMoE EP
53.76 (0.97×)
14.84
41.8
68.1
32.7
52.5
27.6
44.55
MH LatentMoE HP
34.41 (0.62×)
15.02
41.2
67.8
33.1
54.3
29.4
45.17
MH LatentMoE HP G
50.04 (0.90×)
14.82
41.3
67.6
34.1
54.3
29.9
45.43
is faster under high sparsity, likely due to different work
partitioning and FlexAttention’s well-optimized kernels.
We also replicate the finding from Elango et al. (2026) that
LatentMoE EP trains faster than MoE EP. However, our
method is faster while achieving better or comparable per-
formance. We attribute this to the fact that our method is
fundamentally EP-free, requiring only HP.
Following Elango et al. (2026), we reinvest the speedup
advantage into doubled granularity (MH LatentMoE HP G).
At the 4B scale, MH LatentMoE HP G achieves the best
average accuracy (45.43%) while still training 1.1x faster
than MoE EP (50.04 vs 55.34 hours).
These results demonstrate that Multi-Head LatentMoE pro-
vides significant training speedups over MoE. When rein-
vesting the saved compute into increased granularity, Multi-
Head LatentMoE achieves both faster training and better
model quality.
4.2. Per-Component Analysis
The training speedup of Multi-Head LatentMoE with Head
Parallel comes from three factors: reduced all-to-all commu-
nication, efficient routing, and efficient expert computation.
In this section, we isolate each component to verify its con-
tribution.
We first compare Expert Parallel (EP) and Head Parallel
(HP) under varying load imbalance. We measure all-to-
all latency (including time spent waiting for the slowest
GPU) and peak VRAM usage on 4 GPUs. We control
load imbalance by drawing per-token expert assignments
from a Zipf distribution with skew in {0.0, 1.0, 2.0}, where
skew=0.0 corresponds to uniform distribution and skew=2.0
assigns 99.8% of tokens to a single GPU. Figure 3 shows
that EP latency and peak VRAM both increase with larger
k and larger skew. This is expected because EP duplicates
0
1
2
Load Imbalance (skew)
0
5
10
15
20
Latency (ms)
Running speed
0
1
2
Load Imbalance (skew)
0
2
4
6
8
10
VRAM Usage (GiB)
Memory footprint
EP (k=2)
EP (k=4)
EP (k=8)
EP (k=16)
HP (for any k)
Figure 3. Comparison of Expert Parallel (EP) and Head Parallel
(HP) under varying load imbalance. Left: all-to-all latency, includ-
ing time spent waiting for the slowest GPU. Right: peak VRAM
usage across all GPUs. We simulate load imbalance on 4 GPUs
using a Zipf distribution with varying skew: skew=0.0 corresponds
to uniform distribution (25% per GPU), skew=1.0 assigns 80.8%
of tokens to GPU 0, and skew=2.0 assigns 99.8% to GPU 0. EP
latency and memory grow with both k and skew, while HP remains
constant for any k.
and redistributes tokens after routing, and must wait for the
most-loaded device. In contrast, HP remains constant across
all values of k and skew because each token is sent exactly
once before routing, making traffic per GPU fixed.
We next compare naive routing (using torch.matmul)
with our IO-aware routing.
We measure forward and
backward latency and combined memory footprint, using
B = 40, T = 2048, Nh = 8, dh = 128, and varying the
number of experts. Figure 4 shows that our IO-aware routing
maintains constant memory footprint as the number of ex-
perts grows, while the naive implementation scales linearly.
The backward pass of our method remains nearly constant
because gradients are computed only for the selected top-k
6


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
29
210
211
212
Number of Experts
0
20
40
60
80
100
Latency (ms)
Running speed
Naive (fwd)
Naive (bwd)
Ours (fwd)
Ours (bwd)
29
210
211
212
Number of Experts
0
10
20
30
VRAM Usage (GiB)
Memory footprint
Naive (fwd+bwd)
Ours (fwd+bwd)
Figure 4. Comparison of naive routing with torch.matmul and
our IO-aware routing. Left: latency for forward and backward
passes. Right: memory footprint for forward and backward com-
bined. Our IO-aware routing maintains constant memory footprint
regardless of the number of experts, and its backward pass remains
nearly constant due to sparse gradient computation. Experiments
use B = 40, T = 2048, Nh = 8, dh = 128.
24
26
28
Number of Experts
100
101
102
Latency (ms)
Forward
24
26
28
Number of Experts
100
101
102
103
Latency (ms)
Backward
Naive (de=128)
Ours (de=128)
Naive (de=256)
Ours (de=256)
Naive (de=512)
Ours (de=512)
Figure 5. Comparison of naive expert computation (using grouped
GEMM) and our IO-aware expert computation under Multi-Head
LatentMoE, where multiple heads increase the number of sub-
tokens. Left: forward pass latency. Right: backward pass latency
(log scale). de denotes the number of hidden neurons per expert.
Naive grouped GEMM scales poorly with the number of experts,
while our method remains efficient across all configurations. Ex-
periments use B = 4, T = 512, Nh = 8, dh = 128, k = 4.
experts.
Finally, we compare naive expert computation (using
grouped GEMM) with our FlexAttention-based approach
under Multi-Head LatentMoE, where multiple heads in-
crease the number of sub-tokens. We measure forward
and backward latency using B = 4, T = 512, Nh = 8,
dh = 128, k = 4, and varying expert size de and expert
count. Figure 5 shows that naive grouped GEMM scales
poorly with the number of experts, while our method re-
mains efficient across all configurations. The gains are
particularly large in the backward pass.
These results validate that Head Parallel removes the sensi-
tivity to load imbalance inherent in Expert Parallel, and that
our IO-aware routing and expert computation make Multi-
Head LatentMoE practical by avoiding excessive HBM us-
age while remaining performant.
4.3. Ablation: Head Configuration
In Multi-Head LatentMoE, each token is projected into Nh
sub-tokens of dh dimensions. We investigate how Nh and
dh affect model performance and training efficiency.
We measure validation loss and training cost. Following
common practice (Vaswani et al., 2017), we keep Nh · dh =
d and evaluate (Nh, dh) ∈{(16, 64), (8, 128), (4, 256)}.
Table 2 shows the results.
The 8 × 128 configuration
achieves the lowest training cost while maintaining compet-
itive validation loss. The 4 × 256 configuration achieves
the lowest validation loss of 3.41, but incurs higher training
cost and high SRAM pressure. The 16 × 64 configuration
has the lowest SRAM pressure but the highest training cost
and validation loss.
The experiments show Multi-Head LatentMoE is robust to
head configuration choices. Therefore, we select the 8×128
configuration as it provides the best training efficiency and
offers a good balance for model performance and ease of
engineering.
Table 2. How different head configurations affect Multi-Head La-
tentMoE. Nh denotes the number of heads and dh denotes the head
dimension. Training Cost is relative to the 8 × 128 configuration.
SRAM Pressure refers to the on-chip memory pressure during
routing and expert computation under typical work partitioning.
Bold indicates best while underline indicates second best.
Nh × dh
Val Loss
Training Cost
SRAM Pressure
(↓)
(Relative ↓)
(↓)
16 × 64
3.56
1.34×
Low
8 × 128
3.48
1.00×
Medium
4 × 256
3.41
1.07×
High
4.4. Ablation: Separate Routing Tokens
In Multi-Head LatentMoE, the input token xt is projected
into sub-tokens xt,i that are used for both routing and expert
computation. In this section, we assess an alternative design
choice: projecting a separate set of sub-tokens rt,i dedicated
to routing decisions.
Specifically, a single token xt becomes
[xt,1, . . . , xt,Nh, rt,1, . . . , rt,Nh] = split(Winxt)
(9)
where Win ∈R2d×d is a learned linear projection, and
split(·) divides the projected token into 2Nh sub-tokens of
dimension dh. Each head i then uses rt,i for computing
routing scores and xt,i for expert computation. Note that
this design doubles the all-to-all volume under Head Parallel,
as both xt,i and rt,i must be distributed across devices.
7


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
0
2000
Training Step
5
10
Validation Loss
Validation loss curves
Without Routing Token
With Routing Token
0
2000
Training Step
0.05
0.00
0.05
Loss Difference
Without minus with
Figure 6. Comparing Multi-Head LatentMoE with and without
separate sub-tokens for routing decisions. Left: validation loss
curves over training steps. Right: loss difference (without minus
with), where negative values favor the variant with separate routing
tokens.
To investigate whether this design choice improves model
quality, we pretrain Multi-Head LatentMoE with and with-
out separate routing tokens and measure validation perplex-
ity throughout training.
Figure 6 shows the results. Separate routing tokens provide
a small improvement in early training that diminishes as
training progresses. Both configurations achieve nearly
identical perplexity. Given that this design also doubles the
all-to-all volume, we do not use separate routing tokens.
5. Related Work
Latent Representations in MoE
Projecting a token into
a lower-dimensional latent space is a common design choice
in MoEs for both routing and expert computation.
X-MoE (Chi et al., 2022) addressed representation collapse
by routing in a low-dimensional space, but experts still op-
erated on high-dimensional inputs. Multi-Head MoE (MH-
MoE) (Huang et al., 2024) linearly projected a single token
into multiple smaller latent spaces, creating sub-tokens. Dif-
ferent from our work, these sub-tokens shared the same
router and set of experts, so MH-MoE could not benefit
from Head Parallel. MH-MoE did not propose IO-aware
routing or expert computation. All activations are materi-
alized in HBM. Recently, LatentMoE (Elango et al., 2026)
reduced all-to-all communication volume by performing ex-
pert computation in a low-dimensional latent space, but still
relied on EP and inherited its issues with load imbalance
and non-deterministic communication.
Different from both Multi-Head MoE (MH-MoE) (Huang
et al., 2024) and LatentMoE (Elango et al., 2026), Multi-
Head LatentMoE assigns each head an independent router
and set of experts, forming fully independent MoE modules.
Only this design can benefit from Head Parallel.
Distributed Training
Distributed training in MoE parti-
tions expert weights across devices to scale model capacity.
Its communication overhead is the primary bottleneck in
MoE training (Fedus et al., 2022).
GShard (Lepikhin et al., 2020) established Expert Paral-
lel (EP), but its all-to-all operation had O(k) communi-
cation volume, was affected by load imbalance, and was
non-deterministic. DeepEP (Zhao et al., 2025) improved EP
through kernel fusion, communication-computation over-
lap, and avoided sending duplicate tokens to the same node.
However, without fundamentally changing the parallelism,
DeepEP only reduced constant factors without changing
the O(k) complexity. Tensor Parallel (Shoeybi et al., 2019)
provided balanced and deterministic communication, but
had large communication volume. In attention research,
Ulysses (Jacobs et al., 2023), also known as Head Parallel,
distributed attention heads across devices to enable long
sequence modeling. Our work is fundamentally different.
Our Head Parallel distributes expert weights across GPUs
like EP, but moves all-to-all before routing. It is the first
MoE parallelism achieving O(1) communication volume
along with balanced and deterministic communication.
6. Conclusion
In this work, we addressed the fundamental limitations of EP
with Multi-Head LatentMoE and HP. To make Multi-Head
LatentMoE practical, we developed new routing operators
using online top-k and FlexAttention-based expert compu-
tation, achieving FlashAttention-like IO awareness while
being exact.
Head Parallel achieves O(1) communication volume regard-
less of the number of activated experts, perfectly balanced
traffic across GPUs, and deterministic communication pat-
terns. Being static means HP does not require communi-
cating metadata between GPUs. Experiments show that
Multi-Head LatentMoE with HP trains up to 1.61x faster
than traditional MoE. The communication volume reduced
to 25% of EP at k = 4. With doubled granularity, it achieves
higher overall performance while still being 1.11x faster.
Our work accelerates ultra-sparse MoE training, making
multi-billion-scale foundational research more accessible.
Scope, limitations, and future directions. Our approach is
most effective in the ultra-sparse regimes with large expert
counts and small expert sizes, which characterize modern
MoE architectures (Yang et al., 2025). We note that Head
Parallel is bounded by the number of heads. While paral-
lelism on 8 to 16 GPUs are sufficient for most academic
settings, using HP for inter-node traffic and EP for intra-
node traffic is a possible direction for large scale training.
Another future direction is further engineering optimizations
such as communication-computation overlap.
8


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
Acknowledgements
The authors acknowledge Research Computing at Arizona
State University (Jennewein et al., 2023) for providing com-
puting and storage resources that have contributed to the
research results reported within this paper.
This work used Bridges-2 at Pittsburgh Supercomput-
ing Center (Brown et al., 2021) through allocation
CIS250914 from the Advanced Cyberinfrastructure Coordi-
nation Ecosystem: Services & Support (ACCESS) program,
which is supported by National Science Foundation grants
#2138259, #2138286, #2138307, #2137603, and #2138296.
This material is based upon work supported by the National
Science Foundation Graduate Research Fellowship Program
under Grant No. 2233001. Any opinions, findings, and
conclusions or recommendations expressed in this material
are those of the author(s) and do not necessarily reflect the
views of the National Science Foundation.
We thank Keane te Velde for assistance with code develop-
ment. Full acknowledgements will appear in a later version
of this manuscript.
References
Abnar, S., Shah, H., Busbridge, D., Ali, A. M. E., Susskind,
J., and Thilak, V. Parameters vs flops: Scaling laws for
optimal sparsity for mixture-of-experts language mod-
els, 2025. URL https://arxiv.org/abs/2501.
12370.
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.
Besiroglu, T., Bergerson, S. A., Michael, A., Heim, L., Luo,
X., and Thompson, N. The compute divide in machine
learning: A threat to academic contribution and scrutiny?
arXiv preprint arXiv:2401.02452, 2024.
Brown, S. T., Buitrago, P., Hanna, E., Sanielevici, S.,
Scibek, R., and Nystrom, N. A.
Bridges-2: A plat-
form for rapidly-evolving and data intensive research.
In Practice and Experience in Advanced Research Com-
puting 2021: Evolution Across All Dimensions, PEARC
’21, New York, NY, USA, 2021. Association for Com-
puting Machinery.
ISBN 9781450382922.
doi: 10.
1145/3437359.3465593. URL https://doi.org/
10.1145/3437359.3465593.
Chi, Z., Dong, L., Huang, S., Dai, D., Ma, S., Patra, B.,
Singhal, S., Bajaj, P., Song, X., Mao, X.-L., et al. On
the representation collapse of sparse mixture of experts.
Advances in Neural Information Processing Systems, 35:
34600–34613, 2022.
Dao, T.
Flashattention-2:
Faster attention with bet-
ter parallelism and work partitioning. arXiv preprint
arXiv:2307.08691, 2023.
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and R´e, C. Flashat-
tention: Fast and memory-efficient exact attention with io-
awareness, 2022. URL https://arxiv.org/abs/
2205.14135.
Dong, J., Feng, B., Guessous, D., Liang, Y., and He,
H.
Flex attention: A programming model for gener-
ating optimized attention kernels, 2024. URL https:
//arxiv.org/abs/2412.05496.
Elango, V., Bhatia, N., Waleffe, R., Shafipour, R., Asida,
T., Khattar, A., Assaf, N., Golub, M., Guman, J., Mi-
tra, T., Zhao, R., Borkar, R., Zilberstein, R., Patwary,
M., Shoeybi, M., and Rouhani, B.
Latentmoe: To-
ward optimal accuracy per flop and parameter in mixture
of experts, 2026. URL https://arxiv.org/abs/
2601.18089.
Fedus, W., Zoph, B., and Shazeer, N. Switch transformers:
Scaling to trillion parameter models with simple and ef-
ficient sparsity. Journal of Machine Learning Research,
23(120):1–39, 2022.
Gale, T., Narayanan, D., Young, C., and Zaharia, M.
Megablocks: Efficient sparse training with mixture-of-
experts. Proceedings of Machine Learning and Systems,
5:288–304, 2023.
Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,
A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
https://zenodo.org/records/12608602.
Huang, S., Wu, X., Ma, S., and Wei, F. Mh-moe: Multi-head
mixture-of-experts. arXiv preprint arXiv:2411.16205,
2024.
Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E.
Adaptive mixtures of local experts. Neural computation,
3(1):79–87, 1991.
9


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song,
S. L., Rajbhandari, S., and He, Y. Deepspeed ulysses:
System optimizations for enabling training of extreme
long sequence transformer models.
arXiv preprint
arXiv:2309.14509, 2023.
Jennewein, D. M., Lee, J., Kurtz, C., Dizon, W., Shaef-
fer, I., Chapman, A., Chiquete, A., Burks, J., Carlson,
A., Mason, N., Kobwala, A., Jagadeesan, T., Barghav,
P., Battelle, T., Belshe, R., McCaffrey, D., Brazil, M.,
Inumella, C., Kuznia, K., Buzinski, J., Dudley, S., Shah,
D., Speyer, G., and Yalim, J. The Sol Supercomputer
at Arizona State University.
In Practice and Experi-
ence in Advanced Research Computing, PEARC ’23, pp.
296–301, New York, NY, USA, Jul 2023. Association
for Computing Machinery. ISBN 9781450399852. doi:
10.1145/3569951.3597573.
Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary,
B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna,
E. B., Bressand, F., et al. Mixtral of experts. arXiv
preprint arXiv:2401.04088, 2024.
Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y.,
Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling
giant models with conditional computation and automatic
sharding. arXiv preprint arXiv:2006.16668, 2020.
Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao,
C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437, 2024.
NVIDIA. Transformer engine. https://github.com/
NVIDIA/TransformerEngine, 2022.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., et al. Pytorch: An imperative style, high-performance
deep learning library. Advances in neural information
processing systems, 32, 2019.
Penedo, G., Kydl´ıˇcek, H., allal, L. B., Lozhkov, A., Mitchell,
M., Raffel, C., Werra, L. V., and Wolf, T. The fineweb
datasets: Decanting the web for the finest text data
at scale, 2024.
URL https://arxiv.org/abs/
2406.17557.
Qiu, Z., Huang, Z., Zheng, B., Wen, K., Wang, Z., Men,
R., Titov, I., Liu, D., Zhou, J., and Lin, J. Demons in the
detail: On implementing load balancing loss for training
specialized mixture-of-expert models. arXiv preprint
arXiv:2501.11873, 2025.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,
Sutskever, I., et al. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
Q., Hinton, G., and Dean, J. Outrageously large neural
networks: The sparsely-gated mixture-of-experts layer.
arXiv preprint arXiv:1701.06538, 2017.
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper,
J., and Catanzaro, B.
Megatron-lm: Training multi-
billion parameter language models using model paral-
lelism. arXiv preprint arXiv:1909.08053, 2019.
Tillet, P., Kung, H.-T., and Cox, D. Triton: an intermediate
language and compiler for tiled neural network computa-
tions. In Proceedings of the 3rd ACM SIGPLAN Interna-
tional Workshop on Machine Learning and Programming
Languages, pp. 10–19, 2019.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At-
tention is all you need. Advances in neural information
processing systems, 30, 2017.
Wang, L., Gao, H., Zhao, C., Sun, X., and Dai, D. Auxiliary-
loss-free load balancing strategy for mixture-of-experts.
arXiv preprint arXiv:2408.15664, 2024a.
Wang, Z., Zhu, J., and Chen, J. Remoe: Fully differen-
tiable mixture-of-experts with relu routing. arXiv preprint
arXiv:2412.14711, 2024b.
Wu, X., Huang, S., Wang, W., and Wei, F. Multi-head
mixture-of-experts, 2024.
URL https://arxiv.
org/abs/2404.15045.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report. arXiv preprint arXiv:2505.09388, 2025.
Zhao, C., Zhou, S., Zhang, L., Deng, C., Xu, Z., Liu, Y.,
Yu, K., Li, J., and Zhao, L. Deepep: an efficient expert-
parallel communication library.
https://github.
com/deepseek-ai/DeepEP, 2025.
10


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
A. Implementation Details
Environments. All experiments are conducted on NVIDIA H100 GPUs with 80GB, interconnected via NVLink.
Baselines. We compare Multi-Head LatentMoE with Head Parallel against MLP and MoE with Expert Parallel. MoE
follows the definition in Section 2.2. For the MoE implementation, Grouped GEMM uses NVIDIA Transformer Engine
(TE) (NVIDIA, 2022), token clustering uses TE’s moe permute(.) and moe sort chunks by index(.), and
Expert Parallel uses PyTorch’s all to all(.).
Architecture. All models are decoder-only Transformers with 12 layers, embedding size 1024, and context window 2048.
We use rotary positional embeddings and RMSNorm without bias terms. The attention module has 8 heads with head
dimension 128. Following (Liu et al., 2024), we replace the first two MoE layers with MLPs to mitigate router imbalance in
early layers. All routers compute in FP32 for numerical stability, and we use aux-free (Wang et al., 2024a) and global (Qiu
et al., 2025) load balancing.
Initialization. Following standard practices (Radford et al., 2019), we remove all biases, and all weights are randomly
initialized from N(0, 0.02). Output projection weights are further scaled by 1/
√
2 · L where L is the number of layers, also
following Radford et al. (2019).
Training. We use AdamW with β1 = 0.9, β2 = 0.95, and weight decay 0.1. Learning rate is 5.0 × 10−4 (following (Chi
et al., 2022; Wang et al., 2024b; Wu et al., 2024)) with trapezoidal schedule of 8000 warmup steps and 2000 decay steps.
The global batch size is 0.66 million tokens.
Evaluation. At training time, we estimate validation perplexity on a held-out set of 99.6 million tokens from the same
distribution. After training, we use lm-eval-harness (Gao et al., 2024) to perform zero-shot evaluation on five downstream
tasks: HellaSwag, PiQA, LAMBADA, ARC-Easy, and ARC-Challenge.
B. Model Hyperparameters
We report hyperparameters for each experiment setting in Tables 3, 4, and 5. Unless otherwise specified, all models
are decoder-only Transformers with RMSNorm (no bias), rotary positional embeddings (RoPE), and a context length of
T = 2048.
Table 3. Hyperparameters for the 0.2B–0.2B MLP baseline.
Hyperparameter
MLP
Parameters (active–total)
0.2B–0.2B
Transformer layers L
12
Embedding size d
1024
Attention heads
8
Head dimension
128
Context length T
2048
Feedforward
Dense MLP
MoE experts Ne
–
Active experts k
–
Expert hidden size
–
Multi-head count Nh
–
Latent head dim dh
–
Training hyperparameters (shared).
All experiments use AdamW with (β1, β2) = (0.9, 0.95), weight decay 0.1,
learning rate 5.0 × 10−4 with a trapezoidal schedule (8000 warmup steps, 2000 decay steps), and a global batch size of
0.66M tokens. All runs are conducted on NVIDIA H100 80GB GPUs interconnected via NVLink.
11


---

Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism
Table 4. Hyperparameters for the 0.2B–2.2B category.
Hyperparameter
MoE (EP)
LatentMoE (EP)
MH LatentMoE (HP)
MH LatentMoE (HP, G)
Parameters (active–total)
0.2B–2.2B
0.2B–2.2B
0.2B–2.2B
0.2B–2.2B
Transformer layers L
12
12
12
12
Embedding size d
1024
1024
1024
1024
Attention heads
8
8
8
8
Head dimension
128
128
128
128
Context length T
2048
2048
2048
2048
MoE experts Ne
384
384
384
768
Active experts k
4
4
4
8
Expert hidden size
256
256
256
128
Multi-head count Nh
–
–
8
8
Latent head dim dh
–
–
128
128
Table 5. Hyperparameters for the 0.2B–4.2B category.
Hyperparameter
MoE (EP)
LatentMoE (EP)
MH LatentMoE (HP)
MH LatentMoE (HP, G)
Parameters (active–total)
0.2B–4.2B
0.2B–4.2B
0.2B–4.2B
0.2B–4.2B
Transformer layers L
12
12
12
12
Embedding size d
1024
1024
1024
1024
Attention heads
8
8
8
8
Head dimension
128
128
128
128
Context length T
2048
2048
2048
2048
MoE experts Ne
768
768
768
1536
Active experts k
4
4
4
8
Expert hidden size
256
256
256
128
Multi-head count Nh
–
–
8
8
Latent head dim dh
–
–
128
128
12
