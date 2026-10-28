---
title: 'Piper: Efficient Large-Scale MoE Training via'
id: piper-efficient-large-scale-moe-training-via
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-topology-cost-at-1m-tokens
created: '2026-10-27T13:20:48.554734Z'
updated: '2026-10-27T19:00:31.688126Z'
source: https://arxiv.org/abs/2605.05049
source_domain: arxiv.org
fetched_at: '2026-10-27T13:20:48.554090Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: Suggested by scalable-training-of-mixture-of-experts-models-with
raw_file: raw/piper-efficient-large-scale-moe-training-via.pdf
doi: arXiv:2605.05049
utility_score: 13.0
citation_count: 0
venue: arXiv.org
---

*Suggested by [[scalable-training-of-mixture-of-experts-models-with]]*

Piper: Efficient Large-Scale MoE Training via
Resource Modeling and Pipelined Hybrid Parallelism
Sajal Dash
Oak Ridge National Laboratory
Oak Ridge, USA
Email: dashs@ornl.gov
Feiyi Wang
Oak Ridge National Laboratory
Oak Ridge, USA
Email: fwang2@ornl.gov
Abstract—Frontier models increasingly adopt Mixture-of-
Experts (MoE) architectures to achieve large-model performance
at reduced cost. However, training MoE models on HPC
platforms is hindered by large memory footprints, frequent large-
scale communication across heterogeneous networks, and severe
workload imbalance. To characterize these challenges, we develop
a mathematical model that quantifies memory, compute, and com-
munication requirements for MoE configurations under various
parallelization schemes, verified through micro-benchmarking,
code instrumentation, and hardware profiling. Our analysis
identifies performance bottlenecks: all-to-all latency at scale from
expert parallelism, insufficient compute-communication overlap,
low GPU utilization from imbalanced skinny GEMMs, and the
absence of platform-aware hybrid parallelization strategies. To
address these, we introduce Piper, a framework that leverages
resource modeling to identify efficient training strategies for MoE
models on target HPC platforms, applying pipeline parallelism
with optimized schedules. Piper achieves 2–3.5× higher MFU
than state-of-the-art frameworks such as X-MoE, and a novel
all-to-all algorithm delivers 1.2X–9X bandwidth over vendor
implementation.
I. INTRODUCTION
Transformer based Large Language Models (LLMs) have
demonstrated consistent performance gains with increasing
model size [1], [2]. However, training these models at scale
is resource-intensive: a mixed-precision training run requires
approximately 20 bytes of memory per parameter, roughly
6 floating-point operations per parameter per token, and
substantial inter-device communication [3], [4]. The Mixture-
of-Experts (MoE) mechanism, in which only a fraction of the
model parameters get sparsely activated for each token, has
emerged as a promising approach to achieving dense-models’
performance at a reduced computational cost with savings
proportional to the sparsity factor [5], [6]. State-of-the-art
models such as Mixtral [7], DeepSeek [8], Qwen [9], and
Kimi [10] have adopted MoE to deliver superior performance
at substantially reduced training cost. Despite these advantages,
MoE training presents its own unique challenges. Relative to
a parameter-matched dense model, MoE training introduces
computational load imbalance across devices, an elevated
memory footprint from storing both model parameters and
transient activation tensors, and high inter-device communi-
cation volume under expert parallelism — the most widely
used MoE distribution method [11], [12]. Emerging MoE
architectures with fine-grained experts such as DeepSeek-
MoE [13], exacerbate these challenges further: they produce
many tall-and-skinny GEMMs with poor hardware utilization,
inflate activation memory, and require all-to-all collectives
involving a large number of participating processes.
These difficulties compound when training on shared HPC
platforms which are primarily designed for modeling and
simulation workloads and feature non-uniform communication
fabrics across GPU nodes. Prior work such as X-MoE [14]
has demonstrated that communication overhead becomes a
dominant performance bottleneck in exactly this setting, yet
no holistic framework exists for systematically characterizing,
quantifying, and mitigating these inefficiencies across both
model and platform architecture.
In this work, we develop mathematical models to quantify
memory, compute, and communication requirements for diverse
MoE architectures under different distributed training strategies
and their combinations, parameterized by empirically measured
platform characteristics — including memory capacity, GPU
throughput, and network bandwidth. We validate these models
through comprehensive experimental training runs using frame-
works including DeepSpeed-TED [15], DeepSpeed-MoE [12],
and X-MoE [14] on the Frontier supercomputer [16].
Using our resource modeling framework alongside exper-
imental profiling, we identify prominent sources of missed
performance: (i) all-to-all latency at scale caused by expert
parallelism over non-uniform interconnects; (ii) load imbal-
ance from skewed expert assignment, resulting in low GPU
utilization during early stages, and, in case of no external
mechanisms, throughout the majority of training; and (iii) a
lack of system-aware hybrid parallelization strategies. We then
develop targeted solutions for each.
A. Contributions
1) Analytical and Empirical Resource Modeling: We
develop a mathematical model for estimating memory,
compute, and communication utilization during MoE
training across a range of architectures and paralleliza-
tion configurations, and empirically validate it through
micro-benchmarking, code instrumentation, and hardware
profiling.
2) Piper: Pipeline Parallelism for Localizing Commu-
nication: We introduce Piper, a framework that applies
pipeline parallelism to intra-layer parallelization strategies
arXiv:2605.05049v1  [cs.DC]  6 May 2026


---

in order to localize and overlap expensive collective
communications, and uses the resource model to automat-
ically identify efficient training configurations for target
HPC platforms. Piper achieves 2–3.5X higher Model
FLOP Utilization (MFU) compared to state-of-the-art MoE
training framework, X-MoE [14].
3) Topology-Aware All-to-All Algorithm: We design a
Dragonfly-topology-aware hierarchical all-to-all algorithm
that exploits the dependency structure of asynchronous
point-to-point communications, groups traffic over slower
inter-node and inter-cabinet links, and saturates NICs
uniformly, achieving 1.5X–4X the bandwidth of vendor-
provided implementations.
4) Expert Migration for Load Balancing: We develop an
expert migration technique in which GPUs hosting the
same layer periodically exchange experts to re-balance
load, incurring an amortized overhead of less than 5% of
total training time.
5) Trillion-Scale MoE Training: Using Piper and our
resource modeling tool, we devise and validate train-
ing strategies for several state-of-the-art MoE models
at 20–50% MFU, and demonstrate training of trillion-
parameter MoE models at 20% MFU on the Frontier
supercomputer. For reference, X-MoE reported training
of a 545B parameter model at 5.23% MFU.
II. BACKGROUND AND RELATED WORK
We survey prior work across five dimensions relevant
to this paper: MoE model architectures, distributed training
frameworks and hybrid parallelism, collective communication
algorithms, load balancing techniques, and pipeline parallelism.
We also review HPC platform topology characteristics that
motivate our system-level design choices. Table I summarizes
the architectural parameters of representative state-of-the-art
MoE models.
A. MoE Architectures
The MoE mechanism replaces the feed-forward network
(FFN) sublayer in a Transformer block [2] with a collection
of expert FFNs, each receiving only the tokens routed to
it by a learned gating function [5]. There are two primary
architectural streams. Coarse-grained MoE. Early large-scale
MoE models such as GShard [11], Switch Transformer [6],
and the Mixtral family [7] employ a small number (typically
8–64) of large experts whose FFN dimension matches a
comparably sized dense model, with top-1 or top-2 routing per
token. Because individual experts frequently exceed single-GPU
memory capacity, they require tensor parallelism or sharded
data parallelism, substantially complicating the communication
pattern.
Fine-grained MoE. Pioneered by DeepSeek-MoE [13] and
followed by Qwen3 [17], Kimi K2 [18], and others, this
approach decomposes each expert into many smaller experts;
reducing FFN dimension by a factor m while increasing expert
count proportionally and selects a larger top-K (K ∈[6, 8, 16]).
16]) per token. While individual experts now fit within a single
GPU without tensor splitting, fine-grained designs generate
tall-and-skinny GEMMs with poor hardware utilization, inflate
activation memory by factor mm m, and require all-to-all
collectives spanning many GPUs.
B. Distributed Training Frameworks and Hybrid Parallelism
Training large MoE models requires combining data par-
allelism (DP) [4], tensor parallelism (TP) [19], pipeline
parallelism (PP) [20], [21], and expert parallelism (EP) [11],
where different experts are placed on different devices and
tokens are routed via all-to-all collectives.
DeepSpeed-MoE [12] combines expert parallelism with
tensor parallelism and ZeRO memory sharding [4], primarily
targeting coarse-grained architectures. DeepSpeed-TED [15]
extends this by jointly optimizing across Tensor, Expert, and
Data parallelism axes.
X-MoE [14] targets fine-grained expert architectures, iden-
tifying activation memory and all-to-all scope as primary
bottlenecks. It introduces zero-padding for load balancing,
redundancy-based communication bypassing, and sequence-
sharded parallelism, successfully outperforming DeepSpeed-
MoE and Tutel at hundreds-of-billions scale. However, for
500B+ models, X-MoE achieves only 5% MFU.
Tutel [22] provides efficient MoE dispatch and combine
kernels with dynamic top-K routing and adaptive parallelism
switching, but focuses on the dispatch kernel rather than end-to-
end training strategy selection and does not cover attention-layer
parallelization.
A general limitation across these frameworks is the absence
of platform-aware hybrid parallelism planning that jointly
accounts for memory, compute, and communication constraints,
a gap Piper directly addresses (Section III).
C. Pipeline Parallelism
Pipeline parallelism partitions model layers across devices
using micro-batching [20]. The 1F1B schedule [21] reduces
pipeline bubble fraction and peak activation memory over
GPipe [23]; interleaved variants [3] and ZB-H1/H2 sched-
ules [24] reduce bubble overhead further. These techniques
target dense models where communication occurs between lay-
ers rather than within them. Piper extends pipeline parallelism
to the intra-layer axis introduced by expert parallelism, enabling
computation-communication overlap within MoE layers.
D. Load Balancing
Uneven token distribution across experts reduces effective
GPU throughput. Common mitigations include auxiliary load-
balancing losses [6], [11], token dropping [6], expert-choice
routing [25], and the auxiliary-loss-free bias-adjustment strategy
of DeepSeekV3 [8]. All operate at the routing level and cannot
correct device-level imbalance from oblivious expert placement.
Our expert migration approach (Section VI) complements these
methods by physically redistributing experts based on observed
load.


---

TABLE I: State-of-the-Art Mixture-of-Experts (MoE) Model Configurations
Model
Total Params
Active Params
Total Experts
Active / Token
Layers
Hidden Size
FFN Dim (per expert)
Context
Train Tokens
DeepSeek-V2
236B
21B
162a
6R + 2S
60
5,120
1,536
128K
8.1T
DeepSeek-V3
671B
37B
257b
8R + 1S
61
7,168
2,048
128K
14.8T
DeepSeek-V3.2
671B
37B
257b
8R + 1S
61
7,168
2,048
128K
—
Mixtral 8×7B
∼47Bc
∼13B
8
Top-2
32
4,096
14,336
32K
—
Mixtral 8×22B
141B
39B
8
Top-2
56
6,144
16,384
64K
—
Qwen3-30B-A3B
30B
3B
128d
Top-8
48
2,048
768
128K
∼36T
Qwen3-235B-A22B
235B
22B
128d
Top-8
94
7,168
2,048
128K
∼36T
Llama 4 Scout
109B
17B
17e
1R + 1S
∼48
∼5,120
∼8,192
10M
40T
Llama 4 Maverick
400B
17B
129e
1R + 1S
∼48
∼5,120
∼8,192
1M
40T
Arctic
480B
17B
128f
Top-2
—
—
∼3,660
128K
3.5T
Kimi K2
∼1T
32B
384
Top-8
61
7,168
2,048
128K
15.5T
a 160 routed (R) + 2 shared (S) experts.
b 256 routed + 1 shared expert.
c Mixtral replicates only the FFN layers; attention weights are shared, giving
∼47B total rather than 8 × 7B = 56B.
d No shared experts; uses global-batch load-balancing loss.
e Maverick uses alternating dense and MoE layers; values marked ∼are approximate (not fully
disclosed by Meta).
f Arctic is a Dense-MoE hybrid: 10B dense transformer backbone + residual 128 × 3.66B MoE MLP.
R = routed, S = shared.
E. Collective Communication Algorithms
Under expert parallelism, each MoE layer incurs four all-to-
all operations (two per forward pass and two per backward),
making all-to-all a dominant latency contributor at scale [14],
[12].
Flat all-to-all. NCCL [26] and RCCL [27] perform direct
point-to-point transfers between all process pairs, which is
bandwidth-optimal under a uniform topology but performs
poorly on hierarchical networks where inter-node bandwidth
is significantly lower than intra-node bandwidth.
Hierarchical all-to-all. Tutel [22], FasterMoE [28], and
HetuMoE [29] use a two-phase approach, intra-node aggre-
gation followed by reduced inter-node transfers, substantially
reducing cross-node messages. However, these algorithms treat
the inter-node network as homogeneous.
Dragonfly topology. HPC systems such as Frontier [16]
employ Dragonfly networks [30] with high-bandwidth intra-
group links and sparser inter-group links. Topology-oblivious
algorithms cause unnecessary contention on slower inter-group
links. Our topology-aware all-to-all (Section V) explicitly
models this three-level hierarchy (intra-node, intra-group, inter-
group) and coordinates asynchronous point-to-point communi-
cation to eliminate idle cycles.
F. Analytical Performance Modeling
Roofline analysis [31] bounds achievable performance by
arithmetic intensity and memory bandwidth, informing kernel
optimization for attention and FFN layers [32]. Korthikanti
et al. [33] develop analytical models for pipeline bubble and
activation memory in dense Transformer training; PaLM [34]
uses empirical roofline fitting to project hardware efficiency
across configurations. For MoE models, interactions among
expert parallelism, routing, and load imbalance introduce
variables that dense-model frameworks do not capture. No prior
work provides a unified model spanning memory, compute,
and communication jointly across different parallelization
dimensions validated on a real HPC platform, a gap we address
in Section III.
III. PIPER: A FRAMEWORK FOR PIPELINING MOE
TRAINING VIA RESOURCE MODELING
Piper is a framework for efficient MoE model training on
HPC platforms. Its design rests on two observations. First,
existing frameworks such as DeepSpeed-MoE, DeepSpeed-
TED, and X-MoE, distribute all model components across large,
groups of GPUs, forcing expensive collective communications
(4 all-reduce for tensor parallelism, 2 all-gather for sharded data
parallelism, 4 all-to-all for expert parallelism) to span many
ranks simultaneously. Second, pipeline parallelism, which is
the standard tool for bounding communication group size in
dense model training [3], has not been applied to MoE training
due to the added complexity of intra-layer expert parallelism.
Piper closes this gap by composing pipeline parallelism with
expert parallelism. It organizes P GPUs into a PP × EP
device mesh: PP pipeline stages, each staffed by EP GPUs
that handle one partition of experts for L/PP layers via expert-
data parallelism. Confining expert-parallel communication to
a small, topologically local group of GPUs, ideally within a
single node or a single-hop Rosetta switch group on Frontier
allows Piper to exploit fast intra-node interconnects and avoid
the high latency of large-scale all-to-all collectives. We expand
on Tutel [22] to support the underlying expert-parallelism.
Piper consists of four components, described in turn: (i) an
analytical resource model that estimates memory, compute, and
communication for any (PP, EP) configuration and a model
architecture (Section III-A1); (ii) a micro-benchmarking suite
that measures platform-specific bandwidth and throughput to
parameterize the model (Section IV-A); (iii) a performance
estimator that scores valid configurations by predicted MFU
(Section IV-C); and (iv) a pipelined training executor that


---

implements the selected strategy with an efficient 1F1B
schedule (Section VII).
At the core of our framework is Pipeline Parallelism on top
of expert parallelism (Figure 2). We use Tutel for facilitating
expert-parallelism. Piper framework has a resource modeling
component, a micro-benchmarking suit, a performance estima-
tor, and a pipelined training tool that partitions MoE models
across layers.
A. Resource Modeling for MoE Training
Training large MoE models faces two coupled resource
constraints: (i) memory pressure from model parameters,
optimizer states, and expert activations, and (ii) communication
latency from all-to-all collectives under expert parallelism.
We develop analytical models for both, parameterized by the
notation in Table II, and use them to prune the (PP, EP)
search space to configurations that are memory-feasible and to
rank feasible configurations by predicted throughput.
a) Notations: We will use the notations listed in Table II
for resource modeling.
TABLE II: Symbol definitions.
Symbol
Description
d
Model hidden dimension (dmodel)
L
Total transformer layers (assume all are MoE layers)
LMoE
Number of MoE layers; L −LMoE are dense
H
Number of attention heads
dh
Per-head dimension (H · dh = d)
E
Routed experts per MoE layer
Es
Shared (always-active) experts per MoE layer
k
Top-k routing (experts activated per token)
nmat
Weight matrices/expert: 3 (SwiGLU)
dMoE
ffn
Expert FFN intermediate dimension
dd
ffn
Dense FFN intermediate dimension
PP
Pipeline parallel degree
EP
Expert parallel degree
P = PP × EP
Total GPU count
g
GPUs per node
s
Sequence length (tokens)
b
Global batch size (sequences)
bµ = b/M
Microbatch size
M = α · PP
Total microbatches per gradient step
α
Microbatch multiplier
i
Pipeline stage index, 0 ≤i ≤PP −1
βp
Bytes per parameter (on GPU)
Mfw
Framework overhead (RCCL buffers, etc.)
1) Modeling Memory:
a) Modeling memory under expert-data parallelism: The
total memory required for training an MoE model constitutes
a) static memory (parameter + optimizer states + gradients)
and b) activation memory.
Mixed-precision training stores parameters in multiple for-
mats simultaneously. The total bytes consumed on GPU per
parameter is 16, 2 Bytes for fp16 param, 2 Bytes for fp16
grad, 4 Bytes for fp32 master copy, and (4 + 4 = 8) Bytes
fp32 momentum and variance.
We first establish a lower bound on the memory by
assuming everything fits in a hypothetical GPU with infinite
memory so that no model parallelism is required (Table III).
There are 4d2
model attention parameters (WK, WQ, WV
∈
Rdmodel×dmodel) for Multi-Head Attention (MHA). Each of
the FFN experts has 3dmodeldffn weights since there are
three weight matrices (Wup, Wgate
∈Rdmodel×dffn and
Wdown ∈Rdffn×dmodel).
Accounting for the activation memory, intermediate output
is calculated in half precision (2 Bytes), for b sequences with s
sequence length, each expert receives se; E[se] ≈bsk
E
tokens.
For a single expert with SwiGLU activations, each token activat-
ing an expert creates 3dffn (with fused kernel) or 4dffn values
(up, gate, down). Since E[se] = bsk/E, per expert activation
memory is 2Bytes × bsk/E × (3dffn + dmodel). Activation
memory from the attention module is 2Bytes × (6bsdmodel
(Q, K, V projections, Attention output, output projections) +
2Bytes × 2bHs2 (Attention score, softmax output) (Table III).
With, flash attention, 4bHs2 →2bHs.
Model
#Parameters
Model Memory
Activation Memory
Attention
4d2
model
64d2
model
12bs dmodel + 4bHs2
Experts
3Edmodeldffn
48Edmodeldffn
2bsk(3dffn + dmodel)
TABLE III: Estimating training memory
Mu = L

64d2
model + 48Edmodeldffn + 12bsdmodel
+ 4Hbs2 + 2bsk
 3dffn + dmodel
 (1)
2) Memory with Expert Data Parallelism: In expert data
parallelism, the world size is P = EP. The non-expert modules
(attention) get replicated across P GPUs and each GPU gets
E/EP experts. The total memory consumption increases due
to the replicated attention module. However, the per GPU
memory requirement comes down to
Medp = L

64d2
model + 48E
EP dmodeldffn + 12bsdmodel
+ 4Hbs2 + 2bsk
EP
 3dffn + dmodel

(2)
a) Memory Under Pipelined Expert Parallelism: In the
hybrid Pipeline-Expert parallelism, we divide the P GPUs into
PP × EP grid and there are EP pipeline parallel groups of
size PP and there are PP expert-parallel groups of size EP.
In each of the PP pipeline stages, there are EP GPUs hosting
l = L/PP layers. Each of these EP GPUs is replicating the
non-expert parameters (attention, router) and hosting E/EP
experts in an expert-data parallel way. So, each GPU’s memory
is l × (attention-memory + E/EP expert-memory). The
peak activation memory in GPipe schedule is for all of the M


---

Fig. 1: Piper framework for efficient MoE training
Fig. 2: Pipeline Parallelism on Expert Parallelism
microbatches since at the steady state all microbatches needs
to stay alive in the memory.
M GP ipe
edp×pp =
L
PP

64d2
model + 48E
EP dmodeldffn
+ 12bsdmodel + 4Hbs2 + 2bsk
EP
 3dffn + dmodel

(3)
b) 1F1B Pipeline Schedule: In 1F1B scheduling, stage i
(0-indexed, i ∈{0, 1, . . . , PP −1}) holds (PP −i) in-flight
microbatch activations simultaneously at peak. Each microbatch
has size b/M, where M is the total number of microbatches.
Hence, the per-GPU memory for stage i is:
M 1F1B
edp×pp(i) =
L
PP ×
 
64d2
model + 48E
EP · dmodeldffn + (PP −i)
12b
M sdmodel + 4b
M Hs2 +
2bsk
M · EP (3dffn + dmodel)
 !
(4)
The memory difference between the first and last stage is:
∆M = M1F1B(0) −M1F1B(PP −1)
= L(PP −1)
PP
12b
M sdmodel + 4b
M Hs2 +
2bsk
M · EP (3dffn + dmodel)

(5)
That means, the first stage needs to hold (PP −1)-times
more activation memory than the last stage. This creates heavily
skewed memory pressure across pipeline stages.
B. Modeling Communication
1) Communication Under Expert-Data Parallelism: Under
expert data parallelism, major communication happens due
to the activation values. There are two phases of all-to-all
communications.
a) Dispatch and Combine: Each data-parallel router
routes it’s bsk tokens (bs tokens, but each token gets to k
experts) to E experts. Under proper load-balancing, every
GPU sends bsk/EP tokens to every other GPU. The com-
munication volume between a pair of GPUs is 2bsk/EP and
total communication volume during the dispatch all-to-all is
 EP
2

× 2bsk
EP
= (EP −1)bsk. These are tokens, so they
have dmodel dimensions, in fp16/bf16, the total message size
is 2(EP −1)bsk × dmodel bytes. Individual message size is
≈4bskdmodel
EP
bytes.
Combine is the same communication in the reverse direction.
So, the message volumes are same. The per-GPU send volume
during dispatch is 2bskdmodel
EP
bytes (fp16), giving a per-NIC
injection load of 4bskdmodel
EP ·BNIC seconds at NIC bandwidth BNIC.
Since combine is the reverse operation, the total all-to-all
latency per MoE layer in the forward pass is bounded by:
Ta2a ≥4bskdmodel
EP · BNIC
(6)
This bound is tight when NICs are uniformly saturated;
Section V discusses why the flat RCCL all-to-all fails to


---

achieve it on Dragonfly topologies and how our topology-aware
algorithm addresses this.
2) Communication under Pipeline and Expert-Data Paral-
lelism: World size P = EP × PP. For each stage hosting
l-layers, there are 2l all-to-all communication calls in the
forward pass.
Between tow stages a batch of P2P communication happens.
Each of the EP GPUs hosting the last layer of the ith stage
sends 2bsdmodel bytes to it’s counterpart to the first layer of the
(i + 1)st stage. Total message sent between two stages in this
stage is 2EP × bsdmodel bytes. Now, under different pipeline
scheduling, number of stages participating in this concurrent
communication varies.
C. Finding Valid Parallelization Strategies w/o OOM
PP × EP = n × g
(total GPU count)
(7)
EP | E
(EP divides expert count)
(8)
PP ≤L
( ≥1 layer per stage)
(9)
EP ≤g · Nh
(EP within fast-interconnect domain)
(10)
Mpeak(0) ≤CGPU
(worst-case stage fits in HBM)
(11)
where Nh is the number of nodes sharing a single-hop
interconnect (e.g., Nh
= 4 for a Rosetta switch group
on Frontier). The fourth constraint ensures that all-to-all
communication during expert dispatch stays within the fast
intra-group fabric. The fifth uses the stage-0 peak from the
1F1B model, which is the binding memory constraint across
all stages.
D. Pipelined Training Execution
Among valid configurations, Piper ranks them by esti-
mated MFU. Then it interfaces PyTorch Distributed Pipeline
Parallelism with expert-parallel or any other hybrid intra-
node distribution library such as Tutel. We instrumented
1F1B schedule and installed synchronization mechanisms
among expert-parallel group members. We expanded Tutel and
PyTorch’s pipeline parallelism mechanism to work with each
other under a two-dimensional parallelization (three, counting
external data parallelism).
IV. PERFORMANCE MODELING THROUGH
MICRO-BENCHMARKING
Resource modeling gives us a realistic expectation regarding
viable distributed training strategies in terms of number
of nodes, degrees of expert and pipeline data parallelism,
whether to adopt memory saving techniques such as checkpoint
activation, offloading, etc. This can be done largely statically
by the mathematical formulas we developed in Section III-A
and considering system properties such as HMB memory. Once
we find candidate strategies to train our model without running
into out of resource (e.g., OOM) error.
To estimate the model flops utilization or MFU for each
of these strategies, we need to micro-benchmark the HPC
platform for various phases of the MoE training workflow.
With pipelined expert-data parallelism, the typical flow is

attn →routing →dispatch a2a →expert →combine a2a

×l →P2P
So, we need to run micro-benchmarking on computational
performance and communication latency.
A. Micro-benchmarking Computation
Every transformer layer has two major components, non-
expert attention module, and multiple Feed Forward Network
(FFN) experts. In an expert-data parallel setting, we need to
distribute the experts equally across the g gpus in a node and
replicate the attention module in a data-parallel way across all
g gpus. Each GPU needs to host and process one attention
module and E/g experts. We start with separate optimization
of these two parts and identify the suitable kernels optimized
for each part.
1) Attention Performance: For attention part, different
models have different model dimension, and the flash-attention
kernel is optimized for only a fixed set of head-dimension.
We benchmark a single GPU to identify the best performing
head-dimension. We need to choose a set of best throughput
generation < batch size, head dimension > tuples. Figure 3
shows our measure performances for various MoE model
architectures.
512
1K
2K
4K
8K
Sequence Length
60
70
80
90
100
110
120
TFLOPS  (QKV proj + causal SDPA + out proj)
100 TFLOPS
Proj-
dominated
Attention Module Throughput vs Sequence Length
AMD Instinct MI250X  ·  fp16  ·  B=4  ·  8 MoE Models
OLMoE-1B-7B
Mixtral-8×7B
Qwen2-57B-A14B
DeepSeek-V2
Mixtral-8×22B
DBRX
Grok-1
DeepSeek-V3
GQA  (6 models 
 blue/green/purple family)
MHA / MLA  
 DeepSeek-V2
MHA / MLA  
 DeepSeek-V3
Fig. 3: Achievable throughput for various models for different
sequence lengths.
2) Expert Performance: Similarly, for the FFN part, which
mostly requires GEMM operations, we need to identify the
set of (num tokens, batch size, expert dimension) tuples that
gives us the best throughput. For the fine-grained experts where
GEMM between many tall and skinny matrices are involved,
the search becomes more involved (Figure 4).
Then, from these two sets, we need to identify the tuples
with agreeing batch size. Choosing this batch size becomes a
critical choice, especially in the context of pipeline parallelism
since choosing the right number of micro batch size, and right
value for micro batch size will determine pipeline bubble and
computation efficiency.


---

200
400
8001K
2K
4K
8K10K
20K
40K
80K100K
Total Tokens in Batch  (log scale)
0
20
40
60
80
100
120
Avg TFLOPS  (Up + Down) / 2
100 TFLOPS
Memory-latency
bottleneck region
OLMoE-1B-7B
Mixtral-8×7B
Qwen2-57B-A14B
DeepSeek-V2
Mixtral-8×22B
DBRX
Grok-1
DeepSeek-V3
(c)
MoE Expert GEMM Throughput vs Batch Size  ·  Average (Up + Down) / 2
AMD Instinct MI250X  ·  fp16  ·  8 MoE Models
OLMoE-1B-7B
Mixtral-8×7B
Qwen2-57B-A14B
DeepSeek-V2
Mixtral-8×22B
DBRX
Grok-1
DeepSeek-V3
Fig. 4: MoE GEMM Performance
B. Micro-benchmarking Communication
In our framework, we incur two types of communication,
all-to-all within an expert-parallel group and send-recv between
two pipeline stages. Since, between stage send-recv communi-
cation happens following a synchronization across the expert
parallel group, there are EP concurrent P2P communications
between two stages.
For benchmarking all-to-all bandwidth, we vary the number
of GPUs from 2 to 64 spanning 1-8 nodes, for various message
sizes (Figure 5).
Fig. 5: Benchmarking All-to-all bandwidth for various message
sizes. Bandwidth drops significantly as soon as all-to-all
involves inter-node communicaiton
C. MFU Estimation
Let tcompute and tcomm be the attention + expert and A2A
+ P2P totals summed over the full step. Then:
MFU =
Fmodel
ˆπeff · G · tcompute
|
{z
}
hardware efficiency
×
tcompute
tstep
|
{z
}
compute fraction
,
tcompute
tstep
= 1−b−tcomm
tstep
(12)
V. HALO: HIERARCHICAL AFFINITY-AWARE
LOCALITY-OPTIMIZED ALL-TO-ALL
Our algorithm pursues five design goals. It saturates all
four NICs on a single node during inter-node communication,
and identifies the three phases of communication along with
their dependency structure in order to maximize concurrency
across phases. At the intra-node level, it enforces maximum
locality by exploiting GPU-to-NIC affinity. At the inter-node
level, it leverages the Dragonfly grouping structure by treating
the four nodes connected to a common Rosetta switch as a
single communication locality domain. Finally, when possible,
it constrains node allocation to within the same rack entirely,
avoiding the slowest inter-rack communication links.
A. Communication Group Construction
We assign GPUs sharing a NIC to the same inter-node
communicator, ensuring that inter-node traffic from a given
GPU is always injected through its affinitized NIC. This
prevents NIC contention and allows all four NICs on a node
to operate at full bandwidth simultaneously.
Fig. 6: Communication groups for Topological all-to-all
B. Three-Phase Algorithm and Dependency Structure
The three phases exhibit the following dependency structure:
Phase I ∥
 Phase II →Phase III

(13)
Phase I (intra-node all-to-all) is fully independent of the inter-
node phases, since all source-destination pairs are known locally
and require no prior accumulation. Phase III depends on the
completion of Phase II, as it redistributes the received remote
data within the node. We exploit the Phase I independence by
launching it concurrently with Phase II or Phase III, hiding a
significant portion of intra-node communication latency behind
inter-node transfers. Figure 7 demonstrates our three phases
and their overlapping strategies.
C. Rack-Aware Node Allocation
When the job scheduler permits, HALO constrains node
allocation to within a single rack to eliminate inter-rack traffic,
which traverses the slowest links in the Dragonfly topology.
Within a rack, nodes are further grouped by Rosetta switch
affinity: the four nodes sharing a switch form a single locality
domain, and inter-group communication is batched across this
boundary to minimize contention on inter-switch links.


---

Algorithm 1: HALO ALLTOALL
Input: S∈RN×D send buffer, N = num nodes×R
communication groups and rank metadata from
SETUPCOMMS
Output: O∈RN×D fully transposed output buffer
1 local start ←node id · R;
N−←{n ∈[0, num nodes) : n̸ = node id};
M ←|N−|
// Persistent buffers — allocated once on first call, reused
thereafter
2 P1 ∈RR×D
// Ph1 recv
3 Fs
2, Fr
2 ∈RMR×D
// Ph2 send/recv flat buffers
4 Fs
3, Fr
3 ∈RR×MD
// Ph3 send/recv flat buffers
5 O∈RN×D// output
6 Phase 1: Intra-node Extraction
// Contiguous copy of this node’s R rows
7
L ←S[local start : local start+R]
// shape (R, D)
// Intra-node exchange: rank ℓreceives row ℓfrom every
local peer
8
P1 ←ALLTOALLSINGLE(L, internal group)
9 Phase 2: Inter-node Exchange
// Pack all remote-destined rows with a single
INDEXSELECT kernel
10
idx ←

nR, . . . , nR+R−1
 n ∈N−

// pre-built LongTensor
on GPU
11
Fs
2 ←INDEXSELECT(S, dim = 0, idx) // one GPU kernel;
shape (MR, D)
// Batched P2P over world group (RDMA); one send+recv
slice per remote node
12
for i, n ∈ENUMERATE(N−) do
13
peer ←n · R + local rank
14
IRECV(Fr
2[iR : (i+1)R], src = peer)
15
ISEND(Fs
2[iR : (i+1)R], dst = peer)
16
WAITALL()
// flush all RDMA ops
17 Phase 3: Intra-node Redistribution
// Transpose Ph2 recv buffer; one GPU kernel — no
intermediate allocation
18
Fs
3 ←
RESHAPE

PERMUTE
 VIEW(Fr
2, M, R, D), [1, 0, 2]

, [R, MD]

// row ℓ= all remote data for local rank ℓ
// Scatter repacked rows to final owners on this node
19
Fr
3 ←ALLTOALLSINGLE(Fs
3, internal group)
// rank
ℓreceives its (MD)-wide slice
// Write Phase 1 (local) and Phase 3 (remote) results into
output buffer
20 O[local start : local start+R] ←P1
21 T ←PERMUTE
 VIEW(Fr
3, R, M, D), [1, 0, 2]

// shape
(M, R, D)
22 for i, n ∈ENUMERATE(N−) do
23
O[nR : (n+1)R] ←T[i]
24 return O
Fig. 7: Three phases of our HALO all-to-all algorithm. Phase
I can happen concurrently with either of Phase II and Phase
III, but Phase III needs to wait for Phase II to complete.
D. Comparison against torch.dist.all to all
We compare HALO against torch.dist.all_to_all
backed by RCCL across varying node counts and message
sizes (Figure 8). HALO achieves 1.1×–9× lower latency for
configurations of 16 nodes or more. The crossover at 16
nodes reflects the threshold at which inter-rack communication
becomes dominant under the flat RCCL implementation —
precisely the regime where HALO’s rack-aware grouping and
NIC saturation strategy yield the largest gains. At smaller scales,
where all GPUs fit within a single switch group, both algorithms
perform comparably since single-hop communication already
saturates available bandwidth.
2.1
4.2
8.4
16.8
33.6
67.1
134.2
268.4
536.9
2147.5
4295
Data Size / Rank (MB)
0.6×
0.8×
1.0×
1.2×
1.4×
1.6×
1.8×
2.0×
Speedup vs. Reference
1.80×
1.75×
1.78×
16 nodes
128 GPUs
4.2
8.4
16.8
33.6
67.1
134.2
268.4
536.9
1073.7
4295
Data Size / Rank (MB)
1.0×
2.0×
3.0×
4.0×
5.0×
6.0×
7.0×
8.0×
9.0×
Speedup vs. Reference
8.03×
7.92×
8.06×
32 nodes
256 GPUs
8.4
16.8
33.6
67.1
134.2
268.4
536.9
1073.7
2147.5
Data Size / Rank (MB)
1.0×
2.0×
3.0×
4.0×
5.0×
Speedup vs. Reference
4.44×
4.94×
4.67×
64 nodes
512 GPUs
16.8
33.6
67.1
134.2
268.4
536.9
1073.7
2147.5
4295
Data Size / Rank (MB)
1.0×
2.0×
3.0×
4.0×
5.0×
Speedup vs. Reference
1.81×
3.54×
4.40×
128 nodes
1024 GPUs
33.6
67.1
134.2
268.4
536.9
1073.7
2147.5
4295
Data Size / Rank (MB)
1.0×
2.0×
3.0×
4.0×
5.0×
6.0×
Speedup vs. Reference
4.36×
3.44×
5.56×
256 nodes
2048 GPUs
67.1
134.2
268.4
536.9
1073.7
2147.5
4295
Data Size / Rank (MB)
0.5×
1.0×
1.5×
2.0×
2.5×
3.0×
3.5×
4.0×
Speedup vs. Reference
3.50×
3.54×
3.03×
512 nodes
4096 GPUs
V0 Sequential
V1 Ph1 Ph2
Ph3
V2 Ph2
Ph1 Ph3
dist.all_to_all
Baseline
Fig. 8: Latency comparison of Neighborhood-all to all algo-
rithm against the RCCL based torch.dist.all to all algorithm.
VI. LOAD BALANCING THROUGH EXPERT MIGRATION
The load across GPUs emerging from router’s expert
selection can vary over training period. Initially, all experts
are preferred by the router equally; however small random
difference can make some experts more favorable. These
favored experts get more tokens, and consequently more
gradient updates. This creates a positive feedback loop where a
subset of experts become more capable than others, and end up
getting more tokens consistently. This causes expert collapse.
In the middle part of the training, experts become specialized
in different types of input. The load distribution stabilizes, and


---

different experts develop distinct activation patterns across the
feature space. Towards the end, the specialization becomes
more concrete. Some experts remain underutilized.
Unless there is an auxiliary load-balancing loss or other
load balancing mechanism in place, expert workload remains
skewed. So, a static expert parallelism at the beginning of the
training cannot ensure that all the GPUs within a layer are
receiving balanced workload. While load balancing loss helps
with load balancing in the middle stages of the training, it
still suffers from prolonged period of load imbalance. Figure 9
shows expert-routing distribution in different layer becomes
balanced after almost one billion token consumption.
Existing MoE training methods are averse to expert re-
assignment for load balancing due to the perceived overhead
of expert migration. Under existing distribution techniques that
distribute experts among many nodes through slow Ethernet,
this caution is justified. However, when we successfully utilize
pipeline parallelism to localize the experts from the same
layer to a small number of GPUs connected via fast link
(NVLink or Infinity Fabric), we should reconsider the dynamic
load balancing. In the absence of load balancing loss, the
expert demands are still skewed, but load across the GPUs can
be balanced. Even with the expert collapse, the final model
performance does not suffer.
Fig. 9: Expert load distribution during the training process.
The model has a 350M base model, with 16 experts and expert
frequency 0.5.
A. Dynamic Expert Migration
We extend the router class to maintain token distribution
throughout the training. An external scheduler (could be
intermittent interrupt in the training loop) inspects the growing
load imbalance, and whenever it crosses a pre-determined
threshold, it can run our migration algorithm to identify
the minimal number of intra-group (expert-parallel group)
migrations of experts among GPUs. The cost of migration
will be the cost of an all-to-all in the expert parallel group
where individual expert’s migration cost is the latency of
moving 48dmodeldffn (3dmodeldffn expert parameters, and
each parameter needs 16 Bytes for master copy, optimizer states,
and gradients). In the worst case, when we do a complete re-
assignment from scratch, that would mean we will be moving
all E experts, and the individual send message size will
be 48(E/EP)dmodeldffn (3dmodeldffn Bytes. Assuming a
single layer fits in a node (8 GPUs) with EP=8, the individual
send-message sizes for various SOTA model is in Table IV.
TABLE IV: Per-Layer Worst-Case Expert Migration Message
Size and Latency Per GPU (48 × E × dmodel × dffn/G bytes,
G = 8, bandwidth = 50 GB/s)
Model
E/layer
dmodel
dffn
Send Size /
GPU (GB)
Latency
(ms)
Switch-Base
128
768
2,048
1.21
24.2
Mixtral 8×7B
8
4,096
14,336
2.63
52.6
Mixtral 8×22B
8
6,144
16,384
4.50
90.0
Grok-1
8
6,144
32,768
9.00
180.0
GLaM (1.2T)
64
8,192
32,768
102.88
2057.6
DeepSeek-V2
160
5,120
1,536
7.04
140.8
DeepSeek-V3
256
7,168
2,048
21.00
420.0
With average intra-node all-to-all bandwidth 50GB/s, the
worst case expert migration will only take tens of milliseconds
for most models. However, we will perform this migration
incrementally and intermittently so that only a subset of the
experts need to migrate to balance GPU workload. To rebalance
with minimal swaps, we develop a hill-climbing swapping based
algorithm (Algorithm 19).
Algorithm 2: Hill-Climbing Swap-Based Minimal Rebalancing
Input: Groups G = {G1, . . . , GK}, max iterations T = 100
Output: Rebalanced groups G, swap count c
1 c ←0;
2 for t = 1, . . . , T do
3
sk ←P
n∈Gk n
∀k;
4
k+ ←arg maxk sk;
5
k−←arg mink sk;
6
δ ←sk+ −sk−;
7
best swap ←None,
∆∗←0;
8
foreach (i, n1) in Gk+ do
9
foreach (j, n2) in Gk−do
10
δ′ ←|(sk+ −n1 + n2) −(sk−−n2 + n1)|;
11
if δ′ < δ and (δ −δ′) > ∆∗then
12
∆∗←δ −δ′,
best swap ←(i, j);
13
if best swap̸ = None then
14
(i∗, j∗) ←best swap;
15
Gk+[i∗] ↔Gk−[j∗];
16
c ←c + 1;
17
else
18
break;
19 return G, c
VII. SCALABLE TRAINING OF SOTA MOE MODELS
We have developed resource modeling and a throughput
estimator through micro-benchmarking of Frontier super-
computer to identify the viable and performant distributed
training strategies. For example to train a model (super model
with 545B parameters), we computed the required memory
for various node count with different distribution strategies
(Figure 10).


---

4
8
16
32
64
128
256
Expert Parallel Size (EP)
0
25
50
75
100
125
150
175
200
225
Memory per GPU (GB)
64 GB
HBM limit
16 bytes/param: 2B fp16 weight + 2B fp16 grad + 4B fp32 master + 4B fp32 m1 + 4B fp32 m2
Activation: ⌈L/pp⌉⋅pp ⋅2 b s dm ⋅2 B  (2 checkpoint boundaries/layer: attn module input + MoE module input)
Total GPU Memory vs. EP Size
(External Checkpoint: Attn + MoE)
32 nodes (256 GPUs)
64 nodes (512 GPUs)
128 nodes (1024 GPUs)
Total (model + activation)
Model only (weights + optimizer)
Fig. 10: Identifying viable training strategies for a model
by estimating its per-GPU memory requirement. Here, we
investigate training strategy of a 615 B parameter model. And,
just like this analysis suggests, we can train this model with
at least 64 nodes.
A. Trroughput Ceiling : Training a Single Layer in a Node
Every transformer layer has two major components, non-
expert attention module, and multiple Feed Forward Network
(FFN) experts. In an expert-data parallel setting, we need to
distribute the experts equally across the G gpus in a node and
replicate the attention module in a data-parallel way across all
G gpus. Each GPU needs to host and process one attention
module and E
G experts.
After getting the suitable configuration, we train a single
layer of the SOTA models on a single Frontier node using
expert-parallelism and observe their training performance
(Figure 11). We were able to fit a single layer to a single node
for each of the models we tried. With memory constraint under
pipelined expert-parallelism and pipeline bubble under various
schedules, this sets up the upper limit of training performance
of any model using Piper.
Mixtral
8×22B
Mixtral
8×7B
Llama 4
Maverick
Llama 4
Scout
Arctic
DeepSeek
V3
Kimi K2
DeepSeek
V2
0
20
40
60
80
100
120
140
Throughput (TFLOPS)
129.4
117.5
112.4
109.6
104.2
84.3
81.7
78.3
mean = 102.2
MoE Layer Training Throughput
Frontier Supercomputer (Single-Node EDP)
Meta
Mistral
DeepSeek
Moonshot
Snowflake
Fig. 11: Training efficiency of a single layer of the SOTA
models.
B. Training the Full Model
With our resource modeling, we identified valid paralleliza-
tion strategies for various SOTA models and train them using
their original size. We selectively use activation checkpointing
to fit the model into a small number of GPUs while minimizing
Mixtral 8×7B
47B params
Mixtral 8×22B
154B params
Llama 4 Scout
102B params
Llama 4 Maverick
529B params
DeepSeek V2
235B params
0
20
40
60
80
100
120
Throughput (TFLOPS)
102.8
55.4
74.2
37.8
46.8
TFLOPS
MFU (%)
Act. checkpointing
0
10
20
30
40
50
60
70
MFU (%)
53.8%
29.0%
38.8%
19.8%
24.5%
Fig. 12: Training throughput of SOTA MoE models using
Piper (sequence length 4096). Hatching denotes activation
checkpointing.
the impact on the MFU (Figure 12). From Table 12, we observe
that training models with traditional large experts achieves
best performance while the fine-grained experts achieve lower
performance.
C. Comparison Against Other MoE Training Frameworks
We compare Piper’s training throughput for different model
sizes against state of the art training frameworks such as Tutel,
DeepSpeed MoE, DeepSpeed TED, and most prominent X-
MoE. Since X-MoE is the leading open-source framework
for fine-grained experts training, we compare Piper against
these tools using fine-grained MoEs. Our framework Piper can
train these models with only a fraction of their requirement at
2-3.6X throughput (Figure 13).
Fig. 13: Throughput comparison against SOTA MoE training
frameworks. Piper can train the small, medium, large, and
super model using 8, 32, 80, and 512 MI250X GPUS instead
of 256 and 1024 GPUs used by X-MoE.
D. Training Trillion(s) Parameter Models
We take a dense model [dmodel = 5120, dffn = 20480, L =
32, k = 2] (10 Billion parameters, let’s call it M10B) and


---

scale out its parameter count by scaling the number of experts
(E). We start with 16 experts on 8 nodes and then scale E
proportionally. We use 128 experts on 64 nodes (512 GPUs),
and 256 experts on 128 nodes (1024 GPUs). With this scaling,
we can train a 862 Billion parameter model using 512 GPUs
at 39.38 TFLOPs, and a 1.7 Trillion parameter model using
1024 GPUs at 33 TFLOPS. This is a form of weak scaling,
and the scaling efficiency is 73% from 64 GPUs to 1024 GPUs
(Figure 14).
64
128
256
512
1024
Number of GPUs
0
10
20
30
40
50
60
Throughput (TFLOPs)
45.15
40.88
37.52
39.38
33.04
Throughput and Model Size vs. Number of GPUs
Throughput (TFLOPs)
Ideal throughput (45.15 TFLOPs)
Model Size (B params)
0
250
500
750
1000
1250
1500
1750
2000
Model Size (Billion Parameters)
110 B
E=16, PP=4
110 B
E=32, PP=4
432 B
E=64, PP=8
862 B
E=128, PP=4
1721 B
E=256, PP=4
Fig. 14: Scaling M10B models with experts.
VIII. CONCLUSION AND DISCUSSION
We presented Piper, a holistic MoE training framework that
co-designs distributed training strategy through mathemati-
cal resource modeling, empirical micro-benchmarking, and
platform-aware performance estimation. Its central contribution
is the application of pipeline parallelism on top of traditional
expert-data parallelism, which proves critical in mitigating large-
scale all-to-all latency. Piper achieves 2–3.6× the MFU of X-
MoE on SOTA models, demonstrating that deep understanding
of the target HPC platform translates directly into training
throughput gains. Implemented in Python and PyTorch, Piper is
portable across platforms, and its micro-benchmarking suite can
characterize arbitrary HPC systems. As new MoE architectures
emerge, Piper’s modular design allows incremental integration
of kernel optimizations, improved schedules, and dynamic
load-balancing strategies to deliver continued efficiency gains.


---

REFERENCES
[1] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess,
R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws
for neural language models,” arXiv preprint arXiv:2001.08361, 2020.
[Online]. Available: https://doi.org/10.48550/arXiv.2001.08361
[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances
in Neural Information Processing Systems, vol. 30, 2017. [Online].
Available: https://doi.org/10.48550/arXiv.1706.03762
[3] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary,
V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro
et al., “Efficient large-scale language model training on GPU clusters
using Megatron-LM,” in Proceedings of the International Conference for
High Performance Computing, Networking, Storage and Analysis (SC),
2021. [Online]. Available: https://doi.org/10.1145/3458817.3476209
[4] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “ZeRO: Memory
optimizations toward training trillion parameter models,” in Proceedings
of the International Conference for High Performance Computing,
Networking, Storage and Analysis (SC), 2020. [Online]. Available:
https://doi.org/10.1109/SC41405.2020.00024
[5] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton,
and J. Dean, “Outrageously large neural networks: The sparsely-
gated
mixture-of-experts
layer,”
in
International
Conference
on
Learning Representations (ICLR), 2017. [Online]. Available: https:
//doi.org/10.48550/arXiv.1701.06538
[6] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to
trillion parameter models with simple and efficient sparsity,” Journal of
Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022. [Online].
Available: https://doi.org/10.48550/arXiv.2101.03961
[7] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary,
C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand
et al., “Mixtral of experts,” arXiv preprint arXiv:2401.04088, 2024.
[Online]. Available: https://doi.org/10.48550/arXiv.2401.04088
[8] DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in
LLMs via reinforcement learning,” arXiv preprint arXiv:2501.12948,
2025. [Online]. Available: https://doi.org/10.48550/arXiv.2501.12948
[9] Qwen Team, “Qwen1.5-MoE: Matching 7B model performance
with 1/3 activated parameters,” 2024. [Online]. Available: https:
//qwenlm.github.io/blog/qwen-moe/
[10] Kimi
Team,
“Kimi
k1.5:
Scaling
reinforcement
learning
with
LLMs,” arXiv preprint arXiv:2501.12599, 2025. [Online]. Available:
https://doi.org/10.48550/arXiv.2501.12599
[11] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun,
N. Shazeer, and Z. Chen, “GShard: Scaling giant models with conditional
computation and automatic sharding,” arXiv preprint arXiv:2006.16668,
2020. [Online]. Available: https://doi.org/10.48550/arXiv.2006.16668
[12] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan,
J. Rasley, and Y. He, “DeepSpeed-MoE: Advancing mixture-of-experts
inference and training to power next-generation AI scale,” in Proceedings
of the 39th International Conference on Machine Learning (ICML),
2022. [Online]. Available: https://doi.org/10.48550/arXiv.2201.05596
[13] D.
Dai,
C.
Deng,
C.
Zhao,
R.
X.
Xu,
H.
Gao,
D.
Chen,
J. Li, W. Zeng, X. Yu, Y. Wu et al., “DeepSeekMoE: Towards
ultimate expert specialization in mixture-of-experts language models,”
arXiv preprint arXiv:2401.06066, 2024. [Online]. Available: https:
//doi.org/10.48550/arXiv.2401.06066
[14] Z. Chi, L. Dong, S. Ma, R. Pang, S. Huang, X.-L. Mao, and F. Wei,
“X-MoE: Scalable and finetunable sparse mixture-of-experts transformer
for on-device inference,” arXiv preprint arXiv:???, 2024, wARNING:
original arXiv ID 2405.13089 is incorrect and belongs to an unrelated
paper. Correct ID needs to be verified.
[15] Microsoft DeepSpeed Team, “DeepSpeed tensor, expert, and data
parallelism,”
Microsoft,
Tech.
Rep.,
2022.
[Online].
Available:
https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/
[16] S. Atchley, C. Zimmer, J. Lange, B. Grodowitz, S. Oral et al., “Frontier:
Exploring exascale,” in Proceedings of the International Conference for
High Performance Computing, Networking, Storage and Analysis (SC),
2023. [Online]. Available: https://doi.org/10.1145/3581784.3607089
[17] Qwen Team, “Qwen3 technical report,” arXiv preprint arXiv:2505.09388,
2025. [Online]. Available: https://doi.org/10.48550/arXiv.2505.09388
[18] Kimi Team, “Kimi k2: Open agentic intelligence,” arXiv preprint
arXiv:2507.20534, 2025. [Online]. Available: https://doi.org/10.48550/
arXiv.2507.20534
[19] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary,
V. Korthikanti et al., “Efficient large-scale language model training on
GPU clusters using Megatron-LM,” in Proceedings of the International
Conference for High Performance Computing, Networking, Storage
and Analysis (SC), 2021. [Online]. Available: https://doi.org/10.1145/
3458817.3476209
[20] D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur,
G. R. Ganger, P. B. Gibbons, and M. Zaharia, “PipeDream: Generalized
pipeline parallelism for DNN training,” in Proceedings of the 27th ACM
Symposium on Operating Systems Principles (SOSP), 2019. [Online].
Available: https://doi.org/10.1145/3341301.3359646
[21] D. Narayanan, A. Phanishayee, K. Shi, X. Chen, and M. Zaharia,
“Memory-efficient pipeline-parallel DNN training,” in Proceedings of
the 38th International Conference on Machine Learning (ICML), 2021.
[Online]. Available: https://doi.org/10.48550/arXiv.2006.09503
[22] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas,
J. Jose, P. Ram et al., “Tutel: Adaptive mixture-of-experts at scale,” in
Proceedings of Machine Learning and Systems (MLSys), vol. 5, 2023.
[Online]. Available: https://doi.org/10.48550/arXiv.2206.03382
[23] Y. Huang, Y. Cheng, A. Bapna, O. Firat, M. X. Chen, D. Chen, H. Lee,
J. Ngiam, Q. V. Le, Y. Wu, and Z. Chen, “GPipe: Efficient training of
giant neural networks using pipeline parallelism,” in Advances in Neural
Information Processing Systems (NeurIPS), vol. 32, 2019. [Online].
Available: https://doi.org/10.48550/arXiv.1811.06965
[24] P. Qi, X. Wan, G. Huang, and M. Lin, “Zero bubble pipeline
parallelism,” arXiv preprint arXiv:2401.10241, 2023. [Online]. Available:
https://doi.org/10.48550/arXiv.2401.10241
[25] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M. Dai, Z. Chen,
Q. V. Le, and J. Laudon, “Mixture-of-experts with expert choice routing,”
in Advances in Neural Information Processing Systems (NeurIPS),
2022. [Online]. Available: https://doi.org/10.48550/arXiv.2202.09368
[26] NVIDIA Corporation, “NVIDIA Collective Communications Library
(NCCL),” 2023. [Online]. Available: https://developer.nvidia.com/nccl
[27] AMD Inc., “ROCm Collective Communications Library (RCCL),” 2023.
[Online]. Available: https://github.com/ROCm/rccl
[28] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li,
“FasterMoE: Modeling and optimizing training of large-scale dynamic pre-
trained models,” in Proceedings of the 27th ACM SIGPLAN Symposium
on Principles and Practice of Parallel Programming (PPoPP), 2022.
[Online]. Available: https://doi.org/10.1145/3503221.3508418
[29] X. Nie, P. Zhao, X. Miao, T. Zhao, and B. Cui, “HetuMoE: An
efficient trillion-scale mixture-of-expert distributed training system,”
arXiv preprint arXiv:2203.14685, 2022. [Online]. Available: https:
//doi.org/10.48550/arXiv.2203.14685
[30] J. Kim, W. J. Dally, S. Scott, and D. Abts, “Technology-driven,
highly-scalable dragonfly topology,” in Proceedings of the 35th Annual
International Symposium on Computer Architecture (ISCA), 2008.
[Online]. Available: https://doi.org/10.1109/ISCA.2008.19
[31] S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful
visual performance model for multicore architectures,” Communications
of the ACM, vol. 52, no. 4, pp. 65–76, 2009. [Online]. Available:
https://doi.org/10.1145/1498765.1498785
[32] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. R´e, “FlashAttention: Fast
and memory-efficient exact attention with IO-awareness,” in Advances
in Neural Information Processing Systems (NeurIPS), 2022. [Online].
Available: https://doi.org/10.48550/arXiv.2205.14135
[33] V. A. Korthikanti, J. Casper, S. Lym, L. McAfee, M. Andersch,
M. Shoeybi, and B. Catanzaro, “Reducing activation recomputation in
large transformer models,” in Proceedings of Machine Learning and
Systems (MLSys), 2023. [Online]. Available: https://doi.org/10.48550/
arXiv.2205.05198
[34] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts
et al., “PaLM: Scaling language modeling with pathways,” Journal
of Machine Learning Research, vol. 24, no. 240, pp. 1–113, 2023.
[Online]. Available: https://doi.org/10.48550/arXiv.2204.02311
