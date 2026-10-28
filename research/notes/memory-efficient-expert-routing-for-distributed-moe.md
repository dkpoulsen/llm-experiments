---
title: Memory-Efficient Expert Routing for Distributed MoE
id: memory-efficient-expert-routing-for-distributed-moe
tags:
- cmp170hx-training-feasibility-a0d4fc
- moe-training
- memory-efficiency
- expert-parallelism
- activation-memory
- nvlink-scope
created: '2026-10-27T10:12:31.527994Z'
updated: '2026-10-27T19:00:31.152820Z'
source: https://arxiv.org/abs/2610.07333
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:31.526948Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'RelayMoE (Tarafder et al., William & Mary + BSC, arXiv 2610.07333): memory-efficient
  expert routing for distributed MoE training. Its diagnostic contribution is the
  most useful part for a 64 GB-card budget: peak memory in MoE training is set by
  the MoE block, not attention, because every intermediate buffer in the dispatch
  pipeline is scaled by top-k. Profiling a Qwen3-style model (top-8, 4 layers) at
  micro-batch 24 / 16384 tokens on 8x H100 with TP=SP=2, CP=4, EP=8 and full recompute,
  the MoE input activation is only 336 MiB per GPU but the top-k-expanded dispatch
  tensors are ~2.6 GiB each (8x expansion) and per-layer peak exceeds 42 GB during
  backward, with spikes to 51 GB against a 14 GB inter-spike baseline. RelayMoE replaces
  the single-shot all-to-all dispatcher with a ring that circulates expert weights
  or tokens, computes locally at each hop, and reconstructs expert intermediates hop-by-hop
  during backward so they can be freed. Results on 30B-57B production MoE models:
  2x average single-layer speedup over Megatron-LM, up to 2.02x full-model throughput
  at the same memory budget, 3.8x average (7.6x peak) reduction in MoE execution memory,
  and up to 2.85x longer trainable sequence. Scope caveat: the ring relies on GPU-to-GPU
  peer-to-peer hops over NVLink 4.0 within a node and NDR InfiniBand across nodes
  — the exact capability the CMP 170HX box lacks, so the mechanism does not port,
  but the memory arithmetic does.'
raw_file: raw/memory-efficient-expert-routing-for-distributed-moe.pdf
doi: arXiv:2610.07333
utility_score: 12.0
citation_count: 0
---

Memory-Efficient Expert Routing for Distributed MoE
Training
Arnab Kanti Tarafder
aktarafder@wm.edu
William & Mary
Jaume Guasch-Martí
jaume.guasch@bsc.es
Barcelona Supercomputing Center
Gokcen Kestor
gokcen.kestor@bsc.es
Barcelona Supercomputing Center
Jie Ren
jren03@wm.edu
William & Mary
Abstract
As Mixture-of-Experts (MoE) models scale toward hundreds
of experts and higher top-k routing, memory efficiency in dis-
tributed training becomes a critical bottleneck. Peak memory
is dominated by the MoE block, not attention: every inter-
mediate buffer in the MoE dispatch pipeline is individually
scaled by top-k routing. The standard all-to-all dispatcher
sends all routed tokens in a single collective step, requiring
the full top-k-expanded buffer to be constructed at once.
In this work, we propose RelayMoE, a ring-based MoE ex-
ecution model that computes locally as expert weights or to-
kens circulate, avoiding full top-𝑘-expanded dispatch buffers.
RelayMoE selects between expert and token routing accord-
ing to communication volume and overlaps transfers with
computation. The ring structure naturally supports memory-
efficient MoE recomputation during backward: each hop
reconstructs expert intermediates, uses them to compute
gradients, and releases them before the next hop. The saved
memory supports longer sequences and larger batches, or
retains more attention activations to reduce attention re-
computation and improve training throughput. We evaluate
RelayMoE on 30B–57B production MoE models and varied
expert configurations. In single-layer MoE experiments, Re-
layMoE achieves a 2× average speedup over Megatron-LM.
In full-model training under the same GPU memory bud-
get, it improves throughput by up to 2.02× and extends the
largest tested trainable sequence length by up to 2.85×.
1
Introduction
The rapid scaling of large language models (LLMs) has in-
creasingly relied on Mixture-of-Experts (MoE) architectures.
By activating only a subset of experts per token, MoE models
achieve stronger quality at a fraction of the compute cost
of dense models [6, 8, 30], making them a common architec-
ture for frontier systems. This advantage has led to larger
and sparser MoE configurations along multiple dimensions
simultaneously: more experts, higher top-k routing, longer
training sequences, and wider expert Feedforward Network
(FFN) layers, as shown in Table 1. Each of these dimensions
independently improves model quality, but together they
Table 1. MoE architecture trends in frontier models. Buffer
Multiplier shows how each dispatch buffer is scaled relative
to the base activation tensor by top-𝑘routing. Combined
with growing sequence lengths (32K→262K), peak memory
from MoE dispatch has grown by an order of magnitude.
Model
Date
# of
Experts
Top-
k
Expert
FFN
Max
Context
Buffer
Mult.
Mixtral 8x22B [16]
04/24
8
2
16384
32K
2×
DeepSeek-V3 [5]
12/24
256
8
2048
128K
8×
Qwen3-30B-A3B [39]
04/25
128
8
768
131K
8×
Kimi K2 [38]
07/25
384
8
2048
128K
8×
Qwen3-Next-80B-A3B [28]
09/25
512
10
512
128K
10×
Qwen3.5-35B-A3B [29]
02/26
256
8
512
262K
8×
Qwen3.5-397B-A17B [29]
02/26
256
8
512
131K
8×
increase GPU memory requirements during distributed train-
ing.
To train these increasingly large MoE models, expert par-
allelism (EP) [30, 35] distributes experts across GPUs and
dispatches tokens to their assigned experts via all-to-all col-
lective communication. However, the all-to-all dispatcher
requires the full top-k-expanded buffer to be constructed
before communication can begin. Because each token is
replicated to its top-k selected experts, every intermediate
buffer in the dispatch pipeline is 𝑘× larger than the base
activation tensor. This compounds further with increasing
sequence lengths and batch sizes. As a result, peak memory
from MoE dispatch has grown by an order of magnitude in
recent architectures (Table 1). Meanwhile, attention memory
has been effectively reduced through memory-efficient algo-
rithms [3, 4, 15, 19, 24] and hybrid linear-attention designs in
recent architectures [29], leaving MoE dispatch as the domi-
nant contributor to peak memory. As Figure 3 shows, MoE
blocks create transient memory spikes that determine the
OOM boundary and limit the maximum trainable sequence
length and batch size.
Prior work has extensively optimized EP communica-
tion. Libraries such as DeepEP [45], HybridEP [46], and
NCCL-EP [10] have improved dispatch throughput and la-
tency through fusing permutation with communication, and
communication–computation overlap. While these techniques
can reduce the number of coexisting intermediate buffers,
token dispatch and expert computation still create memory
1
arXiv:2610.07333v1  [cs.DC]  5 Oct 2026


---

peaks that limit training (§ 2). Multiple expanded activations
can coexist as dispatch and compute stages allocate new
outputs while retaining their inputs. Reducing this mem-
ory footprint requires rethinking how communication and
computation are organized in the MoE block.
Tutel [14] pipelines token dispatch and expert compu-
tation using multiple streams, while FasterMoE [13] uses
fine-grained scheduling to overlap communication with com-
putation. Janus [25] instead fetches expert weights to process
local tokens. These works primarily target communication
efficiency, while peak GPU memory during MoE execution
remains a limiting factor, especially during backward.
In this paper, we present RelayMoE, a ring-based MoE
execution model that reduces peak GPU memory by com-
puting locally as data circulates through the ring. RelayMoE
circulates expert weights or token activations through a ring
of GPUs. At each hop, a GPU computes partial results using
the data present on that device and forwards the visiting
data to the next GPU; the partial results accumulate into the
final output as the data completes the ring. Because no GPU
ever needs to receive and hold data from all other GPUs, the
memory footprint of the MoE block is capped by the local
data plus a fixed number of in-flight partitions, independent
of the top-𝑘factor and the EP degree. Rather than reserving
a fixed token capacity for each expert, RelayMoE processes
the routed token–expert pairs partition by partition. This
enables dropless, padding-free expert computation and sup-
ports longer sequences and larger batch sizes on the same
hardware.
To turn these memory savings into higher training through-
put, RelayMoE selects between expert and token routing ac-
cording to communication volume and overlaps transfers
with local computation. Memory efficiency must also extend
to backward: activation checkpointing discards forward in-
termediates, but reconstructing them together can recreate
a large memory peak. Reconstructed activations remain in
memory while gradient computation produces additional
intermediate tensors. The ring structure naturally supports
memory-efficient MoE recomputation during backward: at
each hop, RelayMoE reconstructs the intermediates needed
for local gradient computation and releases them immedi-
ately afterward. This prevents expert intermediates from
accumulating across hops and leaves more memory for re-
taining attention activations, reducing attention recomputa-
tion.
We make the following contributions:
• We identify the memory overhead of full top-𝑘-expanded
dispatch buffers and propose a ring-based MoE ex-
ecution model that avoids materializing them, with
adaptive routing between experts and tokens.
• We design memory-efficient MoE recomputation that
reduces the backward memory peak without an addi-
tional ring traversal, and selective attention recompu-
tation that converts these memory savings into higher
training throughput.
• We evaluate RelayMoE on production MoE models and
across varied expert configurations. Compared with
all-to-all, RelayMoE achieves a 2× average speedup
in MoE layer execution at EP=8, reduces peak GPU
memory during MoE execution by 7.6× on average
at EP=64, and extends the largest tested trainable se-
quence length by up to 2.85× in full-model training.
2
Background and Motivation
2.1
MoE Training with Expert Parallelism
MoE-based language models replace the dense feedforward
network (FFN) in each transformer layer with a set of expert
FFNs and a learned router (Figure 1, left). For each input
token, the router computes a score over all experts and se-
lects the top-k highest-scoring experts to process that token
(Figure 1, upper right). The outputs of the selected experts
are weighted by the router scores and summed to produce
the final output. Because only 𝑘out of 𝐸total experts are acti-
vated per token, MoE models scale total parameters without
proportionally increasing per-token compute cost.
Token embedding layer
RMSNorm 1
Attention
Linear output
Final RMS Norm
RMSNorm 2
MoE
+
+
FFN
top-k routing
FFN
FFN
FFN
E0
E1
E2
E15
…
𝑻𝒐𝒌𝒆𝒏𝟏
…
top-k routing
𝑻𝒐𝒌𝒆𝒏𝒏
…
…
EP group of 4 ranks
Rank 0
E0-E3
Rank 1
E4-E7
Rank 2
E8-E11
Rank 3
E12-E15
TP group of 4 ranks
Rank 0
Attn
Rank 1
Attn
Rank 2
Attn
Rank 3
Attn
×N
Attention
MoE
All-to-All
Figure 1. MoE-based transformer architecture with expert
parallelism. Each layer consists of an attention block fol-
lowed by a MoE block, where the router selects top-k experts
per token. Experts are distributed across GPUs via expert
parallelism, with tokens dispatched through all-to-all com-
munication.
When the number of experts exceeds what a single GPU
can hold, expert parallelism (EP) distributes experts across
GPUs, with each GPU hosting a subset of experts (Figure 1,
lower right). Because each token may be routed to experts
residing on different GPUs, cross-GPU communication such
2


---

as all-to-all is required to deliver tokens to their assigned
experts and return results after computation.
The standard mechanism for this communication is the
all-to-all dispatcher. Figure 2 illustrates its stages. First, a
dispatch preprocessing step permutes tokens by expert as-
signment and expands each token into k copies, producing
a tensor of shape [tokens × 𝑘, 𝐻]. This expanded tensor is
then dispatched across GPUs via an all-to-all collective, so
that each GPU receives the tokens assigned to its local ex-
perts. A postprocessing step reorganizes the received tokens
for efficient expert computation via grouped GEMM. After
expert computation, the reverse path combines results: a
combine preprocessing step prepares outputs, a second all-
to-all returns them to the originating GPUs, and a final un-
permutation restores the original token order with weighted
aggregation.
2.2
Peak Memory Consumption in MoE Training
To understand where peak memory arises in distributed
MoE training, we profile a Qwen3-style MoE model (Ta-
ble 3) with results shown in Figure 3 and 4. Each of the 4
transformer layers consists of a multi-head attention block
followed by a MoE block with top-8 routing. We train with
a micro-batch size of 24 and sequence length of 16384 on 8
H100 GPUs. For the attention portion of each layer, we ap-
ply tensor parallelism with sequence parallelism (TP=SP=2)
and context parallelism (CP=4), which together reduce the
local sequence length to 2048. For the MoE portion, we use
expert parallelism (EP=8) with expert tensor parallelism dis-
abled (ETP=1), following MoE Parallel Folding [23] where
attention and MoE adopt independent parallelism mappings.
Activation recomputation is enabled throughout. This con-
figuration represents a well-optimized baseline: existing at-
tention memory techniques are fully applied, reducing the
MoE input activation tensor on each GPU to 336 MiB1.
Figure 3 shows the memory profile across all 4 layers
during training. Sharp spikes appear repeatedly in both for-
ward and backward passes, reaching up to 51 GB while the
memory between spikes remains around 14 GB. Each spike
corresponds to a MoE block within a transformer layer. Al-
though SP and CP reduce the number of tokens per GPU
entering each MoE block to 𝐵× 𝑆/(SP × CP) = 49,152, the
subsequent top-k expansion multiplies this by 8, producing
the dispatch buffers that cause the spikes. As a result, it is the
MoE blocks, not attention, that determine the out-of-memory
(OOM) boundary for a given training configuration.
Figure 4 zooms into a single transformer layer during both
forward and backward passes. The top-k-expanded dispatch
tensors each occupy approximately 2.6 GiB, an 8× expansion
from the 336 MiB input activation. The all-to-all dispatcher
pipeline produces multiple such activations through token
1Per-GPU activation size = micro-batch × local sequence length × hidden
dimension × 2 bytes (bf16) = 24 × 2048 × 3584 × 2 bytes = 336 MiB.
permutation, dispatch, and combine. These expanded tensors
coexist as dispatch and combine stages allocate new outputs
while retaining their inputs, driving the per-layer peak above
42 GB during backward.
These observations reveal that peak memory in distributed
MoE training is driven by two factors. First, each interme-
diate activation in the dispatch pipeline is proportional to
the number of tokens per GPU, the top-k routing factor, and
the hidden dimension, all of which are growing in frontier
architectures (Table 1). Second, multiple such activations
coexist in memory during execution of the dispatch pipeline.
Together, these factors produce the memory spikes observed
in Figure 3. Addressing this bottleneck requires fundamen-
tally redesigning how communication and computation are
organized in the MoE block.
3
Design
RelayMoE reduces peak GPU memory by computing locally
as data circulates through a ring. We describe ring-based
dispatch and its communication–computation overlap in
Section 3.1, adaptive routing in Section 3.2, memory-efficient
MoE recomputation in Section 3.3, and selective attention
recomputation in Section 3.4.
3.1
Ring-Based MoE Dispatch
RelayMoE circulates data through the expert-parallel group
in a ring instead of gathering all routed tokens in a single step
(Figure 5). Both the experts and the tokens are partitioned
across the EP group: each rank hosts a subset of the experts,
as in standard expert parallelism, and holds the tokens of
its local batch. We illustrate ring execution using expert
routing; Section 3.2 discusses token routing and the choice
between the two modes. At each hop, a GPU computes partial
results using its local tokens and the visiting expert partition,
accumulates them into the output, and forwards the expert
weights to its ring neighbor via peer-to-peer communication.
After 𝑁computation hops across the 𝑁GPUs in the EP
group, every partition has visited every GPU, so each token
has been processed by all of its assigned experts and the
output activation is complete.
Forward execution. To illustrate, consider a forward hop
using expert routing. The GPU uses the routing map to find
which local tokens selected any of the visiting experts. For
matching pairs, it locally permutes the relevant tokens, exe-
cutes the expert GEMM, weights the outputs by the router
probabilities, and adds the results to the running output ac-
tivation. Tokens that did not select any visiting expert are
skipped. Thus, each hop computes only the token–expert
pairs selected by the router, while the expert partition follows
the fixed ring schedule.
Backward execution. With expert routing, each backward
hop reconstructs the required expert intermediates and com-
putes gradients (Section 3.3). Gradients of the MoE outputs,
3


---

local
1. router
top-k
build routing map
and probs
local
2. dispatch_
preprocess
permute + expand
[𝑇!"#$!× k, H]
EP
3. token_
dispatch
A2A across
EP degree
TP
4. dispatch_
postprocess
AG(TP=2) + sort
for local experts
compute
5. local
experts
local experts
on this EP rank
TP
6. combine_
preprocess
unsort +
RS(TP=2)
EP
7. token_
combine
A2A back
across EP=8
local
8. combine_
postprocess
unpermute +
scatter-add
Figure 2. Stages of the all-to-all dispatcher in Megatron-LM. Each stage is annotated with its communication scope (local, EP,
TP, or compute). The dispatch path expands tokens by top-k and redistributes them across EP ranks; the combine path reverses
the process after expert computation. “A2A” and “AG” represents all-to-all and all-gather communication, respectively. Dashed
stages are present only when tensor parallelism (TP>1) is enabled for expert computation.
fwd Attention
fwd MoE
bwd MoE
bwd Attention
Memory peak 
unresolvable 
by CP/SP
Figure 3. Memory profile during training of a Qwen3-style
MoE model. MoE blocks create memory spikes in both for-
ward and backward passes that dominate peak memory.
Figure 4. Memory profile of a single transformer layer. The
MoE dispatch pipeline produces multiple 2.6 GiB intermedi-
ate activations (8× expansion of the base activation due to
top-k routing).
token inputs, and router probabilities remain local. Expert-
weight gradients are summed across ranks in reused com-
munication buffers and returned to their owners before pa-
rameter updates.
Figure 5 illustrates backward execution at EP=4. On rank
0, 𝑇0 remains resident as 𝐸0, 𝐸3, 𝐸2, and 𝐸1 visit at hops 0–3.
The colored stack represents in-place accumulation of the
𝑔𝑗contributions into one token-gradient buffer; Σ𝑗on the
arrows denotes the accumulated weight gradient of 𝐸𝑗. At
hop 0, computation with 𝐸0 produces both 𝑔0 for the local
token gradient and the first contribution to Σ0. Rank 0 sends
only 𝐸0 at hop 0, then Σ0 together with 𝐸3 at hop 1. Weights
return to their owners at hop 3, followed by the final gradient-
only return shown on the right before parameter updates.
RelayMoE is mathematically equivalent to all-to-all dis-
patch. Each token–expert pair computes the same GEMM
with the same weights and the same router probabilities. The
difference is the execution order: all-to-all gathers the pairs
at the destination GPU before computing, while RelayMoE
accumulates partial outputs incrementally across hops.
Communication–computation overlap. Ring-based dis-
patch replaces the all-to-all dispatch pipeline with a sequence
of peer-to-peer hops. Although this avoids the full expanded
dispatch buffers, performing computation and communica-
tion sequentially at every hop would add substantial latency.
Overlapping them is therefore critical to making ring-based
dispatch practical. Forward requires overlap between local
computation and weight transfer; backward additionally re-
quires weight gradient transfer. We describe the schedule
with expert routing below. Token routing follows the same
principle, overlapping transfer of the next token partition
with computation on the current partition.
In the forward pass, while a GPU computes using the
currently visiting expert partition and its local tokens, it
prefetches the next partition’s weights from its neighbor via
asynchronous P2P communication. If this transfer completes
before the current computation finishes, the next partition is
available without a communication stall. With effective over-
lap, the per-hop time approaches the larger of computation
and transfer time, rather than their sum.
During backward, RelayMoE overlaps local recomputation
and gradient computation with merged transfers of expert
weights and accumulated weight gradients. Gradients follow
their corresponding weights one hop later, so each trans-
fer combines weights and gradients from different expert
partitions.
The key observation enabling this overlap is that local
partial weight gradients can be computed without wait-
ing for the incoming accumulated gradient buffer. After lo-
cal gradient computation completes, the GPU synchronizes
with the incoming buffer, adds its local partial gradient in-
place, and asynchronously forwards the updated buffer to
the next GPU. If the incoming buffer has not arrived, the GPU
waits at this synchronization point. With effective overlap,
the per-hop backward time approaches the longer of local
4


---

hop 0
rank 0
rank 1
rank 2
rank 3
T0 E0 g0
T1 E1 g1
T2 E2 g2
T3 E3 g3
E0
E2
E1
E3
hop 1
rank 0
rank 1
rank 2
rank 3
T0 E3
g0
g3
+
T1 E0
g1
g0
+
T2 E1
g2
g1
+
T3 E2
g3
g2
+
E3
Σ0
E1
Σ2
E0
Σ1
E2
Σ3
hop 2
rank 0
rank 1
rank 2
rank 3
T0 E2
g0
g3
+
g2
+
T1 E3
g1
g0
+
g3
+
T2 E0
g2
g1
+
g0
+
T3 E1
g3
g2
+
g1
+
E2
Σ3
E0
Σ1
E3
Σ0
E1
Σ2
hop 3
rank 0
rank 1
rank 2
rank 3
T0 E1
g0
g3
+
g2
+
g1
+
T1 E2
g1
g0
+
g3
+
g2
+
T2 E3
g2
g1
+
g0
+
g3
+
T3 E0
g3
g2
+
g1
+
g0
+
E1
Σ2
E3
Σ0
E2
Σ3
E0
Σ1
Gradient return
rank 3
rank 0
Σ0
rank 0
rank 1
Σ1
rank 1
rank 2
Σ2
rank 2
rank 3
Σ3
Parameter update
Figure 5. Expert routing during backward at EP=4. Tokens 𝑇𝑟stay on rank 𝑟. The colored stack shows contributions 𝑔𝑗from
visiting expert partition 𝐸𝑗to the local token gradient. On the arrows, Σ𝑗denotes the weight gradient of 𝐸𝑗, accumulated
across ranks. Weight gradients follow their corresponding weights one hop later. The first transfer carries only weights; the
final gradient-only transfer returns the completed weight gradients to their owners for parameter updates.
computation and communication, plus synchronization and
in-place accumulation. Both forward and backward expert
computation can adopt optimized GEMM kernels such as
GroupedGEMM. Section 4.5 evaluates the contributions of
overlap and GroupedGEMM separately.
Memory consumption. Each GPU retains its resident data,
output or gradient accumulators, and the current and prefetched
partitions with their communication buffers. Local permuta-
tion and expert intermediates cover only the token–expert
pairs matched at the current hop; their size still depends
on top-𝑘and routing. RelayMoE avoids gathering the full
top-𝑘-expanded activations across partitions or construct-
ing a full expanded buffer during combine. This removes
the coexisting dispatch buffers identified in Section 2, while
preserving the same token–expert computation. Section 3.3
explains how reconstructing and consuming expert interme-
diates within each hop also reduces their lifetime and peak
memory during backward.
Peak memory 
usage reduced by 
13.9 GB
Figure 6. Memory profile under RelayMoE with the same
training configuration as Figure 3. The dashed lines mark
the all-to-all and RelayMoE peak levels, with a reduction of
13.9 GiB. The sharp MoE memory spikes are substantially
reduced.
Figure 6 shows the resulting memory profile under the
same training configuration as Figure 3. The sharp MoE mem-
ory spikes are substantially reduced, and peak memory is
reduced by 13.9 GiB. Figure 7 zooms into one layer’s forward
and backward execution, corresponding to Figure 4. The co-
existing 2.6 GiB dispatch buffers no longer appear; instead,
local expert computation produces smaller memory fluctu-
ations across hops. Backward still uses more memory than
forward, but it no longer exhibits the large dispatch-buffer
spike seen with all-to-all.
forward
backward
Figure 7. Forward and backward GPU memory profile of
one transformer layer under RelayMoE, corresponding to
Figure 4.
This design eliminates the need for token dropping and
capacity-factor padding to bound dispatch buffers. In all-to-
all dispatch, routing imbalance can concentrate tokens on the
GPU that hosts a popular expert. With expert routing, each
GPU instead processes its fixed local tokens as expert parti-
tions visit, regardless of which experts the tokens select. This
balances resident token memory across GPUs, although un-
even token–expert matches can still leave ranks idle within
a hop. For architectures with shared experts such as Qwen2-
MoE, shared experts are computed separately outside the
ring, as they are replicated on every GPU and do not require
dispatch communication.
3.2
Adaptive Routing: Experts vs Tokens
RelayMoE can circulate either expert weights or token acti-
vations through the ring while the other remains resident on
each GPU. Both modes produce identical results (Section 3.1),
but differ in communication volume (Table 2). With token
routing, each token partition carries its routing metadata and
5


---

accumulated outputs back to its original GPU. During back-
ward, the token inputs, MoE output gradients, and accumu-
lated token-input and router-probability gradients circulate
together, while expert-weight gradients accumulate locally.
Both modes reuse communication buffers across hops.
Considering expert parallelism alone, the comparison in-
volves two factors: per-hop communication volume and the
tokens or expert weights kept resident on each GPU (both
modes traverse the same 𝑁GPUs). We use the notation in
Table 2: 𝐻is the hidden dimension, 𝑊is the size of one
expert’s weights, and 𝑀is the routing metadata size per to-
ken. In forward, expert routing communicates 𝐸local ×𝑊per
hop while keeping the local tokens resident. Token routing
communicates the local token partition and its routing meta-
data while keeping 𝐸local experts resident. Expert routing is
preferred when token activations are large relative to expert
weights: the communication volume is smaller and the to-
kens, which would be expensive to move, remain stationary.
Token routing is preferred when expert weights are large
relative to token activations: the communication volume is
smaller and the large expert weights remain stationary.
Table 2 includes gradient communication during back-
ward. For expert routing, it reports the merged payload of
expert weights and accumulated weight gradients described
in Section 3.1.
Table 2. Per-hop communication volume and resident mem-
ory for expert routing vs token routing. 𝐸local: experts per
GPU, 𝑊: single expert weight size, 𝐵: micro-batch, 𝑆: se-
quence length, 𝐻: hidden dimension, 𝑀: per-token routing
metadata size.
Routing Expert
Routing Token
Basic (EP only, N = EP hops)
Resident
Tokens: 𝐵× 𝑆× 𝐻
Experts: 𝐸local ×𝑊
Comm. (fwd)
𝐸local ×𝑊
𝐵× 𝑆× (𝐻+ 𝑀)
Comm. (bwd)
2 × 𝐸local ×𝑊
2 × 𝐵× 𝑆× (𝐻+ 𝑀)
With SP, CP, and ETP
Resident
𝐵×𝑆
SP×CP × 𝐻
𝐸
EP × 𝑊
ETP
Comm. (fwd)
𝐸
EP × 𝑊
ETP
𝐵×𝑆
SP×CP × (𝐻+ 𝑀)
Comm. (bwd)
2 × 𝐸
EP × 𝑊
ETP
2 ×
𝐵×𝑆
SP×CP × (𝐻+ 𝑀)
In distributed MoE training, EP typically works along-
side other parallelism dimensions that further adjust both
resident memory and communication volume. Sequence par-
allelism (SP) and context parallelism (CP) partition the se-
quence dimension, reducing tokens per GPU to 𝐵× 𝑆/(SP ×
CP). Expert tensor parallelism (ETP) splits each expert’s
weight matrices across GPUs, reducing per-expert weight
size to 𝑊/ETP. MoE Parallel Folding [23] allows ETP to be
configured independently of attention-layer TP. Both adjust-
ments apply to both modes, as shown in the second section
of Table 2. Because these adjustments can shift the balance in
either direction, the preferred routing mode depends on the
full training configuration rather than the model architecture
alone. For example, a model with many experts may favor
expert routing under high ETP (which shrinks expert vol-
ume) but token routing under low ETP. RelayMoE evaluates
both volumes at training start using the full configuration
and selects the mode with lower per-hop communication.
As shown in Table 1, frontier MoE models have grown
from 8 experts (Mixtral) to several hundred (Qwen3.5, Kimi
K2). At a fixed EP degree, this increases the number of ex-
perts per GPU 𝐸/𝐸𝑃. For a fixed expert size, more experts
increase the weight volume transferred at each hop under
expert routing. Token routing avoids transferring these ex-
pert weights, making it an attractive option when expert
partitions are large relative to token activations. The choice
still depends on expert width and token count.
The physical topology of the ring also matters. When EP
spans multiple nodes, cross-node hops use inter-node links
(e.g., RDMA) with substantially lower bandwidth than intra-
node NVLink. All-to-all dispatchers address this through
hierarchical algorithms and optimized kernels such as Tutel
and DeepEP that pipeline cross-node transfers. RelayMoE’s
ring structure instead serializes communication into P2P
hops, which creates opportunities to overlap each hop’s trans-
fer with local computation (Section 3.1), but requires careful
scheduling to avoid underutilizing bandwidth on the fast
intra-node hops. Its memory benefit comes from avoiding
the full expanded dispatch buffers, while its latency depends
on the balance between local computation and transfers over
the available links.
3.3
Memory-Efficient MoE Recomputation
Figure 8 compares MoE recomputation with all-to-all and Re-
layMoE. In the upper timeline, activation checkpointing [2]
retains the inputs needed to replay the MoE layer during
backward. Reconstructed activations remain live while gradi-
ent computation produces additional temporary tensors. The
memory profile therefore rises during recomputation and
again during gradient computation, producing a backward
peak higher than the forward peak.
The ring structure naturally supports recomputation within
each backward hop, as shown in the lower timeline. With ex-
pert routing, RelayMoE uses the saved local tokens and rout-
ing information together with the visiting expert partition
to reconstruct the required intermediates (R). It immediately
computes and accumulates the corresponding gradients, then
releases the reconstructed intermediates before the next hop.
Gradient computation still raises memory within a hop, but
expert intermediates do not accumulate across hops, so their
peak memory is determined by the largest hop. Saved in-
puts and gradient accumulators remain live across hops;
weight and gradient communication buffers are reused. This
schedule reduces the memory needed for expert computation
6


---

Forward
Backward
Recompute
Gradient compute
Compute
GEMMs
GEMMs
bwd GEMMs
Memory
Reconstruct
Consume
and release
Comm.
A2A
A2A
A2A
A2A
A2A
A2A
All-to-All
Forward
Backward
Compute
fwd
fwd
fwd
fwd
R
bwd
R
bwd
R
bwd
R
bwd
Reconstruct → consume → release
Memory
Comm.
h0
h1
h2
h3
h0
h1
h2
h3
Time saved
Relay-MoE
Figure 8. MoE recomputation with all-to-all and RelayMoE
(EP=4). R denotes recomputation; h0–h3 denote ring hops.
Purple profiles show only intermediate activations and tem-
porary gradients. The final gradient-only transfer is not
shown.
during backward itself, beyond the savings from discarding
forward activations.
Recomputation also reuses the data movement already
required by backward. The visiting weights and local tokens
are available for both reconstruction and gradient compu-
tation, so reconstructing expert intermediates requires no
additional ring traversal. Token routing follows the same
principle with visiting tokens and resident experts. The lo-
cal recomputation participates in the overlap described in
Section 3.1; it still adds computation, and its runtime cost
depends on the balance between computation and commu-
nication.
3.4
Selective Attention Recomputation
The memory saved by RelayMoE can also reduce recomputa-
tion outside the MoE layer. We retain the MoE recomputation
schedule above and use the available memory to selectively
retain attention activations. These activations can then be
reused during backward, reducing attention recomputation
and improving training throughput.
For each attention layer 𝑖, recomputation option 𝑗has
a memory cost 𝑚𝑖,𝑗for retained activations and a compute
saving𝑐𝑖,𝑗from avoiding the corresponding backward recom-
putation. Both are estimated from the model configuration
and parallelism settings. For grouped-query attention with
FlashAttention, the activation sizes depend on the micro-
batch size, local sequence length, number of attention heads,
and head dimension. The compute saving also depends on
sequence length and the attention mechanism used by each
block.
RelayMoE uses the difference between GPU capacity and
peak memory under full recomputation as its memory bud-
get. It ranks the available options by their compute saving
per unit of memory 𝑐𝑖,𝑗/𝑚𝑖,𝑗and greedily selects options
within this budget. The selection is computed once at the
start of training and applied throughout.
4
Evaluation
4.1
Experimental Setup
Platform. All experiments were run on MareNostrum 5 in
the Accelerated (ACC) partition. Each node contains four
NVIDIA H100 GPUs (64 GB HBM2e per GPU) and two Intel
Xeon Platinum 8460Y+ CPUs (40 cores each). GPUs within
a node are interconnected via NVLink 4.0, and nodes are
connected through an NDR InfiniBand network with a fat-
tree topology [1].
Software and baselines. We implement RelayMoE on top
of Megatron-LM [37] in 2.3K lines of code. For the MoE layer
experiments, we compare RelayMoE against FasterMoE [13],
Tutel [14], and Megatron-LM’s all-to-all dispatcher, varying
token counts and expert parallelism degrees. The All-to-
All baseline uses the default MoE dispatcher from a recent
version of Megatron-LM (May 2026). For end-to-end training,
we use the same All-to-All implementation as our primary
baseline. We also enable DeepEP [45] within Megatron-LM,
which optimizes token dispatch and combine while retaining
the same all-to-all formulation.
Models. We evaluate RelayMoE using Qwen3-style configu-
rations that vary the number of experts (64–384), the routing
degree (top-4 to top-16), and the expert FFN width (512–2048).
We adjust the number of layers to keep the total trainable
parameter count comparable across settings, as shown in Ta-
ble 3. Single-layer experiments isolate MoE execution, while
full-model results also reflect the different layer counts. Un-
less stated otherwise, these experiments use 2 nodes (8 GPUs)
with TP=4, CP=2, PP=1, and EP=8.
We also evaluate production-model training with Qwen2-
57B-A14B, Qwen3-30B-A3B, and Qwen3.5-35B-A3B on 8
nodes (32 GPUs). We enable NVIDIA’s MoE Parallel Fold-
ing [23], a state-of-the-art parallelization strategy that allows
attention and MoE layers to use independent configurations.
We use TP=2, CP=4, PP=4, and EP=8 to evaluate the benefits
of RelayMoE on top of these parallelism optimizations.
Metrics. The MoE layer experiments report forward-and-
backward latency, including communication. Average speedups
are arithmetic means over distinct test configurations. For
full-model experiments, we report model throughput in TFLOP/s/GPU,
excluding recomputation FLOPs but including recomputa-
tion time. For a given model and sequence length, higher
throughput means more training tokens processed per sec-
ond.
4.2
MoE Layer Performance
To understand the performance impact of MoE execution,
we compare the forward-and-backward latency of Relay-
MoE with baselines. All comparisons use the same model,
parallelism, batch size, sequence length, and recomputation
settings.
Latency comparison. We study MoE execution latency
using a single SwiGLU MoE layer from each configuration
7


---

40K 65K 98K
0
500
1000
OOM
E64
40K 65K 98K
0
500
1000
1500
OOM
E128
40K 65K 98K
0
1000
2000
OOM
E384
40K 65K 98K
0
250
500
750
K4
40K 65K 98K
0
500
1000
1500
OOM
K8
40K 65K 98K
0
1000
2000
OOM
OOM
K12
40K 65K 98K
0
1000
2000
OOM
OOM
OOM
K16
40K 65K 98K
0
500
1000
OOM
H512
40K 65K 98K
0
500
1000
1500
OOM
H768
40K 65K 98K
0
500
1000
1500
OOM
H2048
Sequence Length
Latency (ms)
All-to-All
RelayMoE
Tutel
FasterMoE
Figure 9. Latency of a single MoE layer at different sequence lengths, with EP=8 and micro-batch size 12.
2k
E64
8
16
32
64
0
500
1k
2k
4k
E128
8
16
32
64
0
500
1k
5k
E384
8
16
32
64
0
500
1k
1k
2k
K4
8
16
32
64
0
200
400
2k
4k
K8
8
16
32
64
0
500
1k
5k
K12
8
16
32
64
EP
0
1.7k
5k
K16
8
16
32
64
EP
0
1.5k
2.5k
H512
8
16
32
64
EP
0
1k
5k
H768
8
16
32
64
EP
0
1k
5k
H2048
8
16
32
64
EP
0
1.2k
Latency (ms)
All-to-All
RelayMoE
Tutel
FasterMoE
Figure 10. Forward-and-backward latency of a single MoE layer as EP increases from 8 to 64, with micro-batch size 12 and
sequence length 8192 fixed per GPU. Missing Tutel points indicate OOM.
in Table 3. Figure 9 shows results at sequence lengths of 40K,
65K, and 98K. All systems use EP=8 on 8 GPUs, batch size
𝑏= 12, and the same recomputation configuration.
RelayMoE outperforms all-to-all by 2× on average. For
E128, which matches the expert configuration of Qwen3-30B-
A3B, the speedup increases from 1.77× at 40K to 2.4× at 98K.
Expert routing keeps tokens local and reduces communica-
tion volume at long sequence lengths. Longer sequences and
larger micro-batches also increase the computation within
each hop, helping amortize transfers and kernel launches.
For E384 and H2048, RelayMoE is slower than all-to-all at
40K but becomes faster at 65K, reaching speedups of 1.51×
and 1.56×, respectively, at 98K. Their larger expert counts or
wider FFNs increase the cost of transferring expert weights.
As more tokens are processed, this weight-transfer volume
stays fixed while all-to-all communication grows. The re-
sulting crossover shows that the benefit of expert routing
depends on the amount of token computation available to
amortize weight transfers. Section 4.5 evaluates token rout-
ing for configurations with larger expert-weight volumes.
Compared to FasterMoE, RelayMoE is 4.03× faster on av-
erage, with speedups ranging from 1.41 to 6.20×. FasterMoE
uses fine-grained scheduling to overlap token exchange with
expert computation. In RelayMoE, expert routing keeps the
transfer volume fixed as sequence length increases, while
more tokens provide more local computation to overlap with
8


---

40K
64K
98K
131K
393K
786K
50
100
150
200
TFLOP/s/GPU
OOM
Qwen3-30B-A3B
40K
64K
98K
131K
393K
786K
50
100
150
OOM
Qwen3.5-35B-A3B
384K
50
100
150
OOM
Qwen2-57B-A14B
Sequence Length
All-to-all
RelayMoE
Figure 11. End-to-end training throughput on Qwen3-30B-A3B, Qwen3.5-35B-A3B, and Qwen2-57B-A14B. Each dispatcher
uses its largest feasible micro-batch.
40K
65K
98K
0
150
300
OOM
OOM
E64
40K
65K
98K
0
150
300
E128
40K
65K
98K
0
150
300
E384
40K
65K
98K
0
150
300
K4
40K
65K
98K
0
150
300
OOM
K12
40K
65K
98K
0
150
300
OOM
K16
40K
65K
98K
0
150
300
OOM
H512
40K
65K
98K
0
150
300
H2048
Sequence Length
TFLOP/s/GPU
DeepEP
RelayMoE
Figure 12. End-to-end training throughput with RelayMoE and DeepEP at EP=4. Where DeepEP fits, both use the same number
of recomputed attention blocks, while RelayMoE uses its largest feasible micro-batch.
Table 3. Model configurations used in evaluation. 𝐿: layers,
𝐸: experts, 𝐾: top-k. Scaling study configs use Qwen3-style
architecture (SwiGLU, RMSNorm, RoPE) with PP=1. 𝐿varies
to keep the number of trainable parameters similar per GPU.
Name
L
Hidden
FFNmoe
E
K
Params
Production architectures (end-to-end evaluation)
Qwen3-30B-A3B
48
2048
768
128
8
30B
Qwen3.5-35B-A3B
40
2048
512
256
8
35B
Qwen2-57B-A14B
28
3584
2560
64
8
57B
Varying expert count
MoE-E64
24
2048
768
64
8
7.8B
MoE-E128
12
2048
768
128
8
7.6B
MoE-E384
4
2048
768
384
8
7.5B
Varying top-k
MoE-K4
12
2048
768
128
4
7.5B
MoE-K8
12
2048
768
128
8
7.6B
MoE-K12
12
2048
768
128
12
7.6B
MoE-K16
12
2048
768
128
16
7.6B
Varying expert FFN width
MoE-H512
18
2048
512
128
8
7.72B
MoE-H768
12
2048
768
128
8
7.6B
MoE-H2048
5
2048
2048
128
8
8.2B
these transfers. This helps amortize communication over
longer sequences.
Compared to Tutel, RelayMoE is 2.94× faster on average.
Tutel pipelines token dispatch and expert computation, but
the communication volume of its expert-parallel execution
still grows with sequence length and top-𝑘. Tutel encounters
OOM in several configurations at 65K and 98K, where more
routed token assignments require larger dispatch buffers and
more memory for expert intermediate activations.
EP scaling performance. We study whether RelayMoE
and the baselines maintain their execution efficiency when
processing proportionally more tokens at higher EP degrees.
Figure 10 compares latency as EP increases from 8 to 64, with
𝑏= 12 and sequence length 8192 fixed per GPU.
RelayMoE outperforms all-to-all at all tested EP degrees,
with its speedup generally increasing as EP grows. For K16,
the speedup increases from approximately 2.5× at EP=8 to
5× at EP=64. Expert routing reduces communication volume
in these configurations and overlaps expert transfers with
local computation.
In these single-layer experiments, RelayMoE reduces peak
GPU memory during MoE execution by an average of 3.8× at
EP=8 and 7.6× at EP=64, reaching 13.4× for K16. Increasing
EP reduces the expert partition and the intermediate activa-
tions processed at each hop. These savings target the MoE
execution peak identified in Figure 3; Section 4.4 examines
their effect on full-model training, which also retains model
states and activations from other layers.
9


---

Compared to FasterMoE and Tutel, RelayMoE’s latency
advantage generally increases with EP, reaching speedups of
up to 27× and 16×, respectively, at EP=64. RelayMoE reduces
communication volume and overlaps expert transfers with
local computation, helping limit latency growth at larger EP
degrees. Tutel encounters OOM in some configurations due
to buffer memory overhead. FasterMoE and Tutel are also
slower than Megatron-LM all-to-all in all tested cases.
4.3
Model Training Performance
MoE-layer latency captures only part of RelayMoE’s effect on
training: lower MoE memory usage also leaves more space
for attention activations and input tokens. We further quan-
tify end-to-end throughput and trainable sequence lengths.
End-to-end performance on production models. We
evaluate throughput and trainable sequence length on pro-
duction MoE architectures. We compare RelayMoE with all-
to-all on Qwen3-30B-A3B, Qwen3.5-35B-A3B, and Qwen2-
57B-A14B using 32 GPUs with TP=2, CP=4, PP=4, and EP=8.
At each sequence length, we select the largest micro-batch
size for each dispatcher that fits in GPU memory. Figure 11
shows that RelayMoE achieves higher throughput on both
Qwen3-30B and Qwen3.5-35B at every sequence length where
both dispatchers run, with the memory savings allowing
larger micro-batches. The largest tested trainable sequence
length increases from 393K to 786K, a 2× improvement.
These gains hold across different expert granularities: Qwen3.5-
35B doubles the expert count with a narrower expert FFN.
On Qwen2-57B, RelayMoE trains at 384K while all-to-all runs
out of memory, demonstrating the capacity benefit on this
wider-expert model. Attention and the remaining training
work limit how much faster MoE execution improves end-
to-end throughput; the larger feasible micro-batches and
contexts provide an additional benefit from lower memory
usage.
Comparison with DeepEP. We next compare RelayMoE
with DeepEP to evaluate training throughput and capac-
ity with optimized all-to-all communication. We use the
Qwen3-style full models in Table 3, including their different
layer counts, and evaluate sequence lengths 40K, 65K, and
98K. Both dispatchers use EP=4 within one node; inter-node
DeepEP was unavailable with the platform’s NVSHMEM
and NIC configuration. Where DeepEP fits, it uses a fixed
batch size and the smallest number of recomputed atten-
tion blocks that fits in GPU memory. RelayMoE uses the
same number of recomputed attention blocks and its largest
feasible micro-batch. Figure 12 reports the results.
When both dispatchers fit in memory, RelayMoE generally
achieves comparable training throughput while supporting
larger micro-batches. Matching the number of recomputed
attention blocks excludes the reduced-recomputation benefit.
The larger micro-batch does not guarantee higher through-
put: H2048 remains slower for RelayMoE, although the gap
narrows at longer sequences, consistent with the higher
expert-transfer cost of its wide FFNs.
DeepEP runs out of memory on K8, K12, K16, and H512 at
98K, while RelayMoE trains these configurations. DeepEP
optimizes token dispatch and combine while retaining the
all-to-all formulation. Optimized dispatch kernels retain the
routed-token activation buffers, so faster communication
alone does not remove their memory cost [47]. Here, lower
memory usage expands the trainable configurations even
when throughput gains are limited. At 98K, both dispatchers
still run out of memory on E64, whose 24-layer configuration
retains attention and checkpoint state across more layers;
fewer experts per layer therefore do not guarantee a smaller
full-model memory footprint.
4.4
Impact of Memory Savings
The end-to-end comparisons combine changes in MoE exe-
cution with the effects of lower memory usage. We examine
three uses of the saved memory: retaining attention activa-
tions to reduce recomputation, increasing the micro-batch,
and extending the sequence length. The sequence-length
comparison holds the micro-batch and full recomputation
settings fixed, showing whether the capacity benefit persists
without either of the first two adjustments.
Recompute reduction for improved throughput.
We first evaluate whether the memory savings of Relay-
MoE can be translated into higher training throughput by
reducing attention activation recomputation. We use the
Qwen3-style configurations from Table 3 on 8 GPUs and
evaluate three long-context settings with sequence lengths
40K, 65K, and 98K. For these three sequence lengths, we use
micro-batch sizes 12, 8, and 5, respectively, which keeps the
number of tokens per step approximately constant. For each
model configuration and each dispatcher, we calculate the
number of recomputed attention blocks from the available
GPU memory budget using the method in Section 3.4.
Figure 13 shows higher throughput for RelayMoE in most
configurations, with speedups of up to 2.02× on K16 at 40K.
The saved MoE memory allows more attention activations
to be retained for backward, reducing attention recompu-
tation and improving throughput. The near-parity on K4 is
consistent with its lower top-𝑘expansion, which means less
memory can be saved by removing dispatch buffers.
All-to-all remains faster on E384 at all three sequence
lengths; H2048 is also slower at 40K and 65K, but reaches
comparable throughput at 98K. For these configurations, the
memory savings do not consistently offset the execution cost
associated with their larger expert-weight volume. Unlike
the fixed-batch latency comparison in Section 4.2, this study
decreases the micro-batch as sequence length increases, keep-
ing the number of tokens processed per step approximately
constant. Longer sequences therefore do not provide more to-
kens to amortize expert transfers. Section 4.5 examines token
routing as an alternative when moving experts is costly.
10


---

40K
65K
90K
0
100
200
300
OOM
OOM
E64
40K
65K
114K
0
100
200
300
OOM
E128
40K
65K
150K
0
100
200
300
OOM
E384
40K
65K
98K
0
100
200
300
K4
40K
65K
114K
0
100
200
300
OOM
OOM
K12
40K
65K
114K
0
100
200
300
OOM
OOM
K16
40K
65K
98K
0
100
200
300
OOM
H512
40K
65K
114K
0
100
200
300
OOM
H2048
Sequence Length
TFLOP/s/GPU
All-to-All
RelayMoE
Figure 13. Training throughput with reduced attention recomputation. Sequence lengths 40K, 65K, and 98K use micro-batch
sizes 12, 8, and 5, respectively.
E64
E128
E384
K4
K12
K16
H512
H2048
0
75
150
225
TFLOP/s/GPU
b=22
b=26
b=28
b=18
b=28
b=32
b=24
b=30
All-to-All (b=12)
RelayMoE (b=12)
RelayMoE (largest b)
Figure 14. Training throughput at sequence length 40,960
with different micro-batch sizes. Labels above the dark-red
bars give RelayMoE’s largest feasible micro-batch sizes.
Batch scaling for improved throughput. We evaluate
larger micro-batches as another use of the saved memory.
We use the same 8-GPU setup as in the recomputation study
and fix the sequence length to 40,960. Figure 14 reports three
settings for each configuration: all-to-all at micro-batch size
𝑏= 12, RelayMoE at the same micro-batch size, and RelayMoE
at the largest micro-batch size that fits in GPU memory.
At 𝑏= 12, RelayMoE is faster in most configurations under
the selected checkpoint schedules and reduces full-model
peak GPU memory by up to 35.7% relative to all-to-all. The
saved memory allows micro-batches of 18–32, depending on
the model. For H2048, RelayMoE is slower than all-to-all at
𝑏= 12 but becomes faster at 𝑏= 30; E384 also closes most
of its throughput gap at 𝑏= 28. Processing more tokens per
step can amortize expert transfers, although the attention-
checkpoint schedules also change, so these results reflect
their combined effect.
The largest batch is not always the fastest: E64, K12, K16,
H512, and H768 achieve higher throughput with RelayMoE
at 𝑏= 12. Larger batches consume activation memory that
could otherwise retain attention activations, potentially re-
quiring more recomputation. These results show that the
memory savings provide a choice between larger batches
and less recomputation; batch size and checkpointing need
to be considered together to improve throughput.
Sequence scaling for longer context. Finally, we evaluate
whether RelayMoE supports longer sequences when both
dispatchers use the same micro-batch size and full recom-
putation. We use the same 8-GPU setup with 𝑏= 12 and
recompute all attention blocks and MoE activations. In Fig-
ure 15, only K4 trains beyond 65K with all-to-all, whereas
RelayMoE reaches 90K–150K across configurations, extend-
ing the largest tested feasible sequence length by up to 2.85×.
This capacity gain persists with attention recomputation held
fixed: checkpointing discards forward activations, but back-
ward still needs temporary expert intermediates. RelayMoE
avoids the full all-to-all dispatch buffers and reconstructs
these intermediates within each backward hop, leaving more
memory for longer inputs. The tested limits also depend on
layer count, retained attention and checkpoint inputs, and
model state.
4.5
Design Analysis
We examine when each routing mode is effective and how
communication overlap and GroupedGEMM improve ring
execution.
Adaptive routing: token vs expert. To evaluate when
token routing is preferable, we compare token routing, expert
routing, and all-to-all on MoE-E384-K12 and MoE-E384-K16,
each with 384 experts, hidden dimension and expert FFN
width 2048, and 19.53B parameters. All three methods use
the same micro-batch size and recomputation settings at
each sequence length.
Figure 16 shows that token routing outperforms expert
routing on both configurations. On MoE-E384-K12, its speedup
over expert routing decreases from 3.7× at 20K to 2.1× at
40K. Token routing also outperforms all-to-all at 20K and
runs at 40K and 65K, where all-to-all encounters OOM. On
MoE-E384-K16, both ring modes run at all tested sequence
lengths while all-to-all encounters OOM. Token routing
therefore retains the capacity benefit in these wide-expert
11


---

40K
65K
90K
0
100
200
300
OOM
OOM
E64
40K
65K
114K
0
100
200
300
OOM
E128
40K
65K
150K
0
100
200
300
OOM
E384
40K
65K
98K
0
100
200
300
K4
40K
65K
114K
0
100
200
300
OOM
K8
40K
65K
114K
0
100
200
300
OOM
OOM
K12
40K
65K
114K
0
100
200
300
OOM
OOM
K16
40K
65K
98K
0
100
200
300
OOM
H512
40K
65K
124K
0
100
200
300
OOM
H768
40K
65K
114K
0
100
200
300
OOM
H2048
Sequence Length
TFLOP/s/GPU
All-to-All
RelayMoE
Figure 15. Training throughput as sequence length increases, with micro-batch size 𝑏= 12 and full activation recomputation.
20K
40K
65K
0
40
80
120
TFLOP/s/GPU
OOM
OOM
MoE-E384-K12
20K
40K
65K
OOM
OOM
OOM
MoE-E384-K16
Sequence Length
All-to-All
Relay-Expert
Relay-Token
Figure 16. Throughput with token routing, expert routing,
and all-to-all on the two 384-expert configurations.
configurations while avoiding the cost of circulating their
weights.
The narrowing gap between the two ring modes follows
the same volume trade-off as the crossover in Section 4.2:
token communication grows with sequence length, while
expert-weight communication remains fixed. Together, these
results show how token routing complements expert routing
when expert-weight volume is large, motivating the routing
choice in Section 3.2.
0
100
200
300
Latency (ms)
RelayMoE
+ W. & G. Overlap
+ W. & G. Overlap +
GroupedGEMM
295.9 ms
203.4 ms
175.8 ms
Forward
Backward
Figure
17.
Forward
and
backward
latency
with
communication–computation overlap and GroupedGEMM
enabled successively. “W.&G. overlap” enables weight and
gradient transfers to overlap with computation.
Communication overlap and GroupedGEMM. To quan-
tify the contribution of RelayMoE’s optimizations, we mea-
sure forward and backward pass time for a single E128 MoE
layer with EP=8 and fixed token counts. In Figure 17, the
naive ring implementation executes communication and
computation sequentially, without GroupedGEMM, taking
295.9 ms per iteration. With communication–computation
overlap in the second bar, total latency decreases to 203.4 ms:
transfers proceed while computation is ongoing at each hop,
reducing exposed communication time. Adding GroupedGEMM
in both forward and backward further reduces latency to
175.8 ms by processing the current partition’s expert compu-
tations together. The additional reduction shows that improv-
ing local expert computation complements communication
overlap within the ring.
0
200
400
600
800
1000
1200
1400
Training Iteration
5.0
7.5
10.0
Loss
RelayMoE
All-to-All
Figure 18. Training loss of Qwen3-30B-A3B with RelayMoE
and all-to-all, using the same initialization and hyperparam-
eters.
Training correctness. We verify that RelayMoE preserves
training correctness relative to the standard all-to-all dis-
patcher. We train Qwen3-30B-A3B with RelayMoE and all-to-
all under the same training settings, including initialization,
random seed, and hyperparameters. Figure 18 shows that
the two loss curves are indistinguishable over 1500 training
iterations. This is consistent with the mathematical equiv-
alence of the two formulations (Section 3.1): both compute
the same token–expert pairs, with possible floating-point
rounding differences from the accumulation order.
5
Related Work
MoE training. Existing work optimizes communication in
MoE expert parallelism through dispatch libraries [10, 26,
45, 46], communication–computation overlap [5, 14, 17, 20,
27, 36, 43], and network design [22]. Other work optimizes
expert computation and activation memory usage [9, 12, 42],
parallelism and pipeline layouts [18, 23, 44, 48], and exe-
cution on HPC platforms [41]. RelayMoE targets the top-𝑘-
expanded inter-GPU dispatch buffers. Its ring execution can
incorporate optimized expert kernels and communication–
computation overlap, and complements attention parallelism.
To reduce communication overhead, existing work ex-
plores transferring experts to GPUs that host tokens. Faster-
MoE [13] replicates overloaded experts near their tokens to
12


---

alleviate load imbalance. In its data-centric mode, Janus [25]
uses credits to bound GPU expert-fetch buffers and caches
fetched weights in host memory for backward. RelayMoE
targets the peak from expert intermediates during backward,
reconstructing and releasing them within each hop using the
same weight transfers as gradient computation. These mem-
ory savings support longer sequences, larger batches, and
larger top-𝑘values within the same GPU memory budget.
MemFine [47] reduces MoE activation memory usage
through token chunking and chunk-level recomputation,
retaining dispatch–compute–combine for each chunk. Ring-
Expert in RelayMoE likewise limits the lifetime of recon-
structed intermediates, but keeps tokens local and circulates
expert partitions, avoiding inter-GPU token-dispatch buffers
while reusing weight transfers for recomputation and gradi-
ent computation.
Memory-efficient training. Offloading [32, 33], model-
state partitioning [31], and activation checkpointing [2, 19]
have been explored to reduce memory pressure in LLM train-
ing. These techniques reduce GPU memory usage for model
states or trade recomputation for fewer retained activations.
RelayMoE complements them by reducing the memory re-
quired during MoE execution, including the intermediates
reconstructed during backward. The memory saved in MoE
layers can be used to retain more attention activations, re-
ducing attention recomputation and improving throughput
for the same training workload.
Other work reduces the memory cost of attention. FlashAt-
tention [3, 4, 34] reduces attention activation memory usage,
while sequence and context parallel systems distribute at-
tention computation and activations across GPUs [7, 11, 15,
21, 24, 37, 40]. RingAttention also uses a ring, but circulates
KV blocks for attention; RelayMoE addresses MoE dispatch
buffers and expert activations. Reducing attention memory
leaves these MoE costs in place, so the MoE sublayer can
still limit the trainable sequence length. These attention opti-
mizations and RelayMoE can be combined to reduce memory
in both sublayers, supporting longer-context training under
the same hardware budget.
6
Conclusion
We presented RelayMoE, a ring-based MoE execution model
that avoids full top-𝑘-expanded dispatch buffers and reduces
the memory required for expert recomputation during back-
ward. The saved memory supports longer sequences and
larger micro-batches, and can be used to reduce attention
recomputation. Our evaluation demonstrates lower peak
GPU memory usage and improved training throughput over
all-to-all dispatch.
References
[1] Fabio Banchelli, Marta Garcia-Gasulla, Filippo Mantovani, Joan
Vinyals, Josep Pocurull, David Vicente, Beatriz Eguzkitza, Flavio CC
Galeazzo, Mario C Acosta, and Sergi Girona.
Introducing
marenostrum5: A european pre-exascale energy-efficient sys-
tem designed to serve a broad spectrum of scientific work-
loads.
Future Generation Computer Systems, page 108125, 2025.
https://doi.org/10.1016/j.future.2025.108125.
[2] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training
deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174,
2016.
[3] Tri Dao. FlashAttention-2: Faster attention with better parallelism
and work partitioning. In Proceedings of the Twelfth International
Conference on Learning Representations (ICLR 2024), 2024.
[4] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
FlashAttention: Fast and memory-efficient exact attention with IO-
awareness. In Advances in Neural Information Processing Systems 35
(NeurIPS 2022), 2022.
[5] DeepSeek-AI et al. Deepseek-v3 technical report. arXiv preprint
arXiv:2412.19437, 2024.
[6] Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lep-
ikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams W. Yu, Orhan
Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou,
Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson,
Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc
Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling
of language models with Mixture-of-Experts. In Proceedings of the
39th International Conference on Machine Learning, volume 162, pages
5547–5569, jul 2022.
[7] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism
approach for long context generative ai, 2024.
[8] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers:
Scaling to trillion parameter models with simple and efficient sparsity.
Journal of Machine Learning Research, 23(120):1–39, 2022.
[9] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia.
Megablocks: Efficient sparse training with mixture-of-experts. In
Proceedings of Machine Learning and Systems (MLSys), 2023.
[10] Amos Goldman, Nimrod Boker, Maayan Sheraizin, Nimrod Admoni,
Artem Polyakov, Subhadeep Bhattacharya, Fan Yu, Kai Sun, Geor-
gios Theodorakis, Hsin-Chun Yin, Peter-Jan Gootzen, Aamir Shafi,
Assaf Ravid, Salvatore Di Girolamo, James Dinan, Xiaofan Li, Man-
junath Gorentla Venkata, and Gil Bloch. Nccl ep: Towards a unified
expert parallel communication api for nccl, 2026.
[11] Diandian Gu, Peng Sun, Qinghao Hu, Ting Huang, Xun Chen, Yingtong
Xiong, Guoteng Wang, Qiaoling Chen, Shangchun Zhao, Jiarui Fang,
Yonggang Wen, Tianwei Zhang, Xin Jin, and Xuanzhe Liu. Loongtrain:
Efficient training of long-sequence llms with head-context parallelism,
2024.
[12] Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, and Tri Dao.
SonicMoE: Accelerating MoE with IO and tile-aware optimizations.
arXiv preprint arXiv:2512.14080, 2025.
[13] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo,
Shangfeng Shi, and Qin Li. FasterMoE: Modeling and optimizing
training of large-scale dynamic pre-trained models. In Proceedings of
the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel
Programming (PPoPP ’22), Seoul, Republic of Korea, 2022. ACM.
[14] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu,
Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, HoYuen Chau,
Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. Tutel: Adap-
tive mixture-of-experts at scale. In D. Song, M. Carbin, and T. Chen,
editors, Proceedings of Machine Learning and Systems, volume 5, pages
269–287. Curan, 2023.
[15] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang,
Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deep-
Speed Ulysses: System optimizations for enabling training of extreme
long sequence transformer models. arXiv preprint arXiv:2309.14509,
2023.
13


---

[16] Albert Q Jiang, Alexandre Sablayrolles, et al. Mixtral of experts. arXiv
preprint arXiv:2401.04088, 2024.
[17] Chenyu Jiang, Ye Tian, Zhen Jia, Chuan Wu, Yida Wang, and Shuai
Zheng. Lancet: Accelerating mixture-of-experts training via whole
graph computation-communication overlapping. In Proceedings of
Machine Learning and Systems (MLSys), 2024.
[18] Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li,
Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma,
Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin
Jin, and Xin Liu. MegaScale-MoE: Large-scale communication-efficient
training of mixture-of-experts models in production. arXiv preprint
arXiv:2505.11432, 2026. To appear in EuroSys 2026.
[19] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee,
Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Re-
ducing activation recomputation in large transformer models. In
Proceedings of Machine Learning and Systems, volume 5, jun 2023.
[20] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. Accelerat-
ing distributed MoE training and inference with Lina. In 2023 USENIX
Annual Technical Conference (USENIX ATC 23), pages 945–959, Boston,
MA, 2023. USENIX Association.
[21] Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang
You. Sequence parallelism: Long sequence training from system per-
spective. In Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 2391–2404,
Toronto, Canada, 2023. Association for Computational Linguistics.
[22] Xudong Liao, Yijun Sun, Han Tian, Xinchen Wan, Yilun Jin, Zilong
Wang, Zhenghang Ren, Xinyang Huang, Wenxue Li, Kin Fai Tse,
Zhizhen Zhong, Guyue Liu, Ying Zhang, Xiaofeng Ye, Yiming Zhang,
and Kai Chen. MixNet: A runtime reconfigurable optical-electrical
fabric for distributed mixture-of-experts training. In Proceedings of
the ACM SIGCOMM 2025 Conference (SIGCOMM ’25), pages 554–574,
Coimbra, Portugal, 2025. ACM.
[23] Dennis Liu, Zijie Yan, Xin Yao, Tong Liu, Vijay Korthikanti, Evan
Wu, Shiqing Fan, Gao Deng, Hongxiao Bai, Jianbin Chang, Ashwath
Aithal, Michael Andersch, Mohammad Shoeybi, Jiajie Yao, Chandler
Zhou, David Wu, Xipeng Li, and June Yang. MoE parallel folding:
Heterogeneous parallelism mappings for efficient large-scale MoE
model training with Megatron Core. arXiv preprint arXiv:2504.14960,
2025.
[24] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with block-
wise transformers for near-infinite context.
In Proceedings of the
Twelfth International Conference on Learning Representations (ICLR
2024), 2024.
[25] Juncai Liu, Jessie Hui Wang, and Yimin Jiang. Janus: A unified dis-
tributed training framework for sparse mixture-of-experts models. In
Proceedings of the ACM SIGCOMM 2023 Conference, pages 486–498,
2023.
[26] Ziming Mao, Yihan Zhang, Chihan Cui, Zhen Huang, Kaichao You,
Zhongjie Chen, Zhiying Xu, Zhenyu Gu, Scott Shenker, Costin Raiciu,
Yang Zhou, and Ion Stoica. UCCL-EP: Portable expert-parallel com-
munication. arXiv preprint arXiv:2512.19849, 2025.
[27] Xinglin Pan, Wenxiang Lin, Lin Zhang, Shaohuai Shi, Zhenheng Tang,
Rui Wang, Bo Li, and Xiaowen Chu. Fsmoe: A flexible and scalable
training system for sparse mixture-of-experts models. In Proceedings
of the 30th ACM International Conference on Architectural Support for
Programming Languages and Operating Systems (ASPLOS), 2025.
[28] Qwen
Team.
Qwen3-next:
Towards
ultimate
train-
ing
&
inference
efficiency.
https://qwen.ai/blog?id=
4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-
advancements-list, September 2025.
[29] Qwen Team. Qwen3.5: Towards native multimodal agents. https:
//qwen.ai/blog?id=qwen3.5, February 2026.
[30] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang,
Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yux-
iong He. DeepSpeed-MoE: Advancing Mixture-of-Experts inference
and training to power next-generation AI scale. In Proceedings of the
39th International Conference on Machine Learning, volume 162, pages
18332–18346, jul 2022.
[31] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He.
Zero: memory optimizations toward training trillion parameter models.
In Proceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis, SC ’20. IEEE Press, 2020.
[32] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith,
and Yuxiong He. ZeRO-Infinity: Breaking the gpu memory wall for
extreme scale deep learning. arXiv preprint arXiv:2104.07857, 2021.
[33] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji
Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He.
ZeRO-Offload: Democratizing billion-scale model training. In 2021
USENIX Annual Technical Conference (USENIX ATC 21), pages 551–564.
USENIX Association, 2021.
[34] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep
Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention
with asynchrony and low-precision. In Advances in Neural Information
Processing Systems 37 (NeurIPS 2024), 2024.
[35] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis,
Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural
networks: The sparsely-gated Mixture-of-Experts layer. In Interna-
tional Conference on Learning Representations, apr 2017.
[36] Shaohuai Shi, Xinglin Pan, Qiang Wang, Chengjian Liu, Xiaozhe Ren,
Zhongzhe Hu, Yu Yang, Bo Li, and Xiaowen Chu. Schemoe: An ex-
tensible mixture-of-experts distributed training system with tasks
scheduling. In Proceedings of the Nineteenth European Conference on
Computer Systems (EuroSys), 2024.
[37] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-
billion parameter language models using model parallelism. arXiv
preprint arXiv:1909.08053, 2019.
[38] Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo
Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Rui-
jue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen,
Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du,
Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenx-
iao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi
Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang
Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He,
Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang,
Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yong-
sheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li,
Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li,
Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu,
Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y.
Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu,
Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo,
Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin
Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen
Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su,
Zhengyuan Su, Lin Sui, Xinjie Sun, Flood Sung, Yunpeng Tai, Heyi
Tang, Jiawen Tao, Qifeng Teng, Chaoran Tian, Chensi Wang, Dinglu
Wang, Feng Wang, Hailong Wang, Haiming Wang, Jianzhou Wang,
Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Si Wang,
Xinyuan Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang,
Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhengtao Wang, Zhexu
Wang, Chu Wei, Qianqian Wei, Haoning Wu, Wenhao Wu, Xingzhe
Wu, Yuxin Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Weimin Xiong,
Boyu Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran
Xu, Yangchuan Xu, Ziyao Xu, Jing Xu, Jing Xu, Junjie Yan, Yuzi Yan,
14


---

Hao Yang, Xiaofei Yang, Yi Yang, Ying Yang, Zhen Yang, Zhilin Yang,
Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye,
Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie
Yuan, Siyu Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu
Zhang, Xiaobin Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang,
Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang,
Zheng Zhang, Haotian Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng,
Shaojie Zheng, Longguang Zhong, Jianren Zhou, Xinyu Zhou, Zaida
Zhou, Jinguo Zhu, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi
k2: Open agentic intelligence, 2026.
[39] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie
Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran
Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai
Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng
Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize
Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang
Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu,
Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu.
Qwen3 technical report, 2025.
[40] Junqi Yin, Mijanur Palash, Mallikarjun Shankar, and Feiyi Wang.
RingX: Scalable parallel attention for long-context learning on HPC.
In Proceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis (SC ’25). ACM, 2025.
[41] Yueming Yuan, Ahan Gupta, Jianping Li, Sajal Dash, Feiyi Wang,
and Minjia Zhang. X-MoE: Enabling scalable training for emerging
mixture-of-experts architectures on HPC platforms. In Proceedings
of the International Conference for High Performance Computing, Net-
working, Storage and Analysis (SC ’25). ACM, 2025.
[42] Jiyuan Zhang, Yining Liu, Siqi Yan, Lisen Deng, Jennifer Cao, Shuqi
Yang, Min Ni, Bi Xue, and Shen Li. MoEBlaze: Breaking the mem-
ory wall for efficient MoE training on modern GPUs. arXiv preprint
arXiv:2601.05296, 2026.
[43] Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei
Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen
Chang, Quan Chen, and Xin Liu. Comet: Fine-grained computation-
communication overlapping for mixture-of-experts. In Proceedings of
Machine Learning and Systems (MLSys), 2025.
[44] Zheng Zhang, Donglin Yang, Yaqi Xia, Liang Ding, Dacheng Tao,
Xiaobo Zhou, and Dazhao Cheng. MPipeMoE: Memory efficient MoE
for pre-trained models with adaptive pipeline parallelism. In 2023 IEEE
International Parallel and Distributed Processing Symposium (IPDPS).
IEEE, 2023.
[45] Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean
Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. DeepEP: An
efficient expert-parallel communication library. https://github.com/
deepseek-ai/DeepEP, 2025.
[46] Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean
Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. HybridEP: Hybrid
intra-node NVLink and inter-node RDMA expert parallelism. https:
//github.com/deepseek-ai/DeepEP/tree/hybrid-ep, 2025.
[47] Lu Zhao, Rong Shi, Shaoqing Zhang, et al.
Memfine: Memory-
aware fine-grained scheduling for moe training, 2025. arXiv preprint
arXiv:2511.21431.
[48] Guichao Zhu, Lintian Lei, Yuhao Qing, Yichao Fu, Fanxin Li, Dong
Huang, Zekai Sun, and Heming Cui. FoldMoE: Efficient long sequence
MoE training via attention-MoE pipelining. In Proceedings of the
63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), Vienna, Austria, 2025. Association for Com-
putational Linguistics.
15
