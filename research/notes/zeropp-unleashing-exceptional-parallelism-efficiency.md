---
title: 'ZeroPP: Unleashing Exceptional Parallelism Efficiency'
id: zeropp-unleashing-exceptional-parallelism-efficiency
tags:
- cmp170hx-training-feasibility-a0d4fc
- pipeline-parallel
- zero
- fsdp
- tp-free
- communication-volume
- sub-q6
created: '2026-10-27T10:14:24.665412Z'
updated: '2026-10-27T19:00:31.181996Z'
source: https://arxiv.org/abs/2402.03791
source_domain: arxiv.org
fetched_at: '2026-10-27T10:14:24.664565Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'ZeroPP (Shanghai AI Lab / Tsinghua): a TP-free framework that combines pipeline
  parallelism with ZeRO-3/FSDP, adding a ZeRO-compatible breadth-first PP schedule,
  decoupled gradient computation for a near-zero pipeline bubble, and recomputation.
  Theory: TP costs 8Bbsh communication volume per transformer block (four all-reduces
  across forward and backward); ZeroPP costs 36Bh^2/U per iteration (two all-gathers
  plus one reduce-scatter per scheduling unit, each three times the ~12h^2 parameter
  size), so ZeroPP is cheaper whenever 2sUb > 9h - a condition easily met at long
  sequence lengths and large batches. Microbenchmark on 8 GPUs with a 6.2B GPT: ZeRO-3
  communication volume and iteration time beat TP at every global batch size, with
  the gap widening as batch grows. End-to-end: up to 64 A800-80GB GPUs across 8 InfiniBand
  nodes; up to 33% faster than Megatron 3D parallelism at comparable memory; a 14.6B
  model on 32 GPUs with PP=4 and global batch 192 shows the advantage growing as the
  memory limit tightens. Directly supports ''drop TP, use PP+FSDP'' on a low-bandwidth
  fabric.'
raw_file: raw/zeropp-unleashing-exceptional-parallelism-efficiency.pdf
doi: arXiv:2402.03791
utility_score: 12.0
citation_count: 3
---

ZeroPP: Unleashing Exceptional Parallelism Efficiency
through Tensor-Parallelism-Free Methodology
Ding Tang, Lijuan Jiang∗, Jiecheng Zhou, Minxi Jin
Hengjie Li, Xingcheng Zhang, Zhilin Pei
Shanghai Artificial Intelligence Laboratory
{tangding,jianglijuan,zhoujiecheng,jinminxi,lihengjie,zhangxingcheng,peizhilin}@pjlab.org.cn
Jidong Zhai
Tsinghua University
zhaijidong@tsinghua.edu.cn
Abstract
Large-scale models rely heavily on 3D parallelism for distributed training, which
utilizes tensor parallelism (TP) as the intra-operator parallelism to partition model
states across GPUs. However, TP introduces significant communication overheads
and complexity in modifying single-GPU code. In this paper, we propose a TP-free
distributed framework ZeroPP, which leverages the hybrid of scalable inter-operator
pipeline parallelism and intra-operator fully sharded data parallelism to train mod-
els at scale, reducing memory consumption and enabling high training efficiency.
Through extensive experimentation, we demonstrate that ZeroPP achieves signif-
icant performance gains of up to 33% compared to conventional 3D parallelism
while maintaining comparable GPU memory consumption.
1
Introduction
Large-scale fundamental models such as BERT [9], GPT [3], LLaMA [23] are flourishing in AI
areas by utilizing massive datasets and stacked transformer-based architectures. Striking performance
gains have been demonstrated in various areas such as NLP [9, 3, 23], CV [6], etc, with giant model
sizes. In recent years, tens to hundreds of billions of model parameters gradually become common
in large-scale network training, and the model scale still grows rapidly [7]. Inevitably, efficient
distributed parallel parallelism becomes the crucial fundamental support for large-scale network
development, due to the enormous training time and the explosive growth of model sizes far beyond
the memory capacity of a single accelerator.
With the increasing demand for distributed training brought about by larger model sizes, various
parallel strategies have been proposed, such as data parallelism (DP) [5], pipeline parallelism (PP) [8,
15], and tensor parallelism (TP) [21]. In addition, concerning memory limitations and training
efficiency, hybrid parallel strategies are further utilized to support the training of large-scale models
with parameter counts ranging from tens of billions to trillions on hundreds to thousands of GPUs [22,
28]. In particular, as the de facto large-scale distributed method, 3D parallelism [14] has been widely
used, which combines DP, TP, and PP to achieve high training efficiency by weighing communication,
computation, and memory usage.
∗Corresponding author.
Preprint. Under review.
arXiv:2402.03791v3  [cs.DC]  24 May 2024


---

TP as an intra-operator parallelism is leveraged by 3D parallelism to distribute model states across
GPUs [13]. However, TP necessitates heavy collective communication for forward and backward
computation since the weight tensors are partitioned along the reduction dimension. Therefore, TP is
typically performed across GPUs within a node with high bandwidth [14]. Moreover, implementing
TP requires intricate modification to the single GPU code and poses a substantial burden on data
scientists and AI practitioners, hindering its widespread adoption and usability.
ZeRO is another widely used intra-operator parallelism method, a refined variant of data parallelism
that does not necessitate modifications to the model implementations [20]. ZeRO typically includes
multiple stages. ZeRO stage 1 (ZeRO-1) solely partitions the optimizer states, exhibiting consistent
communication volume compared to vanilla DP. ZeRO stage 3 (ZeRO-3) partitions the optimizer
states, gradients, and parameters, and requires extra parameter gathering before each forward and
backward computation. But ZeRO-3 remains more communication-efficient than TP. We demonstrate
that through a simple experiment in Figure 1, and provide a more detailed theoretical analysis in
Section 4.
0
0.5
1
1.5
2
2.5
3
0
20
40
60
80
100
120
140
32
48
64
Iteration Time(s)
Communication Size (GB)
Global Batch Size
ZeRO-3 communication size
TP communication size
ZeRO-3 iteration time
TP iteration time
Figure 1: Communication efficiency comparison between TP and ZeRO-3. We test the 6.2B GPT
model parallelized with the two methods, respectively, across 8 GPUs with varying global batch sizes,
and estimate the per-GPU communication volume for each method. Consequently, ZeRO-3 gains
superior communication efficiency, and the performance gap becomes more pronounced as the global
batch size increases.
Considering that ZeRO-3 partitions the full set of model states and achieves superior communication
efficiency as mentioned above, the hybrid parallelization of ZeRO-3 and PP is expected to outperform
the hybrid parallelization of TP and PP. However, the combination of ZeRO-3 and PP as a hybrid
parallelism method entails significant repetitive parameter gathering and gradient synchronization due
to frequent micro-batch computation switches in PP. The high-frequency communication results in
poor efficiency. Lamy-Poirier et al. [11] propose a Breadth-first PP (BF-PP) method which combines
ZeRO-3 efficiently but only targets scenarios with small batch sizes. More specifically, the GPU
memory requirements of BF-PP increase infinitely with the growth of batch sizes and model scales.
The drawback in memory consumption of BF-PP makes it unpractical in large-scale training since
most models are trained at large batch sizes and model scales currently. Besides, as with most PP
methods, BF-PP suffers from pipeline bubbles which hurts the training efficiency to a great extent.
In this paper, we propose a TP-free hybrid parallelism method ZeroPP, which combines a scalable
PP method and ZeRO-3 to achieve balanced computation, communication, and memory utilization.
By employing a blockwise task-interleaved PP schedule, we reduce parameter gathering during
computation and communication between PP and ZeRO-3. Memory cost is kept low through
unit scheduling, and an efficient recomputation scheme is designed to optimize activation memory
usage further. In addition, ZeroPP utilizes a near-zero bubble PP schedule by leveraging gradient
computation decoupling. Our contributions are as follows:
• We present an efficient PP method that seamlessly integrates with ZeRO-3, ensuring efficient
computation, communication, and memory utilization. By carefully decoupling gradient
2


---

computations and reusing memory through unit scheduling, our approach reduces the
memory footprint and promotes the utilization of computational resources outstandingly.
• We propose a hybrid parallel framework ZeroPP, which combines the newly proposed PP,
ZeRO-3, and vanilla DP and is freed from the sophisticated and heavy communication of TP.
ZeroPP leverages the hybrid of scalable inter-operator pipeline parallelism and intra-operator
fully sharded data parallelism to train models at scale, reducing memory consumption and
enabling high training efficiency.
• We involve an effective recomputation method for ZeroPP, which further reduces memory
consumption while introducing a marginal increase in computational overhead.
• We conduct a series of experiments on transformer models of various sizes. The results
demonstrate that ZeroPP achieves significant performance gains of up to 33% compared to
conventional 3D parallelism while maintaining comparable GPU memory consumption.
2
Related Works
2.1
Data, Model and 3D parallelism
DP [5] is popular to accelerate neural network training, where each model replica works independently
but synchronizes gradients through all-reduce to update the parameters. However, state-of-the-art
large-scale networks rely on vast model sizes and training data, which may make DP alone unusable
due to memory constraints.
Model parallelism is introduced to mitigate memory limitations. TP and PP are two typical model
parallelism methods. TP [21, 26, 29, 25, 1] splits parameters into shards along a specific dimension
and each device only holds a shard of each parameter while not affecting the correctness of the
computation graph. However, since heavy collective communication overhead is required when the
hidden dimension is sharded, TP is often restricted to a single node. PP [8, 15, 14, 11] partitions
the model into stages, each of which is composed of contiguous network layers and mapped onto a
device. PP is widely studied and employed on account of the lightweight communication overhead.
3D parallelism [21], integrating DP, PP, and TP, emerges as a compelling solution that attains
commendable throughput and scalability. Notably, the technique has proven its efficacy in training
diverse, expansive large language models [2, 22]. Nevertheless, the popular 3D parallelism faces
substantial obstacles brought by TP as discussed above, i.e. huge complexity to refactor code and a
considerable amount of communication overhead.
2.2
ZeRO
ZeRO [20] presents a memory-optimized approach to data parallel training by partitioning model
states, encompassing parameters, gradients, and optimizer states, across multiple GPUs while selec-
tively retrieving the required model states solely for computing specific layers.
ZeRO entails three distinct stages. ZeRO stage 1 (ZeRO-1) partitions optimizer states across GPUs.
In ZeRO stage 2 (ZeRO-2), optimizer states and gradients undergo partition. Both ZeRO-1 and
ZeRO-2 effectively reduce memory consumption while maintaining the same communication volume
compared to DP. Advancing further, ZeRO stage 3 (ZeRO-3) partitions all three constituents of
the model states, namely parameters, gradients, and optimizer states. ZeRO-3 stands out due to its
remarkable memory efficiency but suffers from parameter gathering in each forward and backward
computation. Substantial extra communication overheads are introduced.
To excavate more memory capability, subsequent works such as ZeRO-Infinity [19] and ZeRO++ [27]
have been proposed. However, these approaches rely on CPU offloading or quantization methods,
which can introduce performance or accuracy losses.
In this paper, unless otherwise specified, ZeRO refers to ZeRO Stage 3.
2.3
Gradient Calculation Decoupling in Pipeline Parallelism
With the progression of distributed training, numerous PP strategies have been proposed [8, 15, 14,
11]. However, a predominant issue encountered by most of these approaches is the presence of
3


---

pipeline bubbles. One effective solution to mitigate pipeline bubbles involves decoupling the gradient
calculations for weights and inputs [16, 18]. 2
In the backward propagation of model training, the loss is backpropagated according to the chain rule
of differentiation to calculate input and weight gradients. However, the computation of gradients for
inputs and weights can be decoupled. Taking a basic linear layer as an example, the gradients for
inputs are obtained by multiplying the gradients of activation values by weights, while the gradients
for weights are obtained by multiplying the gradients of activation values by inputs. After decoupling
gradient computations, we can first backpropagate the input gradients to maintain data dependencies
and then compute the weight gradients to update local parameters where needed in PP.
3
Method
3.1
ZeRO-Compatible Pipeline Parallelism
In this section, the designed ZeRO-compatible PP schedules are introduced. As depicted in Figure 2,
we illustrate three schedules in a top-to-bottom arrangement, each building upon the improvements
of its predecessor.
scheduling unit 1
scheduling unit 2
…
forward
input gradient calculation
weight gradient  calculation
optimizer step
decoupling of input and weight 
gradient computation
reusing scheduling unit in an iteration for large batch size
Figure 2: Illustration of the proposed ZeRO-compatible PP schedules. In the three PP schedules
presented from top to bottom, effective solutions are utilized pertaining to communication, pipeline
bubbles, and memory usage at larger batch sizes in the integration of PP and ZeRO. In particular,
the micro-batches are divided into multiple groups, and the computations corresponding to each
micro-batch group are regarded as a scheduling unit.
We first elaborate on the repetitive parameter gatherings in the combination of conventional PP
strategies with ZeRO. To enhance comprehension, we depict a concrete example in Figure 3, show-
casing the forward pass of a 4-layer model with 3 micro-batches distributed across 2 GPUs. As
observed, in a conventional PP schedule combined with ZeRO, parameter gathering is required before
each micro-batch (and likewise gradient reduction in the backward pass), resulting in significant
communication overhead and decreased GPU utilization.
A ZeRO-compatible PP schedule shown as the first schedule of Figure 2 partitions the model into
stages which are multiples of the pipeline parallelism degree and mapped to the devices with a
looping placement. The micro-batches of each stage are scheduled consecutively to reuse the
gathered parameters in the stage computation of different micro-batches, which reduces parameter
gathering times largely as shown in Figure 3. Similar PP strategy have been proposed [11]. However,
2“Parameter” and “weight” are interchangeable in the article.
4


---

we observe that such a PP strategy brings about two issues. Firstly, when the batch sizes are small,
pipeline bubbles cost a large proportion of training time, resulting in decreased GPU utilization.
Secondly, when the batch sizes are large, the occupied memory by activation values cannot be
released in a timely manner, leading to out-of-memory (OOM) errors. Therefore, we make two
further improvements to enhance its generality.
A
B
C
D
Forward
1
2
3
micro-batch
Model
1A
WA
GPU0
GPU1
Layer
Device
1B
WB
2A
WA
2B
WB
3A
WA
3B
WB
1C
WC
1D
WD
WC
2C
2D
WD
3C
WC
WD
3D
1A
WA
2A
3A
WC
1C
2C
3C
1B
WB
2B
3B
WD
1D
2D
3D
GPU0
GPU1
traditional PP + ZeRO
our ZeRO-Compatiable PP + ZeRO
Figure 3: Illustration on forward pass communications in the combination of PP and ZeRO. Digits
from 1 to 3 are the indices of micro-batches, and letters from A to D are the indices of layers. Wi
represents the parameter gathering communication of the i-th layer.
3.1.1
Near-Zero Bubble Pipeline Parallelism
Pipeline bubbles are bunches of device idling time due to the maintenance of data dependencies
during forward and backward propagation, which cause performance degradation and restrict the use
of pipeline parallelism. We leverage the separation of gradient calculation of inputs and weights for a
delicate pipeline schedule [16, 18], aiming to achieve near-zero bubbles.
Initially, each device is mapped with multiple stages using the looping placement [14]. Besides,
we categorize the stage computation into four types, i.e. forward pass, input gradient computation,
weight gradient computation, and optimizer step. Since input gradients are transferred among devices
to maintain the data dependency of network layers, we schedule input gradient computations as
early as possible to reduce the device idling time. Besides, we insert weight gradients whose data
dependency is satisfied into pipeline bubbles to further improve efficiency.
More specifically, the optimized pipeline schedule includes a forward process, a forward-backward-
interleaved process, and a backward process as shown in the second schedule of Figure 3. In
the forward process, micro-batches of forward passes are breadth-firstly scheduled by stages on
each device. The forward-backward-interleaved process consists of an interleaved schedule on the
calculation tasks of forward, input gradient, and weight gradient corresponding to the last mapped
stage in the looping placement. The forward passes and input gradient computations are scheduled
with higher priority, while weight gradient computations are used for filling in the pipeline bubbles.
In the backward process, the micro-batch calculations of input and weight gradient are breadth-
firstly scheduled by stages, following which the left weight gradient calculations of the last stage
are scheduled. Once the number of micro-batches and the pipeline parallelism degree satisfy a
certain condition as analyzed in Section 4, near-zero pipeline bubbles could be achieved. The high
communication efficiency in the combination of PP and ZeRO is ensured with the blockwise schedule
where micro-batches of computations are breadth-firstly scheduled by stages.
3.1.2
Reusing Allocated Memory of Scheduling Units
In order to reduce activation memory consumption, we utilize a looping scheduling scheme. The
micro-batches are divided evenly into multiple groups. The stage computations are scheduled by
the micro-batch groups. We call the stage computations corresponding to each micro-batch group a
scheduling unit. In each scheduling unit, the near-zero bubble pipeline schedule is performed. In
addition, the size of the scheduling unit is set adaptively according to the batch size.
5


---

The memory allocated by each scheduling unit will be promptly released before the next scheduling
unit begins. Hence, the maximum allocated memory is identical to the allocated memory of the
scheduling unit. By adaptively setting the size of the scheduling unit according to batch sizes, ZeroPP
achieves a satisfying trade-off between memory consumption and communication.
3.2
TP-Free Hybrid Parallelism
In this section, we will describe how the proposed PP strategy is integrated with ZeRO and vanilla
DP to achieve an efficient hybrid parallelism method for large-scale distributed training.
3.2.1
ZeRO-3 + PP + DP
Models are generally trained on large-scale clusters, where the bandwidth between nodes is compara-
tively lower than that within a node. Therefore, in ZeroPP, we combine ZeRO-3, PP, and vanilla DP
as a hybrid parallelism method to train models at scale in which ZeRO-3 is performed intra-node
and DP is performed inter-node. More concretely, each node stores a complete model state copy
that is partitioned evenly inside a node. In each scheduling unit, two all-gather operations and one
reduce-scatter operation for each stage to gather parameters and synchronize gradients, respectively,
are performed within a node. We thus avoid parameter gathering between nodes and simply perform
one inter-node all-reduce gradient synchronization in the last scheduling unit.
3.2.2
ZeRO-3 + PP + ZeRO-1
To further reduce memory usage, we substitute vanilla DP with ZeRO-1 in the hybrid parallelism
method discussed above. Initially, each node stores a parameter and gradient copy that is partitioned
inside a node. The optimizer states are partitioned across all GPUs. After the gradient computations
of the last scheduling unit in each iteration, we perform gradient reduce-scatter across nodes, update
the parameters corresponding to the partitioned optimizer states, and perform an all-gather operation
to assemble the parameter copy across nodes. In this way, we further partition the optimizer states
across the nodes, while maintaining no extra communication overhead.
3.3
Activation Recomputation
Recomputation [4] is an essential technique for reducing activation memory usage during the training
of large-scale models. The technique becomes particularly crucial as the input sequence length of
models continues to increase, and the memory consumption associated with activation values grows
accordingly. For those PP methods [16, 18] that achieve improvement in pipeline bubbles by schedul-
ing the calculations of input gradients and weight gradients separately, the incorporation of activation
recomputation poses a non-trivial challenge. Specifically, when a layerwise recomputation strategy is
adopted, both the input and weight gradient computation depend on the activation recomputation.
Hence, extra unacceptable recomputation overhead might be introduced due to the separation of
gradient computations, thereby limiting further scalability.
As depicted in Figure 2, our proposed PP approach exhibits continuity in the input and weight
gradient computation excluding the computations in the forward-backward-interleaved scheduling
process. The continuity requires a single activation recomputation for consecutive input and weight
gradient computations without extra overhead. Therefore, building upon the proposed PP approach,
we further propose a simple and efficient recomputation scheme in which we perform recomputation
for all mapped stages except the last stage in the interleaved scheduling process on each device. The
recomputation method can either be full [4] or selective [10].
4
Analysis
In this section, a comprehensive theoretical analysis is conducted on ZeroPP. We will focus on two
aspects: comparisons between ZeroPP and conventional 3D parallelism, emphasizing the advantages
of ZeroPP utilizing the intra-node parallelism ZeRO, and comparisons among the PP+DP approaches
where intra-node parallelism is not required. We use symbolic representations listed in Table 1 for
ease of description.
6


---

Table 1: The table of symbols used in the paper.
Symbols
Meanings
s
Sequence length
h
Hidden size
L
Number of total transformer layers
Mw
Weight memory of a transformer layer
Ma
Activation memory of a transformer layer
V
Number of stages on each device
B
Number of micro-batches on a single GPU
U
Number of micro-batches within a scheduling unit
b
Micro-batch size
P
PP size
D
DP size
4.1
Comparison with Conventional 3D Parallelism
We elaborate with the core transformer block [24] with a hidden size of h. Besides, each device deals
with B micro-batches and each micro-batch contains b samples whose sequence length is s in an
iteration. In terms of 3D parallelism, TP requires four all-reduce operations during the forward pass
and backward pass, resulting in a communication volume of 8Bbsh in a transformer block [10].
In ZeroPP, we assume each scheduling unit has the number of micro-batches U. Thus, B
U scheduling
units are processed in each iteration. Within each scheduling unit, two all-gather operations are
performed for parameter gathering, and one reduce-scatter operation is performed for gradient syn-
chronization, resulting in a communication volume three times that of the parameter size. Considering
that the parameter size of a single typical transformer block is around 12h2 (we neglect the item with
small orders), we can calculate the total communication volume in one iteration as 36Bh2
U
. Therefore,
as long as the condition 2sUb > 9h is satisfied, the communication volume of ZeroPP introduced by
intra-operator parallelism is smaller than that of conventional 3D parallelism. Current models prefer
to have larger sequence lengths and are trained with larger batch sizes, which makes the condition
quite easy to meet in practice and enables ZeroPP a competitive hybrid distributed alternative.
In addition, ZeroPP offers multiple other advantages compared with conventional 3D parallelism.
Firstly, the communication in TP is challenging to overlap with computation due to the strong
intra-block data dependency, whereas the communication in ZeroPP can be easily overlapped by
layerwise data prefetching as mentioned in Section 3. Secondly, ZeroPP employs a near-zero
bubble PP method, which effectively improves device utilization. Thirdly, ZeroPP presents more
communication overlapping opportunities for gradient synchronization in vanilla DP since the stages
are scheduled breadth-firstly by micro-batches and can start gradient synchronization earlier compared
to conventional PP strategies [11].
4.2
Comparison among Various PP Methods
ZeroPP mitigates pipeline bubbles by early scheduling the calculation of input gradients which
reduces the waiting time of devices due to data dependency in the backward propagation. Besides,
inserting the calculation of weight gradients that are ready in the pipeline bubbles, ZeroPP could
achieve near-zero bubbles in the whole training process. However, near-zero bubbles are achieved
when enough active micro-batches are ensured concerning data dependency in the forward and
backward propagation. The minimum number of active micro-batches to achieve near-zero bubbles is
2P −1. Therefore, pipeline bubbles related to the number of micro-batches U in a scheduling unit
are described as
#Pipeline Bubbles =
(
≈0,
U ≥2P −1
B(2P −1−U)
U
,
U < 2P −1
.
When the number of active micro-batches is less than 2P −1, pipeline bubbles could still be relieved
due to the split of gradient computation and the early schedule of input gradient computation.
7


---

The activation memory could be fixed to a maximum memory by a scheduling unit. For the sake of
simplicity, we have omitted the analysis of gradients as it follows a similar pattern to that of weight
analysis. Therefore, the memory consumption of ZeroPP is described as,
Max Allocation Memory = LMw/(PD) + LMw/(PV )
|
{z
}
weight_mem
+ MIN(B, U)LMa/P
|
{z
}
activation_mem
.
We also compare ZeroPP with state-of-the-art PP methods combined with vanilla DP. As listed in
Table 2, ZeroPP achieves superior performance in pipeline bubbles and weight memory usage. To
further integrate recomputation, the activation memory is reduced largely at a small cost.
Table 2: Comparison of state-of-the-art PP methods.
Pipeline Methods
Bubble Ratio
Weight Memory
Activation Memory
Gpipe
(P −1)/B
↑
LMw/P ↑
BLMa/P
↓
1F1B
(P −1)/B
↑
LMw/P ↑
LMa
↑↑
Interleaved 1F1B
(P −1)/(V B) ↑↑
LMw/P ↑
LMa(1 + (P −1)/(V P)) ↑
ZeroPP
Near Zero
↑↑↑
LMw(1/(PD) + 1/(PV ))
↑↑↑
LMa(2 −1/P) ↑
ZeroPP + Recomp
Near Zero
↑↑↑
LMw(1/(PD) + 1/(PV ))
↑↑↑
LMa(2/V −1/(V P)) ↑↑↑
5
Evaluation
We conduct a series of experiments to validate our main claim that ZeroPP achieves high training
efficiency compared to 3D parallelism while maintaining comparable GPU memory consumption.
We implement our codebase using PyTorch [17] and conduct our experiments using up to 64 NVIDIA
A800-SXM4-80GB GPUs distributed across 8 nodes that are interconnected by an InfiniBand network.
The details of GPT models we use are shown in Table 3.
We compare our method with 3D parallelism based on Megatron [21]. Besides, the 3D parallelism
methods incorporated with two state-of-art PP methods are tested, respectively, including the one
utilizing the mainstream interleaved 1F1B method [21] and the other employing the latest ZB-V
method [18]. In terms of DP, both our method and 3D parallelism employ the Pytorch DDP [12].
Table 3: Details of the models
MODEL
SIZE
NUM
LAYERS
NUM
HEADS
HIDDEN
SIZE
SEQ
LENGTH
14.6B
48
40
5120
1024
28.3B
64
48
6144
2048
5.1
Comparison to 3D parallelism with Fixed Parallelism Degree
We first demonstrate the performance gap between ZeroPP and 3D parallelism with fixed parallelism
degree. In Table 4, we present the iteration time and the maximum allocated GPU memory under
different settings. The experimental results align with our theoretical expectations, yet there are
notable observations that require further elucidation.
The underperformance of ZB-V can be primarily attributed to two aspects. Firstly, though ZB-B
should maintain fixed GPU memory consumption based on the theoretical analysis, the open-source
codebase that the paper [18] presented exhibits an increase in memory consumption as the batch size
grows in the testings. Secondly, the ZB-V implementation lacks compatibility with numerous 3D
parallel optimization techniques, resulting in suboptimal performance.
For ZeroPP, we employ recomputation to reduce the memory footprint of activations. The throughput
of ZeroPP with recomputation significantly surpasses that of 3D parallelism.
8


---

Table 4: Experimental results of ZeroPP versus 3D parallelism under various settings.
Setup
Model
14.6B
28.3B
DP Size
1
2
TP/ZeRO-3 Size
8
8
PP Size
4
4
Global Batch Size
192
256
Performance
Methods
Memory
Iteration Time
Memory
Iteration Time
3D Parallelism
10.5G
7.5s
21.6G
12.1s
3D Parallelism + SP
9.6G
9.1s
18.9G
11.9s
ZB-V + TP + DP
20.8G
8.4s
40.6G
13.3s
ZeroPP
12.1G
4.1s
23.8G
9.2s
5.2
Comparison to 3D parallelism under Equivalent Memory Constraints
In order to provide a clearer comparison of the performance of 3D parallelism and ZeroPP in practical
applications, we evaluate the maximum throughput of the two methods under equivalent GPU memory
constraints. The experimental results are presented in Figure 4.
0
0.5
1
1.5
2
2.5
3
3.5
4
4.5
5
0
5
10
15
20
25
30
35
40
45
50
20
30
40
50
Iteration Time(s)
Memory Usage (GB)
Memory Limit (GB)
3D Parallelism Actual Memory
ZeroPP Actual Memory
3D Parallelism Iteration Time
ZeroPP Iteration Time
Figure 4: Comparison of the iteration time and actual GPU memory consumption between ZeroPP
and 3D parallelism under equivalent GPU memory constraints.
We conduct experiments using a 14.6B model on 4-node A800 machines with 32 GPUs and explore
the hyperparameter space of ZeroPP and 3D parallelism to find the optimal configurations under the
same GPU memory limitations. The PP size is uniformly set to 4 and the global batch size is fixed to
192. Due to resource constraints, ZeroPP does not perform DP across nodes, whereas 3D parallelism
adjusts the DP and TP sizes within each node. For ZeroPP, we search for the scheduling unit size,
the number of stages on each device, and whether to utilize recomputation. For 3D parallelism, we
explore the TP/DP size and whether to use sequence parallelism.
The experimental results demonstrate that under equivalent memory constraints, ZeroPP exhibits
significantly shorter iteration times compared to 3D parallelism. Moreover, as the memory limitation
becomes more stringent, the advantage of ZeroPP becomes increasingly evident. This highlights the
effectiveness of ZeroPP and its ability to achieve high performance with reduced memory footprint.
6
Conclusion
In conclusion, we propose a scalable TP-free distributed framework that leverages PP and ZeRO-
3 efficiently. A ZeRO-compatible PP schedule is proposed which combines ZeRO-3 with high
communication efficiency and reduced memory usage. By decoupling gradient computations, near-
zero bubble pipeline parallelism is achieved. An efficient recomputation scheme is also proposed to
reduce activation memory consumption in ZeroPP. Furthermore, we also perform theoretical analysis
on ZeroPP and prove its efficiency over conventional 3D parallelism. We look forward to further
applying and optimizing ZeroPP to expansive large-scale training scenes.
9


---

However, ZeroPP encounters challenges pertaining to the activation memory, and memory consump-
tion is suboptimal when recomputation is not included. Nonetheless, we show that our recomputation
strategy can efficiently resolve this problem and believe that further optimization such as offloading
can be achieved by synergistically combining PP with ZeRO-3.
References
[1]
Zhengda Bian et al. “Maximizing parallelism in distributed training for huge neural networks”.
In: arXiv preprint arXiv:2105.14450 (2021).
[2]
Sid Black et al. “Gpt-neox-20b: An open-source autoregressive language model”. In: arXiv
preprint arXiv:2204.06745 (2022).
[3]
Tom Brown et al. “Language models are few-shot learners”. In: Advances in neural information
processing systems 33 (2020), pp. 1877–1901.
[4]
Tianqi Chen et al. “Training deep nets with sublinear memory cost”. In: arXiv preprint
arXiv:1604.06174 (2016).
[5]
Jeffrey Dean et al. “Large scale distributed deep networks”. In: Advances in neural information
processing systems 25 (2012).
[6]
Alexey Dosovitskiy et al. “An image is worth 16x16 words: Transformers for image recognition
at scale”. In: arXiv preprint arXiv:2010.11929 (2020).
[7]
Zhengxiao Du et al. All nlp tasks are generation tasks: A general pretraining framework. 2021.
[8]
Yanping Huang et al. “Gpipe: Efficient training of giant neural networks using pipeline
parallelism”. In: Advances in neural information processing systems 32 (2019).
[9]
Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. “Bert: Pre-training of
deep bidirectional transformers for language understanding”. In: Proceedings of naacL-HLT.
Vol. 1. 2019, p. 2.
[10]
Vijay Anand Korthikanti et al. “Reducing activation recomputation in large transformer
models”. In: Proceedings of Machine Learning and Systems 5 (2023).
[11]
Joel Lamy-Poirier. “Breadth-First Pipeline Parallelism”. In: Proceedings of Machine Learning
and Systems 5 (2023).
[12]
Shen Li. Getting Started with Distributed Data Parallel. 2024. URL: https://pytorch.
org/tutorials/intermediate/ddp_tutorial.html (visited on 2024).
[13]
Shenggui Li et al. “Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel
Training”. In: (2021).
[14]
Deepak Narayanan et al. “Efficient large-scale language model training on gpu clusters us-
ing megatron-lm”. In: Proceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis. 2021, pp. 1–15.
[15]
Deepak Narayanan et al. “PipeDream: Generalized pipeline parallelism for DNN training”. In:
Proceedings of the 27th ACM Symposium on Operating Systems Principles. 2019, pp. 1–15.
[16]
Hyungjun Oh et al. “Out-of-order backprop: An effective scheduling technique for deep
learning”. In: Proceedings of the Seventeenth European Conference on Computer Systems.
2022, pp. 435–452.
[17]
Adam Paszke et al. “Pytorch: An imperative style, high-performance deep learning library”.
In: Advances in neural information processing systems 32 (2019).
[18]
Penghui Qi et al. “Zero Bubble Pipeline Parallelism”. In: arXiv preprint arXiv:2401.10241
(2023).
[19]
Samyam Rajbhandari et al. “Zero-infinity: Breaking the gpu memory wall for extreme scale
deep learning”. In: Proceedings of the international conference for high performance comput-
ing, networking, storage and analysis. 2021, pp. 1–14.
[20]
Samyam Rajbhandari et al. “Zero: Memory optimizations toward training trillion parameter
models”. In: SC20: International Conference for High Performance Computing, Networking,
Storage and Analysis. IEEE. 2020, pp. 1–16.
[21]
Mohammad Shoeybi et al. “Megatron-lm: Training multi-billion parameter language models
using model parallelism”. In: arXiv preprint arXiv:1909.08053 (2019).
[22]
Shaden Smith et al. “Using deepspeed and megatron to train megatron-turing nlg 530b, a
large-scale generative language model”. In: arXiv preprint arXiv:2201.11990 (2022).
10


---

[23]
Hugo Touvron et al. “Llama: Open and efficient foundation language models”. In: arXiv
preprint arXiv:2302.13971 (2023).
[24]
Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information process-
ing systems 30 (2017).
[25]
Boxiang Wang et al. “2.5-dimensional distributed model training”. In: arXiv e-prints (2021),
arXiv–2105.
[26]
Boxiang Wang et al. “Tesseract: Parallelize the tensor parallelism efficiently”. In: Proceedings
of the 51st International Conference on Parallel Processing. 2022, pp. 1–11.
[27]
Guanhua Wang et al. “Zero++: Extremely efficient collective communication for giant model
training”. In: arXiv preprint arXiv:2306.10209 (2023).
[28]
Peng Xu et al. “MEGATRON-CNTRL: Controllable story generation with external knowledge
using large-scale language models”. In: arXiv preprint arXiv:2010.00840 (2020).
[29]
Qifan Xu and Yang You. “An efficient 2d method for training super-large deep learning models”.
In: 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE.
2023, pp. 222–232.
11
