---
title: 'ZeRO-Ofﬂoad: Democratizing Billion-Scale Model Training'
id: zero-ofﬂoad-democratizing-billion-scale-model-training
tags:
- cmp170hx-training-feasibility-a0d4fc
- zero-offload
- cpu-offload
- pcie
- memory-optimization
- sub-q3
created: '2026-10-27T10:38:59.358940Z'
updated: '2026-10-27T19:00:31.255543Z'
source: https://arxiv.org/abs/2101.06840
source_domain: arxiv.org
fetched_at: '2026-10-27T10:38:59.358264Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'ZeRO-Offload (Microsoft / UC Merced): offloads optimizer states and gradients
  to CPU memory to train 13B parameters on a single V100-32GB, a 10x increase over
  the 1.4B ceiling of plain PyTorch, at 40 TFLOPS/GPU for a 10B model versus 30 TFLOPS
  for PyTorch on 1.4B. Scales near-linearly to 128 GPUs and reaches 70B parameters
  on one DGX-2 box when combined with model parallelism (4.5x over MP alone). The
  load-bearing constraint for this run: CPU memory bandwidth is at least an order
  of magnitude faster than PCIe bandwidth between CPU and GPU, and GPU memory is another
  order of magnitude faster again, so the design must minimize CPU-GPU traffic (theoretical
  minimum 4M for a model-state offload). On a PCIe Gen2 x4 box the PCIe tier is roughly
  an order of magnitude slower than a modern x16 Gen4 link, so host-offload is even
  less attractive than in this paper''s V100 testbed.'
raw_file: raw/zero-ofﬂoad-democratizing-billion-scale-model-training.pdf
doi: arXiv:2101.06840
utility_score: 13.0
---

*Suggested by [[zero-memory-optimizations-toward-training-trillion-parameter-models-sc20]] — ZeRO cites CPU offload over PCIe as the alternative it rejects; ZeRO-Offload is that primary source*

ZeRO-Ofﬂoad: Democratizing Billion-Scale Model Training
Jie Ren∗, Samyam Rajbhandari† , Reza Yazdani Aminabadi† , Olatunji Ruwase†
Shuangyan Yang∗, Minjia Zhang† , Dong Li∗, Yuxiong He†
†Microsoft , ∗University of California, Merced
{jren6, syang127, dli35}@ucmerced.edu, {samyamr, yazdani.reza, olruwase, minjiaz, yuxhe}@microsoft.com
Abstract
Large-scale model training has been a playing ground for a
limited few requiring complex model refactoring and access to
prohibitively expensive GPU clusters. ZeRO-Ofﬂoad changes
the large model training landscape by making large model
training accessible to nearly everyone. It can train models with
over 13 billion parameters on a single GPU, a 10x increase
in size compared to popular framework such as PyTorch, and
it does so without requiring any model change from the data
scientists or sacriﬁcing computational efﬁciency.
ZeRO-Ofﬂoad enables large model training by ofﬂoading
data and compute to CPU. To preserve compute efﬁciency,
it is designed to minimize the data movement to/from GPU,
and reduce CPU compute time while maximizing memory
savings on GPU. As a result, ZeRO-Ofﬂoad can achieve 40
TFlops/GPU on a single NVIDIA V100 GPU for 10B pa-
rameter model compared to 30TF using PyTorch alone for a
1.4B parameter model, the largest that can be trained without
running out of memory. ZeRO-Ofﬂoad is also designed to
scale on multiple-GPUs when available, offering near linear
speedup on up to 128 GPUs. Additionally, it can work to-
gether with model parallelism to train models with over 70
billion parameters on a single DGX-2 box, a 4.5x increase in
model size compared to using model parallelism alone.
By combining compute and memory efﬁciency with ease-
of-use, ZeRO-Ofﬂoad democratizes large-scale model train-
ing making it accessible to even data scientists with access to
just a single GPU.
1
Introduction
Since the advent of the attention-based deep learning (DL)
models in 2017, we have seen an exponential growth in DL
model size, fueled by substantial quality gains that these atten-
tion based models can offer with the increase in the number
of parameters. For example, the largest language model in
literature had less than 100M parameters in 2017, it grew
to over 300M with BERT [6] in 2018, increasing to tens of
billions in 2019 with models such as GPT-2 [3], T5 [20],
Megatron-LM [28] and Turing-NLG [25]. Today, the largest
language model GPT-3 [2] has a staggering number of 175B
parameters. With the three orders of magnitude growth in
model size since 2017, the model accuracy continues to im-
prove with the model size [12]. Recent studies in fact show
that larger models are more resource-efﬁcient to train than
smaller ones [12] for a given accuracy target. As a result, we
expect the model size to continue growing in the future.
However, accessibility to large model training is severely
limited by the nature of state-of-art system technologies.
Those technologies make entry into the large model train-
ing space prohibitively expensive. To be more speciﬁc, dis-
tributed parallel DL training technologies such as pipeline
parallelism [10], model parallelism [28], and ZeRO [21]
(Zero Redundancy Optimizer) allow transcending the mem-
ory boundaries of single GPU/accelerator device by splitting
the model states (parameters, gradients and optimizer states)
across multiple GPU devices, enabling massive models that
would otherwise simply not ﬁt in a single GPU memory. All
record-breaking large models such as GPT-2, Megatron-LM,
Turing-NLG, GPT-3, were trained using a combination of the
aforementioned technologies. However, all of these DL par-
allel technologies require having enough GPU devices such
that the aggregated GPU memory can hold the model states
required for the training. For example, training a 10B param-
eter model efﬁciently requires a DGX-2 equivalent node with
16 NVIDIA V100 cards, which cost over 100K, beyond the
reach of many data scientists, and even many academic and
industrial institutions.
Heterogeneous DL training is a promising approach
to reduce GPU memory requirement by exploiting CPU
memory. Many efforts have been made in this direction
[8, 9, 11, 17, 23, 23, 24, 32–34]. Nearly all of them target
CNN based models, where activation memory is the memory
bottleneck, and model size is fairly small (less than 500M).
However, the primary memory bottleneck for recent attention
based large model training are the model states, instead of
activation memory. There is an absence in literature studying
these workloads for heterogeneous DL training. Additionally,
existing efforts on heterogeneous training are further limited
in two major ways: i) nearly all of them exploit CPU memory,
but not CPU compute, which we show can be used to signiﬁ-
cantly reduce the CPU-GPU communication overhead, and
ii) they are mostly designed for and evaluated on single GPU
1
arXiv:2101.06840v1  [cs.DC]  18 Jan 2021


---

[9, 11, 23, 34], without a clear path to scaling efﬁciently on
multiple GPUs that is crucial for large model training.
Addressing the aforementioned limitation, we attempt to de-
mocratize large model training by developing ZeRO-Ofﬂoad,
a novel heterogeneous DL training technology designed
speciﬁcally for large model training. ZeRO-Ofﬂoad exploits
both CPU memory and compute for ofﬂoading, while offering
a clear path towards efﬁciently scaling on multiple GPUs by
working with ZeRO-powered data parallelism [21]. Addition-
ally, our ﬁrst principle analysis shows that ZeRO-Ofﬂoad pro-
vides an optimal and the only optimal solution in maximizing
memory saving while minimizing communication overhead
and CPU compute overhead for large model training.
ZeRO-Ofﬂoad is designed around three main pillars: i)
Efﬁciency, ii) Scalabilty, and iii) Usability.
Efﬁciency: The ofﬂoad strategy is designed with the goal
of achieving comparable compute efﬁciency to the state-of-art
non-ofﬂoad strategies but for signiﬁcantly larger models. To
achieve this goal, we rely on ﬁrst principle analysis to identify
a unique optimal computation and data partitioning strategy
between CPU and GPU devices that is optimal in three key
aspects: i) it requires orders-of-magnitude fewer computa-
tion on CPU compared to GPU, preventing the CPU compute
from becoming a performance bottleneck, ii) it minimizes the
communication volume between CPU and GPU preventing
communication from being a bottleneck, and iii) it provably
maximizes memory savings on GPU while achieving mini-
mum communication volume.
Our analysis shows that to be optimal in the aforemen-
tioned regards, we must ofﬂoad the gradients, optimizer states
and optimizer computation to CPU, while keeping the param-
eters and forward and backward computation on GPU. This
strategy enables a 10x increase in model size, with minimum
communication and limited CPU computation, which allows
us to train 13B parameters on a single NVIDIA V100 GPU
at 40 TFLOPS, compared to 30 TFLOPS on the same GPU
with 1.2B parameters, the largest model that can be trained
without any CPU ofﬂoading.
Ofﬂoading optimizer computation requires CPU to perform
O(M) computation compared to O(MB) on GPU where M
and B are the model size and batch sizes. In most cases, the
batch size is large, and CPU computation is not a bottleneck,
but for small batch sizes, the CPU compute can be a bottle-
neck. We address this using two optimizations: i) an efﬁcient
CPU optimizer that is up to 6x faster than the-state-of-art ,
and ii)One-step delayed parameter update that allows over-
lapping the CPU optimizer step with GPU compute, while
preserving accuracy. Together, they preserve efﬁciency for
ZeRO-Ofﬂoad even with small batch sizes.
Scalability: Good scalability is crucial to take advantage
of multiple GPUs that may be available to some data sci-
entists. In the DL community, data parallelism is generally
used as the de facto standard to scale DL training to multiple
GPUs [5, 26, 35]. However, it is not designed to work with
Figure 1: ZeRO-Ofﬂoad can be enabled with a few lines of change.
The code on left shows a standard training pipeline, while the right
shows the same pipeline with ZeRO-Ofﬂoad enabled.
heterogeneous training and presents scalability challenges
because of the replication of data and computation in data
parallel training. Data parallel training replicates all the model
states such as optimizer states, parameters, and gradients, and
it also replicates the optimizer computation on each GPU.
Therefore, ofﬂoading model states or optimizer computation
to CPU in combination with data parallelism will result in
signiﬁcant replication of communication and CPU compute:
increase the CPU memory requirement proportionally to the
data parallelism degree while limiting throughput scalability
due to the increased communication.
To address these challenges, ZeRO-Ofﬂoad combines our
ofﬂoad strategy with ZeRO [21] powered data parallelism
instead of traditional data parallelism. The symbiosis allows
ZeRO-Ofﬂoad to maintain a single copy of the optimizer
states on the CPU memory regardless of the data parallel
degree. Furthermore, it keeps the aggregate communication
volume between GPU and CPU, as well as the aggregate CPU
computation a constant regardless of data parallelism, allow-
ing ZeRO-Ofﬂoad to effectively utilize the linear increase in
CPU compute with the increase in the data parallelism degree.
As a result, ZeRO-Ofﬂoad achieves excellent scalability on
up to 128 GPUs.
In addition to working with ZeRO powered data paral-
lelism, ZeRO-Ofﬂoad can be combined with model paral-
lelism [27,28] to achieve higher memory savings, when mul-
tiple GPUs are available.
Usability: ZeRO-Ofﬂoad is available as part of an Open-
Source PyTorch library, DeepSpeed (www.deepspeed.ai).
Unlike most strategies discussed in Section 2, ZeRO-Ofﬂoad
does not require model refactoring to work. In fact, PyTorch
users can enable ZeRO-Ofﬂoad with few lines of code change
to their existing training pipeline as shown in Figure 1, al-
lowing to train 10x larger models easily. For detailed tuto-
rial, please see: https://www.deepspeed.ai/tutorials/
zero-offload/.
Contributions Our contributions are as follows:
• A unique optimal ofﬂoad strategy for heterogeneous large
model training on GPU + CPU system that enables 10x
larger model on a single GPU without sacriﬁcing efﬁciency
(Sec. 3, Sec. 4.1).
• Highly scalable multi-GPU design through i) a symbiotic
2


---

combination of ofﬂoad strategy with ZeRO powered data
parallelism (Sec. 4.2), allowing ZeRO-Ofﬂoad to achieve
near linear scalability, and ii) seamless integration with
model-parallel training [28], enabling even larger mod-
els than using ZeRO-Ofﬂoad or model parallelism alone
(Sec. 4.2).
• Optimized CPU execution through: i) High-performance
CPU Adam optimizer design and implementation offering
over 6x speedup over state-of-the-art Adam implementa-
tions (Sec. 5.1), and ii) One-step delayed parameter update
to overlap CPU parameter update with GPU forward and
backward pass (Sec. 5.2).
• Open-source implementation of ZeRO-Ofﬂoad in PyTorch.
• Extensive evaluation demonstrating i) Model Scale: 10x
increase in model size with up to 13B on a single GPU
and 4x increase in model size over model parallelism with
up to 70 B parameters on a DGX-2 node. ii) Efﬁciency:
Over 40 TFlops for a 10B parameters on a single NVIDIA
V100, compared to 30 TFLOPS on the same GPU with
1.2B parameters, the largest model that can be trained with-
out any CPU ofﬂoading. iii) Scalability: Near-perfect linear
scalability for a 10B parameter model on up to 128 GPUs.
iv) CPU overhead reduction with our ADAM implemen-
tation with 6x speedup over PyTorch optimizer and up to
1.5X improvement in end-to-end throughput with delayed
parameter update optimizations (Sec. 6).
2
Background and Related Work
Memory consumption in large model training. The full
spectrum of memory consumption during DNN model train-
ing can be classiﬁed into two parts: i) model states and ii)
residual states [21]. Model states include parameters, gradi-
ents, and optimizer states (such as momentum and variances
in Adam [13]); Residual states include activations, temporary
buffers, and unusable fragmented memory.
Model states are the primary source of memory bottleneck
in large model training. We consider the memory consump-
tion due to model states for large transformer models such
as Megatron-LM (8 billion) [28], T5 (11 billion) [20], and
Turing-NLG [25] (17.2 billion). They are trained with ﬂoat-16
mixed precision training [16] and Adam optimizer [13].
Mixed precision training often keeps two copies of the
parameters, one in ﬂoat-16 (fp16) and the other in ﬂoat-32
(fp32). The gradients are stored in fp16. In addition to the
parameters and gradients, the Adam optimizer keeps track of
the momentum and variance of the gradients. These optimizer
states are stored in fp32. Therefore, training a model in mixed
precision with the Adam optimizer requires at least 2 bytes
of memory for each fp16 parameter and gradient, and 4 byte
of memory for each fp32 parameter, and the moementum and
variance of each gradient. In total, a model with M parameters
requires 16×M bytes of memory. Therefore, the model states
for Megatron-LM, T5 and Turing-NLG require 128 GB, 176
GB and 284 GB, respectively, which are clearly beyond the
memory capacity of even the current ﬂagship NVIDIA A100
GPU with 80 GB of memory.
Signiﬁcant amount of work has been done in the recent
years to enable large model training, which requires more
memory than what is available on a single GPU to ﬁt these
model and residual states. These efforts can be classiﬁed
broadly into two categories: i) scale-out training and ii) scale-
up training based approaches. We discuss them as follows.
Scale out large model training. Scale-out training uses
aggregate memory of multiple GPUs to satisfy the memory
requirement for large model training. Two prominent exam-
ples of scale out training is model parallelism [5, 28] and
pipeline parallelism [7,10], both partitioning the model states
and the residual states across multiple GPUs. Model paral-
lelism [5,28] partitions the model vertically and distributes
the model partitions to multiple GPU devices in order to train
large models. Pipeline parallelism [7,10] on the other hand
parallelizes model training by partitioning the model horizon-
tally across layers. Both of these approaches must change the
user model to work, therefore can limit usability.
A recent work, ZeRO [21], provides an alternative to model
and pipeline parallelisms to train large models. ZeRO splits
the training batch across multiple GPUs similar to data paral-
lel training [5,26,35], but unlike data parallel training which
replicates all the model states on each GPU, ZeRO partitions
them across all GPUs, and uses communication collectives
to gather individual parameters as needed during the training.
ZeRO does not require changes to the user model to work,
making it more generic than model or pipeline parallel train-
ing. It also offers better compute efﬁciency and scalability.
Despite the ability of model and pipeline parallelisms, and
ZeRO to train large models, they all require multiple GPUs
such that the aggregate GPU memory can hold the model and
residual states for training large models. In contrast, ZeRO-
Ofﬂoad is designed to ﬁt a larger model by ofﬂoading model
states to CPU memory and can train a 10x larger model on
a single GPU without sacriﬁcing efﬁciency. When multiple
GPUs are available, ZeRO-Ofﬂoad is designed to work to-
gether with ZeRO to offer excellent scalability, or in conjunc-
tion with model parallelism to ﬁt even larger model sizes that
is not possible with ZeRO-Ofﬂoad or model parallelism alone.
Scale up large model training. Existing work scales up
model size in a single GPU through three major approaches.
The ﬁrst approach trades computation for memory saving
from activations (residual memory) by recomputing from
checkpoints [4]. The second approach uses compression tech-
niques such as using low or mixed precision [16] for model
training, saving on both model states and activations. The
third approach uses an external memory such as the CPU
memory as an extension of GPU memory to increase memory
capacity during training [8,9,11,17,23,24,33].
Our work, ZeRO-Ofﬂoad falls under the third approach.
Unlike ZeRO-Ofﬂoad, the above efforts only ofﬂoad data to
3


---

CPU but not compute, and they use smaller models training.
In contrast, a recent work called L2L [18] can enable multi-
billion parameter training by managing memory usage in
GPU layer by layer. In particular, L2L synchronously moves
tensors needed in the upcoming layer into GPU memory for
computation and keeps the rest of tensors into CPU memory
for memory saving. In comparison on ZeRO-Ofﬂoad, it offers
limited efﬁciency due to extra communication overhead, does
not offer a way to scale out across devices, and requires model
refactoring, making it difﬁcult to use.
ZeRO powered data parallel training. ZeRO-Ofﬂoad
works with ZeRO to scale DL training to multiple GPUs.
ZeRO has three stages, ZeRO-1, ZeRO-2 and ZeRO-3 cor-
responding to the partitioning of the three different model
states, optimizer states, gradients and parameters, respectively.
ZeRO-1 partitions the optimizer states only, while ZeRO-2
partitions gradients in addition to optimizer states, and ZeRO-
3 partitions all model states. ZeRO-Ofﬂoad works symbioti-
cally with ZeRO-2, and therefore we discuss it further.
In ZeRO-2, each GPU stores a replica of all the parameters,
but only updates a mutually exclusive portion of it during
the parameter update at the end of each training step. As
each GPU only updates a portion of the parameters, they
only store optimizer states and gradients required to make
that update. After the update, each GPU sends its portion
of the updated parameters to all the other GPUs using an
all-gather communication collective. ZeRO-2 computation
and communication schedule is described below:
During the forward pass, each GPU computes the loss with
respect to a different mini-batch. During the backward prop-
agation, as each gradient is computed, it is averaged using a
reduce operator at the GPU/GPUs that owns the gradient or
part of the gradient. After the backward pass, each GPU up-
dates its portion of the parameters and optimizer states using
the averaged gradients corresponding to that portion. After
this, an all-gather is performed to receive the rest of the
parameter update computed on other GPUs.
3
Unique Optimal Ofﬂoad Strategy
ZeRO-Ofﬂoad is designed to enable efﬁcient large model
training on a single or multiple GPUs by ofﬂoading some of
the model states from GPU to CPU memory during training.
As discussed in Sec. 2, model states: parameters, gradients,
and the optimizer states, are the primary source of memory
bottleneck in large model training. By ofﬂoading some of
these model states to CPU, ZeRO-Ofﬂoad can enable training
of signiﬁcantly larger models 1. However, identifying the
optimal ofﬂoading strategy is non-trivial. There are numerous
1ZeRO-Ofﬂoad only ofﬂoads model states. Ofﬂoading secondary sources
of memory bottleneck such as activation memory is beyond scope of our
ofﬂoad strategy. Given that they are signiﬁcantly smaller than model states,
we ignore them for the purpose of our analysis. Furthermore, the ﬁrst and
second approaches described in Sec. 2 can be used in conjunction with
ZeRO-Ofﬂoad to reduce activation memory
ways to ofﬂoad model states to CPU memory, each with a
different trade-off in terms of CPU computation, and GPU-
CPU communication, both of which can limit the training
efﬁciency.
To identify the optimal ofﬂoad strategy, ZeRO-Ofﬂoad mod-
els the DL training as data-ﬂow graph and uses ﬁrst principle
analysis to efﬁciently partition this graph between CPU and
GPU devices. ZeRO-Ofﬂoad partitions the graph in a way
that is optimal in three key aspects: i) it requires orders-of-
magnitude fewer computation on the CPU compared to GPU
that prevents CPU from becoming a performance bottleneck,
ii) it guarantees the minimization of communication volume
between CPU and GPU memory, and iii) it provably maxi-
mizes the memory savings while achieving minimum com-
munication volume. In fact, ZeRO-Ofﬂoad can achieve high
efﬁciency during training that is comparable to non-ofﬂoad
training and it is unique optimal, meaning no other solution
can offer better memory savings without increasing the com-
munication volume or increasing CPU computation.
In this section, we discuss the derivation of our unique
optimal ofﬂoad strategy. Our strategy is speciﬁcally designed
for mixed precision training with Adam optimizer which is
the de facto training recipe for large model training.
3.1
DL Training as a data-ﬂow graph
The DL training workload can be represented as a weighted
directed graph of data and computation, as shown in the ﬁg-
ure 2, where the circular nodes represents model states (param-
eter16, gradient16, parameter32, momentum32, variance32),
and the rectangular nodes represents computation (forward,
backward, param update). The edges in the graph represents
the data ﬂow between the nodes, and the weight of an edge
is the total data volume in bytes that ﬂows through it during
any given training iteration. For a model with M parameters,
the weight of the edges in this graph is either 2M where the
source node produces fp16 model states, or 4M where the
source node produces fp32 model states.
An ofﬂoad strategy between GPU and CPU can be repre-
sented using a two-way partitioning of this graph, such that
computation nodes in a partition would be executed on the
device that owns the partition, and the data nodes in the parti-
tion will be stored on device that owns the partition. The total
data volume that must be communicated between the GPU
and CPU is given by the weight of edges running across two
partitions.
There are numerous ways to partition this graph. In the
following sections, we use ﬁrst principles to simplify the data
ﬂow graph to reduce the number possible choices based on
three different efﬁciency metric: i) CPU computation over-
head, ii) communication overhead, and iii) memory savings.
4


---

CPU
gradient 16
2M
parameter 16
2M
GPU
FWD
BWD
activation 16
FWD-BWD 
Super Node
float2half
parameter 32
variance 32
momentum 32
Update Super Node
Param update
Param update
FWD
BWD
float2half
activation 16
gradient 16
momentum 32
variance 32
parameter 32
2M
12 M
4M
parameter 16
activation 16
2M
12 M
momentum 32
variance 32
parameter 32
4M
2M
2M
2M
Figure 2: The dataﬂow of fully connected neural networks with
M parameters. We use activation checkpoint to reduce activation
memory to avoid activation migration between CPU and GPU.
3.2
Limiting CPU computation
The CPU computation throughput is multiple orders of magni-
tude slower than the GPU computation throughput. Therefore,
ofﬂoading large computation graph to CPU will severely limit
training efﬁciency. As such, we must avoid ofﬂoading com-
pute intensive components to the CPU.
The compute complexity of DL training per iteration is
generally given by O(MB), where M is the model size and B
is the effective batch size. To avoid CPU computation form
becoming a bottleneck, only those computations that have a
compute complexity lower than O(MB) should be ofﬂoaded
to the CPU. This means that the forward propagation and
backward propagation both of which have a compute com-
plexity of O(MB) must be done on the GPU, while remaining
computations such as norm calculations, weight updates etc
that have a complexity of O(M) maybe ofﬂoaded to the CPU.
Based on this simple observation we fuse the forward and
backward nodes in our data ﬂow graph into a single super-
node (FWD-BWD) and assign it on the GPU.
3.3
Minimizing Communication Volume
The CPU memory bandwidth is at least an order of magnitude
faster than the PCI-E bandwidth between CPU and GPU,
while the GPU memory is another order of magnitude faster
than even the CPU memory. Therefore, we must minimize
the communication volume between CPU and GPU memory
to prevent the PCI-E bandwidth from becoming a training
performance bottleneck. To do so we must ﬁrst identify the
theoretical minimum communication volume for a model
state ofﬂoad strategy.
The minimum communication volume for any model state
ofﬂoad strategy is given by 4M 2. Note that after fusing the
forward and backward into a single super-node as discussed
in Sec. 3.2, each node in our data ﬂow graph is part of a cycle.
Therefore, any partitioning of this graph would require cutting
2Please note that it is possible to reduce the communication volume
further by only ofﬂoading partial model states. For simpliﬁcation, we assume
that an ofﬂoad of a model state implies that we ofﬂoad the entire model state.
Our analysis on the memory savings per communication volume, still holds
even if we ofﬂoad partial model states
at least two edges, each of which have a edge weight of at
least 2M, resulting in a total communication of at least 4M.
If we choose to limit the communication volume to this
bare minimum, we can greatly simplify our data-ﬂow graph
and reduce the number of partitioning strategies to a handful:
Creating fp32 super-node: Notice that any partitioning
strategy that does not co-locate the fp32 model states its pro-
ducer and consumer nodes cannot achieve the minimum com-
munication volume of 4M. Such a partition must cut at least
one edge with a weight of 4M, and the other with at least
2M, resulting in a communication volume of at least 6M .
Therefore, to achieve the minimum communication volume,
all ofﬂoad strategies must co-locate fp32 model states with its
producer and consumer operators, i.e., the fp32 model states
(momentum 32, variance 32 and p32) must be co-located with
the Param Update and the ﬂoat2half computations.
This constraint allows us to treat all the aforementioned
fp32 data and compute nodes in the data ﬂow graph as a single
super-node that we refer to as Update Super. We show this
reduced data ﬂow graph in ﬁgure 2, consisting of only four
nodes: FWD-BWD Super node, p16 data node, g16 data node,
and Update Super node.
p16 assignment: To achieve the minimum communica-
tion volume, p16 must be co-located with FWD-BWD Super
because the edge weight between these two nodes is 4M. Sep-
arating these two nodes, would increase the communication
volume to 6M(4M +2M). Since, we have already assigned
node FWD-BWD Super to GPU to limit computation on CPU,
p16 must also be assigned to GPU.
3.4
Maximizing Memory Savings
After simplifying the data ﬂow graph to minimize commu-
nication volume, only g16 and Update Super remain to be
assigned. Notice that at this point, all partitions will result
in minimum communication volume, so we can prune the
choices further to maximize the memory savings on the GPU.
Table 1 shows the memory savings of all valid partitioning
strategies that minimize the communication volume. The max-
imum memory savings of 8x can be achieved by ofﬂoading
both g16 and Update Super to CPU.
Table 1: Memory savings for ofﬂoad strategies that minimizes com-
munication volume compared to the baseline.
FWD-BWD
p16
g16
Update
Memory
Reduction
gpu
gpu
gpu
gpu
16M
1x (baseline)
gpu
gpu
cpu
gpu
14M
1.14x
gpu
gpu
gpu
cpu
4M
4x
gpu
gpu
cpu
cpu
4M
8x
3.5
A unique and optimal ofﬂoad strategy
ZeRO-Ofﬂoad allocates all the fp32 model states along with
the fp16 gradients on the CPU memory, and it also computes
the parameter updates on the CPU. The fp16 parameters are
kept on the GPU and the forward and backward computations
are also done on the GPU.
5


---

Computation stream:
Swapping stream:
GPU
CPU
GPU
FWD & BWD
FWD & BWD
p update
GPU->CPU
GPU->CPU
g offload
p swap
Step i
Step i+1
Figure 3: ZeRO-Ofﬂoad training process on a single GPU.
We arrive at this ofﬂoad strategy by simplifying our data
ﬂow graph and eliminating all other partitioning strategies
as they do not limit CPU computation, minimize commu-
nication volume, or maximize memory savings. Therefore,
ZeRO-Ofﬂoad is not only optimal in terms of the aforemen-
tioned metrics, it is also unique; there can be no other strategy
that can offer more memory savings than ZeRO-Ofﬂoad with-
out increasing the compute complexity on the CPU or incur
additional GPU-CPU communication volume.
4
ZeRO-Ofﬂoad Schedule
In this section, we discuss the concrete computation and com-
munication schedule for implementing ZeRO-Ofﬂoad on a
single GPU system based on our ofﬂoad strategy. We then
show how we extend this schedule to work effectively on
multi-GPU systems by combining our ofﬂoad strategy with
ZeRO data parallelism and model parallelism.
4.1
Single GPU Schedule
As discussed in Sec. 3, ZeRO-Ofﬂoad partitions the data such
that the fp16 parameters are stored in GPU while the fp16
gradients, and all the optimizer states such as fp32 momentum,
variance and parameters are stored in CPU.
During the training, we begin by computing the loss via the
forward propagation. Since the fp16 parameters are already
presented on the GPU, no CPU communication is required for
this part of the computation. During the backward propagation
on the loss, the gradient for different parameters are computed
at different point in the backward schedule. ZeRO-Ofﬂoad
can transfer these gradients for each parameter individually
or in small groups to the CPU memory immediately after
they are computed. Therefore, only a small amount of mem-
ory is required to temporarily hold the gradients on the GPU
memory before they are transferred to CPU memory. Fur-
thermore, each gradient transfer can be overlapped with the
backpropagation on the remainder of the backward graph,
allowing ZeRO-Ofﬂoad to hide a signiﬁcant portion of the
communication cost.
After the backward propagation, ZeRO-Ofﬂoad updates the
fp32 parameters and the remaining optimizer states (such as
momentum and variance) directly on the CPU, and copies the
updated fp32 parameters from the CPU memory to the fp16
parameters on the GPU memory. Figure 3 shows the com-
putation and communication in each step of ZeRO-Ofﬂoad
diagrammatically, and Figure 5 shows the concrete schedule
as a pseudo-code.
𝐺𝑃𝑈!
𝑪𝑷𝑼𝑵
…
𝐺𝑃𝑈"
𝐷𝑃!"#
𝐷𝑃!"$
𝐷𝑃!"%
𝐺𝑃𝑈#
𝐺𝑃𝑈!
𝑪𝑷𝑼𝟎
…
𝐺𝑃𝑈"
𝐷𝑃&
𝐷𝑃$
…
𝐷𝑃#
𝐺𝑃𝑈#
Parameters on
GPU memory
Gradients on  
GPU memory
Gradients on 
CPU memory
Optimizer States 
on CPU memory
Figure 4: ZeRO-Ofﬂoad data placement with multiple GPUs
4.2
Scaling to Multi-GPUs
ZeRO-Ofﬂoad in its entirety is a symbiotic integration
of ZeRO-Ofﬂoad strategy described in Sec. 3 and ZeRO-
powered data parallelism discussed in Sec. 2, which allows
ZeRO-Ofﬂoad to scale to hundreds of GPUs efﬁciently. ZeRO-
Ofﬂoad preserves the model state partitioning strategy of
ZeRO Stage-2 (optimizer state and gradient partitioning),
while ofﬂoading the partitioned gradients, optimizer states
and the corresponding parameter updates to the CPU.
The key beneﬁt of doing this partitioning before ofﬂoading
is that for systems with more than 1 GPU, each data parallel
process is only responsible for updating a subset of the pa-
rameters. The aggregated communication volume from all the
data parallel GPUs to the CPU remains constant, and CPU re-
sources are used in parallel to jointly compute a single weight
update. As a result the total CPU update time decreases with
increased data parallelism, since the CPU compute resources
increase linearly with the increase in the number of compute
nodes. This allows ZeRO-Ofﬂoad to achieve very good scal-
ability, as the overhead of communication across GPUs is
offset by the reduction in the CPU optimizer step.
ZeRO-Ofﬂoad partitions gradients and optimizer states
among different GPUs, and each GPU ofﬂoads the partition
it owns to the CPU memory and keeps it there for the en-
tire training. During the backward propagation, gradients are
computed and averaged using reduce-scatter on the GPU, and
each GPU only ofﬂoads the averaged gradients belonging
to its partition to the CPU memory. Once the gradients are
available on the CPU, optimizer state partitions are updated
in parallel by each data parallel process directly on the CPU.
After the update, parameter partitions are moved back to GPU
followed by an all-gather operation on the GPU similar to
ZeRO-2 to gather all the parameters. Figure 4 shows the data
placement model parameters, gradients and optimizer states
for ZeRO-Ofﬂoad and the details of the ZeRO-Ofﬂoad data
parallel schedule is presented in Figure 5. The all gather op-
eration described above is shown as a sequence of broadcast
operations in the Figure.
Model Parallel training ZeRO-Ofﬂoad can also work
together with tensor-slicing based model parallelism (MP)
frameworks such as Megatron-LM [28]. It does so by ofﬂoad-
ing the gradients, optimizer states and the optimizer compu-
tation corresponding to each MP process allowing ZeRO-
Ofﬂoad to train signiﬁcantly larger models than possible than
6


---

Figure 5: Code representing ZeRO-Ofﬂoad that combines unique
optimal CPU ofﬂoad strategy with ZeRO-powered data parallelism.
using model parallelism alone. Sec. 6 provides more details.
5
Optimized CPU Execution
We speedup the CPU execution time for the parameter updates
with two optimizations. First, we implement a fast CPU Adam
optimizer using high performance computing techniques of-
fering signiﬁcant speedup over state-of-art Pytorch imple-
mentation. Second, we develop a one-step delayed parameter
update schedule that overlaps the CPU parameter update com-
putation with the forward and backward computation on the
GPU, hiding the CPU execution time when enabled.
5.1
Implementing the CPU Optimizer
We use three levels of parallelism for improving the perfor-
mance of the CPU optimizer.. 1) SIMD vector instruction [15]
for fully exploiting the hardware parallelism supported on
CPU architectures. 2) Loop unrolling [31], an effective tech-
nique for increasing instruction level parallelism that is crucial
for better memory bandwidth utilization. 3) OMP multithread-
ing for effective utilization of multiple cores and threads on
the CPU in parallel. Using these technique, we present a sig-
niﬁcantly faster implementation of Adam optimizer compared
to state-of-art PyTorch implementation.
Mixed Precision Training with Adam
ADAM is an op-
timization algorithm used for deep-learning training, which
takes the loss gradients together with their ﬁrst and second
momentums to update the parameters. Therefore, in addition
to the model parameters, ADAM requires two more matrices
of the same size (M) saved during the training. In the mixed
precision training mode, there are two versions of the parame-
ters stored in memory: one in FP16 (p16) used for computing
the activations in the forward pass (on GPU), and one master
copy in FP32 (p32) which is updated by the optimizer (on
CPU). The p16 is updated with the p32 through float2hal f
casting, at each training step. Moreover, the momentum and
variance of the gradients are saved in FP32 (on CPU), to pre-
vent the precision loss for updating the parameters. Please
refer to [13] for further detail on ADAM’s algorithm.
Algorithm 2 CPU-ADAM Optimizer
Input: p32, g32, m32, v32, β1, β2, α , step, eps
Output: p16, p32, m32, v32
Parameter: tile_width, simd_width, unroll_width
1: biascorrection1 ←−α/(1−βstep
1
)
2: biascorrection2 ←1/
q
1−βstep
2
3: simd_count ←sizeo f(32) / simd_width
4: unroll omp parallel
5: for i in 1 to (simd_count/unroll_width) do
6:
...
7:
gv, pv, mv, vv = g32[i], p32[i],m32[i],v32[i]
8:
mv = FMA(gv, (1 - β1), β1*mv)
9:
vv = FMA(gv*gv, (1 - β2), β2*vv)
10:
gv = FMA(√vv, biascorrection2, eps)
11:
gv = mm / gv
12:
pv = FMA(gv, biascorrection1, pv)
13:
p32[i],m32[i],v32[i] = pv, mv, vv
14:
...
15:
IF (i == tile_width) Copy_to_GPU(p16, p32)
16: end for
Optimized Implementation Algorithm 1 elaborates the
ADAM’s implementation detail using SIMD operations. As
shown, the Adam function receives the optimizer parameters
such as β1, β2, and α, and the gradient, momentum, variance
and master copy of parameters (p32) as the input. We also
use some parameters speciﬁc to the implementation, like the
simd_width and unroll_width. The Adam optimizer sends
back the updated variance, momentum, and parameter in both
FP16 (to GPU) and FP32 (to CPU) .
7


---

We ﬁrstly read the data, including parameter, gradient, mo-
mentum and variance, into the vector registers (line 7). Then,
we use several fused multiply-add (FMA) vector operations
to preform the main execution pipeline which is repeated by
the unrolling width. Note that the rest of operations, such as
multiply, division, and sqrt, also run in vector mode. For the
best performance we use AVX512 simd instruction set and an
unroll_width of 8 based on auto-tuning results.
In addition to the CPU-Adam optimizer, we implement the
CPU-to-GPU 16FP parameter-copy in a tiled manner (line
15). We overlap the CPU and GPU execution by parallelizing
the Adam computation and copying the parameters over to
GPU. As we process Adam computation of the current tile of
data on CPU, we write the parameters back to GPU for the
previously processed tile. This way, we reduce the idle time
of GPU to start the processing of the next training step.
5.2
One-Step Delayed Parameter Update
Despite using a highly optimized CPU optimizer, the CPU
computation overhead can become a bottleneck during train-
ing with very small batch sizes, when the GPU computation
time is not much larger than CPU compute. For such limited
cases, we develop one-step delayed parameter update (DPU)
that overlaps CPU and GPU compute to hide the CPU compu-
tation overhead by delaying the parameter update by a single
step. We verify that DPU does not impact the ﬁnal accuracy
of training in the evaluation.
DPU training schedule Figure 6 shows the workﬂow of
ZeRO-Ofﬂoad training process with delayed parameter up-
date.  The ﬁrst N −1 steps, are trained without DPU to avoid
destabilizing the training during the early stages where gra-
dients change rapidly.  On step N, we obtain the gradients
from the GPU, but we skip the CPU optimizer step, and do
not update the fp16 parameters on the GPU either.  At step
N +1, we compute the parameter updates on the CPU using
gradients from step N, while computing the forward and back-
ward pass on the GPU in parallel using parameters updated at
step N −1. From this step onwards, the model at (i+1)th step
will be trained using the parameters updated with gradients
from (i−1)th step instead of parameters updated at ith step,
overlapping CPU compute with GPU compute.
Accuracy trade-off. Since DPU changes the semantics of
the training, it is reasonable to ask if there is a trade-off be-
tween model accuracy and training efﬁciency. To answer this
question, we evaluated DPU on multiple training workloads
and found that DPU does not hurt convergence if we intro-
duce DPU after a few dozen iterations instead of introducing
it at the beginning. Our evaluation result in Sec. 6 shows
that compared with training with ZeRO-Ofﬂoad only, training
with delayed parameter update achieves same model training
accuracy with higher training throughput.
Table 2: Hardware overview of experimental system.
DGX-2 node
GPU
16 NVIDIA Tesla V100 Tensor Core GPUs
GPU Memory
32GB HBM2 on each GPU
CPU
2 Intel Xeon Platinum 8168 Processors
CPU Memory
1.5TB 2666MHz DDR4
CPU cache
L1, L2, and L3 are 32K, 1M, and 33M, respectively
PCIe
bidirectional 32 GBps PCIe
6
Evaluation
This section seeks to answer the following questions, in com-
parison to the state-of-the-art:
(i) How does ZeRO-Ofﬂoad scale the trainable model size
compared to existing multi-billion parameter training
solutions on a single GPU/DGX-2 node?
(ii) What is the training throughput of ZeRO-Ofﬂoad on
single GPU/DGX-2 node?
(iii) How does the throughput of ZeRO-Ofﬂoad scale on up
to 128 GPUs?
(iv) What is the impact of our CPU-Adam and delay param-
eter update (DPU) on improving throughput, and does
DPU change model convergence?
6.1
Evaluation Methodology
Testbed. For the evaluation of model scale and throughput,
we conduct our experiments on a single DGX-2 node, whose
details are shown in Table 2. For the evaluation of throughput
scalability, we conduct experiments on 8 Nvidia DGX-2 nodes
connected together with InﬁniBand using a 648-port Mellanox
MLNX-OS CS7500 switch.
Workloads. For the performance evaluation, we focus on
evaluating GPT-2 [19] like Transformer based models [30].
We vary the hidden dimension and the number of Transformer
blocks to obtain models with a different number of parame-
ters. Note that scaling the depth alone is often not sufﬁcient
because it would make training more difﬁcult [12]. Table 3
shows the conﬁguration parameters used in our experiments.
For convergence analysis, such as the delayed parameter
update, we use GPT-2 [19] and BERT [6], both of which are
commonly used as pre-trained language models and have
demonstrated superior performance in many NLP tasks (e.g.,
natural language understanding and inference) than recur-
rent neural networks or convolutional neural networks. We
use BERT-large, same as the one from [6], which has 24-
layer, 1024-hidden, 16-heads, and 336M parameters. Similar
to [21, 28], we ﬁne-tune BERT on the Stanford Question
Answering Dataset (SQuAD) [1], which is one of the most
widely used reading comprehension benchmark [22]. Unless
otherwise stated, we follow the same training procedure and
hyperparameter settings as in [6,19].
Baseline. We compare the effectiveness of ZeRO-Ofﬂoad
with state-of-arts multi-billion parameter training solutions:
• PyTorch DDP: This is the existing PyTorch Transformer
implementation using DistributedDataParallel [14].
8


---

time
Stream 1:
Stream 2:
FWD
GPU
fp16 param (step N-1)
BWD
GPU
fp16 param (step N-1)
gradient offload
GPU->CPU
fp16 gradient 
(step N-1)
Step N
Step N+1
Step N - 1
fp32 param
->
fp16 param
step(N)
CPU->GPU
param swap
…
…
FWD
GPU
fp16 param (step N)
BWD
GPU
fp16 param (step N)
gradient offload
GPU->CPU
fp16 gradient 
(step N)
FWD
GPU
fp16 param (step N)
BWD
GPU
fp16 param (step N)
gradient offload
GPU->CPU
fp16 gradient 
(step N+1)
param update
fp32 param
fp32 momentum
fp32 variance
step(N)
CPU
param update
fp32 param
fp32 momentum
fp32 variance
step(N-1)
CPU
fp32 param
->
fp16 param
step(N)
CPU->GPU
param swap
Figure 6: Delayed parameter update during the training process.
Table 3: Model conﬁguration in evaluation.
# params
batch size
per GPU
MP setting
in ZeRO-Ofﬂoad
# layer
hidden size
1, 2 billion
32
1
20, 40
2048
4 billion
32
1
64
2304
6, 8 billion
16
1
53, 72
3072
10,11 billion
10,8
1
50,55
4096
12, 13 billion
4
1
60, 65
4096
15 billion
8
2
78
4096
20,40,60 billion
8
2
25,50,75
8192
70 billion
8
8
69
9216
• Megatron [28]: One of the current state-of-the-art multi-
billion parameter model training solutions, which em-
ploys model parallelism to train up to 8.3B parameter
models using 512 GPUs.
• L2L [18]: L2L enables training of deep Transformer
networks by keeping one Transformer block at a time in
GPU memory and only moves tensors in the upcoming
Transformer block into GPU memory when needed.
• ZeRO [21]: ZeRO extends data parallelism by eliminat-
ing memory redundancies across multiple GPUs, allow-
ing to train models up to 170B parameters with high
training throughput using 25 DGX-2 nodes. We refer
to the open-sourced implementation of ZeRO as ZeRO-
2. ZeRO-2 achieves the SOTA results for large model
training and is a strong baseline.
6.2
Experimental Results
6.2.1
Model scale
As an important step toward democratizing large model train-
ing, in this part, we ﬁrst test the largest trainable models on a
single GPU as well as 16 GPUs in a single DGX-2 node.
Single GPU.
The largest model can be trained using Py-
Torch DDP on a single GPU with 32GB memory is 1.4B,
before running out of memory, as shown in 7. Both Mega-
tron and ZeRO-2 do not increase the trainable model size on
a single GPU in comparison to PyTorch, because they both
utilize the aggregated GPU memory to ﬁt larger models. In
contrast, ZeRO-Ofﬂoad enables 13B model training on a sin-
gle GPU, which is more than 9X larger than using PyTorch,
Megatron, and ZeRO-2. This is mainly because of ZeRO-
Ofﬂoad’s strategy for maximizing the memory savings on
GPU by ofﬂoading expensive states such as optimizer states
and the majority of gradients to CPU memory. On the other
hand, L2L is able to train even larger models (e.g., 17B) on a
single GPU by frequently moving weights from unused layers
to CPU memory. However, the largest model size does not
increase when training L2L with multiple GPUs, which is
discussed next.
Multi-GPU in single DGX-2.
We further perform model
scale tests with 4 and 16 GPUs in a single DGX-2 node, re-
spectively. As shown, the maximum trainable model size stays
the same for both PyTorch and L2L, because both of them
do not handle memory redundancies in data parallelism. As
a result, their scalability is bounded by the model scale on a
single GPU. Both Megatron and ZeRO-2 support large model
training with more GPUs, but they cannot scale efﬁciently
beyond 15B parameters, even with 16 GPUs. Megatron sup-
ports larger models than ZeRO-2, because ZeRO-2 still incurs
memory redundancies on model weights. On the other hand,
ZeRO-Ofﬂoad easily enables training of up to 70B parameter
models by partitioning and ofﬂoading optimizer states and
gradients to CPU memory combined with model parallelism.
Overall, ZeRO-Ofﬂoad increases the model scale on a sin-
gle DGX-2 node by 50X, 4.5X, 7.8X, and 4.2X than using
PyTorch, Megatron, ZeRO-2, and L2L, respectively.
6.2.2
Training throughput
Single GPU.
Next, we compare the training throughput of
ZeRO-Ofﬂoad and L2L, for models with billion-scale param-
eters, on a single GPU. We do not include Megatron and
ZeRO-2 in this comparison, because both of them cannot
train models bigger than 1.4B parameters due to OOM. We
evaluate ZeRO-Ofﬂoad and L2L with the same training batch
size (e.g., 512) and same micro-batch sizes (shown in table 3),
with gradient accumulation enabled. We also disable delayed
parameter update in this experiment so that the comparison is
only from the system efﬁciency perspective. We evaluate the
performance improvement and its impact on the convergence
of delayed parameter update in Section 6.2.4.
Figure 8 shows that ZeRO-Ofﬂoad outperforms L2L by
14% on average (up to 22%) in throughput (TFLOPS). The
performance beneﬁt of ZeRO-Ofﬂoad comes from the follow-
ing two aspects. First, ZeRO-Ofﬂoad has a lower communica-
9


---

Figure 7: The size of the biggest model that
can be trained on single GPU, 4 and 16 GPUs
(one DGX-2 node).
Figure 8: The training throughput with Py-
Torch, L2L, and ZeRO-Ofﬂoad on a single
GPU with a batch size of 512.
Figure 9: The training throughput is com-
pared for w/o DPU and w/ DPU to GPT-2.
Batch size is set to 8.
tion cost between CPU and GPU than L2L. For a model with
M parameters, L2L requires 28M data communication vol-
ume between GPU and CPU, which is a sum of the weights,
gradients, and optimizer states of each layer of the model. As
analyzed in Sec. 4.1, the communication volume between
CPU and GPU memory in ZeRO-Ofﬂoad is 4M, which is
7x smaller than L2L. The reduced communication volume
signiﬁcantly mitigates the bottleneck from CPU-GPU commu-
nication. Second, compared with L2L, the parameter update
of ZeRO-Ofﬂoad happens on CPU instead of GPU, but our op-
timized CPU-Adam implementation achieves a quite compa-
rable parameter update performance than the PyTorch Adam
implementation on GPU (evaluated in Sec. 6.2.4). Therefore,
although the optimizer update on GPU in L2L is slightly
faster than the optimizer update on CPU in ZeRO-Ofﬂoad,
the communication overhead introduced by L2L leads to an
overall slower throughput than ZeRO-Ofﬂoad.
Multi-GPU in single DGX-2. Next, we compare the training
throughput of PyTorch, ZeRO-2, Megatron, ZeRO-Ofﬂoad
without model parallelism (w/o MP), and ZeRO-Ofﬂoad with
model parallelism (w/ MP) in one DGX-2 node. When using
MP, we use a MP degree that gives the best performance
for both baseline and ZeRO-Ofﬂoad. We use a total batch
size of 512 for all the experiments using a combination of
micro-batch per GPU and gradient accumulation. To get the
best performance for each conﬁguration, we use the largest
micro batch that it can support without OOM. We exclude
L2L [29] in this test because its implementation does not
support multi-GPU training.
Figure 10 shows the throughput per GPU results when train-
ing on multiple GPUs. We make the following observations:
• For 1B to 15B models, ZeRO-Ofﬂoad achieves the high-
est throughput and has up to 1.33X, 1.11X, 1.64X higher
speeds than PyTorch, ZeRO-2, and Megatron, respec-
tively. By ofﬂoading all the optimizer states to CPU
with low overhead, ZeRO-Ofﬂoad can train with larger
micro-batch sizes giving higher throughput.
• ZeRO-2 runs out of memory once the model size is be-
yond 8B due to lack of enough aggregate GPU memory
to store the model states on 16 GPUs. Instead, ZeRO-
Ofﬂoad scales to 13B, without model parallelism be-
cause it ofﬂoads optimizer states and the majority of
Figure 10: Training throughput with PyTorch, ZeRO-2, Megatron-
LM, ZeRO-Ofﬂoad without model parallelism and ZeRO-Ofﬂoad
with model parallelism.
gradients to CPU memory.
• When combined with model parallelism, ZeRO-Ofﬂoad
enables training up to 70B parameter models with more
than 30 TFLOPS throughput per GPU. In contrast, Mega-
tron supports only up to 15B parameter models before
running out of memory, using just model parallelism.
• Compared ZeRO-Ofﬂoad with ZeRO-2 and Megatron,
ZeRO-Ofﬂoad outperforms ZeRO-2 and Megatron in
throughput for 1–8B and 1–13B parameter models, re-
spectively. ZeRO-Ofﬂoad is faster than Megatron, be-
cause it eliminates frequent communication between
different GPUs and can train with larger micro batch
sizes. ZeRO-Ofﬂoad outperforms ZeRO-2 also due to
larger micro batch sizes.
6.2.3
Throughput Scalability
We compare the throughput scalability of ZeRO-2 and ZeRO-
Ofﬂoad3 on up to 128 GPUs in Figure 11 and make the fol-
lowing key observations: First, ZeRO-Ofﬂoad achieves near
perfect linear speedup in terms of aggregate throughput (green
line) running at over 30 TFlops per GPU (blue bars). Second,
from 1 to 16 GPUs, while ZeRO-2 runs out of memory, ZeRO-
Ofﬂoad can effectively train the model, turning the model
training from infeasible to feasible. Third, with 32 GPUs,
3We do not include comparison against Megatron because it consistently
performs worse than ZeRO-Ofﬂoad, as shown in Figure 10. Given the com-
munication overhead added by model parallelism, scaling out Megatron
training can not achieve higher throughput than ZeRO-Ofﬂoad even with
linear scalability.
10


---

Figure 11: Comparison of training throughput between ZeRO-
Ofﬂoad and ZeRO-2 using 1–128 GPUs for a 10B parameter GPT2.
ZeRO-Ofﬂoad slightly outperforms ZeRO-2 in throughput.
The improvement comes from additional memory savings on
GPU from ZeRO-Ofﬂoad, which allows training the model
with larger batch sizes that lead to increased GPU computa-
tion efﬁciency. Fourth, with more GPUs (such as 64 and 128),
ZeRO-2 starts to outperform ZeRO-Ofﬂoad, because both can
now run similar batch sizes, achieving similar computation
efﬁciency, whereas ZeRO-2 does not suffer from the addi-
tional overhead of CPU-GPU communication. In summary,
ZeRO-Ofﬂoad complements ZeRO-2 and enables large model
training from a single device to thousands of devices with
good computation efﬁciency.
6.2.4
Optimized CPU execution
A. CPU-Adam efﬁciency.
In this part, we evaluate our
Adam implementation against the PyTorch Adam on CPU.
Table 4 shows the optimizer execution time of the two im-
plementations for model parameters from 1 to 10 billion.
Compared to PyTorch (PT-CPU), CPU-Adam reduces the ex-
ecution time by over 5X for all the conﬁgurations and 6.4X
for the case with 1B parameters. The CPU-Adam optimizer
achieves high speedups by exploiting the instruction-level par-
allelism, thread-level parallelism, and the tile-based data copy
scheme (as shown in line 15 of Algorithm 1). Meanwhile,
although CPU-Adam has a slower speed than the PyTorch
Adam implementation on GPU (PT-GPU), the performance
gap is not very huge, and the CPU computation is not a bot-
tleneck of the training throughout.
B. One-step Delayed parameter update (DPU).
Figure 9
shows the comparison of the training throughput of GPT-2
with and without DPU. As shown, with DPU enabled, the
training achieves 1.12–1.59, updated times higher throughput
than without it, for a wide range of model sizes for a small
micro batch size of 8. This is expected because DPU allows
the optimizer updates to overlap with the next forward com-
putation such that the GPU does not have to be slowed down
by the CPU computation and CPU-GPU communication. But,
what about accuracy?
Convergence impact We study the convergence impact
of DPU on both GPT-2 and BERT. Figure 12 shows the
pre-training loss curves over 100K training iterations us-
ing PyTorch (unmodiﬁed GPT-2), and Figure 13 shows the
Table 4: Adam latency (s) for PyTorch (PT) and CPU-Adam.
#Parameter
CPU-Adam
PT-CPU
PT-GPU (L2L)
1 billion
0.22
1.39
0.10
2 billion
0.51
2.75
0.26
4 billion
1.03
5.71
0.64
8 billion
2.41
11.93
0.87
10 billion
2.57
14.76
1.00
loss curves of ﬁne-tuning Bert-large model on SQuAD using
ZeRO-Ofﬂoad without DPU, and ZeRO-Ofﬂoad with DPU.
In both cases, DPU is enabled after 40 iterations allowing the
training to stabilize in its early stage before introducing DPU.
We observe that the training curves of the unmodiﬁed GPT-
2 and ZeRO-Ofﬂoad w/o DPU are exactly overlapped, be-
cause ZeRO-Ofﬂoad w/o DPU performs only system opti-
mizations and does not alter training dynamics. On the other
hand, the training curve from ZeRO-Ofﬂoad with DPU con-
verges slightly slower at the very beginning of the training
(e.g., barely can be seen at 2K-5K iterations) and quickly
catches up after 5K iterations. For the remaining of the train-
ing, the training loss matches the original training until the
model converges.
For Bert-Large ﬁne-uning, we can see that although the
training losses are not exactly the same, they converge in the
same trend and are largely overlapped. Without changing any
hyperparameters, ZeRO-Ofﬂoad + DPU achieves the same
ﬁnal F1 score (92.8) as the baseline. From these results on
both GPT-2 pretraining, and Bert-Large ﬁne-tuning, we empir-
ically verify that DPU is an effective technique to improve the
training throughput of ZeRO-Ofﬂoad without hurting model
convergence and accuracy.The 1-step staleness introduced by
DPU is well tolerated by the iterative training process once
the model has passed the initial training phase.
Figure 12: The training loss
curve of unmodiﬁed GPT-2,
ZeRO-Ofﬂoad w/o DPU and
ZeRO-Ofﬂoad with DPU.
Figure 13: The ﬁne-tuning loss
curve of BERT, ZeRO-Ofﬂoad
w/o DPU and ZeRO-Ofﬂoad
with DPU.
7
Conclusions
We presented ZeRO-Ofﬂoad, a powerful GPU-CPU hybrid
DL training technology with high compute efﬁciency and near
linear throughput scalability, that can allows data scientists
to train models with multi-billion parameter models even
on a single GPU, without requiring any model refactoring.
We open-sourced ZeRO-Ofﬂoad as part of the DeepSpeed
11


---

library (www.deepspeed.ai) with the hope to democratize
large model training, allowing data scientist everywhere to
harness the potential of truly massive DL models.
12


---

References
[1] The Stanford Question Answering Dataset (SQuAD)
leaderboard.
https://rajpurkar.github.io/
SQuAD-explorer/.
[2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen
Krueger, Tom Henighan, Rewon Child, Aditya Ramesh,
Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christo-
pher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christopher
Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,
and Dario Amodei. Language models are few-shot learn-
ers, 2020.
[3] Yu Cao, Wei Bi, Meng Fang, and Dacheng Tao. Pre-
trained language models for dialogue generation with
multiple input sources, 2020.
[4] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos
Guestrin. Training deep nets with sublinear memory
cost. arXiv: Learning, 2016.
[5] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen,
Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, An-
drew Senior, Paul Tucker, Ke Yang, Quoc V. Le, and
Andrew Y. Ng. Large scale distributed deep networks.
In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Wein-
berger, editors, Advances in Neural Information Process-
ing Systems 25, pages 1223–1231. Curran Associates,
Inc., 2012.
[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. Bert: Pre-training of deep bidirec-
tional transformers for language understanding. arXiv
preprint arXiv:1810.04805, 2018.
[7] Aaron Harlap, Deepak Narayanan, Amar Phanishayee,
Vivek Seshadri, Nikhil Devanur, Greg Ganger, and Phil
Gibbons. Pipedream: Fast and efﬁcient pipeline parallel
dnn training, 2018.
[8] Mark Hildebrand, Jawad Khan, Sanjeev Trika, Jason
Lowe-Power, and Venkatesh Akella. Autotm: Automatic
tensor movement in heterogeneous memory systems
using integer linear programming. In Proceedings of the
Twenty-Fifth International Conference on Architectural
Support for Programming Languages and Operating
Systems, ASPLOS ’20, 2020.
[9] Chien-Chin Huang, Gu Jin, and Jinyang Li. Swapad-
visor: Pushing deep learning beyond the gpu memory
limit via smart swapping. In Proceedings of the Twenty-
Fifth International Conference on Architectural Support
for Programming Languages and Operating Systems,
ASPLOS ’20, page 1341–1355, New York, NY, USA,
2020. Association for Computing Machinery.
[10] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan
Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee,
Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng
Chen. Gpipe: Efﬁcient training of giant neural networks
using pipeline parallelism, 2018.
[11] Hai Jin, Bo Liu, Wenbin Jiang, Yang Ma, Xuanhua Shi,
Bingsheng He, and Shaofeng Zhao. Layer-centric mem-
ory reuse and data migration for extreme-scale deep
learning on many-core architectures. ACM Trans. Ar-
chit. Code Optim., 15(3), September 2018.
[12] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B.
Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec
Radford, Jeffrey Wu, and Dario Amodei. Scaling laws
for neural language models, 2020.
[13] Diederik P. Kingma and Jimmy Ba. Adam: A method
for stochastic optimization, 2014.
[14] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar,
Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith,
Brian Vaughan, Pritam Damania, and Soumith Chintala.
Pytorch distributed: Experiences on accelerating data
parallel training. Proc. VLDB Endow., 13(12):3005–
3018, 2020.
[15] Gaurav Mitra, Beau Johnston, Alistair Rendell, Eric
McCreath, and Jun Zhou. Use of simd vector opera-
tions to accelerate application code performance on low-
powered arm and intel platforms. pages 1107–1116, 05
2013.
[16] Nvidia.
Automatic Mixed Precision for Deep
Learning.
https://developer.nvidia.com/
automatic-mixed-precision, 2019.
[17] Xuan Peng, Xuanhua Shi, Hulin Dai, Hai Jin, Weiliang
Ma, Qian Xiong, Fan Yang, and Xuehai Qian. Capuchin:
Tensor-based gpu memory management for deep learn-
ing. In Proceedings of the Twenty-Fifth International
Conference on Architectural Support for Programming
Languages and Operating Systems, ASPLOS ’20, page
891–905, New York, NY, USA, 2020. Association for
Computing Machinery.
[18] Bharadwaj Pudipeddi, Maral Mesmakhosroshahi, Jin-
wen Xi, and Sujeeth Bharadwaj. Training large neural
networks with constant memory using a new execution
algorithm. June 2020.
[19] Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. Language models are
unsupervised multitask learners. OpenAI blog, 1(8):9,
2019.
13


---

[20] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei
Li, and Peter J. Liu. Exploring the limits of transfer
learning with a uniﬁed text-to-text transformer, 2020.
[21] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and
Yuxiong He. ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models. In International
Conference for High Performance Computing, Network-
ing, Storage and Analysis (SC), 2020.
[22] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. Squad: 100, 000+ questions for machine
comprehension of text. In Jian Su, Xavier Carreras, and
Kevin Duh, editors, Proceedings of the 2016 Conference
on Empirical Methods in Natural Language Processing,
EMNLP 2016, Austin, Texas, USA, November 1-4, 2016,
pages 2383–2392. The Association for Computational
Linguistics, 2016.
[23] Jie Ren, Jiaolin Luo, Kai Wu, Minjia Zhang, Hyeran
Jeon, and Dong Li. Sentinel: Efﬁcient Tensor Migration
and Allocation on Heterogeneous Memory Systems for
Deep Learning. In International Symposium on High
Performance Computer Architecture (HPCA), 2020.
[24] Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Ar-
slan Zulﬁqar, and Stephen W. Keckler. vdnn: Virtualized
deep neural networks for scalable, memory-efﬁcient neu-
ral network design. In The 49th Annual IEEE/ACM Inter-
national Symposium on Microarchitecture, MICRO-49,
2016.
[25] Corby Rosset. Turing-nlg: A 17-billion-parameter lan-
guage model by microsoft, 2020.
[26] Christopher J. Shallue, Jaehoon Lee, Joseph Antognini,
Jascha Sohl-Dickstein, Roy Frostig, and George E. Dahl.
Measuring the Effects of Data Parallelism on Neural
Network Training. Journal of Machine Learning Re-
search, 20, 2019.
[27] Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin
Tran, Ashish Vaswani, Penporn Koanantakool, Peter
Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff
Young, Ryan Sepassi, and Blake A. Hechtman. Mesh-
tensorﬂow: Deep learning for supercomputers. In Samy
Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen
Grauman, Nicolò Cesa-Bianchi, and Roman Garnett,
editors, Advances in Neural Information Processing Sys-
tems 31: Annual Conference on Neural Information Pro-
cessing Systems 2018, NeurIPS 2018, December 3-8,
2018, Montréal, Canada, pages 10435–10444, 2018.
[28] Mohammad Shoeybi, Mostofa Patwary, Raul Puri,
Patrick LeGresley, Jared Casper, and Bryan Catanzaro.
Megatron-lm: Training multi-billion parameter language
models using model parallelism. CoRR, abs/1909.08053,
2019.
[29] Roman Tezikov. PyTorch implementation of L2L exe-
cution algorithm. https://github.com/TezRomacH/
layer-to-layer-pytorch, 2020.
[30] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser,
and Illia Polosukhin.
Attention is all you need.
In
Isabelle Guyon, Ulrike von Luxburg, Samy Bengio,
Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan,
and Roman Garnett, editors, Advances in Neural Infor-
mation Processing Systems 30: Annual Conference on
Neural Information Processing Systems 2017, Decem-
ber 4-9, 2017, Long Beach, CA, USA, pages 5998–6008,
2017.
[31] G. Velkoski, M. Gusev, and S. Ristov. The performance
impact analysis of loop unrolling. In 2014 37th Interna-
tional Convention on Information and Communication
Technology, Electronics and Microelectronics (MIPRO),
pages 307–312, 2014.
[32] Oreste Villa, Mark Stephenson, David Nellans, and
Stephen Keckler. Buddy Compression: Enabling Larger
Memory for Deep Learning and HPC Workloads on
GPUs. In International Symposium on Computer Archi-
tecture, 2020.
[33] Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang
Li, Shuaiwen Leon Song, Zenglin Xu, and Tim Kraska.
Superneurons: Dynamic gpu memory management for
training deep neural networks. In Proceedings of the
23rd ACM SIGPLAN Symposium on Principles and
Practice of Parallel Programming, PPoPP ’18, page
41–53, New York, NY, USA, 2018. Association for Com-
puting Machinery.
[34] Junzhe Zhang, Sai-Ho Yeung, Yao Shu, Bingsheng He,
and Wei Wang. Efﬁcient memory management for gpu-
based deep learning systems. CoRR, abs/1903.06631,
2019.
[35] M. A. Zinkevich, M.Weimer, A. Smola, and L. Li. Par-
allelized Stochastic Gradient Descent. In International
Conference on Neural Information Processing Systems,
2010.
14
