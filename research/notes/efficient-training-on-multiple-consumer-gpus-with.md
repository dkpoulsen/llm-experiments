---
title: Efficient Training on Multiple Consumer GPUs with
id: efficient-training-on-multiple-consumer-gpus-with
tags:
- cmp170hx-training-feasibility-a0d4fc
- pipeline-parallelism
- cpu-offloading
- consumer-gpu-training
- pcie-fabric
- no-p2p
- zero-offload
created: '2026-10-27T10:12:04.685153Z'
updated: '2026-10-27T19:00:31.102922Z'
source: https://arxiv.org/abs/2604.27085
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:04.684087Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'RoundPipe (Luo et al., Tsinghua, SIGOPS ATC 2026, arXiv 2604.27085): pipeline
  parallelism + CPU offloading for LLM fine-tuning on consumer GPUs. Directly relevant
  to the CMP 170HX topology question because RoundPipe uses NO GPU peer-to-peer communication
  at all — it relies entirely on PCIe host-to-device transfers under its Computation
  Dispatch Paradigm, and still reaches 1.48-2.16x the throughput of the fastest baselines
  (ZeRO-2, FSDP, ZeRO-Infinity, Megatron-PP, Megatron-TP, Mobius) on 8x RTX 4090 (PCIe
  4.0, 32 GB/s) for 1.7B-32B models, and 76%+ of the same frameworks on 8x A800. It
  quantifies the DDP/ZeRO penalty on this class of fabric: DeepSpeed spends ~70% of
  training time on communication on consumer GPU servers, and Megatron-TP''s throughput
  ''becomes impractical under PCIe''. Memory anchors: an 8B model needs 128 GB for
  model states alone (vs 24 GB per 4090), host memory footprint is ~20x parameter
  size, activation recomputation is 2.37-5.75x faster than reloading activations over
  PCIe, and max trainable sequence length is independent of GPU count (73K/49K/39K/28K/31K
  for the five models) because stage-boundary activations live in host DRAM. Bubble
  ratio falls below 4.5% with asynchronous optimizer updates; near-linear scaling
  from 1 to 8 GPUs; LoRA fine-tunes Qwen3-235B at 31K sequence length on 24 GB cards.'
raw_file: raw/efficient-training-on-multiple-consumer-gpus-with.pdf
doi: arXiv:2604.27085
utility_score: 15.0
---

Efficient Training on Multiple Consumer GPUs with
RoundPipe
Yibin Luo, Shiwei Gao, Huichuan Zheng, Youyou Lu, Jiwu Shu∗
Tsinghua University
Abstract
Fine-tuning Large Language Models (LLMs) on consumer-
grade GPUs is highly cost-effective, yet constrained by lim-
ited GPU memory and slow PCIe interconnects. Pipeline
parallelism combined with CPU offloading mitigates these
hardware bottlenecks by reducing communication overhead.
However, existing PP schedules suffer from an inherent limita-
tion termed the weight binding issue. Binding uneven model
stages (e.g., the LM head is large) to GPUs limits the pipeline’s
throughput to that of the GPU with the heaviest load, leading
to severe pipeline bubbles.
In this paper, we propose RoundPipe, a novel pipeline sched-
ule that breaks the weight binding constraint on consumer
GPU servers. RoundPipe treats GPUs as a pool of stateless
execution workers and dynamically dispatches computation
stages across devices in a round-robin manner, achieving
a near-zero-bubble pipeline. To ensure training correctness
and system efficiency, RoundPipe integrates a priority-aware
transfer scheduling engine, a Layerwise Model Consistency
Protocol, and an automated layer partitioning algorithm. Evalu-
ations on an 8× RTX 4090 server demonstrate that RoundPipe
achieves 1.48–2.16× speedups over state-of-the-art baselines
when fine-tuning 1.7B to 32B models. Remarkably, Round-
Pipe enables LoRA fine-tuning of the Qwen3-235B model
with 31K sequence length on a single server.
RoundPipe has been open-sourced as a Python library.
Project Page: https://thustorage.github.io/RoundPipe/
CCS Concepts: • Computing methodologies →Parallel
algorithms.
Keywords: LLM training, Pipeline parallel, CPU offloading
ACM Reference Format:
Yibin Luo, Shiwei Gao, Huichuan Zheng, Youyou Lu, Jiwu Shu. 2026.
Efficient Training on Multiple Consumer GPUs with RoundPipe.
In The 2026 ACM SIGOPS Annual Technical Conference (SIGOPS
ATC 2026), November 15-18, 2026, Hong Kong, China. ACM, New
York, NY, USA, 15 pages. https://doi.org/10.1145/3828523.3857063
∗Jiwu Shu is the corresponding author (shujw@tsinghua.edu.cn).
This work is licensed under a Creative Commons Attribution 4.0 International
License.
SIGOPS ATC 2026, Hong Kong, China
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2818-1/2026/11
https://doi.org/10.1145/3828523.3857063
1
Introduction
Fine-tuning Large Language Models (LLMs) has become the
cornerstone of modern AI applications, enabling open-source
foundation models to master complex reasoning and domain-
specific tasks [31, 39]. Unlike pre-training from scratch, fine-
tuning requires significantly less computation [7]. For such
workloads, consumer-grade GPUs offer a compelling alterna-
tive. The NVIDIA RTX 4090 delivers computation capabilities
comparable to those of datacenter A100 GPUs, while costing
roughly 80% less. Efficiently fine-tuning massive LLMs on
consumer GPU servers is vital for democratizing AI for small
companies and researchers.
However, consumer-grade GPUs impose two hardware
constraints that restrict model scalability and training effi-
ciency. (1) Limited memory capacity: The VRAM of a typical
consumer-grade GPU falls short of training demands. For
instance, training an 8B model needs 128 GB for model states
alone [34], far larger than 24GB for NVIDIA RTX 4090
or 32GB for NVIDIA RTX 5090. Furthermore, expanding
context windows for complex tasks [6, 44] (e.g., reasoning,
video generation) also inflates activation memory. (2) Slow
inter-GPU communication: Consumer-grade GPUs use PCIe
interconnects, offering less than 20% of NVLink bandwidth.
This physical limitation is further compounded by root com-
plex contention in PCIe topologies [10, 18].
To alleviate VRAM constraints during consumer-grade
GPU training, existing methods commonly employ CPU
offloading [35, 37] to shift parameters, optimizer states, and
activations to host memory, often combined with activation
recomputation [4, 19] to further reduce the memory footprint.
When scaling out across multiple consumer-grade GPUs
connected via PCIe, data parallelism techniques like ZeRO-
Infinity [35] distribute model states across devices. However,
they require frequent collective communications at each layer
to reconstruct parameters. Prior studies [10] indicate that
these communications can consume up to 70% of the training
time. To mitigate such PCIe bottlenecks, Mobius [10], as the
state-of-the-art system, integrates pipeline parallelism (PP)
with offloading, replacing extensive multi-GPU collectives
with more efficient P2P communication. Mobius employs a
looped-BFS schedule [20] and supports offloading by loading
and evicting model weights at the granularity of individual
pipeline stages.
However, the looped-BFS schedule, as well as other pipeline
schedules designed for datacenter training, suffers from sig-
nificant pipeline bubbles. In these schedules, the weights of a
arXiv:2604.27085v2  [cs.DC]  18 Oct 2026


---

Forward
Recompute
Backward
Structural Bubbles
Imbalance Bubbles
GPU1
GPU2
GPU3
GPU4
GPU1
GPU2
GPU3
GPU4
GPU1
GPU2
GPU3
GPU4
F1
F2
F3
F4
F5
F6
F7
F8
F9
F10
F11
F12,13
B12,13
B11
B10
B9
B8
B7
B6
B5
B4
B3
B2
B1
F1,2,3
F4,5,6
F7,8,9
F10,11,12
FB13
B12
B11
B10
B9
B8
B7
B6
B5
B4
B3
B2
B1
F1,2,3
F4,5,6
F7,8,9
F10,11,12
FB13
B12
B10
B9
B8
B7
B6
B5
B4
B3
B2
Time
Time
B1
(a) Looped BFS schedule (standard partition)
(c) RoundPipe schedule
B11
F1
F2
F3
F4
F5
F6
F7
F8
F9
F10
F11
B12
B11
B10
B9
B8
B7
B6
B5
B4
B3
B2
B1
Time
(b) Looped BFS schedule (flexible partition)
F12
F13
B13
Figure 1. Looped BFS schedule and RoundPipe schedule when training a 12-layer model with a language model head (layer
13) on 4 GPUs. F/B denotes forward/backward, and numbers indicate the index of layers contained in the stage. Looped BFS
schedule processes 8 microbatches at a time while RoundPipe processes them in two rounds.
stage (and the corresponding forward and backward compu-
tation) are fixed to a specific GPU. We term this the weight
binding issue. No matter how the pipeline is partitioned (stan-
dard looped partitioning in Figure 1 (a) or flexible partitioning
in Figure 1 (b)), GPUs have to wait for the slowest stage (e.g.,
the LM Head), causing structural bubbles between stages or
imbalanced bubbles within stages. In the case of training the
Llama-3.1-8B model, pipeline bubbles can reach up to 30%.
Our core observation is that CPU offloading offers an
opportunity to break the weight binding issue. In the offload
setting, GPUs can be viewed as a stateless execution worker
pool. Master weights reside in host DRAM; during the forward
pass, the weights of each stage are transferred to a GPU for
computation, and the resulting activations are stored back to
host DRAM; during the backward pass, the same stage can
be dispatched to a different GPU dynamically.
Based on this insight, we propose RoundPipe, whose
pipeline schedule is shown in Figure 1 (c). RoundPipe lever-
ages a round-robin task dispatching schedule and assigns
stages to different GPUs in a sequential manner. This allows
RoundPipe to freely select an optimal number of pipeline par-
titions without imbalance bubbles, while existing schedules
are limited to multiples of #GPUs. To pipeline the forward
and backward stages without bubbles, RoundPipe adopts an
asymmetric stage splitting strategy to balance execution time
across stages. For example, we combine three layers into a
forward stage or one layer into a backward stage. 1 This also
removes iteration boundaries, allowing the pipeline to flow
from one iteration to the next without structural bubbles when
performing gradient accumulations or using asynchronous
optimizers.
1Due to activation recomputation, their execution times are equal.
Fully unleashing the potential of RoundPipe requires tack-
ling several challenges. First, massive amounts of data must
continuously move between the host and GPUs. To prevent
large parameter transfers from blocking critical-path activation
transfers and the compute stream, RoundPipe introduces a
priority-aware transfer scheduling engine, packing parameter
transfers into idle windows between critical-path activation
transfers, ensuring seamless overlap with computation.
Second, RoundPipe supports executing optimizer updates
asynchronously to hide latency. This introduces race condi-
tions over host-resident data, where the GPU and the CPU-side
optimizer simultaneously read and write this data. To guaran-
tee data consistency without reintroducing pipeline-stalling
barriers, RoundPipe implements a Layerwise Model Consis-
tency Protocol that enforces strict execution ordering at the
individual layer level.
Finally, to eliminate manual tuning and maintain near-
optimal load balancing across asymmetrically partitioned
stages, RoundPipe features an automated stage-splitting al-
gorithm that efficiently (with 𝑂(𝐿3) complexity) computes a
pipeline partition.
We evaluate RoundPipe across 8×RTX 4090 and 8×A800
SXM servers using 1.7B to 235B models [25, 30, 43]. Com-
pared with state-of-the-art baselines, RoundPipe delivers: (1)
on 4090s, up to 2.16× higher throughput and 7.3× longer
sequences; (2) on A800s, matching their throughput for small
models, with up to 1.47× speedups and 5.6× longer sequences
on large ones. Remarkably, RoundPipe is the only system
capable of LoRA fine-tuning a 235B MoE model on 24 GB
GPUs. Furthermore, its 4090 throughput reaches at least 76%
of existing A800 solutions across all models, bridging the
performance gap between consumer and data-center hardware.


---

In summary, the main contributions of this paper are as
follows:
• We identify limitations of pipeline parallelism on consumer
GPUs and propose RoundPipe as a solution.
• We propose a set of system designs, including a multi-
stream architecture for data overlap and a Layerwise Model
Consistency Protocol, alongside a stage-splitting algorithm
to fully unleash the potential of RoundPipe.
• We evaluate RoundPipe comprehensively to demonstrate
its effectiveness and training efficiency compared to state-
of-the-art solutions.
2
Background and Motivation
Consumer-grade GPUs offer high computational power at a
fraction of the cost of datacenter accelerators. However, their
limited hardware memory capacity and low-bandwidth PCIe
interconnects pose bottlenecks for training large models. In
this section, we first review the memory pressure and existing
mitigation techniques (§2.1). We then discuss multi-GPU
parallelism under the offloading setting (§2.2), highlighting
the limitations of current pipeline schedules in resolving
structural and imbalance bubbles (§2.3).
2.1
Memory-Consuming Training and Mitigations
On memory-constrained consumer GPUs, one primary chal-
lenge in training or fine-tuning large models is memory pres-
sure. This pressure stems from the massive footprint of static
model states (including parameters, gradients, and optimizer
states) and runtime intermediate activations.
First, the continuous scaling of model size to unlock emer-
gent capabilities [17] increases the basic memory required for
model states. Under standard mixed-precision [26] training
with the Adam optimizer [34], a Φ-parameter model state
occupies 16Φ bytes. For a 32B model, the basic memory
usage has reached 512GB.
Second, the continuous expansion of context windows
for complex tasks like reasoning [6], video generation [44],
and document-level understanding [1] drastically inflates the
runtime activation that must be stored for backpropagation.
The activation footprint per transformer layer increases linearly
with input sequence length. For example, training a LLaMA-
3.1-8B [25] model with a single 16k-token sequence generates
68 GB of activations from all layers. Furthermore, the use
of micro-batches (which linearly scale up the activation) is
common during distributed training.
Several techniques can mitigate the training memory pres-
sure on consumer-grade GPU servers. We will review repre-
sentative approaches below and discuss their trade-offs.
2.1.1
Activation Recomputation. Activation recomputa-
tion or gradient checkpointing [4] avoids storing intermediate
activations during the forward pass. Instead, only the input to
each transformer layer is retained. The remaining activations
are recomputed on-the-fly by re-executing the forward pass
Qwen3
1.7B
Llama-3.1
8B
gpt-oss
20B
Qwen3
32B
Qwen3
235B
0
10
20
30
40
50
Time (ms)
Recompute vs Activation Reload Time on 4090
Recompute Time
Activation Reload Time
Figure 2. Time of recomputing and reloading activations of a
transformer layer.
with the stored layer input before the backward pass of every
layer. This reduces the intermediate activations’ size to 2𝑠𝑏ℎ
bytes per layer [19].
While the extra computation may seem expensive, it is
often more efficient than swapping activations to host memory
over PCIe. Figure 2 compares the execution time of activation
recomputation and reloading activations from host memory
on 4090. Recomputing a transformer layer is 2.37× ∼5.75×
faster than reloading activations from host memory. Hence, the
reloading overhead cannot even be mitigated by layer-by-layer
computation-transmission overlap. Activation recomputation
has therefore become a common practice in large-scale train-
ing. It is favored by mainstream LLM training frameworks [51],
and is employed even on datacenter clusters with sufficient
GPU memory [16, 24]. In this paper, we adopt full activation
recomputation as a basic assumption. We also offload the
checkpointed activation to host DRAM.
2.1.2
Weight and Optimizer Offloading. To reduce the
static memory footprint of model states, prior work has pro-
posed to offload optimizer states, gradients, and often param-
eters to host memory, and to execute the optimizer on the
CPU [35, 37]. Model weights are transmitted to the GPU layer
by layer for gradient computation, and the resulting gradients
are sent back to the host memory for optimizer updates. A
host-side optimizer step is typically slow for large models, 9.6s
for a 32B model [37]. Since the execution time of the CPU
optimizer is comparable to the scale of GPU computation, syn-
chronous execution of both reduces training efficiency. Prior
work has therefore adopted staleness-1 asynchronous updates.
Training iteration 𝑇+1 will read the weights produced after
iteration 𝑇−1, while the CPU applies iteration-𝑇gradients
to model weights in the background [21, 37]. These works
demonstrate that one-step staleness does not harm conver-
gence, making asynchronous optimizer a practical option for
memory-constrained training systems.
2.2
Parallelism with Offloading
Given the limited PCIe interconnect of consumer GPUs, an-
other critical factor for training efficiency is how to parallelize
across multiple consumer GPUs.


---

DP-based parameter offloading. A simple approach is to
combine data parallelism with host memory offload. Deep-
Speed ZeRO-Offload [37] and ZeRO-Infinity [35] partition
model states across data-parallel ranks, and swap them to
CPU memory or NVMe storage.
However, as a data-parallel approach, ZeRO requires ex-
changing the full set of parameters between every GPU for
each forward and backward computation. It is usually com-
pleted through a high volume of collective communications
like all-gather. On consumer-grade servers with low PCIe
bandwidth, these communications become the major perfor-
mance bottleneck. A previous study shows that DeepSpeed
spends about 70% of training time on communication on con-
sumer GPU servers [10], limiting the efficiency of multi-GPU
scaling on cost-effective hardware.
PP-based parameter offloading. To overcome the com-
munication bottleneck inherent in data-parallel offloading,
Mobius [10] proposes a pipeline-parallel approach to of-
floaded training. It partitions a model’s layers into pipeline
stages and assigns each stage to a different GPU. These stages
are stored in DRAM, and each stage’s weights are loaded to
its assigned GPU before executing it. This reduces communi-
cation traffic, as only activations and gradients pass between
GPUs. However, Mobius inherits the pipeline bubble problem
from existing pipeline parallelism.
2.3
Motivation: Pipeline Schedules and Bubbles
Pipeline parallelism typically suffers from two types of
performance-degrading bubbles. As shown in Figure 1 (a),
Structural bubbles arise from the data-dependency of forward
and backward propagation under balanced stages. Imbalance
bubbles arise when some stages run longer than others, forcing
dependent stages to stall. Existing pipeline schedules strug-
gle to mitigate both types of bubbles simultaneously when
training large models.
Structural Bubbles. Structural bubbles come from data
dependencies in pipeline execution. Non-looped schedules
such as GPipe [14] and 1F1B [13] assign one stage to each
GPU. The first stage starts first and finishes last, so some GPUs
are idle at the beginning and end of the iteration.Given a model
split into 𝑆stages and trained on 𝑀micro-batches, this results
in a pipeline bubble of
𝑆−1
𝑀+𝑆−1, which can be substantial when
𝑀is not much larger than 𝑆.
Zero-Bubble schedules [33] attempt to fill these idle slots
by reordering computations. However, they require holding
activations for extended periods. This conflicts with activation
recomputation and leads to prohibitive memory consumption
for large-scale models. Looped schedules (e.g., Interleaved
1F1B [29], Looped BFS [20]) offer a more memory-efficient
alternative. They assign an equal number of stages to each
GPU, creating a finer-grained pipeline with a total of 𝑆= 𝑣𝑁
stages across all 𝑁GPUs. This reduces the bubble ratio to
Qwen3
1.7B
Llama-3.1
8B
gpt-oss
20B
Qwen3
32B
Qwen3
235B
0
10
20
30
Bubble Ratio (%)
Ideal vs Real Bubble Ratio
Ideal
Interleaved-1F1B
Looped-BFS
Figure 3. Bubble ratio of Looped schedules under ideal
balanced partition and real-world imbalanced partition on 8
GPUs. Real-world bubble ratios are collected in §5.8.1
approximately
𝑁·(𝑁−1)
𝑆·𝑀+𝑁·(𝑁−1) , which decreases as the number
of stages grows.
Imbalance Bubbles. Theoretical analyses of pipeline bub-
bles assume uniform stage execution times, an assumption
frequently violated by real-world models. In practice, stage
latencies diverge significantly; the slowest bottleneck stage
stalls subsequent operations, inducing imbalance bubbles
within the pipeline. Consequently, this results in a significant
disparity between the actual and ideal bubble ratios for the
looped schedule, as shown in Figure 3.
Limitations. Crucially, existing schedules face a dilemma
because they force the total stage count to be an exact inte-
ger multiple of the GPU count. Coarse-grained partitioning
with fewer stages (e.g., GPipe) incurs high structural bub-
bles. Conversely, fine-grained partitioning with more stages
(e.g., looped schedules) leads to severe load imbalance: as
each stage contains fewer layers, compute-heavy components,
such as the LM head, introduce significant inter-stage imbal-
ance regardless of where they are assigned. Consequently, as
demonstrated in Figure 3, the overall bubble ratio can reach
up to 30% in current pipeline schedules.
3
Introducing RoundPipe
In this section, we introduce RoundPipe. By leveraging the
opportunities presented by offloading, we decouple pipeline
stages from specific physical GPUs and propose a novel Com-
putation Dispatch Paradigm (§3.1). Building on this foun-
dation, we employ round-robin dispatch and asymmetric
splitting to synthesize a new pipeline schedule supporting an
arbitrary number of stages (§3.2). We analyze the bubble ratio
of this pipeline and the communication cost of Computation
Dispatch Paradigm (§3.3).
3.1
Computation Dispatch Paradigm
Strawman: flexible pipeline partition. Recall that existing
pipeline schedules require partitioning stages into integer
multiples of the number of devices 𝑆= 𝑣𝑁, resulting in
imbalance bubbles (as shown in Figure 1 (a)). An intuitive
approach to reduce the imbalance bubble is to apply a flexible


---

stage partitioning (such as 13 stages in Figure 1 (b)). This
approach can reduce imbalance bubbles by reducing the
execution time imbalance between layers.
However, this approach increases structural bubbles. In
Figure 1 (b), GPU 1 is assigned more stages, while GPUs 2-4
have fewer stages. Unfortunately, the weights (and correspond-
ing computations) of a stage in existing pipeline schedules
are fixed to specific GPUs and do not move during training.
Therefore, both the forward and backward passes of a stage
reside on the same physical device. Since each GPU can
only process its pre-assigned and bound stages, GPUs 2-4
are underutilized, creating structural bubbles between their
forward and backward passes when waiting for data from the
busiest GPU (GPU 1).
Opportunity: computation dispatching. We find that of-
floading model states to the CPU presents a unique opportunity
to increase the pipeline schedule efficiency. In PP-based of-
floading, the model states are offloaded to host memory and
transferred to the GPU on demand for execution. It is no
longer necessary to bind a stage to a fixed GPU. This allows
us to dispatch the stage computation to an underutilized GPU
freely. Intuitively, this incurs no additional communication
cost; since weight transfers between the CPU and GPU are
already necessary, this method essentially reassigns the target
GPU for these weights and their associated computations. We
evaluate its cost in §3.3.
Thus, we propose the Computation Dispatch Paradigm.
Model states and activations reside on the host, and computa-
tions (with the corresponding model states and activations)
are dispatched to GPUs for execution. In this way, any GPU
can execute any stage as soon as the requisite data is ready.
By doing so, we can execute a flexible pipeline with a small
structural bubble, achieving a simultaneous reduction in both
imbalance and structural bubbles.
3.2
RoundPipe Schedule
We concretize the idea of computation dispatch with the
RoundPipe schedule (Figure 1c). RoundPipe incorporates a
round-robin stage dispatch pattern and an asymmetric stage
splitting strategy.
Round-robin dispatch. Like looped schedules, RoundPipe
distributes stages across GPUs in a round-robin fashion;
however, it unifies both forward and backward stages into
a single continuous dispatch sequence. RoundPipe divides
the 𝑀micro-batches into multiple rounds, processing 𝑀𝑅
(𝑀𝑅≥𝑁) micro-batches per round on 𝑁GPUs. Within each
round, the 𝑆𝑓forward stages are followed by the 𝑆𝑏backward
stages, forming a linear sequence of 𝑆= 𝑆𝑓+ 𝑆𝑏stage slots.
Stage slot 𝑖(counting from 0 across the concatenated forward-
then-backward sequence) is dispatched to 𝐺𝑃𝑈(𝑔0+𝑖) mod 𝑁,
where 𝑔0 is the starting GPU index for the round. Each GPU
executes all 𝑀𝑅micro-batches of the current round for its
assigned stage before the next stage slot is dispatched.
Optim T
Iter T-1
GPUs
CPU
Iter T
Iter T+1
Optim T-2
Optim T-1
Time
Figure 4. Asynchronous optimizer update in RoundPipe.
Between rounds, the dispatch seamlessly resumes where
the previous round left off. That is, the starting index updates
to 𝑔0 ←(𝑔0 + 𝑆) mod 𝑁, assigning the first stage of the new
round to the GPU next in line. This dispatch logic ensures a
continuous, zero-bubble flow of stages across rounds. Stage
dispatch will repeat 𝑅= 𝑀/𝑀𝑅rounds until all 𝑀micro-
batches have completed their forward and backward passes.
Asymmetric stage splitting. The forward pass of a trans-
former layer is faster than its backward pass with recomputa-
tion. The symmetric stage splitting adopted by conventional
pipeline parallelism enforces identical layer partitions across
both passes, creating idle bubbles at the transition boundary
between the faster forward and slower backward stages. In
contrast, RoundPipe maintains separate partitions for forward
and backward passes, which enables a balanced partition that
equalizes per-stage execution time across the entire forward-
to-backward sequence.
To formalize this asymmetric partitioning, RoundPipe
defines (𝐹1, 𝐹2, . . . , 𝐹𝑆𝑓, 𝐵1) as the forward partition, where
𝐹𝑖is the number of consecutive layers in forward stage 𝑖;
and defines (𝐵1, 𝐵2, . . . , 𝐵𝑆𝑏) as the backward partition, where
𝐵𝑗is the number of consecutive layers in backward stage
𝑗. To reduce redundant computation, RoundPipe fuses the
forward and backward passes for the 𝐵1 layers. In this fused
stage, these forward computations serve as the recomputation
required by the backward pass, saving one forward pass’s
worth of computation for these layers. All stages should have
approximately equal computation time across both directions
and satisfy Í
𝑖𝐹𝑖+ 𝐵1 = 𝐿, Í
𝑗𝐵𝑗= 𝐿, where 𝐿is the number
of model layers.
By breaking the symmetry between forward and back-
ward partitions, asymmetric splitting bridges the gap between
the forward and backward phases, eliminating RoundPipe’s
pipeline bubbles at phase boundaries.
Supporting asynchronous optimizer. RoundPipe natively
supports staleness-1 asynchronous optimizer update (§2.1.2).
RoundPipe’s round-robin dispatch allows the pipeline to flow
from one iteration to the next without structural bubbles. Fig-
ure 4 presents how RoundPipe works with an asynchronous
optimizer. The iteration 𝑇continues the round-robin assign-
ment from where iteration 𝑇−1 left off. With asynchronous
updates enabled, the warm-up and cool-down bubbles that
appear at iteration boundaries are thereby eliminated.


---

50
100
150
200
250
Batch Size
0K
5K
10K
15K
20K
25K
30K
Operational Intensity (FLOPS/Byte)
4090
A800
5090
H100
Qwen3-1.7B
Llama-3.1-8B
GPT-Oss-20B
Qwen3-32B
Qwen3-235B
Figure 5. Operational intensity vs. batch size for representative
dense and MoE models at 2048 sequence length. Horizontal
lines mark the ridge-point OI for each GPU.
3.3
Benefits and Cost Analysis
Bubble Analysis. We now quantify the pipeline bubble
ratio of the RoundPipe schedule. Let 𝑡denote the execution
time per micro-batch for each stage. With the stage count
𝑆= 𝑆𝑓+𝑆𝑏, the workload of all 𝑀micro-batches costs 𝑀·𝑆·𝑡
GPU time. The bubble of RoundPipe comes from the startup
cost of filling the pipeline in the first round and the cool-down
cost of draining the pipeline after the last round, which are
symmetric and together contribute 𝑁· (𝑁−1) · 𝑡GPU time.
The total GPU time is therefore (𝑀· 𝑆+ 𝑁· (𝑁−1)) · 𝑡and
the bubble ratio is
𝑁·(𝑁−1)
𝑀·𝑆+𝑁·(𝑁−1) . Although RoundPipe shares
the same ratio formula with looped schedules (Interleaved
1F1B, Looped BFS), the number of stages in RoundPipe is
the sum of forward and backward stages, which is around
4
3× larger than looped schedules. 2 Thus, the bubble ratio
of RoundPipe is smaller than that of looped schedules. In
addition, RoundPipe’s flexible stage partitioning allows it to
achieve better time balance and therefore smaller imbalance-
induced bubbles than looped schedules.
Communication Cost. RoundPipe employs Computation
Dispatch Paradigm to dispatch stage computations to GPUs,
thereby requiring transferring weights and activations between
the GPU and the CPU memory. We conduct a roofline analy-
sis [46] to evaluate the PCIe communication cost during this
dispatch process.
Figure 5 presents the operational intensity (OI) of the for-
ward pass within a single transformer layer for multiple models.
When the OI of a workload exceeds the GPU’s ridge point,
it indicates that the workload is compute-bound, meaning
that data transfer latency can be fully hidden by computation.
For dense models, OI exceeds the ridge point at batch size
𝐵= 8, the smallest batch size for 8-GPU pipeline parallel
2To balance forward and backward stage runtime, RoundPipe typically
assigns 3 layers to 1 forward stage. Looped schedules have at most #layers
(𝐿) stages, while RoundPipe has 𝐿
3 forward stages and 𝐿backward stages
( 4
3𝐿in total).
Model
Stage Partition (§4.4)
GPU Worker 1
Model Master Copy
(Low Precision)
Optimizer Worker
Model Optimizer Copy
(High Precision)
Dispatch
step()
Data Transfer (§4.2)
Consistency
Protocol (§4.3)
…
GPU Worker N
Figure 6. RoundPipe system overview.
training. For MoE models, the ridge point is crossed when
𝐵< 100, well within the typical training regime. Furthermore,
the backward pass (with activation recomputation) has ∼3×
the forward FLOPS but ∼2× the data movement, so its OI is
higher than the forward pass’s.
This proves that model training under Computation Dis-
patch Paradigm is compute-bound. Therefore, Computation
Dispatch Paradigm is a throughput-preserving method that un-
locks a larger scheduling search space and scales up trainable
model parameters and sequence lengths. Based on this, Round-
Pipe implements an efficient communication-computation
overlap (§4.2) and verifies its minimal overhead nature through
our evaluation in §5.3.
4
Design and Implementation
The previous section presented RoundPipe’s scheduling algo-
rithm in the abstract. Realizing this abstraction as a practical
training framework involves mapping the logical pipeline to
physical hardware efficiently. §4.1 introduces the overall sys-
tem architecture and workflow, followed by the core technical
challenges. Subsequent subsections detail how RoundPipe
addresses data transfer overlap (§4.2), parameter consistency
(§4.3), and stage partitioning (§4.4).
4.1
RoundPipe Overview
RoundPipe adopts a single-controller architecture inspired by
Ray [27] and veRL/HybridFlow [40], separating the control
plane, which manages task scheduling and ordering, from the
data plane, which handles hardware-level execution and device
data transfers. As shown in Figure 6, RoundPipe consists of
a controller, an optimizer worker, and one GPU worker for
each GPU. The RoundPipe controller transparently maps the
computation onto the available hardware via GPU workers,
and handles asynchronous optimizer updates on the optimizer
worker.
4.1.1
RoundPipe Workflow. Users write a sequential train-
ing script using two APIs. During each RoundPipe invocation,
the user’s thread acts as the controller to orchestrate all activity,
directing per-device GPU workers and the optimizer worker
that execute concurrently.


---

Stage 1
Stage 2
MB1
MB2
MB1
MB2
Parameter
Upload
Activation
Upload
Compute
Time
head-of-line
blocking
Figure 7. Illustrated example of simple compute-transfer
overlap on two consecutive stages with two microbatches
(MB) each. Each color represents one stage.
forward_backward(). From the caller’s perspective, this is
identical to a standard PyTorch forward and backward pass. It
accepts input tensors, returns a scalar loss, and accumulates
gradients in place. Internally, RoundPipe runs a complete
pipelined execution across all available GPUs. The controller
constructs micro-batches from the inputs and dispatches the
stage to a GPU worker in round-robin order. The GPU workers
execute in parallel with the controller and with one another,
handling the data movement and computation asynchronously.
step(). This API dispatches gradient post-processing and
weight updates (e.g., gradient scaling, clipping, and the op-
timizer step) to the optimizer worker. The optimizer worker
applies these updates to the full-precision optimizer copy of
the model, while the GPU workers concurrently compute the
next iteration using the master copy without interference.
4.1.2
Challenges. There are several software-design chal-
lenges that, if handled carelessly, would introduce subtle
performance and correctness bugs:
• Data Transfer Overlap: Under the Computation Dispatch
Paradigm, multiple kinds of data move between the host
and GPUs continuously. As shown in Figure 7, a simple
overlap strategy suffers from head-of-line blocking, where
large parameter/gradient transfers delay the critical-path
activation transfers for the current stage. §4.2 addresses
this via a priority-aware transfer scheduling engine that
packs non-critical data into the idle windows of critical-path
communications.
• Asynchronous Consistency: Operating concurrently on the
optimizer copy and the master copy of the model requires
bridging the gap between them. The system must propagate
updates between these two copies to preserve staleness-1
asynchronous optimizer update semantics without stalling
pipeline execution. §4.3 addresses this via a Layerwise
Model Consistency Protocol.
• Pipeline Stage Partition: To balance the load across the
pipeline without manual tuning, the system requires an
efficient mechanism for partitioning layers automatically.
However, a naive search for the optimal partition incurs an
unacceptable exponential time complexity. §4.4 addresses
this via an 𝑂(𝐿3) two-stage partitioning algorithm.
Stage 1
Stage 2
Stage 3
MB1
MB2
MB1
MB2
MB1
MB2
Chunked
Parameter
Upload
Chunked
Gradient
Download
Activation
Upload
Compute
Activation
Download
Time
Figure 8. RoundPipe multi-stream workflow over three con-
secutive stages with two microbatches (MB) each. Each color
represents one stage.
4.2
Data Transfer Overlap
Under the Computation Dispatch Paradigm, every pipeline
stage requires two categories of data movement: (1) per-
micro-batch activation uploads and downloads between host
and GPU, and (2) per-stage parameter uploads and gradient
downloads between host and GPU.
Category (1) lies on the critical path. The next stage cannot
begin until the current stage’s output activation has arrived in
host memory and been re-uploaded to the target GPU. Cate-
gory (2), in contrast, has lower priority and can be scheduled
into any available execution window during the preceding or
subsequent stages.
4.2.1
The Multi-Stream Architecture. To exploit this over-
lap and fully utilize the bidirectional PCIe link, RoundPipe
maintains four dedicated communication streams per device
alongside the default compute stream. These four streams
independently handle the two transfer categories in both
directions (upload and download). Figure 8 illustrates this
overlap. Activation uploads and downloads are performed one
micro-batch early/delayed to avoid blocking compute, while
parameter uploads and gradient downloads are interleaved
into idle intervals between activation transfers. The compute
stream synchronizes with the activation transfer streams at
micro-batch granularity using CUDA events. This divides
the timeline into bounded data-transfer windows, ensuring
dependencies are met while capping in-flight data to avoid
out-of-memory errors.
4.2.2
Scheduling Parameter Transfers. To avoid band-
width contention on the shared PCIe link, RoundPipe con-
fines parameter and gradient transfers to the idle intervals
between activation transfers. A stage processing 𝑀micro-
batches creates 𝑀data-transfer windows, each processing one
micro-batch activation and one chunk of parameter/gradient
transfers. RoundPipe partitions parameter/gradient transfers
into 𝑀chunks by applying longest-processing-time-first sched-
uling [11]. Specifically, we sort the parameter/gradient tensors


---

forward_backward step T+1
optim step T
Optimizer Worker
Controller
GPUs
GPUs
Optimizer Worker
Barrier
P cp
G
forward_backward step T
optim step T-1
optim step T
T
T+1
Parameter Upload
Compute
Gradient Download
T
T+1
T
T+1
Controller
optim step T-1
cp
Master Weight
Version T-1
Version T
forward_backward step T
  forward_backward step T+1
O
P cp
G cp O
Time
Time
(a) Blocking Copy
(b) Layerwise Model Consistency Protocol
Data Ready Signal
Buffer Ready Signal
Figure 9. (a) The blocking approach copies weights (P cp) and gradients (G cp) on the main thread. (b) RoundPipe offloads
copies to the optimizer worker (O) and applies Layerwise Model Consistency Protocol to ensure correctness. The four arrows
from left to right correspond to ordering constraints (1)-(4), respectively.
by size in descending order and assign each tensor to the
interval with the smallest current total assigned transfer size.
In case of very large tensors (e.g., language model head), we
split them into smaller chunks before scheduling to ensure
they can fit within the available data-transfer windows without
causing contention.
4.3
Fine-Grained Model Consistency
4.3.1
The Consistency Problem. Mixed-precision training
with CPU offloading maintains three representations of the
parameters and gradients [37]: (1) a short-lived GPU transient
copy for active computation, (2) a low-precision CPU master
copy in host memory, and (3) a full-precision optimizer copy
synchronized with the master.
To coordinate iterations, RoundPipe utilizes the step()
call as a logical barrier during asynchronous optimizer up-
dates. RoundPipe must precisely control the synchronization
of updated weights into the master copy (P cp) and the ex-
traction of gradients from the master copy for the optimizer
(G cp). To guarantee staleness-1 training semantics, correct-
ness reduces to enforcing five ordering constraints among
these synchronization operations, GPU data transfers, and
optimizer steps:
(1) Protects against early overwriting of weights. P cp must
wait until the GPU has finished uploading parameters for
the previous iteration.
(2) Ensures complete weight updates. The GPU must wait
until P cp completes before uploading parameters for the
next iteration.
(3) Protects against reading incomplete gradients. G cp
must wait until the GPU has finished downloading the
previous iteration’s gradients into the master buffer.
(4) Protects previous gradients from being overwritten.
The GPU must wait until G cp completes before writing
the next iteration’s gradients into the master buffer.
(5) Preserves optimizer step semantics. Data copies must
be scheduled between the just-completed optimizer step
and the next incoming step.
4.3.2
Blocking vs. Layerwise Model Consistency Proto-
col. A straightforward blocking copy (Figure 9(a)) treats the
step() call as a concrete barrier, blocking until all workers
have finished their tasks before performing data synchroniza-
tions (P cp and G cp). It is trivially correct, but it reintro-
duces exactly the pipeline bubble from pipeline flush that
asynchronous execution was designed to eliminate. The data
synchronization itself is also time-consuming as it requires
performing floating-point casting on the CPU.
To eliminate this, RoundPipe uses a Layerwise Model
Consistency Protocol (Figure 9(b)). RoundPipe offloads
P cp and G cp to the optimizer worker and coordinates access
through a set of threading events. The ordering constraints
(1)-(4) should be preserved by four dependency edges, as
shown in Figure 9(b), while (5) is naturally enforced by the
optimizer worker’s internal sequential execution. RoundPipe
utilizes point-to-point event signaling to enforce these edges.
The controller dynamically creates and dispatches events for
workers to wait for or to set, allowing them to resolve ordering
constraints directly in a peer-to-peer manner.
Crucially, synchronization applied to the whole model
would still stall the shallowest layer (Layer 1), which sits at the
tight turnaround between the end of the backward pass and
the beginning of the next forward pass. Therefore, RoundPipe
binds these synchronization events to individual layers. This


---

fine-grained protocol allows the optimizer worker to release
events for early layers immediately after processing them,
empowering GPU workers to begin the next iteration’s forward
pass on Layer 1 while deeper layers are still synchronizing.
4.4
Automatic Stage Partitioning
Optimal stage partitioning is critical to maximizing the ex-
ecution performance of RoundPipe. All stages should have
similar execution times to ensure smooth and bubble-free
round-robin stage dispatching. For the first few iterations,
RoundPipe runs a one-layer-per-stage partition to collect run-
time statistics. RoundPipe uses lightweight CUDA events to
collect per-layer execution times and tracks memory consump-
tion through the PyTorch allocator. Based on these statistics,
RoundPipe computes a well-balanced stage partition and
applies it to subsequent iterations.
4.4.1
Problem Formulation. The objective of the partition-
ing algorithm is to minimize pipeline execution time while
ensuring that the partition does not exceed memory capacity.
We model execution time using per-layer computation latency,
given that RoundPipe effectively overlaps most communi-
cation with computation. As discussed in §3.3, total GPU
time (including startup and cool-down cost) for RoundPipe
to process 𝑀micro-batches on 𝑁GPUs with 𝑆stages is
𝑀·𝑆·𝑡+𝑁· (𝑁−1) ·𝑡under a balanced partition. In practice,
however, the execution time varies across stages. To simplify,
we model the execution time of all stages as the maximum
stage runtime (i.e., the pipeline bottleneck). Our goal is to
minimize (𝑀· 𝑆+ 𝑁· (𝑁−1)) · 𝑡max (§3.3) while preserving
GPU memory constraints on every stage.
4.4.2
Algorithm. RoundPipe leverages the observation that,
since each stage must consist of contiguous layers, the set
of possible values for 𝑡max is bounded to all contiguous sub-
sequence sums of forward and backward times (𝑂(𝐿2) possi-
bilities). For each candidate 𝑡max value, the problem reduces
to: find the minimum number of contiguous partitions such
that no partition’s total time exceeds 𝑡max and no partition’s
memory consumption exceeds the GPU’s memory limit. This
is a classic greedy problem solvable in 𝑂(𝐿) time by scan-
ning layers and packing them into stages until a constraint
is violated, resulting in an overall 𝑂(𝐿3) complexity for the
partitioning algorithm. During the greedy process, Round-
Pipe prioritizes filling the first backward stage to its maximum
capacity. This greedy choice maximizes the size of the first
backward stage, which is beneficial because this stage skips
activation recomputation (§3.2), leading to greater reductions
in total compute cost.
5
Evaluation
In this section, we first evaluate RoundPipe on both consumer-
grade and datacenter-grade GPU servers, then quantify its
host memory consumption and training cost, and study its
scaling with GPU count and sequence length. We finally break
down the pipeline bubbles of RoundPipe and evaluate the
effectiveness of RoundPipe’s design.
5.1
Experimental Setup
Hardware. We use two server configurations. Unless other-
wise stated, all experiments use 8 GPUs on a single server.
• 4090 server: 8× NVIDIA RTX 4090 GPUs (24 GB VRAM
each), Intel Xeon Gold 6330 CPU, 800 GB available DDR4
host memory, PCIe 4.0 (32 GB/s) interconnect.
• A800 server: 8× NVIDIA A800 SXM GPUs (80 GB
HBM2e each), Intel Xeon Platinum 8352Y CPU, 800 GB
available DDR4 host memory, NVLink 3.0 (200 GB/s) in-
terconnect.
Baselines. We compare against six representative train-
ing frameworks: (1) DeepSpeed ZeRO-2 [34], (2) PyTorch
FSDP [32] (ZeRO-3 data parallelism), (3) DeepSpeed ZeRO-
Infinity [35] (CPU memory offloading), (4) Megatron-LM
pipeline parallelism (Megatron-PP) [29], (5) Megatron-LM
tensor parallelism (Megatron-TP) [41], and (6) Mobius [10].
Megatron-PP implements the interleaved-1F1B pipeline sched-
ule, and Mobius implements the looped-BFS pipeline schedule
with CPU memory offloading. We also report RoundPipe-
sync, a variant of RoundPipe that disables asynchronous
optimizer updates (§3.2) and performs a synchronous CPU
optimizer step.
Workloads. We evaluate five open-source models: Qwen3-
1.7B [43], LLaMA-3.1-8B [25], GPT-OSS-20B [30], Qwen3-
32B, and Qwen3-235B-A22B. GPT-OSS-20B and Qwen3-
235B-A22B are Mixture-of-Experts (MoE) models, while
the other three are dense Transformers. We perform full-
parameter training on the first four models and apply LoRA
fine-tuning (𝑟= 32) exclusively to the massive Qwen3-235B.
The sequence length is fixed at 2048, with global batch sizes
of 512, 256, 128, 128, and 64 for each model, respectively.
All frameworks employ mixed-precision (FP16) training, full
activation recomputation, and identical global batch sizes.
We maximize the micro-batch size for each framework and
model individually to ensure full GPU utilization, subject to
two constraints: avoiding device out-of-memory (OOM) and
maintaining a sufficient number of micro-batches (e.g., ≥8)
to sustain high pipeline or data parallel throughput.
Metrics. We report two metrics: (1) Training throughput
(tokens/s): the number of tokens processed per second during
steady-state training, averaged over 10 iterations after warm-
up. (2) Maximum sequence length: the longest sequence
length at which training completes without out-of-memory
errors, searched at micro-batch size=1.


---

Qwen3
1.7B
LLaMA-3.1
8B
GPT-OSS
20B
Qwen3
32B
Qwen3
235B-LoRA
0K
20K
40K
60K
Throughput (tokens/s)
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
N/A
OOM
OOM
OOM OOM
ZeRO-2
FSDP
ZeRO-Infinity
Megatron-TP
Megatron-PP
Mobius
RoundPipe-sync
RoundPipe
Figure 10. Training Throughput on 8×RTX 4090.
Qwen3
1.7B
LLaMA-3.1
8B
GPT-OSS
20B
Qwen3
32B
Qwen3
235B-LoRA
0K
20K
40K
60K
Max Sequence Length
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
OOM
N/A
OOM
OOM
OOM OOM
ZeRO-2
FSDP
ZeRO-Infinity
Megatron-TP
Megatron-PP
Mobius
RoundPipe
Figure 11. Maximum trainable sequence length on 8×RTX
4090.
5.2
End-to-End Performance on 4090
Figures 10 and 11 present end-to-end throughput and maxi-
mum sequence length results on the 4090 server. 3
Throughput. Both RoundPipe and its synchronous vari-
ant achieve the highest throughput across all five models.
RoundPipe outperforms the fastest existing systems across
1.7-32B models by 1.48 ∼2.16× on training throughput, and
RoundPipe-sync outperforms by 1.15 ∼1.63×. This perfor-
mance advantage is driven by our novel pipeline design, which
substantially mitigates pipeline bubbles and yields improved
communication-computation overlap compared with existing
frameworks. Furthermore, by strictly bounding the number
of layers resident on the GPU, RoundPipe is the only system
capable of successfully LoRA-finetuning the massive 235B
model on 24GB GPUs.
Maximum Sequence Length. RoundPipe consistently sup-
ports the longest sequences. Excluding Megatron-TP, whose
throughput becomes impractical under PCIe, RoundPipe
extends the maximum sequence length by 4.7 ∼7.3× over
the next-best baseline. Optimizer asynchrony does not af-
fect this metric because it does not change GPU memory
usage. On Qwen3-1.7B, RoundPipe and Megatron-TP both
handle substantially longer sequences than other baselines.
TP achieves this by sharding activations across GPUs, while
3N/A on Megatron-TP when LoRA finetuning Qwen3-235B because the
model has four key value heads, which does not support TP=8 on Megatron.
Qwen3
1.7B
LLaMA-3.1
8B
GPT-OSS
20B
Qwen3
32B
Qwen3
235B-LoRA
0K
20K
40K
60K
80K
Throughput (tokens/s)
OOM
OOM
OOM OOM
N/A
OOM
ZeRO-2
FSDP
ZeRO-Infinity
Megatron-TP
Megatron-PP
RoundPipe-sync
RoundPipe
RoundPipe-4090
Figure 12. Training Throughput on 8×A800.
Qwen3
1.7B
LLaMA-3.1
8B
GPT-OSS
20B
Qwen3
32B
Qwen3
235B-LoRA
0K
100K
200K
300K
Max Sequence Length
OOM
OOM
OOM
OOM
N/A
OOM
ZeRO-2
FSDP
ZeRO-Infinity
Megatron-TP
Megatron-PP
RoundPipe
Figure 13. Maximum trainable sequence length on 8×A800.
RoundPipe achieves it by storing stage-boundary activations
in host memory and recomputing layer-internal activations on
demand. RoundPipe avoids TP’s heavy communication over-
head, thereby achieving higher training throughput than TP. As
model size scales up, non-offloading systems are bottlenecked
by model footprint and fail to scale sequence lengths, whereas
existing offloading systems (e.g., ZeRO-Infinity) still OOM
early because they do not offload activations. RoundPipe even
supports a slightly longer sequence on Qwen3-235B than
on Qwen3-32B because LoRA fine-tuning produces fewer
intermediate activations and gradients than full-parameter
training during recompute and backward pass.
5.3
End-to-End Performance on A800
To evaluate the overhead of RoundPipe and the Computation
Dispatch Paradigm, we compare RoundPipe with SOTA train-
ing systems on datacenter GPUs with ample GPU memory and
interconnect. Figures 12 and 13 present the results. We omit
Mobius because it reports lower performance on datacenter
GPUs compared to DeepSpeed ZeRO-Infinity [10].
Throughput. Even on hardware-rich servers, RoundPipe
delivers 0.98 ∼1.47× the throughput of SOTA systems. For
smaller models (1.7B and 8B), Data Parallelism (DP) per-
forms best by fully exploiting the high-bandwidth NVLink.
Remarkably, RoundPipe achieves highly competitive per-
formance without utilizing any GPU peer-to-peer commu-
nication (NVLink), relying entirely on PCIe host-to-device


---

Table 1. Host memory consumption (GB) of RoundPipe,
Minimum required and Peak during the benchmark of §5.2.
Qwen3
LLaMA
GPT-OSS
Qwen3
Qwen3
1.7B
3.1-8B
20B
32B
235B-LoRA
Min.
45.6
160.8
401.6
652.5
502.4
Peak
74.3
204.4
446.1
752.3
750.6
transfers under the Computation Dispatch Paradigm. Mean-
while, RoundPipe-sync matches the speed of DeepSpeed
ZeRO-Infinity, with their slowdown attributed to CPU op-
timizer step overhead. For larger models (20B+), TP and
PP overtake DP by avoiding full-parameter synchronization,
but RoundPipe still leads by generating less communication
than DP/TP and fewer bubbles than existing PP systems. On
Qwen3-32B, Megatron-PP encounters an OOM error because
the 64-layer model must be partitioned across 8 GPUs. The
final rank is burdened with 8 layers plus the LM head, en-
countering OOM during backward. Furthermore, Figure 12
plots RoundPipe’s performance on the 4090 server. Although
the 4090’s lower VRAM bandwidth inherently caps its peak
compute utilization compared to the A800, RoundPipe still
consistently achieves over 76% of the throughput of existing
SOTA frameworks running on the A800 across all models.
This proves that RoundPipe elevates consumer GPUs beyond
cost-effective alternatives; they now offer absolute training
times comparable to the expensive datacenter hardware.
Maximum Sequence Length. On the A800 server, Round-
Pipe increases the maximum sequence length by 1.19 ∼5.62×
over the baselines, following the same trend as on the 4090. For
smaller models (≤20B), Megatron-TP uses NVLink to sup-
port 108K∼242K tokens, but RoundPipe pushes this further
to 192K∼288K by storing model states and stage-boundary ac-
tivations in CPU memory. For larger models, TP and the other
baselines again become limited by GPU memory, whereas
RoundPipe maintains its lead.
5.4
Host Memory Consumption
RoundPipe consumes a fixed amount of host memory, approx-
imately 20× the model parameter size. This footprint includes
the FP32 model weights, gradients, and optimizer states, the
low-precision weights and gradients, and a small allocation for
the intermediate activation tensors used during recomputation.
We monitored host memory utilization during the evalua-
tion of §5.2. As shown in Table 1, RoundPipe achieves its
peak training throughput and supports its maximum sequence
length as long as the server is equipped with host memory
matching the minimum requirement.
5.5
Cost Efficiency
The 8×RTX 4090 server used in our experiments is priced
at $2.32 per hour, whereas the 8×A800 server costs $8.16
Qwen3
1.7B
LLaMA-3.1
8B
GPT-OSS
20B
0
50
100
150
USD / Billion Tokens
Mobius
Megatron-PP
Mobius
FSDP
FSDP
Megatron-TP
Qwen3
32B
Qwen3
235B-LoRA
0
1K
2K
Mobius
OOM
Megatron-TP
ZeRO-Inf
RoundPipe
RoundPipe-sync
Best baseline
 
8×RTX 4090
8×A800
Figure 14. Training cost in USD per billion tokens.
0
1
2
3
4
5
6
7
8
Number of RTX 4090 GPUs
0K
10K
20K
30K
40K
50K
60K
Throughput (tokens/s)
Qwen3-1.7B
LLaMA-3.1-8B
GPT-OSS-20B
Qwen3-32B
Qwen3-235B-LoRA
Figure 15. RoundPipe throughput on 1∼8 RTX 4090 GPUs.
per hour. Figure 14 converts the throughput of §5.2 and §5.3
into the training cost per billion tokens. On the 4090 server,
RoundPipe reduces training costs by 30%–53% compared to
the best existing baselines, and it is cheaper than all systems
on the A800 server for every model.
5.6
Scalability
We conducted strong-scaling experiments with fixed work-
loads to evaluate RoundPipe’s multi-GPU efficiency. Fig-
ure 15 illustrates that RoundPipe achieves near-linear through-
put scaling from 1 to 8 GPUs across all model sizes. The
stability of this scaling curve underscores the robustness of
RoundPipe, showing that its partitioning and overlap mecha-
nisms remain effective across GPU counts.
Crucially, RoundPipe exhibits a unique architectural ad-
vantage: the maximum sequence length is independent of
the GPU count. From 1 to 8 GPUs, the maximum sequence
length remains 73K, 49K, 39K, 28K, and 31K for the five
models. This invariant is a direct product of our Computation
Dispatch Paradigm. A GPU holds data required for its cur-
rently assigned stage only, while the remaining model state
and stage-boundary tensors for recompute are stored in host
memory.
5.7
Sequence Length Sensitivity
Figure 16 shows how RoundPipe’s throughput changes as
the sequence length increases over two orders of magnitude


---

512 1K
2K
4K
8K 16K 32K 64K
Sequence Length
0K
20K
40K
60K
Throughput (tokens/s)
Qwen3-1.7B on 8×RTX 4090
Figure 16. Throughput vs. sequence
length for Qwen3-1.7B.
2
4
6
8
10
Microbatch Size
1500
2000
2500
3000
Throughput (tokens/s)
RoundPipe
Simple Prefetch
No Prefetch
Figure 17. Throughput vs. microbatch
sizes for Llama3.1-8B on RTX 4090.
Qwen3
1.7B
LLaMA
3.1-8B
GPT-
OSS
20B
Qwen3
32B
Qwen3
235B
LoRA
0
25
50
75
Iteration time (s)
Blocking Copy
Fine-Grained
Consistency
Figure 18. Blocking copy vs. Layerwise
Model Consistency Protocol.
G
G
G
G
G
1
1
1
1
1
I
I
I
I
I
B
B
B
B
B
S
S
S
S
S
R
R
R
R
R
0
10
20
30
40
Bubble Ratio (%)
Qwen3
1.7B
Llama-3.1
8B
GPT-OSS
20B
Qwen3
32B
Qwen3
235B
Intra (steady state)
Switch (fwd→bwd)
Fill
Drain
Bubble Ratio Breakdown by Model and Schedule
Figure 19. Simulated pipeline bubble ratio of each schedule.
G: GPipe, 1: 1F1B, I: Interleaved-1F1B, B: Looped-BFS, S:
RoundPipe-sync, R: RoundPipe.
on 8×RTX 4090 with Qwen3-1.7B. RoundPipe maintains
robust performance across the entire range, with throughput
decreasing smoothly as attention cost grows. This shows that
RoundPipe supports both short and very long contexts without
prohibitive overhead.
5.8
Ablation Studies
5.8.1
Pipeline Schedule. To isolate the contribution of
RoundPipe’s pipeline schedule from the end-to-end perfor-
mance, we simulate the bubble ratio of five pipeline schedules
using per-layer timing data collected from the 4090 server.
For each model, we measure forward and backward execution
times of each transformer layer and the language model head
using a micro-batch size of 4 × 2048 tokens per micro-batch.
We write a pipeline simulator that faithfully implements each
schedule’s logic. We simulate the pipeline execution for 16
micro-batches on 8 GPUs.
Figure 19 shows the simulation result, breaking the bubbles
into four categories: (1) fill: filling the pipeline with the first
few micro-batches; (2) switch: transitioning between the for-
ward and backward passes; (3) drain: draining the pipeline for
the last few micro-batches; and (4) intra-stage: computational
imbalance within a single stage. Non-overlapped communica-
tion contributes less than 1% of the total runtime across all
experiments and is therefore excluded.
RoundPipe-sync reduces bubbles by 23% ∼55% relative
to the best baseline, and its bubble ratio further decreases as
deeper models are split into more stages. RoundPipe with
asynchronous optimizer updates virtually eliminates inter-
iteration bubbles, driving the absolute bubble ratio down to
below 4.5%. These bubbles are all attributed to stage imbal-
ances, which our partitioning algorithm (§4.4) successfully
bounds to a very low level.
We measured the wall-clock time for RoundPipe’s auto-
matic stage-splitting algorithm (§4.4). Partitioning Qwen3-
1.7B, LLaMA-3.1-8B, GPT-OSS-20B, and Qwen3-32B takes
2.9, 2.9, 2.6, and 5.0 milliseconds, respectively, proving the
algorithm is exceedingly fast in practice. Partitioning the
94 layers Qwen3-235B takes 1.47 seconds, which is still
inconsequential compared to the hours required for model
training.
5.8.2
Priority-aware Data Transfer. RoundPipe’s multi-
stream architecture (§4.2) overlaps parameter uploads and
gradient downloads with GPU computation by interleaving
transfers into the idle intervals between activation transfers.
To isolate the contribution of this design, we train Llama3.1-
8B on an RTX 4090 and compare three variants: (1) No
Prefetch: transfers are issued sequentially and block com-
putation; (2) Simple Prefetch: transfers begin at the start
of each stage without scheduling across micro-batch inter-
vals; (3) RoundPipe: full fine-grained overlap with transfer
chunking.
Figure 17 shows throughput as the microbatch size (mbs)
increases from 1 to 10. RoundPipe’s performance plateaus
early at around mbs=3, reflecting that the transfers are already
fully hidden at that point. It delivers 1.34× training throughput
compared to the no-prefetch variant and 1.15× throughput
compared to the simple prefetch variant. As mbs grows,
each micro-batch provides a longer computation window,
allowing the simple prefetch to eventually overlap most of
the transfer with computation. These results confirm that
RoundPipe’s data transfer design successfully overlaps data
transfer with computations and effectively addresses the head-
of-line blocking during data prefetches.
5.8.3
Layerwise Model Consistency Protocol. Round-
Pipe’s Layerwise Model Consistency Protocol (§4.3.2) avoids
blocking the main thread during the FP32→FP16 weight
copy and gradient collection that bridges the asynchronous


---

optimizer with the GPU workers. To quantify the benefit,
we compare it with the blocking copy baseline that inserts
a global barrier before performing the FP32→FP16 weight
cast on the main thread (§4.3.2). We measure the per-iteration
training time of both methods to quantify the performance
difference.
Figure 18 presents the results. The consistency protocol
reduces a substantial overhead of 2.6 ∼14 seconds per itera-
tion, growing roughly with the size of trainable parameters.
Qwen3-235B-LoRA benefits less because LoRA updates only
a small subset of weights and therefore copies less data. These
results confirm that the Layerwise Model Consistency Proto-
col is essential for realizing the full benefit of asynchronous
optimizer updates. By replacing the iteration barrier with
fine-grained, per-layer event signaling, RoundPipe effectively
overlaps weight and gradient synchronization with GPU com-
putation, converting what would otherwise be idle pipeline
stalls into productive work.
6
Discussion
Multi-node scaling. RoundPipe can be extended to multi-
node setups via two approaches. First, RoundPipe can be com-
bined with ZeRO-1 DP [34], using RoundPipe for intra-node
PP and ZeRO-1 for inter-node DP. Based on RoundPipe’s de-
coupled optimizer worker (§4.1), the CPU can perform ZeRO-
1’s gradient reduce-scatter without interfering with GPU
computation, thereby overlapping cross-node communication
with computation. Second, RoundPipe’s pipeline schedule
can be adapted by interleaving model layers across multiple
nodes. Within each node, RoundPipe’s core principles—the
stateless GPU worker model, the Layerwise Model Consis-
tency Protocol, optimizer placement, and CPU–GPU transfer
coordination—remain applicable and beneficial. Compared
to traditional pipeline parallelism, RoundPipe still offers bet-
ter scheduling flexibility in the multi-node context, reducing
pipeline bubbles and improving GPU load balancing. The
additional overhead is the inter-node transmission of hidden
states, which involves minimal communication volume and
can be hidden behind computation.
7
Related Work
7.1
Pipeline Parallel Schedules
Synchronous pipeline schedules [8, 14, 15, 22, 41] are subject
to the pipeline bubble problem. Asynchronous approaches [3,
5, 12, 13, 28, 47, 48] reduce bubbles via weight stashing,
and the Zero Bubble Schedule [33] reduces bubbles via de-
laying gradient calculations for weights, but both techniques
trade memory consumption for efficiency. Looped pipeline
methods [20, 29] increase stage counts to reduce bubbles,
but they require the stage count to be a multiple of the GPU
count, making balanced partitioning increasingly difficult.
RoundPipe takes a different approach by utilizing heteroge-
neous memory to decouple stages from GPUs, providing a
synchronous schedule with flexible stage partitions, and an
asynchronous schedule with no GPU memory overhead.
7.2
Offloading Training Frameworks
Heterogeneous memory offloading has been widely explored
to enable training models that exceed GPU memory. One
line of work offloads model weights and optimizer states to
CPU or NVMe [9, 35, 42, 49]. Another line targets activation
offloading, swapping intermediate activations to the host to re-
duce peak GPU memory [2, 38, 45, 52]. More recent systems
manage data at tensor granularity to achieve better transfer–
compute overlaps [23, 36, 50]. However, these approaches
are predominantly designed for single-GPU or data-parallel
settings; scaling them to multiple GPUs incurs substantial com-
munication overhead [10]. RoundPipe co-designs distributed
training with host-memory offloading, achieving scaling and
offloading with negligible overhead.
7.3
Novelty Relative to Prior Systems
The primary novelty of RoundPipe lies in unbinding pipeline
stages from specific GPUs and enabling flexible pipeline par-
titioning. All prior pipeline training systems bind stages to
GPUs to minimize communication overhead, a design choice
prevalent in non-offloading systems. Existing works have over-
looked the key insight of RoundPipe: fetching model layers
from host memory to the GPU remains compute-bound (§3.3),
so computational efficiency is not bottlenecked by weight
loading. Driven by this observation, RoundPipe is the first
system to propose and realize dynamic stage dispatching for
pipeline-parallel training, and it exploits the resulting schedul-
ing freedom with flexible stage partitioning to reduce pipeline
bubbles and improve GPU utilization. Importantly, efficiently
implementing stage dispatching requires addressing the chal-
lenges detailed in §4; naively adapting stage dispatching into
existing frameworks would yield suboptimal performance.
8
Conclusion
This paper presents RoundPipe, a system that implements a
new pipeline parallelism schedule for training large models
on consumer GPU servers. We introduced the Computa-
tion Dispatch Paradigm and confirmed that it preserves full
compute-bound throughput. Building on this paradigm, the
RoundPipe schedule uses asymmetric stage splitting and
round-robin dispatch to decouple stages from GPUs, mitigate
stage imbalance, and improve pipeline efficiency. Experiment
results demonstrate its performance gain.
Acknowledgments
This work is supported by the National Key R&D Program
of China (Grant No. 2024YFB4505201), the National Nat-
ural Science Foundation of China (Grant No. U22B2023,
62332011), and the High Performance Computing Center of
Tsinghua University.


---

References
[1] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li,
Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2023. L-Eval: Insti-
tuting Standardized Evaluation for Long Context Language Models.
arXiv:2307.11088 [cs.CL] https://arxiv.org/abs/2307.11088
[2] Jonghyun Bae, Jongsung Lee, Yunho Jin, Sam Son, Shine Kim, Hak-
beom Jang, Tae Jun Ham, and Jae W Lee. 2021. {FlashNeuron}:{SSD-
Enabled}{Large-Batch} training of very deep neural networks. In
19th USENIX conference on file and storage technologies (FAST 21).
387–401.
[3] Chi-Chung Chen, Chia-Lin Yang, and Hsiang-Yun Cheng. 2018. Ef-
ficient and robust parallel dnn training through model parallelism on
multi-gpu platform. arXiv preprint arXiv:1809.02839 (2018).
[4] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Train-
ing Deep Nets with Sublinear Memory Cost. arXiv:1604.06174 [cs.LG]
https://arxiv.org/abs/1604.06174
[5] Zihao Chen, Chen Xu, Weining Qian, and Aoying Zhou. 2023. Elastic
averaging for efficient pipelined DNN training. In Proceedings of the
28th ACM SIGPLAN Annual Symposium on Principles and Practice of
Parallel Programming. 380–391.
[6] DeepSeek-AI. 2025. DeepSeek-R1 incentivizes reasoning in LLMs
through reinforcement learning. Nature 645, 8081 (Sept. 2025), 633–638.
doi:10.1038/s41586-025-09422-z
[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
2019. Bert: Pre-training of deep bidirectional transformers for language
understanding. In Proceedings of the 2019 conference of the North
American chapter of the association for computational linguistics
(NAACL). 4171–4186.
[8] Shiqing Fan, Yi Rong, Chen Meng, Zongyan Cao, Siyu Wang, Zhen
Zheng, Chuan Wu, Guoping Long, Jun Yang, Lixue Xia, et al. 2021.
DAPPLE: A pipelined data parallel approach for training large models.
In Proceedings of the 26th ACM SIGPLAN Symposium on Principles
and Practice of Parallel Programming. 431–445.
[9] Jiarui Fang, Zilin Zhu, Shenggui Li, Hui Su, Yang Yu, Jie Zhou, and
Yang You. 2022. Parallel training of pre-trained models via chunk-based
dynamic memory management. IEEE Transactions on Parallel and
Distributed Systems 34, 1 (2022), 304–315.
[10] Yangyang Feng, Minhui Xie, Zijie Tian, Shuo Wang, Youyou Lu,
and Jiwu Shu. 2023. Mobius: Fine Tuning Large-Scale Models on
Commodity GPU Servers. In Proceedings of the 28th ACM International
Conference on Architectural Support for Programming Languages and
Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023).
Association for Computing Machinery, New York, NY, USA, 489–501.
doi:10.1145/3575693.3575703
[11] R. L. Graham. 1969.
Bounds on Multiprocessing Timing
Anomalies.
SIAM J. Appl. Math. 17, 2 (1969), 416–429.
arXiv:https://doi.org/10.1137/0117039 doi:10.1137/0117039
[12] Lei Guan, Wotao Yin, Dongsheng Li, and Xicheng Lu. 2019. XPipe:
Efficient pipeline model parallelism for multi-GPU DNN training. arXiv
preprint arXiv:1911.04610 (2019).
[13] Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri,
Nikhil Devanur, Greg Ganger, and Phil Gibbons. 2018. PipeDream: Fast
and Efficient Pipeline Parallel DNN Training. arXiv:1806.03377 [cs.DC]
https://arxiv.org/abs/1806.03377
[14] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu
Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le,
Yonghui Wu, and Zhifeng Chen. 2019. GPipe: Efficient Training of Giant
Neural Networks using Pipeline Parallelism. arXiv:1811.06965 [cs.CV]
https://arxiv.org/abs/1811.06965
[15] Arpan Jain, Ammar Ahmad Awan, Asmaa M Aljuhani, Jahanzeb Maq-
bool Hashmi, Quentin G Anthony, Hari Subramoni, Dhableswar K
Panda, Raghu Machiraju, and Anil Parwani. 2020. Gems: Gpu-enabled
memory-aware model-parallelism system for distributed dnn training.
In SC20: international conference for high performance computing,
networking, storage and analysis. IEEE, 1–15.
[16] Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xi-
ang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng,
Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xu-
anzhe Liu, Xin Jin, and Xin Liu. 2025. MegaScale-MoE: Large-Scale
Communication-Efficient Training of Mixture-of-Experts Models in Pro-
duction. arXiv:2505.11432 [cs.LG] https://arxiv.org/abs/2505.11432
[17] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Ben-
jamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu,
and Dario Amodei. 2020. Scaling Laws for Neural Language Models.
arXiv:2001.08361 [cs.LG] https://arxiv.org/abs/2001.08361
[18] Heehoon Kim, Junyeol Ryu, and Jaejin Lee. 2024. TCCL: Discovering
Better Communication Paths for PCIe GPU Clusters. In Proceedings of
the 29th ACM International Conference on Architectural Support for
Programming Languages and Operating Systems, Volume 3 (La Jolla,
CA, USA) (ASPLOS ’24). Association for Computing Machinery, New
York, NY, USA, 999–1015. doi:10.1145/3620666.3651362
[19] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee,
Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2022.
Reducing Activation Recomputation in Large Transformer Models.
arXiv:2205.05198 [cs.LG] https://arxiv.org/abs/2205.05198
[20] Joel Lamy-Poirier. 2023.
Breadth-First Pipeline Parallelism.
arXiv:2211.05953 [cs.DC] https://arxiv.org/abs/2211.05953
[21] Tingfeng Lan, Yusen Wu, Bin Ma, Zhaoyuan Su, Rui Yang, Tekin Bicer,
Masahiro Tanaka, Olatunji Ruwase, Dong Li, and Yue Cheng. 2025.
ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous
Updates. arXiv:2505.12242 [cs.DC] https://arxiv.org/abs/2505.12242
[22] Shigang Li and Torsten Hoefler. 2021. Chimera: efficiently training
large-scale neural networks with bidirectional pipelines. In Proceedings
of the International Conference for High Performance Computing,
Networking, Storage and Analysis. 1–14.
[23] Changyue Liao, Mo Sun, Zihan Yang, Jun Xie, Kaiqi Chen, Bin-
hang Yuan, Fei Wu, and Zeke Wang. 2024. LoHan: Low-Cost High-
Performance Framework to Fine-Tune 100B Model on a Consumer
GPU. arXiv:2403.06504 [cs.DC] https://arxiv.org/abs/2403.06504
[24] Xinzhang Liu, Chao Wang, Zhihao Yang, Zhuo Jiang, Xuncheng Zhao,
Haoran Wang, Lei Li, Dongdong He, Luobin Liu, Kaizhe Yuan, Han
Gao, Zihan Wang, Yitong Yao, Sishi Xiong, Wenmin Deng, Haowei
He, Kaidong Yu, Yu Zhao, Ruiyu Fang, Yuhao Jiang, Yingyan Li,
Xiaohui Hu, Xi Yu, Jingqi Li, Yanwei Liu, Qingli Li, Xinyu Shi, Junhao
Niu, Chengnuo Huang, Yao Xiao, Ruiwen Wang, Fengkai Li, Luwen
Pu, Kaipeng Jia, Fubei Yao, Yuyao Huang, Xuewei He, Zhuoru Jiang,
Ruiting Song, Rui Xue, Qiyi Xie, Jie Zhang, Zilu Huang, Zhaoxi
Zhang, Zhilong Lu, Yanhan Zhang, Yin Zhang, Yanlei Xue, Zhu Yuan,
Teng Su, Xin Jiang, Shuangyong Song, Yongxiang Li, and Xuelong Li.
2025. Training Report of TeleChat3-MoE. arXiv:2512.24157 [cs.CL]
https://arxiv.org/abs/2512.24157
[25] AI @ Meta Llama Team. 2024.
The Llama 3 Herd of Models.
arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
[26] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos,
Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii
Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed Precision
Training. arXiv:1710.03740 [cs.AI] https://arxiv.org/abs/1710.03740
[27] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov,
Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William
Paul, Michael I. Jordan, and Ion Stoica. 2018. Ray: A Distributed
Framework for Emerging AI Applications. arXiv:1712.05889 [cs.DC]
https://arxiv.org/abs/1712.05889
[28] Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, and Matei
Zaharia. 2021. Memory-Efficient Pipeline-Parallel DNN Training.
arXiv:2006.09503 [cs.LG] https://arxiv.org/abs/2006.09503
[29] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick
LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vain-
brand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar


---

Phanishayee, and Matei Zaharia. 2021.
Efficient Large-Scale
Language Model Training on GPU Clusters Using Megatron-LM.
arXiv:2104.04473 [cs.CL] https://arxiv.org/abs/2104.04473
[30] OpenAI.
2025.
gpt-oss-120b
&
gpt-oss-20b
Model
Card.
arXiv:2508.10925 [cs.CL] https://arxiv.org/abs/2508.10925
[31] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wain-
wright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina
Slama, Alex Ray, et al. 2022. Training language models to follow
instructions with human feedback. Advances in neural information
processing systems (NIPS) 35 (2022), 27730–27744.
[32] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James
Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia
Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward
Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chil-
amkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala.
2019. PyTorch: An Imperative Style, High-Performance Deep Learning
Library. arXiv:1912.01703 [cs.LG] https://arxiv.org/abs/1912.01703
[33] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. 2023. Zero
Bubble Pipeline Parallelism. arXiv:2401.10241 [cs.DC] https://arxiv.
org/abs/2401.10241
[34] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong
He. 2020. ZeRO: Memory Optimizations Toward Training Trillion
Parameter Models. arXiv:1910.02054 [cs.LG] https://arxiv.org/abs/
1910.02054
[35] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith,
and Yuxiong He. 2021. ZeRO-Infinity: Breaking the GPU Memory
Wall for Extreme Scale Deep Learning. arXiv:2104.07857 [cs.DC]
https://arxiv.org/abs/2104.07857
[36] Jie Ren, Jiaolin Luo, Kai Wu, Minjia Zhang, Hyeran Jeon, and Dong Li.
2021. Sentinel: Efficient tensor migration and allocation on heteroge-
neous memory systems for deep learning. In 2021 IEEE International
Symposium on High-Performance Computer Architecture (HPCA).
IEEE, 598–611.
[37] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji
Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He.
2021. ZeRO-Offload: Democratizing Billion-Scale Model Training.
arXiv:2101.06840 [cs.DC] https://arxiv.org/abs/2101.06840
[38] Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, and
Stephen W Keckler. 2016. vDNN: Virtualized deep neural networks for
scalable, memory-efficient neural network design. In 2016 49th Annual
IEEE/ACM International Symposium on Microarchitecture (MICRO).
IEEE, 1–13.
[39] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai
Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal
Remez, et al. 2023. Code llama: Open foundation models for code.
arXiv preprint arXiv:2308.12950 (2023).
[40] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang,
Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybrid-
Flow: A Flexible and Efficient RLHF Framework. In Proceedings of the
Twentieth European Conference on Computer Systems (EuroSys ’25).
ACM, 1279–1297. doi:10.1145/3689031.3696075
[41] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper, and Bryan Catanzaro. 2020. Megatron-LM: Training
Multi-Billion Parameter Language Models Using Model Parallelism.
arXiv:1909.08053 [cs.CL] https://arxiv.org/abs/1909.08053
[42] Xiaoyang Sun, Wei Wang, Shenghao Qiu, Renyu Yang, Songfang Huang,
Jie Xu, and Zheng Wang. 2022. StrongHold: fast and affordable billion-
scale deep learning model training. In Proceedings of the International
Conference on High Performance Computing, Networking, Storage and
Analysis (Dallas, Texas) (SC ’22). IEEE Press, Article 71, 17 pages.
[43] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL]
https://arxiv.org/abs/2505.09388
[44] Team Wan. 2025. Wan: Open and Advanced Large-Scale Video Gener-
ative Models. arXiv:2503.20314 [cs.CV] https://arxiv.org/abs/2503.
20314
[45] Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuai-
wen Leon Song, Zenglin Xu, and Tim Kraska. 2018. Superneurons:
Dynamic GPU memory management for training deep neural networks.
In Proceedings of the 23rd ACM SIGPLAN symposium on principles
and practice of parallel programming. 41–53.
[46] Samuel Williams, Andrew Waterman, and David Patterson. 2009.
Roofline: an insightful visual performance model for multicore architec-
tures. Commun. ACM 52, 4 (April 2009), 65–76. doi:10.1145/1498765.
1498785
[47] Bowen Yang, Jian Zhang, Jonathan Li, Christopher Ré, Christopher
Aberger, and Christopher De Sa. 2021.
Pipemare: Asynchronous
pipeline parallel dnn training. Proceedings of Machine Learning and
Systems 3 (2021), 269–296.
[48] PengCheng Yang, Xiaoming Zhang, Wenpeng Zhang, Ming Yang, and
Hong Wei. 2022. Group-based interleaved pipeline parallelism for
large-scale DNN training. In International Conference on Learning
Representations.
[49] Ruijia Yang and Zeyi Wen. 2026. An Efficient Heterogeneous Co-
Design for Fine-Tuning on a Single GPU. arXiv:2603.16428 [cs.DC]
https://arxiv.org/abs/2603.16428
[50] Haoyang Zhang, Yirui Zhou, Yuqi Xue, Yiqi Liu, and Jian Huang.
2023. G10: Enabling an efficient unified gpu memory and storage
architecture with smart tensor migrations. In Proceedings of the 56th
Annual IEEE/ACM International Symposium on Microarchitecture.
395–410.
[51] Pinxue Zhao, Hailin Zhang, Fangcheng Fu, Xiaonan Nie, Qibin Liu,
Fang Yang, Yuanbo Peng, Dian Jiao, Shuaipeng Li, Jinbao Xue, Yangyu
Tao, and Bin Cui. 2025. MEMO: Fine-grained Tensor Management
For Ultra-long Context LLM Training. Proceedings of the ACM on
Management of Data 3, 1 (Feb. 2025), 1–28. doi:10.1145/3709703
[52] Zan Zong, Li Lin, Leilei Lin, Lijie Wen, and Yu Sun. 2023. Str:
Hybrid tensor re-generation to break memory wall for dnn training.
IEEE Transactions on Parallel and Distributed Systems 34, 8 (2023),
2403–2418.
