---
title: 'VarioPath: Workload-Aware All-to-All Communication for PCIe GPU Clusters'
id: variopath-workload-aware-all-to-all-communication-for-pcie-gpu-clusters
tags:
- cmp170hx-training-feasibility-a0d4fc
- all-to-all
- pcie-fabric
- expert-parallelism
- topology-aware-scheduling
- moe-inference
created: '2026-10-27T10:12:15.956971Z'
updated: '2026-10-27T19:00:31.127399Z'
source: https://arxiv.org/abs/2609.34340
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:15.955869Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'VarioPath (Fei et al., USTC + Tsinghua, arXiv 2609.34340): workload-aware
  AlltoAllv scheduling for PCIe GPU clusters — the primitive that EP dispatch/combine
  and sequence parallelism are built from. It documents why PCIe all-to-all is hard:
  no dedicated scale-up fabric means intra-node GPU traffic and NIC traffic share
  the same PCIe hierarchy, so concurrent transfers contend on bottleneck links; demand
  is skewed (top-k routing) and changes every layer (sub-millisecond per layer), defeating
  static schedules. VarioPath splits an offline topology-aware analyzer (records contention-free
  transfer patterns as channels, exploits topology symmetry) from an online demand-aware
  scheduler. On four platforms up to 256 GPUs it averages 5.88x over FAST and 1.72x
  over DeepEP (1.55-2.00x geomean over DeepEP on random-demand AlltoAllv, 3.73-10.61x
  over NCCL), reaching 30.91 GB/s algorithmic bandwidth at 256 GPUs vs 28.11 (DeepEP)
  and 18.85 (NCCL). End-to-end: Qwen3-30B-A3B prefill latency -19.1% to -27.2% at
  EP16/24/32, Wan2.1 generation -1.1% to -6.6%. Ablation: fixing either peer matching
  or path choice roughly doubles completion time (2.03x / 2.00x), fixing both 2.84x
  — peer-path co-design is not optional. Testbeds are Gen4/Gen5 x16 with 32-64 GB/s
  per GPU, so the absolute GB/s figures do not transfer to a Gen2 x4 box; the structural
  findings do.'
raw_file: raw/variopath-workload-aware-all-to-all-communication-for-pcie-gpu-clusters.pdf
doi: arXiv:2609.34340
utility_score: 14.0
citation_count: 0
---

VarioPath: Workload-Aware All-to-All Communication for PCIe GPU Clusters
Yao Fei1
Jin Fang1
Size Zheng2
Gongming Zhao1
Hongli Xu1
Jiacheng Zhu1
Zhijing Xin1
1University of Science and Technology of China
2Tsinghua University
{yao_fei,fangjin98,zhu_jc,zhijingxin}@mail.ustc.edu.cn
{gmzhao,xuhongli}@ustc.edu.cn
zhengsz@mail.tsinghua.edu.cn
Abstract
AlltoAllv communication is a critical primitive in distributed
large-model inference, particularly for mixture-of-experts
(MoE) models. The growing adoption of PCIe GPU systems
for cost-efficient inference makes AlltoAllv performance on
these systems increasingly important. Without a dedicated
scale-up interconnect (e.g., NVLink or Infinity Fabric), PCIe
GPU systems carry both intra-node and inter-node traffic
through the PCIe hierarchy, where concurrent transfers can
contend for PCIe link bandwidth. This link contention, com-
pounded by skewed traffic distributions and dynamic traffic
demand, makes efficient AlltoAllv scheduling challenging.
Existing approaches are either poorly suited to PCIe GPU
systems or incur substantial schedule synthesis overhead that
reduces their practicality in real-world deployments.
We present VarioPath, an efficient AlltoAllv scheduling
framework for PCIe GPU systems. It combines an offline
topology-aware analyzer with an online demand-aware sched-
uler. The analyzer records contention-free transfer patterns as
AlltoAllv channels and exploits topology symmetry to build
a compact catalog for efficient search. The online scheduler
decomposes each AlltoAllv invocation’s demand across a se-
quence of channels, adapting to rapidly changing and skewed
traffic while incurring low planning overhead. Evaluation on
four platforms (up to 256 GPUs) shows average AlltoAllv
speedups of 5.88× over FAST and 1.72× over DeepEP. End-
to-end experiments show that VarioPath reduces Qwen3 infer-
ence latency by up to 27.2% and Wan2.1 generation latency
by 6.1%.
1
Introduction
AlltoAllv is a critical communication primitive for distributed
model inference. In expert parallelism (EP) [8,11,20,21,31,
33], it dispatches activations to experts and returns their out-
puts. In sequence parallelism (SP) [10, 14], it redistributes
attention tensors across the sequence and head dimensions.
An AlltoAllv invocation may require each GPU to send dif-
ferent volumes of data to different peers. For example, input-
dependent expert routing can produce a skewed traffic dis-
tribution [15,16,20], with a few destinations receiving sub-
stantially more data than others. AlltoAllv’s many-to-many
exchange pattern complicates the coordination of concurrent
transfers and limits efficient utilization of interconnect band-
width [9,16,17,22]. Consequently, AlltoAllv communication
often becomes a major bottleneck in distributed large-model
inference.
This communication bottleneck is even more pronounced
on PCIe GPU systems. In tightly integrated GPU servers (e.g.,
NVIDIA HGX H100 [27]), a dedicated NVLink/NVSwitch
scale-up fabric [28] provides high-bandwidth, non-blocking
intra-node GPU connectivity [34]. By contrast, PCIe systems
connect GPUs and NICs through shared PCIe switches and
links. These interconnects vary across server designs and do
not guarantee non-blocking communication [17,22,32]. Their
bandwidth is also substantially lower than that of dedicated
scale-up networks. Despite these limitations, PCIe systems
are widely deployed for inference because their availabil-
ity and customizability help control infrastructure cost and
meet different resource requirements [9, 17]. For example,
Alibaba and ByteDance deploy PCIe-connected NVIDIA T4
and L40S GPUs in production clusters comprising thousands
of GPUs [39,45]. Cloudflare also deploys multiple types of
PCIe GPUs for its Workers AI service [18].
On these widely deployed PCIe GPU systems, AlltoAllv
presents three challenges: link contention, traffic skew, and
dynamic demand. First, the typically non-Clos interconnect
can concentrate traffic on the same bottleneck links [22,32],
while the absence of a separate scale-up network makes intra-
node GPU communication and NIC-based inter-node traf-
fic compete for the same PCIe resources. Traffic therefore
concentrates on some links, creating hotspots while leaving
others underutilized and reducing aggregate throughput. Sec-
ond, AlltoAllv demand can be highly skewed, requiring some
GPUs to transfer much more data than others. This imbal-
ance keeps certain GPUs and PCIe links busy after others
have finished, creating a communication tail that delays the
1
arXiv:2609.34340v1  [cs.DC]  28 Sep 2026


---

entire invocation even without link contention. Finally, dy-
namic AlltoAllv demand makes static schedules impractical
and requires schedulers to adapt in real time.
Prior work [3, 8, 20, 23, 32, 38] primarily targets tightly
integrated servers with dedicated scale-up networks. For ex-
ample, FAST [20] and DeepEP [8] rebalance AlltoAllv traffic
by introducing additional intra-node transfers over dedicated,
high-bandwidth scale-up networks. On PCIe GPU systems,
however, these approaches exacerbate link contention because
the additional intra-node transfers compete with NIC traffic
for the same shared PCIe links. General-purpose libraries
such as NCCL [29] and MSCCL [25] incur little runtime
scheduling overhead, but their static scheduling strategies can-
not effectively adapt to dynamic AlltoAllv demands in real-
world workloads. Only a few systems, including Blink [38],
BlueConnect [4], and TCCL [17], explicitly optimize collec-
tive communication for PCIe GPU systems. However, they
focus primarily on regular collectives such as AllReduce [30]
and AllGather [1], often using ring-based algorithms. Their
optimizations therefore do not directly apply to AlltoAllv.
MILP-based schedulers [3,23,32,40] jointly model network
topology and communication demand but can take seconds
to hours to synthesize a schedule. Because AlltoAllv demand
changes across invocations, repeatedly running this synthesis
is impractical for real-world workloads.
Addressing the above challenges while maintaining low
scheduling overhead is difficult on PCIe GPU systems. This
is because avoiding link contention requires time-consuming
topology analysis, while dynamic AlltoAllv demand requires
schedules to adapt to every invocation. Our key insight is
that the PCIe topology remains stable and determines which
concurrent transfer patterns avoid link contention. These pat-
terns can therefore be identified in advance and recorded
as reusable AlltoAllv channels, each comprising concurrent
GPU pairs and their communication paths. These channels
distill the topology’s communication capabilities into reusable
scheduling primitives. Channels offer different distributions
of transfer capacity, making them suitable for different com-
munication demands. On the demand side, skewed AlltoAllv
demand can be divided into chunks and mapped to a sequence
of channels in real time. This decomposition balances traffic
across different paths to mitigate communication tails. Over-
all, this separation makes effective, low-overhead AlltoAllv
scheduling possible.
Based on this insight, we present VarioPath, combining an
offline topology-aware analyzer with an online demand-aware
scheduler. To avoid enumerating and storing an exponentially
growing channel space, the offline analyzer exploits topology
symmetry to represent channels compactly as channel fami-
lies. It then organizes these families in a tree-structured cata-
log, enabling successive filtering of candidates. At runtime,
the scheduler maps demand chunks to these families, jointly
selecting channels and allocating transfer volumes to their
paths. The scheduler prioritizes communication bottlenecks
when matching channels to the current demand. Reusing the
offline catalog allows it to achieve near-optimal schedules
with low scheduling overhead. We make the following contri-
butions:
• We present VarioPath, an AlltoAllv scheduling frame-
work for PCIe GPU systems. By separating topology-
dependent offline analysis from demand-dependent on-
line scheduling, VarioPath enables topology-aware and
demand-adaptive AlltoAllv scheduling with low runtime
overhead.
• We distill topology information into the AlltoAllv
channel abstraction, with each channel representing a
contention-free concurrent transfer pattern. The offline
analyzer exploits topology symmetry to organize chan-
nels into a compact tree-structured catalog for efficient
search, while the bottleneck-aware online scheduler de-
composes each invocation’s demand and allocates it
across a sequence of channels.
• We validate VarioPath on four PCIe GPU platforms, from
single-node servers to 256 GPUs, using collective bench-
marks and end-to-end Qwen3 and Wan2.1 inference. Var-
ioPath consistently outperforms state-of-the-art systems,
achieving up to a 5.88× AlltoAllv speedup and reducing
inference latency by up to 27.2%.
2
Background and Motivation
2.1
AlltoAllv Scheduling Challenges on PCIe
Systems
An AlltoAllv invocation is described by a matrix of pairwise
data volumes, where each entry specifies the data sent from
one source rank to one destination rank. Support for indepen-
dently sized rank-pair transfers makes AlltoAllv well suited to
key data-redistribution operations in distributed large-model
inference, including token dispatch and output return in expert
parallelism (EP) [8, 12, 20] and attention-tensor redistribu-
tion in sequence parallelism (SP) [10,14]. In MoE inference,
token-to-expert assignments are recomputed at every layer,
producing a new AlltoAllv demand matrix. With per-layer
execution times often below one millisecond [24,42,43], the
scheduler may need to adapt to each new demand matrix on
a similarly short timescale. Otherwise, scheduling can de-
lay the critical path and erase the latency gains of a better
communication plan.
Following prior pairwise-exchange schedules [1,20,26,36],
all-to-all exchanges are commonly divided into endpoint-
disjoint stages. Within each stage, every rank sends to at
most one peer and receives from at most one peer. Because
each rank receives from at most one peer within a stage, this
scheduling structure avoids receiver-side incast, where multi-
ple senders compete for a receiver’s ingress bandwidth and
2


---

G0
G1
G2
G3
G4
G5
N0
N1
Inter-node Network
21.3GB/s
21.3GB/s
21.3GB/s
total_bw. 
128GB/s
src
dst
path
0
3B
1
4B
2
5B
3
0B
4
1B
5
2B
fA
0
fA
1
fA
2
fA
0 fA
1
fA
2 fA
3 fA
4 fA
5
AlltoAllv Stage A
(a) Fastest-path assignment
G0
G1
G2
G3
G4
G5
N0
N1
Inter-node Network
32GB/s
32GB/s
50GB/s
total_bw.   
228GB/s
fB
0
fB
1
fB
2
src
dst
path
0
3B
1
4B
2
5
N0
3
0B
4
1B
5
2
N1
fB
0 fB
1
fB
2 fB
3 fB
4 fB
5
AlltoAllv Stage B
(b) Path-only adaptation
G0
G1
G2
G3
G4
G5
N0
N1
Inter-node Network
64GB/s
50GB/s
64GB/s
total_bw.   
 356GB/s
src
dst
path
0
1I
1
4B
2
5
N0
3
0B
4
3I
5
2
N1
fC
0
fC
1 fC
2 fC
3 fC
4 fC
5
fC
1
fC
2
AlltoAllv Stage C
fC
0
(c) Joint peer–path selection
Figure 1: Joint peer–path selection avoids PCIe contention. Panels (a)–(c) compare fastest-path assignment, path-only
adaptation, and joint peer–path selection, whose modeled aggregate bandwidths are 128, 228, and 356 GB/s, respectively. Each
panel depicts an endpoint-disjoint stage from the same AlltoAllv invocation. Each flow f represents a logical transfer together
with its selected physical path. Transfers not selected in the current stage remain for later stages. For clarity, each topology draws
only f X
0 – f X
2 , while its table and aggregate bandwidth include all six logical transfers selected in the stage. The table below
each topology lists every source’s destination and selected path. I, B, and N0/N1 denote a same-switch PCIe path (PIX), an
inter-switch PCIe path (PXB), and NIC paths through NIC0/NIC1, respectively. For simplicity, we use nominal PCIe and NIC
bandwidths of 64 and 50 GB/s, respectively.
(a) Three AlltoAllv flows
G0
G1
G2
N0
N1
PXB path
NIC path
G3
G4
G5
(b) Whole-flow schedule
PXB 64 GB/s
NIC 50 GB/s
A · 114 MB
idle
B · 64 MB
C · 50 MB
(c) Chunk-level schedule
Stage 1
Stage 2
A1 · 64 MB
B · 64 MB
A2 · 50 MB
Network
flow A
114MB
flow B
64MB
flow C
50MB
Time
Saved
C · 50 MB
Time
PXB 64 GB/s
NIC 50 GB/s
Figure 2: Chunk-level path remapping mitigates commu-
nication tails. (a) Three cross-switch transfers can use ei-
ther a 64-GB/s PXB or a 50-GB/s NIC path; block width in
(b)–(c) denotes modeled transfer time. (b) Whole-flow bind-
ing creates imbalanced path loads: B and C serialize on the
NIC, while the PXB path finishes early and becomes idle. (c)
Chunking A enables path switching in Stage 2, balancing load
and eliminating the tail.
cause severe congestion [13,19]. However, even with pairwise-
exchange scheduling, PCIe GPU systems still present two
additional AlltoAllv scheduling challenges:
Shared-link contention limits AlltoAllv throughput. A
non-blocking NVLink/NVSwitch fabric [22] supports such
endpoint-disjoint logical transfers without internal-link over-
subscription. PCIe-based systems lack an independent scale-
up fabric and therefore cannot guarantee contention-free exe-
cution of endpoint-disjoint transfers. Specifically, this limita-
tion stems from two properties of PCIe systems. First, intra-
node P2P and NIC-based inter-node logical transfers traverse
the same PCIe hierarchy and may contend for GPU-facing
links and other shared PCIe links [32]. Second, PCIe fab-
rics are typically non-Clos [5,17,22,32], so logical transfers
that are endpoint-disjoint may still share bandwidth-limited
PCIe links. Figure 1(a) shows that endpoint-disjoint logical
transfers can still contend when their paths converge on the
same PCIe link. The resulting contention reduces the effec-
tive rates of all paths using that bottleneck and increases the
stage completion time while alternative paths remain under-
utilized. Accumulated across stages, this loss directly lowers
end-to-end AlltoAllv throughput.
Skewed demand creates long communication tails. All-
toAllv demand may be skewed, with some logical transfers
carrying substantially more data than others. Some paths may
consequently carry disproportionately heavy loads relative
to their available rates. These paths remain active after more
lightly loaded paths have finished and become idle, as illus-
trated in Figure 2(b). AlltoAllv completes only after its last
logical transfer, so these stragglers create a long communica-
tion tail that delays the entire invocation.
2.2
Why Existing Designs Fall Short
General-purpose communication libraries. NCCL [29]
incurs little runtime scheduling overhead but uses a static
AlltoAllv schedule that does not adapt to changing commu-
nication demand. MSCCLang [6] schedules logical transfers
by controlling their ordering and concurrency, while leaving
their physical paths fixed. This is insufficient for PCIe sys-
tems, where logically independent transfers may still contend
on shared physical links. Consequently, neither system can
adapt its schedule to rapidly changing AlltoAllv demand.
Demand-aware AlltoAllv schedulers for hierarchical sys-
tems. Some systems optimize AlltoAllv for servers with hier-
3


---

archical architectures that provide independent scale-up and
scale-out networks. FAST [20] uses additional intra-node for-
warding to balance scale-out traffic, whereas DeepEP [8] uses
it to reduce inter-node traffic. On PCIe GPU systems with-
out an independent scale-up fabric, however, these schemes
may fail to deliver their intended gains and instead exacer-
bate PCIe contention: their additional intra-node transfers
and NIC-based inter-node traffic traverse the same switched
PCIe hierarchy, increasing load on shared PCIe links. The
resulting contention reduces the effective bandwidth of con-
current transfers and ultimately lowers end-to-end AlltoAllv
throughput.
Topology-aware synthesis. TACCL [32], TE-CCL [23], and
SCCL [2] synthesize a schedule by jointly considering a speci-
fied communication demand and a detailed model of physical
paths and shared resources. Prior systems report synthesis
times ranging from minutes to hours [2,23,32]. Regenerating
such a schedule for every dynamic AlltoAllv invocation is
therefore impractical, particularly at scale.
2.3
Design Observations and Key Insight
Observation 1: Avoiding PCIe link contention requires
joint peer–path selection. Within an AlltoAllv stage, peer
matching determines which logical transfers execute con-
currently, while path assignment determines which physical
resources they use. These decisions jointly determine con-
tention: optimizing either one in isolation can still map mul-
tiple transfers onto the same bottleneck link while leaving
alternative paths underutilized. Figure 1 compares three pos-
sible stages in an AlltoAllv schedule: fastest-path assignment
(Stage A), path-only adaptation (Stage B), and joint peer–path
selection (Stage C). In Stages A and B, the cyclic match-
ing [20,26,36] uses k = 3, so rank i sends to (i+k) mod 6.
Under this peer matching, path-only adaptation raises mod-
eled aggregate bandwidth from 128 to 228 GB/s but leaves
PXB transfers contending. Stage C jointly changes match-
ing and paths, combining local PIX, noncontending PXB, and
NIC transfers to eliminate modeled PCIe contention and reach
356 GB/s.
Inspired by this observation, we use a channel to represent
the static peer–path configuration of a stage: concurrent GPU
pairs and one physical path per pair. Intuitively, a feasible
channel contains peer–path choices whose transfers can ex-
ecute concurrently without causing PCIe link contention, as
illustrated by Figure 1(c). Because feasible channels com-
bine different peer pairs and path rates, they provide different
transfer capabilities. Their feasibility depends only on the
topology, allowing them to be precomputed offline.
Observation 2: Skewed demand requires adaptive chunk-
to-path mapping. AlltoAllv demand may be skewed, and
each logical transfer may have multiple candidate paths with
different rates. Whole-flow binding can therefore imbalance
path loads even without link contention, leaving faster paths
idle while slower ones determine completion. Figure 2(b)
illustrates this mismatch: the PXB path becomes idle before
the NIC path finishes. In (c), splitting transfer A and remap-
ping its second chunk to NIC balances the two paths over two
rounds. This example shows why path assignments must be
revisited at chunk granularity: between rounds, outstanding
traffic can be remapped to match load to path rates.
Key insight.
Together, these observations reveal a natu-
ral separation between topology and demand. The topology
changes infrequently and determines which peer–path config-
urations can coexist without modeled contention; VarioPath
precomputes these configurations as reusable channels. At
runtime, VarioPath represents each invocation’s demand as
a linear combination of channels and assigns invocation-
specific data volumes to the selected paths. This separa-
tion replaces repeated combinatorial topology search with
lightweight online demand decomposition.
3
Scheduling Model
Model setup. We model an AlltoAllv invocation on a fixed
topology comprising a set V of N GPU ranks and a demand
matrix D ∈ZN×N
≥0 , where Dij denotes the data volume from
rank i to rank j. Self-traffic is handled locally and omitted,
so Dii = 0. For each ordered pair i̸ = j, let Pij denote the
nonempty set of available concrete paths. Each path p ∈Pij
has an estimated rate bw(p) > 0 and uses a set of modeled
resources L(p) ⊆U. The resource set U contains the di-
rected shared PCIe and NIC-facing resources considered by
our model. We exclude endpoint-local GPU-to-switch links
from U because the one-to-one peer matching defined below
ensures that no two lanes in a channel share the same endpoint
link in the same direction.
Table 1 summarizes the notation used in Sections 3 and 4.
Table 1: Key notation used in Sections 3 and 4.
Symbol
Definition
V
Set of all GPU ranks.
Va
Switch-local group of topology-equivalent GPU
ranks, indexed by a.
Rad
Set of group-level path instances from source group
Va to destination group Vd.
π
Nonself permutation mapping each source rank i to a
distinct destination π(i).
C
Channel specifying permutation-defined peer pairs
and one concrete communication path per pair.
E
Channel activation obtained by assigning a nonzero
lane-volume vector to a channel.
X(E)
Rank-level demand-contribution matrix served by
activation E.
M
Channel-shape matrix; Mad is the number of lanes
from Va to Vd.
B,B[M]
Channel family B is a set of group-level path in-
stances whose transfers can execute concurrently
without sharing a modeled resource; catalog bucket
B[M] contains all families that realize shape M.
Drem, bD
Unserved rank-level demand and its aggregation by
ordered rank-group pair.
4


---

Channel definition. A channel C = (π,p) consists of a per-
mutation π over V and a path vector p = (pi)i∈V. The map-
ping π assigns each source rank i to a distinct destination
π(i)̸ = i, and pi ∈Pi,π(i) selects the corresponding concrete
path. Because π is a permutation, each channel contains ex-
actly one outgoing and one incoming lane per rank, preventing
same-direction fan-out and fan-in at an endpoint. A channel
specifies only this peer–path pattern.
A channel activation E = (C,x) represents an AlltoAllv
stage, assigning a nonnegative integer volume xi to each lane
i ∈V. All nonzero-volume lanes execute concurrently in that
stage. Specifically, X(E)i,π(i) = xi for i ∈V, and all other en-
tries are zero. Thus, X(E) records the rank-level demand
served in that stage.
Channel feasibility. A channel is feasible under our model
when no two selected paths share a modeled directed resource:
∑
i∈V
1[ℓ∈L(pi)] ≤1,
∀ℓ∈U.
(1)
Opposite directions and parallel physical links are distinct
resources. We conservatively make each shared PCIe link
exclusive within a channel because concurrent use would
make path rates traffic-dependent; transfers sharing it run in
separate activations.
Schedule validity. A schedule is an ordered sequence of
activations S = ⟨E1,...,EK⟩, where Ek = (Ck,xk). The sched-
ule is valid if (i) every Ck is feasible and (ii) the activations
exactly reconstruct the requested demand, ∑K
k=1 X(Ek) = D.
This equality formalizes the linear combination view from
Section 2: each activation contributes a channel-shaped com-
ponent X(Ek), and these components sum to D.
Optimization objective. We estimate schedule completion
time under an execution model in which activations run se-
quentially and the lanes within each activation run concur-
rently. Let λ ≥0 denote a fixed per-activation overhead. Un-
der this model, an activation’s modeled duration is its fixed
overhead plus the completion time of its slowest lane. For
Ek = (Ck,xk) with permutation πk and paths pk, summing
these durations gives
bT(S) =
K
∑
k=1

λ+max
i∈V
xk,i
bw(pk,i)

,
S⋆∈arg min
S valid
bT(S).
(2)
The objective captures the trade-off between shortening com-
munication tails and limiting the overhead incurred by addi-
tional activations.
4
VarioPath Design
Overview. VarioPath constructs schedules through an offline–
online workflow, as shown in Figure 3. Offline, it represents
sets of feasible channels as channel families and organizes
the channel families in a reusable catalog tree for the fixed
topology. This catalog tree records feasible choices once,
OFFLINE
ONLINE
Topology
Catalog tree
Traffic demand
SW0
SW1
0
1
2
3
Channel family B1
Channel C1
Channel C2
Channel C3
0
1
2
3
(Sec. 4.2)  
decompose demand
NIC0
NIC1
64GB/s
 10MB
 30MB
Activation E1
 10MB
 10MB
Activation E4
 20MB
(Sec. 4.1) 
channel analysis
64GB/s
64GB/s64GB/s
Figure 3: VarioPath’s workflow. Offline analysis stores com-
patible channel families in a reusable catalog tree; online
channel-guided decomposition instantiates concrete channels
and allocates each invocation’s demand across the resulting
activations.
avoiding repeated topology analysis for each demand. Online,
each iteration instantiates candidate channels from the catalog,
assigns lane volumes, and selects the highest-utility activation,
progressively decomposing the demand.
4.1
Topology-Aware Channel Analysis
Topology abstraction with rank groups and path instances.
As discussed in Section 2, concurrent transfers can contend
on endpoint-local GPU-to-switch links and shared PCIe links,
especially inter-switch links. Because every channel con-
tains exactly one outgoing and one incoming lane per rank,
no endpoint-local link is reused in the same direction. We
therefore model only the remaining conflicts on shared PCIe
and NIC-facing resources in U. We partition the N = Gg
ranks into G switch-local rank groups Va, each containing
g topology-equivalent ranks on the same PCIe switch. A
group-level path instance r ∈Rad records one possible shared-
resource route from Va to Vd while omitting endpoint-local
GPU-to-switch attachments. Endpoint binding adds these at-
tachments later; in Figure 4, the same instance r1 can thus
connect GPU 0 to GPU 3 or GPU 1 to GPU 4.
Representing group-pair transfers as channel shapes. Be-
cause ranks within each group are topology-equivalent, their
identities do not affect the channel’s group-level structure.
Aggregating a channel’s lanes by ordered rank-group pair
therefore preserves its concurrency pattern. We encode these
lane counts in a matrix M, called a channel shape. Formally,
M ∈ZG×G
≥0 , where Mad is the number of lanes from Va to Vd.
An admissible shape has balanced rows and columns:
∑
d
Mad = g
∀a,
∑
a
Mad = g
∀d.
(3)
These constraints allocate exactly g outgoing and g incoming
positions to each group. Assigning the positions one-to-one
to its g ranks gives every rank one outgoing and one incoming
lane, as required by the permutation in the channel definition.
5


---

A larger Mad assigns more concurrent lanes to Va-to-Vd traf-
fic, allowing the scheduler to favor group pairs with greater
demand.
Grouping channels into channel families. A shape fixes the
number of lanes between each ordered group pair but leaves
their routes unspecified. A channel family B refines a shape
M by selecting group-level path instances while leaving their
rank-level endpoints unbound. Let Bad(r) ∈{0,1} indicate
whether instance r ∈Rad is selected. A family realizes M only
if it satisfies:
∑
r∈Rad
Bad(r) = Mad,
∀(a,d),
∑
a,d ∑
r∈Rad
Bad(r)1[ℓ∈L(r)] ≤1,
∀ℓ∈U.
(4)
The first constraint enforces shape consistency by selecting
exactly Mad path instances for every ordered group pair (a,d).
The second enforces resource compatibility by allowing each
modeled directed resource ℓ∈U to appear in at most one
selected instance. Each compatible family is stored in the
bucket B[M]; hence, M is realizable exactly when B[M]̸ = ∅.
Figure 4 shows that satisfying the balance constraints alone
does not guarantee realizability: M1 has no compatible family,
whereas B1 ∈B[M2] realizes M2.
Instantiating a channel from a family. A family fixes com-
patible path instances but represents multiple rank-level chan-
nels because their endpoints remain unbound. To instantiate
one channel, endpoint binding assigns each selected instance
r ∈Rad a pair (i, j) ∈Γad, where Γad = {(i, j) ∈Va ×Vd :
i̸ = j} excludes self-transfers. These choices are subject to
one-to-one source and destination assignments within each
group. Together with the balance constraints, a valid bind-
ing uses every rank exactly once as a source and once as a
destination, yielding the permutation required by Section 3.
Assigning (i, j) turns r into a concrete path p ∈Pi j by adding
its GPU-to-switch attachments. Under the group-equivalence
assumption, these attachments are excluded from U and pre-
serve the modeled footprint and configured rate: L(p) = L(r)
and bw(p) = bw(r).
Depth-first construction of the catalog tree. Algorithm 1
enumerates candidate path instances in a fixed order using
a pruned binary DFS. At state (t,B), B is the partial family
and rt the next candidate; the fixed order gives each subset
a unique decision sequence. The exclude branch skips rt,
whereas the include branch is explored only if CANADD
preserves resource disjointness and the group-degree limits.
SUFFIXOK stops when the remaining instances cannot fill
every group’s g outgoing and g incoming slots. Because these
tests reject only incompatible or uncompletable states, all
valid families remain reachable. COMPLETE holds once every
group has g selected outgoing and incoming instances. Then
B is valid; its group-pair counts determine shape M, and B is
inserted into B[M].
Pruning and search cost.
Dominance pruning operates
Algorithm 1 Offline catalog tree construction
Require: Path-instance sets {Rad}; group size g
Ensure: Pruned channel-family catalog tree B
1: order the instances as R = (r1,...,rm); B ←∅
2: EXPLORE(1,∅)
3: B[M] ←PRUNEFAMILIES(B[M]) for all M ∈dom(B)
4: return B
5: procedure EXPLORE(t,B)
6: if COMPLETE(B) then
7:
Mad ←|B∩Rad| for all (a,d)
8:
insert B into B[M]; return
9: end if
10: if t > m or not SUFFIXOK(t,B) then return
11: EXPLORE(t +1,B)
12: let rt ∈Rad
13: if CANADD(rt,B) then
14:
EXPLORE(t +1,B∪{rt})
15: end if
3
3
V0
V1
0
1
2
V0
V1
R01
No compatible family
B[M1] = { }
r3
2
2
V0
V1
1
1
V0
V1
r1,r2
r0
r5
3
4
5
R00
r1
Group-level topology
M1
M2
M3
...
Topo
Channel family B
B1
B2
...
Channel  C
0
1
2
3
4
5
V0
V1
Catalog
Tree
C1
C2
C3
...
unrealizable
 C1
r2
r4 R10
R11
r3,r4
Group-level channel
 shape M
Figure 4: Topology-aware construction of the channel-
family catalog tree. Balance constraints define admissible
shapes, while the group-level topology determines which
shapes have compatible families. Here M1 is unrealizable
because B[M1] = ∅, whereas B1 ∈B[M2] realizes M2. A one-
to-one endpoint binding then instantiates B1 as channel C1.
at two levels. Before the DFS, PRUNEPATHS removes a
path instance r when an alternative r′ for the same group
pair satisfies L(r′) ⊆L(r) and bw(r′) ≥bw(r), with at least
one strict inequality. Such an alternative uses no additional
modeled resource and offers no lower rate. After the DFS,
PRUNEFAMILIES removes exact duplicates and families dom-
inated within the same shape. One family dominates another
when their instances can be paired within every group pair
with identical footprints and no lower rates, with at least one
paired instance having a higher rate. Families with different
footprints remain as alternatives because they expose differ-
ent resource choices to endpoint binding. These two filters
therefore target different costs: path-level pruning reduces the
DFS branching factor, whereas family-level pruning limits the
catalog alternatives exposed to endpoint binding and online
scheduling.
The DFS may visit O(2|R |) candidate subsets in the worst
case, although resource checks, suffix pruning, and the two
dominance filters reduce the explored and stored choices in
6


---

practice. The catalog is built once per machine configuration
and reused across invocations, keeping this topology search
off the per-invocation critical path.
4.2
Channel-Guided Demand Decomposition
Online scheduling procedure. Algorithm 2 repeatedly de-
composes the remaining demand into channel activations;
Figure 5 illustrates one iteration. The scheduler initializes
Drem = D, where Drem is the demand not yet served. Each
iteration traverses the catalog from coarse to fine. It first ag-
gregates Drem by rank-group pair, scores all channel shapes,
and selects the highest-scoring shape M⋆. It then uses the
rank-level demand to bind every family B ∈B[M⋆] to end-
points, yielding a candidate channel CB. For each candidate, it
assigns lane volumes and evaluates the full modeled utility UB.
The scheduler appends the highest-utility activation E⋆, sub-
tracts its served demand from Drem, and repeats until Drem = 0.
This coarse-to-fine search narrows the topology choices at
the group level while retaining demand-aware endpoint and
volume assignment.
Step 1: Bottleneck-aware channel-shape selection. A chan-
nel shape specifies group-pair lane counts, so the scheduler
aggregates the remaining demand at the same granularity:
bDad = ∑
i∈Va ∑
j∈Vd
Drem
i j .
(5)
A group pair with larger bDad has more residual bytes to drain
and is more likely to determine the completion tail if assigned
insufficient concurrency. We therefore use bDad as its current
bottleneck pressure. Because Equation 3 gives every admissi-
ble shape the same total lane budget, allocating more lanes to
one group pair necessarily reduces the concurrency available
to others. We score each shape by
s(M) = ∑
a,d
bDadMad,
M⋆= arg
max
M∈dom(B)s(M).
(6)
The objective directs the fixed concurrency budget toward
high-pressure pairs, while restricting the search to dom(B)
guarantees topology feasibility. Recomputing the pressures
and M⋆after each update to Drem lets the scheduler follow
the evolving communication bottleneck.
Step 2: Binding paths to form candidate channels. For
each family B under M⋆, the scheduler binds its path instances
one-to-one to rank-level GPU pairs, producing a candidate
channel CB. Let τ be a fixed invocation-wide time quantum,
and let sB(r) and dB(r) denote r’s assigned endpoints. For
r ∈Rad and a valid pair (i, j) ∈Γad, define the useful-rate
credit
hijr = min
Drem
i j
τ
,bw(r)

,
(7)
with zero credit outside Γad. The two terms are the demand
rate needed within one quantum and r’s supported rate, so
Algorithm 2 Online channel-guided demand decomposition
Require: Feasible demand D; complete catalog tree B; τ > 0, λ ≥0; Alter-
nating Algorithm (AA) hyperparameters Nstart,IAA > 0
Ensure: Schedule S with ∑E∈S X(E) = D
1: Drem ←D; S ←⟨⟩
2: while ∥Drem∥1 > 0 do
3:
bDad ←∑i∈Va ∑j∈Vd Drem
ij
for all (a,d)
4:
M⋆←argmaxM∈dom(B) ∑a,d bDadMad
5:
for B ∈B[M⋆] do
6:
CB ←MULTISTARTAA(B,Drem,τ,Nstart,IAA)
7:
(xB,UB) ←EVALUATE(CB,Drem,τ,λ)
8:
end for
9:
B⋆←argmaxB∈B[M⋆]UB
10:
E⋆←(CB⋆,xB⋆)
11:
S ←S ∥⟨E⋆⟩
12:
Drem ←Drem −X(E⋆)
13: end while
14: return S
Remaining demand 
V0
V1
V2
V3
V0
V1
V2
V3
0 1
1 0
8 1
2 7
1 0
0 2
0 1
0 0
1 0
0 0
0 1
1 0
7 2
1 7
1 0
0 2
1 0
0 2
0 1
0 0
0 1
1 0
8 1
2 8
7 1
1 7
1 0
0 2
1 0
0 1
0 1
1 0
Group-level
demand view
2 18 3
1
1
2 17 3
3
1
2 19
16 3
2
2
High-demand
group pairs
Channel shape 
0 2 0 0
0 0 2 0
0 0 0 2
2 0 0 0
1.
Aggregate
2. Select
shape
3. Bind
endpoints
4. Assign volumes
& select 
5. Update
remaining demand
Candidate 
Activation 
Figure 5: One iteration of channel-guided demand decom-
position. The group-level view summarizes the remaining
demand Drem by rank-group pair and guides the selection of
channel shape M⋆. For each family B ∈B[M⋆], rank-level end-
point binding instantiates candidate channel CB. Candidate
volumes and utilities determine activation E⋆, which updates
Drem. Channels are shown schematically.
their minimum measures the useful assignment rate. We score
a complete binding by
FB(sB,dB) = ∑
r∈B
hsB(r),dB(r),r.
(8)
Thus, τFB is its provisional useful byte volume.
Because each credit jointly depends on both endpoints,
optimizing both one-to-one assignments yields a bilinear as-
signment problem (BAP) [7]. We search for a high-scoring
binding with the Alternating Algorithm (AA) [35]. Fixing ei-
ther endpoint assignment reduces the remaining optimization
to a standard assignment problem, so AA alternately opti-
mizes the source and destination assignments. Each update
solves one g×g assignment problem per rank group without
decreasing FB. A run ends when neither update improves the
score or the sweep limit is reached. Figure 6 illustrates this
process, where the displayed binding credit rises from 5 to
15.
7


---

(a) Inputs to Endpoint Binding
0
3
1
4
2
5
0
1
2
Family 
Equal rate estimates
3
4
5
(b) Binding-credit matrices and the Alternating Algorithm
3
4
5
0
1
2
4
1
3
5
8
2
7
3
5
Initial binding
3
4
5
0
1
2
4
1
3
5
8
2
7
3
5
After optimization
Alternating
Algorithm
1
2
3
4
Candidate 
Figure 6: Endpoint binding with the Alternating Algo-
rithm. (a) Family B provides two unbound, equal-rate path
instances from V1 to V2. (b) Colored entries in the credit ma-
trices hijr mark the current endpoint assignments. Alternating
source and destination optimization raises FB from 5 to 15
and produces candidate channel CB. Other family slots are
omitted for clarity.
To reduce initialization sensitivity, MULTISTARTAA runs
AA from Nstart feasible initial endpoint bindings. Each run
stops after IAA sweeps or when a complete sweep does not im-
prove FB. The procedure returns the highest-scoring channel
CB = (πB,pB) for evaluation in Step 3.
Step 3: Allocating volumes and selecting an activation.
For candidate CB, EVALUATE assigns each lane the largest
volume allowed by both its path capacity within τ and its
remaining demand:
xB,i = min
n
τbw(pB,i),Drem
i,πB(i)
o
.
(9)
We compare the resulting activations by utility UB: useful
bytes divided by modeled duration, including the slowest-
lane time and per-activation overhead from Section 3. We
define UB = 0 for a candidate with ∑i xB,i = 0; otherwise,
UB =
∑i∈V xB,i
λ+maxi∈V
xB,i
bw(pB,i)
.
(10)
The candidate with the largest UB supplies the next activation
E⋆. For feasible nonzero demand and the complete catalog
produced by Algorithm 1, a positive-scoring M⋆admits at
least one positive-volume candidate. Thus, every iteration
reduces Drem.
Planning complexity. Let J = |dom(B)| and H⋆= |B[M⋆]|.
Aggregating demand and scoring all shapes cost O(N2 +JG2)
per iteration. Each AA sweep solves 2G assignments of size
g×g; with Nstart starts and at most IAA sweeps, binding and
evaluating all families cost O
 H⋆(NstartIAAGg3 +N)

. Thus,
K activations cost O
 K[N2 +JG2 +H⋆(NstartIAAGg3 +N)]

;
Section 6.4 measures the resulting CPU overhead.
5
Implementation
VarioPath comprises about 3,000 lines of C++ for the offline
and online schedulers and 5,000 lines of C++/CUDA for its
runtime communication kernels. VarioPath implements all
inter-node communication with NCCL GPU-Initiated Net-
working (GIN) [29] over the GDAKI backend. Overall, the
implementation combines topology-aware path profiling with
device-resident data movement and synchronization, special-
izes transfers for sparse expert traffic and small messages, and
adapts execution granularity to each demand.
Topology, path, and rate model. At setup, we reconstruct
the topology from the GPU/HCA PCIe hierarchy and socket
affinity, enumerate supported direct-P2P and NIC paths, and
record their directed shared-resource footprints and band-
widths. Using this information, we construct the group-level
path instances described in Section 4.1. The runtime loads the
resulting profile once and reuses it across calls.
Device-resident buffer layout. AlltoAllv may scatter each
destination’s data across noncontiguous user-buffer regions.
Sending fragments separately creates small transfers and un-
derutilizes links. A customized packing kernel groups ele-
ments by destination rank and packs each peer’s metadata
and payload contiguously into the symmetric send buffer; the
receive kernel unpacks them directly into the output layout.
This fused layout conversion eliminates a separate memory-
reordering pass and avoids fine-grained transfers.
To avoid the high cost of a global barrier, measured at ap-
proximately 20 µs on our 32-GPU RTX 5090 testbed, we use
double buffering to eliminate the global barriers otherwise
required before and after resetting signals. We implement
this scheme with two symmetric-memory slots, each contain-
ing independent payloads, signal counts, indices, and other
invocation-related state variables. In practice, CUDA Graphs
capture and replay these communication operations. To avoid
host–device state inconsistencies across replays, the counters,
slot indices, and related state reside in device memory rather
than host memory.
Efficient sparse data movement for expert parallelism. In
sparse AlltoAllv workloads, source–destination rank pairs
with zero demand require neither data transfer nor comple-
tion waits. For a destination rank hosting L local experts,
each active source–destination pair uses a 32-bit control word:
the lower L bits indicate which local experts receive data,
while the remaining usable high-order bits encode the exact
token count for a designated local expert. Each local expert
is mapped to a queue pair (QP), using a dedicated QP when
resources permit and sharing QPs round-robin otherwise. The
sender posts the designated expert’s payload followed by the
control word on the same QP, so NIC-level in-order execution
guarantees data-before-control ordering without an additional
fence. The receiver uses the bitmap to skip inactive experts
and obtains active-expert counts from GIN signals, avoiding
separate count transfers and output-count atomics for empty
8


---

Name
GPU
#GPUs
per node
PCIe
generation
PCIe BW
(GB/s)
NIC
# NIC
per node
NIC BW/node
(Gb/s)
PCIe
topology
R5080-CX8
RTX 5080
8
Gen5
64
ConnectX-8
4
3200
Tree
R5080-BF3
RTX 5080
8
Gen5
64
BF3-mini
4
1600
Tree
R5090-BF3
RTX 5090
16
Gen5
64
BF3-mini
8
3200
Ring
L20-CX7
L20
16
Gen4
32
ConnectX-7
8
3200
Tree
Table 2: Evaluation platforms. PCIe generation and bandwidth describe each GPU’s PCIe version and nominal one-way capacity.
GPU/NIC counts and aggregate NIC bandwidth are per node.
groups.
Low-latency protocol. GIN signal mode appends an RDMA
atomic notification after the payload, incurring additional no-
tification overhead. NCCL LL avoids the separate notification
by embedding 8 bytes of flags alongside every 8 bytes of pay-
load in each 16-byte FIFO line, reducing payload utilization
to 50%. To achieve ultra-low latency without this overhead,
we design a specialized protocol. Receivers initialize idle
buffers to a sentinel value (e.g., negative zero), and senders
issue unsignaled puts. Receivers poll 16-byte vectors until no
sentinel remains. In a 1 KiB protocol microbenchmark on two
RTX 5090 servers, P50 latency is 7.872 µs; GIN signal and
prepacked NCCL LL take 12.928 and 9.726 µs, respectively.
To avoid split PCIe TLPs, we pad the receive-slot stride so
that each token’s receive address is 128-byte aligned.
Global execution granularity. Each invocation begins with a
target round count Ktarget, the approximate number of schedul-
ing rounds over which the current demand should be drained.
Given Ktarget, the current demand, and estimated path rates, the
runtime derives a global time quantum τ so that the transfer
is expected to complete in roughly Ktarget rounds. A smaller
Ktarget therefore yields a larger τ and fewer rounds, reducing
scheduling and synchronization overhead, whereas a larger
Ktarget yields a smaller τ and enables finer-grained adaptation
of peers and paths as demand drains. In practice, for an N-rank
invocation, we set Ktarget between N and 3N.
6
Evaluation
We evaluate VarioPath to answer four key questions:
• How does VarioPath compare with state-of-the-art sys-
tems across collective workloads, transfer sizes, and de-
ployment scales (§6.1)?
• What end-to-end latency reductions does VarioPath pro-
vide for MoE and sequence-parallel inference workloads
(§6.2)?
• How closely does VarioPath’s channel-level peer–path
co-design approach the full MILP optimum (§6.3)?
• How scalable are VarioPath’s offline catalog and online
planning (§6.4)?
Testbeds.
Table 2 lists the GPUs, PCIe generations and
bandwidths, NICs, and topology of our four platforms. Our
testbed comprises 32 R5080-CX8 nodes, 8 R5080-BF3 nodes,
4 R5090-BF3 nodes, and 2 L20-CX7 nodes. The nodes are
interconnected through a two-tier leaf–spine network.
Workloads. We evaluate VarioPath with random-demand
AlltoAllv, skew AlltoAllv, AlltoAll, and end-to-end inference.
In AlltoAll, every rank sends the same volume to every other
rank. Random-demand AlltoAllv uses demand matrices de-
rived from randomized top-k MoE routing over 256 experts,
with k = 8. Following FAST [20], we generate skew AlltoAllv
by drawing pairwise volumes from a Zipfian distribution. Our
end-to-end workloads cover expert-parallel Qwen3 prefill and
sequence-parallel Wan2.1 generation. A separate 256-GPU
experiment evaluates VarioPath’s large-scale scalability.
Metrics. Our primary communication metric is algorith-
mic bandwidth, computed as the total transferred payload
divided by the product of the number of ranks and completion
time [20]. Higher is better. Because skew AlltoAllv can make
ranks finish at different times, invocation completion time
is the maximum communication time across ranks. Aggre-
gate comparisons use geometric-mean speedup over matched
points. End-to-end experiments report latency. Planning ex-
periments report online CPU time per emitted activation and
the ratio of total online-planning time to communication time;
lower is better for both.
Baselines. Across the communication and end-to-end experi-
ments, we compare VarioPath with NCCL [29], MSCCL [6],
FAST [20], and DeepEP [8]1 MSCCL executes user-defined
schedules that control transfer ordering and concurrency.
FAST is built on NVSHMEM and rebalances skewed traf-
fic within each node before balanced one-to-one inter-node
transfers. DeepEP provides specialized MoE dispatch and
combine kernels over NCCL GIN.
6.1
Collective Communication Performance
We first evaluate random-demand AlltoAllv across transfer
sizes and scales, then AlltoAll and skew AlltoAllv, and finally
large-scale random-demand AlltoAllv on 256 GPUs. These
experiments test whether VarioPath’s benefits persist across
payload sizes, deployment scales, and traffic distributions.
Random-demand AlltoAllv. We vary the per-rank payload
on 32- and 64-GPU deployments with different local PCIe
organizations. R5080-BF3 uses four and eight 8-GPU nodes,
1We evaluate DeepEP V2 from its official GitHub repository.
9


---

32
256
2048
32768
Payload/rank (KiB)
0
8
16
24
Bandwidth (GB/s)
(a) R5080-BF3
4 nodes (32 GPUs)
32
256
2048
32768
Payload/rank (KiB)
(b) R5080-BF3
8 nodes (64 GPUs)
32
256
2048
16384
Payload/rank (KiB)
(c) R5090-BF3
2 nodes (32 GPUs)
32
256
2048
1638
Payload/rank (KiB)
(d) R5090-BF3
4 nodes (64 GPUs)
VarioPath
NCCL
FAST
DeepEP
Figure 7: Random-demand AlltoAllv bandwidth on 32- and 64-GPU deployments. Bandwidth is per-rank payload divided by
p50 completion time.
VarioPath
NCCL
MSCCL
1
4
16
64
512
Message size (MiB)
0
8
16
24
Bandwidth (GB/s)
(a) R5080-CX8, 1 node (8 GPUs)
1
4
16
64
512
Message size (MiB)
0
15
30
Bandwidth (GB/s)
(b) R5080-BF3, 1 node (8 GPUs)
1
4
16
64
512
Message size (MiB)
0
10
20
30
Bandwidth (GB/s)
(c) R5090-BF3, 1 node (16 GPUs)
1
4
16
64
512
Message size (MiB)
0
8
16
24
Bandwidth (GB/s)
(d) L20-CX7, 1 node (16 GPUs)
1
4
16
64
512
Message size (MiB)
0
8
16
24
Bandwidth (GB/s)
(e) R5080-CX8, 2 nodes (16 GPUs)
1
4
16
64
512
Message size (MiB)
0
15
30
Bandwidth (GB/s)
(f) R5080-BF3, 2 nodes (16 GPUs)
1
4
16
64
512
Message size (MiB)
0
10
20
30
Bandwidth (GB/s)
(g) R5090-BF3, 2 nodes (32 GPUs)
1
4
16
64
512
Message size (MiB)
0
8
16
24
Bandwidth (GB/s)
(h) L20-CX7, 2 nodes (32 GPUs)
Figure 8: AlltoAll bandwidth on one- and two-node deployments. Bandwidth is per-rank payload divided by completion time.
whereas R5090-BF3 uses two and four 16-GPU nodes. Fig-
ure 7 shows that VarioPath achieves the highest bandwidth
on all 42 matched test points. Its geometric-mean speedup
over DeepEP, the closest baseline, ranges from 1.55× to
2.00×. Over FAST, it ranges from 3.49× to 9.41×, and over
NCCL from 3.73× to 10.61×. By jointly selecting each chan-
nel’s peer matching and physical paths, VarioPath avoids con-
centrating concurrent flows on shared PCIe links and maps
chunks onto concurrently usable PCIe and NIC paths.
To examine scaling, we fix the per-rank payload while in-
creasing the number of GPUs. At 32 MiB on R5080-BF3,
scaling from 32 to 64 GPUs reduces VarioPath bandwidth by
8.6% and FAST bandwidth by 5.7%, while VarioPath remains
1.72× faster. At 16 MiB per rank on R5090-BF3, the corre-
sponding reductions are 3.6% and 12.6%. Thus, VarioPath
retains its bandwidth advantage at 64 GPUs on both platforms
by re-selecting channels across rounds as individual transfers
complete, instead of leaving the additional flows pinned to a
small set of shared links.
AlltoAll. VarioPath’s advantage persists under uniform traf-
fic distributions. Across the 80 configurations in Figure 8, it
achieves geometric-mean speedups of 1.82× over NCCL and
1.99× over MSCCL. It outperforms NCCL in every configu-
ration and MSCCL in all but the 1-MiB, single-node R5080-
BF3 case. NCCL fixes transport paths at communicator setup,
and MSCCL inherits those paths. Neither baseline can adapt
an active pair’s physical path to the current stage. VarioPath in-
stead adapts paths by stage, distributing uniform traffic across
usable PCIe and NIC paths and avoiding concentration on
shared links. The result confirms that this benefit is not limited
to skewed traffic.
Skew AlltoAllv. We increase the Zipf skewness factor from
0.4 to 0.8 on all four two-node deployments. Larger factors
concentrate more traffic in fewer rank pairs, making the de-
mand harder to distribute across concurrent paths. Figure 9
shows that VarioPath remains fastest in all 12 deployment–
skew configurations, with a geometric-mean speedup of
2.16× over FAST, ranging from 1.99× to 2.39× across plat-
10


---

0.4
0.6
0.8
Skewness factor s
0
10
20
30
Bandwidth (GB/s)
(a) R5080-CX8, 2 nodes
(16 GPUs)
0.4
0.6
0.8
Skewness factor s
(b) R5080-BF3, 2 nodes
(16 GPUs)
0.4
0.6
0.8
Skewness factor s
(c) R5090-BF3, 2 nodes
(32 GPUs)
0.4
0.6
0.8
Skewness factor s
(d) L20-CX7, 2 nodes
(32 GPUs)
NCCL
FAST
VarioPath
Figure 9: Skew AlltoAllv bandwidth on four two-node deployments. Each bar averages five Zipfian workload seeds.
1
2
4
8
16
32
64
128
M
0
10
20
30
AlgoBW (GB/s)
NCCL
DeepEP
VarioPath
Figure 10: Large-scale random-demand AlltoAllv on 256
R5080-CX8 GPUs. M is the number of input tokens per rank;
hidden dimension is 7168 and top-k is 8.
EP16
EP24
EP32
0
1000
2000
3000
Latency (ms)
19.1%
23.7%
27.2%
(a) Qwen3 prefill
SP8
H8C1
SP8
H4C2
SP16
H4C4
SP16
H8C2
0
400
800
1200
Latency (s)
6.1%
1.2%
1.1%
5.5%
(b) Wan2.1 generation
NCCL
VarioPath
Figure 11: End-to-end latency for (a) Qwen3 prefill and
(b) Wan2.1 generation. Labels report latency reduction from
NCCL to VarioPath.
forms. From s = 0.4 to 0.8, its bandwidth drops 9.4–12.3%,
versus NCCL’s 15.2–27.0%. FAST declines less (2.0–6.0%),
but starts from a substantially lower bandwidth and remains
slower throughout the sweep. FAST’s node-local rebalancing
is less effective on PCIe systems because its additional intra-
node forwarding competes with GPU-to-NIC transfers for
shared PCIe bandwidth. Balancing NIC traffic can therefore
leave the PCIe bottleneck unresolved or intensify it. By re-
scoring families and rebinding peers, paths, and chunks after
each activation, VarioPath can redirect outstanding traffic to
newly available path capacity.
Large-scale experiment. We further evaluate VarioPath on
32 R5080-CX8 nodes with 256 GPUs using a random-demand
AlltoAllv workload generated by top-k MoE routing with
k = 8 and hidden dimension 7168. Figure 10 sweeps the per-
rank input-token count M from 1 to 128. VarioPath achieves
the highest algorithmic bandwidth at all eight values of M and
reaches 30.91 GB/s at M = 128, compared with 28.11 GB/s
for DeepEP and 18.85 GB/s for NCCL. This result shows
that demand-aware channel scheduling continues to expose
usable PCIe and NIC capacity when the deployment grows to
hundreds of GPUs.
6.2
End-to-End Inference Performance
We replace NCCL-based AlltoAllv communication with Var-
ioPath in uncached Qwen3–30B–A3B prefill [41, 44] and
Wan2.1 video generation [37]. Qwen runs on two to four
R5080-BF3 nodes with EP16, EP24, and EP32; each trial
prefills 8,192 input tokens and generates one token, and
each point is the client median of 15 requests. Wan2.1 runs
on 8 and 16 L20 GPUs and generates a five-second BF16
1280×720 video with sequence length 75,600, 40 attention
heads, and head dimension 128. In Figure 11, H and C de-
note the head- and context-parallel factors within the stated
sequence-parallel degree.
For Qwen3–30B–A3B, replacing the exchange with Var-
ioPath reduces prefill latency by 19.1–27.2% across EP16,
EP24, and EP32. The reduction increases with the expert-
parallel degree because expert dispatch and combine lie on
the prefill critical path, and larger degrees expose more cross-
GPU and cross-node traffic to demand-aware channel schedul-
ing.
For Wan2.1, the speedup ranges from 1.011× to 1.066×
across the four sequence-parallel layouts, with the largest
gains at H8C1 and H8C2. At a fixed sequence-parallel degree,
comparing H8C1 with H4C2 and H8C2 with H4C4 shows
how the head/context decomposition reshapes rank-pair de-
mand. Thus, the end-to-end benefit depends on workload and
layout, not the sequence-parallel degree alone.
6.3
Design Validation
We first use an ablation study to quantify the contribution of
joint peer–path selection, and then compare VarioPath with
a full MILP optimum to measure the optimality gap of its
online scheduler.
11


---

8
16
32
64 128
0
10
20
30
BW (GB/s)
(a) R5080-BF3, 8 GPUs
8
16
32
64 128
0
8
16
24
(b) R5080-BF3, 16 GPUs
97
98
99
100
Ratio to MILP (%)
Input demand per rank (MiB)
MILP solver
Mean ratio
VarioPath
Figure 12: Scheduler quality against a full MILP optimum.
Bars show mean per-rank bandwidth, and the line shows the
mean VarioPath/MILP ratio over five demand matrices per
size.
Table 3: Peer–path co-design ablation on two-node R5080-
BF3. Values are completion time normalized to demand-
aware peers with dynamic paths.
Fixed nearest path
Dynamic path
Fixed peer rotation
2.84×
2.00×
Demand-aware peers
2.03×
1.00×
Peer/path ablation. This ablation tests the offline channel
hypothesis that high-throughput execution requires jointly
retaining peer matchings and path choices that avoid modeled
shared-resource contention. To isolate this effect, we keep
all other system components unchanged and vary only the
eligible channel set. We replay the same five skew AlltoAllv
demand matrices with s = 0.6 on two-node R5080-BF3. Fixed
peers restrict each stage to a cyclic peer rotation, while fixed
paths use each pair’s preassigned nearest path; the other di-
mension remains adaptive. Table 3 shows that fixing either
dimension approximately doubles completion time, while fix-
ing both increases it to 2.84× that of the unrestricted design.
Thus, peer–path co-design is necessary to retain the unre-
stricted design’s performance on this workload. With only
dynamic paths, a cyclic peer rotation can still select flows that
compete for the same shared links; with only demand-aware
peers, preassigned paths cannot redirect those flows to idle
capacity. The similar penalties of the two one-sided variants
show that neither decision substitutes for the other.
Scheduler quality. To evaluate scheduling quality, we com-
pare VarioPath with a full MILP optimum on one- and two-
node R5080-BF3 deployments. We use per-rank input de-
mands of 8, 16, 32, 64, and 128 MiB and evaluate five in-
dependently generated dense Zipf demand matrices for each
topology and input size, giving 50 complete skew AlltoAllv
calls. Following the modeling approach of TE-CCL [23], the
MILP jointly optimizes concrete peer matchings, physical
paths, transfer sizes, and their execution order over the en-
tire call. It searches the complete modeled path space, and
every instance reaches a solver-certified optimum. Figure 12
reports the algorithmic bandwidth of both methods. Across all
Table 4: Catalog tree scale and online planning overhead.
Catalog tree size is the estimated compact serialized payload.
Online is planning time per emitted activation; O/C is total
online-planning time divided by communication time.
Deployment
#
Shapes
#
Families
Catalog
tree
(KiB)
Offline
(s)
Online
(µs/act.)
O/C
(%)
R5080-BF3–8 GPUs
3
6
1.7
0.800
3.2 0.0715
R5080-BF3–16 GPUs
41
438
132.0
4.950
20.6 0.7554
R5080-CX8–256 GPUs
1616
2140 172682.6 5673.603
78.12 4.1241
R5090-BF3–16 GPUs
115
248
123.0
15.520
8.3 0.3249
R5090-BF3–32 GPUs
308
389
806.7
99.250
21.1 1.1051
L20-CX7–16 GPUs
92
270
121.3
11.720
6.8 0.1476
L20-CX7–32 GPUs
352
462
905.0
86.940
20.8 0.7504
50 instances, VarioPath attains 99.18% of the MILP-optimal
bandwidth on average, corresponding to an average optimality
gap of 0.82%. After averaging within each topology–input-
size configuration, the ratio ranges from 98.34% to 99.41%,
with a minimum of 91.92% among individual instances.
6.4
Catalog Tree Scale and Planning Overhead
Catalog tree size. Table 4 reports offline catalog construction
across deployments from 8 to 256 GPUs. The retained group-
level trees contain 3–1,616 shapes and 6–2,140 families, oc-
cupy between 1.7 KiB and 168.6 MiB, and take between
0.8 s and 94.6 min to construct. Construction applies compat-
ibility filtering, structural deduplication, and rate-dominance
pruning without expanding concrete rank bindings, so one
family represents many concrete channels. This one-time cost
is amortized across demands on the same topology. Equal-
size R5090-BF3 and L20-CX7 deployments retain different
numbers of shapes and families, showing that catalog scale
depends on topology as well as GPU count.
Planning cost. We average online decomposition over 100
1-GiB-per-source demands. Planning takes 3.2–78.12 µs per
activation; total planning time accounts for 0.0715–4.1241%
of communication time (O/C in Table 4). Even at 256 GPUs,
planning takes 78.12 µs per activation, while total planning
time accounts for 4.1241% of communication time. Encoding
topology feasibility offline limits online work to family search,
endpoint binding, and transfer-size assignment, enabling per-
invocation adaptation at substantially lower cost than reported
MILP-based synthesis times [23,32].
7
Conclusion
In this paper, we present VarioPath, an AlltoAllv framework
for PCIe GPU systems that separates topology-dependent
channel analysis from demand-dependent scheduling. Its of-
fline analyzer compresses feasible channels into reusable fam-
ilies, while its lightweight online scheduler decomposes each
invocation’s demand into channel activations without repeat-
ing topology search. Across four PCIe platforms and deploy-
12


---

ments of up to 256 GPUs, VarioPath outperforms existing
systems and reduces Qwen3 prefill and Wan2.1 generation
latency by up to 27.2% and 6.1%.
References
[1] Jehoshua Bruck, Ching-Tien Ho, Shlomo Kipnis, Eli
Upfal, and Derrick Weathersby. Efficient algorithms for
all-to-all communications in multiport message-passing
systems. IEEE Transactions on Parallel and Distributed
Systems, 8(11):1143–1156, 1997.
[2] Zixian Cai, Zhengyang Liu, Saeed Maleki, Madanlal
Musuvathi, Todd Mytkowicz, Jacob Nelson, and Olli
Saarikivi. Synthesizing optimal collective algorithms.
In Proceedings of the 26th ACM SIGPLAN Symposium
on Principles and Practice of Parallel Programming,
pages 62–75, 2021.
[3] Jiamin Cao, Shangfeng Shi, Jiaqi Gao, Weisen Liu, Yi-
fan Yang, Yichi Xu, Zhilong Zheng, Yu Guan, Kun Qian,
Ying Liu, et al. Syccl: Exploiting symmetry for efficient
collective communication scheduling. In Proceedings of
the ACM SIGCOMM 2025 Conference, pages 645–662,
2025.
[4] Minsik Cho, Ulrich Finkler, David Kung, and Hillery
Hunter. BlueConnect: Decomposing all-reduce for deep
learning on heterogeneous network hierarchy. In Pro-
ceedings of Machine Learning and Systems, volume 1,
pages 241–251, 2019.
[5] Charles Clos. A study of non-blocking switching net-
works. Bell System Technical Journal, 32(2):406–424,
1953.
[6] Meghan Cowan, Saeed Maleki, Madanlal Musuvathi,
Olli Saarikivi, and Yifan Xiong. MSCCLang: Microsoft
collective communication language. In Proceedings of
the 28th ACM International Conference on Architectural
Support for Programming Languages and Operating
Systems, Volume 2, pages 502–514, 2023.
[7] Ante ´Custi´c, Vladyslav Sokol, Abraham P. Punnen, and
Binay Bhattacharya. The bilinear assignment problem:
Complexity and polynomially solvable special cases.
Mathematical Programming, 166(1–2):185–205, 2017.
[8] DeepSeek. DeepEP: Expert parallel communication li-
brary. https://github.com/deepseek-ai/DeepEP,
2025.
[9] Jiangsu Du, Hongbin Zhang, Taosheng Wei, Zhenyi
Zheng, Jiazhi Jiang, Kaiyi Wu, Zhiguang Chen, and Yu-
tong Lu. Efficient LLM serving on commodity GPU
clusters with data-reduced cross-instance orchestration.
In 20th USENIX Symposium on Operating Systems De-
sign and Implementation (OSDI 26), pages 1787–1802.
USENIX Association, 2026.
[10] Jiarui Fang and Shangchun Zhao. USP: A unified se-
quence parallelism approach for long context generative
AI. arXiv preprint arXiv:2405.07719, 2024.
[11] William Fedus, Barret Zoph, and Noam Shazeer. Switch
transformers: Scaling to trillion parameter models with
simple and efficient sparsity. Journal of Machine Learn-
ing Research, 23(120):1–39, 2022.
[12] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei
Zaharia. MegaBlocks: Efficient sparse training with
mixture-of-experts. In Proceedings of Machine Learn-
ing and Systems, volume 5, pages 288–304, 2023.
[13] Adithya
Gangidi, Rui
Miao, Shengbao
Zheng,
Sai Jayesh Bondu, Guilherme Goes, Hany Morsy, Rohit
Puri, Mohammad Riftadi, Ashmitha Jeevaraj Shetty,
Jingyi Yang, et al. RDMA over ethernet for distributed
training at Meta scale.
In Proceedings of the ACM
SIGCOMM 2024 Conference, pages 57–70, 2024.
[14] Diandian Gu, Peng Sun, Qinghao Hu, Ting Huang, Xun
Chen, Yingtong Xiong, Guoteng Wang, Qiaoling Chen,
Shangchun Zhao, Jiarui Fang, et al. LoongTrain: Effi-
cient training of long-sequence LLMs with head-context
parallelism. arXiv preprint arXiv:2406.18485, 2024.
[15] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang,
Fuwen Luo, Shangfeng Shi, and Qin Li. FasterMoE:
Modeling and optimizing training of large-scale dy-
namic pre-trained models. In Proceedings of the 27th
ACM SIGPLAN Symposium on Principles and Practice
of Parallel Programming, pages 120–134, 2022.
[16] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang,
Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose,
Prabhat Ram, HoYuen Chau, Peng Cheng, Fan Yang,
Mao Yang, and Yongqiang Xiong.
Tutel: Adaptive
mixture-of-experts at scale. In Proceedings of Machine
Learning and Systems, volume 5, pages 269–287, 2023.
[17] Heehoon Kim, Junyeol Ryu, and Jaejin Lee. TCCL:
Discovering better communication paths for PCIe GPU
clusters. In Proceedings of the 29th ACM International
Conference on Architectural Support for Programming
Languages and Operating Systems, Volume 3, pages 999–
1015, 2024.
[18] JQ Lau, Ma Xiong, and Syona Sarma. Cloudflare’s 12th
generation servers—145% more performant and 63%
more efficient. Cloudflare Blog, 2024.
13


---

[19] Yiran Lei, Dongjoo Lee, Liangyu Zhao, Daniar Kur-
niawan, Chanmyeong Kim, Heetaek Jeong, Changsu
Kim, Hyeonseong Choi, Liangcheng Yu, Arvind Krish-
namurthy, et al. FLASH: Fast all-to-all communica-
tion in GPU clusters. arXiv preprint arXiv:2505.09764,
2025.
[20] Yiran Lei, Dongjoo Lee, Liangyu Zhao, Daniar Kur-
niawan, Chanmyeong Kim, Heetaek Jeong, Changsu
Kim, Hyeonseong Choi, Liangcheng Yu, Arvind Krish-
namurthy, Justine Sherry, and Eriko Nurvitadhi. FAST:
An efficient scheduler for all-to-all GPU communication.
In 23rd USENIX Symposium on Networked Systems De-
sign and Implementation (NSDI 26), pages 2515–2531.
USENIX Association, 2026.
[21] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, De-
hao Chen, Orhan Firat, Yanping Huang, Maxim Krikun,
Noam Shazeer, and Zhifeng Chen. GShard: Scaling gi-
ant models with conditional computation and automatic
sharding. In International Conference on Learning Rep-
resentations, 2021.
[22] Ang Li, Shuaiwen Leon Song, Jieyang Chen, Jiajia Li,
Xu Liu, Nathan R Tallent, and Kevin J Barker. Evaluat-
ing modern GPU interconnect: PCIe, NVLink, NV-SLI,
NVSwitch, and GPUDirect. IEEE Transactions on Par-
allel and Distributed Systems, 31(1):94–110, 2019.
[23] Xuting Liu, Behnaz Arzani, Siva Kesava Reddy Kakarla,
Liangyu Zhao, Vincent Liu, Miguel Castro, Srikanth
Kandula, and Luke Marshall. Rethinking machine learn-
ing collective communication as a multi-commodity
flow problem. In Proceedings of the ACM SIGCOMM
2024 Conference, pages 16–37, 2024.
[24] Meituan LongCat Team. LongCat-Flash technical report.
arXiv preprint arXiv:2509.01322, 2025.
[25] Microsoft. Microsoft collective communication library
(MSCCL). https://github.com/microsoft/msccl.
Accessed September 10, 2026.
[26] Naeris Netterville, Ke Fan, Sidharth Kumar, and Thomas
Gilray. A visual guide to mpi all-to-all. In 2022 IEEE
29th International Conference on High Performance
Computing, Data and Analytics Workshop (HiPCW),
pages 20–27. IEEE, 2022.
[27] NVIDIA.
NVIDIA fabric manager user guide.
https://docs.nvidia.com/hgx-platforms/
fabric-manager-user-guide/index.html,
2025.
Accessed September 11, 2026.
[28] NVIDIA.
NVIDIA NVLink and NVLink Switch.
https://www.nvidia.com/en-us/data-center/
nvlink/, 2025.
[29] NVIDIA. NCCL user guide: Point-to-point communi-
cation. https://docs.nvidia.com/deeplearning/
nccl/user-guide/docs/usage/p2p.html, 2026.
[30] Pitch Patarasuk and Xin Yuan. Bandwidth optimal all-
reduce algorithms for clusters of workstations. Journal
of Parallel and Distributed Computing, 69(2):117–124,
2009.
[31] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Min-
jia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad
Awan, Jeff Rasley, and Yuxiong He. DeepSpeed-MoE:
Advancing mixture-of-experts inference and training to
power next-generation AI scale. In Proceedings of the
39th International Conference on Machine Learning,
volume 162, pages 18332–18346. PMLR, 2022.
[32] Aashaka Shah, Vijay Chidambaram, Meghan Cowan,
Saeed Maleki, Madan Musuvathi, Todd Mytkowicz, Ja-
cob Nelson, Olli Saarikivi, and Rachee Singh. TACCL:
Guiding collective algorithm synthesis using communi-
cation sketches. In 20th USENIX Symposium on Net-
worked Systems Design and Implementation (NSDI 23),
pages 593–612, 2023.
[33] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz,
Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff
Dean. Outrageously large neural networks: The sparsely-
gated mixture-of-experts layer. In International Confer-
ence on Learning Representations, 2017.
[34] Brian Slechta, Nick Comly, Ashraf Eassa, Joe DeLaere,
and Shivam Raj.
NVIDIA NVLink and NVIDIA
NVSwitch supercharge large language model inference.
NVIDIA Technical Blog, August 2024.
[35] Vladyslav Sokol, Ante ´Custi´c, Abraham P. Punnen,
and Binay Bhattacharya. Bilinear assignment problem:
Large neighborhoods and experimental analysis of algo-
rithms. INFORMS Journal on Computing, 32(3):730–
746, 2020.
[36] Rajeev Thakur, Rolf Rabenseifner, and William Gropp.
Optimization of collective communication operations in
mpich. The International Journal of High Performance
Computing Applications, 19(1):49–66, 2005.
[37] Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and
advanced large-scale video generative models. arXiv
preprint arXiv:2503.20314, 2025.
[38] Guanhua Wang, Shivaram Venkataraman, Amar Phan-
ishayee, Nikhil Devanur, Jorgen Thelin, and Ion Stoica.
Blink: Fast and generic collectives for distributed ML.
In Proceedings of Machine Learning and Systems, vol-
ume 2, pages 172–186, 2020.
14


---

[39] Qizhen Weng, Wencong Xiao, Yinghao Yu, Wei Wang,
Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin,
and Yu Ding. MLaaS in the wild: Workload analysis
and scheduling in Large-Scale heterogeneous GPU clus-
ters. In 19th USENIX Symposium on Networked Systems
Design and Implementation (NSDI 22), pages 945–960.
USENIX Association, 2022.
[40] Changbo Wu, Zhuolong Yu, Gongming Zhao, and
Hongli Xu. Enabling reconfiguration-communication
overlap for collective communication in optical net-
works. arXiv preprint arXiv:2510.19322, 2025.
[41] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chen-
gen Huang, Chenxu Lv, et al. Qwen3 technical report.
arXiv preprint arXiv:2505.09388, 2025.
[42] Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang,
Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui,
Size Zheng, Li-Wen Chang, et al. Comet: Fine-grained
computation-communication overlapping for mixture-
of-experts. arXiv preprint arXiv:2502.19811, 2025.
[43] Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai
Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan
Huang, Shangyan Zhou, Shirong Ma, et al. Insights
into DeepSeek-V3: Scaling challenges and reflections
on hardware for AI architectures. In Proceedings of
the 52nd Annual International Symposium on Computer
Architecture, 2025.
[44] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue
Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos
Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Bar-
rett, and Ying Sheng. SGLang: Efficient execution of
structured language model programs. arXiv preprint
arXiv:2312.07104, 2024.
[45] Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Ce-
sar A Stuardo, Dongyang Wang, Xinlei Zhang, Huaping
Zhou, Haoran Wei, Yang Cheng, et al. MegaScale-Infer:
Serving mixture-of-experts at scale with disaggregated
expert parallelism. arXiv preprint arXiv:2504.02263,
2025.
15
