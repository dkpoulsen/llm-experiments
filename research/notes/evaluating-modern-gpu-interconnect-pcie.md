---
title: 'Evaluating Modern GPU Interconnect: PCIe,'
id: evaluating-modern-gpu-interconnect-pcie
tags:
- cmp170hx-training-feasibility-a0d4fc
- sub-q3
- pcie-fabric
- nvlink
- no-p2p
- measured
- interconnect-benchmark
- numa-effects
- nccl
- tartan
created: '2026-10-27T11:12:03.874231Z'
updated: '2026-10-27T19:00:31.425351Z'
source: https://arxiv.org/abs/1903.04611
source_domain: arxiv.org
fetched_at: '2026-10-27T11:12:03.873701Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Li, Song, Chen, Li, Liu, Tallent & Barker (PNNL; arXiv 1903.04611, IEEE
  TPDS) is the canonical primary evaluation of modern GPU interconnects - PCIe, NVLink-V1,
  NVLink-V2, NV-SLI, NVSwitch and GPUDirect-RDMA - across six platforms: P100-DGX-1,
  V100-DGX-1, DGX-2, OLCF SummitDev, Summit, and a 2x Turing RTX-2080 SLI system.
  They measure raw startup latency, sustainable uni/bidirectional bandwidth, topology,
  routing and NUMA effects under both P2P and collective (NCCL) patterns.


  Four new NUMA effects are identified: three triggered by NVLink topology/connectivity/routing,
  and one by a PCIe chipset design issue they label "anti-locality" - nearby access
  presenting LOWER performance than remote access, observed only for bandwidth (not
  latency) and appearing for messages >64 KB. PCIe latency is essentially uniform
  across pairs (same through one PCIe switch, across a local CPU socket, or across
  the QPI bridge), i.e. no NUMA effect in PCIe latency; NVLink latency is ~9 us direct
  and rises ~2x (P100) / ~3x (V100) when routing is required. NV-SLI: local 5 us,
  NV-SLI 8 us, PCIe 13 us. NVSwitch gives 50 GB/s bidirectional per link, 300 GB/s
  via a single NVSwitch hop, and 2.4 TB/s baseboard-to-baseboard bisection.


  Message-size saturation points differ by an order of magnitude: NCCL collective
  bandwidth saturates at ~16 MB on PCIe but ~256 MB on NVLink (8 GPUs); on a 2-GPU
  SLI system both PCIe and NV-SLI saturate at ~1 MB.


  THE finding for this run: "the NUMA effects appear quite significant with 5 GPUs,
  implying that there may be a congestion when forming a NCCL ring among 5 GPUs. One
  should avoid adopting 5 GPUs in their application setup." The box under assessment
  has exactly 5 GPUs, so this is a direct, primary-sourced warning that a 5-rank NCCL
  ring is a pathological configuration on PCIe.


  Their discussion adds two structural cautions: NVLink''s advantage is bandwidth,
  not latency, over PCIe; and a faster interconnect alone "can hardly become significant
  enough to lift the entire application''s speedup" unless the CPU-centric master-slave
  programming model is replaced or communication control moves to the GPU (e.g. NVSHMEM).
  NVLink also introduces its own overheads (enable/disable peer access, routing, NCCL
  initialization).'
raw_file: raw/evaluating-modern-gpu-interconnect-pcie.pdf
doi: arXiv:1903.04611
utility_score: 15.0
citation_count: 338
venue: IEEE Transactions on Parallel and Distributed Systems
---

*Suggested by [[efficient-multi-path-nvlinkpcie-aware-ucx-based-collective-communication-for-dee]] — cites Li/Song/Barker 'Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU Benchmark Suite' as the primary multi-GPU interconnect benchmark*

1
Evaluating Modern GPU Interconnect: PCIe,
NVLink, NV-SLI, NVSwitch and GPUDirect
Ang Li, Shuaiwen Leon Song, Jieyang Chen, Jiajia Li, Xu Liu, Nathan Tallent, and Kevin Barker
Abstract—High performance multi-GPU computing becomes an inevitable trend due to the ever-increasing demand on computation
capability in emerging domains such as deep learning, big data and planet-scale simulations. However, the lack of deep understanding
on how modern GPUs can be connected and the real impact of state-of-the-art interconnect technology on multi-GPU application
performance become a hurdle. In this paper, we ﬁll the gap by conducting a thorough evaluation on ﬁve latest types of modern GPU
interconnects: PCIe, NVLink-V1, NVLink-V2, NVLink-SLI and NVSwitch, from six high-end servers and HPC platforms: NVIDIA
P100-DGX-1, V100-DGX-1, DGX-2, OLCF’s SummitDev and Summit supercomputers, as well as an SLI-linked system with two
NVIDIA Turing RTX-2080 GPUs. Based on the empirical evaluation, we have observed four new types of GPU communication network
NUMA effects: three are triggered by NVLink’s topology, connectivity and routing, while one is caused by PCIe chipset design issue.
These observations indicate that, for an application running in a multi-GPU node, choosing the right GPU combination can impose
considerable impact on GPU communication efﬁciency, as well as the application’s overall performance. Our evaluation can be
leveraged in building practical multi-GPU performance models, which are vital for GPU task allocation, scheduling and migration in a
shared environment (e.g., AI cloud and HPC centers), as well as communication-oriented performance tuning.
Index Terms—Performance Evaluation, GPU, Interconnect, NUMA, PCIe, NVLink, NVSwitch, SLI, GPUDirect, RDMA, NCCL
!
1
INTRODUCTION
M
ULTI-GPU
execution
nowadays
becomes
an
in-
evitable trend for warehouse GPGPU computing.
This is due to the ever-increasing demand of computation
capability from emerging domains such as machine learn-
ing, big data and planet-scale simulations [1], [2]. With in-
creasingly larger problem to solve, scalable GPU computing
becomes necessary. Recently, a research group from Sony
leveraged 3,456 GPUs to train a ResNet-50 neural network
for ImageNet in 122 seconds, achieving near optimal GPU
scaling efﬁciency [3]. The Swiss National Supercomputing
Center (CSCS) relied on 4,888 GPUs in the Piz Daint su-
percomputer to simulate near-global climate in ultra-high
resolution [2].
Multi-GPU execution scales in two directions: vertically
scaling-up in a single node and horizontally scaling-out across
multiple nodes. Good examples to describe the intra-node
scale-up scenario are the latest NVIDIA DGX-1 [4] and
DGX-2 [5] super-AI servers, which incorporate 8 and 16
P100/V100 GPUs connected by NVLink and NVSwitch,
respectively. For the inter-node scale-out scenario, the U.S.
Department of Energy (DOE) has recently deployed two GPU-
accelerated supercomputers Summit [6] and Sierra [7] in Oak
Ridge and Livermore National Laboratories, with more than
3400 GPU-integrated nodes interconnected.
Gaining performance from multi-GPU scaling, however,
is not trivial, mainly because (i) There are no mature multi-
GPU parallel programming, execution and performance
models, largely due to the limited knowledge on how mod-
ern GPUs are interconnected as well as their communication
patterns; (ii) Traditionally, inter-GPU communication shares
•
Ang Li is a research scientist in Paciﬁc Northwest National Laboratory.
E-mail: ang.li@pnnl.gov
Manuscript received January 19, 2019.
the same bus interconnect as CPU-GPU communication,
such as PCIe. This situation recently changed due to the
introduction of GPU-oriented interconnect such as NVLink,
NV-SLI and NVSwitch. However, their characteristics, as
well as the performance impact on real-world multi-GPU
applications are still unknown, limiting the efforts to lever-
age them for advanced performance tuning and delivery.
In this paper, we ﬁll this gap by thoroughly characteriz-
ing a variety of modern GPU interconnects, including PCIe,
NVLink Version-1, NVLink Version-2, NV-SLI, NVSwitch,
and GPUDirect. We measured their raw startup latency,
sustainable uni/bi-directional bandwidth, network topol-
ogy, communication efﬁciency, routing, and NUMA effects,
under the two communication patterns: Peer-to-Peer (P2P)
and Collective (CL). Based on these results, we summarize
several observations, challenges to address, and potential
research topics regarding to multi-GPU execution. Through
this evaluation, software designers can gain deeper knowl-
edge about the latest GPU interconnects, paving the way for
building more mature multi-GPU programming, execution
and performance models, and reliable simulators for better
guiding application development and performance tuning.
2
MODERN GPU INTERCONNECT
We focus on six types of modern GPU interconnect: PCIe,
NVLink-V1, NVLink-V2, NV-SLI, NVSwitch, and GPUDirect-
enabled InﬁniBand. Table 1 lists the platforms we used
for evaluation. For GPU-GPU communication, P100-DGX-
1, V100-DGX-1 are for evaluating PCIe, NVLink-V1 and
NVLink-V2. SLI is for NV-SLI. DGX-2 is for NVSwitch. Sum-
mitDev and Summit are for assessing inter-node InﬁniBand
interconnect with GPUDirect-RDMA. We ﬁrst brieﬂy review
every technology.
arXiv:1903.04611v1  [cs.AR]  11 Mar 2019


---

2
TABLE 1: Evaluation Platforms. “Arch” refers to GPU architecture generation. “SP/DP GFlops” refer to GPU theoretical single/double ﬂoating-point
performance. “Rtm” refers to CUDA runtime version.
Platform
Conﬁguration
Interconnect
CPU
Compiler
GPU
GPU Arch SP/DP GFlops
GPU Memory
Rtm
P100-DGX-1 Single node, 8 GPUs
NVLink-V1
Intel Xeon E5-2698
gcc-4.8.4
Tesla-P100
Pascal
10609/5304
16GB HBM2 @ 732 GB/s
8.0
V100-DGX-1 Single node, 8 GPUs
NVLink-V2
Intel Xeon E5-2698
gcc-5.4.0
Tesla-V100
Volta
14899/7450
16GB HBM2 @ 900 GB/s
9.0
DGX-2
Single node, 16 GPUs
NVSwitch
Intel Xeon P-8168
gcc-7.3.0
Tesla-V100
Volta
14899/7450
16GB HBM2 @ 900 GB/s
9.0
SLI
Single node, 2 GPUs
NVLink-SLI Intel Xeon E5-2680
gcc-4.8.5
RTX-2080
Turing
10068/314.6
8GB GDDR6 @ 448 GB/s 10.0
SummitDev 54 nodes, 4 GPUs/node
NVLink-V1
IBM Power-8
xlc-13.1.6
Tesla-P100
Pascal
10609/5304
16GB HBM2 @ 732 GB/s
8.0
Summit
4600 nodes, 6 GPUs/node
NVLink-V2
IBM Power-9
xlc-16.1.1 Tesla-V100
Volta
14899/7450
16GB HBM2 @ 900 GB/s
9.2
PCI-e switch
GPU
NVLink
PCI-e
QPI
C0
C1
QPI
G0
G4
G1
G5
G2
G6
G3
G7
NVLink
PCI-e
(A) P100-based DGX-1 with NVLink-V1
C0
C1
QPI
G0
G4
G1
G5
G2
G6
G3
G7
NVLink
PCI-e
(B) V100-based DGX-1 with NVLink-V2
CPU
Fig. 1: PCIe and NVLink-V1/V2 topology for P100-DGX-1 and V100-DGX-1.
2.1
PCIe
Peripheral-Component-Interconnect-Express-Bus (PCIe), is a
high-speed serial computer expansion bus standard. Tradi-
tionally, a GPU-integrated system connect one or multiple
GPU devices to the CPUs via PCIe. However, compared to
the interconnect between CPU and DRAM, PCIe is much
slower. It often becomes a major performance bottleneck for
GPU-acceleration [8], [9], [10]. Such a condition exacerbates
when PCIe based GPU P2P communication is adopted [4].
The dash-lines in Figure 1-(A) illustrate how the eight
GPUs are interconnected by PCIe (and QPI) in DGX-1. As
is shown, the PCIe network in DGX-1 forms a balanced tree
structure, e.g., GPU-0 and GPU-1 are connected via a PCIe
switch. The switch is further connected to CPU Socket-0.
Similar scenarios apply to other GPUs. Finally, the dual CPU
sockets are bridged by QuickPath Interconnect or QPI [11].
PCIe in DGX-2 also form a tree-based topology, but adopts
two-level PCIe switches, as shown in Figure 4.
2.2
NVLink-V1
Known as the ﬁrst generation of NVLink, NVLink-V1 is a
wire-based communication interface for near-range devices
based on High-Speed-Signaling-Interconnect (NVHS) [4], [12].
It supports P2P communication that enables CPU-GPU or
GPU-GPU linking. It allows direct read and write on remote
CPU’s host-memory and/or peer GPU’s device-memory.
Remote atomic operations are also feasible. NVLink is bidi-
rectional; each link consists of two sublinks — one for each
direction. Each sublink further contains eight differential
NVHS lanes. An embedded clock is integrated for trans-
mission. The packet size varies from 16 bytes to 256 bytes
(one 128-bit ﬂit to sixteen 128-bit ﬂits). The communication
efﬁciency is strongly correlated to the packet size. Overall,
it is reported to be twice as efﬁcient as PCIe [12].
An NVLink can be viewed as a cable with two terminal-
plugs whereas each GPU incorporates several NVLink slots.
How these slots are connected via the NVLink cables dictate
the topology and bandwidth of the GPU network. Multiple
cables can be ganged together to enhance bandwidth when
C0
C1
G0
G1
G2
G3
X-Bus
NVLink-V1
C0
C1
X-Bus
NVLink-V2
G0
G1
G2
G3
G4
G5
(A) SummitDev
(B) Summit
Fig. 2: NVLink interconnect topology for SummitDev and Summit.
they are linking the same endpoints. A Pascal-P100 GPU
has quad NVLink slots. Therefore, for a dual-GPU system,
a direct setting would be two GPUs connected by four
NVLinks, leading to 4× bandwidth of a single link.
P100-DGX-1: The GPU network topology for DGX-1 is
known as Hypercube Mesh. As shown in Figure 1-(A), each
GPU occupies a corner of the cube and all the 12 edges are
NVLink connections. For the upper and lower planes, the
diagonals are also connected, forming two fully-connected
subsets. Such a topology design is balanced, with stronger
connectivity inside a plane. In other words, accessing within
a plane is UMA, while accessing nodes across planes leads
to NUMA (when they are not directly linked, e.g., from
GPU-0 to GPU-7). In fact, NVLink is not self-routed when
the two terminals are not directly linked. It relies on explicit
routing through a user-speciﬁed intermediate node.
SummitDev: The interconnect topology inside a machine
node in SummitDev is illustrated in Figure 2-(A). As can
be seen, the four P100 GPUs per node are partitioned into
two subsets; two GPUs per subset. A subset, together with
a Power-8 CPU socket, constituting a triple-node fully-
connected subnetwork. Every two nodes (either CPU or
GPU) in a subnetwork are connected by two NVLink-V1
links. The two subnetworks communicate via an X-Bus at
38 GB/s. Note, unlike DGX-1, there is no direct connection
between GPUs from separated subnetworks, as all the four
NVLink slots of the P100 GPUs have already been occupied.
2.3
NVLink-V2
The second generation of NVLink improves per-link band-
width and adds more link-slots per GPU: in addition to 4
link-slots in P100, each V100 GPU features 6 NVLink slots;
the bandwidth of each link is also enhanced by 25%. Besides,
a low-power operating mode is introduced for saving power
in case a link is not being heavily exploited. The extra two
link-slots have enabled certain alternation to the original
network topology.
V100-DGX-1: Shown in Figure 1-(B), the V100-DGX-1 does
not choose to further strengthen the connectivity within
a plane, but forming a fast Backbone Ring inside the
Hypercube-Mesh network. Each connection in this ring
enables 2× bandwidth compared to other links. We suspect


---

3
G0
G1
G2
G3
G4
G5
G6
G7
G8
G9
G10
G11
G12
G13
G14
G15
Baseboard
Baseboard
NVLink
NVSwitch
Fig. 3: NVSwitch interconnect topology in DGX-2.
QPI
PCI-e
G8
G9 G10 G11 G12 G13 G14 G15
G0
G1
G2
G3
G4
G5
G6
G7
PCI-e switch
GPU
CPU
Level-1 PCIe switch
Level-2 PCIe switch
Fig. 4: PCIe interconnect topology in DGX-2.
this is to improve the efﬁciency of collective communication,
as further discussed in Section 3.2.
Summit: Figure 2-(B) shows the interconnect network topol-
ogy for a machine node in Summit. The six GPUs are also
organized in two subnetworks, but each with three GPUs.
A subset, together with a Power-9 CPU socket, forming
a quad-node fully-connect subnetwork. Every two nodes
(either CPU or GPU) in a subnetwork are connected by
two NVLink-V2 links. The two subnetworks are connected
again via an X-Bus, at 64 GB/s. With one more node in the
subnetwork, all the 6 NVLink slots per V100 GPU are fully
utilized; there is no GPU direct connection between subsets.
2.4
NVLink-SLI
Scalable Link Interface (SLI) [13], or Crossﬁre [14], are tra-
ditionally used for graphic purposes only [15]. However,
the recent Turing architecture GPUs (e.g., TU102, TU104)
bring with them a new form of high-speed multi-GPU
bridge, based on the NVLink-V2 interconnect technology.
The bridge pairs up two GPUs so they can communicate
with each other and co-render games, co-run GPGPU tasks,
or co-share GPU memory spaces. In our SLI platform, the
TU104 based RTX2080 GPU offers one x8 NVLink-V2 links,
with up to 25 GB/s per direction per link, delivering an
aggregate bidirectional bandwidth of 50 GB/s. Only two-
way NVLink-SLI is supported for Turing GPUs.
2.5
NVSwitch
NVSwitch [16] is proposed mainly to address all-to-all com-
munication in many emerging applications such as deep
neural network training. NVSwitch currently only appears
in DGX-2. The topology for NVSwitch and PCIe in DGX-2
are shown in Figure 3 and Figure 4, respectively. NVSwitch
is an NVLink-V2 based switch chip for intra-node commu-
nication, featuring 18 ports of NVLink per switch. Shown
in Figure 3, there are two baseboards; each contains 6
NVSwitches and hosts 8 GPUs. This is because a V100 GPU
incorporates 6 NVLink slots, being able to connect to 6
NVSwitches simultaneously, each target per link, at 50 GB/s
bidirectional bandwidth. Each NVSwitch is a 18x18 fully
connected non-blocking crossbar: (1) 8 of the 18 ports are
used for intra-baseboard communication, which means any
of the 8 GPUs on one baseboard can talk with any other
GPUs on the same baseboard at a full bandwidth of 50
GB/s × 6 switches =300 GB/s via a single NVSwitch hop;
(2) Another 8 of the 18 ports are are used to connect to the
opposite baseboard, meaning that each of the GPUs on one
baseboard can talk to any GPUs on the other baseboard also
at a full bandwidth of 300 GB/s, but through 2 NVSwitch
hops. The baseboard-to-baseboard raw bisection bandwidth
is thus 25 GB/s × 8 links/switch × 6 switches=2.4 TB/s;
(3) The left 2 ports per NVSwitch are reserved (e.g., the red
circle in Figure 3).
To keep data integrity, Cyclical Redundancy Coding (CRC)
is imposed to detect NVLink transfer errors and replay
the transfer when necessary. Error-Correcting Codes (ECC)
is enabled for the datapaths, routing, and state structures.
In addition, ﬁnal hop-address ﬁdelity checks and buffer
over/under-ﬂow checks can be enforced in NVSwitch. To
avoid illegal out-of-range access, a fabric manager monitors
the routing tables for each particular application.
2.6
InﬁniBand with GPUDirect-RDMA
GPUDirect InﬁniBand: We will not discuss InﬁniBand [17] it-
self since it is already widely used for HPC platforms today
and has been extensively studied. Instead, we focus on its
relation with GPU. Since the Kepler architecture, NVIDIA
GPUs have introduced GPUDirect-RDMA [18] (Correspond-
ingly, AMD proposed ROCn-RDMA [19]). It enables third-
party PCIe devices, especially the IB Host-Channel-Adapter
(i.e., HCA) to directly access GPU device memory via PCIe
without any assistance from CPU or staging through the
main memory, which signiﬁcantly improves the efﬁciency
of inter-node GPU communication. To achieve IB RDMA,
GPU vendors offer an OS kernel extension to return a DMA
bus mapping for GPU device memory. When a user creates
an IB region, it signals the IB driver with the target address
of the GPU device memory. IB driver then calls a routine
to obtain the DMA mapping. Finally, a normal IB virtual
memory structure is returned to the user program as if it
targets normal CPU memory. GPUDirect-RDMA is enabled
for both SummitDev and Summit.
3
GPU INTERCONNECT MICROBENCHMARKING
We evaluate the basic characteristics of the six GPU in-
terconnects using the microbenchmarks (listed in Table 2)
from the Tartan Benchmark Suite [25] on the platforms listed
in Table 1, focusing on both Peer-to-Peer (P2P) and Col-
lective (CL) communication patterns. For intra-node P2P,
we especially concentrate on assessing the new node-level
NVLink, NVSwitch and NV-SLI technologies in terms of
topology, latency, bandwidth, efﬁciency on message size and
NUMA effects. For inter-node P2P, we discuss properties
such as latency, bandwidth and efﬁciency on message size.
We use cudaEvent to measure the latency and calculate the
bandwidth.


---

4
TABLE 2: Tartan Microbenchmarks.
Scaling
Communication
Interconnect
Metrics
Description
Scale-up
Peer-to-Peer
PCIe, NVLink-V1 and V2
Latency, bandwidth and efﬁciency Developed based on p2pBandwidthLatencyTest from CUDA SDK [20]
Scale-up
Collective
PCIe, NVLink-V1 and V2
Latency, bandwidth and efﬁciency Developed based on nccl-tests [21] linking to NCCL-V1 [22] and V2 [23]
Scale-out
Peer-to-Peer
InﬁniBand with GPUDirect Latency, bandwidth and efﬁciency Developed based on MPI-GPU-BW [24]
Scale-out
Collective
InﬁniBand with GPUDirect Latency, bandwidth and efﬁciency Developed based on nccl-tests [21] linking to NCCL-V2 [23]
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
GPU ID (0-7)
(A) P100 PCI-e Latency (us)
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
(B) V100 PCI-e Latency (us)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
GPU ID (0-7)
(C) P100 NVLink-V1 Latency (us)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
(D) V100 NVLink-V2 Latency (us)
0
5
10
15
20
25
30
35
Fig. 5: P100/V100-DGX-1 P2P communication la-
tency. The red blocks along the anti-diagonal are
local communication. The fact that other blocks
are all green in (A) and (B) indicate that no NUMA
effect appears on PCIe for latency. The orange
and blue blocks in (C) and (D) are refer to neigh-
bor nodes and remote nodes respectively which
exhibit clear disparity.
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
GPU ID (0-7)
(A) P100 PCI-e BW (GB/s)
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
(B) V100 PCI-e BW (GB/s)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
GPU ID (0-7)
(C) P100 NVLink-V1 BW (GB/s)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
(D) V100 NVLink-V2 BW (GB/s)
1
2
4
8
16
32
64
128
256
512
1024
Fig. 6: DGX-1 P2P unidirectional bandwidth. Al-
though not very obvious, we can see 2x2 blocks
in (A) and (B) along anti-diagonal, which indicates
the anti-locality NUMA effect for unidirection band-
width on PCIe. (C) and (D) conﬁrm NUMA among
neighbors and remote nodes. The other two types
of NUMA for NVLink-V2 are not quite clear in (D).
They are more obvious for bidirection bandwidth.
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
GPU ID (0-7)
(A) P100 PCI-e BW (GB/s)
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
(B) V100 PCI-e BW (GB/s)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
GPU ID (0-7)
(C) P100 NVLink-V1 BW (GB/s)
0
1
2
3
4
5
6
7
GPU ID (0-7)
0
1
2
3
4
5
6
7
(D) V100 NVLink-V2 BW (GB/s)
1
2
4
8
16
32
64
128
256
512
1024
Fig. 7: DGX-1 P2P bidirectional bandwidth. The 2x2
blue/red blocks along anti-diagonal in (A) and (B)
clearly illustrate the anti-locality NUMA effect on
PCIe. In (D), the yellow blocks compared with the
green blocks show the NUMA among neighboring
nodes. Meanwhile, the light green blocks along the
anti-diagonal (not quite obvious though) imply the
existence of NUMA among remote nodes.
3.1
Intra-Node P2P Communication
3.1.1
Start-up Latency
Figure 5 shows the start-up communication latency (i.e., raw
latency for transmitting the shortest message) among arbi-
trary pair of GPUs via PCIe and NVLink for the P100 and
V100 DGX-1 platforms. As already mentioned, NVLink is
not P2P self-routed; for GPUs that are not directly connected
by NVLink (e.g., G0 and G5 in Figure 1), there are two
routing paths that only require a single transit (e.g., from G0
to G5, either G1 or G4 can be the router). In such scenarios,
Figure 5 shows the path exhibiting shorter latency.
PCIe Latency: Figure 5-(A) and (B) demonstrate that the
communication latency for accessing different pairs of GPUs
via PCIe are similar, implying that no NUMA effects appear
in latency through PCIe. In other words, the three types
of latency by going through one PCIe switch (e.g., G0 and
G1), across local CPU socket (e.g., G0 and G2), and across
the QPI bridge (e.g., G0 and G6) in Figure 1 are roughly
the same. Meanwhile, comparing Figure 5-(A) and (B), the
PCIe latency is increased slightly (e.g., from green to light
blue) from P100-DGX-1 to V100-DGX-1. As the bandwidth
keeps unchanged, this may suggest a deeper communica-
tion pipeline design in V100-DGX-1 with Little’s Law [26].
NVLink V1&V2 Latency: Compared to PCIe in Figure 5-(A)
and (B), NVLink in Figure 5-(C) and (D) shows signiﬁcant
NUMA effects. For nodes that are directly connected, the
latency is around 9µs; for nodes that require manual rout-
ing, the latency is increased by about 2x for P100-DGX-1
and 3x for V100-DGX-1. Again, we observe that NVLink-V2
exhibits higher latency than NVLink-V1, potentially due to
a deeper pipeline or lower operating frequency.
NV-SLI Latency: Figure 8 shows the latency for PCIe and
NV-SLI in the SLI platform. For the dual-GPU system, there
are three latency levels: local access which is about 5µs; NV-
SLI access to the opposite GPU which is about 8µs; and
PCIe access to the opposite GPU, which is about 13µs. Note,
the two GPUs are directly linked by NV-SLI without any
intervention from other units such as CPU, DMA, etc.
NVSwitch Latency: Figure 11 shows the latency for PCIe
and NVSwitch in the DGX-2 platform. There are in total 16
GPUs in the system, so the grid is much ﬁner. The pattern
is very regular: all the remote access are homogeneous,
either for PCIe or NVSwitch, conﬁrming that NVSwitch is
all-to-all fully-connected. Although accessing GPUs on the
other baseboard incurs two switch hops, we can see that the
difference of latency is very small, almost negligible.
3.1.2
Sustainable Bandwidth
Figure 6 and 7 illustrate the uni- and bidirection sustain-
able bandwidth for PCIe and NVLink of the two DGX-
1 platforms, respectively. Figure 9 and 10 show the uni-
and bidirection bandwidth for PCIe and NV-SLI in the SLI
platform. Finally, Figure 12 and Figure 13 show the uni- and
bidirection bandwidth for PCIe and NVSwitch in DGX-2.
PCIe Unidirection Bandwidth: From Figure 6-(A) and (B),
we can observe slight NUMA effects on PCIe accesses: two
GPUs sharing the same PCIe switch (e.g., G2 and G3 in
Figure 1) exhibit lower bandwidth in the measurement. For
other GPUs, no matter whether sharing the same socket, the
bandwidth appears to be the same. Similar effects have also
been observed in Figure 12-(A), in which four GPUs sharing
the same Level-2 PCIe switch (e.g., G0 to G3 in Figure 4)
deliver lower bandwidth.
PCIe Bidirection Bandwidth: The NUMA effects on bidirec-
tional bandwidth for GPUs sharing the same PCIe switch


---

5
0
1
GPU ID (0-1)
0
1
GPU ID (0-1)
(A) SLI PCI-e Latency (us)
0
1
GPU ID (0-1)
0
1
(B) SLI NVLink Latency (us)
0
5
10
15
20
25
30
35
Fig. 8: SLI-System P2P communication latency.
0
1
GPU ID (0-1)
0
1
GPU ID (0-1)
(A) SLI PCI-e BW (GB/s)
0
1
GPU ID (0-1)
0
1
(B) SLI NVLink BW (GB/s)
1
4
16
64
256
1024
Fig. 9: SLI-System P2P unidirectional bandwidth.
0
1
GPU ID (0-1)
0
1
GPU ID (0-1)
(A) SLI PCI-e BW (GB/s)
0
1
GPU ID (0-1)
0
1
(B) SLI NVLink BW (GB/s)
1
4
16
64
256
1024
Fig. 10: SLI-System P2P bidirectional bandwidth.
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
13
14
15
GPU ID (0-15)
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
13
14
15
GPU ID (0-15)
(A) DGX-2 PCI-e Latency (us)
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
13
14
15
GPU ID (0-15)
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
13
14
15
(B) DGX-2 NVLink-V2 Latency (us)
0
5
10
15
20
25
30
35
Fig. 11: DGX-2 P2P communication latency.
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
13
14
15
GPU ID (0-15)
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
13
14
15
GPU ID (0-15)
(A) DGX-2 PCI-e BW (GB/s)
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
13
14
15
GPU ID (0-15)
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
13
14
15
(B) DGX-2 NVLink-V2 BW (GB/s)
1
4
16
64
256
1024
Fig. 12: DGX-2 P2P unidirectional bandwidth.
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
13
14
15
GPU ID (0-15)
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
13
14
15
GPU ID (0-15)
(A) DGX-2 PCI-e BW (GB/s)
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
13
14
15
GPU ID (0-15)
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
13
14
15
(B) DGX-2 NVLink-V2 BW (GB/s)
1
4
16
64
256
1024
Fig. 13: DGX-2 P2P bidirectional bandwidth.
(e.g., Figure 7-(A) and (B), Figure 13-(A)) are much more
prominent than those on unidirection bandwidth. The PCIe
NUMA effect here is an interesting novel observation: it de-
scribes a scenario that nearby access presenting lower performance
than remote access. We label such a NUMA effect as “anti-
locality”. To the best of our knowledge, few existing work
have discussed this phenomenon before, without mention-
ing leveraging it for performance tuning practice. The anti-
locality effect is possibly due to the unbalanced physical
signal paths on the PCIe-switch chipsets [27]. Note, this PCIe
anti-locality effect is only observed for bandwidth.
NVLink Unidirection Bandwidth: The NVLink scenario is
more complicated. For NVLink-V1 in Figure 6-(C), there are
three connection types: local access, neighboring nodes directly
connected by NVLink, and remote nodes requiring addi-
tional routing, corresponding to the topology illustrated
in Figure 1-(A). For NVLink-V2 in Figure 6-(D), there are
four connection types: local access, close neighboring nodes
connected by dual links (i.e., the “backbone ring”), gen-
eral neighboring nodes connected by one link, and remote
nodes, corresponding to the topology in Figure 1-(B). As
such, there are three types of NUMA effects for NVLink:
• NUMA among neighbors and remote nodes for NVLink-
V1 and V2 on both latency and bandwidth.
• NUMA among neighbor nodes for NVLink-V2. This is
due to different number of links (either 1 or 2) to connect
neighboring nodes in V100-DGX-1. Typically, the latency
remains similar but these two types of links introduce
bandwidth difference.
• NUMA among remote nodes for NVLink-V2. This is
caused by the choice of routing. Figure 6-(C) and (D) show
the bandwidth disparity for choosing different paths.
NVLink Bidirection Bandwidth: The three types of NVLink
NUMA effects for bidirectional bandwidth are much more
obvious than that for unidirectional bandwidth, as dis-
cussed in the caption of Figure 7.
NV-SLI Unidirection Bandwidth: Since NV-SLI only incor-
porates two GPUs, where the communication is symmetric,
showing no NUMA effect in Figure 9-(B).
UniBand-P100
BiBand-P100
UniBand-V100
BiBand-V100
0
10
20
30
40
Bandwidth (GB/s)
G0>G1>G5
G0>G4>G5
G0>G2>G6
G0>G4>G6
G0>G3>G7
G0>G4>G7
G0>G1>G5
G0>G4>G5
G0>G2>G6
G0>G4>G6
G0>G3>G7
G0>G4>G7
G0>G1>G5
G0>G4>G5
G0>G2>G6
G0>G4>G6
G0>G3>G7
G0>G4>G7
G0>G1>G5
G0>G4>G5
G0>G2>G6
G0>G4>G6
G0>G3>G7
G0>G4>G7
Fig. 14: NUMA effect on routing choices for remote GPU access via NVLink.
NV-SLI Bidirection Bandwidth: The bidirectional condition
is similar to unidirection condition, except that the band-
width doubles, as shown in Figure 10-(B).
NVSwitch Unidirection Bandwidth: Shown in Figure 12-
(B), the bandwidth for all remote access through NVSwitch
are consistent or UMA, implying that one more NVSwitch
hop does not degrade bandwidth.
NVSwitch Bidirection Bandwidth: Bidirection bandwidth
condition is similar, except that the bandwidth doubles, as
shown in Figure 13-(B).
3.1.3
Routing
For all the GPU interconnects we discuss here, only the one
for remote access in the DGX-1s via NVLink may require
explicit routing. Here, we further explore the NUMA effects
on alternative routing choices. For demonstration purposes,
we take G0 in Figure 1 as the source node for P2P commu-
nication. There are three remote nodes for G0: G5, G6 and
G7. From G0 to G5, either G1 or G4 can be speciﬁed for
routing. From G0 to G6, either G2 or G4 can be selected;
and from G0 to G7, either G3 or G4 can be selected. We use
microbenchmarks from Tartan to measure the latency, unidi-
rection bandwidth and bidirection bandwidth of each rout-
ing path from G0 to G5, G6, G7 respectively on both DGX-1
platforms. Figure 14 shows the results for unidirection and
bidirection bandwidth. As can be seen, for NVLink-V1 in
P100-DGX-1, there are no NUMA effects; all the bars appear
in the same height. This is because the NVLinks in P100-
DGX-1 are isomorphic — any connection incorporates just
one link. However, for NVLink-V2 in V100-DGX-1, different


---

6
scenarios emerge based on how many dual-bandwidth links
a route goes through, e.g., the lowest bandwidth occurs
from G0→G2→G6 while the highest is triggered by routing
G0→G4→G7. Nevertheless, the latency remains similar for
all scenarios (not shown in the ﬁgures).
3.1.4
Efﬁciency on Message Size
Figure 15 illustrates the P2P latency, unidirection and bidi-
rection bandwidth with respect to message size via PCIe,
NVLink-V1 and V2 for P100-DGX-1 and V100-DGX-1. Fig-
ure 16 illustrates the plot for NV-SLI and PCIe in the SLI-
system. Figure 17 illustrates the plot for NVSwitch and PCIe
in the DGX-2 system.
Latency: The latency remains unchanged for data commu-
nication less than or equal to 16KB for P100-DGX-1 (except
local access). For V100-DGX-1, this value increases to 64KB,
suggesting higher link bandwidth to saturate and deeper
communication pipeline on NVLink-V2. The conditions are
similar for SLI and NVSwitch in the SLI and DGX-2 plat-
forms. The interesting observation is that in Figure 17, there
is slight divergence for PCIe-local (including PCIe-neighbor
and PCIe-one-switch) and PCI-remote access latency with
large messages (i.e., ≥4MB).
Bandwidth: For unidirection bandwidth, it can be seen that
the interconnect starts to saturate at about 222 =4MB for
both NVLink-V1 and V2, implying that to reach the optimal
sustainable bandwidth of the interconnect, one needs at
least 4MB data to transmit at a time. This is also true for
bidirection bandwidth in Figure 15-(E) and (F). In addition,
observing the fact that the latency starts to increase at 16KB
and 64KB, but the bandwidth begins to saturate at 4MB,
implies that from 64KB, we suffer from extra delay, but still
gain overall bandwidth improvement until the bandwidth
saturation point. For DGX-2 NVSwitch uni- and bidirection
bandwidth in Figure 17-(B) and (C), we see that NVSwitch-
one-hop and NVSwitch-two-hop curves are fully aligned,
conﬁrming that accessing remote baseboard imposes no ex-
tra overhead. For PCIe bidirection bandwidth in Figure 17-
(C), the observation is that when message size is larger
than 64KB, the PCIe anti-locality effect appears: PCIe-remote
access delivers higher bidirection bandwidth than PCIe-
neighbors (L1 & L2). In fact, the anti-locality effect also
exists between L1-PCIe-neighbors and L2-PCIe-neighbor in
DGX-2, as can be seen, the bandwidth of L2-PCIe-neighbor
is slightly better than L1-PCIe-neighbor, showing a second
level anti-locality. Finally, all GPU local access bandwidth
in these ﬁgures staggers at about 4MB, possibly due to
exceeding page boundary.
3.2
Intra-Node Collective Communication
Different from P2P communication only including a single
sender and receiver, collective communication (CL) involves
multiple senders and receivers so it is more complicated.
CL generally follows certain patterns, including broadcast,
scatter, gather, all-gather, reduce, all-reduce, all-to-all, etc. It is
extensively used in many key applications such as deep
learning, molecular dynamics, graph analytics, etc.
Efﬁciently implementing CL communication is challeng-
ing because (a) it needs to understand the underlying
hardware network topology in order to enable orchestrated
mapping; (b) it needs to handle the issue of synchronization,
overlapping and deadlock; and (c) performance metrics can
differ subject to application features (e.g., latency-oriented
for small transfers but bandwidth-oriented for large trans-
fers). To relieve these burdens from users, NVIDIA provides
Collective Communication Library (NCCL) [22], [23], using
similar primitives as MPI collectives, while AMD offers
RCCL [28]. NCCL currently supports ﬁve CL patterns:
broadcast, all-gather, reduce, all-reduce, and reduce-scatter.
To offer the maximum bandwidth, NCCL constructs
ring network among the communication participants so
that broadcasting and reduction can be efﬁciently realized
by partitioning data into small chunks, and transmitting
them along the ring in a pipeline fashion. It is claimed
that the ring algorithm can provide near optimal bandwidth
for most of the standard CL operations and can be easily
applied to various network topology [4]. Figure 18, as an
example, describes how a ring-network is constructed for
PCIe, NVLink-V1 and NVLink-V2 in DGX-1s, respectively.
There are two versions of the NCCL library: NCCL-V1
[22] is opensource but only supports intra-node PCIe/QPI
interconnect network. NCCL-V2 [22] is closed-source but
supports NVLink, PCIe, NVSwitch, NV-SLI, IB and IP net-
works, and can automatically integrate them to maximize
overall bandwidth. Although the combination improves
overall performance, it also introduces difﬁculty in inde-
pendently measuring the CL communication efﬁciency for
a particular interconnect. Consequently, NCCL-V1 is em-
ployed for PCIe CL evaluation while NCCL-V2 is adopted
for NVLink/NVSwitch/NV-SLI CL evaluation.
3.2.1
DGX-1 CL Communication
CL Latency: Figure 19 illustrates CL communication startup
latency with respect to the number of GPUs involved for
NCCL-V1 and V2 on the two DGX-1 platforms respectively,
corresponding to PCIe/QPI and NVLink. We have the fol-
lowing observations: (1) latency increases with participating
GPUs almost in a linear fashion; (2) comparing (A) and
(C), (B) and (D), the behaviors of NCCL-V1 and V2 on the
two DGX platforms are similar; (3) comparing (A) (B) with
(C) (D), the latency of NVLink increases faster than PCIe
(except all-reduce), while NVLink-V2 increases faster than
NVLink-V1; (4) for PCIe, all reduce shows the largest latency.
The disalignment of the curves in (B) and (D) for odd
number of GPUs (e.g., 3, 5) is likely due to NCCL algorithm
design rather than NVLink-V2 P2P NUMA effects, as will
be discussed later.
CL Bandwidth: Figure 20 shows CL’s sustainable commu-
nication bandwidth with increased number of GPUs under
1GB payload. As can be seen, (1) for PCIe, CL bandwidth
decreases with more GPUs, which is due to bus contention
in a tree-network, see Figure 18-(A); (2) for NVLink, how-
ever, CL bandwidth essentially increases with more GPUs
due to more connected links in a hypercube mesh network,
see Figure 18-(B) and (C); (3) PCIe and NVLink behavior on
P100-DGX-1 and V100-DGX-1 are in similar trend. However,
NVLink-V2 exhibits signiﬁcantly better bandwidth with 4
GPUs (∼1.6x) and 8 GPUs (∼2x) compared to NVLink-
V1, showing the strength of dual-links and backbone ring


---

7
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
2
12
2
13
2
14
2
15
Latency (us)
(A) P100-DGX-1 P2P Latency
Local Access
PCIe-neighbor
PCIe-remote
NVLink-one-link
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
Unidirection BW (GB/s)
(C) P100-DGX-1 Unidirection BW
Local Access
PCIe-neighbor
PCIe-remote
NVLink-one-link
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
Bidirection BW (GB/s)
(E) P100-DGX-1 Bidirection BW
Local Access
PCIe-neighbor
PCIe-remote
NVLink-one-link
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
Data Size (byte)
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
2
12
2
13
2
14
2
15
Latency (us)
(B) V100-DGX-1 P2P Latency
Local Access
PCIe-neighbor
PCIe-remote
NVLink-two-link
NVLink-one-link
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
Data Size (byte)
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
Unidirection BW (GB/s)
(D) V100-DGX-1 Unidirection BW
Local Access
PCIe-neighbor
PCIe-remote
NVLink-two-link
NVLink-one-link
2
0
2
2
2
4
2
6
2
8 2
10 2
12 2
14 2
16 2
18 2
20 2
22 2
24 2
26 2
28
Data Size (byte)
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
2
9
2
10
2
11
Bidirection BW (GB/s)
(F) V100-DGX-1 Bidirection BW
Local Access
PCIe-neighbor
PCIe-remote
NVLink-two-link
NVLink-one-link
Fig. 15: P2P communication latency, unidirection and bidirection bandwidth with increased message size via PCIe and NVLink for DGX-1.
22
26
210
214
218
222
226
Data Size (byte)
22
25
28
211
214
Latency (us)
(A) NV-SLI P2P Latency
Local Access
PCIe
NV-SLI
22
26
210
214
218
222
226
Data Size (byte)
2−4
2−2
20
22
24
26
28
210
Unidirection BW (GB/s)
(B) NV-SLI Unidirection BW
Local Access
PCIe
NV-SLI
22
26
210
214
218
222
226
Data Size (byte)
2−4
2−2
20
22
24
26
28
210
Bidirection BW (GB/s)
(C) NV-SLI Bidirection BW
Local Access
PCIe
NV-SLI
Fig. 16: P2P communication latency, unidirection and bidirection bandwidth with increased message size via PCIe and NV-SLI for the SLI-system.
22
26
210
214
218
222
226
Data Size (byte)
22
24
26
28
210
212
214
Latency (us)
(A) DGX-2 P2P Latency
Local Access
PCIe-L1-neighbor
PCIe-L2-neighbor
PCIe-remote
NVSwitch-one-hop
NVSwitch-two-hop
22
26
210
214
218
222
226
Data Size (byte)
2−4
2−2
20
22
24
26
28
210
Unidirection BW (GB/s)
(B) DGX-2 Unidirection BW
Local Access
PCIe-L1-neighbor
PCIe-L2-neighbor
PCIe-remote
NVSwitch-one-hop
NVSwitch-two-hop
22
26
210
214
218
222
226
Data Size (byte)
2−4
2−2
20
22
24
26
28
210
Bidirection BW (GB/s)
(C) DGX-2 Bidirection BW
Local Access
PCIe-L1-neighbor
PCIe-L2-neighbor
PCIe-remote
NVSwitch-one-hop
NVSwitch-two-hop
Fig. 17: P2P communication latency, unidirection and bidirection bandwidth with increased message size via PCIe and NVSwitch for DGX-2.
PCI-e switch
GPU
CPU
(A) NCCL Ring in PCI-e Network
C0
C1
QPI
G0
G4
G1
G5
G2
G6
G3
G7
PCI-e
(B) NCCL Two Rings 
in NVLink-V1 Network
G0
G4
G1
G5
G2
G6
G3
G7
NVLink
G0
G4
G1
G5
G2
G3
G7
NVLink
G6
(C) NCCL Backbone Ring 
    in NVLink-V2 Network
Fig. 18: NCCL Rings for PCIe, NVLink-V1 and NVLink-V2 interconnect. (A) for
PCIe, the ring is to traverse the binary-tree network; (B) for NVLink-V1, there
are two independent rings, marked in red-solid line and blue-dash line; and
(C) for NVLink-V2, the lines with 2 links form a fast ring (i.e., the backbone
network) while the lines with 1 link form a slow ring.
(Figure 18); (4) the NUMA effects appear quite signiﬁcant
with 5 GPUs, implying that there may be a congestion when
forming a NCCL ring among 5 GPUs. One should avoid
adopting 5 GPUs in their application setup.
CL Efﬁciency on Message Size: Figure 21 shows CL band-
width with respect to message size increasing from 8B to
1GB for 8 GPUs. As can be seen, for PCIe, CL-bandwidth
saturates at about 224 = 16MB; whereas for NVLink, band-
width saturates around 228 = 256MB. Again, the ﬁve CL
patterns exhibit similar trend in terms of bandwidth when
scaling the message size.
3.2.2
NV-SLI CL Communication
CL Latency and Bandwidth: Since the SLI-system contains
only two GPUs, we use histogram ﬁgures to show the la-
tency and bandwidth for the 5 CL communication patterns.
As can be seen in Figure 22 and 23, the latency for NV-SLI
is similar, around 18µs; but for PCIe, reduce and all reduce
show signiﬁcantly lower latency than the other three, even
less than on NV-SLI. The bandwidth are similar for both
PCIe and NV-SLI, except that reduce scatter showing poorer
bandwidth than the others on both PCIe and NV-SLI.


---

8
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
Latency (us)
(A) P100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
(B) P100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
(C) V100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
(D) V100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Fig. 19: Intra-node CL communication latency with variable participant GPUs for NCCL-V1 (PCIe/QPI) and NCCL-V2 (NVLink-V1/2).
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
120
140
Bandwidth (GB/s)
(A) P100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
120
140
(B) P100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
120
140
(C) V100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate GPUs
0
20
40
60
80
100
120
140
(D) V100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Fig. 20: Intra-node CL communication bandwidth with variable participant GPUs for NCCL-V1 (PCIe/QPI) and NCCL-V2 (NVLink-V1/2).
2
5 2
7 2
92
112
132
152
172
192
212
232
252
272
29
Data Size (byte)
2
-6
2
-5
2
-4
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8
Bandwidth (GB/s)
(A) P100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
5 2
7 2
92
112
132
152
172
192
212
232
252
272
29
Data Size (byte)
2
-6
2
-5
2
-4
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8(B) P100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
5 2
7 2
92
112
132
152
172
192
212
232
252
272
29
Data Size (byte)
2
-6
2
-5
2
-4
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8 (C) V100-DGX-1 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
5 2
7 2
92
112
132
152
172
192
212
232
252
272
29
Data Size (byte)
2
-6
2
-5
2
-4
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
2
5
2
6
2
7
2
8(D) V100-DGX-1 NCCL-V2 CL (NVLink)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Fig. 21: Intra-node CL communication bandwidth for 8 GPUs with increased message size for NCCL-V1 (PCIe/QPI) and NCCL-V2 (NVLink-V1/2).
CL Efﬁciency on Message Size: Figure 24 shows CL band-
width with respect to message size for the two GPUs.
Both PCIe and NV-SLI bandwidth start to saturate at about
220 = 1MB. The ﬁgure conﬁrms that reduce scatter band-
width is lower than the others for large message size,
but at the same time indicating that all gather delivers a
relative low bandwidth with the same message size before
saturated.
3.2.3
NVSwitch CL Communication
CL Latency and Bandwidth: Figure 25 illustrates CL com-
munication latency with respect to the number of GPUs
on DGX-2, corresponding to PCIe and NVSwitch. As can
be seen, due to UMA for NVSwitch, the ﬁve curves in
Figure 25-(B) are rather aligned. Figure 25-(B) conﬁrms the
long latency for all reduce on tree-based PCIe interconnect,
in consistent with Figure 19-(A) and (C). Also note that ex-
cept all reduce, the other CL primitives show lower startup
latency on PCIe than on NVSwitch when the participanting
GPUs are more than three, implying that the advantage of
NVSwitch (as well as NVLink) is on bandwidth rather than
latency. This observation is conﬁrmed in Figure 26. As can
be seen, NVSwitch shows tremendously higher bandwidth
than PCIe, particularly for reduce, all reduce and broadcast.
reduce scatter and all gather show staggering behavior on
bandwidth for NVSwitch; the values are much better with
2, 4, 8 and 16 GPUs than other numbers of GPUs, in con-
sistent with NVLink scenarios in Figure 20-(B)/(D). Since
NVSwitch is UMA, it implies that this staggering issue is not
due to interconnect topology but NCCL’s implementation.
CL Efﬁciency on Message Size: Figure 27 shows CL band-
width with respect to message size for 16 GPUs on the DGX-
2 system. As can be seen, the curves for PCIe in Figure 27-
(A) are in consistent with PCIe in Figure 21-(A) and (C): the
ﬁve curves are mostly aligned for small (e.g., ≤32KB) and
large (e.g., ≥32MB) messages, but not the case in-between
for reduce and broadcast. This divergence also appears for
NVLink-V1 & V2 in Figure 21-(B) and (D), but disappears
for NVSwitch in Figure 27-(B). As NVSwitch is UMA, we
suppose this disalignment is brought by the NUMA effect
in the PCIe and NVLink networks.
3.3
Inter-Node P2P Communication
We measure the latency and bandwidth of inter-node P2P
communication on SummitDev Supercomputer [29] and Sum-
mit Supercomputer [6] from Oak Ridge National Laboratory.
SummitDev is a supercomputer system with 54 nodes. Each
node contains two IBM Power-8 CPUs and four NVIDIA
P100 GPUs. The GPU interconnect is NVLink-V1. Summit
features 4,608 nodes, each with two IBM Power-9 CPUs and
six NVIDIA V100 GPUs. The GPU interconnect is NVLink-
V2. Both SummitDev and Summit support GPUDirect.
For inter-node P2P, we conduct our measurement un-
der ﬁve conﬁgurations: (i) GPUDirect-RDMA is to directly
access GPU memory among nodes with GPUDirect enabled
(“GPUDirect” here refers to a system option. On SummitDev,


---

9
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Participate GPUs=2
0
10
20
30
40
Latency (us)
(A) SLI-System NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Participate GPUs=2
0
10
20
30
40
(B) SLI-System NCCL-V2 CL (NV-SLI)
Fig. 22: SLI-system PCIe and NV-SLI CL communication latency.
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Participate GPUs=2
0
5
10
15
20
25
30
Bandwidth (GB/s)
(A) SLI-System NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Participate GPUs=2
0
5
10
15
20
25
30
(B) SLI-System NCCL-V2 CL (NV-SLI)
Fig. 23: SLI-system PCIe and NV-SLI CL communication bandwidth.
26
211
216
221
226
231
Data Size (byte)
2−4
2−1
22
25
28
Bandwidth (GB/s)
(A) SLI-System NCCL-V1 CL (PCI-e)
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
27
212
217
222
227
Data Size (byte)
2−4
2−1
22
25
28
(B) SLI-System NCCL-V2 CL (NV-SLI)
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
Fig. 24: SLI-system PCIe and NV-SLI CL bandwidth efﬁciency.
2
4
6
8
10
12
14
16
Participate GPUs
0
20
40
60
80
100
Latency (us)
(A) DGX-2 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
4
6
8
10
12
14
16
Participate GPUs
0
20
40
60
80
100
(B) DGX-2 NCCL-V2 CL (NVSwitch)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Fig. 25: DGX-2 PCIe and NVSwitch CL communication latency.
2
4
6
8
10
12
14
16
Participate GPUs
0
20
40
60
80
100
120
140
Bandwidth (GB/s)
(A) DGX-2 NCCL-V1 CL (PCI-e)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
4
6
8
10
12
14
16
Participate GPUs
0
20
40
60
80
100
120
140
(B) DGX-2 NCCL-V2 CL (NVSwitch)
Fig. 26: DGX-2 PCIe and NVSwitch CL communication bandwidth.
29
214
219
224
229
Data Size (byte)
2−4
2−1
22
25
28
Bandwidth (GB/s)
(A) DGX-2 NCCL-V1 CL (PCI-e)
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
29
214
219
224
229
Data Size (byte)
2−4
2−1
22
25
28
(B) DGX-2 NCCL-V2 CL (NVSwitch)
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
Fig. 27: DGX-2 PCIe and NVSwitch CL bandwidth efﬁciency with 16 GPUs.
it is OMPI MCA pml pami enable cuda=1. On Summit, it is
–smpiargs=“gpu”.) (ii) PinnedMem-GPUDirect is to ﬁrst copy
data from GPU memory to the pinned host memory, then
transfer the data to another node’s pinned host memory and
ﬁnally copy to the targeted GPU memory, with GPUDirect
enabled; (iii) PinnedMem is similar to (ii) but with GPUDi-
rect disabled; (iv) UnpinnedMem-GPUDirect is to copy via
host unpinned memory with GPUDirect enabled; and (v)
UnpinnedMem is similar to (iv) but with GPUDirect disabled.
The measured latency and bandwidth with respect to
message size (from 4B to 1GB) under the ﬁve testing sce-
narios are illustrated in Figure 28 and 29, for SummitDev
and Summit, respectively. For better illustration, the latency
curves use log-scale Y-axis while bandwidth curves use
normal-scale Y.
SummitDev: From Figure 28, we draw the following obser-
vations: (i) Until 212 = 4KB, there is little difference among
the ﬁve curves in terms of both latency and bandwidth;
(ii) GPUDirect-RDMA shows the worst performance in the
range from 4KB to 64KB for latency and from 4KB to
256KB for bandwidth, especially at 32KB. This is possibly
due to limitations in some chipsets for P2P access through
the CPU/IOH [30]; (iii) From 4MB on, GPUDirect-RDMA
shows its advantage on bandwidth and obtains its optimal
bandwidth — 12GB/s at 64MB. However, this is still lower
than the PinnedMem-GPUDirect scheme, which demon-
strates more than 14GB/s sustainable bandwidth with large
message size; (v) it is also interesting to observe that the
bandwidth of GPUDirect-RDMA actually degrades dramat-
ically after 64MB, implying that breaking large messages
into multiples of 64MB could be a better way to transfer in
practice on SummitDev.
Summit: Regarding Figure 29, we have the following ob-
servations: (i) Unlike SummitDev, GPUDirect-RDMA shows
the best performance among the ﬁve conﬁgurations on Sum-
mit: it always delivers the lowest latency, especially for small
message size (≤1MB); it always exhibits the highest band-
width, especially for large message size (≥16KB). Mean-
while, the strange performance drop (i.e., latency increase
and bandwidth degradation) in Figure 28 for SummitDev,
disappear in Figure 29 for Summit. These two points sug-
gest that the technology of GPUDirect has been improved
signiﬁcantly from SummitDev to Summit, and becomes the
best choice for GPU inter-node communication.
3.4
Inter-Node Collective Communication
Regarding inter-node collective communication, we mea-
sure the latency and bandwidth with respect to the number
of participant nodes on both SummitDev and Summit. We
tune the number of nodes from 2 to 8, with 1 GPU per node
being utilized. Similarly, the startup latency is measured
with 4B data transfer while the sustainable bandwidth is
measured with sufﬁciently large data transfer (1GB). The
latency results are shown in Figure 30-(A) and Figure 31-(A)
for SummitDev and Summit, respectively. The bandwidth
results are shown in Figure 30-(B) and Figure 31-(B). The


---

10
26
212
218
224
230
Data Size
2−4
2−1
22
25
28
Latency (ms)
(A) SummitDev Inter-node P2P Latency
GPUDirect-RDMA
PinnedMem-GPUDirect
PinnedMem
UnpinnedMem-GPUDirect
UnpinnedMem
26
212
218
224
230
Data Size
0
5
10
15
20
Bandwidth (GB/s)
(B) SummitDev Inter-node P2P Bandwidth
GPUDirect-RDMA
PinnedMem-GPUDirect
PinnedMem
UnpinnedMem-GPUDirect
UnpinnedMem
Fig. 28: SummitDev inter-node P2P latency and bandwidth efﬁciency.
26
212
218
224
230
Data Size
2−4
2−1
22
25
28
Latency (ms)
(A) Summit Inter-node P2P Latency
GPUDirect-RDMA
PinnedMem-GPUDirect
PinnedMem
UnpinnedMem-GPUDirect
UnpinnedMem
26
212
218
224
230
Data Size
0
5
10
15
20
Bandwidth (GB/s)
(B) Summit Inter-node P2P Bandwidth
GPUDirect-RDMA
PinnedMem-GPUDirect
PinnedMem
UnpinnedMem-GPUDirect
UnpinnedMem
Fig. 29: Summit inter-node P2P latency and bandwidth efﬁciency.
1
2
3
4
5
6
7
8
9
Participate Nodes
0
20
40
60
80
100
120
140
Latency (us)
(A) SummitDev CL Latency
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
1
2
3
4
5
6
7
8
9
Participate Nodes
0
5
10
15
20
25
30
Bandwidth (GB/s)
(B) SummitDev CL Bandwidth
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
2
5 2
7 2
9 2
11 2
13 2
15 2
17 2
19 2
21 2
23 2
25 2
27 2
29
Data Size (byte)
2
-7
2
-6
2
-5
2
-4
2
-3
2
-2
2
-1
2
0
2
1
2
2
2
3
2
4
Bandwidth (GB/s)
(C) SummitDev CL (8 Nodes)
Reduce
All_Reduce
Broadcast
Reduce_Scatter
All_Gather
Fig. 30: Inter-node CL communication latency and bandwidth with variable participant nodes, as well as bandwidth for 8 nodes with increased message size.
2
3
4
5
6
7
8
Participate Nodes
0
25
50
75
100
125
Latency (us)
(A) Summit CL Latency
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
2
3
4
5
6
7
8
Participate Nodes
0
10
20
30
40
Bandwidth (GB/s)
(B) Summit CL Bandwidth
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
27
211
215
219
223
227
231
Data Size (byte)
2−4
2−1
22
25
28
Bandwidth (GB/s)
(C) Summit CL (8 Nodes)
Reduce
All Reduce
Broadcast
Reduce Scatter
All Gather
Fig. 31: Inter-node CL communication latency and bandwidth with variable participant nodes, as well as bandwidth for 8 nodes with increased message size.
bandwidth change with increasing message size is shown
in Figure 30-(C) and Figure 31-(C). We tried to illustrate the
difference between enabling and disabling GPUDirect, but
found that the results are in fact very similar. We suspect
that GPUDirect-RDMA is internally enabled in NCCL-V2.
CL Latency: As shown in Figure 30-(A), for SummitDev’s
IB fat-tree network, the latency-change for performing the
ﬁve CL operations remains ﬂat when scaling the number of
nodes, except all-reduce. Similar observation is also drawn in
Figure 31-(A) for Summit, the divergence is much less for all-
reduce. This may imply that it is a joint-effect of algorithm
implementation in NCCL and the interconnect technology
of GPUDirect.
CL Bandwidth: In terms of bandwidth in Figure 30-(B), sim-
ilar to NVLink (Figure 20-(B) and (D)), strong NUMA effects
emerge under 3 and 5 nodes for reduce scatter and all gather
on SummitDev. And for Summit, this happens under 3, 5,
6, 7 nodes in Figure 31. Nevertheless, the bandwidth overall
remains unchanged. This is different from the bandwidth
scenarios exhibited by both PCIe (decreasing) and NVLink
(increasing) in the inter-node P2P communication study.
CL Efﬁciency on Message Size: Finally, the bandwidth
for the ﬁve CL operations converge and saturate around
32/64MB message size, demonstrating nearly 16/32 GB/s
sustainable peak bandwidth on SummitDev in Figure 31-
(C), and Summit in Figure 31-(C), respectively. Overall,
Summit delivers much better GPU inter-node communica-
tion performance than SummitDev.
4
GPU INTERCONNECT BENCHMARKING
The microbenchmarking exhibit some basic characteristics
of modern GPU interconnects. However, in terms of real
multi-GPU applications, their impact remains unknown. In
this section, we use the Tartan Benchmark Suite [25] to eval-
uate the impact of the GPU interconnect. The applications
are listed in Table 3. Particularly, we focus on two aspects:
(1) the impact of a faster GPU interconnect such as NVLink,
compared with PCIe on intra-node scale-up applications;
(2) the impact of GPUDirect-RDMA on inter-node scale-out
applications. We perform two types of scaling measurement:
(a) strong scaling, which ﬁxes the problem size and measures
the time reduction when increasing the number of GPUs;
(b) weak scaling, which measures the time reduction when
increasing the number of GPUs with ﬁxed problem size per
GPU. For overall performance, we use the entire application
speedup (measured by CPU-side time command for whole
application elapsed-time) as the performance metric, mak-
ing a fair comparison across applications. For scale-up ap-
plications, we use the vendor-provided nvprof to measure
the three types of GPU communication: HostToDevice (H2D),
DeviceToHost (D2H) and DeviceToDevice (D2D) in order to
gain more knowledge about the underlying communication


---

11
TABLE 3: Tartan Benchmark Suite.
App
Brief Description
abbr.
Domain
Comm
Scaling
Pattern
ConvNet2
Convolution neural networks via data, model and hybrid parallelism
CNN
Machine Learning
CUDA
Scale-up
P2P
Cusimann
Global optimization via parallel simulated annealing algorithm
CSM
Optimization
OpenMP Scale-up
CPU-GPU
GMM
Multivariate data clustering via Expectation Maximization with Gaussian mixture model
GMM
Data Analysis
OpenMP Scale-up
CL-Broadcast
Kmeans
Kmeans clustering for double-precision data on multi-GPUs attached to the same node
KMN
Data Analysis
CUDA
Scale-up CL-AllReduce
MonteCarlo Monte Carlo option pricing from CUDA SDK
MTC
Finance
CUDA
Scale-up
CPU-GPU
Planar
Depth-ﬁrst-search (DFS) and backtracking to solve Planar Langford’s Sequences
PLN
Number Theory
CUDA
Scale-up
CPU-GPU
Trueke
Exchange Monte Carlo for 3D random ﬁeld Ising model
TRK
HPC Simulation
OpenMP Scale-up
CL-Broadcast
B2rEqwp
3D earthquake wave-propogation model simulation using 4-order ﬁnite difference method
BRQ
HPC Simulation
MPI
Scale-out
P2P
Diffusion
A multi-GPU implementation of 3D Heat Equation and inviscid Burgers’ Equation
DFF
HPC Simulation
MPI
Scale-out
P2P
Lulesh
Livermore unstructured Lagrangian explicit shock hydrodynamics
LLH
Molecular Dynamics
MPI
Scale-out
P2P
CoMD
A reference implementation of classical molecular dynamics algorithms and workloads
CMD
Molecular Dynamics
MPI
Scale-out
P2P/CL
Prbench
Page rank computation by multi-GPUs
PRB
Graph Processing
MPI
Scale-out
P2P/CL
HIT
Simulating Homogeneous Isotropic Turbulence by solving Navier-Stokes equations in 3D
HIT
HPC Simulation
MPI
Scale-out
CL All-to-All
Matvec
Matrix multiplication via mpi-scatter, broadcast and gather
MAM
Linear Algebra
MPI
Scale-out
CL
pattern. All the reported data points of the ﬁgures in this
section are the average results of multiple times’ execution.
4.1
Intra-node Scale-up
For intra-node scale-up scenarios, we evaluated the seven
scale-up applications on P100-DGX-1 and V100-DGX-1, with
and without NVLinks. Since many of these applications are
hard-coded to leverage all the available GPUs in the sys-
tem, we conﬁgure the system environment through export
CUDA VISIBLE DEVICES=x to manipulate the number of
GPUs being visible to the applications. Figure 32 and 33
illustrate the break out of the latency for the three types
of communication (i.e., H2D, D2H, and D2D) regarding the
original implementation (i.e., Baseline) and our modiﬁcation
(i.e., via NCCL) as described in Section IV-A, for intra-node
strong and weak scaling on P100-DGX-1. The intention is
that the original implementation will show the performance
of PCIe, while our modiﬁcation would convert a big portion
of CPU-GPU communication to GPU-GPU communication,
so as to show the performance gain from NVLink. Figure 34
and 35 show the results on V100-DGX-1. The performance
change of the entire application is given in the supplemen-
tary ﬁle due to space limitation.
Impact of NVLink. Although our observation from mi-
crobenchmarking in Section 3 show that NVLink can signif-
icantly improve inter-GPU communication efﬁciency, based
on these ﬁgures, it is clear that those improvements do
not directly transform into overall communication latency
reduction, nor the whole application speedup (see the sup-
plementary ﬁle); except CSM and GMM, there is not very sig-
niﬁcant difference between the Baseline and NCCL bars for
both platforms. There are several reasons behind this. First,
based on our experience on assessing the over 50 multi-GPU
application candidates, most of those scale-up cases are
based on master-slave programming model, where the CPU
is the master, handling the sequential portions of code and
GPUs are the slaves, processing the parallel portions. Under
this model, communication only occurs between CPU and
GPUs; no inter-GPU transaction is presumed. In addition,
the CPU-GPU communication is also highly optimized. This
is true for CSM, GMM, KMN, MTC, PLN and TRK. For CSM, PLN,
GMM, and TRK, we manage to convert some CPU-GPU com-
munication into GPU-GPU communication through NCCL.
As can be seen, for CSM, the effect is obvious: a large portion
of the D2H and H2D communication is replaced by D2D.
For GMM, although we gained ∼6% latency reduction, from
the nvprof trace ﬁle, we did not observe any D2D communi-
cation transactions (but rather D2H and H2D) before/after
NCCL’s BroadcastKernelSmall() kernels. This is potentially
caused by the ﬂexible design strategy adopted by NCCL-V2
to efﬁciently leverage all available interconnect bandwidth
(Section III-B). When the data size per transmission is small,
it may not choose D2D communication via NVLink. Similar
conditions also observed for PLN and TRK. For KMN and MTC,
there is no GPU-GPU communication at all. The D2D in
KMN is actually local data movement within the same GPU.
CNN is another case. For CNN, the ﬁgures do not show bars
under 8 GPUs because CNN implementation requires arbi-
trary data copying among arbitrary peers at runtime, which
currently fails when 8 GPUs are utilized, since not every two
GPUs are directly connected by NVLink under the current
NVLink topology. As it internally uses cudaMemcpyDefault
and uniﬁed-space for data copying across GPUs, NVLink
is already internally enabled by CUDA for an NVLink-
equipped machine such as DGX-1. We tried to modify the
CNN code so that PCIe can be essentially enforced but did
not run correctly with success. This is why the two bars of
CNN exhibit the same value for all four ﬁgures. It is also
interesting to see that when more than 4 GPUs participant
in the computation, the D2H communication increases dra-
matically, potentially due to the gradient merging overhead
in the backpropagation. Secondly, since today’s scale-up
applications are mostly based on the master-slave program-
ming model, communication often only accounts for a small
fraction of the total execution time, let alone the inter-GPU
communication which tended to be avoided previously
when creating applications, thus hardly become the system
bottleneck. Finally, employing NVLink (either P2P or CL)
introduces additional overhead (e.g., enable/disable peer
access, routing, NCCL initialization, etc).
To summarize, a faster GPU interconnect such as
NVLink has been reported to be beneﬁcial for accelerat-
ing modern deep-learning frameworks [31], [32]. However,
regarding general GPGPU applications, without (i) replac-
ing the underlying CPU-centric master-slave programming
model by a more distributed parallelization model, or (ii)
migrating the communication master role to a GPU (e.g., off-
loading GPU communication control from CPU to GPU via
techniques such as NVSHMEM [33]), optimized inter-GPU
communication via faster intra-node GPU interconnect such
as NVLinks can hardly become signiﬁcant enough to lift
the entire application’s speedup. Therefore, we believe that
this observation paves the road for developing interconnect-


---

12
1
2
4
8
0.0
0.5
1.0
1.5
2.0
Normalized Latency
16.9
16.9
CNN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
CSM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
GMM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
KMN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
MTC
1
2
4
8
0.0
0.5
1.0
1.5
2.0
PLN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
TRK
H2D
D2H
D2D
Fig. 32: Normalized latency reduction by NVLink-V1 and NCCL-V2 of strong scaling for single-node scaling-up on NVIDIA P100-DGX-1.
1
2
4
8
0.0
0.5
1.0
1.5
2.0
Normalized Latency
17.6
17.7
CNN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
CSM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
GMM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
KMN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
MTC
1
2
4
8
0.0
0.5
1.0
1.5
2.0
PLN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
TRK
H2D
D2H
D2D
Fig. 33: Normalized latency reduction by NVLink-V1 and NCCL-V2 of weak scaling for single-node scaling-up on NVIDIA P100-DGX-1.
1
2
4
8
0.0
0.5
1.0
1.5
2.0
Normalized Latency
16.1
15.7
CNN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
CSM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
GMM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
KMN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
MTC
1
2
4
8
0.0
0.5
1.0
1.5
2.0
PLN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
TRK
H2D
D2H
D2D
Fig. 34: Normalized latency reduction by NVLink-V2 and NCCL-V2 of strong scaling for single-node scaling-up on NVIDIA V100-DGX-1.
1
2
4
8
0.0
0.5
1.0
1.5
2.0
Normalized Latency
16.2
16.5
CNN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
CSM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
GMM
1
2
4
8
0.0
0.5
1.0
1.5
2.0
KMN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
MTC
1
2
4
8
0.0
0.5
1.0
1.5
2.0
PLN
1
2
4
8
0.0
0.5
1.0
1.5
2.0
TRK
H2D
D2H
D2D
Fig. 35: Normalized latency reduction by NVLink-V2 and NCCL-V2 of weak scaling for single-node scaling-up on NVIDIA V100-DGX-1.
friendly programming models for multi-GPU scale-up sce-
narios so that faster interconnect (e.g., NVLinks) can truly
play a role in improving the overall application efﬁciency.
4.2
Inter-node Scale-out
For inter-node scale-out scenarios, we run the seven scale-
out applications from Tartan on SummitDev and Summit (see
Table 1), with each MPI rank binding to a node using only a
single GPU. Similar to the discussion in Section 3.3, we mea-
sured the overall application performance under ﬁve scenar-
ios: GPUDirect-RDMA, PinnedMem-GPUDirect, PinnedMem,
UnpinnedMem-GPUDirect and UnpinnedMem.
Figure 36 and 37 illustrate the speedups with respect to
single-node UnpinnedMem for strong and weak scaling tests,
respectively, on SummitDev. Figure 38 and 39 illustrate the
speedups for strong and weak scaling on Summit. As can be
seen, compared with the intra-node scale-up cases, the MPI-
based inter-node scale-out applications exhibit much better
scaling behavior in both strong and weak scaling tests, im-
plying that compared with the intra-node fast interconnect,
the inter-node network is much easier to become the system
bottleneck. Improving inter-node network speed can lead to
signiﬁcant performance gain for multi-GPU applications.
Regarding to GPUDirect-supported IB interconnect, we
have the following observations: (i) Enabling GPUDirect
can bring immediate performance enhancement, whether
or not the transmitted data reside in CPU memory or
GPU memory; (ii) Using pinned memory is also beneﬁ-
cial, especially in coordination with GPUDirect enabled;
(iii) GPUDirect+RDMA can be especially helpful in certain
applications (e.g., BRQ and MAM) for SummitDev, and overall
for Summit. This is potentially due to the signiﬁcant im-
provement on GPUDirect-RDMA performance in Summit
than SummitDev; (iv) Overall, the multi-node performance
scaling on Summit is less obvious than on SummitDev, espe-
cially for strong scaling. This is not due to any degradation
in communication efﬁciency in Summit, but essentially the
reverse: Summit improves the fundamental communication
bandwidth, which enhances the baseline performance. That
is why the relative speedups drop on Summit, compared
to SummitDev. Overall, we suggest the application devel-
opers to adopt PinnedMem-GPUDirect for SummitDev, and
GPUDirect-RDMA for Summit.
All in all, for scale-out applications to beneﬁt from a
faster inter-node interconnect (e.g., IB-RDMA), the major
difﬁculty is not from the hardware or the application,
but from the communication abstract interfaces such as
MPI. If a new MPI implementation can internally integrate
NCCL, further harvesting multi-GPU interconnect perfor-
mance (e.g., NVLink and IB-RDMA) can be much more
easier. Again, initiating communication completely on the
GPU side without CPU intervention (e.g., via GPUDirect-
Async [34] or GPU-triggered networking [35]) may also be
critical for good GPU performance delivery.


---

13
1
2
4
8
0
1
2
3
4
5
Speedup
BRQ
1
2
4
8
0
1
2
3
4
5
DFF
1
2
4
8
0
1
2
3
4
5
LLH
1
2
4
8
0
1
2
3
4
5
CMD
1
2
4
8
0
1
2
3
4
5
PRB
1
2
4
8
0
1
2
3
4
5
HIT
1
2
4
8
0
1
2
3
4
5
MAM
Unpinned
Unpinned_GPUDirect
Pinned
Pinned_GPUDirect
GPUDirect-RDMA
Fig. 36: Performance speedup by InﬁniBand GPUDirect-RDMA of strong scaling for multi-node scaling-out on ORNL SummitDev.
1
2
4
8
0
1
2
3
4
5
Speedup
BRQ
1
2
4
8
0
1
2
3
4
5
DFF
1
2
4
8
0
1
2
3
4
5
LLH
1
2
4
8
0
1
2
3
4
5
CMD
1
2
4
8
0
1
2
3
4
5
PRB
1
2
4
8
0
1
2
3
4
5
HIT
1
2
4
8
0
1
2
3
4
5
MAM
Unpinned
Unpinned_GPUDirect
Pinned
Pinned_GPUDirect
GPUDirect-RDMA
Fig. 37: Performance speedup by InﬁniBand GPUDirect-RDMA of weak scaling for multi-node scaling-out on ORNL SummitDev.
1
2
4
8
0
1
2
3
4
5
Speedup
BRQ
1
2
4
8
0
1
2
3
4
5
DFF
1
2
4
8
0
1
2
3
4
5
LLH
1
2
4
8
0
1
2
3
4
5
CMD
1
2
4
8
0
1
2
3
4
5
PRB
1
2
4
8
0
1
2
3
4
5
HIT
1
2
4
8
0
1
2
3
4
5
MAM
Unpinned
Unpinned_GPUDirect
Pinned
Pinned_GPUDirect
GPUDirect-RDMA
Fig. 38: Performance speedup by InﬁniBand GPUDirect-RDMA of strong scaling for multi-node scaling-out on ORNL Summit.
1
2
4
8
0
1
2
3
4
5
Speedup
BRQ
1
2
4
8
0
1
2
3
4
5
DFF
1
2
4
8
0
1
2
3
4
5
LLH
1
2
4
8
0
1
2
3
4
5
CMD
1
2
4
8
0
1
2
3
4
5
PRB
1
2
4
8
0
1
2
3
4
5
HIT
1
2
4
8
0
1
2
3
4
5
MAM
Unpinned
Unpinned_GPUDirect
Pinned
Pinned_GPUDirect
GPUDirect-RDMA
Fig. 39: Performance speedup by InﬁniBand GPUDirect-RDMA of weak scaling for multi-node scaling-out on ORNL Summit.
5
DISCUSSION
Modern GPU interconnect technologies such as NVLink are
claimed to be transparent but in reality it is more compli-
cated to be leveraged for high performance. (1) NUMA effect.
Among the ﬁve types of intra-node GPU interconnect tech-
niques, PCIe, NVLink-V1 and V2 show strong NUMA effect
in the tested platforms, due to various reasons including
topology, position, connectivity, routing, sharing, chipset,
etc. NVSwitch and NV-SLI show UMA. (2) Heterogeneity.
All the tested platforms incorporate more than one type
of interconnect network. These networks have their own
characteristics and can work exclusively, concurrently, or
cooperatively, depending on the system design. Therefore,
one has to handle the interconnect heterogeneity: choosing
one interconnect over the others (e.g., NVLink shows strong
advantage on bandwidth rather than latency over PCIe),
leveraging them simultaneously, or cooperatively integrate
them as an inclusive solution. This is especially difﬁcult
at runtime. (3) Communication Efﬁciency. There are several
factors restricting the delivery of optimal communication
performance, including message size, system design (dual
isolated subnetworks in Summit and SummitDev), hard-
ware limitation (e.g., PCIe antilocality, GPUDirect-RDMA
in SummitDev), and library implementation (e.g., NCCL on
DGX-1 with 3 and 5 nodes). All these lead to the difﬁculties
in leveraging modern GPU interconnect for high-efﬁcient
inter-GPU communication.
Our evaluation motivates the following research direc-
tions: (1) Developing novel multi-GPU programming models.
Existing multi-GPU programming models rely on CPU-
oriented parallel programming models, such as OpenMP
and MPI, to manage multiple GPUs. Consequently, either
there is a mismatch (e.g., CPU-master-GPU-slave model
can hardly beneﬁt from inter-GPU network), or there is
a legacy in adopting new GPU interconnect technologies
(e.g., integrating NCCL into MPI, as it is shown that NCCL
delivers better communication efﬁciency than MPI on all-
reduce [36]). Therefore, new multi-GPU programming mod-
els are highly desired, especially those that are adaptive,
portable, tractable, and being able to effectively address
the complexity aforementioned. In addition, existing multi-
GPU algorithms are usually designed to minimize or even
eliminate communications, due to the huge performance
gap between local access and remote access (e.g., via PCIe).
With the emergence of new GPU interconnect, and foresee-
ing their fast development trend, it may be the right time
to reconsider the role of inter-GPU communication when
designing new parallel models and algorithms. (2) Develop-
ing practical multi-GPU performance models. This is for per-
formance prediction, optimization, and analytics in multi-
GPU application development and tuning. In addition, an
appropriate performance model is crucial for GPU task al-


---

14
location, scheduling and migration in a shared environment
(e.g., Cloud and HPC centers). (3) Developing new com-
munication patterns and libraries for better matching the
underlying interconnect and delivering high-performance.
For example, regarding the dual-subnetwork interconnect
topology for NVLink in Summit, it is worth to ﬁgure out
how to efﬁciently distribute and exchange data among the
two islands, taking data reuse and X-bus bandwidth into
consideration. Give another example, a recent work shows
that a 2D-Torus communication pattern can deliver better
communication efﬁciency than NCCL’s ring pattern for all-
reduce. This new pattern can be obviously migrated from
inter-node to the intra-node NVLink interconnect in DGX-
1s, or NVSwitch in DGX-2, where multiple communication
ring-paths can be constructed. This is especially desired
when porting traditional CPU-based HPC applications onto
the new GPU-based exascale systems, such as Summit [6],
Sierra [7] and Perlmutter [37]. As part of the community
effort, we are planning to pursue these research directions
in our future work with our past experience on GPU analytic
modeling [38], [39], [40] and performance optimization [41],
[42], [43], [44], [45], [46], [47], [48], [49], [50], [51].
6
RELATED WORK
Intra-node GPU Computing. Spafford et al. [52] analyzed
the NUMA effects in a multi-GPU node and provided
optimization guidance. Kim et al. [15] proposed to rely
on hybrid-memory-cubes (HMCs) to build a memory net-
work for simplifying multi-GPU memory management and
improving programmability. Wang et al. [53] presented a
design to realize GPU-Aware MPI to support data commu-
nication among intra-node GPUs with standard MPI. Ben-
Nun et al. [54] described an automatic multi-GPU partition
framework to distribute workload based on their memory
access patterns. Cabezas et al. [55] showed a software so-
lution, including programming interfaces, compiler support
and runtime, to partition GPU kernels for multi-GPU ex-
ecution in a single node. Finally, Sun et al. [56] evaluated
the potential performance beneﬁt and tradeoffs of AMD’s
Radeon Open Compute (ROC) platform for Heterogeneous Sys-
tem Architecture (HSA).
Multi-node GPU Computing. For MPI-based multi-node
GPU computing, Wang et al. [57] introduced a MPI design
that integrates CUDA data movement transparently with
MPI. Gysi et al. [58] proposed a hardware approach to
overlap computation and communication in a GPU cluster.
Klenk et al. [59] analyzed the exascale proxy applications
on their communication patterns and proposed a matching
algorithm for GPUs to comply with MPI constraints. Awan
et al. [60] proposed a pipelined chain design for MPI broad-
cast collective operations on multi-GPU nodes to facilitate
various deep learning frameworks.
7
CONCLUSION
In this paper, we characterize and evaluate six types
of modern GPU interconnects, including PCIe, NVLink-
V1, NVLink-V2, NV-SLI, NVSwitch, and InﬁniBand with
GPUDirect-RDMA, using the Tartan Benchmark Suite over
six GPU servers and HPC platforms: NVIDIA’s P100-DGX-
1, V100-DGX-1, DGX-2, RTX2080-SLI systems, and ORNL’s
SummitDev and Summit supercomputers, covering both
Peer-to-Peer and Collective communication patterns. We
addressed four new types of NUMA effects for intra-node
GPU communication, and proposed some insightful obser-
vations for enabling practical optimization guidelines. This
evaluation study attempts to help the HPC community
to push forward multi-GPU research and development,
particularly the construction of more mature multi-GPU
programming, execution, and performance models.
ACKNOWLEDGMENTS
This research was supported by the Application Assessment
program within the Exascale Computing Project (17-SC-
20-SC), a collaborative effort of the U.S. Department of
Energy Ofﬁce of Science and the National Nuclear Security
Administration. This research was supported by the U.S.
DOE Ofﬁce of Science, Ofﬁce of Advanced Scientiﬁc Com-
puting Research, under award 66150: “CENATE - Center
for Advanced Architecture Evaluation”. This research was
supported by the High Performance Data Analytics (HPDA)
program at PNNL. This research used resources of the Oak
Ridge Leadership Computing Facility at the Oak Ridge
National Laboratory, which is supported by the Ofﬁce of
Science of the U.S. Department of Energy under Contract
No. DE-AC05-00OR22725. The Paciﬁc Northwest National
Laboratory is operated by Battelle for the U.S. Department
of Energy under contract DE-AC05-76RL01830.
REFERENCES
[1]
P. Goyal, P. Doll´ar, R. Girshick, P. Noordhuis, L. Wesolowski,
A. Kyrola, A. Tulloch, Y. Jia, and K. He, “Accurate, Large
Minibatch SGD: Training ImageNet in 1 Hour,” arXiv preprint
arXiv:1706.02677, 2017.
[2]
O. Fuhrer, T. Chadha, T. Hoeﬂer, G. Kwasniewski, X. Lapillonne,
D. Leutwyler, D. L¨uthi, C. Osuna, C. Sch¨ar, T. C. Schulthess et al.,
“Near-global climate simulation at 1 km resolution: establishing a
performance baseline on 4888 gpus with cosmo 5.0,” Geoscientiﬁc
Model Development, 2018.
[3]
H.
Mikami,
H.
Suganuma,
P.
U-chupala,
Y.
Tanaka,
and
Y. Kageyama, “Massively Distributed SGD: ImageNet/ResNet-50
Training in a Flash,” arXiv preprint arXiv:1811.05233, 2018.
[4]
NVIDIA, “NVIDIA DGX-1 System Architecture White Paper,”
2017.
[5]
NVIDIA, “NVIDIA DGX-2H The World’s Most Powerful System
for The Most Complex AI Challenges,” 2018.
[6]
OLCF, “Summit: The Next Leap in Leadership-Class Com-
puting Systems for Open Science,” https://www.olcf.ornl.gov/
for-users/system-user-guides/summit/.
[7]
OLCF, “Sierra: Supporting NNSA’s stockpile stewardship mis-
sion through simulation in lieu of underground testing,” https:
//computation.llnl.gov/computers/sierra.
[8]
S. Pabst, A. Koch, and W. Straßer, “Fast and scalable cpu/gpu
collision detection for rigid and deformable surfaces,” in Computer
Graphics Forum.
Wiley Online Library, 2010.
[9]
Q. Xu, H. Jeon, and M. Annavaram, “Graph processing on GPU:
Where are the bottlenecks?” in International Symposium on Workload
Characterization (IISWC).
IEEE, 2014.
[10] “The
System
Bottleneck
Shifts
to
PCI-Express,”
https://www.nextplatform.com/2017/07/14/
system-bottleneck-shifts-pci-express/.
[11] D. Ziakas, A. Baum, R. A. Maddox, and R. J. Safranek, “Intel R⃝
quickpath interconnect architectural features supporting scalable
system architectures,” in High Performance Interconnects (HOTI),
2010 IEEE 18th Annual Symposium on.
IEEE, 2010.


---

15
[12] D. Foley and J. Danskin, “Ultra-Performance Pascal GPU and
NVLink Interconnect,” IEEE Micro, 2017.
[13] NVIDIA, “SLI best practices,” Tech. Rep., 2007.
[14] AMD, “ATI CrossFire Pro User Guide,” Tech. Rep., 2009.
[15] G. Kim, M. Lee, J. Jeong, and J. Kim, “Multi-GPU system design
with memory networks,” in 47th Annual IEEE/ACM International
Symposium on Microarchitecture (MICRO).
IEEE, 2014.
[16] NVIDIA,
“NVIDIA
NVSWITCH
–
The
World’s
Highest-
Bandwidth
On-Node
Switch,”
http://images.nvidia.com/
content/pdf/nvswitch-technical-overview.pdf.
[17] P. Grun, “Introduction to inﬁniband for end users,” White paper,
InﬁniBand Trade Association, 2010.
[18] NVIDIA, “Developing a Linux Kernel Module using GPUDirect
RDMA,”
http://docs.nvidia.com/cuda/gpudirect-rdma/index.
html.
[19] AMD, “ROCm Driver RDMA Peer to Peer Support,” https://
github.com/RadeonOpenCompute/ROCnRDMA.
[20] NVIDIA, “CUDA SDK Code Samples,” 2015.
[21] NVIDIA, “NCCL Tests,” https://github.com/NVIDIA/nccl-tests.
[22] NVIDIA, “NVIDIA Collective Communications Library (NCCL-
V1),” https://github.com/NVIDIA/nccl.
[23] NVIDIA, “NVIDIA Collective Communications Library (NCCL-
V2),” https://developer.nvidia.com/nccl.
[24] “MPI-GPU-BW,” https://github.com/frroberts/MPI-GPU-BW.
[25] A. Li, S. Song, J. Chen, X. Liu, N. Tallent, and K. Barker, “Tartan:
Evaluating Modern GPU Interconnect via a Multi-GPU Bench-
mark Suite,” in International Symposium on Workload Characteriza-
tion (IISWC).
IEEE, 2018.
[26] J. D. Little, “A proof for the queuing formula: L= λ W,” Operations
research, 1961.
[27] Cirrascale, “Cirrascale SR3514: Unexpected Performance Inequal-
ity. Technical Brief M901A-092014.”
[28] AMD, “ROCm Communication Collectives Library (RCCL),”
https://github.com/ROCmSoftwarePlatform/rccl.
[29] OLCF,
“Summit
Early
Access
Development
Platform,”
https://www.olcf.ornl.gov/for-users/system-user-guides/
summitdev-quickstart-guide/.
[30] NVIDIA, “Developing a Linux Kernel Module using GPUDirect
RDMA,” https://docs.nvidia.com/cuda/gpudirect-rdma/index.
html.
[31] S. Shams, R. Platania, K. Lee, and S.-J. Park, “Evaluation of Deep
Learning Frameworks Over Different HPC Architectures,” in 37th
International Conference on Distributed Computing Systems (ICDCS).
IEEE, 2017.
[32] N. R. Tallent, N. A. Gawande, C. Siegel, A. Vishnu, and A. Hoisie,
“Evaluating On-Node GPU Interconnects for Deep Learning
Workloads,” in International Workshop on Performance Modeling,
Benchmarking and Simulation of High Performance Computer Systems.
Springer, 2017.
[33] S. Potluri, A. Goswami, D. Rossetti, C. Newburn, M. G. Venkata,
and N. Imam, “GPU-Centric Communication on NVIDIA GPU
Clusters with InﬁniBand: A Case Study with OpenSHMEM,” in
HiPC-24.
IEEE, 2017.
[34] E. Agostini, D. Rossetti, and S. Potluri, “GPUDirect Async: Explor-
ing GPU synchronous communication techniques for InﬁniBand
clusters,” Journal of Parallel and Distributed Computing, 2018.
[35] M. LeBeane, K. Hamidouche, B. Benton, M. Breternitz, S. K. Rein-
hardt, and L. K. John, “GPU triggered networking for intra-kernel
communications,” in International Conference for High Performance
Computing, Networking, Storage and Analysis (SC).
ACM, 2017.
[36] J. Yin, “Early experiences with Machine Learning and Deep
Learning on Summit/SummitDev,” https://www.olcf.ornl.gov/
wp-content/uploads/2018/12/summit training mldl.pdf, 2018.
[37] NERSC, “Perlmutter: NERSC’s Next Supercomputer,” https://
www.nersc.gov/systems/perlmutter/.
[38] A. Li, Y. Tay, A. Kumar, and H. Corporaal, “Transit: A visual
analytical model for multithreaded machines,” in International
Symposium on High-Performance Parallel and Distributed Computing
(HPDC).
ACM, 2015.
[39] A. Li, S. L. Song, E. Brugel, A. Kumar, D. Chavarria-Miranda, and
H. Corporaal, “X: A comprehensive analytic model for parallel
machines,” in International Parallel and Distributed Processing Sym-
posium (IPDPS).
IEEE, 2016.
[40] A. Li, W. Liu, M. R. Kristensen, B. Vinter, H. Wang, K. Hou, A. Mar-
quez, and S. L. Song, “Exploring and analyzing the real impact
of modern on-package memory on HPC scientiﬁc kernels,” in
International Conference for High Performance Computing, Networking,
Storage and Analysis (SC).
ACM, 2017.
[41] A. Li, G.-J. van den Braak, A. Kumar, and H. Corporaal, “Adap-
tive and transparent cache bypassing for GPUs,” in International
Conference for High Performance Computing, Networking, Storage and
Analysis (SC).
ACM, 2015.
[42] A. Li, G.-J. van den Braak, H. Corporaal, and A. Kumar, “Fine-
grained synchronizations and dataﬂow programming on GPUs,”
in International Conference on Supercomputing (ICS).
ACM, 2015.
[43] A. Li, S. L. Song, M. Wijtvliet, A. Kumar, and H. Corporaal,
“SFU-driven transparent approximation acceleration on GPUs,”
in International Conference on Supercomputing (ICS).
ACM, 2016.
[44] A. Li, S. L. Song, A. Kumar, E. Z. Zhang, D. Chavarr´ıa-Miranda,
and H. Corporaal, “Critical points based register-concurrency
autotuning for GPUs,” in Conference on Design, Automation & Test
in Europe, 2016.
[45] W. Liu, A. Li, J. Hogg, I. S. Duff, and B. Vinter, “A synchronization-
free algorithm for parallel sparse triangular solves,” in European
Conference on Parallel Processing (EuroPar).
Springer, 2016.
[46] J. Li, Y. Ma, C. Yan, and R. Vuduc, “Optimizing sparse tensor times
matrix on multi-core and many-core architectures,” in Proceedings
of the Sixth Workshop on Irregular Applications: Architectures and
Algorithms.
IEEE Press, 2016.
[47] A. Li, S. L. Song, W. Liu, X. Liu, A. Kumar, and H. Corporaal,
“Locality-aware CTA clustering for modern GPUs,” ACM SIGOPS
Operating Systems Review, 2017.
[48] A. Li, W. Zhao, and S. L. Song, “Bvf: enabling signiﬁcant on-chip
power savings via bit-value-favor for throughput processors,” in
Proceedings of the 50th Annual IEEE/ACM International Symposium
on Microarchitecture.
ACM, 2017.
[49] A. Li, W. Liu, L. Wang, K. Barker, and S. L. Song, “Warp-
consolidation: A novel execution model for gpus,” in International
Conference on Supercomputing (ICS).
ACM, 2018.
[50] D. Shen, S. L. Song, A. Li, and X. Liu, “Cudaadvisor: Llvm-
based runtime proﬁling for modern gpus,” in Proceedings of the
2018 International Symposium on Code Generation and Optimization.
ACM, 2018.
[51] L. Wang, J. Ye, Y. Zhao, W. Wu, A. Li, S. L. Song, Z. Xu, and
T. Kraska, “Superneurons: Dynamic GPU memory management
for training deep neural networks,” in ACM SIGPLAN Notices.
ACM, 2018.
[52] K. Spafford, J. S. Meredith, and J. S. Vetter, “Quantifying NUMA
and contention effects in multi-GPU systems,” in GPGPU-4.
ACM, 2011.
[53] H. Wang, S. Potluri, D. Bureddy, C. Rosales, and D. K. Panda,
“GPU-aware MPI on RDMA-enabled clusters: Design, implemen-
tation and evaluation,” IEEE Transactions on Parallel and Distributed
Systems, 2014.
[54] T. Ben-Nun, E. Levy, A. Barak, and E. Rubin, “Memory access pat-
terns: the missing piece of the multi-GPU puzzle,” in International
Conference for High Performance Computing, Networking, Storage and
Analysis (SC).
ACM, 2015.
[55] J. Cabezas, L. Vilanova, I. Gelado, T. B. Jablin, N. Navarro, and
W.-m. W. Hwu, “Automatic parallelization of kernels in shared-
memory multi-GPU nodes,” in International Conference on Super-
computing (SC).
ACM, 2015.
[56] Y. Sun, S. Mukherjee, T. Baruah, S. Dong, J. Gutierrez, P. Mohan,
and D. Kaeli, “Evaluating performance tradeoffs on the radeon
open compute platform,” in International Symposium on Perfor-
mance Analysis of Systems and Software (ISPASS).
IEEE, 2018.
[57] H. Wang, S. Potluri, M. Luo, A. K. Singh, S. Sur, and D. K. Panda,
“MVAPICH2-GPU: optimized GPU to GPU communication for
InﬁniBand clusters,” Computer Science-Research and Development,
2011.
[58] T. Gysi, J. B¨ar, and T. Hoeﬂer, “dCUDA: hardware supported
overlap of computation and communication,” in International Con-
ference for High Performance Computing, Networking, Storage and
Analysis (SC).
IEEE, 2016.
[59] B. Klenk, H. Fr¨oening, H. Eberle, and L. Dennison, “Relaxations
for high-performance message passing on massively parallel SIMT
processors,” in International Parallel and Distributed Processing Sym-
posium (IPDPS).
IEEE, 2017.
[60] A. A. Awan, C.-H. Chu, H. Subramoni, and D. K. Panda, “Op-
timized Broadcast for Deep Learning Workloads on Dense-GPU
InﬁniBand Clusters: MPI or NCCL?” arXiv:1707.09414, 2017.
