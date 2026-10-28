---
title: 'Exploring GPU-to-GPU Communication:'
id: exploring-gpu-to-gpu-communication
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:19:03.299656Z'
updated: '2026-10-27T19:00:31.671576Z'
source: https://htor.inf.ethz.ch/publications/img/SC_DeSensi_2024_Clean.pdf
source_domain: htor.inf.ethz.ch
fetched_at: '2026-10-27T13:19:03.298925Z'
fetch_provider: builtin
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Suggested by p2pmd — Taxonomy and measured comparison of GPU-to-GPU transfer
  modes: trivial host staging vs device-de...'
raw_file: raw/exploring-gpu-to-gpu-communication.pdf
utility_score: 15.0
---

*Suggested by [[p2pmd]] — Taxonomy and measured comparison of GPU-to-GPU transfer modes: trivial host staging vs device-device peer access vs CCL*

Exploring GPU-to-GPU Communication: 
Insights into Supercomputer Interconnects
Daniele De Sensi, Lorenzo Pichetti, Flavio Vella, Tiziano De Matteis, 
Zebin Ren, Luigi Fusco, Matteo Turisini, Daniele Cesarini, Kurt Lust, 
Animesh Trivedi, Duncan Roweth, Filippo Spiga, Salvatore Di Girolamo, Torsten Hoefler
The International Conference for High Performance Computing, Networking, Storage, and Analysis
17–22 Nov 2024 • Atlanta, GA.


---

Motivation
2
Systems are getting larger
(up to 100,000 GPUs)
Nodes are getting denser
(e.g., 8 GPUs on the same node)
Maturity of the software stack
(MPI, NCCL, RCCL, etc…)


---

Systems analyzed
3
Alps (#6 in Top500)
• NVIDIA H100 GPUs
• HPE Cray Slingshot
• Dragonfly
1.2 Tb/s
per dir.
200 Gb/s per direction
Early-access
GPU
GPU
GPU
GPU


---

Systems analyzed
4
Alps (#6 in Top500)
• NVIDIA H100 GPUs
• HPE Cray Slingshot
• Dragonfly
1.2 Tb/s
per dir.
200 Gb/s per direction
GPU
GPU
GPU
GPU
Leonardo (#7 in Top500)
• NVIDIA A100 GPUs
• NVIDIA InfiniBand
• Dragonfly+
800 Gb/s
per dir.
100 Gb/s per direction
GPU
GPU
GPU
GPU
Early-access


---

Systems analyzed
5
Alps (#6 in Top500)
• NVIDIA H100 GPUs
• HPE Cray Slingshot
• Dragonfly
1.2 Tb/s
per dir.
200 Gb/s per direction
GPU
GPU
GPU
GPU
Leonardo (#7 in Top500)
• NVIDIA A100 GPUs
• NVIDIA InfiniBand
• Dragonfly+
800 Gb/s
per dir.
100 Gb/s per direction
GPU
GPU
GPU
GPU
LUMI (#5 in Top500)
• AMD MI250X GPUs
• HPE Cray Slingshot
• Dragonfly
GCD
GCD
GCD
GCD
GCD
GCD
GCD
GCD
400 Gb/s
per dir.
200 Gb/s per direction
Early-access


---

How to Transfer Data Between GPUs
6
• Trivial staging (GPU->Host->Host->GPU)
• Device-device copy (peer access)
• *CCL (i.e., NCCL/RCCL)
• GPU-Aware MPI
We used the software versions suggested by 
each supercomputing center


---

Experiments Executed
7
Point-to-point
Collective Operations 
(allreduce and alltoall)
Intra-Node
Inter-Node


---

Experiments Executed
8
Point-to-point
Collective Operations 
(allreduce and alltoall)
Intra-Node
Inter-Node
GPU
GPU
GPU
GPU
GPU
GPU
GPU
GPU


---

Intra-Node Point-to-Point
9
Alps


---

Intra-Node Point-to-Point
10
Alps


---

Intra-Node Point-to-Point
11
Alps


---

Intra-Node Point-to-Point
12
Alps


---

Intra-Node Point-to-Point
13
Device-device copy 
not enabled at the time of writing
Alps


---

Intra-Node Point-to-Point
14
Beware: Different y-axis ranges across the plots
Alps
Leonardo
GDRCopy


---

Intra-Node Point-to-Point
15
Beware: Different y-axis ranges across the plots
Alps
Leonardo
LUMI
Direct load/store from 
CPU to GPU memory


---

Intra-Node Point-to-Point
16
Beware: Different y-axis ranges across the plots
Alps
Leonardo
LUMI
MPI and *CCL out-of-the-box performance is sub-optimal 
(i.e., up to 6x performance improvements after tuning)
For small point-to-point intra-node transfers, 
MPI can be 10x faster than *CCL


---

Experiments Executed
17
Point-to-point
Collective Operations 
(allreduce and alltoall)
Intra-Node
Inter-Node
GPU
GPU
GPU
GPU
GPU
GPU
GPU
GPU


---

Intra-Node Allreduce
18
Alps
Leonardo
LUMI
Reduction on CPU


---

Intra-Node Allreduce
19
Alps
Leonardo
LUMI
*CCL outperforms MPI up to 2x on collective operations 


---

Experiments Executed
20
Point-to-point
Collective Operations 
(allreduce and alltoall)
Intra-Node
Inter-Node
GPU
GPU
GPU
GPU
GPU
GPU
GPU
GPU


---

Inter-Node Point-to-Point
21
MPI outperforms *CCL up to 10x on small transfers, 
and up to 3x on larger transfers


---

Leonardo Inter-Node Point-to-Point (for Different Distances)
22


---

23
Leonardo Inter-Node Point-to-Point (for Different Distances)


---

24
Leonardo Inter-Node Point-to-Point (for Different Distances)


---

25
Leonardo Inter-Node Point-to-Point (for Different Distances)
Leonardo inter-node average latency increases up to 2x due to 
network noise. Performance drops up to 50% for collectives at scale
On Alps and LUMI, GPU’s network location has a marginal impact
More details in the paper


---

Experiments Executed
26
Point-to-point
Collective Operations 
(allreduce and alltoall)
Intra-Node
Inter-Node
GPU
GPU
GPU
GPU
GPU
GPU
GPU
GPU


---

Inter-Node Alltoall
27


---

Inter-Node Alltoall
28


---

Inter-Node Alltoall
29


---

Inter-Node Alltoall
30


---

Inter-Node Alltoall
31


---

Inter-Node Alltoall
32
On large-scale collectives, *CCL outperforms MPI
However, we observed instabilities for the alltoall


---

MPI vs. *CCL
33
Alltoall
Allreduce
MPI better than RCCL (up to 15x)
RCCL better than MPI (up to 1.5x)
MPI better than RCCL (up to 10x)
RCCL better than MPI (up to 3.5x)


---

34
Conclusions & Main Take-Home Messages
https://danieledesensi.github.io
desensi@di.uniroma1.it
To sysadmins and users:
Non-negligible tuning effort required to achieve good performance
To software designers:
MPI or *CCL? It depends (collectives vs. point-to-point, etc...)
To sysadmins and system architects:
Adaptive routing (at least on DF+) might require further optimization
