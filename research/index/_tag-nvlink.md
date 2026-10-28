---
title: "Tag: nvlink"
id: "_tag-nvlink"
type: index
created: 2026-10-27T19:00:34.893837+00:00
updated: 2026-10-27T19:00:34.893837+00:00
---

# Tag: nvlink

**4** notes.

- [[efficient-multi-path-nvlinkpcie-aware-ucx-based-collective-communication-for-dee]] — Efficient Multi-Path NVLink/PCIe-Aware UCX based Collective Communication for Deep Learning — Temucin, Sojoodi, Alizadeh & Afsahi (HOTI 2021, Queen's University) show that a single NVLink/PCIe path badly under-utilises intra-node GPU-GPU bandwidth, and fix it with a multi-path copy that stripes messages across multiple NVLink channels, memory regions and GPU streams. On UCX this yields 1.69x higher bandwidth (72 -> ~122 GB/s) and 1.84x on Open MPI + UCX (up to 134 GB/s), on a 4xV100 POWER9 node with three intra-socket NVLinks.

They then build a 3-stage hierarchical, pipelined MPI_Allreduce: multi-path NVLink for intra-socket stages 1 and 3, and PCIe / X-Bus channels for the inter-socket stage 2, chosen per message size to minimise interference (PCIe Gen4 and X-Bus are both ~32 GB/s, so the X-Bus is the bottleneck on the NVLink-X-Bus-NVLink path). For 512 KB-1 GB messages the hierarchical allreduce beats Spectrum MPI, Open MPI+UCX, Open MPI+HPC-X, MVAPICH2-GDR and NCCL by up to 12.25x, 15.63x, 3.72x, 1.48x and 1.38x respectively; at 64 MB specifically the speedups are 11.47x, 14.33x, 1.09x, 1.26x and 1.25x. It outperforms NCCL for all messages >8 MB.

At the application layer with Horovod+TensorFlow: VGG16 gets up to 2.98x/3.42x/3.22x/1.23x/3.24x over those five libraries, ResNet50 1.50x/1.57x/1.22x/1.24x/1.23x.

Why it matters here: it establishes that NCCL is not a hard ceiling on a PCIe/NVLink intra-node fabric - a hierarchical collective that uses ALL available paths (including the host X-bus staging path a no-P2P box is forced onto anyway) can beat it by ~1.25x at >=8 MB. The mechanism is dated (2021, V100, NCCL 2.5.6) but the structural insight - one path at a time wastes bandwidth - is exactly the situation on a 5-GPU box with no P2P, where every collective round-trips through host RAM. `review`
- [[evaluating-modern-gpu-interconnect-pcie]] — Evaluating Modern GPU Interconnect: PCIe, — Li, Song, Chen, Li, Liu, Tallent & Barker (PNNL; arXiv 1903.04611, IEEE TPDS) is the canonical primary evaluation of modern GPU interconnects - PCIe, NVLink-V1, NVLink-V2, NV-SLI, NVSwitch and GPUDirect-RDMA - across six platforms: P100-DGX-1, V100-DGX-1, DGX-2, OLCF SummitDev, Summit, and a 2x Turing RTX-2080 SLI system. They measure raw startup latency, sustainable uni/bidirectional bandwidth, topology, routing and NUMA effects under both P2P and collective (NCCL) patterns.

Four new NUMA effects are identified: three triggered by NVLink topology/connectivity/routing, and one by a PCIe chipset design issue they label "anti-locality" - nearby access presenting LOWER performance than remote access, observed only for bandwidth (not latency) and appearing for messages >64 KB. PCIe latency is essentially uniform across pairs (same through one PCIe switch, across a local CPU socket, or across the QPI bridge), i.e. no NUMA effect in PCIe latency; NVLink latency is ~9 us direct and rises ~2x (P100) / ~3x (V100) when routing is required. NV-SLI: local 5 us, NV-SLI 8 us, PCIe 13 us. NVSwitch gives 50 GB/s bidirectional per link, 300 GB/s via a single NVSwitch hop, and 2.4 TB/s baseboard-to-baseboard bisection.

Message-size saturation points differ by an order of magnitude: NCCL collective bandwidth saturates at ~16 MB on PCIe but ~256 MB on NVLink (8 GPUs); on a 2-GPU SLI system both PCIe and NV-SLI saturate at ~1 MB.

THE finding for this run: "the NUMA effects appear quite significant with 5 GPUs, implying that there may be a congestion when forming a NCCL ring among 5 GPUs. One should avoid adopting 5 GPUs in their application setup." The box under assessment has exactly 5 GPUs, so this is a direct, primary-sourced warning that a 5-rank NCCL ring is a pathological configuration on PCIe.

Their discussion adds two structural cautions: NVLink's advantage is bandwidth, not latency, over PCIe; and a faster interconnect alone "can hardly become significant enough to lift the entire application's speedup" unless the CPU-centric master-slave programming model is replaced or communication control moves to the GPU (e.g. NVSHMEM). NVLink also introduces its own overheads (enable/disable peer access, routing, NCCL initialization). `review`
- [[multi-gpu-gemm-algorithm]] — Multi-GPU GEMM Algorithm — Choi & Stegailov (MMST 2022, CCIS 1750, pp. 281-292) benchmark their own multi-GPU SGEMM algorithm on two nodes: 8x A100 80GB over NVLink 3.0 through NVSwitch (cHARISMa), and 4x AMD Radeon RX 6900 XT 16GB over PCIe 4.0 under ROCm 5.2.1. Table 1 gives the parameters that matter: A100 peak FP32 19.5 TFLOPS (real 18.4), FP32 tensor-core peak 156 TFLOPS (real 124), peak GPU-GPU 300 GB/s (real 281); RX 6900 XT peak FP32 22.5 TFLOPS (real 21.4), peak GPU-GPU 32 GB/s (real 25.5).

On the NVLink node the algorithm reaches >80% of peak for 3-8 GPUs at tile size 1024 with N=90000, with strong scaling close to ideal, and beats cuBLAS-XT. Turning tensor cores on raises absolute TFLOPS but drops efficiency below 40% at 8 GPUs. On the PCIe 4.0 node the ported HIP version is much worse: efficiency below 40% for 3 and 4 GPUs, and a single-GPU SGEMM launch only reaches 8 TFLOPS of the 22.5 peak, with "considerable delays between data transfers". The authors conclude the GPU "is a consumer GPU, not a server grade GPU" and struggles "to perform computations and data transfers simultaneously", so "the multi-GPU performance of the algorithm considered can not be transferred easily to the consumer-grade GPU systems."

This is the closest published analogue to a PCIe-attached consumer/mining-class multi-GPU node, and its verdict is a warning: PCIe-attached consumer GPUs lose ~60% of efficiency in multi-GPU GEMM because compute/transfer overlap fails. Note the scale gap - 32 GB/s PCIe 4.0 here versus ~1 GB/s Gen2 x4 on the CMP box - so the effect should be at least as severe, not less. `review`
- [[quantifying-interconnect-energy-efficiency-on-perlmutter-pjbit-measurements-of-n]] — Quantifying Interconnect Energy Efficiency on Perlmutter: pJ/bit Measurements of NVLink, PCIe, Slingshot NICs, and Rosetta Switches — Zhao, Williams, Antepara, Oliker, Austin & Wright (LBNL, eScholarship qt67z0d2wn, 2026-04-28) measure dynamic energy per bit for the interconnects on Perlmutter (HPE Cray EX, A100 GPUs, NERSC) with a software-centric microbenchmark methodology that correlates controlled stress on one component with job- and rack-level power telemetry, separating dynamic energy from constant-power overhead.

Headline pJ/bit figures: NVLink 70.9 +/- 1.0 pJ/bit, PCIe (host<->GPU) 167.9 +/- 10.5 pJ/bit, NIC+PCIe (host<->Cassini NIC) 4.1 +/- 3.2 pJ/bit, and the GPU DMA engine ~48 pJ/bit at 93.55 GB/s. Moving a bit over PCIe into/out of the GPU costs ~2.4x what it costs over NVLink - the quantitative version of "PCIe is the expensive path".

The mechanism they identify is the "Active Tax": waking an A100's command processor, memory fabric and PCIe PHY from deep idle to P0 costs roughly 50 W of uncore power. Their H2D run shows GPU0 drawing 55.00 W at 24.94 GB/s (of a 32 GB/s PCIe Gen4 peak), and D2H 55.50 W at 24.53 GB/s - the transfer is dominated by the wake-up cost, not the payload. The host->NIC path bypasses the GPU entirely and therefore avoids the tax, which is why NIC+PCIe comes out at 4.1 pJ/bit.

Relevance to the CMP box: every collective on a no-P2P fabric stages through host RAM, so each GPU pays the PCIe path energy (167.9 pJ/bit) AND the ~50 W uncore wake-up tax on every transfer. On a card capped at 250 W, a 50 W uncore tax is 20% of the power budget spent moving bits rather than doing math - a real efficiency argument for large-batch, low-frequency communication (DDP with big global batches and gradient accumulation) and against per-layer all-to-alls. Caveat: measured on A100/PCIe Gen4 with full telemetry; the CMP 170HX has weaker telemetry and a Gen2 x4 link, so absolute pJ/bit will differ, but the direction (PCIe >> NVLink, plus a fixed wake-up tax) is architectural. `review`

---
*Auto-generated by hyperresearch. Do not edit manually.*
