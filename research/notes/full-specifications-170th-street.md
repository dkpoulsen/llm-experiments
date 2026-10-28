---
title: Full Specifications | 170th Street
id: full-specifications-170th-street
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- cmp170hx
- specifications
- tflops
- pcie-link
- primary-source
created: '2026-10-27T11:09:29.217141Z'
updated: '2026-10-27T19:01:51.080057Z'
source: https://170th-street.gitbook.io/hx/hardware/full-specifications
source_domain: 170th-street.gitbook.io
fetched_at: '2026-10-27T11:09:28.593929Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: docs
deprecated: false
summary: '170th Street gitbook ''Full Specifications'' page (381 words, last updated
  ~2 months before fetch) — the primary spec sheet that the it-still-boots specifications
  page paraphrases, and the source of the corpus''s lowest per-card TFLOPS figures.
  General: GA100-105F-A1 (8 GB) / GA100-105A-A1 (10 GB), Ampere, TSMC 7 nm N7, 54.2
  billion transistors, 826 mm^2, released 2021-09-01 at ~$4,299 MSRP, dual-slot passive,
  no display outputs. Compute: 70 SM, 4,480 CUDA cores (64/SM), 280 third-generation
  tensor cores (4/SM), 1,140 MHz base / 1,410 MHz boost, 192 KB L1 per SM, 32,768
  KB L2. Memory: HBM2e, 8 GB / 10 GB stock, 64 GB / 40 GB unlocked, 4,096-bit / 5,120-bit
  bus, 1,458 MHz / 1,215 MHz, 1.49 TB/s / 1.56 TB/s theoretical, ECC disabled, resizable
  BAR present but limited to 64 MiB. Performance table (Original vs Unlocked): FP64
  0.19 -> 6.44 TFLOPS, FP32 0.41 -> 12.99 TFLOPS, FP16 49.05 -> 49.05 TFLOPS (unchanged),
  INT64 2.52 -> 2.52, INT32 12.99 -> 12.99, INT16 11.94 -> 11.94, INT8 1.64 -> 48.03.
  Connectivity: physical x16 slot, electrical Gen1 x4 firmware-locked, ~1 GB/s, NVLink
  connectors present but fuse-disabled, 1x 8-pin CPU power connector via adapter.
  Power: TDP default 250 W, ~300 W software max, ~30-40 W idle. API: CUDA CC 8.0,
  OpenCL 3.0, no DirectX/Vulkan/OpenGL/NVENC/NVDEC. Notes section confirms the 8 GB
  and 10 GB variants are genuine hardware binning variants with different memory buses,
  not firmware differences. CAVEAT: this is a spec-sheet table, not a measurement
  — its 49.05 TFLOPS FP16 figure conflicts with the club''s measured 155.7 TFLOPS
  fp16 on a healthy card, so treat it as the weakest tier for compute claims.'
---

*Suggested by [[specifications-cmp-170hx-documentation]] — Named as the primary source for the compute/memory/performance tables the it-still-boots specs page paraphrases*

For the complete documentation index, see [llms.txt](https://170th-street.gitbook.io/hx/llms.txt). This page is also available as [Markdown](https://170th-street.gitbook.io/hx/hardware/full-specifications.md).
This page documents every known specification of the NVIDIA CMP 170HX in one place. Where figures differ between sources, the most accurate measured values are used.
**General**
Property
Value
Model
NVIDIA CMP 170HX
GPU Die
GA100-105F-A1 (8GB) / GA100-105A-A1 (10GB)
Architecture
Ampere
Manufacturing Process
TSMC 7nm N7 FinFET
Transistors
54.2 billion
Die Size
826 mm²
**PCI Device ID**
**0x20C2 (8GB) / 0x2082 (10GB)**
**PCI Subsystem ID**
**0x1585 (8GB) / 0x1557 (10GB)**
Release Date
September 1, 2021
Original MSRP
$4,299 USD
Form Factor
Dual-slot PCIe add-in card
Cooling
Passive
Display Outputs
**Compute**
Property
Value
Streaming Multiprocessors (SM)
CUDA Cores
4,480 (64 per SM)
Tensor Cores (3rd Generation)
280 (4 per SM)
Texture Units (TMUs)
Render Output Units (ROPs)
Base Clock
1140 MHz
Boost Clock
1410 MHz
L1 Cache
192 KB per SM
L2 Cache
32,768 KB (32 MB)
**Memory**
Property
Value
Memory Type
HBM2e
Memory Capacity
8 GB / 10 GB
Unlocked Memory Capacity
64GB (8GB) / 40GB (10GB)
Memory Bus Width
4,096-bit (8GB) / 5,120-bit (10GB)
Memory Clock
1,458 MHz (8GB) / 1215 MHz (10GB)
Memory Bandwidth (theoretical)
1.49TB/s (8GB) / 1.56TB/s (10GB)
Disabled
Resizable BAR
Present but limited to 64 MiB
**Performance (Theoretical vs Actual)**
Metric
Original
Unlocked
0.19 TLFOPS/s
6.44 TFLOPS/s
0.41 TLFOPS/s
12.99 TFLOPS/s
49.05 TFLOPS/s
49.05 TFLOPS/s
INT64
2.52 TFLOPS/s
2.52 TFLOPS/s
INT32
12.99 TFLOPS/s
12.99 TFLOPS/s
INT16
11.94 TFLOPS/s
11.94 TFLOPS/s
1.64 TFLOPS/s
48.03 TFLOPS/s
**Connectivity**
Property
Value
PCIe Interface (physical)
x16 slot
PCIe Interface (electrical)
Gen 1 x4 (firmware locked)
PCIe Bandwidth
~1 GB/s
NVLink
Physical connectors present, disabled via fuse
Power Connector
1× 8-pin CPU power connector (via adapter)
**Power**
Property
Value
TDP (default)
Max Power Limit (software)
Idle Power (typical)
~30–40W
**API Support**
API
Supported
✅ Yes (Compute Capability 8.0)
OpenCL
✅ Yes (3.0)
DirectX
Vulkan
OpenGL
NVENC / NVDEC
**Notes on Specification Discrepancies**
The CMP 170HX exists in multiple hardware configurations: **8GB, and 10GB variants are confirmed real production cards** , not database errors. These are not merely software differences controlled by firmware; they represent genuine hardware binning variants with different memory configurations:
  * **8GB variant:** 4,096-bit memory bus
  * **10GB variant:** 5,120-bit memory bus


[ PreviousHistory & Timeline ](https://170th-street.gitbook.io/hx/introduction/history-and-timeline)[ NextTeardown Guide ](https://170th-street.gitbook.io/hx/hardware/teardown-guide)
Last updated 2 months ago
