---
title: Specifications - CMP 170HX Documentation
id: specifications-cmp-170hx-documentation
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- cmp170hx
- specifications
- tflops
- pcie-link
- community-reported
created: '2026-10-27T10:52:48.031818Z'
updated: '2026-10-27T19:01:48.596009Z'
source: https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/
source_domain: it-still-boots.github.io
fetched_at: '2026-10-27T10:52:47.411751Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: docs
deprecated: false
summary: 'it-still-boots.github.io CMP 170HX Documentation — Hardware Specifications
  page (851 words), a single-page reference paraphrasing the 170th-Street gitbook
  ''Full Specifications'' fact sheet. Compute: 70 SM / 4,480 CUDA / 280 tensor cores,
  base 1,140 MHz, boost 1,410 MHz, L1 192 KB/SM, L2 32 MB, 54.2 billion transistors,
  826 mm^2, TSMC N7. Memory per SKU: 8 GB (0x20C2) unlocks to 64 GB on a 4096-bit
  bus at ~1.49 TB/s; 10 GB (0x2082) unlocks to 40 GB on a 5120-bit bus at ~1.56 TB/s;
  ECC disabled; resizable BAR limited to 64 MiB stock. Performance table is the load-bearing
  part for the query''s ''~810 TFLOPS bf16'' premise and it disagrees with the repo:
  FP32 ~0.41 TFLOPS stock -> ~12.99 TFLOPS unlocked, FP64 ~0.19 -> ~6.44, FP16 ~49.05
  TFLOPS in BOTH stock and unlocked columns, INT8 ~1.64 -> ~48.03. So this source
  puts per-card FP16/BF16 tensor throughput at ~49 TFLOPS (245 TFLOPS for five cards),
  roughly a quarter of the repo''s claimed 162 TFLOPS/GPU, and it does not model any
  compute unlock for FP16 at all. Connectivity: mechanical x16, stock electrical Gen1
  x4 (~1 GB/s), Gen1 x16 (~4 GB/s) after the capacitor mod, Gen2 after the software
  unlock with width still set by the capacitors, Gen3 work-in-progress; NVLink connectors
  present but fuse-disabled. Stock lockdown is two independent layers — firmware Gen
  lock plus missing AC coupling capacitors on 12 of 16 lanes — and neither mod replaces
  the other. Power 250 W default, ~300 W software max, 30-40 W idle. No NVENC/NVDEC,
  no display outputs, CUDA CC 8.0, OpenCL 3.0. Waterblock family Bykski N-TESLA-A100-X-V2.'
utility_score: 16.0
---

# Hardware Specifications[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#hardware-specifications "Permanent link")
Single-page reference for the NVIDIA CMP 170HX as used in the lab. Compute counts follow Full Specifications (**70 SM / 4480 CUDA / 280 tensor** for both SKUs). Clocks, power, APIs, memory geometry, and bandwidth come from the same Full Specs / locked fact sheet.
Primary sources: [Full Specifications](https://170th-street.gitbook.io/hx/hardware/full-specifications), [What is the CMP 170HX?](https://170th-street.gitbook.io/hx/introduction/what-is-the-cmp-170hx), [GitHub mirror](https://github.com/amoghmunikote/170th-Street).
## General[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#general "Permanent link")  
| Property  | Value  |  
| --- | --- |  
| Model  | NVIDIA CMP 170HX  |  
| Architecture  | Ampere (GA100)  |  
| Process  | TSMC 7 nm N7 FinFET  |  
| Transistors  | 54.2 billion  |  
| Die size  | 826 mm²  |  
| Form factor  | Dual-slot PCIe add-in card  |  
| Cooling (stock)  | Passive server heatsink  |  
| Display outputs  | None  |  
| Release  | ~2021-09-01  |  
| Original MSRP  | ~$4299 USD  |  
## SKU identification[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#sku-identification "Permanent link")  
| Property  | 8 GB SKU  | 10 GB SKU  |  
| --- | --- | --- |  
| Stock capacity  | 8 GB HBM2e  | 10 GB HBM2e  |  
| PCI Device ID  | `0x20C2`  | `0x2082`  |  
| PCI Subsystem ID  | `0x1585`  | `0x1557`  |  
| Die  | GA100-105F-A1  | GA100-105A-A1  |  
| Unlocked capacity  | **64 GB**  | **40 GB**  |  
| Memory bus  | 4096-bit  | 5120-bit  |  
| Bandwidth (approx.)  | ~1.49 TB/s  | ~1.56 TB/s  |  
Capacity is per SKU
Consensus-Protocol and cmpunlocker docs agree: 8 GB stock unlocks to 64 GB; 10 GB stock unlocks to 40 GB. An 80 GB profile on 10 GB cards has been built, tested, and rejected as unstable.
## Compute[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#compute "Permanent link")
Full Specs compute (both SKUs):  
| Property  | Value  |  
| --- | --- |  
| Streaming Multiprocessors  | 70  |  
| CUDA cores (64 per SM)  | 4480  |  
| Tensor cores (4 per SM)  | 280  |  
| Base clock  | 1140 MHz  |  
| Boost clock  | 1410 MHz  |  
| L1 cache  | 192 KB per SM  |  
| L2 cache  | 32 MB (32768 KB)  |  
| TMUs  | 280  |  
| ROPs  | 128  |  
## Memory[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#memory "Permanent link")  
| Property  | 8 GB SKU  | 10 GB SKU  |  
| --- | --- | --- |  
| Type  | HBM2e  | HBM2e  |  
| Stock capacity  | 8 GB  | 10 GB  |  
| Unlocked capacity  | 64 GB  | 40 GB  |  
| Bus width  | 4096-bit  | 5120-bit  |  
| Memory clock (Full Specs)  | 1458 MHz  | 1215 MHz  |  
| Bandwidth (theoretical)  | ~1.49 TB/s  | ~1.56 TB/s  |  
| ECC  | Disabled  | Disabled  |  
| Resizable BAR  | Present, limited to **64 MiB** stock  | Present, limited to **64 MiB** stock  |  
Locked bandwidth figures: **~1.49 TB/s** (8 GB SKU) and **~1.56 TB/s** (10 GB SKU).
## Performance (theoretical vs unlocked)[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#performance-theoretical-vs-unlocked "Permanent link")
From the Full Specs performance table (paraphrased; see source for raw rows):  
| Metric  | Stock (throttled)  | Unlocked (class)  |  
| --- | --- | --- |  
| FP32  | ~0.41 TFLOPS  | ~12.99 TFLOPS  |  
| FP64  | ~0.19 TFLOPS  | ~6.44 TFLOPS  |  
| FP16  | ~49.05 TFLOPS  | ~49.05 TFLOPS  |  
| INT8  | ~1.64 TFLOPS  | ~48.03 TFLOPS  |  
Stock FP32 FMA is about **0.39–0.41 TFLOPS**. Unlocked FP32 is about **12–13 TFLOPS**.
PCB note: reference designators match the A100 40GB PCIe family closely enough that A100 schematics and waterblocks are useful (see [Teardown](https://it-still-boots.github.io/cmp-170hx-docs/hardware/teardown/) and [Water Cooling](https://it-still-boots.github.io/cmp-170hx-docs/hardware/water-cooling/)).
## Connectivity[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#connectivity "Permanent link")  
| Property  | Value  |  
| --- | --- |  
| Mechanical slot  | PCIe x16  |  
| Stock electrical link  | Gen1 x4 (~1 GB/s)  |  
| After capacitor mod  | Gen1 x16 (~4 GB/s)  |  
| After software Gen unlock  | Gen2 (width still whatever the caps allow)  |  
| Gen3  | Work in progress  |  
| NVLink  | Connectors present, fuse-disabled  |  
| Power connector  | 1× 8-pin CPU/EPS-style via adapter  |  
Stock lockdown is two independent layers: firmware Gen lock, plus missing AC coupling capacitors on **12 of 16** lanes. Capacitor mod and Gen2 unlock each solve one layer. Neither replaces the other. Procedure: [PCIe capacitor mod](https://it-still-boots.github.io/cmp-170hx-docs/modifications/pcie-capacitor-mod/).
## Power[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#power "Permanent link")  
| Property  | Value  |  
| --- | --- |  
| TDP (default)  | 250 W  |  
| Max power limit (software)  | ~300 W  |  
| Idle (typical)  | ~30–40 W  |  
See [Power](https://it-still-boots.github.io/cmp-170hx-docs/hardware/power/).
## API support[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#api-support "Permanent link")  
| API  | Supported  |  
| --- | --- |  
| CUDA  | Yes (Compute Capability **8.0**)  |  
| OpenCL  | Yes (3.0 per Full Specs)  |  
| DirectX  | No  |  
| Vulkan  | No  |  
| OpenGL  | No  |  
| NVENC / NVDEC  | No  |  
## Form factor and cooling implications[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#form-factor-and-cooling-implications "Permanent link")
  * Dual-slot passive cooler sized for rack airflow
  * No display outs: always keep a second GPU, iGPU, or BMC console for the host
  * Waterblock family: Bykski **N-TESLA-A100-X-V2** (see [Water Cooling](https://it-still-boots.github.io/cmp-170hx-docs/hardware/water-cooling/))


## Related pages[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#related-pages "Permanent link")


## References[¶](https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/#references "Permanent link")
  * <https://170th-street.gitbook.io/hx/hardware/full-specifications>
  * <https://170th-street.gitbook.io/hx/introduction/what-is-the-cmp-170hx>
  * <https://github.com/amoghmunikote/170th-Street>
  * <https://github.com/Consensus-Protocol/cmp170hx/wiki>


Back to top 
