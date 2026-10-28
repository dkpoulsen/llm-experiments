---
title: README.md
id: readmemd
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- nccl
- all-reduce
- interconnect
- measured
- primary-source
created: '2026-10-27T11:14:01.436992Z'
updated: '2026-10-27T19:01:51.599236Z'
source: https://raw.githubusercontent.com/PixelML/club-170hx/main/results/2026-10-01-cmp170hx-3card-p2p-plx/README.md
source_domain: raw.githubusercontent.com
fetched_at: '2026-10-27T11:14:01.436401Z'
fetch_provider: builtin
status: review
type: note
tier: practitioner
content_type: docs
deprecated: false
summary: 'PixelML/club-170hx results/2026-10-01-cmp170hx-3card-p2p-plx/README.md (505
  words) — the sanitized evidence receipt behind the NCCL SHM all-reduce numbers cited
  in BENCHMARKS.md, and therefore the primary measurement underpinning any DDP gradient-reduction
  arithmetic on this hardware. Setup: 3x CMP 170HX memory-unlocked to 65,536 MiB each,
  all at Gen2 x16 on the same CPU socket, GPU0/GPU1 behind one PLX PEX 8747 and GPU2
  behind a second, ACS at firmware default with request/completion redirect enabled
  (ACSCtl 0x001d), 250 W cap, dual Xeon E5-2686 v4, Proxmox VE 9, kernel 7.0.2, bare
  metal, driver 615.71.09 with upstream cmpunlocker @ 88e39ce plus rebar-serialize.patch
  and explicitly NO BAR1-P2P patches. All-reduce measured with torch 2.13.0+cu130
  / NCCL 2.29.7 under torchrun, bf16 sum, 5 warm-ups, median of 20, bus bandwidth
  = algbw x 2(n-1)/n. Results: canAccessPeer/enablePeerAccess returns 0 / CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
  (217) on all 6 pairs and nvidia-smi topo -p2p r reports GNS, yet an alias-proof
  cuMemcpyPeer passes on all 6 pairs because the driver stages through host memory.
  cuMemcpyPeer at 256 MiB runs 6.21-6.30 GB/s against a single-card D2H reference
  of 6.68 GB/s; latency 22 us / 40 us / 343 us at 4 KiB / 64 KiB / 1 MiB versus D2H
  9.5 / 18.4 / 165 us. Same-switch and cross-switch differ by <=1%, i.e. nothing takes
  the switch-local path. NCCL all-reduce bus bandwidth at 256 MiB: 3.69 GB/s (same-switch
  pair), 3.85 GB/s (cross-switch pair), 4.24 GB/s (all three); at 8 KiB the times
  are 144 / 120 / 134 us. NCCL transport is SHM/direct for every pair. Limitations
  stated by the authors: true BAR1 P2P was untested here so this is the no-P2P baseline,
  and there is one probe run per pair with no cross-run error bars.'
---

*Suggested by [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] — The evidence receipt behind the cited NCCL SHM all-reduce measurement (3.69/3.85 GB/s bus bandwidth) — the load-bearing number for DDP gradient-reduction feasibility*

# Interconnect — GPU-to-GPU copies and all-reduce on 3× CMP 170HX behind PLX switches, no P2P

Status: measured
Date: 2026-10-01

Notebook: [`notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb`](../../notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb)

## Hardware

- Cards: 3 × CMP 170HX (GA100, SM80), memory-unlocked to 65,536 MiB each; anonymous labels GPU0, GPU1, GPU2
- Topology / PCIe links: all three at Gen2 x16, same CPU socket; GPU0 and GPU1 behind one PLX PEX 8747 switch, GPU2 behind a second PEX 8747 (`nvidia-smi topo -m`: GPU0–GPU1 PIX, GPU0/1–GPU2 PHB)
- ACS: switch downstream ports at firmware default, request/completion redirect enabled (ACSCtl 0x001d)
- Power limit: 250 W (card default); copies are link-bound
- Cooling: forced airflow; cores 42–48 °C during the runs

## Software

- Host: dual Xeon E5-2686 v4, Proxmox VE 9 (Debian 13), kernel 7.0.2, bare metal (no passthrough); `intel_iommu=on iommu=pt pci=realloc`; BIOS Above 4G on, MMIO high 1024G
- Driver: NVIDIA open kernel modules 615.71.09, host-side memory unlock from upstream cmpunlocker @ 88e39ce, plus [`patches/rebar-serialize.patch`](patches/rebar-serialize.patch); **no BAR1-P2P patches**
- Copy tools: CUDA driver API via ctypes (`libcuda`, CUDA 13.4 UMD), no toolkit
- All-reduce: `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905` (torch 2.13.0+cu130, NCCL 2.29.7), `torchrun`

## Commands

```text
python3 tools/p2p_probe.py   > p2p-copy.json
python3 tools/p2p_latency.py > p2p-latency.json
docker run --rm --gpus '"device=0,1"' --ipc=host -v $PWD/tools:/t --entrypoint torchrun \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 --nproc-per-node 2 /t/nccl_allreduce.py
```

## Method

- Peer access: `cuDeviceCanAccessPeer` and `cuCtxEnablePeerAccess` per ordered pair.
- Alias-proof copy: destination pre-filled with `0xFE` (absent from the source pattern), `cuMemcpyPeer` 256 MiB, every 4 KiB-th byte compared with the source.
- Bandwidth: 256 MiB × 10 `cuMemcpyPeer`; reference D2H+H2D serial loop through one pinned buffer.
- Latency: `cuMemcpyPeer` synchronised on both contexts, 4 KiB–256 MiB, median of 30 (≤16 MiB) or 8 copies; single-card D2H reference.
- All-reduce: bf16 sum, 5 warm-ups, median of 20, bus bandwidth = algbw × 2(n−1)/n.

## Results (measured)

| Measurement | Value |
|---|---|
| `canAccessPeer` / `enablePeerAccess` | 0 / `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` (217) on all 6 pairs; `nvidia-smi topo -p2p r` = GNS |
| Alias-proof copy | pass on all 6 pairs (driver stages through host memory) |
| `cuMemcpyPeer`, 256 MiB | 6.21–6.30 GB/s (single-card D2H: 6.68 GB/s) |
| `cuMemcpyPeer`, 4 KiB / 64 KiB / 1 MiB | 22 µs / 40 µs / 343 µs (D2H: 9.5 / 18.4 / 165 µs) |
| Same switch vs cross switch | no difference (≤1 %) — nothing takes the switch-local path |
| NCCL all-reduce 256 MiB, busbw | 3.69 GB/s same-switch pair, 3.85 GB/s cross-switch pair, 4.24 GB/s all three |
| NCCL all-reduce 8 KiB | 144 µs / 120 µs / 134 µs |
| NCCL transport | SHM/direct for every pair |

## Files

- `p2p-copy.json`, `p2p-latency.json`, `nccl-allreduce.jsonl`, `environment.json` — receipts
- `tools/` — the three measurement scripts and the notebook builder
- `patches/rebar-serialize.patch` — serializes the unlock's BAR1 resize across GPUs (needed on this host; see the notebook)

## Limitations

- True BAR1 P2P (community fork patches) was **untested** here; this is the no-P2P baseline for that comparison.
- One probe run per pair; no cross-run error bars.