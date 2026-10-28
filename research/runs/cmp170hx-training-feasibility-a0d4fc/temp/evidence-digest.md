# Evidence digest — cmp170hx-training-feasibility-a0d4fc

Assembled from 175 claims files (1,371 claims) across 225 notes. 118 claims survive the filter (`evidence_type` empirical/statistical OR a non-empty `numbers` array), ranked by evidence tier and numeric density, capped at 2 claims per source per item so no single note dominates. Every bullet carries its verbatim `quoted_support` and source note id — this is the primary evidence index the draft sub-orchestrators read instead of fetcher summaries.

**Evidence tiers used throughout:** `[T1]` measured on this exact box (user-reported, no external validation) · `[T2]` community report or measured on adjacent hardware · `[T3]` paper-measured · `[T4]` arithmetic estimate.

## Atomic item 0 — negotiated link width/speed and how much x4 vs x16 changes every communication number

- **With BAR1 P2P enabled on a CMP 170HX, NCCL moves 2-GPU all-reduce traffic over P2P/CUMEM instead of shared memory and same-switch bus bandwidth rises from 3.69 to 4.83 GB/s at 256 MiB.** (empirical)
  > "With P2P on, NCCL moves 2-GPU traffic over `P2P/CUMEM` instead of shared memory: same-switch bus bandwidth rises from 3.7 to 4.8 GB/s."
  - numbers: 2 GPU SHM 256 MiB: 72,736 us, busbw 3.69 GB/s; 2 GPU P2P 256 MiB: 55,536 us, busbw 4.83 GB/s; +31% busbw at N=2; 2 GPU SHM 1 MiB: 393 us, busbw 2.67 GB/s; 2 GPU P2P 1 MiB: 317 us, busbw 3.31 GB/s
  - scope: 4x CMP 170HX (GA100, SM80), 65,536 MiB each, BAR1 64 GiB each after the memory unlock, Gen2 x16 per card, two PLX PEX 8747 switches on CPU2, dual Xeon E5-2686 v4, 180 W, driver 615.71.09 + upstream cmpunlocker 88e39ce + rebar-serialize.patch + four BAR1-P2P pa
  [2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]

- **At three ranks the BAR1-P2P all-reduce is SLOWER than host-staged SHM at 256 MiB (3.86 vs 4.24 GB/s busbw), so the P2P bandwidth gain inverts as rank count rises.** (empirical)
  > "| 3 GPU SHM | 256 MiB | 84389 | 4.24 | ... | 3 GPU P2P | 256 MiB | 92732 | 3.86 |"
  - numbers: 3 GPU SHM: 84,389 us, busbw 4.24 GB/s; 3 GPU P2P: 92,732 us, busbw 3.86 GB/s; -9% busbw at N=3; 3 GPU SHM 1 MiB: 458 us, 3.05 GB/s; 3 GPU P2P 1 MiB: 483 us, 2.90 GB/s
  - scope: same 4x CMP 170HX rig, Gen2 x16, three GPUs with one pair cross-switch, 256 MiB bf16 all-reduce
  [2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]

- **PCIe bandwidth on the stock card is locked to Gen1 x4, measured at 0.85 GB/s host-to-device.** (empirical)
  > "CMP 170HX's PCIe bandwidth is restricted to PCIe 1.0 x4, at 0.85 GB/s. Thus, if the application performs frequent data exchanges between CPU and GPU, or between different GPUs, it would be too slow to be acceptable"
  - numbers: 0.85 GB/s enqueueWriteBuffer; 0.84 GB/s enqueueReadBuffer; 10.43 GB/s memcpy from mapped ptr
  - scope: Stock card, clpeak transfer benchmarks, single GPU
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **The PCIe lockdown has two independent layers: an on-chip fuse pinning the device to Gen1, and a board-level x4 limit created by omitting AC-coupling capacitors on the extra lanes.** (empirical)
  > "There are in fact two layers of PCIe bandwidth lockdown. The first layer is the on-chip fuse, hardware or VBIOS lockdown (who knows) to PCIe Gen 1. The next is the circuit board level lockdown to x4 from x16 by omitting the AC coupling capacitors on some PCIe lanes to force a link degradation."
  - numbers: LnkCap 2.5GT/s Width x16; LnkSta 2.5GT/s Width x4 (downgraded); Max Gen 2; Current Gen 1; Device Max 1; Host Max 3
  - scope: lspci -vv and nvidia-smi link reporting on a stock card
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **The cards are limited to PCIe Gen2 x4 in this deployment.** (empirical)
  > "While it lacks display outputs and is limited to PCIe Gen2 x4, it offers 64 GB of HBM2e memory once the VRAM is unlocked."
  - numbers: PCIe Gen2 x4; 64 GB
  - scope: 4x CMP 170HX in one box
  [allover326deepseek-v4-cmp170hx-deepwiki]

- **The intra-node to inter-node bandwidth ratio in a DGX-A100 cluster is about 2 (600 GB/s intra-node bidirectional per GPU vs 400 GB/s inter-node per node).** (empirical)
  > "High-performance DGX-A100 nodes offer 600GB/s intra-node bidirectional bandwidth per GPU and provide 400GB/s inter-node bidirectional bandwidth per node. The bandwidth ratio between intra-node and inter-node measures at 2 in our test-bed."
  - numbers: 600 GB/s intra-node per GPU; 400 GB/s inter-node per node; ratio 2
  - scope: DGX-A100 nodes with NVLink intra-node and 4x InfiniBand HDR inter-node
  [amsp-reducing-communication-overhead-of]

- **With no BAR1-P2P driver patches, cuDeviceCanAccessPeer is 0 on all six ordered pairs of three cards, so the driver stages every GPU-to-GPU copy through host memory at 6.21-6.30 GB/s at 256 MiB (single-card D2H 6.68 GB/s) and 22 us at 4 KiB, identical same-switch and cross-switch.** (empirical)
  > "`cuDeviceCanAccessPeer` is 0 on all six ordered pairs, so the driver stages every GPU-to-GPU copy through host memory: 6.21–6.30 GB/s at 256 MiB (single-card D2H 6.68 GB/s), 22 µs at 4 KiB, identical same-switch and cross-switch."
  - numbers: cuDeviceCanAccessPeer = 0 on 6/6 pairs; 6.21-6.30 GB/s at 256 MiB; 6.68 GB/s single-card D2H; 22 us at 4 KiB
  - scope: Three memory-unlocked cards at Gen2 x16, two behind one PLX PEX 8747 switch and one behind a second, bare metal, no BAR1-P2P patches; measured 2026-10-01
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

- **NCCL falls back to SHM on this fabric and a two-GPU all-reduce reaches 3.69 GB/s (same switch) and 3.85 GB/s (cross switch) bus bandwidth at 256 MiB, with 120-145 us at 8 KiB.** (empirical)
  > "NCCL picks SHM: two-GPU all-reduce 3.69 GB/s (same switch) and 3.85 GB/s (cross switch) bus bandwidth at 256 MiB, 120–145 µs at 8 KiB."
  - numbers: 3.69 GB/s same-switch all-reduce; 3.85 GB/s cross-switch all-reduce; 120-145 us at 8 KiB
  - scope: Three cards at Gen2 x16 behind PLX switches, no BAR1-P2P, NCCL SHM transport, 256 MiB and 8 KiB message sizes
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

- **PCIe link width is x16 or x4 depending on the board, set by whether AC-coupling capacitors on lanes 4-15 are populated; the club's own cards train x16, and the x4 variant is the community-reported norm.** (empirical)
  > "| PCIe width  | x16 or x4 depending on the board (AC-coupling capacitors on lanes 4–15 populated or not)  | community-reported; our cards train x16  |"
  - numbers: lanes 4-15; x16 vs x4
  - scope: Board-level AC-coupling capacitor population on lanes 4-15; club cards measured x16, x4 variant community-reported
  [club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]

- **With the plain unlock there is no peer access and every GPU-to-GPU copy stages through host memory at 6.21-6.30 GB/s at 256 MiB and 22 us at 4 KiB, identical for same-switch and cross-switch pairs.** (empirical)
  > "| No peer access with the plain unlock; copies stage through host memory  | 6.21–6.30 GB/s at 256 MiB, 22 µs at 4 KiB; same-switch = cross-switch  |"
  - numbers: 6.21-6.30 GB/s at 256 MiB; 22 us at 4 KiB
  - scope: Four-card PLX host, Gen2 x16, plain unlock without the BAR1-P2P patch set
  [club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]

- **With the cmpunlocker link patch all four cards on the club's 4-card PLX host train PCIe Gen2 x16, and Gen3 is unreachable because the OPT_DISABLE_GEN3_SPEED fuse reads 1 on every card checked.** (empirical)
  > "**Update 2026-10-02 (measured):** the Gen1 ceiling below is the _stock_ driver's. With the cmpunlocker link patch all four cards on the 4-card PLX host train **Gen2 x16** ; Gen3 is blocked by the `OPT_DISABLE_GEN3_SPEED` fuse, which reads 1 on every card we checked."
  - numbers: Gen2 x16 negotiated; OPT_DISABLE_GEN3_SPEED fuse = 1 on 4/4 cards; 5 GT/s
  - scope: 4-card PLX host (dual-socket Broadwell Xeon E5-2686 v4, two PLX PEX 8747 switches, two cards per switch); cmpunlocker link patch applied; measured 2026-10-01/02
  [club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github]

- **On the stock driver every card advertises LnkCap 2.5GT/s x16 (PCIe Gen1), and a card in a bifurcated slot training at x8 is at that slot's ceiling rather than a riser or passthrough fault.** (empirical)
  > "Host-side `lspci -vv` on the hypervisor shows every CMP 170HX card advertising `LnkCap: Speed 2.5GT/s, Width x16`. PCIe Gen1 is the card's advertised maximum, so a Gen1 link seen in a guest is a card/vBIOS capability, not a hypervisor passthrough bug. One card trains at x8 instead of the advertised x16."
  - numbers: LnkCap 2.5 GT/s; Width x16; one card at x8
  - scope: Hypervisor-side lspci -vv on the four-card test node, stock driver, measured 2026-09-02
  [club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github]

- **On an 8-card PCIe Gen2 x4 box, an 8-stage pipeline's step time follows ~32 ms fixed + 2.1 ms per verified token, and GLM-5.3-Flash NVFP4 PP8 with DFlash2 k=7 delivered 123 tok/s single-stream on code, 37 tok/s on prose, and 276 tok/s aggregate at 16 concurrent streams.** (empirical)
  > "measured with per-stage CUDA events on their 8-card PCIe Gen2 x4 box, step time `≈ 32 ms fixed + 2.1 ms × verified tokens per step` ... single-stream decode 123 tok/s (code), 37 tok/s (prose) ... Steady-state aggregate at 16 concurrent streams: 276 tok/s."
  - numbers: 32 ms fixed per step; 2.1 ms per verified token; 123 tok/s code; 37 tok/s prose; 276 tok/s at 16 streams; 8 cards
  - scope: 8x CMP 170HX at PCIe Gen2 x4, NVFP4 + DFlash2, promisezackr/glm53-flash-170hx-pp8 commit 90ec72e (Apache-2.0, derivative of vLLM); community-reported, not reproduced by the club
  [club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]

- **The overlap is engineered around a two-tier fabric: intra-node NVLink at 160 GB/s versus inter-node IB at 50 GB/s (3.2x), with each token dispatched to at most 4 nodes, averaging 3.2 experts per node, and only 20 SMs dedicated to saturating both links.** (empirical)
  > "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s). To effectively leverage the different bandwidths of IB and NVLink, we limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic. ... only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink."
  - numbers: NVLink 160 GB/s; IB 50 GB/s; 3.2x; at most 4 nodes per token; 3.2 experts per node; up to 13 experts at same cost
  - scope: DeepSeek-V3 H800 cluster: NVLink intra-node, IB inter-node; custom cross-node all-to-all kernels
  [deepseek-v3-technical-report]

## Atomic item 1 — PWRBRK# power-brake trap and its detection

- **Power draw is workload-class dependent: FMA-locked FP32/FP64 workloads draw only 60-75 W, while integer or memory-streaming workloads reach 160-180 W against a 250 W TDP.** (empirical)
  > "The power consumption of gpu_burn for FP32 and FP64 is around 60 watts. When running mixed-precision FP32/FP16 workloads ... the power consumption is around 75 watts with spikes to 100+ watts. ... A typical integer workload such as a password cracker like hashcat can push the power consumption to 160+ watts."
  - numbers: 60 W gpu_burn FP32/FP64; 75 W mixed precision; 160+ W hashcat; 180 W FluidX3D no-FMA; 250 W TDP
  - scope: nvidia-smi power readings under gpu_burn, hashcat, STREAM-like and FluidX3D workloads
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **PWRBRK# on pin B30 can force the cards into a low-power state on certain motherboards, reducing performance by about 4x.** (empirical)
  > "A critical deployment factor is the PWRBRK# (pin B30) diagnostic; on certain motherboards, this signal can force the cards into a low-power state, reducing performance by ~4x."
  - numbers: pin B30; ~4x reduction
  - scope: CMP 170HX on motherboards that assert PWRBRK# on edge pin B30
  [allover326deepseek-v4-cmp170hx-deepwiki]

- **A healthy unlocked card measures 155.7 TFLOPS fp16, while a card held by the PWRBRK# power brake on PCIe sideband pin B30 measures 39.3 TFLOPS and is capped near 88 W of a 250 W budget with clocks around 1,140 MHz.** (empirical)
  > "**Power-brake (`PWRBRK#`, PCIe sideband pin B30)** is a separate, board-level hazard: some workstation boards assert it on x16 slots, permanently capping a card near 88 W of a 250 W budget and dropping clocks to about 1140 MHz (measured fp16 throughput 39.3 vs 155.7 TFLOPS healthy)."
  - numbers: 155.7 TFLOPS fp16 healthy; 39.3 TFLOPS under PWRBRK#; ~88 W cap; 250 W budget; ~1,140 MHz
  - scope: CMP 170HX, unlocked, healthy vs PWRBRK#-asserted; some workstation boards assert PWRBRK# on x16 slots; check with nvidia-smi -q | grep -A1 'HW Power Brake Slowdown'
  [lessonsmd]

## Atomic item 2 — which training topologies are viable on a no-P2P PCIe fabric

- **On a simulated 10 Gbps commodity link, the decomposed 8-bit all-reduce is 3.7x faster than the baseline at a 32 MB message (432.36 ms vs 1,603.78 ms).** (empirical)
  > "Under 10Gbps, the speedup increases to 3.7× (432.36 ms vs 1603.78 ms), confirming that our method delivers substantial communication savings across diverse bandwidth settings."
  - numbers: 10 Gbps; 3.7x; 432.36 ms; 1603.78 ms; 32 MB
  - scope: TP=8, DP=8, 32 MB message, 10 Gbps simulated
  [agoq-activation-and-gradient-quantization-for]

- **Collective communication latency grows with communication scale, so NCCL's effective bandwidth degrades from 8x1 to 8x64 GPU meshes.** (empirical)
  > "Secondly, the latency of collective communication operations demonstrates a positive correlation with communication scale and illustrated in Figure 2(c). Figure 3 further illustrates a reduction in the effective bandwidth of communication operations utilized by ZeRO, scaling from 8 GPUs to 512 GPUs."
  - numbers: message sizes 1M to 1G bytes; 8x1 to 8x64 GPU meshes; 800 Gbps IB HDR
  - scope: NCCL AllGather/ReduceScatter/AllReduce/Broadcast, message sizes 1M-1G bytes, 4x800 Gbps InfiniBand HDR
  [amsp-reducing-communication-overhead-of]

- **Measured end-to-end: on a slow wire (a job forced to cross between NVLink quads over PCIe), training throughput drops 3-5x versus the same job on NVSwitch or inside one quad, and the penalty is strategy-dependent — FSDP falls hardest because it communicates every layer, DDP holds up best because it communicates once per step.** (empirical)
  > "The across-quads case is the cost of getting placement wrong. Throughput drops by 3 to 5x. ... FSDP falls the hardest, because it communicates every layer, and on a slow wire, those per-layer transfers can't hide behind compute. DDP holds up best because it communicates just once per step."
  - numbers: 3-5x throughput loss; 4x H200; Mistral-7B
  - scope: Mistral-7B fine-tune on 4x H200, three wirings (NVSwitch, full NVL quad, cross-quad PCIe), DDP/ZeRO-1/2/3/FSDP, HuggingFace Accelerate defaults, BF16
  [behind-the-scenes-of-distributed-training-and-why-your-gpu-wiring-matters-as-muc]

- **Measured all-reduce bandwidth on identical H200 silicon: NVLink versus forced PCIe differs by ~10-11x, and the same four cards reach 322 GB/s inside an NVLink-bridged quad but only ~35 GB/s when the job spans two quads over PCIe — a ~9x difference decided purely by placement.** (empirical)
  > "The wire alone is worth about 10-11x. ... The same four cards, in the same node, run at 322 GB/s inside one quad and about 35 GB/s across two, a roughly 9x difference decided by nothing but which GPUs the job landed on."
  - numbers: 10-11x NVLink vs PCIe; 322 GB/s inside quad; ~35 GB/s across quads; ~9x placement difference
  - scope: H200 SXM (NVSwitch) and H200 NVL (two bridged quads of four) nodes; all-reduce microbenchmarks at 2/4/8 GPUs
  [behind-the-scenes-of-distributed-training-and-why-your-gpu-wiring-matters-as-muc]

- **Tensor parallelism is link-bound on this fabric: when one card's link retrained from x8 to x1, TP4 fell from 70.5 to 14.4 tok/s under an unchanged protocol while PP4 held 60.8 tok/s on the same box — about 4.2x more link-tolerant.** (empirical)
  > "TP is link-bound on this fabric: with no NVLink, every all-reduce crosses PCIe and the topology runs at the width of its worst rank. When one card's link retrained from x8 to x1, TP4 fell from 70.5 to 14.4 tok/s under an unchanged protocol while PP4 held 60.8 tok/s on the same box — about 4.2x more link-tolerant."
  - numbers: TP4 70.5 -> 14.4 tok/s; PP4 held 60.8 tok/s; 4.2x link tolerance; x8 -> x1 retrain
  - scope: Four-card CMP 170HX, GLM-5.3-Flash, one card retrained x8 -> x1, unchanged protocol, same box
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

- **Tensor parallelism performs one all-reduce per layer per token, making every layer boundary a synchronisation point; on a Gen1/no-P2P fabric the cost is latency-bound rather than bandwidth-bound because the round trip through host memory costs the same whether the tensor is large or small.** (empirical)
  > "Every layer boundary is a synchronization point: all four ranks block until the slowest one's reduce lands. On a Gen1 fabric with no P2P, that is latency-bound, not bandwidth-bound — the round trip through host memory costs the same whether the tensor is large or small."
  - numbers: one all-reduce per layer per token
  - scope: Four-card CMP 170HX, Gen1 x8/x16, no P2P, host-memory-staged collectives; inference (decode) workload
  [club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]

- **On the club's four-card box, GLM-5.3-Flash TP4 with MTP k=3 decoded at 60.4 tok/s at c=1 (median of 5, peak 77.9) and ~38 tok/s aggregate at c=8, while the patched PP4 build reached 87.55 tok/s at c=1 — inverting the earlier verdict so that PP4 became the recipe of record and TP4 was superseded.** (empirical)
  > "Measured on GLM-5.3-Flash AWQ W4A16, TP4, MTP k=3 ...: **60.4 tok/s c=1 decode (median of 5, peak 77.9)** , **~38 tok/s aggregate decode at c=8** ... With the 24-patch port landed, PP4 + native MTP k=3 measures **87.55 tok/s** c=1 (temp 0, median of 3, clean workload) and **67.91 tok/s** (temp 0.7 + `ignore_eos`, median of 5) against TP4's 60.4"
  - numbers: TP4 60.4 tok/s c=1 (peak 77.9); TP4 ~38 tok/s aggregate c=8; PP4 87.55 tok/s c=1; PP4 67.91 tok/s (temp 0.7); 24 patches
  - scope: GLM-5.3-Flash AWQ W4A16, four-card CMP 170HX, image ghcr.io/pixelml/...vllm-glm53-sm80-20260903; PP4 numbers measured 2026-09-05 after a 24-patch port
  [club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-170hx-github-2]

- **The worked example is Llama-2-13B on 8x80GB H100 under FSDP, i.e. a sharded-data-parallel topology.** (empirical)
  > "End-to-end per-GPU memory com-parison when training Llama-2-13B on 8×80G H100 using FSDP."
  - numbers: Llama-2-13B; 8x H100; 80 GB; FSDP
  - scope: Llama-2-13B, 8x H100 80GB, FSDP
  [coat-compressing-optimizer-states-and-activation-for-memory-efficient-fp8-traini]

- **On 4x CMP 170HX, pipeline parallelism beats tensor parallelism by 6.6x on prefill (5,321 vs 801 tok/s at 77k context) and roughly 2x on aggregate decode.** (empirical)
  > "On these cards PP beats TP by 6.6x on prefill (5,321 vs 801 tok/s at 77k context) and roughly 2x on aggregate decode."
  - numbers: 5,321 tok/s PP prefill; 801 tok/s TP prefill; 6.6x; 77k context
  - scope: DeepSeek-V4-Flash-0731 on 4x CMP 170HX, PCIe Gen2 x4, no P2P, vLLM with the repo's patches
  [deepseek-v4-cmp170hxsettingsmd-at-317dac12bbd7f401f0538c76161bc7186086fb65-allov]

- **DeepSeek-V4-Flash (284B total / ~13B active, FP4 routed experts, mostly FP8 non-expert weights) was served on eight MI250 GCDs of ~64 GiB HBM each as a single TP8/EP1 instance with no routed-expert all-to-all — i.e. the accepted topology avoids expert all-to-all entirely and puts everything on tensor parallelism.** (empirical)
  > "Runtime topology: One TP8/EP1 instance, no routed-expert A2A; native decode graph tiers 1/2/4/8/16/32/64. ... DeepSeek-V4-Flash; 284B total parameters, approximately 13B active parameters, FP4 routed experts, and mostly FP8 non-expert weights."
  - numbers: 284B total; ~13B active; 8 MI250 GCDs; ~64 GiB HBM per GCD; TP8/EP1
  - scope: 8x MI250 GCDs (gfx90a/CDNA2), ~64 GiB HBM per GCD, SGLang, DeepSeek-V4-Flash checkpoint (48 safetensors shards)
  [deepseek-v4-flash-on-amd-gfx90a]

- **Moving the same Qwen2.5-3B / 4-rank job from NVLink A100 to PCIe L4 raises ZeRO-2's NCCL share of kernel time from 17.3% to 50.2% and FSDP2 reshard's from 27.7% to 51.1%, i.e. communication becomes roughly half of kernel time once the wire is PCIe.** (empirical)
  > "On PCIe, the NCCL share rises sharply for most configs (FSDP2 no-reshard 26.1%->46.7%, ZeRO-2 17.3%->50.2%)"
  - numbers: ZeRO-2 NCCL share 17.3% (A100) -> 50.2% (L4); FSDP2 no-reshard 26.1% -> 46.7%; FSDP2 reshard 27.7% -> 51.1%; ZeRO-3 42.7% -> 47.5%
  - scope: Qwen2.5-3B bf16, micro-batch 1, seq 2048, 4 ranks, 15 steps median, n=1, defaults-vs-defaults; 4x A100 40GB NVLink vs 4x L4 24GB PCIe
  [deepspeed-zero-vs-pytorch-fsdp2-measured-18x-on-nvlink-barely-finishing-on-pci-2]

- **On the PCIe L4 run ZeRO-2 still reaches 1,923 tok/s and beats FSDP2 no-reshard's 1,761 tok/s, so a 50% NCCL kernel-time share does not translate into a >50% throughput penalty.** (empirical)
  > "Note: the NCCL share is the ratio of NCCL kernel duration reported by the profiler; communication overlaps with compute, so it does not map one-to-one onto pure communication-wait time in wall-clock terms."
  - numbers: ZeRO-2 1,923 tok/s / 4,260.5 ms; FSDP2 no-reshard 1,761 tok/s / 4,651.6 ms; FSDP2 reshard 1,218 tok/s / 6,723.9 ms; ZeRO-3 2,290 tok/s / 3,577.2 ms
  - scope: Same matrix; author caveat that NCCL share 'does not map one-to-one onto pure communication-wait time'
  [deepspeed-zero-vs-pytorch-fsdp2-measured-18x-on-nvlink-barely-finishing-on-pci-2]

- **Delta N-gram's own writer training used expert parallelism 4 alongside data parallelism 4 on four H200 GPUs, so the published recipe depends on the exact topology the CMP 170HX's no-P2P fabric rules out.** (empirical)
  > "We train on four H200 GPUs with expert parallelism 4 and data parallelism 4. Each update uses 12 complete trajectories and backpropagates through the whole write chain of each."
  - numbers: 4 H200 GPUs; EP=4; DP=4; 12 trajectories per update; AdamW lr 3e-4; 100M target tokens
  - scope: Native-model writer training: four H200 GPUs, EP=4, DP=4, 12 complete trajectories per update
  [delta-n-gram-forward-only-continual-writing]

- **On 4x RTX 4090 with all 12 GPU pairs reporting P2P DISABLED, FSDP SHARD_GRAD_OP (the ZeRO-2-equivalent sharding mode) runs at 4.13 samples/s versus DDP's 6.90, a 1.67x step-time ratio (+67% overhead) that exceeds cc-3's 50% overturn threshold.** (empirical)
  > "FSDP Comparison | Strategy | Samples/sec | Memory/GPU | Forward | Backward | Optimizer | DDP 6.90 21.1 GB 0.033s 0.423s 0.128s | SHARD_GRAD_OP 4.13 15.8 GB 0.323s 0.643s 0.009s"
  - numbers: DDP 6.90 samples/s, 21.1 GB/GPU; SHARD_GRAD_OP 4.13 samples/s, 15.8 GB/GPU; FULL_SHARD 2.88, NO_SHARD 3.88; step-time ratio 1.67x; +67% overhead
  - scope: 4x RTX 4090 24GB, PCIe 4.0 x16, topology PHB/NODE, P2P disabled, batch 1/GPU; model and sequence length NOT stated
  [distributed-trainingreadmemd-at-main-cortwavedistributed-training-github]

- **A separate 'bad topology' DDP run on the same 4090 class shows 2 GPUs delivering 0.29x and 4 GPUs 0.91x of single-GPU throughput, i.e. DDP itself can lose more than half its throughput on a badly-wired PCIe host.** (empirical)
  > "1 GPU 15.38 samples/s 1.00x 100% / 2 GPUs 4.50 0.29x 15% / 4 GPUs 14.01 0.91x 23%"
  - numbers: 15.38 samples/s at 1 GPU; 4.50 at 2 GPUs (0.29x); 14.01 at 4 GPUs (0.91x); step time 0.130 s -> 0.526 s -> 0.571 s
  - scope: Vast.ai RTX 4090, batch 2, sdpa attention, rtx_4090_bad_topology.csv
  [distributed-trainingreadmemd-at-main-cortwavedistributed-training-github]

- **At 64 MB the hierarchical allreduce beats Spectrum MPI, Open MPI+UCX, Open MPI+HPC-X, MVAPICH2-GDR and NCCL by 11.47x, 14.33x, 1.09x, 1.26x and 1.25x respectively.** (empirical)
  > "The proposed algorithm outperforms Spectrum MPI, Open MPI + UCX, Open MPI + HPC-X, MVAPICH2-GDR, and NCCL with a speedup of 11.47x, 14.33x, 1.09x, 1.26x, 1.25x, respectively, for 64MB messages."
  - numbers: 11.47x; 14.33x; 1.09x; 1.26x; 1.25x; 64 MB
  - scope: 4x V100, 64 MB message
  [efficient-multi-path-nvlinkpcie-aware-ucx-based-collective-communication-for-dee]

- **The hierarchical allreduce outperforms NCCL for all message sizes greater than 8 MB.** (empirical)
  > "It also outperforms Open MPI + HPC-X for all message sizes and NCCL for message sizes greater than 8MB."
  - numbers: 8 MB
  - scope: 4x V100 single node
  [efficient-multi-path-nvlinkpcie-aware-ucx-based-collective-communication-for-dee]

- **Pipeline parallelism with CPU offloading can train large models on consumer GPUs using no GPU peer-to-peer communication at all, relying entirely on PCIe host-to-device transfers, and still reach 76% or more of the throughput of the same frameworks on NVLink-equipped A800s.** (empirical)
  > "Remarkably, RoundPipe achieves highly competitive performance without utilizing any GPU peer-to-peer communication (NVLink), relying entirely on PCIe host-to-device transfers under the Computation Dispatch Paradigm... RoundPipe still consistently achieves over 76% of the throughput of existing SOTA frameworks running on the A800 across all models."
  - numbers: 76%+ of A800 throughput; PCIe 4.0 32 GB/s; NVLink 3.0 200 GB/s
  - scope: 8x RTX 4090 (24 GB, PCIe 4.0 32 GB/s) vs 8x A800 SXM (80 GB, NVLink 3.0 200 GB/s); 1.7B-235B models; FP16 with full activation recomputation
  [efficient-training-on-multiple-consumer-gpus-with]

- **Data-parallel offloading approaches such as DeepSpeed ZeRO spend about 70% of training time on communication on consumer GPU servers, because they must exchange the full parameter set between every GPU for each forward and backward computation.** (empirical)
  > "A previous study shows that DeepSpeed spends about 70% of training time on communication on consumer GPU servers, limiting the efficiency of multi-GPU scaling on cost-effective hardware."
  - numbers: ~70% of training time on communication
  - scope: consumer GPU servers with PCIe interconnect; ZeRO-Offload / ZeRO-Infinity style parameter partitioning
  [efficient-training-on-multiple-consumer-gpus-with]

- **The study's test platform was a single card over Oculink x4 on a Ryzen 7 7840HS mini-PC, so it measures one card in isolation and says nothing about multi-GPU collectives.** (empirical)
  > "Due to experimental constraints, the GPU was tested via an Oculink-connected PCIe interface on a custom-built mini-PC platform."
  - numbers: 1 GPU; Oculink x4; Ryzen 7 7840HS; driver 570.86.16; CUDA 12.8
  - scope: Single-GPU test bench, Debian 12.9, kernel 6.1.0-31, driver 570.86.16, CUDA 12.8
  [exploration-of-cryptocurrency-mining-specific-gpus-in-ai]

## Atomic item 3 — the DDP overhead model and whether it holds at 1M/2M-token batches

- **Pairing the ~100k-token microbatch with gradient_accumulation=10 (effective global batch ~1M tokens) and DDP.no_sync() so the all-reduce fires only on the 10th accumulation step cut wall-clock from under 18 hours to 13.2 hours.** (empirical)
  > "Zach used a microbatch size of roughly 100k tokens paired with gradient_accumulation=10, forming an effective batch size of approximately 1M tokens ... overall training time naturally plummeted from 18 hours down to 13.2 hours"
  - numbers: gradient_accumulation=10; ~1M-token effective batch; 18 h -> 13.2 h; ~2.26 h of sync -> ~0.23 h
  - scope: Same box; effective batch ~1M tokens; 10B-token dataset
  [a-13-hour-training-experiment-on-a-pc-workstation-why-optimization-begins-with-m]

- **Large minibatches on ImageNet cause optimization difficulty rather than poor generalization: once the difficulty is addressed, there is no loss of accuracy up to minibatch 8,192.** (empirical)
  > "In this paper, we empirically show that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are addressed the trained networks exhibit good generalization. Specifically, we show no loss of accuracy when training with large minibatch sizes up to 8192 images."
  - numbers: minibatch 8,192; no accuracy loss
  - scope: ImageNet, ResNet-50, SGD with momentum, minibatch sizes up to 8,192 images
  [accurate-large-minibatch-sgd]

- **Gradient memory drops 75% versus COAT and activation memory a further 30%, by storing gradients in 8-bit for local accumulation.** (empirical)
  > "compared to COAT, we reduce activation memory by an additional 30% and gradient memory by 75%"
  - numbers: 75%; 30%
  - scope: OLMo-1B, comparison against COAT
  [agoq-activation-and-gradient-quantization-for]

- **ZeRO-1 MFU on LLaMA-7B falls from 63% at 8 GPUs to 36% at 1024 GPUs at the same batch size.** (empirical)
  > "In our experiments, training a LLaMA-7B model on 8 GPUs using ZeRO-1 achieves a model FLOPs utilization (MFU) of 63%, but scaling to 1024 GPUs with the same batch size results in a significant performance reduction, with the MFU dropping to 36%."
  - numbers: 63% MFU at 8 GPUs; 36% MFU at 1024 GPUs
  - scope: LLaMA-7B, ZeRO-1, fixed batch size, 8 to 1024 GPUs
  [amsp-reducing-communication-overhead-of]

- **Holding a 4M-token global batch constant while scaling from 8 to 512 GPUs shrinks per-GPU micro-batches and destroys the compute-to-communication ratio.** (empirical)
  > "As depicted in 2 (b), the computation time of the LLaMA-7B model training linearly reduces from 8 GPUs to 512 GPUs while maintaining a consistent 4M batch size. This reduction adversely affects the compute-to-communication ratio, leading to a communication bottleneck."
  - numbers: 4M-token global batch; 8 to 512 GPUs
  - scope: LLaMA-7B, 4M-token global batch, 8 to 512 GPUs
  [amsp-reducing-communication-overhead-of]

- **The gradient noise scale - the signal-to-noise ratio of the gradient across training examples - approximately predicts the largest useful batch size across supervised learning, RL and generative tasks.** (empirical)
  > "we demonstrate that a simple and easy-to-measure statistic called the gradient noise scale predicts the largest useful batch size across many domains and applications"
  - numbers: 8 tasks
  - scope: 8 tasks: MNIST, SVHN, CIFAR-10, ImageNet, Billion Word, Atari, Dota, SVHN autoencoders
  [an-empirical-model-of-large-batch-training]

- **COAT enables doubling the batch size in distributed training settings, which is directly relevant to amortising communication overhead.** (empirical)
  > "COAT enables efficient full-parameter training of large models on fewer GPUs, and facilitates doubling the batch size in distributed training settings"
  - numbers: 2x batch size
  - scope: distributed training on H100
  [coat-compressing-optimizer-states-and-activation-for-memory-efficient-fp8-traini]

- **Cramming uses sequence length 128 with packed sequences, yielding micro-batches of 64-96 on an 11 GB RTX 2080 Ti that are accumulated into larger batches, and single-epoch training is feasible at this budget.** (empirical)
  > "We pack tokenized data into randomized sequences of length 128 and separate unrelated fragments by <sep>... For the targeted compute settings, this sequence length results in micro-batch sizes of 64 to 96 for most variations of the base BERT architecture on the gtx2080ti, which we will accumulate into larger batch sizes."
  - numbers: sequence length 128; micro-batch 64-96; 11 GB card
  - scope: Cramming implementation, BERT-base variants, packed 128-token sequences
  [cramming-training-a-language-model-on-a]

- **The critical batch size measured directly on OLMo 1B and 7B starts near zero at initialization, grows rapidly within the first 50k tokens, and plateaus around 4,096 documents of 4,096 tokens each (about 16.8M tokens).** (empirical)
  > "The CBS increases over training in a similar way for both model sizes: the CBS starts near 0, grows rapidly within the first 50k tokens, and then plateaus around 4096."
  - numbers: CBS plateaus ~4,096 documents; 4,096 tokens per document; ~16.8M tokens; first 50k tokens
  - scope: OLMo 1B and OLMo 7B pretraining, 4,096-token documents, CBS measured by branched training from checkpoints, up to 600B-token budget
  [critical-batch-size-revisited-a-simple-empirical]

- **Batch-size warmup (start small, double as the CBS grows) reaches slightly better final loss than the small-batch control while using 43% fewer gradient steps, whereas a fixed 4,096-document batch from step 0 uses 75% fewer steps but ends with worse loss.** (empirical)
  > "Overall, batch size warmup slightly exceeds the small-batch control in loss while using 43% fewer gradient steps. In contrast, the large-batch control uses 75% fewer gradient steps, but its final loss degrades compared to the small-batch control."
  - numbers: 43% fewer gradient steps; 75% fewer gradient steps; loss margin 0.0166 after pretraining; loss margin 0.0053 after mid-training
  - scope: OLMo 1B pretraining plus mid-training; small-batch control B=1024, large-batch control B=4096, warmup run doubling at CBS thresholds
  [critical-batch-size-revisited-a-simple-empirical]

- **Sparse-attention backward in V4 uses atomicAdd to accumulate KV gradients, which makes training non-deterministic; the fix is per-SM accumulation buffers plus a deterministic global summation across all buffers.** (empirical)
  > "In conventional implementations of backward propagation for sparse attention, we use atomicAdd to accumulate gradients for the KV tokens. This introduces non-determinism due to the non-associativity of floating-point addition. To address this problem, we allocate separate accumulation buffers for each SM, followed by a global deterministic summation across all buffers."
  - numbers: per-SM accumulation buffers
  - scope: Section on determinism in training infrastructure
  [deepseek-v4]

- **The training recipe is 100B tokens, batch size 4M tokens, sequence length 2048 - i.e. FP8's benefits were demonstrated at very large global batch.** (empirical)
  > "We train the models for 100B tokens in total with a batch size of 4M tokens, and the input sequence length is set to 2048."
  - numbers: 100B tokens; 4M tokens; 2048
  - scope: GPT-7B/13B/175B pretraining
  [fp8-lm-training-fp8-large-language-models]

## Atomic item 4 — is expert parallelism actually ruled out

- **DeepSeek-V3 states that the communication overhead introduced by cross-node expert parallelism produces an inefficient computation-to-communication ratio of approximately 1:1 — i.e. on an NVLink+IB cluster, EP all-to-all cost roughly as much wall time as the compute it fed.** (empirical)
  > "For DeepSeek-V3, the communication overhead introduced by cross-node expert parallelism results in an inefficient computation-to-communication ratio of approximately 1:1."
  - numbers: ~1:1 compute-to-communication ratio; EP32; 2048 H800 GPUs
  - scope: DeepSeek-V3 671B MoE training on 2048 H800 GPUs, intra-node NVLink + inter-node InfiniBand, EP32
  [deepseek-v3-technical-report]

- **DualPipe hides both all-to-all and pipeline communication behind compute by splitting each chunk into attention, all-to-all dispatch, MLP and all-to-all combine and rearranging forward/backward chunks, reaching 'near-zero all-to-all communication overhead' as long as the compute-to-communication ratio is held constant.** (empirical)
  > "In this overlapping strategy, we can ensure that both all-to-all and PP communication can be fully hidden during execution. ... This overlap also ensures that, as the model further scales up, as long as we maintain a constant computation-to-communication ratio, we can still employ fine-grained experts across nodes while achieving a near-zero all-to-all communication overhead."
  - numbers: 8 PP ranks; 20 micro-batches; near-zero all-to-all overhead
  - scope: DeepSeek-V3 training; DualPipe bidirectional pipeline scheduling with 8 PP ranks and 20 micro-batches
  [deepseek-v3-technical-report]

- **Treating host DRAM as an active routing hub rather than a passive bounce buffer raises MoE inference throughput on 8x RTX 5090 by 1.24x to 1.41x per model (up to 1.46x), reaching 85-90% of an NVLink A800 + DeepEP configuration at 23.4% of the hardware cost.** (empirical)
  > "The results show that on the RTX 5090, CoMoE improves inference throughput by up to 1.46x over SGLang. Moreover, it brings TTFT close to that of an A800 system equipped with high-speed NVLink interconnects and reaches 85-90% of the throughput of the strongest A800 configuration (SGLang with DeepEP). To deliver the same serving throughput as that configuration, CoMoE requires at most 23.4% of the equivalent GPU investment."
  - numbers: 1.24x GLM; 1.29x DeepSeek-V2-Lite; 1.41x Qwen3-30B-A3B; up to 1.46x; 85-90% of A800+DeepEP; 23.4% of cost
  - scope: 8x RTX 5090 vs 8x A800 (NVLink 400 GB/s); SGLang 0.5.9 baseline; DeepEP v1.2.1 patched for SM80
  [democratizing-moe-inference-on-commodity-gpus]

- **Decomposing the All-to-All combine into one-sided token-level transfers through host staging yields a 2.46x geometric-mean speedup over NCCL All-to-All and a 1.59x speedup over the same design without the secondary optimizations, cutting average latency 48.9% and 27.5% respectively.** (empirical)
  > "Overall, the full design achieves a 2.46x geometric-mean speedup over NCCL All-to-All and a 1.59x speedup over CoMoE-raw, reducing average latency by 48.9% and 27.5%, respectively."
  - numbers: 2.46x vs NCCL All-to-All; 1.59x vs CoMoE-raw; 48.9% latency reduction; 269.4->213.0 us at 128 tokens; 457.9->399.7 us at 256 tokens
  - scope: DeepSeek-V2-Lite hidden size 2048, 6 experts per token, 1-256 token combine requests
  [democratizing-moe-inference-on-commodity-gpus]

- **A single-step host-broadcast EP collective plus DMA-engine transfers and per-sender completion flags beats NCCL by 2.00x on MoE dispatch and 1.53x on combine on PCIe consumer GPUs without P2P.** (empirical)
  > "Across two PCIe systems equipped with two recent generations of consumer GPUs, ThunderEP improves dispatch and combine communication performance over NCCL by 2.00x and 1.53x, respectively."
  - numbers: 2.00x dispatch; 1.53x combine; 2.30x/2.65x latency at 1KB-1MB dispatch; 1.91x/2.14x combine
  - scope: 6x RTX 4090 and 6x RTX 5090, no GPUDirect P2P between any pair, Qwen3-30B-A3B / GPT-OSS-20B / GPT-OSS-120B, vLLM 0.27.1 baseline, NCCL 2.30.7
  [efficient-expert-parallel-communication]

- **When top-k exceeds the expert-parallel size, an All-Gather plus Reduce-Scatter formulation outperforms two All-to-All collectives and is the default EP backend in Megatron-Core and vLLM.** (empirical)
  > "A recent study similarly reports that All-Gather followed by Reduce-Scatter outperforms two All-to-All collectives when top-k exceeds the number of GPUs. Several state-of-the-art MoE systems, including Megatron-Core and vLLM, therefore adopt an implementation of All-Gather and Reduce-Scatter as their default EP communication backend."
  - numbers: top-k > #GPUs threshold
  - scope: fine-grained MoE where each token reaches many destination GPUs; top-k >= number of EP ranks
  [efficient-expert-parallel-communication]

- **A minimal GPU-initiated RDMA path issues an 8-byte write in 0.7 microseconds and completes in 4.0 microseconds, while production libraries add up to 4.6 microseconds of issue time on top of that floor.** (empirical)
  > "A minimal GPU path issues an operation in 0.7 us and completes in 4.0 us; libraries add up to 4.6 us of issue time through queue management, memory ordering, and completion scope, and issue time scales with the SM clock."
  - numbers: 0.7 us issue; 4.0 us completion; 4.6 us library overhead; GDAKI 1.6 us; NVSHMEM public API 5.3 us
  - scope: NVIDIA H100/H200/B200/GB200 with ConnectX-7 InfiniBand; mini-gda vs NVSHMEM IBGDA, NCCL GIN, DeepEP, UCCL-EP, MSCCL++, fabric-lib
  [gpu-initiated-communication-dissecting-down-to-the-bone]

- **All-to-all traffic loses 59% of its NIC message rate at about 3,000 active connections, and Dynamically Connected transport does not remove this decline.** (empirical)
  > "At the NIC, active connections cost more when sending and receiving together: a send-only NIC keeps about 242 M msg/s through 32,768 active queues, while all-to-all traffic loses 59% of its rate near 3,000 connections, and Dynamically Connected (DC) transport does not remove this decline."
  - numbers: 59% loss; ~3,000 connections; 242 M msg/s send-only; 260 M msg/s peak
  - scope: InfiniBand NIC, 32-PE sweep, RC QPs vs DC transport
  [gpu-initiated-communication-dissecting-down-to-the-bone]

## Atomic item 5 — is tensor parallelism actually ruled out

- **NCCL all-reduce at 8 KiB is latency-dominated on this fabric: 144 us for 2 GPU SHM, 119 us for 2 GPU P2P, 134 us for 3 GPU SHM, 153 us for 3 GPU P2P, i.e. busbw of 0.06-0.08 GB/s.** (empirical)
  > "| 2 GPU SHM | 8 KiB | 144 | 0.06 | ... | 2 GPU P2P | 8 KiB | 119 | 0.07 | ... | 3 GPU SHM | 8 KiB | 134 | 0.08 | ... | 3 GPU P2P | 8 KiB | 153 | 0.07 |"
  - numbers: 144 us / 0.06 GB/s (2 GPU SHM); 119 us / 0.07 GB/s (2 GPU P2P); 134 us / 0.08 GB/s (3 GPU SHM); 153 us / 0.07 GB/s (3 GPU P2P)
  - scope: same rig, 8 KiB bf16 all-reduce
  [2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]

- **With P2P working, vLLM TP=2 prefill is about 1.8 to 1.9 times faster than without P2P, on Gen2.** (empirical)
  > "Result: vLLM TP=2 prefill is about 1.8 to 1.9 times faster than without P2P, on Gen2."
  - numbers: 1.8-1.9x prefill; Gen2
  - scope: 2x CMP 170HX, vLLM 0.30.0 TP=2, Gen2 link, W4A16 MoE model
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

- **TP2 without P2P buys single-stream latency and 2.5x the KV cache but not throughput: Qwen3.8-27B decode 54.0 tok/s at TP1 versus 72.0 at TP2, prefill 1,914 versus 1,695-1,736 tok/s at 6,603 tokens, and aggregate at c=16 546 versus 530-535.** (empirical)
  > "Decode256: TP1 54.0, TP2 72.0 (same switch) / 71.9 (cross switch), TP2 + MTP k=3 103.5 tok/s. Prefill at 6,603 tokens: 1,914 (TP1) vs 1,695–1,736 (TP2) tok/s. Aggregate at c=16: 546 (TP1), 530–535 (TP2), 385 (TP2 + MTP). TP2 buys single-stream latency and 2.5x the KV cache, not throughput."
  - numbers: TP1 54.0 tok/s decode; TP2 72.0 tok/s decode; TP2+MTP k=3 103.5 tok/s; prefill 1,914 vs 1,695-1,736; aggregate c=16 546 vs 530-535; 2.5x KV cache
  - scope: dbirks/Qwen3.8-27B-W4A16-AutoRound, club SM80 vLLM image, 180 W, Gen2 x16 behind PLX switches, no P2P (NCCL SHM), prefix cache off
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

- **P2P helps two-card tensor parallelism (Qwen3.8-27B TP2 decode +11%, +14% with MTP k=3) but hurts four-card TP behind PLX switches on Broadwell (GLM-5.3-Flash TP4 decode -20 to -26% at one user and -38 to -48% at eight users, step 19.4 -> 26.1 ms).** (empirical)
  > "| P2P helps two-card TP  | Qwen3.8-27B TP2 decode +11% (+14% with MTP k=3)  | same  | | P2P hurts four-card TP behind PLX on Broadwell  | GLM-5.3-Flash TP4 decode −20 to −26% (1 user), −38 to −48% (8 users); step 19.4 → 26.1 ms."
  - numbers: TP2 +11% decode; TP2 +14% with MTP k=3; TP4 -20 to -26% at 1 user; TP4 -38 to -48% at 8 users; step 19.4 -> 26.1 ms
  - scope: Four-card PLX host on dual-socket Broadwell Xeon E5-2686 v4; contrast: Morrowmake measured +11% from P2P on EPYC root ports
  [club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]

- **DeepSeek-V3 was trained without tensor parallelism at all — the memory optimizations 'enabl[ed] us to train DeepSeek-V3 without using costly Tensor Parallelism (TP)' — so the lab with the strongest incentive to use TP chose EP32 + DP instead.** (empirical)
  > "Finally, we meticulously optimize the memory footprint during training, thereby enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP)."
  - numbers: EP32; no TP in training
  - scope: DeepSeek-V3 pretraining, 2048 H800 GPUs, EP32
  [deepseek-v3-technical-report]

- **Tensor parallelism is impractical on a PCIe consumer-GPU node: Megatron-TP's throughput becomes impractical under PCIe, while RoundPipe avoids TP's heavy communication overhead and therefore trains faster.** (empirical)
  > "Excluding Megatron-TP, whose throughput becomes impractical under PCIe, RoundPipe extends the maximum sequence length by 4.7 to 7.3x over the next-best baseline... RoundPipe avoids TP's heavy communication overhead, thereby achieving higher training throughput than TP."
  - numbers: 4.7-7.3x longer sequences vs next-best
  - scope: 8x RTX 4090, PCIe 4.0; Megatron-LM tensor parallelism baseline
  [efficient-training-on-multiple-consumer-gpus-with]

## Atomic item 6 — frozen Engram tables and the real 64 GB memory arithmetic

- **This deployment keeps the model's PLE / n-gram tables in host RAM (~128 GB) via VLLM_PLE_CPU_OFFLOAD=1, because leaving them on the GPU causes an OOM.** (empirical)
  > "VLLM_PLE_CPU_OFFLOAD=1 keeps the PLE / n-gram tables in host RAM (about 128 GB; budget roughly 180 GB for the container/host). Without it you get a GPU OOM."
  - numbers: ~128 GB host RAM for tables; ~180 GB container budget; 256 GB DDR4
  - scope: vLLM 0.30.0, W4A16 MoE (Qwen3.8-Flash-Next), 2x CMP 170HX, 256 GB DDR4 host RAM
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

- **For a 7B model, Adafactor state is at most about 4 bytes per parameter (versus 8 for Adam m + v) and far less for large matrices, giving roughly 28 GB of optimizer state instead of 84 GB.** (statistical)
  > "Maintaining these per-parameter second-moment estimators requires memory equal to the number of parameters. For the case of neural network weight matrices, we propose maintaining only the per-row and per-column sums of these moving averages"
  - numbers: ~4 bytes per parameter worst case; ~28 GB optimizer state for 7B; 84 GB with Adam
  - scope: Arithmetic from the paper's O(n+m) state bound applied to 7B parameters; excludes activations; assumes Adafactor quality holds at 7B
  [adafactor-adaptive-learning-rates-with-sublinear-memory-cost]

- **Stock VRAM is restricted to 8 GiB and Resizable BAR is unavailable, with a largest BAR of 64 MiB.** (empirical)
  > "Resizable BAR is unavailable, this capability is surprisingly reported, but the largest BAR size is 64 MiB."
  - numbers: 7961 MB total global mem; 64 MiB largest BAR; BAR 0 16MB; BAR 1 64MB; BAR 3 32MB
  - scope: Stock card, 2023, before the cmpunlocker VRAM unlock existed
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **The deployment runs 4x CMP 170HX with VRAM unlocked to 64 GB per card, giving 256 GB of aggregate HBM2e on the box.** (empirical)
  > "By leveraging the GA100 silicon (sm_80) and unlocking the VRAM to 64 GB per card, the system achieves the model's full 1,047,736-token context ceiling"
  - numbers: 4 cards; 64 GB per card; 1,047,736 tokens
  - scope: allover326/deepseek-v4-cmp170hx at commit 317dac, indexed 2026-08-10
  [allover326deepseek-v4-cmp170hx-deepwiki]

- **Measured memory: Mistral-7B in BF16 with Adam needs 87 GB of model state per GPU (14 GB params + 14 GB gradients + ~58 GB fp32 optimizer states) before any activation, which OOMs on an 80 GB A100 under DDP; FSDP shards it to ~22 GB estimated and ~29 GB actual peak on 4 GPUs.** (empirical)
  > "The parameters take 14 GB: 7 billion of them, two bytes each. The gradients take another 14 GB. Then we have the optimizer states ... roughly 58 GB. Add it up, and we reach 87 GB before the model has seen a single batch ... DDP would need an estimated 87 GB per GPU for the model state ... FSDP shards this across 4 GPUs, bringing the estimated requirement down to about 22 GB per GPU. The actual peak during the run was about 29 GB."
  - numbers: 87 GB model state; 14 GB params; 14 GB grads; 58 GB optimizer; 22 GB estimated FSDP; 29 GB actual peak
  - scope: Mistral-7B, BF16 mixed precision, Adam, 4x A100 80GB, HuggingFace Accelerate
  [behind-the-scenes-of-distributed-training-and-why-your-gpu-wiring-matters-as-muc]

- **Offloading a 100B-parameter Engram table entirely to host DRAM costs at most 2.8% end-to-end inference throughput on an H800, because transfer volume scales with the number of activated slots rather than total table size.** (empirical)
  > "As detailed in Table 4, offloading a 100B-parameter embedding table incurs a negligible throughput penalty, peaking at only 2.8% on the 8B backbone. This confirms that the compute intensity of early dense blocks provides a sufficient temporal window to mask the retrieval latency."
  - numbers: 100B parameters; 4B-Dense 9,031.62 -> 8,858.28 tok/s; 8B-Dense 6,315.52 -> 6,140.02 tok/s; 2.8% peak penalty; <3%
  - scope: nano-vLLM harness, NVIDIA H800, 512 sequences, sequence length Uniform(100,1024), Dense-4B and Dense-8B backbones, Engram layer at block 2, all retrievals forced over PCIe from host memory
  [conditional-memory-via-scalable-lookup]

- **Engram-27B is 26.7B total parameters with only 3.8B activated per token, of which 5.7B are Engram memory parameters, so the memory module is ~21% of total but contributes zero per-token FLOPs.** (empirical)
  > "Engram-27B is iso-parameters with MoE-27B by reallocating parameters from routed experts (72 -> 55) to a 5.7B-parameter Engram memory."
  - numbers: 26.7B total; 3.8B activated; 5.7B Engram params; 72 -> 55 routed experts; top-6 routing; 2 shared experts
  - scope: Engram-27B configuration, 262B training tokens, iso-parameter and iso-FLOPs against MoE-27B
  [conditional-memory-via-scalable-lookup]

- **Engram conditional memory is a 196B-parameter block inside the 552B-backbone model, sparsely accessed via token-based lookup — i.e. frozen/conditional memory can be a large fraction of total parameters rather than a rounding error.** (empirical)
  > "Additional architectural components include Single-Pass mHC (revised residual-stream mixing with an efficient Mega-mHC kernel), Engram conditional memory (196B parameters, sparsely accessed via token-based lookup), and DSpark speculative decoding."
  - numbers: 196B Engram parameters; 552B backbone parameters; 35.5% of backbone params
  - scope: DeepSeek-V4.1-Flash architecture; 552B backbone parameters
  [deepseek-aideepseek-v41-flash-hugging-face]

- **The memory state carries no optimizer state at all — it stores only written row ids, patches and write counts — so a frozen table costs its bf16 rows and nothing else.** (empirical)
  > "The state stores only rows that have been written: their row ids, patches Delta_j and write counts c_j, plus a block clock with the bookkeeping for lazy decay. It carries no optimizer state."
  - numbers: 0 optimizer state per frozen row
  - scope: Delta N-gram memory state design; forward-only write path
  [delta-n-gram-forward-only-continual-writing]

- **The entire trainable component in the Delta N-gram design is a 4,075,681-parameter writer; the 48-layer, 512-expert backbone, the read-out and T0 are all frozen.** (empirical)
  > "Backbone, read-out and T0 are frozen; the writer (4,075,681 parameters) is the only trained component."
  - numbers: 4,075,681 trainable params; 48 layers; 512 routed experts; 10 active per token
  - scope: Qwen3.8-Flash-Next: 48 layers, 512 routed experts with ten active per token
  [delta-n-gram-forward-only-continual-writing]

- **Engram-40B underperforms Engram-27B on HumanEval (38.4 vs 40.8), which the paper attributes to undertraining rather than to a capacity limit.** (empirical)
  > "Engram-40B shows further gains on most benchmarks but underperforms Engram-27B on code tasks (HumanEval 38.4 vs 40.8). The paper attributes this to undertraining."
  - numbers: HumanEval 38.4 vs 40.8
  - scope: Engram-40B vs Engram-27B at 262B tokens
  [dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]

- **EngramEdit demonstrates that a frozen conditional-memory table can be edited without touching the backbone, by keeping the pretrained hashed tables fixed and storing a separate cumulative update vector per edited n-gram indexed by its exact token sequence.** (empirical)
  > "Each vector is added to the embedding output whenever its n-gram is activated, while the pretrained tables remain fixed."
  - numbers: 68.5B total; 31.4B n-gram embedding params; 2x RTX PRO 6000
  - scope: LongCat-Flash-Lite 68.5B MoE with 31.4B n-gram embedding parameters, two NVIDIA RTX PRO 6000 GPUs
  [engramedit-decoupled-knowledge-updates]

- **EngramEdit achieves near-perfect editing success on CounterFact and ZsRE by updating only conditional memory, with generalization to unseen expressions and multi-hop reasoning at nearly 3x the strongest baseline's accuracy.** (empirical)
  > "By updating only conditional memory, EngramEdit achieves near-perfect editing success on both CounterFact and ZsRE... with nearly 3x the strongest baseline's accuracy under [multi-hop reasoning]."
  - numbers: 2,000 edits; batches of 100; 5,000 sequential edits; 3,000 MQuAKE cases; ~3x baseline accuracy
  - scope: 2,000 edits in batches of 100, extended to 5,000 facts; MQuAKE 3,000 cases
  [engramedit-decoupled-knowledge-updates]

- **The official Engram implementation ships no distributed training support: the only code file is a forward-pass demo whose disclaimer states it requires further optimization for production use including 'distributed training support'.** (empirical)
  > "Production Readiness: This implementation requires further optimization for actual production use (e.g., custom CUDA kernels, distributed training support)."
  - numbers: 4 commits; single demo file
  - scope: deepseek-ai/Engram repository, 4 commits, latest Jan 14 2026
  [engramengram_demo_v1py-at-main-deepseek-aiengram-github]

- **The demo Engram config allocates 646,400 table rows per n-gram order for orders 2 and 3, with 8 heads per order, giving 10,342,400 rows total at dimension 64 per head.** (empirical)
  > "engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5]) ... max_ngram_size: int = 3 ... n_embed_per_ngram: int = 512 ... n_head_per_ngram: int = 8 ... layer_ids: List[int] = field(default_factory=lambda: [1, 15])"
  - numbers: 646,400 rows per order; 2 orders; 8 heads; 10,342,400 rows total; dim 64 per head; n_embed_per_ngram 512
  - scope: engram_demo_v1.py default EngramConfig
  [engramengram_demo_v1py-at-main-deepseek-aiengram-github]

- **The card's VRAM capacity of 8 GB is the binding constraint for LLM work in the stock configuration, forcing the study down to a 1.5B model.** (empirical)
  > "Considering the nVidia CMP 170HX has only 8GB HBM2e memory, we selected the Qwen2.5-1.5B model to ensure all layers of LLM weights could be loaded into GPU VRAM."
  - numbers: 8 GB VRAM; 1.54B parameters; 28 layers
  - scope: Stock 8 GB card, llama.cpp with all layers offloaded
  [exploration-of-cryptocurrency-mining-specific-gpus-in-ai]

- **FactorEngram's memory footprint at 1B backbone scale is 2B lookup-table parameters with bigram capacity 250K and trigram capacity 750K entries; at 340M backbone scale it is 1B lookup-table parameters with 250K entries per order.** (empirical)
  > "The 340M backbone configuration uses 1B lookup-table parameters, with both bigram and trigram table capacities of 250K entries. The 1B backbone configuration uses 2B lookup-table parameters, with the bigram and trigram table capacities of 250K and 750K entries, respectively."
  - numbers: 340M backbone + 1B lookup params; 1B backbone + 2B lookup params; 250K bigram; 250K/750K trigram; 8K context; 32K vocab
  - scope: 340M and 1B Transformer backbones, 8K context, 32K vocab
  [factorengram-factorized-n-gram-memory-with-basis-level]

- **FactorEngram improves over a reproduction of Engram on all evaluation metrics at both 340M and 1B scales, with the largest gains on 8K Needle-in-a-Haystack long-context retrieval.** (empirical)
  > "At 340M, FactorEngram reduces WikiText perplexity from 23.19 to 21.29 and LAMBADA perplexity from 24.90 to 21.69, while improving average downstream accuracy by 1.76 percentage points over the Transformer. It also outperforms Engram on all evaluation metrics."
  - numbers: WikiText ppl 23.19 -> 21.29; LAMBADA ppl 24.90 -> 21.69; +1.76 pp downstream; 340M/30B tokens; 1B/120B tokens
  - scope: 340M backbone 30B tokens, 1B backbone 120B tokens, FineWeb-Edu, 8K context
  [factorengram-factorized-n-gram-memory-with-basis-level]

## Atomic item 7 — does CSA2 pay off at 32k training context

- **The evidence base is OLMo 1B and 7B at 4,096-token sequence length, so the CBS numbers are for dense models at short context and do not directly establish the CBS for a 32k-context, Engram-augmented or MoE model.** (empirical)
  > "we use the OLMo 1B and OLMo 7B models for our experiments (OLMo et al., 2025), whose pretraining data is openly available (Soldaini et al., 2024)."
  - numbers: 1B; 7B; 4,096-token documents
  - scope: Scope limitation: dense OLMo models, 4,096-token documents, no MoE, Engram or sparse attention
  [critical-batch-size-revisited-a-simple-empirical]

- **flash-attention's sparse MLA (DeepSeek V4) forward and backward kernels added in commit 940cd96 are SM100-only: the files are flash_fwd_mla_sm100.py, flash_bwd_mla_sm100.py, flash_bwd_mla_dq_dqv_sm100.py and flash_bwd_mla_dk_sm100.py.** (empirical)
  > "[Cute,Bwd,Sm100] add sparse MLA (Deepseek v4) backward kernels (#2621) ... flash_bwd_mla_sm100.py, flash_bwd_mla_dq_dqv_sm100.py, flash_bwd_mla_dk_sm100.py, flash_fwd_mla_sm100.py"
  - numbers: +5685/-1066; 12 files; sm100 only
  - scope: Dao-AILab/flash-attention commit 940cd96, 2026-06-20, +5685/-1066 across 12 files
  [cutebwdsm100-add-sparse-mla-deepseek-v4-backward-kernels-2621-dao-ailabflash-att]

- **DeepSeek-V4.1-Flash trains its sparse attention at a sequence length of 64K, not 32k, and extends context to 1M only at 34T tokens into a 45T-token pretraining run.** (empirical)
  > "DeepSeek-V4.1-Flash is trained from scratch on a multimodal corpus comprising 45T tokens, with sparse attention trained at a sequence length of 64K and context extended to 1M tokens at 34T tokens."
  - numbers: 45T tokens; 64K training sequence length; 34T tokens at extension; 1M context
  - scope: DeepSeek-V4.1-Flash pretraining recipe, 45T multimodal tokens
  [deepseek-aideepseek-v41-flash-hugging-face]

- **FP4 main KV caching (E2M1 with one E4M3 scale per 16 channels) plus the mode-sharing design brings the global KV footprint to 890 bytes per token, roughly 1/4 of DeepSeek-V4-Flash and ~437x below V4/V1.** (empirical)
  > "Combined with FP4 main KV caching (E2M1 format, one E4M3 scale per 16 channels), these designs reduce the global KV cache footprint to 890 bytes per token — roughly 1/4 of DeepSeek-V4-Flash."
  - numbers: 890 bytes/token; 1/4 of V4-Flash; 437x vs V1; E2M1 FP4; E4M3 scale per 16 channels
  - scope: DeepSeek-V4.1-Flash vs DeepSeek-V4-Flash and DeepSeek-V1; global KV cache bytes per token
  [deepseek-aideepseek-v41-flash-hugging-face]

- **DSA's dense warm-up stage freezes all model parameters except the lightning indexer and trains it with a KL-divergence loss against the L1-normalized, head-summed main attention distribution, for 1000 steps of 16 sequences of 128K tokens = 2.1B tokens at LR 1e-3.** (empirical)
  > "In this stage, we keep dense attention and freeze all model parameters except for the lightning indexer. ... For warm-up, we use a learning rate of 10^-3. We train the indexer for only 1000 steps, with each step consisting of 16 sequences of 128K tokens, resulting in a total of 2.1B tokens."
  - numbers: 1000 steps; 16 sequences/step; 128K tokens/sequence; 2.1B tokens; LR 1e-3
  - scope: DeepSeek-V3.2 continued pretraining; dense warm-up stage
  [deepseek-v32-pushing-the-frontier-of-open]

- **DSA's sparse training stage optimizes all parameters, keeps the indexer aligned to the main attention distribution but only over the selected token set, detaches the indexer input from the computational graph, and trains for 15000 steps of 480 sequences of 128K tokens = 943.7B tokens at LR 7.3e-6 with 2048 KV tokens selected per query.** (empirical)
  > "It is worth noting that we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only L_I, while the optimization of the main model is according to only the language modeling loss. In this sparse training stage, we use a learning rate of 7.3 x 10^-6, and select 2048 key-value tokens for each query token. We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens."
  - numbers: 15000 steps; 480 sequences/step; 128K tokens/sequence; 943.7B tokens; LR 7.3e-6; top-k 2048
  - scope: DeepSeek-V3.2 sparse training stage
  [deepseek-v32-pushing-the-frontier-of-open]

- **Retrieval accuracy degrades with context depth regardless of tuning, from ~100% at 150k to ~30% at 900k tokens, and the author calls this architectural rather than a tuning problem.** (empirical)
  > "None of these settings change retrieval accuracy, which degrades with depth regardless (~100% at 150k -> ~30% at 900k). That is architectural, not a tuning problem."
  - numbers: ~100% at 150k; ~30% at 900k
  - scope: DeepSeek-V4-Flash-0731 on 4x CMP 170HX, retrieval probes across depth
  [deepseek-v4-cmp170hxsettingsmd-at-317dac12bbd7f401f0538c76161bc7186086fb65-allov]

- **Prefill at 32K input tokens reaches 9,966.84 input tok/s at concurrency 16 (10,263.02 at 16K, 10,413.87 at ~8K) — long-context prefill of a CSA-family MoE degrades only modestly from 8K to 32K on this stack.** (empirical)
  > "Separate C16 prefill measurements reach 10413.87, 10263.02, and 9966.84 input tokens/s at approximately 8K, 16K, and 32K input length."
  - numbers: 10,413.87 tok/s at ~8K; 10,263.02 tok/s at 16K; 9,966.84 tok/s at 32K; C16; -4.3% from 8K to 32K
  - scope: C16 prefill, ~8K/16K/32K input length, accepted prefill configuration (reduced indexer replication, mHC reuse, paired-GCD attention, fixed-order expert reduction)
  [deepseek-v4-flash-on-amd-gfx90a]

- **DeepSeek-V4 (the CSA predecessor of CSA2) warms up with dense attention for the first 1T tokens and only introduces sparse attention at a sequence length of 64K — the opposite of V4.1-Flash's no-warmup recipe, and in both cases sparse attention is introduced at 64K, never at 32k.** (empirical)
  > "The training starts with a sequence length of 4K, and we gradually extend the training sequence length to 16K, 64K, and 1M. As for the setups of sparse attention, we first warmup the model with dense attention for the first 1T tokens, and introduce sparse attention at the sequence length of 64K and keep sparse attention during the rest of the training."
  - numbers: 1T-token dense warmup; sequence lengths 4K -> 16K -> 64K -> 1M; sparse attention introduced at 64K
  - scope: DeepSeek-V4-Flash pretraining; 45T-scale run
  [deepseek-v4]

- **V4's lightning indexer runs its attention computation in FP4 precision to accelerate the operation under extremely long contexts, and the KV cache uses a mixed format (BF16 for RoPE dimensions, FP8 for the rest) that nearly halves KV cache size versus pure BF16.** (empirical)
  > "BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision, which accelerates the attention operation under extremely long contexts."
  - numbers: FP4 indexer attention; BF16 RoPE dims; FP8 other dims; ~2x KV cache reduction vs BF16
  - scope: Section 2.5 efficiency discussion
  [deepseek-v4]

- **At a 32k training context the 16,384-candidate cap is active but only halves the positions scored by deeper indexers (16,384 of 32,768), whereas at 1M it cuts them 64-fold — so the hierarchical indexer's per-query saving is roughly 32x smaller at 32k than at 1M. The repo's phrasing 'at 32k it barely binds' is directionally right about the size of the win but wrong about the mechanism: the cap does bind at 32k, just weakly.** (statistical)
  > "For a fixed candidate-pool size, the number of positions scored per query by each subsequent indexer is bounded independently of context length. The first Full Mode layer still scans the entire causally visible range."
  - numbers: 16,384 / 32,768 = 50% at 32k (2x reduction); 16,384 / 1,048,576 = 1.6% at 1M (64x reduction); 32x difference in saving between 32k and 1M
  - scope: Arithmetic from the paper's stated 16,384 cap against 32,768- and 1,048,576-token contexts
  [deepseek-v41-flash]

- **DeepSeek-V4.1-Flash trains sparse attention from scratch at a 64K sequence length with no dense-attention warm-up, then extends to 1M at 34T tokens — so 32k is below the regime the architecture was trained in, and the 'no dense warmup needed' result is established at 64K, not 32k.** (empirical)
  > "Sparse attention is trained from scratch at a sequence length of 64K, without any dense attention warmup stages. ... We train the model from scratch with sparse attention at a sequence length of 64K and extend the sequence length to 1M at 34T tokens."
  - numbers: 64K training sequence length; 1M extension at 34T tokens; 45T total tokens; no dense warmup
  - scope: 45T-token pretraining run; sequence length 64K then 1M at 34T tokens
  [deepseek-v41-flash]

- **The cuDNN Frontend DSA reference is pure PyTorch (einsum, gather, autograd) with no CuTe-DSL, cutlass or CUDA extension, so unlike the SM90/SM100-only cuDNN DSA kernels it runs on GA100/SM80.** (empirical)
  > "Contains pure-PyTorch reference implementations for numerical verification of the cudnn-frontend DSA kernels."
  - numbers: 980 lines; 37,804 bytes; 0 CUDA kernels
  - scope: test/python/deepseek_sparse_attention/cutedsl/dsa_reference.py, 980 lines, Apache-2.0, Copyright 2026 NVIDIA
  [dsa_referencepy]

- **ref_indexer_forward materializes dense per-head scores of shape (B, S_q, H_q, S_k) in FP32 and returns (B, S_q, S_k) FP32, with no query-chunked variant; at S=32768, B=1, H_q=16 that is ~68.7 GB, already over a 64 GB CMP 170HX card.** (empirical)
  > "scores_per_head[b, q, h, k] = Q @ K^T ... scores = torch.einsum("bqhd,bkhd->bqhk", q_f, k_exp)"
  - numbers: 68.7 GB at S=32768, H_q=16; 137 GB at H_q=32; 4.29 GB for the (B,S_q,S_k) output alone
  - scope: ref_indexer_forward signature and einsum 'bqhd,bkhd->bqhk'; arithmetic at S=32768, B=1, H_q=16, FP32 (4 B/element)
  [dsa_referencepy]

## Atomic item 8 — is the ~500M dense baseline at 32k feasible at 80-120k tok/s

- **Re-enabling reserved TPCs via cmpunlocker master raises the SM count from 70 to 74 per card, which yields only about +1.8% prefill at a fixed power cap.** (empirical)
  > "Prefill is compute-bound, but both cards sit at the power cap, so +4 SM gave only about +1.8%. With the +200 offset the SMs pay off: about +7% decode and +6% prefill at the same 200 W"
  - numbers: 70 SM -> 74 SM; +1.8% prefill; +7% decode with offset
  - scope: cmpunlocker master 6c442ee versus v0.4, both cards at the power cap
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2-2]

- **cmpunlocker master 6c442ee re-enables reserved TPCs to give 74 SMs per card, whereas release v0.4 gives 70 SMs.** (empirical)
  > "Use master 6c442ee or newer: it opens RECONFIG_PLM and re-enables the reserved TPCs, 70 to 74 SM per card (check with torch multi_processor_count; dmesg shows SM-RECONFIG GPCn). v0.4 gives 70 SM."
  - numbers: 74 SM; 70 SM
  - scope: cmpunlocker master 6c442ee ('Add more SMs', PR #55) versus v0.4
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

- **A +200 MHz GPC V/F offset is stable under real vLLM load at 200 W (~1600-1650 MHz), while +250 MHz hung a card with Xid 175 within seconds.** (empirical)
  > "Under a sustained bf16 GEMM at 200 W: +0 about 1385 MHz, +200 about 1500 MHz, +250 about 1525 MHz. ... +250 hung one card within seconds under real vLLM load (Xid 175 GSP RPC timeout, then Xid 154 ...). +200 is our serving point: clean under real load, about 1600-1650 MHz at 200 W."
  - numbers: +0 -> 1385 MHz; +200 -> 1500 MHz; +250 -> 1525 MHz; 1600-1650 MHz at 200 W; Xid 175
  - scope: 6D VBIOS, nvmlDeviceSetGpcClkVfOffset, sustained bf16 GEMM and real vLLM load
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

- **FP32 FMA/MAD throughput on the stock CMP 170HX is throttled to 394 GFLOPS, roughly 1/32 of the silicon's 12.6 TFLOPS theoretical FP32.** (empirical)
  > "tests show heavy performance restrictions on FMA and MAD floating-point operations, with a FP32 FMA speed of 394 GFLOPS and a FP64 FMA speed of 182 GFLOPS. The FP32 FMA performance is 1/100 of today's high-end GPUs."
  - numbers: 394.77 GFLOPS FP32 FMA; 182.72 GFLOPS FP64 FMA; 25267.20 GFlops theoretical FP32
  - scope: Stock VBIOS/driver, clpeak standard (FMA) build
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **FP16 throughput on the stock card is ~42 TFLOPS and is unaffected by whether FMA is enabled or disabled.** (empirical)
  > "According to mixbench, FP16 performance is around ~42 TFLOPS (41869 GFLOPS), a level similar to a recent-gen low-end GPU. The use of FMA or not doesn't seem to affect the outcome."
  - numbers: 41869 GFLOPS FP16
  - scope: mixbench SYCL build, stock card, no tensor-core path exercised
  [all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te]

- **PP4 with DSpark reaches 98.1 tok/s single-stream decode versus 50.8 tok/s for plain PP4, and 90.0 versus 38.8 tok/s at 100k context.** (empirical)
  > "Decode (Single Stream) 50.8 tok/s -> 98.1 tok/s ... Decode @ 100k Context 38.8 tok/s -> 90.0 tok/s"
  - numbers: 98.1 tok/s; 50.8 tok/s; 90.0 tok/s at 100k; 38.8 tok/s at 100k
  - scope: DeepSeek-V4-Flash-0731, PP4, 4x CMP 170HX, vLLM
  [allover326deepseek-v4-cmp170hx-deepwiki]

- **ZeRO-3 MFU on LLaMA-13B collapses from 47% at 8 GPUs to 4% at 1024 GPUs.** (empirical)
  > "13B training from 8 GPUs to 1024 GPUs with ZeRO-3 leads to a substantial MFU reduction from 47% to 4%."
  - numbers: 47% -> 4% MFU
  - scope: LLaMA-13B, ZeRO-3, 8 to 1024 GPUs
  [amsp-reducing-communication-overhead-of]

- **AMSP reaches up to 52% MFU training a LLaMA-based model on 1024 GPUs, a 1.56x throughput improvement over MiCS and ZeRO++.** (empirical)
  > "Evaluations demonstrate up to 52% Model FLOPs Utilization (MFU) when training the LLaMA-based model on 1024 GPUs, resulting in a 1.56 times improvement in training throughput compared to newly proposed systems like MiCS and ZeRO++."
  - numbers: 52% MFU; 1.56x vs MiCS/ZeRO++; 1024 GPUs
  - scope: LLaMA-based model, 1024 GPUs, A800 + InfiniBand
  [amsp-reducing-communication-overhead-of]

- **BM25Chunk improves in-context learning by 11.6%, knowledge memorisation by 9.8% and context utilisation by 7.2% relative to baseline packing, without sacrificing pre-training efficiency.** (empirical)
  > "our proposed efficient retrieval-based sequence construction method, BM25Chunk, can improve in-context learning (+11.6%), knowledge memorisation (+9.8%), and context utilisation (+7.2%) abilities using causal masking and thus without [sacrificing efficiency]."
  - numbers: +11.6% ICL; +9.8% knowledge memorisation; +7.2% context utilisation
  - scope: Downstream evaluation of BM25Chunk vs MIXChunk, 1.3B models, 2K and 8K
  [analysing-the-impact-of-sequence-composition]

- **The study's scale — 1.3B parameters, 150B tokens, 2K and 8K context — sits within the range a 5x CMP 170HX box (320 GB HBM, ~810 TFLOPS bf16) can plausibly reach, making its packing results directly transferable rather than extrapolated.** (empirical)
  > "We pre-train 1.3B parameters models using context windows of 2,048 (referred to as 2K) and 8,192 (8K) tokens... We sample documents with 150B tokens from SlimPajama as the pre-training corpus."
  - numbers: 1.3B parameters; 150B tokens; 2,048 and 8,192 context; 7 SlimPajama sub-corpora
  - scope: Experimental scale of the sequence-composition study
  [analysing-the-impact-of-sequence-composition]

- **A four-card CMP 170HX rig at c=8 serves DeepSeek-V4-Flash-Vision-Exp at 220 tok/s for about $0.90 per million output tokens, versus two DGX Spark nodes at 81 tok/s for about $2.13 per million, but the Spark nodes win on energy at 3,172 versus 1,457 tok/Wh.** (statistical)
  > "At c=8, the CMP rig moves more tokens per second, but the Spark nodes move more tokens per watt-hour. ... Neither number is free of assumptions: hardware prices, electricity rate, and the three-year/50%-utilization lifetime are all inputs we chose, not measurements."
  - numbers: CMP 220 tok/s at c=8; $0.90/M tok; DGX Spark 81 tok/s; $2.13/M tok; 1,457 vs 3,172 tok/Wh; $8,300 vs $8,000 hardware
  - scope: Same checkpoint and protocol on both platforms; hardware prices, electricity rate and a three-year/50%-utilization lifetime are chosen inputs, not measurements; throughput and GPU power are measured
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

- **NVFP4 checkpoints are incompatible with SM80 because they are SM121-format weights, and a BF16 fallback of a large model on four cards OOMs above roughly 1,024 input tokens and decodes at only 0.88-0.93 tok/s.** (empirical)
  > "| GLM-5.3-Flash  | NVFP4  | 3  | ... | Not compatible (SM121 weights on SM80)  | ... | DeepSeek-V4-Flash-Vision-Exp  | FP8 → BF16 fallback · reference TP4 runtime + SM80 patches  | 4 · TP4  | ≤ ~1,024 input tokens (OOM above)  | batch 1  | 512 tokens in ~8.7–9.8 s  | 0.88–0.93 tok/s"
  - numbers: OOM above ~1,024 input tokens; 0.88-0.93 tok/s decode; 512 tokens in 8.7-9.8 s
  - scope: Three to four CMP 170HX cards, SM80; DeepSeek-V4-Flash-Vision-Exp FP8 -> BF16 fallback on the reference TP4 runtime plus SM80 patches
  [club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]

## Atomic item 9 — data pipeline requirements for run comparability

- **The paper's downstream evaluation covers in-context learning across seven text-classification datasets, knowledge memorisation via RACE/SQuAD and RAG tasks, and context utilisation via a multi-hop needle-style task.** (empirical)
  > "we report the in-context learning performance evaluated by text classification accuracy across seven datasets... knowledge memorisation ability... context utilisation ability."
  - numbers: 7 text-classification datasets
  - scope: Downstream evaluation suite
  [analysing-the-impact-of-sequence-composition]

- **Perplexity filtering is effective but deduplication does not help on the paper's clean benchmark, and the authors recommend reserving filtering for noisy datasets.** (empirical)
  > "Of the filtering approaches, we find perplexity-filtering to be effective, while deduplication does not help... Overall, in a data-constrained regime, we recommend reserving filtering for noisy datasets and using both code augmentation and repeating to increase data tokens."
  - numbers: top 25% lowest perplexity retained; 100-char dedup threshold
  - scope: Filtering comparison on OSCAR-derived data, both noisy and clean variants
  [arxiv230516264v5-cscl-28-jun-2025]

- **The writer's training corpus mixes Stack v2 Edu 20%, arXiv 25%, Project Gutenberg 20%, FineWeb-Edu 15%, CC-News 17.5% and same-event CC-News streams 2.5% across 38,566 documents.** (empirical)
  > "Raw English text and source code from six streams, mixed by target-token share: code from Stack v2 Edu (20%), arXiv papers (25%) and Project Gutenberg books (20%) from the Common Pile, FineWeb-Edu web pages (15%), CC-News articles (17.5%), and time-ordered streams of several CC-News reports on the same event (2.5%)."
  - numbers: Stack v2 Edu 20%; arXiv 25%; Gutenberg 20%; FineWeb-Edu 15%; CC-News 17.5%; event streams 2.5%
  - scope: Raw-text writer training data, six streams, target-token share
  [delta-n-gram-forward-only-continual-writing]

- **Repeating a Llama-124M FineWeb run while changing only the initialisation seed or data-order seed produces a pairwise mean absolute loss-trajectory discrepancy of 1.11e-2 and 1.62e-2 nats respectively - an independent measured small-scale noise floor in the 0.01-nat range.** (empirical)
  > "For an internal calibration, we measure run-to-run variation in our most extensively tested configuration, Llama-124M trained on FineWeb. Repeating the same configuration while changing only the initialization seed or data-order seed yields variations of 1.11x10^-2 and 1.62x10^-2, respectively, under the same metric."
  - numbers: 1.11e-2 nats (init seed); 1.62e-2 nats (data-order seed); Llama-124M; batch 128; sequence 1024
  - scope: Llama-124M on FineWeb with AdamW, batch 128, sequence length 1024; pairwise time-averaged mean absolute loss discrepancy over evaluation steps
  [effective-learning-rate-governs-loss-dynamics-in-language-model-pretraining]

- **The FineWeb-Edu classifier is a Snowflake-arctic-embed-m encoder with a single regression head trained on 450k Llama3-70B-instruct educational-quality annotations scored 0-5, with the embedding and encoder layers frozen during training.** (empirical)
  > "We added a classification head with a single regression output to Snowflake-arctic-embed and trained the model for 20 epochs with a learning rate of 3e-4. During training, the embedding and encoder layers were frozen to focus on the classification head."
  - numbers: 450,000 annotations; 20 epochs; 3e-4 learning rate; scores 0-5
  - scope: HuggingFaceTB/fineweb-edu-classifier; 450,000 samples from FineWeb; 20 epochs at LR 3e-4
  [f43939791a7ceb0b01a4937308317be5]

- **FineWeb-Edu contains 1.3 trillion GPT-2 tokens of educational web text, filtered from the ~15T-token FineWeb corpus at a classifier threshold of 3.** (empirical)
  > "FineWeb-Edu dataset consists of 1.3T tokens and 5.4T tokens (FineWeb-Edu-score-2) of educational web pages filtered from FineWeb dataset. This is the 1.3 trillion version."
  - numbers: 1.3T tokens; 5.4T tokens at threshold 2; 15T-token FineWeb source; 1.53B rows; 5.84 TB on disk
  - scope: FineWeb-Edu v1.4.0, all CC-MAIN dumps 2013-2025, GPT-2 tokenizer
  [huggingfacefwfineweb-edu-datasets-at-hugging-face]

## Ungrouped — load-bearing claims that cut across items

- **A 64 GiB BAR1 obtained from the memory unlock removes the 64 MB-BAR1 blocker that the aikitoria fork's kbusIsPcieBar1P2PMappingSupported_GH100 check could not pass, and NCCL collectives then run over BAR1 P2P without hanging.** (empirical)
  > "Each card needs a 64 GiB BAR1 for static-BAR1 P2P. ... Driver: upstream cmpunlocker + rebar-serialize.patch + the four BAR1-P2P patches from admunch888/cmpunlocker (`p2p-caps-override`, `p2p-bar1`, `p2p-skip-mailbox-preinit`, `p2p-readcap-override`), `RMForceStaticBar1=1;RMPcieP2PType=1`."
  - numbers: BAR1 64 GiB per card; 12/12 ordered pairs verified; 5.79 GB/s per pair
  - scope: CMP 170HX with cmpunlocker memory unlock (BAR1 resized to 64 GiB), static BAR1, RMForceStaticBar1=1, RMPcieP2PType=1, four BAR1-P2P patches from admunch888/cmpunlocker
  [2026-10-01-cmp170hx-4card-bar1-p2p-plx-cudaipynb]

- **The wpnews.pro reprint of the CMP 170HX dual-card writeup reproduces the original gist's benchmark table exactly, so the aggregator did not distort the numbers.** (empirical)
  > "| no P2P (L1T W4A16 reference) | ~2500 | ~2500 | ~2500 | ~2500 | | P2P, mixed VBIOS (67 + 6D), 70 SM | 4634 | 4572 | 4543 | 4486 | | P2P, both 6D, 74 SM | 4766 | 4708 | 4669 | 4599 | | + offset +200 MHz, 200 W | 5033 | 4989 | 4951 | 4881 |"
  - numbers: ~2500 tok/s no P2P; 4634 tok/s 70 SM; 4766 tok/s 74 SM; 5033 tok/s with offset
  - scope: Direct comparison of the gist (ropuls/d2978531ba3a1a183e70f62f3de2eb84) against the wpnews.pro article, both fetched 2026-10-27
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2-2]

- **The first P2P setup was not reproducible for some users, and the revised recipe pins exact commits and adds UEFI and kernel prerequisites.** (empirical)
  > "Update 2026-10-02. The initial setup here was not reproducible for some people (P2P did not come up). This is the revised setting: amogh's cmpunlocker as the base, patched selectively with three P2P patches from bayley/cmpunlocker, pinned to exact commits"
  - numbers: 2026-10-02 revision
  - scope: ROMED8-2T UEFI-only board, Proxmox VE kernel 7.0.14-pve
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2-2]

- **P2P can be re-enabled on CMP 170HX: three bayley/cmpunlocker driver patches plus the regkey ForceP2P=0x11 make cudaDeviceCanAccessPeer 1 in both directions, verified by cuMemcpyPeer and NCCL reporting 'via P2P/CUMEM'.** (empirical)
  > "What works here is a regkey plus three driver patches on top of cmpunlocker. ... Verify: cudaDeviceCanAccessPeer is 1 in both directions; cuMemcpyPeer works both ways; with NCCL_DEBUG=INFO you see Channel NN/0 : 0[0] -> 1[1] via P2P/CUMEM in both directions."
  - numbers: ForceP2P=0x11; 3 patches; cudaDeviceCanAccessPeer 1
  - scope: 2x CMP 170HX on separate root complexes, EPYC 7443 / ROMED8-2T, cmpunlocker master 6c442ee + bayley patches 0011/0013/0015 at commit 5a7bb4b, IOMMU off
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

- **By default, two cards on separate root complexes report cudaDeviceCanAccessPeer 0 and the GSP capabilities report GPU_NOT_SUPPORTED.** (empirical)
  > "Two cards on separate root complexes default to no P2P: cudaDeviceCanAccessPeer is 0, and the GSP capabilities report GPU_NOT_SUPPORTED."
  - numbers: cudaDeviceCanAccessPeer 0
  - scope: 2x CMP 170HX on separate root complexes, stock cmpunlocker install
  [2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t]

## Consensus claims (3+ independent sources)

- **[consensus]** Tensor parallelism is the worst topology on a PCIe fabric without NVLink: it performs 2 all-reduces per layer (86 collectives per forward on 43 layers), and measured PP beats TP by 6.6x on prefill and ~2-3x on aggregate throughput.  
  independence score 6.0 · tier measured · sources: club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-1, github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-l, deepseek-v4-cmp170hxsettingsmd-at-317dac12bbd7f401f0538c76161b, megatron-lm-training-multi-billion-parameter-language-models-u, deepseek-v3-technical-report, on-optimizing-the-communication-of-model-parallelism

- **[consensus]** FSDP/ZeRO-3 moves 3-Psi bytes per step against DDP's 2-Psi, i.e. 1.5x communication volume, not 2x.  
  independence score 4.0 · tier primary-derived · sources: pytorch-fsdp-experiences-on-scaling-fully-sharded-data-paralle, zero-memory-optimizations-toward-training-trillion-parameter-m, zero-extremely-efficient-collective-communication-for-giant, zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum

- **[consensus]** Mixed-precision Adam costs 16 bytes per trainable parameter: 2 (bf16 weight) + 2 (bf16 grad) + 12 (fp32 master + two fp32 moments). Frozen parameters cost only their bf16 weight.  
  independence score 5.0 · tier primary-derived · sources: published-as-a-conference-paper-at-iclr-2018, zero-memory-optimizations-toward-training-trillion-parameter-m, amsp-reducing-communication-overhead-of, behind-the-scenes-of-distributed-training-and-why-your-gpu-wir, zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum

- **[consensus]** PWRBRK# on PCIe edge pin B30 silently caps a CMP 170HX at ~88 W / 39.3 TFLOPS, is detected with `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"`, and is fixed by masking B30 or using a riser that does not route it.  
  independence score 4.0 · tier measured · sources: lessonsmd, allover326deepseek-v4-cmp170hx-deepwiki, all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te, club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github

- **[consensus]** DDP performs one gradient all-reduce per optimizer step, and gradient accumulation requires `model.no_sync()` on all but the final micro-step or DDP all-reduces on every micro-step.  
  independence score 3.0 · tier ground-truth · sources: pytorch-ddp-comm-hooks-default_hookspy-bf16fp16-compress-hooks, gradient-accumulation-with-ddp-no_sync-interface-distributed-p-2, megatron-lm-training-multi-billion-parameter-language-models-u

- **[consensus]** Gradient compression pays off more on slow fabrics, not less: measured gains are larger at 10 Gbps than at 200 Gbps, and coded approximate all-reduce wins on PCIe while losing on NVLink.  
  independence score 4.0 · tier measured · sources: powersgd-practical-low-rank, agoq-activation-and-gradient-quantization-for, communication-efficient-distributed-training, zipccl-efficient-lossless-data-compression-of

- **[consensus]** When P2P is unavailable, NCCL stages collectives through host shared memory rather than GPU-to-GPU; this is why PCIe-only all-reduce bandwidth collapses.  
  independence score 3.0 · tier ground-truth · sources: environment-variables-nccl-2323-documentation, club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github, github-mcrltccl-thunder-research-groups-collective-communicati

- **[consensus]** The CMP 170HX is GA100 (sm_80) with 64 GB HBM2e after unlock, ~1.49-1.6 TB/s HBM bandwidth, 250 W stock cap, no NVLink, no ECC, and no FP8 tensor cores.  
  independence score 5.0 · tier measured · sources: specifications-cmp-170hx-documentation, club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github, github-consensus-protocolcmp170hx-cmp-170hx-technical-wiki-sil, all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te, nvidia-a100-tensor-core-gpu-data-sheet-jun21-1

- **[consensus]** Dataset sizes: FineWeb-Edu is 1.3T tokens; FineMath-4+ is 9.6B tokens; Stack-Edu is ~125B tokens across 15 languages.  
  independence score 5.0 · tier ground-truth · sources: huggingfacefwfineweb-edu-datasets-at-hugging-face, huggingfacetbfinemath-datasets-at-hugging-face-2, huggingfacetbstack-edu-datasets-at-hugging-face, smollm2-when-smol-goes-big, the-fineweb-datasets-decanting-the-web-for-the

- **[consensus]** Sparse attention's advantage is context-length-dependent with a crossover well above 32k: token-granular selection loses at 32k, block-granular wins, and the published KV-cache wins are decode-side numbers.  
  independence score 6.0 · tier measured · sources: longcat-sparse-attention-taming-the-lightning-via, native-sparse-attention-hardware-aligned-and-natively, hysparse2-hybrid-sparse-attention, you-only-cache-once, periodic-weak-spots, deepseek-v41-flash

- **[consensus]** A 1M-token global batch is far below the measured critical batch size late in training (~16.8M tokens on OLMo 1B/7B), but CBS is near zero at initialization, so batch-size warmup is required.  
  independence score 3.0 · tier measured · sources: critical-batch-size-revisited-a-simple-empirical, an-empirical-model-of-large-batch-training, accurate-large-minibatch-sgd

## Top contested pairs (from the contradiction graph)

### link-width-x4-vs-x16  — relevance: ?

**The fight:** Is the box's PCIe fabric Gen2 x4 (the repo's premise) or Gen2 x16 (achievable with the capacitor mod)?

- **Side A:** Gen2 x4 — the stock, community-normal configuration. Width is set by the slot and by 12 of 16 lanes being depopulated.
  - Stock electrical link is Gen1 x4 (~1 GB/s); the software Gen unlock lifts speed to Gen2 but width stays whatever the caps allow
  - Stock card has missing AC-coupling capacitors on 12 of 16 lanes, so width beyond x4 is physically blocked
  - All CMP 170HX report x4 (downgraded) from an x16 capability; x4 is normal, x2 is not
  - sources: specifications-cmp-170hx-documentation, all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te, github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2, github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-l
- **Side B:** Gen2 x16 is achievable and measured — the capacitor mod restores the lanes, and a 4-card PLX host trains all cards at Gen2 x16.
  - With the cmpunlocker link patch all four cards on the 4-card PLX host train Gen2 x16; Gen3 is blocked by the OPT_DISABLE_GEN3_SPEED fuse
  - Width beyond x4 needs 24 x 0402 220 nF X7R capacitors soldered on; a four-card Gen2 x16 field report shows zero PCIe errors at 5.97 GB/s
  - Measured BF16 164.4-192.7 TFLOPS against a 202.1 ceiling, and 6.63-6.67 GB/s single-card PCIe bandwidth
  - sources: club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github, physical-mods, performancemd, cmp-170hx-inference-labdocsunlock-and-mods

### p2p-absent-vs-available-vs-breaks-nccl  — relevance: ?

**The fight:** Is peer-to-peer actually unavailable on this box, or is 'no P2P' a driver mask that a patch lifts — and does lifting it break collectives?

- **Side A:** No P2P. Collectives stage through host RAM; the GA100 mailbox path is a genuine dead end.
  - No peer access with the plain unlock; copies stage through host memory at 6.21-6.30 GB/s at 256 MiB
  - P2P fused off (GNS); D2D host-staged at 5.85 GB/s
  - 0 of 56 directed pairs have peer access
  - sources: club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github, github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-l, p2pmd
- **Side B:** P2P works over BAR1 P2P once patched — verified with real transfers, not cudaDeviceCanAccessPeer.
  - P2P works over BAR1 P2P, measured 1.68 GB/s each way on PCIe Gen2 x4, full 4-GPU mesh (12/12 directed pairs), verified with real transfers
  - P2P re-enabled via three bayley patches plus ForceP2P=0x11; TP=2 prefill then runs 1.8-1.9x faster
  - sources: github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2, 2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2-2

### compute-162-vs-42-tflops  — relevance: ?

**The fight:** Is the box's per-GPU bf16 throughput ~162 TFLOPS (repo) or ~42 TFLOPS (published measurements)?

- **Side A:** ~162 TFLOPS bf16 per GPU, ~810 TFLOPS aggregate — measured on the user's unlocked cards.
  - A healthy card measures 155.7 TFLOPS fp16 versus 39.3 under PWRBRK#
  - Measured BF16 164.4-192.7 TFLOPS against a 202.1 ceiling
  - A100 312 TFLOPS bf16 dense at 108 SM scales to ~202 TFLOPS at 70 SM and ~214 at 74 SM; 162 is ~80% of ceiling
  - sources: lessonsmd, performancemd, nvidia-a100-tensor-core-gpu-data-sheet-jun21-1
- **Side B:** ~42 TFLOPS fp16 with tensor cores disengaged; PyTorch reaches only 6.3 TFLOPS.
  - FP16 ~42 TFLOPS but PyTorch only 6.3; tensor cores unusable on the stock card
  - FP16 performance approximately 42 TFLOPS, unthrottled, the primary path for AI inference
  - 170th Street spec sheet lists 49.05 TFLOPS fp16
  - sources: exploration-of-cryptocurrency-mining-specific-gpus-in-ai, all-gbs-without-flops-nvidia-cmp-170hx-review-performance-lockdown-workaround-te, full-specifications-170th-street

### fsdp-multiplier-1.5-vs-2  — relevance: ?

**The fight:** Does FSDP/ZeRO-3 cost 1.5x or 2x DDP's communication?

- **Side A:** 1.5x — FSDP moves 3-Psi bytes per step against DDP's 2-Psi.
  - Full sharding has 1.5x communication overhead and volume over DDP if using a bandwidth-optimal ring algorithm
  - DP = 2-Psi, ZeRO-3 = 3-Psi = 1.5x
  - sources: pytorch-fsdp-experiences-on-scaling-fully-sharded-data-paralle, zero-memory-optimizations-toward-training-trillion-parameter-m, zero-extremely-efficient-collective-communication-for-giant
- **Side B:** 2x — the repo's research.md section 2 states FSDP costs roughly twice DDP.
  - FSDP costs ~2x DDP on this fabric
  - sources: repo-research-md

### ddp-only-vs-pp-viable  — relevance: ?

**The fight:** Is DDP the only viable training topology on this fabric, or is pipeline parallelism a real alternative?

- **Side A:** DDP is the only viable topology; PP, TP and EP are out.
  - The choice is DDP or FSDP, and DDP is ~2x cheaper on the link
  - Drop EP and TP from the training topology candidates
  - sources: repo-research-md
- **Side B:** PP is viable and beats TP by a wide margin; PP's per-step traffic is O(activations), not O(parameters).
  - PP beats TP 6.6x on prefill (5,321 vs 801 tok/s at 77k); TP runs 86 all-reduces per forward over a no-P2P link
  - PP+offload uses zero GPU P2P and beats ZeRO-2/FSDP/ZeRO-Inf/Megatron-PP/TP by 1.48-2.16x on 8x4090
  - Mobius: PCIe 3.0 16 GB/s vs 900 GB/s NVLink; ZeRO traffic 7.3x model size per step, >70% time; 3.8-5.1x via PP+cross-mapping
  - sources: club-170hxdocstopology-and-parallelismmd-at-main-pixelmlclub-1, efficient-training-on-multiple-consumer-gpus-with, mobius-fine-tuning-large-scale-models-on, zeropp-unleashing-exceptional-parallelism-efficiency
