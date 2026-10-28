---
title: '2x CMP 170HX for LLM inference: unlock, cross-root P2P, serving (vLLM, EPYC/ROMED8-2T)
  — Web Pulse'
id: 2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t
tags:
- cmp170hx-training-feasibility-a0d4fc
- cmp-170hx
- p2p-enablement
- cross-root-complex
- pcie-gen2-x16
- vbios-crossflash
- power-cap-thermal
- host-ram-table-offload
created: '2026-10-27T11:04:00.704201Z'
updated: '2026-10-27T19:01:50.570336Z'
source: https://wpnews.pro/news/2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epyc-romed8-2t
source_domain: wpnews.pro
fetched_at: '2026-10-27T11:04:00.083121Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'Aggregated reprint of Roman Puls''s gist (published 2026-10-01, revised
  2026-10-02) documenting 2x CMP 170HX on an EPYC 7443 / ASRock ROMED8-2T host. The
  load-bearing finding for this run is that P2P is NOT permanently fused off: by default
  cudaDeviceCanAccessPeer is 0 and the GSP capabilities report GPU_NOT_SUPPORTED,
  but three driver patches from bayley/cmpunlocker (0011-p2p-bar1, 0013-skip-mailbox-peer-preinit,
  0015-bar1p2p-readcap-override) layered onto amoghmunikote/cmpunlocker plus the regkey
  ForceP2P=0x11 make it 1 in both directions, verified by cuMemcpyPeer and by NCCL
  reporting ''via P2P/CUMEM''. With P2P working, vLLM TP=2 prefill is about 1.8-1.9x
  faster than without, on Gen2. The author is careful that the regkey alone was never
  verified sufficient.


  Other measured facts: after unlock each card reports 64 GB, 74 SM (cmpunlocker master
  6c442ee; v0.4 gives 70 SM) and trains at PCIe Gen2 x16; the 250 W card can be cross-flashed
  to NVIDIA''s 300 W VBIOS (92.00.6D.00.0A, memory 1728 MHz); thermal test shows 200
  W plateaus at 77 C, 225 W at 83-84 C with no margin, and 250 W runs away at +6 C/min;
  a +200 MHz GPC VF offset is stable under real vLLM load (~1600-1650 MHz at 200 W)
  while +250 hung a card with Xid 175. Prefill table: no-P2P ~2500 tok/s flat across
  8k-128k, P2P at 70 SM 4634 down to 4486, at 74 SM 4766 down to 4599, and with the
  +200 offset 5033 down to 4881.


  Directly relevant to the Engram memory question: this deployment keeps the model''s
  PLE / n-gram tables in host RAM (~128 GB) via VLLM_PLE_CPU_OFFLOAD=1, which is a
  working precedent for large lookup tables living outside VRAM.'
utility_score: 12.0
---

[Skip to content](https://wpnews.pro/news/2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epyc-romed8-2t#main)
cd /news/ai-infrastructure/2x-cmp-170hx-for-llm-inference-unloc… · [home](https://wpnews.pro/) › [topics](https://wpnews.pro/topics) › [ai-infrastructure](https://wpnews.pro/topics/ai-infrastructure) › article
[ARTICLE · art-143865] pub=2026-10-01T21:00Z topic=ai-infrastructure verified=true sentiment=· neutral
# 2x CMP 170HX for LLM inference: unlock, cross-root P2P, serving (vLLM, EPYC/ROMED8-2T)
A developer documented running two unlocked NVIDIA CMP 170HX mining cards (GA100, A100-class) for LLM inference on a single AMD EPYC 7443 / ASRock Rack ROMED8-2T host, unlocking them to 64 GB and 74 SM and getting cross-root-complex P2P working. The revised setup uses amoghmunikote's cmpunlocker patched with three P2P patches from bayley/cmpunlocker, a 300 W VBIOS cross-flash with undervolting under a power cap, and serves a W4A16 MoE model with vLLM 0.30.0 built from source. The writeup notes the initial P2P configuration was not reproducible for some users and details the UEFI settings, kernel parameters and cooling needed to make the 64 GB BARs usable.
byRoman Puls read15 min views148 publishedOct 1, 2026 $ copy permalink [$ source](https://gist.github.com/ropuls/d2978531ba3a1a183e70f62f3de2eb84) [$ cite (json)](https://wpnews.pro/news/2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epyc-romed8-2t.json)
Running two NVIDIA CMP 170HX (GA100, A100-class mining cards) for LLM inference on a single host: unlocking them to 64 GB and 74 SM, getting P2P working across separate root complexes, cross-flashing the 250 W card to NVIDIA's 300 W VBIOS, undervolting under a power cap, and serving a W4A16 MoE model with vLLM.
**Update 2026-10-02.** The initial setup here was not reproducible for some people (P2P did not come up). This is the revised setting: amogh's cmpunlocker as the base, patched selectively with three P2P patches from bayley/cmpunlocker, pinned to exact commits (section 2). Also new: 74 SM, the 300 W VBIOS cross-flash, undervolt, thermal limits and benchmarks.
  * 2x NVIDIA CMP 170HX (8 GB locked, 64 GB after unlock), same board (PN 900-11001-0108-000, GPU 20C2-105-A1).
  * AMD EPYC 7443, ASRock Rack ROMED8-2T. The two PCIe x16 slots are on separate root complexes (`nvidia-smi topo -m` shows NODE), which matters for P2P (section 2).
  * Plenty of host RAM. The MoE model in section 6 offloads its PLE / n-gram tables to system memory (about 128 GB), so this is a real requirement, not just headroom. We run 256 GB DDR4 (8x 32 GB) and budget roughly 180 GB for the serving container.
  * The cards are passive. Ours each have one Same Sky (formerly CUI Devices) CBM-7525B-145-494-22 blower (75x75x25 mm) on a straight shroud, driven at 100% duty via the BMC, case open; without forced airflow they overheat within minutes. This is our current blower solution and it is not optimal: all power and temperature limits in this gist (section 4, Benchmark) are for this setup. Next we will try stronger blowers, Delta BCB0812UHN-TP09 (12 V, 30 CFM) or Delta BFB1012UH-BA40ZYD (14.2 V nominal, 37 CFM); both draw over 2 A, so check what your fan headers can supply.

  
| Component  | Version  | Note  |  
| --- | --- | --- |  
| NVIDIA open kernel modules  | 610.57.04  | built and patched by cmpunlocker  |  
| CUDA Toolkit  | 13.3  | needed for vLLM's flashinfer JIT  |  
| cmpunlocker  | amoghmunikote/cmpunlocker master `6c442ee` ("Add more SMs", PR #55)  | not tagged yet; last release v0.4. Plus the three P2P patches from section 2  |  
| vLLM  | 0.30.0  | built from source  |  
| Kernel  | 7.0.14-pve (Proxmox VE)  | pin it, or re-run the installer after a kernel upgrade  |  
| VBIOS  | 92.00.6D.00.0A (300 W) on both cards  | section 3  |  
The ROMED8-2T is UEFI-only. These make the unlocked 64 GB BARs usable:
  * Above 4G Decoding: Enabled. Each unlocked card exposes a 64 GB BAR (two cards is about 128 GB of BAR space). The firmware has to map that in the 64-bit MMIO window above 4 GB, otherwise the large BARs cannot be assigned and the driver fails. This is the setting that matters most.
  * Resizable BAR: Enabled (Auto). Lets each card grow its BAR to the full 64 GB after the unlock. Pre-unlock the BARs are small, so this only matters afterwards.
  * Secure Boot: Disabled. The cmpunlocker kernel modules are unsigned.
  * Boot in pure UEFI mode (disable CSM if the board has it). Legacy/CSM boot can leave GPU BARs unassigned (symptom: "BARx is 0M"). A `pci=realloc` kernel argument is a stopgap; UEFI-only is the clean fix.
  * IOMMU: off (`amd_iommu=off iommu=off` , no AMD-Vi in dmesg,`/sys/class/iommu` empty). We use the cards from LXC containers, which needs no IOMMU, and cross-root P2P works with it off. (An earlier version of this gist recommended`amd_iommu=on iommu=pt` ; that is only relevant for VM passthrough and is not what this box runs.)


A firmware change needs a full power-off to take effect.

```
console=tty0 console=ttyS1,115200n8 pci=realloc quiet acpi_enforce_resources=lax amd_iommu=off iommu=off iomem=relaxed

```

  * `pci=realloc` : lets the kernel reassign the large 64 GB BARs; we keep it on.
  * `amd_iommu=off iommu=off` : see above.
  * `iomem=relaxed` : only needed if you use 170tune's BAR0 tools (section 4); not needed for the unlock, P2P or serving.
  * `console=ttyS1,115200n8` : serial console for the BMC's Serial-over-LAN.
  * `--no-iommu` and`--no-passthrough` are**installer flags** of cmpunlocker (leave the kernel cmdline alone, do not bind the cards to vfio-pci), not kernel arguments.


```
./install.sh --profile=8gb --no-iommu --no-passthrough

```

  * Use master `6c442ee` or newer: it opens RECONFIG_PLM and re-enables the reserved TPCs, 70 to 74 SM per card (check with torch`multi_processor_count` ; dmesg shows`SM-RECONFIG GPCn` ). v0.4 gives 70 SM.
  * It builds the open-gpu-kernel-modules against your running headers (610.57.04, 615.71.09, 610.43.03, 610.43.02 supported). It removes NVIDIA DKMS modules.
  * Secure Boot off (unsigned modules).
  * Cold boot (full power-off) after the first install and after any card or firmware change. For a driver update, e.g. v0.4 to the +4 SM commit, a plain reboot was enough here (and in the field report in PR #58). If the 64 GB or the extra SMs do not show up after a reboot, power the machine off completely.
  * Changing the SM count invalidates vLLM's torch.compile cache. Move `~/.cache/vllm` of the service user aside before the reboot, or vLLM crash-loops.
  * The installer rewrites `/etc/modprobe.d/cmp-pcie-gen2.conf` and drops any extra regkeys, including ForceP2P from section 2. Re-add them after every install, then`update-initramfs -u` .
  * Result: each card reports 64 GB, 74 SM, and the link trains at PCIe Gen2 x16.


Two cards on separate root complexes default to no P2P: `cudaDeviceCanAccessPeer` is 0, and the GSP capabilities report GPU_NOT_SUPPORTED. What works here is a regkey **plus** three driver patches on top of cmpunlocker.
The patches are from [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) (GPL-2.0, not in amogh's repo). We use them unchanged, pinned to bayley commit `5a7bb4b`:
  * [`0011-p2p-bar1.patch`](https://github.com/bayley/cmpunlocker/blob/5a7bb4b7e5056306fe49e8b824787659abb19914/driver/patches/0011-p2p-bar1.patch) : BAR1 P2P path (bus/BIF/UVM/RM changes).
  * [`0013-skip-mailbox-peer-preinit.patch`](https://github.com/bayley/cmpunlocker/blob/5a7bb4b7e5056306fe49e8b824787659abb19914/driver/patches/0013-skip-mailbox-peer-preinit.patch) : stops the mailbox peer pre-registration at GPU init from blocking BAR1 P2P.
  * [`0015-bar1p2p-readcap-override.patch`](https://github.com/bayley/cmpunlocker/blob/5a7bb4b7e5056306fe49e8b824787659abb19914/driver/patches/0015-bar1p2p-readcap-override.patch) : restores the BAR1 P2P read capability that the bridge discovery loses.


To use them with amogh's cmpunlocker (we run master [`6c442ee`](https://github.com/amoghmunikote/cmpunlocker/commit/6c442eeb6448b97c803e72b61da344a39e0a26ab)): copy the three files into `driver/patches/` and register them in two places.
`driver/build.sh`, array `PATCH_ORDER`, sets the order in which the patches are applied (`patch -p1`, top to bottom). Add the three at the end, after `cmp-sku-mask.patch`, in this order. On `6c442ee` the full array then reads:

```
PATCH_ORDER=(
    sec2-postbl-plm-ss-cfg.patch
    booter-verify.patch
    late-pma.patch
    bar0-pramin-clamp.patch
    ce-scrub-workarounds.patch
    persistent-sw-state.patch
    pcie-gen2.patch
    pcie-gen2-probe-retrain.patch
    name-string.patch
    bar1-resize-unlock.patch
    cmp-sku-mask.patch
    0011-p2p-bar1.patch
    0013-skip-mailbox-peer-preinit.patch
    0015-bar1p2p-readcap-override.patch
)

```

`common/constants.yaml` (YAML), section `unlocks:`, must declare every patch that `PATCH_ORDER` builds; `tools/read-constants.py` cross-checks both and the build aborts with "patch ... is built but not declared in constants.yaml" otherwise. The order of the entries there does not matter. Append:

```
  p2p_bar1:
    patch: 0011-p2p-bar1.patch
    registers: {}

  p2p_skip_mailbox:
    patch: 0013-skip-mailbox-peer-preinit.patch
    registers: {}

  p2p_readcap:
    patch: 0015-bar1p2p-readcap-override.patch
    registers: {}

```

Then run the installer and check its log (`logs/install_<timestamp>.log`). The bayley patches were made against an older tree, so on `6c442ee` some hunks land with an offset or fuzz. That is expected and fine, for example:

```
[INFO]    0011-p2p-bar1.patch
Hunk #1 succeeded at 1839 (offset 14 lines).
[INFO]    0015-bar1p2p-readcap-override.patch
Hunk #1 succeeded at 669 with fuzz 2 (offset -13 lines).
[ OK ]  All patches applied

```

What must not appear is `FAILED`, `rejects` or a `.rej` file; then the patch did not apply. The three do not touch the files of the SM patch. After the install, re-add the regkey below (the installer overwrites that file).

```
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;RMForceStaticBar1=1;RMPcieP2PType=1;RMForceP2PType=1;ForceP2P=0x11"

```

Then `update-initramfs -u` and reboot.
  * ForceP2P=0x11 is READ bit0 plus WRITE bit4. It is read at module load, which avoids an init-order problem where the per-PCI-device-ID default runs before the device ID is populated, so the CMP branch is never taken and the override stays unset (the driver then falls back to GSP caps and reports GNS). The three RM*P2P/StaticBar1 keys are historical and probably no-ops; they do no harm.
  * Not tested: whether the regkey alone, without the three patches, is enough. An earlier version of this gist said so; that was not verified.
  * Verify: `cudaDeviceCanAccessPeer` is 1 in both directions;`cuMemcpyPeer` works both ways; with`NCCL_DEBUG=INFO` you see`Channel NN/0 : 0[0] -> 1[1] via P2P/CUMEM` in both directions.
  * Result: vLLM TP=2 prefill is about 1.8 to 1.9 times faster than without P2P, on Gen2.


Our two cards came with different VBIOS: `92.00.67.00.01` (250 W, memory 1458 MHz, SM max 1410 MHz) and `92.00.6D.00.0A` (NVIDIA's 300 W "OC mining" image, memory 1728 MHz, SM max 1695 MHz). Same board, same IDs (`10DE:20C2 / 10DE:1585`). In TP2 the 67 card sets the pace, and no power limit fixes that, because it sits at its VBIOS clock maximum.
We flashed the 67 card to 6D with nvflash 5.867 (Linux), using TechPowerUp image [#268495](https://www.techpowerup.com/vgabios/268495/268495) (SHA1 `efad37d5…cddfb7`; its code region is byte-identical to a dump of our own 6D card). Following the recipe in [aiultrayang/cmp170hx-vbios-flash](https://github.com/aiultrayang/cmp170hx-vbios-flash):
  * **One nvflash operation per card per cold boot** (Falcon). Any second call fails with "Falcon In HALT or STOP state". So: cold boot 1 =`--save` a rollback dump of each card, cold boot 2 = flash, cold boot 3 = verify. No`--list` or`--check` before the real call.
  * Keep the nvidia driver from (`install nvidia /bin/false` and the same for nvidia_uvm/modeset/drm in modprobe.d) for those boots.
  * Flash: `nvflash -i <idx> --log flash.log 92.00.6D.00.0A.rom` , interactively (in tmux; it needs a TTY), read the prompt (current 67, new 6D, same IDs) before pressing`y` . No extra flags (no`--overridesub` , no`--mergeinforom` : the InfoROM version is identical, 1001.0108.01.02).
  * nvflash keeps the card's identity by default: serial, BRD/OBD/OEM objects, retired pages/row remap (RPR/RRL), erase ledger, license block. The log lists each "Preserve InfoROM ... object". Telemetry objects (BBX, SEN, DEM and a few others) come from the image file, i.e. from the donor card. No effect on serving.
  * A failed write means an external SPI programmer (CH341A on the SOP8 chip). Have one at hand and keep the rollback dump.
  * Result: both cards 6D, 64 GB unlock unaffected, memory 1728 MHz, max 300 W.
  * Cap each card to what your cooling can hold: `nvidia-smi -pl <W>` , address cards by UUID rather than index, and persist it with a systemd oneshot. A module reload resets the limits to the VBIOS default (250 W), so re-apply after any rmmod/modprobe. With 6D the hardware maximum is 300 W.
  * Our thermal test (both cards under gpu-burn at once, blowers at 100%, open case): 200 W plateaus at 77 C, 225 W at 83-84 C (no margin), 250 W runs away (+6 C/min, no plateau). We run 200 W. These numbers reflect our non-optimal cooling (one blower per card), not a limit of the card itself.
  * Undervolt via NVML: the 6D VBIOS allows a GPC VF offset of -1000..+1000 MHz (`nvmlDeviceSetGpcClkVfOffset` ; the unit is MHz, a shift of the V/F curve, not mV). We keep the power cap as the governor and add the offset, so the same watts buy more clock. Under a sustained bf16 GEMM at 200 W: +0 about 1385 MHz, +200 about 1500 MHz, +250 about 1525 MHz. Both +200 and +250 passed full-VRAM pattern sweeps and a bit-exact GEMM check with no Xid.
  * +250 hung one card within seconds under real vLLM load (Xid 175 GSP RPC timeout, then Xid 154 "recovery action PF FLR"; it needed a reboot). The synthetic checks did not catch it; [170tune](https://github.com/cachenetics/170tune) documents the same: a passed gate is necessary, not sufficient. +200 is our serving point: clean under real load, about 1600-1650 MHz at 200 W. The offset is volatile (gone after a reboot or driver reload); re-apply it at boot.
  * 170tune caveat: its apply path sets `nvidia-smi -pl 300` and its reset path`-pl 250` , hardcoded. On a card that cannot dissipate that, use only its test tools or patch those values.
  * 170tune's HBM clock and timing levers need extra FBPA PLMs opened (FBPA_MEM, FBPA PLL). amogh's cmpunlocker does not open them (a PR for that was declined), so `170tune preflight` reports the masks as closed. The SM undervolt does not need them.
  * In LXC: the `/dev/nvidiaN` minor numbers are not stable across boots and do not match the nvidia-smi index. Bind all of them into the container and select the cards with`CUDA_VISIBLE_DEVICES=GPU-<uuid>` . Symptom of the wrong card: an OOM saying "total capacity of 11.63 GiB" (that was a 12 GB card, not a lost unlock).


In-band `ipmitool raw 0x3a 0xda` (get) / `0xd6` (set duty) / `0xd8` (manual mode), the AST2500 command set, duty 0-100 on the ROMED8-2T. The `0x3a 0x05/0x06` from the ASRock FAQ returned "invalid" on these boards. The manual setting survives reboots and BMC power cycles.

```
--tensor-parallel-size 2 --enable-expert-parallel
--gpu-memory-utilization 0.90 --max-num-seqs 16
--disable-custom-all-reduce
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
--speculative-config '{"method":"mtp","num_speculative_tokens":4}'

```

with `VLLM_PLE_CPU_OFFLOAD=1` and `NCCL_P2P_LEVEL=SYS`.
  * VLLM_PLE_CPU_OFFLOAD=1 keeps the PLE / n-gram tables in host RAM (about 128 GB; budget roughly 180 GB for the container/host). Without it you get a GPU OOM.
  * `--disable-custom-all-reduce` is required with P2P. The custom IPC all-reduce backend OOMs the VRAM and throws "invalid argument" across root complexes, which crash-loops the workers. NCCL/PYNCCL still uses P2P, so the speedup stays.
  * cudagraph_mode FULL_AND_PIECEWISE gave about 10% more single-stream decode on code than FULL_DECODE_ONLY.
  * Build fix for compressed-tensors PLE: vLLM 0.30.0 throws `NotImplementedError: Qwen4Exp PLE embedding does not support CompressedTensorsConfig` . Its`from_quant_config` path handles only ModelOpt/Fp8 for excluded PLE layers. Add a CompressedTensorsConfig branch that checks`should_ignore_layer(...)` and returns the unquantized PLE embedding method. Small Python change, no recompile.
  * Does not work: `--kv-cache-dtype fp8` crashes the state-attention backend on this model (only auto/bfloat16 are supported).
  * flashinfer's JIT compiles on first start (about 2 minutes), then it is cached.
  * CUDA tools inside a container need the matching `libcuda.so.1` /`libnvidia-ml.so.1` on the path; the driver userspace has to match the kernel module version.
  * A crashing vLLM worker leaks shared memory in /dev/shm (up to the PLE size) and pinned RAM. If that happens: stop the service, reset-failed, check /dev/shm and free RAM, then start once cleanly.
  * After any driver, VBIOS or clock change, check `dmesg | grep -i xid` . Xid 13/31/43 under load usually means an unstable clock/voltage point; 79 means the card fell off the bus.


2x CMP 170HX, vLLM 0.30 TP=2, W4A16 MoE (Qwen3.8-Flash-Next), MTP 4, both cards capped at 200 W unless noted.
Method: decode counted with the server counter `vllm:generation_tokens_total` (counting SSE chunks undercounts with speculative decoding), real code prompts (random tokens kill the MTP acceptance), 3 reps. Prefill with `vllm bench serve` random input, output 4, 3-rep median, tok/s = input length / TTFT. Use a different `--seed` for every point and rep: the random dataset builds prompts as consecutive tokens from a seed-dependent offset, so with one seed the 128k prompt starts with the 65k prompt and you measure prefix-cache hits (we got a fake 8,600 tok/s at 128k that way). Check `vllm:prefix_cache_hits_total` stays flat.  
| setup  | 8k  | 32k  | 65k  | 128k  |  
| --- | --- | --- | --- | --- |  
| no P2P (L1T W4A16 reference)  | ~2500  | ~2500  | ~2500  | ~2500  |  
| P2P, mixed VBIOS (67 + 6D), 70 SM  | 4634  | 4572  | 4543  | 4486  |  
| P2P, both 6D, 70 SM  | 4682  | 4623  | 4586  | 4527  |  
| P2P, both 6D, 74 SM  | 4766  | 4708  | 4669  | 4599  |  
| **+ offset +200 MHz, 200 W**  | **5033**  | **4989**  | **4951**  | **4881**  |  
| + offset +200 MHz, 225 W  | 5084  | 5037  | 5002  | 4930  |  
| + offset +200 MHz, 250 W  | not stable  |  
| setup  | code c=1  | code c=4 aggregate  | code c=4 per request  | prose c=1  |  
| --- | --- | --- | --- | --- |  
| P2P, mixed VBIOS, 70 SM  | 193  | 483  | 131  | 113  |  
| P2P, both 6D, 70 SM  | 208  | 508  | 140  | 122  |  
| P2P, both 6D, 74 SM  | 209  | 514  | 143  | 123  |  
| **+ offset +200 MHz, 200 W**  | **223**  | **527**  | **150**  | **130**  |  
| + offset +200 MHz, 225 W  | 226  | 554  | 150  | 136  |  
| + offset +200 MHz, 250 W  | not stable  |  
| cap  | SM clock under real load  | power avg (prefill)  | GPU max (decode / prefill)  | HBM max  | gpu-burn, both cards  |  
| --- | --- | --- | --- | --- | --- |  
| 200 W  | ~1600-1620 MHz  | ~192-196 W  | 71 / 72 C  | ~77 C  | plateau  |  
| 225 W  | ~1610-1635 MHz  | ~214-219 W  | 74 / 79 C  | 80 C  | 84-85 C after 4.5 min, aborted at 85 C (no plateau)  |  
| 250 W  | n/a  | n/a  | n/a  | n/a  | 83 C within 2 min, aborted; not stable with our cooling  |  
MTP acceptance about 80% on code, 36% on prose throughout. Decode is mostly memory-bound, so the 6D memory clock helped it and the extra SMs alone did not. Prefill is compute-bound, but both cards sit at the power cap, so +4 SM gave only about +1.8%. With the +200 offset the SMs pay off: about +7% decode and +6% prefill at the same 200 W and the same temperatures. With the offset the cards already run near their practical clock ceiling at 200 W, so 225 W adds only 1-2% for 5-7 C more. 250 W is not stable here: the cards run away thermally under sustained load. We serve at 200 W, offset +200. (+250 hung a card under real load, see section 4.)
source & further reading
[gist.github.com — original article](https://gist.github.com/ropuls/d2978531ba3a1a183e70f62f3de2eb84) [.htaccess (mashup)](https://wpnews.pro/news/htaccess-mashup) [Build a beautiful TV home screen with AI — Samraaj’s prompt, camera watcher workflow, and Samsung / Android TV references](https://wpnews.pro/news/build-a-beautiful-tv-home-screen-with-ai-samraajs-prompt-camera-watcher-workflow) [Finding and fixing asyncio event-loop stalls in a streaming gateway: a stack-capturing watchdog, per-process shared resources, and the benchmark behind the numbers.](https://wpnews.pro/news/finding-and-fixing-asyncio-event-loop-stalls-in-a-streaming-gateway-a-stack-per)
── more in #ai-infrastructure 4 stories · sorted by recency
shattered.io · 6 hours ago · #ai-infrastructure
[ DGX Spark vs Mac Studio: $4,999 vs $2,499 [2026] ](https://wpnews.pro/news/dgx-spark-vs-mac-studio-4999-vs-2499-2026)
servethehome.com · 16 hours ago · #ai-infrastructure
[ ASRock Creator Radeon AI Pro R9700 Review: Workstation Inference, AMD Style ](https://wpnews.pro/news/asrock-creator-radeon-ai-pro-r9700-review-workstation-inference-amd-style)
siliconangle.com · 33 min ago · #ai-infrastructure
[ Akamai expands its AI Cloud platform with dedicated inference and agent execution environments ](https://wpnews.pro/news/akamai-expands-its-ai-cloud-platform-with-dedicated-inference-and-agent)
news.ycombinator.com · 2 hours ago · #ai-infrastructure
[ Show HN: AMD EPYC Windows developer rigs in the UK/EU in ~10 minutes ](https://wpnews.pro/news/show-hn-amd-epyc-windows-developer-rigs-in-the-uk-eu-in-10-minutes)
── more on @nvidia 3 stories trending now
wpnews · 10 hours ago · #ai-agents
[ Cooks – open-source sales engagement (sequences, parallel dialer, MCP) ](https://wpnews.pro/news/cooks-open-source-sales-engagement-sequences-parallel-dialer-mcp)
wpnews · 22 hours ago · #ai-agents
[ Connect Claude Code, Codex and ChatGPT to one understanding ](https://wpnews.pro/news/connect-claude-code-codex-and-chatgpt-to-one-understanding)
wpnews · 10 hours ago · #artificial-intelligence
[ OpenAI launches GPT-6.1 Terra for faster, better performing AI agents at half the price ](https://wpnews.pro/news/openai-launches-gpt-6-1-terra-for-faster-better-performing-ai-agents-at-half-the)
sponsored brought to you by  **4,200+** EU-deployed projects
reading about agents? ship yours in a single git push.
## Run your AI side-project on zahid.host
EU-based hosting, git-push deploys, automatic HTTPS, no cold starts. Free tier with a custom domain — perfect for shipping the agent you just read about.
$git push zahid main
→ Live at https://your-agent.zahid.host ✓
[Get free account →](https://zahid.host/signup) [Pricing](https://zahid.host/pricing)
from €0/mo · no card required 
LIVE [news/2x-cmp-170hx-for-llm…] indexed:**0** read:**15min** 2026-10-01 · 11:03:59
×
strictly-necessary cookies only · [privacy](https://wpnews.pro/privacy)
