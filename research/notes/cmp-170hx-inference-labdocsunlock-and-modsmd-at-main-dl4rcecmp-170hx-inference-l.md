---
title: cmp-170hx-inference-lab/docs/UNLOCK-AND-MODS.md at main · dl4rce/cmp-170hx-inference-lab
  · GitHub
id: cmp-170hx-inference-labdocsunlock-and-modsmd-at-main-dl4rcecmp-170hx-inference-l
tags:
- cmp170hx-training-feasibility-a0d4fc
- cmp-170hx
- unlock-caveats
- sku-lottery
- pcie-width-vs-speed
- tensor-parallel-trap
created: '2026-10-27T11:20:39.985692Z'
updated: '2026-10-27T19:00:31.465191Z'
source: https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md
source_domain: github.com
fetched_at: '2026-10-27T11:20:39.364838Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Short (573-word) but load-bearing caveat doc from the dl4rce lab. Its central
  point is that the two unlocks are orthogonal: the software unlock (patched nvidia-open
  610.43.x) gives compute throughput, VRAM geometry (64 GB on the 8 GB SKU, 40 GB
  on the 10 GB SKU) and PCIe Gen2 *speed*, while PCIe *width* x16 requires populating
  the missing AC-coupling capacitors on 12 of the 16 unpopulated lanes. Neither delivers
  PCIe Gen3/Gen4, NVLink, ECC, or native FP8/FP4.


  SKU lottery detail that matters for capacity planning: device id 10de:20c2 (8 GB,
  often Hynix) is the 64 GB unlock; 10de:2082 (10 GB, often Samsung) is stable at
  40 GB, and the 80 GB geometry has been reported unstable with GSP/DMA faults, so
  ''80 GB 170HX'' listings should not be trusted. The FMA lock is itself a lottery
  until the unlock is applied and a microbench confirms real FP32/BF16 throughput.
  Also states plainly that multi-GPU tensor parallel over this PCIe is a trap, citing
  the public DeepSeek-V4-on-four-170HX write-ups that had to use pipeline parallel
  instead, and that the cards ship at 250 W with no display output and usually no
  NVENC.'
---

*Suggested by [[github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-lab-vllm-scripts-me]] — unlock and hardware-mod caveats referenced by the lab README*

[Skip to content](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md) to refresh your session. Dismiss alert
## Collapse file tree
## Files
Search this repository(forward slash)` forward slash/`
/
# UNLOCK-AND-MODS.md
Copy path
More file actions
More file actions
## Latest commit
and
[Publish CMP 170HX inference lab: 27B MTP and 72B AWQ, scripts, graphs.](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/c68dcebb03ab28c3a7c29279258700c1899160c3)
Open commit details
Aug 21, 2026
[c68dceb](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/c68dcebb03ab28c3a7c29279258700c1899160c3) · Aug 21, 2026
## History
[History](https://github.com/dl4rce/cmp-170hx-inference-lab/commits/main/docs/UNLOCK-AND-MODS.md)
Open commit details
History
40 lines (24 loc) · 2.78 KB
/
# UNLOCK-AND-MODS.md
Copy path
## File metadata and controls
  * Preview
  * 

40 lines (24 loc) · 2.78 KB
Copy raw file
Download raw file
Outline
Edit and raw actions
# Unlock, SKU lottery, and the hardware mod
This lab did **not** perform the unlock. The rented card already reported 64 GB, unlocked FMA, and PCIe **Gen2 ×16**. The notes below are from public community sources so a buyer can price the _work_ , not just the board.
## Two different problems  
| What you want  | How  | Typical stock  | After  |  
| --- | --- | --- | --- |  
| Compute + VRAM geometry  | Software unlock (patched `nvidia-open` 610.43.x)  | 8 GB or 10 GB visible, FMA throttled, often Gen1  | 64 GB (8 GB SKU) or 40 GB (10 GB SKU), full SM throughput, **Gen2 speed**  |  
| PCIe **width** ×16  | Hardware: populate missing AC-coupling capacitors  | Often **×4** (12 of 16 lanes unpopulated)  | ×16. Independent of the Gen2 speed unlock  |  
Gen2 vs ×16 are orthogonal. Software gives you Gen2. Width is solder. Neither one is PCIe Gen3/Gen4, NVLink, ECC, or native FP8/FP4.
Primary public write-ups:
  * Unlock tool: [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
  * Technical wiki (silicon, OTP, cooling, open problems): [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx)
  * Price spike after the unlock landed: [Wccftech, 2026](https://wccftech.com/nvidia-cmp-170hx-8-10-gb-prices-explode-over-1000-usd-as-tool-unlocks-hidden-64-80gb-vram/)


This repository does **not** ship patches, ROP chains, or register recipes. Use the projects above, and budget a cold power-cycle plus Secure Boot off.
## SKU lottery
  * **8 GB / device id`10de:20c2` (often Hynix):** community consensus is this is the 64 GB unlock.
  * **10 GB /`10de:2082` (often Samsung):** 40 GB is the stable story. 80 GB geometry has been tried and reported unstable (GSP/DMA faults). Do not buy “80 GB 170HX” marketing.


FMA lock is also a lottery until the unlock is applied and `nvidia-smi` / a microbench show real FP32/BF16 throughput.
## Cooling and the rest of the board
These were mining cards. Expect a blower or a barely adequate shroud, **250 W** , no display output, and usually no NVENC. Putting one in a quiet desktop without a real cooler is the part people skip in 1 500–1 800 € listings.
NVLink fingers may be present; the interface ICs are typically unpopulated. Multi-GPU tensor parallel over this PCIe is a trap (see also public DeepSeek-V4-on-four-170HX write-ups that had to use **pipeline** parallel, not TP).
## Reddit (field reports, not vendor copy)
  * Unlock announcement / skepticism thread: [r/LocalLLaMA — Nvidia CMP 170HX Unlock 8GB to 64GB](https://www.reddit.com/r/LocalLLaMA/comments/1v0a7wn/)
  * Later field-report thread: [r/LocalLLaMA — Any CMP 170HX field report](https://www.reddit.com/r/LocalLLaMA/comments/1vbf1s0/)


Hype in those threads still outruns the card: “5090 speed with 64 GB” is not what we measured. 64 GB is real. 2025–26 inference formats (FP8/FP4) are not.
You can’t perform that action at this time. 
