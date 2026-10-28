---
title: 'GitHub - dl4rce/cmp-170hx-inference-lab: CMP 170HX LLM inference lab: vLLM
  scripts, measured results, graphs. Not an A100. Unlock/mod caveats. · GitHub'
id: github-dl4rcecmp-170hx-inference-lab-cmp-170hx-llm-inference-lab-vllm-scripts-me
tags:
- cmp170hx-training-feasibility-a0d4fc
- cmp-170hx
- no-p2p-fused
- tensor-parallel-penalty
- pipeline-parallel
- pcie-width-vs-throughput
- measured-benchmarks
- vram-unlock
created: '2026-10-27T11:03:46.187811Z'
updated: '2026-10-27T19:00:31.407868Z'
source: https://github.com/dl4rce/cmp-170hx-inference-lab
source_domain: github.com
fetched_at: '2026-10-27T11:03:45.563883Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Independent measured lab (2026-08-21) on unlocked 64 GB CMP 170HX cards,
  with the dual-card experiment that matters most for this run. Confirms P2P is fused
  off in the CMP SKU: nvidia-smi topo -p2p reports GNS (''GPU not supported''), torch.cuda.can_device_access_peer
  is False in both directions, device-to-device copies are host-staged at 5.85 GB/s
  versus 6.17 GB/s for H2D, and vLLM logs ''Custom allreduce is disabled because your
  platform lacks GPU P2P capability''. The author states explicitly this is not IOMMU
  or ACS, so nothing in BIOS or the kernel recovers it.


  Dual-card numbers (same engine, checkpoint and flags; only the split changes): TP=2
  raises single-stream decode 57.5 to 70.9 tok/s (+23%) but drops 16-agent aggregate
  from 608 to 415 tok/s (-32%) and 48K prefill from 27.1 to 47.4 s (-75%); PP=2 gives
  605 tok/s at 16 agents, 895 at 32 agents and 14.8 s at 48K prefill (3.2x faster
  than TP=2); two independent TP=1 servers behind a load balancer reach 754 tok/s
  at 16 agents and 1,253 at 32, i.e. 3.0x the TP=2 figure on identical silicon. TP=2''s
  crossover is around 3 agents, and PP=2 matches single-card throughput while holding
  2.3x the KV pool.


  Direct evidence on the repo''s open question about link width: measured PCIe x16
  versus x4 gives H2D 6.63 versus 1.7 GB/s, and decode and prefill did not move -
  so width barely matters for inference, though the repo''s training communication
  estimates do depend on it. Unlock path recorded: cmpunlocker, 8 GB SKU to 64 GB,
  10 GB SKU to ~40 GB stable (not 80); Gen2 speed is software while width is a solder
  mod (24x 0402 AC-coupling capacitors).'
utility_score: 12.0
---

[Skip to content](https://github.com/dl4rce/cmp-170hx-inference-lab#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/dl4rce/cmp-170hx-inference-lab) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/dl4rce/cmp-170hx-inference-lab/branches)[](https://github.com/dl4rce/cmp-170hx-inference-lab/tags)
Go to file
Open more actions menu
## Latest commit
and
[Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5)
Open commit detailssuccess
Aug 21, 2026
[88a91bc](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5) · Aug 21, 2026
## History
[11 Commits](https://github.com/dl4rce/cmp-170hx-inference-lab/commits/main/)
Open commit details
11 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   | [Add gitleaks CI, local hook, and Hub shard hash check.](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/b3165befbed8f14ff672f384f6ed6c0bcb9fcee2 "Add gitleaks CI, local hook, and Hub shard hash check.

Public tree must stay free of IPs, tokens, and rental hosts; GitHub secret scanning and push protection are on for this repo.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
| [Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5 "Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endpoint

Pairs the fastest model with the best dual-card mode. Results:

  32 agents    1102.6 tok/s - highest single-endpoint throughput measured
  KV pool      5,254,908 tokens, 40.1x concurrency at 128K, the deepest
  TTFT         26.0 ms, down from 77.7 ms on one card
  53K prefill  18.1 s vs 33.6 s, the same pipeline-overlap win as Qwen

The honest caveat is in the 16-agent row: one card does 790 tok/s and PP=2
only 626. The second card pays off past ~16 agents or when the deeper KV
pool is the point, not before. Power rises with load here \(422 W at 32
agents\), the opposite of the 72B TP=2 run where batch drove power down
because the cards blocked on the interconnect - PP=2 keeps both busy.

Also documents the install path that actually works. The recipe we had
published \(@main\) now 404s, because VLLM_USE_PRECOMPILED needs a commit
with a published wheel and main has moved past it. Adds the wheel-probe
loop, plus the two traps hit today: --system-site-packages lets a stock
0.27.1 satisfy the requirement, and a short hash cannot be checked out
from a shallow clone.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5 "Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endpoint

Pairs the fastest model with the best dual-card mode. Results:

  32 agents    1102.6 tok/s - highest single-endpoint throughput measured
  KV pool      5,254,908 tokens, 40.1x concurrency at 128K, the deepest
  TTFT         26.0 ms, down from 77.7 ms on one card
  53K prefill  18.1 s vs 33.6 s, the same pipeline-overlap win as Qwen

The honest caveat is in the 16-agent row: one card does 790 tok/s and PP=2
only 626. The second card pays off past ~16 agents or when the deeper KV
pool is the point, not before. Power rises with load here \(422 W at 32
agents\), the opposite of the 72B TP=2 run where batch drove power down
because the cards blocked on the interconnect - PP=2 keeps both busy.

Also documents the install path that actually works. The recipe we had
published \(@main\) now 404s, because VLLM_USE_PRECOMPILED needs a commit
with a published wheel and main has moved past it. Adds the wheel-probe
loop, plus the two traps hit today: --system-site-packages lets a stock
0.27.1 satisfy the requirement, and a short hash cannot be checked out
from a shallow clone.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5 "Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endpoint

Pairs the fastest model with the best dual-card mode. Results:

  32 agents    1102.6 tok/s - highest single-endpoint throughput measured
  KV pool      5,254,908 tokens, 40.1x concurrency at 128K, the deepest
  TTFT         26.0 ms, down from 77.7 ms on one card
  53K prefill  18.1 s vs 33.6 s, the same pipeline-overlap win as Qwen

The honest caveat is in the 16-agent row: one card does 790 tok/s and PP=2
only 626. The second card pays off past ~16 agents or when the deeper KV
pool is the point, not before. Power rises with load here \(422 W at 32
agents\), the opposite of the 72B TP=2 run where batch drove power down
because the cards blocked on the interconnect - PP=2 keeps both busy.

Also documents the install path that actually works. The recipe we had
published \(@main\) now 404s, because VLLM_USE_PRECOMPILED needs a commit
with a published wheel and main has moved past it. Adds the wheel-probe
loop, plus the two traps hit today: --system-site-packages lets a stock
0.27.1 satisfy the requirement, and a short hash cannot be checked out
from a shallow clone.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5 "Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endpoint

Pairs the fastest model with the best dual-card mode. Results:

  32 agents    1102.6 tok/s - highest single-endpoint throughput measured
  KV pool      5,254,908 tokens, 40.1x concurrency at 128K, the deepest
  TTFT         26.0 ms, down from 77.7 ms on one card
  53K prefill  18.1 s vs 33.6 s, the same pipeline-overlap win as Qwen

The honest caveat is in the 16-agent row: one card does 790 tok/s and PP=2
only 626. The second card pays off past ~16 agents or when the deeper KV
pool is the point, not before. Power rises with load here \(422 W at 32
agents\), the opposite of the 72B TP=2 run where batch drove power down
because the cards blocked on the interconnect - PP=2 keeps both busy.

Also documents the install path that actually works. The recipe we had
published \(@main\) now 404s, because VLLM_USE_PRECOMPILED needs a commit
with a published wheel and main has moved past it. Adds the wheel-probe
loop, plus the two traps hit today: --system-site-packages lets a stock
0.27.1 satisfy the requirement, and a short hash cannot be checked out
from a shallow clone.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Measure two 170HX cards: TP=2 loses 32% of throughput to missing P2P](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/4ae2c009beb60abd774155f61386d1578c1a7101 "Measure two 170HX cards: TP=2 loses 32% of throughput to missing P2P

A second rental put 2x 170HX in one box, so the obvious question -- does
tensor-parallel scale on this card -- finally has a measured answer. It
does not.

Same engine, checkpoint and flags as the single-card 27B run, changing
only --tensor-parallel-size:

  single decode  57.5 -> 70.9 tok/s  \(+23%\)
  16 agents       608 ->  415 tok/s  \(-32%\)
  48K prefill    27.1 -> 47.4 s      \(-75%\)
  KV pool       1.19M -> 2.98M       \(+150%\)

The cause is visible in nvidia-smi topo -p2p: GNS, "GPU not supported".
P2P is fused off in the mining SKU -- not IOMMU, not ACS -- so device-to
-device copies are host-staged at 5.85 GB/s and vLLM disables its custom
allreduce. Allreduce payload grows with batch size, which is why the
penalty lands hardest exactly where the second card was supposed to help.

Two independent TP=1 servers instead reach 754 tok/s at 16 agents and
1253 at 32 -- 3.0x the TP=2 figure on the same silicon. Shard, don't
tensor-parallel, unless the model genuinely needs 128 GB.

TP=1 on this instance also reproduced the published single-card numbers
to 0.1% \(57.47 vs 57.52 tok/s\), which validates the whole results file
across two separate rentals.

bench_power.py summed only one nvidia-smi row and crashed on a two-GPU
box; it now aggregates across visible GPUs, so dual-card power figures
are box totals.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Add gitleaks CI, local hook, and Hub shard hash check.](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/b3165befbed8f14ff672f384f6ed6c0bcb9fcee2 "Add gitleaks CI, local hook, and Hub shard hash check.

Public tree must stay free of IPs, tokens, and rental hosts; GitHub secret scanning and push protection are on for this repo.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Publish CMP 170HX inference lab: 27B MTP and 72B AWQ, scripts, graphs.](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/c68dcebb03ab28c3a7c29279258700c1899160c3 "Publish CMP 170HX inference lab: 27B MTP and 72B AWQ, scripts, graphs.

72B uses the upstream vLLM random ISL/OSL serving bench. Private LinkedIn copy is not in this tree.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endp…](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/88a91bc900f5ac99f8441bf5dde723dffa74d7c5 "Muse Glimmer on two cards at PP=2: 1103 tok/s, the lab's fastest endpoint

Pairs the fastest model with the best dual-card mode. Results:

  32 agents    1102.6 tok/s - highest single-endpoint throughput measured
  KV pool      5,254,908 tokens, 40.1x concurrency at 128K, the deepest
  TTFT         26.0 ms, down from 77.7 ms on one card
  53K prefill  18.1 s vs 33.6 s, the same pipeline-overlap win as Qwen

The honest caveat is in the 16-agent row: one card does 790 tok/s and PP=2
only 626. The second card pays off past ~16 agents or when the deeper KV
pool is the point, not before. Power rises with load here \(422 W at 32
agents\), the opposite of the 72B TP=2 run where batch drove power down
because the cards blocked on the interconnect - PP=2 keeps both busy.

Also documents the install path that actually works. The recipe we had
published \(@main\) now 404s, because VLLM_USE_PRECOMPILED needs a commit
with a published wheel and main has moved past it. Adds the wheel-probe
loop, plus the two traps hit today: --system-site-packages lets a stock
0.27.1 satisfy the requirement, and a short hash cannot be checked out
from a shallow clone.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
|   |   | [Drop the public 'not in this repo' blurb.](https://github.com/dl4rce/cmp-170hx-inference-lab/commit/e260ae4a1ad25dc1ab9821262d62bc789795a42d "Drop the public 'not in this repo' blurb.

That section named private material on the public README; keep the tree about measured numbers only.

Co-authored-by: Cursor <cursoragent@cursor.com>")  | Aug 21, 2026  |  
| View all files |  
## Repository files navigation
# CMP 170HX inference lab
Unlocked NVIDIA **CMP 170HX** (GA100, 64 GB HBM2e) measured as a local LLM card, 2026-08-21.
**Short answer:** after the community unlock, this is a 64 GB Ampere mining SKU, not an A100 and not a 5090. **1 500–1 800 € plus unlock plus hardware plus a real cooler is the hype price, not the measured one.**
Full numbers: [`results/lab-2026-08-21.json`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/results/lab-2026-08-21.json) · scripts: [`scripts/`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/scripts) · unlock caveats: [`docs/UNLOCK-AND-MODS.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNLOCK-AND-MODS.md) · skips: [`docs/UNSUPPORTED.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/UNSUPPORTED.md) · secrets: [`SECURITY.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/SECURITY.md)
This repo has **no hostnames, IPs, accounts, or private checkpoints**. Scripts default to `http://127.0.0.1:8000`.
## Sorted overview (one card)
Public 4-bit checkpoints. Decode is median tok/s, thinking-on unless a gate required otherwise. The Qwen rows are the frozen **vLLM 0.27.1** recipe with FP8 KV; Muse and Gemma need a newer engine and **bf16 KV** (details below).  
| Rank  | Model  | Fit  | Single tok/s  | 16-agent agg  | KV @ native window  | Notes  |  
| --- | --- | --- | --- | --- | --- | --- |  
| 1  |  **Qwen3.6 35B-A3B** AWQ (~3B active)  | 22.9 GiB  | **134.8**  | **1 261**  | **3.16M / 24.1× @ 128K**  | Hybrid GDN+MoE. MTP-3 is _slower_ single-stream (125.6), 16-agent 1 357  |  
| 2  | Qwen3.8 **27B** W4A16 MTP-3  | 18.6 GiB  | **88.3**  | **641**  | 1.01M / 7.73× @ 128K  | Best 27B depth. No-MTP: 57.5 tok/s, 1.20M KV  |  
| 3  |  **Muse Glimmer 30B** W4A16  | 21.0 GiB  | 58.1  | **790**  | 2.64M / 20.1× @ 128K  | Best single-card aggregate; **1 103** on two cards at PP=2. Needs a dev-branch vLLM  |  
| 4  |  **Gemma 4 31B** QAT W4A16  | 19.8 GiB  | 48.8  | 645  | 0.36M / 2.71× @ 128K  | Official QAT checkpoint. Chat endpoint only  |  
| 5  | Qwen2.5 **72B** AWQ  | 38.8 GiB  | **25.8**  | 100 @ 4-wide  | 0.12M / 3.78× @ **32K**  | vLLM refused 128K. Already ~250 W on one stream  |  
| Same 27B recipe on **2× RTX PRO 2000 16 GB** , TP=2, no NVLink: 31 tok/s (off) / 55 tok/s (MTP-3). That box cannot load 72B AWQ.  |  
## Qwen3.6 35B-A3B (the new number)
`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, 6/6 shards hash-checked against Hub LFS. Native 256K, vision encoder present, served as text at 128K.  
| No MTP  | MTP-3  |  
| --- | --- |  
| Load  | 22.87 GiB in 45 s  | 23.38 GiB in 49 s  |  
| KV  | 3 159 025 · 24.10× @ 128K  | 2 502 283 · 19.09× @ 128K  |  
| Single decode  |  **134.8 tok/s** · TTFT 40 ms  | 125.6 tok/s · TTFT 68 ms  |  
| 16 agents  | 1 261 tok/s agg · ~190 W  | 1 357 tok/s agg · ~191 W  |  
| Packing 128K→8K  | 24 / 48 / 96 / 192 / **385**  | 19 / 38 / 76 / 152 / 305  |  
MTP does not win here the way it did on dense 27B. Use **no MTP** unless you are filling 16 slots.
Power: idle ~43 W; 1 agent ~158 W at 135 tok/s. The 250 W cap is not the story on this MoE.
## Qwen3.8 27B (dense, MTP helps)  
| Workload  | Result  |  
| --- | --- |  
| No MTP  | 57.5 tok/s · TTFT 82 ms · KV 1.20M @ 128K  |  
|  **MTP-3** (best depth)  |  **88.3 tok/s** code · 81.8 tok/s prose · 641 tok/s @ 16 agents  |  
| MTP depths 1/2/3/4  | 66.3 / 77.7 / **88.3** / 86.8 tok/s — plateau at 3  |  
| 256K window  | KV 1.10M · 4.2× @ 262144 · exact needle at **200k** tokens  |  
| Real JPEG vision  | “Dog” in **465 ms** TTFT (text is ~100 ms)  |  
| Power  | idle 40 W · 1 agent 205 W · 16 agents 248–253 W  |  
## Muse Glimmer 30B and Gemma 4 31B (what the frozen recipe could not do)
Both were listed as “did not serve” on vLLM 0.27.1. Neither was a VRAM problem — both were engine problems, and both now run on the same card.
What actually had to change:
  1. **A newer vLLM** , installed in a **second venv** so the measured 0.27.1 numbers above stay reproducible. `MuseGlimmer*` and `Gemma4*` are in that build's registry; 0.27.1 has no Muse class at all.
  2. **The right Gemma checkpoint.** The AWQ community repack trips a per-layer `head_dim` mismatch. The official **QAT compressed-tensors** build (`google/gemma-4-31B-it-qat-w4a16-ct`) loads clean.
  3. **`--kv-cache-dtype bfloat16`, not fp8.** Ampere cc 8.0 has no native `fp8e4nv`, so Triton attention refuses FP8 KV outright. This costs pool size and is the single biggest reason Gemma's KV looks small.
  4. **`VLLM_USE_FLASHINFER_SAMPLER=0`.** FlashInfer's JIT could not build its `sm_80` kernels against the CUDA headers in that wheel — it failed for attention _and_ again, later, for sampling. The torch sampler sidesteps it.

  
| Muse Glimmer 30B  | Gemma 4 31B QAT  |  
| --- | --- |  
| Load  | 21.02 GiB in 8.5 s  | 19.78 GiB in 8.6 s  |  
| KV @ 128K  |  **2 635 350** · 20.11×  | 355 128 · 2.71×  |  
| TTFT  | 78 ms  | **29 ms**  |  
| Single decode  | **58.1 tok/s**  | 48.8 tok/s  |  
| 4 / 8 / 16 agents  | 226 / 357 / **790** tok/s  | 184 / 348 / 645 tok/s  |  
| Prefill ~48–53K  | 33.6 s  | 60.9 s  |  
| Power 1 / 16 agents  | 222 W / 249 W  | 235 W / 250 W  |  
**790 tok/s aggregate is the highest single-card number in this lab** — higher than the 35B MoE at 16 agents on raw decode, though the MoE still wins on single-stream latency and KV depth. Given a second card it goes further still: [1 103 tok/s at PP=2](https://github.com/dl4rce/cmp-170hx-inference-lab#muse-on-two-cards-the-fastest-endpoint-in-the-lab).
Gemma caveat: query it through **`/v1/chat/completions`**. Raw`/v1/completions` on this instruction/thinking checkpoint degenerates into repeated tokens. Its 128K window is nominal — with bf16 KV the pool only holds ~2.7 full contexts.
These two rows are **not directly comparable** to the Qwen rows: different engine, and bf16 KV roughly halves the pool against FP8. Serve commands and the accuracy gaps: [`docs/METHODOLOGY.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/METHODOLOGY.md).
## Two cards: TP=2 is a trap without P2P
A second rental put **2× 170HX** in one box (128 GB, both Gen2 ×16, `PHB` topology). Same engine, same checkpoint, same flags as the single-card 27B run — the only variable is how the model is split across the two cards.  
| TP=1 (one card)  | TP=2 (both)  | PP=2 (both)  | Two TP=1 servers  |  
| --- | --- | --- | --- |  
| Single decode  | 57.5 tok/s  |  **70.9** (+23%)  | 57.5  | 57.5  |  
| 16 agents  | 608 tok/s  | 415 (−32%)  | 605  | **754**  |  
| 32 agents  | —  | —  | 895  | **1 253**  |  
| KV pool @128K  | 1.19M / 9.1×  | **2.98M / 22.8×**  | 2.76M / 21.1×  | 1.19M per server  |  
| 48K prefill  | 27.1 s  | 47.4 s (−75%)  |  **14.8 s** (1.8× faster)  | 27.1 s  |  
**TP=2 makes one stream faster and everything else slower.** The crossover is around 3 agents.
The cause is in `nvidia-smi topo -p2p`: **`GNS`— GPU not supported**. Peer-to-peer is fused off in the CMP SKU, not blocked by IOMMU or ACS, so nothing in BIOS or the kernel recovers it. `torch.cuda.can_device_access_peer` is `False` both ways, device-to-device copies are host-staged at **5.85 GB/s** (no faster than the 6.17 GB/s H2D), and vLLM logs it plainly:
> Custom allreduce is disabled because your platform lacks GPU P2P capability
Every layer's allreduce then crawls over the host bridge. Allreduce payload scales with batch size, so the penalty grows exactly where you wanted the second card to help.
**If you are throughput-bound, do not use TP=2.** Run one TP=1 server per card behind a load balancer: **1 253 tok/s vs 415** , a **3.0×** win on identical silicon.
### Pipeline parallel is the setting we should have reached for first
`--pipeline-parallel-size 2` splits the model by **layer** , not by tensor. The only inter-GPU traffic is one activation handoff at the stage boundary, instead of an allreduce inside every layer — precisely the traffic that the dead link punishes. Same checkpoint, same flags, one word changed:  
| vs TP=2  | PP=2 result  |  
| --- | --- |  
| 16 agents  |  **605 tok/s** vs 415 — **1.46×**  |  
| 32 agents  |  **895 tok/s** — TP=2 could not scale here at all  |  
| 48K prefill  |  **14.8 s** vs 47.4 s — **3.2×** faster  |  
| KV pool  | 2.76M tokens (21.1×) — within 7% of TP=2  |  
**PP=2 matches a single card's throughput (605 vs 608 tok/s) while holding 2.3× the KV pool.** That is the combination we wanted from the second card and failed to get from TP=2: full speed _and_ the capacity. Prefill is better than either alternative — even 1.8× faster than one card, since the stages overlap across a long prompt.
The one thing it gives up is TP=2's single-stream gain: 57.5 tok/s, identical to one card. Pipeline parallel does not split per-token weight reads, so a lone request sees no extra bandwidth.
**Practical ranking on a P2P-less box:** sharding for raw throughput → **PP=2** when you want depth without losing speed → TP=2 only for single-stream latency or a model that will not fit.
### Muse on two cards: the fastest endpoint in the lab
Pairing the best model with the best dual-card mode. Muse Glimmer 30B at PP=2, bf16 KV, dev-branch vLLM:  
| one card  | PP=2 (both)  |  
| --- | --- |  
| 16 agents  | **790 tok/s**  | 626  |  
| 32 agents  | —  | **1 103 tok/s**  |  
| KV pool @128K  | 2.64M / 20.1×  | **5.25M / 40.1×**  |  
| TTFT  | 77.7 ms  | **26.0 ms**  |  
| 53K prefill  | 33.6 s  | **18.1 s**  |  
**1 103 tok/s at 32 agents is the highest single-endpoint throughput measured here** , and the 5.25M-token KV pool is the deepest — 40 full 128K contexts resident at once, even though cc 8.0 forces bf16 KV on this checkpoint. Latency improves too: TTFT drops to **26 ms** from 77.7 ms, and a 53K prefill runs **1.9× faster** (18.1 s vs 33.6 s), the same pipeline-overlap effect seen on Qwen.
But note the row that does not flatter the second card: **at 16 agents one card is faster** (790 vs 626). The second card only pays off past ~16 concurrent agents, or when you want the deeper pool. Power confirms PP=2 keeps both cards genuinely working — 422 W at 32 agents, rising with load, the opposite of the 72B TP=2 run where heavy batch drove power _down_ because the cards sat blocked on the interconnect.
### Does the second card earn its keep on a 72B?
We asserted "use TP=2 when it does not fit", then measured it. Qwen2.5 72B AWQ (39 GiB of weights) at TP=2, 32K window, FP8 KV:  
| one card  | two cards, TP=2  |  
| --- | --- |  
| Single-stream decode  | 25.8 tok/s  |  **37.1 tok/s** (+44%)  |  
| KV pool  | cramped at 32K  |  **494 848 tokens** (15.1× concurrency)  |  
| 16 agents  | —  | 275 tok/s (17.8 per agent)  |  
| 32 agents  | —  | 360 tok/s (11.6 per agent)  |  
| 24K prefill  | —  | 51.0 s  |  
The second card buys **headroom, not speed**. 360 tok/s at 32 agents is still below what a _single_ card does with 27B (608 tok/s at 16). Power inverts what you would guess: **437 W** median on one stream, **304 W** at 32 agents — under batch the cards idle on the host-staged allreduce instead of computing.
**The context window is not a VRAM problem.** Asking for 128K was rejected outright: this checkpoint declares `max_position_embeddings=32768`, and the advertised 128K needs YaRN rope-scaling turned on explicitly. The 32K ceiling on one card was the _model's_ limit all along. A second card does not extend context — only the KV pool at a given length.
Full data: [`results/dual-card-2026-08-21.json`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/results/dual-card-2026-08-21.json).
## Qwen2.5 72B AWQ (it fits, it crawls)
**38.8 GiB** in 20 s · **25.8 tok/s** · native **32K** only · 250 W on one stream.
The 32K is the checkpoint's own `max_position_embeddings`, not a VRAM ceiling — confirmed when [two cards refused 128K too](https://github.com/dl4rce/cmp-170hx-inference-lab#does-the-second-card-earn-its-keep-on-a-72b). A second card lifts this to 37.1 tok/s.
vLLM random ISL/OSL: 25 / 23 / 18 / **10** tok/s at 128 / 512 / 2k / 8k →128; 4-wide **79** tok/s.
70B-class write-up: [`docs/70B.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/70B.md).
PCIe ×16 vs ×4: H2D **6.63 vs 1.7 GB/s**. Decode and prefill did **not** move. Pay for the solder mod only if you care about load, KV restore, and vision H2D.
## Should you pay 1 500–1 800 €?
Prices jumped when the unlock went public ([Wccftech](https://wccftech.com/nvidia-cmp-170hx-8-10-gb-prices-explode-over-1000-usd-as-tool-unlocks-hidden-64-80gb-vram/)). Listings in that band are selling an A100 story.
You still need:
  1. **Software unlock** ([cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)) — or a seller who already did it. Verify `65536 MiB` _and_ a compute microbench. 8 GB SKU → 64 GB; 10 GB SKU → ~40 GB stable, not 80 GB.
  2. **Optional hardware:** 24× 0402 AC-coupling capacitors for PCIe ×16. Gen2 _speed_ is software; _width_ is solder ([wiki](https://github.com/Consensus-Protocol/cmp170hx)).
  3. **Cooling.** 250 W, mining blower, no display. A silent desktop without a proper cooler is the part the price tag pretends is free.


Reddit field threads (hype included):
  * [Unlock 8 GB → 64 GB](https://www.reddit.com/r/LocalLLaMA/comments/1v0a7wn/)


If you want 64 GB in one slot for a lab, this card can make sense **far below** A100 money. If you want a 24/7 27B appliance, a 70 W-class dual-16GB box is still the boring winner on tokens/watt. If you want **~30B-class MoE** at 135 tok/s and a 3M-token KV pool, the 170HX suddenly looks like the point of 64 GB.
## Reproduce
See [`docs/METHODOLOGY.md`](https://github.com/dl4rce/cmp-170hx-inference-lab/blob/main/docs/METHODOLOGY.md). Figures:

```
python3 scripts/plot_results.py
```

After a Hub download, hash-check shards:

```
python3 scripts/verify_hf_dir.py --repo org/name --dir /path/to/snapshot
```

## About
CMP 170HX LLM inference lab: vLLM scripts, measured results, graphs. Not an A100. Unlock/mod caveats. 
### Resources
### Security policy
### Stars
**0** stars
### Watchers
**0** watching
### Forks
[ forks](https://github.com/dl4rce/cmp-170hx-inference-lab/forks)
## 
No releases published
## 
  * [**cursoragent** Cursor Agent](https://github.com/cursoragent)
  * [**dl4rce** Volker](https://github.com/dl4rce)


## Languages
  * [Python92.4%](https://github.com/dl4rce/cmp-170hx-inference-lab/search?l=python)
  * [Shell7.6%](https://github.com/dl4rce/cmp-170hx-inference-lab/search?l=shell)


You can’t perform that action at this time. 
