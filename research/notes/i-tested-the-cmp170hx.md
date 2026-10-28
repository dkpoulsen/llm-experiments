---
title: I tested the CMP170HX
id: i-tested-the-cmp170hx
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-2
- pcie-gen2-x4
- cmp-170hx
- link-rate
- measured
- community-reported
created: '2026-10-27T14:41:24.541183Z'
updated: '2026-10-27T19:02:00.181015Z'
source: https://bittide.aicompass.dev/article/3ea0180a-5bc9-4164-a4c4-f1ed14046b13
source_domain: bittide.aicompass.dev
fetched_at: '2026-10-27T14:41:24.540397Z'
fetch_provider: builtin
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'bittide.aicompass.dev ''I tested the CMP170HX'' (1,055 words) — a 4x CMP
  170HX (GA100, 70 SM, 64 GB each, 150 W) llama.cpp serving report on PCIe Gen2 x4
  with no NVLink, cards mounted through a single x16-to-4x4 M2 bifurcation adapter.
  [T2 community measurement on this exact hardware at this exact width.] The load-bearing
  line for the fabric question: ''the transfer rate is 800MB/s at Gen 1 and 1.6GB/s
  at Gen 2'' — i.e. the measured Gen2 x4 link rate on these cards is ~1.6 GB/s, which
  is the hard ceiling any 5-rank ring all-reduce must live under (busbw <= link rate;
  algbw <= 0.8 x link rate at N=5, so algbw <= ~1.0 GB/s). Serving numbers: 1 card
  gpt-oss-120B MXFP4 78 tg / 1,244 pp; 2 cards 85 tg / 1,870 pp; 3 cards DeepSeek
  V4-Flash 0731 Q4_K_XL ~29 tg / ~365 pp at 1M ctx; 4 cards same model 29 tg / ~450
  pp; dense >30B is a ''dead end''. No NCCL/all-reduce measurement — this is a link-rate
  datapoint, not a collective receipt, so it bounds the fabric but does not by itself
  settle NCCL efficiency at 5 ranks.'
utility_score: 14.0
---

I tested the CMP170HX
BitTide
Menu
Home
Newest
Models
Tools
News
Products
Papers
Events
Daily Report
Search
Subscribe
中文
Sign in
Summary
A hands-on benchmark of Nvidia CMP170HX mining cards repurposed as 64GB VRAM AI inference accelerators, showing they can run large local LLMs like DeepSeek V4-Flash and gpt-oss-120B at useful speeds, with caveats around Ampere-class throughput and PCIe Gen2 x4 connectivity.
Lots of rumor and misinfo bouncing around, so I put some of these old mining cards to the test. I used 4 of the 8GB cards, set to 64GB each. Lots of models fit entirely on a single card, and you can also run several small models at the same time on a single card, as long as their combined VRAM usage is 64GB or less. A setup like comfyui taking 10-12GB and qwen on another 30-40GB works fine all on the same card. I did not exhaustively show results of tiny 8B or 12B running at several hundred t/s, because it is better to run larger, smarter models. What follows is an AI summary of a bunch of different tests on recent interesting models to give a clear overview of what the cards can do. I am not a reseller, just had a handful of these collecting dust. If you picked them up at $200, you won the lottery. If you are considering a purchase now, you have to decide if you are happy with 30xx (Ampere) class performance. It might make sense because of the huge VRAM, but you may want to hold out for Hopper or Blackwell. I can confirm the 8GB run fine at 64GB and the 10GB run fine at 40GB with higher memory throughput. I am only using the 8G cards here because it's a pain to re-rack the server and from my testing there is not much difference. I personally don't see an issue with x4 PCIE - the transfer rate is 800MB/s at Gen 1 and 1.6GB/s at Gen 2. The only time I ever noticed it was loading large models, but since my SSD reads at 550MB/s I could not saturate the PCIE link until I put models on NVME. The upside to x4 PCIE is that I have these 4 GPU installed through a single x16 to 4x4 M2 drive adapter (4x4 bifurcation, M2 to PCIE risers) so my little PC could potentially run 16 of the cards for a full TB of VRAM if I filled all 4 x16 slots with M2 adapter cards. Running local LLMs on 4× cut-down A100 mining cards (GA100, 70 SMs, 64 GB each = 256 GB), ~1215 GB/s HBM, PCIe Gen2 ×4, no NVLink, 150 W power limit. llama.cpp, -sm layer. Numbers are single-stream server measurements (tg = token gen, pp = prefill), f16 KV unless noted. 1 card Model Quant · active tg pp Max ctx Notes gpt-oss-20B MXFP4 · 3.6B MoE 120 2000 131K 503 t/s batched; launch-overhead bound single-stream gpt-oss-120B MXFP4 · 5.1B MoE 78 1244 131K q8 KV; fastest capable coder Qwen3.6-35B-A3B Q6_K +MTP · 3B MoE 110 1700 262K little-MoE default (MTP tg optimistic) Qwen3.6-27B Q5_K_M +MTP · dense 47 812 262K 29 tg without MTP gemma-4-31B Q8_0 · dense 23 767 262K Q8 beats Q6_K on speed and quality 2 card Model Quant · active tg pp Max ctx Notes gpt-oss-120B MXFP4 · 5.1B MoE 85 1870 131K 2nd card buys +48% pp only, tg slightly improved Laguna-S 2.1 Q4_K_M · 8B MoE 59 968 262K "just works" fork, fast GLM-4.5-Air Q6_K · 12B MoE 39 1181 131K smart 2-card partner MiniMax-M2.7 IQ4_XS · 10B MoE 38 800 160K q8 KV; only 4-bit fit on 2 cards Gemma-4-31B-StyleTune Q8_0 · dense 23 ~760 131K 50 ms warm TTFT swa full, mem hog Mistral-Medium-3.5 128B Q4_K_XL · dense 128B 9.8 200 262K Dense >30B dead end, too slow 3 card Model Quant · active tg pp Max ctx Notes DeepSeek V4-Flash 0731 Q4_K_XL · 13B MoE (MLA) ~29 ~365 1M plain no-spec; huge ctx, tiny MLA KV MiniMax-M2.7 Q4_K_M · 10B MoE 47 1135 192K beats the IQ4_XS (+29% pp, better quality) Hy3 Q4_K_M · 16B MoE 29.5 315 65K ? deleted ? V4-Flash speed with 16× less ctx DeepSeek + DSpark drafter does not fit 1M on 3 cards - loads at ~99% VRAM but OOM-crashes on a large prefill (died at 16K of 262K tokens). The 11 GB drafter needs the 4th card at 1M, or cap ctx to ~512-768K.* 4 card Model Quant · active tg pp Max ctx Notes DeepSeek V4-Flash 0731 Q4_K_XL · 13B MoE 29 ~450 1M plain no-spec same + BF16 DSpark drafter speculative 33 400 1M 37 code / 28 prose *GGUFs from unsloth, bartowski, lmstudio-community, poolside. Many models were tested then deleted (quality or a better alternative)
0
Like
0
Add to favorites
Original Article
Similar Articles
CMP170Hx “Spark” Machine
Reddit r/LocalLLaMA
The author shares their DIY CUDA box setup using CMP170 mining cards to run AI models like Qwen Flash Next, achieving strong inference performance.
PSA: Nvidia's CMP 170HX Full Compute and Memory(80GB) may be unlockable via exploit
Reddit r/LocalLLaMA
A potential exploit in Nvidia's Falcon security processor may allow unlocking the crippled CMP 170HX crypto-mining GPU into a full A100 80GB, potentially making high-end AI hardware available for under $1000.
768gb vram for less than the price of one RTX 6000
Reddit r/LocalLLaMA
A user built a 768GB VRAM system using 12x64GB CMP170HX cards for less than the cost of one RTX 6000 Pro, enabling local inference of various large AI models with strong performance.
@ciruai: Testing DeepSeek v4 Flash on the AMD Ryzen AI Max+ 395 Strix Halo with 128GB RAM. Getting ~15 TPS over a decently long …
X AI KOLs Timeline
Testing DeepSeek v4 Flash on the AMD Ryzen AI Max+ 395 with 128GB RAM achieves ~15 TPS for a 284B MoE model (13B active) locally, costing $3,000 versus $25,000+ for a datacenter setup, highlighting the feasibility of running large models on consumer hardware.
Attn CMP170hx 10Gb card owners: unlock from 40Gb —> 48Gb coming along nicely.
Reddit r/LocalLLaMA
Owners of the NVIDIA CMP170HX 10Gb graphics card can look forward to a VRAM unlock increasing memory from 40GB to 48GB, with the mod expected to arrive soon.
Feedback
×
Submit Feedback
Thank you for your feedback!
Submit