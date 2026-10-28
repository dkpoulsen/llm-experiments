---
title: Overview - CMP 170HX Documentation
id: overview-cmp-170hx-documentation
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- benchmarks
- context-scaling
- pcie-link
- measured
created: '2026-10-27T10:53:36.508252Z'
updated: '2026-10-27T19:01:49.116350Z'
source: https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/
source_domain: it-still-boots.github.io
fetched_at: '2026-10-27T10:53:35.885074Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: docs
deprecated: false
summary: 'it-still-boots.github.io CMP 170HX Documentation — Performance Overview
  page (1,515 words): a 1,603-point decode sweep across 16 models at 1k-128k prompt
  depth on ONE unlocked CMP 170HX (64 GiB, ~1.49 TB/s, PCIe Gen2 x4, 150-300 W) in
  vLLM, llama.cpp and SGLang. Inference-only, but it supplies the two numbers this
  run most needs about the exact fabric in the query. First, the link: ''PCIe trained
  at Gen1 x4 (0.84 GB/s) for runs before 2026-09-10, Gen2 x4 (1.69 GB/s) after a reseat''
  — so Gen2 x4 measures 1.69 GB/s per card per direction, which corroborates bayley/cmpunlocker''s
  1.68 GB/s BAR1-P2P figure and pins the query''s ''Gen2 x4'' premise to a real measured
  box. A matched SGLang ladder on the new link reproduced every depth within 0.3%,
  i.e. decode is essentially link-insensitive at single-card scale. Second, context-depth
  scaling, which is the empirical baseline CSA2 has to beat at 32k: median decode
  retained vs the 1k rate is 99.0% at 4k, 96.2% at 16k, 93.5% at 32k, 88.1% at 64k,
  79.4% at 128k (75 measurements per depth to 64k, 56 at 128k). Dense attention therefore
  costs only ~6.5% of decode throughput at 32k on this card, so a sparse-attention
  scheme must beat a ~6.5% penalty plus its own indexer overhead to pay off at 32k.
  Power-limit ladder (210 matched measurements): 250 W keeps 99.3% of the 300 W rate,
  200 W 95.7%, 150 W 85.2%. Speculative decoding by depth: DFlash 2.22x at 1k falling
  to 1.22x at 128k, MTP 1.88x to 0.76x (drafter costs more than it returns at 128k),
  DSpark 2.06x to 1.40x. Concurrency: Nemotron-3.5-Lightning-30B-A3B W4A16 at 32k
  reaches 1,306 tok/s aggregate at 16 streams (237 at n=1) with TTFT 3.3 s -> 31.0
  s; Ornith-1.5-35B W4A16 on SGLang peaks at 270 tok/s at n=2 then falls as queues
  grow. Quality (Qwen3.8-27B): W4A16 vs bf16 loses 1.3 points HumanEval pass@1 (0.927
  vs 0.939) and 8.7 points GSM8K strict (0.444 vs 0.532).'
utility_score: 14.0
---

# Performance Overview[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#performance-overview "Permanent link")
Lab decode numbers from an unlocked **CMP 170HX** (64 GiB HBM2e, ~1.49 TB/s, PCIe Gen2 x4, 150-300 W) running **vLLM** , **llama.cpp** , and **SGLang**. The source sweep covers **1,603 decode points** across **16 models** at prompt depths from 1k to 128k. Workload numbers below focus on 16k, 32k, and 64k.  
| GPU  | VRAM  | Bandwidth  | PCIe  | Engines  | Power  |  
| --- | --- | --- | --- | --- | --- |  
| GA100 sm_80  | 64 GiB HBM2e  | ~1.49 TB/s  | Gen2 x4  | vLLM, llama.cpp, SGLang  | 150-300 W  |  
## Headline numbers[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#headline-numbers "Permanent link")  
| Metric  | Value  |  
| --- | --- |  
| 16k coding  |  **228 tok/s** (Nemotron-3.5-Lightning-30B-A3B W4A16, vLLM, 1,024-token output, 1.7 s TTFT)  |  
| 64k longcode  |  **218 tok/s** (same model, 2,048-token output, 7.8 s TTFT)  |  
| 32k batch  |  **1,306 tok/s** aggregate (Nemotron W4A16, vLLM, 16 streams at 32k prompts)  |  
| At 128k  |  **79%** of the 1k-context decode rate (speculation off)  |  
## What to run[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#what-to-run "Permanent link")  
| Job  | Model  | Engine  | 16k  | 32k  | 64k  |  
| --- | --- | --- | --- | --- | --- |  
| Interactive  | Nemotron-3.5-Lightning-30B-A3B W4A16  | vLLM  | 228 tok/s, 1.7 s TTFT  | 218 tok/s, 7.8 s TTFT  |  
| Fast MoE  | Ornith-1.5-35B-A3B W4A16  | SGLang  | 155 tok/s, 1.3 s TTFT  | 142 tok/s, 7.8 s TTFT  |  
| Batch  | Qwen3.8-27B W4A16-fast  | vLLM  | 602 tok/s, 32 streams  |  
| Batch  | Nemotron-3.5-Lightning-30B-A3B W4A16  | vLLM  | 1,306 tok/s, 16 streams  | n=32 did not fill (17 running)  |  
Ornith at 32k on SGLang peaks at 270 tok/s with two streams.
## Coding and long-context generation[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#coding-and-long-context-generation "Permanent link")
Decode and TTFT with speculation off, 300 W, coding-style prompts.  
| Model  | Quant  | Engine  | 16k decode  | 16k TTFT  | 64k decode  | 64k TTFT  |  
| --- | --- | --- | --- | --- | --- | --- |  
| Nemotron-3.5-Lightning-30B-A3B-W4A16  | W4A16  | vLLM  | 227.7  | 1.7 s  | 217.9  | 7.8 s  |  
| Ornith-1.5-35B-A3B-W4A16  | W4A16  | SGLang  | 154.8  | 1.3 s  | 141.9  | 7.8 s  |  
| Qwen3.6-35B-A3B-FP8  | FP8  | vLLM  | 153.8  | 1.6 s  | 135.9  | 9.1 s  |  
| Ornith-1.5-35B-A3B-W4A16  | W4A16  | vLLM  | 140.2  | 1.5 s  | 125.5  | 8.5 s  |  
| Qwen3.8-27B-INT8-W8A16-MTP  | INT8-W8A16  | vLLM  | 43.5  | 8.6 s  | 39.2  | 40.5 s  |  
| Ornith-1.5-35B-A3B-GGUF  | Q5_K_M  | llama.cpp  | 108.9  | 9.1 s  | 99.7  | 42.2 s  |  
## Context depth[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#context-depth "Permanent link")
Decode against prompt depth with speculation off. The chart shows the three fastest families from the model ladder. The table is the median retained rate across every model and engine that ran both that depth and the 1k reference.
_Figure: Nospec decode versus prompt depth for Qwen3.6-35B FP8, Ornith-1.5-35B W4A16, and Qwen3.8-27B W4A16 at 1k, 4k, 16k, 32k, 64k, and 128k on the unlocked CMP 170HX at 300 W._  
| Depth  | Decode retained vs 1k  | Measurements  |  
| --- | --- | --- |  
| 1,024  | 100.0%  | 75  |  
| 4,096  | 99.0%  | 75  |  
| 16,384  | 96.2%  | 75  |  
| 32,768  | 93.5%  | 75  |  
| 65,536  | 88.1%  | 75  |  
| 131,072  | 79.4%  | 56  |  
Median decode is 96.2% of the 1k rate at 16k and 79.4% at 128k.
## Power limit[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#power-limit "Permanent link")
Every ladder point was repeated at 300, 250, 200, and 150 W (210 matched measurements).
_Figure: Median decode retained versus the 300 W baseline at 300 W, 250 W, 200 W, and 150 W on the unlocked CMP 170HX._  
| Limit  | Decode retained  | p10-p90  | Power budget  |  
| --- | --- | --- | --- |  
| 300 W  | 100.0%  | 100.0-100.0%  | 100%  |  
| 250 W  | 99.3%  | 96.1-101.3%  | 83%  |  
| 200 W  | 95.7%  | 89.8-100.1%  | 67%  |  
| 150 W  | 85.2%  | 77.4-97.6%  | 50%  |  
250 W keeps 99.3% of the 300 W rate. 200 W keeps 95.7% on two-thirds the power. 150 W keeps 85.2%.
## Engines compared[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#engines-compared "Permanent link")
Same underlying Ornith-1.5-35B-A3B at ~4-5 bit, same depth, 300 W, speculation off. llama.cpp is GGUF; vLLM and SGLang are compressed-tensors. The weight files are not identical.
_Figure: Ornith-1.5-35B-A3B at roughly 4 to 5 bit, speculation off, decode across llama.cpp, vLLM, and SGLang from 1k to 128k prompt depth._
SGLang leads on the MoE W4A16 path (163.6 tok/s at 1k, 126.7 at 128k). vLLM sits a step behind on the same model. llama.cpp lands around 70% of SGLang on this 35B MoE.
Compact Qwen3.8-27B 4-bit row at 16k for engine orientation: llama.cpp 36.8, vLLM 62.4, SGLang 67.0 tok/s.
## Speculative decoding[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#speculative-decoding "Permanent link")
Median gain over the same configuration with speculation off, by depth.
_Figure: Median speculative speedup for DFlash, MTP, and DSpark versus prompt depth from 1k to 128k, relative to the matched nospec run._  
| Method  | 1,024  | 4,096  | 16,384  | 32,768  | 65,536  | 131,072  |  
| --- | --- | --- | --- | --- | --- | --- |  
| DFlash  | 2.22×  | 2.02×  | 2.01×  | 1.79×  | 1.56×  | 1.22×  |  
| MTP  | 1.88×  | 1.90×  | 1.61×  | 1.42×  | 1.28×  | 0.76×  |  
| DSpark  | 2.06×  | 2.11×  | 1.53×  | 1.43×  | 1.52×  | 1.40×  |  
DFlash is 2.22× at 1k and 1.22× at 128k. MTP is 0.76× at 128k, so the drafter costs more than it returns at that depth.
## Concurrent decode (workload)[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#concurrent-decode-workload "Permanent link")
256-token outputs. The long-prompt numbers are the 16k and 32k tables. Ornith-1.5-35B W4A16 reaches 3,168 tok/s at 64 streams and 1k. At 32k the same model peaks at 270 tok/s with two streams.
_Figure: Aggregate decode versus concurrent stream count (n=1..16) for Nemotron at 32k on vLLM, Ornith at 32k on SGLang, and Qwen 27B at 16k on vLLM._
### vLLM Nemotron 30B-A3B, 32k context[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#vllm-nemotron-30b-a3b-32k-context "Permanent link")  
| n=1  | n=2  | n=4  | n=8  | n=16  |  
| --- | --- | --- | --- | --- |  
| decode agg  | 237  | 377  | 604  | 934  | 1,306  |  
| per-stream  | 240  | 124  | 65  | 34  | 17  |  
| TTFT  | 3.3 s  | 5.2 s  | 8.7 s  | 16.0 s  | 31.0 s  |  
| KV / running  | 0.6% / 1  | 1.2% / 2  | 2.5% / 4  | 4.9% / 8  | 9.9% / 16  |  
Ornith 32k on SGLang peaks at **270 tok/s aggregate at n=2** , then falls as waiting queues grow. Qwen3.8-27B W4A16-fast at 16k on vLLM reaches **602 tok/s at n=32**.
## Quality (Qwen3.8-27B, this card)[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#quality-qwen38-27b-this-card "Permanent link")  
| Benchmark  | n  | bf16  | INT8  | W4A16  | W4A16 vs bf16  |  
| --- | --- | --- | --- | --- | --- |  
| HumanEval pass@1  | 164  | 0.939  | 0.939  | 0.927  | −1.3%  |  
| HumanEval+ pass@1  | 164  | 0.762  | 0.707  | 0.677  | −11.2%  |  
| GSM8K flexible  | 1,319  | 0.544  | 0.531  | 0.529  | −2.8%  |  
| GSM8K strict  | 1,319  | 0.532  | 0.532  | 0.444  | −16.4%  |  
| IFEval inst strict  | 541  | 0.447  | 0.448  | 0.446  | −0.3%  |  
W4A16 loses 1.3 points of HumanEval pass@1 and 8.7 points of GSM8K strict-match. Flexible-extract, which accepts the number anywhere in the answer, moves about 1.5 points. The GSM8K and HumanEval+ levels come from few-shot completion prompts built for base models, so they sit below a direct chat prompt. The gap between the three quants is the comparison.
## Method[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#method "Permanent link")
  * One CMP 170HX, blower on a temperature-driven curve. Runs without it were discarded.
  * Identical serving flags per engine, `--gpu-memory-utilization 0.85` throughout.
  * Each ladder ran three times after one discarded warmup pass.
  * Prompt depths calibrated against the server's own tokenizer.
  * Output length forced with `ignore_eos` and `min_tokens`.
  * Wall-clock decode is replaced when it disagrees with the server's timing by more than 2×.
  * PCIe trained at Gen1 x4 (0.84 GB/s) for runs before 2026-09-10, Gen2 x4 (1.69 GB/s) after a reseat. A matched SGLang ladder on the new link reproduced every depth within 0.3%.
  * Every run writes its prompts and completions to disk.


## Further reading[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#further-reading "Permanent link")
Community reports and tools on [Localmaxxing](https://www.localmaxxing.com/en/reports):
  * [Community reports index](https://www.localmaxxing.com/en/reports) (includes **CMP 170HX: Qwen3.8-27B AutoRound + DFlash2 at 131k**)
  * [Hardware hub](https://www.localmaxxing.com/en/hardware) (CMP 170HX listed with community runs; single-GPU median around 71 tok/s across a large sample)


Optional stack-comparison context from the same reports index on dual **RTX 3090 TP2** (separate hardware from this 170HX lab): MTP vs DFlash, and vLLM vs SGLang DFlash2.
## References[¶](https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/#references "Permanent link")
  * <https://www.localmaxxing.com/en/reports>
  * <https://www.localmaxxing.com/en/hardware>
  * <https://www.localmaxxing.com/en/decode-calculator>
  * <https://github.com/amoghmunikote/cmpunlocker>
  * <https://170th-street.gitbook.io/hx>


Back to top 
