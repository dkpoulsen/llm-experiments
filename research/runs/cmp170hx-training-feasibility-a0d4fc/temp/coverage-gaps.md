# Coverage gaps — cmp170hx-training-feasibility-a0d4fc

Wave 1: 10 fetchers, 80 seed URLs → **133 notes** (129 substantive, 4 deprecated), **921 claims ingested**.
Wave 2 targeted 5 gap URLs.

## Atomic-item coverage after wave 1

| Atomic item | Status | Sources | Notes |
|---|---|---|---|
| Sub-Q1 negotiated link width/speed | **Well-covered** | 9 | club-170hx HARDWARE + CMP170HX-UNDERSTANDING, it-still-boots specs, bayley/cmpunlocker, dl4rce UNLOCK-AND-MODS, wpnews gist, niconiconi, Consensus-Protocol, 170th-street |
| Sub-Q2 PWRBRK# power brake | **Adequate** | 4 | club-170hx LESSONS (155.7 vs 39.3 TFLOPS), deepwiki, niconiconi, club HARDWARE |
| Sub-Q3 training topologies | **Well-covered** | 14 | FSDP paper, ZeRO, AMSP, ZeroPP, veScale-FSDP, Mobius, RoundPipe, Megatron-LM, BF-PP, model-parallel comm, TDS wiring, HF parallelism, FSDP docs, MiCS |
| Sub-Q4 DDP overhead model | **Well-covered** | 16 | default_hooks, no_sync forum, PowerSGD, ZipCCL, GeoFP8, AGoQ, CERAR, FP8-LM, COAT, dietgpu, NCCL docs, Demystifying NCCL, critical batch size, small-batch, empirical large-batch, TCCL |
| Sub-Q5 expert parallelism | **Well-covered** | 14 | ThunderEP, RoundPipe, CoMoE, VarioPath, GPU-initiated comm, RelayMoE, Head Parallel, MegaBlocks, Megatron-Core MoE, Pangu, DeepSeek-V3, Routing-Free MoE, Fire-Flyer, club CLUSTER |
| Sub-Q6 tensor parallelism | **Well-covered** | 8 | club TOPOLOGY, SETTINGS, dl4rce, Megatron-LM, model-parallel comm, ZeroPP, DeepSeek-V3, promisezackr PP8 |
| Sub-Q7 Engram / Memory Grafting | **Well-covered** | 13 | Engram canonical, Memory Grafting, CXL pooling, TF-Engram, EngramEdit, FactorEngram, drli critique, vLLM docs, Engram demo code, Engram repo, LongCat, UltraMemV2, OverEncoding |
| Sub-Q8 CSA2 sparse attention | **Well-covered** | 16 | V4.1-Flash paper + HF card, StreamIndex, LongCat, IndexCache, KV-Kaizen, LISA, HySparse2, cuDNN DSA, V4, V3.2, NSA, You-Only-Cache-Once, periodic weak spots, KV pitfalls, V4-Flash AMD |
| Sub-Q9 dense baseline / TFLOPS | **Well-covered** | 18 | Qwen3, A100 datasheet, niconiconi, crypto-mining-GPUs, 170th-street, it-still-boots overview, club BENCHMARKS + LESSONS, Cramming, Chinchilla, Mesa, mixed-precision optimizers, 8-bit optimizers, Adafactor, ICLR-2018 mixed precision, memory-efficient-LLM-training |
| Sub-Q10 data pipeline | **Adequate** | 10 | fineweb-edu card, finemath card, smollm-corpus, SmolLM2, FineWeb paper, sequence composition, Cramming, data-constrained scaling, Chinchilla, classifier gist |
| Entity: bf16_compress_hook | **Well-covered** | 5 | default_hooks source, PowerSGD, ZipCCL, GeoFP8, AGoQ |
| Entity: 1M-token global batches | **Well-covered** | 6 | critical batch size, empirical large-batch model, small-batch, SmolLM2, FineWeb, Chinchilla |
| Entity: 32k training context | **Well-covered** | 9 | Qwen3, V4.1, LongCat, HySparse2, NSA, StreamIndex, periodic weak spots, You-Only-Cache-Once, it-still-boots sweep |

## Genuine gaps (flagged, not silently dropped)

1. **Stack-Edu dataset page never fetched.** Two attempts: `FETCH_ERROR` (browser driver closed) then `JUNK_CONTENT` (empty shell). Token counts are covered indirectly by the SmolLM2 paper (~125B tokens, 15 languages) and partially by `smollm-corpus` (python-edu only). **Wave 2 retries this.**
2. **No source measures NCCL all-reduce at the repo's 0.75 GiB/s across 5 GPUs.** Nearest proxies: ~1.0 GB/s bus bandwidth (SETTINGS.md), 3.69 GB/s SHM all-reduce at 256 MiB on a 4-card PLX host (club README), 5.85 GB/s host-staged D2D (dl4rce). These are different quantities. The repo's figure is user-measured and cannot be externally validated — the report must say so.
3. **The repo's "~70× compute time" EP estimate is unvalidated.** ThunderEP gives the ring-traffic multiplier (1.6× at N=5), VarioPath gives 1.7–5.9× off hardware limit, DeepSeek-V3 gives ~1:1 compute:comm on NVLink+IB. No source reproduces 70×. **Treat as estimate-tier, not measured.**
4. **Link-width effect on *training* collectives is unmeasured.** dl4rce's x16-vs-x4 comparison was inference-only (decode/prefill unchanged, H2D 6.63 vs 1.7 GB/s). Repo open question #1 stays open for DDP gradient traffic.
5. **CSA2 has no SM80 kernel path.** cuDNN DSA is SM90/SM100-only; the only Ampere-runnable path is the pure-PyTorch reference in `cudnn-frontend`. **Wave 2 fetches it.**
6. **Frozen-table evidence is thin on the "does it help at 0.9B" question.** Memory Grafting reports 0.92B gains, but the canonical Engram and every production memory model train their tables. **Wave 2 fetches Delta N-gram (arXiv 2610.31125) as the strongest frozen-table counter-datapoint.**
7. **ACL PDFs are login-walled in this environment** (SimSMoE, sequence-composition). Both were recovered via arXiv/landing-page substitutes; SimSMoE remains abstract-only.
8. **Escalation queue holds 6 bot-blocked DOIs.** All were substituted with open-access copies (PVLDB PDF, arXiv). No content loss.

## Retractions

`sources score` → `"retracted": []`. No retracted source in the corpus.

## Wave 2 URLs

1. `https://huggingface.co/datasets/HuggingFaceTB/stack-edu` — gap 1
2. `https://arxiv.org/abs/2610.31125` — gap 6 (Delta N-gram)
3. `https://github.com/NVIDIA/cudnn-frontend/blob/main/python/dsa_reference.py` — gap 5
4. `https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/hardware/pcie.md` — gap 4
5. `https://arxiv.org/abs/2603.28458` — CSA2 indexer prior work
