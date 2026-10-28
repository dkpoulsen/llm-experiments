# Scored URL queue — 80 URLs, 10 batches

Scoring: authority / novelty / stance / coverage / redundancy / freshness (0–3 each, max 18).

| Batch | URL | Score | Serves |
|---|---|---:|---|
| 1 | https://arxiv.org/abs/2601.07372 | 18 | Sub-Q7 Engram canonical |
| 1 | https://arxiv.org/abs/2605.20948 | 18 | Sub-Q7 Memory Grafting canonical |
| 1 | https://arxiv.org/abs/2603.10087 | 14 | Sub-Q7 CXL pooling cost |
| 1 | https://arxiv.org/abs/2607.07388 | 14 | Sub-Q7 SSD-backed lookup latency |
| 1 | https://arxiv.org/abs/2610.10533 | 12 | Sub-Q7 EngramEdit updatability |
| 1 | https://arxiv.org/abs/2609.35578 | 12 | Sub-Q7 FactorEngram variant |
| 1 | https://drli.blog/posts/analysis-engram-conditional-memory/ | 13 | Sub-Q7 adversarial |
| 1 | https://docs.vllm.ai/en/v0.31.0/features/engram/ | 12 | Sub-Q7 runtime cost |
| 2 | https://arxiv.org/abs/2609.19969 | 18 | Sub-Q8 CSA2 canonical |
| 2 | https://arxiv.org/abs/2609.37988 | 12 | Sub-Q8 KV compression choices |
| 2 | https://arxiv.org/abs/2609.26368 | 12 | Sub-Q8 two-level KV sharing |
| 2 | https://arxiv.org/abs/2605.02568 | 15 | Sub-Q8 streaming top-k training cost |
| 2 | https://arxiv.org/abs/2607.19358 | 12 | Sub-Q8 linear-indexed sparse attention |
| 2 | https://arxiv.org/abs/2608.01662 | 12 | Sub-Q8 cross-layer indexing |
| 2 | https://arxiv.org/abs/2603.12201 | 13 | Sub-Q8 index reuse |
| 2 | https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html | 14 | Sub-Q8 DSA backward cost |
| 3 | https://arxiv.org/abs/2609.40093 | 18 | Sub-Q5 EP on PCIe canonical |
| 3 | https://arxiv.org/abs/2604.27085 | 15 | Sub-Q3 RoundPipe consumer-GPU training |
| 3 | https://arxiv.org/abs/2610.09424 | 12 | Sub-Q5 CoMoE commodity GPUs |
| 3 | https://arxiv.org/abs/2609.34340 | 14 | Sub-Q5 all-to-all on PCIe clusters |
| 3 | https://arxiv.org/abs/2610.01380 | 12 | Sub-Q3 GPU-initiated communication |
| 3 | https://arxiv.org/abs/2610.07333 | 12 | Sub-Q5 expert routing memory |
| 3 | https://arxiv.org/abs/2602.04870 | 13 | Sub-Q5 communication-efficient MoE parallelism |
| 3 | https://arxiv.org/abs/2211.15841 | 15 | Entity MoE MegaBlocks |
| 4 | https://doi.org/10.14778/3611540.3611569 | 17 | Sub-Q3 FSDP canonical |
| 4 | https://arxiv.org/abs/2602.22437 | 13 | Sub-Q3 FSDP at scale |
| 4 | https://arxiv.org/abs/2311.00257 | 14 | Sub-Q3 ZeRO communication overhead |
| 4 | https://doi.org/10.1109/sc41405.2020.00024 | 17 | Sub-Q3 ZeRO canonical |
| 4 | https://arxiv.org/abs/2211.05322 | 14 | Sub-Q6 model-parallel communication |
| 4 | https://arxiv.org/abs/2402.03791 | 12 | Sub-Q6 TP-free parallelism |
| 4 | https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py | 16 | Sub-Q4 bf16_compress_hook primary |
| 4 | https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593 | 12 | Sub-Q4 no_sync primary |
| 5 | https://arxiv.org/abs/2609.23898 | 12 | Sub-Q4 coded approximate all-reduce |
| 5 | https://arxiv.org/abs/2604.27844 | 13 | Sub-Q4 ZipCCL lossless compression |
| 5 | https://arxiv.org/abs/2607.07494 | 12 | Sub-Q4 FP8 gradient compression |
| 5 | https://arxiv.org/abs/2605.00539 | 12 | Sub-Q4 activation/gradient quantization |
| 5 | https://doi.org/10.1109/hoti52880.2021.00018 | 14 | Sub-Q3 UCX NVLink/PCIe collectives |
| 5 | https://doi.org/10.1007/978-3-031-24145-1_23 | 13 | Sub-Q3 GEMM NVLink vs PCIe |
| 5 | https://escholarship.org/uc/item/67z0d2wn | 12 | Sub-Q3 interconnect energy |
| 5 | https://clusterbid.com/blog/nccl-tuning-ring-tree-all-to-all-algorithms | 11 | Sub-Q3 NCCL algorithms |
| 6 | https://github.com/PixelML/club-170hx/blob/main/docs/HARDWARE.md | 18 | Sub-Q1 link width measured |
| 6 | https://github.com/PixelML/club-170hx/blob/main/docs/TOPOLOGY-AND-PARALLELISM.md | 18 | Sub-Q6 PP vs TP measured |
| 6 | https://github.com/PixelML/club-170hx/blob/main/docs/CMP170HX-UNDERSTANDING.md | 17 | Sub-Q1 lock layers, P2P |
| 6 | https://github.com/PixelML/club-170hx/blob/main/docs/BENCHMARKS.md | 15 | Sub-Q6 measured recipes |
| 6 | https://github.com/bayley/cmpunlocker | 16 | Sub-Q1 P2P over BAR1 |
| 6 | https://github.com/Consensus-Protocol/cmp170hx | 15 | Sub-Q1 hardware reference |
| 6 | https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/ | 16 | Sub-Q1 stock Gen1 x4 |
| 6 | https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/ | 14 | Sub-Q9 measured perf |
| 7 | https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/ | 16 | Sub-Q9 FP32/FMA throttle, 42 TFLOPS |
| 7 | https://arxiv.org/abs/2505.03782 | 15 | Sub-Q9 mining GPUs in AI |
| 7 | https://github.com/amoghmunikote/170th-street/blob/main/benchmarks-and-performance/fp16-performance.md | 13 | Sub-Q9 fp16 measurement |
| 7 | https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/SETTINGS.md | 16 | Sub-Q6 PP vs TP 6.6x |
| 7 | https://deepwiki.com/allover326/deepseek-v4-cmp170hx | 14 | Sub-Q1 precedent |
| 7 | https://github.com/dl4rce/cmp-170hx-inference-lab | 12 | Sub-Q6 PP vs TP |
| 7 | https://wpnews.pro/news/2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epyc-romed8-2t | 12 | Sub-Q1 cross-root P2P |
| 7 | https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md | 14 | Sub-Q3 cluster design |
| 8 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 16 | Sub-Q10 primary |
| 8 | https://huggingface.co/datasets/HuggingFaceTB/stack-edu | 16 | Sub-Q10 primary |
| 8 | https://huggingface.co/datasets/HuggingFaceTB/finemath | 16 | Sub-Q10 primary |
| 8 | https://arxiv.org/abs/2502.02737 | 15 | Sub-Q10 SmolLM2 mix |
| 8 | https://doi.org/10.18653/v1/2024.acl-long.427 | 14 | Sub-Q10 sequence composition |
| 8 | https://arxiv.org/abs/2212.14034 | 12 | Sub-Q10 Cramming |
| 8 | https://arxiv.org/abs/2505.09388 | 16 | Sub-Q9 Qwen3 architecture |
| 8 | https://doi.org/10.52202/079017-0970 | 15 | Sub-Q10 FineWeb canonical |
| 9 | https://arxiv.org/abs/2111.11124 | 12 | Sub-Q9 Mesa memory saving |
| 9 | https://arxiv.org/abs/2309.12381 | 12 | Sub-Q9 mixed-precision optimizers |
| 9 | https://doi.org/10.52202/085713-3898 | 14 | Sub-Q4 critical batch size |
| 9 | https://arxiv.org/abs/1804.07612 | 11 | Sub-Q4 small-batch adversarial |
| 9 | https://archiesengupta.com/blog/memory-efficient-training-deep-dive | 12 | Sub-Q9 memory math |
| 9 | https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5 | 13 | Sub-Q9 memory math |
| 9 | https://arxiv.org/abs/2609.34849 | 11 | Sub-Q8 distillation dynamics |
| 9 | https://arxiv.org/abs/2605.30992 | 11 | Sub-Q5 non-collapsing routers |
| 10 | https://arxiv.org/abs/2604.00801 | 11 | Sub-Q5 routing-free MoE |
| 10 | https://doi.org/10.18653/v1/2025.findings-naacl.107 | 12 | Sub-Q5 representational collapse |
| 10 | https://arxiv.org/abs/2505.04519 | 13 | Sub-Q5 large MoE training |
| 10 | https://arxiv.org/abs/2609.15627 | 12 | Sub-Q8 V4-Flash on non-NVIDIA |
| 10 | https://arxiv.org/abs/2609.36322 | 11 | Sub-Q8 chunked KV phase sensitivity |
| 10 | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | 15 | Sub-Q8 primary model card |
| 10 | https://lyceum.technology/magazine/zero-3-vs-fsdp-memory-efficiency/ | 11 | Sub-Q3 FSDP vs ZeRO |
| 10 | https://towardsdatascience.com/behind-the-scenes-of-distributed-training-why-your-gpu-wiring-matters-as-much-as-your-strategy/ | 11 | Sub-Q3 wiring matters |
