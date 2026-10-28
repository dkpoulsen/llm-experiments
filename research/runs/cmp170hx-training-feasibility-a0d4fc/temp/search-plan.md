# Search plan — cmp170hx-training-feasibility-a0d4fc

Lenses: **A** breadth, **B** citation-chain depth (academic APIs), **C** adversarial, **D** period-pinned
(not applicable — `time_periods` is empty).

| Atomic item | Search query | Type | Lens | Target |
|---|---|---|---|---|
| Sub-Q1 link width | "CMP 170HX PCIe Gen2 unlock link width training benchmark" | web | breadth | factual |
| Sub-Q1 link width | "CMP 170HX NVLink P2P VFIO passthrough multi-GPU" | web | breadth | factual |
| Sub-Q1 link width | "CMP 170HX specifications stock electrical link Gen1 x4" | web | breadth | primary |
| Sub-Q1 link width | "CMP 170HX Gen2 x16 cmpunlocker link patch" | web | adversarial | contrarian (tests the x4 assumption) |
| Sub-Q2 PWRBRK | "CMP 170HX PWRBRK power brake 88W 39 TFLOPS" | web | breadth | factual |
| Sub-Q2 PWRBRK | "CMP 170HX PWRBRK power brake 88W ASUS WRX80E B30 Kapton tape nvidia-smi HW Power Brake Slowdown" | web | depth | primary |
| Sub-Q3 topologies | "NCCL all-reduce bandwidth PCIe Gen2 x4 no NVLink multi-GPU measured GB/s" | web | breadth | measured |
| Sub-Q3 topologies | "FSDP ZeRO-3 versus DDP communication overhead measured PCIe low bandwidth 2x" | web | adversarial | contrarian |
| Sub-Q3 topologies | "pipeline parallelism tensor parallelism communication cost" | academic | depth | canonical |
| Sub-Q3 topologies | "all-reduce bandwidth PCIe NVLink comparison GPU" | academic | depth | canonical |
| Sub-Q4 DDP model | "PyTorch DDP no_sync gradient accumulation communication overhead bf16_compress_hook" | web | breadth | primary docs |
| Sub-Q4 DDP model | "gradient compression bf16 all-reduce distributed training" | academic | depth | canonical |
| Sub-Q4 DDP model | "critical batch size large batch training neural networks" | academic | adversarial | contrarian |
| Sub-Q5 EP | "expert parallel communication PCIe connected consumer GPUs" | academic | depth | canonical |
| Sub-Q5 EP | "mixture of experts training stability routing collapse" | academic | adversarial | contrarian |
| Sub-Q5 EP | "MoE dropless implementation megablocks" | academic | depth | canonical |
| Sub-Q6 TP | "pipeline parallelism tensor parallelism communication cost" | academic | depth | canonical |
| Sub-Q6 TP | "club-170hx github hardware topology parallelism" | web | breadth | measured |
| Sub-Q7 Engram | "conditional memory scalable lookup Engram" | academic | depth | canonical |
| Sub-Q7 Engram | "memory grafting offline conditional memory pretraining" | academic | depth | canonical |
| Sub-Q7 Engram | "Engram conditional memory criticism limitations DeepSeek hashed n-gram memory" | web | adversarial | contrarian |
| Sub-Q7 Engram | "Memory Grafting offline conditional memory frozen tables donor model 0.92B results limitations" | web | adversarial | contrarian |
| Sub-Q7 Engram | "Pooling Engram Conditional Memory CXL" | academic | depth | systems cost |
| Sub-Q8 CSA2 | "DeepSeek V4.1 Flash KV cache compression sparse attention" | academic | depth | canonical |
| Sub-Q8 CSA2 | "sparse attention indexer distillation training long context" | academic | depth | canonical |
| Sub-Q8 CSA2 | "DeepSeek CSA2 compressed sparse attention training indexer distillation top-k non-differentiable criticism" | web | adversarial | contrarian |
| Sub-Q9 dense baseline | "Qwen3 technical report architecture" | academic | depth | canonical |
| Sub-Q9 dense baseline | "activation checkpointing memory transformer training" | academic | depth | canonical |
| Sub-Q9 dense baseline | "Adam optimizer memory footprint mixed precision training" | academic | depth | canonical |
| Sub-Q9 dense baseline | "transformer training activation memory 32k context activation checkpointing Adam 16 bytes per parameter mixed precision" | web | breadth | measured |
| Sub-Q9 dense baseline | "CMP 170HX tensor core bf16 TFLOPS benchmark measured GA100 mining card" | web | adversarial | contrarian (tests the 162 TFLOPS claim) |
| Sub-Q10 data pipeline | "FineWeb-Edu dataset educational web text" | academic | depth | canonical |
| Sub-Q10 data pipeline | "FineWeb-Edu Stack-Edu FineMath4+ HuggingFace dataset sizes SmolLM2 training mix" | web | breadth | primary |
| Sub-Q10 data pipeline | "packed sequences document masking language model training" | academic | depth | canonical |
| Entity: DDP world size | "PyTorch DDP 5 GPUs odd world size NCCL ring all-reduce non-power-of-two" | web | adversarial | contrarian |
| Entity: CMP 170HX | "deepseek-v4-cmp170hx 4x CMP 170HX DeepSeek V4 Flash inference" | web | breadth | precedent |
| Entity: CMP 170HX | "CMP 170HX LLM training 5 GPU" | web | breadth | gap probe |
| Entity: CMP 170HX | "Exploration of Cryptocurrency Mining-Specific GPUs in AI" | academic | depth | canonical |
| Entity: CMP 170HX | "All GB/s without FLOPS CMP 170HX review" | web | adversarial | contrarian |

Adversarial searches run: 8 (exceeds the 5 minimum).

## Coverage check against coverage-matrix.md

Every row of the coverage matrix has at least one planned search. No gaps.
