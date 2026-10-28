## Coverage Matrix — query phrase → atomic item mapping

| Query phrase (verbatim) | Mapped atomic item(s) | Scope check | Gap? |
|---|---|---|---|
| "5× NVIDIA CMP 170HX box" | Entity: NVIDIA CMP 170HX | OK — full scope, 5-card count preserved | No |
| "320 GB HBM2e" | Entity: CMP 170HX required_fields (64 GB HBM2e after unlock) | OK | No |
| "~810 TFLOPS bf16" | Entity: CMP 170HX required_fields (measured bf16 TFLOPS) | OK — aggregate vs per-GPU both covered | No |
| "PCIe Gen2 x4" | Entity: PCIe Gen2 x4 fabric | OK — includes Gen1/Gen3 fuse state and per-slot width | No |
| "no NVLink and no P2P" | Entity: PCIe Gen2 x4 fabric required_fields | OK | No |
| "which LLM training experiments are feasible" | Sub-Q3 (topologies), Sub-Q9 (dense baseline), heading 7 (verdict) | OK — feasibility is asked per axis, not collapsed to one | No |
| "verify the supporting claims in the repo's documentation" | Required format: claim-by-claim verdict table with measured/reported/estimate tiers | OK | No |
| "DDP is the only viable training topology at 1M-token global batches" | Sub-Q3, Sub-Q4; Entities: DDP, FSDP, PP, TP, EP | OK — "only viable" is tested against all five topologies, not just FSDP | No |
| "expert and tensor parallelism are ruled out by the fabric" | Sub-Q5, Sub-Q6; Entities: EP, TP | OK — each gets its own sub-question so the ~70x estimate is checked separately | No |
| "frozen Engram memory tables (Memory Grafting)" | Sub-Q7; Entities: Engram, Memory Grafting | OK — both vanilla learned tables and frozen grafting covered | No |
| "7B-total / ~1B-trainable model fit DDP on 64 GB cards" | Sub-Q7; Entity: DDP; required format: memory-budget table | OK — total vs trainable distinction preserved | No |
| "CSA2 compressed sparse attention" | Sub-Q8; Entity: CSA2 | OK — Full/Reindex/Reuse modes and indexer caveats included | No |
| "32k training context" | Entity: 32k training context | OK — includes candidate-cap relevance at 32k vs 1M | No |
| Repo doc: PWRBRK# power brake | Sub-Q2; Entity: PWRBRK# power brake | OK — detection and fix both required | No |
| Repo doc: 9B-token data pipeline | Sub-Q10; Entity: Data pipeline | OK — mix, tokenizer, packing, determinism | No |
| Repo doc: bf16_compress_hook | Sub-Q4; Entity: bf16_compress_hook | OK — message-size dependence included | No |
| Repo doc: MoE architecture axis | Entity: MoE | OK — total vs active params, expert count, routing | No |
