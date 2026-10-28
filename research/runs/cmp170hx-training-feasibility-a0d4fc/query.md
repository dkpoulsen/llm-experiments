---
vault_tag: cmp170hx-training-feasibility-a0d4fc
created: 2026-10-27T09:55:09Z
source: user-prompt (clarified)
---

Assess which LLM training experiments are feasible on a 5× NVIDIA CMP 170HX box — 320 GB HBM2e, ~810 TFLOPS bf16, PCIe Gen2 x4, no NVLink and no P2P — and verify the supporting claims in the repo's documentation: whether DDP is the only viable training topology at 1M-token global batches, whether expert and tensor parallelism are ruled out by the fabric, whether frozen Engram memory tables (Memory Grafting) make a 7B-total / ~1B-trainable model fit DDP on 64 GB cards, and whether CSA2 compressed sparse attention pays off at 32k training context.
