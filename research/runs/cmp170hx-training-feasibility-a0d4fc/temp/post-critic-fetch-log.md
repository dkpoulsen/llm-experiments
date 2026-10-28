# Post-critic fetch log — step 13

**Result: 0 fetch-worthy gaps.** No fetcher wave was run. Every critic finding that names a coverage gap points at evidence the vault already holds, so the patcher has ammunition for all 31 findings without new sources.

## Why no fetch

The step-13 criterion is: a gap is fetch-worthy only when the vault has **0–1** relevant notes for the topic. All four critic files were scanned for `severity: major|critical` findings with a coverage-gap failure mode, and each named topic was checked against the vault.

| Critic finding | Topic | Vault coverage | Fetch? |
|---|---|---|---|
| w-01 (critical) | sequence composition / document-boundary masking | 6 notes, incl. `analysing-the-impact-of-sequence-composition` and its ACL reprint | no |
| w-02 (critical) | bf16 gradient compression on slow fabrics | 3 notes on the query, 6 in the cluster: `ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compre`, `agoq-activation-and-gradient-quantization-for`, `powersgd-practical-low-rank`, `zipccl-efficient-lossless-data-compression-of`, `communication-efficient-distributed-training`, `pytorch-ddp-comm-hooks-default_hookspy-bf16fp16-compress-hooks` | no |
| w-03 (major) | 5-rank ring pathology / odd-rank NUMA | `evaluating-modern-gpu-interconnect-pcie` (Tartan, "One should avoid adopting 5 GPUs"), `why-is-the-bandwidth-utilization-of-7-gpus-much-lower-than-that-of-6-8-gpus-for` (NCCL maintainer on offset alignment) | no |
| w-04 (major) | critical batch size / warmup | 20 notes on the query; `critical-batch-size-revisited-a-simple-empirical`, `an-empirical-model-of-large-batch-training`, `accurate-large-minibatch-sgd` | no |
| w-05 (major) | ECC status | 7 notes: `github-amoghmunikotecmpunlocker-…` (Working ✓), `github-bayleycmpunlocker-…` (not restored), `readme_enmd` (v0.5, Enabled at ≤0.3%), `specifications-cmp-170hx-documentation`, `full-specifications-170th-street`, `cmp-170hx-inference-labdocsunlock-and-modsmd-…`, `lessonsmd` (~50% field failure above 8 GiB) | no |
| w-06 (major) | frozen vs trained Engram tables | 6 notes: `delta-n-gram-forward-only-continual-writing`, `longcat-sparse-attention-taming-the-lightning-via`, `representation-drift-in-continual-learning`, `conditional-memory-via-scalable-lookup`, `ultramemv2-memory-networks-scaling-to-120b`, `over-tokenized-transformer-vocabulary-is-generally-worth-transforming` | no |
| w-07 (major) | sequence parallelism as the sixth topology | 20 notes: `deepspeed-ulysses-system-optimizations-for-enabling`, `usp-a-unified-sequence-parallelism-approach-for`, `loongtrain-efficient-training-of-long-sequence-llms`, `ring-attention-with-blockwise`, `seq1f1b-efficient-sequence-level-pipeline-parallelism-for` | no |
| w-08 (major) | Stack-Edu hydration / checkpoint size / dataloader units | `huggingfacetbstack-edu-datasets-at-hugging-face`, `smollm2-when-smol-goes-big`, `published-as-a-conference-paper-at-iclr-2022`, `adafactor-adaptive-learning-rates-with-sublinear-memory-cost`, `memory-efficient-mixed-precision-optimizers` | no |
| ins-01 (critical) | bf16 compression premise + 2M-token case | same cluster as w-02, plus `an-empirical-model-of-large-batch-training` and `critical-batch-size-revisited-a-simple-empirical` for the 2M case | no |
| ins-02 (major) | non-differentiable top-k / parallel indexer | `deepseek-v41-flash`, `streamindex-memory-bounded-compressed-sparse`, `published-as-a-conference-paper-at-colm-2026` (HISA), `dsa_referencepy` | no |
| ins-05 (major) | the impossible d=2048/494M config | `qwen3-technical-report` (the config anchor), `readmemd-faircslmtrainbench-at-main` (the measured 505.5M config) | no |
| ins-07 (minor) | Qwen3 tokenizer | `qwen3-technical-report`, `huggingfacetbstack-edu-datasets-at-hugging-face` (the SmolLM2-tokenizer caveat) | no |
| D1–D8 (dialectic) | thesis direction, 6.6% provenance, wall-clock threshold, T1 anchor, EP availability, algbw reading, freeze-vs-replicate, TP placement | all resolve against notes already cited in the report plus `interim-report-allreduce-ceiling-and-definition`'s calibration | no |
| depth-1–8 | Engram 33M inconsistency, retrieval mechanism, unit scaling, derivation reproducibility, SM80 mechanism, MoE trade rate, 5-GPU pathology, experiment program | all resolve against notes already in the vault | no |

## Unfilled and unfillable

One gap cannot be filled by any fetch and must be acknowledged rather than fabricated:

- **No five-rank all-reduce receipt exists in any accessible source.** The club's result set has one-, two-, three- and four-card runs only; Tartan's Fig. 20 has a five-GPU point but is a plot with an implausible y-scale and is not citable. The user's own `all_reduce_perf` output is the only decisive artifact, and no fetch can obtain it. The patcher must state this limitation explicitly wherever the 6.6% figure appears.

## Note for the patcher

Because no new sources were fetched, `temp/evidence-digest.md` was **not** extended with a post-critic section. All citations the patcher needs are already in the digest or in the vault, and every note id referenced by a critic finding resolves.
