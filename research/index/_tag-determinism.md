---
title: "Tag: determinism"
id: "_tag-determinism"
type: index
created: 2026-10-27T19:00:34.889937+00:00
updated: 2026-10-27T19:00:34.889937+00:00
---

# Tag: determinism

**3** notes.

- [[analysing-the-impact-of-sequence-composition]] — Analysing The Impact of Sequence Composition — Zhao et al., 'Analysing The Impact of Sequence Composition on Language Model Pre-Training' (arXiv 2402.13991; ACL 2024 long paper) — the primary source behind the ACL landing-page note. Setup: LLaMA-architecture 1.3B-parameter models pretrained on 150B tokens sampled from SlimPajama (7 sub-corpora, token proportions preserved), at context windows of 2,048 (2K) and 8,192 (8K). Four sequence-construction / masking variants: MIXChunk (baseline random concatenation), UNIChunk (same-source documents only), BM25Chunk (retrieval-based: for each document, BM25-select related documents from the corpus to fill the chunk), and INTRADoc (intra-document causal masking).

Results: BM25Chunk has the lowest PPL of the three causal-masking variants — -0.62 at 2K and -0.78 at 8K vs MIXChunk; UNIChunk also beats MIXChunk (-0.34 / -0.21). INTRADoc has the lowest PPL of all. GitHub is the most sensitive domain: INTRADoc improves over UNIChunk by -1.3 PPL at 2K and -2.0 at 8K, suggesting code pretraining is disproportionately harmed by unrelated preceding context. Downstream: BM25Chunk improves in-context learning +11.6%, knowledge memorisation +9.8%, context utilisation +7.2%.

Cost: intra-document causal masking costs 4.0% training-efficiency degradation in their implementation, implemented with FlashAttention-2; the authors state they do not fully solve this efficiency issue. This is the number to price against any packing / masking change on the CMP 170HX box: a 4% throughput tax buys double-digit downstream gains, a favourable trade at small scale. `review`
- [[analysing-the-impact-of-sequence-composition-on-language-model-pre-training-acl]] — Analysing The Impact of Sequence Composition on Language Model Pre-Training - ACL Anthology — ACL Anthology landing page for Zhao et al., 'Analysing The Impact of Sequence Composition on Language Model Pre-Training' (ACL 2024 Long Papers, pp. 7897-7912, DOI 10.18653/v1/2024.acl-long.427). Abstract only — full text is in the arXiv primary (2402.13991) fetched separately. Core claims: standard fixed-length concatenation with causal masking lets tokens attend to previous documents, injecting distracting context; intra-document causal masking conditions each token only on prior tokens of the same document and significantly improves LM and downstream performance; and the retrieval-based packing method BM25Chunk improves in-context learning (+11.6%), knowledge memorisation (+9.8%) and context utilisation (+7.2%) without sacrificing efficiency. Directly relevant to the repo's data-pipeline packing/determinism claims: packing strategy is a first-order quality variable, not a neutral implementation detail. `review`
- [[deepseek-v4]] — DeepSeek-V4: — DeepSeek-V4 technical report (DeepSeek-AI, arXiv 2606.19348) — the CSA/HCA predecessor that CSA2 in V4.1-Flash simplifies. V4-Flash is 43 layers at d=4096 with pure SWA in the first two layers and CSA and HCA interleaved thereafter; CSA uses compression rate m=4, 64 indexer query heads, indexer head dimension 128, and attention top-k of 512 (V4-Flash) or 1024 (V4-Pro); HCA uses m'=128. Under this configuration V4-Flash is 284B total parameters.

The training recipe is the load-bearing part for the CSA2-at-32k question. Training starts at a 4K sequence length and is gradually extended to 16K, 64K and 1M. Critically, V4 warms up with DENSE attention for the first 1T tokens and only introduces sparse attention at a sequence length of 64K — the direct opposite of V4.1-Flash's "no dense attention warmup stages" claim, and in both generations sparse attention is introduced at 64K, never at 32k. When sparsity is introduced, V4 first runs a short stage to warm up the lightning indexer in CSA and then trains with sparse attention for most of the run. Batch size is raised to 75.5M tokens and held there; LR is warmed up over 2000 steps, held at 2.7e-4, then cosine-decayed to 2.7e-5. Sample-level attention masking is used during pretraining.

Precision and determinism machinery: KV entries use a mixed storage format (BF16 for RoPE dimensions, FP8 for the rest) which nearly halves KV cache size versus pure BF16; lightning-indexer attention is computed in FP4 precision; FP4 QAT is applied during post-training to MoE expert weights and the indexer QK path. Sparse-attention backward uses atomicAdd to accumulate KV gradients, which makes training non-deterministic, so V4 allocates per-SM accumulation buffers and performs a deterministic global summation across them. Optimizer is Muon for the majority of parameters with AdamW for the embedding module.

Also relevant: V4 explicitly notes that relative to DeepSeek-V3.2 a SMALLER attention top-k is chosen in the V4 series to improve efficiency on short- and medium-length texts — i.e. the top-k budget was tuned down for shorter contexts, which undercuts the repo's framing that the budget is "tuned for 1M context" and therefore mismatched to 32k. Sparse-attention kernels are co-designed with the KV cache layout (padding blocks to align with hardware), and the visible range of compressed KV entries is precomputed by rules. 26,488 words — long source, flag for source-analyst. `review`

---
*Auto-generated by hyperresearch. Do not edit manually.*
