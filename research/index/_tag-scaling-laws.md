---
title: "Tag: scaling-laws"
id: "_tag-scaling-laws"
type: index
created: 2026-10-27T19:00:34.894868+00:00
updated: 2026-10-27T19:00:34.894868+00:00
---

# Tag: scaling-laws

**3** notes.

- [[cramming-training-a-language-model-on-a]] — CRAMMING: TRAINING A LANGUAGE MODEL ON A — Cramming (Geiping & Goldstein, arXiv 2212.14034) is the canonical single-GPU compute-budget study: train a transformer LM from scratch with masked LM on one GPU for 24 hours and measure GLUE. Hardware tested: RTX 2080 Ti (11 GB), RTX A4000 (16 GB), RTX A6000 (48 GB), each paired with 4 CPU cores and 32 GB RAM; the A6000 run reaches BERT-base-class GLUE in one day.

Findings that bear on CMP 170HX feasibility: (1) scaling laws hold in the extreme low-compute regime — per-token training efficiency depends strongly on model size but not on transformer type, so architecture changes only help insofar as they raise throughput at constant parameter count; (2) the authors explicitly find no benefit from CPU/NVMe offloading in this setting; (3) FlashAttention (Dao et al. 2022) is only re-enabled for the final architecture variant after ablations, i.e. it is a throughput lever, not an accuracy lever; (4) the ablation accounting attributes ~2 GLUE points to architecture changes, ~1 point to data changes, and ~0.5 point to training-recipe changes; (5) sequence length 128 with packed sequences gives micro-batches of 64-96 on the 2080 Ti, accumulated into larger batches, and single-epoch training is feasible at this budget.

Relevance: it is the strongest published evidence that a single GPU with no interconnect at all can produce a usable model in a bounded wall-clock budget — the limiting case of the CMP 170HX box (5 cards, no NVLink, no P2P). It also warns that offloading, a plausible workaround for a 64 GB-per-card memory ceiling, bought nothing in their measurements. `review`
- [[arxiv230516264v5-cscl-28-jun-2025]] — Scaling Data-Constrained Language Models (arXiv 2305.16264v5) — Scaling Data-Constrained Language Models (Muennighoff et al., arXiv 2305.16264v5, revised 28 Jun 2025) is the primary behind the '4-5 epoch' repetition threshold that SmolLM2 cites when sizing its data mixture. Setup: more than 400 models from 10M to 9B parameters, trained for up to 1500 epochs, with budgets ranging up to 900B training tokens; the headline experiment uses a 4.2B-parameter model. All models and datasets are released as huggingface/datablations.

Core findings: (1) with constrained data at a fixed compute budget, training with up to 4 epochs of repeated data yields negligible loss change versus unique data, and differences in downstream task performance are insignificant up to around 4 epochs (25% of budget) before dropping; (2) beyond that, the value of additional compute decays to zero — the fitted data-constrained scaling law replaces the raw token count D with an effective-token count D' = U_D + U_D*R_D*(1 - e^(-R_D/R*_D)), where R*_D is the 'half-life' of repeated data; (3) in the data-constrained regime, compute should be allocated to both more parameters and more epochs, with epochs scaled slightly faster than Chinchilla would predict.

Mitigations: mixing Python code into natural-language training provides a 2x increase in effective tokens even when evaluated only on natural-language tasks, and filling up to 50% of the data with code (42B tokens) shows no deterioration. On filtering, perplexity filtering is effective but deduplication does not help on their (clean) benchmark; they recommend reserving filtering for noisy datasets.

Why it matters here: it bounds how far a 9B-token mix can be stretched on a fixed box. If the unique-token pool is ~9B, roughly 36B tokens of training are available at the 4-epoch threshold before returns collapse — the real constraint on how long a run this hardware can justify, not the FLOP budget alone. `review`
- [[training-compute-optimal-large-language-models]] — Training Compute-Optimal Large Language Models — Training Compute-Optimal Large Language Models (Hoffmann et al., DeepMind, arXiv 2203.15556) is the primary behind the '20 tokens per parameter' rule that FineWeb and SmolLM2 both use to size their ablations. The paper fits three independent approaches — minimum over training curves, IsoFLOP profiles, and parametric loss modelling — on models from 70M to over 16B parameters trained on 5B to over 400B tokens, and finds that model size and training tokens should be scaled in approximately equal proportions as compute grows: exponents a = 0.50 / b = 0.50 (approach 1), 0.49 / 0.51 (approach 2), 0.46 / 0.54 (approach 3), versus Kaplan et al.'s 0.73 / 0.27.

The practical frontier (Table 3, approach 1) gives exact FLOPs and token counts: 400M params -> 1.92e19 FLOPs / 8.0B tokens; 1B -> 1.21e20 FLOPs / 20.2B tokens; 10B -> 1.23e22 FLOPs / 205.1B tokens; 67B -> 5.76e23 FLOPs / 1.5T tokens; 175B -> 3.85e24 / 3.7T; 1T -> 1.27e26 / 21.2T. The FLOPs constraint used throughout is FLOPs(N,D) = 6ND.

Headline result: Chinchilla (70B params, 1.4T tokens) trained on the same compute budget as Gopher (280B) uniformly and significantly outperforms Gopher, GPT-3 (175B) and Jurassic-1 (178B), reaching 67.5% on MMLU, a >7% improvement over Gopher. The paper's central claim is that the models of its era were considerably over-sized relative to their compute budgets.

Why it matters here: 6ND plus the Table 3 frontier converts the box's ~810 TFLOPS bf16 into a concrete answer to 'how big a model and how many tokens can this run afford' — a 1B model at 20.2B tokens costs 1.21e20 FLOPs, a multi-day-to-multi-week run at realistic MFU, while a 10B model at 205B tokens (1.23e22 FLOPs) is out of reach. It also justifies the repo's choice of a ~1B-trainable model rather than a larger one at fixed compute. `review`

---
*Auto-generated by hyperresearch. Do not edit manually.*
