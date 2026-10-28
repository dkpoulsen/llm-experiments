# Interim report: token-budget-and-effect-size

**Locus question:** Are 9B tokens enough to resolve the architecture deltas the program wants to measure, when every published ablation at comparable scale used 50-350B tokens and the corpus disagrees on whether architecture changes per-token efficiency at all?
**Flavor:** dialectical

**Evidence tiers used below:** T1 = measured on this box (the user's own numbers); T2 = community-measured on CMP 170HX-class hardware; T3 = measured in a published paper on other hardware; T4 = arithmetic from stated assumptions. The run's binding hierarchy is T1 > T2 > T4; T3 sits alongside T2 as external measurement. Every claim below is labelled.

## What the corpus already said

The repo's plan is explicit and uniform: **one deterministic 9B-token stream, consumed in the same order by every arm**, at 32k context, with the dense control at ~500M active and the other arms matched to it. `data-pipeline.md` says "One deterministic 9B-token stream that every experiment consumes in the same order"; `experiments/01-dense-baseline.md` says "Every later experiment compares against this run's loss curve at matched tokens"; `goal-and-requirements.md` lists four axes (dense control, MoE 7B/1B, Engram learned vs frozen, CSA2 at 32k). There is **no seed replication, no size ladder, and no power analysis anywhere in the three docs** — the comparison is a single-point A/B at one size, one budget, one seed, on mean validation loss. The repo also concedes the budget's inadequacy in its own data doc: "math and code capabilities barely emerge at 9B tokens on a 500M model."

The corpus's published budgets, converted to absolute tokens and tokens-per-parameter, cluster far above that:

| Source | Params | Tokens | Tok/param | Tier |
|---|---:|---:|---:|---|
| [[cramming-training-a-language-model-on-a]] | ~110M (BERT-base) | ~7B (Fig. 1 x-axis) | 64 | T3 |
| [[moba-mixture-of-block-attention-for]] scaling-law sweep | 568M / 822M / 1.1B / 1.5B / 2.1B | 10.8B / 15.3B / 20.6B / 27.4B / 36.9B | 18–19 | T3 |
| [[moba-mixture-of-block-attention-for]] 32K hybrid | 1.5B | 30B | 20 | T3 |
| [[memory-grafting-scaling-language-model]] | 0.92B trainable | 50B | 54 | T3 |
| [[memory-grafting-scaling-language-model]] | 2.8B trainable | 100B | 36 | T3 |
| [[conditional-memory-via-scalable-lookup]] slot sweep | 568M active / 3B total | 100B ("to ensure convergence") | 176 | T3 |
| [[conditional-memory-via-scalable-lookup]] allocation law | 993M active / 9.9B total | ~100B (C = 6e20 FLOPs) | 101 | T3 |
| [[periodic-weak-spots]] | 0.6B (Qwen3 backbone) | 100B | 167 | T3 |
| [[analysing-the-impact-of-sequence-composition]] | 1.3B | 150B | 115 | T3 |
| [[smollm2-when-smol-goes-big]] | 1.7B | 350B per ablation | 206 | T3 |
| [[scaling-embeddings-outperforms-scaling-experts-in-language]] | 280M / 790M / 1.3B | 300B | 231–1071 | T3 |
| [[native-sparse-attention-hardware-aligned-and-natively]] | 27B | 260–270B | 9.6 | T3 |
| [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] | 400M / 1B / OLMoE-1.3B / OLMoE-7B | 400B / 1T / 500B / 500B | 71–1000 | T3 |
| **repo plan** | **~500M active** | **9B** | **18** | — |

Two corrections to the locus's framing fall out of that table. First, the gap is **not** 3–39×: the closest published comparator is MoBA's smallest scaling-law point at **568M / 10.8B tokens**, which is 1.2× the box's planned run, not 3.3× — and cramming's ~7B is actually *below* it at 0.78×. Second, the *median* published budget is **100B tokens, ~11× the box's 9B**, and the top of the range (OE at 1T) is 111×. So the box's run is not unprecedented — it is a near-replica of the low end of one published sweep — but it sits at roughly the 15th percentile of the distribution. The sharper point is that **the two published studies nearest the box's budget are also the two that report the smallest architecture effects in their families**: cramming (~7B at 110M) found no architecture win at all, and MoBA's smallest point (10.8B at 568M) is where its 32K penalty is largest and least asymptotic.

The corpus also carries the three opposing positions the loci analyst named, and each has real evidence behind it:

- **Position A ([[cramming-training-a-language-model-on-a]]):** "Per-token efficiency of training depends strongly on model size, but not transformer type." Cramming trained funnel-transformers, FFN-dropped, recurrent and deep-narrow variants for 24 h each and found "all choices reach a MLM loss around 1.9"; the architecture ablation was worth ~2 GLUE points on 78.6 (2.5% relative).
- **Position B ([[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]]):** input-vocabulary scaling at 400M "matches the training loss of a 1B baseline with no additional training cost", with a 5.7× convergence speedup at 1B/1T and a log-linear law "L = 2.6754 − 0.0256 × log10 m".
- **Position C ([[memory-grafting-scaling-language-model]], [[dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]]):** the memory axis is budget- and scale-dependent and can be *negative* at ~1B — "vanilla Engram underperforms the MoE baseline in this 1B-scale setting, with 45.03 versus 45.62" — recovering only +0.2 to +1.4 after 50B tokens.

What the corpus did **not** contain, and what this locus therefore needed, was three things: (a) a run-to-run noise floor for a single LM training run, so "exceeds noise" could be made quantitative; (b) a published token-budget sweep that shows how an architecture delta *changes* across budgets rather than being measured at one; (c) any test of whether architecture changes the scaling *exponent* rather than the intercept. Those are the three fetches below.

## What the new sources say

### 1. [[datadecide-how-to-predict-best-pretraining-data-with-small-experiments]] — the noise floor and the decision-accuracy framework

DataDecide is the largest controlled small-scale pretraining suite in existence: "1,050 models using 25 data recipes × 14 model scales × 3 random seeds", "a token to parameter ratio of 100, which at 5× 'Chinchilla' (5 × C) optimal ratio ... captures the typical overtraining favored for inference savings", up to 1B parameters and 100B tokens. It is the only source in the vault that measures run-to-run variance at scale *by design*.

Its noise number: "we find that the standard deviation between runs at the 1B 5×C scale can be as high as **2% points of accuracy** for some recipes on most tasks." That is a T3 measurement of seed-to-seed spread at exactly the scale class the box is targeting.

Its decision framework is the load-bearing part. "The evaluation must separate pairs of data recipes by an amount greater than combined noise from run-to-run variance of each of the pair's runs." They decompose decision accuracy into two quantities — "noise—the standard deviation over 3 random seed runs averaged over all recipes, and spread—the standard deviation among the mean performance of the different data recipes" — and show that predictability is governed by the spread/noise ratio, not by compute. The practical lever is the metric: "the switch of metric allows small models to get above the noise floor for these tasks", taking code-task decision accuracy "from trivial to 80%".

Its headline for the box's design question: "the ranking of models at a single, small size (e.g., 150M parameters) is a strong baseline for predicting best models at our larger target scale (1B) (~80% of comparisons correct)." So a 150M-scale experiment is a legitimate decision instrument — **at ~80% fidelity, i.e. one wrong call in five** — and only if the metric is chosen for low noise and high spread.

### 2. [[a-hitchhikers-guide-to-scaling-law-estimation]] — the 3.5%/4% floor, and "5 models is a safe bet"

This is the source that converts "exceeds run-to-run noise" from a slogan into a threshold. Its calibration paragraph is the single most useful passage in this locus:

> "Empirically, we found no widely adopted modeling changes that were motivated with less than a **4% relative difference** between models. Additionally, reported **variance across random restarts of the same model architecture reaches up to 3.5%** ... We take this to mean that this is approximately the minimal effect size experimenters care, as well the minimal effect that can be reliably measured."

So the published floor is: **run-to-run spread up to 3.5% relative; nothing below ~4% relative is treated as a real modelling win.** (The paper's phrasing is ambiguous between variance and spread; read either as a relative spread, it lands in the same place, and DataDecide's 2pp-at-1B corroborates it independently.)

Its design prescription is equally direct: "**5 models is a safe bet**, more would improve the results' robustness. **These models can be small.**" And on where to spend compute: "because of noise—it is often preferable to extrapolate from a large number of small, partially trained models rather than a small number of large models." It also quantifies how much of a run you actually need: "It is generally possible to estimate a model's final loss beginning roughly **1/3 of the way through training**", and "As long as the first ≈10% of checkpoints are discarded, final loss can often be predicted accurately."

That last pair is a direct cost lever the repo does not use: a size ladder trained to one-third of budget, with the first 10% of checkpoints discarded, is sufficient to estimate final loss — at one-third the wall-clock.

### 3. [[scaling-laws-vs-model-architectures-how-does-inductive-bias-influence-scaling]] — architecture changes the exponent, and rankings flip across compute regions

Tay et al. pretrained and finetuned "over 100 different models" across "ten diverse model architectures" from "15M to 40 Billion parameters". Their central finding is the one the corpus was missing:

> "For the first time, we derive scaling laws for different inductive biases and model architectures. **We find that this scaling coefficient differs greatly from model to model.**"

And the consequence for pointwise comparisons — which is precisely what the box plans:

> "models that operate well in one compute-scale region is not necessarily the best in another compute-region. Moreover, we find that certain models have difficulty scaling despite performing decently (comparably) at lower-compute regions. This has implications, since **it is difficult to get the full picture of a model's scalability with pointwise comparisons at a certain compute-region**."

Table 2 gives the fitted slopes, and they are brutal for the two axes the box cares most about:

| Model | α(F→upstream) | α(F→downstream) |
|---|---:|---:|
| Transformer (dense) | **0.54** | **0.28** |
| GLU-Transformer | 0.49 | 0.24 |
| Funnel | 0.47 | 0.22 |
| **Switch (MoE)** | **0.23** | **0.14** |
| Universal | 0.50 | 0.20 |
| ALBERT | 0.08 | **−0.12** |
| Evolved | 0.44 | 0.22 |
| **Performer (linear attention)** | **0.25** | **0.05** |

The dense Transformer has the steepest slope of all ten: "the vanilla Transformer has the best scaling behaviour, **even if its absolute performance at each compute region is not the greatest**." MoE (Switch) scales at less than half the dense slope; the linear-attention variant (Performer) has a downstream slope of 0.05, i.e. essentially flat.

One caveat must be carried with the Switch row, and it cuts against over-reading it: the authors footnote that "The largest Switch transformer was scaled in a pretty sub-optimal way. So we don't think it is representative of the full potential of the Switch family. Take the last data point of Switch with a pinch of salt." The Performer row carries no such caveat.

## Evidence synthesis

### The effect-size ledger: every published delta the box wants to reproduce is at or below the single-run noise floor

This is the quantitative core of the locus. Converting each published delta into nats and into % relative, against a loss anchor of ~2.77 nats (MoBA's fitted 568M/10.8B point, T3) and a noise floor of 3.5% relative (~0.10 nats, or ~2pp on a 45% benchmark):

| Source | Scale / tokens | Published Δ | Δ in nats | % relative | vs 3.5% floor |
|---|---|---|---:|---:|---|
| [[conditional-memory-via-scalable-lookup]] allocation law | 993M active / 100B | 1.7248 → 1.7109 | 0.014 | **0.8%** | **4.4× below** |
| [[memory-grafting-scaling-language-model]] Δ loss vs Engram (Fig. 2) | 0.92B / 10–50B | within ±0.02 | ±0.02 | **0.7%** | **5× below** |
| [[memory-grafting-scaling-language-model]] | 0.92B / 50B | 45.62 → 46.98 (MG-Qwen vs MoE) | — | **3.0%** | at floor |
| [[memory-grafting-scaling-language-model]] | 2.8B / 100B | 51.95 → 53.86 | — | **3.7%** | just above |
| [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] OLMoE-1.3B | 1.3B / 500B | 2.554 → 2.472 | 0.082 | **3.2%** | at floor |
| [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] OLMoE-7B | 7B / 500B | 2.305 → 2.229 | 0.076 | **3.3%** | at floor |
| [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] OLMo2-1B | 1B / 400B → 1T | 0.12 → 0.14 | 0.12–0.14 | **4.6–5.8%** | above |
| [[moba-mixture-of-block-attention-for]] at 8K | 568M–2.1B / 10.8–36.9B | "within 1e−3" | 0.001 | **0.04%** | **90× below** |
| [[moba-mixture-of-block-attention-for]] at 32K, low compute | 568M / 10.8B | fitted 1.546C^−0.108 vs 1.464C^−0.097 | **0.105** | **3.8%** | at floor |
| [[moba-mixture-of-block-attention-for]] at 32K, high compute | 2.1B / 36.9B | same fit | **0.045** | **1.6%** | 2.2× below |
| [[periodic-weak-spots]] mean loss/accuracy | 0.6B / 100B | "nearly matches its full-attention baseline" | ~0 | **~0%** | invisible |
| [[periodic-weak-spots]] phase-stratified | 0.6B / 100B | 78pp vs 6.1pp prefix-padding gap | — | **~13×** | 13× above |
| [[cramming-training-a-language-model-on-a]] architecture ablation | 110M / ~7B | ~2 GLUE points on 78.6 | — | **2.5%** | below |

Three things are visible in that table that no single source states.

**First, the memory axis's effect is small everywhere it has been measured, even at 100–500B tokens.** Engram's own allocation optimum — the paper's headline design law — buys **0.0139 nats (0.8% relative)** at 993M active parameters and ~100B tokens. Memory Grafting's loss delta at 0.92B is within ±0.02 nats over 10–50B tokens. OE's headline at 500B tokens is 3.2–3.3% relative loss, i.e. *at* the noise floor, despite adding 10× the parameters (13.1B embedding params on a 1.3B model). Only OE at 1T tokens (4.6–5.8%) clears the 4% "minimal effect experimenters care about" bar — and that took a trillion tokens.

**Second, the attention axis's effect is budget-dependent by a factor of 2.3× inside one published sweep.** MoBA's fitted curves give a 32K trailing-loss gap of **0.105 nats at its smallest point and 0.045 at its largest** — the paper's own words are "the loss gap is progressively narrowing". At the box's planned compute (2.66 PFLOP/s-days for 500M × 9B at 32k, T4), the fit predicts **~0.060 nats (2.2% relative)** — *below* the noise floor. So a 9B-token CSA2-vs-dense comparison at 32k would be expected to return a null, and that null would be consistent with the true, small, asymptotic penalty.

**Third, the metric can dominate the budget entirely.** Periodic Weak Spots is the cleanest demonstration in the corpus: at **100B tokens**, "the W8/S8 compressed model nearly matches its full-attention baseline in mean accuracy and validation loss, yet its worst residue is only 9.9%, against 58.4% for the baseline" — while the phase-stratified gap is "up to 78 percentage points against at most 6.1 for the baselines". The same 100B-token run yields ~0 on the metric the box plans to use and a 13× effect on a metric chosen for spread. DataDecide says the same thing from the other direction: "the switch of metric allows small models to get above the noise floor."

### Where the box's 9B actually sits: at the Chinchilla knee, which is the worst place to measure a stable ranking

The box's 500M-active model at 9B tokens is at **18 tokens/param**, which is 90% of Chinchilla-optimal for 500M (Table 3 of [[training-compute-optimal-large-language-models]] gives 400M → 8.0B and 1B → 20.2B, so 500M → ~10B by log-interpolation, i.e. the near-universal ~20 tokens/param; T3). For the 1B-active MoE arm it is 9B/20.2B = **45% of optimal** — the locus's arithmetic is right there, but only for that arm.

That placement matters more than the raw token count. At 18 tok/param you are *at* the knee of the loss curve — the steepest, most transient part, where architecture differences are largest and least predictive. At 50–200 tok/param, where most of the corpus's ablations sit (median ~70), you are past the knee, where differences are smaller but stable. This is exactly what cramming describes: "different architectures are unpredictable throughout an initial stage of training (the first 1B tokens), after which the per-token efficiencies differ by only a multiplicative constant". The box's 9B is past cramming's 1B transient, but at 18 tok/param it is still at the knee.

MoBA's sweep is the proof that the knee is where the gap is largest: its smallest point (568M/10.8B, 19 tok/param) has a 32K gap of 0.105 nats, and its largest (2.1B/36.9B, 17.6 tok/param) has 0.045. The tokens/param ratio barely moves across that sweep — what moves is *absolute compute*, by 12×. So the variable that governs the effect size is absolute compute, not tokens/param. The box's 9B/500M run is a near-replica of MoBA's smallest point, and MoBA's own data say that point overstates the asymptotic architecture gap by **2.3×**.

### Position A: architecture does not change per-token efficiency, so a 9B comparison lands within noise

The strongest version of this side is cramming, and it is stronger than it first looks. Cramming did not merely fail to find an architecture win — it found a *mechanism* for the failure: "scaling laws create a strong barrier to scaling down. Per-token efficiency of training depends strongly on model size, but not transformer type." Every variant it tested — funnel, FFN-dropping, recurrent layers, deep-narrow rescaling — reached "a MLM loss around 1.9" after 24 h, and the differences were "only a multiplicative constant (a horizontal shift due to the log axis)" that "depends almost entirely on the model size, not model type". The paper's operational conclusion is that architecture selection should be made "based primarily on how they affect computation time for a single gradient step" — i.e. architecture is a throughput lever, not an accuracy lever.

The steelman gets stronger when you notice that cramming's variants are all *compute-shaped*: they rearrange the same FLOPs at the same parameter count. If per-token efficiency is a function of parameter count, then any rearrangement that holds parameters fixed must land on the same curve, and cramming's null is not an empirical accident but a prediction. Under this reading, the box's four arms are four rearrangements of ~500M active parameters, and the expected result is four loss curves within noise of each other — which is precisely the "9B null" the locus fears.

The weakness of this side is scope. Cramming's null covers compute-shape variants at 110M parameters and ~7B tokens (64 tok/param). It never tested a *capacity-axis* variant — one that adds parameters without adding FLOPs — and it never tested at 500M or above. Its own framing concedes the boundary: "per-token efficiency ... depends strongly on model size", and the memory axis's entire claim is that it changes the effective size at fixed FLOPs.

### Position B: the memory axis is a first-order lever even at small scale

The strongest version of this side is OE, and its numbers are genuinely large in the currency that matters — tokens-to-target, not loss-at-fixed-tokens. "OE-12.8M with 400M parameters matches the baseline with 1B parameters" at 400B tokens; on OLMo2-1B, "training loss achieves a remarkable 5.7× speedup", with downstream accelerations of 3.2×/3.0×/2.6×/3.1×/3.9×. A 5.7× convergence speedup is not a 0.8%-relative effect; it is a claim that the memory axis buys you a different position on the compute axis entirely.

The steelman's second leg is that the effect is *lawful*, not anecdotal: "L = 2.6754 − 0.0256 × log10 m", i.e. "for every 4× increase in m, the training loss decreases by 0.015", measured over 500B tokens on OLMoE-1.3B. A log-linear law is a scaling law, and scaling laws are what you use to decide architecture. Engram corroborates the shape from the other side: "scaling the number of memory slots yields a clear and consistent improvement in validation loss ... the curve follows a strict power law (linear in log-space), indicating that Engram provides a predictable scaling knob: larger memory continues to pay off without requiring additional computation."

The weakness is that the 5.7× figure is a *ratio of tokens to reach a target loss*, not a loss gap at fixed tokens — and the paper is explicit that the gap at fixed tokens is small and slow to appear: "The larger input vocabulary requires longer training to obtain all its gains"; "the 'slow-start' nature of large vocabularies ... larger input vocabularies require more training tokens to realize their expected benefits"; and, at 500B tokens on OLMoE-7B, "The gains of OE have not fully converged." At fixed tokens the measured gaps are 0.082 and 0.076 nats (3.2–3.3% relative) — at the noise floor. So Position B's evidence is real but it is evidence about *convergence speed*, and convergence speed is exactly the thing a 9B-token fixed-token comparison cannot see.

### Position C: the memory axis's benefit is budget- and scale-dependent and can be negative at ~1B

The strongest version of this side is Memory Grafting's 0.92B table, because it contains a sign flip inside one paper. At 0.92B trainable / 0.29B active, 50B tokens: MoE 45.62, vanilla Engram **45.03** — the memory axis is *negative* at this scale. Memory Grafting recovers it to 46.42 (MG-DeepSeek) and 46.98 (MG-Qwen3.5), but the paper attributes the baseline's failure to "the limited Engram embedding table in 0.9B model, which lead to more hash collisions". At 2.8B / 0.55B active, 100B tokens, the sign flips: MoE 51.95, vanilla Engram **52.43**, MG 53.86. So the same architecture axis is −0.59 points at 0.92B and +0.48 points at 2.8B. That is a scale-dependent sign flip, and it is the single strongest piece of evidence that a 9B-token run at one size cannot tell you the direction of the memory-axis effect.

The steelman's second leg is the back-loading evidence. Engram-40B, at 262B tokens, is reported as still improving: "the training loss gap between Engram-40B and the baselines continues to widen towards the end of training, suggesting that the expanded memory capacity has not yet fully saturated within the current token budget" ([[dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]], quoting the primary). And LongCat adds a regime condition rather than a scale one: "at low parameter ratios embedding scaling fails to beat simply adding experts, while at high sparsity the benefit is pronounced" ([[scaling-embeddings-outperforms-scaling-experts-in-language]]). So the memory axis is not merely budget-dependent — it is conditional on the sparsity regime, with a crossover "slightly above a ratio of 20" where embeddings are ~50% of total parameters.

The weakness of this side is that it is a *description*, not a decision rule. "It depends on scale and budget" tells the box nothing about what to run. What it does establish, decisively, is that a single-point measurement at 9B cannot distinguish "the memory axis does nothing" from "the memory axis does something that has not started yet".

### Resolving the dialectic: A and B are not contradicting each other — they are measuring different axes, and C is the only position that survives contact with the box's budget

The cramming-vs-OE dispute is **a definitional confusion, not a factual one**, and naming it is the main analytic gain of this locus.

Cramming's "architecture" means *compute shape*: funnel, FFN-drop, recurrent, deep-narrow — variants that hold parameter count and FLOPs fixed and rearrange them. Its claim, "per-token efficiency depends strongly on model size, but not transformer type", is a claim about that class. OE's "architecture" means *capacity axis*: it adds 13.1B embedding parameters to a 1.3B model, holding activated FLOPs roughly fixed. OE-400M is not a 400M-parameter model in cramming's sense — it is a ~13.5B-total-parameter model with 400M activated (T4: 12.8M × d_model, with d_model ≈ 1024 for the 400M rung). So OE's win is entirely consistent with cramming: it is a *parameter-count* win, achieved by adding parameters that cost no FLOPs. Cramming would predict exactly that, and it does: "Models with more parameters learn more efficiently."

Once the axes are separated, the corpus's apparent contradiction dissolves and a cleaner picture emerges:

- **Compute-shape variants** (funnel, recurrent, deep-narrow): cramming's null holds. Do not expect a loss win; expect a throughput win.
- **Capacity-axis variants** (MoE, Engram, OE): the effect is real but *small at fixed tokens* (0.014–0.082 nats) and *back-loaded* (it grows with budget). Its sign is scale-dependent (Memory Grafting's flip at 0.92B → 2.8B) and regime-dependent (LongCat's crossover).
- **Attention-sparsity variants** (CSA2, MoBA): the effect is *budget-dependent and narrowing* (0.105 → 0.045 nats across MoBA's sweep), and on mean loss it can be invisible even at 100B tokens (Periodic Weak Spots).

And Tay et al. supply the mechanism that makes all three dangerous at a single point: the scaling *exponent* differs by architecture, and the dense Transformer has the steepest one. So any variant that wins at 9B is, by construction, winning in the region where its exponent disadvantage has not yet bitten. **The 9B point systematically flatters non-dense architectures.**

### The wall-clock ceiling: 100B tokens per arm is not an option, and that is what makes this locus decisive

The locus's own arithmetic assumed 100k tok/s. The sustained-throughput investigator's finding ([[interim-report-sustained-throughput-at-32k]]) is that 6ND is invalid at 32k — attention is 88% of step FLOPs and total FLOPs are 8.5× the 6ND estimate — so the realistic rate is **~9–22k tok/s, centre ~15k**, not 80–120k. Recomputing the wall-clock at that rate (T4, anchored in T1/T2 throughput):

| Budget | Wall-clock at 15k tok/s | 5 arms |
|---|---:|---:|
| 9B | 6.9 d | **34.7 d** |
| 20B (Chinchilla for 1B active) | 15.4 d | 77 d |
| 36B (4 epochs on 9B unique) | 27.8 d | 139 d |
| 50B (Memory Grafting's 0.92B budget) | 38.6 d | 193 d |
| 100B (Memory Grafting's 2.8B budget) | **77.2 d** | **386 d** |
| 350B (SmolLM2's ablation budget) | 270 d | 1,351 d |

At the optimistic end of the range (22k tok/s) five arms at 100B is still **263 days**; at the pessimistic end (9k) it is **643 days**. So "match the literature's token budget" is not an available option on this box under any throughput assumption — the published ablation budgets span 0.8–111× the box's 9B (median 11×), and anything above ~20B per arm puts a five-arm program past a quarter of a year of continuous running. **The binding feasibility constraint is not the fabric; it is that the literature's budgets are unaffordable, which means the program must buy statistical power with experimental design rather than with tokens.**

There is one real relief valve, and it is small: [[arxiv230516264v5-cscl-28-jun-2025]] establishes that "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data", so the 9B unique stream can be stretched to ~36B effective tokens. That buys 4× the tokens at 4× the wall-clock — it does not change the affordability calculus, but it does mean the data pipeline is not the constraint (FineWeb-Edu alone has 1.3T tokens available).

### The two failure modes are opposite, and the locus has the sign backwards

The locus's stated risk is "a null result that cannot be distinguished from an underpowered one". The evidence says the dominant risk is different, and it differs by axis:

- **Memory axis (Engram, MoE, OE): false null.** Expected deltas at 9B are ~0.007–0.04 nats (T4: scaling the 100B-token deltas down by ln(9)/ln(100) ≈ 0.48), i.e. 0.25–1.5% relative — **2–14× below the 3.5% floor**. A null here is the *expected* result even if the effect is real and large at 100B. The null carries almost no information: it licenses only the inference "the effect is <0.1 nats at this scale", which is compatible with every published positive result.
- **Attention axis (CSA2 at 32k): false null too, but for a different reason.** The fit predicts ~0.060 nats at the box's compute — 1.6× below the floor. And Periodic Weak Spots shows that even at 100B tokens the mean-loss signal is ~0.
- **Ranking instability: false positive / wrong sign.** This is the risk the locus missed. Tay et al. show the exponent differs and rankings flip across compute regions; MoBA shows the gap shrinks 2.3× with compute; Memory Grafting shows the memory axis's sign flips between 0.92B and 2.8B. So a 9B-token *ranking* of the four arms can reverse at 100B — and the reversal is systematic, not random: the 9B point flatters non-dense architectures because the dense Transformer has the steepest exponent (α = 0.54 vs 0.23 for Switch, 0.25 for Performer).

So the honest answer to "is a 9B-token null interpretable?" is: **the null is interpretable only as a bound, not as a verdict, and the ranking is not interpretable at all.** A null at 9B means "no effect larger than ~0.1 nats at 9B tokens", which is a real and useful bound — but it is not evidence for or against any of the published claims, all of which are smaller than that bound at this budget.

### What the design should be instead, at the same wall-clock

The literature's own answer to "how do I make a small-scale architecture decision" is not "train longer" — it is "fit a law over a ladder, and pick a metric with spread". Three concrete moves, all affordable inside the ~35 wall-clock-day envelope the box already planned:

1. **Replace the single-point A/B with a size ladder.** [[a-hitchhikers-guide-to-scaling-law-estimation]]: "5 models is a safe bet ... These models can be small", and "it is often preferable to extrapolate from a large number of small, partially trained models rather than a small number of large models." A 4-size ladder (50M / 125M / 250M / 500M) × 2 architectures = 8 runs. Fitting L(C) per architecture and comparing *exponents* is what resolved MoBA's 1e-3-nat gap at 8K — a gap ~100× below the noise floor that no single-point comparison could have seen. The dense-vs-variant exponent difference is the decision-relevant quantity, and it is measurable at small sizes.
2. **Train each ladder point to ~1/3 of budget, not to completion.** [[a-hitchhikers-guide-to-scaling-law-estimation]]: "It is generally possible to estimate a model's final loss beginning roughly 1/3 of the way through training", provided "the first ≈10% of checkpoints are discarded". That cuts the ladder's wall-clock by ~3×, which is what makes 8 runs affordable.
3. **Pick the metric for spread, not for convention.** [[datadecide-how-to-predict-best-pretraining-data-with-small-experiments]]: "The evaluation must separate pairs ... by an amount greater than combined noise"; [[periodic-weak-spots]]: "average accuracy is not enough to evaluate models with chunked KV-cache compression. Their reliability should also be measured across phases." Concretely: per-domain validation loss (continuous likelihood, not accuracy), worst-residue rather than mean, and phase-shifted evaluation for CSA2. This is free and it is the single highest-leverage change in the whole program.

And the 32k context requirement should be decoupled from the architecture comparison. The memory and MoE axes do not need 32k — MoBA's own scaling-law sweep ran at 8K and only checked 32K on the trailing-token metric. Running the ladder at 8K instead of 32k cuts FLOPs/token by ~3× (T4: 2.554e10 → 8.64e9 for 500M/28L/2048), which turns a 35-day program into a ~12-day one, or buys 3× more ladder points for the same wall-clock. Only the CSA2 arm needs 32k, and it should get a small budget plus the phase-stratified metric.

## Open questions

- **What is the actual seed-to-seed spread of final loss on *this* box's stack?** The 3.5% figure is a literature aggregate, not a measurement of a 500M Qwen3-architecture model at 18 tok/param with this data order. Two seeds at the largest ladder point would settle it for ~14 wall-clock days, and it is the single number that determines whether any of the published deltas are detectable.
- **Does the memory axis's delta actually grow as ln(tokens)?** I assumed a log-linear growth to project 9B deltas from 100B deltas (T4). If the growth is steeper (as "slow start" implies), the 9B delta is even smaller and the false-null conclusion is stronger; if it is flat, the 9B delta equals the 100B delta and the memory axis would be detectable. A 3-point budget sweep (3B / 9B / 27B) on one architecture would settle it for ~26 wall-clock days.
- **Is the 3.5% floor a spread or a variance?** [[a-hitchhikers-guide-to-scaling-law-estimation]]'s phrasing is ambiguous; if it is a variance, the standard deviation is far larger and every published delta in the ledger is hopeless at any affordable budget. DataDecide's independent 2pp-at-1B figure suggests spread, but this should be pinned down.
- **Does CSA2's indexer change the FLOPs denominator enough to move the 32k wall-clock?** Cross-locus with the CSA2 investigator: if the indexer is a parallel attention, the 88%-attention share may not fall, and the 15k tok/s figure stands. If it does fall, the whole wall-clock table shifts.
- **What is the MoE arm's exponent on this box's architecture?** Tay et al.'s Switch slope (0.23) is caveated by the authors as unrepresentative. A 2-size MoE ladder (250M / 500M active, matched total) would give the box's own exponent for ~14 wall-clock days, and that exponent is what decides whether the 7B/1B MoE arm is worth running at all.

## Sources

1. [[datadecide-how-to-predict-best-pretraining-data-with-small-experiments]] — Magnusson et al., ICML 2025, arXiv 2504.11393. 1,050 runs × 25 recipes × 14 scales × 3 seeds at 100 tok/param; 2pp run-to-run std at 1B/5×C; 150M ranking predicts 1B at ~80%; metric choice moves decision accuracy from trivial to 80%.
2. [[a-hitchhikers-guide-to-scaling-law-estimation]] — Choshen, Zhang, Andreas, ICML 2025, arXiv 2410.11840. 3.5% restart spread / 4% minimal meaningful effect; "5 models is a safe bet"; fit from 1/3 of training after discarding the first 10%.
3. [[scaling-laws-vs-model-architectures-how-does-inductive-bias-influence-scaling]] — Tay et al., Findings of EMNLP 2023, arXiv 2207.10551. Ten architectures, >100 models, 15M–40B params; α table (dense 0.54/0.28, Switch 0.23/0.14, Performer 0.25/0.05, ALBERT 0.08/−0.12); rankings flip across compute regions.
4. [[moba-mixture-of-block-attention-for]] — Moonshot AI, arXiv 2502.13189. Five-point scaling-law sweep 568M/10.8B → 2.1B/36.9B; 8K gap 1e-3; 32K gap 0.105 → 0.045 nats, "progressively narrowing"; 90/10 hybrid at 1.5B/30B/32K.
5. [[conditional-memory-via-scalable-lookup]] — DeepSeek-AI/PKU, arXiv 2601.07372. Allocation law Δ = 0.0139 nats at 993M active / ~100B; slot sweep "trained for 100B tokens to ensure convergence"; log-linear memory scaling.
6. [[memory-grafting-scaling-language-model]] — Tsinghua/MSRA, arXiv 2605.20948. 0.92B/50B (MoE 45.62, Engram 45.03, MG 46.42–46.98) and 2.8B/100B (51.95 / 52.43 / 53.86); Δ loss vs Engram within ±0.02 nats over 10–50B.
7. [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]] — ByteDance Seed, ICML 2025, arXiv 2501.16975. OE-12.8M at 400M matches 1B baseline; 5.7× convergence speedup at 1T; L = 2.6754 − 0.0256·log10 m; OLMoE-1.3B/500B −0.082, OLMoE-7B/500B −0.076; "slow-start" and "gains have not fully converged in 500B".
8. [[periodic-weak-spots]] — ByteDance Seed et al., arXiv 2609.36322. 0.6B/100B/2k; compressed model "nearly matches its full-attention baseline in mean accuracy and validation loss" while worst residue is 9.9% vs 58.4%; 78pp vs 6.1pp phase-stratified.
9. [[cramming-training-a-language-model-on-a]] — Geiping & Goldstein, arXiv 2212.14034. ~110M/~7B/24h; "per-token efficiency depends strongly on model size, but not transformer type"; architecture ablation ~2 GLUE points.
10. [[training-compute-optimal-large-language-models]] — Hoffmann et al., arXiv 2203.15556. Table 3 frontier: 400M → 8.0B tokens, 1B → 20.2B tokens.
11. [[arxiv230516264v5-cscl-28-jun-2025]] — Muennighoff et al. 4-epoch repetition threshold; 9B unique → ~36B effective.
12. [[smollm2-when-smol-goes-big]] — 1.7B/350B per ablation; ~$250k per from-scratch ablation, cited as the reason for online mixture rebalancing instead of multiple runs.
13. [[analysing-the-impact-of-sequence-composition]] — 1.3B/150B, 2K and 8K; BM25Chunk −0.62 PPL at 2K.
14. [[scaling-embeddings-outperforms-scaling-experts-in-language]] — Meituan LongCat, arXiv 2601.21204. 280M/790M/1.3B at 300B tokens; "at low parameter ratios embedding scaling fails to beat simply adding experts"; crossover at ratio ~20 (~50% embedding share).
15. [[native-sparse-attention-hardware-aligned-and-natively]] — DeepSeek-AI, arXiv 2502.11089. 27B/260–270B; sparse attention matches or beats full attention at 32k.
16. [[dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis]] — Engram-40B "training loss gap ... continues to widen towards the end of training" at 262B tokens.
17. [[interim-report-sustained-throughput-at-32k]] — sibling investigator. Realistic 9–22k tok/s at 32k (centre ~15k), not 80–120k; 6ND invalid at 32k.
18. `research.md`, `data-pipeline.md`, `experiments/01-dense-baseline.md`, `goal-and-requirements.md` (repo, not vault notes) — 9B-token single stream, 32k context, four axes, one seed, mean loss at matched tokens, no power analysis.

## Committed position

**A 9B-token null is interpretable only as a bound, never as a verdict — and the program's real failure mode is not the null, it is the ranking.** The evidence adds up to this: every architecture delta the box wants to reproduce is *small at fixed tokens* and *back-loaded*, and the box's planned 9B/500M run sits at the Chinchilla knee — 18 tokens/param, the steepest and least asymptotic part of the loss curve — where the published deltas are 0.014–0.082 nats (0.8–3.2% relative) against a single-run noise floor of 3.5% relative (~0.1 nats, ~2pp). That is 2–14× below the floor for the memory axis and 1.6× below it for CSA2 at 32k, so a null is the *expected* outcome even if both effects are real and large at 100B; it licenses only "no effect larger than ~0.1 nats at 9B", which is compatible with every published positive result and therefore settles nothing. Worse, the 9B point is systematically biased in one direction: Tay et al. show the scaling exponent differs by architecture and that the dense Transformer has the steepest one (α = 0.54 vs 0.23 for Switch, 0.25 for Performer), and MoBA's own sweep shows its 32K penalty shrinking 2.3× from 0.105 to 0.045 nats as compute rises — so a 9B-token ranking flatters every non-dense arm and can reverse at 100B. The fix is not more tokens: at the measured ~15k tok/s, 100B per arm is 77 wall-clock days and five arms is 386 days, so the literature's budgets span 0.8–111× the box's 9B and the 50–350B band is 5.6–39× unaffordable and the fabric is not the binding constraint — the ablation budget is. The affordable fix is design: a 4-size ladder (50M–500M) × 2 architectures, each trained to ~1/3 of budget with the first 10% of checkpoints discarded, fitting L(C) per architecture and comparing *exponents* rather than single losses, with per-domain likelihood and worst-residue metrics instead of mean accuracy — which is ~12–35 wall-clock days at 8K context, the same envelope the box already planned, and is the only design in the evidence that resolves deltas below the noise floor (MoBA resolved 1e-3 nats this way). Concretely: **run the memory and MoE axes at 8K on a size ladder, run CSA2 alone at 32k with a phase-stratified metric and a small budget, and do not spend 9B tokens on a single-point A/B that cannot see the effect it is looking for.**

- **Position:** A 9B-token single-point A/B on mean loss is underpowered by 2–14× against the published deltas and systematically flatters non-dense architectures; 9B tokens is sufficient *only* if spent on a multi-size scaling-law fit with a low-noise metric, and 100B-per-arm is unaffordable (77 days/arm, 386 days for five arms) so the token budget cannot be the fix.
- **Confidence:** high on the arithmetic (the effect-size ledger, the 3.5%/4% floor, the 77-day-per-arm wall-clock, and MoBA's 2.3× narrowing are all direct quotes or simple arithmetic from published numbers). Medium on the prescriptive design: the ladder-plus-exponent recommendation rests on one paper's prescription ("5 models is a safe bet") and one paper's demonstration (MoBA), and no one has run this exact design on this box. Low on the exact 9B deltas: the ln(tokens) projection from 100B to 9B is my own T4 extrapolation, not a published measurement.
- **Boundary conditions:** This applies to architecture comparisons at 0.1–1B activated parameters on a fixed compute budget, where the deltas are 0.01–0.1 nats. It does *not* apply to (a) throughput or memory-feasibility questions, which are measured directly and need no statistical power — the 7B/1B Engram fit test is a feasibility question and 1–2B tokens is plenty; (b) capability-emergence questions, where the repo itself concedes "math and code capabilities barely emerge at 9B tokens on a 500M model" and no affordable budget on this box will answer them; (c) large-model regimes (≥7B activated) where the published deltas are bigger (Memory Grafting's +1.91 points at 2.8B/100B is 3.7% relative, just above the floor) but the box cannot reach them. Outside the 0.1–1B, fixed-compute regime, the evidence is insufficient.
- **What would change this position:** (1) A measured seed-to-seed spread on this box's stack *below* 1% relative — if two seeds of a 500M/9B run land within 0.02 nats, the noise floor drops 5× and the memory axis's 0.014–0.082-nat deltas become detectable at 9B, which would make the simple A/B viable and my design recommendation unnecessary. (2) A budget sweep showing the memory-axis delta is *flat* in tokens rather than growing — if the Engram allocation delta is 0.0139 nats at 3B as well as at 100B, then 9B is representative and the false-null argument collapses. (3) A measured 32k throughput above 40k tok/s on five cards, which would make 100B-per-arm a ~29-day proposition and reopen the "just match the literature" option. (4) Evidence that CSA2's indexer cuts attention FLOPs enough to raise throughput 3×, which would do the same. Conversely, if a 3-point budget sweep (3B/9B/27B) showed the delta growing *faster* than ln(tokens), my false-null conclusion strengthens and the recommended budget should rise to 27B per ladder point.
- **Evidence weight:** 6 published measurements support the effect-size ledger ([[conditional-memory-via-scalable-lookup]], [[memory-grafting-scaling-language-model]], [[over-tokenized-transformer-vocabulary-is-generally-worth-scaling]], [[moba-mixture-of-block-attention-for]], [[periodic-weak-spots]], [[cramming-training-a-language-model-on-a]]); 2 published measurements supply the noise floor ([[datadecide-how-to-predict-best-pretraining-data-with-small-experiments]], [[a-hitchhikers-guide-to-scaling-law-estimation]]); 1 published measurement supplies the exponent-differs mechanism ([[scaling-laws-vs-model-architectures-how-does-inductive-bias-influence-scaling]]); 1 sibling investigator supplies the throughput denominator ([[interim-report-sustained-throughput-at-32k]]); 1 published measurement supplies the repetition relief valve ([[arxiv230516264v5-cscl-28-jun-2025]]). Against: [[cramming-training-a-language-model-on-a]] directly contradicts the "architecture matters" premise but only for compute-shape variants at 110M, which I read as a scope limit rather than a refutation. **0 measurements exist on this box** — the entire noise floor and every effect size in this note is external (T3), and the wall-clock is arithmetic (T4) anchored in T1/T2 throughput.
