---
title: 'Dr. Robert Li | Engram: A Critical Analysis of Conditional Memory as a Sparsity
  Axis - DeepSeek''s Adaptation of Classic N-grams for Early Layer Feature Representation'
id: dr-robert-li-engram-a-critical-analysis-of-conditional-memory-as-a-sparsity-axis
tags:
- cmp170hx-training-feasibility-a0d4fc
- engram
- critical-analysis
- training-vs-inference
- memory-accounting
created: '2026-10-27T10:12:55.356939Z'
updated: '2026-10-27T19:00:31.173539Z'
source: https://drli.blog/posts/analysis-engram-conditional-memory/
source_domain: drli.blog
fetched_at: '2026-10-27T10:12:54.713935Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: blog
deprecated: false
summary: 'Dr. Robert Li''s long adversarial/pedagogical analysis of the Engram paper
  (19 Jan 2026, ~15.4k words). Mostly a section-by-section walkthrough plus a classical
  n-gram tutorial, but it contributes three things the primary paper does not state
  as bluntly. (1) The inference/training asymmetry, spelled out explicitly: prefetch-and-overlap
  works at inference because indices are known from token IDs before the forward pass,
  but training needs gradients for every accessed embedding to travel back to host
  memory, plus optimizer state (Adam momentum and variance) and gradient accumulation
  across distributed batches - ''training requires bidirectional, latency-sensitive
  communication that doesn''t tolerate PCIe bottlenecks as gracefully''. (2) A memory-accounting
  worked example: a 100B model at FP16 needs 200 GB HBM (3x H100 80GB tensor-parallel
  or 2x H200 141GB), whereas an 8B backbone + 92B Engram keeps only ~16 GB on GPU
  with ~184 GB of tables in host DRAM, runnable on 1x RTX 4090 24GB + 256 GB host
  RAM - with the caveat that this buys memory capacity, not reasoning depth. (3) A
  gap table of claims validated vs asserted: the U-shape is tested at only two compute
  budgets (2e20, 6e20 FLOPs) while V3 uses ~1e25; log-linear Engram scaling is tested
  only to ~13B; the 100B offload claim is throughput-only with capability NOT tested
  at 100B; the multi-level Zipfian cache hierarchy is described with no empirical
  validation; context-extension gains are shown at 32k only, with no 128k or 1M-context
  testing. Also quantifies hash collision arithmetic (2e15 trigrams into 3e6 slots
  => ~700M n-grams per slot, ~11 collisions per head per batch, negligible across
  8 heads).'
utility_score: 13.0
---

## Engram: A Critical Analysis of Conditional Memory as a Sparsity Axis - DeepSeek's Adaptation of Classic N-grams for Early Layer Feature Representation
19 Jan 2026
[AI Research](https://drli.blog/posts/analysis-engram-conditional-memory/#!) [Large Language Models](https://drli.blog/posts/analysis-engram-conditional-memory/#!) [Neural Architecture](https://drli.blog/posts/analysis-engram-conditional-memory/#!) [Sparse Models](https://drli.blog/posts/analysis-engram-conditional-memory/#!) [Memory Systems](https://drli.blog/posts/analysis-engram-conditional-memory/#!) [DeepSeek](https://drli.blog/posts/analysis-engram-conditional-memory/#!)
**Paper:** Engram: Conditional Memory via Scalable Lookup **Authors:** DeepSeek-AI & Peking University **Date:** January 2026 **arXiv:** 2601.07372v1
### **_TL;DR_**
  * The Engram paper introduces **conditional memory** as a new sparsity axis for large language models, complementing the established paradigm of conditional computation (MoE). The central thesis is that Transformers lack a native knowledge lookup primitive, forcing them to simulate retrieval through expensive computation. Engram addresses this by storing N-gram pattern embeddings in hash-indexed tables, enabling retrieval of static local patterns while preserving the Transformer backbone for dynamic reasoning.
  * Under rigorous iso-parameter and iso-FLOPs constraints, Engram-27B outperforms the MoE-27B baseline across diverse benchmarks: knowledge-intensive tasks (MMLU +3.4), general reasoning (BBH +5.0), and code/math (HumanEval +3.0). The paper’s U-shaped allocation curve demonstrates that optimal sparse model design allocates approximately 20-25% of inactive parameters to Engram memory rather than MoE experts.
  * The architecture is strategically significant given documented constraints on Chinese AI compute capacity. Engram’s ability to offload parameters to host DRAM (bypassing HBM bottlenecks) and extract more capability per FLOP directly addresses hardware limitations. The paper’s structure—detailed concept validation at modest scale, infrastructure described but not benchmarked at frontier scale—matches DeepSeek’s historical publication-to-deployment pattern, suggesting high probability of inclusion in their forthcoming V4 frontier model.
  * Key technical innovations include: **Tokenizer compression** (surjective mapping collapsing semantically equivalent tokens with 23% vocabulary reduction), **Multi-head hashing** (collision-robust retrieval via independent hash functions), **Context-aware gating** (learned scalar gates filtering irrelevant or collision-contaminated retrievals), **mHC integration** (branch-specific gating enabling expressivity-efficiency tradeoffs), and **Memory-compute decoupling** (prefetch-and-overlap strategy enabling <3% overhead when offloading 100B parameters to host memory).
  * The paper opens significant unexplored territory, including domain-specialized Engram modules for high-accuracy applications (medical, legal, scientific) and the broader design space of memory-expressivity tradeoffs within the conditional memory paradigm.


## Sectional Summary
### Section 1: Introduction
The paper opens by identifying a fundamental limitation in Transformer architectures: the absence of a native knowledge lookup primitive. While MoE provides conditional computation (sparse expert activation), Transformers lack conditional memory (sparse knowledge retrieval). This forces models to reconstruct static patterns—named entities, collocations, idioms—through iterative computation at every forward pass.
Engram proposes treating conditional memory as a first-class modeling primitive. The key insight is that local N-gram patterns are deterministic given the input tokens, enabling hash-based retrieval rather than learned routing. This architectural choice decouples memory scaling from compute scaling: adding Engram parameters increases storage requirements but not per-token FLOPs.
The introduction frames the paper’s core contribution as answering the allocation question: given fixed total parameters and activated parameters, how should sparse capacity be distributed between MoE experts (conditional computation) and Engram embeddings (conditional memory)?
### Section 2: Method
#### Section 2.1: Tokenizer Compression
Standard subword tokenizers create multiple token IDs for semantically equivalent surface forms (“Apple”, “apple”, “ apple”, “APPLE”). This is catastrophic for N-gram lookup, as each variant maps to different hash slots.
Engram implements a surjective mapping P:V→ that collapses equivalent tokens via:
  * NFKC Unicode normalization
  * Lowercasing
  * Whitespace stripping
  * Diacritic removal


This achieves 23% vocabulary reduction, improving coverage per embedding slot and reducing hash collision probability.
#### Section 2.2: Multi-Head Hashing
The combinatorial space of N-grams (|V≈1015 for trigrams with 128k vocabulary) cannot be stored explicitly. Engram uses independent hash functions mapping N-grams to table slots of size :
ϕ:→{0,1,…,M−1}
Multi-head hashing provides collision robustness: if two N-grams collide in one hash function, they almost certainly differ in others. With heads and M=3×106 slots, the probability of total collision is approximately 10−52, making catastrophic collision effectively impossible.
#### Section 2.3: Context-Aware Gating
Static lookup cannot resolve polysemy (“bank” as financial institution vs. riverbank) or filter hash collisions. Engram introduces a learned gating mechanism:
=σ(RMSNorm(⋅RMSNorm())
The gate ∈(0,1) modulates the retrieved embedding based on compatibility with the Transformer’s hidden state . This reintroduces minimal dynamic computation ( for dot product) while preserving the efficiency benefits of static retrieval.
#### Section 2.4: Multi-Branch Integration with mHC
Engram integrates with Manifold-Constrained Hyper-Connections (mHC; Xie et al., 2025), which expand the residual stream to parallel branches. The design shares expensive components (embedding table, value projection) while separating cheap components (key projections, gates):
  * **Shared** : One embedding table , one value projection 
  * **Separate** : key projections WK, scalar gates 


This enables branch-specific decisions about memory utilization while amortizing storage costs.
#### Section 2.5: Decoupling Memory from Compute
The section describes two system designs:
**Training** : Standard model parallelism shards Engram tables across GPUs, using All-to-All communication to gather active embeddings. This distributes but does not eliminate HBM requirements.
**Inference** : Deterministic hash-based addressing enables prefetch-and-overlap strategies. Since indices are known before the forward pass, embeddings can be asynchronously retrieved from host DRAM via PCIe while preceding layers compute. Engram modules at layers 2 and 15 provide sufficient computation buffer to mask transfer latency.
The paper describes (but does not benchmark) a multi-level cache hierarchy exploiting Zipfian N-gram distribution: frequent patterns in GPU HBM, common patterns in host DRAM, rare patterns on NVMe SSD.
### Section 3: Scaling vs. Sparsity
#### Section 3.1: Allocation Under Fixed Constraints
The paper introduces the allocation ratio ρ∈[0,1], where determines the fraction of sparse parameters assigned to MoE experts versus Engram memory. Under fixed total and activated parameters, sweeping reveals a **U-shaped validation loss curve** :
  * ρ=100% (pure MoE): Loss = 1.7248
  * ρ≈75-80% (optimal): Loss = 1.7109
  * (Engram-dominated): Loss increases


The U-shape is replicated at two compute budgets (2×1020 and 6×1020 FLOPs), suggesting stability across regimes. The optimal allocation dedicates approximately 20-25% of sparse capacity to Engram.
#### Section 3.2: Infinite Memory Regime
With a fixed MoE backbone, sweeping Engram capacity from to slots reveals **log-linear scaling** : validation loss decreases linearly with log⁡(slots). This suggests Engram can scale efficiently beyond tested ranges.
Comparison with OverEncoding (Huang et al., 2025a) shows Engram extracts more value from equivalent memory budgets, attributed to deeper injection points, context-aware gating, and tokenizer compression.
### Section 4: Experiments
#### Section 4.1: Pre-training Setup
Models are trained for 50,000 steps on 262B tokens:
  * **Dense-4B** : 4.1B total, 3.8B activated
  * **MoE-27B** : 26.7B total, 3.8B activated (72 routed + 2 shared experts)
  * **Engram-27B** : 26.7B total, 3.8B activated (55 routed + 2 shared experts, 5.7B Engram)
  * **Engram-40B** : 39.5B total, 3.8B activated (55 routed + 2 shared experts, 18.5B Engram)


All models use DeepSeek-V3 tokenizer, MLA attention (DeepSeek-AI, 2024a), mHC (), and Muon optimizer.
#### Section 4.2: Pre-training Results
Engram-27B outperforms MoE-27B across all benchmark categories:  
| Category  | Representative Benchmark  | MoE-27B  | Engram-27B  |  
| --- | --- | --- | --- |  
| Knowledge  | MMLU  | 57.4  | 60.4  | +3.0  |  
| Reasoning  | BBH  | 50.9  | 55.9  | +5.0  |  
| Code  | HumanEval  | 37.8  | 40.8  | +3.0  |  
| Math  | GSM8K  | 58.4  | 60.6  | +2.2  |  
Engram-40B shows further gains on most benchmarks but underperforms Engram-27B on code tasks (HumanEval 38.4 vs 40.8). The paper attributes this to undertraining—the loss gap between Engram-40B and baselines continues widening at training end.
### Section 5: Long-Context Capability
#### Section 5.1: Experimental Setup
Models undergo YaRN context extension (Peng et al., 2023) from 4k to 32k tokens over 5,000 additional steps.
#### Section 5.2: Results and Analysis
The paper makes three controlled comparisons:
  1. **Iso-Loss** (46k vs 50k steps): Matching pre-training loss isolates architectural effects. Engram dramatically outperforms on complex retrieval: Multi-Query NIAH 97.0 vs 84.2 (+12.8), Variable Tracking 87.2 vs 77.0 (+10.2).
  2. **Iso-FLOPs** (50k vs 50k steps): Standard comparison shows Engram advantages compound with its better base quality.
  3. **Extreme** (41k vs 50k steps): Engram at 82% training compute matches MoE on LongPPL while exceeding on RULER tasks.


The mechanism: Engram handles local patterns via lookup, freeing attention capacity for global context management. Tasks requiring broad attention (Frequent Words Extraction: +26.3) show largest gains.
### Section 6: Analysis
#### Section 6.1: Effective Depth
LogitLens analysis (nostalgebraist, 2020) shows Engram representations converge to prediction-ready states earlier (lower KL divergence at early layers). CKA analysis (Kornblith et al., 2019) reveals Engram layer 5 representations match MoE layer ~12 representations for named entities.
Interpretation: By offloading static pattern reconstruction to lookup, Engram effectively increases model depth—early layers can immediately begin reasoning rather than spending capacity on feature composition.
#### Section 6.2: Structural Ablations
Layer placement sweep finds layer 2 optimal for single-module Engram (balances early intervention with contextual precision for gating). Multi-branch integration and context-aware gating are critical; depthwise convolution provides marginal benefit.
#### Section 6.3: Sensitivity Analysis
Suppressing Engram output during inference reveals functional specialization:
  * **Factual knowledge** (TriviaQA): Catastrophic collapse to 29% retained
  * **Reading comprehension** (C3): Resilient at 93% retained


This demonstrates Engram becomes the primary repository for parametric knowledge, while the backbone retains comprehension and reasoning capabilities.
#### Section 6.4: System Efficiency
Table 4 demonstrates 100B parameter Engram offloaded to host DRAM incurs <3% throughput overhead (8,858 vs 9,032 tokens/sec on 4B backbone). This validates the prefetch-and-overlap strategy.
### Section 7: Related Work
The paper positions Engram against:
  * **N-gram language models** (Shannon, 1948; Jurafsky & Martin, 2024): Engram modernizes the concept with learned embeddings and neural integration
  * **OverEncoding** (Huang et al., 2025a): Prior N-gram embedding work limited to input layer averaging
  * **Product key memory** (Lample et al., 2019): Attention-based retrieval vs. Engram’s hash-based deterministic lookup
  * **Retrieval-augmented generation** (Lewis et al., 2020): External document retrieval vs. Engram’s internal parametric memory


## Detailed Technical Analysis
### 1. Classical N-gram Models: Foundation and Connection to Engram
#### Lay Analogy: Your Phone’s Predictive Keyboard
Imagine texting a friend. You type **“I’ll meet you at the”** and your phone suggests **“airport”** , **“office”** , or **“usual”**. How does it know? It has memorized millions of text messages and learned that certain words frequently follow certain phrases.
Your phone doesn’t analyze the entire conversation—it just looks at the **last few words** and consults a giant lookup table: _“When people type ‘at the’, what do they usually type next?”_
This is exactly how an N-gram model works:
  * It memorizes patterns from training text
  * It only looks at the **immediate local context** (the last words)
  * Prediction is a **table lookup** , not computation


The “N” in N-gram refers to the window size. A **3-gram (trigram)** model looks at the previous 2 words to predict the next one.
#### Mathematical Foundation
##### The Goal: Assign Probability to Sequences
Given a sentence W=(,,…,), we want to compute —the probability of this exact word sequence occurring.
Using the **chain rule of probability** , we decompose this as:
P(,,…,)=∏P(∣,,…,w)
**Problem** : Computing P(∣,…,w) requires conditioning on the _entire_ history. For a vocabulary of size and sequence length , we’d need to estimate parameters—astronomically intractable.
##### The Markov Assumption (Key Simplification)
The N-gram model makes the **( N−1)th-order Markov assumption**: the probability of the next word depends only on the previous (N−1) words:
P(∣,…,w)≈P(∣wt−N+1,…,w)
This truncates history to a fixed-size window, making the model tractable.
##### Notation Summary  
| Symbol  | Meaning  |  
| --- | --- |  
| Word at position   |  
| Vocabulary (set of all words)  |  
| Vocabulary size  |  
| The “N” in N-gram (context window + target)  |  
| wt−N+1  | Shorthand for (wt−N+1,…,w) — the context  |  
| Count function (occurrences in training corpus)  |  
##### Maximum Likelihood Estimation
We estimate probabilities by **counting** co-occurrences in a training corpus:
PMLE(∣wt−N+1)=C(wt−N+1,…,w,)C(wt−N+1,…,w)
In plain English:
P(next word∣context)=Times we saw (context + next word) togetherTimes we saw (context) at all
##### Concrete Examples
**Bigram ()** : Conditions on 1 previous word
P(mat∣the)=C("the mat")C("the")=1,24789,432≈0.014
**Trigram ()** : Conditions on 2 previous words
P(mat∣on,the)=C("on the mat")C("on the")=3425,891≈0.058
Notice: The trigram probability is higher because “on the” provides more specific context than just “the”.
##### Full Sentence Probability
For the sentence **“the cat sat on the mat”** using a trigram model:
P(sentence)=P(the∣⟨s⟩,⟨s⟩)×P(cat∣⟨s⟩,the)×P(sat∣the,cat)×…
Where is a special start-of-sentence token.
#### Visual Representation: Trigram Decomposition

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRIGRAM MODEL (N=3)                          │
│                                                                 │
│   Sentence: "the cat sat on the mat"                           │
│                                                                 │
│   ┌─────────┬─────────┬─────────┐                              │
│   │ Context │ Context │ Target  │   Probability                │
│   │  w_{t-2}│  w_{t-1}│   w_t   │                              │
│   ├─────────┼─────────┼─────────┤                              │
│   │   <s>   │   <s>   │   the   │   P(the | <s>, <s>)          │
│   │   <s>   │   the   │   cat   │   P(cat | <s>, the)          │
│   │   the   │   cat   │   sat   │   P(sat | the, cat)          │
│   │   cat   │   sat   │   on    │   P(on | cat, sat)           │
│   │   sat   │   on    │   the   │   P(the | sat, on)           │
│   │   on    │   the   │   mat   │   P(mat | on, the)           │
│   │   the   │   mat   │  </s>   │   P(</s> | the, mat)         │
│   └─────────┴─────────┴─────────┘                              │
│                                                                 │
│   The sliding window moves through the sentence:                │
│                                                                 │
│   [<s>  <s>  the] cat  sat  on  the  mat  </s>                 │
│        [<s>  the  cat] sat  on  the  mat  </s>                 │
│             [the  cat  sat] on  the  mat  </s>                 │
│                  [cat  sat  on] the  mat  </s>                 │
│                       [sat  on  the] mat  </s>                 │
│                            [on  the  mat] </s>                 │
│                                 [the  mat  </s>]               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

```

### The Lookup Table Structure

```
┌──────────────────────────────────────────────────────────────────┐
│                  N-GRAM PROBABILITY TABLE                        │
│                  (Trigram Example)                               │
├──────────────────────────────────┬───────────────────────────────┤
│          Context (Key)           │      Next Word Probabilities  │
├──────────────────────────────────┼───────────────────────────────┤
│      "the cat"                   │  sat: 0.12, is: 0.08, ...    │
│      "cat sat"                   │  on: 0.25, down: 0.18, ...   │
│      "sat on"                    │  the: 0.35, a: 0.22, ...     │
│      "on the"                    │  mat: 0.18, floor: 0.15, ... │
│      "how are"                   │  you: 0.85, things: 0.08, ...|
│           ...                    │            ...                │
└──────────────────────────────────┴───────────────────────────────┘

                    ↓ At inference time ↓

        Input: "sat on"  →  Lookup  →  Output: P(the|sat,on) = 0.35
                              O(1)

```

### Handling Edge Cases: Smoothing
A critical problem: **unseen N-grams get probability zero** , which makes entire sentences have .
**Example** : If “quantum cat” never appeared in training:
P(sat∣quantum,cat)=C("quantum cat sat")C("quantum cat")==undefined
**Solutions** (smoothing techniques):
  1. **Laplace (Add-1) Smoothing** : Add 1 to all counts

PLaplace(∣w)=C(w,)+1C(w)+|V|
  1. **Kneser-Ney Smoothing** : Sophisticated interpolation using lower-order models
  2. **Backoff** : If trigram unseen, fall back to bigram, then unigram


### Computational Complexity  
| Operation  | Time Complexity  | Space Complexity  |  
| --- | --- | --- |  
| Training (counting)  |  where = corpus size  |  O(|V) worst case  |  
| Inference (lookup)  |  per prediction  | —  |  
| Storage  | —  |  where = unique N-grams  |  
The **lookup** is the key property that makes N-grams attractive—and what the Engram paper exploits.
### Key Properties Summary  
| Property  | Assessment  |  
| --- | --- |  
| Lookup complexity  |  —instant retrieval  |  
| Interpretability  | High—direct frequency counts  |  
| Long-range dependencies  | None—limited to (N−1) context  |  
| Generalization  | None—”cat sat” “dog sat”  |  
| Data sparsity  | Severe—most N-grams unseen  |  
### Connection to Engram
Engram modernizes N-gram models by:  
| Classic N-gram  | Engram  |  
| --- | --- |  
| Stores probability distributions  | Stores dense embedding vectors  |  
| Exact string matching  | Hash-based approximate matching  |  
| Zero probability for unseen N-grams  | Hash collisions handled by gating  |  
| No semantic generalization  | Tokenizer compression groups equivalent forms  |  
| Standalone model  | Module within Transformer backbone  |  
The core insight remains the same: **local patterns don’t require deep computation—they can be retrieved in time** , freeing neural network depth for tasks that actually require reasoning.
## 2. Tokenizer Compression: Canonical Projection for N-gram Efficiency
### The Problem: Tokenizer Fragmentation
Standard subword tokenizers (BPE, SentencePiece) are designed for **lossless text reconstruction** , not semantic coherence. This creates a proliferation of token IDs that represent the same underlying concept:

```
Raw text: "Apple"  → Token ID: 12847
Raw text: "apple"  → Token ID: 18234  
Raw text: " apple" → Token ID: 31092  (note the leading space)
Raw text: " Apple" → Token ID: 45123
Raw text: "APPLE"  → Token ID: 67891
Raw text: "äpple"  → Token ID: 89012  (diacritic variant)

```

All six tokens refer to the same semantic concept, but each has a completely different ID. For N-gram lookup, this is catastrophic:
  * The trigram **“the red apple”** and **“The red Apple”** would hash to entirely different embedding slots
  * You’d need to observe both variants in training to learn both patterns
  * The combinatorial explosion is severe: if each position has 6 variants, a trigram has =216 possible ID combinations for semantically identical content


### The Solution: Canonical Projection
The paper implements a **surjective mapping** P:V→ that collapses semantically equivalent tokens into canonical representatives:

```
P: V → V'  (surjective = many-to-one)

P(12847) = P(18234) = P(31092) = P(45123) = P(67891) = P(89012) = 7234
   ↑         ↑          ↑          ↑          ↑          ↑          ↑
"Apple"   "apple"   " apple"  " Apple"   "APPLE"   "äpple"   canonical
                                                              "apple"

```

The mapping applies several normalization steps:
  1. **NFKC Unicode Normalization** : Converts compatibility characters to canonical forms (ﬁ → fi, ² → 2, ä → a in some modes)
  2. **Lowercasing** : A → a
  3. **Whitespace stripping** : “ apple” → “apple”
  4. **Diacritic removal** (implied): café → cafe


### What Appendix C Reveals
The paper’s Table 6 shows the most aggressively merged groups:  
| Canonical Token  | Merge Count  | Original Variants  |  
| --- | --- | --- |  
|  `'␣'` (whitespace)  | 163  |  `\t`, `\n`, `\r`, `␣`, `␣␣`, `\n\n`, `␣␣␣`, `␣\n`, …  |  
| 54  |  `A`, `a`, `␣a`, `␣A`, `á`, `ä`, `ã`, `ą`, `à`, `å`, `â`, …  |  
| 40  |  `O`, `o`, `␣o`, `␣O`, `ó`, `ö`, `ô`, `õ`, `ő`, `ò`, …  |  
| 35  |  `E`, `e`, `␣e`, `␣E`, `é`, `è`, `ę`, `ě`, `ê`, …  |  
| 30  |  `I`, `i`, `␣I`, `␣i`, `í`, `ì`, `î`, `ï`, …  |  
The 23% vocabulary reduction means roughly 30,000 token IDs are collapsed into existing canonical forms.
### Concrete Step-by-Step Example
The complete Engram embedding process for a real input:
#### Input Sentence

```
"The Milky Way galaxy"

```

#### Step 1: Standard Tokenization (DeepSeek-V3 tokenizer)

```
Text:     "The Milky Way galaxy"
Tokens:   ["The", " Mil", "ky", " Way", " galaxy"]
Raw IDs:  [1847,   29341,  8472,  15234,  31847]

```

#### Step 2: Tokenizer Compression (Canonical Projection)
Apply to each raw ID:

```
Raw ID    Text        Normalization Steps              Canonical ID
------    ----        ---------------------            ------------
1847      "The"       lowercase → "the"                → 892
29341     " Mil"      strip space, lower → "mil"       → 4521  
8472      "ky"        lowercase → "ky"                 → 8470
15234     " Way"      strip space, lower → "way"       → 2847
31847     " galaxy"   strip space, lower → "galaxy"    → 12453

Canonical IDs: [892, 4521, 8470, 2847, 12453]

```

**Critical point** : If the input had been “THE MILKY WAY GALAXY” or “ the milky way galaxy”, the canonical IDs would be **identical**.
#### Step 3: Form N-gram Contexts
For a trigram model (), extract suffix N-grams at each position:

```
Position t=0: g_{0,3} = (<s>, <s>, 892)         → "the"
Position t=1: g_{1,3} = (<s>, 892, 4521)        → "the mil"
Position t=2: g_{2,3} = (892, 4521, 8470)       → "the milky"
Position t=3: g_{3,3} = (4521, 8470, 2847)      → "milky way"
Position t=4: g_{4,3} = (8470, 2847, 12453)     → "way galaxy"

```

#### Step 4: Multi-Head Hashing
For each N-gram, apply different hash functions to get embedding indices:

```
For g_{3,3} = (4521, 8470, 2847) representing "milky way":

Hash Head 1: φ_{3,1}(4521, 8470, 2847) = 2847391 mod 3000017 = 847374
Hash Head 2: φ_{3,2}(4521, 8470, 2847) = 9182734 mod 3000017 = 182700
...
Hash Head 8: φ_{3,8}(4521, 8470, 2847) = 1928374 mod 3000017 = 928357

```

The hash function is multiplicative-XOR:
ϕ(,,)=((⋅)⊕(⋅)⊕(⋅))modM
Where ,, are different prime multipliers per head, and is a prime table size.
#### Step 5: Embedding Retrieval
Look up embeddings from each table:

```
e_{3,3,1} = E_{3,1}[847374]  ∈ ℝ^{d/K}    # Head 1 embedding
e_{3,3,2} = E_{3,2}[182700]  ∈ ℝ^{d/K}    # Head 2 embedding
...
e_{3,3,8} = E_{3,8}[928357]  ∈ ℝ^{d/K}    # Head 8 embedding

```

#### Step 6: Concatenate Across Heads and N-gram Orders

```
For position t=3 ("way"):

From bigrams (N=2):
  e_{3,2} = [e_{3,2,1} ∥ e_{3,2,2} ∥ ... ∥ e_{3,2,8}]  # "milky" → "way"

From trigrams (N=3):  
  e_{3,3} = [e_{3,3,1} ∥ e_{3,3,2} ∥ ... ∥ e_{3,3,8}]  # "the milky" → "way"

Final memory vector:
  e₃ = [e_{3,2} ∥ e_{3,3}] ∈ ℝ^{d_{mem}}

```

#### Step 7: Context-Aware Gating
The retrieved embedding is static—it doesn’t know the actual context. The gating mechanism modulates it:
=⋅ =⋅ =σ(RMSNorm(⋅RMSNorm()) =⋅
If the context (from preceding Transformer layers) is incompatible with the retrieved memory (e.g., hash collision retrieved “Milky Way candy bar” context when we need “Milky Way galaxy”), and the memory is suppressed.
### Visual Diagram of the Full Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENGRAM EMBEDDING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: "The Milky Way"                                                     │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────┐                                                    │
│  │  STANDARD TOKENIZER │                                                    │
│  │  (BPE/SentencePiece)│                                                    │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  Raw IDs: [1847, 29341, 8472, 15234]                                        │
│    "The"   " Mil"  "ky"  " Way"                                             │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                                │
│  │     TOKENIZER COMPRESSION P: V → V'     │                                │
│  │  ┌───────────────────────────────────┐  │                                │
│  │  │ • NFKC normalization              │  │                                │
│  │  │ • Lowercasing                     │  │                                │
│  │  │ • Whitespace stripping            │  │                                │
│  │  │ • Diacritic removal               │  │                                │
│  │  └───────────────────────────────────┘  │                                │
│  └──────────┬──────────────────────────────┘                                │
│             │                                                               │
│             ▼                                                               │
│  Canonical IDs: [892, 4521, 8470, 2847]                                     │
│     "the"   "mil"   "ky"   "way"                                            │
│             │                                                               │
│             │     ┌──────────────────────────────────────────┐              │
│             │     │  KEY INSIGHT: These canonical IDs are    │              │
│             │     │  IDENTICAL for "THE MILKY WAY",          │              │
│             │     │  "the milky way", " The Milky Way", etc. │              │
│             │     └──────────────────────────────────────────┘              │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                                │
│  │         FORM N-GRAM CONTEXTS            │                                │
│  │                                         │                                │
│  │  Bigrams:   (the,mil) (mil,ky) (ky,way) │                                │
│  │  Trigrams:  (the,mil,ky) (mil,ky,way)   │                                │
│  └──────────┬──────────────────────────────┘                                │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                                │
│  │         MULTI-HEAD HASHING              │                                │
│  │                                         │                                │
│  │  For each N-gram, K hash functions:     │                                │
│  │                                         │                                │
│  │  (mil,ky,way) ──┬── φ₁ ──→ idx: 847374  │                                │
│  │                 ├── φ₂ ──→ idx: 182700  │                                │
│  │                 ├── φ₃ ──→ idx: 293847  │                                │
│  │                 └── ...                 │                                │
│  └──────────┬──────────────────────────────┘                                │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              EMBEDDING TABLE LOOKUP (O(1))                      │        │
│  │                                                                 │        │
│  │   E_{3,1}[847374] ──→ [0.12, -0.34, 0.87, ...]  (d/K dims)     │        │
│  │   E_{3,2}[182700] ──→ [0.45, 0.23, -0.11, ...]  (d/K dims)     │        │
│  │   ...                                                           │        │
│  │                                                                 │        │
│  │   Concatenate all heads & N-gram orders:                        │        │
│  │   e_t = [e_{2-gram} ∥ e_{3-gram}] ∈ ℝ^{d_{mem}}                │        │
│  └──────────┬──────────────────────────────────────────────────────┘        │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              CONTEXT-AWARE GATING                               │        │
│  │                                                                 │        │
│  │   h_t ─────────────────────┐                                    │        │
│  │   (from Transformer)       │                                    │        │
│  │                            ▼                                    │        │
│  │   e_t ───→ W_K ───→ k_t ───┼───→ α_t = σ(h_t · k_t / √d)       │        │
│  │       └──→ W_V ───→ v_t ───┼───→ ṽ_t = α_t · v_t               │        │
│  │                            │                                    │        │
│  │   If context mismatches retrieved memory: α_t → 0 (suppress)    │        │
│  │   If context aligns with retrieved memory: α_t → 1 (use it)     │        │
│  └──────────┬──────────────────────────────────────────────────────┘        │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                                │
│  │   RESIDUAL CONNECTION TO BACKBONE       │                                │
│  │                                         │                                │
│  │   H^(ℓ) ← H^(ℓ) + Conv(ṽ_t)            │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Why This Matters for N-gram Efficiency
#### Without Tokenizer Compression
Consider training on a corpus with these occurrences:

```
"the Milky Way" appears 10,000 times
"The Milky Way" appears 8,000 times  
"THE MILKY WAY" appears 500 times
" the milky way" appears 3,000 times

```

Without compression, these are **four separate N-gram entries** , each with fewer training examples. The embedding for each variant is learned independently.
#### With Tokenizer Compression
All 21,500 occurrences contribute to a **single canonical N-gram** :

```
(the, milky, way) → single embedding learned from 21,500 examples

```

This provides:
  1. **Better statistics** : More training signal per pattern
  2. **Smaller tables** : 23% fewer slots needed
  3. **Better generalization** : Rare variants (like “THE MILKY WAY”) benefit from common variants
  4. **Reduced hash collisions** : Fewer unique N-grams means lower collision probability per slot


### The Surjective Function in Practice
The mapping is implemented as a precomputed lookup table:

```
# Pseudocode for tokenizer compression
class TokenizerCompressor:
    def __init__(self, tokenizer):
        self.projection = {}  # Raw ID → Canonical ID
        
        # Group tokens by normalized form
        normalized_groups = defaultdict(list)
        for token_id in range(tokenizer.vocab_size):
            token_text = tokenizer.decode([token_id])
            canonical = self.normalize(token_text)
            normalized_groups[canonical].append(token_id)
        
        # Assign canonical IDs
        canonical_id = 0
        for canonical_text, raw_ids in normalized_groups.items():
            for raw_id in raw_ids:
                self.projection[raw_id] = canonical_id
            canonical_id += 1
        
        self.compressed_vocab_size = canonical_id  # ~77% of original
    
    def normalize(self, text):
        text = unicodedata.normalize('NFKC', text)  # Unicode normalization
        text = text.lower()                          # Lowercase
        text = text.strip()                          # Strip whitespace
        # Additional normalizations...
        return text
    
    def compress(self, token_ids):
        return [self.projection[tid] for tid in token_ids]

```

The compression happens **only for Engram indexing** —the main Transformer backbone still uses the original token IDs and embeddings. This is crucial: you want the model to distinguish “Apple” (company) from “apple” (fruit) in its representations, but for N-gram pattern matching, the surface form variations shouldn’t matter.
## 3. Multi-Head Hashing: Principled Collision Mitigation
### The Combinatorial Problem
For vocabulary size |V|=128,000 and trigrams ():
|possible trigrams|=|V=128,0003≈2.1×1015
You cannot allocate 2 quadrillion embedding slots. So you must compress this space.
### Single-Head Hashing: The Naive Approach
A single hash function maps the astronomical N-gram space to a manageable table:
ϕ:→{0,1,…,M−1}
Where might be 3 million (a prime, for better distribution).

```
("the", "milky", "way")  ──φ──→  slot 847,374
("quantum", "field", "theory")  ──φ──→  slot 2,391,847
("the", "red", "apple")  ──φ──→  slot 847,374   ← COLLISION!

```

**The problem** : When two unrelated N-grams collide, they share the same embedding. The model learns a muddled average of all colliding patterns. With 2×1015 N-grams and 3×106 slots, the expected collisions per slot is ~700 million N-grams—catastrophic.
But most N-grams are vanishingly rare or never occur in training. The _effective_ collision rate depends on the training distribution, which follows Zipf’s law. Still, collisions are inevitable and damaging with a single head.
### Multi-Head Hashing: Collision Mitigation
The key insight is probabilistic: **if two N-grams collide in one hash function, they almost certainly won’t collide in independent hash functions**.
With heads, each with table size :
P(collision in all heads)=()K
For M=3,000,017 (prime) and :
P(total collision)=(13×106)8≈10−52
This is astronomically unlikely. In practice, two N-grams will share _some_ heads but not _all_ heads:

```
N-gram A: "the milky way"
  Head 1: slot 847,374  ←─┐
  Head 2: slot 182,700     │ collision
  Head 3: slot 2,918,374   │
  Head 4: slot 501,283  ←──┼─┐
  Head 5: slot 1,847,291   │ │
  Head 6: slot 928,174     │ │
  Head 7: slot 2,103,847   │ │
  Head 8: slot 384,192     │ │
                           │ │
N-gram B: "the red apple"  │ │
  Head 1: slot 847,374  ←──┘ │ same slot (collision)
  Head 2: slot 2,847,123     │ different
  Head 3: slot 918,234       │ different
  Head 4: slot 501,283  ←────┘ same slot (collision)
  Head 5: slot 2,918,473       different
  Head 6: slot 129,384         different
  Head 7: slot 1,029,384       different
  Head 8: slot 2,918,473       different

```

The concatenated embeddings are:
=[[847374]‖[182700]‖[2918374]‖…‖[384192]] =[[847374]‖[2847123]‖[918234]‖…‖[2918473]]
Even with 2 collisions out of 8 heads, 75% of the embedding dimensions are distinct. The representations remain distinguishable.
### Visual: How Multi-Head Reduces Collision Damage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE HEAD vs MULTI-HEAD HASHING                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SINGLE HEAD (K=1):                                                         │
│  ═══════════════════                                                        │
│                                                                             │
│  "the milky way"  ───┐                                                      │
│                      ├───→ slot 847,374 ───→ [shared embedding]             │
│  "the red apple"  ───┘                       (CONTAMINATED)                 │
│                                                                             │
│  Problem: 100% of embedding is shared on collision                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MULTI-HEAD (K=8):                                                          │
│  ═══════════════════                                                        │
│                                                                             │
│  "the milky way"     Head 1   Head 2   Head 3   Head 4   ...   Head 8      │
│         │              │        │        │        │              │          │
│         └──────────→ [847K]  [183K]  [2.9M]  [501K]   ...    [384K]        │
│                        ↓        ↓        ↓        ↓              ↓          │
│                      [ e₁  ∥   e₂   ∥   e₃   ∥   e₄   ∥ ... ∥   e₈  ]     │
│                        ↑                  ↑                                 │
│                     collision           unique                              │
│                        ↓                  ↓                                 │
│                      [ e₁  ∥   e₂'  ∥   e₃'  ∥   e₄   ∥ ... ∥   e₈' ]     │
│                        ↓        ↓        ↓        ↓              ↓          │
│         ┌──────────→ [847K]  [2.8M]  [918K]  [501K]   ...    [2.9M]        │
│         │              │        │        │        │              │          │
│  "the red apple"     Head 1   Head 2   Head 3   Head 4   ...   Head 8      │
│                                                                             │
│  Result: Only 2/8 = 25% of embedding dimensions collide                     │
│          75% of representation is DISTINCT                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COLLISION PROBABILITY:                                                     │
│                                                                             │
│    K=1:  P(full collision) = 1/M           ≈ 3×10⁻⁷                        │
│    K=8:  P(full collision) = (1/M)^8       ≈ 10⁻⁵²                         │
│                                                                             │
│  The multi-head design makes "catastrophic collision" essentially           │
│  impossible while gracefully degrading on partial collisions.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Theoretical Foundation: Established Techniques
Multi-head hashing is a variant of several established methods:
#### 1. Feature Hashing (The “Hashing Trick”)
Introduced by Weinberger et al. (2009) for high-dimensional sparse features:
ϕ(x)=ξ(j)⋅⋅eh(j)
Where is a hash function and is a sign function to reduce bias. Used extensively in large-scale ML (Vowpal Wabbit, scikit-learn’s HashingVectorizer).
#### 2. Count-Min Sketch
A probabilistic data structure using multiple hash functions to estimate frequencies (Cormode & Muthukrishnan, 2005):

```
     h₁  h₂  h₃  h₄
    ┌───┬───┬───┬───┐
    │ 3 │ 0 │ 1 │ 2 │  ← row 1
    ├───┼───┼───┼───┤
    │ 1 │ 4 │ 0 │ 1 │  ← row 2
    ├───┼───┼───┼───┤
    │ 2 │ 1 │ 3 │ 0 │  ← row 3
    └───┴───┴───┴───┘
    
Estimate = min(counts across all hash positions)

```

The minimum across heads gives a collision-robust estimate.
#### 3. Bloom Filters
Test set membership with multiple hash functions—false positives possible, false negatives impossible. Same probabilistic principle: collision in all hashes is exponentially unlikely.
#### 4. Random Projections (Johnson-Lindenstrauss)
The JL lemma guarantees that random projections preserve pairwise distances (Johnson & Lindenstrauss, 1984):
‖f(x)−f(y)=(1±ϵ)‖x−y
With high probability, for appropriate target dimension.
### Why Semantically-Uninformed Hashing Works
The hash functions are **semantically uninformed** —”milky way” (galaxy) and “milky way” (candy bar) might partially collide, while “milky way” and “andromeda galaxy” won’t benefit from any shared structure.
But this isn’t a bug, it’s a feature:
  1. **Semantic similarity is handled by the Transformer backbone** , not Engram
  2. **Engram stores surface-level N-gram patterns** , which are about co-occurrence, not meaning
  3. **The gating mechanism provides semantic filtering** after retrieval


The hash function’s job is simply to provide a _consistent_ , _deterministic_ , _uniform_ mapping. It doesn’t need to be smart—it just needs to avoid systematic bias.
### The Mathematical Guarantee
For independent hash functions with table size , the expected number of “clean” dimensions (no collision with any other active N-gram) follows a balls-into-bins analysis.
If there are active N-grams in a batch:
E[collisions per head]=n−M(1−(1−)n)≈
For typical batch sizes (n∼4096 tokens 2 N-gram orders =8192) and :
E[collisions per head]≈819222×3×106≈11
So ~11 collisions per head per batch, but the probability of the same pair colliding across all 8 heads is negligible.
### The Gating Mechanism as Second Defense
Even when partial collisions occur, the context-aware gating provides semantic filtering:

```
# Retrieved embedding might contain collision noise
e_t = retrieve_ngram_embedding(context)  # Partially contaminated

# But the Transformer hidden state knows the true context
h_t = transformer_layers(input)  # Has global context

# Gating checks: "Does this retrieved memory match my context?"
alpha = sigmoid(dot(h_t, W_K @ e_t) / sqrt(d))

if context_matches_memory:
    alpha → 1.0  # Use the memory
else:
    alpha → 0.0  # Suppress (probably collision noise)

```

This is why Figure 7 in the paper (gating visualization) shows selective activation—the model learns to ignore retrieved embeddings when they don’t match the actual context.
### Summary Assessment  
| Aspect  | Assessment  |  
| --- | --- |  
| Is it dimensionality reduction?  | Yes—from to dimensions  |  
| Is it semantically informed?  | No—the hash function is semantically uninformed, but the design is principled  |  
| Is it lossy?  | Yes, but multi-head makes catastrophic loss exponentially unlikely  |  
| Is it novel?  | No—it’s a neural adaptation of feature hashing / count-min sketch  |  
| Why does it work?  | Probabilistic guarantees + gating mechanism + Zipfian sparsity of N-grams  |  
The semantically-uninformed mapping is actually a strength: it requires no learning, is deterministic (enabling prefetching), and provides theoretical guarantees on collision rates. The semantic heavy lifting is delegated to the Transformer backbone and the learned gating mechanism.
## 4. Context-Aware Gating: Lightweight Dynamic Filtering
### The Fundamental Tension
The paper presents a dichotomy:
  * **Static memory** : lookup, no computation, context-blind
  * **Dynamic computation** : O() per layer, full expressivity, context-aware


But pure static lookup has a fatal flaw: **the same N-gram can mean different things in different contexts.**

```
"The bank was steep"        → riverbank (geography)
"The bank was closed"       → financial institution
"The bank shot was perfect" → billiards term

```

All three share the bigram “The bank”, which would retrieve the **same static embedding**. Without some mechanism to disambiguate, Engram would inject irrelevant or contradictory information.
### The Gating Computation
#### The Static Component (Context-Independent)

```
# These depend ONLY on the N-gram hash—identical every time "the bank" appears
e_t = lookup_ngram_embedding(token_ids)     # Static: ℝ^{d_mem}
k_t = W_K @ e_t                              # Static: ℝ^d (linear projection)
v_t = W_V @ e_t                              # Static: ℝ^d (linear projection)

```

At this point, everything is deterministic. Given the same input tokens, you get identical and regardless of surrounding context.
#### The Dynamic Component (Context-Dependent)

```
# h_t comes from preceding Transformer layers—FULLY context-dependent
h_t = transformer_output[t]  # Has seen full sequence via attention

# The gate computation is where static meets dynamic
q_t = RMSNorm(h_t)           # Normalized query (dynamic)
k_t_norm = RMSNorm(k_t)      # Normalized key (static)

# Scalar attention score
alpha_t = sigmoid(dot(q_t, k_t_norm) / sqrt(d))  # ∈ (0, 1)

# Gated output
v_tilde = alpha_t * v_t      # Dynamic scaling of static vector

```

#### Geometric Interpretation
The output lives on a **one-dimensional ray** in embedding space:

```
                    v_t (static direction)
                     ↑
                     │
                     │  α=1.0 (full activation)
                     │
                     ●──────────────────→
                     │
                     │  α=0.5 (partial)
                     │
         ───────────●───────────────────→
                     │
                     │  α≈0 (suppressed)
                     ●
                   origin

```

**The direction is fixed; only the magnitude varies.** This is fundamentally less expressive than full dynamic computation, where both direction and magnitude are context-dependent.
### Comparison: Gating vs. Full Dynamic Computation
#### Full Attention (What Transformers Do)

```
# Every aspect is context-dependent
Q = W_Q @ H          # Queries from all positions
K = W_K @ H          # Keys from all positions  
V = W_V @ H          # Values from all positions

attention = softmax(Q @ K.T / sqrt(d))  # Full N×N interaction
output = attention @ V                   # Weighted combination of ALL values

```

**Expressivity** : Output can be any linear combination of value vectors, with weights determined dynamically by full sequence context.
**Cost** : O(⋅d) for sequence length , dimension 
#### Full FFN (What MoE Experts Do)

```
# Arbitrary nonlinear transformation
hidden = activation(W_1 @ x + b_1)    # Up-project
output = W_2 @ hidden + b_2            # Down-project

```

**Expressivity** : Can approximate any continuous function (universal approximation)
**Cost** : O(d⋅) where is typically 
#### Engram Gating (What This Paper Does)

```
# Scalar modulation of static vector
alpha = sigmoid(dot(h_t, k_t) / sqrt(d))  # Single dot product
output = alpha * v_t                       # Scalar multiplication

```

**Expressivity** : Output constrained to ray defined by ; only magnitude varies
**Cost** : for the dot product + for the scaling = 
#### Cost Comparison Table  
| Mechanism  | FLOPs per Token  | Expressivity  |  
| --- | --- | --- |  
| Full Attention  | O(N⋅d)  | Any weighted combination of values  |  
| FFN Layer  | O(d⋅)≈O(4)  | Universal function approximation  |  
| MoE Expert  | O(d⋅num_experts)  | Same, but sparse  |  
| **Engram Gating**  | **Scalar scaling of fixed direction**  |  
Engram gating is **orders of magnitude cheaper** but **dramatically less expressive**.
### The Computational Hierarchy
Engram creates a **tiered computation strategy** :

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPUTATIONAL HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: Static Lookup (O(1))                                              │
│  ════════════════════════════                                              │
│  • Hash N-gram → retrieve embedding                                         │
│  • Zero computation, pure memory access                                     │
│  • Handles: stereotyped patterns, named entities, collocations             │
│                                                                             │
│                          ↓                                                  │
│                                                                             │
│  TIER 2: Lightweight Gating (O(d))                                         │
│  ══════════════════════════════════                                        │
│  • Single dot product + sigmoid                                            │
│  • Filters out irrelevant/colliding retrievals                             │
│  • Handles: polysemy, hash collisions, context mismatch                    │
│                                                                             │
│                          ↓                                                  │
│                                                                             │
│  TIER 3: Full Transformer Computation (O(d²))                              │
│  ════════════════════════════════════════════                              │
│  • Attention + FFN/MoE                                                     │
│  • Full dynamic reasoning                                                  │
│  • Handles: composition, inference, long-range dependencies                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

The hypothesis is that **many tokens don’t need Tier 3 processing** —they’re predictable from local context and can be handled by Tiers 1+2. This frees Tier 3 capacity for tokens that actually require reasoning.
### Expressivity Limitations of Gating
#### 1. Cannot Change Direction

```
Context A: "The river bank was steep"
Context B: "The investment bank was profitable"

Both retrieve: v_t = [0.3, -0.2, 0.8, ...]  (same static vector)

Context A gating: α_A * v_t = 0.9 * [0.3, -0.2, 0.8, ...] 
Context B gating: α_B * v_t = 0.7 * [0.3, -0.2, 0.8, ...]

The outputs point in THE SAME DIRECTION—only magnitude differs.
The model cannot rotate "bank" toward financial vs. geographical meanings.

```

**Who handles this?** The Transformer backbone. The Engram contribution is added residually, and subsequent attention layers can still distinguish contexts.
#### 2. Cannot Compose Information

```
"The large red ball" 

Engram retrieves:
  - bigram ("large", "red") → e_1
  - bigram ("red", "ball") → e_2
  - trigram ("large", "red", "ball") → e_3

But these are independent lookups—Engram cannot compute
the COMPOSITIONAL meaning "a ball that is both large and red"
beyond what's stored in e_3.

```

**Who handles this?** The Transformer’s attention mechanism, which can dynamically compose features across positions.
#### 3. Cannot Reason

```
"If it's raining, then the ground is wet. It's raining. Therefore..."

No static N-gram lookup can complete this—it requires:
  - Understanding conditional structure
  - Applying modus ponens
  - Generating "the ground is wet"

```

**Who handles this?** The full Transformer stack, which the paper argues now has more “effective depth” because it’s not wasting early layers on pattern matching.
### The Spectrum of Staticness

```
FULLY STATIC                                              FULLY DYNAMIC
     │                                                           │
     ▼                                                           ▼
┌─────────┬──────────┬───────────┬──────────────┬───────────────┐
│ Raw     │ Engram   │ Engram    │ Per-head     │ Full          │
│ N-gram  │ (single  │ (multi-   │ Gating       │ Attention     │
│ Lookup  │ gate)    │ branch)   │ (alternative)│               │
├─────────┼──────────┼───────────┼──────────────┼───────────────┤
│ e_t     │ α·v_t    │ Σ α_m·v_t │ Σ α_k·e_k    │ softmax(QK)·V │
├─────────┼──────────┼───────────┼──────────────┼───────────────┤
│ O(1)    │ O(d)     │ O(M·d)    │ O(K·d)       │ O(N·d)        │
├─────────┼──────────┼───────────┼──────────────┼───────────────┤
│ No      │ 1D ray   │ M rays    │ K-dim        │ Full span     │
│ control │          │ (M=4)     │ subspace     │ of values     │
└─────────┴──────────┴───────────┴──────────────┴───────────────┘

```

Engram with multi-branch gating sits at a sweet spot: **enough dynamism to filter collisions and polysemy, cheap enough to not defeat the purpose of static lookup.**
### Alternative Design: Per-Head Gating
The current architecture uses a single scalar gate for the entire retrieved embedding:

```
e_t = concat([e_head_1, e_head_2, ..., e_head_K])  # All heads retrieved
alpha = compute_gate(h_t, e_t)                      # Single scalar
output = alpha * project(e_t)                       # All-or-nothing

```

An alternative would be **per-head gating:**

```
e_heads = [e_head_1, e_head_2, ..., e_head_K]       # All heads retrieved
alphas = [compute_gate(h_t, e_k) for e_k in e_heads]  # K separate gates
output = concat([α_k * e_k for α_k, e_k in zip(alphas, e_heads)])

```

**Why this could work:**
  1. **Selective collision filtering** : If head 3 collided but heads 1,2,4-8 didn’t, you could suppress only head 3
  2. **Feature-specific modulation** : Different heads might capture different aspects
  3. **Smoother expressivity gradient** : -dimensional control surface instead of scalar


**Why the paper doesn’t do this:**
  1. **Computational cost** : separate dot products instead of 1
  2. **Multi-branch integration already provides this** : mHC with branches offers similar granularity
  3. **Empirical sufficiency** : The ablations suggest current design works well enough


### Empirical Validation: Does Gating Actually Discriminate?
Figure 7 in the paper shows gating activation qualitatively:

```
"Only Alexander the Great could tame the horse Bucephalus."

Gating activation (α values):
  "Alexander"     → low (not end of pattern)
  "the"           → low  
  "Great"         → HIGH (completes "Alexander the Great")
  "could"         → low
  "tame"          → low
  "the"           → low
  "horse"         → low
  "Bucephalus"    → medium (entity, but less stereotyped)

```

The gating successfully identifies where static patterns END—which is exactly where the retrieved embedding is most reliable (the full N-gram was seen in training). At positions mid-pattern or on novel combinations, gating suppresses the retrieval.
## 5. mHC-Engram Integration: Expressivity-Efficiency Tradeoffs
### Manifold-Constrained Hyper-Connections (mHC) Background
The paper assumes familiarity with mHC (Xie et al., 2025).
#### Standard Residual Connection
Traditional Transformers use a single residual stream:
h(ℓ+1)=h+f(h)
All information flows through one pathway. The residual connection preserves the input, and adds new information.
#### Hyper-Connections (HC)
Hyper-Connections expand this to parallel branches with learnable mixing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STANDARD RESIDUAL vs HYPER-CONNECTIONS                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STANDARD RESIDUAL (M=1):                                                   │
│                                                                             │
│       h ────────────────┬────────────────→ h + f(h)                        │
│                         │                     ↑                             │
│                         └───→ f(·) ───────────┘                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HYPER-CONNECTIONS (M=4):                                                   │
│                                                                             │
│       h₁ ──────┬───────────────────────────────┬──→ h₁'                    │
│                │         ╲    learnable        │                            │
│       h₂ ──────┼────────────╲   mixing  ───────┼──→ h₂'                    │
│                │              ╲  weights       │                            │
│       h₃ ──────┼────────────────╲──────────────┼──→ h₃'                    │
│                │                  ╲            │                            │
│       h₄ ──────┼────────────────────╲──────────┼──→ h₄'                    │
│                │                      ↓        │                            │
│                └───────────→ f(·) ─────────────┘                            │
│                                                                             │
│  Each output h_m' is a learned combination of:                              │
│    • All input branches h₁...h₄                                            │
│    • The transformation output f(combined_input)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

#### The “Manifold-Constrained” Part
The key innovation in mHC is constraining the connection weights to lie on a manifold that preserves certain geometric properties (like gradient flow stability). Without this constraint, having branches with arbitrary mixing weights can cause training instabilities.
Mathematically:
Hout=A⋅+B⋅f(C⋅)
where A,B,C satisfy manifold constraints ensuring stable gradients.
### The Engram-mHC Integration Design
#### What’s Shared (Efficiency)

```
# ONE embedding table for all branches
E = shared_ngram_embedding_table  # Massive: billions of parameters

# ONE value projection for all branches  
W_V = shared_value_projection     # Shape: d × d_mem

# The expensive retrieval happens ONCE
e_t = E[hash(ngram)]              # O(1) lookup, same for all branches
v_t = W_V @ e_t                   # O(d × d_mem), computed once

```

#### What’s Separate (Expressivity)

```
# M DIFFERENT key projections, one per branch
W_K = [W_K_1, W_K_2, W_K_3, W_K_4]  # M separate matrices

# Each branch computes its OWN gate
for m in range(M):
    k_t_m = W_K[m] @ e_t                           # Branch-specific key
    alpha_m = sigmoid(dot(h_t[m], k_t_m) / sqrt(d))  # Branch-specific gate
    u_t[m] = alpha_m * v_t                          # Branch-specific output

```

### Visual Representation of the Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENGRAM × mHC INTEGRATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         N-gram tokens                                       │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │   Hash + Lookup │  ← SHARED (O(1), one retrieval)     │
│                    │   E[hash(·)]    │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│                         e_t (static embedding)                              │
│                             │                                               │
│              ┌──────────────┼──────────────┐                               │
│              │              │              │                                │
│              ▼              ▼              ▼                                │
│         ┌────────┐    ┌────────┐    ┌────────┐                             │
│         │  W_V   │    │  W_V   │    │  W_V   │  ← SHARED (one projection) │
│         │(shared)│    │(shared)│    │(shared)│                             │
│         └───┬────┘    └───┬────┘    └───┬────┘                             │
│             │             │             │                                   │
│             ▼             ▼             ▼                                   │
│            v_t           v_t           v_t      (identical value vectors)  │
│             │             │             │                                   │
│             │             │             │                                   │
│         ┌───┴───┐     ┌───┴───┐     ┌───┴───┐                              │
│         │ W_K_1 │     │ W_K_2 │     │ W_K_3 │   ← SEPARATE (M projections)│
│         └───┬───┘     └───┬───┘     └───┬───┘                              │
│             │             │             │                                   │
│             ▼             ▼             ▼                                   │
│           k_t_1         k_t_2         k_t_3    (different key vectors)     │
│             │             │             │                                   │
│      ┌──────┴──────┐ ┌────┴────┐ ┌─────┴─────┐                             │
│      │   h_t[1]    │ │  h_t[2] │ │   h_t[3]  │  ← From mHC branches       │
│      │   (query)   │ │ (query) │ │  (query)  │                             │
│      └──────┬──────┘ └────┬────┘ └─────┬─────┘                             │
│             │             │             │                                   │
│             ▼             ▼             ▼                                   │
│      α_1 = σ(q·k_1) α_2 = σ(q·k_2) α_3 = σ(q·k_3)  SEPARATE gates        │
│             │             │             │                                   │
│             ▼             ▼             ▼                                   │
│         α_1 · v_t     α_2 · v_t     α_3 · v_t    (different magnitudes)   │
│             │             │             │                                   │
│             ▼             ▼             ▼                                   │
│         Branch 1      Branch 2      Branch 3     → to mHC residual        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### The Expressivity-Efficiency Spectrum

```
MAXIMUM EFFICIENCY                                    MAXIMUM EXPRESSIVITY
(Minimum Parameters)                                  (Maximum Parameters)
        │                                                       │
        ▼                                                       ▼
┌───────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
│   Config A    │   Config B    │   Config C    │   Config D    │   Config E    │
│               │               │  (PAPER'S     │               │               │
│               │               │   CHOICE)     │               │               │
├───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ • 1 embed     │ • 1 embed     │ • 1 embed     │ • 1 embed     │ • M embed     │
│   table       │   table       │   table       │   table       │   tables      │
│ • 1 W_V       │ • 1 W_V       │ • 1 W_V       │ • M W_V       │ • M W_V       │
│ • 1 W_K       │ • M W_K       │ • M W_K       │ • M W_K       │ • M W_K       │
│ • 1 gate      │ • 1 gate      │ • M gates     │ • M gates     │ • M gates     │
│   (shared)    │   (shared)    │ • 1 conv      │ • M conv      │ • M conv      │
├───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ All branches  │ Same key      │ Different     │ Different     │ Completely    │
│ get identical │ space, but    │ gates, same   │ value         │ independent   │
│ contribution  │ branches      │ value         │ projections   │ per branch    │
│               │ vote together │ direction     │ per branch    │               │
├───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Params: ~P    │ Params: ~P+Md²│ Params: ~P+Md²│ Params:       │ Params:       │
│               │               │               │ ~P+2Md²       │ ~M×P          │
├───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Expressivity: │ Expressivity: │ Expressivity: │ Expressivity: │ Expressivity: │
│ 1 scalar      │ 1 scalar      │ M scalars     │ M vectors     │ M independent │
│ for all       │ (joint)       │ (independent) │ (M rays)      │ vectors       │
└───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘

```

### The Tradeoff Relationship
The relationship is partially inverse, but with nuance:
**The inverse relationship:**

```
                    Engram Contribution
                           │
      Pure Static          │           Pure Dynamic
      (no gating)          │           (full attention)
            │              │                  │
            ▼              │                  ▼
     ┌──────────────────┐  │  ┌──────────────────────────┐
     │ Fixed embedding  │  │  │ Query-dependent          │
     │ Same for all     │◄─┼─►│ selection over all       │
     │ contexts         │  │  │ memory positions         │
     └──────────────────┘  │  └──────────────────────────┘
            │              │                  │
       O(1) cost           │             O(N·d) cost
       No expressivity     │             Full expressivity
                           │
                    ┌──────┴──────┐
                    │   Gating    │
                    │  (O(d))     │
                    │             │
                    │ Intermediate│
                    │ cost &      │
                    │ expressivity│
                    └─────────────┘

```

**Orthogonal scaling axes:**
The paper’s design achieves **orthogonal scaling** :
  * mHC scales expressivity via **branch parallelism** (more information pathways)
  * Engram scales capacity via **memory size** (more stored patterns)


These are _somewhat_ independent axes:

```
                        Memory Size (Engram)
                              │
                    Low       │        High
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Low  │   Small,      │   Large       │
              │   simple      │   memory,     │
   Branches   │   model       │   simple      │
   (mHC)      │               │   routing     │
              │───────────────┼───────────────│
              │               │               │
         High │   Small       │   PAPER'S     │
              │   memory,     │   CHOICE      │
              │   complex     │   (27B total) │
              │   routing     │               │
              └───────────────┴───────────────┘

```

### Available Tuning Knobs
  1. **Number of branches** : More branches = more expressive gating, more parameters
  2. **What’s shared vs. separate** : Could separate for direction control per branch
  3. **Gating granularity** : Vector gating instead of scalar for per-dimension control
  4. **Hash heads per branch** : Different heads for different branches


## 6. Memory-Compute Decoupling: Training vs. Inference Strategies
### Training: Distributed HBM Sharding
The paper describes training system design:
> “During training, to accommodate large-scale embedding tables, we employ standard model parallelism by sharding the tables across available GPUs. An All-to-All communication primitive is used to gather active rows in the forward pass and dispatch gradients in the backward pass.”
This is **not** offloading to host memory. During training:
  * The embedding tables are split across multiple GPUs
  * Each GPU holds of the table in its HBM
  * All2All communication gathers the needed embeddings
  * Gradients flow back via the same All2All primitive



```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING: DISTRIBUTED SHARDING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    GPU 0 (HBM)         GPU 1 (HBM)         GPU 2 (HBM)         GPU 3 (HBM) │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐       ┌───────────┐│
│   │ E[0:M/4]  │       │E[M/4:M/2] │       │E[M/2:3M/4]│       │E[3M/4:M]  ││
│   │           │       │           │       │           │       │           ││
│   │ Shard 0   │       │ Shard 1   │       │ Shard 2   │       │ Shard 3   ││
│   └─────┬─────┘       └─────┬─────┘       └─────┬─────┘       └─────┬─────┘│
│         │                   │                   │                   │      │
│         └───────────────────┴───────────────────┴───────────────────┘      │
│                                     │                                       │
│                              All-to-All                                     │
│                           Communication                                     │
│                                     │                                       │
│                                     ▼                                       │
│                    Each GPU receives embeddings it needs                    │
│                    for its local batch of tokens                           │
│                                                                             │
│   MEMORY REQUIREMENT: Total Engram params / Number of GPUs per GPU         │
│   STILL IN HBM: Yes, distributed but still on-device                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

**Training still requires substantial HBM** —it’s just distributed.
### Inference: Host Memory Offloading with Prefetching
The dramatic memory savings apply to inference:
> “During inference, this deterministic nature enables a prefetch-and-overlap strategy. Since memory indices are known prior to the forward pass, the system can asynchronously retrieve embeddings from abundant host memory via PCIe.”

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE: HOST MEMORY OFFLOADING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          HOST MEMORY (DRAM)                                 │
│                    ┌─────────────────────────────┐                          │
│                    │                             │                          │
│                    │   ENGRAM EMBEDDING TABLE    │                          │
│                    │        (100B params)        │                          │
│                    │                             │                          │
│                    │   ~200GB at FP16            │                          │
│                    │   (abundant, cheap)         │                          │
│                    │                             │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                              PCIe Transfer                                  │
│                           (async, prefetched)                               │
│                                   │                                         │
│                                   ▼                                         │
│                    ┌─────────────────────────────┐                          │
│                    │         GPU (HBM)           │                          │
│                    │                             │                          │
│                    │  ┌───────────────────────┐  │                          │
│                    │  │ Transformer Backbone  │  │                          │
│                    │  │ (4B-8B params)        │  │                          │
│                    │  │ ~8-16GB               │  │                          │
│                    │  └───────────────────────┘  │                          │
│                    │                             │                          │
│                    │  ┌───────────────────────┐  │                          │
│                    │  │ Prefetch Buffer       │  │  ← Small buffer for     │
│                    │  │ (active embeddings)   │  │    current batch        │
│                    │  │ ~few MB               │  │                          │
│                    │  └───────────────────────┘  │                          │
│                    │                             │                          │
│                    │  Total HBM: ~10-20GB        │                          │
│                    │  (NOT 100GB+)               │                          │
│                    │                             │                          │
│                    └─────────────────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Why Prefetching Works for Inference But Not Training
The critical enabler is **deterministic addressing** :
**Inference** : Indices Known in Advance

```
# At inference time, the input sequence is fixed
input_tokens = [1847, 29341, 8472, 15234, 31847]

# We can compute ALL Engram indices BEFORE the forward pass
ngram_indices_layer_2 = compute_hashes(input_tokens, layer=2)
ngram_indices_layer_15 = compute_hashes(input_tokens, layer=15)

# Start prefetching while Layer 1 computes
async_prefetch(ngram_indices_layer_2)  # PCIe transfer begins

# By the time we reach Layer 2, embeddings are already in GPU buffer
layer_1_output = transformer_layer_1(input)
engram_embeddings = await_prefetch()  # Already arrived!
layer_2_output = engram_layer_2(layer_1_output, engram_embeddings)

```

**Training** : Gradient-Dependent Updates

```
# During training, we need gradients for EVERY accessed embedding
forward:
    e_t = E[hash(ngram)]  # Retrieved from host? 
    loss = compute_loss(model(e_t))

backward:
    grad_e_t = d_loss / d_e_t  # Need to send gradient BACK to host
    E[hash(ngram)] -= lr * grad_e_t  # Update in host memory
    
# Problem 1: PCIe bandwidth for gradients back to host
# Problem 2: Optimizer states (Adam momentum, variance) where do they live?
# Problem 3: Gradient accumulation across distributed batches

```

Training requires bidirectional, latency-sensitive communication that doesn’t tolerate PCIe bottlenecks as gracefully.
### Memory Accounting Examples
**Traditional 100B model** :
100B params×2 bytes (FP16)=200GB HBM
Requires 3× H100 80GB (tensor parallel) or 2× H200 141GB.
**Engram 100B (8B backbone + 92B Engram)** :  
| Component  | Location  | Size  |  
| --- | --- | --- |  
| Transformer Backbone  | GPU HBM  | ~16GB  |  
| Engram Tables  | Host DRAM  | ~184GB  |  
| **Total GPU HBM**  | **~16GB**  |  
Can run on 1× RTX 4090 24GB + 256GB host RAM.
**Important caveat** : Not equivalent models! The 8B backbone limits reasoning depth. But for knowledge retrieval tasks, may be competitive.
### Key Insight: Breaking the 1GB/1B Rule
The traditional rule of thumb (1GB HBM per 1B parameters) doesn’t apply to Engram inference because:
  1. Engram parameters can reside in host DRAM
  2. Prefetch-overlap masks PCIe latency
  3. Zipfian caching further reduces effective latency


This enables running larger parameter models on memory-constrained hardware, but the Engram parameters provide “memory capacity” not “reasoning depth.”
## 7. The U-Shaped Allocation Curve: Empirical Findings and Open Questions
### What The Paper Says
> “This observed U-shape confirms the structural complementarity between the two modules:
>   * **MoE-dominated ( ρ→100%)**: The model lacks dedicated memory for static patterns, forcing it to inefficiently reconstruct them through depth and computation.
>   * **Engram-dominated ()** : The model loses conditional computation capacity, hurting tasks that require dynamic, context-dependent reasoning; memory cannot replace computation in this regime.”
> 

### What The Paper Doesn’t Explain
  1. **A formal model** predicting the optimal allocation ratio
  2. **An explanation** for why the optimum is at ~75-80% specifically (rather than 50% or 90%)
  3. **Analysis of the curve shape** —why U-shaped rather than V-shaped, linear, or asymmetric?
  4. **Scale dependence** —does the optimal shift as total parameters increase?
  5. **Task dependence** —is different for knowledge-intensive vs. reasoning tasks?


### Plausible Hypotheses
#### Hypothesis 1: Diminishing Returns on Each Axis
Both MoE experts and Engram slots likely have diminishing returns following power laws:
LMoE(n)=A⋅ LEngram(m)=B⋅
Under fixed parameter budget Psparse:
n⋅pexpert+m⋅pslot=Psparse
The U-shape emerges if both functions are convex and the coefficients create an interior optimum.
#### Hypothesis 2: Functional Specialization with Coverage Requirements
If ~30-40% of tokens are predictable from local patterns and ~60-70% require reasoning, optimal allocation should roughly match this split. The paper’s finding (≈0.75-0.80) is slightly higher, suggesting MoE is somewhat less efficient at its task than Engram at its task.
#### Hypothesis 3: Hash Collision Saturation
Beyond some point, additional Engram capacity hits diminishing returns due to training signal sparsity for rare N-grams, while MoE experts can generalize across inputs.
### Assessment
The U-shaped curve is an **empirical finding** , not a **derived scaling law**. The paper provides no predictive equations for optimal allocation in new configurations. Practitioners can use ρ≈0.75-0.80 as a starting point but have no principled way to adjust for specific use cases without running their own sweeps.
## 8. Scaling Experiments: Zero-Sum Allocation vs. Additive Scaling
### Experiment 1: Zero-Sum Allocation (Section 3.1)
**Setup** : Fixed total parameters , fixed activated parameters . Sweep allocation ratio .
**Finding** : U-shaped loss curve with optimal ρ≈75-80%. Reallocating ~20-25% of sparse budget from MoE to Engram improves performance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 1: ZERO-SUM ALLOCATION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Total Sparse Budget: 10B parameters (fixed)                              │
│                                                                             │
│   ρ = 100%:  MoE gets 10B  │  Engram gets 0B     → Loss: 1.7248           │
│   ρ = 80%:   MoE gets 8B   │  Engram gets 2B    → Loss: 1.7109 (optimal) │
│   ρ = 60%:   MoE gets 6B   │  Engram gets 4B    → Loss: ~1.715            │
│   ρ = 40%:   MoE gets 4B   │  Engram gets 6B    → Loss: ~1.725            │
│                                                                             │
│   KEY INSIGHT: Trading MoE capacity for Engram capacity improves loss      │
│                up to a point, then hurts.                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

**Interpretation** : Under fixed constraints, some Engram is better than pure MoE. This is genuine reallocation—MoE capacity decreases as Engram increases.
### Experiment 2: Additive Scaling (Section 3.2)
**Setup** : Fixed MoE backbone (Ptot≈3B, Pact=568M). Sweep Engram capacity from 0.3B to 13B.
**Finding** : Log-linear scaling—loss decreases linearly with log⁡(slots). No saturation observed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 2: ADDITIVE SCALING                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Fixed MoE Backbone: 3B parameters (UNCHANGED across all runs)            │
│                                                                             │
│   Config A:  MoE: 3B  +  Engram: 0.3B   = Total: 3.3B   → Loss: ~1.81     │
│   Config B:  MoE: 3B  +  Engram: 1B     = Total: 4B     → Loss: ~1.78     │
│   Config C:  MoE: 3B  +  Engram: 5B     = Total: 8B     → Loss: ~1.76     │
│   Config D:  MoE: 3B  +  Engram: 13B    = Total: 16B    → Loss: ~1.74     │
│                                                                             │
│   The PROPORTION of Engram increases (9% → 81%), but this is because      │
│   the denominator grows, NOT because MoE shrinks.                          │
│                                                                             │
│   NOTHING IS OFFLOADED—MoE capacity is preserved.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

**Interpretation** : With unconstrained memory, more Engram is always better (within tested range). Nothing is “offloaded”—MoE is unchanged, Engram is added.
### Critical Distinction  
| Aspect  | Experiment 1  | Experiment 2  |  
| --- | --- | --- |  
| MoE params  | Decreases as   | Constant  |  
| Engram params  | Increases as   | Increases  |  
| Total params  | Constant  | Increases  |  
| Nature  | Reallocation  | Addition  |  
### The OverEncoding Comparison
OverEncoding (Huang et al., 2025a) also uses hash-based N-gram embeddings but:
  * Injects at layer 0 only (no prefetch overlap)
  * Uses fixed averaging (no gating)
  * No tokenizer compression



```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OVERENCODING vs ENGRAM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OVERENCODING:                                                             │
│   ─────────────                                                             │
│   • N-gram embeddings retrieved at INPUT LAYER (Layer 0)                   │
│   • Integration: AVERAGING with vocabulary embedding                        │
│   • No gating mechanism                                                     │
│   • No context-awareness                                                    │
│                                                                             │
│       input_embedding = 0.5 * vocab_embed[token] + 0.5 * ngram_embed[hash] │
│                                                                             │
│   ENGRAM:                                                                   │
│   ───────                                                                   │
│   • N-gram embeddings retrieved at INTERMEDIATE LAYERS (e.g., 2, 15)       │
│   • Integration: GATED RESIDUAL addition                                    │
│   • Context-aware gating (can suppress irrelevant retrievals)              │
│   • Tokenizer compression                                                   │
│                                                                             │
│       hidden = hidden + gate(context, ngram_embed) * project(ngram_embed)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

Engram extracts more value from equivalent memory budget due to deeper injection, gating, and compression. The comparison demonstrates that _how_ you integrate N-gram memory matters as much as _whether_ you include it.
## 9. Strategic Publication Patterns and V4 Deployment Probability
### The Undertraining Signal
The paper explicitly acknowledges:
> “Finally, scaling to Engram-40B further reduces pre-training loss and improves performance across most benchmarks. Although it does not yet strictly dominate Engram-27B on every task, **this is likely an artifact of under-training**. We observe that **the training loss gap between Engram-40B and the baselines continues to widen towards the end of training** , suggesting that the expanded memory capacity has not yet fully saturated within the current token budget.”
### Evidence of Undertraining
  1. **Loss gap widening** : Engram-40B advantage over baselines increases toward training end
  2. **Inconsistent benchmark dominance** : Engram-40B regresses on code tasks (HumanEval 38.4 vs Engram-27B 40.8)


### Historical Pattern: DeepSeek Publication → Deployment  
| Innovation  | Paper Date  | Deployed In  | Lag  |  
| --- | --- | --- | --- |  
| DeepSeekMoE  | Jan 2024  | V2 (May 2024)  | ~4 months  |  
| MLA  | May 2024  | V2 (May 2024)  | <1 month  |  
| Aux-loss-free  | Nov 2024  | V3 (Dec 2024)  | ~1 month  |  
| mHC  | Dec 2024  | V3 (Dec 2024)  | <1 month  |  
| **Engram**  | **Jan 2026**  | **V4 (???)**  | **???**  |  
**Pattern** : Every major DeepSeek paper has been deployed. No exception.
### The Strategic Publication Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RESEARCH PAPER AS STRATEGIC SIGNAL                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: INTERNAL R&D (Not Published)                                    │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Full-scale experiments at frontier compute                             │
│   • Production infrastructure development                                  │
│   • Integration with existing model architecture (V3)                      │
│   • Iterative refinement based on internal benchmarks                      │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│   PHASE 2: ACADEMIC PUBLICATION (This Paper)                               │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Establish intellectual priority                                        │
│   • Validate core concepts at reduced scale                                │
│   • Describe (but don't fully benchmark) production design                 │
│   • Signal direction to recruit talent and shape field                     │
│   • Deliberately omit frontier-scale results                               │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│   PHASE 3: PRODUCT LAUNCH (V4 Announcement)                                │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Reveal frontier-scale performance                                      │
│   • Cite own paper as foundation                                          │
│   • Competitors now 6-12 months behind                                     │
│   • Academic paper provides legitimacy                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Probability Assessment  
| Scenario  | Probability  | Description  |  
| --- | --- | --- |  
| Full adoption  | ~35%  | Engram as described, scaled to 100B+  |  
| Modified adoption  | ~40%  | Engram-like module with undisclosed modifications  |  
| Partial/optional  | ~15%  | Engram in specific variants only  |  
| Deferred/abandoned  | ~10%  | Internal issues prevent deployment  |  
**Combined probability of some adoption** : ~75%
**Expected timeline** : Based on historical patterns (papers 1-4 months before deployment), V4 announcement likely in **Q1-Q2 2026**.
## 10. Long-Context Enhancement: The Attention Capacity Mechanism
### The Experimental Design
The paper compares models with matched base quality (iso-loss) to isolate architectural effects from general capability differences. This methodology is itself a contribution—many long-context papers conflate these factors.
### Key Results: Iso-Loss Comparison
Both models at pre-training loss = 1.63:  
| Metric  | MoE-27B (50k)  | Engram-27B (46k)  |  
| --- | --- | --- |  
| LongPPL - Book  | 4.38  | 4.19  | -0.19  |  
| LongPPL - Paper  | 2.91  | 2.84  | -0.07  |  
| LongPPL - Code  | 2.49  | 2.45  | -0.04  |  
| NIAH - Multi-Query  | 84.2  | 97.0  | **+12.8**  |  
| Variable Tracking  | 77.0  | 87.2  | **+10.2**  |  
| Frequent Words Extraction  | 73.0  | 98.6  | **+25.6**  |  
### The Mechanism: Attention Capacity Hypothesis
Standard Transformers use attention for both local patterns and long-range dependencies. At long contexts, these compete:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION CAPACITY HYPOTHESIS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STANDARD TRANSFORMER:                                                     │
│   ─────────────────────                                                    │
│                                                                             │
│       ┌─────────────────────────────────────────────┐                      │
│       │            ATTENTION BUDGET                  │                      │
│       │  ┌─────────────────┬───────────────────┐    │                      │
│       │  │  Local Patterns │  Global Patterns  │    │                      │
│       │  │  "the cat sat"  │  "earlier, John"  │    │                      │
│       │  │                 │  "mentioned that" │    │                      │
│       │  │     ~60%        │       ~40%        │    │                      │
│       │  └─────────────────┴───────────────────┘    │                      │
│       └─────────────────────────────────────────────┘                      │
│                                                                             │
│       At long contexts, local patterns "crowd out" global attention.       │
│                                                                             │
│                                                                             │
│   ENGRAM TRANSFORMER:                                                       │
│   ───────────────────                                                      │
│                                                                             │
│       ┌─────────────────────────────────────────────┐                      │
│       │            ATTENTION BUDGET                  │                      │
│       │  ┌─────────────────────────────────────┐    │                      │
│       │  │        Global Patterns Only         │    │                      │
│       │  │  "earlier, John mentioned that..."  │    │                      │
│       │  │                                     │    │                      │
│       │  │             ~90%                    │    │                      │
│       │  └─────────────────────────────────────┘    │                      │
│       └─────────────────────────────────────────────┘                      │
│                                                                             │
│       ┌─────────────────────────────────────────────┐                      │
│       │          ENGRAM (separate budget)           │                      │
│       │  ┌─────────────────────────────────────┐    │                      │
│       │  │        Local Patterns Only          │    │                      │
│       │  │       "the cat sat", "New York"     │    │                      │
│       │  │            O(1) lookup              │    │                      │
│       │  └─────────────────────────────────────┘    │                      │
│       └─────────────────────────────────────────────┘                      │
│                                                                             │
│       By handling local patterns separately, attention can focus           │
│       entirely on long-range dependencies.                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Task-Specific Pattern  
| Task Type  | Engram Advantage  | Explanation  |  
| --- | --- | --- |  
| LongPPL (average perplexity)  | Modest (+2-5%)  | Most tokens are locally predictable  |  
| Single-needle retrieval  | None (~0%)  | Easy for both architectures  |  
| Multi-query retrieval  | Large (+15%)  | Attention-limited in MoE  |  
| Variable tracking  | Large (+16%)  | Requires multi-hop retrieval  |  
| Frequent word extraction  | Huge (+36%)  | Requires global scanning  |  
The pattern confirms the mechanism: Engram’s advantage scales with how much the task requires global (vs. local) attention.
## 11. The Effective Depth Hypothesis: Bypassing Pattern Reconstruction
### The Claim
> “By equipping the model with an explicit knowledge lookup capability, Engram effectively mimics an increase in model depth by relieving the model of the early stages of feature composition.”
### What Early Layers Do
Standard Transformers spend layers 1-6 progressively reconstructing static patterns. The paper cites Ghandeharioun et al. (2024) to illustrate with “Diana, Princess of Wales”:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         ENTITY RESOLUTION: "Diana, Princess of Wales"                       │
│         (How a standard Transformer reconstructs a known entity)            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: "... Diana, Princess of Wales ..."                                │
│   Task: Internally represent who this entity is                            │
│                                                                             │
│   Layer    Latent State (via PatchScope)           What's Happening        │
│   ─────────────────────────────────────────────────────────────────────────│
│   1-2      "Wales: Country in the United Kingdom"  Just the last token     │
│   3        "Wales: Country in Europe"              Still just geography    │
│   4        "Princess of Wales: Title held by       Starting to see title   │
│             female sovereigns..."                                          │
│   5        "Princess of Wales: Title given to      Title semantics emerge  │
│             the wife of the Prince of Wales..."                            │
│   6        "Diana, Princess of Wales (1961-1997),  FINALLY: Full entity   │
│             the first wife of Prince Charles..."                           │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   OBSERVATION:                                                              │
│   It takes 6 LAYERS just to reconstruct a well-known entity.               │
│   This is static knowledge—it's the same every time this phrase appears.  │
│   These layers are essentially rebuilding a lookup table at runtime.       │
│                                                                             │
│   ENGRAM ALTERNATIVE:                                                       │
│   Layer 2: Engram looks up trigram "Princess of Wales"                     │
│            → Retrieves pre-computed embedding encoding the full entity     │
│   Layer 3+: Can immediately proceed to REASONING about Diana              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### LogitLens Evidence
LogitLens (nostalgebraist, 2020) projects each layer’s hidden state through the final LM head to measure “prediction readiness”:
KL(P‖Pfinal)
**Finding** : Engram shows systematically lower KL divergence at early layers—representations converge to prediction-ready states faster.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KL DIVERGENCE BY LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KL Divergence                                                            │
│       │                                                                     │
│   10  │ ●                                                                  │
│       │  ●                                                                  │
│    8  │   ●●                                                               │
│       │     ●●    MoE-27B                                                  │
│    6  │       ●●●                                                          │
│       │   ○      ●●●                                                       │
│    4  │    ○○        ●●●●                                                  │
│       │      ○○○          ●●●●●●                                           │
│    2  │         ○○○○○           ●●●●●●●●●●                                │
│       │              ○○○○○○○○○○○○○○○○○○○○○                                │
│    0  │──────────────────────────────────────────→ Layer                   │
│       0    5    10   15   20   25   30                                     │
│                                                                             │
│       ● MoE-27B        ○ Engram-27B                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### CKA Evidence
CKA (Centered Kernel Alignment; Kornblith et al., 2019) measures representational similarity between layers.
**Finding** : Engram layer 5 representations match MoE layer ~12 representations for named entities. The “soft alignment index” quantifies this shift:  
| Engram Layer   | Soft Alignment   | “Depth Bonus” ()  |  
| --- | --- | --- |  
| 0  | ~2  | +2 layers  |  
| 5  | ~12  | +7 layers  |  
| 10  | ~17  | +7 layers  |  
| 15  | ~21  | +6 layers  |  
| 20  | ~24  | +4 layers  |  
| 25  | ~27  | +2 layers  |  
Early-to-mid Engram layers gain the most “effective depth” because that’s where static pattern reconstruction would normally occur.
### The Mechanism Summarized

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENGRAM'S "EFFECTIVE DEPTH" MECHANISM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STANDARD TRANSFORMER:                                                     │
│                                                                             │
│   Layer 0  ──→ Raw token embeddings                                        │
│   Layer 1  ──→ Local bigram features                                       │
│   Layer 2  ──→ Trigram patterns emerging                                   │
│   Layer 3  ──→ Entity boundaries detected                                  │
│   Layer 4  ──→ Entity types recognized                                     │
│   Layer 5  ──→ Multi-token entities composed        ← RECONSTRUCTION      │
│   Layer 6  ──→ Entity semantics resolved             ← COMPLETE HERE      │
│   Layer 7  ──→ Begin relational reasoning           ← REASONING STARTS    │
│   ...                                                                       │
│   Layer 30 ──→ Final prediction                                            │
│                                                                             │
│   ENGRAM TRANSFORMER:                                                       │
│                                                                             │
│   Layer 0  ──→ Raw token embeddings                                        │
│   Layer 1  ──→ Basic contextualization                                     │
│   Layer 2  ──→ ENGRAM INJECTION                     ← STATIC PATTERNS     │
│               │                                       RETRIEVED O(1)       │
│               └→ Representations NOW equivalent to MoE Layer 6-7          │
│   Layer 3  ──→ Can immediately begin reasoning      ← REASONING STARTS    │
│   ...                                                                       │
│   Layer 30 ──→ Final prediction                                            │
│                                                                             │
│   NET EFFECT:                                                               │
│   Engram "skips" ~4-5 layers of reconstruction work.                       │
│   Those layers can now contribute to reasoning instead.                    │
│   A 30-layer Engram model has ~34-35 "effective layers" of reasoning.     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

## 12. Functional Separation and Domain-Specialized Engram Potential
### The Sensitivity Experiment
The experiment completely suppresses Engram output during inference ( for all positions).
### Results by Task Category

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RETAINED PERFORMANCE BY TASK TYPE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TASK CATEGORY              BENCHMARK        RETAINED     INTERPRETATION  │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   READING COMPREHENSION      C3               93%          Backbone holds  │
│   (Context provides answer)  RACE-Middle      89%          comprehension   │
│                              RACE-High        84%          capability      │
│                              DROP             81%                          │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   COMMONSENSE REASONING      HellaSwag        85%          Backbone holds  │
│   (World knowledge needed)   ARC-Challenge    81%          most common-    │
│                              PIQA             81%          sense patterns  │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   KNOWLEDGE-INTENSIVE        CMMLU            78%          Moderate        │
│   REASONING                  MMLU             75%          degradation—    │
│   (Facts + reasoning)        MMLU-PRO         72%          needs both      │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   CODE                       CruxEval         76%          Mixed—patterns  │
│   (Patterns + logic)         MBPP             68%          matter for      │
│                              HumanEval        58%          generation      │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   ALGORITHMIC REASONING      BBH              67%          Surprising      │
│   (Multi-step logic)         GSM8K            62%          dependency—     │
│                              MGSM             44%          maybe via       │
│                              MATH             36%          pattern recog   │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   FACTUAL KNOWLEDGE          TriviaQA-ZH      44%          CATASTROPHIC    │
│   (Pure recall)              PopQA            44%          COLLAPSE        │
│                              TriviaQA         29%          ← Engram IS     │
│                                                              the knowledge │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Interpretation: Functional Separation
Engram becomes the **primary repository for parametric knowledge**. Factual knowledge is inherently N-gram structured (“The capital of France” → “Paris”), making it ideal for static lookup. The backbone retains comprehension and reasoning capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURAL CAPABILITY MAPPING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────────────┐                    │
│                         │      FULL MODEL             │                    │
│                         │   (Backbone + Engram)       │                    │
│                         └─────────────┬───────────────┘                    │
│                                       │                                     │
│               ┌───────────────────────┼───────────────────────┐            │
│               │                       │                       │            │
│               ▼                       │                       ▼            │
│   ┌───────────────────────┐          │          ┌───────────────────────┐ │
│   │      BACKBONE         │          │          │       ENGRAM          │ │
│   │   (MoE Experts +      │          │          │   (N-gram Memory)     │ │
│   │    Attention)         │          │          │                       │ │
│   ├───────────────────────┤          │          ├───────────────────────┤ │
│   │ • Reading comprehens. │          │          │ • Factual knowledge   │ │
│   │ • Commonsense reason. │          │          │ • Entity recognition  │ │
│   │ • Context extraction  │          │          │ • Pattern completion  │ │
│   │ • Logical inference   │          │          │ • Stereotyped phrases │ │
│   │ • Long-range deps     │          │          │ • Local collocations  │ │
│   └───────────────────────┘          │          └───────────────────────┘ │
│               │                       │                       │            │
│               │          ┌────────────┴────────────┐          │            │
│               │          │   REQUIRES BOTH         │          │            │
│               └─────────►│ • Knowledge reasoning   │◄─────────┘            │
│                          │ • Code generation       │                       │
│                          │ • Math problem solving  │                       │
│                          └─────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Implication: Domain-Specialized Engram
If Engram disproportionately stores factual knowledge, domain-specialized Engram modules could dramatically improve accuracy in knowledge-intensive applications:
#### Medical Engram Potential

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEDICAL ENGRAM: POTENTIAL DESIGN                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DOMAIN CHARACTERISTICS:                                                   │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   Medical text is HIGHLY N-GRAM STRUCTURED:                                │
│                                                                             │
│   • Drug names: "acetaminophen", "metformin hydrochloride"                 │
│   • Dosages: "500mg twice daily", "10mg/kg body weight"                    │
│   • Conditions: "type 2 diabetes mellitus", "acute myocardial infarction" │
│   • Anatomical terms: "left anterior descending artery"                    │
│   • Procedures: "laparoscopic cholecystectomy"                             │
│   • Interactions: "contraindicated with MAO inhibitors"                    │
│                                                                             │
│   These are LOCAL, STATIC patterns—perfect for Engram.                     │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   POTENTIAL ARCHITECTURE:                                                   │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   Base Model:                                                               │
│   ├─ General backbone (reasoning, comprehension)        ~20B params        │
│   └─ General Engram (common knowledge)                  ~10B params        │
│                                                                             │
│   Medical Specialist:                                                       │
│   ├─ Same backbone                                      ~20B params        │
│   ├─ General Engram (retained)                          ~10B params        │
│   └─ Medical Engram (domain-specific)                   ~50B params        │
│       • Trained on PubMed, clinical notes, FDA data                        │
│       • Drug-drug interactions                                             │
│       • Diagnostic criteria                                                │
│       • Treatment protocols                                                │
│                                                                             │
│   Total: 80B params, but only 20B activated per token                     │
│   Massive factual capacity with modest compute cost.                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

#### Advantages of Domain-Specialized Engram
  1. **Efficiency** : Scales knowledge without scaling inference cost
  2. **Modularity** : Swap domain modules without backbone retraining
  3. **Updateability** : Incremental updates without catastrophic forgetting
  4. **Auditability** : Deterministic retrievals enable knowledge provenance


#### Future Direction: Mixture of Memories (MoM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           FUTURE DIRECTION: MIXTURE OF MEMORIES (MoM)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Just as MoE routes computation to specialized experts,                   │
│   MoM could route retrieval to specialized memory modules:                 │
│                                                                             │
│                        Input Tokens                                         │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │  Memory Router  │                                      │
│                    │  (Learned)      │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│           ┌─────────────────┼─────────────────┐                            │
│           │                 │                 │                             │
│           ▼                 ▼                 ▼                             │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                    │
│   │   General     │ │   Medical     │ │    Legal      │                    │
│   │   Engram      │ │   Engram      │ │   Engram      │                    │
│   │   (50B)       │ │   (50B)       │ │   (30B)       │                    │
│   └───────┬───────┘ └───────┬───────┘ └───────┬───────┘                    │
│           │                 │                 │                             │
│           └─────────────────┼─────────────────┘                            │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │   Gated Fusion  │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │    Backbone     │                                      │
│                    │    (20B MoE)    │                                      │
│                    └─────────────────┘                                      │
│                                                                             │
│   Total: 150B+ parameters                                                  │
│   Activated: ~25B per token                                                │
│   Specialized knowledge for each domain                                    │
│   Single backbone for shared reasoning                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

## 13. Scale Claims vs. Empirical Validation: Strategic Omissions
### Systematic Gap Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALE CLAIMS vs. EMPIRICAL VALIDATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CLAIM                                    VALIDATED?    GAP                │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│   U-shaped allocation is stable            Partially     Only 2 compute    │
│   across compute regimes                                 budgets tested    │
│                                                          (2×10²⁰, 6×10²⁰) │
│                                                          V3 uses ~10²⁵    │
│                                                                             │
│   Log-linear Engram scaling continues      Partially     Tested to ~13B   │
│   indefinitely                                           Claims apply to  │
│                                                          100B+            │
│                                                                             │
│   100B Engram offloading with <3%          Throughput    Capability NOT   │
│   overhead                                 only          tested at 100B   │
│                                                                             │
│   Multi-level cache hierarchy              Described     No empirical     │
│   exploits Zipfian distribution                          validation       │
│                                                                             │
│   Engram-40B undertraining implies         Claimed       No extended      │
│   further gains possible                                 training run     │
│                                                                             │
│   Prefetch-overlap strategy scales         Architecture  Not tested at    │
│   to production serving                    described     high QPS/batch   │
│                                                                             │
│   Context-extension gains persist          32k context   No 128k or 1M    │
│   at longer contexts                       only          context testing  │
│                                                                             │
│   MoE + Engram composition optimal         Asserted      No comparison    │
│   for frontier models                                    at frontier      │
│                                                          scale            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Evidence of Intentional Omission
  1. **Selective precision** : Extremely detailed on some aspects (hyperparameters to 4 decimals), conspicuously vague on scale
  2. **Infrastructure over-specification** : Section 2.5 describes production-grade systems unnecessary for 27B validation
  3. **Explicit undertraining acknowledgment** : They tell you results are incomplete
  4. **Architecture alignment with V3** : Uses V3 tokenizer, MLA, mHC—not generic research


### The Competitive Timing Dimension

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPETITIVE TIMING ANALYSIS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IF DEEPSEEK PUBLISHES FULL FRONTIER RESULTS:                             │
│   ─────────────────────────────────────────────────────────────────────────│
│   • OpenAI/Anthropic/Google immediately start replication                  │
│   • US labs have more compute to iterate faster                            │
│   • DeepSeek's head start erodes quickly                                   │
│                                                                             │
│   BY PUBLISHING CONCEPT WITHOUT SCALE VALIDATION:                          │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Establishes intellectual priority (can cite own work)                  │
│   • Competitors must independently validate scale                          │
│   • Buys time to ship V4 before replication                                │
│   • When V4 launches, competitors are still experimenting                  │
│                                                                             │
│   THE STRATEGIC CALCULUS:                                                  │
│   ─────────────────────────────────────────────────────────────────────────│
│   • DeepSeek's advantage is NOT compute (they have less)                  │
│   • DeepSeek's advantage IS architectural innovation speed                 │
│   • Publishing establishes priority, withholding preserves lead time      │
│   • Optimal strategy: publish concept, withhold scale results             │
│                                                                             │
│   This is EXACTLY what the paper does.                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

# Future Implications
## The Beijing AGI Summit Context
At the January 2026 Beijing AGI summit, senior figures from Chinese AI labs made striking admissions (South China Morning Post, 2026; Wall Street Journal, 2026):
  * **Zhipu AI co-founder Tang Jie** : “The truth may be that the gap is actually widening”
  * **Alibaba scientist Lin Junyang** : “Less than 20%” chance of overtaking US in 3-5 years; US labs enjoy “one to two orders of magnitude more training compute”


External assessments support this pessimism (Asialink, 2026; Science Business, 2026):
  * US controls ~75% of global AI compute; China ~15%
  * US AI supercomputing capacity approximately 9× China’s
  * HBM (not just GPUs) identified as binding constraint (Design-Reuse, 2026)


## Engram as Strategic Response
The Engram architecture directly addresses acknowledged constraints:  
| Constraint  | Engram Response  |  
| --- | --- |  
| HBM bottleneck  | Host DRAM offloading via prefetch  |  
| Compute disadvantage  | More capability per FLOP  |  
| Hardware access limits  | Architectural efficiency as substitute for scale  |  
This alignment is not coincidental. Engram represents the **operationalization of efficiency as competitive strategy**.
## The Paradox of Constraint-Driven Innovation
Export controls may paradoxically accelerate Chinese architectural innovation:
  * US labs with abundant compute have less pressure to innovate on efficiency
  * Chinese labs, facing structural constraints, are forced into architectural creativity
  * Innovations that extract more from less benefit everyone but are _discovered_ under constraint


If US labs have 10× compute but DeepSeek extracts 2× more per FLOP, the effective gap narrows to 5×. With compounding efficiency innovations, this represents the scenario where China narrows the capability gap despite hardware disadvantage.
## DeepSeek V4 Prediction
### Evidence for Engram Adoption
  1. **Track record** : Every major DeepSeek paper has been deployed
  2. **Strategic fit** : Directly addresses HBM constraints and compute efficiency
  3. **Architecture alignment** : Paper uses V3 components (tokenizer, MLA, mHC)
  4. **Publication timing** : Matches historical pattern (paper 1-4 months before deployment)
  5. **Scale validation gap** : Frontier results deliberately withheld


### Signals to Watch
**Strong adoption signals** :
  * Follow-up paper showing Engram at 100B+ scale
  * Production-grade distributed training code release
  * V4 technical report citing this paper in architecture section
  * DeepSeek communications mentioning “conditional memory”


**Weak/negative signals** :
  * Long delay between paper and any follow-up
  * Other DeepSeek papers explore alternative efficiency approaches
  * No mention of Engram in subsequent communications


## Research Directions
### Domain-Specialized Engram
Research questions:
  1. Does domain-specific Engram training improve domain accuracy?
  2. What’s the optimal for knowledge-intensive domains?
  3. Can domain and general Engram compose without interference?
  4. Can Engram retrievals provide citations for knowledge provenance?


### Technical Extensions
  * Integration with other sparse primitives (Mixture-of-Depths, early exit)
  * Learned addressing (beyond fixed hash functions)
  * Higher-order N-grams with larger memory budgets
  * Continual learning for Engram modules


### Broader Implications
If Engram’s thesis is correct—that Transformers “waste” depth on static pattern reconstruction—implications extend beyond efficiency:
  * **Interpretability** : Cleaner separation between knowledge storage and reasoning
  * **Editability** : Modify factual knowledge without affecting reasoning capabilities
  * **Verification** : Audit knowledge sources via retrieval provenance


## Future Potential: Activation-Conditional Memory Modulation
A promising direction for future investigation emerges from the intersection of Engram’s branch-specific gating architecture and recent advances in neural network steering via representation engineering (Turner et al., 2023; Zou et al., 2023). If semantic modes—such as factual retrieval versus abstract reasoning—are geometrically identifiable within activation space, then Engram’s existing architectural hooks may permit dynamic, activation-conditional modulation of memory contribution.
### The Existing Architectural Hook
The Engram design already creates the necessary structure for semantic-conditional behaviour:
=σ(ht⋅kt)
The gate is already a function of the branch-specific hidden state ht, which encodes semantic content from preceding layers. This means the model can _implicitly_ learn to modulate Engram contribution based on whatever semantic features emerge in that activation pattern. The sensitivity analysis results (TriviaQA retaining only 29% performance versus C3 retaining 93% when Engram is suppressed) demonstrate that the model does learn some version of functional separation between factual retrieval and comprehension modes.
The research direction proposed here would make this separation **explicit and controllable** rather than leaving it to emergent learning dynamics.
### Proposed Implementation Approaches
#### Approach 1: Learned Factuality Probe
Following the methodology established in probing classifier literature (Belinkov, 2022), a linear probe could be trained on intermediate activations to detect “factual retrieval mode”:
=σ(wfact⊤+b)
This factuality score ∈(0,1) would then globally modulate Engram gates:
=⋅(1+λ⋅)
Where controls sensitivity to the factuality signal. Higher (factual retrieval detected) increases Engram contribution; lower (reasoning mode detected) decreases it.
#### Approach 2: Steering Vector Modulation
Building on representation engineering methods (Zou et al., 2023), a “factuality direction” vfact could be identified in activation space via contrastive methods applied to factual versus reasoning task completions:
factuality\\_scoret=⋅vfact‖‖‖vfact‖
This directional projection would bias gates toward higher or lower Engram contribution based on the model’s inferred semantic mode at each position.
#### Approach 3: Branch Specialization via Auxiliary Training Signal
Rather than post-hoc steering, functional separation could be encouraged during training through auxiliary losses that reward branch specialization:
  * **Branches 1-2** : High Engram affinity (factual retrieval specialists)
  * **Branches 3-4** : Low Engram affinity (reasoning specialists)


This would create explicit functional roles within the mHC branch structure, analogous to how MoE experts specialize through load balancing losses.
#### Approach 4: Inference-Time Control Surface
For deployment flexibility, a user-controllable “factuality dial” could be exposed:
=σ(ht⋅kt+βuser)
Where βuser∈[−2,+2] allows runtime adjustment of the prior toward factual precision (high Engram, positive ) or flexible reasoning (low Engram, negative ). This would provide practitioners direct control over the knowledge-reasoning tradeoff without model retraining.
### Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           ACTIVATION-CONDITIONAL ENGRAM MODULATION              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Hidden State h_t                                              │
│        │                                                        │
│        ├──→ [Factuality Probe] ──→ f_t ∈ (0,1)                 │
│        │                              │                         │
│        │         ┌────────────────────┘                         │
│        │         │                                              │
│        │         ▼                                              │
│        │    ┌─────────┐                                         │
│        │    │  Scale  │                                         │
│        │    └────┬────┘                                         │
│        │         │                                              │
│        ▼         ▼                                              │
│   ┌─────────────────────────────────────────┐                  │
│   │  Branch 1: α₁' = α₁ · (1 + λ·f_t)      │  High factuality │
│   │  Branch 2: α₂' = α₂ · (1 + λ·f_t)      │  → More Engram   │
│   │  Branch 3: α₃' = α₃ · (1 - λ·f_t)      │  Low factuality  │
│   │  Branch 4: α₄' = α₄ · (1 - λ·f_t)      │  → Less Engram   │
│   └─────────────────────────────────────────┘                  │
│                                                                 │
│   Result: Model dynamically routes between factual retrieval   │
│   and reasoning modes based on detected activation state.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

```

### Toward Unified Conditional Allocation
This direction points toward a more general architectural principle: **conditional memory with semantic routing**. Just as MoE routes computation to specialized experts based on learned gating functions, retrieval could be routed to specialized memory modules based on semantic state:  
| Detected Mode  | Engram Contribution  | Expert Routing  |  
| --- | --- | --- |  
| Factual retrieval  | High  | Low diversity (retrieval experts)  |  
| Abstract reasoning  | Low  | High diversity (reasoning experts)  |  
| Creative generation  | Moderate  | Specific expert clusters  |  
This would create a unified framework where both computation and memory are conditionally allocated based on task demands inferred from activation patterns—extending the conditional computation paradigm to encompass conditional memory as a first-class design axis.
### Research Considerations
Several factors warrant careful investigation:
  1. **Implicit learning sufficiency** : The current gating mechanism may already capture semantic-conditional behaviour through end-to-end learning. Explicit steering could prove redundant or could interfere with learned patterns that are more nuanced than a linear probe can capture.
  2. **Probe fidelity** : Factuality detection is not binary—many tasks require both retrieval and reasoning in interleaved fashion. A linear probe may be insufficiently expressive to capture the full spectrum of semantic modes.
  3. **Training dynamics** : If Engram contribution becomes highly state-dependent, the model might learn inconsistent or unstable dependencies that complicate optimization and hurt convergence.
  4. **Interpretability prerequisites** : Identifying robust semantic directions requires substantial representation engineering work that the Engram paper does not undertake. Applying methods from Zou et al. (2023) and Turner et al. (2023) to Engram-augmented models specifically would be a necessary precursor.
  5. **Evaluation methodology** : Measuring success requires benchmarks that cleanly separate factual retrieval from reasoning—existing benchmarks often confound these capabilities.


### Intended Future Work
The author of this analysis intends to explore these directions through systematic investigation of:
  1. **Semantic geometry in Engram-augmented models** : Applying representation engineering methods to characterize how factual versus reasoning modes manifest geometrically in mHC branch activations.
  2. **Branch specialization dynamics** : Analysing whether branches naturally specialize for different functional roles during standard Engram training, or whether auxiliary losses are required to induce clean separation.
  3. **Controllable factuality-reasoning tradeoffs** : Developing and evaluating inference-time control mechanisms that allow practitioners to explicitly navigate the knowledge-computation tradeoff.
  4. **Interaction effects with MoE routing** : Investigating whether Engram contribution and expert selection exhibit correlated or complementary patterns, and whether joint optimization of both routing decisions yields benefits over independent optimization.


This research direction represents a natural extension of the Engram contribution—moving from static architectural choices about _where_ to inject memory toward dynamic, semantically-informed decisions about _when_ and _how much_ memory to contribute based on the model’s internal state.
# References
Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1), 207-219.
Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: The count-min sketch and its applications. _Journal of Algorithms_ , 55(1), 58-75.
DeepSeek-AI. (2024a). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. _arXiv preprint arXiv:2405.04434_.
DeepSeek-AI. (2024b). DeepSeek-V3 technical report. _arXiv preprint arXiv:2412.19437_.
DeepSeek-AI. (2026). Engram: Conditional memory via scalable lookup—A new axis of sparsity for large language models. _arXiv preprint arXiv:2601.07372_.
Design-Reuse. (2026). China’s AI chip ambitions limited by HBM memory supply, notes report. Retrieved from https://www.design-reuse.com/news/
Ghandeharioun, A., Kim, B., & others. (2024). PatchScope: A framework for inspecting language model representations at scale. _Proceedings of the 41st International Conference on Machine Learning_.
Huang, X., & others. (2025a). OverEncoding: Hash-based N-gram embeddings for vocabulary expansion. _arXiv preprint_.
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. _Contemporary Mathematics_ , 26, 189-206.
Jurafsky, D., & Martin, J. H. (2024). _Speech and language processing_ (3rd ed.). Pearson.
Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. _Proceedings of the 36th International Conference on Machine Learning_ , 3519-3529.
Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., & Jégou, H. (2019). Large memory layers with product keys. _Advances in Neural Information Processing Systems_ , 32.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. _Advances in Neural Information Processing Systems_ , 33, 9459-9474.
nostalgebraist. (2020). Interpreting GPT: The logit lens. _LessWrong_. Retrieved from https://www.lesswrong.com/posts/
Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., … & Zheng, Y. (2023). YaRN: Efficient context window extension of large language models. _arXiv preprint arXiv:2309.00071_.
Reuters. (2026, January 10). China is closing US technology lead despite constraints, AI researchers say. Retrieved from https://www.reuters.com/world/china/
Science Business. (2026). State of AI 2025: Five key charts for Europeans. Retrieved from https://sciencebusiness.net/news/ai/
Shannon, C. E. (1948). A mathematical theory of communication. _Bell System Technical Journal_ , 27(3), 379-423.
South China Morning Post. (2026, January). China AI has less than 20% chance to exceed US over next 3-5 years: Alibaba scientist. Retrieved from https://www.scmp.com/tech/big-tech/
Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. _arXiv preprint arXiv:2308.10248_.
Wall Street Journal. (2026, January). China AI race: US chips. Retrieved from https://www.wsj.com/tech/ai/
Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). Feature hashing for large scale multitask learning. _Proceedings of the 26th International Conference on Machine Learning_ , 1113-1120.
Xie, S., & others. (2025). mHC: Manifold-constrained hyper-connections for improved gradient flow in deep networks. _arXiv preprint_.
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., … & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. _arXiv preprint arXiv:2310.01405_.
Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: The count-min sketch and its applications. _Journal of Algorithms_ , 55(1), 58-75.
DeepSeek-AI. (2024a). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. _arXiv preprint arXiv:2405.04434_.
DeepSeek-AI. (2024b). DeepSeek-V3 technical report. _arXiv preprint arXiv:2412.19437_.
DeepSeek-AI. (2026). Engram: Conditional memory via scalable lookup—A new axis of sparsity for large language models. _arXiv preprint arXiv:2601.07372_.
Design-Reuse. (2026). China’s AI chip ambitions limited by HBM memory supply, notes report. Retrieved from https://www.design-reuse.com/news/
Ghandeharioun, A., Kim, B., & others. (2024). PatchScope: A framework for inspecting language model representations at scale. _Proceedings of the 41st International Conference on Machine Learning_.
Huang, X., & others. (2025a). OverEncoding: Hash-based N-gram embeddings for vocabulary expansion. _arXiv preprint_.
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. _Contemporary Mathematics_ , 26, 189-206.
Jurafsky, D., & Martin, J. H. (2024). _Speech and language processing_ (3rd ed.). Pearson.
Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. _Proceedings of the 36th International Conference on Machine Learning_ , 3519-3529.
Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., & Jégou, H. (2019). Large memory layers with product keys. _Advances in Neural Information Processing Systems_ , 32.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. _Advances in Neural Information Processing Systems_ , 33, 9459-9474.
nostalgebraist. (2020). Interpreting GPT: The logit lens. _LessWrong_. Retrieved from https://www.lesswrong.com/posts/
Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., … & Zheng, Y. (2023). YaRN: Efficient context window extension of large language models. _arXiv preprint arXiv:2309.00071_.
Reuters. (2026, January 10). China is closing US technology lead despite constraints, AI researchers say. Retrieved from https://www.reuters.com/world/china/
Science Business. (2026). State of AI 2025: Five key charts for Europeans. Retrieved from https://sciencebusiness.net/news/ai/
Shannon, C. E. (1948). A mathematical theory of communication. _Bell System Technical Journal_ , 27(3), 379-423.
South China Morning Post. (2026, January). China AI has less than 20% chance to exceed US over next 3-5 years: Alibaba scientist. Retrieved from https://www.scmp.com/tech/big-tech/
Wall Street Journal. (2026, January). China AI race: US chips. Retrieved from https://www.wsj.com/tech/ai/
Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). Feature hashing for large scale multitask learning. _Proceedings of the 26th International Conference on Machine Learning_ , 1113-1120.
Xie, S., & others. (2025). mHC: Manifold-constrained hyper-connections for improved gradient flow in deep networks. _arXiv preprint_.
