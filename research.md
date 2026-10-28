# Deep Research: Training Modern Architectures on the 5× CMP 170HX Box

Working research notes for the experiment program. Claims are labelled **measured** (someone
measured it on this class of box), **reported** (a community/published number), or **estimate**
(arithmetic from the stated assumptions — to be confirmed by an experiment).

## Headline findings

1. **Verify the link width before anything else.** Stock driver trains the cards at PCIe **Gen1**;
   the cmpunlocker link patch lifts them to **Gen2**, and the negotiated *width* is set by the slot
   (x4, x8, or x16). Gen3 is fuse-blocked (`OPT_DISABLE_GEN3_SPEED` reads 1 on every card checked).
   If the cards are actually Gen2 **x16** rather than x4, every communication cost below drops ~4×
   and the "large-batch-only" conclusion weakens a lot. One `lspci -vv` per card settles it.
2. **Expert/tensor parallelism is not viable for training here.** EP does two all-to-alls *per MoE
   layer*; at 1M-token batches that is ~70× the compute time on this fabric (estimate below). TP is
   the same shape. The community inference numbers agree in direction: PP beats TP by **6.6× on
   prefill** on 4× CMP 170HX, because TP runs 86 all-reduces per forward over a no-P2P link.
3. **DDP's overhead depends only on batch size, not model size** — compute and gradient payload both
   scale with params. At 1M-token global batches it is ~36%; at 2M, ~18%. So the brief's "big global
   batches" rule is the right lever, and it is the *only* lever.
4. **The 7B-total target is a memory problem, not a bandwidth problem.** 7B fully trainable with Adam
   needs ~112 GB per card — it does not fit DDP on 64 GB. The fix is to **freeze the Engram memory
   tables**: frozen params carry no gradients and no optimizer state, so 7B total with ~1B trainable
   fits DDP at ~28 GB/card.
5. **Vanilla Engram tables are learned from scratch and are expensive to scale.** Memory Grafting
   (Tsinghua / MSRA, 2026) builds them from a frozen donor model's hidden states instead, and reports
   gains at **0.92B scale** — exactly this box's range.
6. **CSA2's published win is mostly inference-side KV cache.** At 32k training the indexer adds a
   parallel attention and a non-differentiable top-k (needs a distillation loss). Worth testing, but
   it is the least-proven axis on this box.

## 1. Hardware reality

| Fact | Status |
|---|---|
| 5× CMP 170HX, GA100, sm_80, 64 GB HBM2e after unlock, 250 W cap | measured (user) |
| bf16 ≈162 TFLOPS/GPU, fp32 ≈13 | measured (user) |
| PCIe Gen2 x4, no NVLink, no P2P | measured (user) |
| NCCL all-reduce across 5 GPUs ≈0.75 GiB/s | measured (user) |
| Stock driver trains at Gen1; cmpunlocker link patch → Gen2; Gen3 fuse-blocked | measured (club-170hx) |
| No P2P under VFIO passthrough — collectives round-trip through host RAM | measured (club-170hx) |
| Link width is per-slot: x8 on a bifurcated slot is the slot's ceiling, not a fault | measured (club-170hx) |

**The PWRBRK# trap.** Some workstation boards (notably ASUS Pro WS WRX80E-SAGE) assert `PWRBRK#` on
edge pin B30, pinning the card at ~88 W / 39.3 TFLOPS fp16 instead of 105–180 W / 155.7 TFLOPS. It
looks exactly like "mining cards are just slow". Check with
`nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"` — `Active` on a card that is not thermally or
power limited means the platform is braking it. Fix: Kapton tape over B30, or a riser that does not
route B30. The user's measured 162 TFLOPS says these cards are healthy.

**Precedent on this exact hardware class.** `allover326/deepseek-v4-cmp170hx` runs DeepSeek-V4-Flash
(~284B total / ~13B active) on 4× CMP 170HX at Gen2 x4, no P2P: **98 tok/s decode, ~5,300 tok/s
prefill, 1,047,736 verified context**. That is inference, not training, but it establishes that the
box can hold and move a 155 GiB checkpoint across 4 cards. Note the honest limits in that repo:
retrieval accuracy falls to ~30% at 900k context, and TTFT at 1M is 9.2 minutes.

## 2. MoE on a no-P2P PCIe fabric

Communication patterns, cheapest to most expensive on this box:

| Strategy | What crosses the link | Frequency |
|---|---|---|
| **DDP** | gradient all-reduce (P × 2 B) | once per step |
| **FSDP / ZeRO-3** | param all-gather + grad reduce-scatter | per layer |
| **PP** | activation hand-off (T × hidden × 2 B) | per stage boundary |
| **TP** | all-reduce per layer | per layer |
| **EP** | two all-to-alls per MoE layer | per MoE layer |

**Estimate — overhead vs compute at 1M-token global batches.** Assumptions: 0.75 GiB/s effective
collective bandwidth, 810 TFLOPS bf16 aggregate, 6 FLOPs/param/token, bf16 gradients, 24 layers,
hidden 2048, top-k 2, 1B active params.

| Strategy | Payload per step | Time @0.75 GiB/s | Compute time | Overhead |
|---|---:|---:|---:|---:|
| DDP (1B trainable) | 2.0 GB | 2.7 s | 7.4 s | **36%** |
| FSDP (7B trainable) | ~28 GB (gather + scatter) | 37 s | 52 s | **72%** |
| PP (5 stages) | ~16 GB | 21 s | 7.4 s | **~2.9×** |
| EP (24 MoE layers) | ~400 GB | 537 s | 7.4 s | **~72×** |

The DDP row is the important one: its overhead is `3.6e5 / global_batch_tokens` at theoretical peak
(810 TFLOPS, 100% MFU, giving 7.4 s compute and 36% overhead at 1M tokens). At realistic ~55%
achieved MFU (405 TFLOPS aggregate, matching the measured 91–100 TFLOPS/GPU from the brief), compute
time is ~14.8 s, which drops 1M-token all-reduce overhead to ~18–20%. That reproduces the brief's
measured behaviour (494M model at ~140k tok/s with 1M-token batches ≈ 18% overhead). At 2M tokens
it drops further to ~9–10%.

**Consequence:** EP and TP are out for training. The choice is DDP or FSDP, and DDP is ~2× cheaper on
the link. So the design goal is *make the trainable set fit on one card*.

## 3. Engram: the training-cost problem and the frozen-table fix

Engram (DeepSeek, arXiv 2601.07372) is a **conditional memory** module: the preceding few tokens are
hashed into large static embedding tables, and the retrieved rows are fused into the hidden state,
gated by how well they match. It is a sparsity axis orthogonal to MoE:

| | MoE (conditional computation) | Engram (conditional memory) |
|---|---|---|
| What scales | neural parameters | static memory |
| Selection | learned routing (gradient) | deterministic hashing (gradient-free) |
| Lookup cost | O(k), k = active experts | O(1) |
| Per-token FLOPs | grows with active params | **constant** — more slots, same FLOPs |

That last row is why Engram fits this box: **you can add capacity without adding compute**. But the
paper's own tables are *learned from scratch during pretraining*, which makes memory scaling expensive
and sometimes ineffective.

**Memory Grafting** (arXiv 2605.20948) fixes that: run a donor model offline over frequent local
n-grams, store its final-token hidden states as **frozen** memory values, and let the recipient model
retrieve them by exact longest-match suffix lookup, with a hash-based Engram fallback for unmatched
contexts. Reported numbers:

- 33M entries per grafted layer (top 11M n-grams per order), bf16 hidden vectors.
- For Qwen3.5-35B-A3B, roughly **12 GB of frozen memory per layer**.
- 2.8B scale: average benchmark 51.95 (MoE) → 52.43 (vanilla Engram) → **53.86** (grafting).
- 0.92B scale: all grafting variants beat both baselines.

**Why this matters here:** frozen tables cost zero optimizer state and zero per-token FLOPs. They are
pure VRAM. On a box with 320 GB of VRAM and a link that punishes parameter traffic, that is the
cheapest capacity available.

## 4. CSA2: what it buys at 32k, and what it costs to train

DeepSeek-V4.1-Flash (arXiv 2609.19969) is a 552B-total MoE (16B active decode, 8B active prefill),
pretrained on 45T tokens, 1M context. CSA2 assigns each attention layer one of three static modes:

- **Full** — own global KV, own indexer, fresh top-k indices.
- **Reindex** — borrows a preceding layer's global KV, rescoring with its own indexer Q.
- **Reuse** — borrows both KV and top-k indices; no indexer at all.

Plus a hierarchical sparse indexer that caps the candidate pool at 16,384 positions, and FP4 KV
caching. Result: **890 bytes of KV cache per token**, ~1/4 of V4-Flash, with decode FLOPs nearly flat
as context grows.

**Training-side caveats (estimate / inferred):**

- Top-k selection is non-differentiable. The DSA lineage trains the indexer with a separate
  distillation loss against the main attention — extra code and an extra loss term, not a drop-in.
- At 32k the indexer is a second, smaller attention running in parallel; the net FLOP win depends on
  the layer mix (how many Full vs Reindex vs Reuse layers).
- The candidate-pool cap (16,384) is tuned for 1M context. At 32k it barely binds, so the sparsity
  win is smaller than the published numbers suggest.
- The published 890 B/token figure is an inference KV-cache number, not a training activation number.

**Verdict:** worth one experiment, but it is the axis most likely to disappoint at 32k on this box.

## 5. Memory math: what fits on one 64 GB card

Adam state per trainable param: fp32 master + fp32 m + fp32 v = 12 B, plus bf16 weight (2 B) and
bf16 grad (2 B) = **16 B per trainable param**. Frozen params cost only their bf16 weight (2 B).

| Design | Weights | Grads | Adam | Total | Fits DDP? |
|---|---:|---:|---:|---:|---|
| 7B all trainable | 14 GB | 14 GB | 84 GB | **112 GB** | No — needs FSDP |
| 7B total, 1.5B trainable + 5.5B frozen | 14 GB | 3 GB | 18 GB | **35 GB** | Yes |
| 7B total, 1B trainable + 6B frozen | 14 GB | 2 GB | 12 GB | **28 GB** | Yes |

Activations at 32k with full checkpointing and gradient accumulation: with a per-GPU micro-batch of
1 sequence (32,768 tokens) and 6 accumulation steps to reach the 1M global batch, each GPU holds only
1 active sequence during backpropagation: 32,768 tokens × 2048 hidden × 2 B × 24 layers ≈ **3.2 GB**
(not 20 GB — micro-batches do not keep activations across accumulation steps).

So the 1B-trainable / 6B-frozen design lands at ~31 GB/card (28 GB weights/optimizer + ~3.2 GB
activations), and the 1.5B-trainable design lands at ~38 GB/card. Both easily fit a 64 GB card with
over 25 GB of headroom for KV cache, workspace, and CUDA buffers. Anything above ~2B trainable
needs FSDP and pays the doubled link overhead.

## 6. What this changes in the plan

- **Add a pre-flight check to every experiment plan:** `lspci -vv` link width/speed per card, and
  `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"`. Both are cheap and both invalidate results.
- **Add a "frozen memory" variant to the Engram axis.** Vanilla Engram (learned tables) vs Memory
  Grafting (frozen tables) is the comparison that decides whether 7B total is reachable with DDP.
- **Drop EP and TP from the training topology candidates.** The candidates are DDP and FSDP.
- **Reconsider 32k.** The indexer's candidate cap barely binds at 32k, so the sparse-attention win is
  smaller than the published 1M-context numbers. Either accept a modest win or test 128k on one axis.
- **Treat the trainable-param count, not the total, as the budget.** The goal doc says "7B total /
  ~1B active"; the sharper constraint is "≤ ~1.5B trainable, rest frozen".

## Open questions to settle by measurement

1. What is the actual negotiated link width per card? (Gen2 x4 vs x16 changes every number above.)
2. Does the 0.75 GiB/s all-reduce hold with `bf16_compress_hook`, and at what message size?
3. What does a 1B-trainable DDP step actually cost at 1M and 2M tokens? (Validates the 36%/18% model.)
4. How large can the frozen Engram table be before lookup latency (host-side hashing) becomes the
   bottleneck rather than VRAM?
5. Does CSA2 at 32k beat full attention on this box at all, given the indexer overhead?

## Sources

- PixelML/club-170hx — `docs/HARDWARE.md`, `docs/TOPOLOGY-AND-PARALLELISM.md` (link width, PWRBRK#,
  PP-vs-TP measurements)
- allover326/deepseek-v4-cmp170hx — README (DeepSeek-V4-Flash on 4× CMP 170HX, PP vs TP)
- localai.co.kr — CMP 170HX multi-GPU guide (Gen3 fuse-blocked)
- arXiv 2601.07372 — *Conditional Memory via Scalable Lookup* (Engram)
- arXiv 2605.20948 — *Memory Grafting: Scaling LM Pre-training via Offline Conditional Memory*
- arXiv 2609.19969 — *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* (CSA2)
- arXiv 2609.40093 — *Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs*
- github.com/deepseek-ai/Engram — official module demo
