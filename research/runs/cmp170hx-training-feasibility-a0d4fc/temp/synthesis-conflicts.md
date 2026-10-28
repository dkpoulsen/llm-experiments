# Synthesis conflicts — resolved before the synthesizer runs

Five substantive conflicts across the three drafts. Each resolved against the source, not by averaging.

## Conflict 1: EP per-GPU communication volume — 62.9 GB vs 314 GB vs 79 GB

- **Draft A says:** the per-GPU EP volume is ~62.9 GB per step (T4), which is ~6× at the repo's denominator and ~2× at the corrected one.
- **Draft C says:** the per-GPU volume is 314 GB per step across 24 layers (T4), giving ~7× at 32k. It attributes the formula `T·K·h·(EP−1)/EP` to [[scalable-training-of-mixture-of-experts-models-with]].
- **Draft B says:** ~1.64 GB per GPU per all-to-all, ~3.3 GB per MoE layer for the dispatch+combine cycle, ~79 GB per step across 24 layers, "about 47 s at 0.53 GB/s against a 56.8 s window, or roughly 6×".
- **Source check:** the formula `T·K·h·(EP−1)/EP` does **not** appear in [[scalable-training-of-mixture-of-experts-models-with]] — a full-text grep of that note returns no instance of `EP−1`, `EP-1` or the formula. Draft C's citation is unsupported. The convention question is what matters: `T` is the tokens **on that GPU**, not the global batch. At a 1M-token global batch over 5 data-parallel ranks, `T_local = 200,000`. Then per layer, dispatch = 200,000 × 2 × 2048 × 0.8 = 6.55e8 elements = **1.31 GB bf16**; dispatch + combine = **2.62 GB/layer**; × 24 layers = **62.9 GB per GPU per step**.
- **Verdict:** **Draft A is correct at 62.9 GB.** Draft C's 314 GB is a 5× aggregate-vs-per-GPU error — it used `T = 1M` (the global batch) where the formula wants the per-GPU token count. That is precisely the error Draft C spends its MoE section criticising the repo for, so the draft must not carry it. Draft B's 79 GB is close to A but its stated time is inconsistent: 79 GB at 0.53 GB/s is 149 s, not 47 s; 47 s is 79 GB at 1.68 GB/s (the raw link rate, not the achieved algbw). Use **62.9 GB**, and state the ratio against both compute windows explicitly.

## Conflict 2: the EP and TP cost ratios at 32k

- **Draft C says:** at 32k, EP and TP both come to **~7×**, and the repo's ~28× figure is a short-context artefact.
- **Draft B says:** EP is **~6×** "once aggregate volume is divided by GPU count".
- **Draft A says:** EP is ~6× at the repo's denominator and ~2× at the corrected one.
- **Source check:** recompute both against the two compute windows. EP = 62.9 GB → at the repo's 0.805 GB/s that is **78 s**; against the repo's 6ND compute window (7.4 s) that is **10.6×**, and against the corrected 32k window (56.3 s) it is **1.39×**. TP does **4** all-reduces per layer over forward+backward (2 forward, 2 backward), each moving `T_local × h × 2 B = 819 MB` per rank, × 1.6 ring factor = 1.31 GB on the wire, so 4 × 1.31 × 24 = **126 GB** → 156 s → **2.8×** against the corrected window.
- **Verdict:** **EP ≈ 1.4× and TP ≈ 2.8× at 32k under the corrected compute window.** Both are far below the repo's 70×, and both are below the ~7× Draft C asserts. The consequence is load-bearing and must be stated plainly: **the bandwidth case against EP and TP evaporates at 32k.** What remains is (a) availability — no SM80 training-grade EP implementation exists, per [[efficient-expert-parallel-communication]]'s closed list of DeepEP / pplx-kernels / NIXL-EP / NCCL EP / UCCL-EP all requiring NVLink, GPUDirect P2P or RDMA; and (b) critical-path serialization — TP's all-reduces sit inside the layer, EP's all-to-all can overlap with expert compute. The report must not argue EP/TP out on arithmetic; it must argue them out on availability and scheduling.

## Conflict 3: the frozen-table byte count — 12 GB total vs 24 GB

- **Draft B says:** read as "12 GB total", the design closes at ~48 GB/card; read as "12 GB per layer" it breaks at 72 GB/card. B concludes the design is sound and "the specification is off by a factor of three".
- **Draft A and Draft C say:** the frozen table is **12.29B params / 24.58 GB** (two grafted layers at ~12.3 GB each), so the 1B-trainable design lands at **~53 GB/card**.
- **Source check:** [[memory-grafting-scaling-language-model]] specifies 3M entries per grafted layer (top 1M n-grams per order × 3 orders), each a 2048-wide bf16 hidden vector = **12.3 GB per layer**, and states "approximately 12B frozen grafting memory" for the 0.92B-trainable recipient. Twelve billion params at bf16 is 24 GB. So the paper's figure is **per layer**, and the total for the recipient is **24 GB**, not 12 GB.
- **Verdict:** **24 GB is correct; Draft B's "12 GB total" reading is the repo's reading, which the paper contradicts.** The design still fits — 16 GB trainable state + 24 GB frozen + ~3.2 GB checkpointed activations + ~9.9 GB naive logits = **53.1 GB**, or ~43 GB with fused cross-entropy — but the headroom is ~11 GB, not ~20 GB. Draft B's conclusion (the design is sound) survives; its byte count and its "factor of three" characterisation do not. The report should say: the repo's 6B/12 GB is exactly **one** layer's worth, and the paper's recipient carries two.

## Conflict 4: the FLOPs ratio vs the wall-clock ratio at 32k

- **Draft A says:** for the repo's 500M baseline at 28 layers and hidden 2048, attention is 88% of the step and total FLOPs are **8.5×** the 6ND estimate.
- **Draft C says:** the quadratic term inflates compute **4.2×**, and 80–120k tok/s at 32k would require 400–600 TFLOPS/card.
- **Source check:** both are FLOPs-convention numbers, and they differ because they use different model configurations. But the configuration Draft A uses (28 layers × hidden 2048) is arithmetically **1.41B parameters**, not 500M — the repo's own baseline triple is internally inconsistent. More importantly, the FLOPs ratio is **not** the wall-clock ratio: [[readmemd-faircslmtrainbench-at-main]] measures a 505.5M model at 4k = 26,297 tok/s (39.88% MFU) and 32k = 11,990 tok/s (63.88% MFU), a measured per-token cost ratio of **2.19×**, with a 1.60× MFU uplift absorbing 40% of the quadratic penalty.
- **Verdict:** **use 2.19× as the wall-clock ratio and state the FLOPs ratio separately with its convention.** The report must not present 8.5× as a wall-clock penalty. The correct sentence is: attention is ~76–88% of step *FLOPs* at 32k depending on configuration, total FLOPs are 4.2–8.5× the 6ND estimate, but the *measured* per-token wall-clock penalty is **2.19×** because the 32k step runs at 63.88% MFU rather than the 30–50% the FLOPs-only reading assumes. This is the single most important calibration in the report, and all three drafts get it partly wrong.

## Conflict 5: structural and hygiene violations in Draft B

- **Draft B's headings are numbered** — `## 1. Hardware Reality: Link Width, Power Brake, and What the Numbers Assume`, `## 2. Training Topologies on a No-P2P PCIe Fabric`, etc. The structural contract requires the **literal** headings `Hardware Reality`, `Training Topologies`, `DDP Overhead Model`, `MoE & EP`, `Engram & Frozen Memory`, `CSA2 at 32k`, `Verdict`, unnumbered and unembellished.
- **Draft B cites workspace artifacts** — `[[interim-report-topology-cost-at-1m-tokens]]`, `[[interim-report-memory-budget-and-frozen-tables]]`, `[[interim-report-token-budget-and-effect-size]]`. These are pipeline artifacts, not sources, and must be stripped; the underlying source notes must be cited instead.
- **Verdict:** the synthesizer must use the seven literal unnumbered headings and must cite only source notes. Draft A and Draft C already comply.
