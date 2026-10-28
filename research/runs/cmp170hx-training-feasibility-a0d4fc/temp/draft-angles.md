# Draft angles — step 10.1

Three genuinely different drafts. The query is argumentative (does the repo's feasibility case survive scrutiny?), and the corpus has a sharp fight, so the angles follow the tension-based assignment: strongest-thesis / steelman-contrarian / synthesis-reconciler.

## Draft A — Strongest-thesis

**Angle:** The repo's experiment program is directionally right and its quantitative spine is wrong. After correction the PCIe fabric is *not* the binding constraint — DDP overhead at 1M-token batches is single-digit percent — and the real constraint is wall-clock at ~90–100 TFLOPS/card. Argue this forcefully: the repo reached a defensible conclusion (data-parallel, large batches, frozen tables) while both of its premises (Gen2 x4, 162 TFLOPS) and one of its comparisons (FSDP 2×) fail. Every correction is a *measured* number, not arithmetic, and the corrections move in opposite directions, which is why the repo's conclusion survives its own arithmetic being wrong.

**Evidence base:** the measured 32k/4k cost ratio (2.19×), the measured seed-variance floor (0.004 nats), the measured Gen2 x4 link rate (1.6 GB/s), the measured BAR1-P2P N=2/N=3 inversion, the measured 180 W sustained TFLOPS ladder, and the measured end-to-end NSA payoff (1.71×).

## Draft B — Steelman-contrarian

**Angle:** Defend the repo's original framing. The fabric *is* the binding constraint, and the frozen-table design is the right response to it. The strongest case: the repo's 0.75 GiB/s is a user measurement on the actual box, and if it is algbw (not busbw) then DDP overhead at 1M tokens is 54–58% and the whole "wall-clock is binding" correction collapses. No 5-rank all-reduce receipt exists anywhere, so the corrected 6.6% is arithmetic on adjacent-hardware inputs, not measurement. P2P is a pessimization at N=3 and at Gen2 x4. ZeRO-2's real overhead is CPU-side Python hooks, which is fabric-independent and would bite hardest on MoE. Sparse attention's sign at 32k depends on a GQA group size nobody has measured. And the repo's own 140k tok/s datapoint implies 84 TFLOPS/card under 6ND — a plausible 42% MFU — so the "162 TFLOPS is wrong" correction may be over-correcting.

**Evidence base:** the 0.75 GiB/s user measurement, the N=3 P2P inversion, the Gen2-x4 P2P pessimization, the absent 5-rank receipt, the ZeRO-2 CPU-hook mechanism, the MoE total-vs-active payload asymmetry, the 42/49 TFLOPS stock figures, the FSA finding that NSA is *slower* than full attention at 32K when g=1, and the LongCat 0.83× at 32K.

## Draft C — Synthesis-reconciler

**Angle:** Both sides capture part of the truth, and the boundary condition is which premise is true — which is decidable by two cheap measurements the repo never ran. If the link is Gen2 x4 and sustained throughput is ~90–100 TFLOPS/card, the fabric is not binding and wall-clock is; if the link is Gen1 x4 or the 0.75 GiB/s was algbw, the fabric binds and the frozen-table design is right. Build the decision tree explicitly: `lspci -vv` for the negotiated triple, a bf16 matmul benchmark for sustained TFLOPS, an `all_reduce_perf` for the column question, and a GQA-group-size check for the sparse-attention sign. Then show that the recommendation is *robust* across the branches even though the reasoning is not — which is the report's real contribution.

**Evidence base:** the axes-independence warning in the physical-mods page, the four possible link states, the TFLOPS ladder across lock state × power cap × duration, the two measured noise floors, the evidence-tier discipline the club applies to its own numbers, and the boundary conditions each source states for itself.
