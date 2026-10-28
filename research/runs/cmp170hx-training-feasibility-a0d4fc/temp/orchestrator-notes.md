# Orchestrator notes — cmp170hx-training-feasibility-a0d4fc

## Wave 1 dispatched (10 fetchers, 80 URLs)

### Contradictions already visible from the search phase

1. **Link width — the repo's #1 open question has a measured answer that contradicts the premise.**
   `research.md` and `goal-and-requirements.md` both assume PCIe **Gen2 x4**. club-170hx's HARDWARE.md
   (update 2026-10-02) reports that with the cmpunlocker link patch, all four cards on a 4-card PLX
   host train **Gen2 x16**; Gen3 is fuse-blocked (`OPT_DISABLE_GEN3_SPEED` = 1). If the user's box
   is x16 rather than x4, every communication number in `research.md` §2 drops ~4× and the
   "large-batch-only" conclusion weakens a lot. This is the single highest-leverage finding.

2. **P2P — the repo says "no P2P"; the corpus says P2P works over BAR1.**
   `bayley/cmpunlocker` reports P2P working over **BAR1 P2P at 1.68 GB/s each way on PCIe Gen2 x4**,
   verified with real transfers (not `cudaDeviceCanAccessPeer`). club-170hx reports "no peer access
   with the plain unlock; copies stage through host memory, 6.21–6.30 GB/s at 256 MiB". So the
   answer depends on which unlock path — plain vs BAR1 P2P. The repo collapses this into "no P2P",
   which is probably wrong for the patched path.

3. **FSDP multiplier — the repo contradicts itself.**
   `goal-and-requirements.md` says FSDP costs **~1.5×** more communication than DDP.
   `research.md` §2 says FSDP costs **~2×** DDP. The literature (ZeRO paper, FSDP paper) says
   FSDP moves **3Ψ vs DDP's 2Ψ = 1.5×** byte volume. So `research.md`'s 2× is wrong, and its
   overhead table is worse than that: it compares FSDP at **7B trainable** against DDP at
   **1B trainable**, so the 72% vs 36% rows conflate param count with strategy.

4. **TFLOPS — the repo's 162 TFLOPS bf16 claim is contested.**
   Published measurements on CMP 170HX report **~42 TFLOPS fp16** (CUDA cores), with tensor-core
   paths restricted by NVIDIA. The repo claims **bf16 ≈162 TFLOPS measured**. Either the user's
   cards have working tensor cores (possible — the unlock path may restore them) or the number is
   CUDA-core arithmetic rather than measured. This needs an explicit resolution: every throughput
   estimate in the docs is built on 810 TFLOPS aggregate.

5. **CSA2 candidate cap at 32k — the repo says it "barely binds".**
   The corpus says the hierarchical sparse indexer caps the candidate pool at 16,384 positions,
   which *does* bind at 32k (a 32k sequence exceeds the cap by 2×). So the repo's reasoning
   ("at 32k it barely binds") is backwards: at 32k the cap binds harder than at 1M, where it is
   a small fraction of context. This flips the CSA2 verdict's reasoning even if the verdict holds.

6. **PP vs TP — repo and corpus agree.** 6.6× prefill advantage for PP on 4× CMP 170HX
   (5,321 vs 801 tok/s at 77k context), 86 all-reduces per forward for TP. Consensus claim.

### Expected tensions for step 3

- **x4 vs x16** (hardware measurement vs repo assumption) — highest decision impact.
- **"no P2P" vs "BAR1 P2P works"** — changes whether collectives stage through host RAM.
- **1.5× vs 2× FSDP** — internal repo inconsistency plus a confounded table.
- **162 TFLOPS vs 42 TFLOPS** — decides whether the 810-TFLOPS aggregate is real.
- **CSA2 at 32k: win vs disappointment** — the repo's own verdict is "worth one experiment but
  most likely to disappoint"; the candidate-cap argument for that verdict is inverted.
- **Frozen Engram tables: capacity win vs lookup-latency wall** — the repo treats frozen tables as
  "pure VRAM"; the corpus adds host-side hashing latency and SSD-backed lookup stalls as the real
  constraint at scale.

### Thesis forming

The repo's experiment program is directionally right (DDP, large global batches, frozen memory
tables) but its quantitative spine is built on two unverified premises — Gen2 **x4** and
**162 TFLOPS bf16** — and one confounded comparison (FSDP at 7B trainable vs DDP at 1B trainable).
If the link is actually x16, the program is *less* constrained than the docs claim and the
"large-batch-only" rule becomes a preference rather than a necessity. If tensor cores are actually
restricted, the 9B-token budget at 32k is far more expensive than the 21–25 h estimate.

### What could overturn the thesis

A measured `lspci -vv` from the user's actual box showing Gen2 x4, or a measured bf16 matmul
benchmark showing ~42 TFLOPS. Both are cheap pre-flight checks the repo already specifies — the
report should insist they run before any experiment.
