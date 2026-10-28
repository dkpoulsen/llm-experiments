---
title: Interim report — allreduce-ceiling-and-definition
id: interim-report-allreduce-ceiling-and-definition
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-allreduce-ceiling-and-definition
created: '2026-10-27T13:31:43.199205Z'
updated: '2026-10-27T19:00:31.046459Z'
status: review
type: interim
tier: practitioner
content_type: article
deprecated: false
summary: '0.75 GiB/s is the nccl-tests busbw column, not algbw: at N=5 busbw = 1.6x
  algbw by construction, and 0.805 GB/s sits at 75-96% of the only physically available
  Gen2-x4 host-staged band (0.84-1.07 GB/s) while as algbw it would demand 77% of
  the one-way link rate on a double-traversal non-pipelined ring (best ever measured:
  64%). The repo''s 36% DDP overhead at 1M tokens is really 54-58% (constant 5.4e5-5.8e5,
  not 3.6e5), and its 2.0 GB payload already assumes bf16_compress_hook. BAR1 P2P
  is a latency lever (+30% on 2-GPU NCCL at x16, -47% forced across a host bridge,
  hangs collectives on 64 MB BAR1), not a bandwidth fix; the 24-capacitor x16 mod
  is the only order-of-magnitude lever at 3.9-4.0x busbw, saturating near 4.3 GB/s.'
---

# Interim report: allreduce-ceiling-and-definition

**Locus question:** What all-reduce bandwidth can DDP actually get on this box: is 0.75 GiB/s the structural Gen2-x4 host-staged ceiling, what does the number even measure (algbw, busbw, or aggregate), does BAR1 P2P lift it, and does the 24-capacitor x16 mod multiply it?
**Flavor:** dialectical

## What the corpus already said

The corpus holds **four mutually incompatible measurements of one quantity**, and the repo divides every overhead figure by a fifth number that is not obviously the same quantity as any of them.

1. The repo's figure: `research.md` §1 lists "NCCL all-reduce across 5 GPUs ≈0.75 GiB/s | measured (user)", and §2 divides a 2.0 GB DDP payload by it to get 2.7 s and a 36% overhead at 1M-token global batches. It is a **user self-report with no receipt, no message size, no rank count beyond "5", and no column named**. Both [[performancemd]] and [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]] state explicitly that their pages contain **no training, all-reduce, NCCL or DDP measurement at all** — so nothing in the corpus can corroborate it. This is the single most load-bearing unverified number in the run.
2. ~1.0 GB/s "measured bus bandwidth" at Gen2 x4, no P2P — [[lessonsmd]] §c and [[deepseek-v4-cmp170hxsettingsmd-at-317dac12bbd7f401f0538c76161bc7186086fb65-allov]]. Note the phrase *bus bandwidth*: this is already a busbw claim, not an algbw claim.
3. 3.69 / 3.85 / 4.24 GB/s NCCL all-reduce **busbw** at 256 MiB — [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]], receipt in [[readmemd]]. But the receipt's hardware section is unambiguous: **three cards at Gen2 x16 behind PLX PEX 8747 switches, no BAR1-P2P patches, NCCL transport SHM for every pair**, `canAccessPeer`/`enablePeerAccess` returning 0 / `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` on all six pairs. This is an **x16** number, not an x4 number.
4. BAR1 P2P at **1.68 GB/s each way at Gen2 x4** on 12/12 directed pairs — [[github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2]]. The same page records that a card trained at x2 "ran exactly half speed everywhere (0.84 GB/s same-switch)", which is the cleanest confirmation in the corpus that this fabric's bandwidth is exactly linear in negotiated width.
5. Static-BAR1 P2P at **5.79 GB/s** on the x16 host, vs **6.21–6.30 GB/s** host-staged at 256 MiB — [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] §5. P2P is *slower* for bulk copies above 16 MiB and only wins on latency (16/25/194 µs vs 22/41/343 µs).

Analyst b's ceiling derivation (ring per-rank load 2(N−1)/N = 1.6 at N=5, host staging doubles PCIe traversal to 3.2× payload, 1.68/3.2 = 0.525 GB/s = 0.489 GiB/s) is arithmetically correct **as an algbw ceiling**. The repo's 0.805 GB/s sits 1.53× above it. That gap is the dialectical crux: either the fabric is better than assumed, or the number is not algbw.

Two corpus facts sharpen the 5-rank case specifically. [[evaluating-modern-gpu-interconnect-pcie]] (Li/Song/Barker, PNNL, IEEE TPDS) states: *"the NUMA effects appear quite significant with 5 GPUs, implying that there may be a congestion when forming a NCCL ring among 5 GPUs. One should avoid adopting 5 GPUs in their application setup."* And for PCIe, "CL bandwidth decreases with more GPUs, which is due to bus contention in a tree-network", with all-reduce showing "the largest latency" of the five collectives. So the N=5 penalty is likely **worse** than the 1.6× ring-load factor alone — which pushes the ceiling *down*, not up.

## What the new sources say

**[[performancemd-2]] — nccl-tests `doc/PERFORMANCE.md`.** This is the canonical definition and it settles the definitional half of the locus outright. Algorithm bandwidth is `algbw = S/t`. Bus bandwidth is derived: *"Considering that each rank has a bandwidth to the outside world of B, the time to perform an allReduce operation of S elements is at best: t = (S*2*(n-1))/(n*B)"*, therefore *"B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)"*. The page closes: *"The bus bandwidth should reflect the speed of the hardware bottleneck : NVLink, PCI, QPI, or network."* So **busbw is literally the per-rank link rate**, and algbw is busbw discounted by the ring-load factor N/(2(N−1)) — **exactly 5/8 = 0.625 at N=5, i.e. busbw = 1.6 × algbw.**

**[[all_reducecu]] — nccl-tests `src/all_reduce.cu`.** The code confirms the formula and that it is applied unconditionally, regardless of algorithm or transport: `double factor = ((double)(2*(nranks - 1)))/((double)nranks); *busBw = baseBw * factor;`.

**[[whats-the-difference-in-allreduce-for-factor-between-the-nccl-and-nccl-test-issu]] — nccl-tests issue #274, Sylvain Jeaugey (NVIDIA).** The most important sentence in the whole batch for the ceiling question: *"The AlgBw->BusBW is theoretical, on a flat hardware topology, using point-to-point transfers to execute the operation… If BusBW peaks at say 50GB/s, it's telling you that your system is equivalent to a system where each GPU would be connected to a flat fabric at 50GB/s."* The formula's derivation assumes **one point-to-point transfer per hop**. On a host-staged fabric (no P2P) each hop is *two* PCIe traversals, so the achieved busbw is capped below the one-way link rate — the formula's own premise is violated, and the reported busbw silently absorbs the penalty.

**[[pci-express-wikipedia]] — PCIe link physics.** PCIe 2.0 is 5 GT/s per lane with 8b/10b encoding → 500 MB/s per lane per direction, so **Gen2 x4 = 2.0 GB/s per direction** and Gen2 x16 = 8 GB/s. Critically for the ceiling: *"each lane is a dual simplex channel"* and a link *"supports full-duplex communication between any two endpoints, with no inherent limitation on concurrent access"*. So the 1.68 GB/s measured at x4 is a **per-direction** figure, and send and receive can overlap. This matters because it means the host-staging penalty is not a hard 2× — it is a *serialization* penalty, and its size depends on whether NCCL overlaps the D2H and H2D halves of each hop.

**[[exploring-gpu-to-gpu-communication]] — De Sensi et al., SC'24.** Confirms the mechanism taxonomy as four distinct modes: *"Trivial staging (GPU->Host->Host->GPU)"*, *"Device-device copy (peer access)"*, CCL, and GPU-aware MPI. The staging path is named as a two-hop host round trip, which is the corpus's `NCCL_SHM_DISABLE` documentation restated (*"SHM is used between devices when peer-to-peer cannot happen, therefore, host memory is used"* — [[environment-variables-nccl-2323-documentation]]). The deck's quantitative content is in images, so it corroborates the mechanism, not the magnitude.

**[[ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compre]] — a community DDP measurement on a PCIe-only, no-NVLink box (4× RTX A6000, Gen3 x16, every pair PHB).** This is the closest thing in the corpus to an end-to-end DDP measurement on a no-NVLink PCIe fabric, and it is the strongest single source for the *shape* of the problem:

- *"NCCL bus bandwidth between the four cards was 4.3 GB/s (2 GPUs: 3.9 GB/s), flat from 8 MB to 1 GB messages."*
- *"At 2.9 GB/s of algorithm bandwidth that is roughly 170 ms of communication against a 130 ms compute step"* — note 4.3 × 4/6 = 2.87, i.e. the author applies the nccl-tests conversion correctly.
- *"DDP all-reduces fp32 gradients: about 500 MB per step for the 124M model."*
- *"Forcing peer-to-peer made it worse. `NCCL_P2P_LEVEL=SYS` … halved the all-reduce bandwidth to 2.3 GB/s and dropped 124M training from 129k to 80k tokens/s. NCCL's default choice was right; don't override it on a PHB machine."*
- bf16 gradient compression moved 4-GPU speed-up from **2.06× to 2.65×** (124M) and **1.91× to 2.72×** (355M); gradient accumulation ×4 moved it to 3.20×/3.10×; both together 3.54×/3.66×.

Two things follow. First, **a Gen3 x16 host-staged fabric delivers 4.3 GB/s busbw — only 1.01–1.16× the CMP box's Gen2 x16 figure of 3.69–4.24 GB/s, despite roughly double the line rate.** The host-staged path saturates around 4.3 GB/s busbw and stops caring about link generation. Second, forcing P2P across a host bridge *halves* busbw. P2P is not a free bandwidth lever.

**[[reduce_kernelh]] — NCCL `src/device/reduce_kernel.h`.** This kills the "BF16 all-reduce is already 32-bit internally" hypothesis outright. Under `#if __CUDA_ARCH__ >= 800` — which is exactly GA100/SM80, the CMP 170HX's architecture — NCCL specialises `FuncSum` for bfloat16 to **native 16-bit arithmetic**: `SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __hadd(x, y))` and `SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 2, __nv_bfloat162, __hadd2(x, y))`. The fp32 path (`__float2bfloat16(__bfloat162float(x) + __bfloat162float(y))`) is the `#else` branch for pre-SM80 hardware. So on this box the wire dtype is bf16 (2 bytes/element) **and** the accumulation is bf16 — there is no widening anywhere.

**[[half-precision-reduction-accumulation-in-fp32-issue-1026-nvidianccl-github]] — NCCL issue #1026, Jeaugey.** Confirms the same from the maintainer's mouth: *"we can't really accumulate in fp32 because it would mean communicating fp32 between GPUs, which would be the same as running the allreduce operation on fp32 to start with."* And the precision warning that matters here: *"The least stable algorithm is rings, which unfortunately is also the one which gives the best peak bandwidth on most platforms. The tree algorithm … is much more stable because it adds values of equal weights."* On a single-node 5-GPU box the tree degenerates to a chain ([[demystifying-nccl-an-in-depth-analysis-of-gpu]]: tree "builds its branching structure only across nodes … so on a single-node 5-GPU box the tree degenerates to a chain and buys nothing over ring"), so you cannot buy numerical stability by switching algorithm.

**[[does-nccl-allreduce-use-fp16-mixed-precision-pytorch-forums]] — PyTorch maintainer ptrblck.** The other half of the bf16 question: *"The parameters are stored in `float32` using the automatic mixed-precision util. and thus also the gradients. NCCL thus communicates them in `float32`, too. If you are calling `.half()` on the model directly … NCCL should communicate in `float16`."* So under standard AMP the DDP buckets are **fp32**, and `bf16_compress_hook` is a genuine 2× cut in wire bytes — not a no-op.

## Evidence synthesis

### The definitional question is settled; the numeric question is not

`busbw = 1.6 × algbw` at N=5, by construction, and busbw is defined as the per-rank link rate. So the repo's single figure of 0.75 GiB/s = **0.805 GB/s** has three candidate readings, and they differ by 1.6× in opposite directions:

| Reading | Implied algbw | Implied busbw | Verdict against physics |
|---|---|---|---|
| **algbw** | 0.805 GB/s = 0.75 GiB/s | 1.288 GB/s = 1.20 GiB/s | busbw would be **77% of the 1.68 GB/s one-way link rate** on a fabric where every byte crosses PCIe twice |
| **busbw** | 0.503 GB/s = 0.469 GiB/s | 0.805 GB/s = 0.75 GiB/s | busbw is **75–96% of the realistic host-staged band** (0.84–1.07 GB/s) |
| **aggregate** | 0.161 GB/s | 0.258 GB/s | 4–6× below the community's own ~1.0 GB/s busbw at the same link width |

The ceiling itself has three defensible values, all arithmetic estimates (tier 3) built on community measurements (tier 2):

- **Strict serial host-staged model** (analyst b): busbw ≤ 1.68/2 = **0.84 GB/s = 0.78 GiB/s**; algbw ≤ 0.525 GB/s = **0.489 GiB/s**.
- **Empirically calibrated from the club's own x16 receipt**: NCCL achieves 3.69–4.24 GB/s busbw against a 6.63–6.68 GB/s one-way rate, i.e. **55.7–64.0% of one-way**. Scaling by the measured width ratio 1.68/6.63 = 0.253 gives x4 busbw of **0.935–1.074 GB/s** and algbw of **0.584–0.671 GB/s = 0.544–0.625 GiB/s**.
- **Absolute full-duplex bound**: busbw ≤ 1.68 GB/s, so algbw ≤ 1.05 GB/s = **0.978 GiB/s**. This is reachable only with perfect send/receive overlap and zero host-staging penalty.

The community's own Gen2 x4 figure — *"~1.0 GB/s measured bus bandwidth"* in [[lessonsmd]] and the SETTINGS.md mirror — lands inside the calibrated band (0.935–1.074 GB/s) and 19% above the strict serial model. Two independent routes therefore agree that **busbw at Gen2 x4, no P2P, is ~0.9–1.1 GB/s**, and algbw is ~0.55–0.63 GiB/s.

Against that, the repo's 0.805 GB/s is **75–86% of the busbw band** (a plausible, slightly conservative measurement) and **1.29–1.48× the algbw band** (not plausible). As an algbw it would require NCCL to hit 77% of the one-way link rate on a double-traversal fabric, where the best efficiency ever measured on this hardware class is 64%.

### Position A: 0.75 GiB/s is the real x4 host-staged ceiling — no P2P means every byte crosses PCIe twice, and NCCL ring/LL cannot exceed ~1 GB/s busbw at Gen2 x4

This is the strongest version of the "the number is fine" side, and it has real support. The mechanism is documented, not inferred: `NCCL_SHM_DISABLE` says SHM "is used between devices when peer-to-peer cannot happen, therefore, host memory is used"; the club's receipt shows `canAccessPeer` = 0 and `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` on all six pairs with "NCCL transport SHM/direct for every pair"; and SC'24 names the path "Trivial staging (GPU->Host->Host->GPU)". The arithmetic is clean: ring per-rank load 2(N−1)/N = 1.6 at N=5, doubled by host staging to 3.2× payload, divided by the measured 1.68 GB/s one-way rate = 0.525 GB/s. And the corpus's own community statement of the same fabric is *"~1.0 GB/s measured bus bandwidth"* — which is 1.33× the repo's 0.75 GiB/s, i.e. the same order and the same column.

What this position gets right: the ceiling is real, it is ~1 GB/s in busbw terms, and the repo's number is not absurd. What it cannot explain: **if 0.75 GiB/s is busbw, then the repo's arithmetic is wrong by 1.6×**, because it divides a payload by a busbw and calls the result a payload time. Position A as stated ("0.75 GiB/s is the real ceiling") is therefore *compatible* with the repo's number but *incompatible* with the repo's use of it.

### Position B: 3.69–4.24 GB/s is achievable — the club measures it on a 4-card PLX host; that host runs Gen2 x16, so the gap is link width, not NCCL

This side is correct on the facts and correct on the mechanism, and it is the only position backed by a *measured NCCL all-reduce on this exact silicon*. [[readmemd]] is a sanitized receipt: 3 × CMP 170HX, memory-unlocked to 65,536 MiB, **all three at Gen2 x16**, driver 615.71.09 with cmpunlocker @ 88e39ce plus `rebar-serialize.patch` and **explicitly no BAR1-P2P patches**, torch 2.13.0+cu130 / NCCL 2.29.7, bf16 sum, 5 warm-ups, median of 20, "bus bandwidth = algbw × 2(n−1)/n". The result — 3.69 GB/s same-switch, 3.85 cross-switch, 4.24 all-three at 256 MiB — is a genuine no-P2P host-staged NCCL measurement, which is exactly the configuration the user's box is in.

The gap is link width, and the corpus proves width scales linearly: bayley's x2 card "ran exactly half speed everywhere (0.84 GB/s same-switch, ~0.58 cross)", and Gen2 x4 → Gen2 x16 moves one-way from 1.68 to 6.63–6.67 GB/s, a 3.95× step. Applying that ratio to the club's busbw gives 0.93–1.07 GB/s at x4 — which is where the community's ~1.0 GB/s figure sits. So Position B is not an alternative to Position A; **it is the calibration that fixes Position A's magnitude.** What Position B cannot do is make 0.75 GiB/s an algbw: even at the club's best x16 efficiency (64%), x4 algbw tops out at 0.671 GB/s = 0.625 GiB/s.

### Position C: 0.75 GiB/s is not physically reachable as algbw on x4 host-staged; the figure must be busbw or aggregate, and the true overhead constant is ~5.4e5 tokens, not 3.6e5

This is the position the definitions support. Three independent lines converge:

1. **Definition.** busbw = 1.6 × algbw at N=5, and busbw is *defined* as the per-rank link rate. A user reading `all_reduce_perf` output quotes the busbw column, because that is the column nccl-tests exists to produce ("To provide a number which reflects how optimally the hardware is used, NCCL tests introduce the notion of 'Bus Bandwidth'").
2. **Physics.** As algbw, 0.805 GB/s implies busbw of 1.288 GB/s = 77% of the one-way link rate, on a fabric where each hop is two traversals and NCCL's ring is non-pipelined ([[demystifying-nccl-an-in-depth-analysis-of-gpu]]: "Ring AllReduce … is NON-pipelined (each GPU must finish every task in one iteration before starting the next), so it is latency-bound on host-staged links"). No measurement in the corpus shows better than 64%.
3. **Internal consistency with the repo's own premise.** The repo asserts *both* "no P2P" *and* 0.75 GiB/s. Those two claims cannot both be true if the number is algbw: 0.75 GiB/s algbw requires one traversal per hop, i.e. working P2P. Since "no P2P" is the better-corroborated claim (club: 0/6 pairs, GNS, SHM transport; bayley: P2P needs four extra driver patches plus `RMForceStaticBar1=1;RMPcieP2PType=1`), the reading of the bandwidth column is the more likely error.

The downstream consequence is quantitative and large. At 1M-token global batches, 1B trainable params, 6 FLOPs/param/token, 810 TFLOPS bf16 aggregate:

- **algbw reading:** t_comm = 2.0/0.75 = 2.67 s, compute 7.41 s → **36%** overhead, constant 3.60e5 tokens. (The repo's figure.)
- **busbw reading:** algbw = 0.469 GiB/s, t_comm = 4.27 s → **58%** overhead, constant **5.76e5** tokens. Using GB rather than GiB for the 2.0 GB payload gives 3.98 s and 54%, constant 5.37e5 — the range is **54–58%** and **5.4e5–5.8e5**.
- **aggregate reading:** 13.3 s → **180%** overhead. Ruled out: it implies a per-rank rate of 0.15 GiB/s, 6× below the community's own busbw measurement on the same link width.

Position C's weakness is that it cannot *prove* the column. The repo's claimed corroboration — that its model "reproduces the brief's measured behaviour (494M model at ~140k tok/s with 1M-token batches ≈ 18% overhead)" — is not independent: the 18% is inferred from the same 140k tok/s plus an assumed MFU, and at 494M params the payload is only 0.99–1.98 GB against a ~7 s compute step, so the datapoint cannot discriminate a 1.6× difference in all-reduce time with any confidence. It is also tier-1 self-measured and unverified by any third party.

### Position D: BAR1 P2P works at 1.68 GB/s each way at Gen2 x4 and lifts small-collective latency, but the patch reportedly hangs every NCCL collective

Both halves are true and they point in opposite directions. **It works:** bayley gets peer access and keyed 256 MiB copies passing on 12/12 directed pairs at 1.68 GB/s each way at Gen2 x4, and on 56/56 pairs across two PLX switches on one Xeon E5 v4 root complex, with ACS redirection disabled on every port. The club reproduces it on a 4-card x16 host at 5.79 GB/s with `RMForceStaticBar1=1;RMPcieP2PType=1` plus four patches from admunch888/cmpunlocker and a hand-programmed BAR layout adopted via kexec. **It is fragile:** [[p2pmd]] records an unverified single-builder report that the aikitoria fork's P2P half-works — peer *data movement* at 6.25 GB/s vs 5.70 GB/s staged, but peer *synchronisation* does not work at all, so "any NCCL collective HANGS at transport connect (no error, no timeout, both GPUs pinned at 100%)" and vLLM's custom all-reduce hangs identically. The offered mechanism is decisive: `kbusIsPcieBar1P2PMappingSupported_GH100` requires STATIC BAR1 spanning the whole framebuffer at a 512 MB-aligned offset, but BAR1 on the 170HX is 64 MB, so the check cannot pass.

Even granting working P2P, the bandwidth payoff is bounded and topology-dependent:

- **At x4, P2P cannot exceed the link.** 1.68 GB/s each way is exactly the one-way rate. P2P's benefit is removing the second traversal, not raising the rate — so the ceiling moves from ~0.84–1.07 GB/s busbw to at most 1.68 GB/s busbw, i.e. algbw from 0.54–0.62 to ≤0.98 GiB/s.
- **Measured NCCL gain at x16, 2 GPUs, same switch: 3.7 → 4.8 GB/s busbw (+30%).** Real, but not 2×.
- **Forced P2P across a host bridge: 4.3 → 2.3 GB/s busbw (−47%)** on the A6000 box. [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] §5 records the same shape: "P2P hurts four-card TP behind PLX on Broadwell — GLM-5.3-Flash TP4 decode −20 to −26% (1 user), −38 to −48% (8 users); step 19.4 → 26.1 ms."
- **P2P is slower than staging for bulk copies above 16 MiB** (5.79 vs 6.21–6.30 GB/s at 256 MiB).

So analyst a's framing — "P2P is a latency lever, not a bandwidth lever" — is right for raw copies and right for DDP (one large all-reduce per step, where the 43% latency advantage at 1 MiB is amortized into nothing), but wrong for 2-GPU NCCL collectives (+30%) and wrong in sign for 4-card TP behind PLX (−20 to −26%). The correct statement is narrower: **P2P is a conditional +30% for same-switch small-rank collectives, a −47% when forced across a host bridge, and a hang risk for every NCCL collective on a 64 MB-BAR1 card.**

### Does the 24-capacitor x16 mod multiply it? Yes — ~3.9–4.0× on busbw, ~2.9–3.3× on payload rate — and it is the only lever that changes the order of magnitude

The width chain is fully documented and independently reproduced. [[pcie-subsystemmd]]: stock trains Gen1 x4; the speed cap is OTP fuses plus a signed DevInit table (Gen2 reached in software 2026-07-24, not shipped; Gen3 blocked by `OPT_DISABLE_GEN3_SPEED` reading 1 on 4/4 cards); the width cap is **12 of 16 lanes shipping with their AC-coupling capacitors physically omitted — 24 missing 0402 parts**, defeated only by hand-soldering. The two are independent: "a Gen2-patched unmodded card is Gen2 x4; a fully modded unpatched card is Gen1 x16." [[physical-modsmd]] adds the four-state verification table (stock 2.5GT/s x4 → cap-mod-only 2.5GT/s x16 → unlocker-only 5GT/s x4 → both 5GT/s x16), the part (24 × 0402, 220 nF X7R ≥6.3 V, Taiyo Yuden MAASJ105SB7224KFCA01, value read off the A100 GA100-883 reference schematic P1001-B02 p.3), ~20 minutes per card, and a second independent four-card Gen2 x16 rig with zero PCIe errors.

The multiplier, measured: one-way 1.68 → 6.63–6.67 GB/s (**3.95×**); NCCL busbw 0.93–1.07 → 3.69–4.24 GB/s (**3.9–4.0×**); algbw at N=5 from 0.54–0.62 to **2.31–2.65 GB/s = 2.15–2.47 GiB/s**, which is **2.9–3.3× the repo's 0.75 GiB/s**. DDP overhead at 1M tokens falls from 54–58% to **~11% at theoretical peak** (constant ~1.1e5 tokens) and ~6% at 55% MFU.

Two caveats keep this from being a free win. First, **the host-staged path saturates**: the A6000 box at Gen3 x16 (roughly double Gen2 x16's line rate) delivers 4.3 GB/s busbw, only 1.01–1.16× the CMP box's Gen2 x16 figure. So the mod captures essentially all the available gain and a hypothetical Gen3 unlock would buy ~nothing — which is convenient, because Gen3 is fuse-blocked anyway. Second, **no 5-card x16 measurement exists anywhere in the corpus**; every x16 receipt is 2–4 cards behind PLX switches, and the club's own 4-card x16 host shows P2P *hurting* TP4. The 5-rank x16 number is an extrapolation, not a measurement.

### The bf16 sub-question: resolved, and the answer is the opposite of the hypothesis

The unresolved sub-question — "is BF16 all-reduce already 32-bit internally on GA100, making `bf16_compress_hook` useless?" — is **false on both counts**. [[reduce_kernelh]] shows NCCL's `FuncSum` for `__nv_bfloat16` under `#if __CUDA_ARCH__ >= 800` uses `__hadd`/`__hadd2` on `__nv_bfloat16`/`__nv_bfloat162`: the wire dtype is 2 bytes/element and the accumulation is native bf16. Jeaugey confirms the design intent: accumulating in fp32 "would mean communicating fp32 between GPUs, which would be the same as running the allreduce operation on fp32 to start with." There is no hidden widening.

The hook is therefore a genuine 2× cut in wire bytes — **but only if the buckets are fp32**. Under standard AMP they are: ptrblck, "The parameters are stored in `float32` … and thus also the gradients. NCCL thus communicates them in `float32`, too." The community measurement bears this out: bf16 compression moved 4-GPU DDP from 2.06× to 2.65× (124M) and 1.91× to 2.72× (355M) on a PCIe-only box. If instead the model is trained in pure bf16 (params bf16), the buckets are already bf16 and the hook is a no-op cast.

This has a direct consequence for the repo's arithmetic that the repo does not state: **its 2.0 GB payload for 1B trainable params is 2 bytes/param, i.e. it already assumes bf16 buckets.** Without the hook (fp32 buckets) the payload is 4.0 GB and the overhead doubles to ~108% at 1M tokens under the busbw reading. So the hook is not an optional optimisation to be tested later (repo open question #2); it is a **load-bearing premise of the feasibility claim**. And it carries a precision cost the repo does not budget: bf16 ring accumulation is the least numerically stable configuration NCCL has (Jeaugey: "The least stable algorithm is rings"), and on a single-node 5-GPU box the tree degenerates to a chain, so there is no stability escape hatch.

## Open questions

- **Which column?** Nothing short of the user re-running `all_reduce_perf` with both columns printed settles it. The repo should be asked directly, and the answer should be recorded with the message size (1 MB / 100 MB / 1 GB) and the rank count.
- **Is `canAccessPeer` actually 0 on the user's box?** The repo says "no P2P" but does not say how it was established. If it was established by `cudaDeviceCanAccessPeer` alone, that is exactly the test the club shows returns 0 even when `cuMemcpyPeer` succeeds via host staging — so "no P2P" may be a software-configuration statement rather than a hardware one. If BAR1 P2P *is* live on the box, the algbw reading becomes plausible and the whole locus flips.
- **No 5-rank measurement exists at any width.** Every receipt is 2–4 ranks. The Tartan warning that 5-GPU PCIe rings are pathological means the N=5 penalty may exceed the 1.6× ring-load factor, which would push the ceiling *below* the values derived here. A 5-rank run at x4 and at x16 is the single highest-value experiment in the program.
- **Host-staging saturation is inferred across hosts, not measured on one.** The 4.3 GB/s plateau (Gen3 x16 A6000) vs 3.69–4.24 GB/s (Gen2 x16 CMP) comparison spans different CPUs, root complexes and rank counts. A Gen1-vs-Gen2 × x4-vs-x16 2×2 on one rig — which [[performancemd]] explicitly names as the missing experiment — would settle both the width multiplier and the saturation point.
- **The 494M / 140k tok/s datapoint was not resolvable.** It is the only training measurement the repo cites, it is unverified, and at that model size the communication term is too small a fraction of the step to discriminate a 1.6× error. A 1B-trainable run at 1M-token batches with `torch.cuda.Event` timing around the all-reduce would be far more diagnostic.

## Sources

1. [[performancemd-2]] — nccl-tests `doc/PERFORMANCE.md`: algbw/busbw definitions, the `t = (S/B)·2(n−1)/n` derivation, "bus bandwidth should reflect the speed of the hardware bottleneck".
2. [[all_reducecu]] — nccl-tests `src/all_reduce.cu`: `*busBw = baseBw * 2(nranks−1)/nranks`.
3. [[whats-the-difference-in-allreduce-for-factor-between-the-nccl-and-nccl-test-issu]] — nccl-tests issue #274, Jeaugey: busbw is "theoretical, on a flat hardware topology, using point-to-point transfers".
4. [[pci-express-wikipedia]] — PCIe 2.0 = 500 MB/s per lane per direction after 8b/10b; lanes are dual-simplex, links are full-duplex.
5. [[exploring-gpu-to-gpu-communication]] — De Sensi et al., SC'24: "Trivial staging (GPU->Host->Host->GPU)" vs "Device-device copy (peer access)".
6. [[ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compre]] — community DDP on PCIe-only no-NVLink: 4.3 GB/s busbw at Gen3 x16, 2.9 GB/s algbw, forced P2P halves it, bf16 hook 2.06×→2.65×.
7. [[reduce_kernelh]] — NCCL `src/device/reduce_kernel.h`: native `__hadd`/`__hadd2` bf16 reduction under `__CUDA_ARCH__ >= 800`.
8. [[half-precision-reduction-accumulation-in-fp32-issue-1026-nvidianccl-github]] — NCCL issue #1026, Jeaugey: no fp32 accumulation in 16-bit collectives; ring is the least stable algorithm.
9. [[does-nccl-allreduce-use-fp16-mixed-precision-pytorch-forums]] — ptrblck: AMP keeps fp32 params and fp32 gradients, so NCCL communicates fp32.

Corpus sources relied on: [[readmemd]], [[club-170hxdocsbenchmarksmd-at-main-pixelmlclub-170hx-github]], [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]], [[pcie-subsystemmd]], [[physical-modsmd]], [[performancemd]], [[lessonsmd]], [[deepseek-v4-cmp170hxsettingsmd-at-317dac12bbd7f401f0538c76161bc7186086fb65-allov]], [[github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2]], [[p2pmd]], [[evaluating-modern-gpu-interconnect-pcie]], [[demystifying-nccl-an-in-depth-analysis-of-gpu]], [[environment-variables-nccl-2323-documentation]], [[pytorch-ddp-comm-hooks-default_hookspy-bf16fp16-compress-hooks]].

## Committed position

**The repo's 0.75 GiB/s is the nccl-tests busbw column, not algbw — and because the repo divides a payload by it, every DDP overhead figure in `research.md` is understated by exactly the ring-load factor 2(N−1)/N = 1.6.** The load-bearing reason is definitional plus physical at once: nccl-tests defines busbw as the per-rank link rate B in `t = (S/B)·2(n−1)/n`, so at N=5 busbw is 1.6× algbw by construction; read as busbw, 0.805 GB/s sits at 75–96% of the only physically available band for a Gen2 x4 host-staged ring (0.84–1.07 GB/s, derived two independent ways — the strict serial model and the club's own measured x16 efficiency of 55.7–64.0% scaled by the measured width ratio 0.253), whereas read as algbw it demands busbw = 1.288 GB/s = 77% of the one-way link rate on a fabric where every hop is two PCIe traversals and NCCL's ring is explicitly non-pipelined, against a best-ever measured efficiency of 64%. The decisive corroboration is internal: the repo asserts both "no P2P" and 0.75 GiB/s, and those two claims are mutually exclusive if the number is algbw, because algbw at that level requires one traversal per hop. Since "no P2P" is independently corroborated (0/6 pairs, `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`, `nvidia-smi topo -p2p r` = GNS, NCCL transport SHM, and P2P needing four extra driver patches plus `RMForceStaticBar1=1;RMPcieP2PType=1`), the bandwidth column is the more likely error. The correction is concrete: at 1M-token global batches with 1B trainable params, 6 FLOPs/param/token and 810 TFLOPS bf16, DDP overhead is **54–58%, not 36%**, and the overhead constant is **5.4e5–5.8e5 tokens, not 3.6e5** — and the 2.0 GB payload that produces it is only reachable with `bf16_compress_hook`, because AMP keeps gradients fp32 (4.0 GB) and NCCL does not widen bf16 internally (native `__hadd` on SM80, 2 bytes/element on the wire), so the hook is a real 2× and a load-bearing premise rather than an optional test. On the other two sub-questions: **BAR1 P2P is not a bandwidth fix** — at x4 it delivers exactly the 1.68 GB/s one-way link rate (so it removes a traversal, not a bottleneck), it buys only +30% on 2-GPU same-switch NCCL at x16 (3.7→4.8 GB/s busbw), it *loses* 47% when forced across a host bridge and 20–26% on 4-card TP behind PLX, it is slower than staging for bulk copies above 16 MiB, and the patch reportedly hangs every NCCL collective on a 64 MB-BAR1 card — so it must not be a prerequisite for the program; **the 24-capacitor x16 mod is the only order-of-magnitude lever**, multiplying busbw by 3.9–4.0× (0.93–1.07 → 3.69–4.24 GB/s) and the payload rate by 2.9–3.3× (to 2.15–2.47 GiB/s), which takes DDP overhead at 1M tokens from 54–58% to ~11% at theoretical peak, with the caveat that the host-staged path saturates near 4.3 GB/s busbw (a Gen3 x16 box measures 4.3 GB/s, only 1.01–1.16× the Gen2 x16 CMP figure) and that no 5-card x16 measurement exists.

- **Position:** 0.75 GiB/s is busbw, not algbw; the repo's 36% DDP overhead is really 54–58% at 1M-token global batches, and the overhead constant is ~5.4e5–5.8e5 tokens rather than 3.6e5.
- **Confidence:** medium (~72%). The definitional argument is airtight and the physical band is derived two independent ways, but the column is never named in the repo, the number sits almost equidistant in log-space from the community's own Gen2 x4 busbw (~1.0 GB/s) and its implied algbw (~0.58 GiB/s), and the repo's single 494M/140k tok/s datapoint weakly favours the algbw reading without being able to discriminate it.
- **Boundary conditions:** This holds for a 5-rank, Gen2 x4, host-staged (no BAR1 P2P), single-node NCCL ring at message sizes ≥100 MB, where the 1.68 GB/s one-way figure and the 55.7–64.0% efficiency calibration apply. Outside that scope the reading changes: on a cap-modded Gen2 x16 box the ceiling is 3.69–4.24 GB/s busbw and 0.75 GiB/s is trivially reachable as *either* column, so the ambiguity stops mattering; with working BAR1 P2P at x4 the algbw ceiling rises to ~0.98 GiB/s and the algbw reading becomes plausible; and at message sizes below ~16 MB the number is latency-dominated and neither column is a bandwidth measurement at all.
- **What would change this position:** (1) A 5-rank `all_reduce_perf` on the user's x4 box at 100 MB–1 GB printing both columns, with the user confirming which one they quoted — if algbw reads ≥0.75 GiB/s, this position is wrong and the fabric is better than the corpus supports. (2) Evidence that `canAccessPeer`/`cuMemcpyPeer` on the user's box succeeds peer-to-peer rather than staging through host memory, which would make the algbw reading physically available and collapse the argument. (3) A 5-rank x4 measurement showing NCCL efficiency above 70% of the one-way link rate on a host-staged ring, which would invalidate the 55.7–64.0% calibration that supplies the upper end of the band.
- **Evidence weight:** 1 primary vendor definition (nccl-tests PERFORMANCE.md + source) and 1 NVIDIA maintainer statement establish the busbw/algbw relation and the flat-topology assumption; 3 independent community measurements on this silicon class (club x16 receipt 3.69–4.24 GB/s, bayley x4 P2P 1.68 GB/s each way, club LESSONS/SETTINGS ~1.0 GB/s busbw at x4) supply the band; 1 community DDP measurement on a PCIe-only no-NVLink box (4.3 GB/s busbw at Gen3 x16, forced-P2P −47%, bf16 hook +29%) supplies the efficiency calibration and the hook's effectiveness; 1 NCCL source file plus 1 maintainer statement settle the bf16 question against the hypothesis; 1 primary paper (Tartan/TPDS) warns that 5-GPU PCIe rings are pathological, which pushes the ceiling down rather than up. Against the position: 1 unverified user self-report (the number itself) and 1 non-independent corroboration claim (the 494M/18% inference). No measurement exists on the user's actual 5-card box at any width, so every figure here is tier 2 (community) or tier 3 (arithmetic), never tier 1.
