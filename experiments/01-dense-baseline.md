# Experiment 01 — Pre-flight + matched dense baseline

**Status:** planned, not run
**Axis:** control (everything else is measured against this)
**Tokens:** 9B
**Context:** 32k

## Hypothesis

A ~500M dense model trained at 32k context on this box will land within ~20% of the throughput the
DDP overhead model predicts (~36% all-reduce overhead at 1M-token global batches), and will produce a
loss curve clean enough to serve as the control for the MoE, Engram, and CSA2 experiments.

If it does not, the overhead model in `research.md` is wrong and every later estimate needs revising.

## Step 0 — Pre-flight (do this before spending GPU time)

Both checks are cheap and both silently invalidate results.

1. **Negotiated link width and speed, per card.**
   ```
   for d in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
     lspci -vv -s "${d#00000000:}" | grep -E 'LnkCap|LnkSta'
   done
   ```
   Record `LnkSta` speed and width for all 5 cards. They must agree. If they are Gen2 x16 rather
   than x4, the fabric is ~4× faster than assumed and the overhead model drops accordingly — record
   which it is, because every later plan cites it.

2. **Power brake.**
   ```
   nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
   ```
   `Active` on a card that is not thermally or power limited means the board is asserting `PWRBRK#`
   (edge pin B30). A braked card runs at ~88 W / 39.3 TFLOPS fp16 instead of ~155 TFLOPS and looks
   exactly like a slow card. Fix before proceeding: Kapton tape over B30, or a riser that does not
   route B30.

3. **All-reduce benchmark.** Measure effective all-reduce bandwidth across 5 ranks at 1 MB, 100 MB,
   and 1 GB payloads, with and without `bf16_compress_hook`. This is the number the overhead model
   uses (currently 0.75 GiB/s, measured by the user).

4. **Single-GPU smoke test (100 steps).** Before evicting serving from all 5 GPUs, run a 100-step
   smoke test on 1 GPU at 32k context: verify dataloader throughput, activation checkpointing, and
   VRAM high-water mark (~31 GB expected). Only schedule the full 5-GPU run after the 1-GPU smoke
   test completes without stalls or OOM.

## Config

| | |
|---|---|
| Architecture | Qwen3-style dense decoder, ~500M params (matches the proven 494M run) |
| Active params | ~500M (the MoE experiments must match this) |
| Context | 32k |
| Tokens | 9B |
| Data | `data-pipeline.md` stream: 75% FineWeb-Edu / 15% Stack-Edu / 10% FineMath4+, Qwen3 tokenizer |
| Precision | bf16 autocast, fp32 master weights |
| Optimizer | AdamW, cosine schedule, warmup ~1% of steps |
| Checkpointing | full activation checkpointing (32k activations do not fit otherwise) |

## Parallelism and batch plan

- **Topology:** DDP across 5 ranks. No FSDP, no EP, no TP — see `research.md` §2.
- **Global batch:** 1M tokens ≈ 30 sequences of 32,768 tokens (983,040 tokens).
- **Per-GPU micro-batch:** 1 sequence of 32k tokens.
- **Gradient accumulation:** 6 micro-steps per GPU (6 steps × 5 GPUs × 32,768 tokens = 983,040 tokens).
- **Critical DDP sync rule:** wrap the first 5 micro-steps in `model.no_sync()`, allowing PyTorch
  DDP to all-reduce gradients *only* on the 6th micro-step. Without `no_sync()`, DDP all-reduces on
  every micro-step, multiplying communication overhead by 6× on the PCIe link.
- **Grad compression:** `bf16_compress_hook` on.

## Expected throughput and loss

- **Throughput:** 80–120k tok/s (estimate). The proven baseline run reached ~140k tok/s (context length
  unconfirmed, inferred ~4k); 32k attention costs more. At 100k tok/s, 9B tokens ≈ 25 h.
- **Loss:** record training loss and validation loss (overall and per domain: web, code, math)
  at 1B / 3B / 6B / 9B tokens. No target is set yet — the point of the run is to establish the number,
  not to hit one.

## Hardware limits probed

- Negotiated PCIe width/speed (pre-flight)
- `PWRBRK#` power brake (pre-flight)
- DDP all-reduce overhead at 1M-token batches — validates the `3.6e5 / tokens` model
- Dataloader throughput at 32k — CPU-side, not the link (see `data-pipeline.md` bandwidth budget)
- VRAM high-water with 32k activations + full checkpointing

## Acceptance criteria

- All 5 cards report the same negotiated link width and speed.
- No card is power-braked.
- The run completes 9B tokens without OOM or dataloader stalls.
- Throughput is within 20% of the overhead model's prediction; if not, record why.
- Loss at 1B / 3B / 6B / 9B tokens (training and domain validation) is recorded.

## Results

Not run yet. Results go to `results.md` in this shape:

| Run | Date | Tokens | Context | Topology | tok/s | TFLOPS/GPU | VRAM high-water | Val Loss (Overall / Web / Code / Math) @9B | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## What this unblocks

Every later experiment compares against this run's loss curve at matched tokens. The MoE, Engram, and
CSA2 plans must all state their active-param count and confirm it matches ~500M, or the comparison is
confounded.
