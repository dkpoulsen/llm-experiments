# Training Data Pipeline

**Status:** planned, not built
**Scope:** shared by every experiment — one fixed token stream, so runs stay comparable

## Purpose

One deterministic 9B-token stream that every experiment consumes in the same order. Comparability
depends on this: if the MoE run sees different tokens than the baseline, the comparison is
confounded and the whole experiment program is worthless.

## Dataset mix (proposed — confirm or adjust)

| Source | Share | Tokens | Available | Notes |
|---|---:|---:|---:|---|
| FineWeb-Edu | 75% | 6.75B | 1.3T | Educational web text, the proven baseline's source |
| Stack-Edu | 15% | 1.35B | ~125B | Educational code, 15 languages |
| FineMath4+ | 10% | 0.9B | 10B | Math with reasoning, scores 4–5 |

No source is repeated — every share is well under its available size. The shape follows SmolLM2's
stage-1 mix (90% web, 10% code, no math) with a small math slice added, since math and code
capabilities barely emerge at 9B tokens on a 500M model.

## Tokenizer

Same as the proven run (Qwen3 tokenizer). Fixed — changing it invalidates every comparison.

## Pipeline

1. **Download** the three sources from Hugging Face.
2. **Tokenize** to `uint32` arrays (Qwen's vocabulary size is ~152k, which overflows `uint16`).
3. **Interleave** documents across the three sources by sampling with weights 0.75 (FineWeb-Edu),
   0.15 (Stack-Edu), and 0.10 (FineMath4+). Interleaving prevents sequential domain clumps and
   catastrophic forgetting.
4. **Pack** sampled documents greedily into exactly 32,768-token sequences, no padding.
5. **Record document boundaries** per sequence in standard `cu_seqlens` format (`int32` cumulative
   offsets), matching FlashAttention / SDPA varlen kernels so document-boundary masking can be enabled
   without extra collation glue.
6. **Shard** into ~1 GB binary files for zero-copy memory mapping (`np.memmap`).
7. **Serve** via memmap: each of the 5 DDP ranks reads a disjoint, deterministic slice of the
   global sequence order.

## Packing (proposed — confirm)

Packed sequences, no padding, loss on all tokens. Document-boundary masking is **off** by default,
matching standard packed-LM training. The `cu_seqlens` offsets are stored alongside each shard so a
later experiment can turn masking on without re-tokenizing.

Why packed: at 32k context, document-level batching would pad most sequences heavily (FineWeb-Edu
documents average far below 32k), wasting a large fraction of the compute budget on padding.

## Storage

9B tokens × 4 B (`uint32`) = **36 GB**, plus 100M validation tokens = 0.4 GB. Well within the
2.2 TB free NVMe space.

## Bandwidth budget — dataloading is not the bottleneck

| Path | Volume | Time at Gen2 x4 (~2 GB/s) |
|---|---:|---:|
| Training read (100k tok/s × 4 B) | 400 KB/s per GPU, 2 MB/s total | negligible |
| Host → device per step (1M tokens × 4 B) | 4 MB | ~2 ms |
| Checkpoint write (500M params × 2 B) | 1 GB | ~0.5 s |

The real risk is **CPU-side preprocessing** — tokenizing 9B tokens, shuffling, and packing them —
not the PCIe link. That is a one-time cost and can run on the 32c/64t Threadripper while the GPUs
are still serving.

## Determinism

Fixed seed, fixed shard order, fixed global order. Every experiment consumes the same stream in the
same order. Record the manifest (source, share, shard hashes) so a later run can prove it saw the
same data.

## Validation

100M held-out tokens disjoint from the training mix, stratified by source domain:
- 75M tokens FineWeb-Edu
- 15M tokens Stack-Edu
- 10M tokens FineMath4+

Report both overall validation loss and per-domain validation loss at 1B / 3B / 6B / 9B tokens
alongside training loss, giving clear visibility into code and math progress alongside web text.

## Acceptance criteria

- 9B training tokens prepared; realized mix within ±1% of target.
- Validation set provably disjoint from training (13-gram overlap check).
- Dataloader sustains ≥150k tok/s per GPU — headroom over the 100k target.
- Re-running the pipeline reproduces the same token order (manifest hashes match).

## Open questions

1. Is the 75/15/10 mix right, or should code/math be smaller at this token budget?
2. Packed vs document-level — packed proposed, but it changes what the loss curve means.
3. Does the tokenizer's vocabulary cover Stack-Edu's languages without excessive fragmentation?
