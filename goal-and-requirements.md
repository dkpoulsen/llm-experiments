---
id: ai-box-training-experiments
type: goal-and-requirements
status: active
title: AI Box Training Experiments
tags: [llm-training, hardware]
---

# AI Box Training Experiments

Turning a 320 GB, ~810 TFLOPS-bf16 rig into a deliberate experiment program.

## Overview

A working folder for planning, running, and recording LLM training experiments on a single
5× CMP 170HX box. The rig's shape is unusual — A100-class matmul rates and 320 GB of VRAM, but a
PCIe Gen2 x4 fabric with no NVLink or P2P — so what works here is not what works on a normal
multi-GPU node. This project exists to choose experiments against the hardware's real limits and to
keep their results comparable.

## Problem

The box can pretrain up to ~1B params and fully fine-tune up to ~7B, but its interconnect caps
all-reduce at ~0.75 GiB/s, so small-batch DDP stalls and FSDP costs ~1.5× more communication than
plain DDP. Useful throughput comes from large global batches (~1M tokens) and bf16 compression, not
from many small parallel jobs. On top of that, the GPUs are currently occupied by serving work
(TP4 vLLM on GPUs 0–3, a smaller model on GPU4), so training has to be scheduled around it. Without
a written plan, experiments get picked by guesswork and their numbers aren't comparable run to run.

## Capabilities

- **Write experiment plans.** Each plan states the hypothesis, the architecture config (total vs
  active params, expert count, Engram table size, attention variant), the parallelism and batch
  plan, the expected throughput and loss, and which hardware limit it probes.
- **Record comparable results.** One log entry per run — tokens/s, TFLOPS/GPU, VRAM high-water,
  loss at fixed token counts, wall time — so runs stay comparable across architectures.
- **Check plans against the hardware first.** Every plan opens with the pre-flight: negotiated PCIe
  link width and speed per card (`lspci -vv`), and the `PWRBRK#` power-brake check
  (`nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"`). Both silently invalidate results.

The architecture axes under test: a matched dense baseline (~500M active) as the control, then
sparse MoE up to 7B total / ~1B active, Engram conditional memory (learned tables vs frozen
Memory-Grafting tables), and CSA-style compressed sparse attention at 32k context.

The budget is trainable params, not total: ~1.5B trainable with the rest frozen keeps DDP viable on
a 64 GB card. Above that the model needs FSDP and pays roughly twice the link overhead.

## Tech Notes

Verified hardware:

| | |
|---|---|
| GPUs | 5× NVIDIA CMP 170HX (GA100, 64 GB HBM2e), 250 W cap each |
| Compute/GPU | bf16 ≈162 TFLOPS, fp16 ≈178, TF32 ≈83, fp32 ≈13 (measured) |
| Interconnect | PCIe Gen2 x4, no NVLink, no P2P; all traffic crosses the host bridge |
| CPU / RAM | Threadripper PRO 3975WX, 32c/64t, 251 GB DDR4 |
| Storage | 3.7 TB NVMe, ~2.2 TB free |

- PyTorch + NCCL; bf16 autocast everywhere (fp32 is only ~13 TFLOPS; GA100 has no FP8 tensor cores).
- Parallelism: DDP or FSDP only. EP and TP cost ~70× the compute time on this fabric (two
  all-to-alls per MoE layer); FSDP costs ~2× DDP.
- DDP overhead is set by batch size alone — at ~55% achieved MFU, ~20% at 1M-token global batches
  (~36% at theoretical peak).
- Pre-flight: verify negotiated PCIe width/speed per card, and that no card is power-braked
  (`PWRBRK#` on edge pin B30 pins a card at ~88 W / 39.3 TFLOPS and looks like a slow card).
- Batch shape: ~1M-token global batches plus `bf16_compress_hook` to hide the slow fabric.
- Proven baseline: 494M Qwen3-architecture model, 9B tokens FineWeb-Edu + SFT, ~21 h at ~140k tok/s
  on 5 GPUs (91–100 TFLOPS/GPU).
- Serving: vLLM occupies GPUs 0–4 today; training runs need a freed GPU or a schedule.
- Architecture references: DeepSeek Engram (hashed n-gram conditional memory, a sparsity axis
  orthogonal to MoE) and DeepSeek-V4.1-Flash CSA2 (compressed sparse attention).

## Alternatives Considered

| Project | Gap |
|---|---|
| [databricks/megablocks](https://github.com/databricks/megablocks) | Dropless-MoE layers plus pretraining scripts, but built for fast-interconnect clusters; no Engram or sparse attention. |
| [wolfecameron/nanoMoE](https://github.com/wolfecameron/nanoMoE) | Small-scale MoE pretraining with the stability tricks, but single-GPU scale and no fabric-aware batching. |
| [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) | Official Engram module, but demonstration code that mocks attention/MoE — not a training harness. |
| [QingGo/engram-peft](https://github.com/QingGo/engram-peft) | Unofficial PEFT-style Engram injection, but adapter/fine-tune oriented rather than from-scratch pretraining. |
| [PixelML/club-170hx](https://github.com/PixelML/club-170hx) | Community recipes and benchmarks for the CMP 170HX, but inference-focused — no training experiments. |
| [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) | Deep hardware reference on unlock, PCIe Gen2 and multi-GPU, but no training code. |
