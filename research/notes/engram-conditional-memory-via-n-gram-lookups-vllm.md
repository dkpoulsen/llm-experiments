---
title: 'Engram: conditional memory via n-gram lookups - vLLM'
id: engram-conditional-memory-via-n-gram-lookups-vllm
tags:
- cmp170hx-training-feasibility-a0d4fc
- engram
- vllm
- runtime
- cpu-offload
- serving
created: '2026-10-27T10:13:18.993190Z'
updated: '2026-10-27T19:00:31.178902Z'
source: https://docs.vllm.ai/en/v0.31.0/features/engram/
source_domain: docs.vllm.ai
fetched_at: '2026-10-27T10:13:18.355595Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: 'vLLM v0.31.0 Engram feature docs - the ground-truth runtime contract for
  serving Engram models. Engram layers are declared by the checkpoint (engram_layer_ids
  for DeepSeek-V4.1-Flash, ple_layer_ids for Qwen4Exp), not chosen at serve time;
  table sizes, n-gram size and head count are fixed by the checkpoint and not user-configurable.
  Hashing, lookup and gating run as fused Triton kernels over FP8 rows with ue8m0
  block scales. Sizing fact that matters for a 64 GB-card box: DeepSeek-V4.1-Flash
  carries two Engram layers of roughly 384M table rows each with 256-byte FP8 entries
  plus block scales - about 200 GB of table weights alone. cpu_offload defaults to
  true (VLLM_PLE_CPU_OFFLOAD=true): tables live in pinned host memory and are served
  directly from the Triton kernel through unified virtual addressing, prefetched on
  a side CUDA stream to overlap with preceding layers; freed GPU memory goes to KV
  cache. CPU offload requires UVA and vLLM fails fast without it. Topologies: default
  embedding_across_dp=false keeps one TP-sharded table copy per DP replica (no cross-replica
  communication, one host copy per replica); embedding_across_dp=true shards hash
  heads across all TP x DP ranks into a single table copy, gathering hash ids of every
  co-located DP replica per step and exchanging rows back, trading per-step DP collectives
  for a much smaller table footprint. dp_shared_memory stores one copy of each TP
  shard in /dev/shm shared by co-located replicas via cudaHostRegister, dropping host
  memory from one copy per replica to one per node. Limitations: CUDA-alike only;
  DeepSeek V4.1 Engram does not support DBO or microbatching (must disable --enable-dbo
  and set --ubatch-size 0); embedding_across_dp unsupported with elastic expert parallelism.'
utility_score: 12.0
---

# Engram: conditional memory via n-gram lookups[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#engram-conditional-memory-via-n-gram-lookups "Permanent link")
Engram is a _conditional memory_ module attached to specific layers of a transformer backbone. At each position, the preceding few tokens are hashed into large static embedding tables, and the retrieved rows are fused into the hidden state, gated by how well they match it. Where Mixture-of-Experts scales capacity through conditional computation, Engram scales it through conditional memory: parameters that are never computed over, only looked up, so capacity grows at O(1) per-token cost. The mechanism and its design rationale are described in [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372).
vLLM enables Engram automatically whenever the checkpoint declares Engram layers, so basic serving needs no extra flags. This page explains what Engram does and how to control its memory footprint and multi-GPU topology.
## Supported models[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#supported-models "Permanent link")
Engram layers are declared by the checkpoint, not chosen at serve time:  
| Architecture  | HF config field  | Notes  |  
| --- | --- | --- |  
| `engram_layer_ids`  | e.g. `deepseek-ai/DeepSeek-V4.1-Flash`  |  
|  `Qwen4ExpForCausalLM` / [`Qwen4ExpForConditionalGeneration`](https://docs.vllm.ai/en/v0.31.0/api/vllm/models/qwen4_exp/amd/model/#vllm.models.qwen4_exp.amd.model.Qwen4ExpForConditionalGeneration "            Qwen4ExpForConditionalGeneration")  | `ple_layer_ids`  | same mechanism, called PLE  |  
Passing `--engram-config` for a model without n-gram embedding layers fails at startup, as does running on a non-CUDA-alike platform.
## How it works[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#how-it-works "Permanent link")
Engram processes each position in two phases: _retrieval_ and _fusion_.
**Retrieval** maps the local token context to static table rows:
  1. **Tokenizer compression.** At startup, vLLM builds a many-to-one map from the tokenizer that collapses tokens which normalize alike (NFKC, accent stripping, lowercasing, whitespace collapsing), so `" The"`, `"the"` and `"THE"` hash identically. This concentrates semantic density into a much smaller id space; its size must match `engram_compressed_vocab_size` from the checkpoint, or startup fails — every hash multiplier derives from it.
  2. **Multi-head hashing.** For each n-gram order `n` (up to `engram_max_ngram_size - 1`), several hash heads each apply an independent multiplicative-XOR hash over the compressed suffix n-gram. Every (n-gram order, head) pair indexes its own prime-sized bucket range in the layer's table — disjoint across pairs, so collisions cannot collide twice — and the retrieved rows are concatenated into one memory vector per position.


**Fusion** turns the retrieved rows into a context-aware contribution to the hidden state. The rows are context-independent priors and may be noisy (hash collisions, polysemy), so they are gated: a `wkv` projection produces one key per hyper-connection branch plus a shared value, and each branch computes a sigmoid gate over the normalized dot product between its hidden state — which has already aggregated global context through preceding attention — and its key. If the retrieved memory contradicts the current context, the gate tends toward zero and suppresses it. The gated value is added residually into the branch's stream (`hidden + gate * value`); positions that take no part in an n-gram (e.g. image spans) are masked so they pass through untouched.
Engram sits only on the layers listed in the checkpoint. Placement is a deliberate hardware-algorithm trade-off: early enough that the backbone is relieved from reconstructing static patterns in its first layers, deep enough that vLLM can hide table lookups behind the compute of the preceding layers (see below).
In vLLM, hashing, lookup, and gating run as fused Triton kernels over FP8 rows with ue8m0 block scales. The layer layout — table sizes, n-gram size, head count — is fixed by the checkpoint and is not user-configurable. Only the storage and sharding described below are configurable, via `--engram-config`, which takes a JSON object and can also be set field by field as `--engram-config.<field>` ([CLI reference](https://docs.vllm.ai/en/v0.31.0/cli/serve/#-engram-config)).
## CPU offload[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#cpu-offload "Permanent link")
Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) carries two Engram layers of roughly 384M table rows each, with 256-byte FP8 entries plus their ue8m0 block scales — about 200 GB of table weights alone. Because lookups are deterministic — the indices are known from the token ids alone, before the forward pass — the tables need not sit in GPU memory at all. By default vLLM stores them in **pinned host memory** and serves lookups directly from the Triton kernel through unified virtual addressing (UVA), prefetching on a side CUDA stream so transfers overlap with the compute of preceding layers. The freed GPU memory goes to the KV cache.
`cpu_offload` defaults to `true` (on): it follows the `VLLM_PLE_CPU_OFFLOAD` environment variable, which also defaults to `true`. An explicit `--engram-config` value takes precedence over the environment variable.
To keep the tables resident on GPU instead (e.g. when host memory is scarce):

```
vllmservedeepseek-ai/DeepSeek-V4.1-Flash\
--engram-config.cpu_offloadfalse

```

or equivalently `VLLM_PLE_CPU_OFFLOAD=0`. CPU offload requires a GPU with UVA support; vLLM fails fast when it is unavailable.
## Data-parallel topologies (DeepSeek V4.1)[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#data-parallel-topologies-deepseek-v41 "Permanent link")
By default (`embedding_across_dp: false`), each DP replica keeps its own TP-sharded copy of the tables — no cross-replica communication, but one host copy per replica.
With `embedding_across_dp: true`, the hash heads are sharded across all TP × DP ranks into a single table copy. Each step gathers the hash ids of every co-located DP replica, each rank looks up the heads it owns, and the rows are exchanged back. This trades per-step DP collectives for a much smaller table footprint:

```
vllmservedeepseek-ai/DeepSeek-V4.1-Flash\
--tensor-parallel-size2--data-parallel-size4\
--engram-config'{"embedding_across_dp": true}'

```

`embedding_across_dp` is not supported with elastic expert parallelism yet. Qwen4Exp PLE tables are ETP-sharded instead and honor `cpu_offload` only.
## Sharing host tables across DP replicas[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#sharing-host-tables-across-dp-replicas "Permanent link")
When the tables are CPU-offloaded and DP replicas are co-located on one node, `dp_shared_memory` stores **one copy of each TP shard in`/dev/shm`** , shared by all co-located replicas through `cudaHostRegister`. Each replica then prefetches only its own tokens — no per-step Engram DP collectives — and host memory drops from one table copy per replica to one per node.
`dp_shared_memory` defaults to enabled whenever the other settings allow it, and falls back to per-replica (or DP-sharded) tables with a warning when:
  * the DP replicas are not co-located on a single node,
  * `/dev/shm` cannot hold the full tables — in containers, raise the limit with `--shm-size` or `--ipc=host`,
  * expert parallelism is elastic, which is unsupported.


Requirements: `cpu_offload` enabled, `--data-parallel-size > 1`, and a shared IPC namespace across the co-located replicas.
## Limitations[¶](https://docs.vllm.ai/en/v0.31.0/features/engram/#limitations "Permanent link")
  * CUDA-alike platforms only.
  * DeepSeek V4.1 Engram does not support DBO or microbatching; disable `--enable-dbo` and set `--ubatch-size 0`.
  * `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is unsupported with elastic expert parallelism.


Back to top 
