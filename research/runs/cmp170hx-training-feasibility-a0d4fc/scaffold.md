# Scaffold — cmp170hx-training-feasibility-a0d4fc

## User Prompt (VERBATIM — gospel)

> the documentaion in the repo

Clarified intent (user-selected): **Deep-research its claims** — verify the repo's documentation
against real sources, not just audit the prose.

## Run config

| | |
|---|---|
| vault_tag | `cmp170hx-training-feasibility-a0d4fc` |
| query_file_path | `research/runs/cmp170hx-training-feasibility-a0d4fc/query.md` |
| profile / gear | `full` |
| modality | `compare` |
| budget | none set |

## Modality classification rationale

`compare`: the query asks which of four architecture axes (dense control, MoE, Engram, CSA2) and
which topology (DDP vs FSDP) are worth running on this specific box. That needs proportionate
per-axis depth plus a committed recommendation, not enumerative coverage (`collect`) and not a
single defended thesis (`synthesize`).

## Tier rationale

`full` + `argumentative`. The query is not a lookup: it asks whether a set of interlocking claims
about communication cost, memory budget and architecture choice survive scrutiny, and the answer
depends on evidence that is partly measured, partly community-reported and partly arithmetic. That
needs the contradiction graph, loci analysis, depth investigation and adversarial critics — a light
run would produce a summary of the docs rather than a verdict on them. `argumentative` because the
report must commit to which experiments are worth running, not just describe the axes.

## Source corpus (seed documents)

The repo's own documentation is the seed corpus — its claims are what the run verifies:

- `goal-and-requirements.md` — hardware table, capabilities, alternatives considered
- `research.md` — headline findings, overhead estimates, Engram/CSA2 analysis, memory math
- `data-pipeline.md` — 9B-token mix, packing, determinism, acceptance criteria
- `experiments/01-dense-baseline.md` — pre-flight checks, DDP batch plan, expected throughput

Claims in these docs are labelled **measured** / **reported** / **estimate**. The report must
separate those three tiers explicitly and say which estimates survive scrutiny.

## Wrapper requirements

None. No `research/prompt.txt`, no `research/wrapper_contract.json`. Default save path:
`research/notes/final_report_cmp170hx-training-feasibility-a0d4fc.md`.

## Known open questions from the docs (candidate loci)

1. Actual negotiated PCIe link width per card — Gen2 x4 vs x16 changes every communication number ~4×.
2. Does 0.75 GiB/s all-reduce hold with `bf16_compress_hook`, and at what message size?
3. Real cost of a 1B-trainable DDP step at 1M and 2M tokens — validates the `3.6e5 / tokens` model.
4. How large can a frozen Engram table get before host-side hashing lookup latency binds?
5. Does CSA2 at 32k beat full attention on this box at all, given the indexer overhead?
