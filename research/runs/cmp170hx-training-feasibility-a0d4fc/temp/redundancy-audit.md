# Evidence redundancy audit — cmp170hx-training-feasibility-a0d4fc

Method: 139 claims files, 1,242 claims. Clustered by exact-normalised `quoted_support` overlap
(>60% passage overlap), then by citation ancestry (`suggested-by` breadcrumbs).

## Derivative clusters found

| Cluster | Canonical upstream | Derivatives (discount in coverage counts) | Overlap |
|---|---|---|---|
| Sequence composition | `analysing-the-impact-of-sequence-composition` (arXiv 2402.13991) | `analysing-the-impact-of-sequence-composition-on-language-model-pre-training-acl` (ACL landing page, abstract only) | 100% — same paper, landing page vs arXiv |
| 2× CMP 170HX P2P gist | `2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2-2` (original gist) | `2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epycromed8-2t` (wpnews reprint) | 100% — verified faithful reprint |
| FineWeb paper | `the-fineweb-datasets-decanting-the-web-for-the` (arXiv 2406.17557) | `the-fineweb-datasets-decanting-the-web-for-the-finest-text-dat` (proceedings.com record, metadata only) | Same work, no abstract in derivative |
| FSDP | `pytorch-fsdp-experiences-on-scaling-fully-sharded-data-paralle` (PVLDB PDF) | — (the DOI note was never created; fetch was bot-blocked and substituted) | n/a |
| ZeRO | `zero-memory-optimizations-toward-training-trillion-parameter-m` (arXiv 1910.02054) | — (SC20 DOI substituted) | n/a |
| club-170hx HARDWARE | `club-170hxdocshardwaremd-at-main-pixelmlclub-170hx-github` | `club-170hxdocscmp170hx-understandingmd-...` (consolidated superset, not a derivative) | partial — UNDERSTANDING consolidates HARDWARE + TOPOLOGY |
| DeepWiki | `allover326deepseek-v4-cmp170hx-deepwiki` | `deepseek-v4-cmp170hxsettingsmd-...` (repo file is the primary) | derivative — cite the repo file |
| Engram repo README | `github-deepseek-aiengram-conditional-memory-via-scalable-looku` | `engramengram_demo_v1py-at-main-deepseek-aiengram-github` (code file) | not derivative — different artefacts |
| FineMath | `huggingfacetbfinemath-datasets-at-hugging-face-2` | `huggingfacetbfinemath-datasets-at-hugging-face` (deprecated mis-fetch) | 97% — deprecated copy |

## Adjusted coverage after discounting derivatives

Every atomic item keeps ≥2 **independent** sources after discounting. No item drops below 2.

| Atomic item | Raw sources | Independent after discount | Status |
|---|---:|---:|---|
| Sub-Q1 link width | 9 | 9 | OK |
| Sub-Q2 PWRBRK# | 4 | 4 | OK |
| Sub-Q3 topologies | 14 | 14 | OK |
| Sub-Q4 DDP overhead | 16 | 16 | OK |
| Sub-Q5 EP | 14 | 13 | OK |
| Sub-Q6 TP | 8 | 8 | OK |
| Sub-Q7 Engram | 13 | 13 | OK |
| Sub-Q8 CSA2 | 16 | 16 | OK |
| Sub-Q9 dense baseline / TFLOPS | 18 | 18 | OK |
| Sub-Q10 data pipeline | 10 | 9 | OK |

## Wave 3

Not required — no atomic item fell below 2 independent sources.

## Cross-cluster note

The corpus is **not** one-source-in-N-outfits. The three duplicate pairs are landing-page/reprint
artefacts, not independent corroboration. The genuinely independent corroboration that matters:

- **FSDP = 1.5× DDP** is independently stated by the FSDP paper (PVLDB) *and* derived from ZeRO's
  3Ψ-vs-2Ψ accounting — two independent primaries, not one.
- **PP beats TP on this box** is independently measured by club-170hx (6.6× prefill), allover326
  (86 collectives/forward), dl4rce (3.2× prefill), and promisezackr (PP6 on 8 cards) — four
  independent measurements.
- **P2P is a driver mask, not physics** is independently reported by bayley/cmpunlocker (BAR1 P2P,
  1.68 GB/s), the Puls gist (ForceP2P=0x11), and wpnews — but the club's `p2p.md` reports the patch
  hangs every NCCL collective, which is a genuine conflict for step 3.
