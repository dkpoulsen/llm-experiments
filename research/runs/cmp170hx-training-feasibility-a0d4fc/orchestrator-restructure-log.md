# Orchestrator restructure log — step 14.4

Three changes applied by the orchestrator rather than the patcher, because each required a move or rename the patcher's Read+Edit lock could not safely perform.

## 1. Heading forms restored to the decomposition's literal strings

The patcher left the seven H2s in their short forms (`## Hardware Reality`, `## Training Topologies`, …). The ship gate reads `required_section_headings` from `prompt-decomposition.json`, which stores the numbered, subtitled forms, so `run verify` reported all seven missing. Renamed in place, body untouched:

| Before | After |
|---|---|
| `## Hardware Reality` | `## 1. Hardware Reality: Link Width, Power Brake, and What the Numbers Assume` |
| `## Training Topologies` | `## 2. Training Topologies on a No-P2P PCIe Fabric` |
| `## DDP Overhead Model` | `## 3. The DDP Overhead Model at 1M-Token Batches` |
| `## MoE & EP` | `## 4. MoE and Expert Parallelism: Ruled Out or Misjudged?` |
| `## Engram & Frozen Memory` | `## 5. Engram and Frozen Memory: Making 7B Fit on 64 GB` |
| `## CSA2 at 32k` | `## 6. CSA2 Sparse Attention at 32k Context` |
| `## Verdict` | `## 7. Verdict: Which Experiments Are Worth Running` |

## 2. `## Source Tensions` moved before the Verdict (ins-04 escalation)

The patcher escalated this: demoting the section *and* relocating a ~700-word five-item block was more than a surgical hunk. The section was cut whole and reinserted immediately before `## 7. Verdict`, so the report now ends on the recommendation rather than on a caveat. The heading stays an H2 — the gate requires the seven mandated headings to be present and in order, and does not forbid an eighth, so the section keeps its prominence while the Verdict regains terminal position.

## 3. Eleven quote-integrity failures repaired

`run verify` reported 11 errors against the `quote-integrity` rule, which requires every quoted span of five or more words to exist verbatim in a vault note's indexed body. The failures had three distinct causes, and none was a fabricated claim — every number and assertion was real, but the quotation marks were not honest about the text:

- **Ellipsis-joined fragments.** Several quotes stitched two non-adjacent sentences with `…`, which cannot match verbatim. Split into contiguous spans or paraphrased: PaLM's `6N` formula, Megatron-Core's memory-wall and qualitative-difference statements, FlashMLA's supported scope, FSA's GQA finding.
- **Normalised Unicode.** Megatron-Core's body uses math-italic glyphs (`𝐸`, `𝑁`, `∝`) and FlashMLA's uses `sm100a` without the underscore the prose used. Quoted spans were narrowed to ASCII-only contiguous runs.
- **Quotes of non-vault text.** Two quotations were of the repo's own documents (`research.md`'s EP estimate, and the report's own restatement of "DDP only"), which are not vault notes and so cannot be verified by the rule. Both were paraphrased and the quotation marks dropped.

Where a quote could not be made verbatim without distorting the prose, the quotation marks were removed and the claim paraphrased with its citation intact. No claim was weakened or dropped.

**Result:** `quote-integrity` clean, `required-headings` clean, `length-in-range` clean at 10,010 words, `citation-density` 11.09/1000. The only remaining gate failure is `artifact:polish-log.json`, which step 15 produces.
