---
title: DeepSeek-V4-Flash on AMD gfx90a
id: deepseek-v4-flash-on-amd-gfx90a
tags:
- cmp170hx-training-feasibility-a0d4fc
- tensor-parallelism
- moe-inference
- csa
- long-context
- amd-mi250
created: '2026-10-27T11:26:34.679653Z'
updated: '2026-10-27T19:00:31.504946Z'
source: https://arxiv.org/abs/2609.15627
source_domain: arxiv.org
fetched_at: '2026-10-27T11:26:34.679089Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'DeepSeek-V4-Flash inference on AMD MI250 (gfx90a/CDNA2) with SGLang (Siming
  Huang, HKUST-GZ, arXiv 2609.15627v2, Sep 2026) — an enablement and measurement report,
  not a training study. Model: 284B total / ~13B active, FP4 routed experts, mostly
  FP8 non-expert weights, 48 safetensors shards. Host: 8 MI250 GCDs (~64 GiB HBM each)
  on one Supermicro AS-4124GQ-TNMI, 1 TiB DDR4. Topology: a single TP8/EP1 instance
  — explicitly ''no routed-expert A2A'' — with a 1,048,576-token logical KV pool.
  Measured (three rounds per group, native autoregressive): resident decode C1 88.63,
  C32 1041.78, C64 1327.10 output tok/s; C16 prefill 10,413.87 / 10,263.02 / 9,966.84
  input tok/s at ~8K / 16K / 32K input length; 8K public-source prefill 7,035-10,431
  tok/s across C1-64. Accepted prefill config = reduced indexer replication + mHC
  reuse + paired-GCD attention + fixed-order expert reduction. Honest limits stated:
  no filled-1M-context performance or quality certification, dynamic batching and
  cold-shape compilation untested, component equality does not prove numerical equivalence,
  and the report covers V4 not V4.1/Engram offload. Relevance to this run: it is the
  closest published datapoint for running a 284B CSA-family MoE across eight 64 GB
  cards with tensor parallelism and zero expert all-to-all — i.e. TP8 + EP1 is the
  topology a serving stack chose when it wanted to avoid routed-expert A2A entirely.
  Caveat: the interconnect is never named (MI250 GCDs are Infinity-Fabric-linked),
  so the TP8 result is not evidence that TP8 is viable on Gen2 x4.'
raw_file: raw/deepseek-v4-flash-on-amd-gfx90a.pdf
doi: arXiv:2609.15627
utility_score: 12.0
citation_count: 0
---

DeepSeek-V4-Flash on AMD gfx90a
Correctness Recovery and Inference Performance Engineering
Siming HUANG
HKUST(GZ)
Guangzhou, China
Abstract
We report the enablement, correctness recovery, and per-
formance engineering of DeepSeek-V4-Flash inference on
AMD Instinct MI250 (gfx90a/CDNA2) using SGLang. Origi-
nal mixed FP4/FP8 checkpoint weights are retained; execu-
tion uses shape-specific HIP, AIter, Composable Kernel, and
Triton paths. The study extends an earlier four-GCD investi-
gation to a single eight-GCD tensor-parallel instance with
a 1,048,576-token logical KV pool. A fresh-process native
autoregressive campaign on public-source code requests cov-
ers concurrency 1 through 64, with three measured rounds
per group. Single-request resident decode reaches 88.63 out-
put tokens/s, while C32 and C64 reach 1041.78 and 1327.10.
Separate C16 prefill measurements reach 10413.87, 10263.02,
and 9966.84 input tokens/s at approximately 8K, 16K, and
32K input length. The accepted prefill configuration com-
bines reduced indexer replication, mHC reuse, paired-GCD
attention, and fixed-order expert reduction. We distinguish
current measurements from historical ablations and specu-
lative throughput, and resident decode from whole-request
HTTP latency. Component equality and bounded semantic
checks do not establish universal numerical equivalence;
dynamic batching, cold-shape compilation, and untested
million-token occupancy remain explicit limitations.
Keywords: LLM inference, mixture of experts, AMD MI250,
tensor parallelism, quantization, correctness, speculative de-
coding
1
Executive summary
This report updates the evidence through September 18, 2026.
The current campaign uses execution HEAD 147ce11b25,
recorded working-tree hashes, and the explicitly enabled
accepted prefill configuration. The September 14 matrix
archive 3d73a75923 and pilot fb2e39fca9 remain histor-
ical evidence, not freshly repeated ablations.
The report applies three evidence rules:
1. Native AR and full-target DSpark are distinct from
historical approximate verification. Their rates are not
interchangeable.
2. Original checkpoint precision, component equality,
repeated output hashes, and end-to-end answer quality
Technical Report, September 2026,
2026.
are distinct claims. No one check substitutes for the
others.
3. Formal three-round matrices and small ABBA screens
remain separate. Minima/maxima describe observed
runs, not confidence intervals; percentages from dif-
ferent experiments are not added together.
2
Related systems
This work builds on serving systems that reduce schedul-
ing and memory-management overhead for autoregressive
language models. SGLang combines a structured runtime
with RadixAttention and continuous batching, while vLLM
introduced PagedAttention to manage KV-cache fragmen-
tation and sharing efficiently [10, 11]. Our contribution is
narrower and hardware-specific: we retain SGLang’s serving
abstractions while adapting the DeepSeek-V4 execution path
to CDNA2 and validating optimizations against fixed-token
numerical oracles.
DeepSeek-V3 established the MoE, routing, multi-token
prediction, and low-precision training lineage; DeepSeek-
V4 adds compressed sparse/heavily compressed attention
and mHC [5, 7]. MegaBlocks demonstrates block-sparse
MoE training [9]; our serving workload instead spans tiny
per-expert decode matrices and large prefill chunks. These
regimes require different execution geometries rather than
a universal kernel. DSpark supplies a separate confidence-
scheduled speculative design [4]; we report its historical
full-target results separately from the current native-AR ma-
trix.
On AMD GPUs, Composable Kernel and AIter provide
tuned HIP, CK, and CKTile building blocks for GEMM, quan-
tization, attention, and MoE [2, 3]. We use these libraries
where their supported shapes are effective, but introduce
gfx90a-specific paths where generic kernels do not match FP4
storage, wave64 execution, or the small-𝑀decode regime.
The resulting system is therefore an integration and measure-
ment study rather than a replacement for either the serving
runtime or the kernel libraries.
3
System and model context
3.1
Hardware and software stack
The experimental host is a Supermicro A+ Server AS-4124GQ-
TNMI with two AMD EPYC 7763 processors and SMT en-
abled. Host memory comprises 32 32-GiB DDR4-2933 2Rx8
RDIMMs, totaling 1 TiB. Historical TP4 runs used four GCDs;
1
arXiv:2609.15627v2  [cs.DC]  20 Sep 2026


---

Technical Report, September 2026,
Siming HUANG
Table 1. Evidence-backed state as of September 18. C denotes client concurrency; resident decode excludes admission and
drain.
Item
Result
Model
DeepSeek-V4-Flash; 284B total parameters, approximately 13B active parameters, FP4
routed experts, and mostly FP8 non-expert weights
Host platform
Supermicro A+ Server AS-4124GQ-TNMI; dual AMD EPYC 7763 processors with SMT
enabled; 1 TiB DDR4-2933 from 32 × 32 GiB 2Rx8 RDIMMs
Accelerators
Eight MI250 GCDs (gfx90a), approximately 64 GiB HBM per GCD; four dual-GCD
accelerators, not eight MI250 boards
Formal native decode
C1 88.63; C32 1041.78; C64 1327.10 resident output tok/s; full seven-tier table in table 2
Formal prefill
8K public-source inputs: 7035.83–10431.43 input tok/s across C1–64; admission limited
to 16 requests
Long-input prefill
C16: 10263.02 at 16K and 9966.84 at 32K input tokens/request
KV pool
1,048,576 logical tokens allocated; no filled-1M-context performance or quality
certification
Runtime topology
One TP8/EP1 instance, no routed-expert A2A; native decode graph tiers
1/2/4/8/16/32/64
Service state
Benchmark services stopped after testing; no live endpoint is implied by this report
the new TP8 matrix uses all eight GCDs as one model in-
stance, not two independent TP4 replicas. The accelerators
are MI250, not MI250X.
The recorded software environment is:
• Python 3.12.13;
• PyTorch 2.12.0a0+git78258b9;
• Triton 3.7.1+git0263a6a6.rocm7.14.0;
• Transformers 5.12.1 and Tokenizers 0.22.2.
The installed HIP compiler reports 7.15.26333 and AMD
clang 23; this is distinct from the HIP 7.14 build provenance
recorded in earlier runs and does not identify every cached
binary. SGLang, AIter, and CK include local source changes.
The environment ledger lists dependency commits and dirty
paths; a clean SGLang commit alone is not a complete binary
reproducibility guarantee.
The original checkpoint comprises 48 indexed safetensors
shards. Recorded SHA256 values cover configuration, tok-
enizer, generation settings, and the shard index, not every
weight byte. This study does not include V4.1 or Engram
offload results.
CDNA2 provides wave64 execution, 64 KiB LDS per CU,
and INT8/BF16/FP16 matrix or dot instructions, but no na-
tive FP4 matrix core comparable to newer accelerator gener-
ations. DeepSeek-V4-Flash stores routed-expert weights in
FP4. Consequently, the critical gfx90a cost is not raw HBM
bandwidth alone: it is the combination of small-M utilization,
online FP4 unpacking, scale handling, activation quantiza-
tion, expert sorting, and layer-level synchronization. The
AMD MI200 ISA manual guided the custom HIP kernels in
this work [1].
3.2
Execution structure
DeepSeek-V4-Flash has 43 decoder layers, 256 routed ex-
perts, top-6 routing per token, shared experts, mHC, and
compressed sparse attention. The official model card reports
284B total parameters, 13B active parameters, a one-million-
token context, and mixed FP4 expert / FP8 non-expert preci-
sion [6].
4
gfx90a enablement
The initial ROCm implementation targeted gfx942 and gfx950.
Enabling gfx90a required the following changes:
• add the gfx90a build target and select HIP_FP8_TYPE
_FNUZ;
• repair ROCm 7.14 include, library, and rpath discovery
while avoiding the conda host-side hipcc;
• build AOT HIP kernels and enable AIter FP4 CK MoE
execution on gfx90a;
• handle Mori topology and XGMI initialization on a
node without an RDMA NIC;
• validate graph capture and replay on gfx90a, separat-
ing kernel selection failures from hardware faults;
• establish both TP4/EP4 with Mori and TP4/EP1 with-
out A2A as executable reference paths.
The initial bring-up checkpoint was 505b3373794a. It pre-
dates the later correctness and performance work and must
not be treated as the final state.
5
Correctness recovery
5.1
Admission, sampling, and chat encoding
An early request stalled at SWA admission, not in Mori or
RCCL. A 4096-token pool, SWA ratio 0.1, and page size 256
2


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
Attention + mHC
compressor / indexer
FFN mHC
and RMSNorm
Router Top-6
FP4 routed experts
gate/up →down
Shared expert
Add partials
TP all-reduce
Next layer /
final HC and LM head
Figure 1. Schematic native TP-only layer boundaries. Shared and routed experts contribute to a join. TP4/EP1 and TP8/EP1
avoid expert A2A but retain TP collectives; the historical EP variants instead require dispatch/combine.
together fell below the admission floor. A separate AIter
sampler issue on gfx90a could return an invalid token ID on
special logits rows; the replacement uses torch.argmax for
this case.
DeepSeek-V4 does not ship a Jinja chat template. It re-
quires the model-specific encoding and output parser, which
is also stated in the official model card [6].
5.2
W2 permutation: invalidating the early 60 tok/s
result
The early approximately 60 tok/s AIter FP4 route was
not correctness-valid. The legacy CK FP4-by-FP4 stage-1
path produced zeros on gfx90a. After switching to CKTile
BF16-by-FP4, stage 1 reached approximately 0.999996 cosine
similarity to a direct oracle, but stage-2 W2 remained near
0.24.
Output-column fingerprinting identified the following 16-
row block mapping within each 128-wide N tile:
fast →reference = [0, 2, 4, 6, 1, 3, 5, 7].
(1)
Applying the inverse permutation
[0, 4, 1, 5, 2, 6, 3, 7]
(2)
to raw W2 weights and scales during loading restored
the logical output order for top-k=1 and top-k=6. This is a
one-time load transformation with no decode hot-path cost.
The fixed France oracle uses the following input IDs:
[0,128803,3085,344,270,6102,
294,8760,2755,128804,128822]
Its expected completion is:
The capital of France is **Paris**.
token ids: [671,6102,294,8760,344,2619,51119,42499,1]
hash: 6f41fe2f01d52507
The short sentinel was stable in the documented eager,
graph, and independent-start checks. This bounded test can-
not certify a long context, a different batch shape, or the
entire numerical pipeline. Greedy completions can diverge
from the first token under different prefill execution; fixed-
input logits and layer comparisons are needed to locate such
differences.
5.3
Later numerical and integration repairs
The later investigation also identified an independent E8M0
scale-layout issue in large-prefill BF16-CK execution: raw
packed weights do not imply raw logical scales. Runtime
AIter scale shuffling must be inverted or consumed with
the matching address map. Constant-scale synthetic fixtures
can hide this error. The latest scoped large-prefill path uses
unique-slot stores and fixed Top-6 FP32 reduction; other
fallback paths may still use atomics. Neither scale-layout
repair nor a fixed final reduction proves whole-model batch
invariance.
A subsequent TP mapping repair passes the rank-local
attention sink rather than the full checkpoint vector to the
H8 kernel. Previously, nonzero ranks could read rank 0’s
head sinks. Equality to that old kernel was therefore not a
valid model reference. The paired-GCD H16 path used in the
latest campaign is based on the corrected mapping. Further
mHC repairs retain original FP32 coefficient weights in the
scoped first/unfused prefill boundaries. The first boundary’s
static 24-value projection can be tabulated by vocabulary ID,
without replacing context-dependent residuals or later-layer
coefficients.
Deterministic index selection fixes membership ties using
logical IDs and presents selected positions in a fixed order
before rank-local physical mapping. The prefill trivial-row
shortcut retains real sequence lengths and all compressor/-
cache updates; only scores irrelevant to Top-K are skipped.
Subsequent native-decode empty-tile elimination is a sepa-
rate optimization, described in section 8.
Fresh starts after the upstream/V4.1 integration exposed
missing C4 lifecycle callbacks and incorrect compression-
ratio discovery for unified V4 pools. Their repair restored
the requested logical 1M pool and native graph construction.
CPU tests also cover SWA-only draft metadata, but the latest
3


---

Technical Report, September 2026,
Siming HUANG
user scope excluded DSpark: the completed native matrix is
not a fresh DSpark GPU validation.
6
Historical TP4 decode optimization
6.1
Parallel decomposition
After the W2 repair, the corrected TP4/EP4+Mori BS1 base-
line was only 14.2–15.0 tok/s. The TP4/EP1 no-A2A CKTile
reference reached approximately 20.14 tok/s. This compari-
son motivated investigation of dispatch/combine, rank im-
balance, and progress overhead; changing the expert decom-
position also changes compute shapes, so it does not isolate
communication cost.
The historical TP4/EP1 decode route combines:
• packed 4-bit FP4 weights without an offline expansion;
• online BF16 activation quantization to INT8 per 32
elements;
• E2M1 nibble mapping to 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12;
• CDNA2 v_dot4_i32_i8 / mixed-dot accumulation fol-
lowed by activation and E8M0 scaling;
• four 16-lane subgroups for top-6 slots in the down
projection;
• AIter peer-read custom all-reduce in place of the high
fixed-latency RCCL path;
• TP-only mHC launch geometry that no longer reserves
CUs for Mori progress.
The ledger associates the largest jump in figure 2 with
peer-read all-reduce, approximately 25.54 to 60.20 tok/s. The
August short-probe milestone is about 74.5 tok/s. A Septem-
ber 6 isolated-tree reproduction obtained a seven-request
HTTP median of 74.018 and maximum of 74.655 tok/s, with
the historical hash on all seven requests. This was TP4 native
AR, not TP8 or DSpark. The inherited rebased tree ran about
53 tok/s; its different tree contents exposed a lost gfx90a
MHC selector. Historical author dates and copied experi-
ment notes did not establish performance on the rebased
code. The current TP8 C1 result uses a different public-code
protocol and must not be divided by 74.5 to claim TP scaling
efficiency.
7
Prefill optimization and historical
checkpoints
7.1
From per-assignment scalar work to CDNA2
MFMA
The original raw-FP4 direct kernel sent M=256 prefill through
a per-assignment FP16 dot path, with approximately 12.9 s
TTFT for a 1028-token prompt. The optimization sequence
introduced:
1. packed-FP4 unpack reuse across rows, first group8 and
then group32 for M≥1024;
2. v_perm_b32 lookup tables instead of branch-heavy
nibble decoding;
3. MI250 block-FP8 tuned configurations for M512 and
M1024 dense projections;
4. CDNA2 INT8 MFMA routed gate and down kernels;
5. sparse paged-prefill attention reduction from eight
waves to one wave;
6. chunked-prefill growth from 512/1024 to 2048;
7. M2048-specific MFMA grids, native HIP INT8 quanti-
zation, and scale/metadata broadcast;
8. an expert sorter block increase from 32 to 64 rows,
allowing an expert with approximately 48 assignments
to scan packed weights once.
The historical M2048 microbenchmark compares the 32-
row and 64-row sorter variants in figure 4. Gate/up improves
from 7.26 to 5.55 ms (23.6%), down improves from 6.01 to
5.26 ms (12.5%), and the combined kernel time improves
by approximately 18.5%. The recorded component fixtures
reported identical outputs; this is not a full-model equality
claim.
The historical endpoint ABBA sequence was A1=2.184
s, B1=2.061 s, B2=2.062 s, and A2=2.185 s. A later chunk-
only TP4 test used B1=1.861 s, A=1.978 s, and B2=1.852 s (six
steady samples per arm). Chunk 2304 removes the small tail
traversal for 4604 tokens; it is not a universal optimum over
prompt lengths. The first new-shape request still took over
20 s, whereas the warm measurements were near 1.85 s.
7.2
Large-M CK and the meaning of the earlier 6.42k
result
The subsequent BF16-CK prefill route materializes the cur-
rent expert shard in BF16, runs CK stages, and casts the
accumulated output back to BF16. It does not alter the check-
point files, but changes execution arithmetic relative to raw-
FP4/INT8-dot kernels. In the September 7 TP8 migration, 32
distinct requests totaling 73,724 input tokens, admission 16,
and two approximately M36864 forwards reached a warm
median of 6420.39 input tok/s with a 131,072-token pool. This
is an aggregate prefill wave, not single-request decode. The
September 14 matrix used approximately 8K input per re-
quest and a 1M pool, reaching 4.68–5.27k. Its different length,
pool, grouping, and timing prevented a direct regression
claim. The September 18 matrix below replaces that histori-
cal matrix as the current configuration’s measurement.
8
Current TP8 native-AR results
The primary result is a fresh-process remeasurement on Sep-
tember 18 at execution HEAD 147ce11b25, with working-
tree and launch hashes retained. The same process runs the
cells serially, with explicit per-shape warmups and three
measured rounds per cell. All seven native graph tiers have
matching active input and executed rows during full res-
idency. The model remains a single TP8/EP1 instance. No
4


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
EP4
CKTile
EP1
CKTile
INT8-dot
MoE
Peer-read
all-reduce
Router +
geometry
Aug. 24
checkpoint
0
10
20
30
40
50
60
70
80
Native AR / HTTP (tok/s)
14.6
20.1
25.5
60.2
66.1
74.5
Historical TP4 single-request milestones (different checkpoints)
Figure 2. Historical TP4 short-probe milestones retained from the earlier report. Different checkpoints and controls are not a
single controlled speedup series. The invalid early FP4 route is excluded; the later 60.20 point follows a separate repair.
Initial
MFMA
Down
geometry
4-wave
attention
MHC
selector
Chunk
2048
1-wave
attention
Grid
retune
64-row
sorter
1200
1400
1600
1800
2000
2200
Input throughput (tok/s)
1336
1575
1765
1803
1902
2019
2108
2234
Historical TP4 prefill: 4604-token single request
Figure 3. Historical TP4 sequence for a 4604-token prompt. Rates derive from HTTP TTFT; these points are not the latest
8K-input TP8 matrix.
5


---

Technical Report, September 2026,
Siming HUANG
Gate/up
Down
0
1
2
3
4
5
6
7
8
Latency (ms)
7.26
6.01
5.55
5.26
M2048 gate/up and down
32-row
64-row
A1
32-row
B1
64-row
B2
64-row
A2
32-row
1.95
2.00
2.05
2.10
2.15
2.20
TTFT (s), expanded axis
2.184
2.061
2.062
2.185
M2048 sorter: historical ABBA
Figure 4. Left: M2048 routed-MoE kernel latency. Right: strict endpoint-level ABBA validation of the 64-row sorter.
DSpark, MTP, or approximate routed-expert verification con-
tributes to the following table. Earlier TP4, speculative, and
individual optimization ablations remain historical records.
Protocol boundaries. Prefill uses 8191–8192-token public-
source prompts, fresh cache salts, and one output token. At
C64 the wave contains about 524k input tokens, admitted
at most 16 requests at a time with a 32,768-token chunk
budget. C64 therefore does not mean one M64 prefill ker-
nel or 64 simultaneous large prefill requests. Decode uses
511–512-token inputs, greedy sampling, natural EOS, and at
most 2048 output tokens. Each formal round accumulates at
least 30 s of common resident windows; explicit warmups
are excluded. The 64-case corpus extends the earlier 32-case
public-source corpus without changing those first 32 cases.
Duration-dependent wave counts can expose different case
mixtures, which are preserved in raw records.
One consistent launch configuration. The current run
enables the accepted C1-only wo_a GEMV for the entire
matrix, including larger-concurrency drain periods. Unlike
the September 14 table, the new C1 cell is not spliced in from
a separate supplement. The default-off native C32 down-
consumer pilot is not enabled. The rejected fixed-Lt owner-
only query producer is explicitly disabled. Prefill uses the
scoped accepted configuration described below, not an as-
sertion that every launcher default automatically selects the
same path.
Interpretation. The 8K input rates range from 7035.83
to 10431.43 tok/s, with C16 at 10413.87. The native resident
decode endpoints are 88.63 at C1 and 1327.10 at C64. The
HTTP/resident gap includes admission, prefill, uneven an-
swer lengths, and drain; it is not a measured host-only over-
head term. All measured rounds, including outliers, are re-
tained. The older matrix is preserved in the frozen data,
but old/new ratios are not controlled cumulative speedups:
the chunk budget, correctness repairs, and active numerical
paths changed. No matched current TP4 seven-tier matrix is
available, so a TP4-to-TP8 scaling factor is not reported.
8.1
Accepted large-prefill configuration and longer
inputs
The current configuration combines several independently
developed changes: causal empty-tile elimination; query-
group reuse and runtime-𝑀indexer kernels; row-owner
score/Top-K work with logical-index exchange; exact paired
FP32 mHC pre-mix ownership and HIP post-update; and tem-
porary H8-to-H16 attention activation redistribution within
paired GCDs. Full indexer query production and compres-
sor/cache updates remain intact. Wide-owner paths cover
the measured 16K and 32K inputs without truncating their
KV selections.
For routed MoE, the accepted large-prefill path retains
original checkpoint storage, uses layer-local BF16 expansion,
and combines route production with unique-slot CK stage-2
stores and a vectorized fixed-order Top-6 FP32 reducer. This
eliminates the final cross-expert atomic order in the admitted
path; it does not establish batch-invariant projections or
universal determinism. Original FP32 mHC coefficients are
retained at the corrected first/unfused boundaries. A small
static first-layer vocabulary table replaces only its 24-value
raw pre-mix projection, not subsequent context-dependent
operations.
These are separate actual-length workloads, not a same-
shape regression or a filled-million-token-context test. The
6


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
Table 2. Formal native P/D matrix: P and resident D are three-round medians; HTTP aggregates complete measured waves.
Prefill divides aggregate input by wave time to the last first token. Resident and HTTP decode are distinct metrics.
C
Prefill input tok/s
Resident output tok/s
HTTP output tok/s
1
7035.83
88.63
86.76
2
9176.62
107.41
102.89
4
10278.75
187.13
152.72
8
10362.67
333.38
250.49
16
10413.87
598.11
458.78
32
10431.43
1041.78
689.74
64
10381.49
1327.10
829.65
1
2
4
8
16
32
64
Client concurrency C
0
2000
4000
6000
8000
10000
12000
Input tokens / s
(a) Prefill: 8K input, 1 output
Wave median
1
2
4
8
16
32
64
Client concurrency C
0
200
400
600
800
1000
1200
1400
Output tokens / s
(b) Native decode: 512-token input
Resident median
Whole HTTP
Figure 5. Latest TP8 matrix. Colored points/lines show three-round medians; thin vertical ranges and small points are actual
rounds, not confidence intervals. Gray decode points are complete-wave HTTP rates, not a slower kernel configuration.
Concurrency uses a base-2 axis.
Table 3. Current C16 length sweep, three-round medians.
Original weights, 1M allocated logical KV, 32K chunks, zero
prefix-cache hits.
Input tokens/request
Aggregate input tok/s
Approximately 8K
10413.87
Approximately 16K
10263.02
Approximately 32K
9966.84
experiment records resolved settings, eight-rank backend
witnesses, source hashes, input echoes, zero cache-hit counts,
and raw timing records. France and ID/text checks provide
bounded validation, not a task-accuracy benchmark. His-
torical ablation gains are not added together to explain the
combined configuration.
The current three-round C16 prefill tests also expose a
repeatability limit: two of the sixteen 16K requests change
their first token across rounds; the corresponding 8K and
32K C16 first-token sets repeat exactly. These are one-token
observations, not full-answer equality tests. The differing
16K responses are retained in the timing data. Fixed CK re-
duction and local component equality must not be presented
as whole-model determinism.
7


---

Technical Report, September 2026,
Siming HUANG
8.2
Historical ablation: empty C4 tiles
With a 1M logical pool, the C4 logits launch could cover
262,144 key positions even when only a short prefix was
active. Beyond the raw-token threshold near 2048, many
launched tiles had no valid keys but still performed unnec-
essary query loading and dot work. The exact native-decode
guard skips this empty work while retaining defined zero
stores, unchanged active computation, sequence lengths,
Top-K semantics, and physical index mapping.
Seven component shapes, 100 mutations per shape, and
1000 graph replays per shape checked score bits and logical/-
physical indices. Fully populated control cases did not exhibit
the empty-work benefit. At M32, the diagnostic component
dropped from approximately 6934 to 837 microseconds; this
is not a whole-model speed prediction.
A separate matched 8K-input C32 service ABBA measured
169.657 / 611.103 / 610.853 / 169.843 resident output tok/s,
a 3.599-fold candidate/control ratio. Complete HTTP rates
were approximately 83.18 / 128.85 / 128.88 / 83.26 tok/s. The
different ratios illustrate why the measurement boundary
must be explicit. This 8K test is not the 512-input formal
decode table, and its multiplier must not be applied to that
table. This fix is native-decode scoped; the earlier prefill
trivial-row skip is a different path.
8.3
Historical ablation: C1 projection specialization
The restored wave64 wo_a GEMV measured 30.75→6.88 mi-
croseconds in its component test. All 100 tested mutations
were finite and repeatable, but only 70 were bit-equal to the
former einsum path; the maximum relative L2 error against
a float32 reference was about 0.00179 on the test inputs. It re-
tains checkpoint weights while changing floating reduction.
The service ABBA was 79.558 / 87.583 / 87.504 / 79.151
resident output tok/s: +10.32% by mean candidate/control
arms. Candidate repeats had identical completion sequences
in that test, and France passed. The separate formal C1 me-
dian is 87.599. These controls are more informative than an
unpaired comparison with the August 74.5 HTTP result.
8.4
Historical down-consumer pilot: a small C32
gain
The follow-up at fb2e39fca9 replaces the M32 second quan-
tization/down chain with a CTA16 consumer. Each CTA
reads the already-rounded BF16 intermediate, forms group32
INT8 values/scales in LDS, computes its output stripe, and
retains the existing FP32 top-k partial and fixed reduction.
The current gate, sorter, original weights, attention, and TP
collectives are unchanged. The selector is limited to native
TP8/EP1, M32/I256 and A4/R2 geometry; it remains default-
off.
On one GCD, synthetic diverse/skewed inputs reduced
quant+down+reduction from 130.542 to 115.907 microsec-
onds and 81.615 to 70.791 microseconds, respectively. Each
distribution passed 100 mutations of inputs, routes, weights
and scales and 1000 checked graph replays. Comparisons
used numeric element equality, not a signed-zero-bit check.
This is neither the full routed stage nor a captured real-layer
oracle.
The service screen uses two natural-EOS waves per leg
with the same 32 code requests, rather than the duration-
driven formal 64-case protocol. C32 leg medians are 1047.763
/ 1060.206 / 1061.203 / 1041.459 resident tok/s. Mean control
1044.611 versus candidate 1060.705 gives +1.541%; the return
control changes −0.602%, and candidate legs differ 0.094%.
Every candidate wave exceeds every control wave. Whole
HTTP rates vary with answer length and do not establish
the same percentage gain.
The conditional C1 test is an isolation check, not an M1
port: the existing direct M1 down kernel already quantizes
in LDS, with a different rounding contract. Its control/can-
didate means are 89.722/90.024 tok/s; the 0.34% variation is
not attributed to an unselected kernel. All eight 1453-token
C1 completions match across arms. This fixed-case screen
does not supersede the formal multi-case C1 value. For C32,
control outputs themselves vary across waves; neither whole-
model bitwise parity nor code-answer factual correctness
is established. All 264 measured pilot responses passed in-
dependent ID/text integrity checks, and all service France
sentinels passed.
9
Historical speculative decoding: separate
evidence
The latest native-AR matrix deliberately excludes DSpark.
We retain the September 11 full-target result as historical
evidence, not a claim that the latest merged DSpark service
has been revalidated. This distinction also prevents an earlier
approximate-target speed from becoming a recovery target.
9.1
Why the historical TP4 1.5k result is not a strict
baseline
In the old anchor-only/compact routed-MoE path, C32 with
three draft positions produced M128 target rows, but only
32 anchor rows executed the full routed expert branch. Non-
anchor rows omitted routed experts. Keeping stored weights
unchanged did not preserve the full target function at those
positions. Consequently, approximately 1.5k resident tok/s
from that path is an approximate-target result. It cannot be
presented as native AR or lossless full-target verification,
even if an early France answer passed.
M128 is a row count, not inherently speculative: native
C128 would also produce M128, whereas native C32 nor-
mally produces M32. Padding C32 to 128 does not compute
three useful future AR positions for free. The new native
down-consumer pilot therefore tests actual M32 without
reintroducing DSpark.
8


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
A1
B1
B2
A2
Service leg (A: control; B: candidate)
0
100
200
300
400
500
600
700
Resident output tokens / s
169.7
611.1
610.9
169.8
3.60x resident speedup
(a) Empty C4 tiles: C32, 8K input
A1
B1
B2
A2
Service leg (A: control; B: candidate)
75.0
77.5
80.0
82.5
85.0
87.5
90.0
92.5
Resident output tokens / s
79.6
87.6
87.5
79.2
+10.32%; expanded y-axis
(b) Restored C1 wo_a GEMV
Figure 6. Separate native ABBA experiments: empty C4 tiles at C32 with 8K inputs, and the C1 projection specialization with
512-token code inputs. The right axis is expanded; the experiments must not be multiplied together.
A1
B1
B2
A2
Service leg (A: control; B: candidate)
1035
1040
1045
1050
1055
1060
1065
1070
1075
Resident output tokens / s
1047.8
1060.2
1061.2
1041.5
+1.54% in this small screen
(a) C32: scoped down consumer
A1
B1
B2
A2
Service leg (A: control; B: candidate)
89.2
89.4
89.6
89.8
90.0
90.2
90.4
Resident output tokens / s
89.9
90.0
90.0
89.6
No attributed C1 speedup
(b) C1: unchanged kernel / isolation
Expanded y-axes; two natural-EOS code waves per leg
Figure 7. Default-off down-consumer pilot, expanded y-axes. Dots are the two measured waves per leg; short bars mark their
median. C1 never executes the candidate and is a negative-control scope check.
9.2
Strict TP8 MHC fusion checkpoint
The September 11 trial preserved all target experts, original
weights, TP-synchronized draft/accept decisions, a 1M logi-
cal pool, and gamma three. It used 32 real-code requests, 512
outputs with ignore_eos, and chunk 2304. Four measured
resident windows per family gave:
FP32 mixing was selected historically. The FP16–FP32
mean gap of about 1.31% was smaller than the observed FP16-
family range of 2.5%, their ranges overlapped, and only four
9


---

Technical Report, September 2026,
Siming HUANG
Table 4. Historical full-target DSpark only; not a current AR comparison. Acceptance and throughput are reported in their
original trial scope.
Family
Resident tok/s
Mean accept length
Relative to control
Control
1073.16
2.687
—
Fusion + FP32 mixing
1127.48
2.732
+5.06%
Fusion + FP16 mixing
1142.26
2.698
+6.44%
windows per family were available. FP16 also had a larger
component perturbation and the trial’s single observed se-
mantic degeneration. These observations motivate the con-
servative choice, not a statistical proof that either rounding
variant is universally safer.
9.3
Acceptance, timing, and drift
For concurrency 𝐶, average committed tokens per request 𝑎,
and complete speculative step time 𝑇SD, useful throughput is
𝐶𝑎/𝑇SD. It exceeds native AR only if𝑇SD/𝑇AR < 𝑎. Larger ver-
ification batches expand expert work and may not amortize
draft cost. Acceptance alone is not a speed metric.
In the historical trial, chunked admission ramped the ac-
tive batch through several sizes before all requests were resi-
dent. Controls differed from themselves on most complete
output sequences. Such changes in batching and numerical
execution confound attribution from final hashes; they do
not prove that all other numerical or synchronization risks
are absent. Most flagged dot tails were forced continuation
after EOS, distinct from genuine semantic repetition. This
motivated the natural-EOS protocol used in the new AR
matrix. The 1127.48 figure must not be divided by the new
1044.32 AR figure to claim a matched speculative speedup:
output policy, input mixture, chunking and date differ. A
fresh matched AR/strict-DSpark experiment remains outside
this update.
10
Measurement methodology
10.1
Resource and metric controls
Every GPU experiment begins with:
amd-smi process --json
Launchers compare GPU owners with their recorded ser-
vice PID, command, and descendants before each test. The
current campaign additionally checks Linux boot ID and pro-
cess start ticks, avoiding wall-clock-derived process identity.
They refuse unrelated GPU owners and terminate only the
owned test process tree. The native campaign verifies the
original model path, TP size, speculative algorithm being ab-
sent, actual graph row counts, and KV pool size. Logged flags
alone are not proof that a specialized kernel was selected.
Early experiments used NUMA node 1 because the host
had a known node-0 memory fault, independent of MI250
or SGLang. The operator subsequently reported that fault
repaired. The September 13–14 launcher uses NUMA inter-
leaving across the host; it does not retain the old node-1-only
policy. This report does not claim a new hardware-health
diagnosis.
We sampled device memory with AMD-SMI at roughly
five-second intervals. The highest sampled use was 54743
of 65,520 tool-reported MB on a GCD (83.55%). These are
sampled device observations, not exact allocator peaks or
occupied-KV counts. The setting mem_fraction_static=0.
96 controls a budget, not constant 96% utilization. Logical KV
capacity, allocated storage, graph and workspace memory,
and occupied context are different quantities.
10.2
Timing definitions and aggregation
Let 𝑓𝑖and 𝑒𝑖be the first and last streamed-token timestamps
of request 𝑖, and 𝑛𝑖(𝑡) its cumulative output count. The com-
mon resident interval is
𝑡𝑠= max
𝑖
𝑓𝑖,
𝑡𝑒= min
𝑖
𝑒𝑖,
𝑅𝐷=
Í
𝑖[𝑛𝑖(𝑡𝑒) −𝑛𝑖(𝑡𝑠)]
𝑡𝑒−𝑡𝑠
.
(3)
Only intervals with 𝑡𝑒> 𝑡𝑠contribute. Multiple waves con-
tribute token and duration sums within each formal round;
the reported cell is the median of three round rates. This is
streamed service wall time, not pure GPU timing. Whole-
wave HTTP rate instead divides all completed output tokens
by wave start-to-last-response time. Prefill rate divides all
prompt tokens by wave start-to-last-first-token time, includ-
ing admission and first-token overhead. The two-column
P/D matrix does not describe disaggregated serving.
Candidate comparisons use the order
𝐴1 →𝐵1 →𝐵2 →𝐴2.
(4)
A1 and A2 use independently started control processes;
consecutive B1/B2 legs can share one candidate process. This
is stated per experiment, not represented as four indepen-
dent starts. Explicit warmup files are excluded. Measured
outliers remain in the formal three-round matrix. The small
down-consumer pilot uses two fixed waves per leg and mean
arm medians for its ABBA comparison. No statistical confi-
dence interval is inferred from these small samples. Earlier
trimmed-mean or short-probe protocols retain their original
labels.
The initial request can trigger compilation even after an-
other shape was warm. The new corpus includes rows of 511
10


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
1040
1060
1080
1100
1120
1140
1160
1180
Historical resident output tokens / s (not native AR)
Control
Fusion / FP32
Fusion / FP16
1073.16   |   accept 2.687
1127.48   |   accept 2.732
1142.26   |   accept 2.698
Strict TP8 DSpark, C32: September 11 evidence
Figure 8. Historical strict DSpark family means with the ledger’s rounded observed min/max ranges, not confidence intervals.
Native AR, natural EOS, and the September 14 C32 pilot are deliberately not plotted on this axis.
and 512 tokens; admission groups exposed M8190 and M8191
below the M8192 CK selector. Object timestamps and Ninja
records associated four builds totaling about 23.1 s with one
slow wave. This evidence explains that cold-shape event, not
all HTTP variability. Runtime-M repair already removed one
small-prefill specialization problem, but remaining exact-M
tails are an open latency risk. Generated IDs, raw stream sam-
ples, finish reasons, input hashes, and cold/warm separation
are retained outside temporary directories.
10.3
Correctness hierarchy
1. Kernel level: elementwise bitwise equality, maximum
absolute error, relative L2, and cosine similarity.
2. Layer level: router top-k, stage 1, activation, stage 2,
and rank-local sum.
3. Model level: decode completion IDs independently and
check semantic sentinels and logits on fixed inputs.
France is a sanity check, not a quality benchmark.
4. Attribution level: batch shapes, admission, prefixes,
physical cache state, and numerical path must be con-
trolled before comparing whole output hashes. Greedy
divergence alone cannot identify the faulty operator.
5. Remaining coverage: broad teacher-forced tests com-
paring cached decode with recomputation at compres-
sion, window, and indexer boundaries, plus filled-long-
context quality evaluation. Existing local fixtures are
not universal coverage.
11
Rejected or reverted directions
The following historical observations apply to the tested
shapes and configurations. They do not prove a general al-
gorithm or ISA instruction is unusable, and were not all
rerun during this report update. Component timing is never
substituted for a service-level win.
12
Serving, memory, and agent integration
12.1
Reproducible configuration, not a live
deployment
The earlier four-GCD deployment allocated a 560,896-token
pool and exposed graph tiers 1/2/4. The latest tested configu-
ration instead uses TP8/EP1, pool 1,048,576, memory fraction
0.96, seven decode tiers, admission 16 for prefill, and chunk
32,768. Benchmark services are shut down after testing. No
current endpoint availability, LAN exposure, or continuously
running GPU workload is claimed.
The documented launcher starts a loopback service on
port 30021 only when explicitly invoked. A user should dis-
cover its served model ID before sending a chat request; the
performance harness bypasses chat tokenization by supply-
ing the frozen input IDs directly.
curl http://127.0.0.1:30021/v1/models
curl http://127.0.0.1:30021/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{
"model": "<served-model-id>",
"messages": [{"role": "user", "content": "Hello
"}],
"max_tokens": 64
}'
12.2
Historical API and agent integration
Earlier work implemented OpenAI-compatible model dis-
covery, streaming chat, a separate optional system-prompt
proxy, and tool-result round trips for DeepSeek Harness [8].
11


---

Technical Report, September 2026,
Siming HUANG
Table 5. Important negative results and their implications.
Direction
Result
Conclusion
CKTile KSPLIT=2/4
Approximately 25/22 tok/s
or lower
Split-K reduction and CTA pressure do not suit BS1
small-M MoE.
AsyncLL / SDMA Mori
Approximately 20–49 tok/s
The current low-latency transport does not shorten the
critical path.
Offline FP4-to-INT8 expansion
Gate 7.23→10.45 ms
Doubled weight traffic costs more than removing online
unpack.
CK BF16 router instances
Approximately 86.7–98.6 𝜇s
Slower than stable torch/rocBLAS; not integrated.
Simple LDS BF16 MFMA router
Approximately
0.43–0.52
ms
Barrier and bank/layout costs dominate.
32×32 INT8 MFMA gate
Best approximately 7.87 ms
165 VGPR plus 32 KiB LDS collapses occupancy.
Full prefill CUDA Graph
Capture succeeds; replay
faults
Prefill metadata addresses are not capture-stable.
One-CTA/token HIP mHC prefill
Approximately 28% slower
Scalar FMA cannot replace the current decomposition by
launch reduction alone.
RCCL in place of peer-read AR
TTFT 0.679→0.711 s
Reduce collective boundaries instead of substituting RCCL.
BF16 down partial
Approximately 0.8% micro
gain
with
broad
value
changes
Insufficient benefit; retain FP32 partials.
Those endpoints were not part of the new performance mea-
surements, and the proxy’s availability is not revalidated
here. No private host address or persona is needed to repro-
duce the benchmark.
The earlier parser configuration was:
--tool-call-parser deepseekv4
--reasoning-parser deepseek-v4
Streaming/non-streaming tool-call checks were interface
smokes, not throughput or model-quality measurements.
Reproduction should use the archived launcher and input
manifest rather than infer configuration from an old end-
point example.
13
Remaining risks and next steps
1. Capacity is not occupancy. Fill the allocated 1M pool
with controlled long contexts and verify quality, admis-
sion, sustained throughput, and exact memory peaks.
Current measurements do not provide this evidence.
2. Numerical coverage. Preserve fixed Top-K member-
ship and order, test real nonconstant scales, and distin-
guish reduction effects from batch scheduling effects.
Unique-slot CK removes one source of nondetermin-
ism, but different GEMM shapes can still perturb scores
and membership. Tested repeatability does not imply
universal batch invariance.
3. Cold-shape coverage. Audit shape-specific compi-
lation keys and pre-warm supported shapes. Report
cold, restarted, and warm TTFT separately. Do not hide
latency outliers inside resident-only speed claims.
4. Indexer work. Large-prefill score/Top-K ownership
is now accepted, but owner-only query production
is not. A fixed-hipBLASLt producer matched a his-
torical layer fixture yet failed live layer-2 validation:
two query rows changed three Top-K members, pro-
ducing 291 positional differences in sorted lists. Local
Top-K and publication matched their score inputs; pro-
jection arithmetic was upstream of the discrepancy.
This rejected candidate remains disabled in the current
campaign.
5. Controlled comparisons. Matched TP4 and TP8 ma-
trices and a fresh strict-DSpark regression remain miss-
ing. This update ran neither; unavailable temporary
artifacts are not reconstructed as invented measure-
ments.
6. Deployment decision. The C32 consumer remains
opt-in after a small positive screen. Broader request
mixtures and context ranges are required before chang-
ing a global default. The C1 isolation result is not an-
other kernel optimization.
14
Conclusion
The expanded study combines historical TP4 correctness
recovery with a fresh single-instance TP8 evaluation. The
current native-AR matrix reaches 88.63 output tok/s at C1,
1041.78 at C32, and 1327.10 at C64. For C16, the prefill rates
are 10413.87, 10263.02, and 9966.84 input tok/s for approxi-
mately 8K, 16K, and 32K requests with a 1M allocated logical
pool. Reduced replicated work, mHC reuse, paired-GCD at-
tention, and fixed-order expert writeback jointly define this
12


---

DeepSeek-V4-Flash on AMD gfx90a
Technical Report, September 2026,
measured configuration. Historical ABBA percentages are
not summed or represented as a freshly rerun cumulative
ablation.
The evidence emphasizes the mismatch between stored
precision and execution formats, small expert matrices, re-
dundant work, specialization reachability, and fixed synchro-
nization boundaries. It does not provide a hardware-counter
roofline proving a single universal bottleneck. Keeping na-
tive AR, strict speculation, approximate targets, HTTP la-
tency, and resident throughput separate is as important as
the kernel work itself. The final deliverable is an auditable
configuration and evidence archive, not a claim of universal
bitwise equivalence or a completed million-token quality
benchmark.
References
[1] Advanced Micro Devices, Inc. n.d.. AMD Instinct MI200 CDNA2 In-
struction Set Architecture.
Advanced Micro Devices, Inc.
https:
//www.amd.com/content/dam/amd/en/documents/instinct-
tech-docs/instruction-set-architectures/instinct-mi200-cdna2-
instruction-set-architecture.pdf Accessed September 14, 2026.
[2] AMD ROCm. n.d.. AIter: AI Tensor Engine for ROCm.
https:
//github.com/ROCm/aiter Accessed September 14, 2026.
[3] AMD ROCm. n.d.. Composable Kernel. https://github.com/ROCm/
composable_kernel Accessed September 14, 2026.
[4] Xin Cheng et al. 2026. DSpark: Confidence-Scheduled Speculative
Decoding with Semi-Autoregressive Generation. arXiv:2607.05147.
https://arxiv.org/abs/2607.05147
[5] DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. arXiv:2412.19437.
https://arxiv.org/abs/2412.19437
[6] DeepSeek-AI. 2026. DeepSeek-V4-Flash Model Card. https://huggin
gface.co/deepseek-ai/DeepSeek-V4-Flash
[7] DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Efficient Million-
Token Context Intelligence. arXiv:2606.19348. https://arxiv.org/abs/
2606.19348
[8] DeepSeek-AI. n.d.. DeepSeek Harness. https://github.com/deepseek-
ai/deepseek-harness Accessed September 14, 2026.
[9] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia.
2022. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.
arXiv:2211.15841. https://arxiv.org/abs/2211.15841
[10] Woosuk Kwon et al. 2023. Efficient Memory Management for Large
Language Model Serving with PagedAttention. arXiv:2309.06180. ht
tps://arxiv.org/abs/2309.06180
[11] Lianmin Zheng et al. 2024. SGLang: Efficient Execution of Structured
Language Model Programs. arXiv:2312.07104v2. https://arxiv.org/ab
s/2312.07104
A
Key checkpoints
Table 6 lists implementation and evidence revisions. Dates
and protocol boundaries are recorded in the corresponding
experiment ledgers.
B
Reproducible report build
cd reports/gfx90a-dsv4
python generate_figures.py
make check
make
make arxiv
The report follows the supplied PPoPP workshop tem-
plate’s acmart class. Its options are sigplan,10pt; the
review option is deliberately omitted because the public
arXiv version must not contain line numbers. The class con-
trols two-column layout, margins, fonts, headings and cap-
tions; no geometry or body-font override is applied. Bibli-
ography formatting uses ACM-Reference-Format and Bib-
TeX. Figures embed Arial and retain only left/bottom spines.
Normal compilation needs neither Python nor a GPU. The
source bundle contains all TeX inputs, bibliography sources,
the generated main.bbl, and pre-generated PDF figures. Lo-
cal compilation of the extracted bundle is verified; arXiv’s
remote service is not tested here. The 2024 workshop date is
not used as a claim of publication: this is a September 2026
technical report using that template, with no submission-
page limit implied.
C
Evidence provenance and open gaps
In the companion repository, data/results.json freezes
three rounds per formal cell, ABBA samples, hashes, and
archive identifiers. The script prepare_report_data.py
refreshes that snapshot from repository evidence; normal
report builds do not access parent directories. The two repos-
itory evidence directories are:
• .agents/experiments/dsv4_tp8_revalidation_
20260913/
• .agents/experiments/dsv4_tp8_ar_down_consu
mer_20260914/
The formal archive contains 278 verified files; the pilot con-
tains 92. Complete SHA256 values are retained with the
report data. These checks protect recorded artifacts, not un-
recorded measurements or full checkpoint weights.
Historical TP4 records and the September 11 DSpark ta-
ble are ledger evidence with explicitly different protocols.
Older temporary full-matrix files were unavailable during
this update, so no missing TP4 concurrency cells are filled
from memory. Public source excerpts are fixed at repository
revision 54b93c45c2; a source-path list may be longer than
the text retained after prompt trimming. Neither source-
review prose nor a simple France question measures general
code correctness. Dataset-specific performance should not
be mistaken for a model-quality benchmark or a matched
comparison against another hardware platform.
13


---

Technical Report, September 2026,
Siming HUANG
Table 6. Implementation and evidence checkpoints.
Commit
Content
505b3373794a
Initial gfx90a/ROCm enablement
caf80718cf
CKTile FP4 expert W2 output-order repair
9140c03605
Correct TP-only 60 tok/s route
b097d228c2
Native router decode optimization
ea519e2e0f
Routed-MoE prefill MFMA
6680714cf8
Sparse-prefill wave reduction
6547c5b063
M2048 64-row expert block; long-prefill gain of 5.6%
85ab1a40c5
DSV4 reasoning and tool parser wiring
94d630a47c
Full-endpoint OpenAI client compatibility
b00a4e11cf
Historical TP4 tree reproduced at 74.018 HTTP tok/s median on September 6
137170e367
TP8 large-prefill migration with corrected scale interpretation
9edb8ecdde
Historical strict-DSpark FP32 MHC fusion profile (not the latest AR matrix)
c224e9c1ce
Fresh-start unified-V4 pool/metadata integration repair
3d73a75923
Formal seven-tier native P/D archive and C1 GEMV restoration
fb2e39fca9
Default-off native C32 down-consumer pilot and C1 isolation
14
