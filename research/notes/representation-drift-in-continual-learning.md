---
title: 'Representation Drift in Continual Learning: LoRA versus N-gram Memory'
id: representation-drift-in-continual-learning
tags:
- cmp170hx-training-feasibility-a0d4fc
- representation-drift
- ngram-memory
- frozen-table-prior-work
- drift-does-not-grow
- table-51b-params
- optimizer-state-on-touched-rows
- tp-pp-ep-dependency
- long-source
created: '2026-10-27T12:13:40.547661Z'
updated: '2026-10-27T19:00:31.626271Z'
source: https://arxiv.org/abs/2610.14921
source_domain: arxiv.org
fetched_at: '2026-10-27T12:13:40.547100Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Representation Drift in Continual Learning: LoRA versus N-gram Memory (Hao
  Wu, Illinois Institute of Technology, arXiv 2610.14921v1, 10 Oct 2026) — the prior
  work Delta N-gram builds on, and the primary source for the claim that writing into
  a pretrained N-gram table disturbs the base model least. Setup: a 48-layer mixture-of-experts
  model (512 routed experts, top-10 routing plus shared experts, four hyper-connection
  residual streams) carrying a native, jointly pretrained N-gram memory of 51.2B parameters
  — a table of 320,001,536 rows of width 160, read at layer 2, with each token reading
  one row from each of 8 bigram and 8 trigram hash heads (16 rows, 2,560 values);
  addresses depend only on raw token IDs with history reset after EOS; the model runs
  in BF16. Three update channels on identical data and budget (125 optimizer steps
  x global batch 40 = 5,000 presentations per task, max length 128, Adam beta (0.9,0.95),
  peak lr 1e-4 cosine to zero, gradient clipping 1, optimizer and scheduler reset
  at every task boundary): (1) N-gram memory only — only the table rows addressed
  by the current task''s inputs are trainable, everything else frozen; (2) Shared
  LoRA rank 16 alpha 64 across six projection families; (3) Merged LoRA, a fresh adapter
  per task merged into the backbone. Seven-task chain of SIGMORPHON 2023 morphological
  inflection (Northern Sami, Navajo, Armenian, Gulf Arabic, Albanian, Macedonian,
  Amharic), 5,000 examples each.


  THE LOAD-BEARING RESULT FOR THIS RUN: updating only the N-gram memory changes the
  base model''s representations of general math and code inputs by 26%/15% relative
  RMS, against 64%/35% for merged LoRA and 97%/39% for shared LoRA. Critically, this
  drift DOES NOT GROW with the number of tasks — it reaches 25% after the first task
  and stays there for all seven (21-26% on math), while shared LoRA grows almost monotonically
  (45% -> 97%) and merged LoRA grows to 64% with a 76% peak. Generic behaviour is
  essentially unchanged (no code problem lost, math NLL -0.01), and every change is
  confined to inspectable, reversible table rows. The mechanism is exact: an input
  whose addressed rows were never rewritten is bit-for-bit unchanged at every layer
  and in every logit, because the addresses are fixed by raw tokens and everything
  upstream of the read is frozen.


  THE COST, which the run must not elide: N-gram memory acquires new tasks more weakly
  and its stored tasks interfere with one another, losing as much old-task NLL as
  shared LoRA. Forgetting is governed by drift on the learned tasks'' OWN inputs,
  measured from the checkpoint at which each was learned, not by generic drift (Spearman
  rho = 0.85; rho = 0.71 on Llama-3-8B). With a frozen head, every pairwise logit
  margin changes by the projection of drift onto a head-difference direction. The
  design principle the paper draws — ''write new knowledge only into the memory table
  and keep the backbone fixed'' — leaves the remaining problem, interference between
  stored tasks, inside the memory.


  THE FEASIBILITY NUMBERS THAT MATTER ON A 64 GB CARD: the table is 51.2B parameters,
  i.e. ~102 GB in BF16 — the table ALONE exceeds a single CMP 170HX card by ~1.6x,
  before the 48-layer MoE backbone. So the frozen-table literature''s actual scale
  is far beyond the repo''s ''7B total / 6B frozen'' design; the repo''s design is
  at the small end of this design space and cannot inherit these results without scaling
  the table down by roughly an order of magnitude. On the positive side for memory
  math: the N-gram arm''s trainable set is only 51,018-149,809 rows per task (8-24M
  scalars), all seven tasks together touch 666,747 rows = 0.21% of the table, and
  each task shares 1,277-4,911 rows with earlier tasks. Because untouched rows have
  identically zero gradient and optimizer state, ''Adam is exactly equivalent to full-table
  Adam'' on the touched rows — so optimizer state scales with touched rows, not table
  size, which is precisely the property the repo''s 16 B/trainable-param vs 2 B/frozen-param
  arithmetic relies on.


  TOPOLOGY DEPENDENCY: training used Megatron-SWIFT on 4xH200 with TP2/PP2/EP2 — tensor,
  pipeline AND expert parallelism simultaneously, none of which the CMP 170HX''s no-P2P
  Gen2 x4 fabric supports. So this paper''s training recipe, like Delta N-gram''s
  EP4/DP4, is not reproducible on the box without re-deriving it under pure DDP. 11,284
  words — LONG SOURCE, flag for source-analyst delegation.'
raw_file: raw/representation-drift-in-continual-learning.pdf
doi: arXiv:2610.14921
citation_count: 1
---

*Suggested by [[delta-n-gram-forward-only-continual-writing]] — Delta N-gram reference [31]: the prior Representation Drift study that established N-gram table writes disturb the base least*

Representation Drift in Continual Learning:
LoRA versus N-gram Memory
Hao Wu
Illinois Institute of Technology
hwu71@hawk.illinoistech.edu
Abstract
A language model that keeps learning online must not slowly erode its own base capabilities.
We ask which part of a modern LLM should absorb continual updates, by measuring how each
choice changes the model’s internal representations. On a 48-layer mixture-of-experts model with
a native pretrained N-gram memory, we train a seven-task chain under three update mechanisms
with identical data and budgets: updating only the N-gram memory table, continually training
one shared LoRA adapter, and training a fresh LoRA per task that is merged into the backbone.
We capture fixed-context representations at every layer and task boundary. Updating only the
N-gram memory changes the base model’s representations of general math and code inputs
by 26%/15% (relative RMS), against 64%/35% for merged and 97%/39% for shared LoRA.
This drift does not grow with the number of tasks: it reaches 25% after the first task and
stays there for all seven, while both LoRA variants keep accumulating. Its generic behaviour is
essentially unchanged (no code problem lost, math NLL −0.01), and every change is confined to
inspectable, reversible table rows. The cost appears elsewhere: N-gram memory acquires new
tasks more weakly and its stored tasks interfere with one another, losing as much old-task NLL
as shared LoRA. We show that this forgetting is governed by drift on the learned tasks’ own
inputs, measured from the checkpoint at which each was learned, not by generic drift (Spearman
ρ = 0.85; ρ = 0.71 on Llama-3-8B), and we derive the exact relations behind it: with a frozen
head, every pairwise logit margin changes by the projection of drift onto a head-difference
direction. These results support a design principle for long-running online learning: write new
knowledge only into the memory table and keep the backbone fixed; the remaining problem,
interference between stored tasks, then lies inside the memory.
1
Introduction
A deployed language model that learns continually faces a constraint that one-off fine-tuning does
not: updates never stop. Any perturbation of the shared backbone that accumulates with the
number of updates will, over a long enough horizon, erode the general capabilities the model was
deployed for. The practical question is therefore not only how well a model learns each new task,
but where in the model new knowledge should be written so that the base stays intact as updates
pile up.
Modern LLMs offer more than one place to write. Besides low-rank adapters on the backbone
weights (LoRA [11]), recent production models carry large native memories, such as hashed N-gram
embedding tables pretrained jointly with the backbone [23]. These are read through deterministic
token addresses and touched sparsely. We compare three write channels on the same model, data
and budget:
• N-gram memory: only the rows of the native N-gram table that the current task addresses are
trained; every Transformer weight, the router, the memory read-out and the LM head are frozen.
1
arXiv:2610.14921v1  [cs.LG]  10 Oct 2026


---

• Shared LoRA: one LoRA adapter covering attention, linear attention and every routed and
shared expert is trained continually across tasks.
• Merged LoRA: a fresh adapter of the same shape is trained per task and merged into the
backbone before the next task.
We judge them by what they do to the model’s representations. At every task boundary of a
seven-task chain we capture representations for identical fixed contexts at all 48 layers, the LM-head
input and every MLP/MoE block. We measure them against two references. Notation follows the
key–value view of AlphaEdit [6]: a layer’s input representation is a key, the keys of several inputs
are stacked into a matrix K, and the layer’s response is Y = KW. With a frozen output head, every
change in logits comes from drift of the keys at the head. Big-K drift compares general math and
code inputs with the pretrained base and measures erosion of the base. Small-ki drift compares each
learned task’s inputs with the checkpoint at which that task had just been learned and measures
what happens to acquired knowledge afterwards.
Findings.
1. N-gram memory leaves the base nearly untouched, and the effect does not accumulate.
Its big-K drift is 26%/15% on math/code versus 64–97%/35–39% for LoRA. It reaches this level
after the first task and stays flat for all seven, while both LoRA variants keep growing. The
ordering holds at every sampled layer and answer position. Generic NLL and generation are
essentially unchanged (Sections 5.1, 5.2 and 5.4).
2. Its per-round changes cancel. The realised squared drift is 15% of the sum of per-round
squared changes, against 40–56% for LoRA. Successive table updates move the generic represen-
tation back and forth rather than away (Section 5.3).
3. Its cost is interference among stored tasks, not damage to the base. On learned-task
inputs N-gram memory drifts as much as shared LoRA, and it acquires difficult tasks more
weakly. Merged LoRA retains learned tasks best (Sections 5.6 and 6.3).
4. Forgetting follows small-ki drift, not big-K drift, on Qwen3.8 (ρ = 0.85) and on Llama-3-8B
(ρ = 0.71) (Section 6.1).
Theory.
Section 3 gives the relations that connect these measurements.
They are a ra-
dial/directional decomposition of drift, a telescoping identity that exposes cross-round interference,
a bound of retention loss by small-ki drift, and an exact margin identity: with a frozen head, a
learned token is lost only when drift along a head-difference direction exceeds its acquisition margin.
For N-gram memory, an input whose addressed rows were never rewritten is exactly unchanged.
This is why generic drift stays small, and why interference concentrates on inputs that share rows.
Position.
For a model that must learn online indefinitely, the binding constraint is the base, and
among the three channels only N-gram memory keeps the base fixed with a perturbation that does
not grow with the number of tasks. We therefore argue that new knowledge should be written only
into the memory table. The weaknesses we observe, weaker acquisition and inter-task interference in
shared rows, are then problems of memory management, such as addressing, capacity and per-row
update rules, rather than reasons to resume editing the backbone.
2


---

2
Related Work
Catastrophic forgetting in LLM fine-tuning.
Forgetting under sequential training is a long-
standing problem [13, 20]. For LLMs, continual instruction tuning and domain adaptation have
been shown to degrade general capabilities [19], and LoRA has been reported to “learn less and
forget less” than full fine-tuning [3]. Our comparison is between two LoRA continuation strategies
and a non-LoRA memory, under matched data exposure.
Knowledge editing and null spaces.
ROME and MEMIT [21, 22] treat FFN projections as
key–value stores and write facts by solving constraints on KW. AlphaEdit [6] projects updates onto
the null space of the keys to be preserved, K0, so that K0 ∆W = 0. Our big-K/small-k notation
comes directly from this view. The difference is that a null-space constraint protects against weight
change at fixed keys, whereas under continual training the keys themselves drift (Equation (3));
that second term is what we measure.
Orthogonality and subspace methods.
A family of methods constrains successive updates to
be orthogonal to previously used subspaces, including orthogonal gradient descent [7], O-LoRA [27]
and InfLoRA [16]. These methods implicitly use parameter- or gradient-space geometry as a proxy
for forgetting. Our margin identity (Proposition 1) makes explicit the output-space quantity that
such constraints are ultimately trying to protect.
Representation-level analyses of forgetting.
Ramasesh et al. [24] localise forgetting to
higher layers through representation similarity; Davari et al. [5] show that representation forgetting
measured by linear probes can be much smaller than accuracy loss suggests. CKA [14] is the
standard similarity measure; we report it only as an auxiliary diagnostic because it is invariant to
isotropic scaling and orthogonal transforms, both of which change logits under a frozen head.
Memory-augmented and sparse-update continual learning.
Product-key and memory
layers [2, 15] provide large, sparsely addressed parameter stores; Lin et al. [18] argue that sparse
memory fine-tuning forgets less than LoRA or full fine-tuning. Hashed N-gram memories pretrained
jointly with the backbone now appear in production LLMs, including the model we study [23], whose
report motivates deterministic addressing partly by cheap offloading. We study such a memory as a
continual-update channel. Its frozen backbone does not keep learned-task representations fixed, and
low generic drift does not imply better retention.
Margins and robustness.
Logit margins are classical in generalisation and robustness analysis [1].
We use the exact relation between margin change and drift projected onto head-difference vectors
as a bookkeeping device for forgetting, closely related to how certified-robustness arguments bound
prediction changes by input perturbation norms.
Merging and update placement.
Merging task-specific weight increments into a shared model
is studied as task arithmetic [12]; our merged-LoRA arm applies it sequentially, one task at a time.
Forgetting has also been shown to depend on how an update is obtained, not only on its size [25].
We vary where the update is placed (a sparse memory table, one continually trained adapter, or a
sequence of merged adapters) under identical data, and measure where in the network the resulting
drift appears.
3


---

3
Representation Drift and How It Enters Forgetting
Notation follows our experiment reports. Ft is the checkpoint after task t (F0 is the pretrained
base). For a fixed input context x and fixed token position, ht(x) is the hidden state entering the
frozen LM head U, so that
zt(x) = Uht(x) + b.
(1)
All three update mechanisms studied here leave U and b untouched, so every change in logits is a
change in h.
3.1
Keys, responses, and two sources of change
Our K and k follow the key–value view of knowledge editing [6, 21]. A linear layer is read as a map
from keys to values, and the layer’s input representation is the key. Stacking the keys of several
inputs as rows gives a matrix K, and the layer’s response to them is
Y = KW.
(2)
Big K denotes the keys of existing knowledge that should be preserved, the K0 of AlphaEdit.
Complete world knowledge cannot be observed, so we use fixed generic math and code probes as its
proxy. Small k denotes the keys of newly written knowledge. Once task i has been learned, its keys
ki also become old keys to be preserved.
For the same inputs, the change of response between rounds i and t splits exactly as
Yt −Yi =
Ki (Wt −Wi)
|
{z
}
weight change at fixed keys
+
(Kt −Ki) Wt
|
{z
}
key drift through new weights
.
(3)
AlphaEdit’s null-space constraint K0 ∆W = 0 controls the first term. The second term comes from
the keys themselves. Every layer’s input is that layer’s key, so any upstream parameter update
reaches downstream layers as key drift, even if their weights never change.
The three update mechanisms occupy different places in Equation (3). Both LoRA variants
have both terms at every adapted layer. N-gram memory changes no weight outside the table, so
the first term vanishes at every layer and only key drift remains, originating at the table injection
in the second layer. The LM head is frozen in all three. At the head the key is h and the response
is the logits, so the first term is identically zero and
Zt −Zi = (Kt −Ki) U⊤:
(4)
every output change comes from key drift. What this paper calls representation drift is the key
drift Kt −Ki at each layer, and in particular at the head. We first define how to measure it and
then show how it enters loss and predictions exactly.
3.2
Two references: big K and small ki
Task i has a fixed probe set Si. For t > i define the drift of an old-task input and its pooled
magnitude,
et
i(x) = ht(x) −hi(x),
Dt
i =

1
|Si|
X
x∈Si
∥et
i(x)∥2
2


1/2
.
(5)
This is small-ki drift: task-i inputs, measured from the checkpoint at which task i was just learned.
Big-K drift replaces the reference by F0 and the inputs by a fixed generic probe set (math, code)
4


---

that no task trains on. Neither the change F0 →Fi on Si (that is learning) nor big-K drift is a
measurement of forgetting of task i. In tables we report the scale-free pooled ratio
rF = ∥Kafter −Kbefore∥F
∥Kbefore∥F
=
 P
x ∥hafter(x) −hbefore(x)∥2
P
x ∥hbefore(x)∥2
!1/2
,
(6)
where the rows of K are the keys of these inputs at that site. rF is an energy-weighted average of
per-query relative drifts, not their arithmetic mean, and a percentage of representation change, not
of lost knowledge.
Length and direction.
For a fixed query with old representation k and new representation k′,
let ρ = ∥k′∥/∥k∥and c = cos(k, k′). Then exactly
∥k′ −k∥2
∥k∥2
= (ρ −1)2 + 2ρ(1 −c).
(7)
The first term is radial (length) change, the second directional change. Pooled decompositions
weight by old energy; per-query percentages must not be averaged first and then substituted. Linear
CKA [14] is reported only as an auxiliary statistic: it is invariant to isotropic scaling and orthogonal
rotation, both of which change what a fixed head reads.
3.3
Accumulation across tasks
Adjacent-round drift on the same input is vi,j(x) = hj(x)−hj−1(x). On a fixed input the telescoping
identity is exact:
et
i(x) =
t
X
j=i+1
vi,j(x),
∥et
i∥2 =
X
j
∥vi,j∥2 + 2
X
a<b
⟨vi,a, vi,b⟩.
(8)
The inner products are the inter-task interference terms: aligned successive updates make drift grow
linearly in the number of tasks, mutually orthogonal ones only as its square root, and opposing
ones cancel so that the endpoint drift is smaller than the sum of per-round drifts. For a probe set
stacked as K(j) with vj = K(j) −K(j−1), normalise by the base energy,
Dt = ∥K(t) −K(0)∥2
F
∥K(0)∥2
F
,
Ej =
∥vj∥2
F
∥K(0)∥2
F
,
Rt =
Dt
Pt
j=1 Ej
.
(9)
Rt < 1 iff the cross-round inner products sum to a negative number (net cancellation), Rt > 1 iff
they sum to a positive number; Rt = 1 does not imply pairwise orthogonality. Archived adjacent
drifts are normalised by ∥K(j−1)∥F , so Ej is recovered by first multiplying by the pooled norm ratio
qj−1 = ∥K(j−1)∥F /∥K(0)∥F and then squaring. Moreover Dj −Dj−1 −Ej equals twice the inner
product of round j’s change with the accumulated change, so dividing by 2
pDj−1Ej gives their
overall cosine.
3.4
From drift to loss and to argmax
With a frozen head the logit change at the head input is exact,
δzt(x) = zt(x) −zi(x) = Uet
i(x).
(10)
5


---

For a teacher-forced answer token y, since ∇zℓ= p −ey has norm at most
√
2, the cross-entropy
obeys the strict bound
|ℓ(zt, y) −ℓ(zi, y)| ≤
√
2 ∥U∥2 ∥et
i∥2,
(11)
and averaging over the task set (Jensen) gives
|∆NLLt
i| ≤
√
2 ∥U∥2 Dt
i.
(12)
This bound connects small-ki drift directly to retention loss. Big-K drift cannot replace Dt
i in it
unless one additionally assumes that old-task queries are distributed like the generic probes.
The bound ignores direction. The second-order expansion makes this explicit:
∆ℓ= g⊤
i et
i + 1
2(et
i)⊤U⊤ΛiUet
i + O(∥et
i∥3),
gi = U⊤(pi −ey),
Λi = diag(pi) −pip⊤
i .
(13)
Drift that falls in the approximate null space of gi leaves NLL unchanged to first order; small drift
along a sensitive head direction can cause large forgetting.
For greedy correctness, let the acquisition token margin be mi(x) = zi,y −maxc̸=y zi,c. Because
every pairwise logit gap changes by at most ∥uy −uc∥∥e∥≤2∥U∥2→∞∥e∥,
2∥U∥2→∞∥et
i(x)∥2 < mi(x) =⇒the greedy prediction of this token is unchanged.
(14)
If every teacher-forced answer token satisfies Equation (14), exact match of the whole answer is
preserved. This is conservative and sufficient only; generation accuracy must be measured separately
and cannot be inferred from NLL.
3.5
Margin budget along a fixed decision boundary
Equation (14) uses only ∥e∥. The exact statement keeps direction. For reference token y and
competitor c̸ = y with head rows uy, uc, define
ayc = uy −uc,
mi,c = zi,y −zi,c,
di,c =
mi,c
∥ayc∥2
,
Ht
i,c = −a⊤
ycet
i
∥ayc∥2
.
(15)
Proposition 1 (Margin identity). With a frozen affine head, for every fixed competitor c,
mt,c = mi,c + a⊤
ycet
i = mi,c −∥ayc∥2Ht
i,c.
(16)
Hence Ht
i,c = di,c means y and c tie, and Ht
i,c > di,c means c strictly overtakes y.
Proof. mt,c = a⊤
ycht + (by −bc) = a⊤
yc(hi + et
i) + (by −bc) = mi,c + a⊤
ycet
i.
This involves no linearisation; H/d and the margin change are the same quantity, not a prediction
and its validation. When the reference token strictly leads the full vocabulary at Fi (all mi,c > 0,
outside numerical tolerance), di = minc̸=y, ayc̸=0 di,c is a safe radius: ∥et
i∥< di preserves the argmax
(Cauchy–Schwarz), which sharpens Equation (14). Positions where y already trails or ties at Fi
receive no safe radius and are counted separately. The geometrically nearest competitor
cgeom = arg
min
c̸=y, ∥ayc∥2>0
mi,c
∥ayc∥2
(17)
need not be the second-highest logit, and the full-vocabulary argmax at Ft must be recomputed
because a different token may overtake.
6


---

Along a fixed boundary the telescoping identity becomes exact per round:
Ht
i,c =
t
X
j=i+1
Hi,j,c,
Hi,j,c = −n⊤
yc(hj −hj−1),
nyc =
ayc
∥ayc∥2
.
(18)
We summarise sign coherence by Hnet = P
j Hi,j,c, Hgross = P
j |Hi,j,c| and the cancellation coefficient
κ = 1 −|Hnet|/Hgross (for Hgross > 0). A positive projection consumes margin and a negative one
adds to it.
Sequences.
At answer positions s where the reference token strictly leads at Fi, let ρt
i,s =
∥et
i,s∥2/di,s and ρt
i,seq = maxs ρt
i,s. If the greedy output at Fi equals the reference sequence with
every step strictly leading, and decoding at Ft uses the same prompt, head, greedy rule and
termination, then ρt
i,s < 1 at every position (including the end token) suffices to preserve the whole
sequence. The true first flip is where the greedy trajectory first departs; after it, model change and
prefix change are mixed. Initial fragility and later erosion are separated by reporting the distribution
of acquisition radii di,s together with the normalised budget
F t,∗
i
= max
s,c
Ht
i,s,c
di,s,c
,
(19)
where F t,∗
i
≥1 means some boundary is reached under the reference prefix.
3.6
Where drift comes from: layers and update structure
Layer recursion.
For the same full token sequence, let Fℓ, F ′
ℓbe block ℓbefore and after an
update and Eℓthe representation change entering it. Exactly,
Eℓ+1 = Sℓ+ Pℓ,
Sℓ= F ′
ℓ(Kℓ) −Fℓ(Kℓ),
Pℓ= F ′
ℓ(Kℓ+ Eℓ) −F ′
ℓ(Kℓ),
(20)
with ∥Eℓ+1∥2
F = ∥Sℓ∥2
F + ∥Pℓ∥2
F + 2⟨Sℓ, Pℓ⟩F . Locally Eℓ+1 ≈JℓEℓ+ Sℓ: upstream drift propagates
through the block Jacobian and the block’s own parameter change adds to it. Layer curves can
therefore amplify, attenuate or cancel, and the difference between adjacent layer curves is not that
layer’s causal contribution. The recursion separates where drift is created (Sℓ, the source at layer ℓ)
from how it is carried (Pℓ), and gives an accumulation bound.
Proposition 2 (Cross-layer accumulation). Let Lj be a Lipschitz constant of the updated block F ′
j
on the segment between Kj and Kj + Ej, and let the input to the first block be unchanged (E1 = 0).
Then
∥EL+1∥F ≤
L
X
ℓ=1

L
Y
j=ℓ+1
Lj

∥Sℓ∥F .
(21)
Proof. From Equation (20), ∥Eℓ+1∥F ≤∥Sℓ∥F +∥Pℓ∥F ≤∥Sℓ∥F +Lℓ∥Eℓ∥F ; iterate from E1 = 0.
Drift at the output is thus a sum over sources, each amplified by the blocks after it. The bound
is conservative: directions and cancellation between sources matter, and depth alone does not
determine forgetting. Its value here is structural, because the three update mechanisms have very
different sets of sources.
• LoRA: every adapted block changes, so Sℓ̸ = 0 at every depth and drift is created throughout
the network.
7


---

• N-gram memory: every block except the table injection is frozen, so Sℓ= 0 for all ℓbut one.
The whole right-hand side reduces to a single term, the injection change ∆J(x) of Equation (28)
carried through the frozen suffix. In particular, if ∆J(x) = 0 for an input, its representation is
unchanged at every depth. This is the memory-table analogue of the layer-wise induction used
for null-space continual learning [26].
How large that single source is depends on the input: small for generic text, which rarely reads the
rows a task writes, and large for old-task inputs, which share rows with later tasks. Section 5.6
shows both in the layer profile.
If round j injects a direct perturbation s(i)
ℓ,j at layer ℓon old-task inputs, then to first order
et
i ≈
t
X
j=i+1
X
ℓ
J(t)
L←ℓ+1s(i)
ℓ,j + higher-order terms.
(22)
Unlike Equations (7), (8) and (16), this is an approximation, and any attribution built on it must
first report its reconstruction residual ∥v −ˆv∥/∥v∥.
For a block captured at its input and aggregate output, the endpoint change splits exactly as
F (7)
ℓ
(k(7)
ℓ) −F (0)
ℓ
(k(0)
ℓ) = F (0)
ℓ
(k(7)
ℓ) −F (0)
ℓ
(k(0)
ℓ)
|
{z
}
input change, old map
+ F (7)
ℓ
(k(7)
ℓ) −F (0)
ℓ
(k(7)
ℓ)
|
{z
}
map change, new input
.
(23)
Our captures contain the left-hand side and the input change but not the cross-forward F (0)
ℓ
(k(7)
ℓ),
so the two terms are not separately identified.
Shared LoRA.
If the same factors C, D keep training, the round-j weight change is exactly
∆Wℓ,j = δCℓ,jDℓ,j−1 + Cℓ,j−1δDℓ,j + δCℓ,jδDℓ,j,
(24)
decoder overwrite, re-encoding of old codes, and coupling. The first two terms are first order in the
current change because the old factors are given, not small, and they enter s(i)
ℓ,j = ∆Wℓ,jK(i)
ℓ
directly.
The chain ∥δD∥→∥δDki∥→∥CδDki∥→∥JCδDki∥→−n⊤JCδDki separates code overlap and
functional change from signed boundary harm; only the last is forgetting-relevant.
Fresh/merged LoRA.
The old factors are frozen inside the merged weights and each round adds
∆Wℓ,j = Cℓ,jDℓ,j. There is no direct overwrite of old factors, but the new adapter still responds to
old inputs, and cross-layer Jacobians and higher-order products remain. The two arms also differ
in expressible capacity: the cumulative update has rank at most r per module for shared LoRA
and at most tr for merged LoRA. For fixed layer inputs K1, K2 and target increments Z1, Z2, any
cumulative update C of rank at most r that fits the new task to ∥CK2 −Z2∥2
F ≤ε2 must incur
∥CK1 −Z1∥2
F ≥
hX
j>r
σj([Z1 Z2])2 −ε2i
+,
(25)
by the Eckart–Young bound on rank(C[K1 K2]) ≤r. Increasing cumulative rank removes this
obstruction but does not guarantee that optimisation finds a low-interference solution.
8


---

Rank does not set the disturbance; direction does.
Write a weight change in the row
convention of Equation (2) through its SVD, D = P
j σjvju⊤
j , with vj input-side and uj output-side
unit vectors. Its effect on old keys K, the first term of Equation (3), is exactly
∥KD∥2
F =
X
j
σ2
j ∥Kvj∥2
2,
(26)
because KD = P
j σj(Kvj)u⊤
j and the uj are orthonormal. The disturbance is set by the size of
each update direction and by how strongly the old keys project onto it, not by the number of
directions. A low-rank update that points at old keys can be very harmful; a high-rank update
orthogonal to them is harmless. It vanishes exactly when Kvj = 0 for every direction with σj > 0,
which is the null-space condition of AlphaEdit.
When task updates must overlap.
Let the update written by task j be vec(∆Wj) = αjuj
with ∥uj∥2 = 1, for m tasks, and let q = rank[u1, . . . , um] be the dimension they jointly span. Then
1
m(m −1)
X
a̸=b
(u⊤
a ub)2 ≥max

0,
m −q
q(m −1)

.
(27)
This is Welch’s bound [28]. The Gram matrix G = [u⊤
a ub] has trace m and rank at most q, so
∥G∥2
F ≥m2/q, and the off-diagonal part is ∥G∥2
F −m. Once the number of tasks exceeds the
dimension available to their updates, squared overlap between task directions is unavoidable. For
m ≤q the bound is zero: orthogonal updates are geometrically possible, though optimisation need
not find them. This is the structural difference between the two LoRA arms. Shared routes
every task through one factor pair, so successive increments reuse the same row and column spaces
(Equation (24)); Merged gives every task fresh factors, so the jointly spanned dimension can keep
growing. Overlap in parameter space is not yet forgetting. It matters only through Equation (26),
by landing on directions that old keys use, and then through Equation (16), by projecting onto the
old task’s head-difference directions.
N-gram memory.
The native table T is read at a fixed early layer through addresses determined
by hashed token bigrams and trigrams of the raw input; the backbone, gates and projections are
frozen. Write the read pattern of context x as a fixed sparse matrix R(x). Upstream of the injection
nothing changes, so the injected representation changes only through the table,
∆J(x) = G
 H(x), (T + ∆T)R(x)
 −G
 H(x), TR(x)
,
(28)
where G is the frozen read-out (key/value projections, hidden-dependent gate and causal convolution)
and H(x) the unchanged pre-injection state. If ∆TR(x) = 0 on the full causal prefix of an old input,
its entire forward pass is unchanged. Any read of a row written by a later task—a genuinely shared
N-gram or a hash collision—breaks this, and the change then propagates through the frozen suffix,
whose inputs have moved even though its parameters have not. N-gram memory can therefore have
very small big-K drift (generic inputs rarely read rows the tasks write) and still large small-ki drift
(old-task inputs share rows with new tasks).
4
Experimental Setup
Backbone.
We use Qwen3.8-Flash-Next [23] (revision de4b8e4), a 48-layer hybrid-attention (gated
linear attention and full attention) MoE language model with 125B backbone parameters (about
9


---

6B active), 512 routed experts with top-10 routing plus shared experts, and four hyper-connection
residual streams that are contracted before the LM head. It carries a native, jointly pretrained
N-gram memory of 51.2B parameters: a table of 320,001,536 rows of width 160, read at the second
layer. Each token reads one row from each of 8 bigram and 8 trigram hash heads (16 rows, 2,560
values). The addresses depend only on raw token IDs (with history reset after EOS), and the rows
pass through key/value projections, a hidden-state-dependent gate and a causal convolution before
being injected into the residual streams. The model runs in BF16. The final normalisation is the
identity, so the captured head input determines logits exactly.
Task chain.
Seven tasks, learned in the fixed order SME →NAV →HYE →AFB →SQI →
MKD →AMH, are morphological inflection in Northern Sami, Navajo, Armenian, Gulf Arabic,
Albanian, Macedonian and Amharic from the SIGMORPHON 2023 shared task [8] (pinned commit;
lemma, features→form). Each task uses 5,000 training examples in a chat template with thinking
disabled; the model must output only the inflected form. Train and test lemma strings are disjoint
(exact string match). We chose inflection because it is genuinely new knowledge for the base model
in most of these languages, admits unambiguous exact-match scoring, and yields short answers
whose first token dominates correctness.
Update mechanisms.
All arms use identical data order and budget: 125 optimizer steps ×
global batch 40 = 5,000 presentations per task, max length 128, Adam (β = (0.9, 0.95), ϵ = 10−8),
peak learning rate 10−4 with cosine decay to zero, no warm-up, no weight decay, gradient clipping
at 1. Optimizer and scheduler are reset at every task boundary.
• N-gram: only the table rows addressed by the current task’s training inputs are trainable,
starting from their current (pretrained or previously updated) values. Each task has between
51,018 (AMH) and 149,809 (AFB) rows, i.e. 8–24M scalars. All seven tasks together touch
666,747 rows (0.21% of the table), and each task shares between 1,277 and 4,911 rows with earlier
tasks. Hashing, gate, read-out projections, router, all Transformer weights and the head are
frozen. On the touched rows, Adam is exactly equivalent to full-table Adam, because untouched
rows have identically zero gradient and optimizer state. Row values accumulate across tasks.
• Shared LoRA: rank 16, α = 64, no dropout, on six projection families (in_proj, out_proj,
linear_qkv, linear_proj, linear_fc1, linear_fc2) covering linear attention, full atten-
tion, and every routed and shared expert. The same factors continue training on each new
task.
• Merged LoRA: identical configuration, but a freshly initialised adapter per task that is merged
into the backbone weights at the end of the task.
The arms are matched in data exposure, not in parameter count: a full rank-16 adapter bank has
2.79B parameters (80.2M on any single token’s path), while a task’s N-gram support is ∼16M
scalars. We return to this in Section 7.
Probes and captures.
At F0 and after every task we run forward-only passes on fixed probes:
• Generic: 128 MATH-500 problems [10, 17] with official solutions and 64 HumanEval problems [4]
with canonical solutions, in raw (template-free) format.
• Task: 64 fixed held-out probes for each task learned so far, in the training template.
10


---

We record teacher-forced token-weighted NLL of the reference answer and capture, at the last
prompt token (and at 25/50/75/100% answer positions for math/code), the output of all 48 decoder
layers (four streams concatenated), the post-merge representation and the LM-head input. For the
final checkpoints we additionally capture the input and aggregate output of every MLP/MoE block.
Drift uses pooled relative RMS (Equation (6)); we also report mean cosine, pooled norm ratio, and
linear CKA.
Generative evaluation.
Checkpoints are exported to Hugging Face format: the N-gram arm
with its accumulated rows written into the table, Shared with its current factors merged once,
and Merged with its per-task adapters merged in order.
They are then decoded greedily
(do_sample=false) with vLLM (TP8, BF16, N-gram table offloaded to CPU). Task accuracy
is exact match on the 64 task probes. Math accuracy (the same 128 problems) uses symbolic
answer verification (math-verify 0.9.0). Code accuracy (the same 64 problems) is pass@1 under
WASI-sandboxed execution; the sandbox passes all 164 canonical HumanEval solutions.
Second model.
An earlier, separately run chain on Llama-3-8B [9] uses the same seven tasks
and order with a different memory arm: a zero-initialised hashed bigram/trigram lookup table
(2 × 262,144 buckets of width 4,096) added at the input of the second decoder block. Llama-3-8B
has no native memory: the table is new, and because it starts at zero the model at the start of
training is exactly the base, whose existing knowledge does not depend on the table at all. This
differs from Qwen3.8, whose table was pretrained together with the backbone and already carries
part of the model’s function, so that updating its rows edits an existing function. It also uses rank-8
LoRA on all seven projection types, 625 steps × batch 8 at learning rate 2 × 10−4, seed 17, and
representations after the final RMSNorm. We use it only to check whether the direction of the
Qwen findings replicates (Section A).
Infrastructure.
Training uses Megatron-SWIFT on 4×H200 with TP2/PP2/EP2. Checkpoints
are verified by hash before being used as the parent of the next task. All numbers are from a single
training seed (data seed 42); probe sets are fixed across methods and rounds.
5
Representation Drift Results
All three arms completed the seven-round chain with identical boundary evaluations. Unless stated
otherwise, numbers are at the LM-head input, last prompt token; drift is pooled relative RMS
(Equation (6)).
5.1
Big K after seven rounds
Table 1 is the central representation result. On both generic domains the ordering is
N-gram < Merged < Shared,
and it holds for the pooled RMS, the median and the 95th percentile of per-query drift, and the
mean cosine. Three further observations:
1. Drift is directional. In every arm and domain more than 96% of squared drift is the directional
term 2ρ(1 −c) of Equation (7). N-gram keeps norms almost exactly (ρ ≈1.00). Shared on
math is the only case with a substantial radial component: the pooled norm shrinks by 17%.
11


---

Table 1: Big-K drift F0 →F7 (head input, last prompt token). Radial and directional energies are
the two terms of Equation (7), pooled and normalised by base energy; their sum is the squared
relative RMS. ∆NLL is the teacher-forced reference-answer loss change relative to F0.
Method
Domain
Rel. RMS
Mean cos
Median
P95
Norm ratio
Radial
Direct.
Dir. share
CKA
∆NLL
N-gram
math
26.37%
0.9658
13.48%
58.37%
1.0059
0.0020
0.0676
97.17%
0.627
−0.0119
code
15.13%
0.9889
12.49%
23.58%
1.0019
0.0003
0.0226
98.66%
0.986
−0.0022
Shared
math
97.33%
0.4502
97.75%
105.27%
0.8274
0.0358
0.9116
96.22%
0.366
−0.1001
code
38.90%
0.9263
34.76%
54.89%
1.0003
0.0014
0.1499
99.09%
0.896
+0.0802
Merged
math
64.36%
0.7894
52.64%
113.52%
0.9275
0.0143
0.3998
96.54%
0.127
−0.0865
code
34.98%
0.9387
32.58%
51.61%
0.9639
0.0026
0.1198
97.86%
0.913
+0.0246
2. Drift is not NLL. Shared moves math representations the most (97%) and also lowers math
reference NLL the most (−0.100). Code NLL rises for both LoRA arms. Representation change
at a generic probe can be neutral or even helpful for the loss, as Equation (13) allows.
3. CKA disagrees with direct drift. Merged has the lowest math CKA (0.13) although its
direct drift is below Shared’s. CKA measures sample-geometry similarity up to rotation and
scale and is not a substitute for drift in the head’s coordinates.
Figure 1: Big-K drift by layer for the seven rounds (one column per method, one row per generic
domain; one curve per round).
Layer curves are four residual streams concatenated (10,240
dimensions).
5.2
Seven-round trajectory
Table 2 (all rounds in Section B) shows three different temporal profiles. N-gram jumps to about
25% after the first task and then stays flat (21–26% on math for all seven rounds). Shared grows
almost monotonically on math (45%→97%) and monotonically on code. Merged is non-monotone:
12


---

Table 2: Big-K trajectory at the head input: drift from F0 after each round, and reference NLL
(base: math 0.888, code 0.201).
Math
Code
Method
Round
Drift
Cos
NLL
Drift
Cos
NLL
N-gram
1
25.21%
0.968
0.881
13.40%
0.991
0.200
3
24.81%
0.970
0.878
14.66%
0.990
0.199
5
26.14%
0.966
0.874
14.22%
0.990
0.199
7
26.37%
0.966
0.876
15.13%
0.989
0.199
Shared
1
45.51%
0.893
0.750
24.91%
0.969
0.210
3
74.71%
0.717
0.769
30.10%
0.955
0.217
5
66.64%
0.760
0.619
35.53%
0.937
0.205
7
97.33%
0.450
0.788
38.90%
0.926
0.281
Merged
1
45.01%
0.897
0.781
25.12%
0.969
0.209
3
52.41%
0.856
0.796
29.37%
0.958
0.215
5
76.22%
0.714
0.916
32.41%
0.948
0.224
7
64.36%
0.789
0.804
34.98%
0.939
0.224
math drift peaks at round 5 (76%) and falls back to 64% by round 7. After the first round, Shared
and Merged are nearly indistinguishable (45.5% vs 45.0%), as they must be: the first task trains
an identically configured adapter in both. The arms diverge only once Shared starts rewriting its
factors.
5.3
Accumulation and cancellation across rounds
Table 3: Cross-round accumulation of big-K drift (Equation (9)). P
j Ej is the sum of per-round
squared changes, normalised by base energy; R7 < 1 means the cross-round inner products sum to a
negative number.
Method
Domain
Final drift
P
j≤7 Ej
R7
N-gram
math
26.37%
0.4781
0.145
code
15.13%
0.1207
0.190
Shared
math
97.33%
1.6958
0.559
code
38.90%
0.3865
0.392
Merged
math
64.36%
1.0220
0.405
code
34.98%
0.3019
0.405
Using Equations (8) and (9) we recover the total cross-round inner product from archived
pooled norms (Table 3, Figure 2). In all arms R7 < 1: successive task updates partly undo one
another on the generic probes. The decisive point concerns N-gram. Its per-round changes are
not small—P
j Ej = 0.48 on math, about half of Merged’s—but they cancel much more strongly
(R7 = 0.145). Its small endpoint drift is therefore not “nothing changed in each round”; it is
“each round changed the generic representation and the next round largely moved it back.” For
long-running learning this is the property that matters: the table’s effect on the base behaves like
a bounded excursion rather than a random walk. The same back-and-forth acts on learned-task
inputs, which is where its cost appears (Section 5.6).
13


---

1
2
3
4
5
6
7
Task boundary
0
20
40
60
80
100
Pooled relative drift (%)
Math | distance from original base
N-gram
Shared LoRA
Fresh / merged LoRA
1
2
3
4
5
6
7
Task boundary
0.0
0.2
0.4
0.6
0.8
1.0
Net / summed step drift energy
Math | net accumulation ratio
1
2
3
4
5
6
7
Task boundary
0
5
10
15
20
25
30
35
40
Pooled relative drift (%)
Code | distance from original base
N-gram
Shared LoRA
Fresh / merged LoRA
1
2
3
4
5
6
7
Task boundary
0.0
0.2
0.4
0.6
0.8
1.0
Net / summed step drift energy
Code | net accumulation ratio
Across-task accumulation at the frozen head input | last prompt
Same-query archived metrics; lower ratio means greater net cancellation, not necessarily less forgetting
Figure 2: Left: big-K drift relative to F0 over seven rounds. Right: cosine between round j’s change
and the change accumulated before it, (Dj −Dj−1 −Ej)/(2
pDj−1Ej).
14


---

5.4
Depth: residual streams and MLP/MoE blocks
Table 4: Big-K drift F0 →F7 at sampled layer outputs (four residual streams concatenated).
Method
Domain
L1
L12
L24
L36
L48
Max layer
N-gram
math
0.00%
4.25%
8.62%
34.52%
36.39%
L48 (36.39%)
code
0.00%
2.50%
5.06%
14.63%
14.57%
L43 (15.53%)
Shared
math
12.76%
39.50%
56.58%
139.45%
99.73%
L36 (139.45%)
code
6.68%
18.45%
25.35%
41.55%
38.78%
L34 (41.82%)
Merged
math
9.09%
19.02%
34.40%
84.60%
81.09%
L36 (84.60%)
code
7.13%
14.76%
19.78%
35.31%
34.80%
L42 (35.84%)
Table 4 and Figure 1 show where in depth the arms separate. N-gram drift is exactly zero
at layer 1, which precedes the memory read, and stays below 10% through layer 25 even though
the table is read at layer 2. It grows mainly in the last third of the network. Both LoRA arms
change every layer directly, so drift is already 7–13% at layer 1 and rises with depth. All arms peak
between layers 34 and 48, and for both LoRA arms on math the drift falls after its peak (Shared
139%→100% from L36 to L48). This is the behaviour described by Equation (20): upstream drift
is propagated, re-mixed and partly cancelled by later blocks, so drift norms need not increase
monotonically with depth.
At the four final endpoints (base and the three F7) we additionally captured the input and the
aggregated output of every MLP/MoE block, before the hyper-connection injection, for 640 fixed
queries (Table 5 and Figure 3).
Table 5: Big-K drift F0 →F7 at MLP/MoE blocks (%). “Median” is the median over the 48
per-layer pooled drifts. Input and output are each normalised by their own base energy, so the two
columns must not be subtracted.
Domain
Hook
Method
L1
L48
Median
math
input
N-gram
0.05
22.21
10.59
Shared
14.19
88.73
72.91
Merged
8.34
58.05
44.41
output
N-gram
0.09
74.50
19.72
Shared
17.93
123.51
112.13
Merged
11.48
151.05
69.87
code
input
N-gram
0.09
14.50
6.15
Shared
11.43
37.55
29.76
Merged
7.59
34.61
23.88
output
N-gram
0.12
12.19
11.31
Shared
7.86
33.91
43.52
Merged
8.44
30.45
36.14
The method ordering of Table 1 holds block by block, at both hooks. On math the median output
drift exceeds the median input drift in every arm (N-gram 10.6%→19.7%, Shared 72.9%→112.1%,
Merged 44.4%→69.9%). For N-gram this is a pure propagation effect, since its MoE parameters
are frozen and only the second term of Equation (23) vanishes. At the last layer the N-gram MoE
output drifts by 74.5% on math although its input drifts by only 22.2%. For the LoRA arms both
terms of Equation (23) are present.
15


---

Decoder layer (1-based)
0
20
40
60
80
100
120
Pooled relative drift (%)
Math | MLP input | n=128
N-gram
Shared LoRA
Fresh / merged LoRA
Decoder layer (1-based)
0
25
50
75
100
125
150
175
200
Pooled relative drift (%)
Math | MLP aggregate output | n=128
N-gram
Shared LoRA
Fresh / merged LoRA
1
8
16
24
32
40
48
Decoder layer (1-based)
0
20
40
60
80
100
120
Pooled relative drift (%)
Code | MLP input | n=64
1
8
16
24
32
40
48
Decoder layer (1-based)
0
25
50
75
100
125
150
175
200
Pooled relative drift (%)
Code | MLP aggregate output | n=64
Big K | Base to final checkpoint | last prompt
Same-site paired observations; input/output curves are not causal layer gains
Figure 3: Per-layer big-K drift F0 →F7 at MLP/MoE block input and aggregated output.
5.5
Answer positions
For math and code we also captured the positions at 25/50/75/100% of the reference answer under
teacher forcing. Head-input big-K drift is position-dependent (Table 6): it is lower inside the answer
than at the last prompt token for every arm (e.g. Shared math 97% at the prompt vs 50–63%
inside the answer), and largest at the final answer position for both LoRA arms. The method
ordering is unchanged at every position.
Table 6: Big-K drift F0 →F7 at the head input by reference-answer position.
Method
Domain
q25
q50
q75
q100
N-gram
math
12.66%
13.66%
20.40%
15.00%
code
7.95%
10.65%
10.68%
17.95%
Shared
math
50.49%
53.08%
54.36%
62.74%
code
21.98%
26.21%
27.95%
53.03%
Merged
math
36.69%
40.50%
39.90%
46.41%
code
20.47%
23.21%
23.28%
43.42%
5.6
Small ki: from acquisition to the end of the chain
Table 7 measures what big K cannot: how much the representations of learned tasks move after
they are learned. Three facts stand out.
1. The big-K ordering does not transfer. N-gram has the smallest big-K drift but not the
smallest small-ki drift on any task; it is the largest of the three on five of six tasks (all but SME).
16


---

Table 7: Small-ki drift Fi →F7 on each learned task’s 64 probes (head input, last prompt token),
with the change in teacher-forced reference NLL over the same interval. AMH is learned last and
has no subsequent training.
Task
Method
Rel. RMS
Mean cos
Median
P95
Norm ratio
Dir. share
∆NLL
SME
N-gram
40.43%
0.9205
38.00%
56.79%
1.0099
98.18%
+0.236
Shared
50.65%
0.8836
47.32%
67.03%
1.0703
96.83%
+0.356
Merged
37.19%
0.9313
36.03%
49.12%
0.9972
99.01%
+0.210
NAV
N-gram
61.98%
0.8172
61.78%
77.22%
1.0465
99.07%
+0.753
Shared
60.60%
0.8223
57.86%
78.65%
1.0262
99.11%
+1.018
Merged
47.43%
0.8881
46.34%
59.10%
1.0022
99.37%
+0.450
HYE
N-gram
32.61%
0.9462
30.37%
44.01%
0.9577
97.54%
+0.082
Shared
28.37%
0.9599
25.48%
43.61%
0.9963
99.57%
+0.123
Merged
25.84%
0.9663
23.67%
34.99%
0.9585
96.80%
+0.100
AFB
N-gram
37.26%
0.9320
36.31%
51.61%
1.0155
99.29%
+0.353
Shared
30.30%
0.9542
29.05%
39.09%
0.9957
99.14%
+0.163
Merged
22.17%
0.9753
21.66%
28.30%
0.9901
98.99%
+0.094
SQI
N-gram
36.77%
0.9324
33.21%
57.49%
0.9982
99.61%
+0.215
Shared
36.30%
0.9359
32.86%
52.75%
1.0236
98.98%
+0.080
Merged
30.31%
0.9554
25.56%
52.39%
1.0115
99.52%
+0.088
MKD
N-gram
26.00%
0.9668
22.76%
39.94%
1.0092
99.51%
+0.192
Shared
23.25%
0.9738
22.06%
30.11%
1.0242
98.46%
+0.057
Merged
16.04%
0.9874
15.45%
22.73%
1.0166
98.41%
+0.003
Mean
N-gram
39.18%
+0.305
Shared
38.25%
+0.300
Merged
29.83%
+0.157
Its mean small-ki drift (39.2%) equals Shared’s (38.3%). Even with a frozen backbone, old-task
representations move by 26–62%.
2. Merged has the smallest small-ki drift on every task, by 3–15 points.
3. Drift is again almost purely directional (96.8–99.6%), with norm ratios within 7% of one.
Older tasks drift more in every arm (SME and NAV have experienced six and five later tasks), but
task age is not the only factor: HYE (third) drifts less than AFB (fourth) under N-gram.
Figure 4 shows where in depth this happens. On generic inputs (a) the three arms are far apart
at every layer, and N-gram stays below 10% until layer 25. On learned-task inputs (b) the three
curves nearly coincide, and N-gram drift appears immediately after the memory read: 19% at layer
5 and 31% at layer 9, against 4% for generic inputs at the same layers. This is the single-source
picture of Proposition 2: N-gram has one source term, the table injection, and it is small on generic
inputs and large on old-task inputs, which read rows that later tasks rewrite. LoRA creates sources
at every layer for both kinds of input. The two LoRA arms show the opposite asymmetry in deep
layers. There they move generic inputs far more than learned-task inputs (Shared 139% vs 54% at
layer 36).
17


---

10
20
30
40
decoder layer
0
20
40
60
80
100
120
140
relative RMS drift (%)
N-gram read
(a) big K: math probes, F0
F7
10
20
30
40
decoder layer
(b) small ki: six learned tasks, Fi
F7
N-gram
Shared LoRA
Fresh/Merged LoRA
Figure 4: Per-layer drift (four residual streams concatenated, last prompt token). (a) Big K on
math probes, F0 →F7. (b) Small ki, mean over the six learned tasks with min–max band, Fi →F7.
The dotted line marks the N-gram read at layer 2.
Figure 5: Small-ki drift Fi →Ft for every learned task i and later round t (head input, last prompt
token).
18


---

6
Behavioural Correlates and Boundary Checks
We now relate the drift measurements to behaviour: teacher-forced NLL, first-token margins, and
greedy generation accuracy.
6.1
Forgetting in NLL, and which drift tracks it
Table 8: Final (F7) reference NLL: generic probes relative to F0, learned tasks relative to their
acquisition checkpoint Fi (mean over the six tasks with later training), next to the mean big-K
drift.
Method
Math NLL
∆
Code NLL
∆
Old-task mean ∆NLL
Big-K drift
N-gram
0.8759
−0.0119
0.1985
−0.0022
+0.3052
20.75%
Shared
0.7877
−0.1001
0.2809
+0.0802
+0.2995
68.12%
Merged
0.8036
−0.0865
0.2244
+0.0246
+0.1573
49.67%
Table 8 contains the paper’s main dissociation. N-gram memory perturbs generic representations
least and leaves generic NLL essentially unchanged. Yet on the tasks it has learned it forgets as
much as Shared LoRA (+0.305 vs +0.300) and nearly twice as much as Merged LoRA (+0.157).
The big-K ordering N-gram < Merged < Shared does not predict the forgetting ordering
Merged < Shared ≈N-gram.
Small-ki drift does (Figure 6). Its means order the arms exactly as their mean ∆NLL, and
per task it tracks ∆NLL across all 18 task–method pairs (Spearman ρ = 0.85). This is what
Equation (12) leads one to expect: forgetting of task i is bounded by Dt
i, which is measured on
task-i inputs from Fi. Big-K drift enters that bound only under the additional assumption that
generic and task inputs move alike, and N-gram memory violates that assumption by construction
(Equation (28)). The same dissociation appears on Llama-3-8B (Figure 6c,d; Section A). There
the memory arm is not native: Llama-3-8B has no N-gram memory, so we added a lookup table
initialised to zero, on which the base model’s knowledge does not depend. Big-K drift is again
smallest for the memory arm (6.1% vs 30.3% merged, 50.5% shared). It is also much smaller than for
Qwen’s native table (26%/15%), consistent with an empty side path perturbing the base less than
editing rows that pretraining already uses. Its mean old-task ∆NLL (+0.69) nevertheless exceeds
Merged’s (+0.46), and small-ki drift again tracks per-task ∆NLL (ρ = 0.71). The dissociation
therefore holds whether the memory is native and pretrained or newly added and empty.
6.2
First-token margins
From the head inputs at F0 and F7 and the frozen head we computed, for every probe, the
full-vocabulary margin of the first reference token, m = zy −maxc̸=y zc, and its paired change
∆m = mfinal −mbase: 1,920 paired observations over 9 domains and 3 arms (Table 9, Figure 7).
On the seven tasks all 21 domain–method medians are positive, because this F0 →F7 comparison
mixes learning with forgetting. The LoRA arms gain about twice the first-token margin of N-gram,
and reference-token argmax rises from 69.6% to 78.3% (N-gram), 85.0% (Shared) and 87.3%
(Merged). On math, where the first reference token is not a training target, both LoRA arms
lower the margin while N-gram leaves it unchanged. Of the 10 math positions where the reference
token strictly led at F0, 4 (N-gram), 9 (Shared) and 10 (Merged) are overtaken at F7. None of
the 60 strictly leading code positions is lost in any arm.
19


---

20
30
40
50
60
70
big-K drift (%, mean of math, code)
0.16
0.18
0.20
0.22
0.24
0.26
0.28
0.30
mean old-task NLL
(a) Qwen3.8: big K
N-gram
Shared LoRA
Fresh/Merged LoRA
20
30
40
50
60
small-ki drift Fi
F7 (%)
0.0
0.2
0.4
0.6
0.8
1.0
task NLL (Fi
F7)
SME
NAV
HYE
AFB
SQI
MKD
SME
NAV
HYE
AFB
SQI
MKD
SME
NAV
HYE
AFB
SQI
MKD
(b) Qwen3.8: small ki, 
s = 0.85
N-gram
Shared LoRA
Fresh/Merged LoRA
10
20
30
40
50
big-K drift (%, mean of math, code)
0.5
0.6
0.7
0.8
0.9
mean old-task NLL
(c) Llama-3-8B: big K
Lookup
Shared LoRA
Fresh/Merged LoRA
20
40
60
80
100
small-ki drift Fi
F7 (%)
0.0
0.5
1.0
1.5
2.0
task NLL (Fi
F7)
SME
NAV
HYE
AFB
SQI
MKD
SME
NAV
HYE
AFB
SQI
MKD
SME
NAV
HYE
AFB
SQI
MKD
(d) Llama-3-8B: small ki, 
s = 0.71
Lookup
Shared LoRA
Fresh/Merged LoRA
Figure 6: Big-K drift does not track forgetting; small-ki drift does. (a,c) Mean old-task ∆NLL
against big-K drift: the memory arm has the smallest generic drift but not the least forgetting.
(b,d) Per-task ∆NLL against small-ki drift Fi →F7 for all 18 task–method pairs. (a,b) Qwen3.8
with native N-gram memory; (c,d) Llama-3-8B with an added zero-initialised lookup table.
Table 9: Median paired change of first-reference-token margin F0 →F7 (logits).
Domain (n)
N-gram
Shared
Merged
Math (128)
+0.036
−4.350
−3.103
Code (64)
+0.011
−0.038
+0.345
Seven tasks (448)
+2.172
+4.067
+4.412
N-gram
Shared LoRA
Fresh / merged LoRA
SME  (n=64)
NAV  (n=64)
HYE  (n=64)
AFB  (n=64)
SQI  (n=64)
MKD  (n=64)
AMH  (n=64)
+1.596
+2.271
+1.736
+0.259
+3.417
+4.616
+3.276
+5.656
+5.592
+0.793
+3.688
+3.821
+1.986
+3.862
+3.955
+2.314
+4.463
+3.998
+2.959
+6.012
+5.456
First reference token | Seven training-task domains
Base to final: acquisition and forgetting are mixed
6
4
2
0
2
4
6
Median paired delta margin (logit units)
Figure 7: Median first-reference-token margin change F0 →F7 by domain and method.
20


---

The pattern matches the big-K picture: on code, which the table updates barely touch, N-gram
memory leaves the base’s decision margins where they were.
6.3
Generative accuracy
All endpoints were exported to Hugging Face format and decoded greedily with vLLM; every sample
was scored (unresolved = 0).
Table 10: Greedy exact match on the 64 probes of each task, from acquisition (Fi) to F7 (percentage-
point change).
Task
N-gram
Shared
Merged
SME
13 →13 (0.0)
32 →22 (−15.6)
31 →27 (−6.3)
NAV
6 →
4 (−3.1)
23 →
8 (−23.4)
23 →11 (−18.8)
HYE
61 →55 (−9.4)
63 →55 (−12.5)
64 →57 (−10.9)
AFB
42 →35 (−10.9)
52 →49 (−4.7)
51 →49 (−3.1)
SQI
50 →45 (−7.8)
53 →46 (−10.9)
51 →48 (−4.7)
MKD
54 →46 (−12.5)
59 →57 (−3.1)
56 →57 (+1.6)
AMH
57
58
60
Mean change (6 tasks)
−7.3
−11.7
−7.0
Cumulative accuracy at F7
56.9%
65.8%
69.0%
Task accuracy (Table 10) agrees with NLL and small-ki drift on the two LoRA arms: Shared
loses more than Merged (−11.7 vs −7.0 pp on average), with the earliest tasks hit hardest. For
N-gram it exposes what drift and NLL do not. N-gram acquires the hard languages much more
weakly (SME 13/64 vs 31–32; NAV 6/64 vs 23), so its small exact-match decline (−7.3 pp) is
bounded by a floor. In the language of Proposition 1, many N-gram probes never had a positive
sequence margin to lose. Low drift combined with weak acquisition is not retention. Cumulative
accuracy after round 7 is 56.9% (N-gram), 65.8% (Shared) and 69.0% (Merged). Its round-3
jump in every arm comes from adding HYE, which all arms learn almost perfectly, so cumulative
curves should not be read as forgetting curves.
Table 11: Generic capabilities at F7: greedy accuracy on 128 MATH-500 problems and 64 HumanEval
problems (pass@1, sandboxed WASI execution), with paired per-problem transitions relative to the
base, and the number of math outputs whose final answer could not be parsed (counted as wrong).
Math acc.
Lost / gained
Unparsed
Code acc.
Lost / gained
Base
71/128 (55.5%)
–
56
51/64 (79.7%)
–
N-gram
68/128 (53.1%)
10 / 7
59
55/64 (85.9%)
0 / 4
Shared
32/128 (25.0%)
47 / 8
96
52/64 (81.3%)
7 / 8
Merged
94/128 (73.4%)
1 / 24
33
55/64 (85.9%)
6 / 10
Generic generation (Table 11) shows that changes are not uniformly degradations. Most of the
math swings coincide with the number of unparsable final answers (base 56, Shared 96, Merged
33). They therefore reflect changes in answer format and termination behaviour at least as much as
changes in mathematical competence. Shared lowers math reference NLL (−0.100) while losing 47
problems: teacher-forced NLL scores one fixed reference solution, whereas generation follows the
model’s own trajectory and an early format change propagates. This is exactly the gap between
21


---

Equation (11) and Equation (14). N-gram changes generic behaviour least on math (10 lost / 7
gained), consistent with its small big-K drift.
7
Discussion: Write Online Knowledge into the Memory
The base is the binding constraint.
An online learner receives an unbounded sequence of
updates. Any channel whose perturbation of the base grows with the number of updates will
eventually erode the capabilities the model was deployed for, however well it learns each update. The
three channels differ sharply in this respect. Shared LoRA’s generic drift grows almost monotonically
(45%→97% on math). Merged LoRA’s also grows, non-monotonically, to 64% with a peak of 76%.
N-gram memory reaches 25% after the first task and stays there (Table 2), because its per-round
changes cancel (R7 = 0.15, Table 3). In generic behaviour it loses no code problem and changes
math NLL by −0.01. Among the three, only N-gram memory has a base perturbation that behaves
like a bounded excursion rather than an accumulating walk.
Why the memory protects the base.
Equation (28) makes this structural rather than empirical.
The table is read through addresses fixed by raw tokens, and everything upstream of the read is
frozen. An input none of whose prefix rows were written is therefore exactly unchanged, at every
layer and in every logit. General math and code text rarely addresses the rows a new task writes, so
generic drift stays small and does not accumulate. In addition, every change lives in an explicit set
of rows: 8–24M values per task, 0.21% of the table for all seven tasks. These rows can be inspected,
versioned and rolled back without touching the 125B-parameter backbone. Neither LoRA variant
offers this: their updates act on every input through every adapted projection.
Where the cost appears, and why it belongs to the memory.
N-gram memory pays in
two places. It acquires difficult tasks more weakly (SME 13/64 vs 31–32 for LoRA), and its stored
tasks interfere: old-task NLL rises as much as with shared LoRA, and small-ki drift is as large.
Equation (28) also locates this cost. Old-task inputs share rows with later tasks: 1,277–4,911 rows
per task by count, and in particular the template and chat-format rows that every example reads,
including at the last prompt token. Later tasks rewrite those rows, and the layer profile shows it:
on old-task inputs N-gram drift reaches 31% within seven layers of the read, while generic inputs
at the same depth move by 4% (Figure 4). Both weaknesses are thus properties of how the memory
is addressed and written, not of the backbone. They can be attacked inside the memory: by keeping
shared template rows fixed or task-specific, by allotting more rows or a larger learning rate to new
content, or by per-row update rules. None of these requires editing the backbone again.
What merged LoRA shows.
Merged LoRA retains learned tasks best and acquires them as
well as shared LoRA. It also improves code generation, and its large math accuracy gain coincides
with fewer unparsable answers. Its old factors are never edited, and its cumulative rank grows as tr,
which avoids the overwrite terms of Equation (24) and the capacity bound of Equation (25). These
are good properties for an offline consolidation step, in which knowledge that has proved stable is
periodically folded into the weights with full regression testing. As an always-on online channel it
moves the base by 50% on generic inputs after seven tasks, and that drift has no mechanism that
bounds it.
A design for long-running online learning.
Our measurements support a two-speed design.
Online updates are written only to the native memory table, with the backbone and head frozen, so
22


---

the base stays fixed and every change is local and reversible. Memory-side mechanisms then manage
interference between stored items. Weight-level consolidation, if used at all, is an occasional offline
step. Small-ki drift and acquisition margins, rather than generic drift alone, are the quantities to
monitor, because forgetting follows them (Figure 6).
Limitations.
Results come from one task order and one seed on Qwen3.8, with Llama-3-8B as
a second, differently configured model. The arms are matched in data and optimisation, not in
parameter count. Margins are measured at the first reference token from the base, not from each
acquisition checkpoint. Layer- and block-level results are observational.
8
Conclusion
We measured, layer by layer and round by round, how three ways of writing new knowledge change
a 48-layer MoE language model over a seven-task continual chain. Updating only the native N-gram
memory changes the base model’s representations two to four times less than LoRA. That change
does not grow with the number of tasks, because successive updates largely cancel, and it leaves
generic behaviour essentially intact. Its cost, weaker acquisition and interference among stored
tasks, shows up on the learned tasks’ own inputs. There forgetting follows small-ki drift, and exact
margin relations explain why. For a model that must keep learning online, these results point to a
clear division of labour: write new knowledge only into the memory, keep the backbone fixed, and
solve interference inside the memory rather than by editing the backbone.
Acknowledgments
The author used AI assistants for drafting text, writing code for analysis and figures, and checking
the literature. All experiments, results and claims were verified by the author, who takes full
responsibility for the content.
References
[1] Peter L. Bartlett, Dylan J. Foster, and Matus J. Telgarsky. Spectrally-normalized margin
bounds for neural networks. In Advances in Neural Information Processing Systems, 2017.
[2] Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi
Ghosh. Memory layers at scale. In International Conference on Machine Learning, 2025.
[3] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard,
Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney,
and John P. Cunningham. LoRA learns less and forgets less. Transactions on Machine Learning
Research, 2024.
[4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto,
Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating
large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
[5] MohammadReza Davari, Nader Asadi, Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky.
Probing representation forgetting in supervised and unsupervised continual learning.
In
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
23


---

[6] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He,
and Tat-Seng Chua. AlphaEdit: Null-space constrained knowledge editing for language models.
In International Conference on Learning Representations, 2025.
[7] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for
continual learning. In International Conference on Artificial Intelligence and Statistics, 2020.
[8] Omer Goldman, Khuyagbaatar Batsuren, Salam Khalifa, Aryaman Arora, Garrett Nicolai,
Reut Tsarfaty, and Ekaterina Vylomova. SIGMORPHON–UniMorph 2023 shared task 0:
Typologically diverse morphological inflection. In Proceedings of the 20th SIGMORPHON
Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 117–125,
2023.
[9] Aaron Grattafiori, Abhimanyu Dubey, et al. The Llama 3 herd of models. arXiv preprint
arXiv:2407.21783, 2024.
[10] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn
Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset.
In NeurIPS Datasets and Benchmarks Track, 2021.
[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In
International Conference on Learning Representations, 2022.
[12] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt,
Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International
Conference on Learning Representations, 2023.
[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins,
Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska,
et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National
Academy of Sciences, 114(13):3521–3526, 2017.
[14] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. In International Conference on Machine Learning, 2019.
[15] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé
Jégou. Large memory layers with product keys. In Advances in Neural Information Processing
Systems, 2019.
[16] Yan-Shuo Liang and Wu-Jun Li. InfLoRA: Interference-free low-rank adaptation for continual
learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
[17] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee,
Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In
International Conference on Learning Representations, 2024.
[18] Jessy Lin, Luke Zettlemoyer, Gargi Ghosh, Wen-tau Yih, Aram Markosyan, Vincent-Pierre
Berges, and Barlas Oğuz. Continual learning via sparse memory finetuning. arXiv preprint
arXiv:2510.15103, 2025.
24


---

[19] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study
of catastrophic forgetting in large language models during continual fine-tuning. arXiv preprint
arXiv:2308.08747, 2023.
[20] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks:
The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pages
109–165. Academic Press, 1989. doi: 10.1016/S0079-7421(08)60536-8.
[21] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual
associations in GPT. In Advances in Neural Information Processing Systems, 2022.
[22] Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-
editing memory in a transformer. In International Conference on Learning Representations,
2023.
[23] Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui
Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian,
Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu
Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang
Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng,
and Dayiheng Liu. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and
training stability. arXiv preprint arXiv:2608.30320, 2026.
[24] Vinay V. Ramasesh, Ethan Dyer, and Maithra Raghu. Anatomy of catastrophic forgetting:
Hidden representations and task semantics. In International Conference on Learning Represen-
tations, 2021.
[25] Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. RL’s razor: Why online reinforcement
learning forgets less. In International Conference on Learning Representations, 2026.
[26] Shipeng Wang, Xiaorong Li, Jian Sun, and Zongben Xu. Training networks in null space of
feature covariance for continual learning. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2021.
[27] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui,
and Xuanjing Huang. Orthogonal subspace learning for language model continual learning. In
Findings of the Association for Computational Linguistics: EMNLP 2023, 2023.
[28] Lloyd R. Welch. Lower bounds on the maximum cross correlation of signals. IEEE Transactions
on Information Theory, 20(3):397–399, 1974.
A
Llama-3-8B Replication
The Llama-3-8B chain (Section 4) differs from the Qwen chain in one essential respect: its memory
arm is a lookup table added to a model that has none, initialised to zero, so the base model’s
knowledge does not depend on it and big-K drift measures only the perturbation caused by a new
side path. All captures used SDPA attention. Representations are taken after the final RMSNorm
(the LM-head input), at the last prompt token, for 64 math and 32 code probes and 64 probes
per task. All NLL changes are computed against the same-backend base or acquisition checkpoint.
Absolute values are not comparable with Qwen; only within-model orderings are.
25


---

Table 12: Llama-3-8B big-K drift F0 →F7 and reference NLL change.
Math
Code
Method
Drift
Mean cos
∆NLL
Drift
Mean cos
∆NLL
Lookup
4.86%
0.9988
+0.0024
7.29%
0.9973
−0.0002
Shared
69.10%
0.7661
+0.0712
31.87%
0.9496
+0.0590
Merged†
39.04%
0.9245
+0.0266
21.51%
0.9771
+0.0217
Table 13: Llama-3-8B small-ki drift Fi →F7 and task reference ∆NLL.
Small-ki drift
∆NLL
Task
Lookup
Shared
Merged†
Lookup
Shared
Merged†
SME
54.58%
75.91%
73.86%
+1.051
+1.496
+0.838
NAV
86.25%
96.38%
75.07%
+1.730
+2.166
+1.181
HYE
61.74%
66.16%
62.45%
+0.170
+0.186
+0.087
AFB
59.38%
77.42%
52.60%
+0.374
+1.023
+0.221
SQI
57.52%
65.08%
57.99%
+0.422
+0.518
+0.417
MKD
32.50%
33.50%
21.19%
+0.381
−0.026
+0.009
Mean
58.66%
69.08%
57.19%
+0.688
+0.894
+0.459
†Trained on the full training split; Lookup and Shared used a fixed 1,024-example subset per task.
The direction of the Qwen findings replicates. The memory arm has by far the smallest big-K
drift but not the least forgetting. Merged forgets least in NLL and has the smallest mean small-ki
drift, with Lookup essentially tied in drift. Small-ki drift tracks per-task ∆NLL (ρ = 0.71). Final
seven-task accuracy is 34.2% (Lookup), 33.9% (Shared) and 49.8% (Merged). Lookup’s mean
accuracy at acquisition (54.7%) is again lower than Shared’s (60.3%), the same acquisition gap as
on Qwen.
B
Full Seven-Round Big-K Trajectory
26


---

Table 14: Qwen3.8 big-K drift from F0, mean cosine and reference NLL at every task boundary
(head input, last prompt token). Base NLL: math 0.8878, code 0.2007.
Math
Code
Method
Round
Drift
Cos
NLL
Drift
Cos
NLL
N-gram
1
25.21%
0.9682
0.8811
13.40%
0.9914
0.1995
2
22.16%
0.9750
0.8767
14.67%
0.9896
0.1990
3
24.81%
0.9695
0.8780
14.66%
0.9897
0.1987
4
20.95%
0.9778
0.8774
14.19%
0.9903
0.2003
5
26.14%
0.9664
0.8741
14.22%
0.9901
0.1994
6
26.15%
0.9657
0.8757
15.09%
0.9891
0.1988
7
26.37%
0.9658
0.8759
15.13%
0.9889
0.1985
Shared
1
45.51%
0.8930
0.7504
24.91%
0.9692
0.2100
2
49.28%
0.8735
0.7410
25.92%
0.9671
0.2139
3
74.71%
0.7165
0.7685
30.10%
0.9553
0.2170
4
68.62%
0.7571
0.7472
32.36%
0.9487
0.2026
5
66.64%
0.7602
0.6192
35.53%
0.9372
0.2052
6
84.61%
0.6045
0.7725
38.05%
0.9285
0.2346
7
97.33%
0.4502
0.7877
38.90%
0.9263
0.2809
Merged
1
45.01%
0.8973
0.7811
25.12%
0.9689
0.2088
2
48.70%
0.8775
0.7426
27.16%
0.9636
0.2057
3
52.41%
0.8559
0.7956
29.37%
0.9575
0.2151
4
67.10%
0.7752
0.8744
30.97%
0.9521
0.2166
5
76.22%
0.7144
0.9159
32.41%
0.9477
0.2237
6
60.25%
0.8137
0.8525
33.62%
0.9435
0.2237
7
64.36%
0.7894
0.8036
34.98%
0.9387
0.2244
27


---

C
Reproducibility
Checkpoints are verified by SHA-256 before they are used as the parent of the next task. Probe
sets, prompts and tokenisation protocols are fingerprinted, and repeated capture of the same input
gives zero drift.
28
