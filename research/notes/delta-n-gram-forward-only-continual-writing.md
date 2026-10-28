---
title: 'Delta N-gram: Forward-Only Continual Writing'
id: delta-n-gram-forward-only-continual-writing
tags:
- cmp170hx-training-feasibility-a0d4fc
- frozen-table
- delta-ngram
- forward-only-writes
- memory-grafting-counterpoint
- expert-parallelism-dependency
- drift-hot-rows
- no-optimizer-state
- long-source
created: '2026-10-27T11:59:27.714750Z'
updated: '2026-10-27T19:00:31.572507Z'
source: https://arxiv.org/abs/2610.31125
source_domain: arxiv.org
fetched_at: '2026-10-27T11:59:27.714202Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Delta N-gram (Hao Wu, Illinois Institute of Technology, arXiv 2610.31125v1,
  22 Oct 2026) — the strongest counter-datapoint to the run''s frozen-table optimism,
  and the paper that makes the Memory Grafting design space precise. Setup: a hashed
  N-gram table T = T0 + Δ, where T0 is the pretrained table and Δ is a sparse patch
  nonzero only on written rows; every row with Δj = 0 reads exactly its pretrained
  value, and clearing Δ returns the original model. On Qwen3.8-Flash-Next (48 layers,
  512 routed experts, 10 active/token) the native table is read through H = 16 hash
  heads (8 bigram + 8 trigram), row width d = 160, concatenated to R^2560 and injected
  before layer-2 attention. Backbone, read-out and T0 are ALL frozen; the only trained
  component is a 4,075,681-parameter writer. Writes at use time are forward-only:
  no backpropagation, no optimizer state, no labels; the state stores only written
  row ids, patches Δj and write counts cj — explicitly ''It carries no optimizer state.''


  That is the direct support for the repo''s memory math: a frozen table costs its
  bf16 rows and nothing else (no fp32 master, no m/v, no grad), and the paper demonstrates
  a working design whose trainable budget is 4.1M params against a 48-layer 512-expert
  backbone. But the paper''s own training run contradicts the box''s fabric assumptions:
  the writer was trained on four H200 GPUs with expert parallelism 4 and data parallelism
  4, 12 trajectories per update, AdamW lr 3e-4, 100M target tokens over 4,172 updates.
  EP4 is exactly the topology the CMP 170HX''s no-P2P Gen2 x4 fabric rules out, so
  reproducing this recipe on the box would require re-deriving the writer training
  under pure DDP.


  The capability results are the counter-datapoint. Frozen memory is NOT free capacity:
  on a 0.6B prototype the forward writer keeps 16/16 facts after a second write where
  online gradient writes keep 2-4/16, but the binding is tied to the trained question
  form — a paraphrased question stays at the empty-memory 2/16 level, and the rows
  such queries read overlap the written rows in only ~6% of positions. Longer chains
  erode: four writes leave 60/64 facts answerable, sixteen writes leave 46/64 right
  after their own write but only 19 at the end. On 64 held-out late-2025 news events
  the plain writer lowers continuation NLL by 0.068 nats/token against empty memory
  and 0.024 against another document''s memory (0.063 and 0.013 after 16 intervening
  articles); the delta writer is more document-specific and retains 70% vs 54% of
  that specificity under distraction, at a smaller total gain.


  Long-chain stability is real: 35,000 consecutive writes on two hardware stacks with
  no numerical failure, and learned-task accuracy drifts only -0.26pp (plain) / -0.52pp
  (delta) over the rest of the chain versus -7.29pp for the Adam table updates of
  the prior Representation Drift work. But net acquisition on tasks the writer never
  saw is NEGATIVE (-12/448) and recall of written examples is only 23-25%. General-input
  big-K drift is 32.8%, 1.6x the Adam baseline''s 20.75%, concentrated in eight format
  bigram rows that every example writes and whose patches reach 4.4-11.7x their T0
  norm; zeroing those eight rows removes over 98% of the layer-2 drift on math inputs
  — and also removes their NLL gain. Optional decay pulls Δ toward zero (table toward
  T0), giving reversible forgetting.


  Training data mix (relevant because it overlaps the run''s own pipeline): Stack
  v2 Edu 20%, arXiv 25%, Project Gutenberg 20% (Common Pile), FineWeb-Edu 15%, CC-News
  17.5%, same-event CC-News streams 2.5%; 38,566 training documents. Prototype used
  a zero-initialised external table of 16,384 rows each for bigrams and trigrams,
  row width 1024, 1,190,465 writer params. 16,791 words — LONG SOURCE, flag for source-analyst
  delegation.'
raw_file: raw/delta-n-gram-forward-only-continual-writing.pdf
doi: arXiv:2610.31125
utility_score: 15.0
---

Delta N-gram: Forward-Only Continual Writing
into a Language Model’s Native N-gram Memory
Hao Wu
Illinois Institute of Technology
hwu71@hawk.illinoistech.edu
Abstract
Our previous analysis found that a model’s native hashed N-gram table is where continual
updates disturb the base least, but its writes needed backpropagation and changed the pretrained
table. We present Delta N-gram, a forward-only write path into that table. The pretrained
table T 0 is never modified. Evidence is written into a sparse patch ∆on the rows it addresses:
a 4.1M-parameter writer emits a target and a gate per (position, hash head) event, and a
block-averaged delta rule keeps each row a convex combination of its old value and the gated
targets. The writer is trained offline on raw text with the pretraining objective; at use time
writing takes forward passes only, and clearing ∆restores the original model. A 0.6B prototype
trained to answer questions keeps 16/16 facts after a second write; online gradient writes keep
2–4/16. We study two writers: a plain writer that encodes evidence with the T 0-only model,
and a delta writer that reads its current memory while encoding. On Qwen3.8-Flash-Next, the
plain writer’s memory is document-specific: on 64 held-out news events it lowers continuation
NLL by 0.068 nats/token against empty memory and by 0.024 against another document’s
memory, and 0.063 and 0.013 after 16 intervening articles. The delta writer’s memory is more
document-specific and keeps more of it under distraction (70% vs 54%). Over 35,000 writes on
our previous seven-task chain both writers stay stable and the delta writer drifts less (layer-2
math drift 176% vs 188%); later accuracy changes by −0.26/−0.52pp (−7.29pp for gradient
updates) and small-ki drift is 17.9/17.5% (39.2%). Acquisition is negative, and general-input
drift reaches 32.8% (20.75%): format rows read by every example grow to 4–12 times their
pretrained norm. Zeroing eight such rows removes over 98% of layer-2 drift on math inputs,
together with their NLL gain.
1
Introduction
A model that learns while it is used has to write new information somewhere, and every write is paid
for by what it disturbs. In our previous work, Representation Drift [31], we compared three places
to write on the same 48-layer mixture-of-experts model: the rows of its native, pretrained hashed
N-gram table [26], one continually trained LoRA adapter [11], and a fresh LoRA per task merged
into the backbone. Updating only the N-gram table changed the base model’s representations of
general inputs least (big-K drift 26%/15% on math/code, against 64–97%/35–39% for LoRA), and
this drift did not grow with the number of tasks. We concluded that long-running online learning
should write new knowledge only into the memory table and keep the backbone fixed.
That conclusion was reached with an expensive and irreversible write. Every table update in [31]
was an Adam step: a full backward pass through the network per write, optimizer state per row,
labels to define the loss, and a table that, once changed, no longer contains its pretrained values.
A system that writes continually during use needs the opposite properties. Writing should cost a
forward pass, need no labels or task boundaries, and leave the pretrained knowledge recoverable.
1
arXiv:2610.31125v1  [cs.CL]  22 Oct 2026


---

This paper builds that write path. We keep the pretrained table T 0 fixed and write into a sparse
patch ∆, so the model reads T 0 + ∆. A small writer, trained offline, turns each block of evidence
into writes. For every event, that is, every (position, hash head) pair, it emits a target v and a gate
β from the frozen model’s hidden states. Events that address the same row in a block are averaged
and applied with a delta rule we call D1. D1 moves the row toward the gated targets and keeps
it a convex combination of its old value and those targets. Seen from linear attention, this is a
DeltaNet [27, 32] whose keys are one-hot hash addresses, whose state is never reset, and whose write
targets come from a learned fast-weight programmer [28]. The writer is trained with the pretraining
objective on raw text. Each document is split into chunks; each chunk is first predicted from the
memory written so far and then written, and the loss of later chunks backpropagates through the
whole chain of writes. At use time the same writer and the same D1 run forward only.
The writer must read the evidence before it writes, and we study two ways of doing so. The
plain writer encodes evidence with the model reading only T 0. Its targets are then anchored to
the evidence, rows stay separable, and all blocks of a document can be encoded in one batch. The
delta writer encodes evidence with the model reading its current memory, so what it writes can
depend on what it has already stored. That is what long-term continual learning ultimately needs,
for instance to connect a new report to an entity written earlier. Its price is sequential writing and a
feedback loop through the memory. We analyse that loop and train both writers on the same data.
What the design gives by construction.
• No new memory module. Memory is the model’s own N-gram table, addressed by pretrained
hashes of the last two or three tokens and read by the model’s native injection at layer 2.
• Pretrained parameters are untouched. Rows that were never written read exactly T 0;
clearing ∆returns the original model, and each user’s ∆is a separable set of sparse rows.
• Forward-only writes. At use time there is no backpropagation, no optimizer state and no
label.
• Row-level attribution. Every change lives on a known row, so drift can be traced to, and
intervened on, individual rows.
Findings.
We train both writers (4.1M parameters each) on Qwen3.8-Flash-Next (512 experts, 48
layers) for 100M target tokens. We evaluate them on held-out news events, on the seven-task chain
of [31], and on LoCoMo [20].
1. Trained to answer, forward writes bind and keep facts, but only in the trained form.
On a 0.6B prototype trained with a question-answering objective, the writer keeps 16 of 16
facts after a second write, where online gradient writes into the same table keep 2–4. With a
paraphrased question it stays at the empty-memory level (Section 6.1).
2. Raw-text training yields document-specific memory. On 64 late-2025 news events, reading
a document’s own memory beats reading another document’s, which beats empty memory, at
every training stage. At 100M tokens the two gains are 0.068 and 0.024 nats/token, and 0.063
and 0.013 after 16 intervening articles (Section 6).
3. Reading memory while encoding changes how the writer writes. The delta writer’s
memory is more specific to its document (∆donor 0.032 against 0.023 without distractors) and
keeps more of that specificity after 16 distractors (70% against 54%); its total gain over empty
memory on these news events is smaller. A weight × encoding factorial places this change in the
2


---

learned weights (Section 6.5). On the seven-task chain the delta writer drifts less than the plain
writer: math drift is 176% against 188% at layer 2 and 40.2% against 41.8% at the head. It also
answers slightly more (129 against 125 of 448 test items, and 112 against 103 written examples)
(Section 7).
4. Long write chains are stable. Both writers, including the delta writer whose encoding feeds
back through its own memory, made 35,000 consecutive writes on two hardware stacks without
numerical failure (Section 7).
5. What was stored changes little afterwards, but little is stored. After each task is written,
learned-task accuracy changes by −0.26pp (plain) and −0.52pp (delta) over the rest of the chain,
against −7.29pp for the Adam table updates of [31]. Small-ki drift at the head input is 17.9%
and 17.5%, against 39.2%. Net acquisition on these tasks, which the writer never saw, is negative
(−12/448), and recall of written examples is 23–25% (Section 7).
6. Drift concentrates on a few hot rows and can be located. General-input big-K drift is
32.8%, 1.6 times that of [31]. It enters at the injection layer through eight format bigram rows
that every example writes, whose patches reach 4.4–11.7 times


T 0

. Zeroing them removes over
98% of the layer-2 drift on math inputs and also removes their NLL gain (Section 8).
Analysis.
Section 4 gives the relations behind these measurements: the bound of old-task drift by
prefix overlap times row displacement, inherited from [31]; the boundedness of D1; row separability,
under which zeroing a set of rows after the fact equals never committing them; and the fixed-point
dynamics of a writer that encodes evidence through its own memory, which we then test with a
weight × encoding factorial.
Position.
The two properties that a forward write path into the native table had to show, that
writes carry information forward and that long chains of them stay stable without touching T 0,
hold. The two that remain, using what was written at a different position and bounding the
absolute drift of hot rows, are where we take the method next (Section 9). Code is available at
https://github.com/af-74413592/delta-ngram.
2
Related Work
Where to write.
Our previous work, Representation Drift [31], compared update channels on the
model we use and found that writing only into the native N-gram table perturbs general inputs least;
its costs were interference among stored tasks and weak acquisition. This paper keeps that channel
and replaces the write: forward instead of gradient, patch instead of in-place change. Forgetting
under sequential fine-tuning [2, 14, 19, 22] and subspace constraints on successive updates [6, 17, 30]
motivate the comparison but act on dense parameters, not on addressed rows.
Delta rules and fast weights.
Linear attention can be read as a fast-weight memory written by
a slow network [27, 28]. DeltaNet replaces additive writes with error-correcting delta writes [27, 32].
Gated DeltaNet adds a scalar decay [33], and KDA makes the decay channel-wise [13]. Test-time
training generalises the inner update to a learned objective [29]. Qwen3.8 itself contains such layers.
These lines address interference among keys in a finite state that is cleared at the end of each
sequence, which is why they need decay. Our keys are one-hot hash addresses over hundreds of
millions of rows, the state persists across requests, and the writer is added to a frozen pretrained
3


---

Table 1: Ways to learn while in use. “Pretrained parameters” refers to the weights and tables the
model shipped with.
In-context
writes (linear
attn., GDN,
KDA)
Online LoRA
Gradient table
edits [31]
Persistent
memory block
(MemT)
Delta N-gram
Written state sequence state low-rank
backbone
increment
native N-gram
table
added memory
block (dense
GDN-2 state)
sparse patch
on the native
N-gram table
Write signal
forward
projection
backprop
backprop
(Adam)
forward
(GDN-2
update)
forward
writer + D1
Lifetime
cleared at
sequence end
permanent
permanent
permanent
permanent
(optional
decay)
Pretrained
parameters
unchanged
frozen;
adapter
separate
table
overwritten
unchanged;
module added
unchanged; T 0
fixed
Use-time cost near zero
full backward
per write;
optimizer
state per user
same
forward
one T 0-only
forward +
4M-param
writer
Undo / per
user
n/a
yes (adapter
per user)
needs a copy
of the table
yes (state per
user)
clear ∆;
sparse rows
per user
Rows
touched per
write
whole state
all adapter
parameters
rows read
(sparse)
block state
(dense,
routed)
rows read
(sparse)
Addressing
projected keys none
hashed
n-grams
learned keys
hashed
n-grams,
pretrained
Extra
training
with the
backbone
none
none
0.45–3.6B
params, 34B
tokens
4M params,
100M tokens
model rather than trained with it. Decay in our setting pulls toward T 0 rather than toward zero
(Section 3.5).
Sparse fast-weight memories.
Sparse Delta Memory [3] enlarges the state of a gated linear
RNN with an explicit memory whose slots are selected by product keys and written with a gated
delta rule. Fast-weight product-key memory [34] updates the values of a product-key memory by
chunk-level gradient descent on a reconstruction loss, at training and at inference time, and can
carry them across adjacent segments. Both are layers trained together with the backbone, over
learned keys. Ours adds a writer to a frozen pretrained model, writes into a table the model was
pretrained to read, and uses no gradients at use time.
Editing memory tables.
Memory and product-key layers provide large sparsely addressed
stores [1, 15]. Sparse memory finetuning updates only rows selected by their access statistics
and forgets less than LoRA or full fine-tuning [9, 18]. EngramEdit writes facts into the N-gram
embeddings of LongCat-Flash-Lite with a closed-form ridge solution and a penalty on reused rows [4].
ENGRAFT optimises the Engram rows of Qwen3.8-Flash-Next by gradient descent with a KL
4


---

constraint and keeps the result as a removable overlay [7]. All of these compute the write from
labelled facts or tasks, by backpropagation or a solve. Ours writes from raw evidence in a forward
pass, with no labels, into a separable patch. Knowledge editing in FFN weights [5, 23, 24] shares
the key–value view we use for drift but edits dense projections.
Persistent memory for a frozen backbone.
The closest work is MemT-1.0 [16]. It also freezes
the backbone, writes a per-user memory forward-only and trains the memory module offline. Its
memory is a set of added blocks with dense GDN-2 state, routed over storage experts, updated
token by token and committed per turn. It trains 0.45–3.6B parameters on 34B tokens: continued
pretraining, then post-training that distils from the frozen backbone given the full history. Post-
training adds about 10.69 points over the continued-pretraining checkpoint, the largest effect in its
ablations. Our memory is the model’s own N-gram table, read through its native injection, with a
4M-parameter writer trained on 100M raw-text tokens. Because the state is a sparse patch on known
rows, its drift can be traced to individual rows (Section 8). MemT’s results on long-conversation
benchmarks [20] show what a trained cross-position objective buys. Our LoCoMo failure (Section 7.5)
is consistent with that, and its full-context teacher is the first change we would borrow (Section 9).
3
Method
Figure 1 summarises the method. The backbone, the native read-out and the pretrained table T 0
are frozen. The only trained component is a small writer, and it is trained offline; at use time the
model writes into a sparse patch ∆with forward passes only.
3.1
Native N-gram memory
A hashed N-gram table T ∈RN×d is read at every position p through H hash heads. Head h maps
the last two or three tokens ending at p to a row address jh(p) that depends only on token ids and
is fixed at pretraining. We write the table as
T = T 0 + ∆,
(1)
where T 0 is the pretrained table and ∆is a patch that is nonzero only on rows that have been
written. Every row with ∆j = 0 reads exactly its pretrained value.
On Qwen3.8-Flash-Next [26], H = 16 (bigram and trigram, eight heads each) and each row has
d = 160 dimensions. The read-out concatenates the 16 rows and injects the result into the residual
stream before the attention of layer 2:
eq =
Tj1(q); . . . ; Tj16(q)
 ∈R2560,
vq = Wveq,
kq,c = W (c)
k eq,
(2)
gq,c = σ

ϕ
  ⟨N(kq,c), N(zq,c)⟩/
√
D

,
ϕ(x) = sign(x)
q
max(|x|, 10−6),
(3)
bq = flat4
c=1
 gq,cvq
,
uq = bq + SiLU

DWConv
 Nconv(b)

q

∈R10240.
(4)
Here c indexes the four residual streams, zq,c is the hidden state after layer 1, N is RMSNorm, and
the depthwise convolution has four taps with dilation 3. We use this read-out unchanged. The
value path is linear in eq, so a patch acts through its magnitude; the key and convolution paths are
normalised, so there it acts through its direction.
Because every weight after the injection is frozen, a change of the table reaches the backbone
only as a change of its input. In the key–value notation of [5, 31], where a layer computes Y = KW
and Yt −Yi = Ki(Wt −Wi) + (Kt −Ki)Wt, only the second term survives: for the backbone, writing
the table is a pure perturbation of K.
5


---

native N-gram table T = T 0 + ∆
Evidence
block xt
(512 tokens)
Frozen model reads
T 0 + λ∆t, encodes
xt →ˆHt (stop-grad)
Writer θ (4.1M)
per event a = (p, h):
va = shVθ(xa)
βa = σ(Bθ[·])
D1 per addressed row
∆j ←(1 −wj)∆j + Aj
(block average)
T 0: pretrained rows,
never modified
∆: sparse patch
(written rows,
counts cj)
+
write
gate reads
∆j, cj
Query or
next-chunk
tokens
16 hash heads:
jh(p) from the
last 2–3 tokens
Native read-out
(gate, conv),
injected before layer 2
Frozen 48-layer MoE
→next-token loss
addresses
read rows
offline training only: loss backpropagated through the D1 chain into θ
frozen
trained offline
memory state (per user)
λ = 0: T 0-only encoding; λ = 1: memory-aware
Figure 1: Delta N-gram. Top: a block of evidence is encoded by the frozen model, the writer emits
a target va and gate βa for every (position, hash head) event, and D1 updates the addressed rows of
the sparse patch ∆; the gate also reads each row’s current patch and write count. Bottom: every
forward pass reads T 0 + ∆at the hashed addresses of the last two or three tokens through the
model’s native read-out. At use time only the solid arrows run, forward only; during offline training
the next chunk’s loss is backpropagated through the whole write chain into the writer (dashed red).
Clearing ∆restores the original model.
3.2
From DeltaNet to a row-wise delta rule
DeltaNet [27, 32] reads o = Sq and writes S ←S −β(Sk −v)k⊤. We obtain the write rule for the
table in four steps.
1. One-hot keys. Treat each row of the table as a column of S and set k = ej. Reading Sej is a
table lookup, and kk⊤is nonzero only at j, so a write touches only row j. Distinct rows do not
interfere except through hash collisions, and ∥k∥= 1 holds by construction.
2. Write the patch. The update acts on ∆j; T 0 is never written.
3. Average within a block. Evidence arrives in blocks. Every pair (position p, head h) in a block
is a write event a addressed to row j(a) = jh(p). The nj events addressed to row j all see the
same pre-block snapshot of the state, and their effects are averaged.
4. Targets and step sizes come from a writer. Where DeltaNet obtains v and β by forward
projections, here a learned writer provides them (Section 3.3).
Together these give the rule we call D1. For every row touched by the block,
Aj = 1
nj
X
a→j
βava,
wj = 1
nj
X
a→j
βa ∈[0, 1],
∆+
j = (1 −wj) ∆j + Aj,
c+
j = cj + 1
(5)
6


---

The count nj includes events with βa = 0; rows not touched by the block are left exactly unchanged;
cj is the number of blocks that have written row j. Equivalently, ∆+
j −∆j =
1
nj
P
a βa(va −∆j):
each event moves the row in proportion to its residual va −∆j, as in the delta rule. Section 4.1
shows that averaging, rather than summing, is what keeps the rule bounded.
3.3
The writer
When a block of evidence xt arrives, the writer proceeds in three steps.
Encoding.
The frozen model runs over the block, and we take the input of the LM head, normalised
by a parameter-free RMSNorm (Qwen3.8 has no separate final norm, and the per-token scale of this
input varies widely):
ˆHt = N
 E(xt; T 0 + λ∆t)
,
λ ∈{0, 1},
(6)
with gradients stopped at ˆHt. With λ = 0 the evidence is encoded by the model with an empty
patch; we call this T 0-only encoding and the writer trained with it the plain writer. With λ = 1 the
evidence is encoded by the model reading its current memory (memory-aware encoding). The delta
writer is the plain writer forked at 80M training tokens and fine-tuned with λ = 1 for the remaining
20M.
Event features.
For event a at position p and head h,
xa =
 ˆHp; mean( ˆHp+1:p+4); Pool( ˆH); emb(h)
,
(7)
the representation at the position, the mean over the valid tokens among the next four positions (a
zero vector if there are none, as at the end of a block), an attention pool over the block, and an
embedding of the (order, head) pair. The right context lets the writer see what follows the n-gram
whose row it is about to write.
Target and gate.
va = sh Vθ(xa),
βa = σ
 Bθ[ xa, ∆j(a), va −∆j(a), log(1 + cj(a)) ]
.
(8)
Vθ and Bθ are two-layer MLPs with hidden width 256. Vθ outputs a 160-dimensional target and Bθ
a single gate logit, so βa and wj are scalars. The fixed per-head scale sh is calibrated at initialisation
so that the initial targets have RMS ρ · RMS(T 0
h) with ρ = 0.01; it is a calibration, not a cap, and
the target scale is free to change during training. The target does not read the row state directly.
Under memory-aware encoding it still depends on the memory through ˆHt; stopping the gradient
removes only the backward path, not this forward dependence. The gate reads the row state directly:
it sees the row’s current patch, the residual and the write count, so it can learn to write less when
the row already agrees with the target and more when the old value should be replaced. The native
writer has 4,075,681 parameters.
The writer is tied to the backbone and the table it was trained on. Its input ˆH, its target scale
and the effect of a patch through the read-out all depend on them; in our 0.6B prototype, a writer
trained on a base model kept about one fifth of its gain when moved to the post-trained model.
7


---

3.4
Offline training with the pretraining objective
The writer is trained on raw text with next-token prediction, the same objective as pretraining, and
needs no labels or question–answer data.
1. A document, or a time-ordered stream of reports on one event, is split into blocks of 512 tokens.
Each query carries the last 128 tokens of the previous block as prefix.
2. Memory starts empty. The first block of an ordinary document is only written; in an event
stream the whole first article is only written, and scoring starts with the second article. Every
later block is predicted, then written: the model reads the memory as it was before the block, the
block’s cross-entropy is computed, and the block is then written with D1. The last block is not
written. Within an event stream, memory persists across articles.
3. The loss is normalised once by the number of target tokens in the whole training group and
backpropagated through the full chain of writes; only θ receives gradients.
The gradient path is: loss →frozen layers above the injection →native read-out (2)–(4) →patch
∆t →D1 chain (5) →writer (8). Stopping gradients at ˆHt removes the branch “old memory →
encoding →features” under memory-aware encoding. This is a truncation of backpropagation
through time, chosen for cost: the forward dynamics, and hence the agreement between training
and use, are unchanged.
Training and use run the same computation.
The writer and D1 used at training time are
exactly those used at use time; training only adds a backward pass around them. At use time
evidence is written as it arrives and queries read T 0 + ∆, with no backpropagation, no optimizer
and no change to any weight. Algorithms 1 and 2 give both procedures.
3.5
Memory state, reversibility and decay
The state stores only rows that have been written: their row ids, patches ∆j and write counts cj, plus
a block clock with the bookkeeping for lazy decay. It carries no optimizer state. Clearing the state
returns the model to its empty-patch read-out. This is exact within one read-out implementation,
but the split FP32 read-out of Section A is not bitwise equal to the original BF16 read-out, so every
comparison uses the empty patch in the same implementation as its baseline. A user’s state is a
self-contained set of sparse rows that can be stored apart from the shared backbone and writer.
D1 already forgets by content: rewriting a row replaces its old value by the factor 1 −wj. An
optional decay adds forgetting by time. Unlike the decay of Gated DeltaNet and KDA [13, 33],
which pulls the state toward zero, ours pulls the patch toward zero and hence the table toward
T 0: ∆j ←α∆j, i.e. Tj ←T 0
j + α(Tj −T 0
j ). It is applied lazily. Each block of new evidence is first
encoded with the current state; the clock then advances once, and the rows the block addresses
enter the gate and D1 at their decayed values. A read uses each row at its decayed value without
advancing the clock or storing the result. The main method uses α = 1 (no decay); Section 6
compares α = 0.99.
8


---

Algorithm 1 Delta N-gram at use time: forward-only writing and reading.
Require: frozen backbone, read-out and T 0; frozen writer θ; encoding λ ∈{0, 1}; decay α (α = 1:
none)
1: ∆←∅
▷sparse rows (j, ∆j, cj) and a block clock
2: procedure Write(xt)
▷one block of new evidence
3:
ˆH ←N
 E(xt; T 0 + λ∆)

▷(6); current state, gradient stopped
4:
advance the clock; take each row the block addresses at its decayed value αelapsed∆j
5:
for each event a = (p, h), p a new token of xt do
▷prefix and padding never write
6:
j ←jh(p);
build xa from ˆH, p and h
▷right mean over valid tokens; zero if none
7:
va ←shVθ(xa);
βa ←σ
 Bθ[xa, ∆j, va −∆j, log(1 + cj)]

▷pre-block state
8:
end for
9:
for each row j addressed by the block do
▷D1, (5); nj counts all events on j
10:
wj ←
1
nj
P
a→j βa;
Aj ←
1
nj
P
a→j βava
11:
∆j ←(1 −wj)∆j + Aj;
cj ←cj + 1
12:
end for
13: end procedure
14: procedure Read(q)
15:
run the model on q; every position reads T 0 + ∆through (2)–(4), each row at its decayed
value
16:
the clock is not advanced and nothing is written back
17: end procedure
18: To remove a user’s memory, clear ∆: the model reads the empty patch again.
Algorithm 2 Offline training of the writer (native settings). The forward computation is that of
Algorithm 1; training adds one backward pass.
1: for each update do
2:
S ←0;
N ←0
▷summed token loss and target count over the group
3:
for each of the 12 trajectories in the group do
▷a document or an event stream
4:
split each article separately into 512-token blocks x1, . . . , xL ▷no block spans two articles
5:
∆←∅
▷memory persists across articles of one stream
6:
for t = 1, . . . , L do
7:
if xt is scored then
▷documents: t ≥2; event streams: from the second article
8:
query prefix: the previous 128 tokens of the same article, or BOS if xt starts an
article
9:
S ←S + summed cross-entropy of xt reading T 0 + ∆;
N ←N + |xt| ▷memory
before xt
10:
end if
11:
if t < L then Write(xt)
▷the last block is not written
12:
end if
13:
end for
14:
end for
15:
L ←S/N
▷normalised once, across all GPUs
16:
backpropagate L: frozen layers →read-out →∆→D1 chain →θ
17:
update θ with AdamW (learning rate 3 × 10−4, no weight decay, clipping at 1)
18: end for
9


---

4
Analysis
This section states what the construction guarantees and what it does not. Section 4.1 bounds
the state. Section 4.2 carries the drift bound of [31] over to the patch. Section 4.3 shows that, for
the plain writer, the effect of removing any set of rows can be read off the final state. Section 4.4
analyses the dynamics that memory-aware encoding introduces.
4.1
Boundedness of D1
Proposition 1 (Boundedness). If every target satisfies ∥va∥≤V , then D1 gives



∆+
j



 ≤
max(∥∆j∥, V ) for every row. Starting from ∆= 0, ∥∆j∥≤V at all times.
Proof. If wj = 0 then Aj = 0 and the row is unchanged. Otherwise ∆+
j = (1 −wj)∆j + wj¯vj
with ¯vj = P
a βava/ P
a βa, a convex combination of ∆j and a convex combination of targets, since
wj ∈(0, 1].
For a fixed writer the bound V exists under both encodings. ˆH is RMS-normalised, the right-
context mean and the attention pool are convex combinations of normalised vectors, and there are
16 head embeddings, so the features lie in a compact set and Vθ is bounded on it. The bound holds
for each fixed θ, not uniformly over training.
Averaging over the nj events, rather than summing, is what makes this hold. With a sum,
wj = P
a βa can far exceed 1: twenty events with β = 0.25 give 1 −wj = −4, and repeated
application diverges. Averaging bounds the aggregate strength of repeated hits on a row. It is not a
key normalisation, since one-hot keys already have unit norm.
Two things the proposition does not give. First, the backward pass is not a contraction: because
the gate reads the old patch, the state Jacobian gains a term
1
nj
P
a(va −∆j)(∇∆βa)⊤that can
exceed one, so training clips gradients. Second, V is set by the writer’s targets, not by



T 0
j



. The
proposition bounds the state; it does not bound drift relative to the pretrained values. A row
written by every example is pulled toward the writer’s targets whatever its pretrained norm, which
is how the hot rows of Section 8 reach 4–12 times



T 0
j



.
4.2
From row displacement to representation drift
We follow the notation of [31]. F t(q) is the representation of a fixed context q at a measured site
(the LM-head input unless stated otherwise) after the state has absorbed t tasks. Let Ψ be the
frozen map from the injected input to that site, and Φ the read-out (2)–(4) from table rows to u.
The bound below is conditional, with one condition for each map.
• Read-out. The stacked rows e are linear in the table, so the straight segment between the two
tables is also a straight segment in e. Along it, at every point and not only at its ends, every
gate inner product in (3) keeps its sign; there Φ is differentiable.
• Backbone. Φ is nonlinear, so the image of that segment is a curve in u, and controlling Ψ
along the curve does not give a bound between its endpoints. We therefore assume that Ψ is
Lipschitz with constant LΨ on a convex set containing the straight segment between the two
injected inputs ui
≤q and ut
≤q. This holds, for instance, when every token is routed to the same
experts at every point of that segment.
10


---

Where these conditions fail, the bound needs additional jump terms. Because of its 10−6 floor, ϕ
jumps slightly at zero, and a routing switch moves the output by a finite amount, so a uniform LΨ
is not a property of the architecture. Under the conditions, with a single injection site,



F t(q) −F i(q)



 ≤LΨ



ut
≤q −ui
≤q



 ,
(9)
where the right-hand side covers the whole prefix because attention reads all of it. Let r be the
number of convolution taps and δ their dilation (r = 4, δ = 3 for Qwen3.8), so that up depends on
ep, ep−δ, . . . , ep−(r−1)δ. Let Jp,τ (τ = 0, . . . , r −1) be the Jacobian of up with respect to ep−δτ along
the same segment, and let C2
J = maxp
P
τ sup ∥Jp,τ∥2, with the supremum taken over the segment.
By the mean-value inequality and Cauchy–Schwarz, ∥∆up∥2 ≤C2
J
P
τ ∥∆ep−δτ∥2, and each ∆ep′
enters at most r of the ∆up, namely those with p = p′ + δτ. Averaging (9) over the probe set Qi of
task i gives
 Dt
i
2 :=
1
|Qi|
X
q∈Qi



F t(q) −F i(q)



2 ≤rL2
ΨC2
J
X
j
mij ∥∆Tj∥2 ,
(10)
mij = #{(q, p, h) : q ∈Qi, p ≤q, jh(p) = j}
|Qi|
,
with ∆Tj = T t
j −T i
j. The factor mij is the average number of times a probe’s whole causal prefix
reads row j. The bound is prefix overlap times row displacement: a write rule cannot change the
overlap, only the displacement.
Under (1), both references are explicit:
• Big-K drift (general probes Q0, reference T 0): ∆Tj = ∆t
j, the patch itself. A general input
whose prefix reads no written row is exactly unchanged. Drift on general inputs therefore comes
from rows that are both read by them (m0j > 0) and written; template and format n-grams are
the candidates.
• Small-ki drift (task-i probes, reference: the state just after task i): ∆Tj = ∆t
j −∆i
j. A row
that has already converged to the writer’s target when task i finishes, and receives similar targets
afterwards, contributes little, however large its patch is.
The two can therefore move in opposite directions. A format row that every example writes reaches
its displacement early and then changes little: this inflates big-K drift on every input that reads it
but adds little small-ki drift afterwards. Sections 7 and 8 observe exactly this pattern.
Comparison with gradient table updates.
Under a per-row Adam step, as in [31], the
coordinate step α ˆm/(
√
ˆv + ϵ) is independent of the gradient’s scale. Noise therefore still moves a
row whose mean gradient is zero. As a random-walk approximation, valid when the normalised
updates are nearly unbiased, of stable variance and only short-range correlated, the displacement
after S steps is about c α
√
dS, or c
p
d P
s α2s with a varying learning rate. With the dense Adam
of [31], momentum also keeps moving a row for about 1/(1 −β1) steps after it leaves the batch.
This tail does not carry over to the sparse Adam of our prototype baseline, which updates only rows
present in the batch. Under D1 the displacement is set by the residual va −∆j and the learned
gate, with no optimizer state or momentum. Whether the writer actually gives va ≈∆j or a small
βa on rows that should be kept depends on what it learned; the formula does not guarantee it.
11


---

4.3
Row separability
The plain writer has a property that makes row-level interventions cheap.
Proposition 2 (Row separability). Consider the plain writer (λ = 0), a fixed evidence sequence,
the same clock and decay, and an initial state with ∆= 0. Let S be any set of rows. Compare (A)
running the chain and then setting ∆j = 0 for j ∈S, with (B) running the chain without committing
any write to rows in S. The final patches of (A) and (B) are equal, so every query reads the same
table.
Proof. Under λ = 0 the encoding ˆHt does not depend on ∆, so the features xa and targets va are
identical in (A) and (B). The gate βa reads only the patch and count of its own row j(a), and D1
updates row j only from events addressed to j. By induction over blocks, every row outside S has
the same patch and count in (A) and (B). Rows in S end at zero in both.
The effect of removing any set of rows on drift and on gain can therefore be computed by zeroing
them in the final state, without rerunning the write chain. The equivalence covers removal only.
A cap applied at every write is a different intervention, because a row’s later updates depend on
its own history. Take w = 1
2, targets 4 and then −2, and a cap |∆| ≤1: the uncapped row goes
0 →2 →0 and stays 0 when its final value is capped, while a row capped at every write goes
0 →1 →−1
2. Evaluating a cap enforced during writing therefore requires replaying the write chains
of the capped rows; for the plain writer, only those rows change. The equivalence is also limited in
three further ways. It concerns the patch, not the full state: write counts on S differ. It does not
extend to trajectories that keep writing after the intervention, because the gate sees those counts.
And it fails for the delta writer, whose encoding reads every row in the block and therefore couples
rows.
4.4
Memory-aware encoding: anchoring, feedback and reproducibility
With T 0-only encoding a target depends only on the evidence, so the patch always lies in the convex
hull of {0} and the evidence’s targets: the writer can reweight what the evidence says but cannot
invent new targets. This anchoring does not imply a unique fixed point. The gate reads the old
patch, so the effective target ¯v(∆) = P
a βa(∆)va/ P
a βa(∆) depends on the memory even when the
va do not. Take one coordinate, two events with v± = ±1 and gates β± = σ(±4d). Then w = 1
2 and
F(d) = 1
2d + 1
2 tanh(2d),
(11)
which has three fixed points, 0 and ±0.957; the one at zero is unstable (F ′(0) = 1.5). The gate
favours evidence that agrees with the memory, and the memory then locks in. This is a constructed
example, not a measurement on the native model, but it shows that fixed targets plus a convex
update do not rule out self-reinforcement through the gate. The case most exposed to it is revision,
where old and new values conflict on the same row.
Memory-aware encoding adds a second route: the target itself depends on the state. We analyse
the local map with the evidence, the write counts, the clock and all other external inputs held fixed.
In the actual chain a written row’s count grows by one at every write and the gate reads log(1 + cj),
so each step applies a different map. Stability over a long chain depends on products of successive
Jacobians, which the eigenvalues of a single step do not determine. For a single event with fixed
evidence and count, T(s) = s + β(s)
 v(s) −s
, and at a differentiable fixed point v(s⋆) = s⋆,
DT(s⋆) = (1 −β)I + β Dv(s⋆).
(12)
12


---

With exogenous targets (Dv = 0, as in a DeltaNet layer or our T 0-only writer) this is a pure
contraction by 1−β. In the scalar case, |1−β+βµ| < 1, i.e. 1−2/β < µ < 1 for an eigenvalue µ of Dv,
is sufficient for linearised stability. At equality the linearisation is inconclusive: T(s) = 1
2s + 1
2 tanh s
has T ′(0) = 1, yet 0 is asymptotically stable. For a row written by several events,
DFj = (1 −wj)Pj + 1
nj
X
a→j
h
βa Dva + (va −∆j)(∇βa)⊤i
,
(13)
where Pj selects row j from the full state. The first bracketed term is the feedback that memory-aware
encoding adds; it couples rows, because the encoding reads every row in the block. The second is
the gate reweighting present under both encodings; since the targets of different events differ, it
does not vanish at a fixed point.
The gain along one direction decides nothing. J = diag(0.5, 2) contracts one direction and
expands another, while negative feedback can contract despite a large gain: with w = 0.25 and
¯v(d) = −tanh(2d), the gain at the origin is 2, yet F ′(d) = 0.75 −0.5 sech2(2d) ∈[0.25, 0.75] and the
map contracts globally. Rows a block does not touch contribute an identity block. Restricted to the
rows it updates, a spectral radius of DF below one at a fixed point is sufficient for local asymptotic
stability of the map with frozen inputs. On a closed, invariant, complete set, a uniform contraction
bound L < 1 is sufficient for a unique, globally attracting fixed point. A Lipschitz bound alone is
not, and strict contraction is not necessary for uniqueness. Neither can be checked analytically for a
48-layer MoE encoder, so we measure.
Encoding-substitution sensitivity.
To screen the feedback term we hold the direct state fixed
and replace only the encoding:
G =



¯v( ˆHmem, ∆) −¯v( ˆHbase, ∆)



F
∥∆on the rows read by the block∥F
.
(14)
Replacing the encoding changes both the targets and the gates, so G mixes a change of targets
with gate reweighting and does not isolate the term β Dv. Blocks whose rows all have a zero patch,
where the denominator vanishes, are excluded. G is a secant, used as a screen rather than as a
proof of contraction. Section 6 reports it together with twin-trajectory measurements of whether
small state differences grow.
Reproducibility.
The analysis above is differentiable. A sparse MoE encoder is not: an arbitrarily
small change of the memory can switch the experts of some tokens and move v by a finite amount.
This makes the memory-aware state sensitive to perturbations. On its own it does not prevent
identical runs from agreeing bit for bit, but our GPU kernels are not bitwise deterministic: rebuilding
∆from the same ˆH differs by about 10−9. Under memory-aware encoding such differences grow
to a plateau, and the useful question is whether they stay bounded and below the read-side noise.
T 0-only encoding is reproducible only for a fixed batch layout, since batching changes the encoder’s
numerics as well.
5
Experimental Setup
5.1
Models and writer training
Prototype in two stages.
Before the native model we develop the method on a frozen 0.6B
model with an external, zero-initialised hash table (T 0 = 0): 16,384 rows each for bigrams and
13


---

trigrams, row width 1024. The two orders are summed, divided by
√
2 and injected before decoder
layer 1. The prototype went through two stages that differ in the training objective.
• Question-answering objective. On Qwen3-0.6B-Base, the writer is trained on synthetic
episodes. Short evidence blocks stating facts about named entities are written one at a time, and
the loss is placed on the answers to template questions about them. Training uses 2,000 episodes
(500 outer steps, seed 17), with disjoint entity pools for training and evaluation. Evaluation
generates answers greedily on 16 held-out worlds with two writes each. It compares the writer
with online gradient writes into the same table (ten SparseAdam steps per evidence block,
learning rate 0.01 or 0.05), the full evidence in context, and BM25 retrieval of the top two blocks.
Transfer probes, without retraining, test a paraphrased question that keeps the entity name,
chains of 4 and 16 writes, and revision of an earlier value.
• Raw-text objective. The writer is then trained as in Section 3.4, on raw text with no question–
answer data, on the post-trained Qwen3-0.6B (1,190,465 writer parameters, hidden width 128).
A 5M-token code-only pilot with two seeds comes first, followed by a 100M-token run on the
five-source mixture described below. A matched 5M-token run on the base model tests whether
a writer transfers between the two readers.
Prototype and native results are reported separately throughout.
Native model.
The main model is Qwen3.8-Flash-Next [26]: 48 layers, 512 routed experts with
ten active per token, and the native table and read-out of Section 3.1. Backbone, read-out and T 0
are frozen; the writer (4,075,681 parameters) is the only trained component. We train on four H200
GPUs with expert parallelism 4 and data parallelism 4. Each update uses 12 complete trajectories
and backpropagates through the whole write chain of each. The optimizer is AdamW with learning
rate 3 × 10−4, no weight decay and gradient clipping at 1. Two arms start from the same initial
writer: α = 1 is trained to 99,997,072 target tokens (4,172 updates) and α = 0.99 to 39,998,107
(1,669 updates). The α = 1 budget was extended from a shared to a per-arm 100M after the
20M and 40M development evaluations had been seen, so the 60–100M points are exploratory.
Implementation details are in Section A.
Training data.
Raw English text and source code from six streams, mixed by target-token share:
code from Stack v2 Edu (20%), arXiv papers (25%) and Project Gutenberg books (20%) from
the Common Pile [12], FineWeb-Edu web pages (15%) [25], CC-News articles (17.5%) [10], and
time-ordered streams of several CC-News reports on the same event (2.5%). The corpus has 38,566
training documents and 713 event streams backed by 2,002 articles; 3,069 documents and 224 event
streams are held out for validation. Trajectories are 2 to 32 blocks long depending on the source.
Memory starts empty for every trajectory and persists across articles within an event stream. No
question–answer data or labels are used.
Delta writer.
The memory-aware writer is forked from the α = 1 checkpoint at 80M and trained
to 100M on exactly the 835 update groups that the main line used between 80M and 100M. Only
the encoding changes (λ = 1); all writer parameters are trained and the AdamW state is inherited.
The main line over the same groups is the matched control.
5.2
DEV64: memory of new events
DEV64 consists of 64 event families from the second half of 2025, drawn from 146 source articles.
Each family has an earlier and a later real news report on the same event. Memory is built from
14


---

the earlier report, then k ∈{0, 1, 4, 8, 16} unrelated articles are written, and finally the later report
is scored by teacher-forced continuation. The scored report receives the true first 128 tokens as
context and is never written into memory. Three conditions differ only in what was written:
• self: the family’s own earlier report, followed by the k distractors;
• donor: an earlier report from another family, paired by a hash fixed before training, followed by
the same k distractors;
• empty: nothing.
We report two differences in nats per target token:
∆empty = Lempty −Lself,
∆donor = Ldonor −Lself.
(15)
∆empty is the total benefit of the document’s memory, including generic help; ∆donor is the part
specific to the document. Forgetting can raise ∆donor by making the donor worse, so the two are
always reported together. Losses are averaged within each family and then equally over families.
Three controls accompany the main numbers. The first scoring window, the up to 384 targets
right after the 128-token context, tests whether the gain is present before the later report has
unfolded. A prefix split separates families whose 128-token context restates the early facts (34), does
not (23) or is unclear (7). A direction-permutation control permutes the channels of the read-out
vector at scoring time, preserving addresses, counts and row norms, to test whether the gain comes
from what was written.
Intervals are percentile intervals from 10,000 family-paired bootstrap resamples; comparisons
across checkpoints resample the same families. Unless stated otherwise they are pointwise and
uncorrected. The weight × encoding factorial of Section 6.5 uses a pre-registered analysis with
100,000 resamples and Bonferroni correction over its 60 comparisons. The pretraining cutoff of
Qwen3.8 is not documented, so the model may have seen these events; the empty condition measures
what it already knows. For the prototype we also report a historical held-out set of 640 families
from the five sources, scored the same way and macro-averaged over sources.
5.3
Seven-task chain
We reuse the seven morphological inflection tasks of [31], drawn from SIGMORPHON [8], in the
order SME →NAV →HYE →AFB →SQI →MKD →AMH. Each task contributes 5,000 support
examples (prompt and reference answer, in the chat template with thinking disabled), the same
examples and exposures as the 125 Adam steps × batch 40 of [31]. Every example is one evidence
document and one D1 update, so a chain makes 35,000 writes. One state is kept for the whole chain:
it is never cleared or partitioned at task boundaries, and the writer is frozen and was never tuned
on these tasks. The tasks’ earlier results motivated this work, so this is a re-test on historical tasks,
not a fresh benchmark.
Evaluation follows [31]:
• Accuracy: 64 fixed test items per task, greedy decoding of at most 128 new tokens, exact match
after stripping and Unicode NFC normalisation. Each task is also tested before its support
examples are written, so that acquisition is measured as the change from writing.
• Answer NLL: teacher-forced NLL of the reference answers.
• Drift: representations at the last prompt token at all 48 layers (four residual streams concatenated,
10,240 dimensions) and at the LM-head input (2,560 dimensions), as relative RMS change. Big-K
15


---

drift uses 128 math and 64 code probes against the empty state; small-ki drift compares each
task’s probes with the state right after that task was written.
• Seen-example recall: 448 support examples already written (64 per task) are answered from
the final state.
The three arms of [31] (N-gram table trained with Adam, shared LoRA, merged LoRA) are reused,
not rerun.
Accuracy and the other metrics come from two independent runs. Accuracy is measured on
A800 GPUs with vLLM, whose Qwen3.8 read-out we extend with a separate FP32 patch branch.
NLL and representations are measured on H200 GPUs with the native training stack. Each machine
writes its own chain from an empty state, and the two states differ by 4–6% because of cross-stack
numerics. Each writer (plain and delta) is run once per machine.
5.4
LoCoMo
LoCoMo [20] has 10 long dialogues and 1,986 questions; we use the delta writer. Each dialogue is
written once into an empty memory, in session order, rendered as text with a dated header per
session and one “Speaker: utterance” line per turn, in 512-token blocks. All questions of a dialogue
are answered from the same written state, which questions do not modify. Four conditions use
the same question prompt: self (own dialogue in memory, only the question in context), empty,
donor (another dialogue of similar length in memory) and full context (empty memory, the whole
dialogue in context). The prompt instructs the model to answer “No information available” when it
lacks the information. Decoding is greedy with at most 32 new tokens. The three memory conditions
read through the FP32 split read-out of Section A. The full-context condition needs no patch and
runs the native BF16 read-out with expert offloading, so the two kinds of condition use numerically
different read paths. The main score is LoCoMo’s category-aware token F1 on the 1,540 questions of
categories 1–4; the 446 adversarial questions are excluded. We also score the teacher-forced NLL of
the reference answers. We do not compare these numbers directly with [16], which uses a different
backbone, prompts and judging.
6
Document-Specific Memory from Raw Text
On the native model the writer only ever sees raw-text continuation. This section first traces how
the prototype arrived at that objective. It then asks whether raw text is enough for the writer to
learn memory that is specific to the document written, robust to later writes, and carried by the
content of the patch.
6.1
Prototype I: a question-answering objective
Trained to answer questions, the forward writer binds facts and keeps them (Table 2). One write
raises answer accuracy from the empty-memory 2/16 to 16/16. After a second write all 16 first facts
are still answered, and 15 of the 16 new ones. Online gradient writes into the same table learn each
new fact but overwrite the old one, keeping only 2–4 of 16. The forward writer matches putting the
evidence in context and retrieving it with BM25. This is the contrast that motivates the method:
writing by gradient into shared rows forgets, while a learned forward write need not.
The same experiments expose the limits that reappear on the native model.
16


---

Table 2: Prototype I, 16 held-out two-write worlds (Qwen3-0.6B-Base). First: the first fact asked
after the first write; Retained: the same fact asked after the second write; New: the second fact.
Greedy generation, first-word score. Empty memory answers 2/16.
Method
First
Retained
New
Forward writer + D1
16
16
15
Online gradient writes (SparseAdam, lr 0.01)
16
2
16
Online gradient writes (SparseAdam, lr 0.05)
16
4
16
Full evidence in context
16
16
14
BM25 retrieval, top 2
16
16
14
• The binding is tied to the trained question form. Asked with a paraphrased question that
still names the entity, the writer stays at the empty-memory 2/16 before and after writing. The
rows read by such queries overlap the written rows in only about 6% of positions.
• Longer chains erode it. Without retraining, four writes leave 60 of 64 facts answerable. After
sixteen writes, 46 of 64 are answered right after their own write but only 19 at the end. Revised
values are followed immediately in 5 of 16 worlds.
• Template rows carry a trade-off. At an earlier checkpoint, two old answers were lost after
the second write. Restoring the ten rows of the literal question template to their values before
that write rescues both old answers (14 to 16), but new answers fall from 12 to 2. These are the
same shared rows that, on the native model, carry both gain and drift (Section 8).
A question-answering objective also needs labelled episodes in a fixed format. We therefore moved
to the pretraining objective, which needs no labels and applies to any text.
6.2
Prototype II: raw text
On a 5M-token code-only pilot, two seeds agree: ∆empty = 0.036 and 0.037, ∆donor = 0.024 and
0.026, all with intervals above zero. The writer is tied to its reader. A writer trained against the
base model and used with the post-trained model keeps a gain of 0.0066 over empty memory, against
0.0363 for the writer trained on the post-trained model itself.
Trained on the five-source mixture for 100M target tokens, the prototype writer yields memory
with a clear document-specific component. On the historical held-out set, the five-source macro
averages are ∆empty = 0.144 [0.141, 0.147] and ∆donor = 0.097 [0.095, 0.100], positive for every
source.
On the first block of a later article in an event stream, scored before that article is
written, ∆donor = 0.110: memory carries across articles. On DEV64 the 100M prototype gives
∆empty = 0.060 and ∆donor = 0.038 without distractors. After 16 distractors ∆donor falls to 0.012,
31% of its undistracted value. The small table is the likely bottleneck. At k = 16 the number of
distinct addresses is already 26–31% below the number of distinct n-gram keys, although no table is
more than 62% occupied. Most of the prototype’s DEV64 gain is already present after 5M tokens.
6.3
Native model
On the native model the ordering self < donor < empty in NLL holds at every distractor distance
and every training stage (Figure 2). Table 3 gives the final checkpoint. Reading the document’s
own memory lowers continuation NLL by 0.068 nats/token relative to empty memory and by 0.024
relative to another document’s memory, with all intervals above zero.
17


---

Figure 2: DEV64 during native training. Left: benefit over empty memory (∆empty); right: advantage
over a donor document’s memory (∆donor), without distractors (k = 0) and after 16 distractor
articles (k = 16). Bars are 95% family-bootstrap intervals.
Table 3: DEV64 at the final native checkpoint (α = 1, 100M target tokens). Nats per target token,
95% family-paired bootstrap intervals. Empty-memory NLL is 1.8409 at every k.
Distractors k
∆empty
∆donor
self NLL
0
0.0682 [0.0584, 0.0781]
0.0244 [0.0166, 0.0326]
1.7728
1
0.0658 [0.0562, 0.0754]
0.0187 [0.0124, 0.0256]
1.7751
4
0.0648 [0.0545, 0.0754]
0.0167 [0.0113, 0.0226]
1.7761
8
0.0659 [0.0561, 0.0761]
0.0170 [0.0119, 0.0225]
1.7750
16
0.0633 [0.0524, 0.0746]
0.0133 [0.0077, 0.0194]
1.7777
Robustness to later writes.
After 16 unrelated articles, 93% of ∆empty remains (0.063) and
55% of ∆donor (0.013). The decline is significant for both (−0.0049 [−0.0101, −0.0001] and −0.0110
[−0.0194, −0.0031], paired within family). It is much smaller than on the prototype, where ∆donor
kept 31%.
The gain is present early and comes from the written content.
In the first scoring window,
before the later report has unfolded, ∆empty = 0.052 and ∆donor = 0.018 at k = 0, and 0.046
and 0.010 at k = 16. Families whose 128-token context does not restate the early facts still gain
(∆donor = 0.020 at k = 0, 0.012 at k = 16, both intervals above zero). Permuting the channels of
the read-out vector costs the self condition more than the donor condition: the difference of the two
permutation costs is 0.028 [0.020, 0.037] at k = 0 and 0.017 [0.012, 0.023] at k = 16. The direction
of what was written carries information specific to the document.
Training dynamics.
The two quantities develop differently with training. ∆empty keeps rising:
0.037, 0.051, 0.062, 0.062 and 0.068 at 20, 40, 60, 80 and 100M, and from 40M to 100M self
NLL improves by 0.017 [0.013, 0.023]. ∆donor rises from 20M to 40M (0.013 to 0.023) and then
fluctuates between 0.021 and 0.028. Its change from 40M to 100M is not significant at any distance,
because donor NLL improves almost as much as self NLL. Continued raw-text training thus mainly
strengthens help that transfers across documents, while the document-specific part plateaus after
18


---

Table 4: Weight × encoding factorial on DEV64, all cells at batch size 1. Entries are ∆empty /
∆donor in nats per target token. “Main”: plain writer at 100M; “delta”: memory-aware fine-tune at
100M.
Weights
Encoding
k = 0
k = 1
k = 4
k = 8
k = 16
main
T 0-only
.0668 / .0234
.0654 / .0179
.0635 / .0168
.0655 / .0163
.0626 / .0127
main
memory-aware
.0626 / .0270
.0627 / .0206
.0648 / .0172
.0613 / .0152
.0647 / .0122
delta
T 0-only
.0560 / .0315
.0555 / .0296
.0550 / .0271
.0516 / .0249
.0538 / .0219
delta
memory-aware
.0517 / .0320
.0522 / .0290
.0520 / .0252
.0492 / .0232
.0538 / .0217
about 40M tokens. The checkpoint with the highest ∆donor is 80M (0.028 at k = 0, 0.017 at k = 16).
We report the 100M endpoint, fixed by the budget, rather than select on DEV64. The absolute gains
are smaller than on the prototype, as expected for a much stronger backbone whose empty-memory
NLL is already 1.84 and which may have seen these events.
6.4
Decay toward the pretrained table
On the native model, α = 0.99 gives more total benefit at short distances. At equal budgets of 20M
and 40M its ∆empty is higher than that of α = 1 for k ≤8 (at k = 0: by 0.0069 [0.0030, 0.0111]
and 0.0061 [0.0024, 0.0101]); at k = 16 the difference is not resolved. Differences in ∆donor are
not significant at either budget. Decay thus trades persistence for help on recent material. Since
persistence is the point of the method, the main configuration is α = 1.
The prototype shows why such comparisons need both metrics. A decay rate predicted per
evidence block collapses below 0.4, so memory becomes roughly a cache of the last block. ∆donor
then rises by 20% while self NLL gets worse and the cross-article gain halves (0.091 to 0.049).
Forgetting inflates ∆donor by degrading the donor.
6.5
Memory-aware encoding
Stability.
We measure the quantities of Section 4.4 on the 80M writer, using the next 12 training
trajectories, six of which write 4–15 times. The encoding-substitution sensitivity G (Equation (14))
has median 0.30, 99th percentile 0.57 and maximum 0.59 over 49 blocks, and stays in the same range
during fine-tuning (median 0.27–0.42). Twin trajectories with memory-aware encoding, started from
states that differ by 10−8 to 10−2 of their norm, reach a plateau of 1.7–5.3% relative difference
within 2–4 writes and stay there for up to 15 writes. The plateau does not depend on the initial
difference, which is what the routing discontinuity of Section 4.4 predicts; we have not confirmed
that mechanism with fixed routing. With T 0-only encoding a 0.1% perturbation shrinks to about
0.89 of its size, and independent repeats agree to within 10−8. The effect of either kind of difference
on query NLL stays within the read-side numerical noise, which is present under both encodings
(after a 1% perturbation, mean change +0.0005 and −0.0041 nats/token on 512-token queries).
Batch layout matters as well: encoding the same evidence in a different batch layout changes the
writer’s features by a median 7.6%.
Training.
Fine-tuned with memory-aware encoding on the same 835 update groups as the main
line, the delta writer’s training loss is slightly worse throughout (paired +0.00059 [0.00041, 0.00070]
nats/token). On DEV64, ∆donor rises by 0.006–0.010 and ∆empty falls by 0.009–0.017 relative to the
main line. Weights, encoding at evaluation and batch layout all changed at once, so we separate
them.
19


---

Weight × encoding factorial.
Table 4 crosses the two sets of weights with the two encodings at
evaluation, all at the same batch layout. Under Bonferroni correction over 60 comparisons, 17 are
significant:
• The shift lives in the weights. With the encoding held fixed, changing the weights lowers
∆empty significantly at all five distances under both encodings (by 0.008–0.014 with T 0-only
encoding) and raises ∆donor (by 0.008–0.012 with T 0-only encoding, significant at k ≥1; with
memory-aware encoding significant only at k = 16). Under T 0-only encoding, 70% of ∆donor
survives 16 distractors with the delta weights, against 54% with the main weights.
• Switching the encoding at evaluation brings no detected gain. With the weights held
fixed, memory-aware encoding lowers ∆empty significantly only at k = 0, and no corrected
comparison shows a gain. No interaction term excludes zero. The layout control (main weights
and T 0-only encoding at batch 12 versus batch 1) differs by at most 0.0013 in point estimate.
The larger ∆donor must be read together with the absolute losses. At k = 0, changing the weights
moves self NLL from 1.7741 to 1.7849 and donor NLL from 1.7975 to 1.8164. Both get worse, and
the gap grows because the donor gets worse faster. Training with memory-aware encoding therefore
changed how the writer writes: memory became more specific to its document and more robust to
distractors, but less helpful overall on these news events. On DEV64 it did not make the writer
better at predicting the document it wrote. Over the long seven-task chain, the same writer drifts
less than the plain writer while answering slightly more (Section 7.3). This is a single training run,
and the mechanism is unknown.
7
Long Write Chains: Stability, Retention and Acquisition
The seven-task chain puts the writer far outside its training distribution. Training trajectories
have at most 32 blocks of raw text. Here a single state receives 35,000 writes of short, templated
examples from tasks the writer has never seen. This section reports what the chain does to stability,
to what is stored, to what is acquired and to the base.
7.1
Stability
Both writers completed 35,000 consecutive writes on both stacks without numerical failure. Every
state passed its integrity checks before and after each evaluation, and queries never modified the
state. The delta writer, whose encoding reads its own memory, showed no runaway self-reinforcement
over the chain: its final accuracy and drift are close to those of the plain writer (below). The
two stacks write numerically different states from the same evidence. At the end of the chain
∥∆A800 −∆H200∥/ ∥∆H200∥is 6.3% for the plain writer and 4.2% for the delta writer, while row
sets, counts and clocks agree exactly. This is why accuracy (A800) and drift (H200) are reported as
two trajectories.
7.2
Acquisition and retention
Table 5 separates acquisition from retention. Writing a task’s own 5,000 examples does not raise its
accuracy. Summed over the seven tasks, the change from writing is −12/448 for the plain writer
and −7/448 for the delta writer (5 and 4 items turned correct, 17 and 11 turned wrong). Two tasks
improve: SQI for the plain writer (24 to 27) and MKD for the delta writer (26 to 27). After the
seventh task the chains hold 125/448 and 129/448, against 255–309/448 for the three arms of [31].
The two writers do not differ significantly (paired McNemar p = 0.42).
20


---

Table 5: Seven-task accuracy (out of 64 per task) before the task’s own examples are written →
right after they are written →after the seventh task. The last column is the N-gram arm of [31]
right after learning each task.
Task
Plain writer
Delta writer
[31] N-gram
SME
10 →7 →6
10 →7 →7
13
NAV
6 →5 →5
6 →6 →6
6
HYE
33 →28 →29
32 →32 →32
61
AFB
5 →2 →2
6 →2 →2
42
SQI
24 →27 →25
27 →27 →26
50
MKD
29 →27 →28
26 →27 →26
54
AMH
31 →30
31 →30
57
Total after task 7
125/448 (27.9%)
129/448 (28.8%)
255/448 (56.9%)
Net acquisition
−12/448
−7/448
—
Change after learning (tasks 1–6)
−0.26pp
−0.52pp
−7.29pp
Table 6: Small-ki drift (relative RMS) at the last prompt token, from right after each task is written
to the end of the chain (plain writer, H200). Last column: N-gram arm of [31] at the head input
after task 7.
Task
Head input
Layer 48
Layer 9
[31] N-gram
SME
23.3%
42.8%
290%
40.4%
NAV
21.3%
32.7%
211%
62.0%
HYE
12.4%
21.1%
52%
32.6%
AFB
11.2%
21.3%
55%
37.3%
SQI
19.9%
31.7%
180%
36.8%
MKD
19.2%
33.6%
265%
26.0%
Mean
17.9%
30.6%
—
39.2%
What is stored changes little afterwards. From the moment each task is written to the end of
the chain, accuracy on the first six tasks changes by −0.26pp (plain) and −0.52pp (delta), against
−7.29pp for the Adam-trained N-gram arm, −11.72pp for shared LoRA and −7.03pp for merged
LoRA. Answer NLL tells the same story. Writing a task raises its answer NLL by 0.069 (plain)
and 0.081 (delta) on average; afterwards, NLL on the first six tasks changes by −0.001 and −0.019,
against +0.305 for the N-gram arm of [31].
The two halves must be read together. Little changes afterwards largely because little was
acquired. These numbers show that later writes did not erode what the state held; they do not
show that the method learns and then keeps. The positive evidence for storing and keeping new
information is the document-specific memory of Section 6.
7.3
Drift
Small ki.
At the head input, the learned tasks drift by 17.9% on average from the end of their
own writing to the end of the chain (Table 6), against 39.2% for the N-gram arm of [31]. The delta
writer gives 17.5%. The comparison holds at matched rounds. For SME after one, two and three
further tasks: 12.8%, 18.7% and 19.4%, against 32.5%, 40.1% and 41.6%. For NAV after one and
two further tasks: 16.3% and 17.2%, against 50.8% and 51.8%. Inside the network the picture
differs: at layer 9, four of the six tasks drift by more than 180%. The small output-side drift sits on
21


---

Table 7: Plain versus delta writer on the seven-task chain, after the seventh task unless stated. Drift
is relative RMS (H200); accuracy and recall are out of 448 (A800).
Plain writer
Delta writer
Math big K, layer 2
188.0%
175.7%
Math big K, layer 2, after task 4
153.0%
94.8%
Math big K, layer 9
135.0%
125.7%
Math big K, layer 48
51.2%
49.3%
Head big K, math / code
41.8% / 23.8%
40.2% / 22.5%
Small ki at the head, mean of six tasks
17.9%
17.5%
Old-task answer NLL, change after learning
−0.001
−0.019
Net acquisition
−12
−7
Accuracy after task 7
125
129
Recall of written examples
103
112
top of large early-layer changes.
Big K.
On general inputs the chain drifts more than the N-gram arm of [31]. At the head input,
math probes drift by 41.8% and code probes by 23.8% from the empty state (mean 32.8%), against
26%/15% (mean 20.75%) for that arm. This is still below merged LoRA (49.67%) and shared LoRA
(68.12%). The two probe sets reach this level in different ways:
• Math drift enters at the injection. It is 0 at layer 1, before the injection, and 198% at layer 2
after the first task; it then decays with depth to 42.5% at the head. It is fully present after the
first task (5,000 writes) and does not accumulate: after the seventh task it is 188% at layer 2
and 41.8% at the head.
• Code drift is 1–3% at layer 2 and builds up in later layers through attention, reaching 24% at
the head.
The math pattern is what a few hot rows read directly at the probe position would produce, and
Section 8 confirms it. Big-K drift is not a behavioural loss here: math answer NLL falls from 0.904
(empty) to 0.846, and code NLL is unchanged (0.247 to 0.246). The rows that move the math
representations also help predict the math answers (Section 8.2). We nevertheless count 32.8% as a
defect. Drift on general inputs is a budget the method must stay within, not a cost to be offset by
gains.
Plain versus delta writer.
The delta writer, which reads its memory while encoding, drifts less
than the plain writer over the chain (Table 7). On the math probes its drift is smaller at every
reported depth at the end of the chain. The gap is largest in early layers: after the fourth task,
layer-2 drift is 94.8% against 153.0%, and layer-9 drift is 69.4% against 108.8%. At the head input
its big K is smaller at six of seven endpoints on math and at all seven on code. On learned tasks its
small-ki drift is lower for four of six tasks, and their answer NLL improves by 0.019 after learning,
against 0.001. The lower drift costs nothing in accuracy: net acquisition is −7 against −12, 129
against 125 test items are answered after the seventh task, and 112 against 103 written examples
are recalled. Section 8.2 repeats the comparison after zeroing the shared rows.
22


---

Table 8: LoCoMo, categories 1–4 (1,540 questions), delta writer. F1 on 0–100; reference-answer
NLL in nats/token, averaged per question. Intervals: 10,000 dialogue-cluster bootstrap resamples.
Condition
F1
Answer NLL
self (own dialogue in memory)
0.44
4.034
empty
0.44
4.018
donor (other dialogue in memory)
0.44
4.049
full context
56.79 [55.33, 58.34]
—
self −empty
0
+0.016 [−0.006, +0.035]
self −donor
0
−0.015 [−0.025, −0.004]
7.4
Recall of written examples
Weak acquisition could mean the state stores the examples but does not generalise from them. We
test this directly. From the final states, both writers answer the same 448 support examples that were
written into memory (64 per task), using their original prompts. The plain writer answers 103/448
(22.99%) and the delta writer 112/448 (25.00%). The difference is +9 items (paired McNemar
p = 0.049, uncorrected, exploratory). Neither state reproduces a large share of what it was given, so
the data do not support “remembered but not generalised”. The writes did not bind each answer to
an address from which the answer can be read back (Section 8.3).
7.5
LoCoMo
Writing the ten dialogues takes 426 block writes (215,278 tokens), about 12 seconds per 23,000-token
dialogue on A800. With only the question in context, all three memory conditions answer every
question with the same refusal, “No information available”. No greedy output differs between self,
empty and donor, and F1 is 0.44 in each. The same backbone with the whole dialogue in context
reaches 56.79 (Table 8).
Reference-answer NLL separates the sources of memory without showing a benefit. Self is lower
than donor (−0.015, interval below zero), so what was written depends on the dialogue. Relative to
empty memory, the self difference crosses zero under both per-question and per-token averaging,
and its sign depends on the averaging. Split by position, both self and donor make the first answer
token worse than empty memory (+0.13 and +0.17). Given the correct prefix, both improve the
remaining tokens slightly (−0.020 and −0.016), so this improvement is not specific to the dialogue
that was written. These NLLs use a single-prefill path that differed from the token-by-token path
by up to 0.34 nats in a pre-run check, so they are specific to that path. Section 8.3 offers one
explanation for the first-token result: the rows read where the answer starts were never written.
This is consistent with the result but does not by itself establish its cause.
Zero-shot transfer from raw-text continuation to dialogue question answering therefore fails for
this writer. Using written content at a different position from where it was written is the main
capability the method lacks.
8
Locating Drift at the Row Level
In [31] drift could be measured but not attributed: Adam updates had spread over many rows of a
table that no longer stored its pretrained values. Here every change is a patch on a known row, the
addresses are deterministic hashes of tokens, and the plain writer is row-separable (Proposition 2).
23


---

Table 9: Zeroing the patches of the eight shared bigram rows. Big-K drift of the math probes
(relative RMS against the empty state) and their reference-answer NLL. Empty-state math NLL is
0.9039 for both writers.
State
Layer 2
Layer 9
Head input
Math NLL
Plain, task 1
197.7% →2.5%
143.9% →3.9%
42.5% →12.8%
0.841 →0.901
Plain, task 7
188.0% →2.9%
135.0% →5.1%
41.8% →18.8%
0.846 →0.902
Delta, task 1
182.3% →2.9%
131.7% →4.6%
40.6% →12.7%
0.849 →0.902
Delta, task 7
175.7% →3.3%
125.7% →5.7%
40.2% →27.1%
0.855 →0.901
This makes it possible to say which rows a probe reads, how often each was written, how far it
moved, and what removing it does.
8.1
Which rows the math probes read
Using the native hash and tokenizer, we reconstruct the 16 physical rows read at the last position of
every math probe and look them up in the saved states; this is a CPU audit with no model forward.
All 128 math prompts end with the tokens Solution, : and a newline. Of the 16 rows read there:
• the eight trigram rows for (Solution, :, newline) were never written: count and patch are zero;
• the eight bigram rows for (:, newline) are the rows of Input: followed by a newline, which
occurs in every support example of the seven tasks. Each example writes each of these rows
exactly once, with no further hash collisions. After the first task their counts are 5,000, after the
seventh 35,000.
Their patches are several times larger than the pretrained rows: ∥∆j∥/



T 0
j



 ranges over 4.48–10.75
after the first task and 4.41–10.03 after the seventh for the plain writer, and 4.61–11.74 after the
first task for the delta writer. The level is reached within the first task. From the first to the
seventh endpoint the concatenated patch of the eight rows changes by 8.8% (plain writer) and
keeps its direction (cosine 0.997). This matches the layer profile of Section 7.3: math drift appears
at the injection after the first task and does not accumulate. Boundedness (Proposition 1) holds
throughout; what is missing is anything that ties the size of a patch to



T 0
j



.
8.2
Zeroing the eight shared rows
We set the patches of these eight rows to zero in a copy of each state and keep T 0, every other row
and all other fields. Then we rerun the 128 math probes at all seven endpoints of both writers. For
the plain writer, Proposition 2 makes this the state the chain would have reached had these rows
never been committed. For the delta writer it is an intervention on the final state only.
Figure 3 and Table 9 show the result. Layer-2 drift falls by 98.1–98.7%, from 176–198% to
2.5–3.3%. Layer-9 drift falls from 126–144% to 4–6%. These eight rows are the source of the shallow
drift on the math probes. Drift that remains at the head (12.7–27.1%) builds up in later layers
from other rows and from propagation through attention, as on the code probes. The intervention
also compares the two writers. Zeroing the eight rows lowers the delta writer’s final head drift from
40.2% to 27.1%, and the plain writer’s from 41.8% to 18.8%, although the delta writer’s patches on
these rows are no smaller (4.6–11.7 times



T 0
j



 after the first task). Drift norms do not add across
rows, so these numbers do not apportion the drift between the eight rows and the rest. For the
24


---

Figure 3: Big-K drift of the 128 math probes at every layer, before (solid) and after (dashed) zeroing
the patches of the eight shared bigram rows in the same state. Each curve is measured against the
same writer’s empty state.
delta writer the zeroed state is an intervention on its final state, not a state its chain would have
reached (Section 4.3).
The same rows carry a gain. Zeroing them raises math answer NLL in all 14 writer × endpoint
conditions, by 0.026–0.060, to 0.901–0.902, close to the empty-memory level of 0.904. Shared rows
are therefore not simply contamination. With the current write rule, the generic gain on math
inputs and the drift on them come from the same few patches, written by format tokens of unrelated
tasks. Zeroing these patches removes both. Whether a nonzero cap can bound the drift and keep
part of the gain has not been tested. This is the trade-off that a per-row budget must manage. Rows
whose patches exceed their pretrained norms by a set factor are the ones a norm cap ∥∆j∥≤c



T 0
j



would bound. Row separability gives the effect of capping the final state directly; a cap enforced at
every write requires replaying the write chains of those rows (Sections 4.3 and 9).
8.3
The answer entry
The same audit explains part of the failure to use memory at a different position. In both the
seven-task prompts and the LoCoMo prompts, the first answer token is predicted at the end of the
chat template, after </think> and two newlines. All 448 recall prompts of Section 7.4 and all 1,986
LoCoMo questions read the same 16 physical rows there.
• In the seven-task chain the support examples use the same template, so these 16 rows are written
throughout the chain. Answers to different lemmas and features meet in the same rows, where
later writes can mix with or overwrite earlier ones.
25


---

• In LoCoMo, the dialogue text written into memory contains no template, so in all ten dialogue
states these 16 rows were never written: their patches and counts are zero. Across the 426 blocks
and 3.4 million address events of the dialogues, no write lands on them.
Memory can still reach the first answer token, but only indirectly: patches read at earlier prompt
positions must propagate through attention and the convolution. The writer was trained on raw-text
continuation, where the useful pattern is to write at a position and read at the same n-gram later.
It was never trained to make content written in one context readable from an answer position whose
own rows carry nothing. A plausible explanation for both the weak recall of Section 7.4 and the
LoCoMo result is therefore this: the writer learned to write and read at the same position, while
question answering needs retrieval across positions. We regard this as the main open problem of the
method rather than a property of the addressing alone. The writer’s features include right context
and a block-level pool, so it can see the answer when it writes earlier rows, and queries can attend
to rows read earlier in the prompt.
9
Discussion
Where memory should live.
The native N-gram table is sparse and addressable, and the
model already reads it. Writing there needs no new memory module, no change to the backbone
and no external store. If future models ship with such tables, the table may be the natural place
for continual learning: a writable store of knowledge inside the model that leaves the backbone
untouched.
Isolation versus binding.
Exact-token addresses are what make the method’s retention and
attribution possible. A write touches only the rows its n-grams address, and an input whose prefix
reads none of them is unchanged. The same property limits use. A row can only help when the
same n-gram is read again, so content written at one position is not directly available at another;
Section 8.3 shows the answer position reading rows that were never written. Content addressing
would give binding, but writes would overlap again, and with overlap comes interference. This is
the central trade-off of the approach. Our results suggest that what is missing is a writer trained to
make written content usable across positions, and possibly a memory module with its own attention
for reading it (see the last paragraph of this section). The next step is an episodic objective, in
which evidence is written and then queried at other positions, with distillation from the same frozen
model given the evidence in context. That distillation is the component whose removal costs MemT
the most [16]. A CPU check on existing LoCoMo losses can decide first whether the n-grams of the
answers are reachable at all, or whether a small content-addressed memory is needed beside the
table.
Drift as a budget.
Pretrained knowledge is the most valuable thing the model has, so absolute
drift on general inputs must be bounded and not traded against gains. The method already makes
drift locatable: we know which rows moved, how far and through which tokens. The default should
therefore include a per-row cap,
∆j ←∆j · min

1,
c



T 0
j



∥∆j∥+ ε

,
(16)
with c fixed in advance by the drift budget for big K and small ki, not tuned for gains. Since
the read-out is injected at layer 2, bounding each patch relative to its pretrained row bounds the
26


---

perturbation at the injection directly. Section 8.2 shows the cost: the hot rows also carry the generic
gain, and zeroing them removes it; how much of it a nonzero cap keeps is untested. Recovering gains
inside the budget is the role of multi-slot rows. The few most-written rows would get several slots
keyed by context, so that a generic signal and specific content are stored apart and used only where
the context matches; all other rows keep a single value. A larger hash table does not help, because
it removes collisions but not the sharing of one n-gram by many contexts. For the plain writer, the
effect of locking any set of rows can be computed from existing final states. A cap enforced during
writing needs only the write chains of the capped rows to be replayed.
Encoding with or without memory.
Encoding evidence with the T 0-only model is a choice,
not a necessity. It anchors targets to the evidence, keeps rows separable, makes training and use
identical, and lets all blocks be encoded in one batch, which on Qwen3.8 is the difference between
about 3,200 and 940 target tokens per second. The cost is that the writer cannot interpret new
content in the light of what it has stored, for instance resolving a pronoun to an entity written
earlier. Memory-aware encoding removes that limitation. It serialises writing and makes the state
sensitive to small numerical differences, though in our measurements these stayed on a bounded
plateau below read-side noise (Section 6.5). The factorial shows that training with it changes how
the writer writes. Over 35,000 writes, the resulting writer drifts less than the plain writer, most
clearly in early layers, with slightly higher accuracy and recall (Section 7.3). On news continuation
it trades total gain for document specificity. The two writers therefore remain two lines of the
method. The plain writer gives batch training, row-level attribution and reproducibility. The delta
writer is the route to integrating new evidence with what is already stored. What it still lacks is a
training signal that rewards such integration: evidence spread over several related documents, with
corrections and revisions, rather than continuation of one document at a time.
Who decides what is written.
Every token of the evidence produces write events; how much
each writes is decided by the learned gate, not by hand-written masks. Raw-text training teaches
the writer what helps later continuation. It does not teach what a later question will need, and it
does not teach restraint on format rows that every example shares. The one boundary still set by
hand is what counts as evidence. User documents, user corrections and verified outcomes count. The
model’s own generations are not written back by default, because a persistent write that conditions
all of a user’s later requests is where self-reinforcement would do damage.
Memory that belongs to the user.
Each user’s memory is a separable set of sparse rows, so
it can be kept on the client and supplied per request, while the backbone and writer are shared.
A dialogue of 13,000–25,000 tokens gives 78–145 MB of FP32 rows before any compression. A
client can only change its own state; merging the patches of many users into a shared one is where
privacy and poisoning questions arise. A portable state must carry its compatibility information:
the backbone, tokenizer and writer versions, the addressing scheme, the write operator and decay,
and a schema and layout version with a content checksum. A resumable full state, with counts and
clock, must be distinguished from a read-only snapshot for inference.
Fast writing, slow consolidation.
The method is the fast half of a two-speed system. In use, it
writes sparse rows quickly. During a “sleep” phase, signals that have proved stable and general, such
as the generic gain now carried by hot rows, would be consolidated into a protected shared part,
and specific content would stay in sparse rows. This is the division of labour between hippocampus
and neocortex in complementary learning systems [21], and the starting point of our 2023 work on
27


---

online LoRA updates with sleep cycles. A forward write path is also what a self-improving system
needs: the model writes its experience into its own memory and works with that memory afterwards.
Connecting the two is the subject of our next paper.
Limitations.
Two experiments show that the method does not yet work on long-chain tasks that
need written content at a different position. On the seven-task chain, net acquisition is negative and
only 23–25% of written examples can be recalled. On LoCoMo, every memory condition answers
with the same refusal. Our writers went through only the first half of MemT-1.0’s recipe [16]:
continuation on raw text (100M tokens), with no post-training. MemT follows its 20B tokens of
continued pretraining with 14B tokens of post-training on memory-specific data, distilled from the
frozen backbone given the full history. That stage adds about 10.69 points in its ablation, and it
targets exactly the cross-position use our writers lack. Hot rows have no budget relative to their
pretrained norms, and general-input drift is 1.6 times that of the N-gram arm of [31]. Budget limited
us to one model with a native N-gram table. We did not train writers for LongCat-Flash-Lite, the
model edited by EngramEdit [4], which adds its n-gram embeddings at the input without a gate.
Nor did we train one for DeepSeek-V4.1-Flash-1017, which injects two tables at different layers.
A writer is tied to its backbone, so each would need its own; the second would also test the drift
bound with more than one injection point. All results come from one training seed.
Why the writer learns only same-position continuation.
The most likely reason, still to be
tested, is the training objective. Raw-text continuation rewards writing, at the row of an n-gram k,
what follows k, and reading it back at the same position when k next appears. DEV64 continuation
uses memory in exactly this way, which is why it works there. Question answering instead needs
information written at the positions of the question’s content to be moved by attention to the answer
position, and the writer was never asked to write for that. Two observations point the same way.
The 0.6B prototype trained with a question-answering objective answers 16 of 16 facts after one
write, against 2 of 16 with empty memory (Section 6.1); after the switch to raw text this ability was
not kept. And in [31], gradient updates of the same table reached 56.9% on the seven tasks through
the same frozen attention. So the table, read through frozen attention, can carry the answers when
it is trained with answer gradients. Whether a forward writer trained for cross-position use can
reach that, or whether reading across positions also needs a memory module with its own attention,
as in MemT-1.0 [16], remains open.
10
Conclusion
Delta N-gram writes into a language model’s own N-gram table with forward passes only. It needs
no gradients, labels or optimizer state at use time and never touches the pretrained table. A
4M-parameter writer trained on raw text with the pretraining objective learns document-specific
memory that survives later writes. A second writer, which reads its own memory while encoding
evidence, writes memory that is more specific to its document and drifts less over long chains. Both
stay stable over chains of 35,000 writes. Because every change is a patch on a known row, drift
can be traced to and intervened on at individual rows, which is what turns a drift budget from a
measurement into something that can be enforced. Two problems remain. First, written content is
not yet usable at other positions; one explanation still to be tested is that the answer position reads
rows the evidence never wrote. Second, a few format rows shared by every example carry both the
generic gain and the drift on general inputs. A cross-position training objective, a per-row cap set
28


---

by the drift budget, and context-keyed slots for hot rows address these two problems; they are the
next steps.
References
[1] Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi
Ghosh. Memory layers at scale. In Proceedings of the 42nd International Conference on Machine
Learning, volume 267 of Proceedings of Machine Learning Research, pages 3831–3842, 2025.
[2] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard,
Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney,
and John P. Cunningham. LoRA learns less and forgets less. Transactions on Machine Learning
Research, 2024.
[3] Loïc Cabannes, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Matthijs Douze, Maria Lomeli,
Ilze Amanda Auzina, Justin Carpentier, Gabriel Synnaeve, and Hervé Jégou. Sparse delta
memory: Scaling the state of linear RNNs through sparsity. arXiv preprint arXiv:2607.07386,
2026.
[4] Hongru Cai, Ran Wei, Wenjie Wang, Chengfa Wu, Ning Song, Yongqi Li, and Wenjie Li.
EngramEdit: Decoupled knowledge updates in LLMs through conditional memory. arXiv
preprint arXiv:2610.10533, 2026.
[5] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He,
and Tat-Seng Chua. AlphaEdit: Null-space constrained model editing for language models. In
International Conference on Learning Representations, 2025.
[6] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for
continual learning. In Proceedings of the Twenty Third International Conference on Artificial
Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages
3762–3773, 2020.
[7] fulvian. ENGRAFT: Write facts into an LLM’s n-gram (Engram) memory table as a removable
overlay. GitHub repository, https://github.com/fulvian/engraft-ngram, 2026.
[8] Omer Goldman, Khuyagbaatar Batsuren, Salam Khalifa, Aryaman Arora, Garrett Nicolai,
Reut Tsarfaty, and Ekaterina Vylomova. SIGMORPHON–UniMorph 2023 shared task 0:
Typologically diverse morphological inflection. In Proceedings of the 20th SIGMORPHON
Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 117–125,
2023.
[9] Prakhar Gupta, Garv Shah, Satyam Goyal, and Anirudh Kanchi. Sparse memory finetuning
as a low-forgetting alternative to LoRA and full finetuning. arXiv preprint arXiv:2605.03229,
2026.
[10] Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp.
news-please: A
generic news crawler and extractor. In Proceedings of the 15th International Symposium of
Information Science, pages 218–223, 2017. doi: 10.5281/zenodo.4120316. CC-News corpus:
https://huggingface.co/datasets/vblagoje/cc_news.
29


---

[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In
International Conference on Learning Representations, 2022.
[12] Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella Biderman, Baber
Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper, Aviya Skowron, et al. The Com-
mon Pile v0.1: An 8TB dataset of public domain and openly licensed text. arXiv preprint
arXiv:2506.05209, 2025.
[13] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, et al. Kimi
Linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025.
[14] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins,
Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska,
et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National
Academy of Sciences, 114(13):3521–3526, 2017.
[15] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé
Jégou. Large memory layers with product keys. In Advances in Neural Information Processing
Systems, 2019.
[16] Zhiyu Li, Zeyu Zhang, Linyun Liu, Yiwei Bu, Chenyang Xi, et al. MemT-1.0: Towards
memory-native evolution. arXiv preprint arXiv:2610.26825, 2026.
[17] Yan-Shuo Liang and Wu-Jun Li. InfLoRA: Interference-free low-rank adaptation for continual
learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
[18] Jessy Lin, Luke Zettlemoyer, Gargi Ghosh, Wen-tau Yih, Aram Markosyan, Vincent-Pierre
Berges, and Barlas Oğuz. Continual learning via sparse memory finetuning. arXiv preprint
arXiv:2510.15103, 2025.
[19] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study
of catastrophic forgetting in large language models during continual fine-tuning. arXiv preprint
arXiv:2308.08747, 2023.
[20] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and
Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 13851–13870, 2024.
[21] James L. McClelland, Bruce L. McNaughton, and Randall C. O’Reilly.
Why there are
complementary learning systems in the hippocampus and neocortex. Psychological Review, 102
(3):419–457, 1995.
[22] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks:
The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pages
109–165. Academic Press, 1989. doi: 10.1016/S0079-7421(08)60536-8.
[23] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual
associations in GPT. In Advances in Neural Information Processing Systems, 2022.
30


---

[24] Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-
editing memory in a transformer. In International Conference on Learning Representations,
2023.
[25] Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell,
Colin Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the
web for the finest text data at scale. In Advances in Neural Information Processing Systems,
Datasets and Benchmarks Track, 2024.
[26] Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui
Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian,
Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu
Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang
Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng,
and Dayiheng Liu. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and
training stability. arXiv preprint arXiv:2608.30320, 2026.
[27] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast
weight programmers. In Proceedings of the 38th International Conference on Machine Learning,
volume 139 of Proceedings of Machine Learning Research, pages 9355–9366, 2021.
[28] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic
recurrent networks. Neural Computation, 4(1):131–139, 1992. doi: 10.1162/neco.1992.4.1.131.
[29] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois,
Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin.
Learning to (learn at test time): RNNs with expressive hidden states. In Proceedings of the
42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine
Learning Research, pages 57503–57522, 2025.
[30] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui,
and Xuanjing Huang. Orthogonal subspace learning for language model continual learning. In
Findings of the Association for Computational Linguistics: EMNLP 2023, 2023.
[31] Hao Wu. Representation drift in continual learning: LoRA versus N-gram memory. arXiv
preprint arXiv:2610.14921, 2026.
[32] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear
transformers with the delta rule over sequence length. In Advances in Neural Information
Processing Systems (NeurIPS), 2024.
[33] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2
with delta rule. In International Conference on Learning Representations (ICLR), 2025.
[34] Tianyu Zhao and Llion Jones.
Fast-weight product key memory.
arXiv preprint
arXiv:2601.00671, 2026.
31


---

A
Native Implementation and Reproducibility
A.1
Training on Qwen3.8
Layout.
Training uses tensor and pipeline parallelism 1, expert parallelism 4 and data parallelism
4 on four H200 GPUs, with three trajectories per GPU and twelve per update. Gradients are
summed across GPUs and normalised once by the number of target tokens in the whole group. The
pretrained table holds about 51 billion parameters and is kept in CPU memory.
Four-stage schedule.
With T 0-only encoding, ˆH does not depend on the memory, which allows
a staged schedule:
1. encode all blocks of all trajectories in one batch;
2. run the D1 write chain block by block;
3. run all queries forward and backward in one batch, giving the gradient of each block’s patch;
4. backpropagate along the write chain into the writer.
This reaches 3,000–3,300 target tokens per second, against about 940 for the interleaved schedule.
Its loss agrees with the sequential schedule to within 0.001. Gradient differences are at the level of
the routing-noise floor below. Memory-aware encoding requires the sequential schedule.
Small patches in a BF16 model.
The backbone runs in BF16, and a patch of order 0.01·RMS(T 0)
added to T 0 would mostly be rounded away. The read-out is therefore split into a BF16 projection
of the base table and an FP32 projection of the patch, which are added before the injection. The
empty patch read through this split branch is the baseline for every comparison; it is not bitwise
equal to the original BF16 read-out.
Routing-noise floor.
With ten of 512 experts active per token, a patch that differs only in its
last bits can switch the experts of some tokens. Between such near-identical states the writer’s
gradient changes by about 31.6% relative. Schedule equivalence is judged against this floor, not bit
equality.
Training and evaluation agree.
Two fixes made forward passes with and without gradient
tracking bit-identical: fixing the router’s top-k to sorted order in both modes, and running two
fused mixing functions eagerly. Evaluation runs without gradient tracking. With tracking on but
no backward pass, the graph accumulated and memory grew from 68 to 143 GB until evaluation
stalled; without it, memory stays near 78 GB.
A.2
Inference with live writes
For generation we extend vLLM’s Qwen3.8 read-out with the same FP32 patch branch. Normalisa-
tion, gating and convolution of the read-out run in FP32, and the result is cast to BF16 before it
joins the residual stream; the patch is never merged into the BF16 table. Under tensor parallelism 8,
one rank computes each D1 update and all ranks adopt its result. The state is updated only between
requests. Every request checks the state hash before and after, and answering never modifies the
state. Writing a 23,000-token dialogue takes about 12 seconds on A800 GPUs. The state of a
13,000–25,000-token dialogue is 78–145 MB in FP32 without compression.
32


---

A.3
Reproducibility
Every run is bound to the SHA-256 hashes of its source code, checkpoint and data. Exact resumption
of training is allowed only when the code manifest hash matches; a code change goes through a
registered migration that leaves the old sources untouched. Generations are sealed before they
are scored, and key numbers were recomputed independently from raw per-item records. The
DEV64 donor pairs and distractors are fixed by hash before training. The code is available at
https://github.com/af-74413592/delta-ngram.
33
