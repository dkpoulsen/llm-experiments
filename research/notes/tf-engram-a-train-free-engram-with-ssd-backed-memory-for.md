---
title: 'TF-Engram: A Train-Free Engram with SSD-Backed Memory for'
id: tf-engram-a-train-free-engram-with-ssd-backed-memory-for
tags:
- cmp170hx-training-feasibility-a0d4fc
- engram
- ssd-offload
- train-free
- prefetching
- memory-hierarchy
created: '2026-10-27T10:12:13.567634Z'
updated: '2026-10-27T19:00:31.116177Z'
source: https://arxiv.org/abs/2607.07388
source_domain: arxiv.org
fetched_at: '2026-10-27T10:12:13.566747Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'TF-Engram (CUHK + BIT Zhuhai, arXiv 2607.07388) - a train-free Engram that
  builds phrase-specific semantic memory offline from an external corpus, stores tables
  across a GPU-DRAM-SSD hierarchy, and uses Early-Exit Guided Predictive Prefetching
  to hide external-memory latency during autoregressive decoding. Backbone stays frozen;
  memory is injected as a hidden-state pathway. Quality: on Qwen3-0.6B the average
  downstream score rises 57.6 -> 59.4, beating both the frozen backbone and a parameter-matched
  LoRA baseline. Storage numbers that matter for capacity planning: a 100M-entry table
  with 1024-dim FP16 vectors is 215.86 GB on disk; a GPU-resident 10M-entry table
  inflates GPU memory to 24.03 GB, whereas the SSD-backed 100M-entry table uses only
  3.06 GB of GPU memory and shifts the burden to CPU DRAM and NVMe. Offline construction
  cost is modest: 1.612 h for n-gram extraction, 0.231 h embedding generation, 0.157
  h index construction. Runtime cost on the testbed (RTX 5090 32 GB, 64 GB host RAM,
  516 GB SSD, i9-14900K): baseline 481.2 tok/s and 2.08 ms/token; SSD-backed without
  prefetch drops to 439.2 tok/s; Top-64 prefetch recovers to 460.3 tok/s, Top-128
  to 462.4 tok/s. Decode latency 2.108 ms/token at 1M entries -> 2.175 ms/token at
  100M entries (474.6 -> 459.8 tok/s). Best early-exit layer is 8. Note this is a
  single-consumer-GPU inference result, not a training result.'
raw_file: raw/tf-engram-a-train-free-engram-with-ssd-backed-memory-for.pdf
doi: arXiv:2607.07388
utility_score: 14.0
citation_count: 2
venue: arXiv.org
---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for
Large Language Models
Yutang MA
The Chinese University of Hong Kong
ytma2@cse.cuhk.edu.hk
Kecheng HUANG
Beijing Institute of Technology, Zhuhai
huangkecheng@bitzh.edu.cn
Xikun JIANG
The Chinese University of Hong Kong
xkjiang24@cse.cuhk.edu.hk
Zili SHAO
The Chinese University of Hong Kong
shao@cse.cuhk.edu.hk
ABSTRACT
Large Language Models (LLMs) store factual knowledge and domain-
specific patterns implicitly in dense Transformer parameters, mak-
ing knowledge expansion costly through pretraining, fine-tuning,
retrieval augmentation, or longer contexts. Engram-style memory
offers a compact hidden-state injection pathway, but existing GPU-
resident designs often rely on hash-based compression, causing un-
related phrases to collide in shared slots and weakening phrase-level
semantic fidelity. We present TF-Engram, a train-free Engram sys-
tem that constructs phrase-specific semantic memory offline from
external corpora, stores large memory tables across a GPU–DRAM–
SSD hierarchy, and uses Early-Exit Guided Predictive Prefetching
to hide external-memory latency during autoregressive decoding.
On Qwen3-0.6B, TF-Engram improves the average downstream
score from 57.6 to 59.4, outperforming both the frozen backbone
and a parameter-matched LoRA baseline. System evaluation shows
that large TF-Engram tables can be built with moderate offline cost,
SSD-backed storage substantially reduces GPU memory demand,
and predictive prefetching recovers much of the throughput loss
caused by external memory access. These results demonstrate that
static phrase memory can be integrated into LLM inference as a
scalable, train-free, and low-overhead system component.
1
INTRODUCTION
Large Language Models (LLMs) have become the dominant archi-
tecture for natural language understanding, reasoning, code genera-
tion, and knowledge-intensive question answering [3, 7, 10, 28–30].
Their success is largely driven by scaling dense Transformer param-
eters and training data [12, 18]. However, this design also tightly
couples knowledge capacity with model size: factual knowledge,
domain-specific terminology, and frequently occurring semantic
patterns are implicitly stored inside billions of neural parameters.
Improving such knowledge often requires more pretraining, super-
vised fine-tuning, retrieval augmentation, or longer context win-
dows [2, 3, 11, 23]. These solutions increase training cost, inference
latency, memory consumption, or deployment complexity. As LLMs
are increasingly used in domain-specific and knowledge-intensive
applications, it becomes important to decouple static knowledge
storage from dense Transformer computation.
Recent work on memory-augmented LLMs explores this decou-
pling from several directions. Retrieval-Augmented Generation
(RAG) systems maintain an external document corpus and retrieve
relevant passages during inference, usually appending the retrieved
text to the prompt [2, 11, 15, 23]. Long-context and KV-cache based
methods extend the amount of information accessible to the model
by keeping more historical tokens or intermediate states [8, 9, 20].
Adapter-based and parameter-efficient memory methods inject ad-
ditional trainable modules into the model, allowing new knowl-
edge to be stored in a smaller number of parameters than full
fine-tuning [13, 14, 21, 24]. Another line of work, represented by
Engram-style memory, introduces a lightweight side memory path-
way that stores token-level or phrase-level patterns and injects
retrieved memory signals into the hidden states of the backbone
model [5]. These categories differ substantially in memory granu-
larity, training cost, inference overhead, and system scalability.
Although RAG, long-context, and adapter-based methods are
effective in many scenarios, they are not ideal for fine-grained
static phrase memory. RAG retrieves coarse-grained text chunks
rather than compact phrase-level memory entries, and the retrieved
passages increase context length and attention cost [2, 15, 23]. Long-
context methods allow the model to attend to more information,
but they still require the Transformer to process additional tokens,
even when the useful knowledge corresponds to short and stable
phrases [8, 9]. Adapter-based memory methods reduce the cost of
full fine-tuning, but they still require training and are usually tied to
a specific backbone model [13, 14, 21, 24]. In contrast, Engram-style
memory is a more suitable starting point for a system-oriented de-
sign. It provides a compact side pathway, operates closer to token or
phrase granularity, and can be integrated into LLM inference with-
out restructuring the main Transformer computation [5]. Therefore,
this paper focuses on extending Engram-style memory into a scal-
able external memory system for LLMs.
Despite its promise, Engram-style memory is not yet a practical
large-scale static memory system. Existing designs are mainly opti-
mized as compact GPU-resident neural memory modules. To fit a
large number of token patterns into limited accelerator memory,
they often rely on hash-based compression, where many token
sequences are mapped into a fixed-size table. This design reduces
memory footprint, but it introduces a tension between memory
coverage and memory fidelity. To make Engram suitable for large
static phrase knowledge, three critical issues must be addressed:
how to construct phrase-specific memory without additional train-
ing, how to scale memory capacity without relying on ambiguous
hash-collision slots, and how to access external memory without
stalling autoregressive decoding.
arXiv:2607.07388v1  [cs.CL]  8 Jul 2026


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
The first critical issue is that memory construction remains
training-dependent and model-specific. Conventional Engram de-
signs typically learn memory entries, projection layers, gates, or
auxiliary parameters from data, either jointly with the backbone
model or through additional post-training [5, 13, 14]. This makes
the memory difficult to deploy across different LLMs or adapt to
new domains without extra optimization. Moreover, when mem-
ory entries are constructed at token or hash-slot granularity, the
resulting values do not explicitly preserve phrase-level semantics.
A scalable Engram system should therefore be able to construct
phrase-level memory offline, preserve semantic information, and
avoid retraining the backbone model.
The second critical issue is the trade-off between capacity and
semantic fidelity. Existing Engram-style memories can remain com-
pact by mapping many token patterns into a fixed-size hash table.
However, when the candidate phrase space is much larger than
the table, unrelated phrases inevitably collide in the same memory
slots. A single slot may then receive signals from multiple unre-
lated phrases, causing its memory value to become an ambiguous
mixture rather than a clean phrase-specific representation. This
weakens the original goal of static phrase memory: a phrase such as
New York City, quantum field theory, or BGP route reflector should
ideally retrieve a stable memory signal tied to that phrase, not a
value shared with many unrelated patterns. Therefore, simply com-
pressing more phrases into a small GPU table increases apparent
coverage but can reduce the usefulness of the retrieved memory.
Scaling Engram memory requires preserving precise phrase entries
and moving the capacity burden into a storage hierarchy.
The third critical issue is access latency after precise memory
is externalized. Moving memory entries from GPU memory to
host DRAM or NVMe SSD can greatly expand capacity and reduce
hash collisions, but naive external lookup can break the latency
requirements of autoregressive decoding. During generation, each
token depends on previous outputs, and the model has limited
opportunity to wait for random storage accesses. Prior LLM serving
and offloading systems show that memory hierarchy, scheduling,
and data movement are central to efficient inference [1, 20, 27, 32].
A straightforward prefetch design may use the final next-token
distribution produced by the original LLM head. However, this
signal becomes available only after the full forward pass completes,
leaving little remaining computation to overlap with SSD or CPU–
GPU transfer. Therefore, SSD-backed Engram memory requires an
earlier prediction mechanism that can anticipate future memory
requests before the final layers finish.
To address these issues, we propose TF-Engram, a train-free
Engram with SSD-backed memory for large language models. The
key idea is to shift Engram from a small hash-compressed learned
table into a storage-managed external memory system. Instead
of forcing a large phrase space into ambiguous shared slots, TF-
Engram constructs phrase-specific semantic memory offline, places
it across a GPU–DRAM–SSD hierarchy, and uses early-layer pre-
diction signals to prefetch future memory entries before they are
needed.
To address the first issue, TF-Engram constructs semantic phrase
memory entirely offline. It first mines candidate phrases from large-
scale corpora, such as general web text, encyclopedic text, and
domain-specific scientific corpora. These phrases are then tokenized
using the target LLM tokenizer, encoded using an external semantic
encoder, and stored as static key-value memory entries. Unlike
conventional Engram memories that learn memory parameters
from model training, TF-Engram separates memory construction
from backbone optimization. This design allows the memory table
to be built once, reused across inference runs, and extended with
new corpora without retraining the LLM itself.
To address the second issue, TF-Engram organizes memory as an
SSD-backed hierarchy. Frequently accessed entries are kept in GPU
or host-memory caches for low-latency lookup, while cold entries
are offloaded to commodity NVMe SSDs. This hierarchy allows
memory capacity to scale independently from accelerator memory.
Instead of using hash compression as the main mechanism for fitting
memory into GPU, TF-Engram preserves phrase-specific logical
entries and relies on hierarchical storage to absorb the capacity de-
mand. This design makes it possible to support much larger phrase
tables while reserving scarce GPU memory for model weights, acti-
vations, KV cache, and the hottest memory entries [20, 26, 27].
To address the third issue, TF-Engram introduces Early-Exit
Guided Predictive Prefetching. Rather than waiting for the final
language modeling head to produce the next-token distribution,
TF-Engram attaches an auxiliary prefetch head to an intermediate
layer near the end of the LLM, such as layer 𝐿−𝑟. This early-exit
head produces an approximate next-token distribution before the
full forward pass finishes. TF-Engram expands the predicted top-
𝐾tokens into likely future phrase accesses and asynchronously
prefetches their memory entries from SSD or host DRAM while the
remaining Transformer layers continue computation. By using the
remaining layers as a latency-hiding window, TF-Engram reduces
blocking external memory accesses and makes SSD-backed phrase
memory practical for autoregressive inference.
Our evaluation on Qwen3-0.6B shows that TF-Engram improves
end-to-end model quality while keeping the backbone frozen. Across
ten downstream benchmarks, TF-Engram increases the average
score from 57.6 to 59.4, outperforming both the original backbone
and a parameter-matched LoRA baseline. The system analysis fur-
ther shows that large TF-Engram tables can be constructed with
moderate offline cost, while the SSD-backed hierarchy substantially
reduces GPU memory demand compared with GPU-resident mem-
ory. With early-exit guided prefetching, TF-Engram recovers much
of the throughput loss caused by external memory access and keeps
decoding latency overhead modest. These results indicate that TF-
Engram provides a practical path toward train-free, scalable, and
low-overhead memory augmentation for LLM inference.
2
BACKGROUND
2.1
Memory-Augmented LLMs
Large Language Models (LLMs) are typically implemented as decoder-
only Transformers [3, 28–30]. Given an input sequence𝑥= 𝑥1,𝑥2, . . . ,𝑥𝑡,
the model first maps token IDs into continuous embeddings and
then processes them through a stack of Transformer layers. At each
layer, the model refines the hidden representation of every token
by applying self-attention and feed-forward transformations [30].
After the final layer, the hidden state at position 𝑡is projected by
the language modeling head to produce a probability distribution


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
over the next token:
𝑝(𝑥𝑡+1 | 𝑥≤𝑡) = softmax(𝑊lmℎ𝐿
𝑡),
(1)
where ℎ𝐿
𝑡denotes the final-layer hidden state and 𝐿is the number
of Transformer layers.
During autoregressive decoding, this process is repeated token
by token [3, 29]. Since each generated token depends on previously
generated tokens, LLM inference is inherently sequential at the
decoding stage. Modern inference systems reduce redundant com-
putation by caching previous key-value states, but each new token
still requires a forward pass through the model [20, 32]. Therefore,
any additional operation placed on the per-token critical path can
directly affect latency and throughput. This latency-sensitive prop-
erty is especially important when integrating external memory,
because memory lookup, data movement, and memory injection
may all become part of the decoding pipeline.
A fundamental property of standard LLMs is that most knowl-
edge is stored implicitly in dense parameters [3, 12, 18]. Factual
associations, entity knowledge, technical terminology, and recur-
ring linguistic patterns are encoded through pretraining rather
than stored in an explicit external structure. This design is power-
ful because it allows the model to generalize from large-scale data,
but it also creates several system-level limitations. First, increasing
knowledge capacity often requires increasing model size, which
raises both training and inference cost [7, 12, 18]. Second, updating
knowledge is difficult because modifying a small set of facts may
require fine-tuning or retraining [13, 14]. Third, static knowledge
and dynamic computation are tightly coupled: even if a piece of
knowledge is stable and repeatedly used, the model still retrieves it
through dense Transformer computation.
Memory-augmented LLMs aim to address this limitation by in-
troducing external or auxiliary memory [2, 5, 11, 19, 23]. Instead of
relying solely on parametric knowledge, the model can access addi-
tional information through a memory pathway. Existing memory-
augmented methods differ in what they store, how they access
memory, and where the memory is integrated into the inference
pipeline. From a system perspective, these methods can be roughly
categorized into retrieval-based memory, context-based memory,
parameter-based memory, and internal side-pathway memory.
Retrieval-Augmented Generation (RAG) is the most widely used
form of external memory [11, 15, 23]. A RAG system maintains a
corpus of documents or passages outside the model. Given an input
query, the system computes a retrieval query, searches an exter-
nal index, and appends the retrieved passages to the prompt. The
LLM then consumes the original input together with the retrieved
context. This approach is flexible because the corpus can be up-
dated independently from the model parameters. It is also effective
for knowledge-intensive tasks where relevant information can be
found in documents [2, 15, 23]. However, RAG usually operates
at coarse granularity. The retrieved units are typically paragraphs
or passages, not compact semantic memories. As a result, RAG
increases prompt length and attention cost, even when the useful
information is only a short phrase or entity. Moreover, retrieval
and prompt construction are often outside the model’s internal
computation, making RAG less suitable for fine-grained token-level
or phrase-level memory injection.
Context-based memory methods take a different approach. Long-
context models, sliding-window attention, recurrent memory mech-
anisms, and KV-cache reuse techniques allow the model to access
more historical information [8, 9, 20]. These methods are useful
when relevant information already appears in the input context
or previous conversation. However, they do not directly solve the
problem of storing large amounts of reusable external knowledge. A
long-context model still needs to process additional tokens through
attention. This is expensive when the same static knowledge is
repeatedly needed across many requests. KV-cache based meth-
ods preserve intermediate states for previously processed tokens,
but the cached content is request-specific rather than a general-
purpose memory table [20]. Therefore, context-based memory is
better suited for preserving dynamic conversational history than
for storing large-scale static phrase knowledge.
Parameter-based memory methods store new information by
adding or updating trainable parameters. Examples include adapters,
LoRA-style modules, memory layers, and other parameter-efficient
tuning mechanisms [13, 14, 21, 24]. Compared with full fine-tuning,
these methods reduce the number of updated parameters and can
be easier to deploy. They are useful when a model needs to adapt
to a new domain or task. However, they still require optimization.
The learned parameters are usually tied to a specific backbone ar-
chitecture, hidden dimension, tokenizer, and training objective. If
the backbone model changes, the memory module may need to be
retrained or redesigned. Furthermore, parameter-based memory
does not naturally provide explicit indexing or storage hierarchy:
once knowledge is absorbed into parameters, it becomes difficult
to selectively retrieve, update, evict, or offload.
Internal side-pathway memory methods introduce a more com-
pact integration point [5, 19]. Instead of appending retrieved text
to the prompt or modifying a large number of parameters, these
methods add a memory pathway inside the model’s inference pro-
cess. The memory is usually represented as a key-value structure.
During inference, the model forms a query from token history or
hidden states, retrieves one or more memory values, and injects
the retrieved signal into the hidden representation. This design
has two important advantages. First, it avoids increasing the input
context length. Second, it allows memory access to operate at a
finer granularity than document retrieval. Engram-style memory
belongs to this category [5].
Another useful property of decoder-only Transformers is that in-
termediate layers already contain partial predictive information [4,
22]. Although the final next-token distribution is produced only
after all 𝐿layers finish, an intermediate hidden state ℎℓ
𝑡, where
ℓ< 𝐿, can also be projected to produce an approximate next-token
distribution. Such an auxiliary prediction does not need to replace
the final model output. Instead, it can be used for system-level
scheduling decisions, such as prefetching likely future memory
entries before the full forward pass completes. This property is
important for latency-sensitive memory systems because it creates
an opportunity to overlap memory transfer with the computation
of the remaining Transformer layers.
These memory categories reflect different trade-offs. RAG pro-
vides flexible document-level access but adds retrieval and context-
processing overhead [15, 23]. Long-context methods preserve more
dynamic history but are expensive for static knowledge [8, 9].


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
Parameter-based methods can encode new information but require
training and are model-specific [13, 14]. Side-pathway memory
provides a compact mechanism for inference-time knowledge in-
jection, making it attractive for system-level optimization [5]. This
paper focuses on Engram-style side-pathway memory because it
is naturally aligned with phrase-level memory, does not require
prompt expansion, and can potentially be implemented as an effi-
cient external memory system.
2.2
Engram-Style Memory
Engram-style memory is designed to supplement the Transformer
backbone with an auxiliary memory pathway[6]. The core idea is
to store reusable token-level or phrase-level patterns in a memory
table and retrieve relevant memory signals during inference. Unlike
RAG, which retrieves raw text and feeds it back into the model as
additional context [2, 23], Engram-style memory retrieves hidden-
space values and injects them directly into the model representation.
This makes the memory more compact and avoids increasing the
sequence length processed by attention.
A generic Engram-style memory can be abstracted as a key-value
table:
M = (𝑘𝑖, 𝑣𝑖)𝑁
𝑖=1,
(2)
where 𝑘𝑖is a memory key and 𝑣𝑖is the corresponding memory
value. The key is used for lookup, while the value is used to modify
the model’s hidden state. Depending on the design, the key may be
derived from token IDs, token 𝑛-grams, hashed token sequences,
phrase identifiers, or hidden-state representations. The value is
usually represented in the same or a compatible vector space as
the LLM hidden states, so that it can be added to or fused with the
model’s internal representation [5, 19].
During inference, the model computes a hidden state ℎ𝑡for the
current position 𝑡. The memory module then derives a query 𝑞𝑡
from the current context. This query may depend on the recent
token sequence, the current hidden state, or both. The memory
lookup function returns a retrieved memory signal:
𝑚𝑡= 𝑀(𝑞𝑡),
(3)
where 𝑀(·) denotes the memory lookup and aggregation function.
If multiple memory entries are matched, the returned signal may
be computed through weighted aggregation:
𝑚𝑡=
∑︁
𝑖∈N(𝑞𝑡)
𝛼𝑖𝑣𝑖,
(4)
where N (𝑞𝑡) is the set of retrieved entries and 𝛼𝑖is the retrieval
weight for entry 𝑖.
The retrieved memory signal is then injected into the model. A
common formulation is residual addition with a gate:
ℎ′
𝑡= ℎ𝑡+ 𝑔𝑡· 𝑚𝑡,
(5)
where ℎ′
𝑡is the memory-augmented hidden state and 𝑔𝑡controls
how strongly the memory affects the model. The gate may be a
scalar, a vector, or a learned function of the current hidden state and
retrieved memory. This gating mechanism is important because
not every memory retrieval is useful. When the memory signal
is relevant, the gate can amplify it; when the memory is noisy or
unnecessary, the gate can suppress it [5].
Engram-style memory has several properties that make it at-
tractive for LLM systems. First, it is compact. Instead of storing
full documents or long passages, it stores structured memory en-
tries associated with token or phrase patterns. This allows memory
to represent frequently reused knowledge units more efficiently.
Second, it does not require prompt expansion. Retrieved memory
values are injected into hidden states rather than appended to the
input sequence, avoiding additional attention over retrieved text.
Third, it can be integrated as a side pathway. The main Transformer
computation remains largely unchanged, which makes the design
compatible with existing decoder-only LLM architectures [3, 29, 30].
A practical Engram-style system must also control memory size.
The number of possible token 𝑛-grams or phrase patterns grows
rapidly with corpus size. Directly assigning an independent vector
to every possible pattern would quickly exceed GPU memory. To
keep the memory table compact, existing designs often use hash-
based compression. A hash function maps many token sequences
into a fixed-size table:
𝑏= Hash(𝜏) mod 𝐵,
(6)
where 𝜏is a token sequence, 𝐵is the number of physical memory
slots, and 𝑏is the selected slot. This allows the table size to remain
fixed even when the number of possible token patterns is much
larger than 𝐵.
Hash-based compression provides an efficient way to fit memory
into GPU, but it also changes the meaning of a memory entry. A
physical slot no longer necessarily corresponds to a single phrase
or token pattern. Instead, it may be shared by many different se-
quences. If these sequences are semantically related, the shared
value may still provide a useful signal. However, if unrelated phrases
collide in the same slot, the learned value can become an ambiguous
mixture of multiple patterns. The retrieved memory signal may
then be less phrase-specific, or even misleading.
This issue is particularly important for static phrase memory.
Many useful knowledge units are short multi-token expressions,
such as named entities, technical terms, scientific concepts, product
names, organization names, and recurring domain-specific phrases.
These phrases often carry meanings that are not fully captured by
individual token embeddings. A phrase-level memory pathway can
therefore provide a more direct representation of such semantic
units. However, if multiple unrelated phrases are compressed into
the same hash slot, the memory system loses the ability to pre-
serve clean phrase-specific signals. Compared with document-level
retrieval, phrase-level memory is more compact; compared with
token-level lookup, it better matches semantic boundaries; but com-
pared with hash-collided memory, it requires more precise logical
entries.
Conventional Engram-style designs are therefore not yet suffi-
cient for large-scale train-free static memory. First, memory entries
are often learned from data [5]. This means the memory table is
not simply a static external database; it is part of the model opti-
mization process. As a result, constructing or adapting memory
requires additional training. Second, existing Engram memories
are often model-specific. The value vectors must match the hidden
space of a particular LLM, and the lookup or injection mechanism
may depend on the model’s architecture and hidden dimension.


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
Third, compact GPU-resident implementations often rely on com-
pression or hashing, which improves storage efficiency but can
weaken phrase-level semantic fidelity under heavy collision.
These limitations become more serious when the goal is to build
a large static memory from external corpora. A corpus-derived
phrase memory may contain millions or billions of candidate en-
tries. Such a table is too large to fit entirely in GPU memory if every
phrase is represented precisely [20, 26, 27]. At the same time, com-
pressing it aggressively into a small hash table can cause unrelated
phrases to share memory values. Moreover, retraining all memory
entries together with the LLM would be prohibitively expensive.
Static phrase knowledge is naturally suited for offline construction:
once phrases are mined and encoded, many entries can remain un-
changed across inference runs. This suggests that Engram should
be reinterpreted not only as a neural memory module, but also as a
storage-backed memory system.
This paper builds on this reinterpretation. We use Engram-style
memory as the integration mechanism because it provides a com-
pact side pathway for hidden-state injection. At the same time,
we redesign its construction, storage, and prefetch assumptions.
Instead of learning all memory entries through model training,
TF-Engram constructs phrase memories offline. Instead of relying
primarily on hash compression to fit memory into GPU, TF-Engram
preserves phrase-specific logical entries and organizes them across
GPU cache, host memory, and SSD. Instead of waiting for final out-
put predictions to prefetch memory, TF-Engram uses an early-exit
prediction signal from an intermediate layer to overlap memory
transfer with the computation of the remaining Transformer layers.
3
MOTIVATION
3.1
Hash-Compressed Engram Weakens Static
Memory Capability
Engram-style memory is designed to provide a compact side path-
way for injecting reusable knowledge into LLM hidden states. Ide-
ally, each memory entry should represent a meaningful token pat-
tern or phrase, and the retrieved value should provide a stable signal
that helps the model predict the next token. This design is attractive
because many useful knowledge units in language are phrase-level:
named entities, technical terms, scientific concepts, organization
names, geographic expressions, and frequent multi-token expres-
sions. Such phrases often have stable semantics and repeatedly
appear across corpora. Therefore, they are natural candidates for
static memory.
However, directly storing all useful phrases is difficult. The num-
ber of possible token 𝑛-grams grows rapidly with corpus size, and
a corpus-scale phrase table can easily contain millions or billions
of candidates. If each phrase is assigned an independent memory
value, the table quickly exceeds practical GPU memory limits. Ex-
isting Engram-style systems address this scalability problem by
using a fixed-size hash-based table. Instead of allocating one unique
slot for every possible token pattern, many token sequences are
mapped into a smaller number of memory slots through hashing.
This allows the memory table to remain compact and GPU-resident.
This hash-based compression solves the storage problem, but
introduces a more fundamental memory-quality problem. When
Table 1: Hash-compressed memory creates a mismatch be-
tween represented phrase space and physical memory slots.
When the candidate phrase space is much larger than the
table, many phrases must share the same slot.
Candidate Phrases
Hash Slots
Avg. Phrases per Slot
1M
1M
1.0
100M
1M
100.0
1B
1M
1000.0
many phrases are mapped into a fixed-size table, different token pat-
terns inevitably collide in the same memory slot. A single slot may
therefore receive training signals from multiple unrelated phrases.
The learned value in that slot no longer represents a specific phrase
or semantic unit; instead, it becomes an average or mixture of all
collided patterns. If the collided phrases are semantically related,
the shared value may still be useful. However, if they are unrelated,
the retrieved memory value becomes ambiguous. In the worst case,
the memory signal can be actively misleading because the model
retrieves a value influenced by a different phrase.
This issue is especially problematic for static phrase memory.
The goal of static phrase memory is to preserve stable phrase-level
knowledge. For example, a phrase such as New York City, quantum
field theory, or BGP route reflector should ideally correspond to a
memory entry that preserves its specific semantic meaning. Hash
collision breaks this assumption. If multiple unrelated phrases share
the same slot, the table cannot provide a clean phrase-specific mem-
ory signal. Thus, the hash table may be compact, but its memory
capability is weakened by semantic ambiguity.
The severity of this problem increases as the number of candi-
date phrases grows. Suppose 𝑀distinct phrases are mapped into
𝐵hash slots. When 𝑀is much larger than 𝐵, most slots will con-
tain multiple phrases. The expected number of occupied slots is
approximately:
𝐵
 
1 −

1 −1
𝐵
𝑀!
,
(7)
and the average number of phrases per occupied slot grows as the
phrase set expands. Therefore, increasing the corpus size without
increasing the table size does not linearly increase usable memory
capacity. Instead, it increases collision pressure. The memory table
appears to cover more phrases, but many of these phrases are no
longer represented independently.
This creates a tension in existing Engram designs. On one hand,
increasing the number of covered phrases is necessary for stronger
memory capability. A small memory table can only capture a limited
number of useful static patterns. On the other hand, if a large phrase
space is compressed into a small hash table, the resulting collisions
make the memory values less precise. The system therefore faces
an unfavorable trade-off: either maintain a small table with limited
coverage, or use hash compression to increase apparent coverage
while sacrificing phrase-specific semantic clarity.
This trade-off is inconsistent with the original motivation of
static memory. Static phrase knowledge should be explicit, stable,
and reusable. A memory system should retrieve a value because the
current phrase matches a meaningful stored entry, not because the


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
phrase happens to collide with many other patterns in a compressed
hash table. Therefore, the key limitation of existing Engram-style
memory is not only that GPU memory is limited. The deeper issue
is that hash-based compression uses ambiguous shared slots to ap-
proximate a much larger phrase space, which reduces the semantic
fidelity of the retrieved memory.
This observation motivates a different design point. Instead of
forcing a large phrase space into a small GPU-resident hash ta-
ble, we should preserve phrase-specific memory entries and move
the capacity burden into a storage hierarchy. In other words, En-
gram memory should scale by externalizing storage rather than
by increasing hash collision. This allows the memory system to
maintain cleaner phrase-level semantics while supporting much
larger memory capacity.
3.2
Externalizing Precise Engram Memory
Introduces New Challenges
Externalizing Engram memory is a natural way to avoid exces-
sive hash compression. If memory entries can be stored outside
GPU memory, the system no longer needs to map a massive phrase
space into a small number of slots. Instead, each important phrase
can have a more precise memory representation, and the table can
scale with corpus size. However, this design changes the nature
of Engram from a compact neural table into a storage-backed ex-
ternal memory system. Once the memory is externalized, three
challenges must be addressed: how to construct phrase-specific
memory without training a huge parameter table, how to organize
the memory across GPU, DRAM, and SSD, and how to hide the
latency of external lookup during autoregressive decoding.
Challenge 1: Large precise memory cannot be trained as a
conventional Engram table. If each phrase is assigned a distinct
memory entry, the table may contain millions or billions of entries.
Treating such a table as ordinary trainable parameters is impractical.
Training would require large-scale optimization, significant GPU
memory, and repeated access to massive corpora. It would also
make the memory tightly coupled with a specific backbone model,
hidden dimension, and training objective. When the corpus changes
or a new domain is added, the table may need additional training.
This defeats the purpose of using static phrase memory as a reusable
external knowledge structure.
A scalable memory construction method should therefore avoid
learning every memory entry through LLM training. Since many
phrase memories correspond to stable semantic units, they can be
constructed offline. The system can mine candidate phrases from
large corpora, encode their meanings using an external semantic
encoder, and store them as static memory entries. This construction
process better matches the nature of phrase-level knowledge: the
memory entry is tied to the phrase itself rather than to a hash slot
shared by unrelated phrases. The challenge is to build such entries
in a train-free way while still making them compatible with the
LLM inference pathway.
Challenge 2: Precise memory requires storage-aware hier-
archy instead of hash-only compression. Removing or reduc-
ing hash compression increases memory fidelity, but also increases
storage demand. A phrase-specific table can be much larger than a
compressed Engram table. Keeping the entire table in GPU memory
is unrealistic because GPU memory is already occupied by model
weights, KV cache, activations, and runtime buffers. Therefore, pre-
cise Engram memory must be organized across multiple storage
tiers. Hot entries should remain close to the model, while cold en-
tries should be placed in cheaper and larger storage such as host
DRAM or NVMe SSD.
This creates a storage-management problem. GPU memory pro-
vides low latency but limited capacity. Host DRAM provides more
capacity but requires CPU–GPU data movement. SSD provides
much larger capacity but has much higher random-access latency.
The system must decide which entries should stay in GPU cache,
which entries should be staged in DRAM, and which entries should
remain on SSD. It must also support efficient lookup without turn-
ing phrase matching into random blocking I/O. Therefore, scaling
precise Engram memory requires a storage-aware layout, index
structure, and cache policy. The system should use SSD capacity to
preserve phrase-specific memory entries, while keeping frequently
accessed entries in faster tiers.
Challenge 3: External lookup latency must be hidden be-
fore it reaches the decoding critical path. Even if SSD-backed
memory solves the capacity problem, it introduces a latency prob-
lem. Autoregressive decoding is sequential: each generated token
depends on previous tokens, and each step has a strict latency bud-
get. If the model encounters a phrase and then waits for a blocking
SSD read or CPU–GPU transfer, the entire decoding pipeline stalls.
This is unacceptable for LLM serving, where per-token latency di-
rectly affects throughput and user-perceived responsiveness. There-
fore, precise external memory cannot rely on naive on-demand SSD
lookup.
A straightforward predictive design may attempt to use the final
next-token distribution produced by the original language modeling
head. However, this distribution becomes available only after all
Transformer layers have completed the forward pass for the current
token. At that point, there is little remaining computation with
which to overlap memory movement. In other words, final-output-
guided prefetching can predict future memory accesses, but it is
often too late to hide SSD or host-to-device transmission latency.
The system must instead predict memory accesses earlier. Decoder-
only Transformers naturally provide intermediate hidden states
before the final output layer. If an auxiliary early-exit head is at-
tached to an intermediate layer near the end of the network, such as
layer 𝐿−𝑟, it can produce an approximate next-token distribution
before the full forward pass completes. This early prediction can
be used only for memory scheduling, while the final token remains
produced by the original LLM head after all layers finish. The re-
sulting prefetch request can be issued while the remaining layers
𝐿−𝑟+ 1, . . . , 𝐿are still being computed.
This creates a useful latency-hiding window. During the compu-
tation of the remaining Transformer layers, the memory system can
asynchronously fetch candidate phrase entries from SSD or host
DRAM and stage them into faster tiers. If the predicted phrase is ac-
tually needed by the time memory injection occurs, the lookup can
be served from GPU cache or host memory rather than triggering a
blocking external access. The challenge is to balance three factors:
placing the early-exit head early enough to expose a useful overlap
window, placing it late enough to preserve prediction quality, and
limiting the top-𝐾prefetch budget to avoid excessive wasted I/O.


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
These challenges define the design requirements for TF-Engram.
First, memory construction should be train-free and semantic, so
that phrase-specific entries can be built offline without learning
a large hash-compressed parameter table. Second, memory stor-
age should be hierarchical, so that the system can preserve precise
phrase entries without requiring all of them to fit in GPU memory.
Third, memory access should be early-predicted and asynchro-
nous, so that SSD and CPU–GPU transfer latency can be hidden
behind the computation of remaining Transformer layers rather
than exposed on the token-generation critical path. Together, these
requirements motivate TF-Engram as a train-free, SSD-backed, and
early-exit-guided extension of Engram-style memory.
4
TF-ENGRAM DESIGN
4.1
Overview
Figure 1 presents the overall design of TF-Engram. TF-Engram
transforms Engram-style memory from a compact hash-compressed
GPU table into a train-free, SSD-backed external memory system.
Its main goal is to preserve phrase-specific static memory entries
while scaling memory capacity beyond accelerator memory and
hiding the latency of external memory access during autoregressive
decoding.
As shown in Figure 1, TF-Engram consists of three components.
The first component, Train-Free Semantic Memory Construction, is
performed offline. Starting from large-scale corpora such as web
text, Wikipedia, and scientific text, TF-Engram mines reusable
phrase candidates, such as bigrams, trigrams, and other phrase-
level semantic units. These phrases are then organized into a phrase
memory table, where each entry maps a token prefix to a phrase
identifier and a phrase embedding. Unlike conventional Engram
systems that rely on learned or hash-compressed memory slots,
TF-Engram explicitly preserves phrase-specific memory entries
constructed offline.
The second component, SSD-Backed Memory Hierarchy, orga-
nizes the resulting memory table into a hierarchical storage system.
Instead of forcing all memory entries to reside in GPU memory,
TF-Engram separates memory into three tiers: a hot GPU cache, a
warm host DRAM cache, and a cold NVMe SSD store. This hierarchy
allows TF-Engram to preserve a much larger phrase memory table
while keeping frequently accessed entries close to the model. As
a result, TF-Engram scales memory capacity through hierarchical
storage rather than through aggressive hash compression.
The third component, Early-Exit Guided Predictive Prefetching,
operates during online inference. Given an input token stream,
the backbone LLM processes the current decoding step as usual.
Before the full forward pass completes, TF-Engram taps an in-
termediate layer near the end of the backbone, denoted as layer
𝐿−𝑟, and attaches an early-exit prefetch head. This head predicts
top-𝐾candidate phrases that are likely to be accessed in the near fu-
ture. TF-Engram then asynchronously prefetches the corresponding
memory entries from the hierarchical memory system while the re-
maining Transformer layers continue computation. Meanwhile, the
original final LLM head produces the final hidden representation,
and the fetched memory is incorporated through gated residual
injection. By overlapping memory fetch with the computation of
the remaining layers, TF-Engram hides much of the latency of SSD
and host-to-device transfer.
Overall, TF-Engram separates offline construction from online
inference. Offline, it builds precise phrase-level memory entries
without training the backbone model. Online, it combines hier-
archical storage with early-exit-guided predictive prefetching to
make large external phrase memory practical for autoregressive
LLM inference.
4.2
Train-Free Semantic Memory Construction
The first component of TF-Engram is an offline construction pipeline
for phrase-specific semantic memory. The goal is to build a large
phrase memory table without training memory entries as model
parameters. This differs from conventional Engram-style memory,
where memory values are often learned through model training and
compressed into a fixed-size hash table. TF-Engram treats phrase
memory as a static knowledge structure that can be constructed
from corpora, stored externally, and reused during inference.
The construction pipeline begins with phrase mining. Given a
corpus C, TF-Engram extracts candidate multi-token expressions
such as bigrams, trigrams, named entities, technical terms, scientific
concepts, organization names, and domain-specific phrases. Candi-
date phrases can be selected using frequency thresholds, statistical
association, named-entity recognition, or corpus-specific filters.
The objective is not to enumerate every possible token sequence,
but to select phrases that correspond to reusable semantic units.
After phrase extraction, each phrase is tokenized using the target
LLM tokenizer:
𝜏𝑖= Tokenizer(𝜙𝑖).
(8)
The tokenized form 𝜏𝑖is used for inference-time matching, while
the original phrase string 𝜙𝑖is used for semantic encoding and meta-
data management. TF-Engram can filter out phrases with unstable
tokenization, excessive length, or low corpus support. This step en-
sures that the phrase memory table remains useful for token-stream
lookup during LLM inference.
TF-Engram then encodes each phrase using a frozen semantic
encoder, instantiated with Qwen3-Embedding in our implementa-
tion [33]:
𝑒𝑖= 𝐸(𝜙𝑖),
(9)
where 𝐸(·) is an external embedding model and 𝑒𝑖is the semantic
memory vector. This is the central difference from hash-compressed
Engram. Instead of assigning many unrelated phrases to a shared
memory slot and learning a mixed value, TF-Engram assigns each
selected phrase its own semantic representation. As a result, the
memory value remains tied to the phrase itself rather than to an
ambiguous hash bucket.
To support efficient inference-time lookup, TF-Engram builds a
two-level index:
token prefix →phrase id →memory address.
(10)
The first level maps token prefixes or suffix candidates to phrase
identifiers. During inference, the phrase matcher only needs to
verify a compact candidate list instead of scanning the full memory
table. The second level maps phrase identifiers to memory metadata,
including phrase length, semantic vector address, frequency score,
storage tier, and physical offset.


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
Figure 1: Technique overview of TF-Engram.
When multiple phrases match the current context, TF-Engram
uses a deterministic selection policy. A simple policy is longest-
match selection, where longer phrases are preferred because they
usually correspond to more complete semantic units. If several
phrases have the same length, TF-Engram can use frequency, se-
mantic confidence, or corpus priority as tie-breakers. If multiple
matched phrases are retained, their memory vectors can be aggre-
gated:
𝑒𝜙=
∑︁
𝑖∈P𝑡
𝛼𝑖𝑒𝑖,
(11)
where P𝑡is the set of matched phrases at position 𝑡and 𝛼𝑖is
determined by phrase length, frequency, or confidence.
This construction is train-free in the sense that the phrase mem-
ory entries are not optimized through LLM training. The memory
table is built by corpus processing, tokenization, and frozen se-
mantic encoding. Adding a new corpus only requires mining and
encoding new phrases, rather than retraining the backbone model
or re-optimizing a large memory parameter table.
4.3
SSD-Backed Memory Hierarchy
The second component of TF-Engram is an SSD-backed memory
hierarchy. The purpose of this hierarchy is to preserve phrase-
specific memory entries without requiring the entire table to fit in
GPU memory. Instead of using hash compression to force a large
phrase space into a small GPU-resident table, TF-Engram keeps
precise logical entries and uses storage hierarchy to absorb the
capacity demand.
TF-Engram organizes memory into three tiers:
M = M ∗𝑔𝑝𝑢∪M ∗𝑑𝑟𝑎𝑚∪M ∗𝑠𝑠𝑑,
(12)
where M ∗𝑔𝑝𝑢stores hot entries, M ∗𝑑𝑟𝑎𝑚stores warm entries,
and M ∗𝑠𝑠𝑑stores cold entries. The GPU tier provides the lowest
latency and is reserved for the most frequently accessed entries.
The DRAM tier provides a larger staging cache and metadata store.
The SSD tier stores the full memory table and absorbs the long tail
of phrase entries.
A memory lookup proceeds through the hierarchy. When a
phrase is matched, TF-Engram first checks whether the correspond-
ing memory value is already in GPU cache. If so, the value can be
used immediately for hidden-state injection. If the entry is in host
DRAM, it is staged to GPU memory. If the entry is only on SSD, the
system issues an asynchronous read and inserts the entry into a
faster tier when it arrives. To reduce random I/O overhead, memory
entries can be stored in pages or blocks, allowing multiple nearby
vectors and metadata records to be fetched together.
The cache policy is important because Engram memory access
is sparse and data-dependent. Unlike parameter offloading, where
access order is largely determined by model layer order, phrase
memory access depends on the token stream and generated con-
tent. TF-Engram can rank memory entries using access frequency,
recency, phrase length, domain priority, and prefetch confidence.
Frequently used general phrases remain close to the model, while
rare long-tail phrases stay on SSD until needed.
The SSD layout is designed to reduce exposed storage latency.
Memory vectors are stored in contiguous arrays, and phrase meta-
data maps phrase identifiers to SSD offsets. Phrases that are likely
to be accessed together can be placed nearby, such as phrases shar-
ing a token prefix or belonging to the same corpus domain. This
improves spatial locality for batched prefetch requests. TF-Engram
can also use memory-mapped files or asynchronous I/O interfaces
to overlap SSD reads with LLM computation.
This hierarchy allows TF-Engram to scale memory capacity inde-
pendently from GPU memory. GPU memory is reserved for model
weights, KV cache, activations, runtime buffers, and the hottest
phrase entries. SSD storage holds the large static phrase memory.


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
Figure 2: Early-exit guided predictive prefetching.
Therefore, TF-Engram can preserve phrase-specific entries without
collapsing them into ambiguous hash-collision slots.
4.4
Early-Exit Guided Predictive Prefetching
The third component of TF-Engram is Early-Exit Guided Predictive
Prefetching. SSD-backed memory provides large memory capacity,
but a blocking access to host DRAM or NVMe SSD during autore-
gressive decoding can directly stall token generation. Therefore,
TF-Engram must not only store a large static memory table, but also
move likely useful phrase entries into faster memory tiers before
they are accessed.
Figure 2 shows the overall design. A straightforward design
would use the final next-token distribution produced by the orig-
inal LM head to decide which entries to prefetch. However, this
distribution is only available after all Transformer layers complete.
At that point, there is little remaining computation with which to
overlap the memory transfer. TF-Engram instead predicts future
memory accesses from an intermediate layer. The early prediction
is used only for memory scheduling, while the final generated token
is still produced by the original LM head after the full forward pass.
For an LLM with 𝐿Transformer layers, TF-Engram places an
auxiliary early-exit prefetch head at layer 𝐿−𝑟, where 𝑟is the
number of remaining layers. Given the intermediate hidden state
ℎ𝐿−𝑟
𝑡
at decoding step𝑡, the early-exit head produces an approximate
next-token distribution:
𝑝early(𝑥𝑡+1 | 𝑥≤𝑡) = softmax

𝑊prefℎ𝐿−𝑟
𝑡

.
(13)
Here,𝑊pref denotes the projection used by the prefetch head. The re-
sulting distribution is not used for generation. It only predicts likely
future memory accesses. The final output token is still computed
by the original LM head after all 𝐿Transformer layers finish.
TF-Engram then selects the top-𝐾likely future tokens from the
early distribution:
T early
𝑡
= TopK  𝑝early(𝑥𝑡+1 | 𝑥≤𝑡) .
(14)
These candidate tokens are combined with the recent token history
to infer possible future phrases. Using the token-to-phrase index, TF-
Engram maps the predicted token candidates to a set of candidate
phrase entries:
Q𝑡=
n
𝜙𝑖| 𝜏𝑖∈M(T early
𝑡
)
o
.
(15)
Here, M(·) maps the early predicted token candidates to phrase
entries through the token-to-phrase index. The main benefit of
early-exit prediction is latency overlap. Once the prefetch request
is issued at layer 𝐿−𝑟, the backbone LLM continues computing
the remaining layers 𝐿−𝑟+ 1, . . . , 𝐿. During this time, TF-Engram
can move candidate entries from NVMe SSD to host DRAM, or
from host DRAM to the GPU cache. The available latency-hiding
window can be approximated as:
𝑇hide ≈𝑇compute (𝐿−𝑟+ 1, . . . , 𝐿) ,
(16)
where𝑇compute(𝐿−𝑟+1, . . . , 𝐿) denotes the computation time of the
remaining Transformer layers. If a prefetched entry is later selected
by the actual phrase lookup, the access can be served from a faster
tier instead of triggering a blocking SSD access.
The early-exit prefetch head can be implemented in either a train-
free or lightly calibrated form. In the strict train-free setting, TF-
Engram reuses a frozen projection derived from the backbone model
to estimate the early next-token distribution, without updating
the backbone model or the Engram table. If a small calibration
stage is allowed, 𝑊pref can be trained on unlabeled text using the
standard next-token objective while keeping both the backbone
model and the memory table frozen. In both cases, the early-exit
head affects only prefetch scheduling and does not change the final
token probabilities produced by the LLM.
The position of the early-exit head introduces a trade-off be-
tween prediction quality and overlap opportunity. Placing the head
earlier provides a larger latency-hiding window, but the intermedi-
ate hidden state may produce a less accurate next-token prediction.
Placing the head later improves prediction quality, but leaves less re-
maining computation to hide memory-transfer latency. TF-Engram
therefore treats the early-exit layer position, the top-𝐾prefetch
budget, and the prefetch queue size as tunable system parameters.
TF-Engram further prioritizes prefetch candidates according
to both prediction likelihood and expected memory benefit. Each
candidate phrase 𝜙𝑖is assigned a priority score:
𝑠𝑖= 𝑝𝑖· 𝑟𝑖· 𝑐𝑖,
(17)
where 𝑝𝑖represents the probability of the predicted token path, 𝑟𝑖
estimates the usefulness of the phrase entry, and 𝑐𝑖reflects the ex-
pected benefit of prefetching the entry from its current storage tier.
For example, an entry currently residing on SSD receives a higher
prefetch benefit than an entry already cached in GPU memory.
High-priority candidates are prefetched first, while low-priority
candidates are dropped when the I/O budget or queue capacity is
exhausted.
Overall, early-exit guided predictive prefetching enables TF-
Engram to use large SSD-backed memory without exposing the full


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
Table 2: Hardware configuration used in our evaluation.
Component
Configuration
CPU
Intel Core i9-14900K
CPU cores / threads
24 cores / 32 threads
Max CPU frequency
6.0 GHz
GPU
NVIDIA GeForce RTX 5090
GPU memory
32GB
Host memory
64GB
SSD storage
516GB
NUMA nodes
1
SSD access latency to decoding. By predicting likely future phrase
accesses before the forward pass completes, TF-Engram overlaps
memory transfer with the remaining Transformer computation and
makes large static Engram tables practical for low-latency autore-
gressive inference.
5
EVALUATION
We evaluate TF-Engram from four perspectives. First, we study end-
to-end effectiveness by comparing TF-Engram with the original
Qwen3-0.6B backbone and a parameter-matched LoRA baseline on
downstream benchmarks. This evaluates whether external phrase-
level memory can improve model quality without updating the
backbone parameters. Second, we analyze the overhead introduced
by TF-Engram, including offline construction cost, storage foot-
print, runtime memory usage, throughput, and decoding latency.
Third, we study the sensitivity of TF-Engram to three key parame-
ters: memory injection scale, early-exit layer position, and top-𝐾
prefetch budget. Finally, we conduct ablation studies to isolate the
contribution of train-free semantic memory, SSD-backed hierarchy,
and early-exit guided prefetching.
5.1
Experimental Setup
We evaluate TF-Engram on a single-node workstation. Table 2
summarizes the hardware configuration. The machine contains
a high-end desktop CPU, a single NVIDIA RTX 5090 GPU, host
DRAM, and a local NVMe SSD. This setup allows us to evaluate
whether TF-Engram can support large external phrase memory
under limited accelerator memory, without relying on a multi-GPU
server.
Unless otherwise stated, we use Qwen3-0.6B[31] as the backbone
model. The original Qwen3-0.6B without external memory is used
as the backbone baseline. For end-to-end model-quality comparison,
we additionally include a parameter-matched LoRA baseline. The
LoRA baseline represents a lightweight trainable adaptation method
with a comparable additional trainable-parameter budget, while
TF-Engram keeps the backbone model frozen and introduces an
external phrase-level memory pathway.
TF-Engram constructs phrase memory offline from a mixture of
FineWeb-Edu, Wikipedia, C4 English, arXiv, and PubMed. These
corpora cover general web text, encyclopedic factual knowledge,
commonsense text, scientific terminology, and biomedical terminol-
ogy. The constructed phrase entries are stored in a GPU–DRAM–
SSD hierarchy: GPU memory caches hot entries, host DRAM caches
Table 3: End-to-end performance comparison across down-
stream benchmarks.
Benchmark
Qwen3-0.6B
Parameter-Matched LoRA
TF-Engram
MMLU
53.7
55.3
56.2
ARC-Challenge
42.0
44.1
45.3
HellaSwag
51.0
52.9
50.6
PIQA
70.0
69.5
72.1
WinoGrande
63.0
64.3
62.8
BoolQ
76.0
75.6
78.4
OpenBookQA
32.0
34.3
35.1
SciQ
95.0
94.8
95.4
TruthfulQA-MC2
45.0
46.5
48.2
LAMBADA
48.0
49.6
50.3
Average
57.6
58.7
59.4
warm entries, and NVMe SSD stores the full phrase table. The mem-
ory size, prefetch budget, and early-exit layer position are specified
in the corresponding experiments.
For end-to-end effectiveness, we evaluate downstream accuracy
on MMLU, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ,
OpenBookQA, SciQ, TruthfulQA-MC2, and LAMBADA. For over-
head analysis, we report offline construction time, storage foot-
print, GPU memory usage, throughput, and per-token latency. For
sensitivity analysis, we vary memory injection scale, early-exit
layer position, and top-𝐾prefetch budget. For ablation studies, we
compare TF-Engram variants that remove or replace individual
components, including SSD-backed memory without prefetching,
final-head-guided prefetching, and the full early-exit-guided design.
5.2
End-to-End Effectiveness
We first evaluate whether TF-Engram improves end-to-end model
quality on representative knowledge, reasoning, and language un-
derstanding benchmarks. We compare three settings: the original
Qwen3-0.6B backbone, a parameter-matched LoRA baseline, and TF-
Engram. The LoRA baseline uses a comparable trainable-parameter
budget and represents a lightweight trainable adaptation method.
In contrast, TF-Engram keeps the backbone model frozen and in-
troduces external phrase-level memory. Table 3 reports the detailed
results.
Overall, TF-Engram achieves the best average performance among
the three methods. The average score improves from 57.6 for the
original Qwen3-0.6B backbone to 58.7 for the parameter-matched
LoRA baseline and further to 59.4 for TF-Engram. This corresponds
to an absolute improvement of 1.8 points over the backbone and
0.7 points over LoRA.
TF-Engram improves over the original backbone on 8 out of 10
benchmarks. The largest gains are observed on ARC-Challenge,
TruthfulQA-MC2, OpenBookQA, MMLU, BoolQ, and LAMBADA.
These results suggest that phrase-level external memory is espe-
cially useful for knowledge-intensive and factual evaluation tasks.
TF-Engram also improves PIQA and SciQ, indicating that the re-
trieved memory signal can benefit tasks beyond factual recall alone.
Compared with the parameter-matched LoRA baseline, TF-Engram
achieves better results on most benchmarks, including MMLU, ARC-
Challenge, PIQA, BoolQ, OpenBookQA, SciQ, TruthfulQA-MC2,


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
Figure 3: Offline construction cost of TF-Engram.
and LAMBADA. This result indicates that external phrase-level
memory can provide a complementary and effective alternative to
lightweight trainable adaptation. In particular, the gains on MMLU,
ARC-Challenge, and TruthfulQA-MC2 suggest that TF-Engram
is better aligned with knowledge-intensive tasks where reusable
phrase-level memory can provide useful semantic signals.
TF-Engram underperforms LoRA on HellaSwag and WinoGrande,
and slightly underperforms the original backbone on these two
benchmarks. This suggests that static phrase memory does not
benefit all task types equally. Tasks that rely more heavily on con-
textual commonsense completion or sentence-level disambiguation
may be less aligned with phrase-level memory retrieval. Neverthe-
less, these drops are limited and do not offset the overall average
improvement.
These results show that TF-Engram improves end-to-end perfor-
mance while keeping the backbone model frozen. Compared with
parameter-matched LoRA, TF-Engram achieves higher average ac-
curacy by using external phrase-level memory instead of directly
updating model weights.
5.3
Overhead Analysis
We further evaluate the cost of TF-Engram from three perspectives:
offline construction cost, runtime resource overhead, and decode
latency overhead.
Offline construction cost. Figure 3 shows the offline construc-
tion cost when building TF-Engram tables with 1M to 100M entries.
The construction process is dominated by tokenizer-level n-gram
extraction, while embedding generation and index construction
contribute only a small fraction of the total time. For a 100M-entry
table, the total construction time is approximately 2.0 hours, among
which n-gram extraction accounts for 1.612 hours. Embedding gen-
eration and index construction take only 0.231 hours and 0.157
hours, respectively. This indicates that TF-Engram can be con-
structed with moderate offline cost, and that the main bottleneck is
corpus-level n-gram processing rather than vector generation or in-
dex building. The storage cost scales nearly linearly with table size,
reaching 215.86 GB for a 100M-entry table with 1024-dimensional
FP16 vectors.
Runtime resource overhead. Figure 4 reports the runtime
memory usage and throughput of different TF-Engram configura-
tions. A GPU-resident 10M-entry table incurs a large GPU memory
Figure 4: Runtime resource overhead of TF-Engram.
Figure 5: Decode latency overhead of TF-Engram.
footprint, increasing GPU memory usage to 24.03 GB. In contrast,
the SSD-backed 100M-entry table reduces GPU memory usage to
3.06 GB, while shifting the storage burden to CPU memory and
SSD-backed access. This demonstrates the main benefit of the SSD-
backed design: TF-Engram can support a much larger memory table
without keeping the full table in GPU memory.
The SSD-backed configuration without prefetching reduces through-
put from 481.2 tokens/s to 439.2 tokens/s. With prediction-guided
prefetching, throughput is gradually recovered. Prefetching with
Top-64 improves throughput to 460.3 tokens/s, and Top-128 fur-
ther improves it to 462.4 tokens/s. This shows that prefetching
effectively hides part of the SSD access overhead, while the remain-
ing throughput gap mainly comes from memory lookup and data
movement during decoding.
Decode latency overhead. Figure 5 further breaks down the
decode latency overhead. The baseline Qwen3-0.6B model achieves
2.08 ms/token. A GPU-resident 10M-entry TF-Engram table in-
creases the latency to 2.16 ms/token, corresponding to a 3.83
Prediction-guided prefetching substantially reduces this over-
head. With Top-16 prefetching, the latency decreases to 2.24 ms/token.
Increasing the prefetch budget to Top-64 further reduces the latency
to 2.17 ms/token, and Top-128 achieves 2.16 ms/token. The latency
overhead is therefore reduced from 9.57


---

Yutang MA, Kecheng HUANG, Xikun JIANG, and Zili SHAO
Figure 6: Sensitivity analysis of TF-Engram.
5.4
Sensitivity Analysis
We further conduct a sensitivity analysis to understand how TF-
Engram behaves under different memory scales and early-exit layer
positions. The results are shown in Figure 6.
Impact of memory scale. We first vary the memory table
size from 1M to 100M entries. As the memory scale increases, the
average score improves consistently from 58.2145 to 59.4092, corre-
sponding to an improvement from 1.12
The benefit, however, comes with moderate runtime overhead.
Decode latency increases from 2.108 ms/token at 1M entries to 2.175
ms/token at 100M entries, while throughput decreases from 474.6
tokens/s to 459.8 tokens/s. This trend reflects the expected trade-off
between memory coverage and runtime cost. Even at 100M entries,
the throughput degradation remains moderate, suggesting that
TF-Engram can scale to large memory tables while maintaining
practical decoding efficiency.
Impact of early-exit layer position. We then evaluate different
early-exit layer positions while fixing the memory scale to 100M
entries and the prefetch budget to Top-64. The best result is obtained
at Layer 8, achieving an average score of 59.4092 and a 3.20
This shows that the early-exit layer position affects both predic-
tion quality and runtime efficiency. A shallow exit layer has lower
latency but provides weaker predictive signals for prefetching. A
deeper exit layer may produce more accurate predictions, but it
introduces additional computation and reduces throughput. Layer 8
provides the best balance between prediction quality and overhead,
achieving the highest average score while keeping decode latency
at 2.177 ms/token.
Overall, the sensitivity analysis shows that TF-Engram benefits
from larger memory scales and requires a moderate early-exit layer
for effective prefetching. The 100M-entry table with Layer 8 early
exit provides the best accuracy-efficiency trade-off in our setting.
6
RELATED WORK
Retrieval- and memory-augmented language models. Prior
work augments language models with external knowledge through
retrieval or non-parametric memory, such as RAG, REALM, RETRO,
Atlas, and kNN-LM [2, 11, 15, 19, 23]. These methods show that
external corpora or datastores can improve language modeling and
knowledge-intensive tasks. TF-Engram differs from them by con-
structing a static phrase-level memory table and injecting retrieved
entries into hidden states, rather than appending retrieved text to
the prompt or interpolating output probabilities.
Parameter-efficient and train-free adaptation. Parameter-
efficient methods such as adapters, prefix tuning, prompt tuning,
and LoRA adapt pretrained models with a small number of trainable
parameters [13, 14, 21, 24]. In contrast, TF-Engram targets a train-
free setting where the backbone model remains frozen and the
memory table is built offline from external corpora. This shifts
adaptation from gradient-based parameter updates to inference-
time access to static external memory.
LLM serving, offloading, and prefetching. Recent LLM serv-
ing systems improve inference efficiency through batching, sched-
uling, KV-cache management, and memory offloading [20, 27, 32].
TF-Engram addresses a complementary problem: serving a large
static semantic memory table during autoregressive decoding. Since
SSD or host-memory access can stall token generation, TF-Engram
introduces an SSD–DRAM–GPU memory hierarchy and uses early-
exit guided predictive prefetching to overlap memory transfer with
the remaining Transformer computation.
Vector indexing and memory tables. Large embedding stores
are commonly supported by vector indexing and approximate
nearest-neighbor search techniques, such as product quantization
and graph-based indexes [16, 17, 25]. TF-Engram is related to these
systems, but its workload is different: memory lookup is tightly
coupled with token-by-token LLM inference, and retrieved entries
are used for hidden-state injection rather than document retrieval.
Therefore, TF-Engram jointly considers memory construction, stor-
age hierarchy, runtime lookup, and prefetching for low-latency
LLM inference.
7
CONCLUSION
This paper presents TF-Engram, a train-free static memory sys-
tem for large language models. TF-Engram constructs phrase-level
memory from external corpora and injects retrieved memory en-
tries into the hidden states of a frozen backbone model, avoiding
model-parameter updates or expensive fine-tuning. To make this
memory scalable and efficient, TF-Engram combines an SSD-backed
GPU–DRAM–SSD memory hierarchy with early-exit guided predic-
tive prefetching, allowing large phrase tables to be stored beyond
GPU memory while hiding much of the external-memory access
latency.
Our evaluation shows that TF-Engram improves downstream
performance while introducing moderate system overhead. A 100M-
entry table can be constructed in approximately two hours, with
tokenizer-level n-gram extraction dominating the offline cost. At
runtime, SSD-backed TF-Engram substantially reduces GPU mem-
ory usage compared with GPU-resident memory, and predictive
prefetching recovers much of the throughput loss while keeping
decode latency overhead modest. Sensitivity analysis further shows
that larger memory scales improve average task performance, and
that a moderate early-exit layer provides the best trade-off between
prediction quality and latency hiding.
Overall, TF-Engram demonstrates that static external memory
can be integrated into LLM inference as a scalable, train-free, and
low-overhead system component. By decoupling knowledge stor-
age from dense model parameters, TF-Engram provides a practical
path toward scalable memory augmentation for large language
models.


---

TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models
REFERENCES
[1] Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad
Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase,
and Yuxiong He. 2022. DeepSpeed Inference: Enabling Efficient Inference of
Transformer Models at Unprecedented Scale. In Proceedings of the International
Conference for High Performance Computing, Networking, Storage and Analysis.
[2] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George B. M. van den Driessche, Jean-Baptiste Lespiau,
Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick,
Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon
Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Im-
proving Language Models by Retrieving from Trillions of Tokens. In Proceedings
of the International Conference on Machine Learning.
[3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In
Advances in Neural Information Processing Systems.
[4] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Lau-
rent Sifre, and John Jumper. 2023. Accelerating Large Language Model Decoding
with Speculative Sampling. arXiv preprint arXiv:2302.01318 (2023).
[5] Runxi Cheng, Yuchen Guan, Yongxian Wei, Qianpu Sun, Qixiu Li, Sinan Du, Feng
Xiong, Chun Yuan, Yan Lu, and Yeyun Gong. 2026. Memory Grafting: Scaling
Language Model Pre-training via Offline Conditional Memory. arXiv preprint
arXiv:2605.20948 (2026).
[6] Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda
Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai
Zhang, Dongyan Zhao, and Wenfeng Liang. 2026. Conditional Memory via
Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv
preprint arXiv:2601.07372 (2026).
[7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebas-
tian Gehrmann, et al. 2022. PaLM: Scaling Language Modeling with Pathways.
arXiv preprint arXiv:2204.02311 (2022).
[8] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and
Work Partitioning. In Proceedings of the International Conference on Learning
Representations.
[9] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R’e. 2022.
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
In Advances in Neural Information Processing Systems.
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
Proceedings of the North American Chapter of the Association for Computational
Linguistics.
[11] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval-Augmented Language Model Pre-Training. In Proceed-
ings of the International Conference on Machine Learning.
[12] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya,
Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Jo-
hannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George
B. M. van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen
Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. 2022. Train-
ing Compute-Optimal Large Language Models. arXiv preprint arXiv:2203.15556
(2022).
[13] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.
Parameter-Efficient Transfer Learning for NLP. In Proceedings of the International
Conference on Machine Learning.
[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large
Language Models. In Proceedings of the International Conference on Learning
Representations.
[15] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot Learning with Retrieval Augmented Language
Models. Journal of Machine Learning Research 24, 251 (2023), 1–43.
[16] Herv’e J’egou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine
Intelligence 33, 1 (2011), 117–128.
[17] Jeff Johnson, Matthijs Douze, and Herv’e J’egou. 2019. Billion-Scale Similarity
Search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[18] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess,
Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361 (2020).
[19] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike
Lewis. 2020. Generalization through Memorization: Nearest Neighbor Language
Models. In Proceedings of the International Conference on Learning Representations.
[20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
In Proceedings of the ACM Symposium on Operating Systems Principles.
[21] Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The Power of Scale for
Parameter-Efficient Prompt Tuning. In Proceedings of the Conference on Empirical
Methods in Natural Language Processing.
[22] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from
Transformers via Speculative Decoding. In Proceedings of the International Con-
ference on Machine Learning.
[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich K"uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt"aschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Informa-
tion Processing Systems.
[24] Xiang Lisa Li and Percy Liang. 2021. Prefix-Tuning: Optimizing Continuous
Prompts for Generation. In Proceedings of the Annual Meeting of the Association
for Computational Linguistics.
[25] Yu. A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 4 (2020), 824–
836.
[26] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong
He. 2021. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep
Learning. In Proceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis.
[27] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y.
Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang,
Christopher R’e, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput
Generative Inference of Large Language Models with a Single GPU. In Proceedings
of the International Conference on Machine Learning.
[28] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timoth’ee Lacroix, Baptiste Rozi‘ere, Naman Goyal, Eric Hambro, Faisal
Azhar, et al. 2023. LLaMA: Open and Efficient Foundation Language Models.
arXiv preprint arXiv:2302.13971 (2023).
[29] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv
preprint arXiv:2307.09288 (2023).
[30] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All
You Need. In Advances in Neural Information Processing Systems.
[31] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou,
Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao
Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su,
Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui,
Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.
arXiv preprint arXiv:2505.09388 (2025).
[32] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-
Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based
Generative Models. In Proceedings of the USENIX Symposium on Operating Systems
Design and Implementation.
[33] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models. arXiv preprint arXiv:2506.05176 (2025).
