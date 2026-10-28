---
title: 'GeoFP8: Geometry-Aware FP8 Gradient'
id: geofp8-geometry-aware-fp8-gradient
tags:
- cmp170hx-training-feasibility-a0d4fc
- sub-q4
- gradient-compression
- fp8
- quantization
- communication-volume
- ddp
- k-fac
- low-precision
created: '2026-10-27T10:47:09.730087Z'
updated: '2026-10-27T19:00:31.297138Z'
source: https://arxiv.org/abs/2607.07494
source_domain: arxiv.org
fetched_at: '2026-10-27T10:47:09.729566Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'GeoFP8 (Wang et al., Rutgers/NCSU; arXiv 2607.07494) attacks low-precision
  gradient communication from the geometry side rather than the format side. Diagnosis:
  gradients are highly anisotropic, so axis-aligned (Euclidean) FP8 scaling incurs
  direction-dependent distortion; transforming gradients into a near-isotropic space
  before quantisation - using a block-diagonal K-FAC local metric, applied selectively
  and approximated with a rank-32 input-side low-rank factor - makes the FP8 representation
  substantially more faithful. GeoFP8 deliberately does not change the optimizer,
  training recipe, collective, or low-precision format; only the coordinate system
  used for the low-precision communication.


  Numbers: GeoFP8 reduces the error between FP8-averaged and FP32-averaged gradients
  by up to 67.4% over the Euclidean baseline; it cuts gradient communication volume
  by 75.0% relative to FP32, yielding a 7.6% end-to-end pretraining-time reduction
  for Llama-600M on 64 NVIDIA GH200 Superchips. Direct Euclidean FP8 at the same scale
  gives a 10.79% end-to-end reduction, i.e. GeoFP8 deliberately trades ~3 points of
  speed for downstream-task preservation. Experiments use 32 GPUs, global batch 512,
  micro-batch 4, Muon optimizer, lr 5e-4, on TACC Vista GH200 nodes (96 GB HBM3 per
  GPU, NVLink-coupled Grace-Hopper).


  Two caveats matter for the CMP 170HX box. (1) The 7.6% end-to-end figure is measured
  on a fast NVLink fabric where gradient communication is a minority of step time;
  on a ~1 GB/s PCIe fabric the same 75% payload cut would translate into a much larger
  fraction of step time, so the paper''s end-to-end number understates the potential
  gain here. (2) FP8 is not natively supported on GA100/sm_80 - the CMP 170HX is Ampere-generation
  silicon without FP8 tensor-core paths, so an FP8 gradient path would need software
  emulation whose cost is not accounted for in the GH200 measurements. Their BF16-vs-FP32
  comparison is also relevant to the repo''s bf16 gradient plan: BF16 gradient communication
  "matches FP32 on 7 out of 14 downstream tasks" but with slightly lower absolute
  values, which is why they kept FP32 as the reference.'
raw_file: raw/geofp8-geometry-aware-fp8-gradient.pdf
doi: arXiv:2607.07494
utility_score: 12.0
citation_count: 0
---

GeoFP8: Geometry-Aware FP8 Gradient
Compression for Distributed LLM Training
Jieying Wang1, Zizhong Wang3, Fangru Linghu2, Shuyuan Fan2, Jiajia Li3, and Zhao Zhang2
1Department of Computer Science, Rutgers University, Piscataway, NJ, USA
2Department of Electrical and Computer Engineering, Rutgers University, Piscataway, NJ, USA
3Department of Computer Science, North Carolina State University, Raleigh, NC, USA
Abstract—Gradient communication is a primary scaling bottle-
neck in large language model (LLM) pretraining. Communicating
gradients in low-precision formats, such as FP8 and NVFP4,
can significantly reduce the communication volume. Existing
methods quantize gradients via linear or nonlinear mappings
in Euclidean space, often degrading model performance because
highly anisotropic gradients incur direction-dependent distor-
tion. We present GeoFP8, a geometry-informed gradient scaling
method that performs low-precision communication in geometry-
aware coordinates. By transforming gradients into a near-
isotropic space before quantization, GeoFP8 makes low-precision
representations substantially more faithful to their high-precision
counterparts. GeoFP8 only changes the coordinate system used
for low-precision gradient communication and does not change
the optimizer, training recipe, communication collective, or low-
precision format. We also develop a simplified geometry-aware
transformation algorithm with low-rank approximation and se-
lective application to balance the computation overhead and
communication reduction. We examine the empirical convergence
of GeoFP8 using Llama-300M and Llama-600M models. Our
results show that GeoFP8 reduces the end-to-end pretraining
time of Llama-600M by 7.6% on 64 NVIDIA GH200 Superchips,
while improving the downstream task preservation profile over
direct Euclidean FP8 communication under the same optimizer
and communication path.
Index Terms—large language models, distributed pretrain-
ing, gradient communication, communication compression, low-
precision communication, geometry-aware communication, K-
FAC
I. INTRODUCTION
Gradient communication is a major scaling bottleneck in
distributed LLM pretraining [1]. In data-parallel, 3D-parallel,
or more complex parallel pretraining strategies, the optimizer
needs to communicate gradients across graphics processing
units (GPUs). Standard communication libraries, such as
NCCL and MPI, implement the ring allreduce algorithm [2],
whose cost depends on the model size and job size (i.e., the
number of GPUs). Previous work reports that communication
accounts for 40% of the overall pretraining time for GPT-8.3B
pretraining across 128 NVIDIA A100 GPUs [3].
Reducing communication volume via low-precision gradi-
ents is an effective way to lower communication overhead, as
the bits saved for each gradient representation directly translate
into proportional reductions in bandwidth consumption. Exist-
ing low-precision training techniques, such as FP8-LM [4] and
COAT [5], mainly apply scaling and quantization in Euclidean
tensor coordinates, which can still lead to degraded model
performance. They are designed specifically for the AdamW
optimizer [6]. FP8-LM uses FP8 format for model weights,
activations, gradients, and first moment, while the second
moment is stored in 16-bit format for division stability. COAT
uses dynamic range strategies (i.e., SX =
amax(XFP32)
∆E4M3
MAX
) and
mixed-granularity activation quantization to quantize second
moment and activation, while the gradients are still in 16-bit
format for training stability. SDP4Bit [7] leverages the sharded
data parallel model placement and implements 4-bit gradient
communication via Fourier transform and hierarchical all-
to-all communication. However, neither FP8-LM nor COAT
achieves comparable model performance to the BF16 baseline.
FP8-LM shows that seven out of ten downstream tasks perform
worse than the baseline. COAT has only one with a higher
score than the baseline among the four downstream tasks.
SDP4Bit only evaluates model performance using validation
loss.
Most existing works propose holistic quantization ap-
proaches that quantize not only gradients but also model
parameters and optimizer states. It is challenging to isolate
the impact of gradients in low-precision formats and to justify
the additional computational overhead and the reduction in
communication. In this work, we focus exclusively on gradient
communication using the FP8 format (native FP8 support
was added in NCCL 2.24 in early 2025). We systematically
analyze the optimization process using gradients in FP8 for
the Muon optimizer [8] in Llama model training. GeoFP8 only
changes the coordinate system used for low-precision gradient
communication; it does not change the optimizer, model,
training recipe, communication collective, or FP8 format. We
observe that gradients are often highly anisotropic, so direct
quantization in Euclidean space can distort some directions
more than others, causing the synchronized update to drift
from its high-precision counterpart. As shown in Figure 1,
applying a uniform scaling factor on both axes in the Euclidean
space causes different error levels. However, transforming the
gradients onto a Riemannian manifold via Fisher whitening
yields a distribution that is close to isotropic.
Inspired by this observation, we propose GeoFP8, a practical
geometry-informed gradient scaling technique that improves
the fidelity–efficiency tradeoff of FP8 gradient communication.
The key idea is simple: quantize gradients in geometry-aware
coordinates rather than raw Euclidean coordinates, so scaling
arXiv:2607.07494v2  [cs.DC]  30 Sep 2026


---

Fig. 1: Two-dimensional projection of real gradient samples
in different coordinate systems. In the original Euclidean
parameter space (left), the gradients are highly elongated and
anisotropic. After a geometry-aware transform (right), the
same gradients become more rounded and less correlated,
making low-precision quantization less sensitive to direction.
to FP8 introduces comparable errors across dimensions. Ge-
oFP8 is enabled by practical local geometry approximations
based on K-FAC [9]–[11] (see §II-B), and we refer to the full
geometry approximation as K-FAC in the following discus-
sion. We use this approximation only to construct temporary
communication coordinates, not to replace the optimizer up-
date with a K-FAC or natural-gradient step. Naively applying
K-FAC to all gradients yields a better model performance than
existing FP8 scaling techniques. However, the computational
complexity of K-FAC makes it less practical, as the additional
computation time outweighs the reduced communication time.
So we select the layers most sensitive to quantization and
apply simplified K-FAC only to these layers, while other layers
use standard Euclidean scaling. This selective strategy reduces
the computational overhead of K-FAC while preserving the
resulting model performance.
We implement GeoFP8 in the Megatron-LM [12] frame-
work and pretrain Llama-300M and Llama-600M with the
OpenWebText [13] dataset. We compare the pretrained model
performance using GeoFP8, FP32, and a Euclidean baseline
(i.e., a layer-wise scaling). Our error analysis shows that
GeoFP8 reduces the error, i.e., the difference between averaged
gradients in FP8 and FP32, by up to 67.4% over the Euclidean
baseline. The pretraining studies show that Llama models
pretrained with GeoFP8 achieve a better downstream task
preservation profile than the Euclidean scaling approach under
the same optimizer and communication path. Compared to
gradients in FP32, GeoFP8 reduces the gradient communi-
cation volume by 75.0%, resulting in a 7.6% reduction in
overall pretraining time across 64 GH200 superchips. The
75.0% reduction refers to gradient payload size, while the end-
to-end speedup is smaller because total pretraining time also
includes forward and backward computation, optimizer up-
dates, quantization and dequantization, geometry transforms,
synchronization overheads, and non-gradient communication.
At the same scale, direct Euclidean FP8 achieves a 10.79%
reduction, showing that GeoFP8 trades part of the raw FP8
speedup for improved downstream preservation.
This paper makes three main contributions:
• A new diagnosis for low-precision scaling error. We
show that the error of Euclidean gradient scaling is a
mismatch between anisotropic local gradient geometry
and axis-aligned quantization.
• A selective geometry-aware scaling design. We develop
GeoFP8, which keeps the standard Euclidean scaling
for most layers and applies geometry-aware scaling only
where the model performance justifies the cost.
• A substantially better fidelity–efficiency tradeoff. We
show that GeoFP8 improves the downstream preser-
vation profile over direct Euclidean FP8 while retain-
ing a fraction of the FP8 speedup over FP32, making
geometry-aware low-precision communication practical
in our Megatron-LM pretraining experiments.
The 7.6% overall time reduction may be considered
marginal. However, modern LLMs are pretrained with tens
of thousands of GPUs over several months. Even a 7.6%
reduction in time can translate to millions of dollars. On the
other hand, the interconnect in the GPU cluster is a shared
resource. A reduced bandwidth consumption will enable more
efficient network sharing. The communication bottleneck is
becoming increasingly prominent as interconnect bandwidth
increases more slowly than the compute elements, i.e., the
GPUs. Although our experiments are on FP8, the geometry-
transform overhead is mostly independent of the low-precision
communication format, so the same principle can be applied to
future FP4 communication when software and hardware sup-
port mature. This trend makes geometry-aware communication
a promising direction for future lower-bit LLM pretraining
systems.
II. BACKGROUND
This section presents the communication setting considered
in this paper and the K-FAC-based local metric that later
defines the communication coordinates of GeoFP8. We first
summarize how communication arises in data-parallel and 3D-
parallel LLM pretraining. We then introduce the layerwise
block-diagonal K-FAC approximation used to obtain tractable
local geometry [9]–[12], [14]–[19].
A. Communication in Data-Parallel and 3D-Parallel Pretrain-
ing
Distributed LLM pretraining requires communication when-
ever model states, activations, or gradients are partitioned
across devices. The exact communication pattern depends on
the parallelization strategy, but in all cases communication
is on the critical path of each pretraining iteration and can
become a major scaling bottleneck as model size and system
scale grow [1], [3], [12], [14].
a) Data parallelism: In data-parallel (DP) pretraining,
each worker stores a full replica of the model and processes
a different shard of the mini-batch. After the backward pass,
the local gradients computed on different workers must be
aggregated before the optimizer update so that all replicas


---

remain synchronized [12], [14], [17]. If worker r produces
local gradient g(r), the synchronized gradient is
¯g = 1
R
R
X
r=1
g(r),
(1)
where R is the number of DP workers. In practice, this
aggregation is typically implemented by all-reduce, or equiv-
alently by reduce-scatter followed by all-gather in sharded
settings such as ZeRO-style pretraining [2], [17], [18]. Because
the full gradient set is communicated at every iteration, the
communication cost grows with both model size and number
of workers, making DP gradient communication a primary
target for compression [18], [20]–[23].
b) 3D parallelism: Modern large-scale LLM pretraining
often combines data parallelism (DP), tensor parallelism (TP),
and pipeline parallelism (PP), which is commonly referred to
as 3D parallelism [3], [12], [14]. In this setting, communica-
tion arises along three distinct paths.
First, in tensor parallelism, individual layers are partitioned
across devices, so partial tensor results must be exchanged
inside a layer during the forward and backward passes. De-
pending on the operator and partition rule, this communication
is implemented through collectives such as all-reduce, all-
gather, or reduce-scatter to assemble outputs or accumulate
partial gradients across TP ranks [12], [14]. Second, in pipeline
parallelism, the model is divided into stage-wise partitions,
so adjacent stages communicate boundary activations in the
forward pass and boundary gradients in the backward pass
through point-to-point sends and receives [12], [15], [16].
Third, across DP groups, replicated parameters or optimizer
shards still require communication so that the distributed
optimizer applies a consistent update [12], [17], [18].
Therefore, 3D-parallel pretraining contains multiple com-
munication paths with different semantics: TP communicates
intra-layer partial tensor results, PP communicates inter-stage
activations and gradients, and DP communicates gradient or
optimizer-update information across replicas [3], [12], [15],
[16]. In this paper, we focus on the DP gradient communica-
tion path. This choice isolates the communication bottleneck
that is most directly compatible with low-precision compres-
sion and allows us to study the effect of gradient-coordinate
choice while keeping the optimizer semantics unchanged out-
side the communication step [7], [20], [22], [24].
B. K-FAC Block-Diagonal Approximation
The Fisher information matrix provides a local metric for
measuring parameter perturbations in neural-network opti-
mization [9], [10]. Let θ denote the model parameters and let
F(θ) denote the Fisher matrix. A perturbation ∆θ has local
squared length
∥∆θ∥2
F = ∆θ⊤F(θ) ∆θ.
(2)
However, directly forming or inverting the full Fisher matrix is
infeasible for modern LLMs because it couples all parameters
across all layers.
K-FAC makes this local geometry tractable through a struc-
tured approximation to the Fisher. In the standard K-FAC
formulation, the Fisher is treated through layerwise blocks,
and each such block is approximated by a Kronecker product
of two smaller factors [11], [19]. For a model with weight
matrices {Wl}L
l=1, this layerwise approximation can be written
as
F(θ) ≈blockdiag (FW1, FW2, . . . , FWL) ,
(3)
where each diagonal block describes the local geometry of one
layer. This block-diagonal approximation removes cross-layer
curvature terms and makes the metric estimable and applicable
on a per-layer basis.
For a linear layer with weight matrix W ∈Rdout×din,
let a denote the layer input activation and let δ denote the
gradient with respect to the layer pre-activation output. Using
the standard column-major vectorization convention, K-FAC
approximates the layerwise Fisher block as
FW ≈A ⊗G,
A = E[aa⊤],
G = E[δδ⊤],
(4)
where A ∈Rdin×din captures input-side second-order statistics
and G ∈Rdout×dout captures output-gradient-side second-
order statistics [11], [19]. Thus, instead of maintaining a full
Fisher matrix over all model parameters, K-FAC maintains
small structured factors for each layer.
This paper does not use K-FAC as an optimizer and does
not apply the natural-gradient update. We use the K-FAC block
only as a tractable local metric for defining communication
coordinates. The optimizer, model architecture, and distributed
training structure remain unchanged; only the coordinate sys-
tem used during low-precision gradient communication is
modified.
III. RELATED WORK
This section summarizes related works in three categories:
communication compression, low-precision LLM pretraining,
and geometry-aware optimization.
A. Gradient Communication Compression
A long line of work reduces distributed-training overhead
by reducing the communication volume. Example methods
include quantization, sparsification, low-rank approximation,
and optimizer-specific communication schemes [3], [7], [18],
[20]–[23], [25]. These methods show that gradient communi-
cation can often be reduced substantially without fully sacrific-
ing optimization quality, making communication compression
a practical systems direction for large-scale pretraining. Recent
work has further pushed this trend toward lower-bit com-
munication in modern LLM settings, including sharded and
hierarchical communication designs [7], [18], [25]. Our work
aligns with existing approaches but differs in its mechanism:
rather than modifying the communication pattern, reducing
rank, or introducing sparsity, we study how the coordinate
system used for gradient communication affects the fidelity of
low-precision communication. These methods are important
related systems, but they are not the direct controlled baseline


---

for our study. QSGD changes the quantization rule, DGC intro-
duces sparsity, PowerSGD communicates a low-rank gradient
representation, and 1-bit Adam is coupled with Adam-style
optimizer dynamics. LoCo is closely related because it targets
low-bit gradient communication, but its main mechanism is
error compensation in the original gradient coordinates. In con-
trast, GeoFP8 keeps the optimizer, communication collective,
FP8 format, and training recipe fixed, and changes only the
communication coordinate system.
B. Low-precision LLM Pretraining
Another closely related line of work studies FP8 and other
low-precision formats for efficient pretraining [4], [5], [24],
[26], [27]. These methods typically rely on scaling a ten-
sor before quantization so that its values better match the
representable range of the target format [4], [24]. In this
setting, the main design questions concern how to choose
the scaling statistics and how to maintain stable pretraining
under reduced precision [4], [5], [24]. Representative systems
such as FP8-LM and COAT demonstrate that low-precision
pretraining can be extended to broader parts of the pretraining
stack, including gradients, activations, and optimizer states,
under carefully designed scaling rules [4], [5]. SDP4Bit further
studies low-bit communication in sharded data-parallel LLM
pretraining and shows that communication-aware smoothing
and hierarchical quantization can substantially narrow the
quality gap relative to higher-precision baselines [7]. Its setting
changes the sharded communication path and jointly designs
the communication algorithm, while our experiments isolate
the coordinate choice under the same allreduce path. Unlike
these holistic systems, GeoFP8 focuses only on gradient
communication and does not quantize model parameters,
activations, or optimizer states. These works establish the
practical importance of scaling for low-precision pretraining
and communication, but they still operate on tensors in their
original Euclidean coordinates.
C. Geometry-aware Optimization and GeoFP8 Position
Our work is also related to geometry-aware optimiza-
tion, especially natural gradient and K-FAC-style approxi-
mations [10], [11], [19]. These methods use local second-
order structure to precondition gradients, primarily to im-
prove optimization efficiency or curvature adaptation. This
is different from using the same structure as a temporary
representation for communication. In contrast, we use the
same geometric intuition for a different purpose: improving
the fidelity of low-precision communication. GeoFP8 does not
change the optimizer into a second-order method; instead, it
uses local geometry to transform gradients into communication
coordinates that are more amenable to FP8 quantization. The
transformed gradient is mapped back before the optimizer
update, so the optimizer receives a gradient in the original
parameter coordinates. In this sense, GeoFP8 is comple-
mentary to both prior communication-compression methods
and geometry-aware optimization methods. GeoFP8 differs
from existing low-precision techniques in that it explores an
alternative coordinate system to apply quantization instead of
choosing a better scale in Euclidean space.
IV. DESIGN
This section presents the design of geometry-informed low-
precision gradient communication. The central idea is to keep
the standard distributed pretraining and optimizer pipeline
unchanged, but to change the coordinate system in which gra-
dient quantization and communication are performed. Instead
of communicating weight gradients directly in Euclidean co-
ordinates, GeoFP8 maps them into coordinates induced by the
K-FAC local metric, performs low-precision communication in
that space, and then maps the synchronized result back before
the optimizer update.
A. Geometry-Aware Communication Coordinates
We focus on the weight gradient
Wg ∈Rdout×din
(5)
of a linear layer. In conventional low-precision communica-
tion, Wg is scaled, quantized, communicated, dequantized, and
averaged directly in Euclidean parameter coordinates. This
is efficient, but it treats all directions uniformly. When the
local gradient geometry is anisotropic, as shown in Figure 1,
the same low-precision scale must cover directions with very
different magnitudes. As a result, quantization error can be-
come strongly direction-dependent: large-variance directions
dominate the scale, while small-variance directions suffer
relatively larger distortion.
a) Local metric from K-FAC: Following the K-FAC
block introduced in Section II-B, the local metric of a linear
layer is approximated by
FW ≈A ⊗G,
A = E[aa⊤],
G = E[δδ⊤],
(6)
where A captures input-side second-order statistics and G
captures output-gradient-side second-order statistics. For a
small perturbation ∆W of the layer weight, the corresponding
K-FAC local squared length is
∥∆W∥2
FW = vec(∆W)⊤(A ⊗G) vec(∆W).
(7)
Using standard Kronecker and trace identities, this can be
equivalently written as
∥∆W∥2
FW = tr
 A∆W ⊤G∆W

.
(8)
Let
A = LAL⊤
A,
G = LGL⊤
G
(9)
be Cholesky factorizations of the regularized K-FAC factors.
Substituting these factorizations into Eq. (8) gives
∥∆W∥2
FW = tr
 LAL⊤
A∆W ⊤LGL⊤
G∆W

= tr
 L⊤
A∆W ⊤LGL⊤
G∆WLA

=


L⊤
G∆WLA


2
F .
(10)
Therefore, the K-FAC ellipsoid

∆W : ∥∆W∥2
FW ≤ϵ
	
(11)


---

is mapped by the factor transform
U = L⊤
G∆WLA
(12)
to the Euclidean Frobenius ball

U : ∥U∥2
F ≤ϵ
	
.
(13)
This shows that the K-FAC factors turn the local anisotropic
metric into a coordinate system whose metric ball is spherical.
Equivalently, directions that are stretched or compressed by the
local Fisher geometry become more uniformly scaled after the
coordinate transformation.
b) Whitening the communicated gradient: The metric-
ball argument above explains the geometric role of the K-
FAC factors. For communication, we apply the corresponding
inverse factors to the gradient before quantization. Specifically,
the full geometry-informed communication coordinate is de-
fined as
f
Wg = L−1
G WgL−⊤
A .
(14)
This transform can also be viewed as a whitening operation
on the layerwise gradient statistics. Under the Fisher/K-FAC
approximation, the dominant second-order structure associated
with vec(Wg) is represented by the same Kronecker-structured
factors. Therefore,
vec(f
Wg) =
 L−1
A ⊗L−1
G

vec(Wg)
(15)
removes the dominant input-side and output-side scaling cap-
tured by A and G. In the transformed coordinates, the gradient
distribution is therefore closer to isotropic, so an axis-aligned
low-precision quantization lattice interacts with the gradient
more uniformly than it does in the original Euclidean coordi-
nates.
After transformation, GeoFP8 applies the same low-
precision communication procedure used by the Euclidean
baseline:
c
f
W g = CommFP8

f
Wg

,
(16)
where CommFP8(·) denotes scaling, FP8 quantization, col-
lective communication, dequantization, and averaging. The
synchronized transformed gradient is then mapped back to
Euclidean coordinates:
c
Wg = LGc
f
W gL⊤
A.
(17)
The mapped-back gradient c
Wg is passed to the same opti-
mizer used by the high-precision and Euclidean low-precision
baselines.
Importantly, this procedure changes only the communication
coordinates. It does not change the model architecture, the
distributed parallelization strategy, or the optimizer update
rule. Thus, GeoFP8 is not a K-FAC optimizer or a natural-
gradient method; it uses K-FAC-style local geometry only to
improve the fidelity of low-precision gradient communication.
c) Input-side practical form: The full two-sided trans-
form in Eq. (14) makes the geometric principle explicit, but
applying it to every layer can introduce substantial matrix-
computation and memory overhead in large-scale LLM pre-
training. In practice, GeoFP8 uses an input-side version of the
same idea:
f
Wg = WgL−⊤
A ,
(18)
followed by FP8 communication in the transformed coordi-
nates and inverse mapping
c
Wg = c
f
W gL⊤
A.
(19)
This input-side form preserves the most useful part of the
local geometry while avoiding the cost of maintaining and
applying both K-FAC factors. It can also be interpreted as
turning the input-side component of the K-FAC metric into
a more isotropic coordinate system. Specifically, if only the
input-side metric is used, then
tr
 A∆W ⊤∆W

= ∥∆WLA∥2
F ,
(20)
so the input-side ellipsoidal geometry is again mapped to a
Euclidean Frobenius ball.
The full two-sided formulation in Eq. (14) exposes the
geometric principle, while the input-side form in Eq. (18) gives
the practical communication coordinate used by GeoFP8. The
remaining system questions are how much of this geometry is
needed in practice, how accurately the input-side factor must
be represented, and where the geometry-aware path should be
applied. Section V answers these questions empirically and
turns the coordinate transform into the final system through
three simplifications: retaining only the input-side geometry,
approximating it with a low-rank factor, and applying it
selectively to the most numerically vulnerable layers.
V. IMPLEMENTATION
This section turns the full geometry-informed communi-
cation formulation of Section IV into a practical system
implementation. Guided by the roadmap in Figure 2, our
implementation answers four empirical questions: Which part
of the full transform matters most? How accurately must the
remaining geometry be represented? Which layers benefit suf-
ficiently to justify the extra cost? And how should the resulting
design be integrated into a standard distributed pretraining
pipeline?
The final system answers these questions through three
design decisions: (i) retain only the input-side geometry, (ii)
approximate it with a rank-32 low-rank representation, and
(iii) deploy it only on the most numerically vulnerable layers.
These decisions are made sequentially: each step removes
a source of overhead from the full geometry-informed for-
mulation only after the reconstruction study shows that the
corresponding fidelity loss is small.


---

Fig. 2: Roadmap of the practical simplifications used to turn
the full geometry-informed communication formulation into
the final GeoFP8 system.
TABLE I: One-step FP8 round-trip reconstruction error for the
full and one-sided transforms, measured in Euclidean gradient
space against the original FP32 gradient. Lower is better for
RelL2, MaxErr, and MSE; higher is better for Cos.
Method
RelL2
Cos
MaxErr
MSE
Euclidean baseline
5.281973e-02
9.986041e-01
1.625657e-04
1.289936e-10
Output-side only
5.282879e-02
9.986036e-01
1.629426e-04
1.290378e-10
Input-side only
1.768108e-02
9.998500e-01
5.332805e-05
1.445418e-11
K-FAC
1.770532e-02
9.998502e-01
5.332805e-05
1.449384e-11
A. Practical Simplification I: Retaining Only the Input-Side
Geometry
The full two-sided formulation in Section IV is a useful
conceptual starting point, but applying both sides of the
transform to every eligible layer is too expensive for large-
scale end-to-end pretraining. We therefore begin by asking
which side of the transform is actually responsible for the
practical fidelity gain.
To answer this question, we perform a controlled one-
step FP8 round-trip analysis in Euclidean gradient space.
For each method, we start from an FP32 gradient, map it
into the corresponding communication coordinates, apply an
FP32→FP8→FP32 round trip there, map the result back to
Euclidean gradient space, and compare it against the original
FP32 gradient.
We report four reconstruction metrics:
RelL2 = ∥ˆg −g∥2
∥g∥2
,
Cos =
⟨ˆg, g⟩
∥ˆg∥2 ∥g∥2
,
MaxErr = ∥ˆg −g∥∞,
MSE = ∥ˆg −g∥2
2
n
,
(21)
where g is the original FP32 gradient, ˆg is the reconstructed
gradient after the FP8 round trip, and n is the number of entries
in the gradient tensor. Cos measures whether the FP8 round
trip preserves the update direction, while RelL2, MaxErr,
and MSE measure the magnitude and worst-case size of the
communication perturbation. These metrics therefore provide
a mechanism-level fidelity test before running full end-to-end
pretraining. Table I reports the results.
The result is clear: the output-side transform provides only
a marginal contribution over the Euclidean baseline, whereas
the input-side transform already matches the full two-sided
transform almost exactly across all metrics. This shows that,
TABLE II: Ablation of input-side approximations, evaluated
in Euclidean gradient space after FP8 round-trip. Low-rank
approximations recover nearly all of the benefit of the full
input-side transform, and rank 32 provides the best practical
cost–quality tradeoff.
Method
RelL2
Cos
MaxErr
MSE
Euclidean baseline
5.281973e-02
9.986041e-01
1.625657e-04
1.289936e-10
Input-side full-A
1.768108e-02
9.998500e-01
5.332805e-05
1.445418e-11
Input-side diag-A
5.262250e-02
9.986144e-01
1.609139e-04
1.280321e-10
Input-side block-A-32
3.168243e-02
9.994993e-01
1.269338e-04
4.641014e-11
Input-side block-A-64
2.754295e-02
9.996220e-01
1.051468e-04
3.507491e-11
Input-side block-A-128
2.431718e-02
9.997058e-01
1.051468e-04
2.734022e-11
Input-side low-rank-A-8
1.958074e-02
9.998106e-01
7.036887e-05
1.772696e-11
Input-side low-rank-A-16
1.731027e-02
9.998523e-01
5.883700e-05
1.385426e-11
Input-side low-rank-A-32
1.720702e-02
9.998551e-01
6.311573e-05
1.368948e-11
in our profiled setting, nearly all of the measured FP8 round-
trip fidelity benefit of full geometry-aware communication is
captured by the input-side geometry alone.
We therefore remove the output-side transform entirely and
retain only the input-side term. The resulting communication
coordinates can be written as
f
Wg = WgTA,
(22)
where TA is an input-side transform derived from the factor
A. FP8 quantization and communication are then performed
on f
Wg, and the synchronized result is mapped back to Eu-
clidean gradient space by the corresponding inverse right-side
transform. This simplification is the first key step that makes
geometry-informed communication practical.
B. Practical Simplification II: Low-Rank Approximation of the
Input-Side Geometry
Once the full transform is reduced to the input side, the next
question is how accurately the remaining geometry must be
represented. Using the full factor A for every selected layer
is still costly, so we seek a lower-cost approximation that
preserves most of its communication benefit.
Let
A ≈E[xx⊤],
A ⪰0,
(23)
and let
A ≈UrΛrU ⊤
r
(24)
be a rank-r approximation with r ≪din. We construct the
input-side communication transform from this reduced repre-
sentation and again evaluate the resulting methods through the
same Euclidean-space FP8 round-trip analysis. Table II reports
the comparison.
Two conclusions follow from Table II. First, a diagonal
approximation barely improves over the Euclidean baseline,
indicating that coarse per-dimension scaling is insufficient to
capture the useful geometry. Second, low-rank approximations
recover nearly all of the benefit of the full input-side transform.
Among them, rank 32 is already nearly identical to the full-
A result while being substantially cheaper to store and apply.
We use rank 32 as the operating point for our implementation
rather than as a model-independent constant. We therefore
adopt input-side only, low-rank-32 as the geometry-informed
transform in the final implementation.


---

Fig. 3: Ranking MLP layers by average numerical vulnerabil-
ity under the Euclidean baseline during the first 100 training
steps. In this profile, the top 13 vulnerable layers are all fc2
layers, and the visible drop after this group motivates selective
deployment of the geometry-informed branch for this model
and recipe.
C. Practical Simplification III: Selective Deployment on Vul-
nerable Layers
Even after the first two simplifications, applying geometry-
informed communication to every layer would still introduce
unnecessary overhead. The next question is, therefore, whether
all layers need the geometry-aware branch uniformly.
To answer this question, we profile the Euclidean baseline
during a 32-GPU pretraining run of a 600M-parameter Llama-
style model and measure the numerical vulnerability of each
MLP layer (both fc1 and fc2) over the first 100 training steps.
For each profiled module, we flatten its direct parameter gradi-
ents, compute a synchronized FP8 scale, encode the flattened
gradient into FP8, and count how often encoded values hit the
upper and lower FP8 boundaries. These boundary-hit ratios
are accumulated separately for fc1 and fc2 layers and then
averaged over the profiling window. We rank layers by the
sum of their average upper- and lower-boundary hit ratios,
which serves as our numerical-vulnerability score.
Figure 3 shows the resulting profile. The vulnerability is
highly concentrated: after the top 13 entries, the score drops
visibly, suggesting that the most problematic FP8 distortion is
confined to a relatively small subset of layers. In this profile,
the top 13 vulnerable layers are all fc2 layers, while the
first fc1 layer appears immediately after this group. Thus, the
number 13 is not intended as a model-independent constant;
it is the operating point selected by the vulnerability-ranking
procedure for this model and recipe. This motivates a simple
selective-deployment policy: we enable the geometry-informed
branch only for these selected vulnerable MLP layers and keep
the Euclidean baseline for all remaining layers. This policy
concentrates additional computation where geometry-aware
communication is most likely to help, while preserving the
fast path for most of the model. For a new model architecture
or training recipe, the same profiling procedure can be rerun
automatically to select the layer set.
Fig. 4: Pretraining-time structure of GeoFP8. Most layers use
the shared communication core directly in Euclidean coor-
dinates. Selected numerically vulnerable layers additionally
apply an input-side low-rank transform before entering the
core and map the synchronized result back afterward.
D. Pretraining-Time Pipeline
Figure 4 summarizes the pretraining-time structure of the
final GeoFP8 system. All layers share the same communication
core consisting of quantization, communication, and dequanti-
zation. Most layers enter this core directly in Euclidean coordi-
nates, while selected numerically vulnerable layers first apply
the input-side low-rank transform and map the synchronized
result back afterward. This organization makes the central
systems point explicit: GeoFP8 does not replace the Euclidean
baseline globally, but augments it only where geometry is
empirically justified.
In this view, the default path for most layers remains identi-
cal to the Euclidean baseline. The geometry-informed branch
differs only by adding a transform before the shared FP8
communication core and a map-back step afterward. Because
this branch is activated only on a small subset of layers, the
overall system remains close to the baseline in communication
structure while correcting the layers most affected by FP8
distortion.
E. Offline Preparation and Pretraining-Time Workflow
The final implementation consists of two stages. First, we
run an offline profiling phase to identify the vulnerable layer
set. Second, during pretraining, we activate the geometry-
informed branch only on the selected layers, while all other
layers continue to follow the Euclidean baseline. The selected
layer set S is fixed before pretraining, but the input-side
geometry factors are refreshed periodically during training.
In our experiments, the factors are updated every 50 training
steps.
In the current implementation, the selected layer set S
is fixed before pretraining based on the profiling procedure
described above. This organization keeps the layer-selection
logic outside the main training loop and preserves a simple
fast path for the majority of layers. Algorithm 1 summarizes
the resulting pretraining-time communication rule. For the
selected GeoFP8 layers, we use local scaling in the trans-
formed coordinates, which gives better empirical fidelity than
synchronizing a global scale in our ablation.
Two implementation details are worth emphasizing. First,
the geometry-aware branch is activated only on a fixed selected
subset of layers, so the majority of the model follows the orig-
inal Euclidean path without additional geometric processing.


---

Algorithm 1 Selective GeoFP8 Communication with Input-
Side Whitening and Error Feedback
Require: Fixed selected layer set S
Require: For each l ∈S, input-side factorization updated
every K steps
A(l) ≈L(l)
A L(l)⊤
A
, with K = 50 in our experiments
Require: Error buffers {R(l)}l∈S initialized to zero
Require: World size N, FP8 quantizer Q(·; s), scaling rule
Scale(·)
1: for each training step do
2:
for each layer l with weight gradient W (l)
g
do
3:
if l /∈S then
4:
s(l) ←Scale(W (l)
g )
5:
s(l) ←ALLREDUCEMAX(s(l))
6:
Q(l) ←Q(W (l)
g ; s(l))
7:
bQ(l) ←ALLREDUCESUM(Q(l))
8:
c
W (l)
g
←Dequantize( bQ(l); s(l))/N
9:
else
10:
U (l) ←W (l)
g
 L(l)
A
−⊤
11:
eU (l) ←U (l) + R(l)
12:
s(l)
loc ←Scale(eU (l))
13:
eQ(l) ←Q(eU (l); s(l)
loc)
14:
eU (l)
deq ←Dequantize( eQ(l); s(l)
loc)
15:
R(l) ←eU (l) −eU (l)
deq
16:
s(l)
comm ←s(l)
loc
17:
if this is a scale-synchronization step then
18:
s(l)
comm ←ALLREDUCEMAX(s(l)
loc)
19:
end if
20:
bQ(l) ←ALLREDUCESUM( eQ(l))
21:
bU (l) ←Dequantize( bQ(l); s(l)
comm)/N
22:
c
W (l)
g
←bU (l)L(l)⊤
A
23:
end if
24:
end for
25:
Optimizer updates parameters using {c
W (l)
g }
26: end for
Second, the geometry-aware branch uses error feedback in the
transformed coordinates, allowing quantization residuals to be
accumulated where the communication actually occurs. This
keeps the implementation aligned with the communication
formulation rather than treating error feedback as a separate
post hoc correction.
F. Implementation Summary
Our final implementation is obtained by progressively sim-
plifying the full geometry-informed formulation of Section IV.
We first remove the output-side transform after observing that
the input-side geometry captures nearly all of the practical
benefit of the full design. We then compress the remaining
input-side geometry with a rank-32 low-rank approximation.
Finally, we deploy the resulting transform only on a small
subset of numerically vulnerable layers.
These simplifications yield a practical hybrid system: Eu-
clidean remains the default communication path, and geometry
is introduced only where it is empirically justified. As a result,
the implementation stays close to the baseline in communica-
tion structure and cost while recovering most of the measured
fidelity benefit of full geometry-aware communication on the
layers that need it most.
VI. EXPERIMENTAL RESULTS
We evaluate GeoFP8 from four complementary perspec-
tives. First, we measure the system benefit of low-precision
communication as the model size and GPU count increase.
Second, we examine whether GeoFP8 preserves pretraining
behavior in terms of validation loss. Third, we compare
downstream task performance to determine whether geometry-
aware communication better preserves useful model quality
than direct Euclidean communication. Fourth, we examine the
memory overhead of GeoFP8 relative to the direct Euclidean
baseline.
Our goal is not to claim that GeoFP8 improves every metric
uniformly. Rather, the goal is to test a more specific hypoth-
esis: under the same low-precision communication format,
changing only the communication coordinates can improve
the usefulness of the communicated update. This distinction
is important in our setting, because optimization loss and
downstream task quality do not always rank the methods in
the same way.
A. Experimental Setup
We study two LLaMA-style pretraining configurations on
OpenWebText. The first is a model of approximately 300M
parameters pretrained with a sequence length of 4096. The sec-
ond is a model of approximately 600M parameters pretrained
with a sequence length of 2048. We use different sequence
lengths to fit in the GPU memory.
Unless otherwise specified, all end-to-end pretraining exper-
iments use 32 GPUs, global batch size 512, micro batch size
4, Muon optimizer, learning rate 5 × 10−4, cosine decay, and
minimum learning rate 5 × 10−5.
All experiments are conducted on the Vista supercomputer
at the Texas Advanced Computing Center (TACC). Vista
includes 600 Grace Hopper compute nodes based on the
NVIDIA GH200 Grace Hopper Superchip architecture, and
each Grace Hopper node provides one NVIDIA GPU with
96 GB of HBM3 memory together with an NVIDIA Grace
CPU interconnected through a tightly integrated CPU–GPU
design.
For end-to-end pretraining, we compare three practical
methods:
• FP32: full-precision gradient communication.
• Euclidean baseline: direct layer-wise FP8 communica-
tion in Euclidean parameter coordinates.
• GeoFP8: the final practical geometry-aware design,
which applies the geometry-aware branch only where its
benefit justifies the cost, while keeping the Euclidean
baseline path elsewhere.
We choose the stronger baseline in each category based on
additional 600M experiments. BF16 gradient communication


---

matches FP32 on 7 out of 14 downstream tasks, but its absolute
task values are slightly lower overall, so we use FP32 as
the high-precision reference. For Euclidean FP8, standard per-
block scaling with block size 512 matches FP32 on only 4 out
of 14 downstream tasks, which is worse than the layer-wise
Euclidean baseline; therefore, we use the stronger layer-wise
Euclidean FP8 baseline in the main comparison.
B. Scaling Benefit of Low-Precision Communication
We first examine the system benefit of low-precision com-
munication as scale increases. Figure 5 reports the step-time
improvement of Euclidean baseline and GeoFP8 relative to the
FP32 baseline as a function of GPU count, for both the 300M
and 600M models. Positive values indicate that the method
is faster than FP32, while negative values indicate that it is
slower.
We observe three trends. First, the communication ad-
vantage of low-precision communication becomes more pro-
nounced at a larger scale, especially for the 600M model.
Second, the larger model benefits more consistently from
reduced communication cost, which is expected because com-
munication occupies a larger fraction of end-to-end step time
at a larger scale. Third, GeoFP8 remains slower than direct
Euclidean baseline because it introduces additional geometry-
aware computation, but it still retains a substantial fraction
of the low-precision systems’ benefit relative to FP32. Most
importantly, the gap widens as GPU count increases, which is
precisely the regime where communication efficiency matters
most.
These results indicate that low-precision communication is
more effective for larger models at larger scales, creating room
for a geometry-aware method that trades a controlled amount
of extra computation for better preservation of model quality.
C. Validation Loss During Pretraining
We next examine pretraining behavior through validation
loss. Figure 6 compares FP32, Euclidean baseline, and GeoFP8
for both model sizes.
The main observation is that GeoFP8 and Euclidean base-
line exhibit very similar validation-loss trajectories across both
model sizes. Both low-precision methods remain reasonably
close to the FP32 baseline throughout pretraining, but the gap
between GeoFP8 and Euclidean baseline is small in validation
loss.
This point is important for interpreting the rest of the results.
Validation loss is a useful optimization-level signal, but our
method changes communication fidelity rather than the model
architecture or objective. In such a setting, small differences in
validation loss do not necessarily imply degraded downstream
model performance. For that reason, we treat validation loss as
an informative but incomplete measure, and use downstream
evaluation as the decisive test.
D. Downstream Task Performance
We now turn to the main practical question of the paper:
Which method better preserves useful model quality after end-
to-end pretraining?
Fig. 5: Step-time improvement over FP32 communication as
GPU count increases. Positive values indicate faster pretrain-
ing than the FP32 baseline; negative values indicate slower
pretraining. The relative benefit of low-precision commu-
nication becomes more pronounced at larger GPU counts,
especially for the 600M model. GeoFP8 is slower than di-
rect Euclidean baseline because of its additional geometry-
aware computation, but it retains a substantial fraction of the
communication-side speedup over FP32.
Fig. 6: Validation loss during pretraining for the 300M model
(top) and the 600M model (bottom), comparing FP32, direct
Euclidean baseline communication, and GeoFP8. In both
cases, GeoFP8 and Euclidean baseline exhibit very similar
validation-loss curves, with both remaining reasonably close
to the FP32 baseline. This also highlights a broader point:
validation loss is only a partial proxy for pretraining quality,
and differences between communication methods may become
more apparent in downstream performance than in pretraining
loss alone.
Table III reports downstream results for all evaluated meth-
ods. Each column corresponds to one downstream task, and
the rightmost column summarizes the number of task wins


---

TABLE III: Downstream task performance of models pretrained with FP32, BF16, Euclidean baseline, per-block Euclidean
FP8, full K-FAC, and geometry-informed communication. Higher is better. “Wins vs. FP32” counts the number of tasks on
which the method outperforms the FP32 baseline.
Model
Method
BOOLQ
CB-ACC
COPA
MUL-RC
RCD-F1
RTE
WiC
WSC
LAMB.
RACE
M-QA
PIQA
WINO.
LAMB-STD
Wins vs. FP32
600M
FP32
0.5908
0.2679
0.7500
0.5532
0.8352
0.5090
0.5000
0.4423
0.4941
0.3120
0.2251
0.6719
0.5612
0.3716
–
BF16
0.6058
0.2321
0.7100
0.5608
0.8346
0.5307
0.4875
0.4231
0.4865
0.3158
0.2295
0.6806
0.5391
0.3873
7/14
Per-block Euclidean
0.5679
0.1250
0.7100
0.5549
0.8205
0.5487
0.5188
0.6058
0.4795
0.3024
0.2161
0.6665
0.5478
0.3685
4/14
Euclidean
0.6119
0.4107
0.7100
0.5025
0.8271
0.5199
0.4969
0.5865
0.4824
0.3100
0.2201
0.6659
0.5320
0.3850
5/14
GeoFP8
0.6104
0.3036
0.7200
0.5054
0.8197
0.5307
0.5031
0.4712
0.4875
0.3033
0.2157
0.6763
0.5257
0.3722
7/14
300M
FP32
0.5226
0.4107
0.6900
0.5528
0.8020
0.5271
0.5031
0.3846
0.4514
0.2986
0.2261
0.6507
0.5375
0.3223
–
Euclidean
0.5734
0.2857
0.6500
0.4878
0.7954
0.5271
0.5016
0.3654
0.4347
0.3043
0.2224
0.6480
0.5162
0.3334
4/14
Full K-FAC
0.6064
0.4107
0.7000
0.5140
0.7799
0.5343
0.5063
0.3654
0.4225
0.3081
0.2188
0.6458
0.5335
0.3076
6/14
GeoFP8
0.5740
0.2321
0.7200
0.4420
0.7881
0.5848
0.4969
0.4038
0.4256
0.3014
0.2302
0.6621
0.5335
0.3089
7/14
relative to FP32. We present the results task by task because
the main question is not only overall performance, but also
how consistently each communication method preserves useful
model quality across different downstream evaluations.
The downstream results reveal a clearer distinction than
the validation-loss curves. For the 300M model, Euclidean
baseline outperforms FP32 on 4 out of 14 tasks, whereas
GeoFP8 outperforms FP32 on 7 out of 14 tasks. We also
evaluate the full K-FAC variant on the 300M model. Although
full K-FAC uses a more complete geometry-aware state, it
outperforms FP32 on 6 out of 14 tasks, which is still worse
than the final selective GeoFP8 design. This supports our
design choice that applying geometry awareness selectively
can preserve more useful downstream behavior than using a
heavier full K-FAC-style branch everywhere.
For the 600M model, Euclidean baseline outperforms FP32
on 5 out of 14 tasks, whereas GeoFP8 again outperforms
FP32 on 7 out of 14 tasks. We additionally report the results
for the 600M BF16 and per-block Euclidean FP8 ablations.
BF16 matches FP32 on 7 out of 14 tasks, but its absolute task
values are slightly lower overall, so FP32 remains the stronger
high-precision reference. Per-block Euclidean FP8 matches
FP32 on only 4 out of 14 tasks, which is worse than the
layer-wise Euclidean baseline. Therefore, the main Euclidean
comparison uses the stronger layer-wise FP8 baseline rather
than the weaker per-block variant.
Thus, while all methods remain relatively close in validation
loss, GeoFP8 yields a more favorable downstream profile than
direct Euclidean baseline across both model sizes and also
improves over the heavier full K-FAC variant in the 300M
ablation.
These results should not be interpreted as GeoFP8 matching
FP32 on every task. Rather, they show that GeoFP8 improves
the cross-task preservation profile over direct Euclidean FP8
communication. The purpose of communication compression
is not merely to keep the loss curve visually close to FP32,
but to preserve the usefulness of the synchronized update.
Our results indicate that geometry-aware communication better
protects task-relevant pretraining signal under aggressive low-
precision communication. In other words, changing the com-
munication coordinates improves the downstream preservation
profile over the Euclidean baseline, even when the improve-
ment is only weakly reflected in validation loss.
E. Memory Study
We compare the memory footprint of GeoFP8 with that
of the direct Euclidean baseline. For the 300M model, Ge-
oFP8 increases memory usage by approximately 3.33%. For
the 600M model, the increase is approximately 8.98%. This
overhead mainly comes from the stored input-side low-rank
factors and transformed-coordinate error-feedback buffers for
the selected layers. Since the geometry branch is selective,
the overhead remains limited rather than scaling as a full two-
sided per-layer K-FAC state. Overall, these results show that
the geometry-aware design introduces only a modest memory
overhead for the two models.
F. Discussion
The experiments support three main conclusions.
First, the system benefit of low-precision communication
becomes larger as the scale increases. This is especially clear
for the 600M model at higher GPU counts, where communi-
cation occupies a larger fraction of overall step time.
Second, both GeoFP8 and the direct Euclidean baseline
remain reasonably close to FP32 in validation loss across
both model sizes, while also exhibiting very similar loss
trajectories to each other. This suggests that validation loss
is an informative but incomplete proxy for the effect of
communication quality on final model performance.
Third, and most importantly, downstream evaluation shows
that geometry-aware communication better preserves model
quality than direct Euclidean baseline communication. Across
both models, GeoFP8 achieves a more favorable downstream
preservation profile than the Euclidean baseline, although it
does not improve every individual task. The 300M full K-
FAC ablation and the 600M BF16 and per-block Euclidean
ablations further support the choice of the final comparison
setting: full K-FAC is not better than the selective design,
BF16 is not a stronger reference than FP32, and per-block Eu-
clidean FP8 is weaker than the layer-wise Euclidean baseline.
This means the main benefit of GeoFP8 would be understated
if evaluation were restricted to loss curves alone.
Taken together, these results support the central claim of
the paper: Changing communication coordinates can make
low-precision gradient communication more faithful to the
high-precision reference than direct Euclidean FP8, and with
selective deployment, this can be done with a practical
systems cost. Although our experiments focus on FP8, the


---

coordinate-change principle behind GeoFP8 is not specific to
FP8, suggesting that the same idea can be applied to future
FP4 communication once hardware and software support are
mature.
G. Limitations
Our current evaluation has several limitations. First, the
end-to-end experiments focus on two medium-scale Llama
models and a fixed pretraining recipe. It would be valu-
able to test larger models, longer training horizons, and
broader hardware regimes. Second, although the downstream
results already show a consistent advantage of GeoFP8 over
direct Euclidean baseline, the evaluation would be further
strengthened by additional seeds and a larger benchmark suite.
Third, this work focuses specifically on communication-only
low-precision communication rather than a fully quantized
pretraining stack. Future work can study whether the same
geometry-aware principle remains useful when combined with
broader low-precision pretraining recipes.
Even with these limitations, the present results already
establish the main practical point: geometry-aware communi-
cation improves the quality retained under FP8 gradient com-
munication relative to direct Euclidean FP8, while preserving
part of the communication-side efficiency that makes low-
precision communication attractive.
VII. CONCLUSION
This paper studies low-precision gradient communication
for LLM pretraining from a geometry-informed perspective.
Our starting point is the observation that low-precision com-
munication error is not only a numerical-format issue, but
also a coordinate-system issue: Under anisotropic gradient
geometry, direct quantization in Euclidean coordinates can
introduce substantial distortion into the aggregated update.
Motivated by this insight, we formulate a full geometry-
aware communication rule based on a K-FAC-style second-
order structure and progressively simplify it into a practical
system implementation. Our results show that most of the
measured FP8 round-trip fidelity benefit of the full two-sided
formulation can be recovered by a much cheaper hybrid design
that retains only the input-side geometry, approximates it with
a 32-rank representation, and applies it selectively to the
most numerically vulnerable layers. The resulting GeoFP8
system preserves the Euclidean baseline fast path for most
of the model while improving the fidelity–overhead tradeoff
of low-precision communication. GeoFP8 achieves a 7.6%
pretraining time improvement for Llama-600M pretraining
across 64 NVIDIA GH200 Superchips, while improving the
downstream preservation profile over direct Euclidean FP8
communication. As communication pressure continues to grow
with model size and GPU scale, geometry-aware communica-
tion may become increasingly useful for future lower-bit LLM
pretraining systems.
We hope this perspective encourages further work on
communication-aware geometric methods for efficient dis-
tributed LLM pretraining.
REFERENCES
[1] T. Ben-Nun and T. Hoefler, “Demystifying parallel and distributed
deep learning: An in-depth concurrency analysis,” ACM Computing
Surveys, vol. 52, no. 4, pp. 65:1–65:43, 2019. [Online]. Available:
https://doi.org/10.48550/arXiv.1802.09941
[2] P. Patarasuk and X. Yuan, “Bandwidth optimal all-reduce algorithms
for clusters of workstations,” Journal of Parallel and Distributed
Computing, vol. 69, no. 2, pp. 117–124, 2009. [Online]. Available:
https://doi.org/10.1016/j.jpdc.2008.09.002
[3] J. Song, J. Yim, J. Jung, H. Jang, H.-J. Kim, Y. Kim, and
J. Lee, “Optimus-CC: Efficient Large NLP Model Training with
3D Parallelism Aware Communication Compression,” in Proceedings
of the 28th ACM International Conference on Architectural Support
for Programming Languages and Operating Systems, Volume 2.
Association for Computing Machinery, 2023, pp. 560–573. [Online].
Available: https://doi.org/10.1145/3575693.3575712
[4] H. Peng, K. Wu, Y. Wei, G. Zhao, Y. Yang, Z. Liu, Y. Xiong, Z. Yang,
B. Ni, J. Hu, R. Li, M. Zhang, C. Li, J. Ning, R. Wang, Z. Zhang,
S. Liu, J. Chau, H. Hu, and P. Cheng, “FP8-LM: Training FP8 large
language models,” arXiv preprint arXiv:2310.18313, 2023. [Online].
Available: https://doi.org/10.48550/arXiv.2310.18313
[5] H. Xi, H. Cai, L. Zhu, Y. Lu, K. Keutzer, J. Chen, and S. Han, “COAT:
Compressing optimizer states and activation for memory-efficient FP8
training,” arXiv preprint arXiv:2410.19313, 2024. [Online]. Available:
https://doi.org/10.48550/arXiv.2410.19313
[6] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,”
2019. [Online]. Available: https://doi.org/10.48550/arXiv.1711.05101
[7] J. Jia, C. Xie, H. Lu, D. Wang, H. Feng, C. Zhang, B. Sun, H. Lin,
Z. Zhang, X. Liu, and D. Tao, “SDP4Bit: Toward 4-bit Communication
Quantization in Sharded Data Parallelism for LLM Training,” 2024.
[Online]. Available: https://doi.org/10.48550/arXiv.2410.15526
[8] K. Jordan, Y. Jin, V. Boza, Y. Jiacheng, F. Cesista, L. Newhouse, and
J. Bernstein, “Muon: An optimizer for hidden layers in neural networks,”
2024. [Online]. Available: https://kellerjordan.github.io/posts/muon/
[9] S.-i.
Amari,
Information
Geometry
and
Its
Applications.
Springer Tokyo, 2016. [Online]. Available: https://doi.org/10.1007/
978-4-431-55978-8
[10] ——,
“Natural
gradient
works
efficiently
in
learning,”
Neural
Computation, vol. 10, no. 2, pp. 251–276, 1998. [Online]. Available:
https://doi.org/10.1162/089976698300017746
[11] J.
Martens
and
R.
Grosse,
“Optimizing
neural
networks
with
kronecker-factored approximate curvature,” in Proceedings of the 32nd
International Conference on Machine Learning, ser. Proceedings of
Machine Learning Research, vol. 37.
PMLR, 2015, pp. 2408–2417.
[Online]. Available: https://doi.org/10.48550/arXiv.1503.05671
[12] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary,
V.
A.
Korthikanti,
D.
Vainbrand,
P.
Kashinkunti,
J.
Bernauer,
B. Catanzaro, A. Phanishayee, and M. Zaharia, “Efficient large-scale
language model training on gpu clusters using megatron-lm,” in
Proceedings of the International Conference for High Performance
Computing,
Networking,
Storage
and
Analysis,
2021.
[Online].
Available: https://doi.org/10.48550/arXiv.2104.04473
[13] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, “Openwebtext cor-
pus,” http://Skylion007.github.io/OpenWebTextCorpus, 2019. [Online].
Available: http://Skylion007.github.io/OpenWebTextCorpus
[14] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and
B. Catanzaro, “Megatron-lm: Training multi-billion parameter language
models using model parallelism,” arXiv preprint arXiv:1909.08053,
2019. [Online]. Available: https://doi.org/10.48550/arXiv.1909.08053
[15] Y. Huang, Y. Cheng, A. Bapna, O. Firat, D. Chen, M. X. Chen, H. Lee,
J. Ngiam, Q. V. Le, Y. Wu, and Z. Chen, “Gpipe: Efficient training
of giant neural networks using pipeline parallelism,” in Advances
in Neural Information Processing Systems, vol. 32, 2019. [Online].
Available: https://doi.org/10.48550/arXiv.1811.06965
[16] D. Narayanan, A. Phanishayee, K. Shi, X. Chen, and M. Zaharia,
“Memory-efficient pipeline-parallel dnn training,” in Proceedings of the
38th International Conference on Machine Learning, ser. Proceedings
of Machine Learning Research, vol. 139. PMLR, 2021, pp. 7937–7947.
[Online]. Available: https://doi.org/10.48550/arXiv.2006.09503
[17] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “Zero: Memory
optimizations toward training trillion parameter models,” in Proceedings
of the International Conference for High Performance Computing,
Networking, Storage and Analysis, 2020, pp. 1–16. [Online]. Available:
https://doi.org/10.48550/arXiv.1910.02054


---

[18] G.
Wang,
H.
Qin,
S.
A.
Jacobs,
C.
Holmes,
S.
Rajbhandari,
O. Ruwase, F. Yan, L. Yang, and Y. He, “ZeRO++: Extremely efficient
collective communication for giant model training,” arXiv preprint
arXiv:2306.10209, 2023. [Online]. Available: https://doi.org/10.48550/
arXiv.2306.10209
[19] R.
Eschenhagen,
A.
Immer,
R.
E.
Turner,
F.
Schneider,
and
P. Hennig, “Kronecker-factored approximate curvature for modern
neural network architectures,” in Advances in Neural Information
Processing Systems, vol. 36, 2023, pp. 33 624–33 655. [Online].
Available: https://doi.org/10.48550/arXiv.2311.00636
[20] D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, “Qsgd:
Communication-efficient sgd via gradient quantization and encoding,”
in Advances in Neural Information Processing Systems, vol. 30, 2017.
[Online]. Available: https://doi.org/10.48550/arXiv.1610.02132
[21] Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally, “Deep gradient
compression: Reducing the communication bandwidth for distributed
training,” in International Conference on Learning Representations,
2018. [Online]. Available: https://doi.org/10.48550/arXiv.1712.01887
[22] T. Vogels, S. P. Karimireddy, and M. Jaggi, “Powersgd: Practical low-
rank gradient compression for distributed optimization,” in Advances
in Neural Information Processing Systems, vol. 32, 2019. [Online].
Available: https://doi.org/10.48550/arXiv.1905.13727
[23] H. Tang, S. Gan, A. A. Awan, S. Rajbhandari, C. Li, X. Lian, J. Liu,
C. Zhang, and Y. He, “1-bit adam: Communication efficient large-scale
training with adam’s convergence speed,” in Proceedings of the 38th
International Conference on Machine Learning, ser. Proceedings of
Machine Learning Research, vol. 139. PMLR, 2021, pp. 10 118–10 129.
[Online]. Available: https://doi.org/10.48550/arXiv.2102.02888
[24] P. Micikevicius, D. Stosic, N. Burgess, M. Cornea, P. Dubey,
R.
Grisenthwaite,
S.
Ha,
A.
Heinecke,
P.
Judd,
J.
Kamalu,
N. Mellempudi, S. Oberman, M. Shoeybi, M. Siu, and H. Wu, “Fp8
formats for deep learning,” arXiv preprint arXiv:2209.05433, 2022.
[Online]. Available: https://doi.org/10.48550/arXiv.2209.05433
[25] X.
Xie,
Z.
Lin,
K.-C.
Toh,
and
P.
Zhou,
“Loco:
Low-
bit
communication
adaptor
for
large-scale
model
training,”
arXiv
preprint
arXiv:2407.04480,
2024.
[Online].
Available:
https://doi.org/10.48550/arXiv.2407.04480
[26] P.
Micikevicius,
S.
Narang,
J.
Alben,
G.
Diamos,
E.
Elsen,
D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and
H. Wu, “Mixed precision training,” in International Conference
on
Learning
Representations,
2018.
[Online].
Available:
https:
//doi.org/10.48550/arXiv.1710.03740
[27] S.
P.
Perez,
Y.
Zhang,
J.
Briggs,
C.
Blake,
J.
Levy-Kramer,
P. Balanca, C. Luschi, S. Barlow, and A. Fitzgibbon, “Training
and inference of large language models using 8-bit floating point,”
arXiv preprint arXiv:2309.17224, 2023. [Online]. Available: https:
//doi.org/10.48550/arXiv.2309.17224
