---
title: Communication-Efficient Distributed Training
id: communication-efficient-distributed-training
tags:
- cmp170hx-training-feasibility-a0d4fc
- sub-q4
- all-reduce
- gradient-compression
- pcie-fabric
- communication-volume
- ddp
- approximate-allreduce
- coded-computing
created: '2026-10-27T10:46:52.239075Z'
updated: '2026-10-27T19:00:31.292931Z'
source: https://arxiv.org/abs/2609.23898
source_domain: arxiv.org
fetched_at: '2026-10-27T10:46:52.238482Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Munim & Ramamoorthy (arXiv 2609.23898v2, cs.IT, 26 Sep 2026) propose CERAR,
  a Communication-Efficient Ring All-Reduce protocol that trades exactness for bandwidth:
  each worker splits its gradient into c chunks, linearly encodes them (Vandermonde,
  or a condition-number-constrained construction), and runs L = c + N - 2 rounds over
  an N-worker ring, giving normalized per-worker communication 1 + (N-2)/c instead
  of ring all-reduce''s 2(N-1)/N, at the cost of a bounded approximation error (~1.6-2.9%
  relative l2 gradient error in their training runs). The Vandermonde route is asymptotically
  exact as c grows but ill-conditioned; they give an alternate scheme with error O(eps)
  and condition number O(eps^-r*), r* = ceil(c/N) - 1, plus a constrained optimization
  for practical designs.


  The headline empirical result lands on the fabric that matters for this box: on
  a single node of 8x A100 PCIe (interconnect characterised as 31.5 GB/s), CERAR cuts
  raw all-reduce time by 46.0-47.1% versus NCCL - a 1B-entry FP32 tensor goes from
  17,954.4 ms (NCCL) to 9,558.6 ms (CERAR); 500M entries 8,611.2 -> 4,651.0 ms; 100M
  1,775.9 -> 940.1 ms. On 4x H200 SXM over NVLink (>=300 GB/s) the same scheme is
  33.2-33.9% SLOWER than NCCL (24B entries: 392.049 ms NCCL vs 522.366 ms CERAR),
  so the win is explicitly a low-bandwidth-fabric phenomenon, not a general one.


  End-to-end: Pythia-1.4B on 4x A100 PCIe 40GB for 1,800 steps (~117.96M tokens) takes
  3.79 h with NCCL vs 3.14 h with CERAR (-17.3%, ~39 min saved), throughput 8,644
  -> 10,451 tok/s, final validation loss 1.4218 vs 1.4213 (essentially identical),
  peak memory 36.2 vs 38.6 GB. On 4x A100 SXM 80GB over NVLink CERAR is 3.65% slower
  (4.745 h vs 4.578 h).


  Caveats the drafter must carry: (i) the authors call their implementation a proof
  of concept, not optimised to NCCL''s level (NCCL uses multiple channels and compute/comm
  overlap); (ii) encoding/decoding compute is real and grows with c, so on a box whose
  GPUs are also weak the arithmetic overhead is not free; (iii) the tested PCIe fabric
  is 31.5 GB/s, roughly 30x the ~0.75 GiB/s NCCL all-reduce measured on the CMP 170HX
  box, so the communication-bound regime is far deeper here - the direction of the
  effect should hold, the magnitude is not transferable; (iv) the approximation error
  is ~2-3% relative on gradients, a training-quality risk the paper argues is benign
  at 1.4B scale only.'
raw_file: raw/communication-efficient-distributed-training.pdf
doi: arXiv:2609.23898
utility_score: 12.0
citation_count: 0
---

1
Communication-Efficient Distributed Training
via Ring-Based Coded Approximate
All-Reduce
Sifat Munim and Aditya Ramamoorthy
Abstract
Ring All-Reduce is a widely used protocol within large-scale distributed training for aggregating gradients across
workers in each iteration. For a system with N workers, its normalized per-worker communication is 2(N −1)/N.
In this work we present a communication-efficient Ring All-Reduce (CERAR) protocol for “approximate” gradient
aggregation. CERAR partitions each local gradient into c components and crucially relies on linear encoding and
decoding operations. It performs L = c + N −2 communication rounds over an N-worker ring, yielding normalized
communication rate 1+(N −2)/c and normalized storage rate 1+N/c. We present an explicit Vandermonde matrix
based construction, whose approximation error can be made arbitrarily close to zero with communication and storage
rates approaching one with increasing c. However, this limit is achieved through ill-conditioned encoding matrices.
Accordingly, we give an alternate construction whose error is O(ϵ), while the relevant condition numbers are O(ϵ−r⋆),
where r⋆= ⌈c/N⌉−1. This naturally motivates a condition-number-constrained optimization formulation for trading
off the competing objectives and obtaining numerically stable practical designs.
Our numerical experiments demonstrate that when compared with the production-standard NCCL on NVIDIA
A100 PCIe GPU Clusters with low-bandwidth PCIe interconnect, CERAR reduces the raw All-Reduce phase time by
up to 47%, and for training a 1.4-billion-parameter Pythia model it reduces training time by about 17% for the same
number of iterations while achieving essentially identical validation loss. On the other hand, for GPU clusters with
high-bandwidth NVLink interconnects, CERAR performs worse on the All-Reduce phase time and slightly worse in
terms of training. Nevertheless, the gap reduces with experiments that involve systems with larger parameter counts.
In this scenario we expect the gains to manifest with more optimized implementations of the CERAR protocol.
I. INTRODUCTION
A typical scenario within model training involves a dataset D = {(xi, yi)} ˜
N
i=1 of ˜N data points, where xi’s and
yi’s are the features and labels, respectively. The aim is to minimize a loss function L ≜1
˜
N
P ˜
N
i=1 l(xi, yi; w) with
respect to w, where w ∈Rd is the unknown parameter that needs to be learned. The function l measures the
prediction error for each data point. In the distributed setting, portions of the dataset are distributed to different
workers who compute the gradient on the data points assigned to them. At the t-th iteration, the system aims at
computing the overall gradient ∇L =
1
˜
N
P ˜
N
i=1 ∇l(xi, yi; wt), following which a parameter-update is performed
and the iterations continue thereafter. We emphasize that D can also represent a relevant subset of the entire dataset,
e.g., when the cluster wants to run mini-batch stochastic gradient descent [1] on an appropriate batch; this is in fact
September 29, 2026
DRAFT
arXiv:2609.23898v2  [cs.IT]  26 Sep 2026


---

Round 4
W0
W1
W2
W3
x0,4 =
δ0,4 + δ3,3
+ δ2,2 + δ1,1
x1,4 =
δ1,4 + δ0,3
+ δ3,2 + δ2,1
x2,4 =
δ2,4 + δ1,3
+ δ0,2 + δ3,1
x3,4 =
δ3,4 + δ2,3
+ δ1,2 + δ0,1
σ0,4 = x3,4 −δ0,1
σ1,4 = x0,4 −δ1,1
σ2,4 = x1,4 −δ2,1
σ3,4 = x2,4 −δ3,1
Round 5
W0
W1
W2
W3
x0,5 =
δ0,5 + δ3,4
+ δ2,3 + δ1,2
x1,5 =
δ1,5 + δ0,4
+ δ3,3 + δ2,2
x2,5 =
δ2,5 + δ1,4
+ δ0,3 + δ3,2
x3,5 =
δ3,5 + δ2,4
+ δ1,3 + δ0,2
σ0,5 = x3,5 −δ0,2
σ1,5 = x0,5 −δ1,2
σ2,5 = x1,5 −δ2,2
σ3,5 = x2,5 −δ3,2
Fig. 1: CERAR protocol with N = 4, c = 5, and L = 7: rounds 4 and 5. Arrows show transmitted vectors; equations beside
workers show the rolling-state updates after reception in the indicated round. The update equations appear in (2).
the most common mode of usage. Several distributed learning architectures have been investigated in the literature
[2], that differ based on how the dataset is divided into chunks and how the workers are connected to each other.
Background and Motivation: For N workers, the basic Ring All-Reduce training algorithm [3], [4] operates by
partitioning D into N equal-sized disjoint chunks Dj, j = 0, . . . , N −1, such that worker Wj is assigned Dj. Wj
computes the sum of the gradients on its assigned data points, i.e., it computes gDj =
1
˜
N
P
i∈Dj ∇l(xi, yi; wt)
(a vector of length-d). Once the workers have computed their assigned gradients, they send messages in parallel
around a clockwise ring such that each worker can compute the overall gradient ∇L. For a ring with N workers,
each worker transmits and receives a total of 2(N −1)d/N real values. For large N, the load normalized by d
approaches 2, i.e., it does not grow with the number of workers. In contrast, in older architectures such as parameter
server (PS) + workers [5], the bandwidth load on the PS grows with N.
The basic Ring All-Reduce algorithm is nominally considered to be optimal from the point of view of recovering
the sum gD0 + · · · + gDN−1 exactly [3]. From an information-theoretic perspective, one can consider the problem
of recovering the sum with arbitrarily small (but nonzero) error; the lower bound of [3] does not apply in this case.
We note that approximate recovery often suffices in scenarios such as distributed parameter training, i.e., instead
of true gradients, we often settle for approximate gradients which do not affect the quality of training much. We
consider this approximate recovery setting in this paper and investigate the fundamental question of how small the
communication load on a worker can be in this setting.
Problem Formulation: For a given number of workers N, suppose that (for i = 0, . . . , N −1) Wi divides its
gradient gDi into c chunks. A round consists of clockwise communication over the ring where Wi sends Wi+1
(indices reduced modulo-N) a real-valued vector of length ˜c. Suppose that a given protocol communicates over L
rounds. The requirement is that worker Wi should be able to reconstruct an approximate gradient ˆgi at the end of
2


---

these rounds such that
∥ˆgi −∇L∥2 ≤˜δ, for all i = 0, . . . , N −1,
(1)
where ˜δ > 0 is a small constant. Thus, the total number of real values communicated in the protocol per worker
is L˜c. We define the normalized rate of the protocol as L˜c
d , e.g., the basic Ring All-Reduce operates with L =
2(N −1), c = N, ˜c = d/c and with ˜δ = 0 (exact reconstruction).
Related Work: Gradient quantization [6], [7], [8], and sparsification [9], [10], [11] aim at reducing the commu-
nication load of distributed training by “compressing” or “sparsifying” the gradients and then communicating the
compressed representations. These approaches address communication reduction along a direction orthogonal to
ours. Specifically, it may be possible to apply our techniques in conjunction with them.
Contributions in the area of gradient coding [12], [13], [14], [15], [16] aim at making distributed training robust
to worker failures/slowdowns; however, these are in the PS + workers model and moreover require redundancy
in the assignment of chunks to workers. Coded distributed computing [17], [18], [19] is similar in the sense that
it trades off computational redundancy for reduced communication for exact computation. Our work is related to
approximate gradient coding since we rely critically on the judicious usage of linear combinations of gradient
chunks and aim at recovering approximate gradients. However, crucially, our setting remains the same as the basic
Ring All-Reduce; we do not need redundant chunk assignments.
Main Contributions: In this work, we present a Communication-Efficient, Approximate Ring All-Reduce protocol
named CERAR. (i) We demonstrate that CERAR can achieve arbitrarily small error for a so-called Vandermonde
construction, while operating with normalized communication and storage rates close to 1. The caveat is that
this is achieved via increasingly ill-conditioned encoding and decoding matrices. (ii) We characterize the tradeoff
between the error and numerical stability of the scheme by explicitly deriving a scheme where the error is O(ϵ)
and the condition number of the matrices is O(ϵ−r⋆) where r⋆= ⌈c/N⌉−1. (iii) We then propose a constrained
optimization problem that trades off error for numerical stability in our scheme. Numerical experiments demonstrate
the effectiveness of our approach.
II. COMMUNICATION EFFICIENT RING ALL REDUCE (CERAR) PROTOCOL
The gradients are subdivided into c chunks, i.e., gDi = [gDi[0]T · · · gDi[c−1]T ]T . In each of the L = c+N −2
rounds (numbered 1 through L), the transmitted vector is of length p = d/c (we assume c | d, which can be
enforced by zero-padding). Let Gi :=
h
gDi[0]
· · ·
gDi[c −1]
i
∈Rp×c and let G(i) = [Gi | . . . | Gi+N−1]
(indices reduced modulo-N).
Initialization: Worker Wi has a c × c matrix Ui =
h
ui,1
· · ·
ui,c
i
. After computing Gi, it creates encoded
vectors
∆i :=
h
δi,1
· · ·
δi,c
i
= GiUi.
When Ui is nonsingular, the map Gi 7→∆i is an invertible change of basis. Consequently, the same d-length
memory region that initially contains gDi can be transformed in place so that it contains ∆i. We denote the
corresponding p-length storage slots by Di,1, . . . , Di,c, such that Di,j stores δi,j. In addition, we allocate N −1
3


---

additional p-length empty dedicated buffers in Wi denoted Bi,1, . . . , Bi,N−1, and one rolling p-length buffer denoted
Ci. Ci holds the current state used in the next transmission and can also serve as the scratch buffer for the in-place
transform Gi 7→∆i before communication begins. The buffer Ci is initialized to the “rolling state” σi,0 = 0.
The formal definition of the protocol appears in Algorithm 1, which describes the communication and storage
updates at each worker. For round k, where 1 ≤k ≤c, worker Wi adds the local encoded vector δi,k to its rolling
state σi,k−1 and transmits the result xi,k to Wi+1, and for c + 1 ≤k ≤L it simply transmits xi,k = σi,k−1. The
first N −1 received vectors (in rounds 1 through N −1) are stored in dedicated buffers (Bi,j’s for Wi). For round
k = j + N −1 (where j ≥1), Wi subtracts its own returning local contribution, namely δi,j from the received
vector to obtain the next rolling state σi,k, while overwriting the corresponding local buffer Di,j with the received
vector for final decoding. After L = c + N −2 rounds, the stored observations, together with the final unmodified
local encoded vector δi,c, form Yi, from which the approximate gradient is reconstructed using Ri. The update
equations at Wi are (worker indices reduced modulo-N)
xi,k =





δi,k + σi,k−1,
1 ≤k ≤c,
σi,k−1,
c + 1 ≤k ≤L
& σi,k =





xi−1,k,
1 ≤k ≤N −1,
xi−1,k −δi,k−N+1,
N ≤k ≤L.
(2)
A. Illustrative Example with N = 4, c = 5, L = 7
We explain the basic protocol for the case of four workers and seven rounds, i.e., N = 4, c = 5, L = 7. In round
1, worker Wi receives xi−1,1 = δi−1,1 (since σi,0 = 0, for all Wi) from its preceding worker, Wi−1 and stores
it in Bi,1 . At the end of round 1, Wi updates its storage to σi,1 = xi−1,1 = δi−1,1. In round 4 (see Fig. 1), Wi
receives the following vector from Wi−1.
xi−1,4 = δi−1,4 + σi−1,3 = δi−1,4 + δi−2,3 + δi−3,2 + δi−4,1,
where the second equality follows by (2). Note that i −4 ≡i (mod 4), i.e., δi,1 needs to be canceled at Wi.
Therefore,
σi,4 = xi−1,4 −δi,1 = δi−1,4 + δi−2,3 + δi−3,2,
and Di,1 = xi−1,4. This process continues until round 5 (note that c = 5 here), following which the workers
simply forward their stored values. Thus, in round 6, Wi simply receives σi−1,5. For example, W0 receives σ3,5 =
δ2,5 + δ1,4 + δ0,3 and it calculates
σ0,6 = δ2,5 + δ1,4
by canceling δ0,3. It can be observed that the received vectors at Wi at each round can be compactly represented
as matrix Mi of dimension Nc × (L + 1). For example, M0 and M1 are given as
4


---

Algorithm 1 Communication and storage updates at worker Wi
Input: N ≥2, L = c + N −2, and encoded vectors δi,1, . . . , δi,c
Output: Stored observations Yi for final decoding
Worker indices are reduced modulo N. All workers execute each round concurrently.
1: for j = 1, . . . , c do
2:
Di,j ←δi,j
3: end for
4: Ci ←0
▷Initial state σi,0
5: for k = 1, . . . , L do
Step 1: Generate, accumulate, and transmit
6:
if k ≤c then
7:
Ci ←Ci + Di,k
8:
end if
9:
Transmit Ci to Wi+1 as xi,k
Complete the outgoing send before overwriting Ci.
Step 2: Receive, store, and update
10:
Receive xi−1,k from Wi−1 into Ci
11:
if k ≤N −1 then
12:
Bi,k ←Ci
13:
else
14:
j ←k −N + 1
15:
Ci ←Ci −Di,j
▷Cancel the returning local contribution
16:
Di,j ←Di,j + Ci
▷Store xi−1,k for decoding
17:
end if
The buffer Ci now contains σi,k.
18: end for
19: Yi ←[ Bi,1 | · · · | Bi,N−1 | Di,1 | · · · | Di,c ]
▷Logical ordering of the existing buffers
M0 =


0
0
0
u0,1
u0,2
u0,3
u0,4
u0,5
0
0
u1,1
u1,2
u1,3
u1,4
u1,5
0
0
u2,1
u2,2
u2,3
u2,4
u2,5
0
0
u3,1
u3,2
u3,3
u3,4
u3,5
0
0
0


, and
(3)
5


---

M1 =


0
0
0
u1,1
u1,2
u1,3
u1,4
u1,5
0
0
u2,1
u2,2
u2,3
u2,4
u2,5
0
0
u3,1
u3,2
u3,3
u3,4
u3,5
0
0
u0,1
u0,2
u0,3
u0,4
u0,5
0
0
0


.
(4)
M2 and M3 can be obtained in a similar manner. For instance, the interpretation of the fourth column of M0 is
that it receives the p-length vector
y0,4 =
c−1
X
ℓ=0
u0,1[ℓ]gD0[ℓ] +
c−1
X
ℓ=0
u1,2[ℓ]gD1[ℓ] +
c−1
X
ℓ=0
u2,3[ℓ]gD2[ℓ] +
c−1
X
ℓ=0
u3,4[ℓ]gD3[ℓ]
from W3 in round 4. y0,4 is stored in D0,1 after overwriting δ0,1. The last column corresponds to y0,8 = δ0,5 that
is contained in D0,5. It is not overwritten as there are only L = 7 rounds.
The worker Wi wants to obtain ˆgi (see (1)). Thus, W0, e.g., wants to find a decoding vector r1 of length-(L+1)
such that it can approximately recover PN−1
i=0 gDi[0], i.e.,
L+1
X
i=1
r1iy0,i ≈
N−1
X
i=0
gDi[0].
This in turn implies that we need M0r1 ≈[eT
0 eT
0 eT
0 eT
0 ]T = 14 ⊗e0 (⊗- Kronecker product), where ei, i =
0, . . . , c −1 denotes the canonical basis vector in Rc. In a similar manner it can be observed that we need the
existence of other decoding vectors such that 1N ⊗ei for i = 1, . . . , c −1 can be decoded from each of them.
This can be compactly represented as the existence of a (L + 1) × c matrix Ri such that MiRi ≈1N ⊗Ic (1N
- all-ones vector of length-N, Ic - c × c identity matrix). Suppose that the protocol identifies suitable Mi and Ri
such that ∥MiRi −1N ⊗Ic∥F ≤δdec for i = 0, . . . , N −1, where ∥·∥F denotes the Frobenius norm. Then, we
have
∥ˆgi −∇L∥2 =



G(i)(MiRi −1N ⊗Ic)



F ≤



G(i)


2 δdec =



G(0)


2 δdec,
(5)
where δdec is the decoding error, and ∥·∥2 denotes the spectral norm. The inequality is standard, and the second
equality holds since the G(i)’s are related by permutations. Thus, assuming that


G(0)

2 ≤CG uniformly over the
gradients under consideration, the overall error in the gradient computation (cf. (1)) can be made small enough by
choosing δdec small enough. Appendix §V-A shows that the normalized rate of any linear protocol satisfying (1)
for sufficiently small ˜δ (under the bounded spectral norm assumption) is at least 1.
B. Discussion of the CERAR protocol
The total storage per worker includes Di,j, j = 1, . . . , c, the buffers Bi,j, j = 1, . . . , N −1 and the buffer Ci
that contains the rolling state. Thus, the normalized storage rate and communication rate of CERAR are given by
S = 1 + N
c ,
and
R = L
c = 1 + N −2
c
.
We note that the storage rate of the basic Ring All-Reduce is exactly 1. Let Uk, k = 0, . . . , N −1 denote the
collection of the uk,j, 1 ≤j ≤c variables, i.e., Uk = [uk,1 uk,2 . . . uk,c] and let U = (U0, . . . , UN−1); these
are the encoding matrices. Each Mi, 0 ≤i ≤N −1 denotes a Nc × (L + 1) matrix that is obtained as a result of
6


---

the CERAR protocol (examples in (3), (4)); henceforth we denote it as Mi(U) to show the dependence explicitly.
Furthermore, note that the r-th block-row of Mi(U) of dimension c × (L + 1) can be expressed as
Mi(U)[r, :] = [0c×(N−1−r) Ui+r 0c×r] for r = 0, . . . , N −1.
(6)
Here the subscript i + r is reduced modulo-N. Let Ri, 0 ≤i ≤N −1 denote (L + 1) × c decoding matrices; let
R = (R0, . . . , RN−1), and the target T = 1N ⊗Ic. For Ri, we denote the c consecutive rows starting at row s
by Ws(Ri) = Ri[s : s + c −1, :] for s = 0, . . . , N −1, where the rows are numbered 0, . . . , L. The least-squares
fit in our scenario is measured by
Φ(U, R) =
N−1
X
i=0
∥Mi(U)Ri −T∥2
F .
(7)
We note here that both R and S are decreasing in c, and the basic Ring All-Reduce already provides a zero-error
solution for the case when c = N. For the remainder of the paper we only consider the regime c > N. Lemma 1
is proved in Appendix, § V-B.
Lemma 1: Suppose that c > N. There is no finite collection of encoding and decoding matrices (U, R) for
which Φ(U, R) = 0.
1) Multiplicative Perturbation Construction (MPC): We now present a specific construction of the U and R
matrices that are obtained by (i) a N × c base matrix B with rows b0, . . . , bN−1, each of length-c, and (ii) a c × c
nonsingular matrix D that perturbs B by right-multiplication. Define for r = 0, . . . , N −1 and an integer m, the
row-vector
ar+mN = brDm.
(8)
Thus, the above vector is defined for all integer indices for the LHS of (8). Note that we have aℓ+N = aℓD. For
every integer ℓ, define the c × c consecutive-row window
Aℓ=


aℓ
aℓ+1
...
aℓ+c−1


.
(9)
It follows that Aℓ+N = AℓD. We assume that A0, . . . , AN−1 are nonsingular. Define
VN−1−k := Ak,
Uk := V−1
k ,
k = 0, . . . , N −1.
(10)
For worker Wi, define the explicit decoder Ri ∈R(L+1)×c by taking the L + 1 consecutive rows:
Ri[ℓ, :] = aℓ−i,
ℓ= 0, . . . , L.
(11)
Equations (10) and (11) specify the construction of the scheme; the s-th c-row window of Ri is
Ws(Ri) = As−i,
s = 0, . . . , N −1.
(12)
7


---

Proposition 1: Let worker indices be reduced modulo-N. For the construction above,
Ui−1−sWs(Ri) =





Ic,
s ≥i,
D−1,
s < i.
Proof. If s ≥i, then s −i ∈{0, . . . , N −1}. By (10),
As−i = Vi−1−s, (reducing the index of RHS modulo-N)
Therefore, Ui−1−sWs(Ri) = V−1
i−1−sAs−i = Ic.
If s < i, we can assert As−i = As−i+ND−1 where s −i + N ≥0 since s −i ≥−(N −1). Again,
As−i+N = Vi−1−s, and hence
Ui−1−sWs(Ri) = V−1
i−1−sAs−i+ND−1 = D−1.
Proposition 2: Let ˜U and ˜R denote the collections of matrices constructed as in (10) and (11). Then,
inf
U,R Φ(U, R) ≤Φ( ˜U, ˜R) = N(N −1)
2
∥D−1 −Ic∥2
F .
(13)
Proof. Worker Wi has exactly i indices s ∈{0, . . . , N −1} for which s < i. By Proposition 1, these are precisely
the nonzero residual blocks, and each is equal to D−1 −Ic. Therefore
Φ( ˜U, ˜R) =
N−1
X
i=0
i ∥D−1 −Ic∥2
F = N(N −1)
2
∥D−1 −Ic∥2
F .
The matrices ˜U and ˜R specified in (10) and (11) are feasible choices in the least-squares minimization in (7), so
the optimized least-squares residual cannot exceed their residual.
2) A Vandermonde construction that achieves arbitrarily small Φ(U, R): Let ξ1, . . . , ξc be distinct reals, and
for τ > 0, set zj(τ) := eτξj for j = 1, . . . , c. Define
bℓ(τ) :=
h
z1(τ)ℓ
· · ·
zc(τ)ℓ
i
, for ℓ= 0, . . . , N −1, and
Dτ := diag
 z1(τ)N, . . . , zc(τ)N
= diag
 eNτξ1, . . . , eNτξc
.
This defines the sequence specified in (8), and therefore Aℓ. Henceforth, we refer to it as Aℓ(τ) to emphasize
the dependence on τ; it is a shifted Vandermonde matrix with nodes z1(τ), . . . zc(τ) and hence nonsingular
[20]. In particular, this choice satisfies the requirement that A0(τ), . . . , AN−1(τ) are nonsingular. We call this
the Vandermonde construction. The following theorem follows from an application of Proposition 2 (details in
Appendix, §V-C).
Theorem 1: Let ˜Uτ and ˜Rτ be the matrices obtained by applying the Vandermonde construction. Then, Φ( ˜Uτ, ˜Rτ) =
N3(N−1)
2
τ 2 Pc
j=1 ξ2
j + O(τ 3) →0 as τ ↓0 so that infU,R Φ(U, R) = 0. If c > N, the infimum is not attained.
Remark 1: Choosing a sequence τc (dependent on c) such that τ 2
c
Pc
j=1 ξ2
j →0, this shows a rather striking fact
that one can simultaneously achieve a normalized rate and storage of 1 as c →∞1 while driving the approximation
1We need d →∞as well. In practice d is many orders of magnitude larger than c, so this is not a limitation.
8


---

error in the reconstruction arbitrarily close to zero. To our best knowledge, this is the first work that shows such a
result for the well-known All-Reduce setting.
Remark 2: As τ ↓0, all Vandermonde nodes satisfy zj(τ) −→1, so that Vk(τ)Vk(τ)T →c11T . Thus, the
condition number (ratio of maximum to minimum singular values) κ2(Vk(τ)) = κ2(Uk(τ)) −→∞, i.e., this
construction is numerically unstable for small τ. Thus, in practice, one needs to adapt the approach appropriately.
C. Trade-off between LS Error and conditioning
We now examine the role of the multiplicative perturbation D on the conditioning of the Mi matrices. Let
Dϵ = Ic +ϵH with ϵ ∥H∥2 < 1 for a fixed matrix H. This implies that


D−1
ϵ


2 =
1
σmin(Dϵ) ≤
1
1−ϵ∥H∥2 . Moreover,
D−1
ϵ
−Ic = −ϵD−1
ϵ H, so that


D−1
ϵ
−Ic


F ≤ϵ


D−1
ϵ


2 ∥H∥F .
Applying the result of Proposition 2 (nonsingularity of the corresponding Aℓ(ϵ) follows from the forthcoming
assumptions and discussion), we can conclude
Φ(U, R) ≤N(N −1)
2
ϵ2 ∥H∥2
F
(1 −ϵ ∥H∥2)2 = O(ϵ2).
(14)
Consider the c × c consecutive-row window Aℓ(ϵ) where ℓ∈{0, . . . , N −1} that is formed by using B and Dϵ
(cf. (9)). These are the kind of window types that arise in our construction of the decoding matrices Ri’s, except
that they are sometimes right-multiplied by powers of Dϵ. However, κ2(Dϵ) = 1 + O(ϵ) (condition number - ratio
of maximum to minimum singular value) so it does not affect the scaling of κ2(Aℓ(ϵ)).
Let c = βN + s, where 0 ≤s < N and define r⋆=
 c
N

−1. Let Eℓ= {ℓ, ℓ+ 1, . . . , ℓ+ s −1} (mod N).
For j ∈{ℓ, ℓ+ 1, . . . , ℓ+ c −1}, consider its remainder upon division by N. If s > 0, then a given remainder
α ∈{0, . . . , N −1} occurs nα(ℓ) times where
nα(ℓ) =





β + 1,
if α ∈Eℓ
β,
if α /∈Eℓ.
If s = 0, then every remainder occurs exactly β times. For ℓ∈{0, . . . , N −1}, the smallest index j ≥ℓsatisfying
j ≡α (mod N) is j = α + Nmα(ℓ), where
mα(ℓ) =





0,
α ≥ℓ,
1,
α < ℓ.
Thus, the first row associated with remainder α is bαDmα(ℓ)
ϵ
, followed by bαDmα(ℓ)+1
ϵ
, . . . , bαDmα(ℓ)+nα(ℓ)−1
ϵ
.
Let Fn denote the n × n lower-triangular finite-difference matrix (Appendix, §V-E), and Πℓbe a permutation
matrix that groups the rows of Aℓby remainder class, and define ˜Fℓ= diag(Fn0(ℓ), Fn1(ℓ), . . . , FnN−1(ℓ)). Then,
using Lemma 2 (Appendix, §V-E) we have
˜FℓΠℓAℓ(ϵ) = Λℓ(ϵ)Kℓ(ϵ)
(15)
where Λℓ(ϵ) is a diagonal matrix with entries {ϵj : j = 0, . . . , nr(ℓ) −1, r = 0, . . . , N −1} and the corresponding
rows of Kℓ(ϵ) are brDmr(ℓ)
ϵ
Hj. We note that as ϵ ↓0, we have Dmr(ℓ)
ϵ
→Ic. This implies that up to a row-
permutation, Kℓ(ϵ) approaches a matrix denoted K(ℓ)(B, H) where
9


---

K(ℓ)(B, H) =


B
BH
...
BHβ−1


if s = 0,
and
K(ℓ)(B, H) =


B
BH
...
BHβ−1
(BHβ)Eℓ


if s > 0.
Here (BHβ)Eℓdenotes the s rows indexed by the set Eℓ. An example showing the calculation in (15) appears in
Appendix §V-E.
It is not too hard to see that the maximum singular value of K(ℓ)(B, H) is upper bounded by C1σmax(B) where
C1 is a constant that depends on r⋆and σmax(H) (see Appendix, §V-F).
Assumption: We assume that K(ℓ)(B, H) is nonsingular and further that minℓ=0,...,N−1 σmin(K(ℓ)(B, H)) =
γ > 0 . This can be justified through a Schwartz-Zippel type argument, and through multiple trials for selecting
H. Constructions of H that depend on B can also be considered.
In the following theorem, N, c, B, and H are fixed, and all asymptotic notation refers to ϵ →0. The proof
appears in the Appendix, §V-D.
Theorem 2 (Singular value scaling): The singular values of Aℓ(ϵ) have the same powers of ϵ, up to constant
factors, as the diagonal entries of Λℓ(ϵ). Specifically, (i) If s = 0, then for each j = 0, . . . , β −1, exactly N
singular values are Θ(ϵj). (ii) If s > 0, then for each j = 0, . . . , β −1, exactly N singular values are Θ(ϵj) and
the remaining s singular values are Θ(ϵβ).
Corollary 1 (Exact smallest-singular-value exponent): For r⋆=
 c
N

−1 and every ℓ= 0, . . . , N −1,
σmin(Aℓ(ϵ)) = Θ(ϵr⋆),
c0 ≤σmax(Aℓ(ϵ)) ≤˜C1 ∥B∥2 ,
where c0 is a constant independent of ϵ. Thus, σmax(Aℓ(ϵ)) = Θ(1). Thus, κ2(Aℓ(ϵ)) = Θ(ϵ−r⋆). Furthermore,
since the Ui’s are formed as inverses of matrices of the form Aℓ(ϵ), we also have that κ2(Ui) = Θ(ϵ−r⋆). From
the discussion in Appendix, §V-G we can conclude that κ2(Mi) = O(
√
Nϵ−r⋆) for all i = 0, . . . , N −1.
Remark 3: Corollary 1 does not apply to the construction in Section II-B2 since the Vandermonde construction
does not work with a fixed B and H. In particular, the corresponding definitions of B and H change with the
parameter τ that goes to zero.
D. Constrained Optimization Problem Formulation
The construction in Section II-B1 guarantees LS squared error O(ϵ2), and O(
√
Nϵ−r⋆) conditioning of the Mi
matrices. Using this solution as an initialization for the following constrained optimization, it is possible to trade
off these competing objectives. Consider
minimize Φ(U, R)
subject to κ2(Mi(U)) ≤C, for i = 0, . . . , N −1.
(16)
10


---

TABLE I: Raw reduction time in milliseconds; lower is better. M := 106, B := 109.
System / interconnect
Tensor size
NCCL
Our algorithm
Gain
A100 PCIe, 8 GPUs / PCIe
100M
1,775.9
940.1
47.1%
A100 PCIe, 8 GPUs / PCIe
500M
8,611.2
4,651.0
46.0%
A100 PCIe, 8 GPUs / PCIe
1B
17,954.4
9,558.6
46.8%
H200 SXM, 4 GPUs / NVLink
9B
147.250
197.097
−33.9%
H200 SXM, 4 GPUs / NVLink
15B
244.974
327.733
−33.8%
H200 SXM, 4 GPUs / NVLink
24B
392.049
522.366
−33.2%
Although (16) is nonconvex, we use an alternating least-squares heuristic in which the Ui’s are projected onto a
stricter spectral constraint set that is sufficient to guarantee the condition-number constraint. In particular, in one
iteration, we minimize Φ(U, R) with respect to U while keeping R fixed, and vice versa. After this, we project
the computed singular values of the Ui’s into a common interval a0 ≤σmin(Ui) ≤σmax(Ui) ≤b0. Note that this
implies that κ2(Mi) ≤
√
Nb0/a0. Therefore, the constant C in (16) needs to be chosen so that
√
Nb0/a0 ≤C.
III. NUMERICAL EXPERIMENTS
Our numerical experiments compared the performance of the CERAR protocol to the production-level NCCL
All-Reduce by [21]. We considered GPU clusters under two different settings - a low-bandwidth (PCIe, 31.5 GB/s)
and a high-bandwidth (NVLink, ≥300 GB/s) interconnect. All code for recreating these experiments can be found
in an anonymous repository. 2 The CERAR scheme was fit using the methods in Section II-D (details in Appendix,
§V-H).
All-Reduce Phase time: We ran the evaluations on single-node systems with eight A100 GPUs and four H200
GPUs; each GPU started with a FP32 local tensor. CERAR used c = 20, L = 26 for the A100 experiments and
c = 12, L = 14 for the H200 experiments. We measured the elapsed time for each GPU to complete the reduction,
including communication and local computation. For our algorithm, this includes (i) encoding (preparing δi,j’s), (ii)
all communication rounds and their local updates, and (iii) decoding using the stored values and the Ri matrices
to reconstruct the approximate sum. As shown in Table I, CERAR outperforms the NCCL implementation on the
A100 PCIe system, reducing the reduction time by 46.0–47.1%. However, on the H200 NVLink system, CERAR
is 33.2%–33.9% slower than NCCL. Appendix §V-H has more details and more fine-grained information about the
time taken for the different parts of the All-Reduce phase.
We emphasize that NCCL is a highly optimized (open-source) NVIDIA implementation. In particular, it es-
tablishes multiple communication channels and incorporates overlapping computation and communication; these
features greatly improve its performance. Our implementation of CERAR is meant more as a proof of concept and
the implementation has not been optimized anywhere close to the level of NCCL. In addition, we emphasize that
the gap between the methods will typically manifest in the regime when the parameter length is very high. Owing to
2https://anonymous.4open.science/r/CERAR--43C6/
11


---

TABLE II: Training results in W0 for Pythia-1.4B on two four-GPU configurations.
Method
Training time
Throughput
Final val. loss
Peak memory
(hours)
(tokens/s)
(GB)
4× A100 PCIe 40GB (PCIe interconnect): 1,800 steps
NCCL All-Reduce
3.79
8,644
1.4218
36.2
Our algorithm
3.14
10,451
1.4213
38.6
4× A100 SXM 80GB (NVLink interconnect): 7,630 steps
NCCL All-Reduce
4.578
60,682
1.1644904
49.55
Our algorithm
4.745
58,543
1.1644902
51.91
TABLE III: Relative ℓ2 gradient error on four A100 SXM 80GB GPUs, measured against a separate NCCL reference reduction
and maximized across workers.
Method
Step 1
Step 3,815
Step 7,630
NCCL All-Reduce
6.772 × 10−9
5.532 × 10−9
6.437 × 10−9
Our algorithm
0.016078
0.028653
0.027179
access limitations, we were unable to benchmark reduction tensors > 24 billion FP32 entries. However, we believe
that an optimized implementation can yield tangible benefits even in the high-bandwidth NVLink systems.
Training time: We trained a Pythia-1.4B model [22], containing ≈1.41 billion parameters, on the TinyStories
dataset [23]. Both A100 systems used four GPUs with CERAR using c = 12, L = 14. For the PCIe experiment, we
trained for 1800 iterations (≈117.96 × 106 tokens), and for the SXM experiment, we trained for 7630 iterations
(≈1 × 109 tokens); more details in Appendix §V-H.
As shown in Table II, on the PCIe configuration, NCCL requires 3.79 hours, whereas CERAR requires 3.14
hours; the corresponding validation losses at the end of the 1800 iterations are 1.4218 and 1.4213 respectively.
Thus, our algorithm reduces the overall training time by 17.3%, saving approximately 39 minutes. On the SXM
configuration, NCCL and our algorithm require 4.578 and 4.745 hours, respectively. The validation losses of both
schemes are approximately 1.16449. However, our algorithm takes 3.65% longer.
Gradient Error: In the SXM experiment, we measured the relative ℓ2 error of the reduced gradient at every
training step. Table III reports the maximum error across workers at the first, middle, and final steps. For our
algorithm, these errors are approximately 1.61%, 2.87%, and 2.72%, respectively, while those for NCCL are below
7 × 10−9 (consistent with floating-point roundoff). Nevertheless, these errors do not impact the training in an
appreciable manner.
IV. CONCLUSIONS AND FUTURE WORK
This work demonstrates that Ring All-Reduce can be made much more communication-efficient when approximate
recovery of the sum is sufficient and presents theoretical results guaranteeing asymptotic optimality amongst the class
12


---

of linear protocols, and practical constructions that guarantee performance metrics (error and numerical stability).
Future work will involve optimized implementations of our protocol that allow for practical benefits even in high-
bandwidth GPU clusters.
REFERENCES
[1] L. Bottou, F. E. Curtis, and J. Nocedal, “Optimization methods for large-scale machine learning,” SIAM Review, vol. 60, no. 2, pp. 223–311,
May 2018.
[2] M. Langer, Z. He, W. Rahayu, and Y. Xue, “Distributed training of deep learning models: A taxonomic perspective,” IEEE Trans. on
Parall. Distr. Sys., vol. 31, no. 12, pp. 2802–2818, 2020.
[3] P. Patarasuk and X. Yuan, “Bandwidth optimal all-reduce algorithms for clusters of workstations,” Journal of Parallel and Distributed
Computing, vol. 69, no. 2, pp. 117–124, 2009.
[4] A.
Sergeev
and
M.
Del
Balso,
“Horovod:
fast
and
easy
distributed
deep
learning
in
TensorFlow,”
[Online]
Available:
https://arxiv.org/abs/1802.05799, 2018.
[5] M. Li, D. G. Andersen, J. W. Park, A. J. Smola, A. Ahmed, V. Josifovski, J. Long, E. J. Shekita, and B.-Y. Su, “Scaling distributed
machine learning with the parameter server,” in 11th USENIX Symposium on operating systems design and implementation (OSDI 14),
2014, pp. 583–598.
[6] D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, “QSGD: Communication-efficient SGD via gradient quantization and encoding,”
Adv. in Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017.
[7] V. Gandikota, D. Kane, R. K. Maity, and A. Mazumdar, “VQSGD: Vector quantized stochastic gradient descent,” IEEE Trans. on Info.
Th., vol. 68, no. 7, pp. 4573–4587, 2022.
[8] M. Yu, Z. Lin, K. Narra, S. Li, Y. Li, N. S. Kim, A. Schwing, M. Annavaram, and S. Avestimehr, “Gradiveq: Vector quantization for
bandwidth-efficient gradient aggregation in distributed CNN training,” Adv. in Neural Inf. Process. Syst. (NeurIPS), vol. 31, 2018.
[9] D. Alistarh, T. Hoefler, M. Johansson, N. Konstantinov, S. Khirirat, and C. Renggli, “The convergence of sparsified gradient methods,”
Adv. in Neural Inf. Process. Syst. (NeurIPS), vol. 31, 2018.
[10] A. Sahu, A. Dutta, A. M Abdelmoniem, T. Banerjee, M. Canini, and P. Kalnis, “Rethinking gradient sparsification as total error
minimization,” Adv. in Neural Inf. Process. Syst. (NeurIPS), vol. 34, pp. 8133–8146, 2021.
[11] J. Wangni, J. Wang, J. Liu, and T. Zhang, “Gradient sparsification for communication-efficient distributed optimization,” Adv. in Neural
Inf. Process. Syst. (NeurIPS), vol. 31, 2018.
[12] R. Tandon, Q. Lei, A. G. Dimakis, and N. Karampatziakis, “Gradient coding: Avoiding stragglers in distributed learning,” in Intl. Conf.
Mach. Learn. (ICML), August 2017, pp. 3368–3376.
[13] N. Raviv, R. Tandon, A. Dimakis, and I. Tamo, “Gradient coding from cyclic MDS codes and expander graphs,” in Intl. Conf. Mach.
Learn. (ICML), July 2018, pp. 4302–4310.
[14] M. Ye and E. Abbe, “Communication-computation efficient gradient coding,” in Intl. Conf. Mach. Learn. (ICML), 2018, pp. 5610–5619.
[15] A. Ramamoorthy, R. Meng, and V. S. Girimaji, “Leveraging partial stragglers within gradient coding,” in Adv. in Neural Inf. Process. Syst.
(NeurIPS), December 2024.
[16] S.
Munim
and
A.
Ramamoorthy,
“Communication-efficient
approximate
gradient
coding,”
2026.
[Online].
Available:
https:
//arxiv.org/abs/2603.22514
[17] S. Li, M. A. Maddah-Ali, Q. Yu, and A. S. Avestimehr, “A fundamental tradeoff between computation and communication in distributed
computing,” IEEE Trans. on Info. Th., vol. 64, no. 1, pp. 109–128, January 2018.
[18] K. Konstantinidis and A. Ramamoorthy, “Resolvable designs for speeding up distributed computing,” IEEE/ACM Trans. on Networking,
vol. 28, no. 4, pp. 1657–1670, 2020.
[19] Z. Huang, M. Cheng, K. Wan, Q. T. Sun, and Y. Wu, “On the optimality of coded distributed computing for ring networks,” 2026.
[Online]. Available: https://arxiv.org/abs/2507.00091
[20] R. A. Horn and C. R. Johnson, Matrix Analysis.
Cambridge University Press, 2012.
[21] NVIDIA, “NVIDIA Collective Communication Library (NCCL) Documentation,” accessed: September 21, 2026. [Online]. Available:
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
13


---

[22] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff,
A. Skowron, L. Sutawika, and O. Van Der Wal, “Pythia: A suite for analyzing large language models across training and scaling,” in
Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause,
E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202.
PMLR, 23–29 Jul 2023, pp. 2397–2430. [Online].
Available: https://proceedings.mlr.press/v202/biderman23a.html
[23] R. Eldan and Y. Li, “TinyStories: How small can language models be and still speak coherent English?” 2023. [Online]. Available:
https://arxiv.org/abs/2305.07759
14


---

V. APPENDIX
A. Proof that any linear protocol needs to have a normalized rate of at least 1
We say that a protocol is linear if it operates by linear encoding and decoding at the workers. Here, we prove
that any linear protocol that satisfies the requirement in (1) has to have rate at least 1. We proceed by contradiction;
assume that there exists a linear protocol satisfying (1) for all admissible gradients having L˜c < d. In what follows,
we will construct two different gradient vectors such that the input to Wi is the same, but the actual values of the
sum of the gradients are very different.
We note that at the end of the rounds worker Wi reconstructs an approximation to ∇L from the incoming vectors
obtained from Wi−1. Suppose that we stack all the vectors received by Wi in a long vector of length L˜c, denoted
βi and suppose that L˜c < d. Now consider the linear map from the gradients outside Wi (namely gDj, j̸ = i) to
βi, and denote it Γi. This map Γi is from R(N−1)d →RL˜c.
Now, we restrict our attention to gradients where every other worker has the same gradient v ∈Rd and gDi = 0.
This provides a restricted linear map from Rd to RL˜c denoted ˜Γi.
Since L˜c < d, there exists a nonzero vector h ∈Rd such that ˜Γi(h) = 0. Let t > 0. Now, consider two different
sets of gradient vectors: (i) gDj = th for j̸ = i, and (ii) gDj = −th for j̸ = i. Since ˜Γi(h) = 0, worker Wi sees
the same set of input vectors in both cases. Set
t =
CG
√
N −1 ∥h∥2
.
For both sets of inputs


G(0)

2 ≤


G(0)

F =
√
N −1t ∥h∥2 = CG. This means that both sets of gradient
vectors satisfy the bounded spectral norm assumption in (5). However, their corresponding sums differ in norm by
2CG
√
N −1. Thus, Wi cannot recover the gradient within ℓ2-error norm ˜δ (see (1)) that is small enough for both
sets of gradient vectors. This gives the required contradiction.
B. Proof of Lemma 1
Assume, for contradiction, that Φ(U, R) = 0. Then, we have
Mi(U)Ri = T,
for all Wi. We note that the r-th block of Mi(U)Ri is given by Ui+rWN−1−r(Ri) for r = 0, . . . , N −1 (worker
indices reduced modulo-N). This can also be expressed as Ui−1−sWs(Ri) by setting s = N −1−r. This requires
Ui−1−sWs(Ri) = Ic,
s = 0, . . . , N −1.
(17)
Consequently, all Ui, i = 0, . . . , N −1 must be nonsingular. Defining Vi := U−1
i , eq. (17) becomes
Ws(Ri) = Vi−1−s.
(18)
Consecutive windows of the matrices Ri for i = 0, . . . , N −1 overlap in c −1 rows. Taking consecutive values of
s in (18) for all workers, therefore gives, for every k,
Vk[2 : c, :] = Vk−1[1 : c −1, :],
15


---

where the row indices of Vk are one-based.
Equivalently, row by row,
Vk[t + 1, :] = Vk−1[t, :],
t = 1, . . . , c −1.
For any j = 1, . . . , c −N, applying this identity N times yields
Vk[j + N, :] = Vk−1[j + N −1, :]
= · · · = Vk−N[j, :] = Vk[j, :],
where the last equality holds because k ≡(k −N) (mod N). Consequently, every row after the first N repeats an
earlier row with the same index modulo N. Hence
rank(Vk) ≤N < c,
contradicting the nonsingularity of Vk.
C. Error bound of Vandermonde Construction
Let ˜Uτ and ˜Rτ be the matrices obtained by applying the Vandermonde construction. Applying the result of
Proposition 2, we have
0 ≤Φ( ˜Uτ, ˜Rτ) = N(N −1)
2
c
X
j=1
 e−Nτξj −1
2
= N 3(N −1)
2
τ 2
c
X
j=1
ξ2
j + O(τ 3) (when τ is small).
(19)
For fixed ξj, the right-hand side converges to zero quadratically as τ ↓0. Hence infU,R Φ(U, R) = 0. When
c > N, Lemma 1 shows that no finite (U, R) attains this value.
D. Proof of Theorem 2
We first note that since Kℓ(ϵ) →PℓK(ℓ)(B, H) for an appropriate row-permutation Pℓ, we have by the continuity
of singular values and the assumption that minℓσmin(K(ℓ)(B, H)) = γ > 0, there exist ϵ small enough so that
σmin(Kℓ(ϵ)) ≥γ/2 for every ℓ.
From (15),
Aℓ(ϵ) = ΠT
ℓ˜F −1
ℓ
ΛℓKℓ(ϵ).
The matrix Πℓis orthogonal, and ˜Fℓis fixed and invertible. For every singular-value index i, standard product
inequalities [20] give
σmin(Kℓ(ϵ))σi(Λℓ(ϵ)) ≤σi(Λℓ(ϵ)Kℓ(ϵ)) ≤σmax(Kℓ(ϵ)) σi(Λℓ(ϵ)).
Similarly, multiplication on the left by the fixed matrix ˜F −1
ℓ
changes every singular value by at most fixed positive
multiplicative constants:
σmin( ˜F −1
ℓ
) σi(Λℓ(ϵ))σmin(Kℓ(ϵ)) ≤σi( ˜F −1
ℓ
ΛℓKℓ(ϵ)) ≤σmax( ˜F −1
ℓ
) σi(ΛℓKℓ).
Every singular value of Aℓis therefore bounded above and below by positive constants multiplied by the corre-
sponding singular value of Λℓ. The multiplicities follow directly from the characterization of Λℓ.
16


---

E. Finite-difference factorization
For n ≥1, define the n × n lower-triangular finite-difference matrix Fn
(Fn)j+1,k+1 =





(−1)j−k j
k

,
0 ≤k ≤j,
0,
k > j,
where j, k ∈{0, . . . , n −1} and the rows and columns of Fn are numbered 1, . . . , n. The matrix Fn is invertible
and
det Fn = 1.
Lemma 2 (Exact finite-difference identity): For every row vector b, every integer m ≥0, and every j ≥0,
j
X
k=0
(−1)j−k
j
k

bDm+k
ϵ
= ϵjbDm
ϵ Hj.
Proof. Factor bDm
ϵ on the left and use the binomial identity:
j
X
k=0
(−1)j−k
j
k

Dk
ϵ = (Dϵ −Ic)j = ϵjHj.
We show an example of how the finite difference matrix acts below.
Example 1: For n = 3, we have
F3 =


1
0
0
−1
1
0
1
−2
1

.
Thus, we have
F3


a0
a0Dϵ
a0D2
ϵ

=


1
0
0
−1
1
0
1
−2
1




a0
a0Dϵ
a0D2
ϵ


=


a0
ϵa0H
ϵ2a0H2


=


1
0
0
0
ϵ
0
0
0
ϵ2




a0
a0H
a0H2

.
F. Bounds on the singular values of K(ℓ)(B, H)
Let 1s>0 denote the indicator function of the positivity of s and x be a vector with ∥x∥2 = 1. We have



K(ℓ)(B, H)x



2
2 =
β−1
X
j=0


BHjx


2
2 + 1s>0


(BHβ)Eℓx


2
2 ≤σ2
max(B)


r⋆
X
j=0
∥H∥2j

≤C2
1σ2
max(B),
17


---

where the constant C1 = √r⋆+ 1 max(1, ∥H∥r⋆
2 ).
The constant ˜C1 in the upper bound in Corollary 1 is greater than or equal to C1 owing to the multiplicative
factor of the inverse of the finite-difference matrix. Likewise, the constant c0 appears because of the finite-difference
matrix and the limiting K(ℓ)(B, H).
G. Relation between condition number of the Ui’s and Mj matrices
In what follows Mj[i, :] denotes the i-th block-row of Mj (with c rows) for i = 0, . . . , N −1. Based on the
structure of Mj, we note that for ∥x∥2 = 1
∥Mjx∥2
2 =
N−1
X
i=0
∥Mj[i, :]x∥2
2
≤N
max
i=0,...,N−1 ∥Ui∥2
2
= N
max
i=0,...,N−1 σ2
max(Ui).
This means that σmax(Mj) ≤
√
N
p
maxi=0,...,N−1 σ2max(Ui). Moreover, we also have σmax(Mj) ≥maxk σmax(Uk).
This is because every Uk appears as a zero-padded block row within Mj. From Corollary 1, we have that
σmax(Ui) = Θ(ϵ−r⋆). Therefore, for fixed B and H, we have σmax(Mj) = Θ(ϵ−r⋆).
Furthermore, for ∥x∥2 = 1, we have
∥Mjx∥2
2 =
N−1
X
i=0
∥Mj[i, :]x∥2
2
=
N−1
X
i=0
∥Uj+ix[N −1 −i : N + c −2 −i]∥2
2
≥
 N−1
X
i=0
∥x[N −1 −i : N + c −2 −i]∥2
2
!
×

min
k=0,...,N−1 σ2
min(Uk)

≥∥x∥2
2 ×

min
k=0,...,N−1 σ2
min(Uk)

=
min
k=0,...,N−1 σ2
min(Uk).
Thus, we have σmin(Mj) ≥
p
(mink=0,...,N−1 σ2
min(Uk))
From Corollary 1, we have σmin(Ui) = Θ(1). Thus, we can conclude that κ2(Mj) = O(
√
Nϵ−r⋆).
H. Details about numerical experiments
Details about fitting (U, R) parameters for CERAR: The code for running the optimization to determine U
and R is available at this repository.3 The code picks B to be a N × c matrix of i.i.d. N(0, 1) random variables.
Following this, the matrix H is either chosen as a diagonal matrix whose diagonal entries are close to zero, or
chosen in a deterministic manner using the constructed B.
3https://anonymous.4open.science/r/CERAR--43C6/
18


---

Details about All-Reduce Phase time measurement experiments: In our experiments, each GPU is controlled
by one worker process. Before each measured reduction, the worker processes synchronize at a barrier: each waits
until all workers have reached this point. After synchronization, each worker starts its timer with its GPU idle,
executes the reduction, and stops the timer once the reduction has finished on its GPU. Before collecting these
measurements, we perform 10 untimed warmup runs to initialize communication resources and, for our algorithm,
compile the GPU functions. The measured time therefore includes communication and local computation, but
excludes warmup runs, input generation, buffer initialization, synchronization before the timer starts, and result
reporting. We then perform 100 runs measuring only the total reduction time (unprofiled runs) and 100 additional
runs with additional timers measuring the communication and computation phases separately (profiled runs). For
each unprofiled run, we take the maximum elapsed time across all the GPUs; Table I reports the median of these
maximum times across runs.
In the profiled runs, we measure the time spent in our algorithm’s communication and computation phases. For
each GPU, we measure the elapsed time for sending and receiving vectors in each round, including waiting for
these operations to complete. Overlapping sends and receives are timed together, rather than adding their individual
durations. The communication time is the sum of these measurements across all rounds. Computation time includes
encoding, updating received vectors and stored data, preparing outgoing vectors, and decoding, together with the
memory accesses required by these operations. For the profiling results reported in Table IV, we use 1B FP32
entries per GPU on the A100 PCIe system and 24B FP32 entries per GPU on the H200 NVLink system. The
table reports the mean times across the profiled runs, separately for each GPU. For NCCL, we report only the total
profiled reduction time. Communication and computation times may not sum exactly to the total because the total
also includes overhead outside the measured phases.
Using the same profiled runs and tensor sizes as Table IV, Table V further divides our algorithm’s computation
into encoding, outgoing-vector preparation, updates after reception, and decoding. Thus, the A100 PCIe breakdown
corresponds to 1B FP32 entries per GPU, whereas the H200 breakdown corresponds to 24B FP32 entries per GPU.
For the A100 PCIe configuration, outgoing-vector preparation covers the additions after reception in rounds 1–7 that
prepare the sends for rounds 2–8. Updates after reception cover rounds 8–26: storing received vectors for decoding
and, except in the final round, removing the returning local contribution and preparing the next send. For the H200
configuration, the corresponding round ranges are 1–3 for outgoing-vector preparation (preparing sends for rounds
2–4) and 4–14 for updates after reception. The four components sum to the computation time in Table IV, up to
rounding.
The profiling results help explain the different outcomes. On A100 PCIe at 1B entries per GPU, our algorithm
spends approximately 9.29–9.52 seconds per GPU on communication, while its computation takes only 29.1–33.6
ms. On H200 at 24B entries per GPU, our communication phase takes approximately 306.5–306.6 ms, which
is less than NCCL’s complete reduction time of approximately 392.0 ms. However, the additional computation
takes approximately 215.7–215.9 ms, bringing our total reduction time to approximately 522.5–522.6 ms. These
results demonstrate the benefit of our algorithm on the tested low-bandwidth PCIe system, while showing that its
computation cost outweighs the communication savings on the tested high-bandwidth H200 system for our current
19


---

TABLE IV: Profiled reduction times in milliseconds for A100 PCIe (PCIe interconnect; N = 8, c = 20, L = 26, 1B FP32
entries per GPU) and H200 SXM (NVLink interconnect; N = 4, c = 12, L = 14, 24B FP32 entries per GPU). Each entry is
the mean across profiled runs for the indicated GPU. Total is the elapsed time of the complete profiled reduction.
NCCL
Our algorithm
System / interconnect
GPU
Total
Communication
Computation
Total
A100 PCIe, 8 GPUs / PCIe
0
17,927.334
9,509.018
33.613
9,549.329
1
17,924.756
9,517.594
32.108
9,553.995
2
17,919.882
9,520.844
31.662
9,556.541
3
17,923.538
9,519.077
32.479
9,556.751
4
17,927.527
9,519.617
32.251
9,556.561
5
17,927.654
9,357.713
29.141
9,387.553
6
17,927.335
9,289.659
31.173
9,323.855
7
17,927.416
9,478.487
29.150
9,508.829
H200 SXM, 4 GPUs / NVLink
0
392.158
306.509
215.889
522.592
1
392.030
306.503
215.867
522.568
2
391.943
306.459
215.835
522.496
3
391.948
306.591
215.729
522.512
implementation. As discussed in Section III, optimizing our implementation and performing experiments on larger
tensors could further improve performance.
Details about training experiments: Within each configuration, both methods (CERAR and NCCL) use the
same model initialization, data ordering, and training hyperparameters. We use sequences of length 512, gradient
checkpointing, and the AdamW optimizer. The model computation uses a mixture of BF16 and FP32, while model
parameters, stored gradients, and gradient reduction use FP32. The learning rate is linearly increased to 3 × 10−4
and subsequently decayed using a cosine schedule.
For the PCIe experiment, the training set contains 120 million tokens, each GPU processes a batch of 32 sequences,
and the learning-rate warmup lasts 200 steps. We train for 1,800 steps, processing 117.96 million tokens. For the
SXM experiment, the training set contains approximately 474.89 million tokens, the per-GPU batch size is 64, and
the warmup lasts 100 steps. We train for 7,630 steps, processing approximately 1 billion tokens.
20


---

TABLE V: Computation breakdown for our algorithm in milliseconds, using the same profiled runs as Table IV. The A100
PCIe results use a tensor of 1B FP32 entries per GPU, whereas the H200 SXM results use a tensor of 24B FP32 entries per
GPU. Each entry is the mean for the indicated GPU.
System / interconnect
GPU
Encoding
Outgoing-vector
preparation
Updates after
reception
Decoding
A100 PCIe, 8 GPUs / PCIe
0
6.245
4.248
16.197
6.923
1
6.186
4.120
14.907
6.896
2
6.206
3.990
14.594
6.872
3
6.255
4.394
14.955
6.876
4
6.186
4.351
14.842
6.873
5
6.229
3.283
12.752
6.877
6
6.305
3.849
14.087
6.932
7
6.222
3.508
12.544
6.876
H200 SXM, 4 GPUs / NVLink
0
51.390
16.441
93.414
54.645
1
51.382
16.437
93.403
54.645
2
51.366
16.438
93.405
54.626
3
51.386
16.422
93.276
54.645
21
