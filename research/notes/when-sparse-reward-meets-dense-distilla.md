---
title: 'When Sparse Reward Meets Dense Distillation: Training Dynamics of On-Policy
  Distillation'
id: when-sparse-reward-meets-dense-distilla
tags:
- cmp170hx-training-feasibility-a0d4fc
- rlvr
- distillation
- training-stability
- peripheral
created: '2026-10-27T11:25:10.016886Z'
updated: '2026-10-27T19:00:31.486541Z'
source: https://arxiv.org/abs/2609.34849
source_domain: arxiv.org
fetched_at: '2026-10-27T11:25:10.016116Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'When Sparse Reward Meets Dense Distillation (Peking University, arXiv 2609.34849,
  Sep 2026, ~21k words) - NTK analysis of hybrid RLVR plus on-policy teacher-KL training.
  Defines a token-level cross-signal NTK K_DR(n) measuring the alignment between reward
  and distillation gradients at each position, and identifies two failure modes: magnitude
  drowning (the reward gradient exceeds the distillation gradient by orders of magnitude,
  so even weak directional conflict makes the distillation loss rise) and localized
  directional conflict (K_DR(n) < 0, where the sequence-level advantage and the teacher''s
  position-specific distribution push opposite ways). The gradient-norm ratio kappa
  = ||grad L_R|| / ||grad L_D|| is stable within a run but varies by roughly an order
  of magnitude across tasks, with an empirical threshold above which naive scalar
  mixing collapses persistently (Qwen3-1.7B on GSM8K collapses by step 500). Proposes
  M3 (magnitude normalization plus an NTK mask, a soft relaxation, or a fast-slow
  extragradient step) and reports stable training across four backbones and four benchmarks
  where scalar mixing collapses. Relevance to this run is peripheral: it is a post-training
  RL result, not a pretraining-feasibility one. Its only bearing is that any distillation-based
  objective (for example a CSA2 indexer distillation loss) needs explicit gradient-magnitude
  balancing, and that on-policy distillation keeps a teacher and a policy resident
  at once, which is a memory problem on 64 GB cards.'
raw_file: raw/when-sparse-reward-meets-dense-distilla.pdf
doi: arXiv:2609.34849
utility_score: 11.0
citation_count: 0
---

WHEN SPARSE REWARD MEETS DENSE DISTILLA-
TION: TRAINING DYNAMICS OF ON-POLICY DISTIL-
LATION
Xinke Jiang1,2,3∗, Tao Feng∗, Zhibang Yang1,2,3∗, Zhixin Zhang1,2,3,
Weixuan Xu1, Haoyu Zhang, Xu Chu2,3,4†
1National Engineering Research Center of Software Engineering, Peking University, Beijing, China
2School of Computer Science, Peking University, Beijing, China
3Key Laboratory of High Confidence Software Technologies, Ministry of Education, Beijing, China
4Center on Frontiers of Computing Studies, Peking University, Beijing, China
{xinkejiang, yangzb}@stu.pku.edu.cn
ABSTRACT
Reinforcement learning with verifiable rewards provides a sparse post-training
signal: a single binary outcome evaluates the entire rollout, and every token re-
ceives the same sequence-level advantage regardless of its individual contribu-
tion. To complement this sparse supervision, a growing family of methods adds a
scalar-weighted teacher KL term to the policy-gradient objective, providing dense
token-level guidance that may be unreliable at some positions. Despite the benefits
of combining these signals, their interaction during optimization can destabilize
joint training. To understand how this instability develops, we study the learn-
ing dynamics of hybrid reward–distillation training through a neural tangent ker-
nel (NTK) analysis. We introduce the cross-signal NTK KDR(n), a token-level
statistic that measures the alignment between reward and distillation gradients at
position n. Through this analysis, we identify two failure modes: ❶Magnitude
drowning, where the reward gradient exceeds the distillation gradient by orders
of magnitude, so that even weak directional conflict can cause the distillation loss
to rise despite its explicit inclusion in the training objective; and ❷Localized di-
rectional conflict, where the sequence-level advantage and the teacher’s position-
specific distribution induce opposing updates at the same token (KDR(n) < 0).
The severity of these effects depends on the optimization regime: the gradient-
norm ratio κ = ∥∇LR∥/∥∇LD∥varies by roughly an order of magnitude across
tasks, and our experiments reveal an empirical threshold beyond which naive mix-
ing can lead to persistent training collapse. Motivated by these findings, we intro-
duce the M3 family, which combines magnitude normalization with three strate-
gies for coordinating dense teacher supervision and sparse reward updates: a hard
NTK-based mask that retains compatible teacher signals (M3-Select), a contin-
uous relaxation of this mask (M3-Soft), and a fast–slow extragradient step that
temporally separates teacher shaping from reward correction (M3-EG). Experi-
ments across four model backbones and four benchmarks show that M3 maintains
stable training dynamics and achieves superior performance in high-κ regimes
where scalar-mixing baselines collapse.
1
INTRODUCTION
Reinforcement Learning with Verifiable Rewards (RLVR) has become a central approach to post-
training reasoning-capable large language models (DeepSeek-AI, 2025; OpenAI, 2024). However,
its supervision is sparse: a single outcome reward evaluates the entire sequence, and every token
receives the same sequence-level advantage regardless of its individual contribution. Process-reward
∗Equal contribution.
†Corresponding author.
1
arXiv:2609.34849v1  [cs.LG]  28 Sep 2026


---

methods partially address this limitation by evaluating intermediate reasoning steps, but step-level
supervision does not directly distinguish the contributions of individual tokens. As the complemen-
tary, teacher distillation provides dense supervision with target distribution at each token position,
although the teacher’s guidance may be unreliable at some positions. Therefore, a growing family
of hybrid methods therefore augments the policy-gradient objective with a scalar-weighted teacher
KL term (Zhao et al., 2026; Agarwal et al., 2024), aiming to provide token-level guidance.
Despite their empirical success (Zhao et al., 2026; Agarwal et al., 2024), these hybrid methods can
exhibit unstable training dynamics and, in some cases, catastrophic collapse (Figure 1c). The un-
derlying difficulty is that a reliable but sparse outcome signal and a dense but imperfect proxy signal
do not necessarily complement each other. Under naive scalar mixing, they may instead com-
pete: ❶one signal can overwhelm the other, ❷or their opposing updates can cancel, progressively
homogenizing the policy, eliminating reward diversity, and ultimately inducing entropy collapse.
To understand how this instability develops, we study the learning dynamics of hybrid reward–
distillation training through an NTK analysis. Building on the neural tangent kernel (NTK) frame-
work (Ren et al., 2025), we examine how reward and distillation updates interact through the model’s
shared parameters. We introduce the cross-signal NTK KDR(n), the inner product between the pa-
rameter gradients contributed by the two objectives at token position n, which measures their local
alignment while accounting for the mapping from token-level residuals to parameter updates through
the model’s Jacobian. Our analysis identifies two failure modes: ❶Magnitude drowning. The re-
ward gradient can exceed the distillation gradient by orders of magnitude, as measured by the norm
ratio κ = ∥gR∥/∥gD∥. In this regime, even weak negative alignment can make the increase in
distillation loss caused by the reward update exceed the decrease produced by the distillation update
itself. Consequently, the reward or distillation loss can rise despite its explicit inclusion in the train-
ing objective (Figure 1a). ❷Localized directional conflict. The two objectives assign updates using
different information: RL broadcasts a sequence-level advantage to every token, whereas distilla-
tion uses a position-specific teacher distribution. At positions where KDR(n) < 0 (Figure 1b), the
resulting gradient contributions oppose each other. For a positive-advantage rollout, the teacher up-
date can decrease the probability of a sampled token that the reward update seeks to reinforce. For a
negative-advantage rollout, it can instead reinforce a token that the reward update seeks to suppress.
Their contributions can therefore cancel in aggregate diagnostics, obscuring local conflicts. Over
longer training horizons, repeated conflicting updates can reduce diversity among rollouts and di-
minish within-group reward variation. When all rollouts in a group receive the conflict reward, their
relative advantages and policy gradient vanish. Our experiments exhibit a corresponding progression
from initial reward improvement to reduced reward diversity and abrupt collapse (Figure 1c).
The severity of these failure modes depends on the optimization regime. A hybrid run may ini-
tially appear stable, with reward increasing even as the distillation loss rises. Across architectures
and tasks, κ varies by roughly an order of magnitude, and our experiments reveal an empirical
threshold beyond which naive mixing becomes prone to persistent collapse. These findings motivate
examining teacher supervision at two levels. At the run level, κ indicates the degree of magnitude
imbalance and helps distinguish settings where standard scalar mixing remains effective from those
where it collapses. At the token level, the sign of KDR(n) distinguishes locally compatible teacher
updates from conflicting ones. Guided by this diagnosis, we introduce the M3 family, which com-
bines magnitude normalization with three strategies for coordinating teacher supervision and reward
updates: M3-Select applies a hard NTK-based mask, retaining teacher supervision only at positions
where KDR(n) ≥0; M3-Soft replaces this mask with a continuous, temperature-controlled gate;
and M3-EG temporally separates teacher shaping from reward correction through a fast–slow ex-
tragradient step. We make the following contributions:
• We develop an NTK-based framework that characterizes the interaction between RL and distilla-
tion through a single per-token statistic, the cross-signal NTK KDR(n). This framework identifies
two failure modes of linear mixing and explains why weight-space rebalancing methods.
• We show that κ remains stable throughout a run but varies by roughly an order of magnitude
across tasks. A critical threshold separates a stable regime, in which naive mixing is effective,
from a catastrophic regime where it collapses. This yields a simple probe-batch rule for selecting
the mixing strategy before full training.
• In high-κ regimes, the M3 family remains stable over horizons at which scalar-mixing baselines
collapse. M3-Select is the most robust variant in the highest-κ settings, while M3-Soft recov-
2


---

0
100
200
300
400
500
Training step
0
1
2
3
4
5
Distillation loss / initial
initial level
teacher loss drifts ↑
while being "minimized"
(a) Pitfall I: magnitude drowning
OPSD
OPSD+GradNorm
M3-Select
0
25
50
75
100
125
150
175
200
Token position n (one correct rollout, A>0)
−10
−1
−10
−4
0
10
−4
10
−1
KDR(n)  (symlog)
42% of positions conflicting
(b) Pitfall II: localized directional conflict
aligned with reward (KDR ≥0)
opposes reward (KDR <0)
0
100
200
300
400
500
Training step
0.0
0.2
0.4
0.6
0.8
1.0
Mean reward (15-step MA, schematic)
Hybrid collapse ×
@ step 221
GRPO collapse ×
@ step 412
M3 peak 0.97
M3 never collapses,
noisy upward trend
(c) Consequence: long-horizon collapse (schematic)
GRPO
Hybrid (OPSD)
M3-Select
M3-Soft (β=1)
Figure 1: Two failures of linear reward–distillation mixing (Qwen3-1.7B, GSM8K). (a) Distil-
lation loss rises under OPSD, indicating magnitude drowning. (b) Positive and negative KDR(n)
interleave in a correct rollout, revealing token-local conflict hidden by aggregation. (c) Baselines
collapse by step 500, while M3-Select and M3-Soft remain stable.
ers catastrophic cases with order-of-magnitude gains. Combined with weight averaging, which
removes the gate-variance pathology identified in our analysis, M3-Soft matches or exceeds the
strongest baseline across all tested architecture–dataset pairs.
2
RELATED WORK
❶LLM Reasoning via RL and Distillation. DeepSeek-R1 (DeepSeek-AI, 2025) spurred RL reason-
ing (GRPO (Shao et al., 2024), DAPO (Yu et al., 2025), Dr. GRPO (Liu et al., 2025)); OPSD (Zhao
et al., 2026) and MiniLLM (Gu et al., 2024) make distillation on-policy. Hybrids differ in cou-
pling: SDPO (H¨ubotter et al., 2026) self-distills from a reprompt, RLSD (Yang et al., 2026) keeps
the teacher as magnitude-only reweighting, and HDPO (Ding, 2026) and DPKD (Li et al., 2024)
interpolate losses. Unlike these methods, we study the dynamics of the coupling, identifying when
dense teacher supervision conflicts with or is drowned by sparse reward updates.
❷Multi-Objective Gradients and NTK. PCGrad (Yu et al., 2020), CAGrad (Liu et al., 2021),
MGDA (Sener & Koltun, 2018), and NashMTL (Navon et al., 2022) operate on aggregate task
gradients; GradNorm (Chen et al., 2018), UW (Kendall et al., 2018), and DWA (Liu et al., 2019)
balance objectives in loss-weight space. Such global operations do not directly resolve conflicts
that alternate across tokens, and we show that GradNorm becomes ineffective at κ ≫1 (Proposi-
tion 5). Prior work applies NTK to gradient conflict and imbalance (Ren et al., 2025; Qin et al.,
2025); our cross-signal NTK specializes this perspective to hybrid RL–distillation and links token-
level interaction to drowning-induced collapse, complementing known RL failures such as reward
hacking (Skalse et al., 2022), entropy collapse (Yu et al., 2025), and length bias (Liu et al., 2025).
3
PRELIMINARIES AND THEORETICAL ANALYSIS
Notation.
In this paper, scalars use italic symbols, vectors bold lowercase or Greek symbols (e.g.,
g, δ, θ), and matrices bold uppercase symbols (e.g., J, K). Calligraphic symbols denote sets and
losses; R denotes real numbers. The model has dpar parameters, collected in θ ∈Rdpar. At position
n, pn
S is the student’s token distribution and pn
S(v) = [pn
S]v the probability of token v. We distinguish
kernel matrix K(n, m) from the scalar interaction score KDR(n) and use φn for alignment angles.
3.1
PROBLEM SETUP AND NTK PRELIMINARIES
We train student model πθ using teacher predicts and response-level rewards. For a prompt–answer
pair (x, y∗) ∼D, the student generates a response ˆy of length N; ˆy<n is its prefix before position n.
❶On-Policy Self-Distillation (OPSD). OPSD (Zhao et al., 2026) learns from responses sampled by
the student itself. A frozen teacher receives the ground-truth answer as additional context, giving
pn
T = πθ0(· | x, y∗, ˆy<n), while pn
S = πθ(· | x, ˆy<n). Their distributions are compared at each
position, with each contribution capped at τ:
LD(θ) = E(x,y∗)∼D Eˆy∼πθ(· | x)
"
1
N
N
X
n=1
min

Dλ
 pn
T


 pn
S

, τ
 #
,
(1)
3


---

Here, Dλ(p∥q) = λKL(p∥m) + (1 −λ)KL(q∥m) is the generalized Jensen–Shannon divergence,
with m = λp + (1 −λ)q. We use λ = 1/2, giving both distributions equal weight. The teacher
thus provides dense and position-specific supervision; KL limits are given in Appendix AG.1.
❷Group Relative Policy Optimization (GRPO). GRPO (Shao et al., 2024), a critic-free variant of
PPO (Schulman et al., 2017), samples G responses per prompt and assigns each a verifiable reward
r(i) = r(ˆy(i), y∗) ∈{0, 1}. Its advantage Ai = (r(i) −¯r)/(σr + ϵ) compares that reward with the
group mean ¯r and standard deviation σr, where ϵ > 0 prevents division by zero:
LR(θ) = −E(x,y∗)∼D E{ˆy(i)}G
i=1∼πθ(· | x)
"
1
G
G
X
i=1
Ai
1
Ni
Ni
X
n=1
log πθ

ˆy(i)
n
 x, ˆy(i)
<n
#
,
(2)
where Ni = |ˆy(i)|. Positive advantages encourage sampled responses and negative advantages
discourage them. The same advantage weights every token, without identifying which tokens caused
success or failure. Gradients hold the sampled responses and advantages fixed.
❸NTK in Learning Dynamics. Following (Ren et al., 2025), let zn ∈R|V| be the policy’s logits,
the scores converted by softmax into probabilities over vocabulary V. The Jacobian Jn = ∇θzn ∈
Rdpar×|V| describes their dependence on the parameters. For L = N −1 P
n ℓn, the chain rule gives
∇θL = 1
N
X
n
Jnδn,
δn = ∇znℓn,
K(n, m) = (Jn)⊤Jm.
(3)
The residual δn describes how the loss changes with each logit. The empirical NTK K(n, m) (Jacot
et al., 2018) couples positions through their shared parameters: an update driven by position m can
change predictions at n. Following Ren et al. (2025), we use the Action–Kernel–Gradient (AKG)
decomposition to analyze these changes.
3.2
HYBRID UPDATE DYNAMICS AND LOSS INTERACTIONS
In OPSD, we assume the hybrid loss between policy update and distillation loss is LH = (1 −
α)LR + αLD, where α ∈[0, 1] weights distillation. A step of size η gives θt+1 = θt −η∇θLH.
Expanding the logits to first order and summing over T steps yields:
zn
T = zn
0 +
T −1
X
t=0
∆zn
t ,
∆zn
t = −η
N
N
X
m=1
Kt(n, m)
h
(1 −α) δm
R + α δm
D
i
+ O(η2),
(4)
where zn
t = zn(θt) and ∆zn
t = zn
t+1 −zn
t . At each step, the kernel maps the combined residual at
every position m to a change in the logits at n. The residuals are evaluated at step t:
δm
R = −A(eˆym −pm
S ),
δm
D = pm
S −pm
T .
(5)
The one-hot vector eˆym selects the sampled token.
The reward residual encourages or dis-
courages this token according to A; the teacher residual compares the full distributions.
We
use a local forward-KL model of distillation: near agreement, the unclipped JSD gradient is
λ(1 −λ)δm
D + O(∥pm
S −pm
T ∥2), with the leading constant absorbed into the teacher scale. Clipped
tokens contribute zero gradient (Appendix AG.1).
First-Order Loss Changes. With gR = ∇θLR and gD = ∇θLD, the same update gives
∆LR ≈−η
h
(1 −α)∥gR∥2 + α⟨gR, gD⟩
i
,
∆LD ≈−η
h
α∥gD∥2 + (1 −α)⟨gD, gR⟩
i
, (6)
Each squared-gradient term describes an objective’s own decrease; the inner product describes the
other update’s effect. Positive alignment helps both objectives, while negative alignment opposes
their progress. A loss increases only when this opposing contribution exceeds its own decrease
within the first-order approximation.
3.3
TOKEN-LEVEL DECOMPOSITION OF GRADIENT INTERACTIONS
Cross-Position Interactions. The overall inner product can hide local conflicts. Each term pairs
the teacher signal at n with the reward signal at m through the shared kernel. The sum includes
same-position and cross-position interactions, whose positive and negative contributions can cancel.
4


---

Expanding both gradients gives that:
⟨gD, gR⟩=
1
N 2
X
n,m
(δn
D)⊤
| {z }
teacher residual
K(n, m)
|
{z
}
kernel coupling
δm
R
|{z}
reward residual
.
(7)
Gradient Magnitude and Alignment. Expanding the squared norms in the same way gives the
magnitude ratio when ∥gD∥> 0:
κ2 = ∥gR∥2
∥gD∥2 =
P
n,m(δn
R)⊤K(n, m) δm
R
P
n,m(δn
D)⊤K(n, m) δm
D
,
(8)
the numerator measures reward-gradient strength and the denominator teacher-gradient strength;
κ ≫1 indicates strong imbalance. For nonzero gradients, the normalized inner product ΦDR =
⟨gD, gR⟩/(∥gD∥∥gR∥) ∈[−1, 1] measures direction independently of magnitude. Positive values
indicate alignment and negative values interference. These quantities arise from one gradient Gram
matrix: its diagonal contains squared norms and its off-diagonal contains the cross inner product
(Appendix AG.2).
3.4
TOKEN-LEVEL CONFLICT AND MAGNITUDE DROWNING
We first locate conflict at individual positions, then examine how gradient imbalance amplifies its
effect on the teacher loss.
Definition 1 (Cross-Signal Token-Level NTK). Using the local residual model of Section 3.2, define
KDR(n) = ⟨Jnδn
D, Jnδn
R⟩,
δn
D = pn
S −pn
T ,
δn
R = −A (eˆyn −pn
S) .
(9)
where A = (r −¯r)/(σr + ϵ). This scalar compares the teacher and reward parameter gradients
contributed by the same position.
Positive, negative, and zero scores define the sets Ω+, Ω−, and Ω0. A zero score also includes
vanishing gradients. KDR(n) is the m = n summand in Eq. 7, before the common factor N −2; it
does not include cross-position interactions.
❶Token-Level Conflict. On a positive-advantage response, a negative KDR(n) means that the
teacher’s same-position contribution lowers the sampled token’s log-probability. The full update
need not do so, since reward and other-position contributions also matter. Corollary 1 derives this
local effect. The conflict rate CNTK = |{n : KDR(n) < 0}|/N is approximately 40% in our
measurements, an empirical observation rather than a consequence of the definition.
❷Magnitude Drowning. Write cos φ = ΦDR. Equation 6 gives ∆LD ≈−η∥gD∥2[α + (1 −
α)κ cos φ]. For nonzero gradients and 0 < α < 1, its first-order sign condition is ∆LD > 0 ⇐⇒
cos φ < −
α
(1−α)κ. For fixed α, the threshold approaches zero from below as O(κ−1): when the
reward gradient is large, even weak negative alignment can outweigh the teacher’s own descent.
Imbalance alone is insufficient; negative alignment must also satisfy the threshold.
As the student approaches the teacher distribution, δn
D shrinks, while the reward residual need not
shrink at the same rate. The kernel maps both into parameter space, so their magnitudes and direc-
tions jointly determine κ. Our experiments identify magnitude imbalance as the dominant failure
mode, with ratios reaching 1.3 × 104. Ratios are approximately stable across the tested LoRA ranks
8–64 but vary substantially across tasks (Proposition 7). Extended scaling analysis and projection-
baseline limitations appear in Appendices AG.2 and M.
3.5
REWARD-SIGNAL DEGENERATION AND TRAINING COLLAPSE
Repeated teacher contributions at conflicting positions may suppress successful response patterns.
If the resulting responses become less diverse in reward, GRPO receives less information for distin-
guishing them. This is a possible training mechanism, not a direct consequence of the local gradient
identity. The final step is exact: if all G sampled responses receive the same reward, then r(i) = ¯r,
σr = 0, and every advantage Ai = 0. All reward residuals vanish, so that group contributes no
reward gradient, g(x)
R
= 0. Only the teacher term can contribute to its hybrid update. However,
one such group does not establish collapse: it may contain all successes or all failures, and later
sampling or updates from other prompts may restore reward variation. Persistent failure requires
5


---

Privileged solution  y*
Two Signals on the Same On-Policy Rollout
The cat sat 
on the mat.
Dense teacher signal
per-token guidance gD
The
cat
sat
on
the
mat
…
Student rollout (same for both signals)
Frozen 
Teacher 𝜋!∗
（no update）
Question  x
What is the 
next token ?
Student 
Policy 𝜋!
（on-policy）
Reward Model
r (x,y)
Sparse reward signal
sequence-level advantage gR
Naive scalar mix 𝛼𝑔! + 1 −𝛼𝑔"
NKT Diagnosis: Why One Scalar Cannot Work
#1
#2
A. Token-level conflict
B. Magnitude drowning
|| gD ||
（teacher）
||gR||
（reward）
𝜅= ||𝑔!||
||𝑔"|| ≫1
Teacher signal is erased
The
cat
sat
the
mat
…
on
gD
gR
KDR(n) ≥ 0
synergy 
KDR(n) < 0
conflict 
Magnitude decides HOW STRONGLY; KDR(n) decides WHERE.
M3：Normalize, Then Gate
#3
A. Normalize magnitudes
Equalized magnitude, 
comparable scale
B. Gate the teacher per token
Reward signal continues for all tokens
𝑔!ˆ
𝑔"ˆ
Pass teacher 
guidance
Block at 
conflict
M3-Select
M3-Soft
M3-EG
hard gate: 
II 𝐾!" 𝑛≥0
soft gate: 
𝜎𝛽𝑐𝑜𝑠𝜃#
Reward-compatible hybrid update
Update Policy 𝜋!"
…
…
…
…
Passed
Passed
Blocked
Weak
Strong
Strong
Figure 2: From interaction analysis to hybrid updates. Shared parameters couple dense teacher
and sparse reward supervision. M3 controls residual scale and allocates teacher influence by token-
level compatibility; its extragradient extension separates teacher shaping from reward correction.
poor responses to remain dominant without recovery of a useful reward signal. The abrupt drops in
Figure 1c are consistent with this mechanism; Section 5.2 provides empirical tests.
Remark 1 (Empirical Gate-Selection Threshold). Our configurations show a transition near κ∗≈
5 × 103: hard masking is most useful at high imbalance, while soft gating often retains more useful
supervision at lower imbalance. This empirical guideline is not a universal collapse threshold; it
depends on the compatible-token fraction, conflict strength, and gate-estimation error. Section 4
introduces the gating methods.
4
METHODOLOGY
M3 turns the preceding analysis into two coupled decisions: how strongly each signal enters the
update, and where teacher guidance is admitted. We first normalize token residuals, then allocate
teacher weight using local compatibility (Figure 2). M3-Select and M3-Soft implement this alloca-
tion with hard and continuous gates. M3-EG extends the same principle to update timing, letting the
teacher shape where the reward direction is evaluated.
4.1
CONTROLLING SIGNAL SCALE
In a global mixture, comparable weighted gradient norms require α ≈κ/(1+κ); the coefficient must
compensate for scale before it can express a preference between the signals. M3 instead controls
scale at the residual level, before conversion into a parameter update. Using the residual convention
of Section 3.2, define
bδn
q =
δn
q
∥δnq ∥+ ϵ,
hn
q = Jnbδn
q ,
q ∈{D, R}.
(10)
Here ϵ > 0 stabilizes division and keeps zero residuals zero. Normalization reduces the dependence
of mixing on raw residual magnitudes. Its positive scaling preserves the sign of KDR(n), but does
not equalize parameter-gradient norms after multiplication by Jn. It therefore controls residual
scale while leaving the directional coordination problem to the gate. The teacher budget acts on
these rescaled contributions, while a common update scale controls their overall size. This separates
the intended allocation of supervision from raw magnitude disparity that constrains naive mixing.
4.2
ALLOCATING TEACHER INFLUENCE ACROSS POSITIONS
With a token-dependent teacher weight αn ∈[0, αmax] and αmax < 1, the common update:
hn
H =
αnhn
D
| {z }
teacher contribution
+ (1 −αn)hn
R
|
{z
}
reward contribution
,
gH = 1
N
N
X
n=1
hn
H,
θt+1 = θt −η¯s gH.
(11)
The shared exponential-moving-average scale ¯s sets the update magnitude; αn controls its local
composition. Teacher and reward weights vary together: rejecting the teacher restores reward weight
to one. The remaining choice is how compatibility determines αn.
6


---

M3-Select: retain compatible teacher contributions.
The hard gate directly implements the sign
criterion:
α∗
n = αmaxI{KDR(n) ≥0}.
(12)
This removes negative teacher projections onto the same-position reward direction. Since normal-
ization preserves signs, the resulting contribution satisfies
⟨hn
H, hn
R⟩= (1 −α∗
n)∥hn
R∥2 + α∗
n⟨hn
D, hn
R⟩≥(1 −αmax)∥hn
R∥2.
(13)
Rejected positions retain the full local reward contribution. This is a same-position guarantee: cross-
position interactions in the aggregate update remain. A zero score is admitted by convention and
may simply reflect a vanishing signal. The gate ceiling controls how much teacher influence an
admitted position receives; the score’s sign controls whether it receives that influence at all. These
two choices need not be tied to one global loss weight.
M3-Soft: vary teacher influence continuously.
Hard decisions can change abruptly near zero
alignment. M3-Soft smooths this transition, retaining partial supervision when compatibility is weak
or uncertain:
αn = αmaxσ(β cos φn),
cos φn =
KDR(n)
∥gn
D∥∥gn
R∥+ ε.
(14)
Here σ is the logistic sigmoid, β ≥0 controls sharpness, and ε > 0 stabilizes the score. Larger
β approaches hard selection away from zero; at zero, the weight remains αmax/2. Soft gating also
retains some negatively aligned teacher contributions, trading strict local exclusion for smoother
supervision. It therefore does not inherit Eq. 13 (Appendix AG.4). At β = 0, all positions receive
the same teacher weight αmax/2; increasing sharpness progressively makes allocation depend on
compatibility. Both variants thus share the same normalized update, with their distinction confined
to the teacher-allocation rule.
4.3
COORDINATING THE SIGNALS IN TIME
The synchronous variants combine directions evaluated at the same parameters. M3-EG instead uses
the gated teacher field mD(θ) = N −1 P
n α∗
nhn
D to construct a temporary point, then evaluates the
normalized reward field hR(θ) = N −1 P
n hn
R there:
eθ = θ −ηinmD(θ),
teacher look-ahead,
θ+ = θ −ηouthR(eθ),
reward correction.
(15)
The reward correction is applied from the original parameters, after restoring them; no gradient is
propagated through the temporary step. Thus teacher guidance changes the reward evaluation point
rather than entering the committed update additively. Appendix E.1 gives the procedure and the
conditions on the inner displacement and outer step for local reward descent.
4.4
TRAINING RULE AND SCOPE
Each iteration samples student responses, obtains teacher distributions and verifier advantages, and
forms the two residual fields. Their compatibility scores determine the gates; normalized, gated
contributions then define either the synchronous direction or the extragradient step (Algorithms 2
and 1). Gates and normalization factors specify update coefficients and are held fixed when applying
the direction. Clipped teacher terms and zero-advantage reward terms contribute zero; normalization
does not recreate missing supervision.
The local projection bound directly supports the hard selection rule. Aggregate stationarity and
conditional variance bounds require a more restrictive population model: smooth lower-bounded
reward loss, orthogonal position subspaces, matched teacher and mean reward norms, deterministic
conditional teacher directions, and gates fixed before fresh reward noise. These conditions are not
enforced by residual normalization. Theorem 1 and Proposition 2 state the resulting guarantees; the
experiments assess behavior beyond that model.
7


---

0
20
40
60
80
100
Step
0.0
0.2
0.4
0.6
Mean reward
(a) Reward
Pure GRPO
Hybrid (®=0.5)
GradNorm
0
20
40
60
80
100
Step
-0.4
-0.2
0.0
0.2
0.4
cos(gD; gR)
(b) Gradient alignment
Hybrid
GradNorm
0
20
40
60
80
100
Step
0.0
0.2
0.4
0.6
0.8
1.0
GradNorm weight
throttling equilibrium w¤
D = ∙=(1+∙)
(c) GradNorm wD trajectory
wD (distill)
wR (reward)
Figure 3: GradNorm challenge on InternLM2.5-1.8B GSM8K: training reward (left), gradient cosine
(center), and learned distillation weight (right).
Table 1: Cell reports accuracy (top) and its percentage-point change from GRPO (bottom; ↑/ ↓);
†Hard masking is vacuous for single-token ARC.
Qwen3-1.7B
Qwen2.5-1.5B
InternLM2.5-1.8B
Llama-3.2-1B
Method
GSM8K SVAMP
ARC
GSM8K
SVAMP
ARC
GSM8K SVAMP
ARC
GSM8K
SVAMP
ARC
Reference: pure RL
GRPO
0.696
0.0
0.940
0.0
0.732
0.0
0.726
0.0
0.825
0.0
0.689
0.0
0.404
0.0
0.670
0.0
0.599
0.0
0.552
0.0
0.730
0.0
0.533
0.0
Teacher-augmented baselines
Hybrid (OPSD)
0.786
↑+9.0
0.932
↓-0.8
0.796
↑+6.4
0.717
↓-0.9
0.857
↑+3.2
0.693
↑+0.4
0.492
↑+8.8
0.608
↓-6.2
0.604
↑+0.5
0.563
↑+1.1
0.722
↓-0.8
0.544
↑+1.1
OPSD+GradNorm
0.765
↑+6.9
0.917
↓-2.3
0.752
↑+2.0
0.735
↑+0.9
0.837
↑+1.2
0.695
↑+0.6
0.459
↑+5.5
0.687
↑+1.7
0.590
↓-0.9
0.547
↓-0.5
0.717
↓-1.3
0.370
↓-16.3
RLSD
0.8317
↑+13.6
0.923
↓-1.7
0.748
↑+1.6
0.725
↓-0.1
0.860
↑+3.5
0.684
↓-0.5
0.471
↑+6.7
0.737
↑+6.7
0.602
↑+0.3
0.568
↑+1.6
0.648
↓-8.2
0.545
↑+1.2
SDPO
0.587
↓-10.9
0.567
↓-37.3
0.742
↑+1.0
0.085
↓-64.1
0.612
↓-21.3
0.424
↓-26.5
0.316
↓-8.8
0.432
↓-23.8
0.586
↓-1.3
0.220
↓-33.2
0.483
↓-24.7
0.432
↓-10.1
Ours: boundary-gated mixing (M3)
M3-Select†
0.803
↑+10.7
0.943
↑+0.3
0.275
↓-45.7
0.459
↓-26.7
0.585
↓-24.0
0.571
↓-11.8
0.394
↓-1.0
0.538
↓-13.2
0.599
0.0
0.516
↓-3.6
0.673
↓-5.7
0.459
↓-7.4
M3-Soft
0.8324
↑+13.6
0.945
↑+0.5
0.807
↑+7.5
0.752
↑+2.6
0.905
↑+8.0
0.736
↑+4.7
0.501
↑+9.7
0.777
↑+10.7
0.608
↑+0.9
0.568
↑+1.6
0.750
↑+2.0
0.556
↑+2.3
M3-EG
0.8446
↑+14.9
0.952
↑+1.2
0.731
↓-0.1
0.749
↑+2.3
0.868
↑+4.3
0.692
↑+0.3
0.440
↑+3.6
0.468
↓-20.2
0.6135
↑+1.5
0.552
0.0
0.758
↑+2.8
0.534
↑+0.1
M3-Soft ≥best non-M3 baseline in 12/12 cells (11 strict wins, 1 exact tie).
5
EXPERIMENTS
5.1
SETUP
Our main evaluation covers Qwen3-1.7B, Qwen2.5-1.5B, InternLM2.5-1.8B, and Llama-3.2-1B on
GSM8K, SVAMP, and ARC-Challenge, with Qwen3-0.6B and MATH added in supplementary long-
horizon runs. We define ¯κ as the mean of κt over RL-active steps of the corresponding naive-Hybrid
run. Full experimental details are provided in Appendix A.
5.2
DOES MAGNITUDE DROWNING ACTUALLY OCCUR?
Figure 1a shows loss inversion under scalar mixing. With ¯κ ranging from 1.4×103 to 1.3×104, the
reward update overwhelms distillation; panel c shows the resulting long-horizon collapse. We next
test whether global norm balancing fixes it.
A controlled challenge.
GradNorm trails both GRPO and naive Hybrid in training reward (Fig-
ure 3, left), despite a near-zero aggregate gradient cosine (center). Its distillation weight rapidly
approaches w∗
D = κ/(1+κ) ≈1 (right), leaving the effective reward weight at only 1−w∗
D ≈1/κ.
GradNorm therefore achieves global norm balance only by nearly turning off RL; because the same
weight is applied to every token, it also cannot separate compatible from conflicting teacher updates.
This motivates M3’s position-dependent gate; further diagnostics are reported in Appendix B.4.
5.3
CAN TOKEN-LEVEL GATING BEAT GLOBAL MIXING?
Accuracy across architectures.
Across four architectures and three datasets (Table 1), M3-Soft
matches or exceeds the strongest non-M3 baseline in all 12 cells under the per-cell best-observed
8


---

Table 2: Long-horizon reward on GSM8K after 500 steps (last-50 mean). ↓denotes collapse.
Reference and adaptive baselines
Ours: boundary-gated mixing
Architecture
GRPO
Hybrid
GradNorm
M3-Select
M3-Soft
Qwen3-1.7B
↓0.003
↓0.000
↓0.000
0.809
0.866
Qwen3-0.6B
↓0.003
↓0.000
↓0.053
0.644
0.673
Llama-1B
↓0.004
↓0.003
↓0.328
0.497
0.485
LLaMA-3.2
Qwen3
Qwen2.5
InternLM2.5
Architecture
103
104
Mean gradient ratio ¹∙
7.7k
3.4k
2.5k
2.7k
1.6k
2.5k
2.2k
1.4k
12.7k
5.1k
2.2k
2.7k
catastrophic threshold ∙⋆= 5;000
GSM8K
ARC
SVAMP
Figure 4: Mean gradient-magnitude ratio ¯κ for
12 architecture–dataset pairs under naive Hybrid;
dashed line: empirical threshold κ∗= 5,000.
0
100
200
300
400
500
Training step
103
104
105
Gradient ratio ∙= kgRk=kgDk
∙⋆= 5;000
catastrophic regime
Hybrid diverges above ∙⋆,
collapses @221
OPSD+GradNorm (collapses @358)
Hybrid, naive (collapses @221)
Figure 5: Runtime magnitude ratio. Hybrid and
GradNorm cross κ∗= 5,000 before collapsing at
steps 221 and 358.
protocol (11 strict wins, one tie). The largest gains occur on Qwen2.5-SVAMP (0.905, +4.5pp over
RLSD), Qwen2.5-ARC (0.736, +4.1pp over OPSD+GradNorm), and InternLM-SVAMP (0.777,
+4.0pp over RLSD). On the two Llama arithmetic cells, M3-Soft ties RLSD on GSM8K (0.568)
and improves over GRPO on SVAMP (0.750, +2.0pp). The Qwen3 margins are tighter: 0.8324 on
GSM8K (+0.07pp over RLSD) and 0.945 on SVAMP (+0.5pp over GRPO).
Long-horizon stability.
Table 2 compares last-50 training reward after 500 steps on GSM8K.
Across all three architectures, every reference or globally balanced baseline triggers the collapse
criterion, whereas M3-Select and M3-Soft remain non-collapsed. M3-Soft attains the highest final
reward on both Qwen3 scales, while M3-Select is marginally higher on Llama-3.2-1B. Detailed
phase-wise trajectories and collapse times for Qwen3-1.7B are reported in Table 8 of Appendix B.2.
Regime Dependence of Token-Level Gating.
Across the 12 architecture–dataset cells, ¯κ ranges
from 1,423 to 12,705 (Figure 4). Figure 5 provides the within-run counterpart: on Qwen3-1.7B
GSM8K, Hybrid crosses the empirical threshold κ∗= 5,000 near step 200 before collapsing
at step 221, while GradNorm crosses later and collapses at step 358. In the two high-κ Llama
arithmetic cells (7,690/12,705), Hybrid’s reference-rate training reward falls to 0.031/0.000 on
GSM8K/SVAMP, whereas M3-Soft reaches 0.647/0.822 (Appendix B.1). At lower κ, hard mask-
ing can be overly restrictive: on Qwen3-ARC (¯κ = 2,465), M3-Select reaches 0.275 accuracy,
compared with 0.807 for M3-Soft (Table 1). Thus, ¯κ indicates the severity of magnitude imbalance
and the appropriate gate strength, rather than directly predicting accuracy. Together, these results
support κ as a regime indicator for instability and appropriate gate strength, rather than a predictor
of absolute accuracy; exact per-cell ¯κ values and reference-rate rewards are in Appendix B.1.
Local Consequence of Token-Level Conflict.
We test the sign-specific prediction on 30 correct
Qwen3-1.7B GSM8K trajectories comprising 8,492 tokens. Tokens are partitioned by the pre-update
cross-signal NTK, after which we apply one OPSD, GRPO, or hybrid update. On compatible posi-
tions, the hybrid update increases the sampled-token log-probability by 2.0 × 10−3; on conflicting
positions, it decreases it by 2.4 × 10−4 (Table 3). Over the same conflicting subset, the pure OPSD
Table 3: One-step sampled-token log-probability change by pre-update conflict region (30 trajecto-
ries, 8,492 tokens). Negative values indicate suppression by the update.
Region
OPSD
GRPO
Hybrid
Compatible (KDR > ϵ, n=851)
+1.6 × 10−3
+2.7 × 10−3
+2.0 × 10−3
Neutral (|KDR| ≤ϵ, n=5191)
≈0
≈0
≈0
Conflicting (KDR < −ϵ, n=2450)
+0.3 × 10−4
+7.3 × 10−4
−2.4 × 10−4
9


---

and GRPO updates yield positive changes. Because the partition is computed before the update, the
result tests the local sign prediction rather than defining conflict from the observed logit change.
6
CONCLUSION AND FUTURE WORK
We study when dense teacher supervision can complement sparse verifiable rewards in reasoning-
model post-training. Our NTK analysis separates their interaction into token-level compatibility,
captured by the cross-signal NTK KDR(n), and scale imbalance, captured by κ, exposing localized
directional conflict and magnitude drowning. This diagnosis leads to the M3 family, which combines
magnitude normalization with hard, soft, or temporally decoupled compatibility gating. Across four
model families and three reasoning benchmarks, M3-Soft matches or exceeds the strongest non-M3
baseline, while M3 variants remain stable over 500-step GSM8K training where non-gated baselines
collapse. In the future, we plan to scale this application to support larger and more complex agentic
scenarios such as coding and deep research.
AI USE STATEMENT
In this work, large language models (LLMs) were used for language polishing, figure design assis-
tance, coding support, and mathematical proof assistance. Specifically, LLMs were used to improve
the clarity, grammar, and readability of the manuscript, refine its stylistic quality, and suggest alter-
native phrasings to reduce redundancy. They also provided suggestions for figure design, visualiza-
tion layouts, and graphical presentation; all final figures were created, verified, and curated by the
authors using the authors’ experimental data. In addition, LLMs assisted with code development,
intermediate mathematical derivations, proof construction, and consistency checking. All assump-
tions, formal statements, derivations, proofs, code, figures, and other LLM-assisted content were
independently reviewed, verified, and revised by the authors before inclusion. The research ques-
tions, core methodology, scientific contributions, experimental design, critical analyses, and final
decisions were independently determined by the authors.
ETHICS STATEMENT
All experiments in this work were carried out using publicly available language models and standard
reasoning benchmarks, including GSM8K, MATH, SVAMP, and ARC-Challenge, in accordance
with their respective licenses and terms of use. The study does not involve human or animal subjects,
and we did not collect, use, or disclose any personally identifiable information or private user data.
REPRODUCIBILITY STATEMENT
All experiments use publicly available base models (Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B,
InternLM2.5-1.8B, and Llama-3.2-1B) and standard benchmarks (GSM8K, MATH, SVAMP, and
ARC-Challenge). Full hyperparameters, training protocols, held-out split construction, and the
rerun/best-of-candidates selection procedure are specified in Appendix A and Section 5.1. The
cross-signal NTK diagnostic KDR(n) and the M3 gating rules are specified in closed form in Sec-
tions 3.4–4 and Algorithm 2; no proprietary data or infrastructure is required to reproduce the main
results.
REFERENCES
Rishabh Agarwal et al. On-policy distillation of language models: Learning from self-generated
mistakes. ICLR, 2024.
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. GradNorm: Gradient
normalization for adaptive loss balancing in deep multitask networks. In International Conference
on Machine Learning, 2018.
DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learn-
ing. arXiv preprint arXiv:2501.12948, 2025.
Ken Ding. HDPO: Hybrid distillation policy optimization via privileged self-distillation. arXiv
preprint arXiv:2603.23871, 2026.
10


---

Yuxian Gu et al. MiniLLM: Knowledge distillation of large language models. In International
Conference on Learning Representations, 2024.
Jonas H¨ubotter, Frederike L¨ubeck, Lejs Behric, Anton Baumann, et al. Reinforcement learning via
self-distillation. arXiv preprint arXiv:2601.20802, 2026.
Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and gen-
eralization in neural networks. Advances in Neural Information Processing Systems, 31, 2018.
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses
for scene geometry and semantics. In IEEE Conference on Computer Vision and Pattern Recog-
nition, 2018.
Yixing Li, Yuxian Gu, Li Dong, Dequan Wang, Yu Cheng, and Furu Wei. Direct preference knowl-
edge distillation for large language models. arXiv preprint arXiv:2406.19774, 2024.
Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for
multi-task learning. In Advances in Neural Information Processing Systems, volume 34, 2021.
Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention.
In IEEE Conference on Computer Vision and Pattern Recognition, 2019.
Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee,
and Min Lin.
Understanding R1-Zero-Like training: A critical perspective.
arXiv preprint
arXiv:2503.20783, 2025.
Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and
Ethan Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine
Learning, 2022.
OpenAI. Learning to reason with LLMs. 2024.
Xiaohan Qin, Xiaoxing Wang, Ning Liao, and Junchi Yan. NTKMTL: Mitigating task imbalance
in multi-task learning from neural tangent kernel perspective. Advances in Neural Information
Processing Systems, 2025. arXiv:2510.18258.
Yi Ren, Danica J Sutherland, et al. Learning dynamics of LLM finetuning. In International Confer-
ence on Learning Representations, 2025.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances
in Neural Information Processing Systems, volume 31, 2018.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of math-
ematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
Joar Skalse, Nikolaus H R Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and char-
acterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35,
2022.
Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weip-
ing Wang, Jiaqi Wang, and Nan Duan. Self-distilled RLVR. arXiv preprint arXiv:2604.03128,
2026.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian
Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng,
Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen,
Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing
Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO:
An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476,
2025.
11


---

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.
Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems,
volume 33, 2020.
Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover.
Self-distilled reasoner: On-policy self-distillation for large language models.
arXiv preprint
arXiv:2601.18734, 2026.
12


---

Table 4: Default experimental hyperparameters.
Component
Hyperparameter
Value
LoRA
Rank r
64
LoRA
Scaling αLoRA
128
Distillation
Divergence
Generalized JSD
Distillation
JSD coefficient λ
1
2
Distillation
Token clipping τ
0.05
Reinforcement learning
Rollouts per group G
8
Hybrid baseline
Default mixing coefficient α
0.5
Optimization
Reference learning rate
5 × 10−5
A
EXPERIMENTAL DETAILS
Rollout and optimization protocol.
We use the on-policy OPSD construction of Section 3.1.
With LoRA disabled, the frozen teacher receives a privileged prompt containing the ground-truth
solution, whereas the student receives only the problem and its sampled prefix. Unless otherwise
stated, experiments use the defaults in Table 4.
Evaluation metrics and regime statistics.
Training reward is the last-20-step mean unless a table
states otherwise, and accuracy is measured by greedy decoding on GSM8K (n = 1319), SVAMP
(n = 600), and ARC (n = 1172). The SVAMP split excludes all prompts used by the 100- and
200-step training runs. For each architecture–dataset pair, ¯κ is the mean of κt = ∥gR(t)∥/∥gD(t)∥
over RL-active steps of the corresponding naive-Hybrid run and is used as a regime indicator. A run
is marked as collapsed when reward remains below 0.05 for ten consecutive steps.
Candidate selection and post-processing.
For each method–cell pair, the evaluated candidates
include the reference run and, where available, seed replicates, checkpoints every 20 steps for hori-
zons up to T ∈{100, 200, 400}, and conservative-rate reruns at 1–2 × 10−5. M3-Soft additionally
includes the evaluated (β, αmax, T) sweep. SDPO uses its selected conservative-rate candidate; on
InternLM-ARC, this is the 20-step pre-collapse checkpoint. Conservative-rate candidates are made
available to all methods on Llama GSM8K and SVAMP. Collapsed runs are retained and reported
at their measured reward and accuracy rather than excluded. SWA is applied to the Qwen3- and
InternLM-GSM8K chains and to the Llama-GSM8K conservative-rate seed-42 chain using last-
two-checkpoint averaging. Because candidates and checkpoints are chosen using accuracy and
no independent validation split is defined, these entries are reported as best observed rather than
validation-selected results. Seed-level robustness and SWA values are reported in Section B.1.
B
EXTENDED EXPERIMENTS
B.1
DETAILED CROSS-ARCHITECTURE RESULTS
Accuracy.
Table 5 directly compares M3-Soft with the strongest non-M3 baseline in each
architecture–dataset cell. Under the per-cell best-observed protocol, M3-Soft matches or exceeds
the strongest baseline in all 12 cells (11 strict wins, one tie). The largest gains occur on Qwen2.5-
SVAMP (0.905, +4.5pp over RLSD), Qwen2.5-ARC (0.736, +4.1pp over OPSD+GradNorm), and
InternLM-SVAMP (0.777, +4.0pp over RLSD); Llama-GSM8K is the exact tie with RLSD at 0.568.
Seed robustness and SWA.
Seed replicates qualify the best-observed results in Table 5. Wide-
margin cells. The ranking is stable on Qwen2.5-SVAMP, where the M3-Soft seed mean remains
+2.4pp above RLSD, and on Llama-SVAMP, where both replicates exceed GRPO by +7.7 and
+10.2pp. Qwen2.5-GSM8K also wins for both available replicates, while two of three Qwen2.5-
ARC seeds exceed the strongest baseline and the third ties it. Tight-margin cells. The Qwen3-
GSM8K replicates remain within 1.2pp below RLSD before SWA, and the InternLM-GSM8K win
is likewise obtained only after averaging. SWA raises the selected Qwen3-, InternLM-, and Llama-
GSM8K chains by 0.013, 0.015, and 0.004, respectively, yielding accuracies of 0.8324, 0.501, and
0.568; it is not uniformly beneficial, decreasing the Llama seed-43 chain from 0.562 to 0.552.
Llama-GSM8K is seed-fragile at the reference learning rate, where two of three replicates collapse,
but both conservative-rate replicates survive (0.568/0.552). We therefore treat the tight cells as
best-observed parity rather than seed-robust separation.
13


---

Table 5: Accuracy of M3-Soft versus the strongest non-M3 baseline.
Best non-M3 baseline
M3-Soft
M3-Soft
Architecture
GSM8K
SVAMP
ARC
GSM8K
SVAMP
ARC
≥Baseline
Qwen3-1.7B
0.8317
0.940
0.796
0.8324
0.945
0.807
3/3
Qwen2.5-1.5B
0.735
0.860
0.695
0.752
0.905
0.736
3/3
InternLM2.5-1.8B
0.492
0.737
0.604
0.501
0.777
0.608
3/3
Llama-3.2-1B
0.568
0.730
0.545
0.568
0.750
0.556
3/3
Total: 11 strict wins and 1 exact tie
12/12
Training reward and magnitude ratio.
Table 6 reports per-cell training rewards and the exact
¯κ measured on the corresponding naive-Hybrid runs. The largest reference-rate separations occur
on Llama-GSM8K (¯κ = 7,690; M3-Soft 0.647, 20.9× over Hybrid 0.031) and Llama-SVAMP
(¯κ = 12,705; M3-Soft 0.822, 11.4× over GRPO 0.072, while Hybrid reaches 0.000). At the low-κ
end, Qwen3-ARC (¯κ = 2,465) favors Hybrid in training reward (0.919 vs. 0.775 for the extended-
training M3-Soft candidate). For Llama-GSM8K, this table reports the reference-rate M3-Soft run
(0.647), whereas Table 1 uses the conservative-rate SWA chain selected by accuracy (reward 0.591,
accuracy 0.568). Because the M3-Soft column includes tuned and, where marked, extended-training
candidates, this table is an optimization diagnostic rather than a matched-budget comparison.
Table 6: Per-cell training reward (last-20 mean) and mean gradient-magnitude ratio ¯κ. M3-Soft
reports the selected evaluated configuration; ∗denotes extended training, and ¯κ is measured on the
corresponding naive-Hybrid run.
Architecture
Dataset
GRPO
OPSD
OPSD+GradNorm
M3-Select
M3-Soft
¯κ
Qwen3-1.7B
GSM8K
0.838
0.825
0.731
0.844
0.881
3,404
SVAMP
0.922
0.894
0.894
0.947
0.969
5,080
ARC
0.741
0.919
0.769
0.263
0.775∗
2,465
Qwen2.5-1.5B
GSM8K
0.806
0.747
0.653
0.153
0.828∗
2,467
SVAMP
0.769
0.766
0.681
0.331
0.778∗
2,234
ARC
0.659
0.650
0.644
0.319
0.700
2,184
InternLM2.5-1.8B
GSM8K
0.369
0.428
0.306
0.100
0.472∗
2,745
SVAMP
0.594
0.563
0.594
0.366
0.688∗
2,690
ARC
0.616
0.634
0.666
0.416
0.682∗
1,423
Llama-3.2-1B
GSM8K
0.000
0.031
0.006
0.597
0.647
7,690
SVAMP
0.072
0.000
0.038
0.488
0.822
12,705
ARC
0.522
0.469
0.353
0.406
0.650∗
1,562
B.2
LONG-HORIZON STABILITY AND RUNTIME DIAGNOSTICS
Cross-dataset long-horizon results.
Table 7 extends the 500-step evaluation to MATH, SVAMP,
and ARC-Challenge using Qwen3-1.7B. Under the reference configurations, prolonged training can
still trigger collapse beyond GSM8K: GRPO and OPSD+GradNorm collapse on MATH, while
Hybrid collapses on SVAMP. In contrast, both M3 variants remain non-collapsed across all three
datasets.
Phase-wise collapse dynamics.
Table 8 resolves the Qwen3-1.7B GSM8K runs into 100-step
phases. Collapse denotes reward below 0.05 for ten consecutive steps. Unless noted otherwise, the
runs use the reference learning rate 5×10−5; all entries are training rewards rather than held-out
accuracies.SDPO collapses first at step 50, followed by Hybrid at 221, OPSD+GradNorm at 358,
RLSD at 362, and GRPO at 412. In contrast, M3-Select does not trigger the collapse criterion within
500 steps. Phase-wise M3-Soft results are unavailable, but its last-50 reward is reported in Table 2.
Post-collapse conflict diagnostic.
After Hybrid collapses, its measured conflict rate falls to zero
because the RL gradient vanishes, not because the two signals become compatible. M3-Select in-
stead maintains an active conflict rate near CNTK = 0.3 while preserving reward (Figure 6).
14


---

Table 7: Long-horizon reward across datasets on Qwen3-1.7B after 500 steps (last-50 mean). M3-
Soft uses the best evaluated gate configuration per dataset; other methods use the reference configu-
ration. Bold marks the column maximum, and ↓denotes collapse.
Method
MATH
SVAMP
ARC-Challenge
Reference and adaptive baselines
GRPO
↓0.018
0.900
0.773
Hybrid (OPSD)
0.448
↓0.170
0.765
OPSD+GradNorm
↓0.015
0.973
0.790
Ours: boundary-gated mixing (M3)
M3-Select
0.367
0.943
0.282
M3-Soft
0.523
0.950
0.667
Table 8: Phase-wise training reward over 500 steps on Qwen3-1.7B GSM8K. “Full” is the all-step
mean, “Trend” gives the first collapse step, and † denotes batch size 1.
Method
1–100 101–200 201–300 301–400 401–500
Full
Trend
Pure GRPO
0.840
0.801
0.809
0.802
0.075
0.665 collapse @412
Hybrid (OPSD, α=0.5)
0.859
0.762
0.114
0.006
0.001
0.348 collapse @221
OPSD+GradNorm†
0.877
0.823
0.868
0.464
0.000
0.606 collapse @358
RLSD (Yang et al., 2026)
0.855
0.874
0.871
0.259
0.005
0.573 collapse @362
SDPO (H¨ubotter et al., 2026) 0.306
0.000
0.000
0.000
0.000
0.061
collapse @50
M3-Select (αmax =0.15)
0.846
0.818
0.853
0.846
0.810
0.835
stable
B.3
GATE SENSITIVITY ACROSS REGIMES
Controlled sharpness sweep.
We first isolate gate sharpness on Qwen3-1.7B GSM8K at 100
steps. With αmax = 0.5, the gentle β = 0.5 gate attains the largest last-20 reward (0.881), compared
with 0.856 for β = 1, 0.850 for GRPO, and 0.844 for hard M3-Select (Table 9). The soft-gate
configurations have similar values of the count-based budget proxy αeff (0.439–0.446), consistent
with gate shape, rather than a large change in this proxy—driving the observed differences.
Table 9: M3-Soft sharpness sweep on Qwen3-1.7B GSM8K at 100 steps (αmax = 0.5). αeff de-
notes the mean teacher weight. For hard selection, the measured conflict rate is CNTK ≈0.40, so
approximately 60% of tokens retain weight αmax, giving αeff ≈0.5 × 0.60 = 0.30.
Method
β
Last-10
Last-20
αeff
GRPO
—
0.858
0.850
—
OPSD (uniform)
—
0.833
0.825
0.500
M3-Select (hard)
∞
0.855
0.844
∼0.30
M3-Soft
10
0.875
0.847
0.446
M3-Soft
5
0.813
0.838
0.439
M3-Soft
1
0.906
0.856
0.444
M3-Soft
0.5
0.913
0.881
0.446
Cross-architecture reward sensitivity.
Table 10 fixes the horizon at 100 steps and compares β ∈
{0.5, 1, 5} across all 12 architecture–dataset cells. A gentle gate (β ∈{0.5, 1}) is the best evaluated
M3-Soft setting in 11/12 cells. Both high-κ Llama arithmetic cells favor β = 1, whereas InternLM-
SVAMP is the sole β = 5 exception. This table diagnoses sensitivity within M3-Soft; it is not the
source of the best-observed headline in Table 1.
Sensitivity.
Training reward does not by itself select the best gate for accuracy. On Qwen2.5-
SVAMP, the very-soft recipe (β, αmax) = (0.1, 0.001) reaches 0.884±0.026 across three seeds
(0.855/0.892/0.905), exceeding RLSD’s 0.860 by 2.4pp in the seed mean. On InternLM-SVAMP,
the same recipe gives 0.710/0.777/0.722: the mean (0.736) is at parity with RLSD (0.737), while
the best observed seed reaches 0.777. Thus, the Qwen2.5 gain is seed-robust, whereas the InternLM
headline is best-observed rather than a mean separation.
15


---

0
100
200
300
400
500
Training step
0.0
0.2
0.4
0.6
Conflict rate CNTK
Hybrid collapse @ 139
(CNTK ! 0: reward = 0, trivial)
conflict persists,
reward survives
Hybrid (naive)
M3-Select
Figure 6: Runtime token-level conflict on Qwen3-0.6B GSM8K. Hybrid’s post-collapse drop reflects
a vanishing RL gradient; M3-Select remains active near CNTK = 0.3.
Table 10: M3-Soft sharpness sensitivity at 100 steps (last-20 mean reward). αmax = 0.05 by default
and 0.025 on Qwen3-SVAMP.
Model
Dataset
β = 0.5
β = 1
β = 5
GRPO
Qwen3-1.7B
GSM8K
0.834
0.856
0.850
0.838
SVAMP
0.991
0.969
0.981
0.953
ARC
0.516
0.350
0.356
0.741
Llama-3.2-1B
GSM8K
0.569
0.647
0.353
↓0.000
SVAMP
0.456
0.822
0.609
0.072
ARC
0.559
0.500
0.388
0.522
Qwen2.5-1.5B
GSM8K
0.697
0.728
0.169
0.806
SVAMP
0.794
0.691
0.463
0.769
ARC
0.700
0.475
0.369
0.659
InternLM2.5-1.8B
GSM8K
0.234
0.269
0.138
0.369
SVAMP
0.594
0.366
0.744
0.594
ARC
0.469
0.575
0.472
0.616
B.4
MECHANISTIC VALIDATION
The following diagnostics test the mechanism at progressively coarser levels. We first verify the pre-
dicted one-step effect after partitioning tokens by their pre-update cross-signal NTK, then compare
response-level and token-level notions of conflict, and finally summarize the aggregate behavior
across settings.
Table 11: Response-level semantic conflict (Qwen3-0.6B, 208 trajectories): on-policy = teacher
evaluates the student’s own trajectory, off-policy = ground-truth contexts; true opposition = wrong
trajectories with opposing RL and distillation gradients.
Mode
Conflict %
cos(gR, gD)
True opposition
κmedian
¯LD
On-policy OPSD
75%
+0.08
25/157 (16%)
388
0.098
Off-policy
75%
−0.01
90/157 (57%)
68
0.810
Response-Level Semantic Conflict.
Response-level labels and token-level interactions answer
different questions. For each trajectory in a GRPO group, we measure its reward, normalized ad-
vantage, trajectory-level gradient cosine cos(gR, gD), and magnitude ratio κ. Across 20 analysis
steps with Qwen3-0.6B (batch size 2, group size 8), groups with zero reward variance are excluded
because GRPO assigns zero advantage and hence no reward gradient, leaving 208 trajectories. Both
modes label 75% of trajectories as semantically conflicting, but they differ sharply in gradient be-
havior. Among the 157 incorrect trajectories, true opposition occurs in 25 cases (16%) on-policy
and 90 cases (57%) off-policy; the corresponding median magnitude ratios are 388 and 68. Within
the on-policy sample, the mean cosine is +0.14 on incorrect trajectories and −0.11 on correct tra-
16


---

0
200
400
600
800
1000
1200
1400
∙= kgRk=kgDk
0.000
0.002
0.004
0.006
0.008
0.010
0.012
Density
(a) Magnitude ratio ∙
On-policy (median=388)
Off-policy (median=68)
On-policy
correct
On-policy
incorrect
Off-policy
correct
Off-policy
incorrect
0.0
0.2
0.4
0.6
0.8
Mean LD
0.098
0.098
0.815
0.810
(b) Teacher signal LD by correctness
-0.4
-0.2
0.0
0.2
0.4
cos(gR; gD)
0
2
4
6
8
10
Density
(c) Gradient alignment
On-policy (mean=+0.078)
Off-policy (mean=-0.007)
Figure 7: On-policy versus off-policy semantic conflict. On-policy distillation has a higher median κ
but fewer incorrect trajectories with genuinely opposed reward and distillation gradients; off-policy
supervision reverses this pattern.
Qwen3-1.7B
GSM8K
Qwen3-1.7B
MATH
Qwen3-0.6B
GSM8K
Qwen3-0.6B
MATH
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Conflict rate CNTK
CNTK ¼ 0:41 across scales and datasets (dashed)
0.40
0.41
0.46
0.37
Figure 8: Per-position conflict rate CNTK across model scales and datasets. The fraction of positions
with negative cross-signal NTK remains between 0.37 and 0.46 (mean ≈0.41, dashed line).
jectories, matching the inversion predicted by Proposition 13. Thus, response-level rejection alone
does not determine whether the two gradients oppose each other.
Token-Level NTK Conflict.
We next measure the same-position cross-signal NTK KDR(n). Po-
sitions with KDR(n) ≥0 are locally compatible, whereas positions with KDR(n) < 0 receive a
teacher component that opposes reward progress. Across the measured model scales and datasets,
the count-based conflict rate ranges from 0.37 to 0.46, with a mean of approximately 0.41 (Fig-
ure 8). The narrow range shows that the conflict observed in the main-text rollout is not isolated to
one model or dataset, without implying that all settings have identical conflict structure.
Figure 9 restores token identities for one correct and one incorrect response to the same prompt.
Across the four paired case studies, conflict covers 44.3% of positions on correct rollouts and 54.9%
on incorrect rollouts, a difference of 10.6 percentage points. Every rollout nevertheless interleaves
compatible and conflicting positions, so these examples support token-level localization rather than
a response-level cutoff. The count difference is descriptive for the eight visualized rollouts and is
not presented as a population-level estimate.
Aggregate Conflict Diagnostics.
Figure 11 summarizes the reference Qwen diagnostics. Across
the measured scales and datasets, the mean magnitude ratio remains in the 3,400–4,000 range, while
the parameter-level cosine averages −0.009. The M3-Select trace also shows higher count-based
conflict at lower-reward steps. This last relationship is descriptive: the plot does not establish that
conflict rate alone causes or predicts method performance.
B.5
ADDITIONAL ABLATIONS AND ILLUSTRATIVE EXAMPLES
LoRA Rank Ablation.
We vary the LoRA rank from 8 to 128 on Qwen3-1.7B GSM8K while
holding the remaining M3-Soft configuration fixed. The mean reward ranges from 0.824 to 0.844
across this 16× change in trainable rank, with the best value at r = 64 and a total spread of 0.020
(Table 12). Thus, performance in this setting is not sensitive to the tested LoRA rank. Because κ
and the token-level conflict rate were not separately recorded for every rank, this ablation supports
reward robustness rather than rank invariance of the underlying diagnostics.
17


---

Prompt: If left( r + frac{1}{r} right)^2 = 3, then find r^3 + frac{1}{r^3}.
Ground-truth answer: 0
(a) correct response, reward=1, advantage=+0.72, conflict rate CNTK=42%
We
·are
·given
:
$$
left
(
·r
·+
·
frac
{
1
}{
r
}
·
right
)^
2
·=
·
3
$$
We
·are
·to
·find
:
$$
r
^
3
·+
·
frac
{
1
}{
r
^
3
}
$$
---
###
·Step
·
1
:
·Expand
·the
·given
·equat
$$
(b) wrong response, reward=0, advantage=-1.21, conflict rate CNTK=61%
We
·are
·given
·the
·equat
:
$$
left
(
·r
·+
·
frac
{
1
}{
r
}
·
right
)^
2
·=
·
3
$$
We
·are
·to
·find
:
$$
r
^
3
·+
·
frac
{
1
}{
r
^
3
}
$$
---
###
·Step
·
1
:
·Expand
·the
·given
·equat
$$
conflict: KDR < 0 (masked by M3-Select)
synergy: KDR
0 (teacher retained)
pale: weak interaction
Figure 9: A readable token-level case study for a correct (A > 0) and an incorrect (A < 0) response
to the same prompt. Darker red denotes stronger local conflict (KDR(n) < 0), darker blue denotes
stronger compatibility, and pale colors denote weak interaction. Both responses interleave the two
signal types.
Table 12: LoRA-rank ablation for M3-Soft on Qwen3-1.7B GSM8K over 100 training steps.
LoRA Rank
r = 8
r = 16
r = 32
r = 64
r = 128
Mean Reward
0.826
0.824
0.828
0.844
0.841
Illustrative Failure Patterns.
The following cases are stylized examples distilled from qualitative
patterns observed in the GSM8K and SVAMP runs; they are not verbatim training trajectories or
additional controlled experiments. Their purpose is to show how magnitude imbalance, group-level
cancellation, and token-level sign conflict can appear in concrete reasoning traces. In the examples,
+ and −denote positive and negative group-relative advantages, respectively.
Bad Case 1: length inflation under drowning (κ ≫κ∗)
On Llama-3.2-1B SVAMP (¯κ ≈1.3×104), the RL gradient dominates the teacher gradient by
roughly four orders of magnitude. Because the verifier rewards only the final numeric answer,
the group-normalized advantage rewards any continuation that eventually reaches the correct
digit and penalizes short-but-wrong ones. When distillation is drowned out, the model no longer
receives the teacher’s “brevity + structure” prior, response length inflates steadily as training
proceeds, and reward collapses to 0.072 (Table 6, row Llama/SVAMP/GRPO). A stylized post-
drowning rollout of the kind we observe:
Let x be the answer.
We are told that Melanie has 3
·
4
apples...so x = 12.
But wait, let us re-verify:
3 · 4 = 12,
so indeed x = 12.
Actually, checking once more, x = 12.
...Answer:
12.
The re-verification loops are reward-neutral (verifier only checks “12”) but teacher-
negative. Under M3 the magnitude normalization restores the teacher’s unit-scale voice—the
brevity+structure prior is no longer drowned by the 104 norm ratio—and the boundary ad-
mits this anti-repetition pressure exactly where it is reward-compatible (KDR(n) ≥0, e.g. in
negative-advantage rollouts whose RL residual also pushes against the loops), so length infla-
tion is suppressed and reward reaches 0.822 (same table row, M3-Soft; the hard-gated M3-
Select variant recovers 0.488). This is the token-level dual of the length-drift phenomenon
18


---

Prompt: Compute det( 2 & 0 & -1 ; 7 & 4 & -3 ; 2 & 2 & 5 ).
Ground-truth answer: 46
(a) correct response, reward=1, advantage=+0.72, CNTK=49% (full response; first 48 tokens shown)
We
·are
·given
·the
·deter
·of
·a
·$
·
3
·
times
·
3
·$
·matrix
:
$$
begin
{
vm
atrix
}
·
2
·&
·
0
·&
·-
1
·
·
7
·&
·
4
·&
·-
3
·
·
2
·&
·
2
(b) wrong response, reward=0, advantage=-1.21, CNTK=52% (full response; first 48 tokens shown)
We
·are
·asked
·to
·compu
·the
·deter
·of
·the
·follo
·
3
×
3
·matrix
:
$$
begin
{
vm
atrix
}
·
2
·&
·
0
·&
·-
1
·
·
7
·&
·
4
·&
·-
3
·
·
2
·&
·
2
·&
Prompt: On a balance scale, 3 green balls balance 6 blue balls, 2 yellow balls balance 5 blue balls, and 6
blue balls balance 4 white balls. How many blue balls are needed to balance 4 green, 2
Ground-truth answer: 16
(c) correct response, reward=1, advantage=+0.35, CNTK=42% (full response; first 48 tokens shown)
We
·are
·given
·sever
·balan
·relat
,
·and
·we
·need
·to
·deter
·how
·many
·**
blue
·balls
**
·are
·needed
·to
·balan
·**
4
·green
·balls
**,
·**
2
·yellow
·balls
**,
·and
·**
2
·white
·balls
**
.
---
###
·Step
·
1
:
·List
·the
·given
(d) wrong response, reward=0, advantage=-2.47, CNTK=62% (full response; first 48 tokens shown)
We
·are
·given
·a
·set
·of
·balan
·scale
·relat
·and
·need
·to
·find
·how
·many
·**
blue
·balls
**
·are
·needed
·to
·**
balance
**
:
-
·**
4
·green
·balls
**
-
·**
2
·yellow
·balls
**
-
·**
2
·white
·balls
**
---
###
·Step
·
Prompt: Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles. How many
different groups of two marbles can Tom choose?
Ground-truth answer: 7
(e) correct response, reward=1, advantage=+1.62, CNTK=42% (full response; first 48 tokens shown)
We
·are
·given
:
-
·Tom
·has
·**
4
·mar
bles
**
:
·
·-
·
1
·red
·marble
·
·-
·
1
·green
·marble
·
·-
·
1
·blue
·marble
·
·-
·
3
·ident
·yellow
·mar
bles
We
·are
·to
·find
·how
(f) wrong response, reward=0, advantage=-0.54, CNTK=45% (full response; first 48 tokens shown)
We
·are
·given
:
-
·**
Tom
·has
**:
··
·
·-
·
1
·red
·marble
··
·
·-
·
1
·green
·marble
··
·
·-
·
1
·blue
·marble
··
·
·-
·
3
·ident
·yellow
·mar
bles
··
We
·are
·to
·find
·how
·many
·**
differ
conflict: KDR < 0 (masked by M3-Select)
synergy: KDR
0 (teacher retained)
pale: weak interaction
Figure 10: Three additional paired token-level case studies. Headers report CNTK over each full
200-token response; the first 48 tokens are displayed. The incorrect rollout has the higher conflict
rate in each pair (49% →52%, 42% →62%, and 42% →45%), while every rollout contains both
compatible and conflicting positions.
19


---

1.7B
GSM8K
1.7B
MATH
0.6B
GSM8K
0.6B
MATH
1000
2000
3000
4000
5000
6000
7000
∙= kgRk=kgDk
¹∙= 3; 404
¹∙= 4; 028
¹∙= 3; 397
¹∙= 3; 371
(a) ∙ distribution across configurations
0
100
200
300
400
500
Training step
-1.0
-0.5
0.0
0.5
1.0
cos(gD; gR)
(b) Parameter-level cosine similarity
Hybrid, 500-step
mean -0.009
Hybrid, 100-step
0.0
0.2
0.4
0.6
0.8
1.0
CNTK (conflict rate)
0.0
0.2
0.4
0.6
0.8
1.0
Mean reward
(c) CNTK vs. reward (M3-Select)
1.7B GSM8K
1.7B MATH
0.6B GSM8K
0.6B MATH
Figure 11: Aggregate conflict diagnostics: (a) mean magnitude ratio across settings; (b) parameter-
level gradient cosine; and (c) token-level conflict rate versus reward for M3-Select.
reported in prior exploration-boundary work: the drowning threshold acts as an implicit “eos-
suppression” bias.
Bad Case 2: shared-prefix contamination in the same rollout group
GRPO computes advantages per rollout, so a group of G trajectories that share a long prefix
will apply opposite-sign updates to that prefix whenever their final answers disagree. Consider
a GSM8K group of two rollouts from the same prompt:
Rollout 1 (+):
Let x be the number of apples.
x = 5 · 3 =
15.
Answer:
15.
Rollout 2 (−):
Let x be the number of apples.
x = 5+3 =
8.
Answer:
8.
Denote the shared prefix as s. The naive RL contribution at any s-token yn is
Gn
R

yn∈s = −

(r1 −b) + (r2 −b)
 eyn −pn
S

= 0,
since the group-mean baseline b = 1
2(r1 + r2) centers the two advantages: the RL signal on
the shared prefix cancels exactly in this two-rollout group—and nearly so for general G, where
unbalanced +/−counts and length-normalization weights leave a small residual. But the dis-
tillation signal Gn
D = pn
S −pn
T remains fully alive on the same tokens, and the teacher pushes
pS toward its preferred verbalization—while the small residual RL noise flips sign randomly
across steps. The result is an unstable prefix whose gradient variance is dominated by cancel-
lation noise (large γn, Definition 3 extension in §F). M3-Select computes cos φn between Gn
D
and the per-rollout Gn
R, detects ¯cn = +1 on the shared prefix of the winning rollout (teacher
and RL prefer the same continuation there), and admits the teacher at its full budget αmax: af-
ter per-token normalization the distillation direction is the only persistent signal on the prefix,
anchoring it against the sign-flipping residual noise. The RL term is never masked—its group-
averaged contribution self-cancels as shown above—so reward supervision effectively acts only
on the divergence region (“5·3=15” vs “5+3=8”), where the two rollouts’ residuals no longer
cancel. This is the on-policy hybrid analogue of the shared-prefix contamination reported by
Ren et al. (2025) (“I-ate-lunch, happy/unhappy” example).
Bad Case 3: correct intermediate step suppressed by wrong final answer
Even when the group contains only one (−) rollout, an internally-correct intermediate step is
penalized because GRPO’s outcome reward propagates the trajectory-level sign to every token.
Consider a two-step arithmetic rollout:
Rollout A (−):
5·4 =
20 , then 20+3 =
24 .
Answer:
24.
Rollout B (+):
5·4 =
20 , then 20+3 =
23 .
Answer:
23.
The intermediate token “20” is arithmetically correct in both rollouts, and the teacher’s distri-
butional prior pT places > 0.9 mass on it. Under naive hybrid,
Gn
R
Rollout A
yn=“20′′ = −(rA −b)(e20 −pn
S),
20


---

Table 13: Stylized failure patterns, observable signatures, and corresponding interventions.
Failure pattern
Observable signature
Intervention
Length inflation
Magnitude drowning
Large κ; repetitive reasoning not penalized
by the final-answer verifier
Magnitude normalization
Shared-prefix cancellation
Group-level cancellation
Weak aggregate reward signal on tokens
shared by opposite-advantage rollouts
Per-token
compatibility
gate
Correct-intermediate
sup-
pression
Token-level sign conflict
KDR(n) < 0 on a locally correct token in-
side a rejected response
M3-Select or M3-Soft
Scaffold-token instability
Boundary uncertainty
|KDR(n)| ≈0 on structural or low-content
tokens
M3-Soft
with rA −b < 0, so the RL update lowers the probability of the correct intermediate. The
per-position diagnostic (Proposition 11) yields ⟨Gn
D, Gn
R⟩∝−(rA −b)σn < 0, so ¯cn =−1 and
M3-Select withholds teacher supervision from Rollout A’s update at this token—the gate acts
on the teacher, never on RL. The repair instead comes from the group: Rollout B’s equal-and-
opposite RL contribution cancels Rollout A’s in the aggregate, while on Rollout B the diagnos-
tic reads ¯cn = +1 and the teacher is admitted, so the surviving net update pushes pS(“20′′) ↑.
Correct-intermediate tokens under negative advantage arise whenever a group mixes right and
wrong final answers built on shared sub-computations; protecting them through this admitted
teacher vote is a driver of the 0.647 vs 0.000 reference-rate gap between M3-Soft and GRPO
(Table 6). This is the dense–sparse counterpart of the “1+1=2” intermediate-step cancellation
phenomenon documented in prior exploration-boundary analyses.
Bad Case 4: directional cancellation on high-entropy scaffold tokens
Certain tokens are structurally common to almost every correct and wrong solution: the “=”
sign, “Answer:”, “\n”, punctuation, and low-content connectives (“so”, “then”). The teacher
assigns pT >0.9 on these, and the RL signal is noisy and near-zero in expectation (they appear
roughly equally in + and −rollouts). Yet the per-step RL contribution is non-zero and has large
variance, producing a gradient-cancellation rate
γn = 1 −


 1
G
P
g ∇θℓg(yn)


2
1
G
P
g


∇θℓg(yn)


2
approaching 1 at these positions, in sharp contrast to content tokens, whose group updates are
directionally aligned. Here the soft gate is the right tool: M3-Soft assigns these positions a
stable intermediate budget αn = αmaxσ(βcos φn) ≈αmax/2 that, at small β, is insensitive
to the noisy sign of \
cos φn, whereas M3-Select’s hard indicator flips with that sign and evicts
the teacher on roughly half the scaffold positions at random. Since the group-averaged RL
residual is near zero-mean here while the normalized teacher direction is persistent, distillation
dominates the expected update without any signal being masked—which is why M3-Soft with
β =0.1 recovers scaffold-token fluency on the low-κ Qwen2.5-SVAMP cell (§B.3).
Table 13 summarizes what each example is intended to illustrate. The first case concerns magnitude
imbalance, the next two concern sign structure under outcome-level credit assignment, and the fourth
concerns uncertainty near the compatibility boundary.
C
EXPLORATION BOUNDARY FRAMEWORK: FULL STATEMENTS AND
BOUNDARY DEFINITION
This appendix states the intra-RL quantities underlying the cross-signal analysis in Section 3.4 and
defines the reward–teacher compatibility boundary.
21


---

C.1
EXPLORATION BOUNDARY AND GRADIENT FOLDING IN RL
Definition 2 (Gradient Diversity and Exploration Boundary). For G rollouts, let g(i)
=
Ai∇θ log πθ(yi | x), with Ai = (r(i) −¯r)/σr, and write ¯g = G−1 P
i g(i). For ¯g̸ = 0, define
∆(θ) = G−1 P
i ∥g(i)∥2
∥¯g∥2
≥1,
BS(θ) = Nmax∆(θ).
(16)
The inequality is Jensen’s inequality; equality holds when all gradients coincide, while equal-norm
orthogonal gradients give ∆= G. Large ∆measures cancellation relative to the individual gra-
dient energy. Here Nmax is the rollout budget per prompt, and BS is the exploration-boundary
index.
Definition 3 (Gradient Folding and Cancellation Rate). With binary rewards, positive- and negative-
advantage trajectories can oppose one another at shared token positions; we call this gradient
folding. Its cancellation rate and surviving gradient magnitude satisfy
γ = 1 −∆−1 = 1 −∥P
i g(i)∥2
G P
i ∥g(i)∥2 ,
∥¯g∥=
p
1 −γ
 
G−1 X
i
∥g(i)∥2
!1/2
.
(17)
Thus γ ∈[0, 1) measures the fraction of mean squared gradient magnitude canceled. Complete
cancellation gives γ = 1 and ∆= +∞, provided the individual gradients are not all zero.
Definition 4 (Token-Level NTK). Let Jn = ∇θzn ∈Rdpar×|V|. The vocabulary-space kernel is
K(s, t) = (Js)⊤Jt, and its sampled-token contraction is
Kt(τ, s, t) = ⟨∇θ log πθ(as | τ<s), ∇θ log πθ(at | τ<t)⟩.
(18)
For an update from position n alone, ˙zn
= −K(n, n)δn; a summed update gives ˙zn
=
−P
m K(n, m)δm. These kernels separate local effects from interactions through shared parame-
ters and provide the geometry for position-level masking.
Definition 5 (Reward–Teacher Compatibility Boundary). For KDR(n) = ⟨Jnδn
D, Jnδn
R⟩, define
∂B = B0 = {n : KDR(n) = 0},
B± = {n : ±KDR(n) > 0}.
(19)
The compatible region B+ contributes positively to the local reward projection, the orthogonal
region B0 contributes zero, and the conflicting region B−contributes negatively. M3-Select admits
the teacher on B+ ∪B0 and masks B−.
D
EXTENDED THEORETICAL ANALYSIS
This appendix contains extended theoretical results referenced in Sections 3.4 and 3.5.
Corollary 1 (One-Step Logits Degradation). For advantage A̸ = 0, write gn
R = Jnδn
R and gn
D =
Jnδn
D. The first-order contribution of position n’s hybrid update to its sampled-token log-probability
is
∆n log pS(ˆyn | ˆy<n) ≈

−gn
R
A , −η[(1 −α)gn
R + αgn
D]

= η
A [(1 −α)KRR(n) + αKDR(n)] ,
(20)
where KRR(n) = ∥gn
R∥2 and ∇θ log pS(ˆyn | ˆy<n) = −gn
R/A. At a conflicting position, distillation
lowers the sampled-token log-probability when A > 0 and raises it when A < 0. For A > 0,
the total diagonal contribution becomes negative exactly when α|KDR(n)| > (1 −α)KRR(n);
Section ?? measures this effect.
Exact κ-growth identity.
For nonzero gradients, direct logarithmic differentiation gives
d log κ
dt
= d log ∥gR∥
dt
−d log ∥gD∥
dt
.
(21)
Thus κ grows whenever the reward norm decays more slowly than the teacher norm. If the difference
of these logarithmic rates is the positive constant λ, then κ(t) = κ(0)eλt.
Corollary 2 (Aggregate Synergy Condition). Let Φτ = cos(gD(τ), gR(τ)) be the trajectory-level
alignment, with fixed conditional means µ0 = E[Φτ | r = 0] > 0 and µ1 = E[Φτ | r = 1] < 0. At
accuracy p,
E[Φτ] = (1 −p)µ0 + pµ1 > 0
⇐⇒
p < p∗:=
µ0
µ0 −µ1
.
(22)
22


---

Corollary 3 (Diminishing Synergy Under Increasing Accuracy). Under the fixed-conditional-mean
model of Corollary 2, dE[Φτ]/dp = µ1 −µ0 < 0, so expected trajectory alignment crosses zero at
p∗and approaches µ1 as p →1. The threshold is determined by the two conditional alignments.
Remark 2 (Why NTK, not just cosine?). The aggregate cosine summarizes the same parameter-
space inner product represented by the NTK: ⟨gD, gR⟩= P
n,m(δn
D)⊤K(n, m)δm
R for summed
gradients. The kernel decomposition exposes which positions and cross-position interactions pro-
duce that scalar, enabling local gating that an aggregate cosine alone cannot specify.
The measured threshold separates cells in which hard masking helps from cells in which a softer
gate preserves more teacher signal. The local score identifies the teacher’s immediate reward pro-
jection; gradient magnitude, estimation noise, and cross-position interactions determine how this
local decision translates into training progress. The experiments in Sections 5.3 and 5.3 compare
these regimes.
D.1
CONFLICT AS INFORMATION DESTRUCTION
At conflicting positions of a positive-advantage trajectory, the teacher contribution lowers the
sampled-token log-probability (Corollary 1). Repeated contributions of this sign can erode rewarded
behavior. The 500-step experiments (Section 5.3) show collapse under uniform mixing in high-κ
cells, while M3-Select removes these negative local teacher projections before the update. This
links the local mechanism to the observed training trajectories.
E
EXTENDED METHOD THEORY
This appendix presents the full theoretical analysis of M3 and M3-Select. All formal statements and
proof sketches summarized in Section 4 are collected here.
E.1
M3-EG: FAST–SLOW EXTRAGRADIENT UPDATE
The extragradient variant of Section 4 keeps the boundary gate α∗
n of Eq. (12) but replaces the
synchronous mixture with a two-timescale schedule. From the anchor θ, the inner fast step applies
only the boundary-gated teacher to reach a look-ahead point
˜θ = θ −ηin 1
N
PN
n=1 Jnα∗
nˆδn
D.
(23)
The on-policy reward group is then scored at ˜θ, but its gradient is applied as an outer correction
anchored at the original θ,
θ+ = θ −ηout 1
N
PN
n=1 Jnˆδn
R
 ˜θ,
(24)
using a first-order approximation that does not backpropagate through the inner step: in practice one
caches the slow-gradient direction
1
N
P
n Jnˆδn
R| ˜θ, restores θ, and then applies Eq. (24). The step
sizes (ηin, ηout) play the roles of the inner (look-ahead) and outer (correction) rates of a standard
extragradient scheme. Algorithm 1 states the full procedure.
Algorithm 1 M3-EG: Fast–Slow Extragradient Update
Input: gate ceiling αmax, inner (look-ahead) rate ηin, outer (correction) rate ηout, smoothing ϵ
1: For step t = 1, . . . , T do
2: Compute per-position residuals δn
D, δn
R, Jacobians Jn, and scores KDR(n) ←⟨Jnδn
D, Jnδn
R⟩.
3: Gate α∗
n ←αmax1{KDR(n) ≥0} (hard) or αmaxσ(βcos φn) (soft); normalize ˆδn
D, ˆδn
R.
4: Fast / look-ahead (teacher only): ˜θ ←θt −ηin 1
N
P
n Jnα∗
nˆδn
D.
▷hard gate masks Ω−
5: Re-score at ˜θ (reward only): hR(˜θ) ←1
N
P
n Jnˆδn
R
 ˜θ (no backprop through ˜θ).
6: Slow / correction (anchored at θt): θt+1 ←θt −ηout hR(˜θ).
End for
23


---

Conflict isolation.
Write hR(θ) = N −1 P
n Jnˆδn
R for the normalized reward field and mD =
N −1 P
n Jnα∗
nˆδn
D for the gated teacher field. The executed move is −ηouthR(θ −ηinmD), so
teacher information enters through the look-ahead location. Positive rescaling of an unsmoothed
teacher residual leaves mD unchanged.
Proposition 1 (Local Reward Descent of the Extragradient Field). Let gR = ∇LR(θ), assume
aR = ⟨hR(θ), gR⟩> 0, and let hR be Lh-Lipschitz along the inner step. Then
⟨θ −θ+, gR⟩= ηout⟨hR(θ −ηinmD), gR⟩
≥ηout (aR −ηinLh∥mD∥∥gR∥) > 0
(25)
whenever ηinLh∥mD∥∥gR∥< aR. For sufficiently small outer step ηout, this is a reward-descent
step. When hR = gR, its first-order expansion is ηout[∥gR∥2 −ηin⟨HRmD, gR⟩] + O(ηoutη2
in),
with HR = ∇2LR(θ).
The inner normalization controls the teacher field’s scale, while the hard gate removes its nega-
tive local reward projections. The outer descent condition above quantifies the additional effect of
moving the reward evaluation point (empirical comparison: Table 1).
Algorithm 2 M3-Select: Boundary-Guided Token-Level Mixing
Input: αmax, step size η, smoothing ϵ, EMA update scale ¯s
1: For step t = 1, . . . , T do
2: Compute per-position residuals δn
D, δn
R and Jacobians Jn.
3: Compute compatibility scores KDR(n) ←⟨Jnδn
D, Jnδn
R⟩.
4: Set αn ←αmax1{KDR(n) ≥0} and record CNTK.
5: Normalize residuals ˆδn
D ←δn
D/(∥δn
D∥+ ϵ) and ˆδn
R ←δn
R/(∥δn
R∥+ ϵ).
6: Update θt+1 ←θt −η¯s 1
N
P
n Jn[αnˆδn
D + (1 −αn)ˆδn
R].
End for
Quadratic sub-optimality of a fixed mixing coefficient.
Let d = gD −gR and let α∗∈[0, 1]
minimize ∥gH(α)∥2. Expanding around the constrained minimizer gives
∥gH(α)∥2 −∥gH(α∗)∥2 = ∥d∥2(α −α∗)2 + 2(α −α∗)⟨gH(α∗), d⟩
≥∥d∥2(α −α∗)2.
(26)
The last term on the first line is nonnegative by constrained optimality and vanishes at an interior
minimizer. The curvature is ∥d∥2 = rD2 + rR2 −2ΦDRrDrR. For a random vector g, we use the
total variance
Var(g) := E∥g −Eg∥2 = tr Cov(g),
with Vart denoting conditioning on the history available before the current gradient sample.
Theorem 1 (Boundary-Gated Reward Convergence). Assume LR is L-smooth and bounded be-
low.
In the conditional model of Appendix L, each position has equal-norm reward and de-
terministic teacher means un, dn, position subspaces are orthogonal, gates are fixed before
fresh reward noise is sampled, and ∇LR = N −1 P
n un. Define cn = ⟨un, dn⟩/∥un∥2 and
wn = ∥un∥2/ P
m ∥um∥2, taking cn = 0 at zero-norm positions. The population-sign gate is
αn = αmax1{cn ≥0}, and ρR,t = P
n wn[(1 −αn) + αncn]. For αmax < 1, choose deterministic
bounds 0 < ρR ≤ρR,t and Vart(ˆgsel
H ) ≤vασ2
R valid at every iteration and history; ρR = 1−αmax
is admissible. If η ≤ρR/L, then
1
T
T −1
X
t=0
E∥∇LR(θt)∥2 ≤1
ρR
2(LR(θ0) −L∗
R)
ηT
+ Lηvασ2
R

.
(27)
Proposition 2 (Conditional Variance of Boundary-Gated Mixing). In the preceding conditional
model, let νn = Et∥ξn∥2 be the fresh reward-noise variance at position n. Then
Vart(gsel
H ) =
1
N 2
X
n
(1 −αn)2νn = vtσ2
R,t,
(28)
24


---

where σ2
R,t = N −2 P
n νn and vt = P
n(1 −αn)2νn/ P
n νn ≤1 when noise is nonzero; take
vt = N −1 P
n(1 −αn)2 when all νn = 0. Choose a deterministic vα ≥vt valid uniformly
over iterations and histories; vα = 1 is always admissible when σ2
R,t ≤σ2
R. For equal position
variances and admitted fraction ρadm = |Ω+ ∪Ω0|/N, vt = 1 −(2αmax −α2
max)ρadm = (1 −
¯α)2 + α2
maxρadm(1 −ρadm), where ¯α = αmaxρadm.
The local reward-projection guarantee underlying both statements is
ΠR(n) ≥(1 −αmax)∥Jnˆδn
R∥2 for n ∈Ω+ ∪Ω0,
ΠR(n) = ∥Jnˆδn
R∥2 for n ∈Ω−.
(29)
Theorem 2 (M3-Norm Convergence Guarantee). Assume LR is L-smooth and bounded below, with
an unbiased reward estimator of conditional variance at most σ2
R. Conditional on each iterate, let
the teacher direction be deterministic, matched to the population reward-gradient norm, and have
alignment at least ϕ∗(Appendix J). For fixed α ∈(0, 1), put c = 1 −α + αϕ∗> 0. If η ≤c/L, then
1
T
T −1
X
t=0
E∥∇LR(θt)∥2 ≤1
c
2(L0
R −L∗
R)
ηT
+ Lη(1 −α)2σ2
R

.
(30)
Proposition 3 (Variance Reduction via Distillation). Under the conditional deterministic-teacher
model of Theorem 2, Vart(ˆgH) = (1 −α)2 Vart(ˆgR) ≤(1 −α)2σ2
R.
Proposition 4 (Optimal Per-Position Mixing). For unit parameter-space directions with cosine cn,
let ΠR(n; α) = 1 −α + αcn. Maximizing the calibrated objective ΠR(n; α) + α = 1 + αcn
over [0, αmax] gives the hard gate α∗
n = αmax1{cn ≥0}, with ties assigned to the teacher. Under
additive logistic score noise of scale 1/β, its expected allocation is αmaxσ(βcn), the M3-Soft gate.
Proposition 5 (GradNorm Degeneracy under κ ≫1). At equal target training rates, impose wD +
wR = 1 and norm balance wD∥gD∥= wR∥gR∥. Then w∗
D = κ/(1 + κ) and w∗
R = 1/(1 + κ),
giving w∗
R ≈0.03% at κ = 3,400. While wD/wR = Θ(1), the teacher’s share of weighted gradient
magnitude is wD/(wD + wRκ) = Θ(κ−1).
A natural loss-level alternative to fixed mixing is to gate α on the positive part of the alignment
functional, ΦDR+ = max(ΦDR, 0), yielding the adaptive schedule
αadaptive = αmax · ΦDR
+/ max(ΦDR
+, ϵ).
(31)
When the gate uses an exponentially smoothed alignment estimate, its response to a sign change has
the following delay.
Proposition 6 (Phase Delay in Adaptive Mixing). Let bΦt+1 = ρbΦt + (1 −ρ)Φt, with 0 < ρ < 1.
If alignment changes from Φ+ > 0 to Φ−< 0 at t1 and bΦt1 = Φ+, then
bΦt1+k = Φ−+ (Φ+ −Φ−)ρk,
k∗=
log[(Φ+ −Φ−)/|Φ−|]
log(1/ρ)

.
(32)
The estimate remains positive for k < k∗, so the teacher gate stays active during that interval.
Proposition 7 (Empirical Invariance of κ under LoRA Rank). Within the measured LoRA-rank
range r ∈{8, 16, 32, 64}, ¯κ = 3,476 ± 301 (CV = 8.7%). On Qwen3-GSM8K, the corresponding
scale comparison is κ ≈3,397 at 0.6B and ≈3,400 at 1.7B. These are within-task observations; at
rank 128 the measured ratio rises to 6,296, and ratios vary substantially across tasks (Section B.1).
Remark 3 (κ Invariance Across LoRA Ranks). A shared rank factor cancels from κ when the
reward and teacher gradients have the same rank-dependent norm scaling. Together with weak token
correlations, this supplies the approximation developed in Appendix O. A common LoRA subspace
alone does not enforce equal scaling for two different directions.
Proposition 8 (Distillation Mode Determines κ). Holding gR fixed gives κon/κoff = ∥goff
D ∥/∥gon
D ∥.
For mode s ∈{on, off}, let Es = Ls
D −min Ls
D and assume positive local curvature bounds
λ−
s I ⪯∇2Ls
D ⪯λ+
s I. Then
κon
κoff
≥
s
λ−
offEoff
λ+
onEon
.
(33)
A large off-policy excess KL relative to the on-policy excess therefore raises this lower bound, with
the curvature ratio accounting for the different prefix distributions.
25


---

Proposition 9 (Conflict-Free Guarantee for M3-Select). With egn
D = 1{KDR(n) ≥0}gn
D, masking
gives ⟨egn
D, gn
R⟩= max{KDR(n), 0} ≥0 at every position.
Proposition 10 (Distillation as Implicit Regularization). In the matched-context isotropic quadratic
model LD(θT + ∆θ) = 1
2fD∥∆θ∥2, with fD > 0, constant gR, and ∆θ(0) = 0, hybrid gradient
flow satisfies
∆θH(T) = −(1 −α)gR
αfD
(1 −e−ηαfDT ),
∥∆θH(T)∥≤(1 −α)∥gR∥
αfD
.
(34)
If degradation is proportional to parameter deviation with a common coefficient, then δH ≤δR/(1+
αρreg), where ρreg = ηfDT/2. A finite rank-dependent reversal follows in the model when the
pure-RL degradation grows continuously without bound, the drowning penalty is bounded, ρreg is
bounded away from zero, and the initial reward gap is positive; Appendix AD gives this conditional
argument and the observed reversal at rank 128.
F
GRADIENT FOLDING, FIVE BRIDGES, AND UNIFIED FRAMEWORK: FULL
STATEMENTS
This section collects the formal statements of the propositions, theorems, and corollaries whose
proofs appear in subsequent appendix sections and whose summaries appear in Section 3.5 and the
method discussion.
Proposition 11 (Per-Position Conflict Decomposition). The per-position residual inner product has
the exact decomposition
(δn
D)⊤δn
R = −A(σn + ηn),
(35)
where σn = pn
S(ˆyn) −pn
T (ˆyn) measures student-teacher disagreement on the sampled token, and
ηn = (pn
T )⊤pn
S −∥pn
S∥2 captures cross-token probability redistribution. Under K(n, n) = λnI
with λn > 0, conflict (KDR(n) < 0) on positive-advantage trajectories (A > 0) occurs exactly
when σn + ηn > 0.
Proposition 12 (Asymmetric Harm from Magnitude Drowning). For gH = (1 −α)gR + αgD with
0 < α < 1 and ΦDR < 0, dividing the harmful cross-terms by their respective self-progress terms
gives
HR→D = (1 −α)|⟨gD, gR⟩|
α∥gD∥2
= (1 −α)κ|ΦDR|
α
,
HD→R = α|ΦDR|
(1 −α)κ.
(36)
Hence HR→D/HD→R = [(1−α)/α]2κ2. At α = 1/2 and κ = 3,400, the ratio is about 1.16×107
(Eq. 6).
Proposition 13 (Cosine Inversion in On-Policy OPSD). For binary rewards with baseline p ∈(0, 1),
write gR = −(r −p)s, where s = ∇θ log πθ(ˆy), and let q = ⟨s, gD⟩/(∥s∥∥gD∥). If E[q | r =
0] + E[q | r = 1] > 0, then
E[cos(gR, gD) | r = 0] −E[cos(gR, gD) | r = 1] = E[q | r = 0] + E[q | r = 1] > 0.
(37)
Thus the conditional score–teacher alignment determines whether the teacher aligns more strongly
with reward updates on incorrect trajectories.
Proposition 14 (Folding–Drowning Coupling). Let γ be the cancellation rate and κraw the ratio
formed from the root-mean-square trajectory gradient. Then κeff = √1 −γ κraw. Under the con-
centrated two-class model with equal class-mean score norms and overlap cosine coverlap,
γ = 1 −2p(1 −p)(1 −coverlap).
(38)
The overlap contribution 2p(1 −p)coverlap peaks at balanced accuracy when coverlap > 0. Reduced
effective magnitude and positive cross-signal alignment jointly favor hybridization whenever both
conditions hold.
Proposition 15 (Distillation Boundary Bound). Let ∆DR = (∥gD∥2 + ∥gR∥2)/∥gD + gR∥2 and
∆τ = (1 −γ)−1. If κraw ≫√∆τ, then
∆DR ≤1 + 2|ΦDR|√∆τ
κraw
+ O
 ∆τ
κ2raw

.
(39)
The index BD = αmax∆DR is the cross-signal analogue of BS. Its reference value ∆DR = 1
separates negative from positive gradient cross-terms.
26


---

Proposition 16 (NTK-Guided Token Masking (Bridge 4)). For un = Jnˆδn
D, vn = Jnˆδn
R, and
S−= {n : ⟨un, vn⟩< 0}, the average local reward-projection gain over uniform M3-Norm is
∆ΠR = αmax
N
X
n∈S−
 ∥vn∥2 −⟨un, vn⟩

.
(40)
It is positive when S−̸ = ∅and αmax > 0. For unit parameter-space directions this reduces to
∆ΠR = αmaxCNTK(1 + |¯c−|), where |¯c−| is the mean absolute cosine on S−.
Proposition 17 (RL Projection Under M3-Norm). For unit directions, bilinearity gives ΠR(α) =
⟨αˆgD + (1 −α)ˆgR, ˆgR⟩= 1 −α + αΦDR. With the same reward-norm reference scale, norm-
balanced GradNorm has projection (1 + ΦDR)/(1 + κ). Hence their projection ratio is Θ(κ) when
both cosine factors remain positive and bounded away from zero.
Theorem 3 (Unified NTK Learning Efficiency). Define the composite efficiency index by
ηeff =
(1 −γ)
| {z }
ηexplore=1/∆τ
[1 −α + αΦDR]
|
{z
}
ηhybrid
.
(41)
Both factors have NTK decompositions: cross-trajectory interactions determine γ, and cross-signal
interactions determine ΦDR. Under a normalized hybrid step, first-order reward progress relative
to the root-mean-square reward-gradient scale is instead proportional to √1 −γ ηhybrid. For fixed
α, the composite index decreases with accuracy only when −γ′(p)ηhybrid + (1 −γ)αΦ′
DR(p) ≤0.
Corollary 4 (CNTK-Based α Scheduling). The hard gate allocates the mean coefficient αeff =
αmax(1 −CNTK) and admits the fraction 1 −eCNTK of absolute cross-signal NTK mass. Under the
fixed-cohort population model of Proposition 21, eCNTK increases with accuracy, so this admitted
mass fraction decreases automatically. The count-based coefficient budget follows the unweighted
conflict rate.
Full proofs of the above statements appear in the dedicated appendix sections below, together with
the PCGrad degeneracy analysis in Appendix M. The five bridges connecting exploration boundary
theory to hybrid dynamics are: (1) shared NTK geometry (Theorem 3); (2) cancellation-modulated
drowning (Proposition 14); (3) the distillation boundary (Proposition 15); (4) token masking (Propo-
sition 16); and (5) conflict-rate–driven α scheduling (Corollary 4).
G
EXTENDED DISCUSSION
This appendix collects the optimality and benefit-condition results referenced in the Conclusion,
together with the practitioner’s decision tree that operationalises them.
Corollary 5 (Local Projection and Teacher Allocation under Low Accuracy). Assume the unit-
direction setting of Proposition 16 and the fixed conditional alignments of Corollary 2. For p < p∗,
trajectory alignment is positive in expectation, and M3-Select weakly improves the average local
reward projection over M3-Norm. At the same ceiling αmax ∈(0, 1), its mean teacher allocation
relative to naive mixing’s norm-based teacher share is
αmax(1 −CNTK)
αmax/[αmax + (1 −αmax)κ] = (1 −CNTK)[αmax + (1 −αmax)κ].
(42)
Thus normalization restores teacher allocation at large κ, and the gate retains only positions with
nonnegative local compatibility.
Proof. The two projection claims follow from E[Φτ] = (1−p)µ0+pµ1 > 0 and ΠSelect
R
−ΠNorm
R
=
αmaxCNTK(1+|¯c−|) ≥0. The allocation identity follows by dividing the admitted mean coefficient
by naive mixing’s norm-based teacher share.
Corollary 6 (Hybrid Benefit Condition). At the same step size η ≤c/L, the M3-Norm bound of
Theorem 2, with c = 1−α+αϕ∗> 0, is strictly smaller than its pure-GRPO instance exactly when
Lησ2
R(1 −α + ϕ∗) > 2(1 −ϕ∗)(L0
R −L∗
R)
ηT
.
(43)
For fixed η, positive variance, and ϕ∗> −(1 −α), this condition holds for sufficiently large T.
27


---

Proof. Let D = 2(L0
R −L∗
R)/(ηT) and V = Lησ2
R. Comparing the two bounds and using α > 0
gives the single equivalence chain
D + (1 −α)2V
c
< D + V ⇐⇒V [c −(1 −α)2] > D(1 −c)
⇐⇒V (1 −α + ϕ∗) > D(1 −ϕ∗),
(44)
which is the stated condition.
Remark 4 (Practitioner’s Decision Tree). Estimate ¯κ from a 10-step probe run on the target cell,
then apply the following gate-selection rule (thresholds are the ones used throughout this work;
κ∗=5,000 is the catastrophic threshold of Remark 1).
(1) Catastrophic regime, ¯κ > κ∗= 5,000: use M3-Select (hard gate). Empirically verified on
Llama-GSM8K (¯κ = 7,690) and Llama-SVAMP (¯κ ≈1.3×104), where every uniform-mixing
baseline collapses to reward < 0.05 while M3-Select survives (§5.3, §5.3); at these κ the soft-
gate rescue is seed-fragile (two of three Llama-GSM8K replicates collapse during training, §B.1),
making the hard gate the robust long-horizon choice.
(2) Stable regime, 1,000 < ¯κ ≤κ∗: use M3-Soft with the empirically effective sharpness β ≈1;
Proposition ?? describes its local variance sensitivity. Empirically verified on Qwen3-GSM8K
(¯κ = 3,404), Qwen3-SVAMP (¯κ ≈5,080, sitting essentially at the threshold), and every In-
ternLM/Qwen2.5 cell in this range. Hard masking over-prunes here (Remark 1); Soft-gate pre-
serves the beneficial synergy tokens the estimator misclassifies.
(3) Low-κ regime, ¯κ ≤1,000: naive uniform mixing (Hybrid, α≈0.5) is already sufficient; special-
ized gating offers diminishing marginal returns. This regime is not instantiated in our sweep—the
nearest cells are the ARC cells across architectures (¯κ ∈[1.4, 2.5]×103), where gentle M3-Soft
(β ∈{0.5, 1}, small αmax) still helps but the gap over Hybrid is within the seed-level standard
deviation (§B.1).
(4) On-policy vs. off-policy OPSD choice. Use the measured conditional alignments to locate the
synergy threshold (Corollary 2); the observed on-policy crossover is near p∗≈0.56. Compare
on- and off-policy validation curves when choosing the sampling mode.
(5) Conflict-rate override. If the observed conflict rate CNTK > 50% persists into training, switch
from M3-Soft to M3-Select regardless of ¯κ: the mass of Ω−tokens is large enough that the Soft
gate’s residual bias on Ω−dominates its variance reduction on Ω+.
Two calibration remarks. (i) The Hybrid→Soft boundary at ¯κ ≈1,000 is set by an empirical gate-
calibration heuristic, and is more forgiving than the Soft→Select boundary at κ∗: below it the Soft
gate is still safe, only unnecessary. (ii) In cells where dynamic-gating M3-Soft is at parity with static-
mixing Hybrid at the peak-training regime (InternLM-GSM8K is the canonical example), apply the
SWA post-step to reduce late-stage checkpoint fluctuations (Proposition ??, §B.3) before falling back
to Hybrid.
H
PROOF OF PROPOSITION 18: SPECTRAL CONFLICT BOUND
Proposition 18 (Spectral Conflict Bound). Let K = [K(n, m)]N
n,m=1 ∈RN|V|×N|V| be the global
token-level NTK, and let uD and uR stack the residuals δn
D and δn
R, respectively. Then
⟨gD, gR⟩≥λmin(K)∥uD + uR∥2 −λmax(K)∥uD −uR∥2
4N 2
,
(45)
⟨gD, gR⟩≤λmax(K)∥uD + uR∥2 −λmin(K)∥uD −uR∥2
4N 2
.
Thus local residual alignment alone does not determine the aggregate interaction.
28


---

Proof. Stacking the Jacobians gives J = [J1, . . . , JN] and K = J⊤J ⪰0. Polarization and the
Rayleigh bounds yield the single chain
⟨gD, gR⟩= N −2u⊤
DKuR
= (uD + uR)⊤K(uD + uR) −(uD −uR)⊤K(uD −uR)
4N 2
≥λmin(K)∥uD + uR∥2 −λmax(K)∥uD −uR∥2
4N 2
.
Interchanging the upper and lower Rayleigh bounds proves the second inequality.
Geometric reading.
Conflict occurs exactly when (uD −uR)⊤K(uD −uR) > (uD +
uR)⊤K(uD + uR): the difference field has more kernel-weighted energy than the sum field. This
depends on the residuals’ projections onto the kernel eigenspaces. If the trainable parameter dimen-
sion is below N|V|, then rank(K) ≤dim(θ) implies λmin(K) = 0, reducing the lower bound to
−λmax(K)∥uD −uR∥2/(4N 2).
I
PROOF OF PROPOSITION 11: PER-POSITION CONFLICT DECOMPOSITION
Proof. Substitution of the two residuals directly gives
(δn
D)⊤δn
R = −A(pn
S −pn
T )⊤(eˆyn −pn
S)
= −A

pn
S(ˆyn) −pn
T (ˆyn) + (pn
T )⊤pn
S −∥pn
S∥2
= −A(σn + ηn).
The residual identity is exact. Under K(n, n) = λnI with λn > 0, KDR(n) has the same sign, so a
positive-advantage trajectory conflicts precisely when σn + ηn > 0.
Interpretation.
σn isolates disagreement on the sampled token, whereas ηn = P
v[pn
T (v) −
pn
S(v)]pn
S(v) aggregates vocabulary-wide redistribution weighted by student confidence. The mea-
sured ratio |ηn|/|σn| ≈8.6 (Section 5) identifies redistribution as the larger contribution in our pilot
experiments.
J
PROOF OF THEOREM 2: M3-NORM CONVERGENCE GUARANTEE
Proof. Let Et condition on the history before the fresh reward-gradient sample, and write gt
R =
∇LR(θt). The population-norm-matched update is ˆgt
H = (1 −α)(gt
R + ξt) + α˜gt
D, where ˜gt
D is
conditionally deterministic, ∥˜gt
D∥= ∥gt
R∥, Etξt = 0, and Et∥ξt∥2 ≤σ2
R. At a stationary point set
˜gt
D = 0. For the uniform alignment lower bound ϕ∗, put c = 1 −α + αϕ∗> 0 and ¯gt
H = Etˆgt
H.
Then
⟨gt
R, ¯gt
H⟩= (1 −α)∥gt
R∥2 + α⟨gt
R, ˜gt
D⟩≥c∥gt
R∥2,
(46)
Et∥ˆgt
H∥2 = ∥¯gt
H∥2 + (1 −α)2Et∥ξt∥2 ≤∥gt
R∥2 + (1 −α)2σ2
R.
(47)
The last inequality uses ∥¯gt
H∥≤(1 −α)∥gt
R∥+ α∥˜gt
D∥= ∥gt
R∥. Smoothness, η ≤c/L, and
telescoping now give
EtLR(θt+1) ≤LR(θt) −η⟨gt
R, ¯gt
H⟩+ Lη2
2 Et∥ˆgt
H∥2
≤LR(θt) −ηc
2 ∥gt
R∥2 + Lη2
2 (1 −α)2σ2
R,
ηc
2
T −1
X
t=0
E∥gt
R∥2 ≤L0
R −ELR(θT ) + T Lη2
2
(1 −α)2σ2
R
≤L0
R −L∗
R + T Lη2
2
(1 −α)2σ2
R.
Dividing by ηcT/2 proves the rate. This argument applies to the stated population-normalized
update; the EMA scale in Algorithm 2 estimates its scale.
Remark 5 (Additional properties of the M3 update). At the minimum-norm coefficient α∗, the
convex-hull optimality condition gives ⟨gH(α∗), gi⟩≥∥gH(α∗)∥2 for i ∈{D, R}, establishing
29


---

first-order Pareto descent (Sener & Koltun, 2018). By comparison, naive mixing with α = 1
2 in-
creases distillation loss to first order exactly when ΦDR < −1/κ, since ⟨gH, gD⟩=
1
2rD2(1 +
κΦDR). The normalized reward projection instead follows Eq. 46. For an interior minimum-norm
coefficient, completing the square gives Eq. 26; at α = 1
2 its gap is (rR2 −rD2)2/(4∥gD −gR∥2).
A clipped boundary minimizer also contributes the corresponding one-sided linear term.
K
PROOF OF PROPOSITION 3: VARIANCE REDUCTION VIA DISTILLATION
Proof. In the conditional model above, ˆgt
H −Etˆgt
H = (1 −α)ξt, hence
Vart(ˆgt
H) = (1 −α)2Et∥ξt∥2 ≤(1 −α)2σ2
R.
Its contribution to the convergence bound is Lη(1 −α)2σ2
R/c. Relative to the pure-RL bound at the
same admissible step size, the noise-floor factor satisfies
(1 −α)2
c
< 1
⇐⇒
c −(1 −α)2 = α(1 −α + ϕ∗) > 0
⇐⇒
ϕ∗> −(1 −α).
For ϕ∗≥0 this factor is at most 1 −α. The variance contraction follows from the reward weight;
normalization additionally controls the mean projection through c.
L
JOINT PROOF OF THEOREM 1 AND PROPOSITION 2
We use the conditional, orthogonal-position model of the two statements. At each iteration, let un
be the mean reward contribution and dn its conditionally deterministic teacher counterpart, with
∥dn∥= ∥un∥. The reward noise ξn has Etξn = 0 and νn = Et∥ξn∥2. Contributions from
distinct positions lie in mutually orthogonal parameter subspaces, as in the exact block-diagonal
NTK model. The gates are fixed before this fresh noise is sampled. Suppressing t locally, define
αn = αmax1[⟨un, dn⟩≥0],
ρadm = |Ω+ ∪Ω0|
N
,
¯α = αmaxρadm = αmax(1 −CNTK).
(48)
Here u = N −1 P
n un = ∇LR(θt) and ˆgH = N −1 P
n[(1 −αn)(un + ξn) + αndn]. For a
zero-norm position set dn = 0 and cn = 0; it has zero reward-energy weight. At u = 0, all mean
contributions vanish and the projection bound holds directly.
Proof. The same decomposition yields both projection and variance, so it suffices to establish their
constants once. With wn = ∥un∥2/ P
m ∥um∥2, cn = ⟨un, dn⟩/∥un∥2, and ¯gH = EtˆgH, orthog-
onality gives
⟨u, ¯gH⟩=
1
N 2
X
n
[(1 −αn) + αncn]∥un∥2 = ρR,t∥u∥2,
ρR,t :=
X
n
wn[(1 −αn) + αncn],
(49)
ρR,t ≥1 −αmax
X
n∈Ω+∪Ω0
wn ≥1 −αmax > 0,
(50)
Vart(ˆgH) = Et



 1
N
X
n
(1 −αn)ξn



2
=
1
N 2
X
n
(1 −αn)2νn = vtσ2
R,t,
(51)
where σ2
R,t = N −2 P
n νn and vt = P
n(1 −αn)2νn/ P
n νn ≤1; use vt = N −1 P
n(1 −αn)2
when all νn = 0. Cross-position noise terms vanish by the subspace orthogonality. These identities
show why the aggregate projection uses reward-energy weights and the variance uses noise-energy
weights.
For equal position variances, the variance factor has the closed form
vt = (1−αmax)2ρadm +1−ρadm = 1−(2αmax −α2
max)ρadm = (1−¯α)2 +α2
maxρadm(1−ρadm).
(52)
This proves Proposition 2, including the correction due to heterogeneity of the gate.
For convergence, take uniform bounds ρR ≤ρR,t, vt ≤vα, and σ2
R,t ≤σ2
R. The local triangle
inequality gives ∥(1 −αn)un + αndn∥≤∥un∥, so orthogonality and Eq. 51 imply
Et∥ˆgt
H∥2 = ∥¯gt
H∥2 + Vart(ˆgt
H) ≤∥∇LR(θt)∥2 + vασ2
R.
(53)
30


---

Substituting Eqs. 49 and 53 into the smoothness-and-telescoping chain in Appendix J, with c re-
placed by ρR and (1 −α)2 by vα, gives for η ≤ρR/L,
1
T
T −1
X
t=0
E∥∇LR(θt)∥2 ≤1
ρR
2(L0
R −L∗
R)
ηT
+ Lηvασ2
R

.
(54)
The choice ρR = 1 −αmax is always valid in this model.
Comparison with uniform mixing.
For the same matched local directions, masking increases the
energy-weighted reward projection by αmax
P
n∈Ω−wn(1 −cn) ≥0. It also leaves more reward
noise than uniform mixing: vt ≥(1 −αmax)2. The convergence comparison therefore depends
on the ratio of variance to projection. Against pure RL, the variance contracts strictly whenever an
admitted position carries nonzero reward noise. These are reward-descent guarantees; at a rejected
position the pure reward direction can still oppose the teacher direction.
M
PCGRAD DEGENERACY IN THE ASYMMETRIC REGIME
Proposition 19 (PCGrad Magnitude Preservation). Let gD, gR be nonzero gradients with cosine
−1 < Φ < 0. Their symmetric PCGrad projections preserve the magnitude ratio:
∥g⊥
D∥
∥g⊥
R∥= ∥gD∥
∥gR∥= rD
rR
= 1
κ.
(55)
Proof. For (i, j) ∈{(D, R), (R, D)}, orthogonal projection gives
∥g⊥
i ∥2 =




gi −⟨gi, gj⟩
∥gj∥2 gj




2
= ∥gi∥2 −⟨gi, gj⟩2
∥gj∥2
= ri
2(1 −Φ2).
(56)
Canceling the common positive factor proves the ratio. Consequently, for 0 < α < 1, the norm-
based teacher share in αg⊥
D + (1 −α)g⊥
R is wD = α/[α + (1 −α)κ], approximately 0.03% at
α = 0.5 and κ = 3,400. At exact antiparallelity both projections vanish.
N
PROOF OF PROPOSITION 6: PHASE DELAY IN ADAPTIVE MIXING
Proof. Consider an alignment estimate with exponential smoothing ˆΦt+1 = ρˆΦt + (1 −ρ)Φt,
0 < ρ < 1. Suppose the input changes from Φ+ > 0 to Φ−< 0 at t1, with ˆΦt1 = Φ+. Solving the
recurrence and its zero-crossing condition in one chain gives
ˆΦt1+k = ρkΦ+ + (1 −ρ)Φ−
k−1
X
j=0
ρj = Φ−+ (Φ+ −Φ−)ρk,
(57)
ˆΦt1+k ≤0 ⇐⇒ρk ≤
|Φ−|
Φ+ −Φ−
⇐⇒k ≥log[(Φ+ −Φ−)/|Φ−|]
log(1/ρ)
.
(58)
Thus the first nonpositive estimate occurs after the ceiling of the last expression. A rule that re-
tains the high teacher weight while ˆΦ > 0 continues doing so during this delay, although the cur-
rent alignment is negative. For a boxcar average of width W, replacing k old observations gives
ˆΦ = Φ+ + k(Φ−−Φ+)/W, so the corresponding delay is ⌈WΦ+/(Φ+ −Φ−)⌉; equal transi-
tion magnitudes give approximately W/2. The result applies to loss-based schedules when their
smoothed control statistic follows this assumed alignment transition.
O
HEURISTIC DERIVATION FOR PROPOSITION 7: RANK-INVARIANCE OF κ
Heuristic derivation. Write the distillation gradient as gD = N −1 P
n dn. To isolate the role of
the LoRA subspace, suppose both signals share a rank-dependent second-moment factor sr > 0,
with E∥gR∥2 = aRsr, E∥dn∥2 = aDsr, and E⟨dn, dm⟩= 0 for n̸ = m. Here aR, aD > 0 are
rank-independent signal constants. Then
E∥gD∥2 =
1
N 2
X
n,m
E⟨dn, dm⟩= aDsr
N
,
s
E∥gR∥2
E∥gD∥2 =
r aR
aD
N.
(59)
31


---

The common factor cancels; concentration of the squared norms transfers this root-mean-square
ratio to typical observed ratios. For example, an isotropic LoRA model may give sr ∝Lr/h for
L adapted layers of width h. The cancellation depends on shared scaling and weak cross-token
correlations, which are the modeling assumptions behind this heuristic. The measured rank sweep,
r ∈{8, 16, 32, 64} and ¯κ = 3,476 ± 301 (CV 8.7%), supplies the empirical evidence in Remark 3;
the signal constants determine its absolute scale.
P
PROOF OF PROPOSITION 8: DISTILLATION MODE DETERMINES κ
Proof. For mode s ∈{on, off}, let Es = Ls
D(θ) −Ls
D(θ∗
s) be the excess distillation loss above its
local minimum. In an exact local quadratic model, Es = 1
2∆θ⊤
s Hs∆θs and gs
D = Hs∆θs, where
∆θs = θ −θ∗
s and Hs is positive definite on the active parameter subspace. Writing λ−
s , λ+
s for its
extremal eigenvalues yields
2λ−
s Es = λ−
s ∆θ⊤
s Hs∆θs ≤∆θ⊤
s H2
s∆θs = ∥gs
D∥2 ≤λ+
s ∆θ⊤
s Hs∆θs = 2λ+
s Es,
(60)
κon
κoff
= ∥goff
D ∥
∥gon
D ∥≥
s
λ−
offEoff
λ+
onEon
,
(61)
where the second line uses the same nonzero reward gradient in both modes.
The same in-
equality follows from local strong-convexity and smoothness bounds with the corresponding con-
stants.
If both losses have zero minimum and share a curvature matrix F, this specializes to
q
Doff
KL/Don
KL/
p
χ(F).
A nearly deterministic teacher on ground-truth tokens gives Doff
KL(n) ≈−log pS(y∗
n). On-policy
evaluation changes the prefix distribution, and its KL gap must be measured. Table 11 reports mean
teacher losses 0.098 on-policy and 0.810 off-policy, with median magnitude ratios 388 and 68,
respectively. These observations establish the cross-mode gap in the evaluated setting; the bound
explains how excess loss and curvature jointly control gradient magnitude.
Q
PROOF OF PROPOSITION 9: CONFLICT-FREE GUARANTEE
Proof. By Definition 1, gn
D = Jnδn
D, gn
R = Jnδn
R, and ⟨gn
D, gn
R⟩= KDR(n). With the gate treated
as a fixed coefficient during the update,

1[KDR(n) ≥0]gn
D, gn
R

= 1[KDR(n) ≥0]KDR(n) = max{KDR(n), 0} ≥0.
(62)
Positive norm rescaling preserves this sign, proving the token-level guarantee.
For comparison, if the aggregate normalized gradients have cosine ΦDR ≥0 and 0 < α < 1, their
mixture satisfies ⟨gH, ˆgR⟩= 1 −α + αΦDR > 0 and ⟨gH, ˆgD⟩= α + (1 −α)ΦDR > 0. The latter
projection changes sign at ΦDR = −α/(1 −α), compared with −α/[(1 −α)κ] for unnormalized
mixing. Cross-position interactions enter the aggregate condition through Proposition 20.
R
PROOF OF PROPOSITION 5: GRADNORM DEGENERACY
Proof. At a norm-balanced GradNorm equilibrium with equal target training rates and wD + wR =
1, the weighted norms coincide. Consequently,
wDrD = wRrR =⇒(w∗
D, w∗
R) = (κ, 1)
1 + κ,
(63)
gGN
H
=
rR
1 + κ(ˆgD + ˆgR),
∥gGN
H ∥≤2rR
1 + κ,
(64)
⟨gR, gGN
H ⟩= rR
2 1 + ΦDR
1 + κ
.
(65)
For an L-smooth reward loss, LR(θ)−LR(θ−ηgH) = η⟨gR, gH⟩+O(Lη2∥gH∥2). The first-order
reward decrease is therefore (1 + ΦDR)/(1 + κ) of the pure-RL decrease at the same learning rate.
Before equilibrium, comparable positive weights give teacher share wD/(wD + wRκ) = Θ(1/κ).
32


---

To compare with M3-Norm at a common reward scale, use rR[αˆgD + (1 −α)ˆgR]. Its reward
projection relative to GradNorm is
ΠM3
R
ΠGN
R
= (1 + κ)[1 −α + αΦDR]
1 + ΦDR
,
(66)
which is Θ(κ) when both cosine-dependent factors stay positive and bounded away from zero. At
α = 0.15, κ = 4,381, and near-zero cosine, this first-order ratio is about 3,725. It describes
the specified update scaling; learning-rate rescaling or a different weight optimizer changes the
comparison.
S
PROOF OF PROPOSITION 14: FOLDING–DROWNING COUPLING
Proof. Let M 2
R = G−1 P
i ∥Ai∇θ log π(yi | x)∥2 > 0. The definitions of cancellation and raw
magnitude immediately give
κeff = ∥gR∥
∥gD∥=
p
(1 −γ)M 2
R
∥gD∥
=
p
1 −γ κraw.
(67)
For the binary-outcome model, let p ∈(0, 1) be the correct fraction, ¯g± the class-mean score
gradients, and σr =
p
p(1 −p). The standardized advantages imply
gR = p1 −p
σr
¯g+ −(1 −p) p
σr
¯g−=
p
p(1 −p)(¯g+ −¯g−),
(68)
∥gR∥2 = p(1 −p)
 ∥¯g+∥2 + ∥¯g−∥2 −2⟨¯g+, ¯g−⟩

.
(69)
If score gradients concentrate at class means with common norm s, then M 2
R = s2 and
γ = 1 −2p(1 −p)(1 −coverlap) = 1 −2p(1 −p)
|
{z
}
averaging contribution
+
2p(1 −p)coverlap
|
{z
}
cross-class overlap contribution
.
(70)
For nonnegative overlap this yields γ ≥2p(1−p)coverlap; the overlap contribution peaks at p = 1/2
and decreases for p > 1/2. It is this contribution, rather than the full cancellation rate, that vanishes
as p →1 in the model.
Combining γ > 0 with the expected-alignment condition of Corollary 2 gives reduced effective
magnitude imbalance and positive expected alignment whenever that corollary’s low-accuracy con-
dition holds. As accuracy increases beyond 1/2, the overlap contribution decreases; Corollary 3
separately describes the decline in expected alignment. These are the two quantities tracked by the
accuracy-adaptive interpretation of Eq. 31.
T
PROOF OF PROPOSITION 20: AGGREGATE CONFLICT AS A CROSS-SIGNAL
NTK SUM
Proposition 20 (Aggregate Conflict as Cross-Signal NTK Sum). For nonzero aggregate gradients,
their cosine is the normalized sum of diagonal cross-signal NTK terms and cross-position interac-
tions. Dropping the latter gives the diagonal NTK approximation.
Proof. Define Ecross = N −2 P
n̸=m(δn
D)⊤K(n, m)δm
R . Using K(n, m) = (Jn)⊤Jm and Defini-
tion 1, the decomposition follows directly:
ΦDR =
1
N 2∥gD∥∥gR∥
X
n,m
(δn
D)⊤K(n, m)δm
R
(71)
= N −2 P
n⟨Jnδn
D, Jnδn
R⟩+ Ecross
∥gD∥∥gR∥
= N −2 P
n KDR(n) + Ecross
∥gD∥∥gR∥
.
(72)
The diagonal approximation is accurate to the extent that the normalized cross-position remainder
is small.
33


---

U
PROOF OF PROPOSITION 15: DISTILLATION BOUNDARY BOUND
Proof. Let c = ΦDR and q = √∆τ/κraw = 1/κeff, using 1 −γ = 1/∆τ and Proposition 14. For
q →0, the exact diversity identity and its expansion are
∆DR =
1 + κ2
eff
1 + κ2
eff + 2cκeff
=
1 + q2
1 + 2cq + q2 = 1 −2cq + O(q2)
(73)
≤1 + 2|c|√∆τ
κraw
+ O
 ∆τ
κ2raw

.
(74)
The expansion is uniform for c ∈[−1, 1] with q sufficiently small; its regime is κraw ≫√∆τ.
Hence BD = αmax∆DR = αmax[1 + O(q)].
For nonzero gradients with nonzero sum, the exact denominator also gives ∆DR < 1, = 1, or > 1
according as c > 0, = 0, or < 0. Thus unity marks the sign change of the cross-signal contribution
to ∥gD + gR∥2. Relative to pure RL, the separate condition for a smaller squared update norm is
∥gD∥2 + 2⟨gD, gR⟩< 0.
V
PROOF OF PROPOSITION 16: NTK-GUIDED TOKEN MASKING
Proof. Write un = Jnˆδn
D, vn = Jnˆδn
R, and S−= {n : KDR(n) < 0}. These are parameter-
space directions obtained by positive residual rescaling, so ⟨un, vn⟩has the sign of KDR(n). The
gate replaces αmaxun + (1 −αmax)vn by vn on S−and leaves the other positions unchanged.
Consequently, the mean local reward-projection gain is
∆ΠR = 1
N
X
n
⟨gsel,n
H
−gM3,n
H
, vn⟩
(75)
= αmax
N
X
n∈S−
⟨vn −un, vn⟩= αmax
N
X
n∈S−
 ∥vn∥2 −⟨un, vn⟩

> 0
(76)
whenever αmax > 0 and S−is nonempty. In the unit-direction model, ∥un∥= ∥vn∥= 1, this
specializes to
∆ΠR = αmax
N
X
n∈S−
(1 + |cos φn|) = αmaxCNTK(1 + |¯c−|),
(77)
where CNTK = |S−|/N and |¯c−| is the mean absolute parameter-space cosine on S−. The gain is
zero when S−is empty. Under the diagonal NTK model, cross-position inner products vanish, so
the aggregate projection onto N −1 P
n vn has the same sign, with gain ∆ΠR/N.
W
M3-SELECT PROJECTION AND CONVERGENCE GUARANTEES
Theorem 4 (M3-Select Projection Improvement over M3-Norm). Under the decoupled-position
model of Proposition 16, with 0 < αmax < 1, M3-Select satisfies:
(a) Reward projection: its aggregate update has at least the reward projection of uniform
M3-Norm, strictly larger whenever CNTK > 0.
(b) Teacher allocation: the sum of its teacher coefficients is the fraction 1 −CNTK of the
uniform allocation, entirely on positions with KDR(n) ≥0.
(c) Reward convergence: under the additional smoothness, matched-norm, and noise assump-
tions of Theorem 1, it satisfies the reward-stationarity bound in Eq. 54.
Proof. Part (a) is Proposition 16, using the positive rescaling from the mean local projection to the
aggregate reward projection. For part (b), summing the gate gives P
n αn = αmax(N −|S−|) =
Nαmax(1−CNTK), and every retained coefficient has nonnegative local compatibility. Part (c) is the
descent-and-telescoping argument of Theorem 1, applied to the same gate and reward objective.
34


---

X
NTK CONFLICT RATE DYNAMICS
Proposition 21 (Accuracy Dependence of the Population Weighted Conflict Rate). Let ¯
M ± > 0 be
the expected per-trajectory absolute cross-signal NTK masses on correct and incorrect trajectories,
and let eC± be the corresponding ratios of expected negative mass to expected absolute mass. If
these four quantities are fixed as accuracy p varies, the pooled population conflict rate is
eCNTK(p) = p ¯
M + eC+ + (1 −p) ¯
M −eC−
p ¯
M + + (1 −p) ¯
M −
.
(78)
It increases strictly with p if eC+ > eC−, interpolating between eC−at p = 0 and eC+ at p = 1.
Proof. Conditioning the expected negative and absolute masses on trajectory correctness gives the
displayed ratio. Differentiating and canceling the common terms yields
d eCNTK
dp
=
¯
M + ¯
M −( eC+ −eC−)
[p ¯
M + + (1 −p) ¯
M −]2 > 0.
(79)
The endpoint values follow by substitution. Exact sign masking retains the fraction 1 −eCNTK of
absolute NTK mass, which therefore decreases under the same assumptions. This mass fraction
differs from the position fraction 1 −CNTK used in αeff: a change in mass allocation need not
change the number of admitted positions.
Y
PROOF OF PROPOSITION 13: COSINE INVERSION IN ON-POLICY OPSD
Proof. For a sampled trajectory, write s = ∇θ log πθ(ˆy
| x), gR
= −As, and gD
=
N −1 P
n ∇θKL(pn
T ∥pn
S), where A = (r −b)/σr, 0 < b < 1, and σr > 0. Assume s and gD
are nonzero, and define the normalized score–distillation alignment q = ⟨s, gD⟩/(∥s∥∥gD∥). Then
cos(gR, gD) = −sign(A)q,
(80)
E[cos(gR, gD) | r = 0] −E[cos(gR, gD) | r = 1] = E[q | r = 0] + E[q | r = 1] > 0,
(81)
where the last inequality is the proposition’s conditional alignment assumption. In particular, pos-
itive conditional means of q give positive cosine on incorrect trajectories and negative cosine on
correct ones.
The inner product underlying this condition includes all token pairings:
⟨s, gD⟩= 1
N
X
m,n
⟨∇θ log pm
S (ˆym), ∇θKL(pn
T ∥pn
S)⟩.
(82)
Thus the assumption concerns gradient-weighted alignment, including cross-position interactions.
For an on-policy teacher, positive q means its descent direction reduces the sampled trajectory’s log
probability; changing the sign of the reward advantage reverses whether that change agrees with RL.
Table 11 reports the corresponding conditional cosines +0.14 and −0.11.
Z
PROOF OF THEOREM 3: UNIFIED NTK LEARNING EFFICIENCY
Proof. Let S2
R = G−1 P
i ∥g(i)
R ∥2 and gR = G−1 P
i g(i)
R . The definitions of diversity, folding, and
the normalized hybrid direction give the complete factorization
ηexplore = ∥gR∥2
S2
R
= ∆−1
τ
= 1 −γ,
(83)
ηhybrid = ⟨(1 −α)ˆgR + αˆgD, ˆgR⟩= (1 −α) + αΦDR,
(84)
ηeff := ηexploreηhybrid = (1 −γ)

(1 −α) + αΦDR

.
(85)
Both factors depend on NTK inner products. Writing si = ∇log π(ˆy(i) | x) and g(i)
R
= −Aisi
gives
G2∥gR∥2 =
X
i,j
AiAj⟨si, sj⟩=
X
i,j
AiAj
X
n,m
Kt(τi, n; τj, m),
(86)
35


---

while Proposition 20 expands ΦDR into within-position and cross-position signal interactions.
These two expansions establish the shared geometry.
The composite index uses the squared cancellation factor. For a unit-scale hybrid update, the actual
first-order RL decrease relative to the RMS single-trajectory scale is instead √1 −γ ηhybrid. At
fixed α, differentiation of the composite index yields
dηeff
dp
= −γ′(p)

(1 −α) + αΦDR(p)

+ (1 −γ(p))αΦDR
′(p).
(87)
Hence decreasing alignment lowers the hybrid factor; monotonicity of the product follows when
the displayed derivative is nonpositive. Under fixed conditional trajectory cosines µ0 > µ1, their
mixture has derivative µ1 −µ0, with zero crossing p∗= µ0/(µ0 −µ1) when µ0 > 0 > µ1.
This crossing characterizes the alignment factor, while the cancellation factor contributes separately
through γ′(p).
AA
PROOF OF PROPOSITION 4: OPTIMAL PER-POSITION MIXING
Proof. Write cn = cos φn. Valuing admitted distillation by ραn in units of RL projection gives the
separable allocation objective
J = 1
N
X
n

(1 −αn) + αncn + ραn

= 1 + 1
N
X
n
αn(cn −τ),
τ = 1 −ρ,
(88)
subject to 0 ≤αn ≤αmax. If the optional budget N −1 P
n αn ≤¯α is imposed, its multiplier λ ≥0
shifts each coefficient to cn −τ −λ. Maximizing these linear terms gives
α∗
n =



αmax,
cn > τ + λ,
0,
cn < τ + λ,
any feasible value in [0, αmax],
cn = τ + λ.
(89)
For ρ = 1 and a slack budget, choose the upper endpoint at ties to obtain α∗
n = αmax1[cn ≥0].
Under the isotropic within-position kernel assumption K(n, n) = λnI with λn > 0, KDR(n) =
λn⟨δn
D, δn
R⟩has the same sign, yielding the M3-Select gate.
For a noisy score ˆcn = cn + εn with logistic error of scale β−1, the expected hard allocation is
E[αmax1[ˆcn ≥0]] = αmax Pr(εn ≥−cn) = αmaxσ(βcn).
(90)
This is M3-Soft. As β →∞, it approaches the hard gate for cn̸ = 0 and assigns half the budget at
the indifferent point cn = 0.
AB
CONVERGENCE RATE COMPARISON: M3-NORM VS GRADNORM
Theorem 5 (Descent-Bound Comparison at a Common Gradient Scale). Let LR be L-smooth and
bounded below, with gR = ∇LR. Compare the norm-restored M3 direction hM3 = ∥gR∥[(1 −
α)ˆgR + αˆgD] with the norm-balanced GradNorm direction hGN = ∥gR∥(ˆgR + ˆgD)/(1 + κ).
Assume 0 ≤α ≤α0 < 1/2, ΦDR ≥−1+δ for fixed δ > 0, and a fixed magnitude ratio κ > 0. For
method j, let the stochastic update be hj + ξj with conditional mean E[ξj] = 0 and E∥ξj∥2 ≤s2
j.
Define
(cM3, bM3) = (1 −2α0, 1),
(cGN, bGN) =

δ
1 + κ,
2
1 + κ

.
(91)
For a common step size η ≤minj cj/(Lb2
j) and ∆R = LR(θ0) −inf LR,
1
T
X
t<T
E∥∇LR(θt)∥2 ≤2∆R
ηcjT + Lηs2
j
cj
.
(92)
When Lηs2
j/cj ≤ϵ/2, sufficient iteration budgets scale as O(∆R/(ηϵ)) for M3 and O((1 +
κ)∆R/(ηϵ)) for GradNorm.
36


---

Proof. Bilinearity and the triangle inequality give ⟨gR, hj⟩≥cj∥gR∥2 and ∥hj∥≤bj∥gR∥. Ap-
plying smoothness conditionally on the current iterate and using the step-size restriction yields the
single descent chain
Et[LR(θt+1)] ≤LR(θt) −η⟨gR, hj⟩+ Lη2
2 (∥hj∥2 + s2
j)
(93)
≤LR(θt) −η
 
cj −Lηb2
j
2
!
∥gR∥2 + Lη2s2
j
2
(94)
≤LR(θt) −ηcj
2 ∥gR∥2 + Lη2s2
j
2
.
(95)
Telescoping and dividing by ηcjT/2 proves the bound; taking T ≥4∆R/(ηcjϵ) gives the stated
sufficient budgets. Their optimization terms differ by a factor proportional to 1 + κ under the
specified common scale and step size.
AC
CONCENTRATION OF THE NTK CONFLICT RATE
Proposition 22 (Conflict-Rate Variance under Mixing). Let ιn = 1[KDR(n) < 0] and CNTK =
N −1 P
n ιn. Suppose | Corr(ιn, ιn+k)| ≤Cρe−βmixk whenever both variances are nonzero. Define
Dρ = 1 + 2Cρ/(eβmix −1). Then
Var(CNTK) ≤Dρ
4N ,
Pr[|CNTK −ECNTK| > t] ≤min

1, Dρ
4Nt2

.
(96)
For the mean over B independent length-N rollouts, with probability at least 1 −δ, the deviation is
at most
p
Dρ/(4BNδ).
Proof. Since Var(ιn) ≤1/4, the covariance expansion reduces to a geometric series:
Var
 
1
N
X
n
ιn
!
=
1
N 2
"X
n
Var(ιn) + 2
N−1
X
k=1
N−k
X
n=1
Cov(ιn, ιn+k)
#
(97)
≤
1
4N
"
1 + 2
N−1
X
k=1

1 −k
N

Cρe−βmixk
#
≤Dρ
4N .
(98)
Chebyshev’s inequality gives the tail bound; independence divides the variance by B. Using Dρ ≤
1 + 2Cρ/βmix, the illustrative setting N = 256, Cρ = 1, βmix = 0.5, and δ = 0.05 gives deviations
at most 0.313 for one rollout and 0.079 for B = 16.
AD
PROOF OF PROPOSITION 10: DISTILLATION AS IMPLICIT
REGULARIZATION
Proof. Consider matched teacher and student distributions evaluated on the same contexts, with
πθT = πT . The local KL expansion is LD(θT +∆θ) = 1
2∆θ⊤FD∆θ +o(∥∆θ∥2), where FD ⪰0
is the Fisher matrix. In the proposition’s isotropic quadratic model, FD = fDI with fD > 0, so
∇LD = fD∆θ and ⟨∇LD, ∆θ⟩= fD∥∆θ∥2: distillation provides a restoring direction.
With constant gR, initial condition ∆θ(0) = 0, and 0 < α < 1, solve the resulting linear flow in
one step:
˙
∆θH = −η[(1 −α)gR + αfD∆θH],
(99)
∆θH(T) = −(1 −α)gR
αfD
(1 −e−ηαfDT ),
∥∆θH(T)∥≤(1 −α)∥gR∥
αfD
.
(100)
Pure RL has ∆θR(T) = −ηTgR. If degradation is proportional to deviation with the same coeffi-
cient for both flows, setting x = ηαfDT and ρreg = ηfDT/2 gives
δH
δR
= (1 −α)1 −e−x
x
≤
1 −α
1 + x/2 ≤
1
1 + αρreg
.
(101)
37


---

The scalar inequality follows from (1 + x/2)(1 −e−x) ≤x for x ≥0; the difference has derivative
[1 −(1 + x)e−x]/2 ≥0 and vanishes at zero.
For the rank-dependent model G(r) = ϵdrown−[δR(r)−δH(r)], a finite crossing follows if ϵdrown >
0 is bounded, δR(r) is continuous and grows without bound, and ρreg(r) ≥ρ0 > 0. Indeed,
G(r) ≤ϵdrown −
αρ0
1 + αρ0
δR(r) −→−∞.
(102)
Together with a positive initial gap, continuity gives a crossing of this model. The reported positive
gap at r = 64 and negative gap at r = 128 locate the observed reversal between the tested ranks.
AE
PROOF OF PROPOSITION: LSGV VARIANCE AMPLIFICATION
Proof. Write u = gR, v = gD, µu = Eu, µv = Ev, and let the joint estimator covariance be
Ω/B, including its cross-signal blocks. Assume λ−I ⪯Ω⪯λ+I for fixed positive constants, 0 <
m ≤∥µv∥≤M, and |c0| ≤1−δc, where c0 = ⟨µu, µv⟩/(∥µu∥∥µv∥). Set ¯κ = ∥µu∥/∥µv∥≤1.
In the small-noise regime, with delta-method remainders negligible relative to the leading variance,
the cosine gradient, evaluated at (µu, µv) with ˆµj = µj/∥µj∥, and its squared norm are
∇uf = ˆµv −c0 ˆµu
∥µu∥
,
∇vf = ˆµu −c0 ˆµv
∥µv∥
,
(103)
∥∇f∥2 = (1 −c2
0)

1
∥µu∥2 +
1
∥µv∥2

,
f(u, v) = ⟨u, v⟩
∥u∥∥v∥.
(104)
Consequently, retaining the joint covariance throughout,
Var(c) = 1
B ∇f ⊤Ω∇f + o
∥∇f∥2
B

= Θ
1 −c2
0
B

1
∥µu∥2 +
1
∥µv∥2

= Θ
 1
B¯κ2

. (105)
This rate describes amplification while relative noise is small. The global bound Var(c) ≤1 contin-
ues to hold when that approximation ceases to apply.
For α = αmaxσ(βc), a second delta expansion and the Lipschitz constant αmaxβ/4 give, respec-
tively,
Var(α) = α2
maxβ2σ′(βc0)2 Var(c) + o(Var(c)),
(106)
Var(α) ≤min
α2
max
4
, α2
maxβ2
16
Var(c)

.
(107)
Static Hybrid has Var(α) = 0. To identify the corresponding contribution to update variance, let
a0 = αmaxσ(βc0), d = µv −µu, and δh0 = (1 −a0)(u −µu) + a0(v −µv). Linearizing
h = (1 −α)u + αv yields
Cov(h) ≃Cov(δh0) + Var(α)dd⊤+ Cov(δh0, α)d⊤+ d Cov(α, δh0).
(108)
A constant gate removes the gate fluctuation terms. When the gate error is uncorrelated with δh0,
the additional covariance is the positive semidefinite term Var(α)dd⊤; for a gate computed from
the same gradients, the displayed cross-covariances determine its net effect.
AF
PROOF OF PROPOSITION: CUMULATIVE CONFLICT BUDGET
Proof. At step s, let qs = N −1
s
P
n αn,s1[Ks
DR(n) < 0] measure retained conflicting teacher
weight, and define the scalar budget E(t) = P
s≤t κsqs. For naive mixing qs = αCNTKs; exact
sign masking gives qs = 0. If the time-averaged product stabilizes, the entire accumulation law is
E(t) = t

1
t
X
s≤t
κsqs

= t ¯d + o(t),
TΓ ≈Γ
¯d ,
(109)
where TΓ is the first crossing of a fixed budget threshold Γ in this model. For approximately constant
κ and conflict rate, ¯dnaive = α¯κ ¯CNTK. If imperfect masking retains weight qs ≈αeffδmask, then
under the same ¯κ,
EM3(t)
Enaive(t) ≈αeffδmask
α ¯CNTK
.
(110)
38


---

The illustrative values αeff = 0.09, δmask = 0.05, α = 0.5, and ¯CNTK = 0.40 give 0.0225, or about
44 times slower budget accumulation.
This calculation concerns retained conflict. A parameter deviation formed from conflict updates
obeys ∥P
s ηsds∥≤P
s ηs∥ds∥, so connecting the scalar budget to reward collapse requires a
model of update directions and a collapse threshold. The reported ordering—Hybrid at step 221,
GradNorm at 358, GRPO at 412, and no M3 collapse through step 500—is an experimental obser-
vation. In particular, pure GRPO has qs = 0 and its collapse is governed by dynamics outside this
conflict budget.
AG
SUPPLEMENTARY DETAILS FOR THE MAIN ANALYSIS AND METHOD
AG.1
DISTILLATION DIVERGENCE AND LOCAL RESIDUAL MODEL
For Dλ(p∥q) = λKL(p∥m) + (1 −λ)KL(q∥m), with m = λp + (1 −λ)q, strictly positive
distributions satisfy
Dλ(p∥q)
λ
→KL(p∥q)
(λ →0),
Dλ(p∥q)
1 −λ
→KL(q∥p)
(λ →1).
The experiments use the symmetric point λ = 1/2. For an unclipped token near student–teacher
agreement,
∇znDλ(pn
T ∥pn
S) = λ(1 −λ)(pn
S −pn
T ) + O(∥pn
S −pn
T ∥2).
The main analysis absorbs the leading constant into the teacher update scale and uses the local
forward-KL residual δn
D = pn
S −pn
T . Tokens above the clipping threshold have zero distillation
gradient. This is a local approximation, not an identity for arbitrary distributions.
For completeness, the GRPO group statistics in Section 3.1 are ¯r = G−1 P
i r(i) and σ2
r =
G−1 P
i(r(i) −¯r)2. Their values and the sampled responses are held fixed during differentiation.
AG.2
GRADIENT GEOMETRY AND MAGNITUDE ASYMMETRY
The NTK sums in Eqs. 7–8 form the gradient Gram matrix

∥gD∥2
⟨gD, gR⟩
⟨gR, gD⟩
∥gR∥2

.
Its diagonal entries measure gradient strength; its off-diagonal entries measure interaction. The
residual quadratic sums are kernel energies, equal to squared gradient norms up to the common
factor N −2. Together, the entries determine the first-order loss changes for given α and η. The
normalized score ΦDR retains direction but removes magnitude.
For 0 < α < 1 and ΦDR < 0, normalize each harmful cross-effect in Eq. 6 by the corresponding
objective’s own descent term. The ratios are
qD = 1 −α
α
κ|ΦDR|,
qR =
α
1 −α
|ΦDR|
κ
,
qD
qR
=
1 −α
α
2
κ2.
Thus the relative asymmetry scales as Θ(κ2) for fixed interior mixing weights. A shrinking teacher
residual can increase κ when the reward gradient remains substantial, but the reward residual can
also vanish when A = 0. Both residual directions and the kernel affect their parameter-gradient
norms.
Proposition 7 reports the observed LoRA-rank and model-scale comparisons, and Ap-
pendix F develops the extended connections.
AG.3
MINIMUM-NORM GLOBAL MIXING
For gH(α) = αgD + (1 −α)gR, let rD = ∥gD∥, rR = ∥gR∥, and ΦDR = ⟨gD, gR⟩/(rDrR),
assuming nonzero gradients. The minimum-norm mixture is the point closest to zero on the segment
joining the two gradients. Minimizing ∥gH(α)∥2 over [0, 1] gives, for gD̸ = gR,
α∗= clip[0,1]

r2
R −ΦDRrDrR
r2
D + r2
R −2ΦDRrDrR

,
(111)
39


---

The clipping operation restricts the unconstrained minimizer to [0, 1]. If the gradients coincide, every
coefficient produces the same update. For κ = rR/rD ≫1, 1 −α∗= O(κ−1). This direction-
dependent criterion need not equal the norm-balancing choice α = κ/(1+κ) in Section 4.1. Neither
global coefficient can remove teacher contributions only at conflicting positions. The quadratic gap
around α∗is given in Eq. 26.
AG.4
NORMALIZATION AND CONTINUOUS-GATE LIMITS
The full-gradient normalization principle is
gMA
H (α) = ¯s[αˆgD + (1 −α)ˆgR],
ˆgD = gD/∥gD∥,
ˆgR = gR/∥gR∥,
for nonzero gradients, with ¯s an EMA-smoothed update scale. Algorithm 2 implements normal-
ization on residuals before applying the Jacobians. This preserves compatibility signs but need not
equalize parameter-gradient norms. The population-norm-matched convergence model is therefore
distinct from the residual-normalized implementation.
For M3-Soft, β →∞recovers hard selection wherever KDR(n)̸ = 0. At an exact zero score the
weight remains αmax/2, whereas M3-Select admits it at weight αmax. As β →0, all positions
receive αmax/2. The mean teacher weight is αeff = N −1 P
n αn. Larger β makes selection sharper
but increases sensitivity near zero alignment; Appendix B.3 evaluates this trade-off.
AG.5
CONDITIONAL VARIANCE AND GLOBAL REBALANCING
In Theorem 1, ρR lower-bounds the mean reward projection weighted by each position’s squared
mean reward-gradient norm; ρR = 1 −αmax is admissible. The noise factor vα uniformly bounds
the noise-energy-weighted mean of (1 −αn)2. The effective step size includes the update scale,
estimated in the algorithm by an EMA.
For equal position noise variances, admitted fraction ρadm, and ¯α = αmaxρadm, the exact factor is
vt = (1 −ρadm) + ρadm(1 −αmax)2 = (1 −¯α)2 + α2
maxρadm(1 −ρadm).
Rejected positions retain the full reward-noise contribution, while admitted positions reduce it.
These statements condition on the history before fresh reward noise and treat the teacher and gates
as fixed. Extra variability in estimated gates or teacher directions is not included.
Remark 6 (Limitations of Global Gradient Rebalancing). Global rebalancing does not select
teacher contributions by position. For symmetric two-gradient projections with −1 < ΦDR < 0,
PCGrad (Yu et al., 2020) removes conflicting components but preserves the norm ratio (Ap-
pendix M). At κ = 3400 and equal mixing weights, the teacher accounts for approximately 0.03%
of the sum of weighted gradient norms. Under the equal-target-rate, unit-sum weighting model of
Proposition 5, GradNorm (Chen et al., 2018) balances norms with reward coefficient 1/(1 + κ),
reducing its step scale by order 1/κ. M3 combines residual normalization with token-level control
of the teacher signal.
40
