---
title: 'Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?'
id: scaling-laws-vs-model-architectures-how-does-inductive-bias-influence-scaling
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-token-budget-and-effect-size
created: '2026-10-27T14:01:55.877568Z'
updated: '2026-10-27T19:00:31.720275Z'
source: https://arxiv.org/abs/2207.10551
source_domain: arxiv.org
fetched_at: '2026-10-27T14:01:55.876944Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Scaling Laws vs Model Architectures:'
raw_file: raw/scaling-laws-vs-model-architectures-how-does-inductive-bias-influence-scaling.pdf
doi: arXiv:2207.10551
citation_count: 142
venue: Conference on Empirical Methods in Natural Language Processing
---

Scaling Laws vs Model Architectures:
How does Inductive Bias Inﬂuence Scaling?
Yi Tay∗
Mostafa Dehghani∗
Samira Abnar
Hyung Won Chung
William Fedus
Jinfeng Rao
Sharan Narang
Vinh Q. Tran
Dani Yogatama†
Donald Metzler
Google Research & DeepMind†
{yitay,dehghani}@google.com
Abstract
There have been a lot of interest in the scal-
ing properties of Transformer models (Kaplan
et al., 2020).
However, not much has been
done on the front of investigating the effect
of scaling properties of different inductive bi-
ases and model architectures.
Do model ar-
chitectures scale differently? If so, how does
inductive bias affect scaling behaviour? How
does this inﬂuence upstream (pretraining) and
downstream (transfer)? This paper conducts
a systematic study of scaling behaviour of ten
diverse model architectures such as Transform-
ers, Switch Transformers, Universal Trans-
formers, Dynamic convolutions, Performers,
and recently proposed MLP-Mixers. Via ex-
tensive experiments, we show that (1) archi-
tecture is an indeed an important considera-
tion when performing scaling and (2) the best
performing model can ﬂuctuate at different
scales. We believe that the ﬁndings outlined in
this work has signiﬁcant implications to how
model architectures are currently evaluated in
the community.
1
Introduction
There have been a lot recent interest in the scaling
properties of Transformer models (Kaplan et al.,
2020; Hernandez et al., 2021; Bahri et al., 2021;
Henighan et al., 2020; Tay et al., 2021b; Abnar
et al., 2021). However, not much is understood
about the scaling properties of different inductive
biases imposed by model architectures. Improve-
ments at a a speciﬁc scale (compute, size etc) are
often assumed to transfer to different scales and
compute regions (So et al., 2019; Choromanski
et al., 2020; Lan et al., 2019; Dehghani et al., 2018)
and new research is often presented in a point-wise
fashion with respect to scale. In short, it is not un-
common for new methods to be presented with data
points at very speciﬁc or limited compute regions
∗Yi and Mostafa contributed equally. Samira is now at
Apple.
(e.g., base size). We believe that understanding the
interaction between architecture and scaling laws
is crucial as designing models that perform well at
diverse scales will likely have signiﬁcant impact.
This paper is an attempt to understand the ef-
fect of inductive bias (architecture) on scaling laws
of language models. To this end, we pre-train and
ﬁnetune over ten diverse model architectures across
multiple compute region and scales (e.g., from 15M
to 40 Billion parameters). In total, we pre-train and
ﬁnetune over 100 different models of different ar-
chitectures and sizes and present insights and chal-
lenges at scaling these ten diverse architectures.
We consider a broad spectrum of models in
our extensive experiments. Concretely, we con-
sider several well-established Transformer vari-
ants (Vaswani et al., 2017) such as Evolved Trans-
former (So et al., 2019), Universal Transformers
(Dehghani et al., 2018) and Switch Transformers
(Fedus et al., 2021). We also consider lightweight
models such as ALBERT (Lan et al., 2019) and/or
efﬁcient Transformers (Tay et al., 2020) such as
Performer (Choromanski et al., 2020) and Funnel
Transformers (Dai et al., 2020). In our comparison,
we are also interested in ﬁnding out if general im-
provements to the Transformer architectures such
as Mixture-of-Softmax (Yang et al., 2017) and/or
Gated Linear Units (Dauphin et al., 2017; Shazeer,
2020) inﬂuence the scaling behaviour of models.
Finally, we also evaluate models outside the fam-
ily of Transformers including Lightweight convo-
lutions (Wu et al., 2019), Dynamic convolutions
(Wu et al., 2019) and the recently proposed MLP-
Mixers (Tolstikhin et al., 2021). Figure 1 illustrates
an overview about the experiments we run.
We also note that scaling these models is not as
straightforward as it seems, i.e., there are intricate
details of scale that are intertwined with architec-
tural choices which we study in detail in this pa-
per. For example, a distinct feature of Universal
Transformers (and ALBERT) is parameter sharing.
arXiv:2207.10551v1  [cs.LG]  21 Jul 2022


---

1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
NL2
NL4
NL8
NL12
NL16
NL24
NL32
NL36
NL48
FF1K
FF2K
FF6K
FF9K
FF12K
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
GLU NL2
GLU NL4
GLU NL8
GLU NL16
GLU NL24
GLU NL32
GLU NL36
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
LConv FF6K
LConv FF9K
LConv FF12K
LConv NL2
LConv NL4
LConv NL8
Lconv NL16
LConv NL24
LConv NL36
DConv Tiny
DConv Small
DConv Base
DConv Large
Synthesizer Small
Synthesizer Base
Synthesizer Large
SynthesizerN Tiny
SynthesizerN Small
SynthesizerN Base
SynthesizerN Large
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Funnel FF6K
Funnel FF9K
Funnel FF12K
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Switch NL8
UT Tiny
UT Small
UT Base
UT Large
UT NR1
UT NR2
UT NR4
UT NR5
UT NR6
UT FF16K
UT FF32K
Albert Small
Albert Base
Albert Large
Albert FF6K
Albert FF9K
Albert FF12K
Albert NL2
Albert NL4
Albert NL8
Albert NL16
Albert NL24
Albert NL36
Albert NH8
Albert NH16
Albert NH24
Albert NH32
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Evolved FF6K
Evolved FF9K
Evolved FF12K
Evolved NL2
Evolved NL4
Evolved NL8
Evolved NL16
Evolved NL24
Evolved NL36
Evolved NH 8
Evolved NH 16
Evolved NH 24
Evolved NH 32
Performer Tiny
Performer Small
Performer Base
Performer Large
Performer PK Small
Performer PK Base
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
MoS- NL4
MoS- NL8
MoS- NL16
MoS- NL24
MoS- NL32
MoS- NL36
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
(a) Upstream: Negative Log-Perplexity
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
NL2
NL4
NL8
NL12
NL16
NL24
NL32
NL36
NL48
FF1K
FF2K
FF6K
FF9K
FF12K
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
GLU NL2
GLU NL4
GLU NL8
GLU NL16
GLU NL24
GLU NL32
GLU NL36
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
LConv FF6K
LConv FF9K
LConv FF12K
LConv NL2
LConv NL4
LConv NL8
Lconv NL16
LConv NL24
LConv NL36
DConv Tiny
DConv Small
DConv Base
DConv Large
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Funnel FF6K
Funnel FF9K
Funnel FF12K
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Switch NL8
UT Tiny
UT Small
UT Base
UT Large
UT NR1
UT NR2
UT NR4
UT NR5
UT NR6
UT FF16K
UT FF32K
Albert Small
Albert Base
Albert Large
Albert FF6K
Albert FF9K
Albert FF12K
Albert NL2
Albert NL4
Albert NL8
Albert NL16
Albert NL24
Albert NL36
Albert NH8
Albert NH16
Albert NH24
Albert NH32
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Evolved FF6K
Evolved FF9K
Evolved FF12K
Evolved NL2
Evolved NL4
Evolved NL8
Evolved NL16
Evolved NL24
Evolved NL36
Evolved NH 8
Evolved NH 16
Evolved NH 24
Evolved NH 32
Performer Tiny
Performer Small
Performer Base
Performer Large
Performer PK Small
Performer PK Base
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
MoS- NL4
MoS- NL8
MoS- NL16
MoS- NL24
MoS- NL32
MoS- NL36
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
(b) Downstream: Accuracy
Figure 1: An overview compute-performance (FLOPs vs performance) plot of all the diverse models and archi-
tectures we pretrained and ﬁnetuned in this study. Colors represent different model architectures and size of the
circles represent the size of the model (parameters).
Hence, compared with standard Transformers, this
architectural choice signiﬁcantly warps the scaling
behaviour not only with respect to performance
but also amongst compute metrics such as FLOPs,
speed and number of parameters (Dehghani et al.,
2021a). Conversely, models such as Switch Trans-
formers are on the other end of the spectrum with
an uncommon relationship between FLOPs and
number of parameters, i.e., they have high parame-
ter to FLOPs ratio. This difﬁculty makes navigating
this landscape challenging.
Our Contributions and Insights
The key con-
tributions of this paper are as follows:
• For the ﬁrst time, we derive scaling laws for
different inductive biases and model architec-
tures. We ﬁnd that this scaling coefﬁcient dif-
fers greatly from model to model. We believe
this is an important consideration in model
development. It turns out that amongst all
ten architectures that we consider, the vanilla
Transformer has the best scaling behaviour,
even if its absolute performance at each com-
pute region is not the greatest.
• We observe that models that operate well in
one compute-scale region is not necessarily
the best in another compute-region. Moreover,
we ﬁnd that certain models have difﬁculty scal-
ing despite performing decently (comparably)
at lower-compute regions. This has implica-
tions, since it is difﬁcult to get the fulll picture
of a model’s scalability with pointwise com-
parisons at a certain compute-region.
• We ﬁnd that when it comes to scaling differ-
ent model architectures, upstream pre-training
perplexity might not correlate well with down-
stream transfer. Hence, the underlying archi-
tecture and inductive bias is also crucial for
downstream transfer.
• We highlight the difﬁculties of scaling with
certain architectures and show that some mod-
els do not scale (or scale with a negative trend).
We also ﬁnd concerning trends where linear-
time attention models such as Performer strug-
gle with scaling up.
2
Related Work
Kaplan et al. (2020) studied empirical scaling laws
of the decoder-only Transformer language mod-
els. They focused on the standard left-to-right lan-
guage modeling objective with the cross-entropy
loss as the performance metric. One of the main
ﬁndings is that the loss scales as a power-law with
three major characteristics of the model training:
model size, dataset size and the training compute.
Another somewhat surprising ﬁnding is that the
model shapes such as width or depth of the Trans-


---

former network have minimal effects on the cross-
entropy loss for a wide range of scales. Subsequent
works (Henighan et al., 2020; Hernandez et al.,
2021) made similar conclusions for autoregressive
generative modeling and for transfer learning, re-
spectively. This ﬁnding is also generally supported
by (Tay et al., 2021b) but discrepancies were found
for the gap between pretraining and ﬁnetuning -
highlighting the fact that observing downstream
performance of large language model is indeed im-
portant. In (Tay et al., 2021b), the effect of depth
was unusually pronounced for downstream perfor-
mance.
Raffel et al. (2019) studied the effect of pre-
training objectives, model structures (e.g., encoder-
decoder, decoder-only), pre-training dataset size
and training strategy on the transfer learning. They
showed that the downstream performance mono-
tonically increases with the model scale (from 60M
to 11B parameters). While they studied several
model structures, the Transformer implementation
is mostly the same as the original Transformer
by Vaswani et al. (2017). Conneau et al. (2020);
Goyal et al. (2021) scaled-up multilingual encoder-
only architectures up to 11B parameters while
maintaining the original Transformer implemen-
tation. They found that scaling the model improves
its cross-lingual ability. Fedus et al. (2021) scaled
a sparse model based on Mixture of Experts (MoE)
models up to trillion parameters.
While previous studies have repeatedly shown
the beneﬁts of scale for language understanding
tasks for both dense and sparse Transformers and
cross-lingual abilities, all of these used the same
Transformer implementation within each studies.
With a plethora of improved Transformer architec-
tures proposed in the literature, it is timely to inves-
tigate which of these improved architecture has the
best scaling properties. The main goal of this pa-
per is to systematically study how inductive biases
imposed by these Transformer variants affect the
scaling behavior in a shared software and hardware
settings. This is in similar spirit to (Narang et al.,
2021) that studies the impact of architectures on
performance. Our analysis extends that of (Narang
et al., 2021) to the model scale axis.
3
Methods
This section outlines our experimental setup.
3.1
Models
This section describes the models we evaluate
in our experiments. Our models are largely im-
plemented in a sequence to sequence framework
(Sutskever et al., 2014) following the convention of
T5 (Raffel et al., 2019). Encoder-decoder models
are a natural choice for this experimentation be-
cause they can universally express both encoding
and decoding tasks.
Transformer Variants
We consider several stan-
dard Transformer variants.
• Transformers (Vaswani et al., 2017) - The
basic vanilla Transformer architecture. Our
basic setup considers the T5-style of Trans-
formers (Raffel et al., 2019), which largely
follows the vanilla Transformer except that
it uses relative attention instead of sinusoidal
position embeddings and pre-layer normaliza-
tion, i.e. layer normalization is applied before
each sublayer.
• Evolved Transformers (So et al., 2019) - A
transformer architecture learned via AutoML.
The architecture comprises of convolutions
and attention. We scale Evolved Transformers
following the same pattern as vanilla Trans-
formers.
• Universal Transformers (UT) (Dehghani
et al., 2018) - A Transformer architecture with
shared parameters and recurrent-like com-
putation for transform layers. Scaling UTs
are challenging because of parameter shar-
ing. While we are able to also increase dFF
or dmodel, the increase in parameters is of
magnitude Nlayers than standard Transform-
ers. Another axis of exploration is to scale r
the number of repeated computation at each
UT layer - this increases computation (number
of FLOPs) but does not increase the parameter
size of the model.
• Switch Transformer (Fedus et al., 2021) -
a sparsely activated mixture-of-experts archi-
tecture. The Sparse Transformer is another
model with an unusual relationship between
number of parameters and compute. When
we scale this model uniformly, the number of
parameters easily reaches the ballpark of 40B.


---

Efﬁcient Transformer Variants
These class of
models are mainly concerned at reducing computa-
tional costs, memory usage, or parameter count of
models.
• Performer (Choromanski et al., 2020) - A lin-
ear time attention model using generalizable
kernel attention. For simplicity, we adopt the
relu kernel variant for our experiments. We
scale Performer in the similar fashion (i.e.,
uniform scaling) as vanilla Transformers.
• Funnel Transformer (FT) (Dai et al., 2020)
A Transformer architecture that downsamples
the input sequence across the layer stack. Our
implementation uses FT only in the encoder
and reverts to vanilla Transformer in the de-
coder following Narang et al. (2021).
• ALBERT (Lan et al., 2019) - A lightweight
transformer architecture that shares parame-
ters across all layers and factorizes the em-
bedding and output softmax layers. For our
seq2seq ALBERT, we also share the weights
of encoder and decoder.
General Improvements
We consider general
improvements that are not necessarily tied to Trans-
formers. We select candidates that have shown to
do well in Narang et al. (2021).
• Mixture of Softmaxes (Yang et al., 2017) -
A transformer architecture adopting the MoS
method at the Softmax layer.
• Gated Linear Units with GeLU (GLU-
Transformer) - Replacing position-wise feed-
forward-networks in Transformers with Gated
Linear Units (Dauphin et al., 2017).
Non-Transformer Architectures
We are inter-
ested in the scaling behaviour of non-Transformer
based architectures such as convolutions and/or
mixer architectures.
• Lightweight Convolutions (Wu et al., 2019) -
Lightweight depthwise convolutions that have
shown promise over Transformer architec-
tures.
• Dynamic Convolutions (Wu et al., 2019) -
An extension of the Lightweight Convolution
to create time-dependent kernels.
• MLP-Mixers (Tolstikhin et al., 2021) - Mix-
ers are recently proposed architectures that
learn a lightweight mixing of tokens. Since
Mixers have not been used in autoregressive
decoding, we only use token-mixers on the
input encoder.
3.2
Experiment Setup
Our setup, along with all models, are implemented
in Mesh TensorFlow (Shazeer et al., 2018), a li-
brary with similar interface to TensorFlow but en-
ables distributed model parallelism across multiple
workers. For fair comparison, all models are pre-
trained for 219 steps on the english C4 corpus op-
timized using an inverse square root learning rate
with Adafactor (Shazeer and Stern, 2018). All mod-
els use the same SentencePiece tokenizer (Kudo
and Richardson, 2018) containing 32K subwords.
This closely follows the setup in the T5 paper (Raf-
fel et al., 2019). Finetuning is performed for 100K
steps on a mixture of GLUE (Wang et al., 2018),
SuperGLUE (Wang et al., 2019) and SQuAD (Ra-
jpurkar et al., 2016). We evaluate on both upstream
(pre-training) validation perplexity as well as down-
stream transfer for NLU tasks (GLUE + Super-
GLUE + SQuAD) after ﬁne-tuning. We pretrain
and ﬁnetune our models with 16 TPU-v3 chips with
data parallelism. All large models have a model
parallelism of 2 and XL models have a model par-
allelism of 8.
Model Sizes
We consider several different model
sizes for each architecture. For models that are
straightforward to scale, we simply follow the stan-
dard convention in Raffel et al. (2019), moving
from small to base, to large and XL. We include a
tiny version of each model to observe how differ-
ent models behave at lower compute regions. For
models where it was not straightforward to scale
(e.g., Universal Transformers, ALBERT), we tried
to scale them in a similar fashion but faced obvi-
ous limitations such as getting ALBERT to have
the same number of parameters as T5 XL without
incurring a huge number of cost in terms of FLOPs.
For convolutional models, we consider dmodel to
be the hidden size (i.e., channel depth) for the one-
dimensional convolution layers. Values such as
dkv, NH then become redundant. Details on scal-
ing details1 of each architecture can be found in the
1The largest Switch transformer was scaled in a pretty
sub-optimal way. So we don’t think it is representative of the
full potential of the Switch family. Take the last data point of
Switch with a pinch of salt.


---

supplementary material.
3.3
Main Results
We report the main results of this paper in Table
1. We report the number of trainable parameters,
FLOPs (of a single forward pass) and speed (steps
per second). We also report on validation perplex-
ity (on upstream pre-training) and results on 17
downstream tasks. The results are reported aggre-
gates of GLUE, SuperGLUE and SQuAD. While
we use the same Mesh TensorFlow-based codebase
used by Raffel et al. (2019) and hence expect our
experimental results to match theirs, we verify that
our T5 base does achieve similar results to what is
reported in Raffel et al. (2019).
3.4
Do all models scale the same way?
This section investigates if all model architectures
scale in the same way.
Upstream Perplexity
Figure 2 reports the scal-
ing behaviour of all models as we increase the
number of FLOPs. We observe that the scaling be-
haviour of all models are quite unique and distinct,
i.e., most of them are quite different from standard
Transformers. Perhaps the biggest ﬁnding here is
that most models (e.g., LConv, Evolved) all seem to
be on-par or better than standard Transformers but
fail to scale with a higher compute budget. Another
interesting trend is that “linear" Transformers such
as Performer fail to scale as shown in Figure 2i.
The pre-training perplexity metric only decreases
by 2.7% going from base to large scale compared
to 8.4% of the vanilla Transformer.
Downstream Transfer
Figure 3 reports the scal-
ing curves of all models on downstream transfer.
The overall ﬁnding that most models have distinct
scaling curves compared to Transformers is also
evident in downstream tasks. It is also noteworthy
that most models have a different upstream and
downstream scaling curve. We ﬁnd that some mod-
els such as Funnel Transformer and LConvs that
seem to hold out pretty well on upstream but suffer
substantially on downstream. As for Performer, the
performance (disparity) seems to be even greater
in downstream as compared to upstream. Notably,
the SuperGLUE downstream tasks generally re-
quire pseudo cross-attention on the encoder, which
models such as convolutions are not equipped to
handle (Tay et al., 2021a). To this end, we ﬁnd
that certain models may have difﬁculty learning
the downstream tasks despite good upstream per-
formance.
3.5
Are the best models at each scale
different?
Figure 1 shows the Pareto-frontier when plotting
compute against upstream and downstream perfor-
mance. Since the colors of the plot represent dif-
ferent models, we can observe that the best model
for every scale and compute region might be differ-
ent. Moreover, from Figure 3, we can also observe
this. For example, the Evolved Transformer seems
to do well against the standard Transformer at
tiny to small region (downstream) but this quickly
changes when scaling the model up. We also ob-
serve this with MoS-Transformer where it clearly
outperforms vanilla Transformers at some regions
but not at others.
3.6
Scaling Law for Each Model
Table 2 presents the slope of the ﬁtted linear line
α for each model across multiple scenarios. We
derive α by plotting F (FLOPs), U (upstream per-
plexity), D (downstream accuracy), P (number of
parameters). In general, most values of α depict
how well a model scales. For example αF,U is plot-
ting FLOPs against Upstream performance. The
only exception is αU,D which is a measure of up-
stream vs downstream performance. A high αU,D
value means that the transfer to the downstream
tasks is better as a model scales. Overall, the α
value is a metric that represents how well a model
performs relatively across all scales
Analysis of Slope for each Model
In general,
we ﬁnd that the vanilla Transformer has the highest
values of α. Models such as Evolved Transformer,
GLU-Transformer, MoS-Transformer and Funnel
Transformer tend to have similar scaling properties
to the vanilla Transformer. The GLU-Transformer
has similar and slightly worse scaling properties
to the vanilla Transformer, even if it was observed
to do better in absolute sense on some compute-
regions. On the other hand, we also observe that
there are models which are difﬁcult to scale such
as LConv, UT, MLP-Mixer and Performer. This is
even more evident on downstream task. We also
note that ALBERT scales (trends) negatively2 (gets
worse) as we scale the model up. On the other hand,
2This version of ALBERT shares parameters across en-
coder and decoder which may partially explain why we had a
hard time scaling up.


---

Table 1: Results on pre-training and ﬁnetuning ten different model architectures. Full results (further varying
hyperparameters of these models) can be found in the Appendix.
Model
#Params
FLOPs
Speed
Neg Log Ppl
GLUE
SGLUE
SQuAD
Transformer Tiny
16M
1.21
38.4
-2.47
69.3
56.9
73.6
Transformer Small
60M
3.70
22.7
-2.02
78.1
65.3
81.9
Transformer Base
223M
11.4
9.3
-1.75
83.8
74.0
86.3
Transformer Large
738M
34.3
3.6
-1.61
86.4
78.3
88.6
Transformer XL
2.9B
63.8
1.3
-1.49
87.8
81.5
89.5
Evolved Transformer Tiny
19M
1.31
39.7
-2.45
69.6
57.1
69.6
Evolved Transformer Small
79M
4.23
23.7
-2.04
75.7
66.2
80.2
Evolved Transformer Base
218M
10.2
8.9
-1.79
83.0
70.5
84.8
Evolved Transformer Large
1.0B
49.3
2.1
-1.62
86.2
77.1
88.0
Evolved Transformer XL
2.2B
71.3
0.8
-1.55
87.0
78.3
88.2
Universal Transformer Tiny
11M
1.77
38.1
-2.73
69.8
56.1
62.3
Universal Transformer Small
52M
7.30
18.3
-2.12
76.8
64.2
75.4
Universal Transformer Base
127M
20.3
8.4
-1.91
80.0
67.9
80.1
Universal Transformer Large
283M
27.6
1.6
-1.67
84.0
73.4
85.4
Switch Transformer Tiny
174M
3.25
29.7
-2.01
78.2
63.8
80.7
Switch Transformer Small
460M
4.63
22.3
-1.85
80.3
68.0
82.9
Switch Transformer Base
2.0B
12.7
8.4
-1.66
84.2
74.1
86.5
Switch Transformer Large
3.9B
23.0
4.1
-1.56
84.6
75.8
87.9
Switch Transformer XL
29.6B
43.3
0.8
-1.62
84.0
75.2
87.5
Performer Tiny
16M
1.14
42.0
-2.88
50.5
48.8
15.0
Performer Small
61M
3.50
39.0
-2.44
57.8
51.1
31.1
Performer Base
224M
10.8
11.7
-2.23
61.4
53.4
37.8
Performer Large
739M
32.8
4.4
-2.16
62.4
52.4
30.8
Funnel Transformer Tiny
16M
1.10
39.9
-2.58
63.4
49.4
54.6
Funnel Transformer Small
61M
2.96
32.7
-2.11
70.0
58.5
75.1
Funnel Transformer Base
223M
8.10
11.9
-1.83
76.3
62.9
81.6
Funnel Transformer Large
739M
22.6
5.0
-1.69
79.8
67.1
83.8
Funnel Transformer XL
2.9B
40.3
1.89
-1.61
79.8
68.0
83.7
ALBERT Small
15M
3.57
42.0
-2.36
73.7
62.0
77.1
ALBERT Base
21M
9.40
16.4
-2.28
69.0
57.2
64.3
ALBERT Large
34M
31.6
5.1
-2.20
62.9
54.1
27.3
MoS-Transformer Tiny
27M
1.29
39.7
-2.37
70.6
57.9
74.1
MoS-Transformer Small
81M
3.70
26.3
-1.98
79.7
67.1
83.1
MoS-Transformer Base
257M
11.4
8.6
-1.70
84.5
73.9
86.8
MoS-Transformer Large
800M
35.0
3.4
-1.56
86.5
79.7
89.1
MoS-Transformer XL
2.9B
112
1.2
-1.45
88.2
81.4
90.0
GLU-Transformer Tiny
26M
1.29
31.7
-2.35
70.5
57.0
74.2
GLU-Transformer Small
77M
3.70
26.4
-1.97
79.1
67.4
83.0
GLU-Transformer Base
248M
11.4
8.6
-1.71
84.6
74.5
87.2
GLU-Transformer Large
748M
35.0
3.4
-1.56
84.2
74.3
86.2
GLU-Transformer XL
2.85B
61.3
1.0
-1.49
87.6
82.9
89.4
LConv Tiny
17M
1.20
31.2
-2.50
51.1
51.3
49.5
LConv Small
67M
3.80
12.8
-2.10
71.8
59.9
64.7
LConv Base
210M
10.6
12.8
-1.95
73.8
63.6
70.3
LConv Large
741M
41.0
3.0
-1.76
76.8
65.6
76.3
LConv XL
2.3B
77.0
1.0
-1.75
73.3
64.1
72.9
DConv Tiny
22M
1.39
27.3
-2.46
51.1
48.9
30.2
DConv Small
96M
4.97
19.8
-2.08
68.6
57.4
64.3
DConv Base
324M
15.3
7.6
-1.90
72.9
60.1
63.7
DConv Large
1.2B
78.0
1.1
-1.82
70.8
58.5
58.2
MLP-Mixer Small
67M
3.83
22.3
-2.15
65.4
55.1
58.7
MLP-Mixer Base
233M
12.4
10.7
-1.90
64.4
58.1
60.5
MLP-Mixer Large
739M
38.3
3.9
-1.73
52.2
47.8
60.9
MLP-Mixer XL
2.86B
48.3
1.2
-1.61
57.3
58.9
65.7


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
Albert Small
Albert Base
Albert Large
(a) ALBERT
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
DConv Tiny
DConv Small
DConv Base
DConv Large
(b) DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
(c) Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
(d) Funnel
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
(e) Transformer-GLU
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
(f) LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
(g) MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
(h) MoS Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
Performer Tiny
Performer Small
Performer Base
Performer Large
(i) Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
(j) Switch Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Tiny
Mini
Small
Base
Large
XL
UT Tiny
UT Small
UT Base
UT Large
(k) Universal Transformer
Figure 2: Upstream Negative Log-Perplexity of vanilla Transformer compared to other models.
the metric αU,D measures how the downstream per-
formance scales with upstream performance. Over-
all, the Switch Transformer does the best on this
metric where downstream performance scales well
with upstream performance. Generally, models
that make less changes to the main Transformer ar-
chitecture (GLU-Transformer, MoS-Transformer)
tend to retain similar scaling behaviours and chang-
ing the inductive bias also signiﬁcantly alters the
scaling property of the model.
3.7
Do Scaling Protocols inﬂuence model
architectures in the same way?
We are interested in how different scaling protocols
inﬂuence the model architectures. Figure 4 shows
the effect of scaling depth of four model archi-
tectures (MoS-Transformer, Transformer, Evolved
Transformer and LConv). Figure 5 shows the ef-
fect of scaling width on the same four architec-
tures. Firstly, on upstream (negative log perplex-
ity) curves, we note that while different architec-
tures have a distinct difference in absolute perfor-
mance, the scaling trend remains quite similar. On
downstream, depth scaling (Figure 4) seems to act
equally on most architectures with the exception of
LConv. Meanwhile, for width scaling, it seems that
Evolved Transformers scale slightly better when
applying width-scaling. It is also interesting to
note that depth-scaling has a much more substan-
tial impact on downstream scaling as opposed to
width-scaling.
3.8
Epilogue and Conclusion
In this paper, we conducted extensive experiments,
pretraining and ﬁnetuning of up to 100 models
ranging from 10 well-established Transformer and
non-Transformer architectures. We showed that
different model architectures can have different
scaling behaviours and models performing well in
one compute region (or model size) may not do
identically well in another compute region.
We also showed that model architectures may
do well on upstream perplexity but fail to transfer
to downstream tasks. Hence, practitioners should


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
Albert Small
Albert Base
Albert Large
(a) ALBERT
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
DConv Tiny
DConv Small
DConv Base
DConv Large
(b) DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
(c) Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
(d) Funnel
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
(e) Transformer-GLU
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
(f) LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
(g) MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
(h) MoS Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
Performer Tiny
Performer Small
Performer Base
Performer Large
(i) Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
(j) Switch Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
SuperGlue Accuracy
Tiny
Mini
Small
Base
Large
XL
UT Tiny
UT Small
UT Base
UT Large
(k) Universal Transformer
Figure 3: Downstream accuracy of vanilla Transformer compared to other models.
2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+13
FLOPS
-2.30
-2.25
-2.20
-2.15
-2.10
-2.05
-2.00
-1.95
-1.90
-1.85
-1.80
-1.75
-1.70
-1.65
-1.60
-1.55
-1.50
Negative Log-Perplexity
NL2
NL4
NL8
NL12
NL16
NL24
NL32
NL36NL48
LConv NL2
LConv NL4
LConv NL8
Lconv NL16
LConv NL24
LConv NL36
Evolved NL2
Evolved NL4
Evolved NL8
Evolved NL16
Evolved NL24
Evolved NL36
MoS- NL2
MoS- NL4
MoS- NL8
MoS- NL16
MoS- NL24
MoS- NL32
MoS- NL36
(a) Upstream Neg. Log-PPL.
2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+13
FLOPS
52
54
56
58
60
62
64
66
68
70
72
74
76
78
SuperGlue Accuracy
NL2
NL4
NL8
NL12
NL16
NL24
NL32
NL36NL48
LConv NL2
LConv NL4
LConv NL8
Lconv NL16
LConv NL24
LConv NL36
Evolved NL2
Evolved NL4
Evolved NL8
Evolved NL16
Evolved NL24
Evolved NL36
MoS- NL2
MoS- NL4
MoS- NL8
MoS- NL16
MoS- NL24
MoS- NL32
MoS- NL36
(b) Downstream Accuracy.
Figure 4: Scaling depth
be cautious about developing architectures that not
only scale well with respect to the upstream per-
plexity, but also based on downstream performance.
While we certainly do not expect researchers to al-
ways report model performance across all scales
(especially large-scale), we believe that it is good to
keep in mind that architectures can perform quite
differently at different compute regions. Hence,
this might be a good dimension to consider when
1.0e+13
1.2e+13
1.4e+13 1.6e+131.8e+132.0e+13
2.4e+13
2.8e+13 3.2e+13
FLOPS
-1.92
-1.90
-1.88
-1.86
-1.84
-1.82
-1.80
-1.78
-1.76
-1.74
-1.72
-1.70
-1.68
-1.66
-1.64
Negative Log-Perplexity
FF2K
FF6K
FF9K
FF12K
LConv FF6K
LConv FF9K
LConv FF12K
Evolved FF6K
Evolved FF9K
Evolved FF12K
(a) Upstream Neg. Log-PPL.
1.0e+13
1.2e+13
1.4e+13 1.6e+131.8e+132.0e+13
2.4e+13
2.8e+13 3.2e+13
FLOPS
64
65
66
67
68
69
70
71
72
73
74
75
SuperGlue Accuracy
FF2K
FF6K
FF9K
FF12K
LConv FF6K
LConv FF9K
LConv FF12K
Evolved FF6K
Evolved FF9K
Evolved FF12K
(b) Downstream Accuracy.
Figure 5: Scaling width of FFN
designing new inductive biases. As such, perform-
ing evaluation at a certain compute region may be
insufﬁcient to capture the full picture. It is also
good to consider if different inductive biases will
result in different extends of emergent capabilities
(Wei et al., 2022; Abnar et al., 2020).
We also showed that different model architec-
tures may react differently to different scaling pro-
tocols, which further expands on the narrative that


---

Table 2: Slope of a ﬁtted linear line for each model,
when we compare FLOPs vs. upstream performance
(F, U), FLOPs vs. downstream performance (F, D),
parameter size vs. upstream performance (F, U), pa-
rameter size vs. downstream performance (P, D), and
ﬁnally upstream performance vs. downstream perfor-
mance (U, D).
Model
αF,U
αF,D
αP,U
αP,D
αU,D
Transformer
0.54
0.28
0.47
0.24
0.49
GLU-Trans.
0.49
0.24
0.42
0.22
0.46
LConv
0.32
0.13
0.29
0.11
0.48
Funnel
0.47
0.22
0.38
0.18
0.46
Switch
0.23
0.14
0.13
0.08
0.58
Universal
0.50
0.20
0.56
0.22
0.35
ALBERT
0.08
-0.12
0.13
-0.21
-1.67
Evolved
0.44
0.22
0.42
0.21
0.47
Performer
0.25
0.05
0.24
0.05
0.24
MoS-Trans.
0.43
0.21
0.43
0.20
0.47
MLP-Mixer
0.32
-0.03
0.26
0.65
-0.02
comparing and benchmarking these models can be
very challenging (Dehghani et al., 2021b). When
it comes to scaling large models, this paper shows
that novel inductive biases can be indeed quite
risky which might explain why most state-of-the-
art large language models (Rae et al., 2021; Chowd-
hery et al., 2022; Tay et al., 2022) are based on
relatively vanilla architectures. Our advice is to
be cautious when staking an expensive run on a
Transformer architecture that drastically modiﬁes
the attention mechanism (e.g., Mixers and Perform-
ers are generally high risk options as seen in our
experiment results). Finally, we acknowledge that
not every practitioner or researcher would require
models that are able to scale to billion of parame-
ters. In that case, inductive biases that are tailored
to small or low compute will be sufﬁcient.
References
Samira Abnar, Mostafa Dehghani, Behnam Neyshabur,
and Hanie Sedghi. 2021.
Exploring the lim-
its of large scale pre-training.
arXiv preprint
arXiv:2110.02095.
Samira
Abnar,
Mostafa
Dehghani,
and
Willem
Zuidema. 2020.
Transferring inductive biases
through knowledge distillation.
arXiv preprint
arXiv:2006.00555.
Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon
Lee, and Utkarsh Sharma. 2021. Explaining neural
scaling laws. arXiv preprint arXiv:2102.06701.
Krzysztof Choromanski, Valerii Likhosherstov, David
Dohan, Xingyou Song, Andreea Gane, Tamas Sar-
los, Peter Hawkins, Jared Davis, Afroz Mohiuddin,
Lukasz Kaiser, et al. 2020.
Rethinking attention
with performers. arXiv preprint arXiv:2009.14794.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts,
Paul Barham, Hyung Won Chung, Charles Sutton,
Sebastian Gehrmann, et al. 2022.
Palm: Scaling
language modeling with pathways. arXiv preprint
arXiv:2204.02311.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. 2020. Unsupervised
cross-lingual representation learning at scale.
In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 8440–
8451, Online. Association for Computational Lin-
guistics.
Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V Le.
2020. Funnel-transformer: Filtering out sequential
redundancy for efﬁcient language processing. arXiv
preprint arXiv:2006.03236.
Yann N. Dauphin, Angela Fan, Michael Auli, and
David Grangier. 2017.
Language modeling with
gated convolutional networks. In Proceedings of the
34th International Conference on Machine Learning,
volume 70 of Proceedings of Machine Learning Re-
search, pages 933–941. PMLR.
Mostafa Dehghani,
Anurag Arnab,
Lucas Beyer,
Ashish Vaswani, and Yi Tay. 2021a. The efﬁciency
misnomer. arXiv preprint arXiv:2110.12894.
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals,
Jakob Uszkoreit, and Łukasz Kaiser. 2018. Univer-
sal transformers. arXiv preprint arXiv:1807.03819.
Mostafa Dehghani, Yi Tay, Alexey A Gritsenko, Zhe
Zhao, Neil Houlsby, Fernando Diaz, Donald Met-
zler, and Oriol Vinyals. 2021b. The benchmark lot-
tery. arXiv preprint arXiv:2107.07002.
William Fedus, Barret Zoph, and Noam Shazeer. 2021.
Switch transformers: Scaling to trillion parameter
models with simple and efﬁcient sparsity.
arXiv
preprint arXiv:2101.03961.
Naman Goyal, Jingfei Du, Myle Ott, Giri Ananthara-
man, and Alexis Conneau. 2021.
Larger-Scale
Transformers for Multilingual Masked Language
Modeling. arXiv e-prints, page arXiv:2105.00572.
Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen,
Christopher Hesse, Jacob Jackson, Heewoo Jun,
Tom B Brown, Prafulla Dhariwal, Scott Gray, et al.
2020.
Scaling laws for autoregressive generative
modeling. arXiv preprint arXiv:2010.14701.
Danny Hernandez, Jared Kaplan, Tom Henighan, and
Sam McCandlish. 2021. Scaling laws for transfer.
arXiv preprint arXiv:2102.01293.
Jared Kaplan,
Sam McCandlish,
Tom Henighan,
Tom B Brown, Benjamin Chess, Rewon Child, Scott
Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
2020.
Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361.


---

Taku Kudo and John Richardson. 2018. Sentencepiece:
A simple and language independent subword tok-
enizer and detokenizer for neural text processing.
arXiv preprint arXiv:1808.06226.
Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut.
2019. Albert: A lite bert for self-supervised learn-
ing of language representations.
arXiv preprint
arXiv:1909.11942.
Sharan Narang, Hyung Won Chung, Yi Tay, William
Fedus, Thibault Fevry, Michael Matena, Karishma
Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong
Lan, et al. 2021. Do transformer modiﬁcations trans-
fer across implementations and applications? arXiv
preprint arXiv:2102.11972.
Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie
Millican, Jordan Hoffmann, Francis Song, John
Aslanides, Sarah Henderson, Roman Ring, Susan-
nah Young, et al. 2021. Scaling language models:
Methods, analysis & insights from training gopher.
arXiv preprint arXiv:2112.11446.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2019. Exploring the limits
of transfer learning with a uniﬁed text-to-text trans-
former. arXiv preprint arXiv:1910.10683.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016.
Squad: 100,000+ questions
for machine comprehension of text. arXiv preprint
arXiv:1606.05250.
Noam Shazeer. 2020.
Glu variants improve trans-
former. arXiv preprint arXiv:2002.05202.
Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin
Tran, Ashish Vaswani, Penporn Koanantakool, Peter
Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff
Young, et al. 2018. Mesh-tensorﬂow: Deep learning
for supercomputers. In Advances in Neural Informa-
tion Processing Systems, pages 10414–10423.
Noam Shazeer and Mitchell Stern. 2018. Adafactor:
Adaptive learning rates with sublinear memory cost.
In International Conference on Machine Learning,
pages 4596–4604. PMLR.
David So, Quoc Le, and Chen Liang. 2019.
The
evolved transformer. In International Conference on
Machine Learning, pages 5877–5886. PMLR.
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014.
Sequence to sequence learning with neural networks.
arXiv preprint arXiv:1409.3215.
Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald
Metzler. 2020.
Efﬁcient transformers: A survey.
arXiv preprint arXiv:2009.06732.
Yi Tay, Mostafa Dehghani, Jai Gupta, Dara Bahri,
Vamsi Aribandi,
Zhen Qin,
and Donald Met-
zler. 2021a.
Are pre-trained convolutions better
than pre-trained transformers?
arXiv preprint
arXiv:2105.03322.
Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fe-
dus, Samira Abnar, Hyung Won Chung, Sharan
Narang, Dani Yogatama, Ashish Vaswani, and Don-
ald Metzler. 2021b. Scale efﬁciently: Insights from
pre-training and ﬁne-tuning transformers.
arXiv
preprint arXiv:2109.10686.
Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Gar-
cia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng,
Neil Houlsby, and Donald Metzler. 2022.
Unify-
ing language learning paradigms.
arXiv preprint
arXiv:2205.05131.
Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov,
Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner,
Jessica Yung, Daniel Keysers, Jakob Uszkoreit,
Mario Lucic,
et al. 2021.
Mlp-mixer:
An
all-mlp architecture for vision.
arXiv preprint
arXiv:2105.01601.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In Advances in neural information pro-
cessing systems, pages 5998–6008.
Alex Wang,
Yada Pruksachatkun,
Nikita Nangia,
Amanpreet Singh, Julian Michael, Felix Hill, Omer
Levy, and Samuel R Bowman. 2019.
Super-
glue:
A stickier benchmark for general-purpose
language understanding systems.
arXiv preprint
arXiv:1905.00537.
Alex Wang, Amanpreet Singh, Julian Michael, Felix
Hill, Omer Levy, and Samuel R Bowman. 2018.
Glue: A multi-task benchmark and analysis platform
for natural language understanding. arXiv preprint
arXiv:1804.07461.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel,
Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler, et al.
2022. Emergent abilities of large language models.
arXiv preprint arXiv:2206.07682.
Felix Wu, Angela Fan, Alexei Baevski, Yann N
Dauphin, and Michael Auli. 2019. Pay less attention
with lightweight and dynamic convolutions. arXiv
preprint arXiv:1901.10430.
Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and
William W Cohen. 2017. Breaking the softmax bot-
tleneck: A high-rank rnn language model.
arXiv
preprint arXiv:1711.03953.


---

4
Appendix
4.1
Scaling Details for Individual Models
For most models, it was reasonable to follow the
uniform scaling method in the main T5 sizes. At
each size, the hyperparameters are as follows:
Model
NL
dff
dmodel
dkv
NH
#Params
Tiny
4/4
1024
256
32
4
16M
Small
6/6
2048
512
32
8
60M
Base
12/12
3072
768
64
12
220M
Large
24/24
4096
1024
64
16
738M
XL
24/24
16384
1024
128
32
3B
Table 3: Table of model conﬁgurations. NL is the num-
ber of layers, dff is the size of the MLP, dmodel is the
hidden size of the model. dkv is the size of each key-
value vector. NH is the number of heads.
Scaling for Switch Transformer
For Switch
Transformers, we use the following scaling:
Model
NL
dff
dmodel
dkv
NH
NE
#Params
Tiny
4
1024
512
64
12
32
173M
Small
6
2048
512
64
12
32
460M
Base
12
3072
768
64
12
32
2B
Large
24
3072
768
64
12
32
8B
XL
48
3072
768
64
12
128
30B
Table 4: Scaling for Switch Transformer. NE is the
number of experts.
Scaling for Universal Transformer
Scaling
UTs are generally difﬁcult as described in the main
text. There were two main considerations for scal-
ing UTs. Initially we tried scaling the number of
recurrent operations. However, we found that even
with an increase of FLOPS, this does not lead to
improved performance. Overall, the UT model
might be pretty slow and therefore a model with
the same hparams as vanilla XL might be infeasible
to run. Hence, we explored increasing the width of
the MLPs to 32K to see if UTs would scale in this
manner.
Model
NR
dff
dmodel
dkv
NH
#Params
UT Tiny
3/3
1024
128
32
8
11M
UT Small
3/3
2048
512
32
8
52M
UT Base
3/3
3072
768
64
12
127M
UT Large
3/3
32768
1024
64
16
283M
Table 5: Table of model conﬁgurations. NR is the num-
ber of recurrent operations, dff is the size of the MLP,
dmodel is the hidden size of the model. dkv is the size
of each key-value vector. NH is the number of heads.


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(a) FLOPs
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
8.4e+6
3.4e+76.7e+71.3e+82.7e+85.4e+81.1e+92.1e+94.3e+98.6e+9 3.4e+10
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(b) Number of Parameters
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
LConv Tiny
LConv Small
LConv Base
LConv Large LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
-3.0
-2.8
-2.6
-2.4
-2.2
-2.0
-1.8
-1.6
-1.4
Negative Log-Perplexity
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(c) Throughput
Figure 6: Quality-cost trade of for the upstream Negative Log-Perplexity of vanilla Transformer compared to other
models, with respect to FLOPs, number of parameters, and throughput.


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(a) FLOPs
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
8.4e+6
3.4e+76.7e+71.3e+82.7e+85.4e+81.1e+92.1e+94.3e+98.6e+9 3.4e+10
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
GLU Tiny
GLU Small
GLU BaseGLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(b) Number of Parameters
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel LargeFunnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
GLU Tiny
GLU Small
GLU Base GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
45
50
55
60
65
70
75
80
85
90
SuperGlue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(c) Throughput
Figure 7: Quality-cost trade of for the downstream SuperGlue Accuracy of vanilla Transformer compared to other
models, with respect to FLOPs, number of parameters, and throughput.


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
50
55
60
65
70
75
80
85
90
Glue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(a) FLOPs
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
8.4e+6
3.4e+76.7e+71.3e+82.7e+85.4e+81.1e+92.1e+94.3e+98.6e+9 3.4e+10
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
GLU Tiny
GLU Small
GLU BaseGLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
50
55
60
65
70
75
80
85
90
Glue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(b) Number of Parameters
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel LargeFunnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
MoS- Tiny
MoS- Small
MoS- Base
MoS- Large
MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
GLU Tiny
GLU Small
GLU Base GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
50
55
60
65
70
75
80
85
90
Glue Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(c) Throughput
Figure 8: Quality-cost trade of for the downstream Glue Accuracy of vanilla Transformer compared to other
models, with respect to FLOPs, number of parameters, and throughput.


---

1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
DConv Tiny
DConv SmallDConv Base
DConv Large
Tiny
Mini
Small
Base
Large XL
DConv
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large XL
Evolved
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
Funnel Tiny
Funnel Small
Funnel BaseFunnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large XL
LConv
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.1e+122.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13
7.0e+131.4e+14
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
MoS- Tiny
MoS- Small
MoS- Base MoS- Large MoS- 3B
Tiny
Mini
Small
Base
Large XL
MoS-Transformer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
GLU Tiny
GLU Small
GLU Base
GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.1e+12 2.2e+12
4.4e+12
8.8e+12
1.8e+13
3.5e+13 7.0e+13
FLOPS
10
20
30
40
50
60
70
80
90
Squad Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(a) FLOPs
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
DConv Tiny
DConv Small
DConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel Large
Funnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
MoS- Tiny
MoS- Small
MoS- BaseMoS- LargeMoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
8.4e+6
3.4e+76.7e+71.3e+82.7e+85.4e+81.1e+92.1e+94.3e+98.6e+9 3.4e+10
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
GLU Tiny
GLU Small
GLU BaseGLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
8.4e+6
3.4e+7 6.7e+7 1.3e+8 2.7e+8 5.4e+8 1.1e+9 2.1e+9
Params
10
20
30
40
50
60
70
80
90
Squad Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(b) Number of Parameters
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
Albert Small
Albert Base
Albert Large
Tiny
Mini
Small
Base
Large
XL
Albert
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
DConv Tiny
DConv SmallDConv Base
DConv Large
Tiny
Mini
Small
Base
Large
XL
DConv
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
Evolved Tiny
Evolved Small
Evolved Base
Evolved Large
Evolved 3B
Tiny
Mini
Small
Base
Large
XL
Evolved
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
Funnel Tiny
Funnel Small
Funnel Base
Funnel LargeFunnel 3B
Tiny
Mini
Small
Base
Large
XL
Funnel
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
LConv Tiny
LConv Small
LConv Base
LConv Large
LConv 3B
Tiny
Mini
Small
Base
Large
XL
LConv
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
MLP Mixer Small
MLP Mixer Base
MLP Mixer Large
MLP Mixer 3B
Tiny
Mini
Small
Base
Large
XL
MLP Mixer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
MoS- Tiny
MoS- Small
MoS- Base MoS- Large MoS- 3B
Tiny
Mini
Small
Base
Large
XL
MoS-Transformer
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
Performer Tiny
Performer Small
Performer Base
Performer Large
Tiny
Mini
Small
Base
Large
XL
Performer
1.6e-2 3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0 2.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
Switch 173M
Switch 460M
Switch 2B
Switch 4B
Switch XL3
Tiny
Mini
Small
Base
Large
XL
Switch
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
GLU Tiny
GLU Small
GLU Base GLU Large
GLU 3B
Tiny
Mini
Small
Base
Large
XL
TransformerGLU
1.6e-2
3.1e-2
6.3e-2
1.3e-1
2.5e-1
5.0e-1
1.0e+0
1/Throughput
10
20
30
40
50
60
70
80
90
Squad Accuracy
UT Tiny
UT Small
UT Base
UT Large
Tiny
Mini
Small
Base
Large
XL
UT
(c) Throughput
Figure 9: Quality-cost trade of for the downstream Squad Accuracy of vanilla Transformer compared to other
models, with respect to FLOPs, number of parameters, and throughput.
