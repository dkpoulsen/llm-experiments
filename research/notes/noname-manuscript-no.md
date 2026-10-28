---
title: 'Mesa: A Memory-saving Training Framework for Transformers'
id: noname-manuscript-no
tags:
- cmp170hx-training-feasibility-a0d4fc
- mesa
- activation-quantization
- memory-speed-tradeoff
- int8-activations
created: '2026-10-27T11:23:53.921759Z'
updated: '2026-10-27T19:00:31.471225Z'
source: https://arxiv.org/abs/2111.11124
source_domain: arxiv.org
fetched_at: '2026-10-27T11:23:53.921178Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Mesa (Monash University, arXiv 2111.11124) - memory-saving training framework
  for Transformers that keeps the forward pass in exact precision but stores only
  a low-precision (int8) copy of each activation, dequantizing it on the backward
  pass to recompute gradients. Two additions make it work: head-wise activation quantization
  (per-attention-head clipping ranges and offsets instead of per-tensor, because MSA
  heads have heterogeneous activation distributions) and running estimates (RE) of
  the quantization parameters so no extra forward pass is needed. Reports up to 50%
  training-memory reduction at comparable or slightly better accuracy: DeiT-Ti 4,149
  MB -> 2,000 MB (RE) with ImageNet Top-1 71.9 -> 72.1; DeiT-B -51%, Swin-Ti -55%;
  compressing MSA alone -27%, FFN alone -21%. The cost is throughput: DeiT-Ti falls
  from 1,196 to 431 images/s (about 2.8x slower) with RE, and compressing the whole
  model halves training speed. Nearest rounding without stochastic rounding fails
  to converge on ImageNet (Table 3). Under the same memory budget it doubles DeiT-Ti
  batch size 1,024 -> 2,048 (33.4 GB vs 28.8 GB) for +0.8% Top-1. Crucially, the paper
  argues the throughput penalty is partly hidden in distributed training because communication
  dominates, and reports PVT-Ti costs only +15% GPU hours. Vision-only (ImageNet,
  CIFAR-100, ADE20K): no LLM or long-context experiments, so the 32k-activation-memory
  case is untested.'
raw_file: raw/noname-manuscript-no.pdf
doi: arXiv:2111.11124
utility_score: 12.0
citation_count: 27
venue: arXiv.org
---

Noname manuscript No.
(will be inserted by the editor)
Mesa: A Memory-saving Training Framework for Transformers
Zizheng Pan, Peng Chen, Haoyu He, Jing Liu, Jianfei Cai
and Bohan Zhuang †
Abstract There has been an explosion of interest in
designing high-performance Transformers. While Trans-
formers have delivered signiﬁcant performance improve-
ments, training such networks is extremely memory in-
tensive owing to storing all intermediate activations
that are needed for gradient computation during back-
propagation, especially for long sequences. To this end,
we present Mesa, a memory-saving training framework
for Transformers. Speciﬁcally, Mesa uses exact activa-
tions during forward pass while storing a low-precision
version of activations to reduce memory consumption
during training. The low-precision activations are then
dequantized during back-propagation to compute gra-
dients. Besides, to address the heterogeneous activation
distributions in the multi-head self-attention layers, we
propose a head-wise activation quantization strategy,
which quantizes activations based on the statistics of
each head to minimize the approximation error. To fur-
ther boost training eﬃciency, we learn quantization pa-
rameters by running estimates. More importantly, by
re-investing the saved memory in employing a larger
batch size or scaling up model size, we may further
improve the performance under constrained computa-
tional resources. Extensive experiments on ImageNet,
CIFAR-100 and ADE20K demonstrate that Mesa can
achieve ﬂexible memory-savings (up to 50%) during
training while achieving comparable or even better per-
formance. Code is available at https://github.com/
ziplab/Mesa.
Keywords Vision Transformers, Eﬃcient Training
Zizheng Pan, Peng Chen, Haoyu He, Jing Liu, Jianfei Cai and
Bohan Zhuang
Faculty of Information Technology, Monash University
E-mail:
{zizeng.pan,
haoyu.he,
jing.liu1,
jianfei.cai,
bo-
han.zhuang}@monash.edu, blueardour@gmail.com
† Corresponding author.
1 Introduction
Transformers have demonstrated stunning success in a
wide range of natural language processing (NLP) [1,2]
and computer vision (CV) tasks [3–5]. Inspired by the
previous works on model scaling [6, 7], the recent re-
searches on Transformers further push the performance
forward with an increasing model size [3, 8, 9]. How-
ever, training Transformer models requires a formidable
amount of memory footprints, prohibiting common users
with limited computational resources from doing re-
lated researches. For example, training a Swin-T [9]
with a batch size of 128 on 224×224 images consumes
at least 12G GPU memory, while Swin-B cannot ﬁt
into a 32GB V100 GPU under the same settings. Con-
sequently, only a few parties can aﬀord to train such
large models. The huge memory consumption and the
increasing resource inequalities make it diﬃcult for the
academic community to follow up on this area, based on
the fact that most of the recent advanced Transformers
are developed with industry participation.
Fortunately, great eﬀorts have been made to train
deep neural networks with low memory. For example,
existing representative techniques include sparsity [11],
low-precision training [10], micro-batching [12], check-
pointing [13] and gradient quantization [14]. Among the
existing memory-saving techniques, one promising di-
rection is the activation compressed training (ACT) [15].
Speciﬁcally, ACT stores low-precision approximate copies
of activations at each layer while computing the forward
pass exactly, which helps to reduce the overall memory
consumption during training. The saved activations are
then dequantized to the original precision in the back-
ward pass to calculate gradients. This approach has
been successfully applied to train ResNet [6] variants.
However, existing works on ACT [15–17] either focus
on dedicated architectures or speciﬁcally target on con-
arXiv:2111.11124v3  [cs.CV]  29 Aug 2022


---

2
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
Swin-Ti
Swin-Ti
0
5000
10000
15000
20000
25000
Training Memory (MB)
70
72
74
76
78
80
82
84
Top-1 Accuracy (%)
DeiT-Ti
DeiT-S
DeiT-B
PVT-Ti
DeiT-Ti
DeiT-S
DeiT-B
PVT-Ti
Mesa + AMP
AMP
Fig. 1 Comparisons of memory footprints during training
with several state-of-the-art vision Transformers on Ima-
geNet. “AMP” denotes the default automatic mixed-precision
training [10]. “Mesa + AMP” means that we train mod-
els with Mesa along with AMP. The memory footprint is
measured with a batch size of 128 and an image resolution
of 224×224 on a single GPU. The proposed Mesa reduces
around half of the memory consumption during training while
achieving similar or even better performance compared to the
default AMP training.
volutional neural works (CNNs). At present, there is
no literature on compressing the commonly used oper-
ations in Transformers (e.g., Softmax, GELU [18] and
LayerNorm [19]). Moreover, no previous work consid-
ers the heterogeneously distributed activations in the
multi-head self-attention (MSA) layers, especially the
attentions. Consequently, we empirically observe that
applying the naive activation bucketing strategy as in
previous works [16] for the attention maps will make
Transformers fail to converge. Therefore, none of the
existing ACT frameworks can be directly applied to
Transformer-based models. Targeting common opera-
tions in Transformer remains challenging and the eﬀect
is unknown.
To tackle the above challenges, we present Mesa,
a Memory-saving 8-bit training framework for Trans-
formers. Mesa covers all memory-hungry operations in
Transformers, including matrix multiplications (Mat-
Mul), Softmax, LayerNorm and GELU. Moreover, we
propose a head-wise activation quantization strategy,
which quantizes the activations based on each self-attention
head. The motivation comes from two aspects. First,
group-wise quantization, derived from product quan-
tization [20], has shown to be eﬀective in minimizing
quantization error. The proposed head-wise quantiza-
tion can be seen as a special case of group-wise quan-
tization where the number of groups is equal to the
number of self-attention heads. Second, previous stud-
ies have revealed that diﬀerent self-attention heads in
Transformers tend to learn diﬀerent attention patterns
[21]. Empirically, we can also observe a large divergence
of statistics among diﬀerent heads as shown in Figure 4.
Therefore, using shared quantization parameters across
self-attention heads may result in highly degraded esti-
mates of statistical quantities.
Besides, Mesa learns quantization parameters with
eﬃcient running estimates during training. Such ap-
proach brings additional beneﬁts compared with gra-
dient based approaches [22, 23] or per-sample statis-
tics [16] as it avoids extra computational and memory
cost for learning quantization parameters. In practice,
we also observe that the running estimates performs
favourably against the per-sample statistics in terms of
both training speed and performance.
To the best of our knowledge, Mesa is the ﬁrst ACT
framework for Transformer-based models. It is also or-
thogonal to other memory saving techniques such as
low-precision training [10] and checkpointing [13]. There-
fore, in practice, one can simultaneously apply auto-
matic mixed-precision training (AMP) and checkpoint-
ing along with Mesa to achieve more memory-saving.
As a bi-product of signiﬁcant memory reduction during
training, we can use a larger batch size and/or train
a larger Transformer model to enable fully-utilization
of available GPU cores. Furthermore, we are able to
re-invest the saved memory during training by con-
structing 3.3× deeper or 2.2× wider DeiT-B, or training
DeiT-B with 1.5× larger image resolution. Note that as
in common practice [13,16,24], there is a trade-oﬀbe-
tween the memory cost and training speed for Mesa.
In this case, Mesa is designed to ﬂexibly compress dif-
ferent operations/modules to achieve a target memory-
saving with acceptable training overhead. Moreover, for
a large number of researchers with only entry/middle-
level GPUs, their ﬁrst thing is that Transformer can
be run in their machines even with small-size models,
before considering throughput. To this end, what Mesa
oﬀers to the community is a comprehensive study of
customizing ACT to Transformers, along with an eﬃ-
cient and eﬀective plug-and-play module with CUDA
implementation that can be easily used in a variety of
Transformer models for ﬂexible memory-saving. In a
nutshell, we summarize our contributions as follows:
1. We propose a memory-saving 8-bit training frame-
work for Transformers, namely Mesa. Mesa is imple-
mented with a fast CUDA kernel and can be easily
adapted to any Transformer projects.
2. We observe the heterogeneously distributed ac-
tivations in self-attention heads. For this, we pro-
pose a head-wise activation quantization strategy
to minimize the approximation error in MSA layers.
Besides, we use running estimates to learn quantiza-


---

Mesa: A Memory-saving Training Framework for Transformers
3
tion parameters, which performs well with negligible
additional cost.
3. Extensive experiments on ImageNet, CIFAR-100
and ADE20K have demonstrated that Mesa can re-
duce ∼50% memory footprint during training with
comparable or even better performance than the
standard mixed-precision training scheme.
2 Related Work
2.1 Transformers
Transformer is initially proposed by Vaswani et al. [25]
for machine translation. A standard Transformer con-
sists of an embedding layer, several Transformer blocks
and a task-speciﬁc head, where each block contains
an MSA layer and a position-wise feed-forward (FFN)
layer. Later on, Transformer has been extended into a
wide range of tasks. In the area of computer vision,
vision Transformers (ViTs) have attracted great atten-
tions recently. For example, Dosovitskiy et al. [26] pro-
posed a standard Transformer architecture for image
recognition, which achieved competitive results on Im-
ageNet compared to CNNs. Subsequent works have im-
proved ViTs from diﬀerent aspects, such as incorporat-
ing pyramid features [3,9,27] , adopting convolutional
layers to enhance the model locality [28], or exploring
a well performed architecture with neural architecture
search (NAS) [29,30]. However, to train a Transformer
usually requires intensive computational resources. For
example, the typical setting [31] to train a ViT on Ima-
geNet requires a batch size of 1,024 on 8 NVIDIA V100
GPUs. As a result, only a few parties are capable of
running such experiments. Besides, it also makes it dif-
ﬁcult for researchers to explore a larger design space
for Transformer architectures. To address this problem,
we propose to reduce the memory cost of Transformers
during training by 8-bit activation compressed training,
making the experiments aﬀordable.
2.2 Quantized Training
Quantized training aims to improve the model eﬃciency
at training time or inference time by quantizing model
weights, activations or gradients into low-precision val-
ues. Existing methods can be roughly categorised into
two folds: 1) quantizing a pretrained model with or
without retraining [32–35], and 2) training a quantized
model from scratch [10,14,36,37]. In Transformers, the
majority of the literature belongs to the ﬁrst category.
For example, 8-bit [38,39] or even lower-bits [40] quan-
tization has been proposed to speed up the inference.
In contrast, this paper focuses on training Transform-
ers from scratch. Diﬀerent from previous works [38,41]
that carry out low-precision computations during either
the forward pass or the backward pass, we store the ap-
proximated low-precision activations for memory saving
during training while still computing the forward pass
exactly. Therefore, we do not change the forward pass
behavior of models.
2.3 Memory-eﬃcient Training
Low-memory training is appealing as it enables large-
scale model training in resource-constraint scenarios. A
plethora of methods have been proposed in this area.
For example, Mostafa et al. [11] proposed to reduce
the model and optimizer memory by dynamic sparse
reparameterization. Micikevicius et al. [10] introduced
mixed precision training, which utilizes mixed half pre-
cision (16-bits) and full precision (32-bits) for training.
Huang et al. [12] proposed to split the mini-batch into
smaller subsets and then accumulate the gradients until
the entire minibatch has been processed. Besides, gra-
dient checkpointing [13] is also a common practice to
reduce the activation memory.
Orthogonal to the above approaches, activation com-
pressed training (ACT) [15] has recently been proposed
to reduce the storage of activations that are required
for gradient computation. This method was ﬁrst intro-
duced by Chakrabarti et al. [15] for ResNet training.
Subsequent works such as TinyScript [17] and ActNN
[16] extend this framework by introducing non-uniform
quantization and mixed-precision quantization, respec-
tively. However, all these methods speciﬁcally target on
CNNs and do not consider the unique components of
Transformers, such as LayerNorm, Softmax and GELU.
Moreover, the quantization scheme for them is prob-
lematic for Transformers due to the heterogeneous ac-
tivation distributions in an MSA layer. Consequently,
none of the existing methods can be directly applied to
Transformer based models. In this paper, we customize
the ACT framework to signiﬁcantly reduce the resource
requirement of training Transformers while keeping their
outstanding performance.
3 Method
In this section, we ﬁrst describe the overall framework
of Mesa. Then we introduce our proposed head-wise
activation quantization and the strategy for learning
quantization parameters. Lastly, we provide the details
of the system implementation and discuss the overhead
of Mesa.


---

4
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
𝐗𝐗0
F1
Quantize
G1
Dequantize
F𝐿𝐿
Quantize
G𝐿𝐿
Dequantize
𝓛𝓛
…
…
8-bit compressed activations
෡∇𝐗𝐗1
𝐖𝐖1
෡∇𝐖𝐖1
෡∇𝐖𝐖𝐿𝐿
𝐖𝐖𝐿𝐿
𝐗𝐗1
ഥ𝐗𝐗0
𝐗𝐗𝐿𝐿−1
𝐗𝐗𝐿𝐿
ഥ𝐗𝐗𝐿𝐿−1
෡∇𝐗𝐗𝐿𝐿−1
∇X𝐿𝐿
…
Fig. 2 The pipeline of the proposed Mesa for training Trans-
formers, where we compress the activations into low-precision
values during training to achieve memory reduction while still
propagating the exact activations during the forward pass.
The dequantized activations are used to compute gradients
during backpropagation. Blue and red lines represent the for-
ward and backward passes, respectively
3.1 Overview of Mesa
To reduce the memory consumption of Transformers at
training time, we introduce Mesa, a generic memory-
saving training framework for Transformers. The over-
all pipeline of Mesa is depicted in Figure 2. In general,
Mesa saves low-precision approximated activations dur-
ing training for backpropagation while still using exact
activations for the forward pass. Speciﬁcally, denoting
Xl−1 as the input to the l-th layer in a Transformer,
the output of the l-th layer can be formulated by
Xl = Fl(Wl, Xl−1),
(1)
where Fl and Wl represent the function and learnable
parameters of the l-th layer, respectively.
In a standard training, the input Xl−1 is saved in
the GPU memory in order to calculate gradients dur-
ing backpropagation, where the saved activations at all
layers take up the majority of the memory consumption
during training, especially when equipped with a large
batch size. To reduce the memory footprint, we pro-
pose to only save the compressed activations ¯Xl−1 in-
stead of the full-precision counterparts Xl−1 during the
forward pass. Such compression is achieved by quanti-
zation, which quantizes the exact activations into low-
precision values. During backpropagation, ¯Xl−1 is de-
quantized into the original precision values ˆXl−1 for
gradient calculation. In this way, the gradients at the
l-th layer can be approximated by
ˆ∇Xl−1, ˆ∇Wl = Gl( ˆ∇Xl, ˆXl−1, Wl),
(2)
where ˆ∇Xl−1, ˆ∇Wl represent the approximated gradi-
ents for the input Xl−1 and the learnable parameters
Wl, and Gl is the gradient function at the l-th layer.
It is worth noting that this strategy has little eﬀect
on the training performance, since it only introduces
modest approximation errors to the natural gradient
noise from distributed training and stochastic gradient
descent (SGD) [42,43]. In the next section, we introduce
a head-wise activation quantization scheme to further
improve the ﬁdelity of gradients during training.
3.2 Head-wise Activation Quantization
As shown in [20], a ﬁne quantization granularity is fa-
vorable to minimize the approximation error. In prac-
tice, layer-wise quantization is fast but may introduce
a large quantization error, while channel-wise quantiza-
tion can be more accurate but comes with extra compu-
tational and memory cost. In light of this, group-wise
quantization balances the two sides and has been widely
adopted in the literature [38,39].
A naive grouping strategy in the previous works [16,
17] is to slice a tensor into ﬁxed-size buckets regardless
of the tensor dimensions, which is however problematic
for Transformers as it ignores the fact that diﬀerent self-
attention heads usually have quite diﬀerent attention
patterns [21], i.e., the activation distributions across
heads are with distinct means and variances in an MSA
layer. In Figure 4, we visualize the activations before the
Softmax layer at the 11-th block of DeiT-S. Clearly, the
activation at each self-attention head should have its
unique clipping range and oﬀset. Such phenomenon can
be observed for the queries, keys, values and attentions
across all Transformer blocks.
Quantization scheme. To address the heterogeneously
distributed activations in self-attention heads, we pro-
pose a head-wise activation quantization strategy. Specif-
ically, with x ∈Xh being one element in the input acti-
vations at the h-th head in an MSA layer, we quantize
the activations to 8-bit by
¯x = clip(round((x −βh) · 255
αh ), 0, 255),
(3)
where αh and βh are learnable parameters represent-
ing the quantization clipping range and the oﬀset at
the h-th head, respectively. clip(x, xlow, xup) clips any
number x into the range of [xlow, xup]. round(·) denotes
the rounding function. Here we adopt the stochastic
rounding
[37] as it theoretically guarantees smaller
probabilistic error bounds [44] compared to the near-
est rounding. Speciﬁcally, it can be formulated as
round(x) =
(
⌈x⌉
with probability p = x −⌊x⌋,
⌊x⌋
otherwise.
(4)


---

Mesa: A Memory-saving Training Framework for Transformers
5
𝑁𝑁
𝐵𝐵
𝐷𝐷
Standard input sequence
𝑁𝑁or 𝐷𝐷ℎ
𝑁𝑁
𝐵𝐵
Queries, Keys, Values or Attentions
Head 1
Head 2
Head 𝑁𝑁ℎ
…
Fig. 3 The diﬀerent group-wise bucketing strategies in Mesa. Diﬀerent colors represent diﬀerent quantization groups. Let B,
N and D be the batch size, the sequence length and the number of hidden channel dimensions, respectively. Nh and Dh refer
to the number of self-attention heads and the head dimensions, respectively. For a standard input sequence, we group the
activations based on a certain number of hidden channels (D/2 in this example). For the queries, keys, values and attentions,
we group the activations based on self-attention heads. Best viewed in color.
12.5
10.0
7.5
5.0
2.5
0.0
2.5
5.0
Values
0
200
400
600
800
1000
1200
1400
Count
Head-1
Head-2
Head-3
Head-4
Head-5
Head-6
Fig. 4 Activation distributions before the Softmax layer at
the 11-th block of DeiT-S [31]. We slice the activation ten-
sor into diﬀerent groups based on self-attention heads and
visualize each group in diﬀerent colors. Best viewed in color.
During backpropagation, we dequantize the activations
at this layer into the original precision by
ˆx = ¯x · αh
255 + βh.
(5)
The dequantized activations are then used to calculate
gradients as in Eq. (2). In Section 5.2, we will show
the eﬀectiveness of the proposed quantization scheme
by comparing it to other strategies, such as symmetric
quantization and nearest rounding.
Learning quantization parameters with running
estimates. To calculate the quantization parameters
α and β at each layer, some previous works propose
to use per-sample statistics [16, 17] or gradient-based
approaches [22, 23]. Speciﬁcally, the methods of per-
sample statistics utilise the current min-max values of
each sample at each layer to calculate quantization pa-
rameters, which is ineﬃcient and consumes additional
memory for storing the quantization parameters. For
gradient-based approaches, asymmetric quantization meth-
ods such as LSQ+ [23] may increase the memory foot-
print as they need to save both the compressed activa-
tions and a large amount of context (e.g., the rounding
errors for learning α) that are needed to calculate gradi-
ents during backpropagation, contradicting our aim of
memory saving. Alternatively, symmetric quantization
approaches such as PACT [22] do not have this prob-
lem as they only require binary values to calculate the
gradients for α. However, we will show in Section 5.2.3
that symmetric quantization achieves poor performance
on Transformers compared to both baselines and asym-
metric quantization.
To this end, we propose to utilise running estimates
[45] to update the quantization parameters during train-
ing, which can be expressed as
αh = λαh + (1 −λ)(max(Xh) −min(Xh)),
(6)
βh = λβh + (1 −λ)min(Xh),
(7)
where λ is a hyperparameter. As the quantization pa-
rameters are shared within each head across diﬀerent
samples, using running estimates can save more mem-
ory at training time. For example, letting Nh be the
number of self-attention heads in an MSA layer and
B be the batch size, with running estimates, we only
need to store 2Nh additional parameters for the Soft-
max layer, which is negligible compared to the overall
memory footprint. In contrast, using per-sample statis-
tics requires to save 2BNh additional parameters for
one layer. Furthermore, we will show in Section 5.2 that
using running estimates can achieve faster throughput
at training time.
Other types of activations. It is worth noting that
we adopt the head-wise bucketing strategy for queries,
keys, values and attentions in a Transformer model. For
other types of activations such as the standard input se-
quence in FFN layers, we apply the channel group-wise
quantization scheme where multiple channel dimensions
of activations are grouped. Figure 3 summarizes the dif-
ferent bucketing strategies of Mesa.


---

6
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
3.3 System Implementation
Mesa is built upon the popular PyTorch [46] frame-
work. It is a standalone package that can be directly
adopted into any Transformer projects. To further ac-
celerate the quantization procedure (i.e., Eq. (3) and
Eq. (5)) during training, we implement a fast CUDA
kernel to improve the training eﬃciency. Overall, Mesa
is able to compress all memory-hungry operations in a
Transformer, including MatMul, LayerNorm, Softmax
and GELU. Moreover, to support downstream tasks
such as semantic segmentation, we also cover commonly
used layers in CNNs, such as the convolutional layer and
ReLU. For these layers, we adopt the standard group-
wise quantization strategy as mentioned in Section 3.2
where we group the convolutional activations based on
a ﬁxed number of channels. By default, we set the num-
ber of quantization groups to be the number of heads
at a Transformer block.
4 Discussions
4.1 The Speed and Memory Trade-oﬀ
While both the training speed and the memory con-
sumption are critical bottlenecks for Transformer train-
ing, this work focuses on the latter. As in common prac-
tice (e.g., checkpointing [13] and hashing [24]), there is
a trade-oﬀbetween the speed and memory consump-
tion at training time for all ACT frameworks [16, 17].
In Mesa, the additional training overhead includes the
quantization/de-quantization, the computation of min/-
max values, stochastic rounding and tensor reshaping
for highly accurate CUDA dispatching. These opera-
tions can happen in almost all layers to achieve sig-
niﬁcant memory-saving while sacriﬁcing the training
speed. In an extreme case, Mesa could result in the
throughput decreases by half on a single GPU when
compressing a whole model.
On the other hand, considering that distributed train-
ing has been widely adopted in modern deep learn-
ing [1,3,26], the training cost can be dominated by data
loading and communication overhead among GPUs, es-
pecially for small models. For example, we will show in
Section 5.1 that training PVT-Ti [3] with Mesa only re-
quires additional 15% GPU hours. Besides, to further
mitigate the additional training overhead, we modu-
larize Mesa such that it can ﬂexibly target diﬀerent
components in a model during training. In practice, we
suggest to compress the most memory-hungry modules
(e.g., backbones) to achieve a target memory-saving
with acceptable training speed. In Section 5.2, we show
in Table 7 that compressing diﬀerent operations (e.g.,
Softmax and GELU) can achieve diﬀerent speed and
memory trade-oﬀs. We will also provide the experi-
ments on compressing diﬀerent modules (e.g., MSA and
FFN) in Table 8 for reference.
4.2 Mixed-precision and Lower Bits Quantization
Apart from the 8-bit quantization used in Mesa, mixed-
precision quantization [47] could potentially bring more
beneﬁts. However, it will also introduce extra training
overhead as it has to calculate the quantization bits for
each layer, thus again slowing down the training speed.
Furthermore, compared to 8-bit integer, the computa-
tion and storage of lower bit values have not been well
supported in existing CPUs/GPUs and deep learning
frameworks (e.g., PyTorch). For this reason, we choose
the ﬁxed 8-bit quantization in Mesa for a better trade-
oﬀbetween training speed and model performance.
4.3 Relation to Other Low-memory Training
Techniques
As mentioned in Section 2, Mesa is orthogonal to check-
pointing [13], micro-batching [12] and mixed-precision
training [10]. Therefore, Mesa can be applied along with
these techniques for more memory reduction, as shown
in Table 14 and Figure 1. Besides, checkpointing tar-
gets the computational graph and keeps performance
constant. In practice, one has to manually set the check-
pointing positions, which is not optimal for the memory-
speed trade-oﬀ. In contrast, Mesa is more ﬂexible for
compressing speciﬁed operations to match a target train-
ing memory, as shown in Table 7. Mesa can also poten-
tially improve performance as well as easily identify-
ing the most memory-intensive operations/modules for
training. For micro-batching [12], it imitates a larger
batch size but may aﬀect model performance. More-
over, training with diﬀerent batch sizes makes it diﬃ-
cult for a fair comparison with related works. For the
existing ACT works, they either cannot apply to Trans-
formers [15,17] or fully apply to Transformers [16], and
thus they are not directly comparable. One closely re-
lated work is ActNN [16], for which we show in Sec-
tion 5.2.11 that Mesa achieves better memory-saving
and speed trade-oﬀ.
5 Experiments
5.1 Main Results
Dataset and evaluation metrics. We conduct exper-
iments on the ImageNet (ILSVRC2012) [48] dataset,


---

Mesa: A Memory-saving Training Framework for Transformers
7
0
100
200
300
Epoch
0
50
100
Top-5 Acc. (%)
DeiT-B
DeiT-B w/ Mesa
0
100
200
300
Epoch
3
4
5
6
7
Train Loss
DeiT-B
DeiT-B w/ Mesa
0
100
200
300
Epoch
2
4
6
Test Loss
DeiT-B
DeiT-B w/ Mesa
0
100
200
300
Epoch
0
20
40
60
80
Top-1 Acc. (%)
DeiT-B
DeiT-B w/ Mesa
Fig. 5 Training curves comparison for DeiT-B with/without Mesa on ImageNet.
Table 1 Classiﬁcation results on ImageNet. “*” denotes our retrained baseline. The training time memory footprint is mea-
sured with a batch size of 128. All models use the default mixed-precision training [10] as in baselines [3, 9, 31]. We measure
the total training time by GPU hours w.r.t. a single NVIDIA V100 GPU. Note that for inference, Mesa does not increase the
model parameter (1st column) and does not aﬀect the computational cost measured in FLOPs (2nd column).
Method
Param (M)
FLOPs (G)
Train Memory (MB)
GPU Hours
Top-1@Acc. (%)
DeiT-Ti* [31]
5
1.3
4,171
440
71.9
DeiT-Ti w/ Mesa
5
1.3
1,858
540
72.1
DeiT-S
22
4.6
8,459
520
79.8
DeiT-S w/ Mesa
22
4.6
3,840
620
80.0
DeiT-B
86
17.5
17,691
600
81.8
DeiT-B w/ Mesa
86
17.5
8,616
1,170
81.8
Swin-Ti [9]
29
4.5
11,812
480
81.3
Swin-Ti w/ Mesa
29
4.5
5,371
820
81.3
PVT-Ti [3]
13
1.9
7,800
520
75.1
PVT-Ti w/ Mesa
13
1.9
3,782
600
74.9
which contains ∼1.2M training images from 1K cate-
gories and 50K validation images. Following the com-
mon practice [26, 31], we measure the model perfor-
mance by Top-1 accuracy. In addition, we also report
the memory consumption at training time and the over-
all GPU hours for training each model.
Compared methods. To demonstrate the eﬀective-
ness of Mesa, we evaluate our framework on several
state-of-the-art vision Transformers, including DeiT [31],
Swin [9] and PVT [3]. DeiT is a standard vision Trans-
former which inherits the similar architecture from the
original Transformer [25]. Swin and PVT are recently
proposed hierarchical vision Transformers (HVTs) which
achieve promising results on various vision tasks. More-
over, as recent models usually have multiple variants in
terms of the model depth and width, we denote them
as “Model-Ti/S/B” to represent their tiny, small and
base settings.
Implementation details. By default, all models are
trained on 8 V100 GPUs with a total batch size of 1,024
(128 per GPU) on ImageNet. We adopt AdamW [49]
optimizer with a cosine decay learning rate scheduler.
We set the initial learning rate and weight decay as
1 × 10−3 and 5 × 10−2, respectively. Furthermore, we
adopt the same training strategies when comparing to
each baseline. All experiments have adopted automatic
mixed-precision training [10] (also called FP16 or half-
precision training) as it is widely used in recent work to
accelerate the training process. The λ in Eqs. (6) and
(7) is set to 0.9, which is determined by a simple grid
search on CIFAR-100. α and β are initialised by the
min-max values from the activations at the ﬁrst train-
ing iteration. In Section 5.2, we conduct experiments of
using diﬀerent λ to train DeiT-Ti on CIFAR-100 and
provide visualisations for the evolution of quantization
parameters.
Results. In Table 1, we report the ImageNet classiﬁ-
cation results of training DeiT and recent HVTs with
Mesa. In general, Mesa can reduce around half of the
memory consumption at training time while achieving
comparable or even better performance than the strong
baselines. For example, on DeiT-B and Swin-Ti, Mesa
achieves the same performance as the baselines while
reducing the memory footprint by 51% and 55% 1, re-
spectively. Remarkably, DeiT-Ti/S with Mesa even out-
perform the baselines by 0.2% in the Top-1 accuracy.
Our conjecture is that the approximated activations
help to regularize the stochastic gradients when training
Transformers, which therefore improves the model per-
formance. Apart from this, we also visualize the training
curves of DeiT-B with Mesa in Figure 5. As it shows, all
1 AMP [10] training stores mixed FP16 and FP32 activa-
tions. With Mesa only Int8 activations are saved, and thus
we can achieve more than 2× memory reduction.


---

8
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
Table 2 Performance comparisons on DeiT-Ti with Mesa by using per-sample statistics (PS) and running estimates (RE).
We report the Top-1 accuracy on ImageNet and CIFAR-100. “*” denotes our retrained baseline. Both experiments adopt the
PyTorch implementation of Mesa.
Method
Train Memory (MB)
Train Throughput (images/s)
ImageNet Top-1(%)
CIFAR-100 Top-1(%)
DeiT-Ti* [31]
4,149
1,196
71.9
64.8
+ Mesa w/ PS
2,117
372
71.9
65.1
+ Mesa w/ RE
2,000
431
72.1
65.2
Table 3 Performance comparisons on DeiT with Mesa under stochastic rounding (SR) and nearest rounding (NR). We report
the Top-1 accuracy on CIFAR-100 and ImageNet.
Method
Train Memory (MB)
Train Throughput (images/s)
ImageNet Top-1(%)
CIFAR-100 Top-1(%)
DeiT-Ti [31]
4,149
1,196
71.9
64.8
+ Mesa w/ NR
1,855
635
failed
64.7
+ Mesa w/ SR
1,855
586
72.1
65.2
curves under Mesa are consistent with those of baseline
DeiT-B or even perform slightly better. For PVT-Ti, we
observe a slight performance drop of 0.2% in the Top-1
accuracy. Our speculation is that PVTs are quite sensi-
tive to train as the authors ﬁnd that a deeper PVT even
cannot converge with the same settings of PVT-Ti/S 2.
In practice, the impact of Mesa on the model perfor-
mance can be architecture-dependent. In general, Mesa
performs consistently well on representative models. As
shown in Table 1, the eﬀect on the performance is only
around ±0.2% for the compared baselines.
Besides, we notice that Mesa slows down the train-
ing speed. As discussed in Section 4, the space-time
trade-oﬀis a common practice in the literature. In Ta-
ble 1, the experiments with Mesa compress almost all
operations to achieve signiﬁcant memory reduction, thus
slowing down the throughput the most. However, in
practice, one can determine the most memory-hungry
operations with Mesa to achieve the ideal memory-saving
target with acceptable training speed. We will show in
Table 7 and Table 13 that Mesa is able to compress
diﬀerent operations to achieve ﬂexible memory-savings.
Furthermore, the training overhead can be largely oﬀ-
set by the data loading and communication cost in dis-
tributed training. For example, the training time of
PVT-Ti with Mesa on ImageNet is only 15% longer
(520 vs. 600 GPU hours). However, we also notice that
the total GPU hours for training DeiT-B and Swin-Ti
with Mesa are almost doubled. This suggests that train-
ing speed reduction varies among diﬀerent architectures
and model sizes, while the worst case may double the
training time.
5.2 Ablation Studies
In this section, we provide ablation studies of Mesa. By
default, we use a CUDA implementation of Mesa for
2 https://github.com/whai362/PVT/issues/2
the following experiments. The throughput and mem-
ory consumption at training time are measured with a
batch size of 128 and an image resolution of 224×224
on a single NVIDIA RTX 3090 GPU. Unless otherwise
speciﬁed, we adopt the same training strategy as in Sec-
tion 5.1 for ImageNet experiments. On CIFAR-100, we
train models with a total batch size of 256 on 2 GPUs
and keep all other experiment settings as same as in
Section 5.1 except that the initial learning rate is lin-
early scaled to 2.5 × 10−4.
5.2.1 Per-sample Statistics vs. Running Estimates for
Updating Quantization Parameters
Previous ACT frameworks such as TinyScript [17] and
ActNN [16] use per-sample statistics (PS) to calculate
the quantization parameters. As discussed in 3.2, such
approach can result in more additional memory con-
sumption and computational cost during training. In
Table 2, we compare the approach of using per-sample
statistics with using running estimates (RE) on Ima-
geNet and CIFAR-100. From the results, we observe
that while both methods perform favourably against
the baseline in terms of the Top-1 accuracy, the strat-
egy of using running estimates achieves better perfor-
mance, less memory consumption and faster through-
put than using per-sample statistics. Note that both of
the PS/RE experiments in Table 2 adopt a PyTorch im-
plementation of Mesa, thus the memory consumption
and throughput at training time are slightly diﬀerent
from that of CUDA implementation. Besides, although
the throughput on a single GPU is reduced by using
Mesa, the total training time on ImageNet is similar
to that of baseline DeiT-Ti as the communication over-
heads have more impact on the training speed under
the distributed training scheme.


---

Mesa: A Memory-saving Training Framework for Transformers
9
Table 4 Performance comparisons between symmetric quan-
tization and asymmetric quantization on DeiT-Ti with Mesa.
Both experiments adopt a PyTorch implementation of Mesa.
We report the Top-1 accuracy on CIFAR-100. “sym” and
“asym” denote symmetric and asymmetric quantization, re-
spectively.
Method
Train Memory
(MB)
Train Throughput
(images/s)
Top-1
(%)
DeiT-Ti [31]
4,149
1,196
64.8
+ Mesa w/ sym
2,045
472
63.2
+ Mesa w/ asym
2,000
431
65.2
Table 5 Performance comparisons on DeiT-Ti with Mesa
under diﬀerent quantization granularities on CIFAR-100.
“Mesa w/ Layer” means we train DeiT-Ti with Mesa under
the layer-wise quantization. “Head” indicates our proposed
head-wise quantization
Method
Train Memory
(MB)
Train Throughput
(images/s)
Top-1
(%)
DeiT-Ti [31]
4,168
1,196
64.8
+ Mesa w/ Layer
1,855
655
64.7
+ Mesa w/ Head
1,855
586
65.2
Table 6 Performance comparisons of diﬀerent λ based on
DeiT-Ti. We report the Top-1 accuracy on CIFAR-100.
Method
λ
Top-1 (%)
DeiT-Ti [31]
-
64.8
+ Mesa
0.0
64.5
0.9
65.2
0.99
65.0
0.999
failed
5.2.2 Stochastic Rounding vs. Nearest Rounding
To explore the eﬀect of stochastic rounding in Mesa, we
compare it with the commonly used nearest rounding
by training DeiT-Ti with Mesa on CIFAR-100 and Im-
ageNet. We report the results in Table 3. As it shows,
although training DeiT-Ti with nearest rounding can
achieve competitive results on CIFAR-100, it fails to
converge on ImageNet. Therefore, we speculate that
stochastic rounding is important to guarantee good per-
formance in Mesa. Also note that stochastic rounding
does not increase the memory footprint during training.
However, it is slightly slower than that of using nearest
rounding, which attributes to the additional overhead
from the implementation of stochastic rounding.
5.2.3 Asymmetric Quantization vs. Symmetric
Quantization
Apart from the asymmetric quantization that used in
Mesa, symmetric quantization is also widely adopted
in the previous model quantization works [22,32]. Un-
der this scheme, the input X is quantized by a scale
factor s = (2b −1)/max(|X|) only, where b is the bit
width. In Table 4, we compare the two quantization
schemes based on CIFAR-100. From the results, we ob-
serve that although symmetric quantization achieves
faster throughput during training, it does not surpass
the baseline in terms of the Top-1 accuracy. The re-
sult is also consistent with a previous observation [41]
for quantizing BERT models. In fact, the activations
in Transformers are skewed into negative values. We
assume asymmetric quantization can provide a more
tighter clipping range such that it helps to minimize
the quantization error during training.
5.2.4 Eﬀect of Diﬀerent Quantization Granularities
To explore the eﬀect of diﬀerent quantization granular-
ities in Mesa, we train DeiT-Ti with Mesa and com-
pare the proposed strategy to layer-wise quantization
on CIFAR-100. The results are shown in Table 5. Over-
all, beneﬁting from using running estimates in Mesa, all
strategies consume a similar amount of memory during
training as each layer only needs to save a few quan-
tization parameters. With the layer-wise quantization,
all activations at the same layer are quantized based on
the same clipping range and oﬀset, but it does not out-
perform the baseline. On the other hand, the proposed
head-wise quantization utilizes the bucketing strategies
as in Figure 3 which distinguishes diﬀerent statistics
over diﬀerent heads and channel groups. It achieves
better performance than both the layer-wise strategy
and the baseline, while making a good trade-oﬀto the
training speed. Additionally, we notice that grouping
activations (except queries, keys, values and attentions)
based on each hidden channel makes DeiT-Ti fail to
converge on ImageNet. This suggests an appropriate
quantization granularity is needed to stabilize Trans-
former training.
5.2.5 Eﬀect of Diﬀerent Decay Rates to Learn
Quantization Parameters
In Mesa, we utilise running estimates to learn the quan-
tization parameters, which requires a decay rate λ to
tune. To understand the eﬀect of diﬀerent λ in Mesa,
we conduct experiments with DeiT-Ti on CIFAR-100
and report the results in Table 6. Overall, we ﬁnd that
a suitable λ is quite essential to help Mesa achieve good
performance. Particularly, when λ is 0, the quantization
parameters only rely on the current batch statistics at
each training iteration. However, such an approach can-
not outperform the baseline. Besides, we ﬁnd a large
λ make DeiT-Ti fail to converge as it cannot timely
adapt to the dynamic activation distributions during


---

10
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
Table 7 Performance comparisons of compressing diﬀerent operations based on DeiT-Ti with Mesa. We report the Top-1
accuracy on CIFAR-100.
Method
Train Memory (MB)
Train Throughput (images/s)
Top-1 (%)
DeiT-Ti [31]
4,149
1,196
64.8
+ Mesa w/ MatMul
3,505 (-15.5%)
729
65.3
+ Mesa w/ GELU
3,540 (-14.7%)
1,031
64.9
+ Mesa w/ LayerNorm
3,844 (-7.4%)
1,059
64.4
+ Mesa w/ Softmax
3,485 (-16.0%)
998
64.8
+ Mesa w/ All
1,855 (-55.3%)
586
65.2
Table 8 Performance comparisons of compressing diﬀerent modules based on DeiT-Ti with Mesa. We report the Top-1
accuracy on CIFAR-100.
Method
Train Memory (MB)
Train Throughput (images/s)
Top-1 (%)
DeiT-Ti [31]
4,149
1,196
64.8
+ Mesa w/ MSA
3,037 (-26.8%)
772
65.0
+ Mesa w/ FFN
3,294 (-20.6%)
888
64.8
+ Mesa w/ MSA + FFN
1,856 (-55.3%)
597
64.9
Table 9 Comparisons of the largest models Mesa can train
before out-of-memory with the same batch size of 128 on
one NVIDIA V100 32GB GPU. “Depth” refers to the model
depth or the number of Transformer blocks. “Width” means
the model width or the number of self-attention heads at
each block. “Resolution” denotes the input image resolution
during training.
Dim
DeiT-B
DeiT-B w/ Mesa
Ratio
Depth
12
40
3.3×
Width
12
26
2.2×
Resolution
224
336
1.5×
training. Overall, we ﬁnd 0.9 achieves the best perfor-
mance in practice, in which case we set λ as 0.9 for all
experiments by default.
5.2.6 Eﬀect of Compressing Diﬀerent Operations
It is worth to mention that Mesa can compress diﬀerent
operations, or any combinations (e.g., GELU and Lay-
erNorm) to achieve diﬀerent memory-speed trade-oﬀs.
For example, a standard Transformer [26, 31] consists
of matrix multiplication (MatMul), GELU, LayerNorm
and Softmax. The activations that are generated from
these operations consume most of the GPU memory
during training. In Table 7, we report the results of
compressing diﬀerent operations in DeiT-Ti based on
CIFAR-100. In general, training with Mesa can achieve
on par or even better performance than the baseline.
At the same time, compressing diﬀerent operations can
achieve diﬀerent memory-savings while also introducing
diﬀerent training overheads. When compressing all op-
erations, we achieve the most memory-saving (around
50%) while slowing down the training speed on a single
GPU by half. In practice, Mesa allows researchers to
ﬂexibly set the trade-oﬀaccording to their own target
memory-saving and acceptable training overhead. We
show in Section 5.2.12 that by using diﬀerent combina-
tions of the compressed operations we can train larger
models with a better memory-speed trade-oﬀ.
5.2.7 Eﬀect of Compressing Diﬀerent Modules
MSA and FFN layers are the main modules of a Trans-
former model. Meanwhile, they consume most of the
GPU memory at training time. To study the eﬀect of
compressing diﬀerent modules, we train DeiT-Ti with
Mesa on CIFAR-100 and report the results in Table 8.
Overall, training DeiT-Ti with Mesa achieves on par or
better performance compared to the baseline. In partic-
ular, compressing the MSA or FFN layers in DeiT-Ti
can reduce 27% and 21% memory footprint at training
time, respectively. However, while compressing MSA
layers can save more memory, it also results in slower
throughput during training compared to compressing
FFN layers. Finally, compressing MSA and FFN layers
simultaneously brings the most memory savings, but it
also leads to the slowest training speed. However, this
overhead could be oﬀset by the communication cost in
distributed learning. In practice, one can also target
diﬀerent modules in a model to achieve an acceptable
memory-speed trade-oﬀ.
5.2.8 Largest Models that Mesa can Train
With the help of Mesa, we can re-invest the reduced
memory by constructing a larger model or training with
a larger image resolution. In Table 9, we report the
largest models that Mesa can train before out-of-memory
based on DeiT-B. Overall, Mesa is able to scale up the
model depth by 3.3× and width by 2.2×. Moreover,


---

Mesa: A Memory-saving Training Framework for Transformers
11
Table 10 Performance comparisons with larger batch size based on DeiT-Ti and Swin-S with Mesa. “Total Memory” indicates
the total memory consumption over 8 GPUs. The GPU hours are calculated w.r.t. a single NVIDIA V100 GPU. We report
the Top-1 accuracy on ImageNet.
Method
Batch Size
Total Memory (GB)
GPU Hours
Top-1 (%)
DeiT-Ti [31]
1,024
33.4
440
71.9
DeiT-Ti w/ Mesa
2,048
28.8
500
72.9
Swin-S [9]
1,024
150.7
968
83.0
Swin-S w/ Mesa
2,048
120.8
1,160
83.1
Table
11 Sementic
segmentation
performance
on
ADE20K
[50]
based
on
Semantic
FPN
[51].
We
use
Swin-Ti as backbone and measure the performance by mIoU.
Backbone
Batch Size
GPUs
mIoU (%)
Swin-Ti [9]
16
8
41.5
Swin-Ti w/ Mesa
16
8
42.2
Swin-Ti w/ Mesa
16
2
42.4
Table 12 Performance comparison between ActNN and the
proposed Mesa based on DeiT-Ti. We adopt the default set-
ting (L3, 2-bit) for ActNN. We report the Top-1 accuracy on
CIFAR-100.
Method
Train Memory
(MB)
Train Throughput
(images/s)
Top-1
(%)
DeiT-Ti [31]
4,149
1,196
64.8
+ ActNN [16]
3,514 (-15.3%)
635 (-46.9%)
64.9
+ Mesa
1,855 (-55.3%)
586 (-51.0%)
65.2
Mesa can train DeiT-B with 1.5× larger image reso-
lution. In practice, one can also decrease the expan-
sion ratios in the FFN layers to scale up more on the
depth/width/resolution. In Table 9, we adopt the de-
fault expansion ratio of four in DeiT-B. Furthermore,
we will show in Section 5.2.10 that Mesa enables us to
train models with a larger batch size under the same
memory budget.
5.2.9 Semantic Segmentation
To explore the performance of Mesa on downstream
tasks, we use Swin-Ti as backbone and Semantic FPN [51]
as the framework to evaluate the performance of seman-
tic segmentation on ADE20K [50]. Following common
practice [52], we use AdamW optimizer with a poly
learning rate schedule and train models with 80,000 it-
erations. We set the initial learning rate to 1×10−4. All
backbones are pretrained on the ImageNet dataset. As
Table 11 shows, training with Mesa outperforms the
baseline by 0.7% in mIoU. Furthermore, with the re-
duced memory footprint, we train the model with only
two GPUs under the same setting while achieving 0.2%
more gains in mIoU, which again demonstrates the ad-
vantage of Mesa under the ﬁnite hardware resources.
5.2.10 Eﬀect of Larger Batch Size under the Same
Memory Budget
In our main results on ImageNet classiﬁcation, we have
shown that Mesa is able to reduce around half of the
memory consumption at training time. More impor-
tantly, with Mesa, it is possible to train a Transformer
with a larger batch size under the same memory budget.
For example, training DeiT-Ti with the default strat-
egy requires at least 34GB GPU memory under a total
batch size of 1,024. However, with the same memory
budget, we can double the batch size to better utilise
available GPU cores and explore potential beneﬁts of
using larger batch size. Speciﬁcally, we train DeiT-Ti
with Mesa using a total batch size of 2,048 on Ima-
geNet with 8 NVIDIA V100 GPUs. As we can see from
Table 10, Mesa still consumes less memory than that
of mixed-precision training with a batch size of 1,024
(33.4GB vs. 28.8GB). Furthermore, it achieves 0.8%
gains in the Top-1 accuracy. Moreover, while the default
training for Swin-S with a batch size of 2,048 will result
in out of memory on 8 V100 GPUs, training with Mesa
under the same batch size again consumes less memory
and even achieves 0.1% improvement in the Top-1 accu-
racy compared to the baseline. The overall performance
indicates that Mesa enables us to train models with a
larger batch size under the same memory budget, while
still performing favourably against baselines.
5.2.11 Compared to ActNN
Existing ACT frameworks include BLPA [15], TinyScript
[17] and ActNN [16]. However, both BLPA and TinyScript
target on speciﬁc CNN architectures (e.g., ResNet [6]),
making it diﬃcult for applying their methods into Trans-
formers. Besides, ActNN only experiments with CNN-
based models which ignores the heterogeneously dis-
tributed activations in MSA layers. In Table 12, we
train DeiT-Ti with the default setting (L3, 2-bit) of
ActNN on CIFAR-100. Compared to ActNN, Mesa cov-
ers more layers and achieves better memory-saving and
performance with a slightly slower throughput. Further-
more, Mesa can achieve a better memory-speed trade-
oﬀthan ActNN by targeting diﬀerent operations. For


---

12
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
G
G+LN+S
G+LN
L0
L1
L3
L4
2000
2500
3000
3500
4000
4500
TrainMemory (MB)
DeiT-Ti
w/ ActNN
w/ Mesa
G+LN+S+M
L0
L1
L3
L4
600
700
800
900
1000
1100
G
G+LN+S
G+LN+S+M
G+LN
1200
Train Throughput (images/s)
DeiT-Ti
w/ ActNN
w/ Mesa
Fig. 6 Memory consumption and throughput comparison at training time between ActNN and Mesa based on DeiT-Ti.
“L0”,“L1”,“L3” and “L4” refer to diﬀerent optimization levels in ActNN [16]. Note that L2 in ActNN achieves the same
memory reduction as L1, and L5 does not further reduce memory cost. Therefore, we exclude L2 and L5 for brevity. We also
denote “G”, “LN”, “S” and “M” as the GELU, LayerNorm, Softmax and MatMul operations, respectively. “+” means we
simultaneously compress multiple operations with Mesa. For example, “G+LN” represents for compressing both the GELU and
LayerNorm in DeiT. Compared to the diﬀerent optimization levels in ActNN, Mesa can ﬂexibly combine diﬀerent operations
and achieve better memory-speed trade-oﬀ. Best viewed in color.
Table 13 Experiments with Swin-B/L on CIFAR-100. Memory footprint is measured with a batch size of 64. “G”, “S”, “LN”
denote GELU, Softmax, LayerNorm, respectively. “all” means we compress all types of operations.
Method
Epochs
Train Memory (MB)
Top-1 (%)
Throughput (images/s)
Swin-B
300
13,717
78.0
111
w/ Mesa all
300
6,708 (-51%)
78.0
55 (-50%)
w/ Mesa G+S
300
9,714 (-29%)
78.4
97 (-13%)
Swin-L
90
21,851
64.5
83
w/ Mesa all
90
11,708 (-46%)
64.6
38 (-54%)
w/ Mesa G+S+LN
90
13,943 (-36%)
65.0
67 (-19%)
Table 14 Comparison with checkpointing (CP) and gradi-
ent accumulation (GA) based on Swin-Ti. Mesa is comple-
mentary to these techniques.
Method
Batch Size
Memory (MB)
Throughput
Swin-Ti
128
11,812
356
w/ Mesa
128
6,741 (-43%)
276 (-22%)
w/ CP
128
6,885
303
w/ Mesa + CP
128
4,353 (-63%)
244 (-31%)
w/ GA
64
6,209
333
w/ Mesa + GA
64
3,682 (-69%)
257 (-28%)
example, in Figure 6, we show the advantage of Mesa
by comparing it with the diﬀerent optimization levels
in ActNN.
5.2.12 Eﬀect of Training on Larger Models
To explore the eﬀect of training larger models with
Mesa, we train Swin-Base and Swin-Large with Mesa on
CIFAR-100. Following the default setting in Swin [9],
we train Swin-Large for 90 epochs. As shown in Ta-
ble 13, Mesa reduces half memory footprint while keep-
ing their performance when compressing all operations.
Moreover, since the throughput is controllable, we can
compress the selected operations to achieve a better
speed-memory trade-oﬀwith even slightly higher accu-
racy on CIFAR-100.
5.2.13 Compared to Other Memory-saving Techniques
As discussed in Section 4, ACT frameworks including
Mesa are complementary to checkpointing (CP), gra-
dient accumulation (GA) and AMP. In fact, CP aims
to store fewer tensors while ACT compresses saved ten-
sors. Besides, GA saves memory by reducing the batch
size, which makes it diﬃcult for a fair comparison with
related works. Moreover, even if most Transformers adopt
LayerNorm instead of BatchNorm [53], training ViTs
such as DeiT can be sensitive to batch size. Therefore,
reducing batch size can hurt model performance 3. In
contrast, Mesa saves memory without modifying the ex-
perimental hyperparameters. Furthermore, Mesa is or-
thogonal to AMP. During training, Mesa uses exact ac-
tivations during forward pass while storing Int8 activa-
tions for backpropagation to reduce memory consump-
tion, while AMP quantizes weights, activations and gra-
dients into FP16.
In Table 14, we compare Mesa with CP and GA.
Overall, Mesa can save more memory while achieving
comparable speed with CP. Note that CP has been
highly optimized by PyTorch team, which is slightly
not fair compared to our implementation. More impor-
3 https://github.com/facebookresearch/deit/issues/
63


---

Mesa: A Memory-saving Training Framework for Transformers
13
Block-1
Block-12
0
100
200
300
Epoch
0.6
0.4
0.2
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
0.5
0.4
0.3
0.2
Value
beta-1
beta-2
beta-3
Query
0
100
200
300
Epoch
0.25
0.50
0.75
1.00
1.25
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
0.4
0.6
0.8
1.0
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
6
4
2
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
3.5
3.0
2.5
2.0
1.5
Value
beta-1
beta-2
beta-3
Value
0
100
200
300
Epoch
2.5
5.0
7.5
10.0
12.5
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
4
6
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
15
10
5
0
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
20
10
0
Value
beta-1
beta-2
beta-3
Attention
(before
softmax)
0
100
200
300
Epoch
0
10
20
30
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
0
20
40
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
30
20
10
0
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
5
4
3
2
Value
beta-1
beta-2
beta-3
Key
0
100
200
300
Epoch
0
20
40
60
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
4
6
8
10
Value
alpha-1
alpha-2
alpha-3
Block-1
Block-12
Fig. 7 Evolution of α and β in the MSA layers of DeiT-Ti. We visualize the results for diﬀerent blocks (columns) and
activations (rows). Diﬀerent colors represent diﬀerent quantization groups. Best viewed in color.
Block-1
FC1
GELU
FC2
Block-12
0
100
200
300
Epoch
5
10
15
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
2
4
6
8
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
8
10
12
14
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
10
15
20
25
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
2
3
4
5
6
Value
alpha-1
alpha-2
alpha-3
0
100
200
300
Epoch
4
6
8
10
Value
alpha-1
alpha-2
alpha-3
Block-1
Block-12
0
100
200
300
Epoch
7
6
5
4
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
8
6
4
2
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
12
10
8
6
4
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
6
4
2
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
0.175
0.170
0.165
Value
beta-1
beta-2
beta-3
0
100
200
300
Epoch
0.175
0.170
0.165
Value
beta-1
beta-2
beta-3
Fig. 8 Evolution of α and β in the FFN layers of DeiT-Ti.


---

14
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
tantly, Mesa can be combined with these techniques for
more memory-saving due to complements.
6 Visualization
In this section, we visualize the evolution of α and β
in DeiT-Ti during training on CIFAR-100. We set the
number of quantization groups at each layer to 3.
6.1 Evolution of α and β in MSA Layers
In Figure 7, we show the evolution of α and β in MSA
layers during training, respectively. In general, the quan-
tization parameters at each head evolve diﬀerently, em-
phasizing the necessity of head-wise activation quan-
tization in MSA layers. Besides, we ﬁnd the quantiza-
tion clipping range α increases signiﬁcantly at the early
stages, then becomes stable or deceases at the later
stages. On the other hand, the quantization oﬀset β
keeps decreasing at the beginning while tends to be sta-
ble or increases later during training. In particular, we
ﬁnd the quantization oﬀsets are skewed to negative val-
ues, which indicates that the activations in MSA layers
contain more negative values, except for the attentions
after Softmax as they are non-negative values.
6.2 Evolution of α and β in FFN Layers
Figure 8 shows the evolution of α and β in FFN layers
during training, respectively. Overall, the phenomenon
is quite similar to that of MSA layers. In particular,
we ﬁnd the quantization oﬀset at the second FC layer
of FFN is always the minimum value (-0.17) of GELU.
This indicates that β can be ﬁxed to -0.17 at this layer
during training, which may help to reduce more training
overhead. We leave the space of further optimization for
future work.
7 Conclusion
In this work, we have presented Mesa, a memory-saving
resource-eﬃcient training framework for Transformers.
Speciﬁcally, we save a low-precision approximated acti-
vations during training to achieve memory saving while
using exact activations for the forward pass. The saved
activations are used to calculated gradients during the
backpropagation. Moreover, we identify the heteroge-
neous activation distributions in an MSA layer of a
Transformer. For this, we proposed a head-wise acti-
vation quantization strategy, which groups the activa-
tions based on each self-attention head to capture bet-
ter quantization clipping ranges and oﬀsets. With Mesa,
we can re-invest the reduced memory footprint by con-
structing a larger model or training with a larger batch
size to explore potential beneﬁts. Extensive experiments
on ImageNet, CIFAR-100 and ADE20K have shown
that Mesa can achieve ﬂexible memory-savings during
training while achieving comparable or even better per-
formance than the default training strategy.
References
1. J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT:
pre-training of deep bidirectional transformers for lan-
guage understanding,” in NAACL-HLT, 2019, pp. 4171–
4186.
2. Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma,
and R. Soricut, “ALBERT: A lite BERT for self-
supervised learning of language representations,” in
ICLR, 2020.
3. W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang,
T. Lu, P. Luo, and L. Shao, “Pyramid vision transformer:
A versatile backbone for dense prediction without convo-
lutions,” in ICCV, 2021.
4. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kir-
illov, and S. Zagoruyko, “End-to-end object detection
with transformers,” in ECCV, 2020, pp. 213–229.
5. S. Zheng, J. Lu, H. Zhao, X. Zhu, Z. Luo, Y. Wang,
Y. Fu, J. Feng, T. Xiang, P. H. S. Torr, and L. Zhang,
“Rethinking semantic segmentation from a sequence-to-
sequence perspective with transformers,” in CVPR, 2021,
pp. 6881–6890.
6. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual
learning for image recognition,” in CVPR, 2016, pp. 770–
778.
7. M. Tan and Q. Le, “EﬃcientNet: Rethinking model scal-
ing for convolutional neural networks,” in ICML, 2019,
pp. 6105–6114.
8. T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Ka-
plan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sas-
try, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger,
T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu,
C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin,
S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish,
A. Radford, I. Sutskever, and D. Amodei, “Language
models are few-shot learners,” in NeurIPS, 2020.
9. Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin,
and B. Guo, “Swin transformer: Hierarchical vision trans-
former using shifted windows,” in ICCV, 2021.
10. P.
Micikevicius,
S.
Narang,
J.
Alben,
G.
F.
Di-
amos, E. Elsen, D. Garc´ıa, B. Ginsburg, M. Houston,
O. Kuchaiev, G. Venkatesh, and H. Wu, “Mixed preci-
sion training,” in ICLR, 2018.
11. H. Mostafa and X. Wang, “Parameter eﬃcient training
of deep convolutional neural networks by dynamic sparse
reparameterization,” in ICML, 2019, pp. 4646–4655.
12. Y. Huang, Y. Cheng, A. Bapna, O. Firat, D. Chen, M. X.
Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, and Z. Chen,
“Gpipe: Eﬃcient training of giant neural networks using
pipeline parallelism,” in NIPS, 2019, pp. 103–112.
13. T. Chen, B. Xu, C. Zhang, and C. Guestrin, “Training
deep nets with sublinear memory cost,” arXiv preprint
arXiv:1604.06174, 2016.
14. D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vo-
jnovic, “QSGD: communication-eﬃcient SGD via gradi-
ent quantization and encoding,” in NIPS, 2017, pp. 1709–
1720.


---

Mesa: A Memory-saving Training Framework for Transformers
15
15. A. Chakrabarti and B. Moseley, “Backprop with approx-
imate activations for memory-eﬃcient network training,”
in NIPS, 2019, pp. 2426–2435.
16. J. Chen, L. Zheng, Z. Yao, D. Wang, I. Stoica, M. W.
Mahoney, and J. Gonzalez, “Actnn: Reducing training
memory footprint via 2-bit activation compressed train-
ing,” in ICML, 2021, pp. 1803–1813.
17. F. Fu, Y. Hu, Y. He, J. Jiang, Y. Shao, C. Zhang, and
B. Cui, “Don’t waste your bits! squeeze activations and
gradients for deep neural networks via tinyscript,” in
ICML, 2020, pp. 3304–3314.
18. D. Hendrycks and K. Gimpel, “Gaussian error linear
units (gelus),” arXiv: Learning, 2016.
19. J. Ba, J. Kiros, and G. E. Hinton, “Layer normalization,”
ArXiv, vol. abs/1607.06450, 2016.
20. H. Jegou, M. Douze, and C. Schmid, “Product quanti-
zation for nearest neighbor search,” PAMI, no. 1, pp.
117–128, 2010.
21. J. Cordonnier, A. Loukas, and M. Jaggi, “On the relation-
ship between self-attention and convolutional layers,” in
ICLR, 2020.
22. J.
Choi,
Z.
Wang,
S.
Venkataramani,
P.
I.-J.
Chuang,
V.
Srinivasan,
and
K.
Gopalakrishnan,
“PACT: Parameterized clipping activation for quan-
tized
neural
networks,”
2018.
[Online].
Available:
https://openreview.net/forum?id=By5ugjyCb
23. Y. Bhalgat, J. Lee, M. Nagel, T. Blankevoort, and
N.
Kwak,
“LSQ+:
improving
low-bit
quantization
through learnable oﬀsets and better initialization,” in
CVPR, 2020, pp. 2978–2985.
24. P. Indyk and R. Motwani, “Approximate nearest neigh-
bors: Towards removing the curse of dimensionality,” in
Proceedings of the Thirtieth Annual ACM Symposium
on the Theory of Computing, Dallas, Texas, USA, May
23-26, 1998.
ACM, 1998, pp. 604–613.
25. A.
Vaswani,
N.
Shazeer,
N.
Parmar,
J.
Uszkoreit,
L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin,
“Attention is all you need,” in NeurIPS, 2017, pp. 5998–
6008.
26. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn,
X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer,
G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An
image is worth 16x16 words: Transformers for image
recognition at scale,” ICLR, 2021.
27. Z. Pan, B. Zhuang, J. Liu, H. He, and J. Cai, “Scalable
visual transformers with hierarchical pooling,” in ICCV,
2021.
28. L. Yuan, Y. Chen, T. Wang, W. Yu, Y. Shi, F. E. Tay,
J. Feng, and S. Yan, “Tokens-to-token vit: Training vision
transformers from scratch on imagenet,” in ICCV, 2021.
29. B. Chen, P. Li, C. Li, B. Li, L. Bai, C. Lin, M. Sun,
J. Yan, and W. Ouyang, “Glit: Neural architecture search
for global and local image transformer,” in ICCV, 2021.
30. M. Chen, H. Peng, J. Fu, and H. Ling, “Autoformer:
Searching transformers for visual recognition,” in ICCV,
2021.
31. H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablay-
rolles, and H. J´egou, “Training data-eﬃcient image trans-
formers & distillation through attention,” in ICML, 2021.
32. S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy,
and D. S. Modha, “Learned step size quantization,” in
ICLR, 2020.
33. S. Jung, C. Son, S. Lee, J. Son, J. Han, Y. Kwak,
S. J. Hwang, and C. Choi, “Learning to quantize deep
networks by optimizing quantization intervals with task
loss,” in CVPR, 2019, pp. 4350–4359.
34. P. Wang, Q. Chen, X. He, and J. Cheng, “Towards accu-
rate post-training network quantization via bit-split and
stitching,” in ICML, 2020, pp. 9847–9856.
35. M.
Nagel,
M.
van
Baalen,
T.
Blankevoort,
and
M. Welling, “Data-free quantization through weight
equalization and bias correction,” in ICCV, 2019.
36. M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi,
“Xnor-net: Imagenet classiﬁcation using binary convolu-
tional neural networks,” in ECCV, 2016, pp. 525–542.
37. M. Courbariaux, Y. Bengio, and J. David, “Binarycon-
nect: Training deep neural networks with binary weights
during propagations,” in NIPS, 2015, pp. 3123–3131.
38. O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat,
“Q8BERT: quantized 8bit BERT,” in EMC2@NeurIPS,
2019, pp. 36–39.
39. S. Shen, Z. Dong, J. Ye, L. Ma, Z. Yao, A. Gholami,
M. W. Mahoney, and K. Keutzer, “Q-BERT: hessian
based ultra low precision quantization of BERT,” in
AAAI, 2020, pp. 8815–8821.
40. H. Bai, W. Zhang, L. Hou, L. Shang, J. Jin, X. Jiang,
Q. Liu, M. R. Lyu, and I. King, “Binarybert: Pushing
the limit of BERT quantization,” in ACL/IJCNLP, 2021,
pp. 4334–4348.
41. W. Zhang, L. Hou, Y. Yin, L. Shang, X. Chen, X. Jiang,
and Q. Liu, “Ternarybert: Distillation-aware ultra-low bit
BERT,” in EMNLP, 2020, pp. 509–521.
42. J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin,
Q. V. Le, M. Z. Mao, M. Ranzato, A. W. Senior, P. A.
Tucker, K. Yang, and A. Y. Ng, “Large scale distributed
deep networks,” in NIPS, 2012, pp. 1232–1240.
43. W. Wen, C. Xu, F. Yan, C. Wu, Y. Wang, Y. Chen, and
H. Li, “Terngrad: Ternary gradients to reduce communi-
cation in distributed deep learning,” in NIPS, 2017, pp.
1509–1519.
44. M. Croci, M. Fasi, N. Higham, T. Mary, and M. Mikaitis,
“Stochastic rounding: Implementation, error analysis,
and applications,” 2021.
45. B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. G.
Howard, H. Adam, and D. Kalenichenko, “Quantiza-
tion and training of neural networks for eﬃcient integer-
arithmetic-only inference,” in CVPR, 2018, pp. 2704–
2713.
46. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury,
G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga,
A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison,
A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai,
and S. Chintala, “Pytorch: An imperative style, high-
performance deep learning library,” in NIPS, 2019, pp.
8024–8035.
47. K. Wang, Z. Liu, Y. Lin, J. Lin, and S. Han, “HAQ:
hardware-aware automated quantization with mixed pre-
cision,” in CVPR, 2019, pp. 8612–8620.
48. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein
et al., “Imagenet large scale visual recognition challenge,”
IJCV, pp. 211–252, 2015.
49. I. Loshchilov and F. Hutter, “Decoupled weight decay
regularization,” in ICLR, 2019.
50. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Bar-
riuso, and A. Torralba, “Semantic understanding of
scenes through the ADE20K dataset,” IJCV, pp. 302–
321, 2019.
51. A. Kirillov, R. B. Girshick, K. He, and P. Doll´ar, “Panop-
tic feature pyramid networks,” in CVPR, 2019, pp. 6399–
6408.


---

16
Z. Pan, P. Chen, H. He, J. Liu, J. Cai, B. Zhuang
52. X. Chu, Z. Tian, Y. Wang, B. Zhang, H. Ren, X. Wei,
H. Xia, and C. Shen, “Twins: Revisiting the design of spa-
tial attention in vision transformers,” in NeurIPS 2021,
2021.
53. S. Ioﬀe and C. Szegedy, “Batch normalization: Accelerat-
ing deep network training by reducing internal covariate
shift,” in ICML, ser. JMLR Workshop and Conference
Proceedings, vol. 37.
JMLR.org, 2015, pp. 448–456.
