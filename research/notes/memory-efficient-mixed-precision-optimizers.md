---
title: MEMORY EFFICIENT MIXED PRECISION OPTIMIZERS
id: memory-efficient-mixed-precision-optimizers
tags:
- cmp170hx-training-feasibility-a0d4fc
- optimizer-memory
- fp32-master-copy
- gradient-accumulation-incompatible
- mixed-precision
created: '2026-10-27T11:24:11.199372Z'
updated: '2026-10-27T19:00:31.474491Z'
source: https://arxiv.org/abs/2309.12381
source_domain: arxiv.org
fetched_at: '2026-10-27T11:24:11.198720Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'Lewandowski and Kosson (EPFL, arXiv 2309.12381, Sep 2023) - mixed-precision
  optimizer that removes the fp32 master copy of the weights by storing only the residual
  extra bits between the bf16/fp16 value and its fp32 original, and fuses the optimizer
  step into the backward pass so gradients never persist in memory. Claims up to 25%
  lower peak memory and 15% faster training at equal accuracy; measured 54% lower
  peak memory on a synthetic 2B-parameter SGD run (against a stated theoretical maximum
  of 60%), 11% lower peak memory when fine-tuning T5 on GLUE with no meaningful slowdown,
  16% faster on ResNet-18, and 20-25% reduction in realistic training. States the
  baseline budget explicitly: standard AMP keeps an fp32 weight plus fp16 copy (6
  bytes), an fp32 gradient (4 bytes) and Adam state (8 bytes) per parameter, and notes
  Adafactor and 8-bit Adam cut state to 4 and 2 bytes. Two caveats that matter for
  this run: round-to-zero on the stored significand makes the compute-side bf16 value
  less accurate than round-to-nearest mixed precision, and fusing backward with the
  optimizer step makes gradient accumulation impossible - which rules the technique
  out for any batch plan built on accumulation to reach a 1M-token global batch.'
raw_file: raw/memory-efficient-mixed-precision-optimizers.pdf
doi: arXiv:2309.12381
utility_score: 12.0
---

MEMORY EFFICIENT MIXED PRECISION OPTIMIZERS
Basile Lewandowski∗, Atli Kosson
Machine Learning Optimization laboratory
´Ecole Polytechnique F´ed´erale de Lausanne
firstname.lastname@epfl.ch
ABSTRACT
Traditional optimization methods rely on the use of single-precision floating point
arithmetic, which can be costly in terms of memory size and computing power.
However, mixed precision optimization techniques leverage the use of both sin-
gle and half-precision floating point arithmetic to reduce memory requirements
while maintaining model accuracy. We provide here an algorithm to further re-
duce memory usage during the training of a model by getting rid of the floating
point copy of the parameters, virtually keeping only half-precision numbers. We
also explore the benefits of getting rid of the gradient’s value by executing the
optimizer step during the back-propagation. In practice, we achieve up to 25%
lower peak memory use and 15% faster training while maintaining the same level
of accuracy.
1
INTRODUCTION
The global trend in machine learning networks is that larger models yield more accurate results.
Consequently, new models have been designed with an ever-increasing number of parameters. How-
ever, training such models can be consuming in term of computing power, which is why recent work
has been aimed towards alternatives to the single-precision arithmetic (fp32). Most recent models
are so large they cannot fit on a single GPU using a traditional training framework: LLaMA (Meta
AI’s latest large language model) would require for instance 560 GB of memory, which is far more
than state of the art GPU can offer.
As neural networks grow larger and larger, the need to reduce their memory footprint has become
increasingly imperative. While prior research has focused on increasing speed, there remain room
for improvements in the way GPU memory is used. Indeed, the common approach to maintaining
some accuracy on the parameters while training on a half precision (fp16) model is to keep a master
copy of them in full floating-point precision. The drawback of doing so is that every parameter now
has to be saved in memory in both fp16 and fp32, further expending the charge for GPU memory.
Typical mixed precision training uses half precision values during the forward pass and single pre-
cision for the parameters’ update. For each parameter, the components stored on GPU memory are
then: the single precision value of the model weight and its half precision copy (6 bytes), the opti-
mizer state (dependent on the optimizer), the gradient (usually 4 bytes), and some additional values
like the forward activations whose size may vary depending on the model. To lower the memory
pressure, solutions have already been developed towards smaller optimizer footprint. Indeed, mem-
ory requirements for modern model training are often dictated by the optimizer state, with up to
twice as much memory required for each parameter. Alternative optimizers, such as Adafactor or
8bit-Adam, already offer a remedy to this problem by changing the way state memory is stored.
Where a standard Adam optimizer would require 8 bytes of state memory by parameter, these opti-
mizers respectively uses 4 bytes and 2 bytes only. The model’s parameters and their gradients then
becomes the main challenge to enhance memory use.
∗Corresponding author. Work carried out as a Master’s thesis from Sorbonne University’s computer science
degree, supervised by Atli Kosson
1
arXiv:2309.12381v1  [cs.LG]  21 Sep 2023


---

Our work aims at removing the additional memory cost incurred by the fp32 copy of the parameter
by keeping in memory only the difference between the original parameter and its fp16 value. We
also get rid of the gradient value by directly applying the update during the backward pass, thus
relieving the need to keep those values at all time. For a given parameter, this leads to - at least - 6
bytes less having to remain stored on the memory.
Our method does not necessitate any alterations to the hyperparameters or the training framework.
It is designed to fit models that require an extensive amount of memory for their training, such as
Large Language Models, Image Classification or Generation, Natural Language Processing, Object
Detection.
1.1
FLOATING POINT FORMAT BASICS
We remind here the basic principles of floating point arithmetic to frame our work thereafter. The
representation of common floating point numbers is ruled by the IEEE-754 Standard 461 (2008).
According to it, a binary floating point number x is represented in memory by :
• its sign sx (0 if positive, 1 if negative)
• its exponent ex, an integer representing the order of magnitude of x
• its mantissa (significand) mx, representing its decimal values w.r.t. ex. The first non-zero
bit is considered implicit and is not actually stored in memory.
such that x = (−1)sx ∗mx ∗2ex
This representation is somewhat analog to the standard scientific notation (for instance π = +3.14∗
100). The encoding used for ex and mx depends on which format is used, as described in Table 1.
The exponent value is represented with a bias so that numbers can be ordered lexicographically.
Some exponent values are reserved for special numbers : emax (maximum value) for infinities and
NaN, zero for subnormal numbers (that do not have an implicit bit), including zero itself. Some
compilers (eg. fast-math option) or non-standard formats (like bfloat16) do not use the subnormal
representation.
2
RELATED WORK
The standard framework for mixed precision training on neural networks is the one described by
Micikevicius et al. (2018). Especially, this work describes the use of a full precision copy for each
parameter to prevent the training from failing because weight updates are too small compared to
weight values. It also proposes the use of loss scaling and full precision arithmetic in some cases
where it is necessary to prevent the gradient from instabilities.
Alternatives solutions have considered the use of smaller floating point format (16bits and less),
with different ways of maintaining some level of accuracy. One common method to ensure the
values used remain accurate enough to allow for an efficient training is quantization : splitting the
data in several chunks and representing each number with only a few bits to describe their value on
a scale from the minimum of the chunk to its maximum. This allow for significant decrease in the
memory footprint of the optimizer states (Dettmers et al. (2022)) or for the gradients in a context of
distributed training (Alistarh et al. (2017)). Quantization has also been extensively considered for
inference-side computations (Gholami et al. (2021)), but the scope of our work is the improvement
of the training phase.
Table 1: Classical floating point formats
Name
Length
Sign
Exponent length
Significand length
Float
32 bits
1 bit
8 bits
1 + 23 bits
BFloat16
16 bits
1 bit
8 bits
1 + 7 bits
Half
16 bits
1 bit
5 bits
1 + 10 bits
2


---

The use of Brain floating point format (bf16) is often preferred for 16bits only training, as it is more
robust against gradient absorption in the updates and does not always necessitate some change in
the hyper-parameters (Zamirai et al. (2020); Kalamkar et al. (2019)). 8bits floating point format can
lead to promising results, and it is yet to determine which format is the best in a context of machine
learning training (Micikevicius et al. (2022); Wang et al. (2018)). More extreme solutions have even
considered the use of 1-bit binary networks (Qin et al. (2020)).
Another method proposed to train models on lower precision is the use of fixed point arithmetic.
This solution represents floating values with a fixed number of fractional and integer bits. In this
representation, a fixed number of bits are assigned to both parts of a decimal number. The position
of the binary point determines the scaling and precision of the fixed-point value. This system may
enable some interesting performance improvements but it is usually hard to maintain a sufficient
accuracy.
3
IMPLEMENTATION
We have developed two methods to reduce the need for memory in the context of mixed precision
training. The first approach aims at reducing the memory weight of the parameters by removing the
full precision copy stored to maintain accuracy during the training. The second one is to get rid of
the gradient values as soon as they are computed in order to avoid having all of the gradients stored
on the GPU memory.
3.1
16BITS ONLY MIXED-PRECISION
As stated earlier, a classic implementation of mixed precision optimization typically stores in mem-
ory both a fp16 and a fp32 value, whereas our approach consists of storing only the difference
between the two formats, thus resulting of at least a third less memory dedicated to the storage of a
model’s parameter. The stored part of the parameter is then used during its update in the backward
pass.
Classic 16bits floating point format (fp16) contains 10 bits of significand whereas alternative Brain-
floating point (bf16) contains only 7bits. There is therefore respectively 13 and 16 bits of precision
to keep if we want to maintain a full fp32 accuracy. We also explore the performance when keeping
only part of those bits.
To do so, we have developed an overload of the arithmetic operators used in the parameters’ update
(elementwise add, multiply, divide and their classic combinations) that performs the operation in
full precision using the extra bits saved separately and outputs both the updated 16 bits float and its
extra bits. We use for that a custom CUDA kernel where each thread handles the values stored in
the same memory slot. Since there is no efficient way to access arbitrarily sized bit strings in the
memory, the accesses are made through chunks of 32 bits (int32). Each GPU thread then handles
the operation on several values, depending on how many different extra-stored values can fit on 32
bits.
To ensure using all of the memory when the bit-sizes are not multiple of 32, we operate the threads
by groups corresponding to that bit-size, for instance values with 12 extra bits would require 12
threads working on 32 values stored in 12 int32 (see fig. 1). The overlapping extra-bits (stored on
two different int32) are modified using shared-memory to avoid data-race.
We provide an implementation for classic optimizers (Adam and SGD) to use transparently on a
fp16 or bf16 model. Extra bits are stored by the optimizer and do not require any modification in the
training framework to perform the weight updates. The extra precision provided ensures that small
gradients are accurately represented, reducing the risk of gradient underflow and enabling successful
model training.
This solution may incur some loss in accuracy on the 16bits floating point value. Indeed, saving only
the first part of the 32bit significand is equivalent to applying a round-to-zero operation on the full
precision value, which is known to be less accurate than the standard round-to-nearest used in this
case. Even though the value stored is as precise as a classic fp32 training, the 16bits value used in
the computations is then less accurate than a traditional mixed-precision framework. To overcome
3


---

 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32 * 12 bits
12 * 32 bits
Figure 1: Storage example for 12 extra-bits: the extra storage is created and accessed using twelve
32-bits integers, the bits stored to keep some accuracy (12 bits per parameter) are distributed among
32 slots. With this solution, we ensure that extra-storage fits closely to the size of the parameter and
that it is accessed efficiently through chunks of 32bits.
this issue, we have developed the option to use stochastic rounding when splitting the full precision
value. To do so, we store one additional extra-bit to keep in memory whether or not the value was
changed when rounded-up, so as to ‘un-round‘ the value before updating it.
Finally, we provide a Fused-version of the optimizers, since too many kernel launches can hinder
performance on smaller parameters. The principle of fused optimizers is to reduce the amount of
CUDA kernels launched during the optimization by handling the parameters as one only stream
of values combined all together. That way, one can launch kernels on a larger number of values,
so that we ensure the time needed for a kernel launch is amply covered by the time taken by the
computations. However, since fused-optimizers access values independently of which parameter
they belong to, they are for now only supported for 8 and 16 extra-bits.
3.2
FUSING BACKWARD PASS AND OPTIMIZER STEP
The classical implementation of the parameters update in a neural network is to compute the gradient
for each parameter and then to use this gradient as input for an optimizer algorithm. The drawback
of this system is that every gradient of every parameter has to be stored in the memory at all time
(or at least as long as the optimizer has not stepped). This generates a consequent need for GPU
memory as it nearly doubles the size of the parameters. Our solution to get rid of this pressure on
the memory is to operate the optimization step as soon as the gradient is computed.
In practice, we use PyTorch (Paszke et al. (2019)) automatic differentiation package to change the
way gradients are computed during the backward pass, which enables us to update directly the
parameters without saving any of their values. The optimization step is then performed by our
backward pass function, which means every operations on the gradient (eg. clipping or scaling) has
to be done through the optimizer. To accommodate every optimizer, our design requires that the
optimizer’s step function is called for each parameters, which is unusual for a training framework.
Our experiments shows that it is not a problem when using classic optimizers. This solution prohibits
any operation that would require every gradients of the model at the same time, but it is – to our
knowledge – very uncommon. The case of gradient accumulation is discussed in subsection 4.1.
4
EXPERIMENTAL RESULTS
To validate the ability of our setup to maintain training performances comparable to a fp32 setup
while using less memory, we carried out several model training from scratch. We consider the
following models : the Deep Learning Recommendation Model (DLRM, Naumov et al. (2019)), the
Resnet-18 image classification model, T5 text-to-text transformer, DCGAN image generation.
Memory Savings
Our solution enables for a smaller memory use in every phase of the training,
and especially during the backward pass. The results displayed in Figure 2 show we achieve up to
54% lower peak memory use on a sample network compared to a standard mixed-precision training.
This is close to the theoretical maximum loss of 60% in this case. Fusing the back-propagation
and the optimization provides on its own a 11% decrease on peak memory while fine-tuning a T5
model on the GLUE benchmark, without significantly slowing down the training (detailed results
are presented in Table 2). Overall, we achieve a reduction of 20 to 25% in real training conditions, as
4


---

Task
Fused backward pass
(enabled ✓or not ✘)
Peak memory
usage
Time to complete
MNLI
✘
29 930 MB
10:41:28
✓
26 818 MB
11:07:34
QNLI
✘
30 239 MB
03:11:09
✓
26 826 MB
03:16:48
MRPC
✘
29 964 MB
06:23
✓
26 649 MB
06:40
Table 2: Finetuning Performance on the GLUE Benchmark. We trained a large flan-T5 model on
several tasks of the GLUE benchmark, comparing its performance when the backward pass is fused
or not. Results show that peak memory usage is down by 11%, while computation time increased
only slightly.
Training framework
Execution Time
Accuracy (top 1)
Global GPU Memory usage
fp32
42 min
94.64 %
4 720 MB
amp fp16 + fp32
24 min
94.04 %
4 520 MB
fp16+13 rstoc
38 min
94.10 %
3 868 MB
fp16+8
20 min
94.06 %
3 776 MB
fp16
23 min
93.79 %
3 840 MB
fp8+8
–
93.96 %
–
fp8+4
–
65.53 %
–
fp8
–
8.78 %
–
Table 3: Training performance on Resnet18 with CIFAR10. Our solution outperforms classic mixed-
precision training both in terms of speed and memory use. This context shows too little difference in
accuracy between the different formats to draw any conclusion. Training was performed on a Nvidia
V100-SXM GPU.
This hardware does not natively support fp8, therefore we emulated 8bits precision on fp16 values
(e5m2 fp8 is equivalent to a truncated fp16), which is why execution time and memory usage is not
relevant in that case. Moreover, fp8-only training could yield better results if it were part of a more
adapted framework, it is displayed here only for comparison purposes.
it is the case with our ResNet18 experiment. We also observe that keeping only part of the mantissa
(for instance 8bits on a fp16 training) can be enough to maintain the accuracy while further reducing
the memory pressure (see fig. 4 for instance).
Accuracy
Our experiments show our solution achieve the same accuracy levels as mixed preci-
sion. In most of the models we tested, accuracy obtained with 16bits formats is close to the results
of full precision training. However, when 16bits training tends to diverge, our system prevent any
massive loss of accuracy. On the recommendation model we trained, we match the accuracy of the
full precision training by adding only 8bits of precision. As can be seen in fig. 4, this allows us
to run fp16 training where it would diverge in a standard context. In the case of ResNet18, our
experiments showed no relevant difference in accuracy between the different format, which led us
to experiment with 8bits floating point format (fp8). The results summarized in Table 3 indicates
our mixed-precision framework could also provide a fp16 accuracy on a 8bits model. This format
however lacks compatible hardware for now and is not yet supported by our solution. On the DC-
GAN model, our optimizer produces significantly better results than a standard bf16 training, but it
does not exactly match the results of full precision training. This is hard to evaluate performance
difference in this case because the results are better or worse than the fp32 training depending on
which metric we use.
Efficiency
The performance of our optimization techniques varies with the number of extra-bits
we use. The absolute throughput of our operators shows that they are as efficient as torch’s standard
ones, and somewhat weaker for extra-bits sizes that rely heavily on shared memory (see Figure 3).
On actual model training, our solution achieve up to 16% faster training on the Resnet-18 model, as
compared to classic mixed-precision training.
5


---

Figure 2: Memory Usage During A Synthetic Training. Left graph shows memory usage when
using torch’s standard mixed-precision framework, while the right one uses ours. Training consists
of three steps on a dummy 2 billion parameters model using SGD with momentum optimizer. Our
solution uses fp16 for the models parameters, the activation values and the gradients, and fp32 for
the optimizer state. The extra-bits are considered as part of the optimizer state memory footprint.
The profiler also considers the automatic mixed precision fp16 copy of the model’s parameters as
activation values, which is why they appear larger in the first figure. In total, peak memory usage is
down by 54% and could be further reduced by using an optimizer with reduced state memory.
Concerning fused optimizer, we notice a small improvement (2% to 10% faster training) when train-
ing a fused optimizer as compared to our standard optimizer. The difference appears to be smaller
than between classic optimizers, we can suppose that is because the extra-bits values are here ac-
cessed separately, whereas they form chunks of 32bits in our base optimizer. The results displayed
in Figure 5 show that our solution does not perform better than fused-fp32 when using 8 extra-bits,
however in this case bf16 training is slower than full precision training (likely because the input data
is in fp32 format).
Stochastic Rounding
Stochastic rounding provides a better accuracy as compared to classic 16bits
formats or basic extra-bit format (see Figure 6). However, stochastic rounding is less efficient due
to the additional operations required to round and ‘unround‘ the values, thus leading to performance
closer to fp32 training. Moreover, the gain in accuracy compared to the standard formats is preemi-
nent when few operations have been done, which is where the error due to the data is maximal.
6


---

55%
60%
65%
70%
75%
80%
85%
90%
95%
fp32
fp16
amp fp16+8 fp16+13
Figure 3: Throughput for an arithmetic operation on tensors. Data is shown as percentage of the
theoretical GPU bandwidth. Experiments were run on a NVIDIA A100-40GB-SXM GPU
0,76
0,77
0,78
0,79
0,80
0,81
0 k
50 k
100 k
150 k
200 k
250 k
300 k
ROC- AUC
Iterations
fp16
fp32
fp16+8
Figure 4: Accuracy improvement during training w.r.t classical fp16 framework on a DLRM model.
In this case, 8 extra-bits of storage are sufficient to achieve the same accuracy as a standard single-
precision setup. While a vanilla fp16 model leads to losing part of the model’s accuracy, our opti-
mizer enables us to avoid this loss.
00:50:00
01:00:00
01:10:00
01:20:00
01:30:00
01:40:00
fused
std
fused
std
fused
std
fused
std
fp32
bf16+16
bf16+8
bf16
Time to complete
60
80
100
120
140
160
180
200
2,3
2,4
2,5
2,6
2,7
2,8
2,9
3
fp32
bf16+16
bf16+8
bf16
Fr´echet Inception distance
Inception Score
IS
FID
Figure 5: Impact of the fused optimizer on training performance of a DCGAN model. We trained
a DCGAN model on the lsun-bedroom dataset, first graph shows the time nedded to complete the
training in the different configurations, second graph shows the final accuracy, measured with the
inception score and Fr´echet inception distance.
7


---

1,00E-05
1,00E-04
1,00E-03
1,00E-02
1,00E-01
1,00E+00
1,00E+01
1
10
100
1000
10000
100000
Forward Error
n
fp16+8
fp16+8 rstoc
fp16
nε
1,00E-04
1,00E-03
1,00E-02
1,00E-01
1,00E+00
1
10
100
1000
10000
Relative Forward Error
Condition Number
εcond
ε + ucond
Figure 6: Forward Error On 16bits Tensor Addition. We measure the round-off error of our different
half float formats on operations with random values following the standard normal distribution. First
graph shows the error after only one operation (in-place add) given some tensor size n. Second graph
shows the error after several operations, given their condition number. We denote ε the fp16 round-
off precision and u the precision when using 8 extra-bits. We observe on the first operation that
basic fp16+8 behaves somewhat poorer than classical fp16, whereas both extra-bit formats produce
significantly better results on cumulative operations.
4.1
LIMITATIONS
Gradient Accumulation
As stated earlier, fusing the backward pass and the optimizer step pre-
vents the use of the gradients after the backward pass. The use of gradient accumulation (stepping
the optimizer after several back-propagation) therefore becomes impossible. However, the main
point of gradient accumulation is to reduce the need on GPU memory by splitting the batches into
smaller mini-batches, and our solution allows for larger batches by getting rid of the gradient, thus
reducing the need for such a mechanism.
Gradient Clipping
Gradient clipping is a method designed to prevent exploding gradients by
setting a threshold on their values. It is usually applied between the backward pass and the optimizer
step, it then becomes impossible when both are fused. One simple solution to mitigate this issue is
to perform the clipping through a parameter hook, for instance:
for p in model.parameters():
p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
The same applies to loss scaling.
Closure
Some training frameworks make use of a ”closure” function, which is called during the
optimizer step. It is for instance the case with the L-BFGS optimizer, that uses several call to the
loss function during the step process. In this case, fusing the backward pass and the stepping may
have unexpected consequences on the optimizer behavior, since gradients are not available where
they should be. Although such frameworks are – to our knowledge – rather uncommon, we warn
that stepping the optimizer after each gradient computation might overall trigger issues with some
training configuration that we are not aware of.
5
CONCLUSION
Mixed precision training is an efficient technique to accelerate neural networks training without
massive loss in accuracy. We have shown that its memory consumption can be further reduced by
tweaking the optimization process. In many cases, the full precision copy can be dispensed with
only part of it. Moreover, fusing the backward-pass and the optimizer step reduces peak memory
usage by removing the pressure issued by gradients.
Further work could be focused on extending these mechanisms with smaller floating point arith-
metic, like 8bits models. We could also investigate their integration in parallel and distributed train-
ing framework, as well as less in common optimization setup.
8


---

REFERENCES
Ieee standard for floating-point arithmetic. IEEE Std 754-2008, pp. 6–14, 2008. doi: 10.1109/
IEEESTD.2008.4610935.
Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic.
Qsgd:
Communication-efficient sgd via gradient quantization and encoding, 2017.
Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise
quantization. 9th International Conference on Learning Representations, ICLR, 2022.
Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. A
survey of quantization methods for efficient neural network inference, 2021.
Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee,
Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen,
Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan,
Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A Study of BFLOAT16
for Deep Learning Training, June 2019.
URL http://arxiv.org/abs/1905.12322.
arXiv:1905.12322 [cs, stat].
Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Gar-
cia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu.
Mixed Precision Training, February 2018. URL http://arxiv.org/abs/1710.03740.
arXiv:1710.03740 [cs, stat].
Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisen-
thwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi,
Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. FP8 Formats for Deep Learn-
ing, September 2022. URL http://arxiv.org/abs/2209.05433. arXiv:2209.05433
[cs].
Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundara-
man, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro
Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, An-
sha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill
Jia, Liang Xiong, and Misha Smelyanskiy. Deep learning recommendation model for personal-
ization and recommendation systems, 2019.
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Ed-
ward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance
Deep Learning Library, December 2019. URL http://arxiv.org/abs/1912.01703.
arXiv:1912.01703 [cs, stat].
Haotong Qin, Ruihao Gong, Xianglong Liu, Xiao Bai, Jingkuan Song, and Nicu Sebe. Binary
neural networks: A survey. Pattern Recognition, 105:107281, 02 2020. doi: 10.1016/j.patcog.
2020.107281.
Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Train-
ing Deep Neural Networks with 8-bit Floating Point Numbers, December 2018. URL http:
//arxiv.org/abs/1812.08011. arXiv:1812.08011 [cs, stat].
Pedram Zamirai, Jian Zhang, Christopher R. Aberger, and Christopher De Sa. Revisiting bfloat16
training. CoRR, abs/2010.06192, 2020. URL https://arxiv.org/abs/2010.06192.
9


---

A
APPENDIX
In this section, we detail the frameworks used in our experiments. All experiments were run using
Python 3.9, CUDA 11.8 and Pytorch 2.0.1 or above
ResNet18 & CIFAR10 1
We trained a ResNet18 model on the CIFAR10 dataset, results are listed
in 3. We did not make use of loss scaling for the experiments as it has not proved necessary. We also
kept all modules, including Batch Norm, in fp16 for the same reason. Further hyper parameters are
listed below.
Batch size:
256
Epochs:
200
Optimizer:
SGD with momentum
Learning rate:
0.3
Warm-up learning rate:
1e-6
Minimum learning rate:
1e-5
Weight Decay:
2e-4
Scheduler:
cosine (one cycle)
Seed:
0xB0B
Warm-up epochs:
5
GLUE2
We fine-tuned the bert-base model using the GLUE benchmark. The original weights are
the ones provided with standard huggingface transformers package.
Batch size:
16
Epochs:
3
Optimizer:
AdamW
Learning rate:
2e-5
Maximum sequence length:
128
Seed:
0xB0B
DLRM3
We used data from the Kaggle-Criteo dataset 4. For the purposes of the demonstration, we
did not use any scaling on the gradients during the fp16 trainings. Performance is evalued through
the Area Under the Receiver Operating Characteristic Curve (ROC-AUC)5 on a binary case.
Sparse feature size:
16
Bottom MLP:
13-512-256-64-16
Top MLP:
512-256-1
Optimizer:
SGD
Batchsize:
128
Training epochs:
1
Learning rate:
0.1
Loss function:
BCE
Seed:
2023
1https://github.com/huggingface/pytorch-image-models
2https://github.com/huggingface/transformers/tree/main/examples/
pytorch/text-classification
3https://github.com/facebookresearch/dlrm
4https://ailab.criteo.com/ressources/
5www.ncbi.nlm.nih.gov/pubmed/2668680
10


---

DCGAN6
We used data from the LSUN dataset 7. We used bf16 in order to avoid needing loss-
scaling. We used the package pytorch-gan-metrics8 to evaluate our model. Evaluation was com-
pleted on a set of 50 000 samples.
Loss function:
BCE with logistic
Optimizer:
Adam
Learning rate:
7.10−5
β1 (first order momentum):
0.64
Batchsize:
128
Image size:
64
Latent vector length (nz):
100
Training epochs:
10
Seed:
0xC0FFEE
6https://github.com/NVIDIA/apex/tree/master/examples/dcgan
7www.yf.io/p/lsun
8https://github.com/w86763777/pytorch-gan-metrics
11
