---
title: Behind the Scenes of Distributed Training and Why Your GPU Wiring Matters as
  Much as Your Strategy | Towards Data Science
id: behind-the-scenes-of-distributed-training-and-why-your-gpu-wiring-matters-as-muc
tags:
- cmp170hx-training-feasibility-a0d4fc
- ddp
- fsdp
- zero
- pcie-fabric
- nvlink-topology
- measured-benchmark
created: '2026-10-27T11:37:38.331924Z'
updated: '2026-10-27T19:00:31.547358Z'
source: https://towardsdatascience.com/behind-the-scenes-of-distributed-training-why-your-gpu-wiring-matters-as-much-as-your-strategy/
source_domain: towardsdatascience.com
fetched_at: '2026-10-27T11:37:38.330703Z'
fetch_provider: builtin
status: review
type: note
tier: practitioner
content_type: article
deprecated: false
summary: 'Measured practitioner study (Towards Data Science, 2026) of how the GPU
  fabric, not the strategy, sets distributed-training throughput. Part 1 (four A100
  80GB, HuggingFace Accelerate, BF16): Mistral-7B with Adam needs 87 GB of model state
  per GPU (14 GB bf16 params + 14 GB grads + ~58 GB fp32 Adam momentum/variance) before
  any activation, so DDP OOMs on an 80 GB card while FSDP shards it to ~22 GB estimated
  and ~29 GB actual peak. Measured peak memory for the same 7B job on 4 GPUs across
  the dial: DDP 87 GB, ZeRO-1 55, ZeRO-2 51, ZeRO-3 40, FSDP 37 — the floor is activations,
  not model state, so sharding never reaches 87/4. On a fast wire ZeRO-1/2 give up
  ~20% of DDP throughput and FSDP claws nearly all of it back at the same memory;
  on a slow wire the whole dial flattens toward the same low throughput. Part 2 (H200
  machines, same GPUs wired two ways): NVLink vs forced PCIe on identical silicon
  is worth ~10-11x on the same collectives; all-reduce bandwidth is 330 GB/s at 2
  GPUs rising to 467 at 8 on NVSwitch, 114 GB/s at 2 GPUs and 322 GB/s at 4 inside
  an NVLink-bridged quad (NV6 links per pair), but only 28/35/39 GB/s at 2/4/8 GPUs
  when a job must cross between quads over PCIe — the same four cards run 322 GB/s
  inside a quad vs ~35 GB/s across two, a ~9x difference decided purely by placement.
  End-to-end, the cross-quad job loses 3-5x throughput, and the loss is not uniform:
  FSDP falls hardest (it communicates every layer, and per-layer transfers cannot
  hide behind compute on a slow wire) while DDP holds up best (one all-reduce per
  step). The operational rule: run nvidia-smi topo -m, and on a slow wire use plain
  DDP if the model fits replicated; if it does not, pick the least-sharding stage
  that fits and accept the wire''s price. Caveat for this run: the slow wire here
  is PCIe Gen5 x16 across NVLink quads (~35 GB/s all-reduce), still ~45x faster than
  the CMP 170HX box''s measured ~0.75 GiB/s, so the direction transfers but the magnitudes
  do not.'
utility_score: 11.0
---

Behind the Scenes of Distributed Training and Why Your GPU Wiring Matters as Much as Your Strategy | Towards Data Science
Training a model on one GPU is straightforward. You load the weights, load the data, and wait for it to finish. For most models, that’s fine, and the only real cost is time. When the wait gets too long, the usual fix is to add GPUs. Each one trains on a different slice of the data in parallel, and the work finishes faster. Nothing about the model state changes; you’ve just thrown more hands at it.
That covers the common case, where the problem is time. Large models add a second problem that more hands can’t solve: space. Over the past year, a major part of my work has involved training and fine-tuning large foundation models across different domains, including single-cell models and vision-and-language transformer models. Once you get past a few billion parameters, the model, its gradients, and its optimizer states no longer fit on a single GPU. You can’t just replicate the work across GPUs because a copy doesn’t fit on any of them. You have to split the model itself.
These two problems map onto two strategies. DDP (Distributed Data Parallel) attacks time. Every GPU keeps a full copy of the model and trains on a different slice of the batch. FSDP (Fully Sharded Data Parallel) attacks space. The model is split into pieces so that no GPU has to hold the entire model.
There aren’t really just two options, though. These are just the two ends of one dial. In between these sit the ZeRO stages from DeepSpeed (Microsoft's training library), which let you slide from one end to the other, step by step, trading a little memory for a little communication at each stop rather than jumping straight from “everybody holds everything” to “nobody does.”
Where to set that dial and what it actually controls is covered in the first part of the article. But there’s a second thing that decides how well any of these strategies work. Each one of them works by moving data between GPUs. And how fast that data moves depends on the physical wires connecting them. This was something I didn’t think about until the same code, on the same kind of GPUs, ran several times slower on one node than on another. So in the second half, we’ll discuss the hardware underneath it, and how the wiring should also be something you consider. (For folks who are not much into hardware, trust me, it’s not that bad!)
One note before we start. Everything here is measured, not just theory. The strategy comparisons ran on four A100 80GB GPUs; the hardware experiments ran on H200 machines, including two that have the exact same GPUs wired together in different ways, which turns out to be the whole point.
Part 1: The Software: What Each GPU Holds
Every distributed training strategy is really an answer to one question: what does each GPU keep its own copy of? For that to make sense, let’s look at an example.
Take Mistral-7B and fine-tune it with Adam in BF16 mixed precision. The parameters take 14 GB: 7 billion of them, two bytes each. The gradients take another 14 GB. Then we have the optimizer states, which are the most expensive part. Adam keeps two running averages for every parameter, momentum and variance, both stored in FP32, which works out to roughly 58 GB. Add it up, and we reach 87 GB before the model has seen a single batch or a single activation has been computed.
For an A100 GPU that has 80 GB of VRAM, the model doesn’t fit. So we distribute. And the question, as mentioned at the beginning of this section, is: what does each GPU hold?
Distributed Data Parallel (DDP): Everyone Holds Everything
The simple answer is that every GPU holds all of it. This is DDP, and it’s the one most people first encounter (which was the case for me as well). Each GPU keeps a complete copy of the model: all the parameters, all the gradients, all the optimizer states. What gets divided is the data. Each GPU takes a different slice of the batch, runs its own forward and backward pass, and computes gradients from its own slice.
The catch is that those gradients are all different because each GPU saw different data. If every GPU just stepped its optimizer now, the copies would drift apart, and you'd no longer have one model, you'd have four. So before stepping, the GPUs average their gradients. Every GPU sends its gradients out, receives everyone else's, and they all end up with the same averaged gradient. They then take an identical optimizer step and the copies stay in sync. That averaging is the one piece of communication DDP needs, and it happens once per step. It is called
all-reduce
, and we’ll come back to it later when we start counting what each strategy costs the wire.
Figure 1
: How DDP works (Image by author)
DDP is fast and simple precisely because it communicates so little. One all-reduce per step, and PyTorch overlaps even that with the backward pass, so on good hardware, you barely notice it. But its limitation is built into the design. Every GPU holds everything, which means adding GPUs does nothing for the memory problem. If 87 GB doesn't fit on one A100, it doesn't fit on four either, because each of the four is still holding the full 87 GB. Adding GPUs to DDP buys you throughput, but it doesn’t reduce the memory each GPU has to hold.
Fully Sharded Data Parallel (FSDP): Nobody Holds Everything
If the problem is that every GPU holds everything, the obvious fix is to make sure no GPU does. This is FSDP. Instead of each GPU keeping a full copy, the model is split into pieces, and each GPU owns just one piece of the parameters, one piece of the gradients, and one piece of the optimizer states. On four GPUs, each one holds roughly a quarter of the 87 GB, which allows the model to fit.
But this raises an immediate problem. You can't run a layer through a GPU that only has a quarter of that layer's weights. So FSDP reassembles each layer just before it's needed and takes it apart again right after. When the forward pass reaches a layer, the GPUs trade their pieces so that every GPU briefly has that layer's complete weights, the layer runs, and then each GPU throws away the pieces it doesn't own. The next layer comes up and the same thing happens again. The full model is never resident on any single GPU; only one layer is, and only for the instant it's computing. The same thing happens in reverse during the backward pass.
Figure 2
: How FSDP works (Image by author)
That trading has names, the same way DDP's averaging did. Gathering everyone's pieces into a full layer is an
all-gather
, and scattering the summed gradients back to their owners is a
reduce-scatter
. Keep in mind this contrast. DDP communicates once per step. FSDP communicates constantly. An all-gather before every layer in the forward pass, another before every layer in the backward pass, and a reduce-scatter after. That's basically the trade that you do when switching between them. FSDP buys memory and pays for it in communication.
One clarification matters here, because it could be kind of confusing. FSDP is still data parallelism. Every GPU runs the entire model on its own slice of the batch, exactly like DDP. The only difference is that the weights are stored in pieces and reassembled on demand instead of being kept as a full copy. Each GPU still computes the whole forward and backward pass from start to finish.
This is different from
model parallelism
, where the model itself is split across GPUs, and each one is responsible for only part of the computation, one set of layers, or one slice of every layer. In this paradigm, a single forward pass has to hop between GPUs because no single one can finish it alone. To add to this, even within model parallelism, we have
layer
and
tensor
parallelism
, which dictate whether we divide the model vertically or horizontally among the layers. But before we digress too much, these are different techniques with different communication patterns, and it’s not what this article is about. Everything here, DDP, FSDP, and the ZeRO stages in between, is data parallel. Same computation on every GPU, just with different data. Only the storage of the model differs from one strategy to the next.
Coming back to it, now that we’ve discussed the theory, let’s see what happens in practice. To see how these two strategies compare, I ran experiments on two models.
DINOv2
(a vision foundation model) and
Mistral-7B
(a large language model). Both experiments ran on four A100 80GB GPUs on a single node, using HuggingFace Accelerate with BF16 mixed precision. The only thing that changed between runs was the strategy.
DINOv2 on Food-101
. The first experiment fine-tunes DINOv2 on Food-101, a classification dataset with 101 food categories and about 101k images. The task is to predict an image category given an image. I tested both the base (86M parameters) and large (304M parameters) backbones, using a batch size of 32 per GPU for the base model and 256 per GPU for the large model. This is a case where both strategies work since the model fits comfortably in GPU memory. But even here, we can see the memory difference between the two approaches.
Figure 3
: Memory usage comparison between DDP and FSDP (Image by author)
FSDP uses significantly less memory per GPU: 0.39 GB vs. 1.79 GB for the base model, and 1.39 GB vs. 6.26 GB for the large model. That is roughly 4.5x less memory with FSDP. This matters less when you have plenty of headroom, but it shows us why FSDP will be essential for larger models. So we saw the memory, what about the speed?
Figure 4
: Throughput comparison of DDP vs. FSDP (Image by author)
We can see that DDP is consistently faster, around 6% higher throughput for the base model and 4% for the large one. This is expected since DDP has less communication overhead. It just has one all-reduce per step, against FSDP’s gather-and-discard at every layer. And when we see the training time:
Figure 5
: Training time comparison between DDP vs. FSDP (just 5 epochs for Base model and 2 epochs for Large) (Image by author)
The takeaway here is that when the model fits in memory, DDP is the better choice. It’s simpler and faster. But notice the memory gap. What happens when we scale up to a model where that gap actually matters?
Mistral-7B on Alpaca
. For the second part, I tried fine-tuning the Mistral model on the Alpaca instruction-following dataset. With DDP, the job crashed immediately with an out-of-memory error. With FSDP, it ran without an issue.
Figure 6
: Estimated and actual memory usage for DDP and FSDP (Image by author)
DDP would need an estimated 87 GB per GPU for the model state (parameters, gradients, optimizers), which exceeds the 80 GB available on the A100. FSDP shards this across 4 GPUs, bringing the estimated requirement down to about 22 GB per GPU. The actual peak during the run was about 29 GB, which includes activation and other overhead. This is why FSDP exists. It makes training possible when DDP simply can’t fit the model.
So that’s the rule I used to stop at. Does the model, with its gradients and optimizer states, fit on one GPU? If yes, use DDP. If no, use FSDP. It’s a good rule. But there are two assumptions hiding inside it. The first is that DDP and FSDP are the only two settings. In the beginning, I already hinted that they aren’t. There’s a dial between them, and in the next part, we’ll discuss these stages, where you shard some of the model state but not all of it.
The second assumption is that we’re treating all machines the same way. These strategies don’t behave the same everywhere. The same strategy on the same model can run several times faster or slower depending on how the GPUs are wired together. That’s what the second half of the article is about. First, though, the dial.
ZeRO: The Dial Between DDP and FSDP
To see the stops on the dial, start with a small fact about the operations from Part 1. DDP’s all-reduce, the one that averages gradients across GPUs, is actually two simpler operations glued together. First is reduce-scatter, which sums everyone’s gradients but gives each GPU only its slice of the result instead of giving everyone the full result. Then an all-gather, which collects those slices back into a full copy on every GPU. So all-reduce is basically reduce-scatter plus all-gather.
Those are the same two operations that FSDP uses. Now that we know that the operations are shared, it makes sense that the amount of sharding could be in stages. The dial is then just a question of how much you’re willing to keep sharded between steps instead of reassembling.
To make that concrete, look at what each GPU stores per parameter. In mixed-precision Adam, there are three things: the parameters, the gradients, and the optimizer states, with the optimizer states being by far the heaviest. DDP keeps full copies of all three on every GPU. The ZeRO stages, from DeepSpeed, peel them off one at a time, heaviest first.
ZeRO-1
shards only the optimizer states. Each GPU still holds the full parameters and full gradients, but it keeps only its slice of the optimizer states. Since those are the heaviest items, this gives the biggest memory win, and it costs very little in extra communication, because nothing you need during the forward and backward pass has moved.
ZeRO-2
adds the gradients to the shards. Now each GPU keeps only its slice of the gradients and its slice of the optimizer states, but still a full copy of the parameters. Communication is still close to DDP’s, since the parameters (the thing you actually compute with) are all still local.
ZeRO-3
adds the parameters as well. Now, no GPU holds the full model, so before a layer can run, its parameters have to be gathered on the fly, used, and discarded. ZeRO-3 and FSDP implement the same basic idea. One is built by DeepSpeed, and the other is native to PyTorch. So, since this is where you stop keeping the parameters local, you will also have the extra communication we discussed earlier.
The following diagram shows the dial from full replication to full sharding:
Figure 7
: The different stages of ZeRO and what they shard vs. replicate (Image by author)
Two things in the figure are worth pausing on. The first is the memory column on the right. Those are the measured peaks for Mistral-7B on four GPUs, and they fall in a clean pattern as you move down the dial. 87 GB at DDP, then 55, 51, 40, and 37 GB. But notice it never drops to a quarter of 87 GB, which is what you’d expect if four GPUs each held a quarter of everything. The floor is mostly activations and runtime overhead. Activations are still produced independently on each GPU, so sharding the model state does not make them disappear.
Second, every step down that dial buys memory by spending communication. ZeRO-1 spends almost none. ZeRO-3 and FSDP spend the most. That is the software view. DDP, ZeRO, and FSDP are different answers to the same question: what should each GPU keep locally, and what should it fetch from the others? Moving toward sharding buys memory, but it also creates more communication.
So far, we have only counted how much data has to move. We have not asked how expensive that movement is. That depends on the hardware. More specifically, it depends on the fabric connecting the GPUs.
Part 2: The Hardware: How Fast the GPUs Can Talk
The Wires Between the GPUs
Every strategy in Part 1 had the same hidden cost line: communication. DDP pays it once per step. FSDP pays it at every layer. ZeRO lets you choose how much to pay. But communication is not an abstract cost. It is data moving between chips, and the path it takes matters.
In this article, I’m only looking at communication inside a single node. Once training spans multiple servers, another layer appears. GPUs now have to communicate across InfiniBand or Ethernet, which brings a different set of bottlenecks. That matters a lot as well, but it’s a separate article.
Inside one node, when GPUs need to exchange data, how do they do it? Well, there are different roads they can take, and the speed between them can differ by up to 10×. The same FSDP code can run at full speed on one machine and crawl on another with the exact same GPUs, purely because of how those GPUs are connected to each other.
There are two basic paths GPU traffic can take inside a server: PCIe and NVLink. PCIe is the default system path through the motherboard. NVLink is NVIDIA’s direct GPU-to-GPU interconnect.
But NVLink is only the link. The topology matters just as much. On the machines I tested, NVLink appeared in two different arrangements. NVL bridge groups and NVSwitch. That gives us four terms to keep straight. PCIe, NVLink, NVL, and NVSwitch. They are related, but they are not the same kind of thing. PCIe and NVLink are the paths. NVL and NVSwitch are ways of arranging the fast NVLink path across multiple GPUs.
PCIe: the default connection
PCIe (PCI Express) is the standard high-speed bus that connects components to a computer’s motherboard, the slots your graphics card, SSD, and network all plug into. Every GPU sits on it by default.
When two GPUs need to exchange data over PCIe, it goes through the system: out of one GPU, across the shared bus, often through the CPU and host memory, and into the other. It works between any two devices, but it’s the slowest option, and every device on the bus is contending for the same lanes. A PCIe Gen5 x16 slot moves somewhere around  64 GB/s in one direction.
NVLink: a direct GPU-to-GPU connection
NVLink is NVIDIA’s dedicated interconnect. Links that run from one GPU to another without going through the CPU or system memory. On an H100 or H200, each GPU has 18 NVLink connections, which together come to roughly 450 GB/s in each direction, about 7x what a single PCIe slot provides.
But NVLink is only the link. What matters just as much is how those links are arranged across the GPUs in a server. Two machines can both have H200 GPUs and still behave very differently if one uses NVLink through bridges and the other uses NVSwitch.
NVL: links within groups, PCIe between them
The cheaper arrangement connects NVLink only within small groups of GPUs, using physical bridges between cards. NVIDIA sells these as “NVL” cards, like the H200 NVL. On the machines I tested, eight cards were bridged into two groups of four. Within a group, every pair of GPUs has a fast NVLink bridge. Between groups, there’s no NVLink at all, and traffic falls back to PCIe.
So the fabric is uneven. Some pairs of GPUs talk over NVLink, others talk over PCIe. And so which pair you get depends on which physical GPUs your job lands on, which is decided by the cluster scheduler. On an NVL machine, placement is important as it can affect performance.
NVSwitch: every GPU connected to every other
The premium arrangement puts a switch in the middle. Dedicated NVSwitch chips connect every GPU to every other GPU at full NVLink speed, with no fast pairs and no slow pairs. Every GPU is one hop away from every other. This is what’s on NVIDIA’s HGX baseboards, the SXM form factor, and it’s the layout that large training runs use.
To get a sense of how far this idea scales, at Computex 2025, Jensen Huang showed off what NVIDIA calls an NVLink Spine, a column of around 5000 cables wiring 72 GPUs into a single all-to-all fabric, and claimed it moves about 130 TB/s. This is more than the peak traffic of the entire internet. Take the comparison as you will, but the underlying point is real. An NVSwitch fabric is a crossbar where every GPU reaches every other at full speed, and it scales to a whole rack. The machine I tested is a much smaller version of the same idea. It has 8 GPUs instead of the 72, but wired on the same principle.
Same GPU, two different machines
It’s worth clearing up a confusion that can happen. When we say “H200,” we are mostly naming the GPU. H200 systems can be built in different form factors and fabrics: as SXM GPUs on an NVSwitch baseboard, or as NVL cards connected by bridges. Both are H200-class GPUs with 141 GB of memory, but they are not the same system. Power limits, board design, and cooling can differ. But the point here is that you cannot infer the GPU-to-GPU fabric from the GPU model name alone. You have to look at the topology. Here are the two machines used for the experiments below. You can check the wiring on your node with
`nvidia-smi topo -m`
.
Figure 8
: Topology comparison of an NVSwitch node vs just NVL (Image by author)
On the SXM node, every pair reads NV18. This is uniform, full speed, every GPU is equal. On the NVL node, the same H200 cards split into two groups of four, fast NVLink within a group and a plain PCIe across. A job that lands on four GPUs inside one group runs near full speed. A job scattered across groups drops down to PCIe. The next section is about exactly what that costs.
Measuring the Wires
We have two kinds of measurements here. The first is a microbenchmark that times the exact collective operations the strategies depend on. All-reduce for DDP, all-gather and reduce-scatter for FSDP, and reports the bandwidth each one achieves. The second is end-to-end training, the same Mistral-7B fine-tune from Part 1, now timed for steady-state throughput, so we can see whether the bandwidth differences actually show up in training. Everything below ran on the two machines described above.
Let’s start with the cleanest possible comparison. Take two H200 SXM GPUs and run the same collectives twice. For the first run, let them use NVLink, and for the second, force them onto PCIe instead. Same GPUs, same code, only the wire changes.
Figure 9
: Collective bandwidth over NVLink versus forced PCIe (Image by author)
The wire alone is worth about 10-11x. This is the same silicon and the same operation, where the only difference was the physical connection the data travels over.
Next, I measured how all-reduce bandwidth changes as you add GPUs to the job. For this, I had to control which physical GPUs each run used, so I could keep a job inside a single NVL group or force it to cross between groups.
Figure 10
: All-reduce bandwidth versus GPU count across three fabrics (Image by author)
We can see three behaviors here. NVSwitch stays flat and fast. 330 GB/s at two GPUs, rising gently to 467 at eight, and it never matters which physical GPUs the job gets. Every pair is NV18, so the fabric looks the same no matter how the scheduler fills it. This is the behavior large training runs assume.
The NVL quad is a surprise, and it comes down to how NVLink links are shared. Each of these GPUs has 18 NVLink links to distribute among its neighbors. Inside a quad of four, every GPU has three neighbors and gives six links to each one, which is the NV6 you’d see in the topology matrix. Six links per pair, eighteen links total per GPU.
That sharing is why job size matters so much. A job using only two GPUs of the quad talks across just the six links connecting that one pair. The other 12 links on each card go to GPUs that aren’t in the job, so they sit idle, and the pair runs at about 114 GB/s, which is a third of the card’s NVLink capacity. Add the other two GPUs, and every card now talks to all three of its neighbors at once, lighting up all 18 links. That’s the same number of links an SXM module uses, which is why a full quad reaches about 322 GB/s, nearly matching NVSwitch.
So the NVL quad was never slow. It was just underused. A small job leaves most of its links idle. When we fill the quad, we fill the links.
The third line is the cross quad case, and it never leaves PCIe territory. 28 GB/s at two GPUs, 35 at four, 39 at eight. Adding GPUs doesn't help because the problem isn't how many links are active; it's that the job has to cross between quads, and there's no NVLink there at all. A collective is only as fast as its slowest required hop, so a single PCIe link between the quads drags the entire all-reduce down to PCIe speed, no matter how fast every other link runs. And since a quad is only four GPUs wide, any job larger than four on this machine is forced to cross. Eight GPUs can't avoid it.
The clearest view is the 4-GPU point on the figure. The same four cards, in the same node, run at 322 GB/s inside one quad and about 35 GB/s across two, a roughly 9× difference decided by nothing but which GPUs the job landed on. On an NVL machine, placement really does make a difference.
What it does to real training
The bandwidth numbers prove the wire matters in a microbenchmark. I also wanted to see if it shows up in real training. So I ran the same Mistral-7B fine-tune from Part 1 on four GPUs, three ways: on NVSwitch, on a full NVL quad, and on an NVL job deliberately spanning two quads.
Figure 11
: Throughput comparison of end-to-end training on different wirings using different strategies (Image by author)
Look at the two fast fabrics first, NVSwitch and the full NVL quad. They’re nearly identical for every strategy. The cards were never the bottleneck, it was the placement. (If FSDP looks like it edges out DDP inside the quad, don’t read too much into it. Repeat runs put the noise at a few percent, so those are a tie. The honest summary is that, on either fast fabric, DDP and FSDP are even.)
The across-quads case is the cost of getting placement wrong. Throughput drops by 3 to 5x. That’s a smaller hit than ~10x bandwidth drop from figure 9, and the gap is worth understanding. Training isn’t pure communication. Each GPU is still doing real compute that doesn’t care about the wire, and that compute partly hides the slow communication behind it. So the wire’s full penalty gets somewhat diluted. But 3 to 5x is still a very meaningful difference, especially for long jobs.
And the slowdown isn't even across the strategies. The pattern is exactly what Part 1 predicts. FSDP falls the hardest, because it communicates every layer, and on a slow wire, those per-layer transfers can't hide behind compute. DDP holds up best because it communicates just once per step. When the wire is fast, sharding is nearly free, so FSDP keeps pace with DDP. When the wire is slow, every extra bit of communication is exposed, so the more a strategy shards, the more it loses.
Put it all together. The whole dial, on three wires
Earlier, we discussed that DDP and FSDP are two ends of one dial, with ZeRO-1, -2, and -3 as the stops between them. That was a diagram, and now I can measure it. Here's the same Mistral-7B job on four GPUs, run at every stop on the dial, on a fast wire and on slow ones.
Figure 12
: The full dial measured end-to-end. Throughput at each stage on three wires (left), and peak GPU memory at each stage (right) (Image by author)
Start with the memory panel on the right because it does exactly what the theory promised. Moving down the dial, peak memory falls in a clean pattern. 87 GB at DDP, then 55 at ZeRO-1, 51 at ZeRO-2, and 40 at ZeRO-3. Each stop shards one more thing, and each one buys headroom. FSDP sits at the bottom at 37 GB, but notice it's not a big step down from ZeRO-3's 40, because FSDP and ZeRO-3 shard exactly the same things. As I mentioned above, they shard the same things, but the frameworks implement and schedule things differently, which explains the small gap that we see. This staircase is the part that always works, regardless of hardware. If your problem is fitting the model, the dial delivers, predictably.
The throughput panel is interesting because the dial behaves like two different things depending on the wire.
On the fast wire, the middle of the dial loses. ZeRO-1 and ZeRO-2 give up around 20% of DDP's throughput, and ZeRO-3 gives up more. But FSDP, sharding the same things as ZeRO-3, claws nearly all of that throughput back while using about the same memory. That's interesting because the last two are implementations of the identical algorithm, and land far apart on a fast wire.
The difference is implementation, not method. The two frameworks schedule and overlap their communication differently, and on a fast wire, where communication is cheap, that overhead is the main thing you're measuring. (This is Hugging Face accelerate's default DeepSpeed config, with no tuning. A tuned setup would likely narrow the gap; the point is what the defaults give you, not that one framework is inherently slower.)
On a slow wire, the dial flattens. Everything collapses toward the same low throughput, and the differences between stages mostly wash out. The expensive thing on a slow wire is communication itself, and once that dominates, the question of which exact tensors you shard barely moves the number. In this case, the wire is setting the pace, not the strategy.
Putting everything above together, we can now make a more informed decision.
If you're on NVSwitch (SXM), stop thinking about the wire. Every GPU is equally fast, and any GPUs the scheduler gives you are as good as any other. Use DDP if the model fits, FSDP if it doesn't. And since FSDP matches DDP's speed here while using far less memory, it's a reasonable default even when the model does fit.
If you're on NVL and your job fits inside one bridged group, pin it there, and you get near-NVSwitch speed. On the machines I tested, that means four GPUs, and you control the placement with CUDA_VISIBLE_DEVICES. FSDP is a strong default in this case. It gets you almost DDP-class speed at roughly half the memory.
If you're on NVL and your job has to span groups, expect a considerable slowdown, worst for the communication-heavy strategies. Don't expect a milder ZeRO stage to win the speed back either, because the dial flattens on a slow wire. If the model fits replicated, plain DDP is the least-bad option. If it doesn't, pick the stage that fits in memory and just accept that you’re going to pay the wire's price.
Conclusion
We’ve covered a lot of ground. In the first half, we looked at the different strategies. DDP, ZeRO, and FSDP. Each of them answers the same question in a different way: what should each GPU store locally, and what should it fetch from the others? DDP keeps everything replicated. FSDP shards everything. ZeRO gives you the stops in between.
In the second half, we looked at the hardware layer, the fabric those requests travel over. On a fast fabric, sharding can be cheap enough that FSDP keeps pace with DDP while using much less memory. On a slow fabric, the same extra communication becomes visible, and can slow things down significantly.
That is the main lesson. The strategy decides how much data moves, but the wire decides how expensive that movement is. So before launching a distributed job, run one command:
`nvidia-smi topo -m`
It tells you whether your GPUs are connected by NVLink, NVSwitch, or a slower host path.
Thanks for reading, and I hope you found this helpful!
References and Further Reading
PyTorch FSDP documentation:
https://pytorch.org/docs/stable/fsdp.html
Getting Started with FSDP (PyTorch tutorial):
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
Hugging Face Accelerate, FSDP guide:
https://huggingface.co/docs/accelerate/usage_guides/fsdp
Hugging Face Accelerate, DeepSpeed guide:
https://huggingface.co/docs/accelerate/usage_guides/deepspeed
DeepSpeed ZeRO:
https://www.deepspeed.ai/tutorials/zero/
Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (the ZeRO paper):
https://arxiv.org/abs/1910.02054
NVIDIA, "What Is NVLink?" (NVLink and NVSwitch overview):
https://blogs.nvidia.com/blog/what-is-nvidia-nvlink/
NCCL documentation (collective operations and topology):
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
DINOv2 (Oquab et al.):
https://arxiv.org/abs/2304.07193
Mistral 7B (Jiang et al.):
https://arxiv.org/abs/2310.06825
Acknowledgments
This work is done at Memorial Sloan Kettering Cancer Center, and I want to thank all my colleagues at the Single-Cell Analytics Innovation Lab (
SAIL
) and the
Pe'er Lab
for their input and guidance.
Related Articles
Three Popular Machine Learning Methods
Artificial Intelligence
Understanding the Basic Machine Learning Types
Mike Wolfe
July 24, 2021
7 min read
The Model Validation Playbook for GenAI: Lessons from Banking
Artificial Intelligence
How model validation standards are changing for LLM-based systems: what breaks, what carries over, and how to test output quality
Ananya Bhattacharyya
September 8, 2026
14 min read
I Replaced Vector DBs with Google’s Memory Agent Pattern for my notes in Obsidian
Artificial Intelligence
Persistent AI memory without embeddings, Pinecone, or a PhD in similarity search.
Nick Lawson
April 3, 2026
11 min read
A Step-By-Step Guide to Approaching Complex Research Projects
How to Drive Focused Research in Data Science
Tal Rosenwein
October 5, 2021
14 min read
Google's New "Hum to Search" AI-Powered Feature to Search Song
Artificial Intelligence
AI-Powered way to Search Your Favourite Song
Pranjal Saxena
October 18, 2020
2 min read
Researching a Humanitarian Disaster Situation Report Chatbot - Using GPT-4-Turbo and full-context...
Exploring a Brute Force Technique for Information Retrieval
Matthew Harris
November 15, 2023
18 min read
How to Turn Your AI Idea Into a Scalable Product: A Technical Guide
Artificial Intelligence
Time to leave localhost behind and start acquiring users
Matt Chapman
July 2, 2024
8 min read
The Explainable Boosting Machine
Artificial Intelligence
As accurate as gradient boosting, as interpretable as linear regression.
Robert Kübler
April 2, 2021
11 min read
Designing the relationship between LLMs and user experience
Product Management
How to make your LLM do the right things, and do them right
Janna Lipenkova
April 19, 2024
14 min read