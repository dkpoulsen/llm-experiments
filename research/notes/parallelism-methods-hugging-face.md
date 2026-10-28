---
title: Parallelism methods · Hugging Face
id: parallelism-methods-hugging-face
tags:
- cmp170hx-training-feasibility-a0d4fc
- ddp
- zero
- fsdp
- parallelism-selection
- docs
created: '2026-10-27T11:40:47.754657Z'
updated: '2026-10-27T19:00:31.560263Z'
source: https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many
source_domain: huggingface.co
fetched_at: '2026-10-27T11:40:47.133642Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: 'Hugging Face Transformers documentation page ''Parallelism methods'' (main
  branch, v5.x) — a short (~1.4k word) strategy-selection guide, no benchmarks or
  numbers. It gives the decision table this run''s topology question needs: single
  node / fits on one GPU -> DistributedDataParallel or ZeRO; doesn''t fit -> PipelineParallel,
  ZeRO or TensorParallel; largest layer doesn''t fit -> TensorParallel or ZeRO; multi-node
  with fast inter-node connectivity (NVLink/NVSwitch) -> ZeRO or 3D parallelism; multi-node
  with slow inter-node connectivity -> ZeRO or 3D parallelism. It defines DP vs DDP
  (DDP recommended because it reduces communication overhead and scales past one machine),
  the three ZeRO stages (shard optimizer states / gradients / parameters), and notes
  that model, pipeline and tensor parallelism all add communication overhead and can
  become the bottleneck. Relevance: it is a ground-truth doc for the claim that DDP
  is the default when a model fits replicated, and that sharding is the fallback when
  it does not — but it says nothing about how a PCIe-only, no-P2P fabric changes those
  recommendations, so it cannot support the repo''s stronger claim that DDP is the
  *only* viable topology on this box.'
---

*Suggested by [[zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum]] — Cited by the Lyceum article as the primary reference for the DDP/ZeRO/FSDP memory and communication trade-off table*

Transformers documentation
Parallelism methods
# Transformers
🏡 View all docs AWS Trainium & Inferentia Accelerate Argilla AutoTrain Bitsandbytes CLI Chat UI Dataset viewer Datasets Deploying on AWS Diffusers Distilabel Evaluate Google Cloud Google TPUs Gradio Hub Hub Python Library Huggingface.js Inference Endpoints (dedicated) Inference Providers Kernels LeRobot Leaderboards Lighteval Microsoft Azure OpenEnv Optimum PEFT Reachy Mini Safetensors Sentence Transformers TRL Tasks Text Embeddings Inference Text Generation Inference Tokenizers Trackio Transformers Transformers.js Xet smolagents timm
Search documentation
`Ctrl+K`
main v5.21.0 v5.20.2 v5.19.0 v5.17.0 v5.15.1 v5.14.0 v5.13.1 v5.12.0 v5.11.0 v5.10.4 v5.9.0 v5.8.1 v5.7.0 v5.6.2 v5.5.4 v5.4.0 v5.3.0 v5.2.0 v5.1.0 v5.0.0 v4.57.6 v4.56.2 v4.55.4 v4.53.3 v4.52.3 v4.51.3 v4.50.0 v4.49.0 v4.48.2 v4.47.1 v4.46.3 v4.45.2 v4.44.2 v4.43.4 v4.42.4 v4.41.2 v4.40.2 v4.39.3 v4.38.2 v4.37.2 v4.36.1 v4.35.2 v4.34.1 v4.33.3 v4.32.1 v4.31.0 v4.30.0 v4.29.1 v4.28.1 v4.27.2 v4.26.1 v4.25.1 v4.24.0 v4.23.1 v4.22.2 v4.21.3 v4.20.1 v4.19.4 v4.18.0 v4.17.0 v4.16.2 v4.15.0 v4.14.1 v4.13.0 v4.12.5 v4.11.3 v4.10.1 v4.9.2 v4.8.2 v4.7.0 v4.6.0 v4.5.1 v4.4.2 v4.3.3 v4.2.2 v4.1.1 v4.0.1 v3.5.1 v3.4.0 v3.3.1 v3.2.0 v3.1.0 v3.0.2 v2.11.0 v2.10.0 v2.9.1 v2.8.0 v2.7.0 v2.6.0 v2.5.1 v2.4.1 v2.3.0 v2.2.2 v2.1.1 v2.0.0 v1.2.0 v1.1.0 v1.0.0 doc-builder-html AR DE EN ES FR HI IT JA KO PT RO TE TR ZH
You are viewing main version, which requires [installation from source](https://huggingface.co/docs/transformers/installation#install-from-source). If you'd like regular pip install, checkout the latest stable version ([v5.21.0](https://huggingface.co/docs/transformers/v5.21.0/perf_train_gpu_many)).
Join the Hugging Face community
and get access to the augmented documentation experience
Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
[Sign Up](https://huggingface.co/join)
to get started
Copy page
#  Parallelism methods
Multi-GPU setups are effective for accelerating training and fitting large models in memory that otherwise wouldn’t fit on a single GPU. It relies on parallelizing the workload across GPUs. There are several types of parallelism such as data parallelism, tensor parallelism, pipeline parallelism, and model parallelism. Each type of parallelism splits the workload differently, whether it’s the data or the model.
This guide will discuss the various parallelism methods, combining them, and choosing an appropriate strategy for your setup. For more details about distributed training, refer to the [Accelerate](https://hf.co/docs/accelerate/index) documentation.
For a comprehensive guide on scaling large language models, check out the [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), which provides detailed strategies and best practices for training at scale.
##  Scalability strategy
Use the [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) to calculate how much memory a model requires. Then refer to the table below to select a strategy based on your setup.  
| setup  | scenario  | strategy  |  
| --- | --- | --- |  
| single node/multi-GPU  | fits on single GPU  | DistributedDataParallel or ZeRO  |  
| doesn’t fit on single GPU  | PipelineParallel, ZeRO or TensorParallel  |  
| largest model layer doesn’t fit  | TensorParallel or ZeRO  |  
| multi-node/multi-GPU  | fast inter-node connectivity (NVLink or NVSwitch)  | ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel)  |  
| slow inter-node connectivity  | ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel)  |  
##  Data parallelism
Data parallelism evenly distributes data across multiple GPUs. Each GPU holds a copy of the model and concurrently processes their portion of the data. At the end, the results from each GPU are synchronized and combined.
Data parallelism significantly reduces training time by processing data in parallel, and it is scalable to the number of GPUs available. However, synchronizing results from each GPU can add overhead.
There are two types of data parallelism, DataParallel (DP) and DistributedDataParallel (DDP).
###  DataParallel
[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) supports distributed training on a _single machine_ with multiple GPUs.
  1. The default GPU, `GPU 0`, reads a batch of data and sends a mini batch of it to the other GPUs.
  2. An up-to-date model is replicated from `GPU 0` to the other GPUs.
  3. A `forward` pass is performed on each GPU and their outputs are sent to `GPU 0` to compute the loss.
  4. The loss is distributed from `GPU 0` to the other GPUs for the `backward` pass.
  5. The gradients from each GPU are sent back to `GPU 0` and averaged.


###  DistributedDataParallel
[DistributedDataParallel](https://pytorch.org/docs/main/notes/ddp.html) supports distributed training across _multiple machines_ with multiple GPUs.
  1. The main process replicates the model from the default GPU, `GPU 0`, to each GPU.
  2. Each GPU directly processes a mini batch of data.
  3. The local gradients are averaged across all GPUs during the `backward` pass.


DDP is recommended because it reduces communication overhead between GPUs, efficiently utilizes each GPU, and scales to more than one machine.
###  ZeRO data parallelism
[Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/) is a more memory efficient type of data parallelism. It significantly improves memory efficiency by partitioning parameters, gradients, and optimizer states across data parallel processes to reduce memory usage. There are three ZeRO stages:
  * Stage 1 partitions the optimizer states
  * Stage 2 partitions the optimizer and gradient states
  * Stage 3 partitions the optimizer, gradient, and parameters


##  Model parallelism
Model parallelism distributes a model across multiple GPUs. There are several ways to split a model, but the typical method distributes the model layers across GPUs. On the `forward` pass, the first GPU processes a batch of data and passes it to the next group of layers on the next GPU. For the `backward` pass, the data is sent backward from the final layer to the first layer.
Model parallelism is a useful strategy for training models that are too large to fit into the memory of a single GPU. However, GPU utilization is unbalanced because only one GPU is active at a time. Passing results between GPUs also adds communication overhead and it can be a bottleneck.
##  Pipeline parallelism
Pipeline parallelism is conceptually very similar to model parallelism, but it’s more efficient because it reduces the amount of idle GPU time. Instead of waiting for each GPU to finish processing a batch of data, pipeline parallelism creates _micro-batches_ of data. As soon as one micro-batch is finished, it is passed to the next GPU. This way, each GPU can concurrently process part of the data without waiting for the other GPU to completely finish processing a mini batch of data.
Pipeline parallelism shares the same advantages as model parallelism, but it optimizes GPU utilization and reduces idle time. But pipeline parallelism can be more complex because models may need to be rewritten as a sequence of [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) modules and it also isn’t possible to completely reduce idle time because the last `forward` pass must also wait for the `backward` pass to finish.
##  Tensor parallelism
Tensor parallelism distributes large tensor computations across multiple GPUs. The tensors are sliced horizontally or vertically and each slice is processed by a separate GPU. Each GPU performs its calculations on its tensor slice and the results are synchronized at the end to reconstruct the final result.
Tensor parallelism is effective for training large models that don’t fit into the memory of a single GPU. It is also faster and more efficient because each GPU can process its tensor slice in parallel, and it can be combined with other parallelism methods. Like other parallelism methods though, tensor parallelism adds communication overhead between GPUs.
Refer to the [Tensor parallelism](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi) guide to learn how to use it for inference.
##  Hybrid parallelism
Parallelism methods can be combined to achieve even greater memory savings and more efficiently train models with billions of parameters.
###  Data parallelism and pipeline parallelism
Data and pipeline parallelism distributes the data across GPUs and divides each mini batch of data into micro-batches to achieve pipeline parallelism.
Each data parallel rank treats the process as if there were only one GPU instead of two, but GPUs 0 and 1 can offload micro-batches of data to GPUs 2 and 3 and reduce idle time.
This approach optimizes parallel data processing by reducing idle GPU utilization.
###  ZeRO data parallelism, pipeline parallelism, and model parallelism (3D parallelism)
Data, pipeline and model parallelism combine to form [3D parallelism](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) to optimize memory and compute efficiency.
Memory efficiency is achieved by splitting the model across GPUs and also dividing it into stages to create a pipeline. This allows GPUs to work in parallel on micro-batches of data, reducing the memory usage of the model, optimizer, and activations.
Compute efficiency is enabled by ZeRO data parallelism where each GPU only stores a slice of the model, optimizer, and activations. This allows higher communication bandwidth between data parallel nodes because communication can occur independently or in parallel with the other pipeline stages.
This approach is scalable to extremely large models with trillions of parameters.
[ Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_train_gpu_many.md)
[←Debugging](https://huggingface.co/docs/transformers/main/en/debugging) [Overview→](https://huggingface.co/docs/transformers/main/en/hardware)
