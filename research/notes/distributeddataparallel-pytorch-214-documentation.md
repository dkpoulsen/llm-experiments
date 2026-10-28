---
title: DistributedDataParallel — PyTorch 2.14 documentation
id: distributeddataparallel-pytorch-214-documentation
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-memory-budget-and-frozen-tables
created: '2026-10-27T13:16:24.016316Z'
updated: '2026-10-27T19:00:31.656514Z'
source: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
source_domain: pytorch.org
fetched_at: '2026-10-27T13:16:23.395107Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: 'Suggested by distributed-data-parallel-pytorch-214-documentation — DDP class
  reference: whether params with requiresg...'
---

*Suggested by [[distributed-data-parallel-pytorch-214-documentation]] — DDP class reference: whether params with requires_grad=False are excluded from gradient synchronization*

Opens in a new window Opens an external website Opens an external website in a new window  
<!---->Close this dialog<!---->
This website utilises technologies such as cookies to enable essential site functionality, as well as for analytics, personalisation, and targeted advertising. To learn more, view the following link:  [Privacy Policy](https://lfprojects.org/policies/privacy-policy/)
Manage Preferences 
<!---->Close Cookie Preferences<!---->
[Skip to main content](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#main-content)
Back to top
v2.14.0 (stable)
[v2.15.0 (unstable)](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.14.0 (stable)](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.13.0](https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.12.0](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.11.0](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.10.0](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.9.1](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.8.0](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.7.0](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.6.0](https://docs.pytorch.org/docs/2.6/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.5.0](https://docs.pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.4.0](https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.3.0](https://docs.pytorch.org/docs/2.3/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.2.0](https://docs.pytorch.org/docs/2.2/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.1.0](https://docs.pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v2.0.0](https://docs.pytorch.org/docs/2.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.13](https://docs.pytorch.org/docs/1.13/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.12](https://docs.pytorch.org/docs/1.12/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.11](https://docs.pytorch.org/docs/1.11/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.10](https://docs.pytorch.org/docs/1.10/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.9.1](https://docs.pytorch.org/docs/1.9.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.9.0](https://docs.pytorch.org/docs/1.9.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.8.1](https://docs.pytorch.org/docs/1.8.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.8.0](https://docs.pytorch.org/docs/1.8.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.7.1](https://docs.pytorch.org/docs/1.7.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.7.0](https://docs.pytorch.org/docs/1.7.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.6.0](https://docs.pytorch.org/docs/1.6.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.5.1](https://docs.pytorch.org/docs/1.5.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.5.0](https://docs.pytorch.org/docs/1.5.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.4.0](https://docs.pytorch.org/docs/1.4.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.3.1](https://docs.pytorch.org/docs/1.3.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.3.0](https://docs.pytorch.org/docs/1.3.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.2.0](https://docs.pytorch.org/docs/1.2.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.1.0](https://docs.pytorch.org/docs/1.1.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.0.1](https://docs.pytorch.org/docs/1.0.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v1.0.0](https://docs.pytorch.org/docs/1.0.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.4.1](https://docs.pytorch.org/docs/0.4.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.4.0](https://docs.pytorch.org/docs/0.4.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.3.1](https://docs.pytorch.org/docs/0.3.1/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.3.0](https://docs.pytorch.org/docs/0.3.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.2.0](https://docs.pytorch.org/docs/0.2.0/generated/torch.nn.parallel.DistributedDataParallel.html)[v0.1.12](https://docs.pytorch.org/docs/0.1.12/generated/torch.nn.parallel.DistributedDataParallel.html)
[ Go to pytorch.org  ](https://pytorch.org)
Custom Search  
|  Sort by Relevance Date  |  
| --- |  
  * [ GitHub](https://github.com/pytorch/pytorch)
  * [ PyTorch Forum](https://discuss.pytorch.org/)


Rate this Page 
★ ★ ★ ★ ★
# DistributedDataParallel[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#distributeddataparallel "Link to this heading") 

_class _torch.nn.parallel.DistributedDataParallel(_module_ , _device_ids_ , _output_device_ , __,_broadcast_buffers_ , _init_sync_ , _process_group_ , _bucket_cap_mb_ , _find_unused_parameters False_, _check_reduction False_, _gradient_as_bucket_view False_, _static_graph False_, _delay_all_reduce_named_params_ , _param_to_hook_all_reduce_ , _mixed_precision_ , _device_mesh_ , _skip_all_reduce_unused_params False_, _bucket_cap_mb_list_ , _batched_grad_copy False_, _forward_sync_buffers_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/parallel/distributed.py#L466)[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "Link to this definition") 
    
Implement distributed data parallelism based on `torch.distributed` at module level.
This container provides data parallelism by synchronizing gradients across each model replica. The devices to synchronize across are specified by the input `process_group`, which is the entire world by default. Note that `DistributedDataParallel` does not chunk or otherwise shard the input across participating GPUs; the user is responsible for defining how to do so, for example through the use of a `DistributedSampler`.
See also: [Basics](https://docs.pytorch.org/docs/2.14/distributed.html#distributed-basics) and [Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-nn-ddp-instead). The same constraints on input as in [`torch.nn.DataParallel`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") apply.
Creation of this class requires that `torch.distributed` to be already initialized, by calling [`torch.distributed.init_process_group()`](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group "torch.distributed.init_process_group").
`DistributedDataParallel` is proven to be significantly faster than [`torch.nn.DataParallel`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") for single-node multi-GPU data parallel training.
To use `DistributedDataParallel` on a host with N GPUs, you should spawn up processes, ensuring that each process exclusively works on a single GPU from 0 to N-1. This can be done by either setting `CUDA_VISIBLE_DEVICES` for every process or by calling the following API for GPUs,

```
>>> torch.cuda.set_device(i)

```
Copy to clipboard
or calling the unified API for [accelerator](https://docs.pytorch.org/docs/2.14/torch.html#accelerators),

```
>>> torch.accelerator.set_device_index(i)

```
Copy to clipboard
where i is from 0 to N-1. In each process, you should refer the following to construct this module:

```
>>> if torch.accelerator.is_available():
>>>     device_type = torch.accelerator.current_accelerator().type
>>>     vendor_backend = torch.distributed.get_default_backend_for_device(device_type)
>>>
>>> torch.distributed.init_process_group(
>>>     backend=vendor_backend, world_size=N, init_method='...'
>>> )
>>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

```
Copy to clipboard
Or you can use the latest API for initialization:

```
>>> torch.distributed.init_process_group(device_id=i)

```
Copy to clipboard
In order to spawn up multiple processes per node, you can use either `torch.distributed.launch` or `torch.multiprocessing.spawn`.
Note
Please refer to [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) for a brief introduction to all features related to distributed training.
Note
`DistributedDataParallel` can be used in conjunction with [`torch.distributed.optim.ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/2.14/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer "torch.distributed.optim.ZeroRedundancyOptimizer") to reduce per-rank optimizer states memory footprint. Please refer to [ZeroRedundancyOptimizer recipe](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html) for more details.
Note
`nccl` backend is currently the fastest and highly recommended backend when using GPUs. This applies to both single-node and multi-node distributed training.
Note
This module also supports mixed-precision distributed training. This means that your model can have different types of parameters such as mixed types of `fp16` and `fp32`, the gradient reduction on these mixed types of parameters will just work fine.
Note
If you use `torch.save` on one process to checkpoint the module, and `torch.load` on some other processes to recover it, make sure that `map_location` is configured properly for every process. Without `map_location`, `torch.load` would recover the module to devices where the module was saved from.
Note
When a model is trained on nodes with `batch=N`, the gradient will be times smaller when compared to the same model trained on a single node with `batch=M*N` if the loss is summed (NOT averaged as usual) across instances in a batch (because the gradients between different nodes are averaged). You should take this into consideration when you want to obtain a mathematically equivalent training process compared to the local training counterpart. But in most cases, you can just treat a DistributedDataParallel wrapped model, a DataParallel wrapped model and an ordinary model on a single GPU as the same (E.g. using the same learning rate for equivalent batch size).
Note
Parameters are never broadcast between processes. The module performs an all-reduce step on gradients and assumes that they will be modified by the optimizer in all processes in the same way. Buffers (e.g. BatchNorm stats) are broadcast from the module in process of rank 0, to all other replicas in the system in every iteration.
Note
If you are using DistributedDataParallel in conjunction with the [Distributed RPC Framework](https://docs.pytorch.org/docs/2.14/rpc.html#distributed-rpc-framework), you should always use [`torch.distributed.autograd.backward()`](https://docs.pytorch.org/docs/2.14/rpc.html#torch.distributed.autograd.backward "torch.distributed.autograd.backward") to compute gradients and [`torch.distributed.optim.DistributedOptimizer`](https://docs.pytorch.org/docs/2.14/distributed.optim.html#torch.distributed.optim.DistributedOptimizer "torch.distributed.optim.DistributedOptimizer") for optimizing parameters.
Example:

```
>>> importtorch.distributed.autogradasdist_autograd
>>> fromtorch.nn.parallelimport DistributedDataParallel as DDP
>>> importtorch
>>> fromtorchimport optim
>>> fromtorch.distributed.optimimport DistributedOptimizer
>>> importtorch.distributed.rpcasrpc
>>> fromtorch.distributed.rpcimport RRef
>>>
>>> t1 = torch.rand((3, 3), requires_grad=True)
>>> t2 = torch.rand((3, 3), requires_grad=True)
>>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))
>>> ddp_model = DDP(my_model)
>>>
>>> # Setup optimizer
>>> optimizer_params = [rref]
>>> for param in ddp_model.parameters():
>>>     optimizer_params.append(RRef(param))
>>>
>>> dist_optim = DistributedOptimizer(
>>>     optim.SGD,
>>>     optimizer_params,
>>>     lr=0.05,
>>> )
>>>
>>> with dist_autograd.context() as context_id:
>>>     pred = ddp_model(rref.to_here())
>>>     loss = loss_func(pred, target)
>>>     dist_autograd.backward(context_id, [loss])
>>>     dist_optim.step(context_id)

```
Copy to clipboard
Note
DistributedDataParallel currently offers limited support for gradient checkpointing with [`torch.utils.checkpoint()`](https://docs.pytorch.org/docs/2.14/utils.html#module-torch.utils.checkpoint "torch.utils.checkpoint"). If the checkpoint is done with use_reentrant=False (recommended), DDP will work as expected without any limitations. If, however, the checkpoint is done with use_reentrant=True (the default), DDP will work as expected when there are no unused parameters in the model and each layer is checkpointed at most once (make sure you are not passing find_unused_parameters=True to DDP). We currently do not support the case where a layer is checkpointed multiple times, or when there unused parameters in the checkpointed model.
Note
To let a non-DDP model load a state dict from a DDP model, [`consume_prefix_in_state_dict_if_present()`](https://docs.pytorch.org/docs/2.14/nn.aliases.html#torch.nn.modules.utils.consume_prefix_in_state_dict_if_present "torch.nn.modules.utils.consume_prefix_in_state_dict_if_present") needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
Warning
Constructor, forward method, and differentiation of the output (or a function of the output of this module) are distributed synchronization points. Take that into account in case different processes might be executing different code.
Warning
This module assumes all parameters are registered in the model by the time it is created. No parameters should be added nor removed later. Same applies to buffers.
Warning
This module assumes all parameters are registered in the model of each distributed processes are in the same order. The module itself will conduct gradient `allreduce` following the reverse order of the registered parameters of the model. In other words, it is users’ responsibility to ensure that each distributed process has the exact same model and thus the exact same parameter registration order.
Warning
This module allows parameters with non-rowmajor-contiguous strides. For example, your model may contain some parameters whose [`torch.memory_format`](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch.memory_format "torch.memory_format") is `torch.contiguous_format` and others whose format is `torch.channels_last`. However, corresponding parameters in different processes must have the same strides.
Warning
This module doesn’t work with [`torch.autograd.grad()`](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad") (i.e. it will only work if gradients are to be accumulated in `.grad` attributes of parameters).
Warning
If you plan on using this module with a `nccl` backend or a `gloo` backend (that uses Infiniband), together with a DataLoader that uses multiple workers, please change the multiprocessing start method to `forkserver` or `spawn`. Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will likely experience deadlocks if you don’t change this setting.
Warning
You should never try to change your model’s parameters after wrapping up your model with `DistributedDataParallel`. Because, when wrapping up your model with `DistributedDataParallel`, the constructor of `DistributedDataParallel` will register the additional gradient reduction functions on all the parameters of the model itself at the time of construction. If you change the model’s parameters afterwards, gradient reduction functions no longer match the correct set of parameters.
Warning
Using `DistributedDataParallel` in conjunction with the [Distributed RPC Framework](https://docs.pytorch.org/docs/2.14/rpc.html#distributed-rpc-framework) is experimental and subject to change. 

Parameters: 
    
  * **module** ([_Module_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – module to be parallelized
  * **device_ids** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _of_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") _or_[ _torch.device_](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch.device "torch.device")) – 
CUDA devices. 1) For single-device modules, `device_ids` can contain exactly one device id, which represents the only CUDA device where the input module corresponding to this process resides. Alternatively, `device_ids` can also be `None`. 2) For multi-device modules and CPU modules, `device_ids` must be `None`.
When `device_ids` is `None` for both cases, both the input data for the forward pass and the actual module must be placed on the correct device. (default: `None`)
  * **output_device** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") _or_[ _torch.device_](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch.device "torch.device")) – Device location of output for single-device CUDA modules. For multi-device modules and CPU modules, it must be `None`, and the module itself dictates the output location. (default: `device_ids[0]` for single-device modules)
  * **broadcast_buffers** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _or_ _None_) – 
Flag that enables syncing (broadcasting) buffers of the module at beginning of the `forward` function. (default: `None`)
Deprecated since version 2.13: Use `forward_sync_buffers` instead.
  * **init_sync** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Whether to sync during initialization to verify param shapes and broadcast parameters and buffers. Note: the deprecated `broadcast_buffers=False` excludes buffers from this init sync. The replacement `forward_sync_buffers` does not affect init sync. WARNING: if this is set to False the user is required to ensure themselves that the weights are the same on all ranks. (default: `True`)
  * **process_group** – The process group to be used for distributed data all-reduction. If `None`, the default process group, which is created by [`torch.distributed.init_process_group()`](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group "torch.distributed.init_process_group"), will be used. (default: `None`)
  * **bucket_cap_mb** – `DistributedDataParallel` will bucket parameters into multiple buckets so that gradient reduction of each bucket can potentially overlap with backward computation. `bucket_cap_mb` controls the bucket size in MebiBytes (MiB). If `None`, a default size of 25 MiB will be used. (default: `None`)
  * **find_unused_parameters** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Traverse the autograd graph from all tensors contained in the return value of the wrapped module’s `forward` function. Parameters that don’t receive gradients as part of this graph are preemptively marked as being ready to be reduced. In addition, parameters that may have been used in the wrapped module’s `forward` function but were not part of loss computation and thus would also not receive gradients are preemptively marked as ready to be reduced. (default: `False`)
  * **check_reduction** – This argument is deprecated.
  * **gradient_as_bucket_view** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – When set to `True`, gradients will be views pointing to different offsets of `allreduce` communication buckets. This can reduce peak memory usage, where the saved memory size will be equal to the total gradients size. Moreover, it avoids the overhead of copying between gradients and `allreduce` communication buckets. When gradients are views, `detach_()` cannot be called on the gradients. If hitting such errors, please fix it by referring to the [`zero_grad()`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad "torch.optim.Optimizer.zero_grad") function in `torch/optim/optimizer.py` as a solution. Note that gradients will be views after first iteration, so the peak memory saving should be checked after first iteration.
  * **static_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – 
When set to `True`, DDP knows the trained graph is static. Static graph means 1) The set of used and unused parameters will not change during the whole training loop; in this case, it does not matter whether users set `find_unused_parameters = True` or not. 2) How the graph is trained will not change during the whole training loop (meaning there is no control flow depending on iterations). When static_graph is set to be `True`, DDP will support cases that can not be supported in the past: 1) Reentrant backwards. 2) Activation checkpointing multiple times. 3) Activation checkpointing when model has unused parameters. 4) There are model parameters that are outside of forward function. 5) Potentially improve performance when there are unused parameters, as DDP will not search graph in each iteration to detect unused parameters when static_graph is set to be `True`. To check whether you can set static_graph to be `True`, one way is to check ddp logging data at the end of your previous model training, if `ddp_logging_data.get("can_set_static_graph") == True`, mostly you can set `static_graph = True` as well. 

Example::
    

```
>>> model_DDP = torch.nn.parallel.DistributedDataParallel(model)
>>> # Training loop
>>> ...
>>> ddp_logging_data = model_DDP._get_ddp_logging_data()
>>> static_graph = ddp_logging_data.get("can_set_static_graph")

```
Copy to clipboard
  * **delay_all_reduce_named_params** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _of_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _of_ _str and torch.nn.Parameter_) – a list of named parameters whose all reduce will be delayed when the gradient of the parameter specified in `param_to_hook_all_reduce` is ready. Other arguments of DDP do not apply to named params specified in this argument as these named params will be ignored by DDP reducer.
  * **param_to_hook_all_reduce** (_torch.nn.Parameter_) – a parameter to hook delayed all reduce of parameters specified in `delay_all_reduce_named_params`.
  * **skip_all_reduce_unused_params** – When set to True, DDP will skip reducing unused parameters. This requires that unused parameters remain the same across all ranks throughout the entire training process. If this condition is not met, it may cause desynchronization and result in training hang.
  * **batched_grad_copy** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – When set to `True`, individual per-parameter gradient-to-bucket copy and division operations are deferred and flushed as a single `_foreach_copy_` plus one flat `div_` when a bucket becomes ready. This reduces per-parameter kernel launches down to 2 kernels per bucket, which can improve throughput for models with many small parameters. The optimization is most effective with `optimizer.zero_grad(set_to_none=True)` (the default), where `gradient_as_bucket_view` alone cannot avoid copies because the bucket view alias is destroyed every iteration. (default: `False`)
  * **forward_sync_buffers** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _or_ _None_) – Flag that enables syncing (broadcasting) buffers of the module at runtime, including at the beginning of `forward` and after uneven-input joins. Does not affect initialization sync (see `init_sync`); buffers are always synced at init regardless of this flag. Replaces the deprecated `broadcast_buffers` argument. When `None`, defaults to `True`. (default: `None`)



Variables: 
    
**module** ([_Module_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – the module to be parallelized.
Example:

```
>>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
>>> net = torch.nn.parallel.DistributedDataParallel(model)

```
Copy to clipboard 

(_divide_by_initial_world_size_ , _enable_ , _throw_on_early_termination False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/parallel/distributed.py#L1990)[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join "Link to this definition") 
    
Context manager for training with uneven inputs across processes in DDP.
This context manager will keep track of already-joined DDP processes, and “shadow” the forward and backward passes by inserting collective communication operations to match with the ones created by non-joined DDP processes. This will ensure each collective call has a corresponding call by already-joined DDP processes, preventing hangs or errors that would otherwise happen when training with uneven inputs across processes. Alternatively, if the flag `throw_on_early_termination` is specified to be `True`, all trainers will throw an error once one rank runs out of inputs, allowing these errors to be caught and handled according to application logic.
Once all DDP processes have joined, the context manager will broadcast the model corresponding to the last joined process to all processes to ensure the model is the same across all processes (which is guaranteed by DDP).
To use this to enable training with uneven inputs across processes, simply wrap this context manager around your training loop. No further modifications to the model or data loading is required.
Warning
If the model or training loop this context manager is wrapped around has additional distributed collective operations, such as `SyncBatchNorm` in the model’s forward pass, then the flag `throw_on_early_termination` must be enabled. This is because this context manager is not aware of non-DDP collective communication. This flag will cause all ranks to throw when any one rank exhausts inputs, allowing these errors to be caught and recovered from across all ranks. 

Parameters: 
    
  * **divide_by_initial_world_size** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, will divide gradients by the initial `world_size` DDP training was launched with. If `False`, will compute the effective world size (number of ranks that have not depleted their inputs yet) and divide gradients by that during allreduce. Set `divide_by_initial_world_size=True` to ensure every input sample including the uneven inputs have equal weight in terms of how much they contribute to the global gradient. This is achieved by always dividing the gradient by the initial `world_size` even when we encounter uneven inputs. If you set this to `False`, we divide the gradient by the remaining number of nodes. This ensures parity with training on a smaller `world_size` although it also means the uneven inputs would contribute more towards the global gradient. Typically, you would want to set this to `True` for cases where the last few inputs of your training job are uneven. In extreme cases, where there is a large discrepancy in the number of inputs, setting this to `False` might provide better results.
  * **enable** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Whether to enable uneven input detection or not. Pass in `enable=False` to disable in cases where you know that inputs are even across participating processes. Default is `True`.
  * **throw_on_early_termination** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Whether to throw an error or continue training when at least one rank has exhausted inputs. If `True`, will throw upon the first rank reaching end of data. If `False`, will continue training with a smaller effective world size until all ranks are joined. Note that if this flag is specified, then the flag `divide_by_initial_world_size` would be ignored. Default is `False`.


Example:

```
>>> importtorch
>>> importtorch.distributedasdist
>>> importos
>>> importtorch.multiprocessingasmp
>>> importtorch.nnasnn
>>> # On each spawned worker
>>> defworker(rank):
>>>     dist.init_process_group("nccl", rank=rank, world_size=2)
>>>     torch.cuda.set_device(rank)
>>>     model = nn.Linear(1, 1, bias=False).to(rank)
>>>     model = torch.nn.parallel.DistributedDataParallel(
>>>         model, device_ids=[rank], output_device=rank
>>>     )
>>>     # Rank 1 gets one more input than rank 0.
>>>     inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
>>>     with model.join():
>>>         for _ in range(5):
>>>             for inp in inputs:
>>>                 loss = model(inp).sum()
>>>                 loss.backward()
>>>     # Without the join() API, the below synchronization will hang
>>>     # blocking for rank 1's allreduce to complete.
>>>     torch.cuda.synchronize(device=rank)

```
Copy to clipboard 

join_hook(_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/parallel/distributed.py#L2096)[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook "Link to this definition") 
    
DDP join hook enables training on uneven inputs by mirroring communications in forward and backward passes. 

Parameters: 
    
**kwargs** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")) – a containing any keyword arguments to modify the behavior of the join hook at run time; all `Joinable` instances sharing the same join context manager are forwarded the same value for `kwargs`. 

The hook supports the following keyword arguments:
     

divide_by_initial_world_size (bool, optional):
    
If `True`, then gradients are divided by the initial world size that DDP was launched with. If `False`, then gradients are divided by the effective world size (i.e. the number of non-joined processes), meaning that the uneven inputs contribute more toward the global gradient. Typically, this should be set to `True` if the degree of unevenness is small but can be set to `False` in extreme cases for possibly better results. Default is `True`. 

no_sync()[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/parallel/distributed.py#L1659)[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync "Link to this definition") 
    
Context manager to disable gradient synchronizations across DDP processes.
Within this context, gradients will be accumulated on module variables, which will later be synchronized in the first forward-backward pass exiting the context.
Example:

```
>>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
>>> with ddp.no_sync():
>>>     for input in inputs:
>>>         ddp(input).backward()  # no synchronization, accumulate grads
>>> ddp(another_input).backward()  # synchronize grads

```
Copy to clipboard
Warning
The forward pass should be included inside the context manager, or else gradients will still be synchronized. 

register_comm_hook(_state_ , __)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/parallel/distributed.py#L2179)[#](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook "Link to this definition") 
    
Register communication hook for user-defined DDP aggregation of gradients across multiple workers.
This hook would be very useful for researchers to try out new ideas. For example, this hook can be used to implement several algorithms like GossipGrad and gradient compression which involve different communication strategies for parameter syncs while running Distributed DataParallel training. 

Parameters: 
    
  * **state** ([_object_](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")) – 
Passed to the hook to maintain any state information during the training process. Examples include error feedback in gradient compression, peers to communicate with next in GossipGrad, etc.
It is locally stored by each worker and shared by all the gradient tensors on the worker.
  * **hook** (_Callable_) – 
Callable with the following signature: `hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]`:
This function is called once the bucket is ready. The hook can perform whatever processing is needed and return a Future indicating completion of any async work (ex: allreduce). If the hook doesn’t perform any communication, it still must return a completed Future. The Future should hold the new value of grad bucket’s tensors. Once a bucket is ready, c10d reducer would call this hook and use the tensors returned by the Future and copy grads to individual parameters. Note that the future’s return type must be a single tensor.
We also provide an API called `get_future` to retrieve a Future associated with the completion of `c10d.ProcessGroup.Work`. `get_future` is currently supported for NCCL and also supported for most operations on GLOO and MPI, except for peer to peer operations (send/recv).


Warning
Grad bucket’s tensors will not be predivided by world_size. User is responsible to divide by the world_size in case of operations like allreduce.
Warning
DDP communication hook can only be registered once and should be registered before calling backward.
Warning
The Future object that hook returns should contain a single tensor that has the same shape with the tensors inside grad bucket.
Warning
`get_future` API supports NCCL, and partially GLOO and MPI backends (no support for peer-to-peer operations like send/recv) and will return a `torch.futures.Future`. 

Example::
    
Below is an example of a noop hook that returns the same tensor.

```
>>> defnoop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
>>>     fut = torch.futures.Future()
>>>     fut.set_result(bucket.buffer())
>>>     return fut
>>> ddp.register_comm_hook(state=None, hook=noop)

```
Copy to clipboard 

Example::
    
Below is an example of a Parallel SGD algorithm where gradients are encoded before allreduce, and then decoded after allreduce.

```
>>> defencode_and_decode(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
>>>     encoded_tensor = encode(bucket.buffer())  # encode gradients
>>>     fut = torch.distributed.all_reduce(encoded_tensor).get_future()
>>>     # Define the then callback to decode.
>>>     defdecode(fut):
>>>         decoded_tensor = decode(fut.value()[0])  # decode gradients
>>>         return decoded_tensor
>>>     return fut.then(decode)
>>> ddp.register_comm_hook(state=None, hook=encode_and_decode)

```
Copy to clipboard
On this page 
[ Show Source ](https://docs.pytorch.org/docs/2.14/_sources/generated/torch.nn.parallel.DistributedDataParallel.rst.txt)
PyTorch Libraries
  * [PyTorch on XLA Devices](https://docs.pytorch.org/xla)


## Docs
Access comprehensive developer documentation for PyTorch
[View Docs](https://docs.pytorch.org/docs/stable/index.html)
## Tutorials
Get in-depth tutorials for beginners and advanced developers
[View Tutorials](https://docs.pytorch.org/tutorials)
## Resources
Find development resources and get your questions answered
[View Resources](https://pytorch.org/resources)
To analyze traffic and optimize your experience, we serve cookies on this site. By clicking or navigating, you agree to allow our usage of cookies. As the current maintainers of this site, Facebook’s Cookies Policy applies. Learn more, including about available controls: [Cookies Policy](https://opensource.fb.com/legal/cookie-policy).
