---
title: FullyShardedDataParallel — PyTorch 2.13 documentation
id: fullyshardeddataparallel-pytorch-213-documentation
tags:
- cmp170hx-training-feasibility-a0d4fc
- fsdp
- sharding-strategy
- frozen-parameters
- communication-volume
- docs
created: '2026-10-27T11:41:09.939639Z'
updated: '2026-10-27T19:00:31.563362Z'
source: https://docs.pytorch.org/docs/2.13/fsdp.html
source_domain: docs.pytorch.org
fetched_at: '2026-10-27T11:41:09.317520Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: 'PyTorch 2.13 FullyShardedDataParallel API documentation (torch.distributed.fsdp)
  — the primary reference for exactly what FSDP moves over the wire and how it treats
  frozen parameters. Sharding strategies: FULL_SHARD all-gathers parameters before
  the forward, reshards after it, all-gathers again before backward, reshards after
  backward, and reduce-scatters gradients after backward, with optimizer states updated
  locally per rank; SHARD_GRAD_OP unshards before forward but does not reshard until
  after backward (so parameters stay gathered between passes); NO_SHARD is DDP-equivalent
  and all-reduces gradients; HYBRID_SHARD does FULL_SHARD within a node and replicates
  across nodes, cutting communication volume because the expensive all-gathers and
  reduce-scatters stay intra-node. Memory/overlap knobs: backward_prefetch=BACKWARD_PRE
  holds current params, next params and current gradients at peak (most overlap, most
  memory) while BACKWARD_POST frees current params before allocating the next set;
  limit_all_gathers (default True) is a rate limiter that caps GPU memory to two consecutive
  FSDP instances'' worth of gathered parameters. Frozen-parameter constraint, directly
  relevant to a frozen-Engram plan: with use_orig_params=False each FSDP instance
  must manage parameters that are all frozen or all non-frozen; with use_orig_params=True
  FSDP supports mixing frozen and non-frozen parameters but recommends against it
  because of higher-than-expected gradient memory. Also: FSDP cannot run the forward
  of a submodule contained in an FSDP instance (its params are sharded but no all-gather
  happens), and mixed precision can be disabled per module at the cost of separate
  all-gathers/reduce-scatters. Relevance: this is the authoritative source for the
  claim that FSDP''s per-layer all-gather/reduce-scatter pattern is what makes it
  expensive on a slow fabric, and it shows the frozen-table plan needs its own all-frozen
  FSDP unit rather than a mixed unit.'
---

*Suggested by [[zero-3-vs-fsdp-memory-efficiency-for-llm-training-lyceum]] — Cited by the Lyceum article as the primary reference for FSDP's FULL_SHARD lifecycle and frozen-parameter handling*

Opens in a new window Opens an external website Opens an external website in a new window  
<!---->Close this dialog<!---->
This website utilises technologies such as cookies to enable essential site functionality, as well as for analytics, personalisation, and targeted advertising. To learn more, view the following link:  [Privacy Policy](https://lfprojects.org/policies/privacy-policy/)
Manage Preferences 
<!---->Close Cookie Preferences<!---->
[Skip to main content](https://docs.pytorch.org/docs/2.13/fsdp.html#main-content)
Back to top
v2.13.0
[v2.15.0 (unstable)](https://docs.pytorch.org/docs/main/fsdp.html)[v2.14.0 (stable)](https://docs.pytorch.org/docs/2.14/fsdp.html)[v2.13.0](https://docs.pytorch.org/docs/2.13/fsdp.html)[v2.12.0](https://docs.pytorch.org/docs/2.12/fsdp.html)[v2.11.0](https://docs.pytorch.org/docs/2.11/fsdp.html)[v2.10.0](https://docs.pytorch.org/docs/2.10/fsdp.html)[v2.9.1](https://docs.pytorch.org/docs/2.9/fsdp.html)[v2.8.0](https://docs.pytorch.org/docs/2.8/fsdp.html)[v2.7.0](https://docs.pytorch.org/docs/2.7/fsdp.html)[v2.6.0](https://docs.pytorch.org/docs/2.6/fsdp.html)[v2.5.0](https://docs.pytorch.org/docs/2.5/fsdp.html)[v2.4.0](https://docs.pytorch.org/docs/2.4/fsdp.html)[v2.3.0](https://docs.pytorch.org/docs/2.3/fsdp.html)[v2.2.0](https://docs.pytorch.org/docs/2.2/fsdp.html)[v2.1.0](https://docs.pytorch.org/docs/2.1/fsdp.html)[v2.0.0](https://docs.pytorch.org/docs/2.0/fsdp.html)[v1.13](https://docs.pytorch.org/docs/1.13/fsdp.html)[v1.12](https://docs.pytorch.org/docs/1.12/fsdp.html)[v1.11](https://docs.pytorch.org/docs/1.11/fsdp.html)[v1.10](https://docs.pytorch.org/docs/1.10/fsdp.html)[v1.9.1](https://docs.pytorch.org/docs/1.9.1/fsdp.html)[v1.9.0](https://docs.pytorch.org/docs/1.9.0/fsdp.html)[v1.8.1](https://docs.pytorch.org/docs/1.8.1/fsdp.html)[v1.8.0](https://docs.pytorch.org/docs/1.8.0/fsdp.html)[v1.7.1](https://docs.pytorch.org/docs/1.7.1/fsdp.html)[v1.7.0](https://docs.pytorch.org/docs/1.7.0/fsdp.html)[v1.6.0](https://docs.pytorch.org/docs/1.6.0/fsdp.html)[v1.5.1](https://docs.pytorch.org/docs/1.5.1/fsdp.html)[v1.5.0](https://docs.pytorch.org/docs/1.5.0/fsdp.html)[v1.4.0](https://docs.pytorch.org/docs/1.4.0/fsdp.html)[v1.3.1](https://docs.pytorch.org/docs/1.3.1/fsdp.html)[v1.3.0](https://docs.pytorch.org/docs/1.3.0/fsdp.html)[v1.2.0](https://docs.pytorch.org/docs/1.2.0/fsdp.html)[v1.1.0](https://docs.pytorch.org/docs/1.1.0/fsdp.html)[v1.0.1](https://docs.pytorch.org/docs/1.0.1/fsdp.html)[v1.0.0](https://docs.pytorch.org/docs/1.0.0/fsdp.html)[v0.4.1](https://docs.pytorch.org/docs/0.4.1/fsdp.html)[v0.4.0](https://docs.pytorch.org/docs/0.4.0/fsdp.html)[v0.3.1](https://docs.pytorch.org/docs/0.3.1/fsdp.html)[v0.3.0](https://docs.pytorch.org/docs/0.3.0/fsdp.html)[v0.2.0](https://docs.pytorch.org/docs/0.2.0/fsdp.html)[v0.1.12](https://docs.pytorch.org/docs/0.1.12/fsdp.html)
[ Go to pytorch.org  ](https://pytorch.org)
Custom Search  
|  Sort by Relevance Date  |  
| --- |  
  * [ GitHub](https://github.com/pytorch/pytorch)
  * [ PyTorch Forum](https://discuss.pytorch.org/)


Rate this Page 
★ ★ ★ ★ ★
# FullyShardedDataParallel[#](https://docs.pytorch.org/docs/2.13/fsdp.html#module-torch.distributed.fsdp "Link to this heading")
Created On: Feb 02, 2022 | Last Updated On: May 12, 2026 

_class _torch.distributed.fsdp.FullyShardedDataParallel(_module_ , _process_group_ , _sharding_strategy_ , _cpu_offload_ , _auto_wrap_policy_ , _backward_prefetch BackwardPrefetch.BACKWARD_PRE_, _mixed_precision_ , _ignored_modules_ , _param_init_fn_ , _device_id_ , _sync_module_states False_, _forward_prefetch False_, _limit_all_gathers_ , _use_orig_params False_, _ignored_states_ , _device_mesh_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L118)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "Link to this definition") 
    
A wrapper for sharding module parameters across data parallel workers.
This is inspired by [Xu et al.](https://arxiv.org/abs/2004.13336) as well as the ZeRO Stage 3 from [DeepSpeed](https://www.deepspeed.ai/). FullyShardedDataParallel is commonly shortened to FSDP.
Example:

```
>>> importtorch
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> torch.cuda.set_device(device_id)
>>> sharded_module = FSDP(my_module)
>>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
>>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
>>> loss = x.sum()
>>> loss.backward()
>>> optim.step()

```
Copy to clipboard
Using FSDP involves wrapping your module and then initializing your optimizer after. This is required since FSDP changes the parameter variables.
When setting up FSDP, you need to consider the destination CUDA device. If the device has an ID (`dev_id`), you have three options:
  * Place the module on that device
  * Set the device using `torch.cuda.set_device(dev_id)`
  * Pass `dev_id` into the `device_id` constructor argument.


This ensures that the FSDP instance’s compute device is the destination device. For option 1 and 3, the FSDP initialization always occurs on GPU. For option 2, the FSDP initialization happens on module’s current device, which may be a CPU.
If you’re using the `sync_module_states=True` flag, you need to ensure that the module is on a GPU or use the `device_id` argument to specify a CUDA device that FSDP will move the module to in the FSDP constructor. This is necessary because `sync_module_states=True` requires GPU communication.
FSDP also takes care of moving input tensors to the forward method to the GPU compute device, so you don’t need to manually move them from CPU.
For `use_orig_params=True`, `ShardingStrategy.SHARD_GRAD_OP` exposes the unsharded parameters, not the sharded parameters after forward, unlike `ShardingStrategy.FULL_SHARD`. If you want to inspect the gradients, you can use the `summon_full_params` method with `with_grads=True`.
With `limit_all_gathers=True`, you may see a gap in the FSDP pre-forward where the CPU thread is not issuing any kernels. This is intentional and shows the rate limiter in effect. Synchronizing the CPU thread in that way prevents over-allocating memory for subsequent all-gathers, and it should not actually delay GPU kernel execution.
FSDP replaces managed modules’ parameters with `torch.Tensor` views during forward and backward computation for autograd-related reasons. If your module’s forward relies on saved references to the parameters instead of reacquiring the references each iteration, then it will not see FSDP’s newly created views, and autograd will not work correctly.
Finally, when using `sharding_strategy=ShardingStrategy.HYBRID_SHARD` with the sharding process group being intra-node and the replication process group being inter-node, setting `NCCL_CROSS_NIC=1` can help improve the all-reduce times over the replication process group for some cluster setups.
**Limitations**
There are several limitations to be aware of when using FSDP:
  * FSDP currently does not support gradient accumulation outside `no_sync()` when using CPU offloading. This is because FSDP uses the newly-reduced gradient instead of accumulating with any existing gradient, which can lead to incorrect results.
  * FSDP does not support running the forward pass of a submodule that is contained in an FSDP instance. This is because the submodule’s parameters will be sharded, but the submodule itself is not an FSDP instance, so its forward pass will not all-gather the full parameters appropriately.
  * FSDP does not work with double backwards due to the way it registers backward hooks.
  * FSDP has some constraints when freezing parameters. For `use_orig_params=False`, each FSDP instance must manage parameters that are all frozen or all non-frozen. For `use_orig_params=True`, FSDP supports mixing frozen and non-frozen parameters, but it’s recommended to avoid doing so to prevent higher than expected gradient memory usage.
  * As of PyTorch 1.12, FSDP offers limited support for shared parameters. If enhanced shared parameter support is needed for your use case, please post in [this issue](https://github.com/pytorch/pytorch/issues/77724).
  * You should avoid modifying the parameters between forward and backward without using the `summon_full_params` context, as the modifications may not persist.



Parameters: 
    
  * **module** ([_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – This is the module to be wrapped with FSDP.
  * **process_group** (_Optional_ _[__Union_ _[_[_ProcessGroup_](https://docs.pytorch.org/docs/2.13/distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup") _,__Tuple_ _[_[_ProcessGroup_](https://docs.pytorch.org/docs/2.13/distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup") _,_[_ProcessGroup_](https://docs.pytorch.org/docs/2.13/distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup") _]__]__]_) – This is the process group over which the model is sharded and thus the one used for FSDP’s all-gather and reduce-scatter collective communications. If `None`, then FSDP uses the default process group. For hybrid sharding strategies such as `ShardingStrategy.HYBRID_SHARD`, users can pass in a tuple of process groups, representing the groups over which to shard and replicate, respectively. If `None`, then FSDP constructs process groups for the user to shard intra-node and replicate inter-node. (Default: `None`)
  * **sharding_strategy** (_Optional_ _[_[_ShardingStrategy_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardingStrategy "torch.distributed.fsdp.ShardingStrategy") _]_) – This configures the sharding strategy, which may trade off memory saving and communication overhead. See [`ShardingStrategy`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardingStrategy "torch.distributed.fsdp.ShardingStrategy") for details. (Default: `FULL_SHARD`)
  * **cpu_offload** (_Optional_ _[_[_CPUOffload_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.CPUOffload "torch.distributed.fsdp.CPUOffload") _]_) – This configures CPU offloading. If this is set to `None`, then no CPU offloading happens. See [`CPUOffload`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.CPUOffload "torch.distributed.fsdp.CPUOffload") for details. (Default: `None`)
  * **auto_wrap_policy** (_Optional_ _[__Union_ _[__Callable_ _[__[_[_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _,_[_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,_[_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") _]__,_[_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _]__,__ModuleWrapPolicy_ _,__CustomPolicy_ _]__]_) – 
This specifies a policy to apply FSDP to submodules of `module`, which is needed for communication and computation overlap and thus affects performance. If `None`, then FSDP only applies to `module`, and users should manually apply FSDP to parent modules themselves (proceeding bottom-up). For convenience, this accepts `ModuleWrapPolicy` directly, which allows users to specify the module classes to wrap (e.g. the transformer block). Otherwise, this should be a callable that takes in three arguments `module: nn.Module`, `recurse: bool`, and `nonwrapped_numel: int` and should return a `bool` specifying whether the passed-in `module` should have FSDP applied if `recurse=False` or if the traversal should continue into the module’s subtree if `recurse=True`. Users may add additional arguments to the callable. The `size_based_auto_wrap_policy` in `torch.distributed.fsdp.wrap.py` gives an example callable that applies FSDP to a module if the parameters in its subtree exceed 100M numel. We recommend printing the model after applying FSDP and adjusting as needed.
Example:

```
>>> defcustom_auto_wrap_policy(
>>>     module: nn.Module,
>>>     recurse: bool,
>>>     nonwrapped_numel: int,
>>>     # Additional custom arguments
>>>     min_num_params: int = int(1e8),
>>> ) -> bool:
>>>     return nonwrapped_numel >= min_num_params
>>> # Configure a custom `min_num_params`
>>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

```
Copy to clipboard
  * **backward_prefetch** (_Optional_ _[_[_BackwardPrefetch_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.BackwardPrefetch "torch.distributed.fsdp.BackwardPrefetch") _]_) – This configures explicit backward prefetching of all-gathers. If `None`, then FSDP does not backward prefetch, and there is no communication and computation overlap in the backward pass. See [`BackwardPrefetch`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.BackwardPrefetch "torch.distributed.fsdp.BackwardPrefetch") for details. (Default: `BACKWARD_PRE`)
  * **mixed_precision** (_Optional_ _[_[_MixedPrecision_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.MixedPrecision "torch.distributed.fsdp.MixedPrecision") _]_) – This configures native mixed precision for FSDP. If this is set to `None`, then no mixed precision is used. Otherwise, parameter, buffer, and gradient reduction dtypes can be set. See [`MixedPrecision`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.MixedPrecision "torch.distributed.fsdp.MixedPrecision") for details. (Default: `None`)
  * **ignored_modules** (_Optional_ _[__Iterable_ _[_[_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__]_) – Modules whose own parameters and child modules’ parameters and buffers are ignored by this instance. None of the modules directly in `ignored_modules` should be [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instances, and any child modules that are already-constructed [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instances will not be ignored if they are nested under this instance. This argument may be used to avoid sharding specific parameters at module granularity when using an `auto_wrap_policy` or if parameters’ sharding is not managed by FSDP. (Default: `None`)
  * **param_init_fn** (_Optional_ _[__Callable_ _[__[_[_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__,__None_ _]__]_) – 
A `Callable[torch.nn.Module] -> None` that specifies how modules that are currently on the meta device should be initialized onto an actual device. As of v1.12, FSDP detects modules with parameters or buffers on meta device via `is_meta` and either applies `param_init_fn` if specified or calls `nn.Module.reset_parameters()` otherwise. For both cases, the implementation should _only_ initialize the parameters/buffers of the module, not those of its submodules. This is to avoid re-initialization. In addition, FSDP also supports deferred initialization via torchdistX’s ([pytorch/torchdistX](https://github.com/pytorch/torchdistX)) `deferred_init()` API, where the deferred modules are initialized by calling `param_init_fn` if specified or torchdistX’s default `materialize_module()` otherwise. If `param_init_fn` is specified, then it is applied to all meta-device modules, meaning that it should probably case on the module type. FSDP calls the initialization function before parameter flattening and sharding.
Example:

```
>>> module = MyModule(device="meta")
>>> defmy_init_fn(module: nn.Module):
>>>     # E.g. initialize depending on the module type
>>>     ...
>>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
>>> print(next(fsdp_model.parameters()).device) # current CUDA device
>>> # With torchdistX
>>> module = deferred_init.deferred_init(MyModule, device="cuda")
>>> # Will initialize via deferred_init.materialize_module().
>>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

```
Copy to clipboard
  * **device_id** (_Optional_ _[__Union_ _[_[_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") _,_[_torch.device_](https://docs.pytorch.org/docs/2.13/tensor_attributes.html#torch.device "torch.device") _]__]_) – An `int` or `torch.device` giving the CUDA device on which FSDP initialization takes place, including the module initialization if needed and the parameter sharding. This should be specified to improve initialization speed if `module` is on CPU. If the default CUDA device was set (e.g. via `torch.cuda.set_device`), then the user may pass `torch.cuda.current_device` to this. (Default: `None`)
  * **sync_module_states** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then each FSDP module will broadcast module parameters and buffers from rank 0 to ensure that they are replicated across ranks (adding communication overhead to this constructor). This can help load `state_dict` checkpoints via `load_state_dict` in a memory efficient way. See [`FullStateDictConfig`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullStateDictConfig "torch.distributed.fsdp.FullStateDictConfig") for an example of this. (Default: `False`)
  * **forward_prefetch** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP _explicitly_ prefetches the next forward-pass all-gather before the current forward computation. This is only useful for CPU-bound workloads, in which case issuing the next all-gather earlier may improve overlap. This should only be used for static-graph models since the prefetching follows the first iteration’s execution order. (Default: `False`)
  * **limit_all_gathers** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP explicitly synchronizes the CPU thread to ensure GPU memory usage from only _two_ consecutive FSDP instances (the current instance running computation and the next instance whose all-gather is prefetched). If `False`, then FSDP allows the CPU thread to issue all-gathers without any extra synchronization. (Default: `True`) We often refer to this feature as the “rate limiter”. This flag should only be set to `False` for specific CPU-bound workloads with low memory pressure in which case the CPU thread can aggressively issue all kernels without concern for the GPU memory usage.
  * **use_orig_params** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Setting this to `True` has FSDP use `module` ‘s original parameters. FSDP exposes those original parameters to the user via `nn.Module.named_parameters()` instead of FSDP’s internal `FlatParameter` s. This means that the optimizer step runs on the original parameters, enabling per-original-parameter hyperparameters. FSDP preserves the original parameter variables and manipulates their data between unsharded and sharded forms, where they are always views into the underlying unsharded or sharded `FlatParameter`, respectively. With the current algorithm, the sharded form is always 1D, losing the original tensor structure. An original parameter may have all, some, or none of its data present for a given rank. In the none case, its data will be like a size-0 empty tensor. Users should not author programs relying on what data is present for a given original parameter in its sharded form. `True` is required to use `torch.compile()`. Setting this to `False` exposes FSDP’s internal `FlatParameter` s to the user via `nn.Module.named_parameters()`. (Default: `False`)
  * **ignored_states** (_Optional_ _[__Iterable_ _[__torch.nn.Parameter_ _]__]__,__Optional_ _[__Iterable_ _[_[_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__]_) – Ignored parameters or modules that will not be managed by this FSDP instance, meaning that the parameters are not sharded and their gradients are not reduced across ranks. This argument unifies with the existing `ignored_modules` argument, and we may deprecate `ignored_modules` soon. For backward compatibility, we keep both `ignored_states` and ignored_modules`, but FSDP only allows one of them to be specified as not `None`.
  * **device_mesh** (_Optional_ _[_[_DeviceMesh_](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh") _]_) – DeviceMesh can be used as an alternative to process_group. When device_mesh is passed, FSDP will use the underlying process groups for all-gather and reduce-scatter collective communications. Therefore, these two args need to be mutually exclusive. For hybrid sharding strategies such as `ShardingStrategy.HYBRID_SHARD`, users can pass in a 2D DeviceMesh instead of a tuple of process groups. For 2D FSDP + TP, users are required to pass in device_mesh instead of process_group. For more DeviceMesh info, please visit: <https://pytorch.org/tutorials/recipes/distributed_device_mesh.html>



apply()[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L569)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.apply "Link to this definition") 
    
Apply recursively to every submodule (as returned by `.children()`) as well as self.
Typical use includes initializing the parameters of a model (see also [torch.nn.init](https://docs.pytorch.org/docs/2.13/nn.init.html#nn-init-doc)).
Compared to `torch.nn.Module.apply`, this version additionally gathers the full parameters before applying . It should not be called from within another `summon_full_params` context. 

Parameters: 
    
**fn** (`Module` -> None) – function to be applied to each submodule 

Returns: 
    
self 

Return type: 
     

check_is_root()[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L542)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.check_is_root "Link to this definition") 
    
Check if this instance is a root FSDP module. 

Return type: 


clip_grad_norm_(_max_norm_ , _norm_type_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1073)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_ "Link to this definition") 
    
Clip the gradient norm of all parameters.
The norm is computed over all parameters’ gradients as viewed as a single vector, and the gradients are modified in-place. 

Parameters: 
    
  * **max_norm** ([_float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") _or_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – max norm of the gradients
  * **norm_type** ([_float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") _or_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – type of the used p-norm. Can be `'inf'` for infinity norm.



Returns: 
    
Total norm of the parameters (viewed as a single vector). 

Return type: 
    
If every FSDP instance uses `NO_SHARD`, meaning that no gradients are sharded across ranks, then you may directly use [`torch.nn.utils.clip_grad_norm_()`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_ "torch.nn.utils.clip_grad_norm_").
If at least some FSDP instance uses a sharded strategy (i.e. one other than `NO_SHARD`), then you should use this method instead of [`torch.nn.utils.clip_grad_norm_()`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_ "torch.nn.utils.clip_grad_norm_") since this method handles the fact that gradients are sharded across ranks.
The total norm returned will have the “largest” dtype across all parameters/gradients as defined by PyTorch’s type promotion semantics. For example, if _all_ parameters/gradients use a low precision dtype, then the returned norm’s dtype will be that low precision dtype, but if there exists at least one parameter/ gradient using FP32, then the returned norm’s dtype will be FP32.
Warning
This needs to be called on all ranks since it uses collective communications. 

_static _flatten_sharded_optim_state_dict(_sharded_optim_state_dict_ , _model_ , _optim_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1555)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict "Link to this definition") 
    
Flatten a sharded optimizer state-dict.
The API is similar to [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict"). The only difference is that the input `sharded_optim_state_dict` should be returned from [`sharded_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict"). Therefore, there will be all-gather calls on each rank to gather `ShardedTensor` s. 

Parameters: 
    
  * **sharded_optim_state_dict** (_Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]_) – Optimizer state dict corresponding to the unflattened parameters and holding the sharded optimizer state.
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Refer to [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict").
  * **optim** ([_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer")) – Optimizer for `model` ‘s parameters.



Returns: 
    
Refer to [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict"). 

Return type: 
    
[dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")] 

forward(__,_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L835)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.forward "Link to this definition") 
    
Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic. 

Return type: 


_static _fsdp_modules(_module_ , _root_only False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L546)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules "Link to this definition") 
    
Return all nested FSDP instances.
This possibly includes `module` itself and only includes FSDP root modules if `root_only=True`. 

Parameters: 
    
  * **module** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module, which may or may not be an `FSDP` module.
  * **root_only** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Whether to return only FSDP root modules. (Default: `False`)



Returns: 
    
FSDP modules that are nested in the input `module`. 

Return type: 
    
List[[FullyShardedDataParallel](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel")] 

_static _full_optim_state_dict(_model_ , _optim_ , _optim_input_ , _rank0_only_ , _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1376)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "Link to this definition") 
    
Return the full optimizer state-dict.
Consolidates the full optimizer state on rank 0 and returns it as a following the convention of [`torch.optim.Optimizer.state_dict()`](https://docs.pytorch.org/docs/2.13/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict"), i.e. with keys `"state"` and `"param_groups"`. The flattened parameters in `FSDP` modules contained in `model` are mapped back to their unflattened parameters.
This needs to be called on all ranks since it uses collective communications. However, if `rank0_only=True`, then the state dict is only populated on rank 0, and all other ranks return an empty .
Unlike `torch.optim.Optimizer.state_dict()`, this method uses full parameter names as keys instead of parameter IDs.
Like in [`torch.optim.Optimizer.state_dict()`](https://docs.pytorch.org/docs/2.13/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict"), the tensors contained in the optimizer state dict are not cloned, so there may be aliasing surprises. For best practices, consider saving the returned optimizer state dict immediately, e.g. using `torch.save()`. 

Parameters: 
    
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module (which may or may not be a [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instance) whose parameters were passed into the optimizer `optim`.
  * **optim** ([_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer")) – Optimizer for `model` ‘s parameters.
  * **optim_input** (_Optional_ _[__Union_ _[__List_ _[__Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]__]__,__Iterable_ _[__torch.nn.Parameter_ _]__]__]_) – Input passed into the optimizer `optim` representing either a of parameter groups or an iterable of parameters; if `None`, then this method assumes the input was `model.parameters()`. This argument is deprecated, and there is no need to pass it in anymore. (Default: `None`)
  * **rank0_only** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, saves the populated only on rank 0; if `False`, saves it on all ranks. (Default: `True`)
  * **group** (_dist.ProcessGroup_) – Model’s process group or `None` if using the default process group. (Default: `None`)



Returns: 
    
A containing the optimizer state for `model` ‘s original unflattened parameters and including keys “state” and “param_groups” following the convention of [`torch.optim.Optimizer.state_dict()`](https://docs.pytorch.org/docs/2.13/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict"). If `rank0_only=True`, then nonzero ranks return an empty . 

Return type: 
    
Dict[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), Any] 

_static _get_state_dict_type(_module_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L753)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type "Link to this definition") 
    
Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at `module`.
The target module does not have to be an FSDP module. 

Returns: 
    
A `StateDictSettings` containing the state_dict_type and state_dict / optim_state_dict configs that are currently set. 

Raises: 
    
  * **AssertionError` if the StateDictSettings for differen** – 
  * **FSDP submodules differ.** – 



Return type: 
    
[_StateDictSettings_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictSettings "torch.distributed.fsdp.api.StateDictSettings") 

_property _module[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.module "Link to this definition") 
    
Return the wrapped module. 

named_buffers(__,_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L975)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_buffers "Link to this definition") 
    
Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.
Intercepts buffer names and removes all occurrences of the FSDP-specific flattened buffer prefix when inside the [`summon_full_params()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params") context manager. 

Return type: 
    
[_Iterator_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator "\(in Python v3.14\)")[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [_Tensor_](https://docs.pytorch.org/docs/2.13/tensors.html#torch.Tensor "torch.Tensor")]] 

named_parameters(__,_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L994)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_parameters "Link to this definition") 
    
Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself.
Intercepts parameter names and removes all occurrences of the FSDP-specific flattened parameter prefix when inside the [`summon_full_params()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params") context manager. 

Return type: 
    
[_Iterator_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator "\(in Python v3.14\)")[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [_Parameter_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter "torch.nn.parameter.Parameter")]] 

no_sync()[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1033)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync "Link to this definition") 
    
Disable gradient synchronizations across FSDP instances.
Within this context, gradients will be accumulated in module variables, which will later be synchronized in the first forward-backward pass after exiting the context. This should only be used on the root FSDP instance and will recursively apply to all children FSDP instances.
Note
This likely results in higher memory usage because FSDP will accumulate the full model gradients (instead of gradient shards) until the eventual sync.
Note
When used with CPU offloading, the gradients will not be offloaded to CPU when inside the context manager. Instead, they will only be offloaded right after the eventual sync. 

Return type: 
     

_static _optim_state_dict(_model_ , _optim_ , _optim_state_dict_ , _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1796)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict "Link to this definition") 
    
Transform the state-dict of an optimizer corresponding to a sharded model.
The given state-dict can be transformed to one of three types: 1) full optimizer state_dict, 2) sharded optimizer state_dict, 3) local optimizer state_dict.
For full optimizer state_dict, all states are unflattened and not sharded. Rank0 only and CPU only can be specified via [`state_dict_type()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type "torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type") to avoid OOM.
For sharded optimizer state_dict, all states are unflattened but sharded. CPU only can be specified via [`state_dict_type()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type "torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type") to further save memory.
For local state_dict, no transformation will be performed. But a state will be converted from nn.Tensor to ShardedTensor to represent its sharding nature (this is not supported yet).
Example:

```
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> fromtorch.distributed.fsdpimport StateDictType
>>> fromtorch.distributed.fsdpimport FullStateDictConfig
>>> fromtorch.distributed.fsdpimport FullOptimStateDictConfig
>>> # Save a checkpoint
>>> model, optim = ...
>>> FSDP.set_state_dict_type(
>>>     model,
>>>     StateDictType.FULL_STATE_DICT,
>>>     FullStateDictConfig(rank0_only=False),
>>>     FullOptimStateDictConfig(rank0_only=False),
>>> )
>>> state_dict = model.state_dict()
>>> optim_state_dict = FSDP.optim_state_dict(model, optim)
>>> save_a_checkpoint(state_dict, optim_state_dict)
>>> # Load a checkpoint
>>> model, optim = ...
>>> state_dict, optim_state_dict = load_a_checkpoint()
>>> FSDP.set_state_dict_type(
>>>     model,
>>>     StateDictType.FULL_STATE_DICT,
>>>     FullStateDictConfig(rank0_only=False),
>>>     FullOptimStateDictConfig(rank0_only=False),
>>> )
>>> model.load_state_dict(state_dict)
>>> optim_state_dict = FSDP.optim_state_dict_to_load(
>>>     model, optim, optim_state_dict
>>> )
>>> optim.load_state_dict(optim_state_dict)

```
Copy to clipboard 

Parameters: 
    
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module (which may or may not be a [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instance) whose parameters were passed into the optimizer `optim`.
  * **optim** ([_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer")) – Optimizer for `model` ‘s parameters.
  * **optim_state_dict** (_Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]_) – the target optimizer state_dict to transform. If the value is None, optim.state_dict() will be used. ( Default: `None`)
  * **group** (_dist.ProcessGroup_) – Model’s process group across which parameters are sharded or `None` if using the default process group. ( Default: `None`)



Returns: 
    
A containing the optimizer state for `model`. The sharding of the optimizer state is based on `state_dict_type`. 

Return type: 
    
Dict[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), Any] 

_static _optim_state_dict_to_load(_model_ , _optim_ , _optim_state_dict_ , _is_named_optimizer False_, _load_directly False_, _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1892)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load "Link to this definition") 
    
Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.
Given a `optim_state_dict` that is transformed through [`optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict"), it gets converted to the flattened optimizer state_dict that can be loaded to `optim` which is the optimizer for `model`. `model` must be sharded by FullyShardedDataParallel.

```
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> fromtorch.distributed.fsdpimport StateDictType
>>> fromtorch.distributed.fsdpimport FullStateDictConfig
>>> fromtorch.distributed.fsdpimport FullOptimStateDictConfig
>>> # Save a checkpoint
>>> model, optim = ...
>>> FSDP.set_state_dict_type(
>>>     model,
>>>     StateDictType.FULL_STATE_DICT,
>>>     FullStateDictConfig(rank0_only=False),
>>>     FullOptimStateDictConfig(rank0_only=False),
>>> )
>>> state_dict = model.state_dict()
>>> original_osd = optim.state_dict()
>>> optim_state_dict = FSDP.optim_state_dict(
>>>     model,
>>>     optim,
>>>     optim_state_dict=original_osd
>>> )
>>> save_a_checkpoint(state_dict, optim_state_dict)
>>> # Load a checkpoint
>>> model, optim = ...
>>> state_dict, optim_state_dict = load_a_checkpoint()
>>> FSDP.set_state_dict_type(
>>>     model,
>>>     StateDictType.FULL_STATE_DICT,
>>>     FullStateDictConfig(rank0_only=False),
>>>     FullOptimStateDictConfig(rank0_only=False),
>>> )
>>> model.load_state_dict(state_dict)
>>> optim_state_dict = FSDP.optim_state_dict_to_load(
>>>     model, optim, optim_state_dict
>>> )
>>> optim.load_state_dict(optim_state_dict)

```
Copy to clipboard 

Parameters: 
    
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module (which may or may not be a [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instance) whose parameters were passed into the optimizer `optim`.
  * **optim** ([_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer")) – Optimizer for `model` ‘s parameters.
  * **optim_state_dict** (_Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]_) – The optimizer states to be loaded.
  * **is_named_optimizer** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – Is this optimizer a NamedOptimizer or KeyedOptimizer. Only set to True if `optim` is TorchRec’s KeyedOptimizer or torch.distributed’s NamedOptimizer.
  * **load_directly** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If this is set to True, this API will also call optim.load_state_dict(result) before returning the result. Otherwise, users are responsible to call `optim.load_state_dict()` (Default: `False`)
  * **group** (_dist.ProcessGroup_) – Model’s process group across which parameters are sharded or `None` if using the default process group. ( Default: `None`)



Return type: 
    
[dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")] 

register_comm_hook(_state_ , __)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1982)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook "Link to this definition") 
    
Register a communication hook.
This is an enhancement that provides a flexible hook to users where they can specify how FSDP aggregates gradients across multiple workers. This hook can be used to implement several algorithms like [GossipGrad](https://arxiv.org/abs/1803.05880) and gradient compression which involve different communication strategies for parameter syncs while training with [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel").
Warning
FSDP communication hook should be registered before running an initial forward pass and only once. 

Parameters: 
    
  * **state** ([_object_](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")) – 
Passed to the hook to maintain any state information during the training process. Examples include error feedback in gradient compression, peers to communicate with next in [GossipGrad](https://arxiv.org/abs/1803.05880), etc. It is locally stored by each worker and shared by all the gradient tensors on the worker.
  * **hook** (_Callable_) – Callable, which has one of the following signatures: 1) `hook: Callable[torch.Tensor] -> None`: This function takes in a Python tensor, which represents the full, flattened, unsharded gradient with respect to all variables corresponding to the model this FSDP unit is wrapping (that are not wrapped by other FSDP sub-units). It then performs all necessary processing and returns `None`; 2) `hook: Callable[torch.Tensor, torch.Tensor] -> None`: This function takes in two Python tensors, the first one represents the full, flattened, unsharded gradient with respect to all variables corresponding to the model this FSDP unit is wrapping (that are not wrapped by other FSDP sub-units). The latter represents a pre-sized tensor to store a chunk of a sharded gradient after reduction. In both cases, callable performs all necessary processing and returns `None`. Callables with signature 1 are expected to handle gradient communication for a NO_SHARD case. Callables with signature 2 are expected to handle gradient communication for sharded cases.



_static _rekey_optim_state_dict(_optim_state_dict_ , _optim_state_key_type_ , _model_ , _optim_input_ , _optim_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1673)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict "Link to this definition") 
    
Re-keys the optimizer state dict `optim_state_dict` to use the key type `optim_state_key_type`.
This can be used to achieve compatibility between optimizer state dicts from models with FSDP instances and ones without.
To re-key an FSDP full optimizer state dict (i.e. from [`full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict")) to use parameter IDs and be loadable to a non-wrapped model:

```
>>> wrapped_model, wrapped_optim = ...
>>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
>>> nonwrapped_model, nonwrapped_optim = ...
>>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
>>> nonwrapped_optim.load_state_dict(rekeyed_osd)

```
Copy to clipboard
To re-key a normal optimizer state dict from a non-wrapped model to be loadable to a wrapped model:

```
>>> nonwrapped_model, nonwrapped_optim = ...
>>> osd = nonwrapped_optim.state_dict()
>>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
>>> wrapped_model, wrapped_optim = ...
>>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
>>> wrapped_optim.load_state_dict(sharded_osd)

```
Copy to clipboard 

Returns: 
    
The optimizer state dict re-keyed using the parameter keys specified by `optim_state_key_type`. 

Return type: 
    
Dict[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), Any] 

_static _scatter_full_optim_state_dict(_full_optim_state_dict_ , _model_ , _optim_input_ , _optim_ , _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1594)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict "Link to this definition") 
    
Scatter the full optimizer state dict from rank 0 to all other ranks.
Returns the sharded optimizer state dict on each rank. The return value is the same as [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict"), and on rank 0, the first argument should be the return value of [`full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict").
Example:

```
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> model, optim = ...
>>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
>>> # Define new model with possibly different world size
>>> new_model, new_optim, new_group = ...
>>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
>>> new_optim.load_state_dict(sharded_osd)

```
Copy to clipboard
Note
Both [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict") and [`scatter_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict") may be used to get the sharded optimizer state dict to load. Assuming that the full optimizer state dict resides in CPU memory, the former requires each rank to have the full dict in CPU memory, where each rank individually shards the dict without any communication, while the latter requires only rank 0 to have the full dict in CPU memory, where rank 0 moves each shard to GPU memory (for NCCL) and communicates it to ranks appropriately. Hence, the former has higher aggregate CPU memory cost, while the latter has higher communication cost. 

Parameters: 
    
  * **full_optim_state_dict** (_Optional_ _[__Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]__]_) – Optimizer state dict corresponding to the unflattened parameters and holding the full non-sharded optimizer state if on rank 0; the argument is ignored on nonzero ranks.
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module (which may or may not be a [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instance) whose parameters correspond to the optimizer state in `full_optim_state_dict`.
  * **optim_input** (_Optional_ _[__Union_ _[__List_ _[__Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]__]__,__Iterable_ _[__torch.nn.Parameter_ _]__]__]_) – Input passed into the optimizer representing either a of parameter groups or an iterable of parameters; if `None`, then this method assumes the input was `model.parameters()`. This argument is deprecated, and there is no need to pass it in anymore. (Default: `None`)
  * **optim** (_Optional_ _[_[_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer") _]_) – Optimizer that will load the state dict returned by this method. This is the preferred argument to use over `optim_input`. (Default: `None`)
  * **group** (_dist.ProcessGroup_) – Model’s process group or `None` if using the default process group. (Default: `None`)



Returns: 
    
The full optimizer state dict now remapped to flattened parameters instead of unflattened parameters and restricted to only include this rank’s part of the optimizer state. 

Return type: 
    
Dict[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), Any] 

_static _set_state_dict_type(_module_ , _state_dict_type_ , _state_dict_config_ , _optim_state_dict_config_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L626)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type "Link to this definition") 
    
Set the `state_dict_type` of all the descendant FSDP modules of the target module.
Also takes (optional) configuration for the model’s and optimizer’s state dict. The target module does not have to be a FSDP module. If the target module is a FSDP module, its `state_dict_type` will also be changed.
Note
This API should be called for only the top-level (root) module.
Note
This API enables users to transparently use the conventional `state_dict` API to take model checkpoints in cases where the root FSDP module is wrapped by another `nn.Module`. For example, the following will ensure `state_dict` is called on all non-FSDP instances, while dispatching into sharded_state_dict implementation for FSDP:
Example:

```
>>> model = DDP(FSDP(...))
>>> FSDP.set_state_dict_type(
>>>     model,
>>>     StateDictType.SHARDED_STATE_DICT,
>>>     state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
>>>     optim_state_dict_config = OptimStateDictConfig(offload_to_cpu=True),
>>> )
>>> param_state_dict = model.state_dict()
>>> optim_state_dict = FSDP.optim_state_dict(model, optim)

```
Copy to clipboard 

Parameters: 
    
  * **module** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module.
  * **state_dict_type** (_StateDictType_) – the desired `state_dict_type` to set.
  * **state_dict_config** (_Optional_ _[_[_StateDictConfig_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictConfig "torch.distributed.fsdp.StateDictConfig") _]_) – the configuration for the target `state_dict_type`.
  * **optim_state_dict_config** (_Optional_ _[_[_OptimStateDictConfig_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig "torch.distributed.fsdp.OptimStateDictConfig") _]_) – the configuration for the optimizer state dict.



Returns: 
    
A StateDictSettings that include the previous state_dict type and configuration for the module. 

Return type: 
    
[_StateDictSettings_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictSettings "torch.distributed.fsdp.api.StateDictSettings") 

_static _shard_full_optim_state_dict(_full_optim_state_dict_ , _model_ , _optim_input_ , _optim_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1481)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "Link to this definition") 
    
Shard a full optimizer state-dict.
Remaps the state in `full_optim_state_dict` to flattened parameters instead of unflattened parameters and restricts to only this rank’s part of the optimizer state. The first argument should be the return value of [`full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict").
Example:

```
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> model, optim = ...
>>> full_osd = FSDP.full_optim_state_dict(model, optim)
>>> torch.save(full_osd, PATH)
>>> # Define new model with possibly different world size
>>> new_model, new_optim = ...
>>> full_osd = torch.load(PATH)
>>> sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, new_model)
>>> new_optim.load_state_dict(sharded_osd)

```
Copy to clipboard
Note
Both [`shard_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict") and [`scatter_full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict") may be used to get the sharded optimizer state dict to load. Assuming that the full optimizer state dict resides in CPU memory, the former requires each rank to have the full dict in CPU memory, where each rank individually shards the dict without any communication, while the latter requires only rank 0 to have the full dict in CPU memory, where rank 0 moves each shard to GPU memory (for NCCL) and communicates it to ranks appropriately. Hence, the former has higher aggregate CPU memory cost, while the latter has higher communication cost. 

Parameters: 
    
  * **full_optim_state_dict** (_Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]_) – Optimizer state dict corresponding to the unflattened parameters and holding the full non-sharded optimizer state.
  * **model** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module (which may or may not be a [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel") instance) whose parameters correspond to the optimizer state in `full_optim_state_dict`.
  * **optim_input** (_Optional_ _[__Union_ _[__List_ _[__Dict_ _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]__]__,__Iterable_ _[__torch.nn.Parameter_ _]__]__]_) – Input passed into the optimizer representing either a of parameter groups or an iterable of parameters; if `None`, then this method assumes the input was `model.parameters()`. This argument is deprecated, and there is no need to pass it in anymore. (Default: `None`)
  * **optim** (_Optional_ _[_[_torch.optim.Optimizer_](https://docs.pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer "torch.optim.Optimizer") _]_) – Optimizer that will load the state dict returned by this method. This is the preferred argument to use over `optim_input`. (Default: `None`)



Returns: 
    
The full optimizer state dict now remapped to flattened parameters instead of unflattened parameters and restricted to only include this rank’s part of the optimizer state. 

Return type: 
    
Dict[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), Any] 

_static _sharded_optim_state_dict(_model_ , _optim_ , _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1447)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict "Link to this definition") 
    
Return the optimizer state-dict in its sharded form.
The API is similar to [`full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict") but this API chunks all non-zero-dimension states to `ShardedTensor` to save memory. This API should only be used when the model `state_dict` is derived with the context manager `with state_dict_type(SHARDED_STATE_DICT):`.
For the detailed usage, refer to [`full_optim_state_dict()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict "torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict").
Warning
The returned state dict contains `ShardedTensor` and cannot be directly used by the regular `optim.load_state_dict`. 

Return type: 
    
[dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")] 

_static _state_dict_type(_module_ , _state_dict_type_ , _state_dict_config_ , _optim_state_dict_config_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L790)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type "Link to this definition") 
    
Set the `state_dict_type` of all the descendant FSDP modules of the target module.
This context manager has the same functions as [`set_state_dict_type()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type "torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type"). Read the document of [`set_state_dict_type()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type "torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type") for the detail.
Example:

```
>>> model = DDP(FSDP(...))
>>> with FSDP.state_dict_type(
>>>     model,
>>>     StateDictType.SHARDED_STATE_DICT,
>>> ):
>>>     checkpoint = model.state_dict()

```
Copy to clipboard 

Parameters: 
    
  * **module** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Root module.
  * **state_dict_type** (_StateDictType_) – the desired `state_dict_type` to set.
  * **state_dict_config** (_Optional_ _[_[_StateDictConfig_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictConfig "torch.distributed.fsdp.StateDictConfig") _]_) – the model `state_dict` configuration for the target `state_dict_type`.
  * **optim_state_dict_config** (_Optional_ _[_[_OptimStateDictConfig_](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig "torch.distributed.fsdp.OptimStateDictConfig") _]_) – the optimizer `state_dict` configuration for the target `state_dict_type`.



Return type: 
     

_static _summon_full_params(_module_ , _recurse_ , _writeback_ , _rank0_only False_, _offload_to_cpu False_, _with_grads False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/fully_sharded_data_parallel.py#L862)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params "Link to this definition") 
    
Expose full params for FSDP instances with this context manager.
Can be useful _after_ forward/backward for a model to get the params for additional processing or checking. It can take a non-FSDP module and will summon full params for all contained FSDP modules as well as their children, depending on the `recurse` argument.
Note
This can be used on inner FSDPs.
Note
This can _not_ be used within a forward or backward pass. Nor can forward and backward be started from within this context.
Note
Parameters will revert to their local shards after the context manager exits, storage behavior is the same as forward.
Note
The full parameters can be modified, but only the portion corresponding to the local param shard will persist after the context manager exits (unless `writeback=False`, in which case changes will be discarded). In the case where FSDP does not shard the parameters, currently only when `world_size == 1`, or `NO_SHARD` config, the modification is persisted regardless of `writeback`.
Note
This method works on modules which are not FSDP themselves but may contain multiple independent FSDP units. In that case, the given arguments will apply to all contained FSDP units.
Warning
Note that `rank0_only=True` in conjunction with `writeback=True` is not currently supported and will raise an error. This is because model parameter shapes would be different across ranks within the context, and writing to them can lead to inconsistency across ranks when the context is exited.
Warning
Note that `offload_to_cpu` and `rank0_only=False` will result in full parameters being redundantly copied to CPU memory for GPUs that reside on the same machine, which may incur the risk of CPU OOM. It is recommended to use `offload_to_cpu` with `rank0_only=True`. 

Parameters: 
    
  * **recurse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__Optional_) – recursively summon all params for nested FSDP instances (default: True).
  * **writeback** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__Optional_) – if `False`, modifications to params are discarded after the context manager exits; disabling this can be slightly more efficient (default: True)
  * **rank0_only** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__Optional_) – if `True`, full parameters are materialized on only global rank 0. This means that within the context, only rank 0 will have full parameters and the other ranks will have sharded parameters. Note that setting `rank0_only=True` with `writeback=True` is not supported, as model parameter shapes will be different across ranks within the context, and writing to them can lead to inconsistency across ranks when the context is exited.
  * **offload_to_cpu** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__Optional_) – If `True`, full parameters are offloaded to CPU. Note that this offloading currently only occurs if the parameter is sharded (which is only not the case for world_size = 1 or `NO_SHARD` config). It is recommended to use `offload_to_cpu` with `rank0_only=True` to avoid redundant copies of model parameters being offloaded to the same CPU memory.
  * **with_grads** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__Optional_) – If `True`, gradients are also unsharded with the parameters. Currently, this is only supported when passing `use_orig_params=True` to the FSDP constructor and `offload_to_cpu=False` to this method. (Default: `False`)



Return type: 
     

_class _torch.distributed.fsdp.BackwardPrefetch(_value_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L72)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.BackwardPrefetch "Link to this definition") 
    
This configures explicit backward prefetching, which improves throughput by enabling communication and computation overlap in the backward pass at the cost of slightly increased memory usage.
  * `BACKWARD_PRE`: This enables the most overlap but increases memory usage the most. This prefetches the next set of parameters _before_ the current set of parameters’ gradient computation. This overlaps the _next all-gather_ and the _current gradient computation_ , and at the peak, it holds the current set of parameters, next set of parameters, and current set of gradients in memory.
  * `BACKWARD_POST`: This enables less overlap but requires less memory usage. This prefetches the next set of parameters _after_ the current set of parameters’ gradient computation. This overlaps the _current reduce-scatter_ and the _next gradient computation_ , and it frees the current set of parameters before allocating memory for the next set of parameters, only holding the next set of parameters and current set of gradients in memory at the peak.
  * FSDP’s `backward_prefetch` argument accepts `None`, which disables the backward prefetching altogether. This has no overlap and does not increase memory usage. In general, we do not recommend this setting since it may degrade throughput significantly.


For more technical context: For a single process group using NCCL backend, any collectives, even if issued from different streams, contend for the same per-device NCCL stream, which implies that the relative order in which the collectives are issued matters for overlapping. The two backward prefetching values correspond to different issue orders. 

_class _torch.distributed.fsdp.ShardingStrategy(_value_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L32)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardingStrategy "Link to this definition") 
    
This specifies the sharding strategy to be used for distributed training by [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel "torch.distributed.fsdp.FullyShardedDataParallel").
  * `FULL_SHARD`: Parameters, gradients, and optimizer states are sharded. For the parameters, this strategy unshards (via all-gather) before the forward, reshards after the forward, unshards before the backward computation, and reshards after the backward computation. For gradients, it synchronizes and shards them (via reduce-scatter) after the backward computation. The sharded optimizer states are updated locally per rank.
  * `SHARD_GRAD_OP`: Gradients and optimizer states are sharded during computation, and additionally, parameters are sharded outside computation. For the parameters, this strategy unshards before the forward, does not reshard them after the forward, and only reshards them after the backward computation. The sharded optimizer states are updated locally per rank. Inside `no_sync()`, the parameters are not resharded after the backward computation.
  * `NO_SHARD`: Parameters, gradients, and optimizer states are not sharded but instead replicated across ranks similar to PyTorch’s `DistributedDataParallel` API. For gradients, this strategy synchronizes them (via all-reduce) after the backward computation. The unsharded optimizer states are updated locally per rank.
  * `HYBRID_SHARD`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes. This results in reduced communication volume as expensive all-gathers and reduce-scatters are only done within a node, which can be more performant for medium -sized models.
  * `_HYBRID_SHARD_ZERO2`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes. This is like `HYBRID_SHARD`, except this may provide even higher throughput since the unsharded parameters are not freed after the forward pass, saving the all-gathers in the pre-backward.



_class _torch.distributed.fsdp.MixedPrecision(_param_dtype=None_ , _reduce_dtype=None_ , _buffer_dtype=None_ , _keep_low_precision_grads=False_ , _cast_forward_inputs=False_ , _cast_root_forward_inputs=True_ , __module_classes_to_ignore=( <class 'torch.nn.modules.batchnorm._BatchNorm'>_, )[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L111)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.MixedPrecision "Link to this definition") 
    
This configures FSDP-native mixed precision training. 

Variables: 
    
  * **param_dtype** (_Optional_ _[_[_torch.dtype_](https://docs.pytorch.org/docs/2.13/tensor_attributes.html#torch.dtype "torch.dtype") _]_) – This specifies the dtype for model parameters during forward and backward and thus the dtype for forward and backward computation. Outside forward and backward, the _sharded_ parameters are kept in full precision (e.g. for the optimizer step), and for model checkpointing, the parameters are always saved in full precision. (Default: `None`)
  * **reduce_dtype** (_Optional_ _[_[_torch.dtype_](https://docs.pytorch.org/docs/2.13/tensor_attributes.html#torch.dtype "torch.dtype") _]_) – This specifies the dtype for gradient reduction (i.e. reduce-scatter or all-reduce). If this is `None` but `param_dtype` is not `None`, then this takes on the `param_dtype` value, still running gradient reduction in low precision. This is permitted to differ from `param_dtype`, e.g. to force gradient reduction to run in full precision. (Default: `None`)
  * **buffer_dtype** (_Optional_ _[_[_torch.dtype_](https://docs.pytorch.org/docs/2.13/tensor_attributes.html#torch.dtype "torch.dtype") _]_) – This specifies the dtype for buffers. FSDP does not shard buffers. Rather, FSDP casts them to `buffer_dtype` in the first forward pass and keeps them in that dtype thereafter. For model checkpointing, the buffers are saved in full precision except for `LOCAL_STATE_DICT`. (Default: `None`)
  * **keep_low_precision_grads** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `False`, then FSDP upcasts gradients to full precision after the backward pass in preparation for the optimizer step. If `True`, then FSDP keeps the gradients in the dtype used for gradient reduction, which can save memory if using a custom optimizer that supports running in low precision. (Default: `False`)
  * **cast_forward_inputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then this FSDP module casts its forward args and kwargs to `param_dtype`. This is to ensure that parameter and input dtypes match for forward computation, as required by many ops. This may need to be set to `True` when only applying mixed precision to some but not all FSDP modules, in which case a mixed-precision FSDP submodule needs to recast its inputs. (Default: `False`)
  * **cast_root_forward_inputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then the root FSDP module casts its forward args and kwargs to `param_dtype`, overriding the value of `cast_forward_inputs`. For non-root FSDP modules, this does not do anything. (Default: `True`)
  * **_module_classes_to_ignore** ([_collections.abc.Sequence_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "\(in Python v3.14\)") _[_[_type_](https://docs.python.org/3/library/functions.html#type "\(in Python v3.14\)") _[_[_torch.nn.modules.module.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module") _]__]_) – (Sequence[Type[nn.Module]]): This specifies module classes to ignore for mixed precision when using an `auto_wrap_policy`: Modules of these classes will have FSDP applied to them separately with mixed precision disabled (meaning that the final FSDP construction would deviate from the specified policy). If `auto_wrap_policy` is not specified, then this does not do anything. This API is experimental and subject to change. (Default: `(_BatchNorm,)`)


Note
This API is experimental and subject to change.
Note
Only floating point tensors are cast to their specified dtypes.
Note
In `summon_full_params`, parameters are forced to full precision, but buffers are not.
Note
Layer norm and batch norm accumulate in `float32` even when their inputs are in a low precision like `float16` or `bfloat16`. Disabling FSDP’s mixed precision for those norm modules only means that the affine parameters are kept in `float32`. However, this incurs separate all-gathers and reduce-scatters for those norm modules, which may be inefficient, so if the workload permits, the user should prefer to still apply mixed precision to those modules.
Note
By default, if the user passes a model with any `_BatchNorm` modules and specifies an `auto_wrap_policy`, then the batch norm modules will have FSDP applied to them separately with mixed precision disabled. See the `_module_classes_to_ignore` argument.
Note
`MixedPrecision` has `cast_root_forward_inputs=True` and `cast_forward_inputs=False` by default. For the root FSDP instance, its `cast_root_forward_inputs` takes precedence over its `cast_forward_inputs`. For non-root FSDP instances, their `cast_root_forward_inputs` values are ignored. The default setting is sufficient for the typical case where each FSDP instance has the same `MixedPrecision` configuration and only needs to cast inputs to the `param_dtype` at the beginning of the model’s forward pass.
Note
For nested FSDP instances with different `MixedPrecision` configurations, we recommend setting individual `cast_forward_inputs` values to configure casting inputs or not before each instance’s forward. In such a case, since the casts happen before each FSDP instance’s forward, a parent FSDP instance should have its non-FSDP submodules run before its FSDP submodules to avoid the activation dtype being changed due to a different `MixedPrecision` configuration.
Example:

```
>>> model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
>>> model[1] = FSDP(
>>>     model[1],
>>>     mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
>>> )
>>> model = FSDP(
>>>     model,
>>>     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
>>> )

```
Copy to clipboard
The above shows a working example. On the other hand, if `model[1]` were replaced with `model[0]`, meaning that the submodule using different `MixedPrecision` ran its forward first, then `model[1]` would incorrectly see `float16` activations instead of `bfloat16` ones. 

_class _torch.distributed.fsdp.CPUOffload(_offload_params False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L229)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.CPUOffload "Link to this definition") 
    
This configures CPU offloading. 

Variables: 
    
**offload_params** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – This specifies whether to offload parameters to CPU when not involved in computation. If `True`, then this offloads gradients to CPU as well, meaning that the optimizer step runs on CPU. 

_class _torch.distributed.fsdp.StateDictConfig(_offload_to_cpu False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L275)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictConfig "Link to this definition") 
    
`StateDictConfig` is the base class for all `state_dict` configuration classes. Users should instantiate a child class (e.g. `FullStateDictConfig`) in order to configure settings for the corresponding `state_dict` type supported by FSDP. 

Variables: 
    
**offload_to_cpu** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP offloads the state dict values to CPU, and if `False`, then FSDP keeps them on GPU. (Default: `False`) 

_class _torch.distributed.fsdp.FullStateDictConfig(_offload_to_cpu False_, _rank0_only False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L292)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullStateDictConfig "Link to this definition") 
    
`FullStateDictConfig` is a config class meant to be used with `StateDictType.FULL_STATE_DICT`. We recommend enabling both `offload_to_cpu=True` and `rank0_only=True` when saving full state dicts to save GPU memory and CPU memory, respectively. This config class is meant to be used via the `state_dict_type()` context manager as follows:

```
>>> fromtorch.distributed.fsdpimport FullyShardedDataParallel as FSDP
>>> fsdp = FSDP(model, auto_wrap_policy=...)
>>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
>>> with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
>>>     state = fsdp.state_dict()
>>> # `state` will be empty on non rank 0 and contain CPU tensors on rank 0.
>>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
>>> model = model_fn()  # Initialize model in preparation for wrapping with FSDP
>>> if dist.get_rank() == 0:
>>> # Load checkpoint only on rank 0 to avoid memory redundancy
>>>     state_dict = torch.load("my_checkpoint.pt")
>>>     model.load_state_dict(state_dict)
>>> # All ranks initialize FSDP module as usual. `sync_module_states` argument
>>> # communicates loaded checkpoint states from rank 0 to rest of the world.
>>> fsdp = FSDP(
...     model,
...     device_id=torch.cuda.current_device(),
...     auto_wrap_policy=...,
...     sync_module_states=True,
... )
>>> # After this point, all ranks have FSDP model with loaded checkpoint.

```
Copy to clipboard 

Variables: 
    
**rank0_only** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then only rank 0 saves the full state dict, and nonzero ranks save an empty dict. If `False`, then all ranks save the full state dict. (Default: `False`) 

_class _torch.distributed.fsdp.ShardedStateDictConfig(_offload_to_cpu False_, __use_dtensor False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L339)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig "Link to this definition") 
    
`ShardedStateDictConfig` is a config class meant to be used with `StateDictType.SHARDED_STATE_DICT`. 

Variables: 
    
**_use_dtensor** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP saves the state dict values as `DTensor`, and if `False`, then FSDP saves them as `ShardedTensor`. (Default: `False`)
Warning
`_use_dtensor` is a private field of [`ShardedStateDictConfig`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig "torch.distributed.fsdp.ShardedStateDictConfig") and it is used by FSDP to determine the type of state dict values. Users should not manually modify `_use_dtensor`. 

_class _torch.distributed.fsdp.LocalStateDictConfig(_offload_to_cpu False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L334)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig "Link to this definition") 


_class _torch.distributed.fsdp.OptimStateDictConfig(_offload_to_cpu_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L358)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig "Link to this definition") 
    
`OptimStateDictConfig` is the base class for all `optim_state_dict` configuration classes. Users should instantiate a child class (e.g. `FullOptimStateDictConfig`) in order to configure settings for the corresponding `optim_state_dict` type supported by FSDP. 

Variables: 
    
**offload_to_cpu** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP offloads the state dict’s tensor values to CPU, and if `False`, then FSDP keeps them on the original device (which is GPU unless parameter CPU offloading is enabled). (Default: `True`) 

_class _torch.distributed.fsdp.FullOptimStateDictConfig(_offload_to_cpu_ , _rank0_only False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L376)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig "Link to this definition") 
     

Variables: 
    
**rank0_only** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then only rank 0 saves the full state dict, and nonzero ranks save an empty dict. If `False`, then all ranks save the full state dict. (Default: `False`) 

_class _torch.distributed.fsdp.ShardedOptimStateDictConfig(_offload_to_cpu_ , __use_dtensor False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L393)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig "Link to this definition") 
    
`ShardedOptimStateDictConfig` is a config class meant to be used with `StateDictType.SHARDED_STATE_DICT`. 

Variables: 
    
**_use_dtensor** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `True`, then FSDP saves the state dict values as `DTensor`, and if `False`, then FSDP saves them as `ShardedTensor`. (Default: `False`)
Warning
`_use_dtensor` is a private field of [`ShardedOptimStateDictConfig`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig "torch.distributed.fsdp.ShardedOptimStateDictConfig") and it is used by FSDP to determine the type of state dict values. Users should not manually modify `_use_dtensor`. 

_class _torch.distributed.fsdp.LocalOptimStateDictConfig(_offload_to_cpu False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L388)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig "Link to this definition") 


_class _torch.distributed.fsdp.StateDictSettings(_state_dict_type torch.distributed.fsdp.api.StateDictType_, _state_dict_config [torch.distributed.fsdp.api.StateDictConfig](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictConfig "torch.distributed.fsdp.api.StateDictConfig")_, _optim_state_dict_config [torch.distributed.fsdp.api.OptimStateDictConfig](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig "torch.distributed.fsdp.api.OptimStateDictConfig")_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/api.py#L412)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.StateDictSettings "Link to this definition") 


torch.distributed.fsdp.wrap.always_wrap_policy(__,_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L117)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.always_wrap_policy "Link to this definition") 
    
A simple recursive wrap policy that always returns `True`. This means that every submodule is wrapped by the wrapper class in `_recursive_wrap()`. 

Return type: 


torch.distributed.fsdp.wrap.enable_wrap(, _wrapper_cls_ , _wrapper_kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L400)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.enable_wrap "Link to this definition") 
    
Context manager to wrap modules using a wrapper.
Useful for when you’d like to apply the same configuration arguments to all child modules that you wrap. A particularly important use case is wrapping large layers so that they get sharded (in-place) during initialization, to avoid running out of system memory. Large layers can indicate that they should be sharded via the `wrap` annotation and this context manager can provide the exact configuration for these nested instances.
Usage:

```
with enable_wrap(wrapper_cls, **params):
    # Wraps layer in FSDP by default if within context
    self.l1 = wrap(torch.nn.Linear(5, 5))

```
Copy to clipboard 

Parameters: 
    
  * **wrapper_cls** ([_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")) – Class that wrap annotation will wrap modules with, such as FullyShardedDataParallel.
  * ****wrapper_kwargs** ([_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")) – Configuration settings that will be passed to all `wrap` instances inside the context



Return type: 
    
[_Generator_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator "\(in Python v3.14\)")[None, None, None] 

torch.distributed.fsdp.wrap.lambda_auto_wrap_policy(_module_ , _recurse_ , _nonwrapped_numel_ , _lambda_fn_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L270)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.lambda_auto_wrap_policy "Link to this definition") 
    
A convenient auto wrap policy to wrap submodules based on an arbitrary user function. If lambda_fn(submodule) == True`, the submodule will be wrapped as a wrapper_cls unit.
Return if a module should be wrapped during auto wrapping.
The first three parameters are required by `_recursive_wrap()`. 

Parameters: 
    
  * **module** ([_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Current module being considered.
  * **recurse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `False`, then this function must decide whether `module` should be wrapped as an FSDP instance or not. If `True`, then the function is still recursing down the module tree as a part of the DFS.
  * **nonwrapped_numel** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Parameter numel not yet wrapped.
  * **lambda_fn** (_Callable_ _[__[_[_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__,_[_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _]_) – If this returns `True`, then this module will be wrapped.



Return type: 


torch.distributed.fsdp.wrap.size_based_auto_wrap_policy(_module_ , _recurse_ , _nonwrapped_numel_ , _min_num_params 100000000_, _force_leaf_modules_ , _exclude_wrap_modules_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L341)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.size_based_auto_wrap_policy "Link to this definition") 
    
A size-based auto wrap policy. 

Parameters: 
    
  * **module** ([_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – Current module being considered.
  * **recurse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")) – If `False`, then this function must decide whether `module` should be wrapped as an FSDP instance or not. If `True`, then the function is still recursing down the module tree as a part of the DFS.
  * **nonwrapped_numel** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Parameter numel not yet wrapped.
  * **min_num_params** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Customizable policy input that controls the size threshold over which a module is ready to be wrapped. This is in units of numel.
  * **force_leaf_modules** (_Optional_ _[_[_set_](https://docs.python.org/3/library/stdtypes.html#set "\(in Python v3.14\)") _[_[_type_](https://docs.python.org/3/library/functions.html#type "\(in Python v3.14\)") _[_[_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__]__]_) – Set of module types to keep as leaves, i.e. their children will never be wrapped.
  * **exclude_wrap_modules** (_Optional_ _[_[_set_](https://docs.python.org/3/library/stdtypes.html#set "\(in Python v3.14\)") _[_[_type_](https://docs.python.org/3/library/functions.html#type "\(in Python v3.14\)") _[_[_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") _]__]__]_) – Set of module types to be excluded in wrapping.



Returns: 
    
Whether `module` should be wrapped. 

Return type: 


torch.distributed.fsdp.wrap.transformer_auto_wrap_policy(_module_ , _recurse_ , _nonwrapped_numel_ , _transformer_layer_cls_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L298)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.transformer_auto_wrap_policy "Link to this definition") 
    
See `_module_wrap_policy()`, where `transformer_layer_cls` is the same as `module_classes`. Note that shared parameters must be wrapped in the same FSDP instance, so this auto wrap policy can help wrap shared embeddings into the same FSDP instance for transformer models. 

Return type: 


torch.distributed.fsdp.wrap.(_module_ , _wrap_overrides_)[[source]](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/wrap.py#L436)[#](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.wrap "Link to this definition") 
    
Annotate that a module should be wrapped. Annotated modules will only be wrapped if inside of an [`enable_wrap()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.enable_wrap "torch.distributed.fsdp.wrap.enable_wrap") context manager. This allows a module to be initialized both with and without a wrapper without code change.
The class that this function wraps the passed in `nn.Module` with is the passed in `wrapper_cls` argument into `enable_wrap`. Both `enable_wrap` and `wrap` can take in kwargs specifying how to construct the `wrapper_cls` instance. In the case of duplicate kwargs in `enable_wrap` and `wrap`, the argument passed into `wrap` will be respected.
Usage:

```
with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
    # Wraps layer in FSDP by default if within context
    self.l1 = wrap(torch.nn.Linear(5, 5))

```
Copy to clipboard 

Parameters: 
    
  * **module** ([_nn.Module_](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – module to wrap (if in [`enable_wrap()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.enable_wrap "torch.distributed.fsdp.wrap.enable_wrap") context)
  * ****wrap_overrides** ([_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)")) – configuration overrides that will take priority over the values provided by the [`enable_wrap()`](https://docs.pytorch.org/docs/2.13/fsdp.html#torch.distributed.fsdp.wrap.enable_wrap "torch.distributed.fsdp.wrap.enable_wrap") context



Return type: 
    
On this page 
[ Edit on GitHub ](https://github.com/pytorch/pytorch/edit/main/docs/source/fsdp.md)
[ Show Source ](https://docs.pytorch.org/docs/2.13/_sources/fsdp.md.txt)
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
