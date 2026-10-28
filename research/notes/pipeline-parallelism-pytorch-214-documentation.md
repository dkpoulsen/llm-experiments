---
title: Pipeline Parallelism — PyTorch 2.14 documentation
id: pipeline-parallelism-pytorch-214-documentation
tags:
- cmp170hx-training-feasibility-a0d4fc
- locus-topology-cost-at-1m-tokens
created: '2026-10-27T13:24:08.563572Z'
updated: '2026-10-27T19:00:31.695882Z'
source: https://pytorch.org/docs/stable/distributed.pipelining.html
source_domain: pytorch.org
fetched_at: '2026-10-27T13:24:07.940044Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: docs
deprecated: false
summary: Suggested by efficient-large-scale-language-model-training-on-gpu-clusters
utility_score: 13.0
---

*Suggested by [[efficient-large-scale-language-model-training-on-gpu-clusters]]*

Opens in a new window Opens an external website Opens an external website in a new window  
<!---->Close this dialog<!---->
This website utilises technologies such as cookies to enable essential site functionality, as well as for analytics, personalisation, and targeted advertising. To learn more, view the following link:  [Privacy Policy](https://lfprojects.org/policies/privacy-policy/)
Manage Preferences 
<!---->Close Cookie Preferences<!---->
[Skip to main content](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#main-content)
Back to top
v2.14.0 (stable)
[v2.15.0 (unstable)](https://docs.pytorch.org/docs/main/distributed.pipelining.html)[v2.14.0 (stable)](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)[v2.13.0](https://docs.pytorch.org/docs/2.13/distributed.pipelining.html)[v2.12.0](https://docs.pytorch.org/docs/2.12/distributed.pipelining.html)[v2.11.0](https://docs.pytorch.org/docs/2.11/distributed.pipelining.html)[v2.10.0](https://docs.pytorch.org/docs/2.10/distributed.pipelining.html)[v2.9.1](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)[v2.8.0](https://docs.pytorch.org/docs/2.8/distributed.pipelining.html)[v2.7.0](https://docs.pytorch.org/docs/2.7/distributed.pipelining.html)[v2.6.0](https://docs.pytorch.org/docs/2.6/distributed.pipelining.html)[v2.5.0](https://docs.pytorch.org/docs/2.5/distributed.pipelining.html)[v2.4.0](https://docs.pytorch.org/docs/2.4/distributed.pipelining.html)[v2.3.0](https://docs.pytorch.org/docs/2.3/distributed.pipelining.html)[v2.2.0](https://docs.pytorch.org/docs/2.2/distributed.pipelining.html)[v2.1.0](https://docs.pytorch.org/docs/2.1/distributed.pipelining.html)[v2.0.0](https://docs.pytorch.org/docs/2.0/distributed.pipelining.html)[v1.13](https://docs.pytorch.org/docs/1.13/distributed.pipelining.html)[v1.12](https://docs.pytorch.org/docs/1.12/distributed.pipelining.html)[v1.11](https://docs.pytorch.org/docs/1.11/distributed.pipelining.html)[v1.10](https://docs.pytorch.org/docs/1.10/distributed.pipelining.html)[v1.9.1](https://docs.pytorch.org/docs/1.9.1/distributed.pipelining.html)[v1.9.0](https://docs.pytorch.org/docs/1.9.0/distributed.pipelining.html)[v1.8.1](https://docs.pytorch.org/docs/1.8.1/distributed.pipelining.html)[v1.8.0](https://docs.pytorch.org/docs/1.8.0/distributed.pipelining.html)[v1.7.1](https://docs.pytorch.org/docs/1.7.1/distributed.pipelining.html)[v1.7.0](https://docs.pytorch.org/docs/1.7.0/distributed.pipelining.html)[v1.6.0](https://docs.pytorch.org/docs/1.6.0/distributed.pipelining.html)[v1.5.1](https://docs.pytorch.org/docs/1.5.1/distributed.pipelining.html)[v1.5.0](https://docs.pytorch.org/docs/1.5.0/distributed.pipelining.html)[v1.4.0](https://docs.pytorch.org/docs/1.4.0/distributed.pipelining.html)[v1.3.1](https://docs.pytorch.org/docs/1.3.1/distributed.pipelining.html)[v1.3.0](https://docs.pytorch.org/docs/1.3.0/distributed.pipelining.html)[v1.2.0](https://docs.pytorch.org/docs/1.2.0/distributed.pipelining.html)[v1.1.0](https://docs.pytorch.org/docs/1.1.0/distributed.pipelining.html)[v1.0.1](https://docs.pytorch.org/docs/1.0.1/distributed.pipelining.html)[v1.0.0](https://docs.pytorch.org/docs/1.0.0/distributed.pipelining.html)[v0.4.1](https://docs.pytorch.org/docs/0.4.1/distributed.pipelining.html)[v0.4.0](https://docs.pytorch.org/docs/0.4.0/distributed.pipelining.html)[v0.3.1](https://docs.pytorch.org/docs/0.3.1/distributed.pipelining.html)[v0.3.0](https://docs.pytorch.org/docs/0.3.0/distributed.pipelining.html)[v0.2.0](https://docs.pytorch.org/docs/0.2.0/distributed.pipelining.html)[v0.1.12](https://docs.pytorch.org/docs/0.1.12/distributed.pipelining.html)
[ Go to pytorch.org  ](https://pytorch.org)
Custom Search  
|  Sort by Relevance Date  |  
| --- |  
  * [ GitHub](https://github.com/pytorch/pytorch)
  * [ PyTorch Forum](https://discuss.pytorch.org/)


Rate this Page 
★ ★ ★ ★ ★
# Pipeline Parallelism[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#pipeline-parallelism "Link to this heading")
Created On: Jun 16, 2025 | Last Updated On: Jul 24, 2026
Note
`torch.distributed.pipelining` is currently in alpha state and under development. API changes may be possible. It was migrated from the [PiPPy](https://github.com/pytorch/PiPPy) project.
## Why Pipeline Parallel?[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#why-pipeline-parallel "Link to this heading")
Pipeline Parallelism is one of the primitive forms of parallelism for deep learning. It allows the **execution** of a model to be partitioned such that multiple **micro-batches** can execute different parts of the model code concurrently. Pipeline parallelism can be an effective technique for:
  * large-scale training
  * bandwidth-limited clusters
  * large model inference


The above scenarios share a commonality that the computation per device cannot hide the communication of conventional parallelism, for example, the weight all-gather of FSDP.
## What is `torch.distributed.pipelining`?[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#what-is-torch-distributed-pipelining "Link to this heading")
While promising for scaling, pipelining is often difficult to implement because it needs to **partition the execution** of a model in addition to model weights. The partitioning of execution often requires intrusive code changes to your model. Another aspect of complexity comes from **scheduling micro-batches in a distributed environment** , with **data flow dependency** considered.
The `pipelining` package provides a toolkit that does said things **automatically** which allows easy implementation of pipeline parallelism on **general** models.
It consists of two parts: a **splitting frontend** and a **distributed runtime**. The splitting frontend takes your model code as-is, splits it up into “model partitions”, and captures the data-flow relationship. The distributed runtime executes the pipeline stages on different devices in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc.
Overall, the `pipelining` package provides the following features:
  * Splitting of model code based on simple specification.
  * Rich support for pipeline schedules, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS, and providing the infrastructure for writing customized schedules.
  * First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).
  * Composability with other PyTorch parallel techniques such as data parallel (DDP, FSDP) or tensor parallel. The [TorchTitan](https://github.com/pytorch/torchtitan) project demonstrates a “3D parallel” application on the Llama model.


## Step 1: build `PipelineStage`[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#step-1-build-pipelinestage "Link to this heading")
Before we can use a `PipelineSchedule`, we need to create `PipelineStage` objects that wrap the part of the model running in that stage. The `PipelineStage` is responsible for allocating communication buffers and creating send/recv ops to communicate with its peers. It manages intermediate buffers e.g. for the outputs of forward that have not been consumed yet, and it provides a utility for running the backwards for the stage model.
A `PipelineStage` needs to know the input and output shapes for the stage model, so that it can correctly allocate communication buffers. The shapes must be static, e.g. at runtime the shapes can not change from step to step. A class `PipeliningShapeError` will be raised if runtime shapes do not match the expected shapes. When composing with other parallelisms or applying mixed precision, these techniques must be taken into account so the `PipelineStage` knows the correct shape (and dtype) for the output of the stage module at runtime.
Users may construct a `PipelineStage` instance directly, by passing in an `nn.Module` representing the portion of the model that should run on the stage. This may require changes to the original model code. See the example in [Option 1: splitting a model manually](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#option-1-manual).
Alternatively, the splitting frontend can use graph partitioning to split your model into a series of `nn.Module` automatically. This technique requires the model is traceable with `torch.Export`. Composability of the resulting `nn.Module` with other parallelism techniques is experimental, and may require some workarounds. Usage of this frontend may be more appealing if the user cannot easily change the model code. See [Option 2: splitting a model automatically](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#option-2-tracer) for more information.
## Step 2: use `PipelineSchedule` for execution[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#step-2-use-pipelineschedule-for-execution "Link to this heading")
We can now attach the `PipelineStage` to a pipeline schedule, and run the schedule with input data. Here is a GPipe example:

```
fromtorch.distributed.pipeliningimport ScheduleGPipe

# Create a schedule
schedule = ScheduleGPipe(stage, n_microbatches)

# Input data (whole batch)
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`
# `x` will be divided into microbatches automatically
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()

```
Copy to clipboard
If your input pipeline already produces microbatches, pass them directly:

```
arg_mbs = [(x0,), (x1,)]
kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
target_mbs = [target0, target1]

schedule.step(
    arg_mbs=arg_mbs,
    kwarg_mbs=kwarg_mbs,
    target_mbs=target_mbs,
)

```
Copy to clipboard
Note that the above code needs to be launched for each worker, thus we use a launcher service to launch multiple processes:

```
torchrun--nproc_per_node=2example.py

```
Copy to clipboard
## Options for Splitting a Model[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#options-for-splitting-a-model "Link to this heading")
### Option 1: splitting a model manually[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#option-1-splitting-a-model-manually "Link to this heading")
To directly construct a `PipelineStage`, the user is responsible for providing a single `nn.Module` instance that owns the relevant `nn.Parameters` and `nn.Buffers`, and defines a `forward()` method that executes the operations relevant for that stage. For example, a condensed version of the Transformer class defined in Torchtitan shows a pattern of building an easily partitionable model.

```
classTransformer(nn.Module):
    def__init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(...)

        # Using a ModuleDict lets us delete layers without affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(...)

        self.output = nn.Linear(...)

    defforward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output

```
Copy to clipboard
A model defined in this manner can be easily configured per stage by first initializing the whole model (using meta-device to avoid OOM errors), deleting undesired layers for that stage, and then creating a PipelineStage that wraps the model. For example:

```
with torch.device("meta"):
    assert num_stages == 2, "This is a simple 2-stage example"

    # we construct the entire model, then delete the parts we do not need for this stage
    # in practice, this can be done using a helper function that automatically divides up layers across stages.
    model = Transformer()

    if stage_index == 0:
        # prepare the first stage model
        del model.layers["1"]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # prepare the second stage model
        model.tok_embeddings = None
        del model.layers["0"]

    fromtorch.distributed.pipeliningimport PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )

```
Copy to clipboard
When composing with other Data or Model parallelism techniques, `output_args` may also be required, if the output shape/dtype of the model chunk will be affected.
### Option 2: splitting a model automatically[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#option-2-splitting-a-model-automatically "Link to this heading")
If you have a full model and do not want to spend time on modifying it into a sequence of “model partitions”, the `pipeline` API is here to help. Here is a brief example:

```
classModel(torch.nn.Module):
    def__init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.layers = torch.nn.ModuleList(
            Layer() for _ in range(2)
        )
        self.lm = LMHead()

    defforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm(x)
        return x

```
Copy to clipboard
If we print the model, we can see multiple hierarchies, which makes it hard to split by hand:

```
Model(
  (emb): Embedding(10, 3)
  (layers): ModuleList(
    (0-1): 2 x Layer(
      (lin): Linear(in_features=3, out_features=3, bias=True)
    )
  )
  (lm): LMHead(
    (proj): Linear(in_features=3, out_features=3, bias=True)
  )
)

```
Copy to clipboard
Let us see how the `pipeline` API works:

```
fromtorch.distributed.pipeliningimport pipeline, SplitPoint

# An example micro-batch input
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=mod,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)

```
Copy to clipboard
The `pipeline` API splits your model given a `split_spec`, where `SplitPoint.BEGINNING` stands for adding a split point _before_ execution of certain submodule in the `forward` function, and similarly, `SplitPoint.END` for split point _after_ such.
If we `print(pipe)`, we can see:

```
GraphModule(
  (submod_0): GraphModule(
    (emb): InterpreterModule()
    (layers): Module(
      (0): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
  )
  (submod_1): GraphModule(
    (layers): Module(
      (1): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
    (lm): InterpreterModule(
      (proj): InterpreterModule()
    )
  )
)

defforward(self, x):
    submod_0 = self.submod_0(x);  x = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    return (submod_1,)

```
Copy to clipboard
The “model partitions” are represented by submodules (`submod_0`, `submod_1`), each of which is reconstructed with original model operations, weights and hierarchies. In addition, a “root-level” `forward` function is reconstructed to capture the data flow between those partitions. Such data flow will be replayed by the pipeline runtime later, in a distributed fashion.
The `Pipe` object provides a method for retrieving the “model partitions”:

```
stage_mod : nn.Module = pipe.get_stage_module(stage_idx)

```
Copy to clipboard
The returned `stage_mod` is a `nn.Module`, with which you can create an optimizer, save or load checkpoints, or apply other parallelisms.
`Pipe` also allows you to create a distributed stage runtime on a device given a `ProcessGroup`:

```
stage = pipe.build_stage(stage_idx, device, group)

```
Copy to clipboard
Alternatively, if you would like to build the stage runtime later after some modification to the `stage_mod`, you can use a functional version of the `build_stage` API. For example:

```
fromtorch.distributed.pipeliningimport build_stage
fromtorch.nn.parallelimport DistributedDataParallel

dp_mod = DistributedDataParallel(stage_mod)
info = pipe.info()
stage = build_stage(dp_mod, stage_idx, info, device, group)

```
Copy to clipboard
Note
The `pipeline` frontend uses a tracer (`torch.export`) to capture your model into a single graph. If your model is not full-graph’able, you can use our manual frontend below.
## Hugging Face Examples[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#hugging-face-examples "Link to this heading")
In the [PiPPy](https://github.com/pytorch/PiPPy) repo where this package was original created, we kept examples based on unmodified Hugging Face models. See the [examples/huggingface](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface) directory.
Examples include:
  * 

## Technical Deep Dive[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#technical-deep-dive "Link to this heading")
### How does the `pipeline` API split a model?[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#how-does-the-pipeline-api-split-a-model "Link to this heading")
First, the `pipeline` API turns our model into a directed acyclic graph (DAG) by tracing the model. It traces the model using `torch.export` – a PyTorch 2 full-graph capturing tool.
Then, it groups together the **operations and parameters** needed by a stage into a reconstructed submodule: `submod_0`, `submod_1`, …
Different from conventional submodule access methods like `Module.children()`, the `pipeline` API does not only cut the module structure of your model, but also the **forward** function of your model.
This is necessary because model structure like `Module.children()` merely captures information during `Module.__init__()`, and does not capture any information about `Module.forward()`. Said differently, `Module.children()` lacks information about the following aspects key to pipelininig:
  * Execution order of child modules in `forward`
  * Activation flows between child modules
  * Whether there are any functional operators between child modules (for example, `relu` or `add` operations will not be captured by `Module.children()`).


The `pipeline` API, on the contrary, makes sure that the `forward` behavior is truly preserved. It also captures the activation flow between the partitions, helping the distributed runtime to make correct send/receive calls without human intervention.
Another flexibility of the `pipeline` API is that split points can be at arbitrary levels within your model hierarchy. In the split partitions, the original model hierarchy related to that partition will be reconstructed at no cost to you. At a result, fully-qualified names (FQNs) pointing to a submodule or parameter would be still valid, and services that relies on FQNs (such as FSDP, TP or checkpointing) can still run with your partitioned modules with almost zero code change.
## Implementing Your Own Schedule[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#implementing-your-own-schedule "Link to this heading")
You can implement your own pipeline schedule by extending one of the following two classes:
  * `PipelineScheduleSingle`
  * `PipelineScheduleMulti`


`PipelineScheduleSingle` is for schedules that assigns _only one_ stage per rank. `PipelineScheduleMulti` is for schedules that assigns multiple stages per rank.
For example, `ScheduleGPipe` and `Schedule1F1B` are subclasses of `PipelineScheduleSingle`. Whereas, `ScheduleInterleaved1F1B`, `ScheduleLoopedBFS`, `ScheduleInterleavedZeroBubble`, and `ScheduleZBVZeroBubble` are subclasses of `PipelineScheduleMulti`.
## Logging[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#logging "Link to this heading")
You can turn on additional logging using the `TORCH_LOGS` environment variable from [torch._logging](https://pytorch.org/docs/main/logging.html#module-torch._logging):
  * `TORCH_LOGS=+pp` will display `logging.DEBUG` messages and all levels above it.
  * `TORCH_LOGS=pp` will display `logging.INFO` messages and above.
  * `TORCH_LOGS=-pp` will display `logging.WARNING` messages and above.


## API Reference[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#module-torch.distributed.pipelining "Link to this heading")
### Model Split APIs[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#model-split-apis "Link to this heading")
The following set of APIs transform your model into a pipeline representation. 

_class _torch.distributed.pipelining.SplitPoint(_value_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/_IR.py#L1192)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint "Link to this definition") 
    
Enum representing the points at which a split can occur in the execution of a submodule. :ivar BEGINNING: Represents adding a split point _before_ the execution of a certain submodule in the forward function. :ivar END: Represents adding a split point _after_ the execution of a certain submodule in the forward function. 

torch.distributed.pipelining.pipeline(_module_ , _mb_args_ , _mb_kwargs_ , _split_spec_ , _split_policy_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/_IR.py#L1247)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.pipeline "Link to this definition") 
    
Split a module based on a specification.
See Pipe for more details. 

Parameters: 
    
  * **module** ([_Module_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")) – The module to be split.
  * **mb_args** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)") _,__...__]_) – Example positional inputs, in micro-batch form.
  * **mb_kwargs** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,_[_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)") _]__|__None_) – Example keyword inputs, in micro-batch form. (default: None)
  * **split_spec** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,_[_SplitPoint_](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint "torch.distributed.pipelining._IR.SplitPoint") _]__|__None_) – A dictionary using submodule names as split marker. (default: None)
  * **split_policy** ([_Callable_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "\(in Python v3.14\)") _[__[_[_GraphModule_](https://docs.pytorch.org/docs/2.14/fx.html#torch.fx.GraphModule "torch.fx.graph_module.GraphModule") _]__,_[_GraphModule_](https://docs.pytorch.org/docs/2.14/fx.html#torch.fx.GraphModule "torch.fx.graph_module.GraphModule") _]__|__None_) – The policy to use for splitting the module. (default: None)



Return type: 
    
A pipeline representation of class Pipe. 

_class _torch.distributed.pipelining.(_split_gm_ , _num_stages_ , _has_loss_and_backward_ , _loss_spec_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/_IR.py#L561)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.Pipe "Link to this definition") 


torch.distributed.pipelining.pipe_split()[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/_IR.py#L363)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.pipe_split "Link to this definition") 
    
pipe_split is a special operator that is used to mark the boundary between stages in a module. It is used to split the module into stages. It is a no-op if your annotated module is run eagerly.
Example

```
>>> defforward(self, x):
>>>     x = torch.mm(x, self.mm_param)
>>>     x = torch.relu(x)
>>>     pipe_split()
>>>     x = self.lin(x)
>>>     return x

```
Copy to clipboard
The above example will be split into two stages.
### Microbatch Utilities[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#module-torch.distributed.pipelining.microbatch "Link to this heading") 

_class _torch.distributed.pipelining.microbatch.TensorChunkSpec(_split_dim_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/microbatch.py#L60)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec "Link to this definition") 
    
Class used to specify chunking of inputs 

torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks(__,_kwargs_ , _chunks_ , _args_chunk_spec_ , _kwargs_chunk_spec_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/microbatch.py#L378)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks "Link to this definition") 
    
Given a sequence of args and kwargs, split them into a number of chunks according to their respective chunking specs. 

Parameters: 
    
  * **args** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)") _,__...__]_) – Tuple of args
  * **kwargs** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,_[_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)") _]__|__None_) – Dict of kwargs
  * **chunks** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Number of chunks to split the args and kwargs into
  * **args_chunk_spec** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_TensorChunkSpec_](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec "torch.distributed.pipelining.microbatch.TensorChunkSpec") _,__...__]__|__None_) – chunking specs for args, in same shape as args
  * **kwargs_chunk_spec** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,_[_TensorChunkSpec_](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec "torch.distributed.pipelining.microbatch.TensorChunkSpec") _]__|__None_) – chunking specs for kwargs, in same shape as kwargs



Returns: 
    
List of sharded args kwargs_split: List of sharded kwargs 

Return type: 
    
args_split 

torch.distributed.pipelining.microbatch.merge_chunks(_chunks_ , _chunk_spec_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/microbatch.py#L493)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.microbatch.merge_chunks "Link to this definition") 
    
Given a list of chunks, merge them into a single value according to the chunk spec. 

Parameters: 
    
  * **chunks** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_Any_](https://docs.python.org/3/library/typing.html#typing.Any "\(in Python v3.14\)") _]_) – list of chunks
  * **chunk_spec** – Chunking spec for the chunks



Returns: 
    
Merged value 

Return type: 
    
value
### Pipeline Stages[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#module-torch.distributed.pipelining.stage "Link to this heading") 

_class _torch.distributed.pipelining.stage.PipelineStage(_submodule_ , _stage_index_ , _num_stages_ , _device_ , _input_args_ , _output_args_ , _output_grads_ , _input_grads_ , _group_ , _dw_builder_ , _get_mesh_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L1737)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage "Link to this definition") 
    
A pipeline stage for pipeline parallelism with sequential model partitioning.
Supports both **static** and **dynamic** metadata inference: 

Static mode:
    
All of `input_args`, `output_args` (and `input_grads`/`output_grads` when DTensors are present) are provided at construction time. 

Dynamic mode:
    
Metadata is inferred from the first microbatch at runtime; any statically provided args are used for validation only. 

Parameters: 
    
  * **submodule** ([_Module_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")) – The `nn.Module` wrapped by this stage.
  * **stage_index** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Zero-based stage ID.
  * **num_stages** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Total number of stages in the pipeline.
  * **device** ([_device_](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch.device "torch.device")) – Device this stage runs on.
  * **input_args** ([_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _,__...__]__|__None_) – Example input tensors (single tensor or tuple). Optional.
  * **output_args** ([_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _,__...__]__|__None_) – Example output tensors. Optional.
  * **output_grads** ([_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|__None_ _,__...__]__|__None_) – Example output gradients (received from next stage). Optional.
  * **input_grads** ([_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _[_[_Tensor_](https://docs.pytorch.org/docs/2.14/tensors.html#torch.Tensor "torch.Tensor") _|__None_ _,__...__]__|__None_) – Example input gradients (sent to previous stage). Optional.
  * **group** (_ProcessGroup_ _|__None_) – Process group for P2P communication. Defaults to the world process group.
  * **dw_builder** ([_Callable_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "\(in Python v3.14\)") _[__[__]__,_[_Callable_](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "\(in Python v3.14\)") _[__[__...__]__,__None_ _]__]__|__None_) – Builder for deferred weight-update runners used by zero-bubble (F/I/W) schedules.
  * **get_mesh** (_GetMeshCallback_ _|__None_) – GetMeshCallback used during dynamic DTensor inference. Ignored in fully static DTensor mode.



torch.distributed.pipelining.stage.build_stage(_stage_module_ , _stage_index_ , _pipe_info_ , _device_ , _group_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L1707)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.stage.build_stage "Link to this definition") 
    
Create a pipeline stage given a stage_module to be wrapped by this stage and pipeline information. 

Parameters: 
    
  * **stage_module** ([_torch.nn.Module_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")) – the module to be wrapped by this stage
  * **stage_index** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – the index of this stage in the pipeline
  * **pipe_info** (_PipeInfo_) – information about the pipeline, can be retrieved by pipe.info()
  * **device** ([_torch.device_](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch.device "torch.device")) – the device to be used by this stage
  * **group** (_Optional_ _[__dist.ProcessGroup_ _]_) – the process group to be used by this stage



Returns: 
    
a pipeline stage that can run with PipelineSchedules. 

Return type: 
    
_PipelineStage
### Pipeline Schedules[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#module-torch.distributed.pipelining.schedules "Link to this heading") 

_class _torch.distributed.pipelining.schedules.ScheduleGPipe(_stage_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1083)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe "Link to this definition") 
    
The GPipe schedule. Will go through all the microbatches in a fill-drain manner. 

_class _torch.distributed.pipelining.schedules.Schedule1F1B(_stage_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1206)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.Schedule1F1B "Link to this definition") 
    
The 1F1B schedule. Will perform one forward and one backward on the microbatches in steady state. 

_class _torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B(_stages_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_ , _backward_requires_autograd_ , _defer_pp_recv False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L3149)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B "Link to this definition") 
    
The Interleaved 1F1B schedule. See <https://arxiv.org/pdf/2104.04473> for details. Will perform one forward and one backward on the microbatches in steady state and supports multiple stages per rank. When microbatches are ready for multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch (also called “depth first”).
This schedule is mostly similar to the original paper. It differs by being relaxing the requirement of num_microbatch % pp_size == 0. Using the flex_pp schedule, we will have num_rounds = max(1, n_microbatches // pp_group_size) and it works as long as n_microbatches % num_rounds is 0. As a few examples, support
  1. pp_group_size = 4, n_microbatches = 10. We will have num_rounds = 2 and n_microbatches % 2 is 0.
  2. pp_group_size = 4, n_microbatches = 3. We will have num_rounds = 1 and n_microbatches % 1 is 0.



_class _torch.distributed.pipelining.schedules.ScheduleLoopedBFS(_stages_ , _n_microbatches_ , _loss_fn_ , _output_merge_spec_ , _scale_grads_ , _backward_requires_autograd_ , _defer_pp_recv False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L2922)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS "Link to this definition") 
    
Breadth-First Pipeline Parallelism. See <https://arxiv.org/abs/2211.05953> for details. Similar to Interleaved 1F1B, Looped BFS supports multiple stages per rank. What is different is that when microbatches are ready for multiple local stages, Loops BFS will prioritizes the earlier stage, running all available microbatches at once. 

_class _torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble(_stages_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_ , _backward_requires_autograd_ , _defer_pp_recv False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L3265)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble "Link to this definition") 
    
The Interleaved Zero Bubble schedule. See <https://arxiv.org/pdf/2401.10241> for details. Will perform one forward and one backward on inputs for the microbatches in steady state and supports multiple stages per rank. Uses the backward for weights to fill in the pipeline bubble.
In particular this is implementing the ZB1P schedule in the paper. 

_class _torch.distributed.pipelining.schedules.ScheduleZBVZeroBubble(_stages_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_ , _backward_requires_autograd_ , _defer_pp_recv False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L3457)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleZBVZeroBubble "Link to this definition") 
    
The Zero Bubble schedule (ZBV variant). See <https://arxiv.org/pdf/2401.10241> Section 6 for details.
This schedules requires exactly two stages per rank.
This schedule will perform one forward and one backward on inputs for the microbatches in steady state and supports multiple stages per rank. Uses backward with respect to weights to fill in the pipeline bubble.
This ZB-V schedule would have the “zero bubble” property only if time forward == time backward input == time backward weights. In practice, this is not likely true for real models so alternatively a greedy scheduler could be implemented for unequal/unbalanced time. 

_class _torch.distributed.pipelining.schedules.ScheduleDualPipeV(_stages_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_ , _backward_requires_autograd_ , _defer_pp_recv False_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L3651)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleDualPipeV "Link to this definition") 
    
The DualPipeV schedule. A more efficient schedule variant based on the DualPipe schedule introduced by DeepSeek in <https://arxiv.org/pdf/2412.19437>
Based on the open sourced code from [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) 

_class _torch.distributed.pipelining.schedules.PipelineScheduleSingle(_stage_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _scale_grads_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L852)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle "Link to this definition") 
    
Base class for single-stage schedules. Implements the step method. Derived classes should implement _step_microbatches.
Gradients are scaled by num_microbatches depending on the scale_grads argument, defaulting to True. This setting should match the configuration of your loss_fn, which may either average losses (scale_grads=True) or sum losses (scale_grads=False). 

(__,_target_ , _losses_ , _return_outputs_ , _loss_kwargs_ , _arg_mbs_ , _kwarg_mbs_ , _target_mbs_ , _kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L906)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle.step "Link to this definition") 
    
Run one training iteration of a single-stage pipeline schedule.
By default, `args`, `kwargs`, and `target` are full-batch values that the schedule splits into microbatches. When the caller has already split inputs into microbatches, pass them through `arg_mbs`, `kwarg_mbs`, and `target_mbs` instead. 

Parameters: 
    
  * ***args** (_Any_) – Whole-batch positional inputs for the first pipeline stage. Do not pass positional inputs with pre-split inputs.
  * **target** (_Any_ _,__optional_) – Whole-batch target for loss computation. When passing pre-split inputs, pass targets through `target_mbs` instead. Default: `None`.
  * **losses** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _,__optional_) – Mutable list populated with one loss per microbatch when this schedule owns the last stage and a `loss_fn` was configured. Default: `None`.
  * **return_outputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__optional_) – Whether to merge and return output chunks on the last stage. Default: `True`.
  * **loss_kwargs** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _,__optional_) – Extra keyword arguments forwarded to the configured `loss_fn`. Default: `None`.
  * **arg_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _]__,__optional_) – Pre-split positional inputs, one tuple per microbatch. Default: `None`.
  * **kwarg_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _]__,__optional_) – Pre-split keyword inputs, one dict per microbatch. Default: `None`.
  * **target_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _,__optional_) – Pre-split targets, one entry per microbatch. Default: `None`.
  * ****kwargs** (_Any_) – Whole-batch keyword inputs for the first pipeline stage. Do not pass keyword inputs with pre-split inputs.



Returns: 
     

The merged output from the last stage when this rank
    
owns the last stage and `return_outputs=True`; otherwise `None`. 

Return type: 
    
Any or None 

Raises: 
    
  * [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "\(in Python v3.14\)") – If backward computation is enabled and `step` is called under `torch.no_grad()`. Use `eval()` for forward-only execution.
  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If a pre-split microbatch container has the wrong type.
  * [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "\(in Python v3.14\)") – If whole-batch and pre-split inputs are mixed, or if a pre-split container does not have one entry per microbatch.


Examples:

```
>>> output = schedule.step(x, mask=mask, target=target)
>>> arg_mbs = [(x0,), (x1,)]
>>> kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
>>> target_mbs = [target0, target1]
>>> output = schedule.step(
...     arg_mbs=arg_mbs,
...     kwarg_mbs=kwarg_mbs,
...     target_mbs=target_mbs,
... )

```
Copy to clipboard 

_class _torch.distributed.pipelining.schedules.PipelineScheduleMulti(_stages_ , _n_microbatches_ , _loss_fn_ , _args_chunk_spec_ , _kwargs_chunk_spec_ , _output_merge_spec_ , _use_full_backward_ , _scale_grads_ , _backward_requires_autograd_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1960)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti "Link to this definition") 
    
Base class for multi-stage schedules. Implements the step method.
Gradients are scaled by num_microbatches depending on the scale_grads argument, defaulting to True. This setting should match the configuration of your loss_fn, which may either average losses (scale_grads=True) or sum losses (scale_grads=False). 

(__,_target_ , _losses_ , _return_outputs_ , _loss_kwargs_ , _arg_mbs_ , _kwarg_mbs_ , _target_mbs_ , _kwargs_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L2145)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti.step "Link to this definition") 
    
Run one training iteration of a multi-stage pipeline schedule.
By default, `args`, `kwargs`, and `target` are full-batch values that the schedule splits into microbatches. When the caller has already split inputs into microbatches, pass them through `arg_mbs`, `kwarg_mbs`, and `target_mbs` instead. 

Parameters: 
    
  * ***args** (_Any_) – Whole-batch positional root inputs when this rank owns the first pipeline stage. Do not pass positional inputs with pre-split inputs.
  * **target** (_Any_ _,__optional_) – Whole-batch target for loss computation. When passing pre-split inputs, pass targets through `target_mbs` instead. Default: `None`.
  * **losses** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _,__optional_) – Mutable list populated with one loss per microbatch when this schedule owns the last stage and a `loss_fn` was configured. Default: `None`.
  * **return_outputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") _,__optional_) – Whether to merge and return output chunks on the last stage. Default: `True`.
  * **loss_kwargs** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _,__optional_) – Extra keyword arguments forwarded to the configured `loss_fn`. Default: `None`.
  * **arg_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") _]__,__optional_) – Pre-split positional inputs, one tuple per microbatch. Default: `None`.
  * **kwarg_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _]__,__optional_) – Pre-split keyword inputs, one dict per microbatch. Default: `None`.
  * **target_mbs** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _,__optional_) – Pre-split targets, one entry per microbatch. Default: `None`.
  * ****kwargs** (_Any_) – Whole-batch keyword root inputs when this rank owns the first pipeline stage. Do not pass keyword inputs with pre-split inputs.



Returns: 
     

The merged output from the last stage when this rank
    
owns the last stage and `return_outputs=True`; otherwise `None`. 

Return type: 
    
Any or None 

Raises: 
    
  * [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "\(in Python v3.14\)") – If backward computation is enabled and `step` is called under `torch.no_grad()`. Use `eval()` for forward-only execution.
  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If a pre-split microbatch container has the wrong type.
  * [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "\(in Python v3.14\)") – If whole-batch and pre-split inputs are mixed, or if a pre-split container does not have one entry per microbatch.


Examples:

```
>>> output = schedule.step(x, mask=mask, target=target)
>>> arg_mbs = [(x0,), (x1,)]
>>> kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
>>> target_mbs = [target0, target1]
>>> output = schedule.step(
...     arg_mbs=arg_mbs,
...     kwarg_mbs=kwarg_mbs,
...     target_mbs=target_mbs,
... )

```
Copy to clipboard 

torch.distributed.pipelining.schedules.get_schedule_class(_schedule_name_)[[source]](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L3878)[#](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.get_schedule_class "Link to this definition") 
    
Maps a schedule name (case insensitive) to its corresponding class object. 

Parameters: 
    
**schedule_name** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The name of the schedule.
On this page 
[ Edit on GitHub ](https://github.com/pytorch/pytorch/edit/main/docs/source/distributed.pipelining.md)
[ Show Source ](https://docs.pytorch.org/docs/2.14/_sources/distributed.pipelining.md.txt)
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
