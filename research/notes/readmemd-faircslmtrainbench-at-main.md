---
title: README.md · FAIRC/SLMTrainBench at main
id: readmemd-faircslmtrainbench-at-main
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-1
- measured-throughput
- sequence-length-scaling
- mfu
- a100
created: '2026-10-27T14:55:02.420392Z'
updated: '2026-10-27T19:02:00.685481Z'
source: https://huggingface.co/datasets/FAIRC/SLMTrainBench/blob/main/README.md
source_domain: huggingface.co
fetched_at: '2026-10-27T14:55:01.800000Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: docs
deprecated: false
summary: 'SLMTrainBench (FAIRC, HuggingFace, v1.0.0, 2026) is the measurement dataset
  behind ''When Peak Floating-Point Throughput Misleads: Utilization and Cost Frontiers
  for Small Language Model Pretraining''. It reports batch-saturated single-GPU steady-state
  training-step throughput (AdamW state zeroing + BF16-autocast forward + shifted-token
  cross-entropy + backward + FP32 AdamW update; eager PyTorch 2.8.0 + OLM, torch.compile,
  no activation checkpointing, no gradient accumulation) for nine dense decoder-only
  models (150M/250M/350M/500M/700M/1B/2B/4B/8B) across ten GPUs and context lengths
  512-32,768: 2,963 configurations, seeds 11/22 as timing replicates. THE LOAD-BEARING
  RESULT for this run, computed from benchmark_catalog.csv: on A100 80GB PCIe the
  500M model does 26,297 tok/s at 4k (MFU 39.88%) and 11,990 tok/s at 32k (MFU 63.88%)
  - a measured per-token cost ratio of 2.19x against a FLOPs ratio of 3.51x, so a
  1.60x MFU uplift absorbs 40% of the quadratic penalty. The pattern is systematic
  across sizes on A100: 150M 2.21x (MFU 32.9->62.4%), 250M 2.27x (35.1->62.7%), 350M
  2.23x (37.0->63.2%), 500M 2.19x (39.9->63.9%); on H100 80GB HBM3 the same models
  give 2.31x/2.36x/2.35x/2.33x with MFU 21-27% -> 38-41%. The 1B model OOMs at 32k
  on A100 PCIe but gives 4k=15,757 tok/s (42.61%) and 16k=10,920 (55.42%), a 1.44x
  ratio for 4x sequence length. Fitting a two-component MFU model (attention vs non-attention)
  to the 500M A100 points gives attention at ~73% of peak and the rest of the step
  at ~21%, which reproduces the measured curve to within a point. Caveats: single-GPU,
  no context parallelism, no activation checkpointing, and the best batch-saturated
  config at each length (bs=8 at 4k, bs=1 at 32k), so this isolates the compute-scaling
  component of the long-context penalty rather than a distributed-training measurement.'
---

[Publish SLMTrainBench v1.0.0](https://huggingface.co/datasets/FAIRC/SLMTrainBench/commit/5744535943803b7c0811d2122ea860d9bd60f509)
[5744535](https://huggingface.co/datasets/FAIRC/SLMTrainBench/commit/5744535943803b7c0811d2122ea860d9bd60f509) verified 2 months ago
|
Download
[ History ](https://huggingface.co/datasets/FAIRC/SLMTrainBench/commits/main/README.md)[ Blame ](https://huggingface.co/datasets/FAIRC/SLMTrainBench/blame/main/README.md)[ Contribute ](https://huggingface.co/datasets/FAIRC/SLMTrainBench/edit/main/README.md)[ Delete ](https://huggingface.co/datasets/FAIRC/SLMTrainBench/delete/main/README.md)
9.18 kB
[metadata ](https://huggingface.co/docs/hub/datasets-cards) 
```
license: cc-by-4.0
pretty_name: SLMTrainBench
size_categories:
  - 1K<n<10K
tags:
  - benchmark
  - gpu
  - hardware
  - pytorch
  - openlanguagemodel
  - small-language-model
  - training
  - mfu
  - tabular
configs:
  - config_name: default
    data_files:
      - split: benchmark
        path: benchmark_catalog.csv

```

#  SLMTrainBench 
SLMTrainBench is the measurement dataset for **When Peak Floating-Point Throughput Misleads: Utilization and Cost Frontiers for Small Language Model Pretraining**. It maps batch-saturated, single-GPU training performance for nine dense decoder-only models from 150 million to 8 billion parameters across ten NVIDIA GPUs and context lengths from 512 to 32,768 tokens.
The dataset contains 2,963 tested batch configurations, including successful measurements and out-of-memory boundaries. It reports tokens per second (TPS), model floating-point operations utilization (MFU), peak allocated and reserved video memory (VRAM), timing stability, attention backend, seed, and provenance.
##  What was measured 
Each timed step performs:
  1. AdamW gradient/state zeroing;
  2. BF16-autocast forward propagation;
  3. shifted-token cross-entropy;
  4. backward propagation; and
  5. an AdamW optimizer update with FP32 parameters, gradients, and optimizer state.


The runs use eager PyTorch 2.8.0 and OpenLanguageModel (OLM), with `torch.compile`, activation checkpointing, and gradient accumulation disabled. Synthetic token tensors remain on the GPU. These are therefore steady-state model-step measurements, not end-to-end dataloader throughput, total training cost, or time-to-quality measurements.
For every GPU, model, and context combination, the harness tests power-of-two batch sizes and records 20 timed steps after adaptive warmup. Seeds 11 and 22 change model initialization and token values; they are timing replicates, not independent machines or training-quality trials.
##  Coverage   
| Dimension  | Values  |  
| --- | --- |  
| GPUs  | A100 80GB PCIe, B200, B300 SXM6 AC, RTX 4090, RTX 5090, H100 80GB HBM3, H200, RTX 6000 Ada, RTX A6000, RTX PRO 6000 Blackwell Server Edition  |  
| Model labels  | 150M, 250M, 350M, 500M, 700M, 1B, 2B, 4B, 8B  |  
| Context lengths  | 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768  |  
| Seeds  | 11, 22  |  
| Precision  | BF16 autocast compute; FP32 parameters, gradients, and Adam states  |  
| Software  | PyTorch 2.8.0, CUDA 12.8, Python 3.12.3, OLM eager mode  |  
The grid is intentionally ragged: long contexts and large models are present only when they fit, and the batch sweep stops at an out-of-memory event or the protocol's saturation rule.
##  Files 
  * `benchmark_catalog.csv`: viewer-friendly flat table containing all 2,963 tested configurations.
  * `benchmark-data.json`: canonical nested release artifact, including protocol, model, GPU, pricing, provenance, row, and interpolation metadata.
  * `metadata/provider_pricing_snapshot.json`: 43 dated provider quotes from RunPod, Vast.ai, Lambda, Amazon Web Services, and Google Cloud.
  * `metadata/runpod_pricing_snapshot.json`: earlier RunPod-only price input kept as a historical audit record.
  * `metadata/rtx4090_metadata_erratum.json`: source-URL-only correction; no performance measurement changed.


Provider prices were captured on 27 July 2026 and are historical observations. Only performance on RunPod-hosted machines was measured. Costs for other providers transfer their dated hourly prices onto RunPod-measured TPS and are projections, not provider-specific benchmarks.
##  Loading 
With Hugging Face Datasets:

```
from datasets import load_dataset

benchmark = load_dataset("FAIRC/SLMTrainBench", split="benchmark")

```

With pandas:

```
import pandas as pd

catalog = pd.read_csv(
    "https://huggingface.co/datasets/FAIRC/SLMTrainBench/resolve/main/benchmark_catalog.csv"
)

```

The CSV is the only file configured for automatic loading. Download `benchmark-data.json` directly when the full nested protocol and interpolation metadata are needed.
##  CSV schema   
| Column  | Description  |  
| --- | --- |  
| `source_file`  | Provenance filename in the original benchmark archive.  |  
| `source_kind`  |  `context_frontier` measurement or reused context-2,048 baseline.  |  
| `gpu_name`  | Captured NVIDIA device name.  |  
| `gpu_uuid`  | Device UUID used to distinguish physical boards; not a credential.  |  
| `seed`  | Input/model-initialization seed (11 or 22).  |  
|  `model_key`, `model_label`  | Machine- and human-readable model-size labels.  |  
| `actual_unique_parameters`  | Exact number of unique trainable parameters.  |  
| `sequence_length`  | Tokens per sequence.  |  
| `batch_size`  | Sequences per optimizer step.  |  
| `tokens_per_step`  |  `batch_size * sequence_length` for completed rows.  |  
| `status`  |  `complete`, `oom`, or `oom_during_model_build`.  |  
| `stable`  | Whether adaptive warmup passed; absent for failed rows.  |  
| `warmup_steps_executed`  | Warmup steps before retained timing began.  |  
| `measured_steps`  | Number of retained timed steps.  |  
| `median_step_time_ms`  | Median complete-step time in milliseconds.  |  
| `mean_based_tokens_per_second`  | Tokens divided by arithmetic-mean step time.  |  
| `measured_robust_relative_jitter`  | Median absolute deviation divided by median retained step time.  |  
| `tokens_per_second`  | Primary TPS, computed from median step time.  |  
| `achieved_tflops`  | Modeled training floating-point operations per second in TFLOP/s.  |  
| `dense_bf16_peak_tflops`  | Nominal vendor dense-BF16 peak used as the primary MFU denominator.  |  
| `model_flops_utilization_pct`  | Nominal-reference MFU percentage.  |  
| `configured_clock_dense_bf16_peak_tflops`  | Peak linearly adjusted to the captured application clock.  |  
| `configured_clock_model_flops_utilization_pct`  | Configured-clock MFU sensitivity value.  |  
|  `peak_allocated_gb`, `peak_allocated_vram_pct`  | Peak PyTorch-allocated VRAM.  |  
|  `peak_reserved_gb`, `peak_reserved_vram_pct`  | Peak PyTorch-reserved VRAM.  |  
| `selected_sdpa_backend`  | PyTorch scaled dot-product attention backend when captured.  |  
| `error`  | Failure text for out-of-memory rows.  |  
Missing values are expected for metrics that cannot be produced by an out-of-memory run. Two model-build failures also lack model- and batch-level fields.
##  Metric definitions 
For batch size (b), sequence length (s), and median step time (t):
[ \mathrm{TPS}=\frac{bs}{t}. ]
For unique parameters (P), layers (n_l), hidden width (h), and context length (s), the benchmark models training work per token as
[ f_{\mathrm{token}} = 6P + 12n_lhs. ]
Achieved modeled throughput is (A=\mathrm{TPS},f_{\mathrm{token}}/10^{12}), and nominal-reference MFU is (100A/F_{\mathrm{BF16,nom}}). MFU is a modeled fraction of vendor peak, not a hardware-counter measurement; the FLOP model omits elementwise operations.
##  Intended use 
Use SLMTrainBench to:
  * compare measured single-GPU TPS, MFU, and memory use for this workload family;
  * locate tested batch sizes and out-of-memory boundaries;
  * reproduce the paper's batch-selection and cost-frontier analyses; and
  * form planning hypotheses for nearby model sizes before validating the focal configuration on the intended machine.


##  Limitations 
  * Results cover one OLM Llama-style architecture, eager PyTorch, BF16 compute, and NVIDIA GPUs. They do not establish rankings for compiled graphs, FP8, alternate kernels, other frameworks, or other accelerators.
  * Most GPU models were tested on one rented board; B300 used two boards. The two seeds do not measure host-to-host or provider-to-provider variance.
  * Synthetic resident tokens exclude input pipelines, checkpointing, evaluation, networking, failures, and setup/idle time.
  * This is a single-GPU benchmark. Do not estimate multi-GPU wall time by simply multiplying TPS; communication and scaling efficiency must be measured.
  * Prices and marketplace availability change. Treat the supplied quotes only as dated, auditable snapshots.
  * The released surrogate is validated only within the measured 150M-8B and context-512-32,768 region and should not replace a focal validation run.


##  Related resources 
  * [Interactive cost explorer](https://olm-cost-frontier-demo.pages.dev/)
  * [Benchmark and analysis source](https://github.com/openlanguagemodel/slmtrainbench)


##  Citation 

```
@misc{mankash2026peakthroughput,
  title        = {When Peak Floating-Point Throughput Misleads: Utilization and Cost Frontiers for Small Language Model Pretraining},
  author       = {Tavish Mankash and Vardhaman Kalloli and Keshava Prasad and Deepan Muthirayan},
  year         = {2026},
  howpublished = {Preprint},
  note         = {SLMTrainBench dataset, version 1.0.0},
  url          = {https://huggingface.co/datasets/FAIRC/SLMTrainBench}
}

```

##  License 
The measurement dataset and its metadata are released under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). Cite the accompanying preprint and identify SLMTrainBench version 1.0.0 when redistributing or adapting the data.
