const query = `Assess which LLM training experiments are feasible on a 5× NVIDIA CMP 170HX box — 320 GB HBM2e, ~810 TFLOPS bf16, PCIe Gen2 x4, no NVLink and no P2P — and verify the supporting claims in the repo's documentation: whether DDP is the only viable training topology at 1M-token global batches, whether expert and tensor parallelism are ruled out by the fabric, whether frozen Engram memory tables (Memory Grafting) make a 7B-total / ~1B-trainable model fit DDP on 64 GB cards, and whether CSA2 compressed sparse attention pays off at 32k training context.`;

const shim = `## Run directives

Auto-selected for this run in step 1; binding wherever they adjust a default in your prompt. Absent instructions here leave your prompt's defaults untouched.

- register: analyze
- inference depth: standard

### Domain notes

Sourcing: academic APIs first for architecture claims (arXiv for Engram, Memory Grafting, CSA2, expert-parallel-over-PCIe); GitHub repos and community benchmarks for hardware measurements (club-170hx, deepseek-v4-cmp170hx); NVIDIA and PCI-SIG documentation for PWRBRK# and link negotiation. Evidence norms: a number measured on this exact box outranks a community report, which outranks arithmetic from stated assumptions — label every claim with its tier. Recency: hardware and model-architecture claims within 12 months matter; PCIe and NCCL fundamentals are stable.

### Inference depth

Standard depth (this run confirms your prompt's defaults). Follow the
normal sourcing playbook.`;

const batches = [
  [
    ["https://arxiv.org/abs/2601.07372", 18],
    ["https://arxiv.org/abs/2605.20948", 18],
    ["https://arxiv.org/abs/2603.10087", 14],
    ["https://arxiv.org/abs/2607.07388", 14],
    ["https://arxiv.org/abs/2610.10533", 12],
    ["https://arxiv.org/abs/2609.35578", 12],
    ["https://drli.blog/posts/analysis-engram-conditional-memory/", 13],
    ["https://docs.vllm.ai/en/v0.31.0/features/engram/", 12]
  ],
  [
    ["https://arxiv.org/abs/2609.19969", 18],
    ["https://arxiv.org/abs/2609.37988", 12],
    ["https://arxiv.org/abs/2609.26368", 12],
    ["https://arxiv.org/abs/2605.02568", 15],
    ["https://arxiv.org/abs/2607.19358", 12],
    ["https://arxiv.org/abs/2608.01662", 12],
    ["https://arxiv.org/abs/2603.12201", 13],
    ["https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html", 14]
  ],
  [
    ["https://arxiv.org/abs/2609.40093", 18],
    ["https://arxiv.org/abs/2604.27085", 15],
    ["https://arxiv.org/abs/2610.09424", 12],
    ["https://arxiv.org/abs/2609.34340", 14],
    ["https://arxiv.org/abs/2610.01380", 12],
    ["https://arxiv.org/abs/2610.07333", 12],
    ["https://arxiv.org/abs/2602.04870", 13],
    ["https://arxiv.org/abs/2211.15841", 15]
  ],
  [
    ["https://doi.org/10.14778/3611540.3611569", 17],
    ["https://arxiv.org/abs/2602.22437", 13],
    ["https://arxiv.org/abs/2311.00257", 14],
    ["https://doi.org/10.1109/sc41405.2020.00024", 17],
    ["https://arxiv.org/abs/2211.05322", 14],
    ["https://arxiv.org/abs/2402.03791", 12],
    ["https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py", 16],
    ["https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593", 12]
  ],
  [
    ["https://arxiv.org/abs/2609.23898", 12],
    ["https://arxiv.org/abs/2604.27844", 13],
    ["https://arxiv.org/abs/2607.07494", 12],
    ["https://arxiv.org/abs/2605.00539", 12],
    ["https://doi.org/10.1109/hoti52880.2021.00018", 14],
    ["https://doi.org/10.1007/978-3-031-24145-1_23", 13],
    ["https://escholarship.org/uc/item/67z0d2wn", 12],
    ["https://clusterbid.com/blog/nccl-tuning-ring-tree-all-to-all-algorithms", 11]
  ],
  [
    ["https://github.com/PixelML/club-170hx/blob/main/docs/HARDWARE.md", 18],
    ["https://github.com/PixelML/club-170hx/blob/main/docs/TOPOLOGY-AND-PARALLELISM.md", 18],
    ["https://github.com/PixelML/club-170hx/blob/main/docs/CMP170HX-UNDERSTANDING.md", 17],
    ["https://github.com/PixelML/club-170hx/blob/main/docs/BENCHMARKS.md", 15],
    ["https://github.com/bayley/cmpunlocker", 16],
    ["https://github.com/Consensus-Protocol/cmp170hx", 15],
    ["https://it-still-boots.github.io/cmp-170hx-docs/hardware/specifications/", 16],
    ["https://it-still-boots.github.io/cmp-170hx-docs/performance/overview/", 14]
  ],
  [
    ["https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/", 16],
    ["https://arxiv.org/abs/2505.03782", 15],
    ["https://github.com/amoghmunikote/170th-street/blob/main/benchmarks-and-performance/fp16-performance.md", 13],
    ["https://github.com/allover326/deepseek-v4-cmp170hx/blob/317dac12/SETTINGS.md", 16],
    ["https://deepwiki.com/allover326/deepseek-v4-cmp170hx", 14],
    ["https://github.com/dl4rce/cmp-170hx-inference-lab", 12],
    ["https://wpnews.pro/news/2x-cmp-170hx-for-llm-inference-unlock-cross-root-p2p-serving-vllm-epyc-romed8-2t", 12],
    ["https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md", 14]
  ],
  [
    ["https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu", 16],
    ["https://huggingface.co/datasets/HuggingFaceTB/stack-edu", 16],
    ["https://huggingface.co/datasets/HuggingFaceTB/finemath", 16],
    ["https://arxiv.org/abs/2502.02737", 15],
    ["https://doi.org/10.18653/v1/2024.acl-long.427", 14],
    ["https://arxiv.org/abs/2212.14034", 12],
    ["https://arxiv.org/abs/2505.09388", 16],
    ["https://doi.org/10.52202/079017-0970", 15]
  ],
  [
    ["https://arxiv.org/abs/2111.11124", 12],
    ["https://arxiv.org/abs/2309.12381", 12],
    ["https://doi.org/10.52202/085713-3898", 14],
    ["https://arxiv.org/abs/1804.07612", 11],
    ["https://archiesengupta.com/blog/memory-efficient-training-deep-dive", 12],
    ["https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5", 13],
    ["https://arxiv.org/abs/2609.34849", 11],
    ["https://arxiv.org/abs/2605.30992", 11]
  ],
  [
    ["https://arxiv.org/abs/2604.00801", 11],
    ["https://doi.org/10.18653/v1/2025.findings-naacl.107", 12],
    ["https://arxiv.org/abs/2505.04519", 13],
    ["https://arxiv.org/abs/2609.15627", 12],
    ["https://arxiv.org/abs/2609.36322", 11],
    ["https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash", 15],
    ["https://lyceum.technology/magazine/zero-3-vs-fsdp-memory-efficiency/", 11],
    ["https://towardsdatascience.com/behind-the-scenes-of-distributed-training-why-your-gpu-wiring-matters-as-much-as-your-strategy/", 11]
  ]
];

const tasks = batches.map((urls, i) => ({
  key: "batch-" + (i + 1),
  agent: "hyperresearch-fetcher",
  cwd: "/home/dkp/.thinkrail/worktrees/llm-experiments/workspace-1",
  task:
    "RESEARCH QUERY (verbatim, gospel):\n> " + query +
    "\n\nQUERY FILE: research/runs/cmp170hx-training-feasibility-a0d4fc/query.md" +
    "\n\nPIPELINE POSITION: You are step 2 (width-sweep fetcher) of the hyperresearch V8 pipeline. The orchestrator partitioned the URL queue into non-overlapping batches; you fetch ONLY the URLs in your batch. Do not search for additional URLs. After you return, the orchestrator runs a coverage check (step 2.5) and may dispatch wave 2 fetchers." +
    "\n\nYOUR INPUTS:\n- vault_tag: cmp170hx-training-feasibility-a0d4fc\n- batch_id: " + (i + 1) +
    "\n- urls:\n" + urls.map((u) => "  - " + u[0] + " (utility: " + u[1] + ")").join("\n") +
    "\n\nRUN DIRECTIVES:\n" + shim
}));

const results = await runs.all(tasks);
return results.map((r, i) => ({ batch: i + 1, ok: r.ok, output: r.output }));
