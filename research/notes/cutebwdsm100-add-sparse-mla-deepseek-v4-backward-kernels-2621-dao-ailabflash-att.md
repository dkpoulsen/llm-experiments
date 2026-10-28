---
title: '[Cute,Bwd,Sm100] add sparse MLA (Deepseek v4) backward kernels (#2621) · Dao-AILab/flash-attention@940cd96
  · GitHub'
id: cutebwdsm100-add-sparse-mla-deepseek-v4-backward-kernels-2621-dao-ailabflash-att
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-4
created: '2026-10-27T14:37:12.861711Z'
updated: '2026-10-27T19:00:31.783847Z'
source: https://github.com/Dao-AILab/flash-attention/commit/940cd9680f3315f2f06b43ab5bea2c2cf2d96806
source_domain: github.com
fetched_at: '2026-10-27T14:37:12.860244Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: Dao-AILab/flash-attention commit 940cd96 '[Cute,Bwd,Sm100] add sparse MLA
  (Deepseek v4) backward kernels (#2621)', 2026-06-20, +5685/-1066 across 12 files.
  The added files are `flash_bwd_mla_sm100.py`, `flash_bwd_mla_dq_dqv_sm100.py`, `flash_bwd_mla_dk_sm100.py`,
  `flash_fwd_mla_sm100.py`, `topk_gather_kv.py`, `flash_bwd_preprocess.py` — the sparse
  MLA forward+backward is SM100 (Blackwell) only. No sm_80, no sm_90. This is the
  upstream flash-attention project's sparse-MLA training path, and it confirms cc-4's
  claim that the training-capable DSA kernels are Blackwell-or-newer.
---

[Cute,Bwd,Sm100] add sparse MLA (Deepseek v4) backward kernels (#2621) · Dao-AILab/flash-attention@940cd96 · GitHub
Skip to content
You signed in with another tab or window.
Reload
to refresh your session.
You signed out in another tab or window.
Reload
to refresh your session.
You switched accounts on another tab or window.
Reload
to refresh your session.
Dismiss alert
File tree
Expand file tree
Collapse file tree
Open diff view settings
Filter options
benchmarks
benchmark_attn.py
flash_attn/cute
bench_utils.py
flash_bwd_mla_dk_sm100.py
flash_bwd_mla_dq_dqv_sm100.py
flash_bwd_mla_sm100.py
flash_bwd_preprocess.py
flash_fwd_mla_sm100.py
interface.py
named_barrier.py
tile_scheduler.py
topk_gather_kv.py
tests/cute
test_flash_attn.py
Expand file tree
Collapse file tree
Open diff view settings
Collapse file
`‎benchmarks/benchmark_attn.py‎`
Copy file name to clipboard
Expand all lines: benchmarks/benchmark_attn.py
+
18
-
9
Lines changed: 18 additions & 9 deletions
Original file line number
Diff line number
Diff line change
`@@ -146,6 +146,7 @@ def setup_fa4(ctx):`
`146`
`146`
` gather_kv_indices = ctx.get("gather_kv_indices")`
`147`
`147`
` k_use = ctx.get("k_paged", k) if ctx["page_size"] is not None else k`
`148`
`148`
` v_use = ctx.get("v_paged", v) if ctx["page_size"] is not None else v`
`149`
`+ num_splits = ctx["num_splits"]`
`149`
`150`
` if ctx["varlen"]:`
`150`
`151`
` qu = ctx["q_unpad"]`
`151`
`152`
` ku = ctx.get("k_paged", ctx["k_unpad"]) if ctx["page_size"] is not None else ctx["k_unpad"]`
`@@ -154,17 +155,17 @@ def setup_fa4(ctx):`
`154`
`155`
` csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]`
`155`
`156`
` pt = ctx["page_table"]`
`156`
`157`
` gather_kv_indices_unpad = ctx.get("gather_kv_indices_unpad")`
`157`
`- fwd_fn = lambda: flash_attn_varlen_func_python(qu, ku, vu, qvu, csq, csk, page_table=pt, causal=causal, window_size=window_size, softcap=softcap, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices_unpad)`
`158`
`+ fwd_fn = lambda: flash_attn_varlen_func_python(qu, ku, vu, qv=qvu, cu_seqlens_q=csq, cu_seqlens_k=csk, page_table=pt, causal=causal, window_size=window_size, softcap=softcap, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices_unpad, num_splits=num_splits)`
`158`
`159`
` else:`
`159`
`- fwd_fn = lambda: flash_attn_func_python(q, k_use, v_use, qv=qv, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices)`
`160`
`+ fwd_fn = lambda: flash_attn_func_python(q, k_use, v_use, qv=qv, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices, num_splits=num_splits)`
`160`
`161`
` bwd_fn = None`
`161`
`162`
` if ctx["has_backward"] and ctx["dtype"] != torch.float8_e4m3fn:`
`162`
`163`
` if ctx["varlen"]:`
`163`
`164`
` qu, ku, vu, gu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"], ctx["g_unpad"]`
`164`
`165`
` qvu = ctx["qv_unpad"]`
`165`
`166`
` csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]`
`166`
`167`
` gather_kv_indices_unpad = ctx.get("gather_kv_indices_unpad")`
`167`
`- bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_python(qu, ku, vu, qvu, csq, csk, causal=causal, softcap=softcap, deterministic=deterministic, gather_kv_indices=gather_kv_indices_unpad), gu, [qu, ku, vu, qvu])`
`168`
`+ bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_python(qu, ku, vu, qv=qvu, cu_seqlens_q=csq, cu_seqlens_k=csk, causal=causal, softcap=softcap, deterministic=deterministic, gather_kv_indices=gather_kv_indices_unpad), gu, [qu, ku, vu, qvu])`
`168`
`169`
` else:`
`169`
`170`
` bwd_fn = _make_bwd_fn(lambda: flash_attn_func_python(q, k, v, qv=qv, causal=causal, softcap=softcap, deterministic=deterministic, gather_kv_indices=gather_kv_indices), g, [q, k, v, qv])`
`170`
`171`
` return fwd_fn, bwd_fn`
`@@ -320,9 +321,10 @@ def parse_args():`
`320`
`321`
` parser.add_argument('--backend', type=csv_strs, default=['all'],`
`321`
`322`
` help='Which backends to benchmark, comma-separated (choices: all,standard,fa2,fa3,fa4,cudnn)')`
`322`
`323`
` parser.add_argument('--gather-kv', type=int, default=None,`
`323`
`- help='kv sparsity length for MLA (hdim=64, hdim_v=512 only). '`
`324`
`+ help='kv sparsity length (supported: hdim-hdim_v=64-512 or 512-512 with shared-kv).'`
`324`
`325`
` 'When set, passes random kv indices (without repeats) to FA4 and uses gather-kv as '`
`325`
`326`
` 'the effective KV length for flops/bandwidth accounting.')`
`327`
`+ parser.add_argument('--shared-kv', action='store_true', help='shared KV mode')`
`326`
`328`
` parser.add_argument('--num-splits', type=int, default=0,`
`327`
`329`
` help='Override kernel num_splits heuristic. 0 = auto (default). '`
`328`
`330`
` '>1 forces SplitKV with that many splits.')`
`@@ -379,6 +381,7 @@ def main():`
`379`
`381`
` seqlen_q_list = resolve_seqlen_q_list(seqlen_list, args.seqlen_q)`
`380`
`382`
` varlen = args.varlen`
`381`
`383`
` gather_kv_length = args.gather_kv`
`384`
`+ shared_kv = args.shared_kv`
`382`
`385`
`383`
`386`
` # Filter backends to those requested and available`
`384`
`387`
` enabled = set(args.backend)`
`@@ -426,6 +429,8 @@ def main():`
`426`
`429`
` v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)`
`427`
`430`
` qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward) if has_qv else None`
`428`
`431`
` q, k, v, qv = [x.detach().to(dtype).requires_grad_(has_backward) if x is not None else None for x in [q, k, v, qv]]`
`432`
`+ if shared_kv:`
`433`
`+ v = k`
`429`
`434`
` g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)`
`430`
`435`
`431`
`436`
` # Varlen tensors`
`@@ -435,6 +440,8 @@ def main():`
`435`
`440`
` g_unpad = rearrange(g.detach(), "b s h d -> (b s) h d")`
`436`
`441`
` cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q`
`437`
`442`
` cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen if page_size is None else None`
`443`
`+ if shared_kv:`
`444`
`+ v_unpad = k_unpad`
`438`
`445`
`439`
`446`
` # Paged KV tensors`
`440`
`447`
` k_paged = v_paged = page_table = None`
`@@ -447,7 +454,7 @@ def main():`
`447`
`454`
` # kv sparsity indices — only meaningful for MLA (hdim=64, hdim_v=512)`
`448`
`455`
` gather_kv_indices = gather_kv_indices_unpad = None`
`449`
`456`
` gather_kv_eff = None # effective KV length for this config`
`450`
`- if gather_kv_length is not None and has_qv:`
`457`
`+ if gather_kv_length is not None:`
`451`
`458`
` assert gather_kv_length <= seqlen, f"--gather_kv {gather_kv_length} > seqlen_kv {seqlen}"`
`452`
`459`
` gather_kv_indices = (`
`453`
`460`
` torch.rand(batch_size, seqlen_q, gather_kv_length, device=device)`
`@@ -533,13 +540,15 @@ def main():`
`533`
`540`
` headdim, headdim_v, causal, seqlen_q, seqlen, batch_size, nheads, nheads_kv, gather_kv_eff = cfg`
`534`
`541`
` has_qv = (headdim == 64 and headdim_v == 512)`
`535`
`542`
` seqlen_k_eff = gather_kv_eff if gather_kv_eff is not None else seqlen`
`536`
`- nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen_k_eff, headdim, headdim_v, causal=causal, has_qv=has_qv)`
`543`
`+ causal_eff = False if gather_kv_eff is not None else causal`
`544`
`+ nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen_k_eff, headdim, headdim_v, causal=causal_eff, has_qv=has_qv)`
`537`
`545`
` dtype_bytes = 1 if dtype == torch.float8_e4m3fn else 2`
`538`
`546`
` if direction == "FWD":`
`539`
`- nbytes = bandwidth_fwd_bytes(batch_size, nheads, nheads_kv, seqlen_q, seqlen_k_eff,`
`540`
`- headdim, headdim_v, dtype_bytes=dtype_bytes, has_qv=has_qv)`
`547`
`+ nbytes = bandwidth_fwd_bytes(batch_size, nheads, nheads_kv, seqlen_q, seqlen,`
`548`
`+ headdim, headdim_v, dtype_bytes=dtype_bytes,`
`549`
`+ has_qv=has_qv, shared_kv=shared_kv)`
`541`
`550`
` else:`
`542`
`- nbytes = bandwidth_bwd_bytes(batch_size, nheads, nheads_kv, seqlen_q, seqlen_k_eff,`
`551`
`+ nbytes = bandwidth_bwd_bytes(batch_size, nheads, nheads_kv, seqlen_q, seqlen,`
`543`
`552`
` headdim, headdim_v, dtype_bytes=dtype_bytes)`
`544`
`553`
` hdim_str = str(headdim) if headdim == headdim_v else f"{headdim}-{headdim_v}"`
`545`
`554`
` row = f"{hdim_str:>9} {str(causal):>6} {batch_size:>5}"`
Collapse file
`‎flash_attn/cute/bench_utils.py‎`
Copy file name to clipboard
Expand all lines: flash_attn/cute/bench_utils.py
+
11
-
2
Lines changed: 11 additions & 2 deletions
Original file line number
Diff line number
Diff line change
`@@ -51,13 +51,22 @@ def flops(`
`51`
`51`
`52`
`52`
`53`
`53`
`def bandwidth_fwd_bytes(`
`54`
`- batch, nheads, nheads_kv, seqlen_q, seqlen_k, headdim, headdim_v, dtype_bytes=2, has_qv=False`
`54`
`+ batch,`
`55`
`+ nheads,`
`56`
`+ nheads_kv,`
`57`
`+ seqlen_q,`
`58`
`+ seqlen_k,`
`59`
`+ headdim,`
`60`
`+ headdim_v,`
`61`
`+ dtype_bytes=2,`
`62`
`+ has_qv=False,`
`63`
`+ shared_kv=False,`
`55`
`64`
`):`
`56`
`65`
` """HBM traffic for one attention pass: read Q,K,V + write O."""`
`57`
`66`
` q = batch * nheads * seqlen_q * headdim`
`58`
`67`
` qv = batch * nheads * seqlen_q * headdim_v if has_qv else 0`
`59`
`68`
` k = batch * nheads_kv * seqlen_k * headdim`
`60`
`- v = batch * nheads_kv * seqlen_k * headdim_v`
`69`
`+ v = batch * nheads_kv * seqlen_k * headdim_v if not shared_kv else 0`
`61`
`70`
` o = batch * nheads * seqlen_q * headdim_v`
`62`
`71`
` return (q + qv + k + v + o) * dtype_bytes`
`63`
`72`
0
commit comments
Comments
0
(
0
)
You can’t perform that action at this time.