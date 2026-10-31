# SenseNova-Loop-9B-Preview-NVFP4 — runtime v2 (published companion update)

Faster serving runtime for the **unchanged** published checkpoint
[`gitcommit90/SenseNova-Loop-9B-Preview-NVFP4`](https://huggingface.co/gitcommit90/SenseNova-Loop-9B-Preview-NVFP4) (weights, config and the v1 `runtime/` are untouched).
Same pinned SGLang image and PR #47281 overlay as v1; W4A4 NVFP4 (flashinfer cuDNN FP4 GEMMs) + MTP (NEXTN, 3 steps).

## Use
```bash
runtime-v2/serve_sglang.sh /path/to/SenseNova-Loop-9B-Preview-NVFP4        # default: MHC_MODE=fast + 64k draft vocab + MTP
MHC_MODE=exact runtime-v2/serve_sglang.sh <model>    # bitwise-identical target math to v1 (greedy outputs = v1)
MHC_MODE=reference TOKEN_MAP=none runtime-v2/serve_sglang.sh <model>   # = v1 runtime behaviour and speed
runtime-v2/serve_sglang.sh <model> --no-mtp           # target only (MTP still recommended, see results)
```
Env as in v1 (`PORT`, `HOST`, `MEMFRAC`, `CTX`, `MAXREQ`, `SERVED`, `NAME`) plus `MHC_MODE`, `TOKEN_MAP`, `SGL_CACHE`.
**Recommended conservative command:** `MHC_MODE=exact bash runtime-v2/serve_sglang.sh /path/to/model`.
The launcher’s existing default remains `fast`; choose explicitly. Exact describes target routing math, not a
guarantee of identical sampled sequences on every workload/batch/hardware. Fast may change greedy outputs.

Startup to ready: ~75 s (v1: ~90 s). Rollback: stop the container and start `bash runtime/serve_sglang.sh /path/to/model` — nothing is shared
except the JIT cache dir (`~/.cache/sensenova-sglang`, safe to delete).

## What changed (two things)
1. **Fused mHC router** (`overlay/python/sglang/srt/layers/sensenova_mhc_fused.py`, wired in `models/sensenova_loop.py`,
   diff: `patches/v2_mhc_fused_router.patch`). The original router ran FP32 log-space Sinkhorn (20 iterations on the 2x2 stream
   matrix) as ~300 tiny ATen kernels per call; 96 router calls per forward (24 looped layers x 2 visits x 2 routers) = ~39k kernels,
   ~60 ms of a 157 ms decode cycle. Now Triton:
   * Sinkhorn/sigmoid kernel: same operation order, 20 iterations from config, torch.logsumexp semantics (NaN-propagating max,
     inf-masked shift), precise libdevice exp/log/div, no FTZ: **bitwise equal** to eager (all T 1..4096, extreme values, CUDA graphs).
   * `MHC_MODE=fast` (default) additionally fuses the FP32 RMS-normalize + 8x8192 FP32 router linear + branch, and the merge.
     Still FP32 everywhere and the same dtype boundaries, but the FP32 dot-product summation order differs from cuBLAS
     (coefficient diff <= 6.6e-7; vs a float64 reference the fused result is *more* accurate: 2.4e-7 vs cuBLAS 6.6e-7).
     Not bitwise, so greedy text can diverge at near-ties (see correctness). +7..16% over `exact`.
   * Eager reference kept verbatim (`_forward_reference`/`_merge_reference`, `MHC_MODE=reference`).
2. **FR-Spec hot-vocab MTP draft head** (`--speculative-token-map tokenmap/hot_64k.pt`). The draft proposed tokens through the
   full 2 GB BF16 lm_head 3-4x per cycle (~35 ms). The draft now scores only the 65,536 most frequent tokens (+ all 33 special
   tokens) counted on TRAIN-split corpora (oasst2, glaive-fc-v2, mbpp, gsm8k train; 37 M tokens; 99.1% weighted coverage;
   `tokenmap/build_token_map.py`). **The target still verifies over the full vocabulary**. Vocabulary restriction applies only to drafting.
   Speculative decoding preserves the target distribution in exact arithmetic with correct verification; finite-precision
   kernels/batching can change actual outputs. Only greedy decoding was validated here; sampling equivalence was not tested. No weights changed (the draft head is a row subset of the same lm_head).

Nothing else: same image digest, same flags (cuDNN FP4, steps 3, mem 0.45, ctx 32768, max 16 running), no KV/state precision change,
no change to loop passes, layers, routing, Sinkhorn iterations, sampling, context or output budgets.

## Rejected after measurement (see LEDGER.md)
MTP steps 2/4/5 (no gain), flashinfer_cutlass FP4 (noise-level), Marlin W4A16+MTP (decode parity, 2x worse TTFT), 32k token map
(64k equal speed, better fidelity), no-MTP throughput profile (MTP wins on realistic text up to c=16; no-MTP only wins on
random-word synthetic at c=16).

## Results (full tables: RESULTS.md; screening history: LEDGER.md)
Per-user decode speed and aggregate throughput, W4A4 + MTP, v1 frozen -> v2 default (`fast`) [`exact`]:

| workload | c | aggregate tok/s | per-request decode tok/s p50 | TTFT p50 |
|---|---:|---|---|---|
| matched (1051 in / 256 out) | 1 | 12.0 -> **27.8** [26.0] | 12.4 -> **28.9** [27.3] | 0.74 -> 0.32 s |
| matched | 2 | 23.4 -> **49.2** [45.5] | 12.2 -> 28.4 | 0.89 -> 0.41 s |
| matched | 4 | 42.0 -> **87.0** [72.6] | 12.1 -> 25.4 | 1.84 -> 0.95 s |
| matched | 8 | 64.3 -> **119.4** [117.2] | 12.1 -> 19.8 | 5.02 -> 2.90 s |
| code (90 in / 256 out) | 1 | 15.6 -> **35.3** [32.8] | 16.0 -> 35.9 | 355 -> 134 ms |
| code | 8 | 97.4 -> **196.7** [172.7] | 14.1 -> 26.3 | 744 -> 332 ms |
| document (~4k in / 256 out) | 1 | 12.8 -> **29.1** [25.6] | 14.3 -> 33.3 | 2.03 -> 1.11 s |
| document | 8 | 52.4 -> **92.3** [70.4] | 9.0 -> 16.1 | 5.49 -> 3.91 s |

MTP acceptance unchanged within noise (code 2.43-2.51 -> 2.31-2.41; doc 2.16-2.26 -> 2.16-2.19). 0 errors; every request produced
exactly the requested 256 tokens. 300-item regression: frozen 251/300, v2_exact 249, v2_fast 247. These are small, noisy regression tests, not proof of equivalence.
The first fast MMLU-Pro run dropped 11.7 percentage points (70.0% → 58.3%); reruns varied substantially.
See RESULTS.md for every run. Fast also had 20 truncated outputs vs 14 frozen, above the original ref+3 tripwire.
**Fast does not pass every original per-task/stability gate on the first run; use exact for the conservative profile.**

## Correctness evidence
* `tests/test_mhc_fused.py` (result: `tests/test_mhc_fused.result.txt`): Sinkhorn kernel bitwise vs eager for T in
  {1..4096} (decode, MTP verify 4/16/32, prefill 1051/4096), logit scales 0.1..1e4, +-inf, NaN, 3e38, denormals, non-contiguous
  inputs, iterations 1/5/20/50, and under CUDA-graph capture/replay; exact router path bitwise end-to-end on real layer 4/15/27
  router weights; fast path coefficient diff <= 6.6e-7 (closer to fp64 than cuBLAS), bf16 branch identical.
* Greedy feature check `tests/greedy_outputs.py`: exact == frozen 12/12; fast valid on all 12 (text, thinking, code, zh, JSON, long,
  tool call, image).
* MTP verified throughout (acceptance logged per request); target-only vs MTP divergence characterised (inherent to v1 too).

## Remaining bottleneck (profile, c=1, exact mode, ~63-69 ms per speculative cycle; GPU ~100% busy, no launch gaps left)
* FP4 W4A4 GEMMs ~35 ms (55%): 6.8 GB of FP4 weights per forward because the 24 looped layers are read twice; cuDNN/CUTLASS
  already at 85-95% of the 273 GB/s DRAM roofline in isolation. Floor ~25 ms.
* Target BF16 lm_head 8.7 ms (2 GB, required for verification) — only reducible by quantizing lm_head (a weight change; needs approval).
* Draft (BF16 MTP layer + 64k head) ~13 ms; GDN/attention/norms ~6 ms; router linear ~5.8 ms in exact mode (~0.5 ms in fast).
* Honest ceiling: ~8.8 GB of weights per target forward => <=31 forwards/s at 100% bandwidth; with accept length 2.3-2.9 that is
  ~70-90 tok/s theoretical, realistically ~40-45 with this stack. Measured: 28-36 tok/s single-user. Reaching 60 tok/s single-user
  would need changes outside this brief: FP8/NVFP4 lm_head and/or draft weights (weight changes), a better-trained draft (higher
  acceptance), or a fused-weight persistent decode kernel. Each is a separate approval/effort decision.
