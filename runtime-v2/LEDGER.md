# Experiment ledger — SenseNova-Loop-9B NVFP4 runtime optimization (GB10, 2026-10-30)
Screen = bench/screen_set.sh: synthetic(1062 in)/code(90 in) prompts, 256 out, c=1 (4 req) + c=4 (8 req), unique prompts, temp 0, ignore_eos.
Greedy check = tests/greedy_outputs.py 12 prompts (text/math/think/code/zh/json/long/tool/image) vs frozen first run on a fresh server.

| # | Hypothesis | Change | Correctness | synth c1 / c4 | code c1 / c4 | Keep? |
|---|---|---|---|---|---|---|
| 0 | baseline | frozen release runtime, W4A4+MTP(3) | ref; same-server repeat 9/12 (prefix-cache rerun) | 11.6 / 40.8 | 15.4 / 58.4 | ref |
| P | profile c1 | — | — | cycle 157 ms: verify 117 (mHC ~60 incl gaps, FP4 35, lm_head 8.7, router cuBLAS 5.5), draft 27.5+13.7 (BF16 lm_head x3) | | |
| 1 | mHC is launch-bound (39k kernels/fwd) | fused Triton Sinkhorn (bitwise), cuBLAS linear + torch merge kept ("exact") | 11/12 identical (only random tool-call id) | 18.7 / 62.9 | 24.8 / 87.0 | KEEP |
| 1b | router linear+merge also fused | single-kernel rmsnorm+linear+Sinkhorn+branch, fused merge ("full") | 2/12 identical; FP32 sum-order diffs (closer to fp64 than cuBLAS) flip near-ties | 21.2 / 73.8 | 25.7 / 92.6 | option (not bitwise) |
| 2 | draft lm_head (2 GB BF16, 3x/cycle) dominates draft | FR-Spec hot-vocab draft head --speculative-token-map (train-split corpora), target verify stays full-vocab | 32k: 10/12; 64k: 11/12 | 32k 27.6/72.0; 64k 29.0/80.7 | 32k 32.9/105.8; 64k 32.2/106.7 | KEEP 64k |
| X | target-only vs MTP (same math) | --no-mtp, exact | 1/12 identical vs MTP: M=1 decode vs M=4 verify kernels differ -> inherent spec/target near-tie flips (also true for frozen) | 19.5 / 63.0 | 20.0 / 70.2 | info |
| P2 | profile exact+tm64k c1 | — | — | cycle ~69 ms: verify 56 (FP4 34.8, lm_head 8.7, router cuBLAS 5.8), draft 8.6 + extend 4.3 | | |
| 3 | more/fewer MTP steps pay after draft got cheap | steps 2/4/5 (draft tokens = steps+1), exact+tm64k | — | s2 26.4/79.0; s4 25.9/68.5 | s2 32.8/108.9; s4 30.8/104.1; s5 31.6/94.3 | REJECT (keep 3) |
| 4 | cutlass FP4 faster on down_proj (microbench -7% FP4) | --fp4-gemm-backend flashinfer_cutlass | 10/12 (different FP4 accumulation) | 26.3/80.8 | 33.2/108.6 | REJECT (noise-level) |
| 5 | fused router linear+merge worth it once mHC is not dominant | full fused (default env) + tm64k | 2/12 (FP32 sum order, see 1b) | 31.0/90.6 (+7%/+12% vs exact) | 37.5/117.5 (+16%/+10%) | FINALIST (needs 300-item) |
| 6 | Marlin W4A16 + MTP better single-user | marlin + SILU fusion off, exact+tm64k | 3/12 (W4A16 numerics) | 27.9/77.4, TTFT c4 2.2 s | 33.8/107.7 | REJECT (parity decode, worse TTFT) |
| 7 | no-MTP throughput profile for high concurrency | --no-mtp vs MTP, fast+tm64k, c=8/16 | — | c8 129.4 vs 130.2; c16 194.3 vs 171.4 | c8 150.6 vs 198.4; c16 258.0 vs 295.0 | REJECT (single MTP profile) |
| F | final validation | frozen_v1 vs v2_fast vs v2_exact (RESULTS.md) | 300-item: 251 / 247 / 249; exact == frozen 12/12; mmlu_pro reruns show frozen spans 0.65-0.70 | matched c1 12.0 -> 27.8 / 26.0; c8 64.3 -> 119.4 / 117.2 | code c1 15.6 -> 35.3; c8 97.4 -> 196.7 | SHIP v2 (default fast, exact switch) |
