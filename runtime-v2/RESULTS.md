# Runtime v2 — final validation results (GB10, 2026-10-31, same session, clocks 2.4 GHz, 47-65 C)

frozen_v1 = published release runtime (W4A4 + MTP, eager mHC). v2_fast = default launcher. v2_exact = MHC_MODE=exact. All W4A4 NVFP4 + MTP(3), identical flags/budgets, unique prompts (no prefix-cache hits), temp 0.

### Matched workload (eval/perf_bench.py: ~1051 prompt tok, 256 out, unique prompts, 8 req/level, 2 warmup)

| runtime | c | aggregate tok/s | per-req decode tok/s p50 | TTFT p50 / p90 (s) | ITL p50 / p90 (ms) | E2E p50 / p90 (s) | out tok min-max | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 1 | 12.0 | 12.4 | 0.74 / 0.74 | 81 / 87 | 21.3 / 22.8 | 256-256 | 0 |
| frozen_v1 | 2 | 23.4 | 12.2 | 0.89 / 1.27 | 82 / 89 | 21.9 / 23.5 | 256-256 | 0 |
| frozen_v1 | 4 | 42.0 | 12.1 | 1.84 / 2.78 | 82 / 92 | 22.9 / 24.4 | 256-256 | 0 |
| frozen_v1 | 8 | 64.3 | 12.1 | 5.02 / 5.11 | 82 / 94 | 26.0 / 29.1 | 256-256 | 0 |
| v2_fast | 1 | 27.8 | 28.9 | 0.32 / 0.34 | 35 / 39 | 9.2 / 10.2 | 256-256 | 0 |
| v2_fast | 2 | 49.2 | 28.4 | 0.41 / 0.57 | 35 / 42 | 9.5 / 11.1 | 256-256 | 0 |
| v2_fast | 4 | 87.0 | 25.4 | 0.95 / 1.23 | 39 / 44 | 10.9 / 12.2 | 256-256 | 0 |
| v2_fast | 8 | 119.4 | 19.8 | 2.90 / 2.91 | 51 / 55 | 15.8 / 17.0 | 256-256 | 0 |
| v2_exact | 1 | 26.0 | 27.3 | 0.51 / 0.51 | 37 / 39 | 9.8 / 10.5 | 256-256 | 0 |
| v2_exact | 2 | 45.5 | 25.7 | 0.58 / 0.98 | 39 / 47 | 10.5 / 12.6 | 256-256 | 0 |
| v2_exact | 4 | 72.6 | 21.7 | 1.15 / 1.81 | 46 / 50 | 12.7 / 14.0 | 256-256 | 0 |
| v2_exact | 8 | 117.2 | 21.2 | 4.35 / 4.36 | 47 / 51 | 16.4 / 17.4 | 256-256 | 0 |

### code: short interactive coding/reasoning prompt (~90 tok in, 256 out)

| runtime | c | n | aggregate tok/s | per-req decode p50 | TTFT p50 (ms) | ITL p50 / p95 (ms) | MTP accept len | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 1 | 4 | 15.6 | 16.0 | 355 | 63 / 66 | 2.51 | 0 |
| frozen_v1 | 4 | 8 | 56.3 | 15.0 | 615 | 67 / 70 | 2.45 | 0 |
| frozen_v1 | 8 | 16 | 97.4 | 14.1 | 744 | 71 / 81 | 2.43 | 0 |
| v2_fast | 1 | 4 | 35.3 | 35.9 | 134 | 28 / 29 | 2.31 | 0 |
| v2_fast | 4 | 8 | 116.7 | 30.9 | 264 | 32 / 34 | 2.33 | 0 |
| v2_fast | 8 | 16 | 196.7 | 26.3 | 332 | 38 / 41 | 2.41 | 0 |
| v2_exact | 1 | 4 | 32.8 | 33.4 | 148 | 30 / 31 | 2.37 | 0 |
| v2_exact | 4 | 8 | 106.5 | 29.0 | 282 | 35 / 38 | 2.41 | 0 |
| v2_exact | 8 | 16 | 172.7 | 23.4 | 398 | 43 / 46 | 2.37 | 0 |

### doc: long document (~4k tok in, 256 out)

| runtime | c | n | aggregate tok/s | per-req decode p50 | TTFT p50 (ms) | ITL p50 / p95 (ms) | MTP accept len | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 1 | 4 | 12.8 | 14.3 | 2029 | 70 / 74 | 2.26 | 0 |
| frozen_v1 | 4 | 8 | 36.9 | 12.1 | 5490 | 83 / 108 | 2.16 | 0 |
| frozen_v1 | 8 | 16 | 52.4 | 9.0 | 5491 | 111 / 156 | 2.18 | 0 |
| v2_fast | 1 | 4 | 29.1 | 33.3 | 1111 | 30 / 31 | 2.19 | 0 |
| v2_fast | 4 | 8 | 69.9 | 25.0 | 4012 | 40 / 51 | 2.16 | 0 |
| v2_fast | 8 | 16 | 92.3 | 16.1 | 3909 | 62 / 85 | 2.16 | 0 |
| v2_exact | 1 | 4 | 25.6 | 30.6 | 1791 | 33 / 33 | 2.26 | 0 |
| v2_exact | 4 | 8 | 56.1 | 20.3 | 4736 | 49 / 64 | 2.13 | 0 |
| v2_exact | 8 | 16 | 70.4 | 14.2 | 7989 | 71 / 121 | 2.16 | 0 |

### 300-item regression (thinking off, temp 0, eval concurrency 16)

| runtime | ifeval | gsm8k | mmlu_pro | humaneval | tools | vqa | pooled | truncated | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 0.767 | 0.950 | 0.700 | 0.825 | 0.900 | 0.900 | 251/300 = 0.837 | 14 | 0 |
| v2_fast | 0.817 | 0.963 | 0.583 | 0.775 | 0.900 | 0.925 | 247/300 = 0.823 | 20 | 0 |
| v2_exact | 0.750 | 0.975 | 0.683 | 0.800 | 0.900 | 0.875 | 249/300 = 0.830 | 9 | 0 |

### MMLU-Pro run-to-run spread (identical settings, no changes between reps; 60 items, 1024-token budget)

| runtime | runs (acc) | mean | truncated per run |
|---|---|---:|---|
| frozen_v1 | 0.700, 0.700, 0.650 | 0.683 | 9, 7, 10 |
| v2_fast | 0.583, 0.717, 0.650 | 0.650 | 14, 10, 9 |
| v2_exact | 0.683 | — | 6 |
| BF16 original (v1 eval) | 0.650 | — | 13 |

The first v2_fast mmlu_pro result (8 losses / 1 gain vs frozen, McNemar p=0.039) did not reproduce: the same config scored 0.717 and
0.650 on reruns, and frozen itself spans 0.650-0.700. Losses are dominated by long CoT hitting the 1024-token budget; v2_fast's mean
length (~600-618 tok) is between frozen (~572-594) and BF16 (622). Pooled 300: frozen 251, v2_exact 249, v2_fast 247 (delta -1.3 pp,
bootstrap 95% CI [-4.3, +1.7]).

### Greedy feature check (12 prompts: text x2, math, thinking, code x2, translate, zh, JSON, long, tool call, image)
* v2_exact vs frozen_v1 (same validation round): **12/12 identical** (tool-call function + arguments; ids are random).
* v2_fast vs frozen_v1: 3/12 identical; all 12 outputs valid (tool call correct, image description correct, thinking present).
* Reference variance: frozen vs frozen across rounds 10/12; frozen target-only vs frozen MTP 1/12 (decode M=1 vs verify M=4 kernels).

### Startup / memory (same --mem-fraction-static 0.45)
| | frozen_v1 | v2_fast |
|---|---:|---:|
| launch -> ready | 92 s | 77 s |
| target-verify CUDA graph capture | 22.4 s, 6.16 GB | 4.4 s, 1.14 GB |
| KV capacity (tokens) | 402,460 | 393,742 (-2%: 0.5 GB sliced draft head) |
| 300-item eval wall time | 549 s | 332 s |
