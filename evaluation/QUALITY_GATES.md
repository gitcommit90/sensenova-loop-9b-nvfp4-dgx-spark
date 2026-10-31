# Quality / performance gates (declared 2026-10-30 ~15:25 PDT, BEFORE any NVFP4 eval result was seen)

Eval: eval/ (300 fixed items: IFEval 60, GSM8K 80, MMLU-Pro 60, HumanEval 40, tools 20, TextVQA 40),
temperature 0, thinking off, same SGLang build/flags for reference (BF16 backbone, FP32 mHC) and candidate.

Quality (candidate vs BF16 reference, paired):
- G1 pooled accuracy (all 300 items, item-weighted): delta point estimate >= -3.0 pp AND paired-bootstrap 95% CI lower bound >= -7.0 pp.
- G2 no single task point-estimate drop > 8 pp (n=20-80 per task, so this is a coarse tripwire, not a significance test).
- G3 tool calling: 0 malformed tool calls beyond the reference's count; image task (TextVQA) drop <= 8 pp.
- G4 stability: 0 request errors, all outputs finite/non-degenerate (no repetition collapse in truncation counts beyond ref + 3).
- G5 MTP: mean accepted length (spec) with NVFP4 target >= 85% of the BF16-target value under identical spec settings.
Performance (same engine, same flags, fixed 1024-in/256-out, concurrency 1/2/4/8):
- P1 0 errors/OOM at c=1..8. P2 report decode tok/s, TTFT, aggregate throughput honestly; speedup is reported, not gated.
If a recipe fails G1-G5, try the next recipe (max 3: decoder-all -> MLP-only -> decoder w/ first/last layers BF16).
