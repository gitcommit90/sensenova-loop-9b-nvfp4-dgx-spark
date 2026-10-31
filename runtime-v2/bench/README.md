# Reproduce runtime v2 measurements

Run one GPU workload at a time. Historical `runs/` contains the original final JSONs, per-item evaluation outputs,
screening history and logs; paths inside archived JSONs may refer to the original machine. No trace files or
session-local GPU-lock/orchestration wrappers are required for serving.

From the repository/snapshot root, after downloading model files as in the main README:

```bash
python3 -m venv .venv-eval
.venv-eval/bin/pip install -r evaluation/harness/requirements.lock
MODEL=/absolute/path/to/SenseNova-Loop-9B-Preview-NVFP4
# Select one runtime, wait for readiness, then measure. Stop it before switching to another.
bash runtime/serve_sglang.sh "$MODEL"                          # frozen v1
# MHC_MODE=exact bash runtime-v2/serve_sglang.sh "$MODEL"      # conservative optimized
# MHC_MODE=fast bash runtime-v2/serve_sglang.sh "$MODEL"       # fastest, different summation order
curl -fsS http://127.0.0.1:30080/v1/models
mkdir -p results
.venv-eval/bin/python evaluation/harness/perf_bench.py --base-url http://127.0.0.1:30080/v1 \
  --model sensenova-loop-9b-nvfp4 --concurrency 1,2,4,8 --num-requests 8 --warmup 2 \
  --input-tokens 1024 --output-tokens 256 --tokenizer "$MODEL/tokenizer.json" --out results/matched.json
.venv-eval/bin/python runtime-v2/bench/screen.py --base-url http://127.0.0.1:30080 \
  --model sensenova-loop-9b-nvfp4 --concurrency 1,4,8 --output-tokens 256 --prompt-kind code \
  --tokenizer "$MODEL/tokenizer.json" --out results/code.json
# Repeat with --prompt-kind doc for ~4K input tokens.
# Build the pinned small evaluation set into evaluation/harness/data/ before running quality tests:
.venv-eval/bin/python evaluation/harness/build_eval_data.py
.venv-eval/bin/python evaluation/harness/run_eval.py --base-url http://127.0.0.1:30080/v1 \
  --model sensenova-loop-9b-nvfp4 --out results/quality --concurrency 16
```

For the 12-case feature test, supply `--image` (the recorded image was TextVQA validation image 34730):
`python runtime-v2/tests/greedy_outputs.py results/features.json --image /path/to/image.jpg`.
Kernel differential tests run in the pinned SGLang container; see `tests/test_mhc_fused.py`.
All run counts and exact prompts/lengths are in saved JSONs. Code/doc final tests use 4/8/16 requests at c=1/4/8;
matched tests use 8 per level. Screening c=16 results are not final matched-sweep numbers.
