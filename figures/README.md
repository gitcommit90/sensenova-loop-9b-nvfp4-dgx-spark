# Reproducible publication figures

House style: [figures4papers Scientific Figure Making](https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making).

```bash
python3 -m venv .venv-figures
.venv-figures/bin/pip install -r figures/requirements.txt
.venv-figures/bin/python figures/plot_benchmarks.py
```

No hand-entered benchmark values: performance is read from `evaluation/perf_sglang_*.json`, quality from
`evaluation/compare_vs_ref.json`. The header and captions identify the October 30, 2026 functional baseline.
Performance panels distinguish median per-request decode, time to first token, and aggregate output throughput.
Quality panels show task scores and paired-bootstrap 95% intervals from the original saved comparisons.
These intervals reflect this small item set, not run-to-run throughput variation or full-benchmark accuracy.

The originals are 300-DPI PNG plus vector PDF and SVG. No new GPU benchmarks were run to make these figures.

## Runtime v2 final figures

`python figures/plot_runtime_v2.py` reads only `runtime-v2/runs/final_*/` JSONs.
Original v1 figures remain unchanged. New charts separate final matched runs from screening, and
show the first complete quality runs and paired intervals, including fast-profile regressions.
