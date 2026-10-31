#!/usr/bin/env python
"""Paired comparison of two run_eval.py output dirs (reference vs candidate).

  .venv-eval/bin/python eval/compare.py eval/runs/bf16 eval/runs/nvfp4 [--json out.json]

Per task (on the intersection of item ids):
  * ref / cand accuracy, delta, paired bootstrap 95% CI of the delta (10k resamples, seed 0),
    discordant counts (ref-only correct / cand-only correct) and exact McNemar p-value;
  * reference agreement (label-free): fraction of items whose extracted answer == reference's;
  * identical raw outputs %, mean completion tokens ref/cand, truncation counts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_jsonl  # noqa: E402


def load(d):
    d = Path(d)
    out = {}
    for p in sorted(d.glob("raw_*.jsonl")):
        task = p.stem[4:]
        out[task] = {r["id"]: r for r in read_jsonl(p)}
    return out


def boot_ci(a, b, iters=10000, seed=0):
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    stats = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    return (stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1])


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def norm_out(r):
    return json.dumps([r.get("content") or "", r.get("tool_calls") or []], sort_keys=True)


def compare(ref_dir, cand_dir):
    ref, cand = load(ref_dir), load(cand_dir)
    res = {"ref": str(ref_dir), "cand": str(cand_dir), "tasks": {}}
    tot = {"n": 0, "ref": 0, "cand": 0, "agree": 0, "ident": 0, "a": [], "b": []}
    for task in ref:
        if task not in cand:
            continue
        ids = sorted(set(ref[task]) & set(cand[task]))
        a = [int(ref[task][i]["correct"]) for i in ids]
        b = [int(cand[task][i]["correct"]) for i in ids]
        n = len(ids)
        agree = sum(1 for i in ids if json.dumps(ref[task][i].get("agree_key"), sort_keys=True)
                    == json.dumps(cand[task][i].get("agree_key"), sort_keys=True))
        ident = sum(1 for i in ids if norm_out(ref[task][i]) == norm_out(cand[task][i]))
        ref_only = sum(1 for x, y in zip(a, b) if x and not y)
        cand_only = sum(1 for x, y in zip(a, b) if y and not x)

        def mtok(rs):
            t = [r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None]
            return round(sum(t) / len(t), 1) if t else None
        lo, hi = boot_ci(a, b)
        res["tasks"][task] = {
            "n": n, "ref_acc": sum(a) / n, "cand_acc": sum(b) / n, "delta": (sum(b) - sum(a)) / n,
            "delta_boot95": [lo, hi], "ref_only_correct": ref_only, "cand_only_correct": cand_only,
            "mcnemar_p": mcnemar_exact(ref_only, cand_only),
            "ref_agreement": agree / n, "identical_output": ident / n,
            "mean_tokens_ref": mtok([ref[task][i] for i in ids]),
            "mean_tokens_cand": mtok([cand[task][i] for i in ids]),
            "truncated_ref": sum(1 for i in ids if ref[task][i].get("finish_reason") == "length"),
            "truncated_cand": sum(1 for i in ids if cand[task][i].get("finish_reason") == "length"),
            "errors_cand": sum(1 for i in ids if cand[task][i].get("error")),
        }
        tot["n"] += n
        tot["ref"] += sum(a)
        tot["cand"] += sum(b)
        tot["agree"] += agree
        tot["ident"] += ident
        tot["a"] += a
        tot["b"] += b
    if tot["n"]:
        lo, hi = boot_ci(tot["a"], tot["b"])
        res["overall_micro"] = {"n": tot["n"], "ref_acc": tot["ref"] / tot["n"], "cand_acc": tot["cand"] / tot["n"],
                                "delta": (tot["cand"] - tot["ref"]) / tot["n"], "delta_boot95": [lo, hi],
                                "ref_agreement": tot["agree"] / tot["n"], "identical_output": tot["ident"] / tot["n"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = compare(args.ref_dir, args.cand_dir)
    hdr = f"{'task':10s} {'n':>4s} {'ref':>6s} {'cand':>6s} {'delta':>7s} {'boot95':>17s} {'r-/c+':>6s} {'mcnemar':>7s} {'agree':>6s} {'ident':>6s} {'tok r/c':>13s}"
    print(hdr)
    rows = list(res["tasks"].items()) + ([("ALL(micro)", res["overall_micro"])] if "overall_micro" in res else [])
    for t, s in rows:
        print(f"{t:10s} {s['n']:4d} {s['ref_acc']:6.3f} {s['cand_acc']:6.3f} {s['delta']:+7.3f} "
              f"[{s['delta_boot95'][0]:+.3f},{s['delta_boot95'][1]:+.3f}] "
              f"{str(s.get('ref_only_correct', '')) + '/' + str(s.get('cand_only_correct', '')):>6s} "
              f"{s.get('mcnemar_p', float('nan')):7.3f} {s['ref_agreement']:6.3f} {s['identical_output']:6.3f} "
              f"{str(s.get('mean_tokens_ref'))+'/'+str(s.get('mean_tokens_cand')):>13s}")
    out = args.json or str(Path(args.cand_dir) / "compare_vs_ref.json")
    Path(out).write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
