#!/usr/bin/env python
"""Select the FIXED eval items from pinned dataset revisions and write them to eval/data/.

Run once (needs HF access only if eval/data/raw is missing; see download list in DATASETS):
    .venv-eval/bin/python eval/build_eval_data.py
Outputs: eval/data/<task>.jsonl, eval/data/images/*.jpg, eval/data/MANIFEST.json (with sha256s).
Re-running produces byte-identical files (fixed seeds, sorted inputs).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pyarrow.parquet as pq  # noqa: E402

import ifeval_lite  # noqa: E402
from common import EVAL_VERSION, write_jsonl  # noqa: E402
from tool_cases import CASES  # noqa: E402

DATA = HERE / "data"
RAW = DATA / "raw"
SEED = 20261030
N = {"ifeval": 60, "gsm8k": 80, "mmlu_pro": 60, "humaneval": 40, "tools": len(CASES), "vqa": 40}

DATASETS = {
    "ifeval": dict(repo="google/IFEval", rev="966cd89545d6b6acfd7638bc708b98261ca58e84",
                   file="ifeval_input_data.jsonl", split="train (only split; = the IFEval eval set)",
                   license="Apache-2.0"),
    "gsm8k": dict(repo="openai/gsm8k", rev="740312add88f781978c0658806c59bc2815b9866",
                  file="main/test-00000-of-00001.parquet", split="main/test", license="MIT"),
    "mmlu_pro": dict(repo="TIGER-Lab/MMLU-Pro", rev="b189ec765aa7ed75c8acfea42df31fdae71f97be",
                     file="data/test-00000-of-00001.parquet", split="test", license="MIT"),
    "humaneval": dict(repo="openai/openai_humaneval", rev="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                      file="openai_humaneval/test-00000-of-00001.parquet", split="test", license="MIT"),
    "tools": dict(repo="(hand-written, eval/tool_cases.py)", rev="in-repo", file="eval/tool_cases.py",
                  split="n/a", license="Apache-2.0 (this project); BFCL-style categories, no BFCL text"),
    "vqa": dict(repo="lmms-lab/textvqa", rev="9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                file="data/validation-00000-of-00003.parquet",
                split="validation (shard 0 of 3, 1667 of 5000 rows)",
                license="TextVQA annotations CC-BY-4.0; images from OpenImages (CC-BY-2.0, per-image Flickr attribution)"),
}


def raw_path(task):
    d = DATASETS[task]
    return RAW / d["repo"].replace("/", "__") / d["file"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(rows, n, task):
    rng = random.Random(f"{SEED}-{task}")
    idx = sorted(rng.sample(range(len(rows)), n))
    return idx


def build():
    DATA.mkdir(exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    out = {}

    # IFEval: restrict to prompts whose instructions are all supported by ifeval_lite
    rows = [json.loads(l) for l in open(raw_path("ifeval"))]
    rows.sort(key=lambda r: r["key"])
    elig = [r for r in rows if all(i in ifeval_lite.SUPPORTED for i in r["instruction_id_list"])]
    sel = [elig[i] for i in pick(elig, N["ifeval"], "ifeval")]
    out["ifeval"] = [dict(id=f"ifeval_{r['key']}", key=r["key"], prompt=r["prompt"],
                          instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"]) for r in sel]
    DATASETS["ifeval"]["eligible_pool"] = f"{len(elig)}/{len(rows)} prompts with only supported instruction ids"

    rows = pq.read_table(raw_path("gsm8k")).to_pylist()
    idx = pick(rows, N["gsm8k"], "gsm8k")
    out["gsm8k"] = [dict(id=f"gsm8k_test_{i}", source_index=i, question=rows[i]["question"],
                         answer=rows[i]["answer"].split("####")[-1].strip().replace(",", ""),
                         solution=rows[i]["answer"]) for i in idx]

    rows = pq.read_table(raw_path("mmlu_pro"), columns=["question_id", "question", "options", "answer",
                                                          "category", "src"]).to_pylist()
    rows.sort(key=lambda r: r["question_id"])
    idx = pick(rows, N["mmlu_pro"], "mmlu_pro")
    out["mmlu_pro"] = [dict(id=f"mmlupro_{rows[i]['question_id']}", **rows[i]) for i in idx]

    rows = pq.read_table(raw_path("humaneval")).to_pylist()
    idx = pick(rows, N["humaneval"], "humaneval")
    out["humaneval"] = [dict(id=rows[i]["task_id"].replace("/", "_"), **rows[i]) for i in idx]

    out["tools"] = [dict(c) for c in CASES]

    tbl = pq.read_table(raw_path("vqa"), columns=["question_id", "question", "answers", "image_id",
                                                    "image", "flickr_original_url"])
    qids = tbl.column("question_id").to_pylist()
    order = sorted(range(len(qids)), key=lambda i: qids[i])
    # only questions with a consensus answer (some normalized answer given by >=3 of 10 annotators)
    from collections import Counter
    from tasks import vqa_norm
    answers = tbl.column("answers").to_pylist()
    order = [i for i in order if Counter(vqa_norm(a) for a in answers[i]).most_common(1)[0][1] >= 3]
    DATASETS["vqa"]["eligible_pool"] = f"{len(order)}/{len(qids)} questions with a >=3-annotator consensus answer"
    idx = [order[i] for i in pick(order, N["vqa"], "vqa")]
    items = []
    for i in idx:
        r = tbl.slice(i, 1).to_pylist()[0]
        img = r["image"]["bytes"]
        ext = "png" if img[:4] == b"\x89PNG" else "jpg"
        rel = f"images/textvqa_val_{r['question_id']}.{ext}"
        (DATA / rel).write_bytes(img)
        items.append(dict(id=f"textvqa_val_{r['question_id']}", question_id=r["question_id"],
                          image_id=r["image_id"], image=rel, image_sha256=hashlib.sha256(img).hexdigest(),
                          question=r["question"], answers=r["answers"],
                          image_source_url=r["flickr_original_url"]))
    out["vqa"] = items

    manifest = {"eval_version": EVAL_VERSION, "seed": SEED, "tasks": {}}
    for task, rows in out.items():
        assert len(rows) == N[task], (task, len(rows))
        p = DATA / f"{task}.jsonl"
        write_jsonl(p, rows)
        manifest["tasks"][task] = {"n": len(rows), "file": p.name, "sha256": sha256(p), **DATASETS[task]}
        if raw_path(task).exists() if task != "tools" else False:
            manifest["tasks"][task]["raw_file_sha256"] = sha256(raw_path(task))
    with open(DATA / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({t: v["n"] for t, v in manifest["tasks"].items()}), "total", sum(N.values()))


if __name__ == "__main__":
    build()
