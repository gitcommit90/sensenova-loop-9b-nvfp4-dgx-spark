#!/usr/bin/env python
"""Build calib/calib_v1.jsonl: 256 chat-format calibration examples for NVFP4 (ModelOpt) PTQ.

  .venv-eval/bin/python calib/build_calib.py            # raw files are expected under calib/raw/
  (downloads: see SOURCES; fetched with huggingface_hub.hf_hub_download at the pinned revisions)

Each line: {"id", "category", "source", "messages": [...OpenAI-style...], "tools"?: [...],
            "enable_thinking": bool, "approx_text_tokens": int, "n_images": int}
Images: {"type": "image", "image": "calib/images/<file>.jpg"} (path relative to project root),
which HF processors' apply_chat_template understand; the chat template also accepts image_url.
Thinking examples carry the trace in the assistant's `reasoning_content` (the chat template renders
it as <think>\n...\n</think>\n\n before the answer); all other final assistant turns render with
the empty <think>\n\n</think>\n\n block, matching enable_thinking=False inference.
Disjointness from eval/data is enforced by construction (different datasets/splits) AND checked
by 13-gram overlap against every eval prompt (summary in calib/calib_v1.meta.json).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "calib"
RAW = CAL / "raw"
SEED = 20261030
MAX_TEXT_TOKENS = 1800
TARGET = {"chat": 77, "code": 51, "math": 51, "tools": 38, "image": 39}  # = 256 (30/20/20/15/15 %)
N_THINK_MATH = 26

SOURCES = {
    "chat": dict(repo="OpenAssistant/oasst2", rev="179dd21fc55192153d94adb0e0ce8f69e222bf75",
                 file="data/train-00000-of-00001-88ba0162028a73fc.parquet", split="train", license="Apache-2.0"),
    "code": dict(repo="google-research-datasets/mbpp", rev="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                 file="full/train-00000-of-00001.parquet", split="full/train", license="CC-BY-4.0"),
    "math": dict(repo="openai/gsm8k", rev="740312add88f781978c0658806c59bc2815b9866",
                 file="main/train-00000-of-00001.parquet", split="main/train", license="MIT"),
    "tools": dict(repo="glaiveai/glaive-function-calling-v2", rev="e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
                  file="glaive-function-calling-v2.json", split="train", license="Apache-2.0"),
    "image": dict(repo="lmms-lab/textvqa", rev="9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                  file="data/train-00000-of-00020.parquet", split="train (shard 0 of 20)",
                  license="TextVQA annotations CC-BY-4.0; images OpenImages CC-BY-2.0 (Flickr attribution)"),
}


def raw(cat):
    s = SOURCES[cat]
    return RAW / s["repo"].replace("/", "__") / s["file"]


_tok = None


def ntok(text):
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(ROOT / "upstream/SenseNova-Loop-9B-Preview/tokenizer.json"))
    return len(_tok.encode(text, add_special_tokens=False).ids)


def ex_text(ex):
    parts = [json.dumps(ex.get("tools") or "")]
    for m in ex["messages"]:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c)
        parts.append(c or "")
        parts.append(m.get("reasoning_content") or "")
        parts.append(json.dumps(m.get("tool_calls") or ""))
    return "\n".join(parts)


# ----------------------------------------------------------------------------- chat (oasst2)
def build_chat(rng):
    t = pq.read_table(raw("chat"), columns=["message_id", "parent_id", "text", "role", "lang", "deleted",
                                             "review_result", "rank", "tree_state", "synthetic"]).to_pylist()
    kids = {}
    roots = []
    for m in t:
        if m["deleted"] or m["review_result"] is False or m["synthetic"]:
            continue
        if m["parent_id"] is None:
            if m["role"] == "prompter" and m["tree_state"] == "ready_for_export":
                roots.append(m)
        else:
            kids.setdefault(m["parent_id"], []).append(m)

    def best(mid, role):
        cs = [c for c in kids.get(mid, []) if c["role"] == role]
        cs.sort(key=lambda c: (c["rank"] if c["rank"] is not None else 99, c["message_id"]))
        return cs[0] if cs else None
    roots.sort(key=lambda m: m["message_id"])
    rng.shuffle(roots)
    want = {"en": 58, "zh": 10, "other": 9}
    got = {"en": [], "zh": [], "other": []}
    for r in roots:
        bucket = r["lang"] if r["lang"] in ("en", "zh") else "other"
        if len(got[bucket]) >= want[bucket]:
            continue
        msgs, cur = [{"role": "user", "content": r["text"]}], r
        turns = rng.choice([1, 1, 2])
        for k in range(turns):
            a = best(cur["message_id"], "assistant")
            if a is None:
                break
            msgs.append({"role": "assistant", "content": a["text"]})
            if k + 1 < turns:
                u = best(a["message_id"], "prompter")
                a2 = best(u["message_id"], "assistant") if u else None
                if not (u and a2):
                    break
                msgs.append({"role": "user", "content": u["text"]})
                cur = u
        if msgs[-1]["role"] != "assistant":
            continue
        ex = {"category": "chat", "source": f"oasst2:{r['message_id']}", "lang": r["lang"], "messages": msgs}
        if ntok(ex_text(ex)) > MAX_TEXT_TOKENS:
            continue
        got[bucket].append(ex)
        if all(len(got[b]) >= want[b] for b in want):
            break
    return got["en"] + got["zh"] + got["other"]


# ----------------------------------------------------------------------------- code (MBPP train)
def build_code(rng):
    rows = pq.read_table(raw("code")).to_pylist()
    rows.sort(key=lambda r: r["task_id"])
    idx = sorted(rng.sample(range(len(rows)), TARGET["code"]))
    out = []
    for i in idx:
        r = rows[i]
        tests = "\n".join(r["test_list"])
        user = f"{r['text']} Your code should pass these tests:\n\n```python\n{tests}\n```"
        asst = f"```python\n{r['code'].strip()}\n```"
        out.append({"category": "code", "source": f"mbpp:task_id={r['task_id']}",
                    "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]})
    return out


# ----------------------------------------------------------------------------- math (GSM8K train)
def build_math(rng):
    rows = pq.read_table(raw("math")).to_pylist()
    idx = sorted(rng.sample(range(len(rows)), TARGET["math"]))
    out = []
    for j, i in enumerate(idx):
        q, sol = rows[i]["question"], rows[i]["answer"]
        steps, final = sol.split("####")
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        final = final.strip()
        think = j < N_THINK_MATH
        if think:
            asst = {"role": "assistant", "reasoning_content": steps,
                    "content": f"The final answer is {final}."}
        else:
            asst = {"role": "assistant", "content": f"{steps}\nThe final answer is {final}."}
        out.append({"category": "math", "source": f"gsm8k:train:{i}", "enable_thinking": think,
                    "messages": [{"role": "user", "content": q}, asst]})
    return out


# ----------------------------------------------------------------------------- tools (glaive v2)
def parse_glaive(rec):
    sysm = rec["system"]
    if "following functions" not in sysm:
        return None
    body = sysm.split("Use them if required -", 1)[-1].strip()
    dec, pos, fns = json.JSONDecoder(), 0, []
    while pos < len(body):
        while pos < len(body) and body[pos] in " \n\t":
            pos += 1
        if pos >= len(body):
            break
        try:
            obj, pos = dec.raw_decode(body, pos)
        except json.JSONDecodeError:
            return None
        fns.append({"type": "function", "function": obj})
    if not fns:
        return None
    turns = re.split(r"\n*(USER|ASSISTANT|FUNCTION RESPONSE): ", "\n" + rec["chat"])
    msgs, n_calls = [], 0
    it = iter(turns[1:])
    for role, text in zip(it, it):
        text = text.replace("<|endoftext|>", "").strip()
        if role == "USER":
            msgs.append({"role": "user", "content": text})
        elif role == "FUNCTION RESPONSE":
            msgs.append({"role": "tool", "content": text})
        else:
            if "<functioncall>" in text:
                pre, call = text.split("<functioncall>", 1)
                m = re.match(r'\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', call, re.S)
                if not m:
                    return None
                try:
                    args = json.loads(m.group(2))
                except json.JSONDecodeError:
                    return None
                if not isinstance(args, dict):
                    return None
                msgs.append({"role": "assistant", "content": pre.strip(),
                             "tool_calls": [{"id": f"call_{n_calls}", "type": "function",
                                             "function": {"name": m.group(1), "arguments": args}}]})
                n_calls += 1
            else:
                msgs.append({"role": "assistant", "content": text})
    if n_calls == 0 or not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "tool" for m in msgs):
        return None
    return fns, msgs


def build_tools(rng):
    data = json.load(open(raw("tools")))
    order = list(range(len(data)))
    rng.shuffle(order)
    out = []
    for i in order:
        p = parse_glaive(data[i])
        if not p:
            continue
        ex = {"category": "tools", "source": f"glaive-fc-v2:{i}", "tools": p[0], "messages": p[1]}
        if ntok(ex_text(ex)) > MAX_TEXT_TOKENS:
            continue
        out.append(ex)
        if len(out) >= TARGET["tools"]:
            break
    return out


# ----------------------------------------------------------------------------- image (TextVQA train)
def build_image(rng, eval_image_ids):
    from tasks import vqa_norm  # eval/ is on sys.path
    t = pq.read_table(raw("image"), columns=["question_id", "question", "answers", "image_id"]).to_pylist()
    cand = [i for i, r in enumerate(t) if r["image_id"] not in eval_image_ids
            and Counter(vqa_norm(a) for a in r["answers"]).most_common(1)[0][1] >= 3]
    rng.shuffle(cand)
    seen_img, pickd = set(), []
    for i in cand:
        if t[i]["image_id"] in seen_img:
            continue
        seen_img.add(t[i]["image_id"])
        pickd.append(i)
        if len(pickd) >= TARGET["image"]:
            break
    pickd.sort()
    imgs = pq.read_table(raw("image"), columns=["image"]).take(pickd).column("image").to_pylist()
    (CAL / "images").mkdir(exist_ok=True)
    out = []
    for i, im in zip(pickd, imgs):
        r = t[i]
        b = im["bytes"]
        ext = "png" if b[:4] == b"\x89PNG" else "jpg"
        rel = f"calib/images/textvqa_train_{r['question_id']}.{ext}"
        (ROOT / rel).write_bytes(b)
        ans = Counter(a.strip() for a in r["answers"]).most_common(1)[0][0]
        out.append({"category": "image", "source": f"textvqa:train:qid={r['question_id']}", "image_id": r["image_id"],
                    "image_sha256": hashlib.sha256(b).hexdigest(), "messages": [
                        {"role": "user", "content": [{"type": "image", "image": rel},
                                                     {"type": "text", "text": r["question"].strip().capitalize()
                                                      + "\nAnswer the question using a single word or phrase."}]},
                        {"role": "assistant", "content": ans}]})
    return out


# ----------------------------------------------------------------------------- overlap check
def shingles(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    sys.path.insert(0, str(ROOT / "eval"))
    eval_dir = ROOT / "eval/data"
    eval_sh, eval_imgs = set(), set()
    for p in eval_dir.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            eval_sh |= shingles(json.dumps(r, ensure_ascii=False))
            if "image_id" in r:
                eval_imgs.add(r["image_id"])
    exs = []
    exs += build_chat(random.Random(f"{SEED}-chat"))
    exs += build_code(random.Random(f"{SEED}-code"))
    exs += build_math(random.Random(f"{SEED}-math"))
    exs += build_tools(random.Random(f"{SEED}-tools"))
    exs += build_image(random.Random(f"{SEED}-image"), eval_imgs)
    cnt = Counter(e["category"] for e in exs)
    assert dict(cnt) == TARGET, cnt
    overlaps = []
    for k, e in enumerate(exs):
        e["id"] = f"calib_v1_{k:03d}"
        e.setdefault("enable_thinking", False)
        e["approx_text_tokens"] = ntok(ex_text(e))
        e["n_images"] = sum(1 for m in e["messages"] if isinstance(m["content"], list)
                            for p in m["content"] if p.get("type") == "image")
        ov = shingles(ex_text(e)) & eval_sh
        if ov:
            overlaps.append({"id": e["id"], "source": e["source"], "n_shared_13grams": len(ov),
                             "example": sorted(ov)[0]})
    order = ["id", "category", "source", "enable_thinking", "approx_text_tokens", "n_images"]
    with open(CAL / "calib_v1.jsonl", "w") as f:
        for e in exs:
            f.write(json.dumps({**{k: e[k] for k in order}, **{k: v for k, v in e.items() if k not in order}},
                               ensure_ascii=False) + "\n")
    toks = [e["approx_text_tokens"] for e in exs]
    meta = {"version": "calib_v1", "seed": SEED, "n": len(exs), "counts": dict(cnt),
            "n_thinking_examples": sum(e["enable_thinking"] for e in exs),
            "langs_chat": dict(Counter(e.get("lang") for e in exs if e["category"] == "chat")),
            "text_tokens": {"min": min(toks), "max": max(toks), "mean": sum(toks) / len(toks), "sum": sum(toks)},
            "max_text_tokens_filter": MAX_TEXT_TOKENS, "sources": SOURCES,
            "eval_overlap_13gram": overlaps,
            "sha256": hashlib.sha256((CAL / "calib_v1.jsonl").read_bytes()).hexdigest()}
    (CAL / "calib_v1.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("n", "counts", "n_thinking_examples", "langs_chat", "text_tokens")}))
    print("13-gram overlaps with eval:", len(overlaps), overlaps[:3])


if __name__ == "__main__":
    main()
