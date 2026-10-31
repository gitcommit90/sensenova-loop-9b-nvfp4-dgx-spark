"""Task definitions: prompt construction, max_tokens, answer extraction and scoring.

Every task exposes:
  build_messages(item, data_dir) -> (messages, tools|None)
  score(item, resp) -> dict(pred=..., correct=bool, **details)   (resp = common.chat() result)
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import ifeval_lite
from common import calls_from_message, strip_think

# Fixed per-task generation budgets (thinking OFF). Changing these => bump EVAL_VERSION.
MAX_TOKENS = {"ifeval": 1024, "gsm8k": 1024, "mmlu_pro": 1024, "humaneval": 1024,
              "tools": 512, "vqa": 64}

# ============================================================================ IFEval
def ifeval_messages(item, data_dir):
    return [{"role": "user", "content": item["prompt"]}], None


def ifeval_score(item, resp):
    text = strip_think(resp["content"])
    per = ifeval_lite.check(text, item["instruction_id_list"], item["kwargs"])
    return {"pred": None, "correct": all(per), "per_instruction": per}


# ============================================================================ GSM8K
GSM_TMPL = ("Solve the following math problem step by step. At the end, write the final "
            "answer on its own line in the form \"The final answer is <number>\".\n\n{q}")
NUM_RE = r"-?\$?\s?\d[\d,]*(?:\.\d+)?"


def _norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return None
    return str(int(x)) if x == int(x) else repr(round(x, 6))


def gsm_extract(text):
    text = strip_think(text)
    m = re.findall(r"final answer is[:\s]*\**\s*\\?\(?\\?(?:boxed\{)?\s*(" + NUM_RE + ")", text, re.I)
    if m:
        return _norm_num(m[-1])
    m = re.findall(r"\\boxed\{\s*(" + NUM_RE + ")", text)
    if m:
        return _norm_num(m[-1])
    m = re.findall(NUM_RE, text)
    return _norm_num(m[-1]) if m else None


def gsm8k_messages(item, data_dir):
    return [{"role": "user", "content": GSM_TMPL.format(q=item["question"])}], None


def gsm8k_score(item, resp):
    pred = gsm_extract(resp["content"])
    return {"pred": pred, "correct": pred is not None and pred == _norm_num(item["answer"])}


# ============================================================================ MMLU-Pro
LETTERS = "ABCDEFGHIJ"
MC_TMPL = ("The following is a multiple choice question about {cat}. Think step by step briefly, "
           "then finish with \"The answer is (X)\" where X is the letter of the correct option.\n\n"
           "Question: {q}\n{opts}")


def mmlu_pro_messages(item, data_dir):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return [{"role": "user", "content": MC_TMPL.format(cat=item["category"], q=item["question"], opts=opts)}], None


def mc_extract(text, n_opts):
    text = strip_think(text)
    valid = LETTERS[:n_opts]
    m = re.findall(r"answer is:?\s*\**\(?([A-J])\)?", text, re.I)
    m = [x.upper() for x in m if x.upper() in valid]
    if m:
        return m[-1]
    m = re.findall(r"(?:^|\s)\(?([A-J])\)?[\.\s]*$", text.strip())
    if m and m[-1] in valid:
        return m[-1]
    m = re.findall(r"\(([A-J])\)", text)
    m = [x for x in m if x in valid]
    return m[-1] if m else None


def mmlu_pro_score(item, resp):
    pred = mc_extract(resp["content"], len(item["options"]))
    return {"pred": pred, "correct": pred == item["answer"]}


# ============================================================================ HumanEval
HE_TMPL = ("Complete the following Python function. Return the complete function definition "
           "(including the signature and any needed imports) in a single ```python code block.\n\n"
           "```python\n{p}```")


def humaneval_messages(item, data_dir):
    return [{"role": "user", "content": HE_TMPL.format(p=item["prompt"])}], None


def extract_code(text, item):
    text = strip_think(text)
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", text, re.S)
    code = None
    for b in blocks:
        if f"def {item['entry_point']}" in b:
            code = b
            break
    if code is None and blocks:
        code = blocks[0]
    if code is None:
        code = text
    if f"def {item['entry_point']}" not in code:  # body-only completion
        code = item["prompt"] + code
    else:  # keep the prompt's imports/helper defs available
        code = item["prompt"].split(f"def {item['entry_point']}")[0] + "\n" + code
    return code


def run_python_sandboxed(program: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute untrusted code in a fresh isolated interpreter in a temp dir with rlimits."""
    with tempfile.TemporaryDirectory(prefix="he_") as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(program)

        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
            os.setsid()
        env = {"PATH": "/usr/bin:/bin", "HOME": td, "PYTHONHASHSEED": "0"}
        try:
            p = subprocess.run([sys.executable, "-I", "-B", path], cwd=td, env=env, capture_output=True,
                               timeout=timeout, preexec_fn=limits, text=True)
            return p.returncode == 0, (p.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def humaneval_score(item, resp):
    code = extract_code(resp["content"], item)
    prog = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    ok, err = run_python_sandboxed(prog)
    return {"pred": code, "correct": ok, "exec_error": None if ok else err}


# ============================================================================ Tool calling
def tools_messages(item, data_dir):
    msgs = []
    if item.get("system"):
        msgs.append({"role": "system", "content": item["system"]})
    msgs.append({"role": "user", "content": item["user"]})
    return msgs, item["tools"]


def _val_match(got, allowed):
    for a in allowed:
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            try:
                if abs(float(got) - float(a)) < 1e-6:
                    return True
            except (TypeError, ValueError):
                pass
        elif isinstance(a, str) and isinstance(got, str):
            if got.strip().lower() == a.strip().lower():
                return True
        elif got == a:
            return True
    return False


def _call_match(call, gold):
    if call.get("name") != gold["name"]:
        return False
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return False
    spec = gold["arguments"]  # name -> list of allowed values; [] or containing "__optional__"
    for k, allowed in spec.items():
        optional = "__optional__" in allowed
        allowed = [a for a in allowed if a != "__optional__"]
        if k not in args:
            if not optional:
                return False
            continue
        if allowed and not _val_match(args[k], allowed):
            return False
    return all(k in spec for k in args)  # no hallucinated params


def tools_score(item, resp):
    calls, source, fmt = calls_from_message(resp["message"] or {"content": resp["content"]}, item["tools"])
    gold = item["gold_calls"]
    if not gold:  # irrelevance: must NOT call any tool
        correct = len(calls) == 0
    else:
        remaining = list(calls)
        correct = len(calls) == len(gold)
        for g in gold:  # order-insensitive matching for parallel calls
            hit = next((c for c in remaining if _call_match(c, g)), None)
            if hit is None:
                correct = False
                break
            remaining.remove(hit)
    names_ok = sorted(c.get("name") or "" for c in calls) == sorted(g["name"] for g in gold)
    return {"pred": calls, "correct": correct, "names_correct": names_ok, "parse_source": source,
            "format": fmt}


# ============================================================================ VQA (TextVQA)
VQA_SUFFIX = "\nAnswer the question using a single word or phrase."
_CONTRACT = {"dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "cant": "can't", "wont": "won't",
             "im": "i'm", "ive": "i've", "youre": "you're", "thats": "that's"}
_NUMW = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
         "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}


def vqa_norm(s: str) -> str:
    s = (s or "").strip().lower().replace("\n", " ").replace("\t", " ")
    s = s.rstrip(".")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$:~|]", " ", s)
    s = re.sub(r"(?<!\d)\.(?!\d)", "", s)
    words = [_NUMW.get(w, _CONTRACT.get(w, w)) for w in s.split()]
    return " ".join(w for w in words if w not in _ART)


def vqa_messages(item, data_dir):
    img_path = Path(data_dir) / item["image"]
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": item["question"] + VQA_SUFFIX},
    ]}], None


def vqa_score(item, resp):
    pred = vqa_norm(strip_think(resp["content"]).split("\n")[0])
    golds = [vqa_norm(a) for a in item["answers"]]
    matches = sum(1 for g in golds if g == pred)
    soft = min(1.0, matches / 3.0)
    # primary binary metric (relaxed): normalized exact match with >=2 of the 10 annotator answers
    # (robust to a single noisy annotation); standard VQA soft accuracy min(1, m/3) also recorded.
    return {"pred": pred, "correct": matches >= 2 and bool(pred), "vqa_soft_acc": soft,
            "any_annotator_match": matches >= 1}


TASKS = {
    "ifeval": (ifeval_messages, ifeval_score),
    "gsm8k": (gsm8k_messages, gsm8k_score),
    "mmlu_pro": (mmlu_pro_messages, mmlu_pro_score),
    "humaneval": (humaneval_messages, humaneval_score),
    "tools": (tools_messages, tools_score),
    "vqa": (vqa_messages, vqa_score),
}


def agreement_key(task: str, scored: dict, resp: dict):
    """Label-free comparable answer used for reference agreement."""
    if task == "ifeval":
        return strip_think(resp.get("content", "")).strip()
    if task == "humaneval":
        return (scored.get("pred") or "").strip()
    if task == "tools":
        return json.dumps(scored.get("pred"), sort_keys=True, default=str)
    return scored.get("pred")
