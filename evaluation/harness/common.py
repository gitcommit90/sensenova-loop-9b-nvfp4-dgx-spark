"""Shared helpers for the SenseNova-Loop NVFP4 eval harness (stdlib + requests only)."""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

EVAL_VERSION = "sn-loop-eval-v1"


# ----------------------------------------------------------------------------- stats
def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------- text utils
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str | None) -> str:
    """Remove any <think>...</think> block (and a dangling '...</think>' prefix)."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ----------------------------------------------------------------------------- tool-call parsing
# Chat template (Qwen3.5-style XML) emits:
# <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n...</function>\n</tool_call>
# String values are written raw; dict/list values as JSON (tojson); numbers/bools via |string
# (so Python-ish 'True'/'False' are possible).
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict | None) -> Any:
    typ = (schema or {}).get("type")
    if typ == "string":
        return value
    v = value.strip()
    if typ == "boolean" or v in ("True", "False"):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
    try:
        return json.loads(v)
    except Exception:
        pass
    if typ in ("integer", "number"):
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v
    return value


def parse_xml_tool_calls(text: str, tools: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Fallback parser for raw-text tool calls. Returns (calls, format_report)."""
    schemas: dict[str, dict] = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        name = m.group(1).strip()
        args = {}
        for pm in PARAM_RE.finditer(m.group(2)):
            pname = pm.group(1).strip()
            args[pname] = _coerce(pm.group(2), schemas.get(name, {}).get(pname))
        calls.append({"name": name, "arguments": args})
    n_open = (text or "").count("<tool_call>")
    n_close = (text or "").count("</tool_call>")
    tail = (text or "").split("</tool_call>")[-1].strip() if n_close else ""
    report = {
        "n_tool_call_tags": n_open,
        "balanced_tags": n_open == n_close,
        "all_blocks_parsed": n_open == len(calls),
        "suffix_after_last_call": bool(tail),  # template says: NO suffix after the call
        "well_formed": n_open > 0 and n_open == n_close == len(calls),
    }
    return calls, report


def calls_from_message(msg: dict, tools: list[dict] | None) -> tuple[list[dict], str, dict]:
    """Return (calls, source, format_report). source in {'server','fallback_xml','none'}."""
    tcs = msg.get("tool_calls") or []
    if tcs:
        calls, ok = [], True
        for tc in tcs:
            fn = tc.get("function", tc)
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except Exception:
                    args, ok = {"__unparseable__": raw}, False
            else:
                args = raw or {}
            calls.append({"name": fn.get("name"), "arguments": args})
        return calls, "server", {"well_formed": ok, "args_valid_json": ok}
    content = strip_think(msg.get("content") or "")
    calls, rep = parse_xml_tool_calls(content, tools)
    return calls, ("fallback_xml" if calls else "none"), rep


# ----------------------------------------------------------------------------- HTTP client
def chat(session, base_url: str, model: str, messages: list, *, max_tokens: int,
         tools: list | None = None, enable_thinking: bool = False, timeout: float = 600,
         retries: int = 2, extra: dict | None = None) -> dict:
    """One deterministic chat completion. Returns a dict with response + timing (never raises)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra:
        body.update(extra)
    url = base_url.rstrip("/") + "/chat/completions"
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = session.post(url, json=body, timeout=timeout)
            dt = time.time() - t0
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code < 500:
                    break
                time.sleep(1 + attempt)
                continue
            j = r.json()
            ch = j["choices"][0]
            msg = ch.get("message") or {}
            return {
                "ok": True,
                "content": msg.get("content") or "",
                "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "message": msg,
                "finish_reason": ch.get("finish_reason"),
                "usage": j.get("usage") or {},
                "latency_s": round(dt, 3),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1 + attempt)
    return {"ok": False, "content": "", "reasoning_content": "", "tool_calls": [], "message": {},
            "finish_reason": None, "usage": {}, "latency_s": None, "error": last_err}


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
