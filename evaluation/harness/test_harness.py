#!/usr/bin/env python
"""Offline tests for scorers + end-to-end harness against eval/mock_server.py (no GPU/model needed).

  .venv-eval/bin/python eval/test_harness.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ifeval_lite  # noqa: E402
from common import parse_xml_tool_calls, read_jsonl, strip_think, wilson_ci  # noqa: E402
from tasks import (extract_code, gsm_extract, humaneval_score, mc_extract, run_python_sandboxed,  # noqa: E402
                   tools_score, vqa_norm, vqa_score)

PY = sys.executable
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def unit():
    lo, hi = wilson_ci(50, 100)
    check(abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, f"wilson {lo},{hi}")
    check(strip_think("<think>\nabc\n</think>\n\nHi") == "Hi", "strip_think")
    # GSM8K
    check(gsm_extract("so 3+4=7.\nThe final answer is 1,234.") == "1234", "gsm final")
    check(gsm_extract("The final answer is $18.00") == "18", "gsm dollars")
    check(gsm_extract("answer \\boxed{42}") == "42", "gsm boxed")
    check(gsm_extract("it is -3.5 or so") == "-3.5", "gsm last num")
    # MC
    check(mc_extract("blah (B) ... The answer is (C).", 4) == "C", "mc 1")
    check(mc_extract("The answer is J", 10) == "J", "mc 2")
    check(mc_extract("The answer is (J)", 4) is None, "mc invalid letter")
    # IFEval
    C = ifeval_lite.check
    check(C("hello world", ["punctuation:no_comma"], [{}]) == [True], "no_comma +")
    check(C("hello, world", ["punctuation:no_comma"], [{}]) == [False], "no_comma -")
    check(C("<<My Title>> text", ["detectable_format:title"], [{}]) == [True], "title")
    check(C("* a\n* b\n- c", ["detectable_format:number_bullet_lists"], [{"num_bullets": 3}]) == [True], "bullets")
    check(C('```json\n{"a": 1}\n```', ["detectable_format:json_format"], [{}]) == [True], "json")
    check(C("p1\n***\np2", ["length_constraints:number_paragraphs"], [{"num_paragraphs": 2}]) == [True], "paras")
    check(C("one two three", ["length_constraints:number_words"], [{"num_words": 5, "relation": "less than"}]) == [True], "words")
    check(C("THIS IS ALL CAPS ENGLISH TEXT ABOUT THE WEATHER TODAY", ["change_case:english_capital"], [{}]) == [True], "caps")
    check(C("text\nP.S. bye", ["detectable_content:postscript"], [{"postscript_marker": "P.S."}]) == [True], "ps")
    check(C("a ****** b", ["combination:two_responses"], [{}]) == [True], "two resp")
    check(C("Is it? My answer is yes.", ["detectable_format:constrained_response"], [{}]) == [True], "constrained")
    check(C("Das ist ein deutscher Satz über das Wetter in Berlin heute.", ["language:response_language"],
            [{"language": "de"}]) == [True], "lang")
    check(C("We like apples", ["keywords:forbidden_words"], [{"forbidden_words": ["apples"]}]) == [False], "forbidden")
    # Sandbox
    ok, _ = run_python_sandboxed("assert 1+1==2\n")
    check(ok, "sandbox pass")
    ok, _ = run_python_sandboxed("assert 1+1==3\n")
    check(not ok, "sandbox fail")
    t = time.time()
    ok, err = run_python_sandboxed("while True: pass\n", timeout=2)
    check(not ok and time.time() - t < 6, f"sandbox timeout {err}")
    ok, _ = run_python_sandboxed("import os; open('/tmp/../' + 'x','r')\n")  # expected to fail cleanly
    he = read_jsonl(HERE / "data/humaneval.jsonl")[0]
    r = humaneval_score(he, {"content": f"```python\n{he['prompt']}{he['canonical_solution']}```"})
    check(r["correct"], "humaneval canonical passes")
    r = humaneval_score(he, {"content": f"```python\n{he['canonical_solution']}```"})  # body-only
    check(r["correct"], "humaneval body-only passes")
    check("def " in extract_code("no code", he), "extract fallback")
    # XML tool-call parser (format from chat_template.jinja)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "s": {"type": "string"}, "n": {"type": "integer"}, "b": {"type": "boolean"}, "o": {"type": "object"}}}}}]
    txt = ("Sure.\n\n<tool_call>\n<function=f>\n<parameter=s>\n42\n</parameter>\n<parameter=n>\n7\n</parameter>\n"
           "<parameter=b>\nTrue\n</parameter>\n<parameter=o>\n{\"k\": [1, 2]}\n</parameter>\n</function>\n</tool_call>")
    calls, rep = parse_xml_tool_calls(txt, tools)
    check(calls == [{"name": "f", "arguments": {"s": "42", "n": 7, "b": True, "o": {"k": [1, 2]}}}], f"xml {calls}")
    check(rep["well_formed"] and not rep["suffix_after_last_call"], f"xml rep {rep}")
    calls, rep = parse_xml_tool_calls("<tool_call>\n<function=f>\n<parameter=s>\nx", tools)
    check(calls == [] and not rep["well_formed"], "xml truncated")
    tc = read_jsonl(HERE / "data/tools.jsonl")
    par = next(c for c in tc if c["id"] == "tool_parallel_03")
    msg = {"content": None, "tool_calls": [
        {"function": {"name": "set_timer", "arguments": '{"minutes": 10}'}},
        {"function": {"name": "set_light_state", "arguments": '{"room": "Bedroom", "on": false}'}}]}
    check(tools_score(par, {"message": msg, "content": ""})["correct"], "tools parallel order-insensitive")
    msg["tool_calls"][0]["function"]["arguments"] = '{"minutes": 10, "bogus": 1}'
    check(not tools_score(par, {"message": msg, "content": ""})["correct"], "tools hallucinated param")
    irr = next(c for c in tc if c["category"] == "irrelevance")
    check(tools_score(irr, {"message": {"content": "Jane Austen"}, "content": "Jane Austen"})["correct"], "irrelevance")
    # VQA
    check(vqa_norm("The Two Dogs.") == "2 dogs", f"vqa norm {vqa_norm('The Two Dogs.')}")
    v = {"answers": ["dakota"] * 6 + ["x"] * 4}
    check(vqa_score(v, {"content": "Dakota"})["correct"], "vqa")
    check(not vqa_score({"answers": ["a1"] + ["z"] * 9}, {"content": "a1"})["correct"], "vqa single-annotator")


def e2e():
    with tempfile.TemporaryDirectory() as td:
        procs = []
        try:
            for port, extra in ((18991, []), (18992, ["--tool-mode", "xml"]), (18993, ["--corrupt-every", "5"])):
                procs.append(subprocess.Popen([PY, "-I", str(HERE / "mock_server.py"), "--port", str(port), *extra],
                                              stdout=subprocess.DEVNULL))
            time.sleep(2.5)
            for port, name in ((18991, "gold"), (18992, "xml"), (18993, "corrupt")):
                subprocess.run([PY, "-I", str(HERE / "run_eval.py"), "--base-url", f"http://127.0.0.1:{port}/v1",
                                "--model", "mock", "--out", f"{td}/{name}", "--concurrency", "8"], check=True)
            g = json.load(open(f"{td}/gold/summary.json"))["tasks"]
            for t in ("gsm8k", "mmlu_pro", "humaneval", "tools", "vqa"):
                check(g[t]["accuracy"] == 1.0, f"gold run {t} acc={g[t]['accuracy']}")
            x = json.load(open(f"{td}/xml/summary.json"))["tasks"]["tools"]
            check(x["accuracy"] == 1.0 and x["parse_source_counts"].get("fallback_xml") == 17,
                  f"xml fallback {x}")
            check(x["format_well_formed"] == 1.0, "xml well formed")
            out = subprocess.run([PY, "-I", str(HERE / "compare.py"), f"{td}/gold", f"{td}/corrupt"],
                                 check=True, capture_output=True, text=True).stdout
            print(out)
            cmp_ = json.load(open(f"{td}/corrupt/compare_vs_ref.json"))
            o = cmp_["overall_micro"]
            check(o["delta"] < 0 and o["delta_boot95"][1] < 0 and o["ref_agreement"] < 1, f"compare {o}")
            check(cmp_["tasks"]["ifeval"]["ref_agreement"] == 1.0, "ifeval agreement identical")
            pb = subprocess.run([PY, "-I", str(HERE / "perf_bench.py"), "--base-url", "http://127.0.0.1:18991/v1",
                                 "--model", "mock", "--concurrency", "1,4", "--num-requests", "4",
                                 "--input-tokens", "256", "--output-tokens", "32", "--out", f"{td}/perf.json"],
                                capture_output=True, text=True)
            check(pb.returncode == 0, f"perf_bench rc {pb.stderr[-800:]}")
            if pb.returncode == 0:
                p = json.load(open(f"{td}/perf.json"))
                lv = p["levels"][-1]
                check(lv["n_ok"] == 4 and lv["output_tokens_mean"] == 32, f"perf {lv}")
        finally:
            for p in procs:
                p.terminate()


if __name__ == "__main__":
    unit()
    if "--unit-only" not in sys.argv:
        e2e()
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILURES")
    sys.exit(1 if FAILS else 0)
