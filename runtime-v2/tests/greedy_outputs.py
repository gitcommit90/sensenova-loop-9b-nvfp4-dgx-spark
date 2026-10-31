"""Collect greedy (temperature 0) outputs for a fixed prompt set; optionally diff against a previous run.
usage: greedy_outputs.py OUT.json [--compare REF.json] [--concurrency N] [--port P]"""
import argparse, base64, json, sys, requests, os
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--compare"); ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--image", default=os.environ.get("TEST_IMAGE")); ap.add_argument("--port", default="30080"); ap.add_argument("--max-tokens", type=int, default=256); a = ap.parse_args()
IMG = a.image
if not IMG: ap.error("--image PATH is required for the 12-case image/feature check")
P = [
 ("text", "Explain in one paragraph why the sky is blue.", False),
 ("text2", "List five creative uses for a paperclip, numbered.", False),
 ("math", "A store sells pens at 3 for $4. How much do 27 pens cost? Show your reasoning briefly.", False),
 ("think", "If a train goes 60 km in 45 minutes, what is its speed in km/h?", True),
 ("code", "Write a Python function that returns the longest palindromic substring of a string, with a short docstring.", False),
 ("code2", "Implement binary search in C and explain its complexity in two sentences.", False),
 ("translate", "Translate into French and German: 'The meeting has been moved to Thursday afternoon.'", False),
 ("zh", "用三句话介绍一下长城。", False),
 ("json", "Return a JSON object with keys name, age, hobbies for a fictional tennis player. JSON only.", False),
 ("long", "Summarize the following text in 3 bullet points:\n" + ("The river carried old letters from the mountain village to the harbor, where the market opened each winter morning. " * 40), False),
]
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.", "parameters": {"type": "object",
          "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
def run(item):
    name, content, think = item; kw = {}
    if name == "tool": kw["tools"] = tools
    msgs = [{"role": "user", "content": content}]
    r = requests.post(f"http://127.0.0.1:{a.port}/v1/chat/completions", json=dict(model="x", messages=msgs, max_tokens=a.max_tokens if not think else 768,
        temperature=0, chat_template_kwargs={"enable_thinking": think}, **kw), timeout=900).json()
    m = r["choices"][0]["message"]
    return name, {"content": m.get("content"), "reasoning": m.get("reasoning_content"), "tool_calls": m.get("tool_calls"),
                  "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage", {}).get("completion_tokens")}
P.append(("tool", "What's the weather in Paris in celsius?", False))
b64 = base64.b64encode(open(IMG, "rb").read()).decode()
P.append(("image", [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": "Describe this image in two sentences."}], False))
with ThreadPoolExecutor(a.concurrency) as ex: res = dict(ex.map(run, P))
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
if a.compare:
    ref = json.load(open(a.compare)); same = 0
    for k in res:
        r0, r1 = ref[k], res[k]
        eq = all(r0[f] == r1[f] for f in ("content", "reasoning", "tool_calls"))
        same += eq
        if not eq:
            t0 = (r0["reasoning"] or "") + (r0["content"] or "") + json.dumps(r0["tool_calls"]); t1 = (r1["reasoning"] or "") + (r1["content"] or "") + json.dumps(r1["tool_calls"])
            i = next((i for i in range(min(len(t0), len(t1))) if t0[i] != t1[i]), min(len(t0), len(t1)))
            print(f"DIFF {k}: first divergence at char {i}/{len(t0)}: ref=...{t0[max(0,i-40):i+40]!r}\n      new=...{t1[max(0,i-40):i+40]!r}")
    print(f"identical {same}/{len(res)}")
else:
    for k, v in res.items(): print(k, v["finish"], v["usage"], repr((v["content"] or "")[:100]), v["tool_calls"] and v["tool_calls"][0]["function"])
