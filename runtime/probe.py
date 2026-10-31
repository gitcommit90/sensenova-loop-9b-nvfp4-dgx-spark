#!/usr/bin/env python3
"""Quick functional probe of an OpenAI-compatible SenseNova-Loop server (text, thinking, tools, image)."""
import base64, json, sys, requests
base, model = sys.argv[1], sys.argv[2]
img = sys.argv[3] if len(sys.argv) > 3 else None
def chat(messages, max_tokens=128, thinking=False, **kw):
    r = requests.post(f"{base}/chat/completions", json=dict(model=model, messages=messages, max_tokens=max_tokens,
        temperature=0, chat_template_kwargs={"enable_thinking": thinking}, **kw), timeout=600).json()
    m = r["choices"][0]["message"]
    return {"content": m.get("content"), "reasoning": (m.get("reasoning_content") or "")[:300],
            "tool_calls": m.get("tool_calls"), "usage": r.get("usage")}
out = {}
out["mult"] = chat([{"role": "user", "content": "What is 13 times 7? Reply with the number only."}], 16)
out["haiku"] = chat([{"role": "user", "content": "Write a haiku about a tennis ball."}], 64)
out["thinking"] = chat([{"role": "user", "content": "If a train goes 60 km in 45 minutes, what is its speed in km/h?"}], 512, thinking=True)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
out["tool"] = chat([{"role": "user", "content": "What's the weather in Paris in celsius?"}], 128, tools=tools)
if img:
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    out["image"] = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "Describe this image in two sentences."}]}], 96)
print(json.dumps(out, indent=1, ensure_ascii=False))
