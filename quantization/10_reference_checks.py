#!/usr/bin/env python3
"""HF-transformers functional checks for SenseNova-Loop (reference BF16 backbone + FP32 mHC).

Checks: mHC router dtype, shared-weight looping, text greedy, thinking, tool-call format,
image understanding, and cached-vs-uncached decoding consistency (teacher-forced argmax).
Writes a JSON report; does not modify the model directory.

usage: python -I scripts/10_reference_checks.py MODEL_DIR OUT_JSON [IMAGE]
"""
import json, sys, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_dir, out_path = sys.argv[1], sys.argv[2]
image_path = sys.argv[3] if len(sys.argv) > 3 else None
dev = "cuda:0"
report = {"model_dir": model_dir, "torch": torch.__version__}

t0 = time.time()
proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": dev}).eval()
report["load_s"] = round(time.time() - t0, 1)
report["gpu_mem_after_load_GiB"] = round(torch.cuda.memory_allocated() / 2**30, 2)

text = model.model.language_model
report["loop_plan_len"] = len(text._loop_plan)
routers = {n: p.dtype for n, p in model.named_parameters() if ".mhc_" in n}
report["mhc_param_count"] = len(routers)
report["mhc_all_fp32"] = all(d == torch.float32 for d in routers.values())
report["layer_param_dtypes"] = sorted({str(p.dtype) for n, p in model.named_parameters() if ".mhc_" not in n})

# Each physical looped layer must be called twice per forward (shared weights).
calls = {}
hooks = [l.register_forward_hook(lambda m, i, o, k=k: calls.__setitem__(k, calls.get(k, 0) + 1))
         for k, l in enumerate(text.layers)]
tok = proc.tokenizer


def chat(messages, max_new=128, tools=None, thinking=False, images=None):
    prompt = proc.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=thinking)
    inputs = proc(text=[prompt], images=images, return_tensors="pt").to(dev)
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return {"prompt_tokens": int(inputs["input_ids"].shape[1]), "gen_tokens": int(gen.shape[0]),
            "seconds": round(time.time() - t, 2), "text": tok.decode(gen, skip_special_tokens=False)}, out, inputs


r, _, _ = chat([{"role": "user", "content": "What is 13 times 7? Reply with the number only."}], 16)
report["text_mult"] = r
report["loop_calls_per_layer"] = {str(k): v for k, v in sorted(calls.items())}
report["looped_layers_called_twice"] = all(calls[k] == (2 if 4 <= k < 28 else 1) * calls[0] for k in calls)
for h in hooks:
    h.remove()

report["text_haiku"] = chat([{"role": "user", "content": "Write a haiku about a tennis ball."}], 64)[0]
report["thinking"] = chat([{"role": "user", "content": "If a train goes 60 km in 45 minutes, what is its speed in km/h?"}],
                          384, thinking=True)[0]
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
r = chat([{"role": "user", "content": "What's the weather in Paris in celsius?"}], 96, tools=tools)[0]
r["has_tool_call_xml"] = ("<tool_call>" in r["text"] and "<function=get_weather>" in r["text"]
                          and "<parameter=city>" in r["text"])
report["tool_call"] = r

if image_path:
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "Describe this image in two sentences."}]}]
    report["image"] = chat(msgs, 96, images=[img])[0]

# Cached generation consistency: greedy tokens from cached decoding must equal the
# argmax of a single uncached forward over prompt+generation (teacher forcing).
r, out, inputs = chat([{"role": "user", "content": "Explain in one paragraph why the sky is blue."}], 64)
plen = inputs["input_ids"].shape[1]
with torch.inference_mode():
    logits = model(input_ids=out, use_cache=False).logits[0, plen - 1:-1].float()
pred = logits.argmax(-1)
gen = out[0, plen:]
report["cache_consistency"] = {"n": int(gen.numel()), "match": int((pred == gen).sum()),
                               "first_mismatch": int((pred != gen).nonzero()[0]) if (pred != gen).any() else None,
                               "finite_logits": bool(torch.isfinite(logits).all())}
report["peak_gpu_mem_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
