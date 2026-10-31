"""Drive a short torch-profiler capture: warmup, then start_profile(num_steps) + N concurrent requests."""
import sys, time, json, threading, requests
sys.path.insert(0, "/home/tux/sensenova-loop-nvfp4/eval")
from perf_bench import build_prompt_text
base = "http://127.0.0.1:30080"
conc = int(sys.argv[1]); steps = int(sys.argv[2]); prefix = sys.argv[3]
text, _, _ = build_prompt_text(1024, "/home/tux/sensenova-loop-nvfp4/upstream/SenseNova-Loop-9B-Preview/tokenizer.json")
def req(i, n=64):
    r = requests.post(base + "/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": f"[{time.time()}-{i}] " + text}],
        "max_tokens": n, "min_tokens": n, "ignore_eos": True, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    return r.json()["usage"]
print("warm", req(-1, 32))
ths = [threading.Thread(target=req, args=(i, 600)) for i in range(conc)]
for t in ths: t.start()
time.sleep(2.0 + 0.5 * conc)  # past prefill, into decode
r = requests.post(base + "/start_profile", json={"num_steps": steps, "activities": ["CPU", "GPU"], "profile_prefix": prefix})
print("profile", r.status_code, r.text[:200])
for t in ths: t.join()
print(json.dumps(requests.get(base + "/get_server_info").json().get("internal_states", [{}])[0].get("avg_spec_accept_length")))
