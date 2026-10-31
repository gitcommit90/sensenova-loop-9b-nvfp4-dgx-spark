#!/usr/bin/env python3
"""ModelOpt NVFP4 PTQ of SenseNova-Loop-9B-Preview through the real looped (2-visit) forward.

Recipes:
  mlp      - NVFP4 on decoder MLP gate/up/down only (conservative)
  decoder  - NVFP4 on all decoder linears (full-attn q/k/v/o, GDN in_proj_qkv/in_proj_z/out_proj, MLP)
Always higher precision: vision tower, lm_head, embeddings, GDN in_proj_a/in_proj_b (tiny gate projs),
GDN conv1d/A_log/dt_bias, norms, mHC routers (FP32, not nn.Linear), MTP head (not loaded by HF; copied BF16).
KV cache: not quantized.

usage: python -I scripts/20_quantize.py --recipe decoder --calib calib/calib_v1.jsonl --out work/nvfp4-decoder [--limit N]
"""
import argparse, copy, json, os, sys, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import modelopt
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="upstream/SenseNova-Loop-9B-Preview")
ap.add_argument("--recipe", choices=["mlp", "decoder"], required=True)
ap.add_argument("--calib", default="calib/calib_v1.jsonl")
ap.add_argument("--root", default=".", help="base dir for calib image paths")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-len", type=int, default=2048)
args = ap.parse_args()
dev = "cuda:0"
torch.manual_seed(0)

proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": dev}).eval()
text = model.model.language_model

# --- quantization config -------------------------------------------------------------
cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
qc = cfg["quant_cfg"]  # ModelOpt >=0.48: ordered list, later entries win
for pat in ["*lm_head*", "*visual*", "*vision*", "*mtp*", "*in_proj_a*", "*in_proj_b*",
            "*mhc_*", "*embed_tokens*", "*conv1d*"]:
    qc.append({"quantizer_name": pat, "enable": False})
if args.recipe == "mlp":
    for pat in ["*self_attn*", "*linear_attn*"]:
        qc.append({"quantizer_name": pat, "enable": False})

# --- calibration data: rendered with the model's own template + processor ----------------
rows = [json.loads(l) for l in open(args.calib)]
if args.limit:
    rows = rows[: args.limit]


def render(row):
    msgs, images = [], []
    for m in row["messages"]:
        m = dict(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image":
                    images.append(Image.open(os.path.join(args.root, p["image"])).convert("RGB"))
                    parts.append({"type": "image"})
                else:
                    parts.append(p)
            m["content"] = parts
        msgs.append(m)
    prompt = proc.apply_chat_template(msgs, tools=row.get("tools"), tokenize=False,
                                      add_generation_prompt=False,
                                      enable_thinking=bool(row.get("enable_thinking")))
    batch = proc(text=[prompt], images=images or None, return_tensors="pt")
    if batch["input_ids"].shape[1] > args.max_len and not images:
        batch = {k: v[:, : args.max_len] for k, v in batch.items()}
    return batch


batches = [render(r) for r in rows]
ntok = sum(int(b["input_ids"].shape[1]) for b in batches)
print(f"calib: {len(batches)} samples, {ntok} tokens, {sum(r.get('n_images', 0) for r in rows)} images", flush=True)

visits = {}
hooks = [l.register_forward_hook(lambda m, i, o, k=k: visits.__setitem__(k, visits.get(k, 0) + 1))
         for k, l in enumerate(text.layers)]


def forward_loop(m):
    t = time.time()
    for i, b in enumerate(batches):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.inference_mode():
            m(**b, use_cache=False)
        if i % 32 == 0:
            print(f"  calib {i}/{len(batches)} {time.time() - t:.0f}s", flush=True)


t0 = time.time()
model = mtq.quantize(model, cfg, forward_loop)
for h in hooks:
    h.remove()
calib_s = time.time() - t0
mtq.print_quant_summary(model) if hasattr(mtq, "print_quant_summary") else None

# --- bookkeeping: what got quantized ----------------------------------------------------
quantized, excluded = [], []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        wq = getattr(mod, "weight_quantizer", None)
        (quantized if (wq is not None and wq.is_enabled) else excluded).append((name, mod.weight.numel()))
total_params = sum(p.numel() for p in model.parameters())
q_params = sum(n for _, n in quantized)
looped = [k for k, v in visits.items() if v == 2 * len(batches)]
stats = {
    "recipe": args.recipe, "modelopt": modelopt.__version__, "torch": torch.__version__,
    "calib_file": args.calib, "calib_samples": len(batches), "calib_tokens": ntok, "calib_seconds": round(calib_s, 1),
    "layers_visited_twice_per_sample": looped,
    "quantized_linear_modules": len(quantized), "excluded_linear_modules": [n for n, _ in excluded],
    "quantized_params": q_params, "total_params_hf_loaded": total_params,
    "fraction_params_quantized": round(q_params / total_params, 4),
    "quant_cfg": qc,
}

# Sanity: logits stay finite through the fake-quant model.
with torch.inference_mode():
    b = {k: v.to(dev) for k, v in batches[0].items()}
    stats["fakequant_logits_finite"] = bool(torch.isfinite(model(**b, use_cache=False).logits).all())

os.makedirs(args.out, exist_ok=True)
t0 = time.time()
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=args.out)
stats["export_seconds"] = round(time.time() - t0, 1)
json.dump(stats, open(os.path.join(args.out, "quantization_stats.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "excluded_linear_modules"}, indent=2))
print("excluded linear modules:", len(excluded))
