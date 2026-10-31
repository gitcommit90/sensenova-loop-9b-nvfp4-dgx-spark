#!/usr/bin/env python3
"""Independent structural verification of an exported NVFP4 checkpoint (reads files only).

Checks against the upstream checkpoint: every upstream tensor is present (as BF16/FP32 or as an
NVFP4 packed weight + scales), packed shapes are consistent (uint8 [out, in/2], fp8 e4m3 block scales
[out, in/16], fp32 global scales), all scales finite/positive, mHC routers FP32, exactly one copy of
each physical (shared, looped) layer, MTP tensors present, config/quant metadata consistent.

usage: python -I scripts/30_verify_export.py EXPORT_DIR UPSTREAM_DIR OUT_JSON
"""
import json, os, sys, collections
import torch
from safetensors import safe_open

exp, up, out = sys.argv[1:4]


def index(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        wm = {}
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(d, f), "pt") as fh:
                    wm.update({k: f for k in fh.keys()})
    return wm


def header(d, wm):
    meta = {}
    for f in set(wm.values()):
        with safe_open(os.path.join(d, f), "pt") as fh:
            for k in fh.keys():
                sl = fh.get_slice(k)
                meta[k] = (sl.get_dtype(), tuple(sl.get_shape()), f)
    return meta


ewm, uwm = index(exp), index(up)
E, U = header(exp, ewm), header(up, uwm)
qcfg = json.load(open(os.path.join(exp, "hf_quant_config.json")))["quantization"]
cfg = json.load(open(os.path.join(exp, "config.json")))
rep = {"export_dir": exp, "quant_algo": qcfg.get("quant_algo"), "group_size": qcfg.get("group_size"),
       "kv_cache_quant_algo": qcfg.get("kv_cache_quant_algo"),
       "config_has_quantization_config": "quantization_config" in cfg,
       "n_tensors_export": len(E), "n_tensors_upstream": len(U)}
problems = []

packed = [k[: -len(".weight")] for k, (dt, _, _) in E.items() if k.endswith(".weight") and dt == "U8"]
rep["nvfp4_linear_modules"] = len(packed)
q_params = 0
for m in packed:
    dt, shp, _ = E[m + ".weight"]
    ref = U.get(m + ".weight")
    if ref is None:
        problems.append(f"packed {m} not in upstream"); continue
    o, i = ref[1]
    q_params += o * i
    if shp != (o, i // 2):
        problems.append(f"{m}: packed shape {shp} != {(o, i // 2)}")
    s = E.get(m + ".weight_scale")
    if s is None or s[0] != "F8_E4M3" or s[1] != (o, i // qcfg["group_size"]):
        problems.append(f"{m}: weight_scale {s}")
    for k in ("weight_scale_2", "input_scale"):
        t = E.get(f"{m}.{k}")
        if t is None or t[0] != "F32" or (len(t[1]) and t[1] != (1,)):
            problems.append(f"{m}: {k} {t}")

# Load all scales and check finite / positive.
bad_scales, n_scales = 0, 0
by_file = collections.defaultdict(list)
for k in E:
    if k.endswith(("weight_scale", "weight_scale_2", "input_scale")):
        by_file[E[k][2]].append(k)
for f, keys in by_file.items():
    with safe_open(os.path.join(exp, f), "pt") as fh:
        for k in keys:
            t = fh.get_tensor(k).float()
            n_scales += 1
            if not torch.isfinite(t).all() or (t[t != 0] <= 0).any() or (k.endswith("input_scale") and (t <= 0).any()):
                bad_scales += 1
                problems.append(f"bad scale {k}")
rep["scale_tensors_checked"] = n_scales
rep["bad_scale_tensors"] = bad_scales

# Upstream coverage: every upstream tensor present in export (either same name or packed).
missing = [k for k in U if k not in E]
rep["upstream_tensors_missing_in_export"] = missing[:20]
rep["n_upstream_missing"] = len(missing)
extra = [k for k in E if k not in U and not k.endswith(("weight_scale", "weight_scale_2", "input_scale"))]
rep["unexpected_extra_tensors"] = extra[:20]

mhc = {k: E[k][0] for k in E if ".mhc_" in k}
rep["mhc_tensors"] = len(mhc)
rep["mhc_all_fp32"] = all(v == "F32" for v in mhc.values()) and len(mhc) == 96
mtp = [k for k in E if k.startswith("mtp.")]
rep["mtp_tensors"] = len(mtp)
rep["mtp_dtypes"] = sorted({E[k][0] for k in mtp})

# Shared weights: layer ids present must be exactly 0..31 (no duplicated per-visit copies).
lids = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in E if "language_model.layers." in k})
rep["physical_layer_ids"] = f"{lids[0]}..{lids[-1]} ({len(lids)})"
if lids != list(range(32)):
    problems.append(f"layer ids {lids}")

dtypes = collections.Counter()
nbytes = collections.Counter()
size = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}
import math
for k, (dt, shp, _) in E.items():
    n = math.prod(shp)
    dtypes[dt] += n
    nbytes[dt] += n * size.get(dt, 0)
rep["elements_by_dtype"] = dict(dtypes)
rep["GB_by_dtype"] = {k: round(v / 1e9, 3) for k, v in nbytes.items()}
rep["total_file_GB"] = round(sum(os.path.getsize(os.path.join(exp, f)) for f in set(ewm.values())) / 1e9, 3)
total_up = sum(math.prod(s) for (_, s, _) in U.values())
rep["upstream_params_incl_mtp"] = total_up
rep["nvfp4_params"] = q_params
rep["fraction_params_nvfp4"] = round(q_params / total_up, 4)
rep["problems"] = problems
rep["ok"] = not problems and not missing and rep["mhc_all_fp32"] and len(mtp) > 0
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep, indent=2))
