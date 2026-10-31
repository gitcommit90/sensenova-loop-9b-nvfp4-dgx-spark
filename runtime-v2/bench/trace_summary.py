"""Summarize a torch-profiler chrome trace: GPU kernel time by category, busy vs wall, per-graph-replay breakdown."""
import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
k = [e for e in ev if e.get("cat") == "kernel"]
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
def cat(n):
    n2 = n.lower()
    if "block_scaled" in n2 or "fp4" in n2 or "nvfp4" in n2 or "cudnn" in n2 and "gemm" in n2 or "sm120" in n2 or "quantize_with_block" in n2 or "cvt_fp16_to_fp4" in n2 or "e2m1" in n2: return "FP4 GEMM/quant"
    if "gemv" in n2 or "gemm" in n2 or "sgemm" in n2 or "cublas" in n2 or "dot_kernel" in n2 or "wmma" in n2 or "splitk" in n2 or "nvjet" in n2: return "other GEMM/GEMV"
    if "delta_rule" in n2 or "causal_conv" in n2 or "gdn" in n2 or "fla" in n2 or "chunk_" in n2: return "GDN/recurrent"
    if "attn" in n2 or "attention" in n2 or "flash" in n2 and "norm" not in n2 or "decode" in n2 and "rope" not in n2: return "attention"
    if any(s in n2 for s in ("exp_kernel", "log_kernel", "masked_fill", "absfunctor", "compareeq", "maxnan", "sum_functor", "add<float>", "sigmoid", "functor_add", "logsumexp")): return "elementwise/reduce (fp32, mHC-like)"
    if "norm" in n2: return "norm"
    return "other"
tot = collections.Counter(); cnt = collections.Counter(); byname = collections.Counter(); bync = collections.Counter()
for e in k:
    c = cat(e["name"]); tot[c] += e["dur"]; cnt[c] += 1; byname[e["name"][:110]] += e["dur"]; bync[e["name"][:110]] += 1
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
busy = sum(e["dur"] for e in k)
print(f"kernels={len(k)} window={((t1-t0)/1e3):.1f}ms gpu_kernel_sum={busy/1e3:.1f}ms ({100*busy/(t1-t0):.0f}% of window)")
for c, v in tot.most_common(): print(f"  {c:40s} {v/1e3:9.1f} ms  {100*v/busy:5.1f}%  n={cnt[c]}")
print("top kernels:")
for n, v in byname.most_common(22): print(f"  {v/1e3:8.1f} ms {bync[n]:7d}x  {n}")
# graph replays / forward annotations
ann = collections.Counter(); annd = collections.Counter()
for e in cpu:
    n = e["name"]
    if any(s in n for s in ("replay", "forward_", "run_batch", "verify", "draft", "cudaGraphLaunch")):
        ann[n[:80]] += 1; annd[n[:80]] += e["dur"]
print("cpu annotations (count, total ms):")
for n, c in ann.most_common(30): print(f"  {c:5d} {annd[n]/1e3:9.1f}  {n}")
rt = [e for e in ev if e.get("cat") == "cuda_runtime"]
rc = collections.Counter(); rd = collections.Counter()
for e in rt: rc[e["name"]] += 1; rd[e["name"]] += e["dur"]
print("cuda runtime:", [(n, rc[n], round(rd[n]/1e3, 1)) for n, _ in rd.most_common(8)])
