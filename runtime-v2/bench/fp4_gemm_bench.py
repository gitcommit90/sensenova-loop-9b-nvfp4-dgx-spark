"""Microbench NVFP4 W4A4 GEMM backends at SenseNova-Loop decoder shapes (run inside the SGLang image)."""
import torch, sys
from flashinfer import fp4_quantize, mm_fp4
torch.manual_seed(0); dev = "cuda"
SHAPES = {"gate_up": (4096, 24576), "down": (12288, 4096), "gdn_qkvz": (4096, 12288), "attn_qkv": (4096, 10240), "o_proj": (4096, 4096)}
MS = [int(m) for m in (sys.argv[1] if len(sys.argv) > 1 else "1,4,8,16,32,64").split(",")]
BACKENDS = ["cudnn", "cutlass", "b12x", "cake", "cutedsl_low_latency", "auto"]
def gs(t): return (6.0 * 448.0 / t.abs().max().float()).float()
def timeit(fn, n=50):
    for _ in range(3): fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1e3
kda = None
try:
    import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as q
    q._has_runtime = lambda d: True
    from sglang.kernels.ops.gemm import try_qwen3x_nvfp4_gemm as kda
except Exception as ex: print("kda import fail", ex)
tot = {}
for name, (K, N) in SHAPES.items():
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02; wg = gs(w)
    wq, ws = fp4_quantize(w, wg, 16, False, True)
    gbytes = N * K * (0.5 + 1 / 16) / 1e9
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); xg = gs(x)
        xq, xs = fp4_quantize(x, xg, 16, False, True); alpha = (1.0 / (xg * wg)).reshape(1)
        ref = (x.float() @ w.float().t())
        row = []
        for be in BACKENDS + (["kda"] if kda else []):
            try:
                out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                if be == "kda":
                    f = lambda: kda(xq, wq, xs, ws, alpha, torch.bfloat16, N)
                    if f() is None: raise RuntimeError("unsupported shape")
                else:
                    f = lambda: mm_fp4(xq, wq.t(), xs, ws.t(), alpha, torch.bfloat16, out, backend=be)
                o = f(); o = out if o is None else o
                err = ((o.float() - ref).norm() / ref.norm()).item()
                us = timeit(f); row.append(f"{be}={us:6.1f}us({gbytes/us*1e6:4.0f}GB/s,e{err:.3f})")
                tot.setdefault((be, M), 0); tot[(be, M)] += us
            except Exception as ex:
                row.append(f"{be}=FAIL[{str(ex)[:40]}]")
        print(f"{name:9s} M={M:3d} " + " ".join(row), flush=True)
print("sum over 5 shapes (us):", {f"{b}@{m}": round(v, 1) for (b, m), v in sorted(tot.items(), key=lambda t: (t[0][1], t[1]))})
