"""Numerical tests for the fused SenseNova mHC router kernels.

Run inside the SGLang image (add the checkpoint mount for the real-weight test):

  docker run --rm --gpus all -v ~/sensenova-loop-nvfp4/runtime-v2:/w \
    -v ~/sensenova-loop-nvfp4/release/SenseNova-Loop-9B-Preview-NVFP4:/model:ro \
    --entrypoint python3 lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17 \
    /w/tests/test_mhc_fused.py [--bench]
"""

import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OVL = os.path.join(ROOT, "overlay/python/sglang/srt")
CKPT = os.environ.get("MHC_CKPT", "/model/model.safetensors")
ITERS = 20
DEV = "cuda"
FAIL = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fused = load("sglang.srt.layers.sensenova_mhc_fused", f"{OVL}/layers/sensenova_mhc_fused.py")
load("sglang.srt.configs.sensenova_loop", f"{OVL}/configs/sensenova_loop.py")
model = load("sglang.srt.models.sensenova_loop", f"{OVL}/models/sensenova_loop.py")
Router = model.SenseNovaMHCRouter
SETTINGS = dict(streams=2, iterations=ITERS, fixed_identity=False, pre_scale=1.0)


def ref_coeffs(logits, iterations=ITERS):
    """Verbatim tail of SenseNovaMHCRouter._forward_reference (streams=2)."""
    pre = logits[..., :2].sigmoid()
    post = 2 * logits[..., 2:4].sigmoid()
    mix = logits[..., 4:].unflatten(-1, (2, 2))
    for _ in range(iterations):
        mix = mix - torch.logsumexp(mix, dim=-1, keepdim=True)
        mix = mix - torch.logsumexp(mix, dim=-2, keepdim=True)
    return pre, post, mix.exp()


def ulp(a, b):
    """Max ULP distance between finite values (sign-aware, fp32 or bf16)."""
    def ordered(x):
        if x.dtype == torch.bfloat16:
            i = x.view(torch.int16).int()
            return torch.where(i < 0, -(i & 0x7FFF), i)
        i = x.view(torch.int32).long()
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    ok = torch.isfinite(a.float()) & torch.isfinite(b.float())
    d = (ordered(a) - ordered(b)).abs()[ok]
    return int(d.max()) if d.numel() else 0


def compare(name, refs, outs, require_bitwise=True, quiet=False):
    def same(r, o):
        if r.shape != o.shape or r.dtype != o.dtype:
            return False
        nr, no = torch.isnan(r), torch.isnan(o)
        if not torch.equal(nr, no):
            return False
        # bit pattern equality on non-NaN entries (distinguishes -0.0 / +0.0)
        it = torch.int16 if r.element_size() == 2 else torch.int32
        return torch.equal(r[~nr].view(it), o[~no].view(it))

    bit = all(same(r, o) for r, o in zip(refs, outs))
    diffs, ulps = [], []
    for r, o in zip(refs, outs):
        fin = torch.isfinite(r) & torch.isfinite(o)
        diffs.append(float((r.float() - o.float())[fin].abs().max()) if fin.any() else 0.0)
        ulps.append(ulp(r, o))
        nan_ok = torch.equal(torch.isnan(r), torch.isnan(o))
        inf_ok = torch.equal(torch.isinf(r), torch.isinf(o))
        if not (nan_ok and inf_ok):
            bit = False
    status = "BITWISE" if bit else "DIFF"
    if not (quiet and bit):
        print(f"  {name:<52} {status:<8} maxabs={max(diffs):.3e} maxulp={max(ulps)}")
    if require_bitwise and not bit:
        FAIL.append(name)
    return bit, max(diffs), max(ulps)


def test_coeffs():
    print("== fused Sinkhorn/sigmoid kernel vs torch reference ==")
    g = torch.Generator(device=DEV).manual_seed(0)
    Ts = [1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 1051, 4096]
    for scale in [0.1, 1.0, 10.0, 100.0, 1e4]:
        for T in Ts:
            l = torch.randn(T, 8, device=DEV, generator=g) * scale
            r, o = ref_coeffs(l), fused.mhc_coeffs(l, ITERS)
            bit = all(torch.equal(a, b) for a, b in zip(r, o))
            if not bit:
                compare(f"randn scale={scale} T={T}", r, o)
        print(f"  randn scale={scale:<8g} T in {Ts}: "
              f"{'all BITWISE' if not FAIL else 'see failures'}")
    # pre/post/mix scale-dependent denormal outputs (exp/sigmoid in (2^-149, 2^-126))
    x = torch.linspace(-110, -80, 4096, device=DEV)
    l = torch.zeros(4096, 8, device=DEV)
    l[:, 0] = x; l[:, 2] = x; l[:, 3] = -x
    l[:, 4] = 0; l[:, 5] = x; l[:, 6] = x; l[:, 7] = 0
    compare("denormal-range sigmoid / exp", ref_coeffs(l), fused.mhc_coeffs(l, ITERS))
    inf, nan = float("inf"), float("nan")
    rows = [
        [0] * 8,
        [5.0] * 8,
        [-3.0] * 8,
        [1e30, -1e30, 1e30, -1e30, 1e30, -1e30, -1e30, 1e30],
        [3e38, -3e38, 88.8, -88.8, 3e38, 3e38, 3e38, 3e38],
        [inf, -inf, inf, -inf, inf, 0, 0, 0],
        [0, 0, 0, 0, -inf, 0, 0, 0],
        [0, 0, 0, 0, -inf, -inf, 0, 0],
        [0, 0, 0, 0, -inf, -inf, -inf, -inf],
        [0, 0, 0, 0, inf, inf, inf, inf],
        [0, 0, 0, 0, inf, -inf, -inf, inf],
        [nan, 0, nan, 0, nan, 0, 0, 0],
        [0, nan, 0, nan, 0, 0, 0, nan],
        [0, 0, 0, 0, 1e-45, -1e-45, 1e-40, 0],
        [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0],
        [100, -100, 50, -50, 100, -100, -100, 100],
        [0, 0, 0, 0, 0, -95, -95, 0],
    ]
    l = torch.tensor(rows, device=DEV, dtype=torch.float32)
    compare("extremes (+/-inf, nan, 3e38, ties, -0)", ref_coeffs(l), fused.mhc_coeffs(l, ITERS))
    for i, row in enumerate(rows):
        lr = torch.tensor([row], device=DEV, dtype=torch.float32)
        r, o = ref_coeffs(lr), fused.mhc_coeffs(lr, ITERS)
        if not all(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
                   and torch.equal(torch.isnan(a), torch.isnan(b)) for a, b in zip(r, o)):
            print("   mismatch row", row, r, o)
    # non-contiguous logits: slice of a wider buffer and a transposed view
    big = torch.randn(1051, 20, device=DEV, generator=g) * 3
    l = big[:, 5:13]
    compare("non-contiguous slice [T, 8] of [T, 20]", ref_coeffs(l), fused.mhc_coeffs(l, ITERS))
    lt = torch.randn(8, 333, device=DEV, generator=g).t()
    compare("transposed view (stride (1, T))", ref_coeffs(lt), fused.mhc_coeffs(lt, ITERS))
    l3 = torch.randn(4, 7, 8, device=DEV, generator=g)
    compare("3-D logits [4, 7, 8]", ref_coeffs(l3), fused.mhc_coeffs(l3, ITERS))
    for it in (1, 5, 50):
        l = torch.randn(64, 8, device=DEV, generator=g) * 4
        compare(f"iterations={it}", ref_coeffs(l, it), fused.mhc_coeffs(l, it))
    z = torch.empty(0, 8, device=DEV)
    assert [t.shape for t in fused.mhc_coeffs(z, ITERS)] == [(0, 2), (0, 2), (0, 2, 2)]
    print("  T=0                                          ok (no launch)")


def graph_run(fn, static_in):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(*static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*static_in)
    return g, out


def test_graph():
    print("== CUDA graph capture/replay vs eager ==")
    gen = torch.Generator(device=DEV).manual_seed(1)
    for T in (1, 4, 8, 128):
        buf = torch.randn(T, 8, device=DEV, generator=gen)
        g, out = graph_run(lambda x: fused.mhc_coeffs(x, ITERS), (buf,))
        for trial in range(3):
            buf.copy_(torch.randn(T, 8, device=DEV, generator=gen) * 10 ** trial)
            g.replay()
            torch.cuda.synchronize()
            compare(f"coeffs graph T={T} trial={trial} vs reference", ref_coeffs(buf), out)
        r = Router(hidden_size=4096, settings=SETTINGS).to(DEV)
        with torch.no_grad():
            r.weight.normal_(0, 0.02, generator=gen); r.bias.normal_(0, 1, generator=gen)
        hs = torch.randn(T, 8192, device=DEV, generator=gen).bfloat16()
        g, out = graph_run(lambda x: r(x)[:3], (hs,))
        hs.copy_(torch.randn(T, 8192, device=DEV, generator=gen).bfloat16() * 3)
        g.replay()
        torch.cuda.synchronize()
        eager = r(hs)[:3]
        compare(f"router graph T={T} vs fused eager", eager, out)
        compare(f"router graph T={T} vs reference (fused linear, info)",
                r._forward_reference(hs)[:3], out, require_bitwise=False)
        r.use_fused_linear = False
        g, out = graph_run(lambda x: r(x)[:3], (hs,))
        hs.copy_(torch.randn(T, 8192, device=DEV, generator=gen).bfloat16())
        g.replay()
        torch.cuda.synchronize()
        compare(f"exact-path router graph T={T} vs reference", r._forward_reference(hs)[:3], out)
        r.use_fused_linear = True
        br = torch.randn(T, 4096, device=DEV, generator=gen).bfloat16()
        mix = torch.rand(T, 2, 2, device=DEV, generator=gen)
        post = torch.rand(T, 2, device=DEV, generator=gen) * 2
        g, out = graph_run(lambda b_, h_, m_, p_: Router.merge(b_, h_, m_, p_), (br, hs, mix, post))
        mix.copy_(torch.rand(T, 2, 2, device=DEV, generator=gen))
        g.replay()
        torch.cuda.synchronize()
        compare(f"merge graph T={T} vs reference", (Router._merge_reference(br, hs, mix, post),), (out,))


def checkpoint_routers():
    if not os.path.exists(CKPT):
        print(f"  (checkpoint {CKPT} not mounted: using random router weights)")
        return None
    from safetensors import safe_open
    out = {}
    with safe_open(CKPT, "pt", device="cpu") as f:
        for layer in (4, 15, 27):
            for kind in ("mhc_attn", "mhc_ffn"):
                p = f"model.language_model.layers.{layer}.{kind}"
                out[p] = (f.get_tensor(p + ".weight"), f.get_tensor(p + ".bias"))
    return out


def make_routers():
    weights = checkpoint_routers()
    routers = {}
    gen = torch.Generator(device="cpu").manual_seed(2)
    if weights is None:
        weights = {"random": (torch.randn(8, 8192, generator=gen) * 0.02,
                              torch.randn(8, generator=gen))}
    for name, (w, b) in weights.items():
        r = Router(hidden_size=4096, settings=SETTINGS)
        assert w.dtype == torch.float32 and tuple(w.shape) == (8, 8192)
        r.weight.data = w.to(DEV)
        r.bias.data = b.to(DEV)
        routers[name] = r
    return routers


def fp64_coeffs(r, hs):
    """Router coefficients from float64 norm + linear (rounded to fp32 logits),
    with the fp32 reference tail: a ground truth for the summation-order error."""
    x = hs.double()
    n = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    logits = F.linear(n, r.weight.double(), r.bias.double()).float()
    pre, post, mix = ref_coeffs(logits, r.iterations)
    return pre, post, mix


def maxdiff(xs, ys):
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(xs, ys))


def test_router_e2e():
    print("== end-to-end router forward + merge: exact fused path (bitwise) ==")
    routers = make_routers()
    gen = torch.Generator(device=DEV).manual_seed(3)
    Ts = (1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 1051, 4096)
    merge_big = [0.0, 0]
    for name, r in routers.items():
        assert r.use_fused and r.use_fused_linear
        r.use_fused_linear = False
        ok = True
        for T in Ts:
            for scale in (1.0, 30.0):
                hs = (torch.randn(T, 8192, device=DEV, generator=gen) * scale).bfloat16()
                ref = r._forward_reference(hs)
                out = r(hs)
                assert ref[3] is False and out[3] is False
                ok &= compare(f"{name} T={T} x{scale} fwd", ref[:3], out[:3], quiet=True)[0]
                ref_m = Router._merge_reference(ref[0], hs, ref[1], ref[2])
                out_m = Router.merge(out[0], hs, out[1], out[2])
                if T <= 32:
                    ok &= compare(f"{name} T={T} x{scale} merge", (ref_m,), (out_m,),
                                  quiet=True)[0]
                else:
                    # cuBLAS switches to tensor-core bmm kernels whose fp32
                    # accumulation is not IEEE round-to-nearest; the fused merge
                    # rounds p0 + p1 exactly once (RN). Record the rate.
                    frac = float((ref_m != out_m).float().mean())
                    ulpd = ulp(ref_m, out_m)
                    merge_big[0] = max(merge_big[0], frac)
                    merge_big[1] = max(merge_big[1], ulpd)
                    if frac > 1e-4:
                        FAIL.append(f"{name} T={T} merge frac {frac}")
        r.use_fused_linear = True
        print(f"  {name:<40} fwd T in {Ts} + merge T<=32: {'all BITWISE' if ok else 'DIFF'}")
    print(f"  merge T>=64 vs cuBLAS bmm: worst frac of differing bf16 elems {merge_big[0]:.2e},"
          f" max {merge_big[1]} bf16 ulp")

    print("== fused RMS-norm + linear path (not bitwise; FP32, different sum order) ==")
    print("  diffs: fused-vs-ref | ref-vs-fp64 | fused-vs-fp64 (max abs over pre,post,mix);"
          " branch/merge: frac of bf16 elems differing, max abs")
    worst = {}
    for name, r in routers.items():
        for T in (1, 4, 8, 16, 32, 64, 1051):
            hs = (torch.randn(T, 8192, device=DEV, generator=gen)).bfloat16()
            ref = r._forward_reference(hs)
            out = r(hs)
            truth = fp64_coeffs(r, hs)
            ref_c, out_c = (ref[2], ref[1]), (out[2], out[1])
            pre_ref = ref_coeffs(F.linear(
                hs.float() * torch.rsqrt(hs.float().square().mean(-1, keepdim=True) + 1e-6),
                r.weight, r.bias))[0]
            d_fr = maxdiff(out_c, ref_c)
            d_rt = maxdiff(ref_c, (truth[1], truth[2]))
            d_ft = maxdiff(out_c, (truth[1], truth[2]))
            bfrac = float((out[0] != ref[0]).float().mean())
            bmax = float((out[0].float() - ref[0].float()).abs().max())
            m_ref = Router._merge_reference(ref[0], hs, ref[1], ref[2])
            m_out = Router.merge(out[0], hs, out[1], out[2])
            mfrac = float((m_out != m_ref).float().mean())
            mmax = float((m_out.float() - m_ref.float()).abs().max())
            for k, v in (("coef", d_fr), ("ref_vs_fp64", d_rt), ("fused_vs_fp64", d_ft),
                         ("branch_frac", bfrac), ("merge_frac", mfrac)):
                worst[k] = max(worst.get(k, 0.0), v)
            if T in (1, 8, 64, 1051):
                print(f"  {name.split('layers.')[-1]:<16} T={T:<3} coef {d_fr:.1e} | {d_rt:.1e} | "
                      f"{d_ft:.1e}; branch {bfrac:.2e} ({bmax:.1e}); merge {mfrac:.2e} ({mmax:.1e})")
    print(f"  worst: {worst}")
    if worst["fused_vs_fp64"] > 4 * max(worst["ref_vs_fp64"], 1e-6):
        FAIL.append("fused linear error vs fp64 much larger than reference's")


def bench():
    print("== microbenchmark: per call, CUDA graph replay of 20 back-to-back calls ==")
    r = list(make_routers().values())[0]
    gen = torch.Generator(device=DEV).manual_seed(4)

    def fused_exact(x):
        r.use_fused_linear = False
        try:
            return r(x)
        finally:
            r.use_fused_linear = True

    def fused_linear(x):
        r.fused_linear_max_t = 1 << 30
        try:
            return r(x)
        finally:
            r.fused_linear_max_t = model._MHC_FUSED_LINEAR_MAX_T

    for T in (1, 4, 8, 16, 32, 64, 128, 512, 1051, 4096):
        hs = torch.randn(T, 8192, device=DEV, generator=gen).bfloat16()
        _, mix, post, _ = r._forward_reference(hs)
        res = {}
        for label, fn in (
            ("ref_fwd", lambda x: r._forward_reference(x)),
            ("exact_fwd", fused_exact),
            ("linear_fwd", fused_linear),
            ("ref_merge", lambda x: Router._merge_reference(x[:, :4096], x, mix, post)),
            ("fused_merge", lambda x: Router.merge(x[:, :4096], x, mix, post)),
        ):
            n = 20 if T <= 512 else 4
            g, _ = graph_run(lambda x: [fn(x) for _ in range(n)], (hs,))
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            reps = 10
            t0 = time.perf_counter()
            for _ in range(reps):
                g.replay()
            torch.cuda.synchronize()
            res[label] = (time.perf_counter() - t0) / (reps * n) * 1e6
            del g
        print(f"  T={T:<4} " + " ".join(f"{k}={v:7.1f}us" for k, v in res.items())
              + f" | x96 fwd {res['ref_fwd'] * 0.096:.2f}->{res['exact_fwd'] * 0.096:.2f}"
              f"/{res['linear_fwd'] * 0.096:.2f} ms, merge {res['ref_merge'] * 0.096:.2f}"
              f"->{res['fused_merge'] * 0.096:.2f} ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        test_coeffs()
        test_graph()
        test_router_e2e()
        if "--bench" in sys.argv:
            bench()
    print("FAILED: " + ", ".join(FAIL) if FAIL else "ALL BITWISE CHECKS PASSED")
    sys.exit(1 if FAIL else 0)
