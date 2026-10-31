def cat(n):
    n2 = n.lower()
    if "block_scaled" in n2 or "fp4" in n2 or "nvfp4" in n2 or "cudnn" in n2 and "gemm" in n2 or "sm120" in n2 or "quantize_with_block" in n2 or "cvt_fp16_to_fp4" in n2 or "e2m1" in n2: return "FP4 GEMM/quant"
    if "gemv" in n2 or "gemm" in n2 or "sgemm" in n2 or "cublas" in n2 or "dot_kernel" in n2 or "wmma" in n2 or "splitk" in n2 or "nvjet" in n2: return "other GEMM/GEMV"
    if "delta_rule" in n2 or "causal_conv" in n2 or "gdn" in n2 or "fla" in n2 or "chunk_" in n2: return "GDN/recurrent"
    if "attn" in n2 or "attention" in n2 or "flash" in n2 and "norm" not in n2 or "decode" in n2 and "rope" not in n2: return "attention"
    if any(s in n2 for s in ("exp_kernel", "log_kernel", "masked_fill", "absfunctor", "compareeq", "maxnan", "sum_functor", "add<float>", "sigmoid", "functor_add", "logsumexp")): return "elementwise/reduce (fp32, mHC-like)"
    if "norm" in n2: return "norm"
    return "other"
