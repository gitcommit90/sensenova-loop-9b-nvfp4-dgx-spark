# Source and artifact provenance

This companion mirrors the runtime/quantization/evaluation files from the original NVFP4 release.

- Upstream model: `sensenova/SenseNova-Loop-9B-Preview` @ `27abea779b1e6634b136805b6519371739d551f7`.
- Original NVFP4 Hugging Face publication: `a57aa9f5d18ae48aa23a5bd53cb7624f514e0e87`.
- Weight file and large tokenizer are distributed through [the HF model repository](https://huggingface.co/gitcommit90/SenseNova-Loop-9B-Preview-NVFP4), not GitHub.
- `RELEASE_SHA256SUMS` is the **complete HF snapshot** manifest; it is not a checksum command for this source-only checkout.
- Figures and documentation added by @dgx-agent. No model weights, runtime semantics, or measured results changed.
- Future optimization experiments are intentionally separate from this functional baseline.

## Runtime v2 update — October 31, 2026

Imported the finished Spark `~/sensenova-loop-nvfp4/runtime-v2` source/evidence snapshot.
Core launcher, overlay, tests' numerical logic and token-map tensors are preserved. Public packaging
repoints the screening helper to `evaluation/harness`, parameterizes the feature-test image and token-map
input paths, and omits session-local orchestration wrappers. No model weights or measured JSONs changed.
See `runtime-v2/bench/README.md` for portable reproduction commands and `runtime-v2/RESULTS.md` for results.
The original `runtime/` is retained byte-for-byte.
