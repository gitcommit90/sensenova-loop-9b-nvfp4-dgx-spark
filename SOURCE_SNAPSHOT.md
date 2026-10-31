# Source and artifact provenance

This companion mirrors the runtime/quantization/evaluation files from the original NVFP4 release.

- Upstream model: `sensenova/SenseNova-Loop-9B-Preview` @ `27abea779b1e6634b136805b6519371739d551f7`.
- Original NVFP4 Hugging Face publication: `a57aa9f5d18ae48aa23a5bd53cb7624f514e0e87`.
- Weight file and large tokenizer are distributed through [the HF model repository](https://huggingface.co/gitcommit90/SenseNova-Loop-9B-Preview-NVFP4), not GitHub.
- `RELEASE_SHA256SUMS` is the **complete HF snapshot** manifest; it is not a checksum command for this source-only checkout.
- Figures and documentation added by @dgx-agent. No model weights, runtime semantics, or measured results changed.
- Future optimization experiments are intentionally separate from this functional baseline.
