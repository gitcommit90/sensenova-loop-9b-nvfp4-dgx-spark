# SenseNova-Loop-9B NVFP4 + MTP on GB10 (sm_121): compatibility scout, 2026-10-30

Labels: **[V]** verified here (code read at a pinned SHA, or a command run on tux). **[I]** inferred, not yet run.
Top criterion (added by the user): **MTP speculative decoding plus an NVFP4 target in one runtime.**

## Ranking
| # | Path | GEMM | MTP | Status |
|---|---|---|---|---|
| 1 | **SGLang PR #47281 overlay + 2 small patches** on `lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17` | **W4A4** flashinfer `mm_fp4` (cutlass/cudnn) | NEXTN via the stock `Qwen3_5ForCausalLMMTP`, BF16 draft (needs patch 2) | best; patches untested end-to-end |
| 2 | Same as #1 with `--fp4-gemm-backend marlin` | **W4A16** Marlin NVFP4 | same | fallback if W4A4 accuracy or kernels misbehave |
| 3 | vLLM 0.27.1 (local image) + out-of-tree sensenova_loop | W4A4 CUTLASS works on sm121 [V] | must port loop **and** wire MTP | big effort |
| 4 | LightLLM `main` | **no NVFP4/ModelOpt** (awq/w8a8/fp8 only) [V] | has sensenova_loop + MTP (`eagle_with_att`) [V] | BF16/FP8 reference only; fa3 defaults are Hopper-only |
| 5 | HF transformers + custom NVFP4 Linear | W4A4 `torch._scaled_mm` fp4 or flashinfer [V] | **none**: we would write draft+verify ourselves | correctness oracle only |

## 1. SGLang PR #47281 [V]
- Open, unmerged, CI red. Fork `WANDY666/sglang` branch `feat/sensenova-loop`. **Head `336c54ac1e721197806d94b6929f567446a59693`.**
  Merge base `c10e0e17735f7981c7cc46891eed6908b7701e58`, which is exactly the commit of nightly `nightly-dev-cu13-20261029-c10e0e17`, so the overlay is exact.
  Validated by the author on H800 BF16 only (logit cosine 0.984 to 0.9999 vs HF). **No MTP and no quantized test.**
- Overlay = 7 runtime files (in `overlay/python/sglang/`): `srt/models/sensenova_loop.py` (new), `srt/configs/sensenova_loop.py` (new),
  `srt/configs/{__init__,model_config}.py`, `srt/layers/rotary_embedding/mrope_rope_index.py`,
  `srt/multimodal/processors/qwen_vl.py`, `srt/utils/hf_transformers/common.py`. They need the very new `layer_boundary.MHCState`,
  `sglang.kernels.ops.layernorm.mhc`, and `runtime_context`, so **don't overlay onto release images ≤ v0.5.22**.
- Loop: the config expands to 56 execution layers (0..3, 4..27×2, 28..31). The KV and GDN pools are per execution layer. Re-visits are
  `copy.copy` of the physical layer with new RadixAttention/RadixLinearAttention objects (layer_id = execution index) and shared params.
  `load_weights` remaps `model.language_model.layers.{phys}.<any suffix>` to the first execution index, so weight_scale, weight_scale_2 and input_scale
  need no extra mapping. mHC routers are raw FP32 Parameters with F.linear and are never quantized.
- **Patch 1, `patches/sensenova_loop_quant.patch` (required for NVFP4):** `get_layer()` builds decoder layers **without
  `quant_config`**, so every decoder Linear would be BF16 and packed uint8 weights fail to load. Pass `quant_config=quant_config`. Also build
  the physical layer with prefix `...layers.{physical_id}`, because ModelOpt `exclude_modules` uses physical names and layers 28..31 run at
  execution ids 52..55 (otherwise their `in_proj_a/b` would not be excluded). Alternative: rewrite exclude_modules to globs
  (`*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*`, `lm_head`, `*visual*`, `mtp*`).
- Base code that already handles NVFP4 [V]: the GDN packed loaders cover weight/weight_scale/input_scale. Scalar `weight_scale_2` with a tuple shard
  is broadcast by `MergedColumnParallelLinear.weight_loader_v2`. The BF16 fused in_proj fast-path turns itself off when quantized.
- FP4 backend: `initialize_fp4_gemm_config()` picks `auto`, which resolves to cutedsl on sm100, marlin on 8.0 ≤ cc < 10, and **`flashinfer_cutlass`** otherwise (sm121).

## 2. MTP (top criterion)
- (a) **The PR does not implement MTP.** The draft-arch rewrite in `configs/model_config.py` (Qwen3.5 list, ~L1169) does not include
  `SenseNovaLoopForConditionalGeneration`, so a NEXTN draft would instantiate the whole loop model [V]. Even with the arch added,
  `SenseNovaLoopTextConfig.layers_block_type` follows `layer_types` (56 entries, [0] = linear_attention), so the single MTP layer would be
  built as GDN instead of full attention [V, code]. **Patch 2, `patches/model_config_mtp_draft.patch`:** for draft + SenseNovaLoop, deep-copy the
  config, set `num_hidden_layers=1`, `layer_types=["full_attention"]`, `full_attention_interval=1`, `execution_plan=(0,)`, `mhc=None`,
  `num_nextn_predict_layers=1`, and arch `Qwen3_5ForCausalLMMTP`. This mirrors the existing Qwen4Exp block. [Patch written, not run.]
- Draft class `Qwen3_5ForCausalLMMTP` (`models/qwen3_5_mtp.py`) loads names `mtp.*` (fc, pre_fc_norm_embedding/hidden,
  layers.0.{self_attn,mlp,norms}, norm). Upstream `mtp.safetensors` has exactly these 15 BF16 tensors [V], and `embed_tokens` and
  `lm_head` are shared from the target via `get_embed_and_head`. The draft takes the target's final-norm hidden state. The PR's loop forward returns
  `final_norm(...)`, the same as Qwen3.5 [V code]. Whether SenseNova trained MTP on that same tensor is [I] (the README says MTP is "Available").
- Flags [I, standard SGLang]: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4` (the draft model path defaults to the target). GDN spec-verify state is per layer_id, so visits are distinct [I].
- (b) NVFP4 target + BF16 draft: `_mtp_quant_config()` returns **None** for `modelopt_fp4` with a serialized NVFP4 checkpoint, so the draft is built BF16 with
  no patch [V]. A draft-vs-main quantization mismatch is auto-resolved for draft models [V]. Keep `mtp*` in exclude_modules regardless.
- Checkpoint layout SGLang expects: `mtp.*` keys must be **listed in `model.safetensors.index.json`**. The loader iterates the index. The
  "add unindexed mtp.safetensors" hack is GLM-only. Upstream already indexes them into `mtp.safetensors` (871 keys) [V]. ModelOpt
  `export_hf_checkpoint` carries unplaced weights such as MTP heads into the export and adds them to exclusions [V code]. **After export, verify that `mtp.*` (15 keys)
  is in the new index. If not, copy `mtp.safetensors` and add its keys to the index.**
- (c) LightLLM: has `lightllm/models/sensenova_loop` and `test/start_scripts/sensenova_loop/sensenova_loop_mtp.sh`
  (`--mtp_mode eagle_with_att --mtp_step 3 --mtp_draft_model_dir <separate dir>`, fa3 attention) at main `ccf4e33d0ce9…`, but there is
  no FP4 quantization module in `lightllm/common/quantization`, so it does not meet the NVFP4 requirement. Useful as an MTP acceptance-rate reference on another GPU.
- (d) The HF runtime cannot use `mtp.*` (ignored keys), so draft and verify would have to be written by hand. It ranks last.

## 3. Images [V]
- Local: `vllm/vllm-openai:v0.27.1-aarch64` and `...-dflash2-nvfp4-hotfix-aarch64` (vllm 0.27.1, torch 2.13.0+cu130, flashinfer
  0.6.16.post3, transformers 5.15.0). No local SGLang image or venv (`~/sglang-cache` holds only JIT caches; `~/venvs` holds vllm025 and vllm-026).
- Pulled (the 1 allowed pull, since it is the exact PR base): `lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17`,
  index `sha256:78142f9f4b7fb3ee27f59065615dc7bb263f643ed256c9299c3861086ade4d47`, arm64
  `sha256:5efa08069f2822f313fc3e5764ce72238f72cb9ca8744e9fa0f46358fd2dd643` (15.7 GB). Its pyproject pins flashinfer 0.7.2.

## 4. vLLM 0.27.1 [V facts / I effort]
Has `Qwen3_5ForConditionalGeneration` and `Qwen3_5MTP`, plus the NVFP4 linear kernels cutlass (works on sm121), marlin and emulation (flashinfer cutedsl is sm10x only).
A loop port must give each visit distinct Attention/GDN instances (vLLM keys KV and mamba state by layer prefix) and make `Qwen3_5MTP` see
the loop's output. That is about 1 to 2 days with CUDA-graph and prefix-cache risk. Not recommended while SGLang is viable.

## 5. Kernel smoke tests on GB10 [V]
`docker run --rm --gpus all -v $PWD:/w:ro --entrypoint python3 vllm/vllm-openai:v0.27.1-aarch64 /w/smoke_fp4.py` (and `smoke_fp4_b.py`).
Peak GPU memory 141 MB. N = K = 4096, ModelOpt scale convention (gs = 448·6/amax, alpha = 1/(gx·gw)).
```
flashinfer mm_fp4: cutlass OK, cudnn OK, auto OK; trtllm & cute-dsl -> BackendSupportedError (capability 121)
kernel vs emulated-FP4 reference rel_err 0.00166 (cutlass & cudnn)   [FP4 vs BF16 noise on random data: 0.134]
M=128 : cutlass 44us (98 TFLOPS)  cudnn 73us
M=2048: cutlass 225us (306 TFLOPS) cudnn 221us | bf16 torch 779us
M=1   : cutlass 1066us (!)  cudnn 63us        -> decode: prefer --fp4-gemm-backend flashinfer_cudnn (or autotune)
vllm ops.cutlass_scaled_fp4_mm OK; torch._scaled_mm(float4_e2m1fn_x2, e4m3 swizzled scales) OK (identical results)
flashinfer mm_bf16_fp4 (W4A16): unsupported on cc121 (cutlass/trtllm); cudnn path needs cuDNN >= 9.23.1 (image 9.20)
```
SGLang-image checks (PR unit tests with overlay, flashinfer 0.7.2 mm_fp4, Marlin W4A16, config expansion): `./run_sglang_smoke.sh`.
**NOT YET RUN:** the image pull was still in progress at hand-off (`pull.log`; pid of `docker pull` still alive). Run the script once it finishes.

## 6. ModelOpt [V unless marked]
- PyPI stable **0.47.0** pins `transformers<5.15` and is **incompatible** with 5.18.0. **0.48.0rc1** (2026-10-19) pins `>=5.5,<5.19`, so use it in a fresh venv.
- `default_disabled_quantizers.yaml` disables `*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*`, `*linear_attn.conv1d*`, `*lm_head*`,
  `*visual*`, `*vision_tower*`, `mtp.*`, `*router*`, `*mlp.gate.*`, and Embedding. GDN-state quantizers are off by default.
  `NVFP4_DEFAULT_CFG` = max calibration with w4a4_nvfp4_nvfp4. mhc_* are not nn.Linear, so they are not wrapped.
- Modules visited twice: quantizers are per module, and max calibration takes the running max over all calls, so both visits are covered [I]. Export walks each physical module once.
- Remote code: `hf_model_type` is only a key for MoE specs and special cases, and the dense path is generic. It copies `*.py`. The resmooth step does `model(torch.ones(1,2))` [I: fine, text-only].
- Output: `hf_quant_config.json` `{"producer":{"name":"modelopt",...},"quantization":{"quant_algo":"NVFP4","group_size":16,
  "kv_cache_quant_algo":null,"exclude_modules":[prefix-wildcards, physical names]}}`, plus config.json `quantization_config.quant_method="modelopt"`.
  Tensors: `weight` uint8 [N,K/2], `weight_scale` e4m3 [N,K/16], `weight_scale_2` fp32, `input_scale` fp32.
- NVIDIA playbook (build.nvidia.com/station/nvfp4-quantization): Spark uses `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` with ModelOpt
  0.41.0 (`examples/llm_ptq/scripts/huggingface_example.sh --quant nvfp4 --tasks quant`) and `trtllm-serve`. Station uses vllm 25.12.post1 with ModelOpt 0.47.0.
  Both are too old for transformers 5.18, so use them only as a pattern.

## 7. Serve command (recipe, [I])
```
docker run --rm --gpus all --ipc=host -p 30000:30000 -v $OUT:/model:ro <7 overlay -v mounts, see run_sglang_smoke.sh> \
  lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17 python3 -m sglang.launch_server --model-path /model \
  --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cudnn --mem-fraction-static 0.5 --port 30000 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
Port 30000 avoids 8000/3000/18080-18105.

## 8. Risks
- The PR is unmerged with CI red and BF16-only validation. Patches 1 and 2 are new and unrun against real weights. MTP on the loop model in SGLang is untested by anyone [I].
- The physical-prefix change only renames layers 28..31. Anything that parses the layer id from the prefix would see the physical id [I].
- GDN triton/fla kernels and spec-decode GDN state tracking on sm121 have not been exercised. M=1 cutlass mm_fp4 is slow, so benchmark cudnn.
- Looping doubles the use of every quantized weight. Keep mHC, in_proj_a/b, lm_head, vision and MTP in BF16, and evaluate against BF16 (and measure MTP acceptance against BF16).
