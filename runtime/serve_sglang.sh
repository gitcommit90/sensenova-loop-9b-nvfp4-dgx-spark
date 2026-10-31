#!/usr/bin/env bash
# Serve SenseNova-Loop-9B-Preview-NVFP4 with the pinned, tested SGLang runtime (DGX Spark / GB10, SM121).
#
#   runtime/serve_sglang.sh [MODEL_DIR] [--no-mtp] [extra sglang args...]
#
# MODEL_DIR defaults to the directory containing this runtime/ folder. Binds 127.0.0.1:${PORT:-30080}.
# Image: lmsysorg/sglang nightly-dev-cu13-20261029-c10e0e17 (pinned by digest) + SGLang PR #47281 model files
# (runtime/overlay, mounted read-only) incl. our two patches (runtime/patches): quant_config passthrough and
# MTP draft config. FP4 GEMMs: flashinfer cuDNN backend (native W4A4 block-scaled FP4 tensor-core kernels).
# W4A16 alternative: append `--fp4-gemm-backend marlin` AND export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
# (SGLang c10e0e17 otherwise feeds pre-quantized FP4 activations to Marlin and crashes at graph capture).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=$(realpath "${1:-$HERE/..}"); [ $# -gt 0 ] && shift
IMG=lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17@sha256:78142f9f4b7fb3ee27f59065615dc7bb263f643ed256c9299c3861086ade4d47
SGL=/sgl-workspace/sglang/python/sglang
OVL=$HERE/overlay/python/sglang
SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
if [ "${1:-}" = "--no-mtp" ]; then SPEC=(); shift; fi
MOUNTS=()
for f in $(cd "$OVL" && find . -name '*.py'); do MOUNTS+=(-v "$OVL/${f#./}:$SGL/${f#./}:ro"); done
NAME=${NAME:-sensenova-loop-nvfp4}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e SGLANG_DISABLE_SILU_FP4_QUANT_FUSION="${SGLANG_DISABLE_SILU_FP4_QUANT_FUSION:-0}" \
  -v "$MODEL:/model:ro" "${MOUNTS[@]}" \
  --entrypoint python3 "$IMG" -m sglang.launch_server \
  --model-path /model --served-model-name "${SERVED:-sensenova-loop-9b-nvfp4}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30080}" \
  --dtype bfloat16 --fp4-gemm-backend flashinfer_cudnn \
  --mem-fraction-static "${MEMFRAC:-0.45}" --context-length "${CTX:-32768}" --max-running-requests "${MAXREQ:-16}" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  "${SPEC[@]}" "$@"
echo "started container $NAME; wait for: curl -s http://${HOST:-127.0.0.1}:${PORT:-30080}/v1/models"
