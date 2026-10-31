#!/usr/bin/env bash
# Serve SenseNova-Loop-9B-Preview-NVFP4 (W4A4 NVFP4 + MTP) with the optimized runtime v2 (DGX Spark / GB10, SM121).
#
#   serve_sglang.sh [MODEL_DIR] [--no-mtp] [extra sglang args...]
#
# MODEL_DIR defaults to the directory containing this runtime folder. Binds 127.0.0.1:${PORT:-30080}.
# Same pinned image, PR #47281 overlay and patches as the v1 release runtime, plus:
#   * fused Triton mHC router kernels (overlay/.../layers/sensenova_mhc_fused.py)
#       MHC_MODE=fast      (default) single-kernel FP32 rmsnorm+linear+Sinkhorn(20 it)+branch, fused merge
#       MHC_MODE=exact     fused bitwise-exact Sinkhorn, cuBLAS FP32 router linear + torch merge
#                          (greedy outputs bitwise identical to the v1 runtime)
#       MHC_MODE=reference original eager PyTorch router (v1 behaviour and speed)
#   * FR-Spec hot-vocab draft head for MTP: TOKEN_MAP=<runtime>/tokenmap/hot_64k.pt (default) | none.
#     Only the draft proposes from the 64k ids; the target verifies against the full vocabulary.
# FP4 GEMMs: flashinfer cuDNN backend (native W4A4 block-scaled FP4 tensor-core kernels).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=$(realpath "${1:-$HERE/..}"); [ $# -gt 0 ] && shift
CACHE=${SGL_CACHE:-$HOME/.cache/sensenova-sglang}; mkdir -p "$CACHE"
IMG=lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17@sha256:78142f9f4b7fb3ee27f59065615dc7bb263f643ed256c9299c3861086ade4d47
SGL=/sgl-workspace/sglang/python/sglang
OVL=${OVERLAY:-$HERE/overlay}/python/sglang
case "${MHC_MODE:-fast}" in
  fast) MHC_ENV=(-e SENSENOVA_MHC_FUSED=1 -e SENSENOVA_MHC_FUSED_LINEAR=1 -e SENSENOVA_MHC_FUSED_MERGE=1) ;;
  exact) MHC_ENV=(-e SENSENOVA_MHC_FUSED=1 -e SENSENOVA_MHC_FUSED_LINEAR=0 -e SENSENOVA_MHC_FUSED_MERGE=0) ;;
  reference) MHC_ENV=(-e SENSENOVA_MHC_FUSED=0) ;;
  *) echo "MHC_MODE must be fast|exact|reference" >&2; exit 2 ;;
esac
TOKEN_MAP=${TOKEN_MAP:-$HERE/tokenmap/hot_64k.pt}
SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
AUX=()
if [ "$TOKEN_MAP" != none ]; then
  TOKEN_MAP=$(realpath "$TOKEN_MAP")
  AUX=(-v "$TOKEN_MAP:/aux/token_map.pt:ro"); SPEC+=(--speculative-token-map /aux/token_map.pt)
fi
if [ "${1:-}" = "--no-mtp" ]; then SPEC=(); shift; fi
MOUNTS=()
for f in $(cd "$OVL" && find . -name '*.py'); do MOUNTS+=(-v "$OVL/${f#./}:$SGL/${f#./}:ro"); done
NAME=${NAME:-sensenova-loop-nvfp4}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e SGLANG_DISABLE_SILU_FP4_QUANT_FUSION="${SGLANG_DISABLE_SILU_FP4_QUANT_FUSION:-0}" \
  "${MHC_ENV[@]}" \
  -v "$MODEL:/model:ro" "${MOUNTS[@]}" "${AUX[@]}" \
  -v "$CACHE:/root/.cache" \
  ${DOCKER_EXTRA:-} \
  --entrypoint python3 "$IMG" -m sglang.launch_server \
  --model-path /model --served-model-name "${SERVED:-sensenova-loop-9b-nvfp4}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30080}" \
  --dtype bfloat16 --fp4-gemm-backend flashinfer_cudnn \
  --mem-fraction-static "${MEMFRAC:-0.45}" --context-length "${CTX:-32768}" --max-running-requests "${MAXREQ:-16}" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  "${SPEC[@]}" "$@"
echo "started container $NAME (MHC_MODE=${MHC_MODE:-fast}, token map: ${TOKEN_MAP}); wait for: curl -s http://${HOST:-127.0.0.1}:${PORT:-30080}/v1/models"
