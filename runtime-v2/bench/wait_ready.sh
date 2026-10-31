#!/usr/bin/env bash
# wait until server on ${PORT:-30080} answers a generation; prints seconds waited
P=${PORT:-30080}; t0=$(date +%s)
for i in $(seq 1 360); do
  if curl -sf -m 5 http://127.0.0.1:$P/v1/models >/dev/null; then echo "ready after $(( $(date +%s)-t0 ))s"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME:-sensenova-loop-nvfp4}"; then echo "container died"; docker logs --tail 40 "${NAME:-sensenova-loop-nvfp4}"; exit 1; fi
  sleep 5
done; echo timeout; exit 1
