#!/usr/bin/env bash
# Shows the status and health of the vllm-reranker (:8000) and vllm-embed (:8001) containers.
set -euo pipefail

docker ps -a --filter "name=vllm-reranker" --filter "name=vllm-embed" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

echo
for pair in "vllm-reranker:8000" "vllm-embed:8001"; do
  name="${pair%%:*}"; port="${pair##*:}"
  if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
    echo "[$name] :$port  -> healthy"
  else
    echo "[$name] :$port  -> unhealthy / unreachable"
  fi
done
