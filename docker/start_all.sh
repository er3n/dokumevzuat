#!/usr/bin/env bash
# Starts the reranker (Qwen3-Reranker-8B, :8000) + first-stage embedding (Nomic v2-moe,
# :8001) vLLM servers and waits until both report /health.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

wait_healthy() {
  local name="$1" port="$2" tries=60
  echo -n "[$name] waiting for health (:$port)"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
      echo "  ready."
      return 0
    fi
    echo -n "."
    sleep 5
  done
  echo "  TIMED OUT — check with 'docker logs $name'."
  return 1
}

bash "$REPO_DIR/docker/serve_qwen3_reranker_vllm.sh"
bash "$REPO_DIR/docker/serve_nomic_embed_vllm.sh"

wait_healthy vllm-reranker 8000
wait_healthy vllm-embed 8001

echo
echo "Both ready. Status: bash docker/status.sh   Stop: bash docker/stop_all.sh"
