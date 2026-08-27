#!/usr/bin/env bash
# Stops and removes the vllm-reranker and vllm-embed containers.
set -euo pipefail
docker rm -f vllm-reranker vllm-embed 2>&1 || true
echo "Stopped."
