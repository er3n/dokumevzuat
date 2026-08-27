#!/usr/bin/env bash
# Serves Qwen3-8B (generative, chat) in bf16 on vLLM (nightly, Docker).
# Purpose: a one-off, separate VRAM stage for query rewriting
# (src/benchmark/generate_rewrites.py) — MUST NOT run at the same time as the
# reranker (:8000) or first-stage embedding (:8001), all three don't fit in 32GB VRAM
# at once. Deliberately left out of docker/start_all.sh for that reason; usage:
#   docker rm -f vllm-reranker vllm-embed 2>/dev/null || true
#   bash docker/serve_qwen3_instruct_vllm.sh
#   python src/benchmark/generate_rewrites.py
#   docker rm -f vllm-instruct
#   bash docker/start_all.sh
#
# --runner: unlike the reranker script, no "pooling" here — this is a generative chat
# model, so vLLM's default chat-completion endpoint is used, and Qwen3's own chat
# template already ships in the HF repo (no extra --chat-template needed).
# enable_thinking=false is turned off via chat_template_kwargs on the
# /v1/chat/completions request by generate_rewrites.py (not in this script — that's
# the client's responsibility).

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-instruct}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "VLLM_WSL2_ENABLE_PIN_MEMORY=1" \
  -p 8002:8002 \
  --ipc=host \
  vllm/vllm-openai:nightly \
  --model "$MODEL" \
  --port 8002 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

echo "Container: $CONTAINER_NAME  Model: $MODEL"
echo "Logs: docker logs -f $CONTAINER_NAME"
echo "Health: curl -s http://localhost:8002/health"
