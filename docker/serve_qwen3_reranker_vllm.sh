#!/usr/bin/env bash
# Serves Qwen3-Reranker-8B in bf16 (unquantized) on vLLM (nightly, Docker).
# Purpose: measure the speed of vLLM's own (FlashInfer/FLASH_ATTN) backend against the
# SDPA/eager fallback that the local sentence-transformers CrossEncoder falls back to on
# RTX 5090/sm_120 due to missing flash-attn. bitsandbytes/FP8 quantization is currently
# broken/unverified for Qwen3-Reranker in vLLM (vllm-project/vllm#33970) — hence bf16 first.
#
# --hf_overrides: the original Qwen3-Reranker is a generative model (scores via yes/no
# token probability); vLLM converts it to Qwen3ForSequenceClassification and loads the
# yes/no vector difference from lm_head into the classifier head
# (is_original_qwen3_reranker=true).
# --chat-template: the query/document/instruct template that same conversion expects
# (Qwen's official example, docker/qwen3_reranker_template.jinja).
# --max-model-len 8192: the longest chunk in the corpus is ~2558 tokens (p99=1331), but
# once train+test are combined (--include-train) the longest question also reaches 1945
# tokens (never seen in the n=154 test-only set) — the two combined plus the chat-template
# overhead exceeded 4096 and vLLM returned "400 Bad Request" (silently blowing up around
# query ~200 in the n=813 raw set). 8192 is well under the model's native 32768 and doesn't
# strain the KV cache budget (gpu-memory-utilization 0.85).

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-reranker}"
MODEL="${MODEL:-Qwen/Qwen3-Reranker-8B}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$REPO_DIR/docker/qwen3_reranker_template.jinja:/qwen3_reranker_template.jinja:ro" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "VLLM_WSL2_ENABLE_PIN_MEMORY=1" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:nightly \
  --model "$MODEL" \
  --runner pooling \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template /qwen3_reranker_template.jinja \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

echo "Container: $CONTAINER_NAME  Model: $MODEL"
echo "Logs: docker logs -f $CONTAINER_NAME"
echo "Health: curl -s http://localhost:8000/health"
