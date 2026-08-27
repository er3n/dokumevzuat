#!/usr/bin/env bash
# Serves the first-stage dense model (Nomic v2-moe) on vLLM (nightly, Docker).
#
# WARNING - ACCURACY ISSUE, NOT RECOMMENDED: in live testing, the embeddings vLLM
# produces do not match local sentence-transformers. For the same 174 corpus documents,
# local-vs-vLLM cosine similarity is min=0.82 / mean=0.94 (expected ~0.999+ at matching
# dtype) - IDENTICAL in both fp16 and bf16 (recall@10=0.5206 vs. 0.6566 locally), so the
# cause isn't dtype/precision. The pooling config (mean pool, all tokens, L2-normalize)
# matches the HF modules.json exactly, so the likely source is an inconsistency in vLLM's
# NomicMoE (bert_with_rope.py) reimplementation (e.g. a RoPE/router detail) - the root
# cause was not investigated, since the first stage isn't the actual bottleneck (locally
# it finishes from cache in ~1s) and this wasn't worth more time. The script and the
# --first-stage-backend vllm code path remain for reference/experimentation but should
# NOT be used in place of the default (local). See README.md's vLLM backend note.
#
# vLLM natively supports nomic-ai/nomic-embed-text-v2-moe (registry.py:
# "NomicBertModel" -> bert_with_rope.NomicBertModel, the MoE config fields --
# moe_top_k/moe_every_n_layers/num_experts -- match the model's config.json exactly);
# no trust_remote_code or custom hf_overrides needed, --runner pooling alone resolves to
# the embed task automatically.
#
# --gpu-memory-utilization is kept low (NOT the 0.85 default): the reranker container
# (serve_qwen3_reranker_vllm.sh) already reserves ~27GB/32GB on the same physical GPU -
# since both containers run at once, trying 0.85 here too would OOM. For a 305M
# active-parameter, non-generative (pooling-only) model, 0.15 (~5GB) is more than enough.
#
# --max-model-len 512: the model's config.json has max_position_embeddings=512 (RoPE) -
# the local sentence-transformers setup was also observed at max_seq_length=512
# (dense_baseline.py's MAX_SEQ_LENGTH_OVERRIDES has no entry for nomic, so it falls back
# to the ST default, already 512). Anything higher is rejected by pydantic validation
# (RoPE NaN risk).
#
# --dtype bfloat16: vLLM's default ("auto") picks float16 for this model. float32 was
# tried but NomicMoE's triton kernel exceeds the shared-memory limit in fp32 and raises
# OutOfResources (131072 > hardware limit 101376). bfloat16 both works and is safer than
# fp16 (same exponent range as fp32) - though per the WARNING above, dtype was never the
# actual root cause (fp16 and bf16 gave identical results).

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-embed}"
MODEL="${MODEL:-nomic-ai/nomic-embed-text-v2-moe}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "VLLM_WSL2_ENABLE_PIN_MEMORY=1" \
  -p 8001:8000 \
  --ipc=host \
  vllm/vllm-openai:nightly \
  --model "$MODEL" \
  --runner pooling \
  --max-model-len 512 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.15

echo "Container: $CONTAINER_NAME  Model: $MODEL"
echo "Logs: docker logs -f $CONTAINER_NAME"
echo "Health: curl -s http://localhost:8001/health"
