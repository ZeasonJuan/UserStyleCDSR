#!/usr/bin/env bash
set -euo pipefail

# ===== 路径与环境 =====
CACHE=hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 只用本地缓存（已经下完模型时可开）
export HF_HOME="$CACHE"
export TRANSFORMERS_OFFLINE=0

# (可选) Transformers 也用同一缓存目录
export TRANSFORMERS_CACHE="$CACHE"

# ===== 解析本地快照目录 =====
SNAPROOT="$CACHE/qwen3-8b"
MODEL_DIR="$SNAPROOT"

# 防御：确保有 config.json
test -f "$MODEL_DIR/config.json" || {
  echo "ERROR: $MODEL_DIR 里找不到 config.json，请确认模型已完整缓存。"
  exit 1
}
export VLLM_USE_RESPONSES_API=0
# ===== 启动 vLLM OpenAI 兼容服务 =====
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3-8b \
  --download-dir "$CACHE" \
  --tensor-parallel-size 1 \
  --max-model-len 8196 \
  --gpu-memory-utilization 0.9 \
  --port 8011