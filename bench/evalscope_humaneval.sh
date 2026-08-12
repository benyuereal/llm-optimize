#!/bin/bash
# ============================================================
# evalscope HumanEval 精度评测脚本 (对齐 start.sh 配置)
# 端口: 8001, served-model-name: gemma4, --dtype float16
# 用法:
#   bash evalscope_humaneval.sh                 # 默认本地模型
#   BASE_URL=http://x.x.x.x:8001 bash evalscope_humaneval.sh   # 远程容器
# ============================================================
set -e

BASE_URL=${BASE_URL:-http://127.0.0.1:8001/v1}
MODEL_NAME=${MODEL_NAME:-gemma4}
PARALLEL=${PARALLEL:-16}

echo "[evalscope] BASE_URL   = $BASE_URL"
echo "[evalscope] MODEL_NAME = $MODEL_NAME"
echo "[evalscope] PARALLEL   = $PARALLEL"

evalscope eval \
    --model-source vllm_chat \
    --model-args "base_url=${BASE_URL},model_name=${MODEL_NAME}" \
    --tasks humaneval \
    --dataset-human-eval-loc bigcode/humaneval \
    --eval-batch-size ${PARALLEL} \
    --work-dir ./outputs/evalscope \
    --generation-params "temperature=0.0,stop=</response>"
