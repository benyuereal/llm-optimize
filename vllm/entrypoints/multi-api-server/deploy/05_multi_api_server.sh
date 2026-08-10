#!/bin/bash
# 多 API server 甜点配置: 1 引擎 + 4 API server, 单端口, 单卡 tp=1
# 用法: bash 05_multi_api_server.sh [count]   (默认 4)
export VLLM_USE_FUSED_RMS_ROPE=0
export VLLM_ROCM_TRANSPOSE_WEIGHT=1

ASC=${1:-4}

vllm serve /data/Qwen3-ASR-1.7B \
    --trust-remote-code \
    --port 8001 \
    --limit-mm-per-prompt '{"audio": 1}' \
    -tp 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --api-server-count $ASC
