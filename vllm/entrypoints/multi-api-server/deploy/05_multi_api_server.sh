#!/bin/bash
# 生产配置: 囊括本项目全部优化, 70 并发端到端 ~440ms
#   - --api-server-count 4: 多 API server 进程并行化 CPU 前端 (核心, 零源码改动)
#   - VLLM_ROCM_TRANSPOSE_WEIGHT=1: 权重转置 (原生环境变量)
#   - VLLM_USE_FUSED_RMS_ROPE=0: 关 fused RMS+RoPE
#   - --max-num-seqs 128 / --max-num-batched-tokens 8192: 调大 batch
# 拓扑: 1 EngineCore (单卡 tp=1) + 4 API server, ZMQ 共享引擎, 同端口 SO_REUSEPORT
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
