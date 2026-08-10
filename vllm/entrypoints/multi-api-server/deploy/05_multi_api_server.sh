#!/bin/bash
# 参数化启动: bash start_apiserver.sh <count>
export VLLM_USE_FUSED_RMS_ROPE=0
export VLLM_ROCM_TRANSPOSE_WEIGHT=1
export VLLM_ASR_PROFILE=1

ASC=${1:-4}

vllm serve /data/Qwen3-ASR-1.7B \
    --trust-remote-code \
    --port 8001 \
    --limit-mm-per-prompt '{"audio": 1}' \
    -tp 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --api-server-count $ASC \
    --middleware asr_timing_mw.asr_timing_middleware
