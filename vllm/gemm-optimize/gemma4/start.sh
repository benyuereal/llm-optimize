#!/bin/bash
# ============================================================
# gemma-4-31B-it-AWQ-4bit vllm serve 启动脚本
# 配置: TP=4 最优 + aiter w4a16 gemm 加速 (group_size=32)
# ============================================================
set -e

export PATH=/opt/hyhal/bin:/opt/dtk/bin:$PATH

# ---- 锁定 4 卡高频 ----
echo "[start.sh] 锁定 DCU 高性能模式..."
for i in 0 1 2 3; do
  rocm-smi -d $i --setperflevel manual >/dev/null 2>&1 || true
  rocm-smi -d $i --setsclk 6 >/dev/null 2>&1 || true
done
echo "[start.sh] 锁频完成: sclk=760MHz"

# ---- 环境变量 ----
export HIP_VISIBLE_DEVICES=0,1,2,3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export VLLM_RPC_TIMEOUT=1800000
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_MLA_DISABLE=1
export VLLM_USE_FLASH_MLA=0
export LLAMA_NN=0
export VLLM_USE_FLASH_ATTN_PA=1
export VLLM_USE_V1=0
export VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- 启用 aiter w4a16 gemm 加速 (替换 triton w4a16) ----
export VLLM_AITER_W4A16_PATCH=0
export AITER_ROOT=/public/home/weishb/aiter

# ---- 启动 vllm serve (TP=4) ----
echo "[start.sh] 启动 vllm serve (TP=4, aiter w4a16 patch 启用)..."
vllm serve /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name gemma4 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --optimization-level 3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --language-model-only \
    --async-scheduling \
    --performance-mode throughput \
    --max-num-batched-tokens 16384 \
    --speculative-config '{"method": "mtp", "model": "/data/zq/models/gemma-4-31B-it-assistant", "num_speculative_tokens": 3}'
