#!/bin/bash
# ============================================================
# gemma-4-31B-it-AWQ-4bit vllm serve 启动脚本
# 配置: TP=4 + aiter w4a16 gemm + FLASH attn (ROCM_AITER_UNIFIED) + fp8_e5m2 KV + MTP
# 与基准 start.sh 唯一差异: attention-backend + kv-cache-dtype
# ============================================================
set -e

export PATH=/opt/hyhal/bin:/opt/dtk/bin:$PATH

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,4,2,3}
export HF_HUB_OFFLINE=1

echo "[start_flash.sh] 锁定 DCU 高性能模式 (cards: $HIP_VISIBLE_DEVICES)..."
for i in ${HIP_VISIBLE_DEVICES//,/ }; do
  rocm-smi -d $i --setperflevel manual >/dev/null 2>&1 || true
  rocm-smi -d $i --setsclk 6 >/dev/null 2>&1 || true
done
echo "[start_flash.sh] 锁频结果确认:"
for i in ${HIP_VISIBLE_DEVICES//,/ }; do
  sclk=$(rocm-smi -d $i --showclocks 2>/dev/null | grep -oE "sclk clock level: [0-9]+ \([0-9]+Mhz\)" | head -1)
  echo "  HCU[$i]: ${sclk:-未获取到频率}"
done

export VLLM_AITER_W4A16_PATCH=1
export ATTN_FLASH_PREFILL=1
export ATTN_FLASH_HEAD512=1  # draft模型head_size=512也走flash  # prefill也走flash (gemma4 fp8 KV已验证cos0.9986, 2.6-3.3x faster)

MODEL_DIR=${MODEL_DIR:-/data/zq/models/gemma-4-31B-it-AWQ-4bit/}
# MTP draft 模型路径: 默认与主模型同父目录下的 gemma-4-31B-it-assistant
# 可用 DRAFT_MODEL_DIR 环境变量覆盖
DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-$(dirname "$MODEL_DIR")/gemma-4-31B-it-assistant}

echo "[start_flash.sh] 启动 vllm serve (TP=4, FLASH + fp8_e5m2 KV)..."
echo "[start_flash.sh] 模型: $MODEL_DIR"
echo "[start_flash.sh] draft: $DRAFT_MODEL_DIR"
vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name gemma4 \
    --dtype float16 \
    --kv-cache-dtype fp8_e5m2 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
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
    --speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL_DIR\", \"num_speculative_tokens\": 3}"
