#!/bin/bash
# Qwen3.8-27B FP8 rowwise (W8A8) — 参考 Qwen3-32B 8005 优化配置
# 卡 1 (空闲), 端口 8009, served-model-name = Qwen3.8-27B (匹配 eval 命令)
# max-model-len 262144 (模型 max_position_embeddings 上限; KV cache ~32GB 放得下)
# 日志统一放 /workspace/_logs/
set -e

MODEL_DIR=/models/Qwen3.8-27B-FP8-ROWWISE/
PORT=8009
CARD=1
LOG_DIR=/workspace/_logs
LOG=$LOG_DIR/vllm_qwen38_27b_fp8_serve.log
mkdir -p $LOG_DIR

echo "=== 检查端口 $PORT ==="
if fuser $PORT/tcp >/dev/null 2>&1; then
  echo "错误: 端口 $PORT 已被占用 (不自动杀, 避免误伤别人的服务)。换端口或手动处理。"
  exit 1
fi

echo "=== 启动 Qwen3.8-27B FP8 (卡 $CARD, 端口 $PORT) ==="
HIP_VISIBLE_DEVICES=$CARD nohup vllm serve $MODEL_DIR \
  --host 0.0.0.0 \
  --port $PORT \
  --served-model-name Qwen3.8-27B \
  --dtype auto \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --max-num-seqs 512 \
  --attention-backend FLASH_ATTN \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --language-model-only \
  --async-scheduling \
  --performance-mode throughput \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}' \
  > $LOG 2>&1 &

echo "PID: $!"
echo "日志: $LOG"
echo ""
echo "等待 ready ..."
for i in $(seq 1 90); do
  # 必须确认是 vllm 服务 (created=vllm), 避免误判其它框架的响应
  if curl -s -m 3 http://localhost:$PORT/v1/models 2>/dev/null | grep -q '"created":"vllm"'; then
    echo "READY (after ~$((i*10))s)"
    curl -s -m 5 http://localhost:$PORT/v1/models | head -c 300
    echo ""
    exit 0
  fi
  # 进程死了就提前退出
  if ! kill -0 $! 2>/dev/null; then
    echo "错误: vllm 进程已退出, 查看日志: tail -50 $LOG"
    exit 1
  fi
  sleep 10
done
echo "900s 内未 ready，查看日志: tail -f $LOG"
