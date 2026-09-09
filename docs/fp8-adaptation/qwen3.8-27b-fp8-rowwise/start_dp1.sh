#!/bin/bash
# ============================================================
# Qwen3.8-27B-FP8-ROWWISE 单卡 DP1 server (卡 4) — 混合架构 TPOT 对比
#
# 用途:
#   与 Qwen3-32B DP1 (start_dp1.sh) 同配置对比 batch=128 decode TPOT。
#   Qwen3.8-27B 是混合架构 (Qwen3_5ForConditionalGeneration):
#     full_attention_interval=4 → 64 层里 16 层 full attention +
#     48 层 linear attention (mamba 式, 固定状态, 不走 paged flash attn)。
#   预期: flash attn (paged_attention) 占比大幅下降, GEMM 占比上升。
#
# 关键配置 (与 Qwen3-32B DP1 对齐, 公平对比):
#   1. 无投机解码 (原 start_qwen38_27b_fp8.sh 有 MTP, 会让 TPOT 虚低,
#      这里关掉以便和 Qwen3-32B 公平比 TPOT)。
#   2. --optimization-level 1 + custom_ops → 激活 A/B 融合。
#   3. --gpu-memory-utilization 0.85 (单卡 144GB)。
#   4. profiler-config → trace 写 /workspace/prof_traces_qwen38_dp1。
#
# 模型参数 (text_config):
#   hidden=5120, layers=64, heads=24, kv_heads=4, head_dim=256,
#   intermediate=17408, vocab=248320, full_attention_interval=4
#
# 用法:
#   bash start_dp1.sh
#   抓 trace:
#     curl -X POST http://localhost:8011/start_profile
#     vllm bench serve --base-url http://localhost:8011 \
#       --model /models/Qwen3.8-27B-FP8-ROWWISE/ --served-model-name Qwen3.8-27B \
#       --max-concurrency 128 --num-prompts 128 \
#       --input-len 1024 --output-len 1024 --ignore-eos --backend openai
#     curl -X POST http://localhost:8011/stop_profile
#
# 注意: 只按端口杀 (fuser -k 8011/tcp), 不动其它 server。
# ============================================================
set -e

export PATH=/opt/hyhal/bin:/opt/dtk/bin:$PATH

# 单卡: 卡 4
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4}
export HF_HUB_OFFLINE=1

# ---- aiter 融合开关: 只开 norm/act 量化融合 (opt1) ----
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_ROCM_USE_AITER_TRITON_ROPE=0

MODEL_DIR=${MODEL_DIR:-/models/Qwen3.8-27B-FP8-ROWWISE/}
PORT=${PORT:-8011}
LOG_DIR=/workspace/_logs
LOG=$LOG_DIR/vllm_qwen38_27b_dp1_serve.log
mkdir -p $LOG_DIR

# ---- 停掉旧 server (只按端口杀, 不动其它 server) ----
echo "[serve] 停掉旧 vllm server (port $PORT, 只按端口杀)..."
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 3

echo "[serve] 启动 Qwen3.8-27B DP1 vllm serve (卡 $HIP_VISIBLE_DEVICES, 端口 $PORT, NUMA node 7, 日志 $LOG)..."
# NUMA: 卡4=node7
numactl --cpunodebind=7 --membind=7 vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name Qwen3.8-27B \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --attention-backend FLASH_ATTN \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --optimization-level 1 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --language-model-only \
    --async-scheduling \
    --performance-mode throughput \
    --max-num-batched-tokens 262144 \
    --compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/prof_traces_qwen38_dp1", "torch_profiler_record_shapes": true, "torch_profiler_with_memory": false, "torch_profiler_with_stack": false}' \
    2>&1 | tee $LOG

echo "前台运行, Ctrl+C 停止服务。日志同时写入: $LOG"
