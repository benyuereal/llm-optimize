export VLLM_USE_FUSED_RMS_ROPE=0

vllm serve /data/Qwen3-ASR-1.7B \
    --trust-remote-code \
    --port 8001 \
    --limit-mm-per-prompt '{"audio": 1}' \
    -tp 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8
