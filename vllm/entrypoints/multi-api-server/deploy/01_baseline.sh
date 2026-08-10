export VLLM_USE_FUSED_RMS_ROPE=0

vllm serve /data/Qwen3-ASR-1.7B \
    --trust-remote-code \
    --port 8001 \
    --limit-mm-per-prompt '{"audio": 1}' \
    -tp 1 \
    --profiler-config '{
        "profiler": "torch",
        "torch_profiler_dir": "/public/home/weishb/vllm_profile"
    }'
