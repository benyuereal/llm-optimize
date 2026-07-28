export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
vllm serve \
        --model /models/GLM-5.2-Channel-INT8-w8a8 \
        -q slimquant_marlin \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 490036 \
        --port 8001 \
        --max-num-batched-tokens 18192 \
        -tp 8 \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs 64 \
        --block-size 64 \
        --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
	--kv-cache-dtype fp8_ds_mla \
        --served-model-name glm-5.2 \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --enable-auto-tool-choice \
        --chat-template-content-format string
