# Qwen3.8-27B-FP8-ROWWISE — Hygon DCU (BW1100) FP8 量化 + 性能优化

Qwen3.8-27B (qwen3_5 架构) 在海光 Hygon DCU (BW1100, gfx938) 上的
**FP8 rowwise (W8A8) 量化 + decode 性能优化** 方案。

量化配方与性能优化与 [qwen3-32b-fp8-rowwise](../qwen3-32b-fp8-rowwise/) 完全一致
(同一套 llm-compressor 配方 + 同一行 `--compilation-config` 优化), 本目录是
Qwen3.8-27B 的适配实例。

## 模型架构

| 参数 | 值 |
|------|-----|
| model_type | qwen3_5 |
| hidden_size | 5120 |
| intermediate_size | 17408 |
| num_attention_heads | 24 |
| num_key_value_heads | 4 (GQA) |
| head_dim | 256 |
| num_hidden_layers | 64 |
| max_position_embeddings | 262144 |

## 量化方案

使用 **llm-compressor** 的 `model_free_ptq` (data-free, 纯 safetensors 处理,
不加载模型) 把 Qwen3.8-27B (BF16, 源 `/models/Qwen3.8-27B`) 量化为
per-block FP8 rowwise (W8A8):

- **权重**: per-channel (rowwise) 量化, `strategy=channel`,
  observer 用 `memoryless_minmax` (data-free, 直接从权重张量算 per-channel scale)
- **激活**: per-token **动态**量化 (`strategy=token, dynamic=true`),
  推理时在线计算, 不存储
- **格式**: compressed-tensors `float-quantized`
- **忽略层**: `model.embed_tokens`, `lm_head` (保持 BF16)

量化脚本: `quantize_qwen3_8_27b_fp8_rowwise.py` (本目录)。

产物: `/models/Qwen3.8-27B-FP8-ROWWISE/`

> 环境备注: llm-compressor 跑在 `/workspace/venv-llmc`
> (`--system-site-packages`, 共享系统 ROCm torch), compressed-tensors 0.18.1。
> 直接用 `scheme="FP8_DYNAMIC"` 字符串会因 preset 解析问题失败,
> 必须用 `preset_name_to_scheme("FP8_DYNAMIC", [])` 再 model_copy 改 observer。

## 性能优化 (与 Qwen3-32B 相同)

启动命令必须带:

```
--compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}'
```

**根因**: 不带这行时 (custom_ops 默认 `none`), vllm 的 norm/act 量化 fusion pass
不生效, inductor 把激活量化自动融合成慢内核
`triton_red_fused__to_copy_abs_clamp_div_...` (171.9us, 占 decode GPU 时间 26.7%)。
带上后量化走 vllm 自定义 C++ 内核 (13.4us)。

**A/B 实测** (Qwen3-32B 同配方, 16 并发 × 512 输出):

| 指标 | 不带 (OFF) | 带 (ON) | 提升 |
|------|-----------|---------|------|
| Output tok/s | 371.97 | 488.00 | **+31%** |
| Mean TPOT | 42.43 ms | 32.17 ms | **−24%** |

详见 [qwen3-32b-fp8-rowwise/docs/custom_ops_quant_rootcause.md](../qwen3-32b-fp8-rowwise/docs/custom_ops_quant_rootcause.md)。

## 环境要求

| 项目 | 配置 |
|------|------|
| DCU | 海光 BW1100 (gfx938) |
| vLLM | 0.21.0 (vllm_hcu 定制版, HCU platform plugin) |
| lightop | 定制版 (HCU 侧 FP8 GEMM / 量化内核) |
| llm-compressor | editable 安装于 `/workspace/venv-llmc` |
| 量化源模型 | `/models/Qwen3.8-27B` (BF16) |
| 量化产物 | `/models/Qwen3.8-27B-FP8-ROWWISE/` |

## 全流程 (从零到跑通)

### 第 1 步: 量化

```bash
# 在 /workspace/venv-llmc 环境里跑 (共享系统 ROCm torch)
source /workspace/venv-llmc/bin/activate
python quantize_qwen3_8_27b_fp8_rowwise.py
# 产物: /models/Qwen3.8-27B-FP8-ROWWISE/
```

### 第 2 步: 启动

```bash
bash start.sh
# 卡 1, 端口 8009, served-model-name Qwen3.8-27B
# max-model-len 262144 (模型上限; humaneval max_tokens 50000 需要足够大的上下文)
# 日志: /workspace/_logs/vllm_qwen38_27b_fp8_serve.log
```

### 第 3 步: humaneval 评估

**先应用 evalscope 思考模式提取 patch**（必须，否则得分是假象）:

```bash
bash evalscope/evalscope_patch.sh install
```

**为什么必须 patch**: Qwen3.8-27B 是思考模式模型，HumanEval 回答 = 思考文本 +
闭标签 + 代码，且代码常不带 ``` 代码块。evalscope 1.11.1 原版 `_postprocess`
只认 ``` 代码块且取 `blocks[0]`，无代码块时返回整个字符串（含思考文本）→
当 Python 执行 → 语法错误 → **pass@1 仅 3.1%（提取假象，代码本身全对）**。

patch 后（保留思考开启，只修提取器）: 164 题真实失败样本回放，
可编译率 4.3% → **98.8%**。参考 `../glm-5.2-channel-int4-w4a8/evalscope/`
（同思路，本 patch 额外处理"无代码块 + 开标签在 prompt"的情况）。

```bash
evalscope eval \
    --model Qwen3.8-27B \
    --model-id Qwen3.8-27B \
    --eval-type openai_api \
    --api-url http://127.0.0.1:8009/v1/chat/completions \
    --api-key sk-dummy \
    --datasets humaneval \
    --eval-batch-size 64 \
    --generation-config '{"max_tokens": 50000, "temperature": 0.2, "top_p": 0.95, "batch_size": 64}'
```

> 注意: **不要**用 `enable_thinking: false` 关思考——思考模式对代码质量有帮助，
> 关掉反而可能降分。正确做法是保留思考 + 修提取器（本 patch）。

## 精度评测结果

### HumanEval (164 题, pass@1)

| 模型 | 端口 | 得分 |
|---|---|---|
| **Qwen3.8-27B FP8** (per-block rowwise) | 8009 | **98.17%** (161/164) |
| **Qwen3.8-27B BF16** | 8010 | **98.78%** (162/164) |
| Qwen3-32B (参考) | 8005 | 98.78% (162/164) |

FP8 与 BF16 差 1 题 (0.61pp), 在 164 题的采样噪声范围内 → **量化基本无损**。
(注: 未打 evalscope 提取 patch 时 FP8 只有 3.05%, 是提取假象, 见上"第 3 步"。)

### GPQA Diamond (198 题, 240k 重测)

GPQA 之前 `max_tokens=8192` 太小, 近一半题被截断判错 (假错)。加大到
`max_tokens=240000` 重测, 截断率降到 0%, 分数大幅回升:

| | 旧 8192 | 新 240k | 提升 |
|---|---------|---------|------|
| **FP8** | 67.68% | **91.92%** | **+24.24pp** |
| **BF16** | 68.69% | **91.92%** | **+23.23pp** |

两个评测均 198/198, 截断 0%。

**关键结论**: FP8 240k = 91.92%, BF16 240k = 91.92%, **两者完全一致** →
per-block rowwise FP8 量化对 GPQA 精度**零损失** (量化无损)。

> 教训: 思考模式模型 (Qwen3.8-27B) 的 `max_tokens` 必须给足, 否则长思考链被
> 截断 → 答案缺失 → 假错。GPQA 这类需要长推理的 benchmark 尤其敏感。

## 与 Qwen3-32B 的差异

| 项 | Qwen3-32B | Qwen3.8-27B |
|----|-----------|-------------|
| 架构 | qwen3 | qwen3_5 |
| 层数 | 64 | 64 |
| hidden | 5120 | 5120 |
| intermediate | 25600 | 17408 |
| head_dim | 128 | 256 |
| max_position | 40960 | 262144 |
| 量化配方 | 相同 | 相同 |
| 启动优化 | 相同 (custom_ops 一行) | 相同 |
