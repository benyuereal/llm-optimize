# gemma-4-31B-it-AWQ · flash attention fp8 KV (head_dim=512) 部署指南

> 第二阶段优化：让 MTP draft 模型的 full_attention 层 (head_dim=512) 走 flash mixed kernel，替代慢的 aiter 2D kernel。
>
> **效果**：batch 4 TPOT 35.63ms（triton 基准 ~79ms，**1.68~2x 加速**），MTP 接受率 96.18%（精度无损）。

## 硬件 / 软件前提

- Hygon DCU **BW10** (gfx936, 48 CUs, 32GB/卡)，TP4 用 GPU 0,4,2,3
- vllm 0.23.0（已含第一阶段 w4a16 gemm 优化）
- 已安装第一阶段 w4a16 优化（`vllm/aiter-w4a16/`），本阶段在其基础上叠加

## 部署 4 步

### 第 0 步 · 下载 flash_attn whl

whl 体积较大（200M），不随仓库分发，放在 GitHub Release 附件。先下载：

1. 到本仓库 GitHub Release 页面，下载 `flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl`
2. 放到 `vllm/flash-attn/dist/` 目录下

> 若已有 whl 在别处，也可不放进 dist，安装时指定：`WHL=/path/to/flash_attn-*.whl bash patch.sh install`

### 第 1 步 · 安装 flash_attn whl

```bash
cd /path/to/llm-optimize/vllm/flash-attn
bash patch.sh install
```

脚本会：卸载旧 flash_attn → 装 dist/ 下的 whl → 验证 import + 512 prefill 符号（fp16/bf16）。

### 第 2 步 · 验证安装

```bash
bash patch.sh status
```

应看到 `flash_attn version: 2.8.3+das.opt1...`。

### 第 3 步 · 启动服务

```bash
bash models/gemma4/start_flash.sh
```

该脚本设置关键环境变量，启动 vllm 服务（TP4, GPU 0,4,2,3），MTP draft 的 full_attention 自动走 flash mixed kernel。

启动成功日志应能看到：
- draft 模型 head512 的 `full_attention` 走 flash kernel（不再是 aiter 2D）
- CUDA graph capture 通过

### 第 4 步 · 性能 + 精度验证

**性能**（batch 4，1024 input + 1024 output）：

```bash
# 用 vllm 自带 benchmark_serving
python3 -m vllm.entrypoints.openai.api_serving_benchmark \
    --backend vllm-sse --model gemma-4-31b-it-awq-4bit \
    --num-prompts 4 --request-rate 1 \
    --prompt-len 1024 --output-len 1024
```

基准结果（本环境实测）：

| 指标 | triton 基准 | flash (本优化) | 提升 |
|------|-----------|--------------|------|
| Mean TPOT (ms) | ~79 | **35.63** | **2.2x** |
| Median TPOT (ms) | ~79 | **35.68** | 2.2x |
| MTP 接受率 (%) | ~96 | **96.18** | 持平 |
| 接受长度 | ~3.8 | **3.89** | 持平 |

**精度**：

```bash
# 用 HumanEval 测试（FP16 triton 基准 96.95%，flash 96.95%，完全一致）
# 或用固定 prompt 对比
python3 /path/to/precision_compare.py    # 8 固定 prompt, temperature=0
```

## 回退

```bash
bash patch.sh revert    # 卸载本 whl
# 重装官方 flash-attn 即可恢复
```

## 原理简述

gemma-4 MTP draft 模型有 `full_attention` 层，global_head_dim=512，num_global_key_value_heads=4。
原来走 aiter 的 2D attention kernel（head_dim=512 时很慢）。本优化扩展 flash-attention-cutlass
的 fp8 mixed kernel，使其支持 head_dim=512 的 prefix decode + prefix prefill，draft 侧改走 flash。

详细改动见上级目录 `flash_fp8_hdim512_patch.md`。
