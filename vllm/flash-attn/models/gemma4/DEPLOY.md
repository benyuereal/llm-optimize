# gemma-4-31B-it-AWQ · flash attention fp8 KV (head_dim=512) 部署指南

> 第二阶段优化：让 MTP draft 模型的 full_attention 层 (head_dim=512) 走 flash mixed kernel，替代慢的 aiter 2D kernel。
>
> **效果**：batch 4 TPOT 35.02ms（triton 基准 ~79ms，**~2.25x 加速**），MTP 接受率 98.08%、接受长度 3.94（精度无损）。

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

**性能**（batch 4，5120 input + 1024 output，全并发）：

```bash
# 与阶段一同一套 bench 命令，公平对比
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm bench serve \
    --backend vllm \
    --base-url http://localhost:8001 \
    --model gemma4 \
    --tokenizer /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
    --dataset-name random \
    --random-input-len 5120 \
    --random-output-len 1024 \
    --num-prompts 4 \
    --seed 42
```

> **务必连跑两轮取第二轮稳态**：第一轮含 warmup（编译/图捕获），偏慢。
> 关键指标：`Mean TPOT (ms)`、`Mean TTFT (ms)`、`Benchmark duration (s)`。
>
> 注意：不要加 `--request-rate 1`（每秒发 1 个请求会让后发请求排队，把排队时间
> 算进 TTFT，造成 TTFT 虚高）。默认 `request-rate=inf` 全并发灌入，与阶段一一致。

基准结果（本环境实测，4 并发，5120 input + 1024 output，request-rate=inf）：

| 指标 | triton 基准 | flash (本优化) | 提升 |
|------|-----------|--------------|------|
| Mean TPOT (ms) | ~79 | **35.02** | **~2.25x** |
| Median TPOT (ms) | ~79 | **34.81** | ~2.25x |
| P99 TPOT (ms) | — | **36.16** | — |
| Output 吞吐 (tok/s) | — | **107.02** | — |
| MTP 接受率 (%) | ~96 | **98.08** | 持平/略升 |
| 接受长度 (mean) | ~3.8 | **3.94** | 持平/略升 |
| 位置0/1/2 接受率 (%) | — | **98.56 / 97.98 / 97.69** | — |

> TTFT 与阶段一（TRITON_ATTN）持平（prefill 走 flash mixed kernel 在 kernel 级比
> triton 快 1.46~4.33x，端到端 TTFT 不退化）。

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

详细改动见 [`flash-attention-cutlass/README.md`](../../../../flash-attention-cutlass/README.md) 的"改动详情"章节。
