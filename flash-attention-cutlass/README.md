# flash-attention-cutlass: fp8 KV + head_dim=512 支持 (gfx936)

针对 Hygon DCU BW10 (gfx936) 优化的 flash-attention，让 gemma-4-31B-AWQ-4bit MTP draft 模型的 full_attention 层（global_head_dim=512）走 flash mixed kernel，替代慢的 aiter 2D kernel。

## 效果

| 指标 | triton 金标准 | flash (本包) | 提升 |
|------|--------------|--------------|------|
| Mean TPOT (batch4) | ~79ms | **35.63ms** | **2.2x** |
| MTP 接受率 | ~96% | **96.18%** | 持平 |
| 接受长度 | ~3.8 | **3.89** | 持平 |
| HumanEval pass@1 | 96.95% | **96.95%** | 精度无损 |

测试条件：TP4 + MTP(num_spec=3)，batch 4 (input 1024 / output 1024)。
（早期单次对比 TPOT 47.50ms vs 79.68ms = 1.68x；batch4 稳态 35.63ms = 2.2x。）

## 与 vllm/flash-attn/ 的关系

本目录是 **flash-attention-cutlass 源码 patch 的归属目录**（源码改了哪些、怎么编译）。
部署产物和一键安装脚本在 [`vllm/flash-attn/`](../vllm/flash-attn/)（客户直接用那个安装）。
两边的 patch / whl / new_files 内容一致，只是组织视角不同：
- 本目录 = 源码视角（patch 怎么来的、怎么重新编译）
- `vllm/flash-attn/` = 部署视角（whl 怎么装、服务怎么起、性能怎么验）

## 快速安装（客户直接用）

> whl 体积较大（200M），不随仓库分发。从本仓库 GitHub Release 附件下载
> `flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl`，放到 `dist/` 目录。

```bash
pip3 install --force-reinstall --no-deps dist/flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl
```

验证：
```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

## 启动服务

```bash
# TP4, FLASH attn + fp8_e5m2 KV + MTP
bash scripts/start_flash.sh
```

关键环境变量（已写在脚本里）：
- `ATTN_FLASH_HEAD512=1` — draft head_size=512 走 flash
- `ATTN_FLASH_PREFILL=1` — prefill 走 flash
- `--attention-backend ROCM_AITER_UNIFIED_ATTN`
- `--kv-cache-dtype fp8_e5m2`
- `--speculative-config '{"method":"mtp","model":"<draft模型路径>","num_speculative_tokens":3}'`

> 注意：`start_flash.sh` 里 `MODEL_DIR`、draft 模型路径、`HIP_VISIBLE_DEVICES` 需按实际环境调整。

## 目录结构

```
flash-attention-cutlass/
├── README.md                              # 本文件
├── dist/                                  # 编译产物 (whl 不入库, 从 GitHub Release 下载放到此)
│   └── flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl
├── patch/
│   ├── flash_fp8e5m2_512.patch            # 源码 patch (git apply)
│   ├── flash_fp8e5m2_512.md               # 改动清单详细说明
│   └── new_files/                         # patch 中新建的 2 个文件（便于审查）
│       ├── flash_fp8_fwd_hdim512_prefix_prefill_fp16.cpp
│       └── flash_fp8_fwd_hdim512_prefix_prefill_bf16.cpp
└── scripts/
    └── start_flash.sh            # 启动脚本
```

## 从源码重新编译（如需）

```bash
# 1. 拿到 flash-attention-cutlass 源码 (分支 path2-e4m3-qfp16, commit 6519c7f)
cd flash-attention-cutlass
git apply /path/to/patch/flash_fp8e5m2_512.patch

# 2. 编译
export FLASH_ATTN_OPT=1
export PATH=/opt/dtk/bin:$PATH
python3 setup.py sdist bdist_wheel

# 3. 安装
pip3 install --force-reinstall --no-deps dist/flash_attn-*.whl
```

## 改动概述

patch 共 19 个文件（17 改 + 2 新建），分两层：

1. **fp8 e4m3 mixed kernel 支持**：让 gfx936 的 fp8 mixed kernel 同时支持 e5m2 和 e4m3 KV 存储（e4m3 走软件 `__e4m32float` dequant，复用 e5m2 的 compact-LDS pipeline）
2. **head_dim=512 支持**：解除 flash 入口的 TORCH_CHECK 限制，让 fp8 + 512 在 decode/prefill 双路径走 flash mixed kernel

详细改动见 `patch/flash_fp8e5m2_512.md`。

## 环境要求

- Hygon DCU BW10 (gfx936)，DTK 驱动
- Python 3.10，vllm 0.23.0，torch 2.10.0
- 模型：gemma-4-31B-it-AWQ-4bit + gemma-4-31B-it-assistant (MTP draft)
