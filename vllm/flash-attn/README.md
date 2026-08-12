# 第二阶段优化 · flash attention fp8 KV + head_dim=512

> 优化 gemma-4 MTP draft 模型 full_attention 层的 attention kernel。
>
> **batch 4 TPOT 35.63ms（triton 基准 ~79ms，1.68~2.2x 加速），MTP 接受率 96.18%，精度无损（HumanEval 96.95% 与 triton 一致）。**

## 背景

gemma-4-31B-it-AWQ 的 MTP draft 模型有一个 `full_attention` 层：
- `global_head_dim = 512`（注意是 512，不是主模型的 256）
- `num_global_key_value_heads = 4`
- `num_speculative_tokens = 3`

这个 512 head_dim 的 attention 原来走 aiter 的 2D kernel，很慢，是 draft 侧的性能瓶颈。

## 优化内容

扩展 `flash-attention-cutlass` 源码，使其 fp8 mixed kernel 支持 head_dim=512：

1. **fp8 KV 支持**（Layer A，前一阶段已做）：flash mixed kernel 支持 fp8 e4m3/e5m2 KV cache，InputElement 模板参数化
2. **head_dim=512 支持**（Layer B，本阶段）：在 flash dispatch 链路加入 512 分支，新增 prefix decode + prefix prefill 的 512 入口和模板实例化

闪存路由：`aiter unified_attention()` + `ATTN_FLASH_HEAD512=1` → `hg_prefix_decode_varlen_fwd` / `hg_prefix_prefill_varlen_fwd` → flash mixed kernel。

## 与第一阶段的关系

| 阶段 | 目录 | 优化对象 | 效果 |
|------|------|---------|------|
| 第一阶段 | `vllm/aiter-w4a16/` | aiter triton w4a16 GEMM 替换 vllm 自带 kernel | TPOT 1.2~1.8x |
| **第二阶段** | **`vllm/flash-attn/`** | **flash fp8 KV + head_dim=512 替换 aiter 2D attention** | **draft full_attention 1.68~2.2x** |

两个阶段互不冲突，可叠加。本阶段在第一阶段基础上叠加。

## 目录结构

```
vllm/flash-attn/
├── README.md                          # 本文件
├── patch.sh                           # 一键安装/回退/状态 (装 whl + 打 vllm 侧 patch)
├── flash_fp8e5m2_512.patch            # flash-attention-cutlass 源码改动 patch (编译 whl 用, 已含在 dist/whl 里)
├── flash_vllm_fp8e5m2.patch           # vllm 侧 fp8_e5m2 patch (改 3 个 vllm 文件, install 时自动打)
├── flash_fp8_hdim512_patch.md         # flash 源码改动详细说明（两层改动, 19 文件）
├── dist/                              # whl 不入库, 从 GitHub Release 下载放到此
│   └── flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl  # 安装产物
├── new_files/                         # patch 中新增的两个 target 文件
│   ├── flash_fp8_fwd_hdim512_prefix_prefill_fp16.cpp
│   └── flash_fp8_fwd_hdim512_prefix_prefill_bf16.cpp
└── models/gemma4/
    ├── DEPLOY.md                      # 部署指南（4 步傻瓜式）
    └── start_flash.sh        # 启动脚本（含环境变量）
```

## 快速开始

```bash
# 0. 下载 whl (从 GitHub Release 附件) 放到 dist/
#    flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl

# 1. 安装
bash patch.sh install

# 2. 启动
bash models/gemma4/start_flash.sh

# 3. 验证（见 models/gemma4/DEPLOY.md）
```

## 源码位置

本阶段改两层：

1. **flash-attention-cutlass 源码**（与 vllm 同级目录，不在本目录）—— 新增 fp8_e5m2
   mixed kernel + head_dim=512 prefill 符号，编译成 whl。改动见 `flash_fp8_hdim512_patch.md`，
   对应 patch 文件 `flash_fp8e5m2_512.patch`。**已包含在 `dist/` 的 whl 里**，用户无需再编译，
   只需从 GitHub Release 下载 whl 即可。

2. **vllm 源码**（3 个文件，`patch.sh install` 自动打）—— 让 vllm 支持 `fp8_e5m2` KV cache：
   - `vllm/model_executor/layers/attention/attention.py` — 放行 e5m2（上游对 compressed-tensors
     模型一律报错 `fp8_e5m2 kv-cache is not supported with fp8 checkpoints.`，但 AWQ-4bit 模型
     checkpoint 无 KV scale，只是运行时想存 e5m2）
   - `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` — 读侧 view 用 e5m2 + 写侧走 triton
     （C++ `reshape_and_cache_flash` op 不支持 e5m2）
   - `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` — 写侧按字符串选 e5m2 dtype

   对应 patch 文件 `flash_vllm_fp8e5m2.patch`。不打这层，新容器启动会直接报上面的 ValueError。

## 编译说明

- 只编译 gfx936（BW10），不编译 gfx938（gfx938 的 fp8 GEMM utils 不支持 head_dim=512，会 static_assert）
- fp16 + bf16 两个 target 文件都需要（FP16_SWITCH 运行时派发，链接器需要两个实例化）
- 编译环境：dtk2604, python3.10

## 回退

```bash
bash patch.sh revert
```
