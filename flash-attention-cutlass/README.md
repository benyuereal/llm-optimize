# flash-attention-cutlass: fp8 KV + head_dim=512 支持 (gfx936)

针对 Hygon DCU BW10 (gfx936) 优化的 flash-attention，让 gemma-4-31B-AWQ-4bit MTP draft 模型的 full_attention 层（global_head_dim=512）走 flash mixed kernel，替代慢的 aiter 2D kernel。

## 效果

| 指标 | triton 金标准 | flash (本包) | 提升 |
|------|--------------|--------------|------|
| Mean TPOT (batch4) | ~79ms | **35.02ms** | **~2.25x** |
| Median TPOT (batch4) | ~79ms | **34.81ms** | ~2.25x |
| MTP 接受率 | ~96% | **98.08%** | 持平/略升 |
| 接受长度 (mean) | ~3.8 | **3.94** | 持平/略升 |
| 位置0/1/2 接受率 | — | **98.56 / 97.98 / 97.69** | — |
| HumanEval pass@1 | 96.95% | **96.95%** | 精度无损 |

测试条件：TP4 + MTP(num_spec=3)，batch 4 (input 1024 / output 1024)。
（早期单次对比 TPOT 47.50ms vs 79.68ms = 1.68x；batch4 稳态 35.02ms = ~2.25x。）

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
│   ├── flash-attn.patch                   # 源码 patch (git apply)
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
git apply /path/to/patch/flash-attn.patch

# 2. 编译
export FLASH_ATTN_OPT=1
export PATH=/opt/dtk/bin:$PATH
python3 setup.py sdist bdist_wheel

# 3. 安装
pip3 install --force-reinstall --no-deps dist/flash_attn-*.whl
```

## 改动详情

统一 patch 文件 `patch/flash-attn.patch`（74K, 1243 行），在干净 HEAD（commit 6519c7f，
分支 path2-e4m3-qfp16）上 `git apply --check` 通过，无冲突。共 19 个文件（17 改 + 2 新建），分两层：

1. **Layer A（fp8 e4m3 mixed kernel 支持）**：让 gfx936 的 fp8 mixed kernel 同时支持 e5m2 和 e4m3
   KV 存储（e4m3 走软件 `__e4m32float` dequant，复用 e5m2 的 compact-LDS pipeline）
2. **Layer B（head_dim=512 支持）**：解除 flash 入口的 TORCH_CHECK 限制，让 fp8 + 512 在
   decode/prefill 双路径走 flash mixed kernel

### Layer A：fp8 e4m3 mixed kernel 支持（10 个文件）

**A1. `csrc/flash_attn_hipc/include/intrinsic.h`** (+14/-14)
- `__e4m32float()`：去掉多余的 `uint8_t __src` 中转，用 `__builtin_memcpy` 替代 `*(float*)&`（修复严格别名 UB）

**A2. `csrc/flash_attn_hipc/include/kernel_traits.h`** (+10/-7)
- `Flash_fp8_bf16_fwd_kernel_traits`：加 `typename elem_type_k = fp8_e5m2` 模板参数，让 e5m2/e4m3 共用此 traits

**A3. `csrc/flash_attn_hipc/include/fwd/pv_gemm_utils.h`** (+37/-5)
- 新增 `FwdInputPipeline<fp8_e4m3, BFloat16>` 特化（复用 e5m2 的 compact-LDS）
- 新增 `e4m3x2_to_bf16x2()`：软件 dequant 两个 e4m3 字节为 bf16
- `mixed_prefill_raw_lds_v_to_bf16_regs<>` 加 `InputElement` 参数，if-else 选 e5m2/e4m3 cast

**A4. `csrc/flash_attn_hipc/include/fwd/qk_gemm_utils.h`** (+14/-4)
- `mixed_prefill_raw_lds_qk_to_bf16_regs<>` 和 `mixed_prefill_raw_qk_cast_half<>` 加 `InputElement`，if-else 选 cast

**A5. `csrc/flash_attn_hipc/include/fwd/qk_gemm_prefetch_v.h`** (+17/-17)
- 16 处调用 `mixed_prefill_raw_qk_cast_half` 透传 `InputElement`

**A6. `csrc/flash_attn_hipc/include/fwd/qk_gemm_prefetch_v_headdim128.h`** (+9/-9)
- 8 处调用 `mixed_prefill_raw_lds_qk_to_bf16_regs` 透传 `InputElement`

**A7. `csrc/flash_attn_hipc/include/fwd/pv_gemm_prefetch_k.h`** (+17/-17)
- 16 处调用 `mixed_prefill_raw_v_cast_half` 透传 `InputElement`

**A8. `csrc/flash_attn_hipc/include/fwd/pv_gemm_prefetch_k_headdim128.h`** (+9/-9)
- 8 处调用 `mixed_prefill_raw_lds_v_to_bf16_regs` 透传 `InputElement`

**A9. `csrc/flash_attn_hipc/include/kvcache/kvcache_qk_gemm_utils_tile16x32.h`** (+28/-16)
- decode tile16x32 路径的 q/k cast 函数加 `InputElement`，if-else 选 e5m2/e4m3

**A10. `csrc/flash_attn_hipc/include/kvcache/kvcache_pv_gemm_utils_tile16x32.h`** (+17/-8)
- decode tile16x32 路径的 v cast 函数加 `InputElement`，if-else 选 e5m2/e4m3

### Layer A+B 共享：dispatch 层（4 个文件）

**D1. `csrc/flash_attn_hipc/src/flash_fwd_launch_template.h`** (+42/-16)
- `run_fp8_flash_fwd_prefix_prefill`（prefill dispatch）：gate 从 `params.is_e5m2` 改为 `(params.is_e5m2 || params.is_e4m3)`，内层拆 if(is_e5m2)/else(is_e4m3) 各自实例化 traits；`Headdim >= 256` 路径走 64x64 tile，512 也走这里（Layer B 间接生效）

**D2. `csrc/flash_attn_hipc/src/flash_fwd_launch_template_pa.h`** (+31/-10)
- `run_mha_fwd_splitkv_dispatch`（decode dispatch）：`use_gfx936_fp8_bf16` 加 `(Headdim == 512 and HeaddimV == 512)`（**Layer B 核心**），gate 改 `(is_e5m2 || is_e4m3)`，内层拆 e5m2/e4m3 实例化；`run_flash_mixed_splitkv_fwd_tile16x32` static_assert 放宽允许 e4m3

**D3. `csrc/flash_attn_hipc/src/flash_fwd_b16_fa.h`** (+3/-2)
- `compute_mixed_attn_mha_prefix_prefill_1rowblock`：static_assert 放宽 `InputElement ∈ {e5m2, e4m3}`

**D4. `csrc/flash_attn_hipc/src/flash_fwd_b16_pa.h`** (+8/-4)
- `compute_mixed_attn_1rowblock_splitkv_tile16x32`（decode mixed kernel）：static_assert 放宽，3 处调用透传 `InputElement`

### Layer B：head_dim=512 入口 + 实例化（3 改 + 2 新建）

**B1. `csrc/flash_attn_hipc/flash_api.cpp`** (+18/-8)
- `hg_prefix_prefill_varlen_fwd`（prefill 入口）：`use_fp8_bf16_mixed` 改为 `(fp8_e5m2_used || fp8_e4m3_used)`；prefill TORCH_CHECK 的 512 条件去掉 `!fp8_used` 改 `!int8_used`，fp8 内层 check 加 512
- `hg_prefix_decode_varlen_fwd`（decode 入口）：`use_fp8_bf16_mixed` 同上；decode TORCH_CHECK 的 512 条件去掉 `!fp8_used`，fp8 内层 check 加 512；`enable_split`（约4173行）gfx936 分支加 `|| fp8_e4m3_used`

**B2. `csrc/flash_attn_hipc/flash_c_api.h`** (+4/-1)
- `run_mha_fwd` → FP8 prefix prefill dispatch：加 `params.d == 512` 分支调用 `run_fp8_mha_fwd_prefix_prefill_<elem_type, 512, 512>`

**B3. `setup.py`** (+8/-4)
- mode "1" 列表新增 2 个 target 文件；3 处架构改为只 `gfx936`（main_ext_archs / offload-arch / GFX_VERSION），绕过 gfx938 fp8 GEMM utils 的 static_assert

**B4. 新建 `csrc/flash_attn_hipc/src/target/flash_fp8_fwd_hdim512_prefix_prefill_fp16.cpp`**
- 实例化 `run_fp8_mha_fwd_prefix_prefill_<Float16, 512, 512>`

**B5. 新建 `csrc/flash_attn_hipc/src/target/flash_fp8_fwd_hdim512_prefix_prefill_bf16.cpp`**
- 实例化 `run_fp8_mha_fwd_prefix_prefill_<BFloat16, 512, 512>`
- 注：`FP16_SWITCH` 运行时分发需要 fp16+bf16 两个符号都在，漏 bf16 会链接报 undefined symbol

### 不包含在 patch 里的文件

- `csrc/flash_attn_cutlass/src/*.hip` / `*_hip.h`（200+）：hipify 自动生成产物
- `csrc/cutlass`（submodule typechange）：非源码改动
- `flash_attn/version.py`：空改动
- `test/`（调试脚本）：可选，非功能必需

### 路由链路（draft head_dim=512 验证）

- aiter `unified_attention()` + `ATTN_FLASH_HEAD512=1` → `varlen_fwd_unified`
- **decode**：num_kv_heads=1 不满足 paged fast path → `hg_prefix_decode_varlen_fwd` → `run_mha_fwd_kvcache` → `PA_HEADDIM_SWITCH(512)` → `run_mha_fwd_splitkv_dispatch<512,512>` → `use_gfx936_fp8_bf16(512)=true` → `run_flash_mixed_splitkv_fwd_tile16x32`
- **prefill**：fp8+512 不满足 q.shape[-1]==256 fast path → `hg_prefix_prefill_varlen_fwd` → `run_mha_fwd` → flash_c_api.h 512 分支 → `run_fp8_mha_fwd_prefix_prefill_<512,512>` → `run_fp8_flash_fwd_prefix_prefill` → gcn_arch==936 → `run_flash_mixed_fwd_prefix_prefill_launcher`

## 环境要求

- Hygon DCU BW10 (gfx936)，DTK 驱动
- Python 3.10，vllm 0.23.0，torch 2.10.0
- 模型：gemma-4-31B-it-AWQ-4bit + gemma-4-31B-it-assistant (MTP draft)
