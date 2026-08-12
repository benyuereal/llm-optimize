# flash-attention-cutlass 完整改动清单 (fp8 e5m2+e4m3 KV + head_dim=512, gfx936)

## 生成物
- **统一 patch 文件**: `flash-attn.patch` (74K, 1243行)
- **验证**: 在干净 HEAD (commit 6519c7f) 上 `git apply --check` 通过, 无冲突
- **分支**: path2-e4m3-qfp16
- **应用**: `cd flash-attention-cutlass && git apply flash-attn.patch`

## 改动分两层
1. **Layer A (之前session)**: 让 gfx936 fp8 mixed kernel 同时支持 e5m2 和 e4m3 KV 存储
   - 原: 只支持 e5m2 (bit-trick dequant)
   - 改: 加 `InputElement`/`elem_type_k` 模板参数, e4m3 走软件 `__e4m32float` dequant, 复用同一 mixed pipeline
2. **Layer B (本次session)**: 让 fp8 + head_dim=512 通过 (decode + prefill), 走 flash mixed kernel 而非慢的 aiter 2D

## 效果 (本次 512 支持)
- TPOT: 79.68ms (triton金标准) → 47.50ms (flash) = **1.68x 加速**
- draft MTP acceptance: 96.42% → 96.76% (稳定)
- 精度: HumanEval 验证中; prompt对比 5/8 完全一致, 其余语义正确

## 文件清单 (19个: 17改 + 2新建)

### === Layer A: fp8 e4m3 mixed kernel 支持 (10个文件) ===

**A1. csrc/flash_attn_hipc/include/intrinsic.h** (+14/-14行)
- `__e4m32float()`: 去掉多余的 `uint8_t __src` 中转, 用 `__builtin_memcpy` 替代 `*(float*)&` (修复严格别名 UB)

**A2. csrc/flash_attn_hipc/include/kernel_traits.h** (+10/-7行)
- `Flash_fp8_bf16_fwd_kernel_traits`: 加 `typename elem_type_k = fp8_e5m2` 模板参数, 让 e5m2/e4m3 共用此 traits

**A3. csrc/flash_attn_hipc/include/fwd/pv_gemm_utils.h** (+37/-5行)
- 新增 `FwdInputPipeline<fp8_e4m3, BFloat16>` 特化 (复用 e5m2 的 compact-LDS)
- 新增 `e4m3x2_to_bf16x2()`: 软件 dequant 两个 e4m3 字节为 bf16
- `mixed_prefill_raw_lds_v_to_bf16_regs<>` 加 `InputElement` 参数, if-else 选 e5m2/e4m3 cast

**A4. csrc/flash_attn_hipc/include/fwd/qk_gemm_utils.h** (+14/-4行)
- `mixed_prefill_raw_lds_qk_to_bf16_regs<>` 和 `mixed_prefill_raw_qk_cast_half<>` 加 `InputElement`, if-else 选 cast

**A5. csrc/flash_attn_hipc/include/fwd/qk_gemm_prefetch_v.h** (+17/-17行)
- 16 处调用 `mixed_prefill_raw_qk_cast_half` 透传 `InputElement`

**A6. csrc/flash_attn_hipc/include/fwd/qk_gemm_prefetch_v_headdim128.h** (+9/-9行)
- 8 处调用 `mixed_prefill_raw_lds_qk_to_bf16_regs` 透传 `InputElement`

**A7. csrc/flash_attn_hipc/include/fwd/pv_gemm_prefetch_k.h** (+17/-17行)
- 16 处调用 `mixed_prefill_raw_v_cast_half` 透传 `InputElement`

**A8. csrc/flash_attn_hipc/include/fwd/pv_gemm_prefetch_k_headdim128.h** (+9/-9行)
- 8 处调用 `mixed_prefill_raw_lds_v_to_bf16_regs` 透传 `InputElement`

**A9. csrc/flash_attn_hipc/include/kvcache/kvcache_qk_gemm_utils_tile16x32.h** (+28/-16行)
- decode tile16x32 路径的 q/k cast 函数加 `InputElement`, if-else 选 e5m2/e4m3

**A10. csrc/flash_attn_hipc/include/kvcache/kvcache_pv_gemm_utils_tile16x32.h** (+17/-8行)
- decode tile16x32 路径的 v cast 函数加 `InputElement`, if-else 选 e5m2/e4m3

### === Layer A+B 共享: dispatch 层 (4个文件) ===

**D1. csrc/flash_attn_hipc/src/flash_fwd_launch_template.h** (+42/-16行)
- `run_fp8_flash_fwd_prefix_prefill` (prefill dispatch):
  - gate 从 `params.is_e5m2` 改为 `(params.is_e5m2 || params.is_e4m3)`
  - 内层拆 if(is_e5m2)/else(is_e4m3), 各自实例化 `Flash_fp8_bf16_fwd_kernel_traits<...,fp8_e5m2>` 和 `<...,fp8_e4m3>`
  - 注: `Headdim >= 256` 路径走 64x64 tile, 512 也走这里 (Layer B 间接生效)

**D2. csrc/flash_attn_hipc/src/flash_fwd_launch_template_pa.h** (+31/-10行)
- `run_mha_fwd_splitkv_dispatch` (decode dispatch):
  - `use_gfx936_fp8_bf16` 加 `(Headdim == 512 and HeaddimV == 512)` ← **Layer B 核心**
  - gate 从 `params.is_e5m2` 改为 `(params.is_e5m2 || params.is_e4m3)`
  - 内层拆 if(is_e5m2)/else(is_e4m3), 各自实例化 traits
- `run_flash_mixed_splitkv_fwd_tile16x32`: static_assert 放宽允许 e4m3

**D3. csrc/flash_attn_hipc/src/flash_fwd_b16_fa.h** (+3/-2行)
- `compute_mixed_attn_mha_prefix_prefill_1rowblock`: static_assert 放宽 `InputElement ∈ {e5m2, e4m3}`

**D4. csrc/flash_attn_hipc/src/flash_fwd_b16_pa.h** (+8/-4行)
- `compute_mixed_attn_1rowblock_splitkv_tile16x32` (decode mixed kernel):
  - static_assert 放宽
  - 3 处调用透传 `InputElement` (prefetch_q, qk_gemm_prefetch_v, pv_gemm_32x64)

### === Layer B: head_dim=512 入口 + 实例化 (3个文件改 + 2新建) ===

**B1. csrc/flash_attn_hipc/flash_api.cpp** (+18/-8行)
- `hg_prefix_prefill_varlen_fwd` (prefill 入口):
  - `use_fp8_bf16_mixed` 从 `fp8_e5m2_used` 改为 `(fp8_e5m2_used || fp8_e4m3_used)`
  - prefill TORCH_CHECK: 512 条件去掉 `!fp8_used`, 改 `!int8_used`; fp8 内层 check 加 512
- `hg_prefix_decode_varlen_fwd` (decode 入口):
  - `use_fp8_bf16_mixed` 同上
  - decode TORCH_CHECK: 512 条件去掉 `!fp8_used`; fp8 内层 check 加 512
  - `enable_split` (约4173行): gfx936 分支加 `|| fp8_e4m3_used` (已支持 512 splitkv)

**B2. csrc/flash_attn_hipc/flash_c_api.h** (+4/-1行)
- `run_mha_fwd` → FP8 prefix prefill dispatch: 加 `params.d == 512` 分支调用 `run_fp8_mha_fwd_prefix_prefill_<elem_type, 512, 512>`

**B3. setup.py** (+8/-4行)
- mode "1" 列表新增 2 个 target 文件
- 3 处架构改为只 `gfx936` (main_ext_archs / offload-arch / GFX_VERSION), 绕过 gfx938 fp8 GEMM utils 的 static_assert

**B4. 新建 csrc/flash_attn_hipc/src/target/flash_fp8_fwd_hdim512_prefix_prefill_fp16.cpp**
- 实例化 `run_fp8_mha_fwd_prefix_prefill_<Float16, 512, 512>`

**B5. 新建 csrc/flash_attn_hipc/src/target/flash_fp8_fwd_hdim512_prefix_prefill_bf16.cpp**
- 实例化 `run_fp8_mha_fwd_prefix_prefill_<BFloat16, 512, 512>`
- 注: FP16_SWITCH 运行时分发需要 fp16+bf16 两个符号都在, 漏 bf16 会链接报 undefined symbol

## 不包含在 patch 里的文件
- `csrc/flash_attn_cutlass/src/*.hip` / `*_hip.h` (200+): hipify 自动生成产物
- `csrc/cutlass` (submodule typechange): 非源码改动
- `flash_attn/version.py`: 空改动
- `test/` (调试脚本): 可选, 非功能必需

## 路由链路 (draft head_dim=512 验证)
- aiter `unified_attention()` + `ATTN_FLASH_HEAD512=1` → `varlen_fwd_unified`
- **decode**: num_kv_heads=1 不满足 paged fast path → `hg_prefix_decode_varlen_fwd`
  → `run_mha_fwd_kvcache` → `PA_HEADDIM_SWITCH(512)` → `run_mha_fwd_splitkv_dispatch<512,512>`
  → `use_gfx936_fp8_bf16(512)=true` → `run_flash_mixed_splitkv_fwd_tile16x32`
- **prefill**: fp8+512 不满足 q.shape[-1]==256 fast path → `hg_prefix_prefill_varlen_fwd`
  → `run_mha_fwd` → flash_c_api.h 512 分支 → `run_fp8_mha_fwd_prefix_prefill_<512,512>`
  → `run_fp8_flash_fwd_prefix_prefill` → gcn_arch==936 → `run_flash_mixed_fwd_prefix_prefill_launcher`

## 编译安装
```bash
cd /public/home/weishb/flash-attention-cutlass
export FLASH_ATTN_OPT=1
export PATH=/opt/dtk/bin:$PATH
python3 setup.py sdist bdist_wheel
pip3 install --force-reinstall --no-deps dist/flash_attn-*.whl
```

## 启动配置 (start_flash.sh 关键项)
```bash
export VLLM_AITER_W4A16_PATCH=1
export ATTN_FLASH_PREFILL=1
export ATTN_FLASH_HEAD512=1
vllm serve ... \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --speculative-config '{"method":"mtp","model":".../gemma-4-31B-it-assistant","num_speculative_tokens":3}'
```
