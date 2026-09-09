# custom_ops 根因分析: decode 激活量化占 36.7% GPU 时间

> 平台: Hygon DCU BW1100 (HCU, gfx938) ROCm, vllm 0.21.0
> 模型: Qwen3-32B-FP8-ROWWISE (W8A8)
> 日期: 2026-09-03

## 1. 现象

用户困惑: 为什么"激活量化"能花掉 decode 总耗时的 34%+?

用 vllm torch profiler (`/start_profile` / `/stop_profile`) 抓 chrome trace,
`/workspace/analyze_quant_trace.py` 解析:

- 量化内核 `triton_red_fused__to_copy_abs_clamp_div_hipblaslt_gemm_channelwise_max`:
  **171.9 us × 32832 次 = 26.7% 真实 GPU 时间** (加上相关量化开销, 量化 + norm
  合计约 **36.7%**)。
- 这不是 launch overhead, 而是**真实执行的慢内核** —— 由 inductor 自动生成的
  fused 内核 (把 abs/clamp/div 等量化数学融合进去)。

## 2. 根因

baseline 启动命令里 **`custom_ops: ['none']`** 禁用了
`quant_fp8` / `rms_norm` / `silu_and_mul` 三个 custom op, 导致:

1. vllm 的 `enable_norm_fusion` / `enable_act_fusion` 要求
   `is_custom_op_enabled(...)` 为真才生效 → **fusion pass 关闭**;
2. inductor 于是把量化数学 (abs/clamp/div 等) 自动融合成那个慢的
   `triton_red_fused_...` 内核;
3. 补充: `rocm_aiter_ops.is_enabled(): False`, 所以 ROCm aiter 的 fusion pass
   不会跑, 只有通用的 `RMSNormQuantFusionPass` / `ActivationQuantFusionPass`
   可用 —— 而它们同样依赖 custom op 开启。

即: `custom_ops: ['none']` → fusion pass 全关 → inductor 兜底生成慢量化内核。

## 3. 修复

启动命令加一行 compilation config, 让 `fuse_norm_quant` / `fuse_act_quant`
生效, 量化改走 vllm 自定义 C++ 内核:

```
--compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}'
```

**纯启动参数改动: 不需重新量化、不需改任何代码。**

## 4. trace 证据 (修复前后对比)

| 项 | 修复前 (custom_ops OFF) | 修复后 (custom_ops ON) |
|---|---|---|
| 量化内核 | `triton_red_fused__to_copy_abs_clamp_div_hipblaslt_gemm_channelwise_max` | `vllm::dynamic_per_token_scaled_fp8_quant_kernel` |
| 单次耗时 | **171.9 us** | **13.4 us** (快 ~13 倍) |
| 量化 + norm 占 GPU 时间 | **36.7%** | **16.3%** |
| GEMM 占 GPU 时间 | 53% | **69.1%** (瓶颈转移到 GEMM) |

## 5. A/B 验证 (端到端)

- A/B 服务器: 8005 (卡 3, custom_ops ON, `/workspace/start_fp8_ab.sh`)
- baseline: 8001 (custom_ops OFF)
- 压测: `/workspace/ab_verify.sh` (16 并发 × 512 输出 token, 16 prompts,
  random 数据集, input 1024)

| 指标 | Baseline (OFF) | A/B (ON) | 提升 |
|---|---|---|---|
| Output tok/s | 371.97 | 488.00 | **+31%** |
| Mean TPOT | 42.43 ms | 32.17 ms | **−24%** |
| Total tok/s | 1115.91 | 1464.00 | **+31%** |
| Mean TTFT | 338.08 ms | 345.53 ms | ~持平 |

**解读**:

- decode 吞吐 +31%、TPOT −24%, 与"消除 26.7% 慢量化内核"的预期一致;
- **TTFT 基本不变**, 符合预期: 优化的是 decode 逐 token 的激活量化,
  prefill 阶段不受影响。

52 并发实测: 优化后 TPOT ~51 ms (baseline 预估 ~67 ms)。

## 6. 后续优化空间

量化已榨干 (36.7% → 16.3%), 瓶颈转移到 GEMM (69.1%)。W4 量化被排除 (精度),
剩余方向是 **hipBLASLt tile 调优** (decode M=1-64 区间)。
