# W8A16 探索: 为什么放弃 (死路)

> 平台: Hygon DCU BW1100 (HCU, gfx938) ROCm, vllm 0.21.0
> 模型: Qwen3-32B-FP8-ROWWISE
> 日期: 2026-09-03

## 1. 动机

用户想试 **W8A16** (权重 FP8 + 激活不量化、保持 bf16), 看是否比 W8A8 快 ——
直觉上省掉了激活量化这一步。

## 2. 关键发现: W8A16 不需要重新量化

W8A16 和 W8A8 用的是**完全相同的 FP8 rowwise 权重**, 唯一区别是 config 里的
`input_activations` 字段:

| | `input_activations` |
|---|---|
| W8A8 (当前产物) | `strategy=token, dynamic=true` |
| W8A16 | `null` (激活不量化) |

所以 W8A16 理论上只需改 config, 无需重新量化。

## 3. 但 vllm 在 ROCm 上没有 W8A16 kernel

- vllm 的 kernel 选择表: `_POSSIBLE_WFP8A16_KERNELS[ROCM] = []` (空列表,
  注释 "To be added") → `choose_scaled_mm_linear_kernel` 直接抛 `ValueError`。
- 底层 lightop 的 `gemm_w8a16_fp8(a, b, scale, ...)` 限制极多:
  - 要求 `b.is_contiguous() == False` (即非连续 `[n,k]` 视图 = `w_fp8.t()`);
  - 硬限制 **`1 <= m <= 16`** (`w8a16.hip:337`), M>16 直接崩;
  - 用的是手写慢内核 `gemm_w8_a16_kernel_colmajor`。

## 4. 微基准数据 (`/workspace/bench_w8a16_vs_w8a8.py`)

在 Qwen3-32B 的 5 个 linear 层形状 (qkv/o/gate/up/down proj) 上,
M ∈ {1,4,8,16} 对比:

- **W8A16 比 W8A8 慢 3–9 倍** (speedup 0.11x–0.36x, 即 W8A8_total / W8A16 < 1);
- M > 16 直接崩溃 (硬限制)。

## 5. profiler 确认 (`/workspace/profile_w8a16.py`)

| 路径 | 实际内核 | 耗时 |
|---|---|---|
| W8A16 | `gemm_w8_a16_kernel_colmajor` (手写慢内核) | **563.6 us** |
| W8A8 | hipBLASLt 调优内核 `Cijk_Alik_Bljk_F8BS_MT64x16x256` | **100.3 us** |

## 6. 结论

**W8A16 在此 HCU build 上是死路, 放弃。** 原因:

1. ROCm 侧 vllm 没有 W8A16 kernel 注册 (`_POSSIBLE_WFP8A16_KERNELS[ROCM] = []`);
2. 底层 lightop 内核是未调优的手写实现, 比 hipBLASLt 调优的 W8A8 内核慢 3–9 倍;
3. M>16 不可用 (硬限制, 直接崩)。

省掉激活量化省下的时间远不够弥补 GEMM 内核本身的差距。**W8A8 是本平台正确选择。**
