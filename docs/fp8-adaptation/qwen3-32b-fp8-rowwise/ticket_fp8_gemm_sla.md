# 【算子优化单】FP8 W8A8 GEMM 在 M=17~64 区间带宽利用率偏低

## 【性能指标要求 / SLA 描述】

**目标算子**：FP8 W8A8 GEMM（per-channel 权重 scale + per-token 动态激活 scale，NT 布局，bf16 输出），即 Qwen3-32B decode 每层的 4 个 GEMM（QKV / O / gate+up / down）。

**当前生产路径**：`rocblas_gemm_ex`（OCP e4m3），经 `lightop.hipblaslt_w8a8_channelwise_gemm` 调用。

**优化目标**：

- **主目标**：M=17~64 区间（decode batch=32/64 落点）的 gate+up GEMM 带宽利用率从 **68% 提升到 ≥80%**（对齐 M=16 已达到的 83%）。对应 gate+up 单算子从 ~214us 降到 ~175us（**~1.2x**）。
- **验收口径**：真实 Qwen3-32B 权重（gate+up N=51200, K=5120），card 4，峰值按 1.8 TB/s。M=64 时 layer 4-GEMM 合计从 0.458ms 降到 **≤0.38ms**。
- **无损约束**：必须保持 fp8 输入、数值等价（仅允许累加顺序变化），**不接受 int8/int4 降级**。

---

## 【当前性能或基线】

实测（真实权重，card 4，3 次重复误差 <1%；M=128 layer × 64 层 = 39.1ms，与 profiler trace 40.64ms/step 吻合）：

| M | gate+up 耗时 | 带宽利用率 | 算术强度 AI | 相对脊点(248) |
|---|---|---|---|---|
| 16 | 175.7us | **83%** | 31.8 | 远低于 |
| 17 | 214.3us | 68% | 33.8 | 远低于 |
| 32 | 211.6us | 69% | 63.2 | 1/4 |
| 64 | 218.5us | 68% | 124.7 | 1/2 |
| 80 | 302.8us | 50% | 154.9 | 2/3 |
| 128 | 309.4us | 50% | 243.2 | **≈脊点** |

**核心异常**：M=16 和 M=64 读的是**同一份 262MB 权重**（激活/输出可忽略），理论上 M=64 应和 M=16 一样是纯权重读取瓶颈、能到 ~83%。实际 M=64 只有 68%，**慢了 22%**。而 M=64 的 AI=125 远低于脊点 248，**不是算力受限——是 kernel 在该区间的 tile 配置不优**。

**tile 切换台阶**（逐点扫描）：跳变点在 **M=17**（175→214us）和 **M=80**（214→303us），M=17~64 整个平台卡在 68%。这是 auto-tuner 的 tile 选择问题，不是物理 regime 变化。

**硬件参数**（BW1101 / gfx938）：HBM 峰值 1.8 TB/s，FP8 算力峰值 447 TFLOPS（M=4096 实测饱和），脊点 AI = 447/1.8 = **248 FLOP/byte**。

---

## 【Profiling 证据】

**瓶颈定位：kernel（tile 配置），非 host / 通信 / 调度。**

1. **排除 host**：GEMM 是纯 device kernel，launch 开销相对 200us 的 kernel 时间可忽略；计时用 CUDA event 包 kernel，不含 host。
2. **排除通信**：单卡 DP1，无 all-reduce/all-gather，不涉及多机多卡。
3. **排除调度**：cudagraph 已启用，decode step 内 GEMM 是连续 kernel，无调度气泡。
4. **锁定 kernel**：
   - M=16 能到 83% BW（证明硬件 + 权重读取路径本身没问题，kernel 有能力跑满）；
   - M=17~64 掉到 68%（同样权重量，纯 kernel 效率损失）；
   - 已尝试库内 algo 枚举（hipBLASLt 标准 API，FNUZ W8A8，64 个 algo 全带 split-K 能力）：M=32/64 最佳 algo 与生产路径持平（1.00x），M=16 反而更差（0.82x），M=128 仅 1.08x。说明**现成库内没有更优 tile，需要算子团队针对 M=17~64 重新调 tile/CTA 配置或写专用 kernel**。
