# 【算子优化单】FP8 W8A8 GEMM 在 M=17~64 区间带宽利用率偏低（Qwen3-32B decode）

> 目标算子：`lightop.hipblaslt_w8a8_channelwise_gemm` → `rocblas_gemm_ex`（OCP e4m3，NT，per-channel 权重 scale + per-token 动态激活 scale，bf16 输出）
> 现象：decode batch=32/64 落点（M=17~64）gate+up GEMM 带宽利用率卡在 68%，明显低于 M=16 已达到的 83%，且该区间算术强度远低于算力脊点，**不是算力受限，是 kernel tile 配置问题**。

---

## 【环境信息】

| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| 内核 | 4.19.90-89.11.v2401.ky10.x86_64（Hygon 定制内核） |
| CPU | Hygon C86-4G (OPN:7490)，128 核 |
| 内存 | 1506 GB |
| 部署形态 | 单机 8 卡；本问题为**单卡 DP1** 复现，不涉及多机 |

---

## 【DCU 信息】

| 项 | 值 |
|---|---|
| 型号 | **BW1101**（C-3000 IC Design Co., Ltd.） |
| 数量 | 8 张（HCU 0~7） |
| 架构 | gfx938（CDNA 系），64 CU，wavefront 64，L2 8MB，LDS 64KB/CU |
| HBM | 154.6 GB/卡，**峰值带宽按 1.8 TB/s 计** |
| FP8 算力 | **447 TFLOPS**（本机 M=4096 GEMM 实测饱和值） |
| 功耗 | PwrCap 800W，空闲 ~157W |
| 固件版本 | 本机 `rocm-smi --showfwversion` 无输出，**待 FAE 补充** |
| 总线 | PCIe（`lspci` 未列出 HCU 设备条目，**待 FAE 补充**） |

---

## 【驱动与 DTK 版本】

| 项 | 值 |
|---|---|
| DTK | **26.04**（`/opt/dtk`，clang 17.0.0，dcc 25.10.0-0） |
| 内核模块 | `hycu`（+ `hycu_sched` / `hyttm` / `hydrm_ttm_helper` / `hyfs`）已加载 |
| 驱动版本 | `modinfo hycu` 本机无输出，**待 FAE 补充** |

---

## 【软件栈版本】

| 包 | 版本 |
|---|---|
| Python | 3.10.12 |
| torch | 2.10.0+das.opt1.dtk2604.2608181852.gddc08a |
| vllm | 0.21.0+das.dtk2604.torch2100.2606111143.g8c979d |
| lightop | 0.6.0+das.dtk2604.torch2100.2608171808.g43c17a |
| aiter | 0.1.5+dtk2604.torch2100.2608221558.g092be0 |
| triton | 3.6.0+dtk2604.torch2100.2608222214.g5ca092 |
| transformers | 5.5.0 |
| GEMM 生产路径 | `lightop.hipblaslt_w8a8_channelwise_gemm` → **`rocblas_gemm_ex`**（OCP e4m3，NT，per-channel 权重 scale + per-token 动态激活 scale，bf16 输出） |

---

## 【复现状态】

**可完整复现**，脚本在 `/workspace/_src/hipblaslt_splitk_probe/`：

```bash
# 1. 带宽利用率 vs M（真实权重，主证据）
HIP_VISIBLE_DEVICES=4 python3 bw_real.py
# 2. 逐点扫描 + 算力峰值/脊点（tile 台阶证据）
HIP_VISIBLE_DEVICES=4 python3 ridge_probe.py
# 3. 库内 algo 枚举 A/B（排除"换 algo 就行"）
HIP_VISIBLE_DEVICES=4 python3 ab_m64.py
```

依赖：`lightop`（生产 GEMM 路径）、`safetensors`、模型 `/models/Qwen3-32B-FP8-ROWWISE/`（layer-0 权重）。
`ab_m64.py` 另需编译 `probe.cpp`：

```bash
g++ -O2 -std=c++17 -shared -fPIC -D__HIP_PLATFORM_AMD__ \
    -I/opt/dtk/include probe.cpp \
    -L/opt/dtk/lib -lhipblaslt -lamdhip64 -o libprobe.so
```

**注意**：card 0 有 run-to-run 波动（疑似干扰），**请用空闲卡（如 card 4）复现**；3 次重复取一致值。

---

## 【实际结果】

实测（真实 Qwen3-32B 权重，card 4，3 次重复误差 <1%；M=128 layer × 64 层 = 39.1ms，与 profiler trace 40.64ms/step 吻合）：

| M | gate+up 耗时 | 带宽利用率 | 算术强度 AI | 相对脊点(248) |
|---|---|---|---|---|
| 16 | 175.7us | **83%** | 31.8 | 远低于 |
| 17 | 214.3us | 68% | 33.8 | 远低于 |
| 32 | 211.6us | 69% | 63.2 | 1/4 |
| 64 | 218.5us | 68% | 124.7 | 1/2 |
| 80 | 302.8us | 50% | 154.9 | 2/3 |
| 128 | 309.4us | 50% | 243.2 | **≈脊点** |

**核心异常**：M=16 与 M=64 读**同一份 262MB 权重**（激活/输出可忽略），M=64 理论上应与 M=16 同为纯权重读取瓶颈（~83%），实际只有 68%，**慢 22%**；而 M=64 的 AI=125 远低于脊点 248，**不是算力受限**。

**tile 切换台阶**（逐点扫描）：跳变点在 **M=17**（175→214us）和 **M=80**（214→303us），M=17~64 平台卡在 68%。

**瓶颈定位：kernel（tile 配置），非 host / 通信 / 调度。**

1. **排除 host**：纯 device kernel，CUDA event 计时不含 host；launch 开销相对 200us kernel 可忽略。
2. **排除通信**：单卡 DP1，无 all-reduce/all-gather。
3. **排除调度**：cudagraph 已启用，decode step 内 GEMM 连续执行，无调度气泡。
4. **锁定 kernel**：
   - M=16 能到 83% BW → 硬件与权重读取路径本身没问题，kernel 有能力跑满；
   - M=17~64 掉到 68% → 同样权重量，纯 kernel 效率损失；
   - **已尝试库内 algo 枚举**（hipBLASLt 标准 API，FNUZ W8A8，64 个 algo 全带 split-K 能力）：M=32/64 最佳 algo 与生产路径持平（1.00x），M=16 反而更差（0.82x），M=128 仅 1.08x → **现成库内无更优 tile，需针对 M=17~64 重调 tile/CTA 配置或写专用 kernel**。

**对业务影响**：decode batch=64 时 GEMM 占 step ~40%，该区间 1.2x 空间对应 **TPOT 约 -5%**。

---

## 【期望结果】

- M=17~64 区间 gate+up GEMM 带宽利用率 **≥80%**（对齐 M=16 的 83%），M=64 单算子 ≤175us；
- 或给出明确结论：若 68% 是该硬件/该 shape 的 kernel 硬上限，请提供 profiling 证据（如 CU 占用、L2 命中率、HBM 读写比）说明瓶颈所在；
- 优化需保持 fp8 无损（数值等价，仅允许累加顺序变化），可接受 tile/CTA/split-K 重调或专用 kernel，**不接受 int8/int4 降级**。

---

## 附：性能指标要求 / SLA 描述

- **主目标**：M=17~64 区间（decode batch=32/64 落点）gate+up GEMM 带宽利用率 **68% → ≥80%**（对齐 M=16 已达到的 83%），单算子 ~214us → ~175us（**~1.2x**）。
- **验收口径**：真实 Qwen3-32B 权重（gate+up N=51200, K=5120），card 4，峰值 1.8 TB/s；M=64 时 layer 4-GEMM 合计 0.458ms → ≤0.38ms。
- **无损约束**：保持 fp8 输入、数值等价（仅允许累加顺序变化），不接受 int8/int4 降级。

## 附：通信环境

不适用。单卡 DP1 任务，无多机多卡通信。
