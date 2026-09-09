# Qwen3-32B FP8 量化与性能优化报告

> 平台：Hygon DCU (BW1100) ROCm（HCU）
> 模型：Qwen3-32B
> 量化产物：`/models/Qwen3-32B-FP8-ROWWISE/`
> 日期：2026-09-03

---

## 1. 背景与平台

- **硬件**：Hygon DCU BW1100（HCU），ROCm 平台，`HCUPlatform._enum = PlatformEnum.ROCM`。机器共 8 张 HCU，torch 2.10.0 + hip 6.3.26113，vllm 0.21.0。
- **重要约束**：DeepGEMM 是 CUDA-only（依赖 sm_90a 的 WGMMA/TMA），**在本平台无法编译和运行**，只能作为算法与数据布局的参考。本报告的量化配方参考了 DeepGEMM 的 `sm90_fp8_gemm_1d2d`（per-block FP8 1d2d recipe），但实际 GEMM 走的是 HCU 侧的 hipBLASLt / lightop 内核。
- **目标**：把 Qwen3-32B 量化为 FP8 并在 vllm 上高效推理，定位并消除 decode 阶段的性能瓶颈。

## 2. 量化方案

使用 **llm-compressor** 的 `model_free_ptq`（data-free，纯 safetensors 处理，不加载模型）把 Qwen3-32B（BF16，源 `/models/Qwen3-32B-local`）量化为 **FP8 rowwise（per-channel 权重）**，即 **W8A8 配方**：

- **权重**：FP8 e4m3，`strategy=channel`（rowwise，1D scale `[out_features]`），observer 用 `memoryless_minmax`（直接从权重张量算 per-channel scale，data-free）。
- **激活**：per-token **动态**量化（`strategy=token, dynamic=true`），scale 在推理时计算，不存储。
- **忽略层**：`model.embed_tokens`、`lm_head`（保持 BF16）；`*norm*` 层自动忽略。
- **产物**：`/models/Qwen3-32B-FP8-ROWWISE/`（17 个 safetensors 分片，compressed-tensors 格式，scale 形状 `(out,1)` bf16）。

量化脚本：`/workspace/quantize_qwen3_32b_fp8_rowwise.py`。

产物 `config.json` 中的 `quantization_config` 关键字段（已核实）：

```json
"weights":          { "strategy": "channel", "dynamic": false, "observer": "memoryless_minmax", "num_bits": 8, "type": "float" },
"input_activations": { "strategy": "token",   "dynamic": true,  "num_bits": 8, "type": "float" }
```

> 环境备注：llm-compressor 跑在 `/workspace/venv-llmc`（`--system-site-packages`，共享系统 ROCm torch；compressed-tensors 0.18.1，llm-compressor 以 editable 安装）。直接用 `scheme="FP8_DYNAMIC"` 字符串会因 weights observer 为 None 而崩溃，需按脚本方式 `preset_name_to_scheme` + `model_copy` 指定 `memoryless_minmax`。

## 3. W8A16 探索（死路，已放弃）

### 3.1 动机

用户想试 **W8A16**（权重 FP8 + 激活不量化、保持 bf16），看是否比 W8A8 快——直觉上省掉了激活量化这一步。

### 3.2 关键发现：W8A16 不需要重新量化

W8A16 和 W8A8 用的是**完全相同的 FP8 rowwise 权重**，唯一区别是 config 里的 `input_activations` 字段：

| | `input_activations` |
|---|---|
| W8A8（当前产物） | `strategy=token, dynamic=true` |
| W8A16 | `null`（激活不量化） |

所以 W8A16 理论上只需改 config，无需重新量化。

### 3.3 但 vllm 在 ROCm 上没有 W8A16 kernel

- vllm 的 kernel 选择表：`_POSSIBLE_WFP8A16_KERNELS[ROCM] = []`（空列表，注释 "To be added"）→ `choose_scaled_mm_linear_kernel` 直接抛 `ValueError`。
- 底层 lightop 的 `gemm_w8a16_fp8(a, b, scale, ...)` 限制极多：
  - 要求 `b.is_contiguous() == False`（即非连续 `[n,k]` 视图 = `w_fp8.t()`）；
  - 硬限制 **`1 <= m <= 16`**（`w8a16.hip:337`），M>16 直接崩；
  - 用的是手写慢内核 `gemm_w8_a16_kernel_colmajor`。

### 3.4 微基准数据（`/workspace/bench_w8a16_vs_w8a8.py`）

在 Qwen3-32B 的 5 个 linear 层形状（qkv/o/gate/up/down proj）上，M ∈ {1,4,8,16} 对比：

- **W8A16 比 W8A8 慢 3–9 倍**（speedup 0.11x–0.36x，即 W8A8_total / W8A16 < 1）；
- M > 16 直接崩溃（硬限制）。

### 3.5 profiler 确认（`/workspace/profile_w8a16.py`）

| 路径 | 实际内核 | 耗时 |
|---|---|---|
| W8A16 | `gemm_w8_a16_kernel_colmajor`（手写慢内核） | **563.6 us** |
| W8A8 | hipBLASLt 调优内核 `Cijk_Alik_Bljk_F8BS_MT64x16x256` | **100.3 us** |

### 3.6 结论

**W8A16 在此 HCU build 上是死路，放弃。** 原因：ROCm 侧 vllm 没有 W8A16 kernel 注册；底层 lightop 内核是未调优的手写实现，比 hipBLASLt 调优的 W8A8 内核慢 3–9 倍，且 M>16 不可用。省掉激活量化省下的时间远不够弥补 GEMM 内核本身的差距。

## 4. 性能问题定位：decode 激活量化占 34%

### 4.1 现象

用户困惑：为什么"激活量化"能花掉 decode 总耗时的 34%？

### 4.2 抓 trace 分析

用 vllm torch profiler（`/start_profile` / `/stop_profile`）抓 chrome trace，`/workspace/analyze_quant_trace.py` 解析：

- 量化内核 `triton_red_fused__to_copy_abs_clamp_div_hipblaslt_gemm_channelwise_max`：
  **171.9 us × 32832 次 = 26.7% 真实 GPU 时间**（加上相关量化开销合计约 34%）。
- 这不是 launch overhead，而是**真实执行的慢内核**——由 inductor 自动生成的 fused 内核。

### 4.3 根因

baseline 启动命令里 **`custom_ops: ['none']`** 禁用了 `quant_fp8` / `rms_norm` / `silu_and_mul` 三个 custom op，导致：

1. vllm 的 `enable_norm_fusion` / `enable_act_fusion` 要求 `is_custom_op_enabled(...)` 为真才生效 → **fusion pass 关闭**；
2. inductor 于是把量化数学（abs/clamp/div 等）自动融合成那个慢的 `triton_red_fused_...` 内核；
3. 补充：`rocm_aiter_ops.is_enabled(): False`，所以 ROCm aiter 的 fusion pass 不会跑，只有通用的 `RMSNormQuantFusionPass` / `ActivationQuantFusionPass` 可用——而它们同样依赖 custom op 开启。

## 5. 修复方案

在启动命令加一行 compilation config，让 `fuse_norm_quant` / `fuse_act_quant` 生效，量化改走 lightop 的 HCU 内核（约 **7 us**，比 171.9 us 快 ~25 倍）：

```
--compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}'
```

**纯启动参数改动：不需重新量化、不需改任何代码。**

## 6. A/B 验证结果

- **A/B 服务器**：8005（卡 3，custom_ops ON，`/workspace/start_fp8_ab.sh`）
- **baseline**：8001（custom_ops OFF）
- **压测**：`/workspace/ab_verify.sh`（16 并发 × 512 输出 token，16 prompts，random 数据集，input 1024）

| 指标 | Baseline (OFF) | A/B (ON) | 提升 |
|---|---|---|---|
| Output tok/s | 371.97 | 488.00 | **+31%** |
| Mean TPOT | 42.43 ms | 32.17 ms | **−24%** |
| Total tok/s | 1115.91 | 1464.00 | **+31%** |
| Mean TTFT | 338.08 ms | 345.53 ms | ~持平 |

**解读**：

- decode 吞吐 +31%、TPOT −24%，与"消除 26.7% 慢量化内核"的预期一致；
- **TTFT 基本不变**，符合预期：优化的是 decode 逐 token 的激活量化，prefill 阶段不受影响。

## 6.5 并行策略：DP4 优于 TP4（2026-09-05 实测）

4 卡（4/5/6/7）下对比 **DP4**（数据并行，每卡全量模型，无 AllReduce）vs **TP4**（权重切 4 份，有 AllReduce）。压测：256 并发 × 512 输出，random，input 1024。

| 指标 | TP4 | **DP4** | 变化 |
|---|---|---|---|
| Mean TPOT | 69.96 ms | **53.54 ms** | **−24%** |
| Total tok/s | 9630 | **11363** | **+18%** |
| Output tok/s | 3210 | 3788 | +18% |
| Mean TTFT | 4614 ms | 6716 ms | +46%（DP4 prefill 略慢） |

**为什么 DP4 反而更快**（decode 是权重带宽瓶颈）：

- TP4 每卡读 1/4 权重（7.8GB），GEMM 快；但每层 2 次 AllReduce，占 GPU 时间 **33.9%**（12.37ms/step）。
- DP4 每卡读全量权重（31.2GB），GEMM 慢；但**完全没有 AllReduce**。
- 实测：DP4 省掉的 AllReduce（~12.6ms/step）**盖过了**多读权重的代价（GEMM 27ms vs 13.26ms，差 13.7ms，但其中含计算地板，非纯带宽），净收益 TPOT −24%。
- **关键**：AllReduce 是**延迟主导**（每次 95.5us，86% 是 launch+同步延迟，仅 14% 带宽），130 次/step 的延迟开销是 TP4 的主要负担。DP4 直接消除它。

**注意**：DP4 多 ApiServer 会禁用 stats logging（日志无 `Avg generation throughput` 行），看吞吐用 `curl /metrics` 或 `vllm bench` 结果。

**落地**：`start.sh` 默认 DP4（`PARALLEL_MODE=dp4`），切回 TP4 用 `PARALLEL_MODE=tp4`。

## 6.6 采样优化：fused top-k+top-p kernel（2026-09-05）

decode 采样原用 triton `_topk_topp_kernel`（多趟百分位/三分搜索），在**尖峰分布**（真实 decode logits）下退化到 **~10ms/step**（profiler 实测 median 10.1ms）。

**fused kernel**（`vllm/v1/sample/ops/fused_topk_topp.py`）：`torch.topk(POOL=1024)` 选候选（对分布不敏感）+ 单 triton kernel 做 mask 填充 + top-k 归一化 nucleus（严格复刻 triton 语义）。

| 指标 | 原 triton | **fused** |
|---|---|---|
| 采样 GPU 时间/step | ~10 ms | **0.955 ms**（4.3x） |
| 正确性 | — | 45/45 通过（vs triton + naive 参考） |

profiler 确认：trace 中 `_topk_topp_kernel` 消失，替换为 `_fused_topk_topp_mask_kernel` + `torch.topk` 的 mbtopk/radix kernel，采样合计 0.955ms/step。详见 `docs/fused_topk_topp.md`。

## 6.7 DP1 单卡 batch 对比：GEMM 亚线性 vs Attention 线性（2026-09-07）

DP1（单卡全量权重）下，对比 **batch=16**（44ms trace，`prof_traces/rank0.1788494705088657334`）与 **batch=128**（78.8ms，`prof_traces_dp1/rank0.1788589642284912465`，文件已清理，数据存于历史分析）的 decode 步耗时分布：

| 类别 | batch=16（44ms） | batch=128（78.8ms） | 增长 |
|---|---|---|---|
| GEMM (hipblaslt) | 22.78ms (52%) | 40.64ms (51.6%) | ×1.78 |
| Attention | 3.79ms (8.6%) | 20.00ms (25.4%) | **×5.3** |
| Quant/SiLU | 2.53ms | 9.44ms | ×3.7 |
| Norm | 2.52ms | 7.23ms | ×2.9 |
| KV | 0.39ms | 1.49ms | ×3.8 |
| **合计** | **43.8ms** | **78.79ms** | ×1.8 |

**核心结论：GEMM 亚线性、Attention 线性。** batch 从 16→128（×8）：

- **GEMM 只涨 ×1.78**：权重字节（31.2GB/step）与 M 无关，M 个 token 共享同一份权重，固定成本被摊薄 → 越大的 batch 越划算。
- **Attention 涨 ×5.3（接近线性）**：KV 每序列独立读、不共享，M 个序列 = M 份 KV，字节和 FLOPs 都随 M 线性增。

**batch=128 的 GEMM 细分**（按 hipblaslt 选的 tile 分组，511 decode 步）：

| tile | 对应 GEMM | 调用数 | 每次 | 小计 | 带宽利用率 |
|---|---|---|---|---|---|
| MT128x128x256 | gate+up (N=51200) | 32640 | 317us | 20.27ms | 45.2% |
| MT96x128x256 | O + down (N=5120) | 65344 | 120us | 15.34ms | 39.5% / 41.6% |
| MT256x128x128 | QKV (N=10240) | 32640 | 78.8us | 5.03ms | 36.3% |
| **合计** | 4 GEMM/层 | | | **40.64ms** | **41.9%** |

**GEMM 带宽利用率 ~42%（峰值 HBM 1.83 TB/s）**，是 M=128 FP8 channelwise kernel 的天花板：M=128 时 CTA 内算术强度 128 FLOP/byte 远低于 ridge 点 382，CTA 内部即 memory-bound，叠加 per-block scale 的 ALU 开销。对比 NVIDIA（cuBLASLt 对 M=128 自动 split-K，可达 HBM 峰值 70-85%），我们的 42% 差距主要来自**缺少 split-K**——但 `hipblaslt_w8a8_channelwise_gemm` 无 split-K 参数、无源码，强制 API 下拿不回来（3 个隐藏变体 `_fast`/`_kme` 实测与 baseline 一致）。

**优化优先级（batch=128）**：

1. **Attention 20ms**（25.4%，唯一随 batch 线性增长）：int4 KV（字节减半，~10ms）> combine（2.25ms，长序列受 smem 限制去不掉）。
2. **Quant/SiLU 9.44ms**（12%）：融合进 GEMM epilogue（~2-4ms）。
3. **GEMM 40.64ms**（51.6%，最大但最难）：M=128 天花板，split-K 仅 ~2.5ms，API 内无解。

## 6.8 单 step 算子流程图与融合空间（2026-09-07）

### 6.8.1 一个 decode step 的算子流程（batch=128，64 层）

```
hidden [128,5120] bf16
  │
  │  ┌─────────────────────────── 每层 ×64 ───────────────────────────┐
  ├─▶│ ① fused_add_rms_norm (residual add + RMSNorm 已融合)            │
  │   │    输入 hidden+residual → 输出 normed [128,5120] bf16 + 新残差  │
  │   │    kernel: vllm custom op (custom_ops: +rms_norm)              │
  │   ▼                                                                │
  │  ② quant_fp8 (激活量化, 独立 kernel)                                │
  │     [128,5120] bf16 → fp8 + per-token scale [128]                  │
  │     kernel: vllm custom op (custom_ops: +quant_fp8, ~13.4us)       │
  │   ▼                                                                │
  │  ③ QKV GEMM (fp8 channelwise, hipblaslt)  78.8us/层                │
  │     [128,5120]fp8 × W[10240,5120]fp8 → [128,10240] bf16            │
  │     拆出 Q[128,8192] K[128,1024] V[128,1024] (GQA 8:1)             │
  │   ▼                                                                │
  │  ④ q_norm / k_norm (head_dim=128 的 RMSNorm, 2 个小 kernel)         │
  │   ▼                                                                │
  │  ⑤ KV cache write (reshape_and_cache, fp8_e4m3)                    │
  │   ▼                                                                │
  │  ⑥ Paged Attention (flash-attention-cutlass, fp8 KV, block=64)     │
  │     → [128,8192] bf16   (main 17.75ms + combine 2.25ms /step)      │
  │   ▼                                                                │
  │  ⑦ O GEMM (fp8 channelwise)  57.7us/层                             │
  │     [128,8192] → 需先 quant? 否: attention 输出 bf16,               │
  │     O GEMM 输入走 quant_fp8 (第 2 次量化)                           │
  │     → [128,5120] bf16                                              │
  │   ▼                                                                │
  │  ⑧ fused_add_rms_norm (第 2 次, post_attention)                     │
  │   ▼                                                                │
  │  ⑨ quant_fp8 (第 3 次量化) [128,5120] → fp8                        │
  │   ▼                                                                │
  │  ⑩ gate+up GEMM (fp8 channelwise)  317us/层                        │
  │     [128,5120]fp8 × W[51200,5120]fp8 → [128,51200] bf16            │
  │     拆 gate[128,25600] up[128,25600]                               │
  │   ▼                                                                │
  │  ⑪ silu_and_mul (独立 kernel)                                      │
  │     SiLU(gate)·up → [128,25600] bf16                               │
  │   ▼                                                                │
  │  ⑫ quant_fp8 (第 4 次量化) [128,25600] → fp8                       │
  │   ▼                                                                │
  │  ⑬ down GEMM (fp8 channelwise)  172us/层                           │
  │     [128,25600]fp8 × W[5120,25600]fp8 → [128,5120] bf16            │
  │   ▼                                                                │
  │  ⑭ residual add (融进下一层 ① 的 fused_add_rms_norm)                │
  └──┴──────────────────────────────────────────────────────────────────┘
  │
  ▼
  ⑮ final RMSNorm → LM Head GEMM (vocab 151936) → 采样 (fused topk+topp)
```

**每层 kernel 清单（7 个主 kernel + 2 个小 norm）：**

| # | kernel | 次数/step | 单 kernel 耗时 | 小计/step | 现状 |
|---|---|---|---|---|---|
| ① | fused_add_rms_norm | 128 | ~30us | ~3.8ms | **add+norm 已融合** |
| ②⑨ | quant_fp8 (hidden 侧) | 128 | ~13.4us | ~1.7ms | 独立 |
| ⑦前 | quant_fp8 (attn 输出) | 64 | ~13.4us | ~0.9ms | 独立 |
| ⑫ | quant_fp8 (mlp 中间) | 64 | ~20us | ~1.3ms | 独立 |
| ⑪ | silu_and_mul | 64 | ~50us | ~3.2ms | 独立 |
| ③⑦⑩⑬ | GEMM ×4 | 256 | 57~317us | 40.64ms | hipblaslt |
| ⑥ | paged attention + combine | 64 | ~310us | 20.0ms | flash-attn |
| ④ | q/k norm (head_dim) | 128 | ~5us | ~0.6ms | 独立小 kernel |
| ⑤ | KV write | 64 | ~23us | 1.49ms | lightop |

**合计 ~78.8ms/step。其中"小 kernel 区"（①②⑨⑪⑫④）≈ 16.7ms，带宽利用率仅 12.7%——launch-bound，不是带宽问题。**

### 6.8.2 已融合 vs 未融合（2026-09-07 trace 实测修正）

**重要修正**：之前以为 A/B 未融合，trace 证实**已融合且已生效**。

| 操作对 | 状态 | 证据（batch=16 trace，÷513 steps） |
|---|---|---|
| residual add + RMSNorm | ✅ 已融合 | `fused_add_rms_norm` |
| **RMSNorm + quant（A）** | ✅ **已融合** | `rms_norm_dynamic_per_token_quant_kernel<true>` 127 次/step，1.857ms |
| **silu_and_mul + quant（B）** | ✅ **已融合** | `triton_..._quant_mul_silu` 64 次/step，0.688ms |
| SiLU + mul | ✅ 已融合 | 融进 B |
| top-k + top-p 采样 | ✅ 已融合 | fused_topk_topp（10ms→0.955ms） |
| **attn 输出 → O GEMM 的 quant** | ❌ 独立，**融不掉** | `dynamic_per_token_scaled_fp8_quant` 128 次/step，2.041ms（前面无 norm 可搭车） |
| q/k norm | ❌ 独立（triton） | `triton_..._rms_norm_split` ~1.3ms/step |
| GEMM epilogue + quant | ❌ 未融合 | 需改 DTK 闭源 kernel，不可行 |

**启用路径**：`--optimization-level 1` → `fuse_norm_quant = enable_norm_fusion = True`（custom_ops 含 rms_norm/quant_fp8）→ `RMSNormQuantFusionPass`（`compilation/passes/fusion/rms_quant_fusion.py`）自动把 `fused_add_rms_norm + dynamic_per_token_quant` 匹配替换成 `rms_norm_dynamic_per_token_quant`。B 走 `ActivationQuantFusionPass`（triton 实现）。

**C++ 融合 kernel**（`csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu:50`），`has_residual=true` 时一个 kernel 干三件事：
```cpp
vllm::compute_rms<scalar_t, has_residual>(&rms, input, ..., residual);            // ① add+RMS
vllm::compute_dynamic_per_token_scales<...>(&token_scale, scales, input, weight, rms, ..., residual);  // ② per-token scale
vllm::norm_and_quant<...>(out, input, weight, rms, &token_scale, ..., residual);  // ③ norm+quant 出 fp8
```

### 6.8.3 剩余可优化空间（修正后）

**A/B 已融合，Quant/Norm/SiLU 区基本优化到位。** 剩余：

| 项 | 现状 | 空间 | 说明 |
|---|---|---|---|
| attn→O 独立 quant | 2.041ms/step | **融不掉** | attention 直接输出 bf16，无 norm 可搭车；除非改 attention 输出直接 fp8（动 flash-attn kernel） |
| q/k norm | ~1.3ms/step | ~0.5ms | 可试融进 attention 前处理，收益小、要动 attention 入口 |
| GEMM epilogue 融合 quant | — | — | ✗ DTK 闭源 |

**结论：之前估的"再省 4ms"不成立——A/B 已经融了。** Quant/Norm/SiLU 区（batch=128 的 16.7ms）已接近该区的下限，主要剩余是 attn→O 的独立 quant（结构性融不掉）。真正的优化空间回到 Attention（int4 KV）和 GEMM（推 lightop split-K）。

### 6.8.4 为什么 QKV/O 保留 bf16 不划算（2026-09-07 实测否决）

想法：QKV/O 权重小（52/42MB），保留 bf16 省掉 2 次激活量化。实测（M=128）：

| GEMM | fp8 | bf16 TN | /step 增加 |
|---|---|---|---|
| QKV | 4.54ms | 10.35ms | +5.81ms |
| O | 3.69ms | 8.37ms | +4.68ms |
| 合计 | 8.23ms | 18.72ms | **+10.49ms** |

省 2 次量化仅 ~3.1ms，**净亏 +7.4ms**。根因：M=128 GEMM 是 memory-bound，耗时 ∝ 权重字节，bf16 字节翻倍 = 时间翻倍；量化 kernel 搬的是 1.3MB 小激活，省不了多少。若 KV cache 也随 QKV 变 bf16，Attention 再 +20ms，直接否决。**维持全 fp8。**

### 6.9 FP8 GEMM 带宽利用率实测 + split-K 验证（2026-09-09，card 4，真实权重）

**目的**：确认 decode GEMM 还有没有优化空间（之前记忆里"M=128 带宽瓶颈、40% 利用率、缺 split-K"的说法是否成立）。

**方法**：加载 layer-0 真实 fp8 权重（OCP e4m3，per-channel scale [N,1] bf16），跑生产路径 `hipblaslt_w8a8_channelwise_gemm`（= `rocblas_gemm_ex`），扫 M ∈ {1,4,8,16,32,64,128,256,512}，算有效 HBM 带宽。峰值按 **1.8 TB/s**。脚本 `/workspace/_src/hipblaslt_splitk_probe/bw_real.py`。

**带宽口径**：bytes = N×K（fp8 权重）+ M×K（fp8 激活）+ M×N×2（bf16 输出）。

#### 6.9.1 带宽利用率 vs M（真实权重，card 4，3 次重复误差 <1%）

| M | QKV (10240×5120) | O (5120×8192) | gate+up (51200×5120) | down (5120×25600) | **layer 合计** |
|---|---|---|---|---|---|
| 1 | 49.9us / 58% | 50.0us / 47% | 179.2us / 81% | 79.6us / 92% | 0.359ms / 1360 GB/s / **75.6%** |
| 4 | 50.3us / 58% | 50.4us / 46% | 179.5us / 81% | 79.6us / 92% | 0.360ms / 1357 GB/s / **75.4%** |
| 8 | 50.5us / 58% | 50.6us / 46% | 179.8us / 81% | 79.1us / 92% | 0.360ms / 1358 GB/s / **75.5%** |
| 16 | 50.8us / 58% | 50.8us / 46% | 177.0us / 83% | 78.9us / 93% | 0.357ms / 1373 GB/s / **76.3%** |
| 32 | 50.7us / 58% | 50.5us / 47% | 214.2us / 69% | 102.8us / 71% | 0.418ms / 1180 GB/s / **65.6%** |
| 64 | 56.0us / 54% | 50.7us / 47% | 218.4us / 68% | 133.0us / 56% | 0.458ms / 1091 GB/s / **60.6%** |
| **128** | 70.1us / 44% | 60.0us / 41% | 307.6us / 50% | 173.8us / 43% | 0.611ms / 837 GB/s / **46.5%** |
| 256 | 84.9us / 39% | 97.6us / 27% | 422.5us / 38% | 290.7us / 27% | 0.896ms / 598 GB/s / **33.2%** |
| 512 | 163.0us / 22% | 110.7us / 26% | 749.1us / 24% | 382.9us / 22% | 1.406ms / 415 GB/s / **23.1%** |

（每格 = 耗时 / 带宽利用率；layer = 4 个 GEMM 之和。每层权重 487.6 MB × 64 层 = 31.2 GB/step，是 decode 固定带宽下限。）

**数据可信度**：M=128 layer = 0.611ms × 64 层 = **39.1ms**，与 profiler trace 的 **40.64ms/step** 几乎完全吻合。card 0 曾出现 run-to-run 波动（gate+up@M=1 在 177~221us 跳），是那张卡有干扰；card 4 干净，3 次重复误差 <1%。

#### 6.9.2 关键结论（修正旧判断）

1. **利用率随 M 下降，不是上升**——旧记忆"M=128 带宽瓶颈、40% 利用率没榨干"是错的。
   - **M≤16 是带宽瓶颈区**，layer 利用率 ~75%，kernel 效率已经很好（不是 40% 那种低效）。
   - **M 越大利用率越低**（M=128 → 46%，M=512 → 23%），这是**混合区→算力受限**的自然结果，不是 kernel 缺陷。
   - 脊点（BW/compute 平衡）约在 M=16~32。
2. **M=32 的 cliff 是 tile 切换 artifact**：gate+up/down 在 M=32 跳一下（177→214us）然后变平，QKV/O 不受影响（N×K 小，tile 选择没在那个点变）。不是物理 regime 变化，跟"能不能优化"无关。
3. **decode GEMM 处于混合区**：M=16→128 的 layer 时间 0.357→0.611ms = **×1.72**，既非纯带宽（×1.0）也非纯算力（×8），与 profiler 的 batch 16→128 GEMM ×1.78 一致。

#### 6.9.3 split-K 验证（hipBLASLt 标准 API，FNUZ W8A8）

**目的**：验证"用现成 split-K kernel 能否提速 M=128 GEMM"。

**发现**：
- 已安装 `/opt/dtk/lib/libhipblaslt.so` 有完整 `hipblaslt_ext` + 标准 `hipblasLtMatmul` API。hipBLASLt 的 W8A8 用 **`HIP_R_8F_E4M3_FNUZ`**（AMD FNUZ 变体，**非** OCP e4m3），双 fp8 + bf16 输出 + `OUTER_VEC_32F` scale。
- **device 库确实有 gfx938 的 split-K fp8 kernel**：枚举的 64 个 algo **全部**报 `ALGO_CAP_SPLITK_SUPPORT`（推翻"所有现成 split-K 路径都死"的旧判断）。
- **但最佳 algo 只比生产路径快 1.00~1.17x**（A/B 同卡同脚本，M=128）：

| shape | 生产 rocblas_gemm_ex | hipBLASLt 最佳 algo | speedup |
|---|---|---|---|
| QKV | 0.0720ms | 0.0617ms | 1.17x |
| O | 0.0606ms | 0.0608ms | 1.00x |
| gate+up | 0.3091ms | 0.2894ms | 1.07x |
| down | 0.1735ms | 0.1731ms | 1.00x |
| per-layer | 0.6153ms | 0.5849ms | **1.05x** |

**为什么不是 2-3x**：split-K 是提升**带宽**利用率的手段，但 M=128 的瓶颈已转向**算力**（见 6.9.2），不在 split-K 作用的地方。生产 `rocblas_gemm_ex` 内部已做最优 K 维并行，hipBLASLt 的 split-K 变体没有一个显著超越它。1.05x 的 GEMM 提升 → 整体 step 仅 ~2%，**不值得**为它改 vllm GEMM 路径（还要处理 FNUZ vs OCP 的 scale 布局差异 + 正确性回归）。

#### 6.9.4 最终判断

**GEMM 这条路彻底没有优化空间**：小 M（≤16）已近峰值（75%），大 M 是混合/算力墙，中间的 cliff 是库的 tile 选择（动不了）。要提速 decode 只能从**非 GEMM 部分**（Attention 随 batch 线性增长、Quant-Norm 已接近下限）或**减少 GEMM FLOPs/权重读取**（更狠量化，但用户要求无损，排除）下手。

> 注：`/workspace/_src` 下 8 个源树（blockwise_w8a8_asm / ck_upstream / DeepGEMM / flash-attention-cutlass / hipBLASLt / lightop-master / tilelang / vllm）评估结论——只有 **hipBLASLt** 有现成可用的 fp8 split-K 入口（已验证，无收益）；ck_upstream 的 fp8 MFMA 只 gate 在 gfx942/950（不含 gfx938）；lightop-master 的 dense `.co` 全是 int8、smooth 路径 M≤16；DeepGEMM 是 CUDA-only。

## 7. 结论与落地建议

1. **量化**：Qwen3-32B → FP8 rowwise（per-channel 权重 + per-token 动态激活，W8A8）产物 `/models/Qwen3-32B-FP8-ROWWISE/` 可用，W8A8 是本平台正确选择。
2. **W8A16 放弃**：ROCm 侧无 vllm kernel、底层内核慢 3–9 倍且 M>16 崩溃，是死路。
3. **性能瓶颈已定位并修复**：decode 34% 耗时来自 inductor 自动融合的慢量化内核，根因是 `custom_ops: ['none']` 关掉了 fusion pass。
4. **并行策略：DP4 优于 TP4**（4 卡场景）：DP4 消除 AllReduce（占 TP4 的 33.9%），实测 TPOT −24% / 吞吐 +18%。`start.sh` 默认 DP4。
5. **采样优化**：fused top-k+top-p kernel 把采样从 ~10ms 降到 0.955ms/step（4.3x），45/45 正确性通过。
6. **落地建议（明确）**：

   - 启动：`bash start.sh`（默认 DP4，custom_ops ON，opt1）；
   - 关键启动参数：`--data-parallel-size 4 --tensor-parallel-size 1` + `--compilation-config '{"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul", "+rotary_embedding"]}'`；
   - 采样：集成 fused kernel（`vllm/v1/sample/ops/fused_topk_topp.py`，见 `docs/fused_topk_topp.md`）。

   综合收益（相对最初 baseline）：decode 吞吐 +31%（custom_ops）+ 并行/采样优化，TPOT 从 ~70ms 降到 ~53ms。

## 8. 附录

### 8.1 关键脚本清单

| 脚本 | 用途 |
|---|---|
| `/workspace/quantize_qwen3_32b_fp8_rowwise.py` | llm-compressor 量化脚本（W8A8 rowwise） |
| `/workspace/bench_w8a16_vs_w8a8.py` | W8A16 vs W8A8 微基准（decode 形状） |
| `/workspace/profile_w8a16.py` | torch profiler 对比两条路径的实际内核 |
| `/workspace/analyze_quant_trace.py` | 解析 decode chrome trace，统计量化内核 GPU 时间 |
| `/workspace/start_fp8_ab.sh` | A/B 服务器启动脚本（8005，custom_ops ON） |
| `/workspace/ab_verify.sh` | A/B 压测脚本（16 并发 × 512 输出） |
| `start.sh`（本目录） | **默认 DP4** 启动脚本（`PARALLEL_MODE=tp4` 切回 TP4） |
| `/workspace/start_dp4.sh` | DP4 启动脚本（端口 8009，对比用） |
| `/workspace/start_tp4.sh` | TP4 启动脚本（端口 8007） |
| `/workspace/start_tp4_profiler.sh` | TP4 + torch profiler（`/start_profile`/`/stop_profile`） |
| `/workspace/_bench/test_fused_vs_triton.py` | fused 采样 kernel 正确性测试（45 用例） |
| `/workspace/_bench/bench_sampling_3way_peaked.py` | 采样 3 路对比（triton/fast/fused，尖峰分布） |
| `vllm/v1/sample/ops/fused_topk_topp.py` | fused top-k+top-p kernel（集成进 vllm） |

### 8.2 服务器启动命令

**8001（baseline，custom_ops OFF）**：

```bash
vllm serve /models/Qwen3-32B-FP8-ROWWISE/ --host 0.0.0.0 --port 8001 --served-model-name qwen3-32b --dtype auto --kv-cache-dtype fp8_e4m3 --max-model-len 32768 --max-num-seqs 512 --attention-backend FLASH_ATTN --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --optimization-level 3 --trust-remote-code --enable-prefix-caching --enable-chunked-prefill --language-model-only --async-scheduling --performance-mode throughput --max-num-batched-tokens 16384 --profiler-config {"profiler": "torch", "torch_profiler_dir": "/workspace/prof_traces", "torch_profiler_with_stack": false}
```

**8005（A/B，custom_ops ON）**——唯一区别是多了这一行（profiler 目录为 `/workspace/prof_ab`）：

```bash
--compilation-config {"custom_ops": ["+quant_fp8", "+rms_norm", "+silu_and_mul"]}
```
