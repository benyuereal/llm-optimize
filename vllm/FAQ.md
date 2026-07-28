# vLLM 架构演进 Q&A

围绕 vLLM 从 PagedAttention 到 MRv2 的代际更替，整理几个常被问到的问题。

---

## Q1：v0.25.0 到底删没删 PagedAttention？

**删了。** 删除的是 **2023 年让 vLLM 一战成名的原始 PagedAttention CUDA kernel**，由 commit `d715b3aa1` "Delete PagedAttention (#47361)" 完成，在 v0.25.0rc1 起生效。

**源码铁证**（逐版本查证）：

| 版本 | 老 PagedAttention kernel 文件 `csrc/libtorch_stable/attention/paged_attention_v1.cu` |
|---|---|
| v0.24.0 | ✅ 存在 |
| v0.25.0rc1 | ❌ 已删除 |
| v0.25.0 | ❌ 已删除 |
| v0.26.0 | ❌ 已删除 |

删除规模：约 **1472 行**——`paged_attention_v1.cu`（190 行）+ `paged_attention_v2.cu`（202 行）+ `attention_kernels.cuh`（667 行）+ `vllm/_custom_ops.py` 相关 op（94 行）等。

### 容易混淆的点：vLLM 里有两个" PagedAttention "

| | 老 PagedAttention（被删的）| v1 PagedAttention 类（没删）|
|---|---|---|
| **是什么** | 2023 年原始的 CUDA kernel 实现 | V1 架构里一个同名 Python wrapper 类 |
| **路径** | `csrc/libtorch_stable/attention/paged_attention_v1.cu` 等 | `vllm/v1/attention/ops/paged_attn.py` |
| **v0.25.0 状态** | ❌ **删除** | ✅ 仍在（是 V1 新代码，不是退役的那个）|

查证时容易查到后者（`vllm/v1/attention/ops/paged_attn.py` 的 `class PagedAttention`）发现它还在，就误以为"没删"。但它跟真正被退役的 PagedAttention kernel 是两回事。

---

## Q2：删的只是 kernel，PagedAttention 的"思想"删了吗？

**思想没删，也删不掉。** PagedAttention 的核心思想是**用操作系统的虚拟内存分页机制管理 KV Cache**：

```
传统做法：每个请求预分配一整块连续显存放 KV Cache
  → 请求长短不一 → 显存碎片化严重 → 利用率只有 ~60%

PagedAttention 思想：把 KV Cache 切成固定大小的"页"(block)
  → 按需分配，用 block table 记录逻辑→物理映射
  → 像 OS 的虚拟内存分页一样 → 利用率拉到 90%+
```

这个"分页管理 KV Cache"的思想是 vLLM 的立身之本，**新引擎 MRv2 继承了它**，只是不再用那套老的 CUDA kernel 实现。

> 比方：PagedAttention 思想 = "用分页管理内存"这个设计理念；被删的 kernel = 实现这个理念的一版旧代码。理念活在新架构里，旧代码退役了。

---

## Q3：MRv2 是什么？全称？它跟 PagedAttention 什么关系？

### 全称

**MRv2 = Model Runner V2**。对应代码里的 **V1 engine**（`vllm/v1/` 目录，核心文件如 `vllm/v1/worker/gpu_model_runner.py`）。

> 命名易混点：vLLM 有两套命名。
> - **V0 / V1 engine**：内部对推理引擎架构的代号。V0 是老架构（`vllm/engine/`、`vllm/executor/`），**V1 是新架构**（`vllm/v1/`）。"V1 engine"就是新引擎。
> - **Model Runner V2 (MRv2)**：新引擎（V1）里的执行核心。社区/新闻常把"V1 engine + 它的 Model Runner"合称为 MRv2。
>
> 所以 **MRv2 ≈ V1 engine 的 Model Runner**。新闻里"MRv2 全面接管"= V1 engine 成为唯一路径。

### 它是干嘛的

MRv2 是 vLLM 的**新一代推理执行引擎**（执行器/运行时），是整个推理流程的"底盘"。它负责：请求怎么排队、KV Cache 怎么管、算子怎么调、多 GPU 怎么协作、CUDA Graph 怎么用……

### 为什么要有 V2（老架构的硬伤）

| 问题 | 老架构（V0 + 早期 V1）| MRv2 |
|---|---|---|
| **CUDA Graphs** | 部分支持，很多场景用不了 | 完整支持（推理大加速）|
| **投机解码** | 有限，常跟 CUDA Graphs 冲突 | 原生支持，动态投机解码 + 完整 CUDA Graphs 兼容 |
| **多模态** | 后期打补丁式适配 | 原生设计，一等公民 |
| **代码维护** | 多套执行路径并存，测试复杂 | 统一一条路径，好维护 |

### 跟 PagedAttention 思想的关系

**MRv2 继承了 PagedAttention 的分页思想，但用更现代的方式实现。** MRv2 里 KV Cache **仍然按 block（页）管理**（分页思想保留），但：

- 不再依赖那套老的 `paged_attention_v1/v2.cu` kernel
- 改用统一的 attention backend（flash-attention / flashinfer / TRT-LLM MLA 等）来访问这些 block
- 跟 CUDA Graphs、投机解码、多模态深度整合

是**"思想留存、实现换代"**。新闻里的比喻：*"你不再用 VHS 录像带，但'录制回放'的概念永远存在。PagedAttention 退役，但它开创的显存管理范式活在每一行新代码里。"*

---

## Q4：MRv2 是什么时候加的？

**v0.7.0 引入**（`vllm/v1/` 目录 + `vllm/v1/worker/gpu_model_runner.py` 首次出现）。但那时是**实验性**，不默认。

源码查证：

| 版本 | `vllm/v1/` 目录 | `gpu_model_runner.py` |
|---|---|---|
| v0.6.0 | ❌ 无 | ❌ 无 |
| **v0.7.0** | ✅ **有** | ✅ **有** |
| v0.8.0+ | ✅ 有 | ✅ 有 |

### MRv2 从引入到"全面接管"的完整时间线

| 版本 | V1/MRv2 状态 | 说明 |
|---|---|---|
| **v0.7.0** | 🆕 引入 | `vllm/v1/` + `gpu_model_runner.py` 首次出现，V1 engine 架构落地（实验性）|
| v0.8.0 ~ v0.16.0 | 实验性打磨 | 逐步补能力（attention / kv_cache / spec_decode 等），默认仍走 V0 |
| v0.17.0 ~ v0.19.0 | 可选启用 | V1 可通过环境变量启用，开始覆盖更多模型 |
| v0.20.0 ~ v0.23.0 | 默认化过渡 | V1 逐步成为默认，覆盖面扩大 |
| **v0.24.0** | 默认引擎 | MRv2 从"实验性"走向"默认"，四大场景（稠密 / MoE / 量化 / 推测解码）补齐 |
| **v0.25.0** | 唯一路径 | **MRv2 成为所有稠密模型唯一默认执行路径**，删除老 PagedAttention kernel |

> MRv2 不是 v0.25.0 才加的——它从 v0.7.0 就开始孵化，经过约 18 个版本（v0.7 → v0.25）的打磨，到 v0.25.0 才正式"全面接管"。v0.25.0 删 PagedAttention，是因为这时 MRv2 终于成熟到可以独挑大梁了。

---

## Q5：MRv2 全面接管，对实际使用有什么影响？

**大多数用户无感**——用户层 API（请求格式、模型加载、部署流程）没变，可能完全感知不到底层换了引擎。

**需要关注的**：
- 如果代码/配置引用了 PagedAttention 的**内部 API**，必须移除
- 如果有**自定义 attention 后端**，迁移成本不可忽视
- v0.25.0 同时移除了 6 个旧模型（Baichuan / Aquila / Grok / Tarsier / AyaVision / MusicFlamingo 等），用这些模型的需先确认替代方案（可走 Transformers 后端，v0.25.0 其性能已追平原生）
