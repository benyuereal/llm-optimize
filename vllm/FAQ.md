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

**MRv2 = Model Runner V2**，是 V1 引擎内部 GPU 执行器的**第二代重写**，首个 commit 为 `#25266 GPU Model Runner V2`。

### ⚠️ 最容易搞混的点：vLLM 有三层概念，不是一个东西

很多人（包括早期版本的本文档）把下面三个混为一谈，其实它们是**不同层级、不同时间**的换代：

| 层级 | 名称 | 代码位置 | 首次出现 | 说明 |
|---|---|---|---|---|
| 引擎层 | **V0 engine**（旧）| `vllm/engine/`、`vllm/core/` | 早期 | 老架构，`LLMEngine` 在 v0.11.0 被掏空成重导出 V1 的垫片 |
| 引擎层 | **V1 engine**（新）| `vllm/v1/` | **v0.7.0**（`#9289`）| 新引擎架构，v0.8.0 默认开 |
| 执行器层 | **MRv2 / Model Runner V2** | `vllm/v1/worker/gpu/`（新目录）| **v0.16.0**（`#25266`，2025-11-21）| V1 引擎**内部**执行器的第二代重写，与老 `gpu_model_runner.py` 并存 |

**关键区别**：
- **V1 engine ≠ MRv2**。V1 engine 是 v0.7.0 起的引擎层换代；MRv2 是 v0.16.0 起在 V1 引擎**内部**对 GPU 执行器的**又一次重写**。MRv2 比 V1 engine 晚了约 9 个版本。
- **MRv2 有用户开关 `VLLM_USE_V2_MODEL_RUNNER`**（v0.24.0 起）。未设时按场景**条件启用**：PCP>1、dspark 投机解码、DFlash 混合草稿、diffusion 模型等**强制走 V2**；默认 V2 模型列表内 + 有 Triton + 不在不支持特性黑名单上的模型才走 V2，否则回退老执行器。所以 MRv2 是"渐进式接管"，不是某版本一刀切全局默认。
- 新闻里"MRv2 全面接管"≈ V1 engine + 它的 Model Runner V2 成为唯一路径，是**两层换代的叠加结果**。

### 它是干嘛的

MRv2 是 vLLM 推理流程的"底盘"执行器：请求怎么排队、KV Cache 怎么管、算子怎么调、多 GPU 怎么协作、CUDA Graph 怎么用，都由它驱动。

### 为什么要有 MRv2（老执行器的硬伤）

| 问题 | 老 Model Runner（V1 早期）| MRv2 |
|---|---|---|
| **CUDA Graphs** | 部分支持，很多场景用不了 | 完整支持（推理大加速）|
| **投机解码** | 有限，常跟 CUDA Graphs 冲突 | 原生支持，动态投机解码 + 完整 CUDA Graphs 兼容 |
| **多模态** | 后期打补丁式适配 | 原生设计，一等公民 |
| **采样** | 路径分散 | Gumbel sampling 等融合优化 |
| **代码维护** | 多套执行路径并存，测试复杂 | 统一一条路径，好维护 |

### 跟 PagedAttention 思想的关系

**MRv2 继承了 PagedAttention 的分页思想，但用更现代的方式实现。** MRv2 里 KV Cache **仍然按 block（页）管理**（分页思想保留），但：

- 不再依赖那套老的 `paged_attention_v1/v2.cu` kernel
- 改用统一的 attention backend（flash-attention / flashinfer / TRT-LLM MLA 等）来访问这些 block
- 跟 CUDA Graphs、投机解码、多模态深度整合

是**"思想留存、实现换代"**。新闻里的比喻：*"你不再用 VHS 录像带，但'录制回放'的概念永远存在。PagedAttention 退役，但它开创的显存管理范式活在每一行新代码里。"*

### 🔍 源码实证（当前仓库 HEAD）

> 以下代码均为当前仓库真实文件内容，作为上述结论的实证。当前 HEAD：`90245f419`（`v0.26.1rc0` 之后 21 个 commit，约 v0.26.1 开发中）。

**实证 1：V0 已成垫片 —— `vllm/engine/llm_engine.py` 全文（7 行）**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

LLMEngine = V1LLMEngine  # type: ignore
"""The `LLMEngine` class is an alias of [vllm.v1.engine.llm_engine.LLMEngine][]."""
```

`vllm/engine/async_llm_engine.py` 同理（6 行），`AsyncLLMEngine = AsyncLLM`（重导出 `vllm.v1.engine.async_llm`）。V0 的引擎类早已被 V1 替换，文件名仅为兼容老 import 保留。

**实证 2：V1 引擎独立目录 —— `vllm/v1/` 结构**

```
vllm/v1/
├── attention/          # attention 后端（含 mla/ 子目录、20+ backend）
├── core/               # 调度 + KV 管理（sched/、block_pool.py、kv_cache_manager.py）
├── engine/             # 引擎核心（core.py、async_llm.py、llm_engine.py）
├── executor/           # 执行器（uniproc/multiproc）
├── kv_offload/         # KV 卸载
├── spec_decode/        # 投机解码内建（eagle/dflash/gemma4/ngram/...）
├── structured_output/  # 结构化输出
├── worker/             # ← 执行器层在这里
│   ├── gpu_model_runner.py   # 老 GPU 执行器（GPUModelRunner, 7902 行）
│   ├── gpu/                  # ← MRv2 新目录（见实证 3）
│   ├── cpu_model_runner.py
│   └── ...
├── kv_cache_interface.py
└── ...
```

**实证 3：MRv2 是独立新目录 —— `vllm/v1/worker/gpu/README.md` 全文**

```
# [Experimental] Model Runner V2

This directory contains the new model runner which is under active development.
Ping [Woosuk Kwon](https://github.com/WoosukKwon) for any changes.
```

该目录由 `#25266 GPU Model Runner V2` 创建，核心文件 `gpu/model_runner.py`（`class GPUModelRunner`，约 73KB），与老执行器 `vllm/v1/worker/gpu_model_runner.py`（`class GPUModelRunner`，7902 行）**并存**。两个同名类、两条执行路径同时存在，正是"渐进式接管"的代码形态。

**实证 4：MRv2 有开关 —— `vllm/envs.py`**

```python
VLLM_USE_V2_MODEL_RUNNER: bool | None = None   # 第 275 行
...
# Flag to control the v2 model runner. If unset, use config defaults.
"VLLM_USE_V2_MODEL_RUNNER": lambda: maybe_convert_bool(
    os.getenv("VLLM_USE_V2_MODEL_RUNNER", None)
),
```

**实证 5：MRv2 启用判定 —— `vllm/config/vllm.py` `use_v2_model_runner`**

```python
def use_v2_model_runner(self) -> bool:
    use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        return use_v2_model_runner              # 环境变量优先

    # PCP runtime 仅 V2 实现 → 强制 V2
    if self.parallel_config.prefill_context_parallel_size > 1:
        return True
    # DSpark 投机解码仅 V2 实现 → 强制 V2
    if (self.speculative_config is not None
            and self.speculative_config.method == "dspark"):
        return True
    # DFlash 混合草稿需多 KV 组（仅 V2）→ 强制 V2
    if self._dflash_needs_multi_kv_group():
        return True
    # diffusion 模型 → 强制 V2
    if self.model_config is not None and self.model_config.is_diffusion:
        return True

    # 其余：仅"默认 V2 模型列表"内的模型 + 有 Triton + 不在支持特性黑名单 → 才走 V2
    if not self._is_default_v2_model_runner_model():
        return False
    if not HAS_TRITON:
        logger.warning_once("Model Runner V2 requires Triton; using the V1 ...")
        return False
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        logger.warning_once("Model Runner V2 does not yet support %s; ...", ...)
        return False
    return True
```

读这段代码即可确认：MRv2 不是"全局默认开/关"，而是**按模型与场景条件启用**，不支持时回退老执行器并打 warning。这正是"渐进式接管"的机制——也解释了为何没有一个干净的"MRv2 默认版本"断点。

> 三层概念与完整时间线的更多细节见 [V1-vs-MRv2.md](V1-vs-MRv2.md)。

---

## Q4：MRv2 是什么时候加的？

**纠正一个常见误解**：MRv2 **不是** v0.7.0 加的（那是 V1 engine），也**不是** v0.24/0.25 才出现。MRv2 从 **v0.15/v0.16 就开始开发**，到 v0.18 已相当活跃——经 git 逐版本核实（统计含 `MRv2`/`Model Runner V2` 的 commit 数）：

| 版本 | MRv2 相关 commit 累计数 | 节点 |
|---|---|---|
| v0.15.0 | 52 | 开发中（首个 commit `#25266` 日期 **2025-11-21**）|
| **v0.16.0** | 55 | **`#25266 GPU Model Runner V2` 已合入**，MRv2 正式落地 |
| v0.17.0 | 93 | 快速增长 |
| **v0.18.0** | 112 | 已成熟，多模态/投机解码/CUDA Graph 持续补齐 |
| v0.19.0 | 131 | |
| v0.20.0 | 144 | |
| v0.22.0 | 173 | |
| v0.24.0 | 200 | |
| v0.25.0 | 220 | 删除老 PagedAttention kernel |
| v0.26.0 | 225 | |

### 三层换代的完整时间线（git 核实）

| 版本 | V1 engine（引擎层）| MRv2（执行器层）| PagedAttention kernel | V0 LLMEngine |
|---|---|---|---|---|
| v0.6.0 | ❌ 无 | ❌ 无 | ✅ 在 | ✅ 2061 行真实实现 |
| **v0.7.0** | ✅ 引入（`#9289`），默认关 | — | ✅ 在 | ✅ 在 |
| **v0.8.0** | 默认开（`VLLM_USE_V1=True`）| — | ✅ 在 | ✅ 在 |
| **v0.11.0** | 已默认 | — | ✅ 在 | ⚰️ **掏空成 6 行垫片** |
| v0.12.0 | `use_v1` 参数废弃（`#28112`），固化 | — | ✅ 在 | 垫片 |
| **v0.16.0** | 已默认 | ✅ **MRv2 引入**（`#25266`）| ✅ 在 | 垫片 |
| v0.18.0 | 已默认 | 112 commit，成熟中 | ✅ 在 | 垫片 |
| v0.24.0 | 已默认 | 200 commit，能力补齐 | ✅ 在 | 垫片 |
| **v0.25.0** | 已默认 | 220 commit | ❌ **删除**（`d715b3aa1`）| 垫片 |

> ⚠️ **关于 `#25033`**：commit `[V0 Deprecation] Remove LLMEngine (#25033)` 标题容易让人以为"v0.25 删了 LLMEngine"，但它实际日期是 2025-10-03、位于较早分支，不在 v0.24/v0.25/v0.26 tag 内。主线 V0 `LLMEngine` 早在 **v0.11.0** 就被掏空成重导出 V1 的 6 行垫片（v0.10.0 还是 2061 行真实实现）。判版本不能只看 commit 标题。

> **结论**：MRv2 从 v0.15/v0.16 起步，经约 10 个版本（v0.16 → v0.25）打磨。v0.25.0 删 PagedAttention kernel，是因为这时 MRv2 + V1 engine 终于成熟到可以独挑大梁。**说"MRv2 在 v0.15~0.18 就已存在"是准确的。**

---

## Q5：MRv2 全面接管，对实际使用有什么影响？

**大多数用户无感**——用户层 API（请求格式、模型加载、部署流程）没变，可能完全感知不到底层换了引擎。

**需要关注的**：
- 如果代码/配置引用了 PagedAttention 的**内部 API**，必须移除
- 如果有**自定义 attention 后端**，迁移成本不可忽视
- v0.25.0 同时移除了 6 个旧模型（Baichuan / Aquila / Grok / Tarsier / AyaVision / MusicFlamingo 等），用这些模型的需先确认替代方案（可走 Transformers 后端，v0.25.0 其性能已追平原生）
