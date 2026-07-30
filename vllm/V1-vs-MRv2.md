# V1 engine vs MRv2：vLLM 引擎换代的三层概念辨析

> 这篇文档专门回答一个高频混淆点：**"MRv2 到底是什么？什么时候加的？"**
>
> 很多人（包括本文档的早期版本）把 vLLM 的引擎换代当成"一次事件"，其实它是**三个不同层级、不同时间**的换代叠加。本文每条结论均经 vLLM 仓库 `git tag` 逐版本核实。
>
> 官方设计文档：[docs.vllm.com.cn/en/latest/design/model_runner_v2](https://docs.vllm.com.cn/en/latest/design/model_runner_v2/)

---

## 一句话结论

vLLM 的引擎换代分**三层**，别混成一个：

| 层级 | 名称 | 首次出现 | 一句话 |
|---|---|---|---|
| 引擎层（旧） | **V0 engine** | 早期 | 老架构 `vllm/engine/llm_engine.py`，v0.11.0 被掏空成垫片 |
| 引擎层（新） | **V1 engine** | **v0.7.0** | 新架构 `vllm/v1/`，v0.8.0 默认开 |
| 执行器层 | **MRv2 / Model Runner V2** | **v0.16.0** | V1 引擎**内部**执行器的第二代重写，落地于新目录 `vllm/v1/worker/gpu/`，与老 `gpu_model_runner.py` 并存 |

**MRv2 ≠ V1 engine。** V1 engine 是 v0.7.0 起的引擎层换代；MRv2 是 v0.16.0 起、在 V1 引擎**内部**对执行器的又一次重写，比 V1 engine 晚约 9 个版本。

---

## 三层分别是什么

### 第 1 层：V0 engine（旧引擎）

- **代码位置**：`vllm/engine/llm_engine.py`、`vllm/core/scheduler.py`、`vllm/core/block_manager.py`（BlockSpaceManager）
- **是什么**：vLLM 最初的推理引擎架构。`LLMEngine` 是它的核心类。
- **命运**：v0.10.0 时 `llm_engine.py` 还有 **2061 行**真实实现；到 **v0.11.0 骤降到 6 行**——内容被掏空，`LLMEngine` 变成重导出 V1 的兼容垫片：

```python
# v0.11.0+ 的 vllm/engine/llm_engine.py 全部内容（6~7 行）
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
LLMEngine = V1LLMEngine  # type: ignore
```

> 即：**V0 的 LLMEngine 在 v0.11.0 就已被 V1 替换**，只是文件名保留以兼容老 import。到 v0.25/v0.26 仍是这个 7 行垫片。

### 第 2 层：V1 engine（新引擎）

- **代码位置**：`vllm/v1/`（独立目录，**不依赖** V0 的 `vllm/core` scheduler/block_manager）
- **首个 commit**：`#9289 [V1] Implement vLLM V1 [1/N]`，**v0.7.0** 合入（v0.6.0 尚无 `vllm/v1/` 目录）
- **核心组件**：
  - `vllm/v1/engine/core.py`（核心循环）
  - `vllm/v1/core/sched/scheduler.py` + `async_scheduler.py`（异步调度）
  - `vllm/v1/core/block_pool.py` + `kv_cache_manager.py` + `kv_cache_coordinator.py`（分页 KV 管理）
  - `vllm/v1/attention/backends/`（20+ attention 后端，registry 注册）
  - `vllm/v1/spec_decode/`（投机解码内建）
- **默认化时间线**（`vllm/envs.py` 逐版本核实）：

| 版本 | `VLLM_USE_V1` 默认值 | 说明 |
|---|---|---|
| v0.7.0 | `False` | V1 引入，默认关 |
| **v0.8.0** | `True` | **默认开** |
| v0.11.0 | `True` | V0 LLMEngine 同期被掏空 |
| v0.12.0 | （变量消失）| `use_v1` 参数废弃（`#28112`），V1 固化为唯一引擎 |

### 第 3 层：MRv2 / Model Runner V2（执行器重写）

- **代码位置**：`vllm/v1/worker/gpu/`（V1 引擎**内部**新目录，核心文件 `gpu/model_runner.py` 的 `class GPUModelRunner`），与老执行器 `vllm/v1/worker/gpu_model_runner.py`（同名类，7902 行）并存。该目录的 `README.md` 自述 "# [Experimental] Model Runner V2"。
- **首个 commit**：`#25266 GPU Model Runner V2`，日期 **2025-11-21**，**v0.16.0** 合入
- **是什么**：V1 引擎虽然 v0.7.0 就有了，但它内部的 `gpu_model_runner`（负责实际把请求跑过模型、管 CUDA Graph、采样等）在 v0.16.0 又被**整体重写了一遍**——这次重写叫 Model Runner V2（MRv2）。
- **为什么重写**（老 Model Runner 的硬伤）：

| 问题 | 老 Model Runner（V1 早期）| MRv2 |
|---|---|---|
| CUDA Graphs | 部分支持，多场景用不了 | 完整支持，推理大加速 |
| 投机解码 | 有限，常与 CUDA Graphs 冲突 | 原生支持，动态投机解码 + 完整 CUDA Graphs 兼容 |
| 多模态 | 后期打补丁 | 原生一等公民 |
| 采样 | 路径分散 | Gumbel sampling 等融合优化 |

- **MRv2 有用户开关 `VLLM_USE_V2_MODEL_RUNNER`**（v0.24.0 起，定义在 `vllm/envs.py`）。但未设该环境变量时，`vllm/config/vllm.py` 的 `use_v2_model_runner` 按**模型与场景条件启用**：PCP>1、dspark 投机解码、DFlash 混合草稿、diffusion 模型等**强制走 V2**；其余仅"默认 V2 模型列表"内 + 有 Triton + 不在支持特性黑名单上的模型才走 V2，否则回退老执行器并打 warning。所以 MRv2 是"渐进式接管"，"MRv2 何时默认"不是个干净断点。源码实证见 [FAQ.md](FAQ.md) Q3。

---

## git 核实的 MRv2 成长曲线

统计每个 tag 上标题含 `MRv2` / `Model Runner V2` 的 commit 累计数：

| 版本 | MRv2 commit 累计数 | 节点 |
|---|---|---|
| v0.15.0 | 52 | 开发中（`#25266` 日期 2025-11-21）|
| **v0.16.0** | 55 | **`#25266` 合入，MRv2 正式落地** |
| v0.17.0 | 93 | 快速增长 |
| **v0.18.0** | 112 | 已成熟（多模态/投机解码/CUDA Graph 持续补齐）|
| v0.19.0 | 131 | |
| v0.20.0 | 144 | |
| v0.22.0 | 173 | |
| v0.24.0 | 200 | |
| v0.25.0 | 220 | 同期删除老 PagedAttention CUDA kernel |
| v0.26.0 | 225 | |

**这条曲线直接证明**：MRv2 从 v0.15/v0.16 起步，到 v0.18 已有 112 个相关 commit、相当活跃。**"MRv2 在 v0.15~0.18 就已存在"是准确的事实**，并非 v0.24/0.25 才出现。

---

## 三层换代的完整时间线（git 核实）

| 版本 | V1 engine（引擎层）| MRv2（执行器层）| PagedAttention kernel | V0 LLMEngine |
|---|---|---|---|---|
| v0.6.0 | ❌ 无 | ❌ 无 | ✅ 在 | ✅ 2061 行真实实现 |
| **v0.7.0** | ✅ 引入（`#9289`），默认关 | — | ✅ 在 | ✅ 在 |
| **v0.8.0** | 默认开（`VLLM_USE_V1=True`）| — | ✅ 在 | ✅ 在 |
| **v0.11.0** | 已默认 | — | ✅ 在 | ⚰️ **掏空成 6 行垫片** |
| v0.12.0 | `use_v1` 废弃（`#28112`），固化 | — | ✅ 在 | 垫片 |
| **v0.16.0** | 已默认 | ✅ **MRv2 引入**（`#25266`）| ✅ 在 | 垫片 |
| v0.18.0 | 已默认 | 112 commit，成熟中 | ✅ 在 | 垫片 |
| v0.24.0 | 已默认 | 200 commit，能力补齐 | ✅ 在 | 垫片 |
| **v0.25.0** | 已默认 | 220 commit | ❌ **删除**（`d715b3aa1`）| 垫片 |
| v0.26.0 | 已默认 | 225 commit | ❌ 已删 | 垫片 |

---

## 为什么会混

1. **命名错位**：代码里新引擎叫 `vllm/v1/`（"第一版重写"），但对外宣传叫 "Model Runner V**2**"（"第二代"）。V1 和 V2 指的是**不同东西**的代数，凑在一起极易混淆。
2. **新闻简化**：外部报道常把"V1 engine + MRv2 + 删 PagedAttention"打包成"v0.25 换底盘"一句话，丢了中间的层级和时间差。
3. **`#25033` 的误导**：commit `[V0 Deprecation] Remove LLMEngine (#25033)` 标题看着像"v0.25 删 LLMEngine"，但它实际日期是 2025-10-03、在较早分支，主线 V0 LLMEngine 早在 **v0.11.0** 就被掏空了。单看 commit 标题会判错版本。

---

## 与 PagedAttention 的关系

- **PagedAttention kernel**（被删的）：v0.25.0 由 `d715b3aa1` 删除，是 2023 年那套原始 CUDA kernel。
- **PagedAttention 思想**（没删的）：分页管理 KV Cache 的理念，由 V1 engine 的 `BlockPool` / `KVCacheManager` / `KVCacheCoordinator` 继承。
- **MRv2 的角色**：MRv2 跑在 V1 engine 上，用的是 V1 的分页 KV 管理，只是执行器（CUDA Graph、采样、投机解码调度）更现代。

### MRv2 的 block table 管理创新

MRv2 在 block tables 管理上有一个关键设计创新：**将 block tables 的权威副本从 CPU 彻底转移到 GPU**，并采用"只传差异"（diff-only）的更新策略。

| 方面 | 老 Model Runner | MRv2 |
|---|---|---|
| **block tables 权威副本** | CPU + GPU 各一份 | **仅 GPU 一份** |
| **更新方式** | 全量重传 | **只传差异**，Triton 内核就地更新 |
| **内存管理** | 请求结束释放 | 请求结束**标记可重用**，不释放 |

具体实现：
- 预分配 `(max_num_reqs, max_num_blocks)` 的 GPU 持久化张量（`StagedWriteTensor`，仅 GPU 一份，无 `self.cpu`）
- 每次请求变动仅暂存新增的 block_id 到 CPU 列表，再通过 `_apply_write_kernel` Triton 内核写入 GPU
- 请求结束后不释放 GPU 内存，仅将 `req_idx` 归还到 `free_indices` 池，供新请求覆盖重用
- 全部 GPU 张量仅在 `shutdown()` 时统一释放

**数据流对比：**
> **老架构**（`CpuGpuBuffer`）：`CPU(numpy 写入完整表)` → `commit_block_table()` → `copy_(cpu → gpu)` 全量拷贝
> **MRv2**（`StagedWriteTensor`）：`CPU(暂存差异列表)` → `async_tensor_h2d(仅差异)` → `Triton 内核就地更新 GPU 大表`

详见 [FAQ.md](FAQ.md) Q6（含完整源码实证）。

### MRv2 的其他关键变化

见 [FAQ.md](FAQ.md) Q7，涵盖：
- **持久化批处理解耦**：消除 `CachedRequestState`，预分配固定行索引
- **异步优先设计**：non-blocking 拷贝、`AsyncOutput` 机制、UVA 并发池
- **代码复杂度降低**：核心文件从 7,916 行降至 1,702 行
- **Triton 原生采样器**：Gumbel sampling 等基于 Triton 实现
- **PagedAttention 演进**：思想保留，实现换代

---

## 给排查者的速查清单

如果你在确认"我的 vLLM 用的是不是 MRv2 / V1"：

- **vLLM ≥ v0.8.0**：默认就是 V1 engine（`vllm/v1/`）。
- **vLLM ≥ v0.16.0**：V1 engine 内部已有 MRv2 执行器（`vllm/v1/worker/gpu/`），逐版本成熟。
- **vLLM ≥ v0.25.0**：老 PagedAttention CUDA kernel 已删。
- 查当前版本：`pip show vllm | grep Version`
- 查是否走 V1 engine：v0.8~v0.11 可看环境变量 `VLLM_USE_V1`；v0.12+ 该变量已废弃，V1 engine 是唯一引擎层。
- 查是否走 MRv2（V2 执行器）：v0.24+ 设 `VLLM_USE_V2_MODEL_RUNNER=1` 强制开、`=0` 强制关；不设则按 `use_v2_model_runner` 的条件判定（见上文）。启动日志若出现 `"Model Runner V2 does not yet support ...; using the V1 model runner instead."` 说明你的配置回退到了老执行器。
