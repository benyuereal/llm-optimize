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

> 以下代码均为当前仓库真实文件内容，作为上述结论的实证。当前 HEAD：`437e0b7f8`（`v0.26.1rc0` 之后若干 commit，约 v0.26.1 开发中）。

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

**纠正一个常见误解**：MRv2 **不是** v0.7.0 加的（那是 V1 engine），也**不是** v0.24/0.25 才出现。MRv2 从 **v0.15/v0.16 就开始开发**，到 v0.18 已相当活跃——并且是有用户开关的**渐进式接管**，到 v0.25.0 才删除老 PagedAttention kernel 完成换代。

| 版本 | 节点 |
|---|---|
| **v0.15.0** | MRv2 开始开发（首个 commit `#25266` 日期 2025-11-21）|
| **v0.16.0** | `#25266` 合入，MRv2 正式落地 |
| **v0.18.0** | 112 个 MRv2 相关 commit，已成熟 |
| **v0.25.0** | 220 个 commit，删除老 PagedAttention kernel，稠密模型默认走 MRv2 |

### 三层换代的重大时间节点

| 版本 | V1 engine | MRv2 | PagedAttention kernel | V0 LLMEngine |
|---|---|---|---|---|
| **v0.7.0** | ✅ 引入，默认关 | ❌ 无 | ✅ 在 | ✅ 在 |
| **v0.8.0** | ✅ 默认开 | ❌ 无 | ✅ 在 | ✅ 在 |
| **v0.11.0** | 已默认 | ❌ 无 | ✅ 在 | ⚰️ 掏空成垫片 |
| **v0.16.0** | 已默认 | ✅ MRv2 引入 | ✅ 在 | 垫片 |
| **v0.25.0** | 已默认 | ✅ 全面接管 | ❌ 删除 | 垫片 |

> ⚠️ **关于 `#25033`**：commit `[V0 Deprecation] Remove LLMEngine (#25033)` 标题容易让人以为"v0.25 删了 LLMEngine"，但它实际日期是 2025-10-03、位于较早分支，不在 v0.24/v0.25/v0.26 tag 内。主线 V0 `LLMEngine` 早在 **v0.11.0** 就被掏空成重导出 V1 的 6 行垫片（v0.10.0 还是 2061 行真实实现）。判版本不能只看 commit 标题。

> **结论**：MRv2 从 v0.15/v0.16 起步，经约 10 个版本（v0.16 → v0.25）打磨。v0.25.0 删 PagedAttention kernel，是因为这时 MRv2 + V1 engine 终于成熟到可以独挑大梁。**说"MRv2 在 v0.15~0.18 就已存在"是准确的。**

---

## Q5：MRv2 全面接管，对实际使用有什么影响？

**大多数用户无感**——用户层 API（请求格式、模型加载、部署流程）没变，可能完全感知不到底层换了引擎。

**需要关注的**：
- 如果代码/配置引用了 PagedAttention 的**内部 API**，必须移除
- 如果有**自定义 attention 后端**，迁移成本不可忽视
- v0.25.0 同时移除了 6 个旧模型（Baichuan / Aquila / Grok / Tarsier / AyaVision / MusicFlamingo 等），用这些模型的需先确认替代方案（可走 Transformers 后端，v0.25.0 其性能已追平原生）

---

## Q6：MRv2 的 block tables 管理

MRv2 最核心的设计创新，就是把 block tables 的管理**从 CPU 彻底转移到 GPU 上**，并以此为基础实现了"只传差异、不回收"的高效内存管理。

> 官方设计文档：[docs.vllm.com.cn/en/latest/design/model_runner_v2](https://docs.vllm.com.cn/en/latest/design/model_runner_v2/)（第 4 节 StagedWriteTensor）

### 核心设计：GPU 持久化 + "只传差异"

| 方面 | 老 Model Runner（V1 早期） | MRv2 |
|---|---|---|
| **block tables 权威副本** | CPU + GPU 各一份，需同步 | **仅 GPU 一份**，CPU 无完整副本 |
| **更新方式** | 每次请求变动全量重传 | **只传差异**（diff-only）|
| **更新 Kernel** | CPU 全量拷贝到 GPU | Triton 内核**就地更新** GPU 上持久化的大表 |
| **内存管理** | 请求结束释放 GPU 显存 | 请求结束标记"可重用"，**不释放** |

**关键源码实证**（当前仓库 HEAD `437e0b7f8`）：

#### 对比：老架构 vs MRv2 的数据存储方式

**老架构**用 `CpuGpuBuffer`（`vllm/v1/utils.py` 第 110-137 行）——CPU 和 GPU 各一份完整副本，每次更新需全量拷贝：

```python
class CpuGpuBuffer:
    def __init__(self, *size, dtype, device, pin_memory=True):
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)  # CPU 副本
        self.gpu = torch.zeros_like(self.cpu, device=device)                              # GPU 副本
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n=None):
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)  # CPU → GPU 全量拷贝
```

每次请求变动后调用 `commit_block_table()`（`vllm/v1/worker/block_table.py` 第 189-190 行）：`self.block_table.copy_to_gpu(num_reqs)`。

**MRv2** 用 `StagedWriteTensor`（`vllm/v1/worker/gpu/buffer_utils.py` 第 135-137 行）——**仅 GPU 一份**，CPU 只暂存增量差异：

```python
# 没有 self.cpu，没有 self.np，没有完整的 CPU 副本
self.gpu = torch.zeros(size, dtype=dtype, device=device)  # 只有 GPU 张量
```

CPU 暂存差异（`stage_write`，第 155-165 行），然后用 Triton 内核就地写入 GPU（`apply_write`，第 174-201 行）：

```python
def stage_write(self, index, start, x):
    self._staged_write_indices.append(index)   # 行号
    self._staged_write_starts.append(start)     # 起始偏移
    self._staged_write_contents.extend(x)       # 仅新增的 block_id（小列表，非完整表）

def apply_write(self):
    write_contents = async_tensor_h2d(self._staged_write_contents, device=self.device)
    _apply_write_kernel[(n,)](self.gpu, ...)   # Triton 内核就地更新
    self.clear_staged_writes()                  # 清空暂存
```

**数据流变化：**
> **老架构**：`CPU(numpy 写入完整表)` → `commit_block_table()` → `copy_(cpu → gpu)` 全量拷贝
> **MRv2**：`CPU(暂存差异列表)` → `async_tensor_h2d(仅差异)` → `_apply_write_kernel Triton 内核就地更新 GPU 大表`

#### 请求生命周期内"锁定"固定行

`vllm/v1/worker/gpu/states.py` —— 预分配固定大小的索引池，每个请求弹出一个固定索引，请求结束仅归还：

```python
self.free_indices = list(range(max_num_reqs))     # 预分配索引池
req_idx = self.free_indices.pop()                  # 新请求弹出一个固定索引

def remove_request(self, req_id: str) -> int | None:
    req_idx = self.req_id_to_index.pop(req_id, None)
    self.free_indices.append(req_idx)              # 仅归还索引，GPU 张量不动
    return req_idx
```

#### 仅在 shutdown 时释放 GPU 张量

`vllm/v1/worker/gpu/model_runner.py` 第 1633-1653 行：

```python
def shutdown(self) -> None:
    """Release GPU tensors (model weights, KV caches, workspace)"""
    torch.accelerator.synchronize()
    if hasattr(self, "kv_caches"):  self.kv_caches.clear()
    if hasattr(self, "attn_groups"):  self.attn_groups.clear()
    del self.model
    gc.collect()
    torch.accelerator.empty_cache()
```

### "不回收"的真正含义：空间重用，而非内存泄漏

"不回收"指的是**不主动逐请求释放 GPU 显存**，而是通过以下方式复用：

1. **预分配固定大小的池**：`BlockTables` 和 `RequestState` 在初始化时分配 `(max_num_reqs, max_num_blocks)` 的 GPU 张量，整个生命周期不再增减。
2. **请求生命周期内"锁定"位置**：每个请求获得一个**固定行索引**（`req_idx`），在其整个生命周期内独占该行。
3. **"释放" = "清空并重用"**：请求结束后，仅将 `req_idx` 归还到 `free_indices` 池。新请求到来时直接覆盖该行（`append_block_ids` 的 `overwrite=True` 参数）。
4. **仅在关闭时统一清理**：`shutdown()` 方法才真正释放所有 GPU 张量。

这是**以空间换时间**的策略——用预分配 + 行锁定 + 就地更新，避免在高频推理步骤中重复进行昂贵的内存分配与释放，将 block tables 的管理完全在 GPU 内部闭环完成。

---

## Q7：MRv2 整体架构还有哪些关键变化？

MRv2 不只是在 block tables 上做了优化，而是一次从第一性原理出发的**执行引擎全面重写**。以下是 v0.25.0 中 MRv2 已落地的核心变化。

> 官方设计文档：[docs.vllm.com.cn/en/latest/design/model_runner_v2](https://docs.vllm.com.cn/en/latest/design/model_runner_v2/)（含 Persistent Batch、Async-First、StagedWriteTensor、Triton 原生采样器、模块化等完整章节）

### 1️⃣ 持久化批处理（Persistent Batch）—— 解耦持久状态与输入张量

**老问题**：V1 早期 Model Runner 将持久状态（persistent state）与输入张量（input tensors）**强耦合**，请求变化时需进行昂贵的全张量重排序，并维护了冗余的 `CachedRequestState` 备份（代码位置 `vllm/v1/worker/gpu_model_runner.py` 第 223 行 `from vllm.v1.worker.gpu_input_batch import CachedRequestState`）。

**MRv2 方案**：预分配固定大小的张量作为"车位"，每个请求分配一个生命周期内固定的行索引，请求结束仅标记可重用，消除重排序逻辑。

**源码实证：**

```python
# vllm/v1/worker/gpu/states.py 第 28-34 行：预分配索引池
self.free_indices = list(range(max_num_reqs))

# 第 95-96 行：新请求弹出一个固定索引
req_idx = self.free_indices.pop()

# 第 122-128 行：请求结束仅归还索引
def remove_request(self, req_id: str) -> int | None:
    req_idx = self.req_id_to_index.pop(req_id, None)
    self.free_indices.append(req_idx)  # 仅归还索引，GPU 张量不动
    return req_idx
```

`CachedRequestState` 在 MRv2 目录 `vllm/v1/worker/gpu/` 中**已不存在**，MRv2 官方设计文档 `docs/design/model_runner_v2.md` 第 23-39 行明确说明此举"removes the need for CachedRequestState"。

### 2️⃣ GPU-native block tables 管理（已在 Q6 详述）

见 Q6。

### 3️⃣ 异步优先的设计

**老问题**：旧版设计未考虑异步调度，是后期打补丁实现的。

**MRv2 方案**：提供了异步执行的基础设施，包括：

- **`async_copy_to_gpu()`**（`buffer_utils.py` 第 26-41 行）：non-blocking 拷贝，`out.copy_(pinned, non_blocking=True)`
- **`AsyncOutput`**（`model_runner.py` 第 1508-1516 行）：将 D2H 拷贝与后续规约算子重叠，注释写明"Start async output copy here so that it can overlap with speculator proposal"
- **`set_default_max_concurrency()`**（`model_runner.py` 第 167 行）：UVA buffer pool 大小设为并发批次数，支撑多步流水

> ⚠️ 注意：模型 runner 本身不直接展示"CPU 准备第 N+1 步，同时 GPU 执行第 N 步"的完整流水线。这种重叠主要发生在引擎/调度器层，模型 runner 提供了异步安全的构建块（separate streams、non-blocking copies、UVA pools）。

### 4️⃣ 代码复杂度降低

| 文件 | 行数 |
|---|---|
| 老 Model Runner `vllm/v1/worker/gpu_model_runner.py` | **7,916 行** |
| MRv2 `vllm/v1/worker/gpu/model_runner.py` | **1,702 行** |

核心文件减少 **78%**。但注意 MRv2 将功能拆到了 `gpu/` 子目录下的多个文件中（`states.py`、`buffer_utils.py`、`block_table.py`、`sample/`、`model_states/` 等），总代码量可能相当。其核心优势是**模块化**——新增功能不用往一个巨大文件里塞。

### 5️⃣ Triton 原生采样器

MRv2 重新实现了采样器，主要基于 Triton kernel：

- `vllm/v1/worker/gpu/sample/gumbel.py`：完整 Triton 实现的 Gumbel sampling（`_gumbel_sample_kernel`、`_temperature_kernel`）
- 采样器 `sampler.py` 使用 `gumbel_sample` 和 `flashinfer_sample` 进行实际采样

### 6️⃣ PagedAttention 的演进（而非简单的"移除"）

PagedAttention 的变化可以用**"思想保留，实现换代"**来概括：

| 方面 | 被移除的 | 保留的 |
|---|---|---|
| **CUDA kernel** | `paged_attention_v1.cu` / `v2.cu` / `attention_kernels.cuh`（`d715b3aa1` 删除） | — |
| **分页管理 KV Cache 思想** | — | 由 V1 engine 的 `BlockPool` / `KVCacheManager` 继承 |
| **block tables 数据结构** | — | 仍为核心，MRv2 以更高效的方式（GPU 原生）管理 |
| **PagedAttention 类** | — | 仍保留在 `vllm/v1/attention/ops/paged_attn.py`，用于 `split_kv_cache` 等工具方法 |
| **底层注意力 kernel** | 旧 CUDA 实现 | 由各个 attention backend（FlashAttention、FlashInfer、Triton 等）替代 |

### 7️⃣ 关于性能数据

> 一些文章提到"单张 GB200 上 Qwen3-0.6B 吞吐量提升约 56%"等具体数字。**经核查当前代码库（commit `437e0b7f8`），该数字未出现在任何文档、commit message 或 PR 描述中**。MRv2 官方设计文档 `docs/design/model_runner_v2.md` 仅做了定性描述："we believe it is a substantial improvement over V1"。此类性能数据可能存在于外部博客文章或 PR 讨论中，但当前代码库中无可验证来源。
