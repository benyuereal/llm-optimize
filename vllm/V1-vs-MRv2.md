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

| 版本 | 节点 |
|---|---|
| **v0.15.0** | MRv2 开始开发（`#25266` 日期 2025-11-21）|
| **v0.16.0** | `#25266` 合入，MRv2 正式落地 |
| **v0.18.0** | 112 个相关 commit，已成熟 |
| **v0.25.0** | 220 个 commit，删除老 PagedAttention kernel |

**结论**：MRv2 从 v0.15/v0.16 起步，到 v0.18 已有 112 个相关 commit，**并非 v0.24/0.25 才出现**。

---

## 三层换代的重大时间节点

| 版本 | V1 engine | MRv2 | PagedAttention kernel | V0 LLMEngine |
|---|---|---|---|---|
| **v0.7.0** | ✅ 引入，默认关 | ❌ 无 | ✅ 在 | ✅ 在 |
| **v0.8.0** | ✅ 默认开 | ❌ 无 | ✅ 在 | ✅ 在 |
| **v0.11.0** | 已默认 | ❌ 无 | ✅ 在 | ⚰️ 掏空成垫片 |
| **v0.16.0** | 已默认 | ✅ MRv2 引入 | ✅ 在 | 垫片 |
| **v0.25.0** | 已默认 | ✅ 全面接管 | ❌ 删除 | 垫片 |

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

### MRv2 源码逐级拆解：它是怎么工作的

以下从 `execute_model` 入口开始，逐层拆解 MRv2 的执行流程。

#### 入口：`execute_model`（`vllm/v1/worker/gpu/model_runner.py`）

```python
# 第 1190 行
def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, ...):
    if not dummy_run:
        # 1. 更新 PP 解码请求（流水线并行）
        self.update_pp_decode_requests()
        # 2. 清理已完成/被抢占的请求
        self.finish_requests(scheduler_output)
        # 3. 释放编码器缓存
        self.free_states(scheduler_output)
        # 4. 添加新请求
        self.add_requests(scheduler_output)
        # 5. 更新已有请求（新增 block、更新 num_computed_tokens）
        self.update_requests(scheduler_output)
        # 6. 将暂存的差异写到 GPU
        self.block_tables.apply_staged_writes()

    # 7. 获取批处理描述符，同步 DP 间状态
    batch_desc = dispatch_cg_and_sync_dp(...)
    # 8. 准备输入张量
    input_batch = self.prepare_inputs(scheduler_output, batch_desc)
    # 9. 准备 attention 元数据（block tables + slot mappings）
    block_tables, slot_mappings = self.prepare_attn(input_batch)
    # 10. 模型前向 + 采样
    hidden_states = self.model.forward(input_ids, positions, block_tables, ...)
    sampler_output = self.sample(hidden_states, input_batch)
    return sampler_output
```

#### 请求生命周期：`add_requests` 和 `finish_requests`

**添加请求**（第 822 行）：

```python
def add_requests(self, scheduler_output):
    for new_req_data in scheduler_output.scheduled_new_reqs:
        # 为请求分配一个固定行索引 req_idx
        self.req_states.add_request(req_id=req_id, ...)
        req_index = self.req_states.req_id_to_index[req_id]

        # 将 block_id 写入 GPU 张量（暂存到 CPU 列表，等 apply_staged_writes 再写 GPU）
        self.block_tables.append_block_ids(
            req_index, new_req_data.block_ids, overwrite=True)
```

**`RequestState.add_request`**（`states.py` 第 87 行）：

```python
def add_request(self, req_id, prompt_len, all_token_ids, ...):
    # 从预分配的 free_indices 池弹出一个索引
    req_idx = self.free_indices.pop()
    self.req_id_to_index[req_id] = req_idx
    self.index_to_req_id[req_idx] = req_id

    # 用 stage_write 暂存数据到 CPU 列表
    self.num_tokens.stage_write_elem(req_idx, prompt_len)
    self.total_len.stage_write_elem(req_idx, max_tokens)
    self.all_token_ids.np[req_idx] = all_token_ids
```

**移除请求**（`states.py` 第 122 行）：

```python
def remove_request(self, req_id):
    req_idx = self.req_id_to_index.pop(req_id, None)
    self.index_to_req_id.pop(req_idx, None)
    self.free_indices.append(req_idx)  # 仅归还索引，GPU 张量不动
    return req_idx
```

#### StagedWriteTensor：只传差异的核心机制

`block_table.py` 第 107-132 行 —— 暂存 + 批量写入：

```python
def append_block_ids(self, req_index, block_ids, overwrite=False):
    # 计算起始位置
    start = self.num_blocks.np[i, req_index] if not overwrite else 0
    # 暂存到 CPU 列表（只存新增的 block_id，不存完整表）
    self.block_tables[i].stage_write(req_index, start, block_ids)

def apply_staged_writes(self):
    if self.num_kv_cache_groups == 1:
        self.block_tables[0].apply_write()     # 单组：直接写
    else:
        self.fused_writer.apply(                # 多组：融合到一次 kernel 启动
            self.block_tables, self.block_table_ptrs, self.block_table_strides)
```

`buffer_utils.py` 第 155-201 行 —— `StagedWriteTensor` 的核心：

```python
def stage_write(self, index, start, x):
    # CPU 列表暂存差异（只有行号、偏移、内容）
    self._staged_write_indices.append(index)
    self._staged_write_starts.append(start)
    self._staged_write_contents.extend(x)

def apply_write(self):
    # 仅将差异拷贝到 GPU
    write_contents = async_tensor_h2d(self._staged_write_contents, device=self.device)
    # Triton 内核就地更新 GPU 大表
    _apply_write_kernel[(n,)](self.gpu, self.gpu.stride(0),
                               indices_uva, starts_uva, write_contents, ...)
    self.clear_staged_writes()  # 清空暂存
```

#### 输入准备：`prepare_inputs`（第 913 行）

```python
def prepare_inputs(self, scheduler_output, batch_desc):
    # 从 persistent state 中 gather 出 step 的输入张量
    req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
    # req_id → req_index 映射
    idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

    # 用 GPU gather 代替 CPU 拼装
    self.input_buffers.input_ids.copy_(token_ids, non_blocking=True)
    self.input_buffers.positions.copy_(positions, non_blocking=True)
    # ... 更多输入准备
```

#### Attention 准备：`prepare_attn`（第 1094 行）

```python
def prepare_attn(self, input_batch):
    # GPU 原生 gather block tables：从 GPU 持久化张量收集到输入张量
    block_tables = self.block_tables.gather_block_tables(
        input_batch.idx_mapping, num_reqs_padded=...)

    # GPU 原生计算 slot mappings
    slot_mappings = self.block_tables.compute_slot_mappings(
        input_batch.idx_mapping, input_batch.query_start_loc,
        input_batch.positions, num_tokens_padded=...)

    return block_tables, slot_mappings
```

#### Triton 原生采样器：`sample`（第 1124 行）

```python
def sample(self, hidden_states, input_batch, grammar_output=None):
    logits = self.model.compute_logits(sample_hidden_states)
    # 无 draft token → 直接采样
    sampler_output = self.sampler(logits, input_batch)
    # 有 draft token → rejection sampling
    sampler_output = self.rejection_sampler(logits, input_batch, ...)
    return sampler_output
```

采样器底层使用 Triton kernel（`sample/gumbel.py`），避免显式的 softmax 全量计算。

#### 关闭时统一释放：`shutdown`（第 1633 行）

```python
def shutdown(self):
    torch.accelerator.synchronize()
    self.kv_caches.clear()
    self.attn_groups.clear()
    del self.model
    gc.collect()
    torch.accelerator.empty_cache()
```

### 执行流程总结

```
每个推理 step，MRv2 的执行流水线：

  1. finish_requests()    → 归还 req_idx 到 free_indices（不清 GPU 内存）
  2. add_requests()       → 从 free_indices 弹出 req_idx，stage_write 暂存差异
  3. update_requests()    → stage_write 新增 block_id，更新 num_computed_tokens
  4. apply_staged_writes()→ Triton 内核将差异写入 GPU 持久化张量
  5. prepare_inputs()     → 从 persistent state gather 出输入张量
  6. prepare_attn()       → GPU 原生 gather block tables + compute slot mappings
  7. model.forward()      → 模型前向计算
  8. sample()             → Triton 原生采样器
  9. shutdown()           → 仅在进程结束时释放所有 GPU 张量
```

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
