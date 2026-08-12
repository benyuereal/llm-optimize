# vLLM 重大更新盘点（v0.19.0 → v0.26.0）

本目录盘点 vLLM 从 v0.19.0 到 v0.26.0 共 8 个 minor 版本（v0.20.0 未单独成文）的重大更新，聚焦新特性、性能、核心架构、模型支持、Kernel/Attention/量化/MoE 等方向，已过滤 Bugfix/CI/Docs 等日常维护提交。所有版本合订在 [v0.19.0.md](v0.19.0.md)（含目录跳转），每版含"本版导读"（通俗定调）+ 分类详情（带 commit 号核实）。

> 数据来源：vLLM 官方仓库 git tag 间的 commit，按 `[Feature]/[Perf]/[Core]/[Model]/[Kernel]/[Attention]/[Quantization]/[MoE]/[SpecDecode]/[MTP]` 等标签筛选，每版约 67-95 条重大更新。
>
> 🔍 **源码实证**：FAQ.md Q3 附当前仓库 HEAD（`90245f419`，约 v0.26.1 开发中）的真实代码片段，实证 V0 垫片（`vllm/engine/llm_engine.py` 7 行重导出 V1）、V1 独立目录（`vllm/v1/`）、MRv2 新目录（`vllm/v1/worker/gpu/`，README 自述 "Model Runner V2"）、MRv2 开关（`VLLM_USE_V2_MODEL_RUNNER`）与启用判定逻辑。

## 版本索引

| 版本 | 文档 | 发布日期 | commit 数 | 重大更新数 |
|---|---|---|---|---|
| v0.19.0 | [v0.19.0.md](v0.19.0.md) | 2026-04-02 | ~466 | 60 |
| v0.20.0 | — | 2026-04-27 | ~765 | （未单独成文，见 v0.21.0 起的相对盘点）|
| v0.21.0 | [v0.19.0.md](v0.19.0.md) | — | ~395 | 67 |
| v0.22.0 | [v0.19.0.md](v0.19.0.md) | — | ~474 | 75 |
| v0.23.0 | [v0.19.0.md](v0.19.0.md) | — | ~427 | 70 |
| v0.24.0 | [v0.19.0.md](v0.19.0.md) | — | ~576 | 70 |
| v0.25.0 | [v0.19.0.md](v0.19.0.md) | — | ~576 | 75 |
| v0.26.0 | [v0.19.0.md](v0.19.0.md) | 2026-07-27 | ~429 | 95 |

> v0.21.0 ~ v0.26.0 已合并入 [v0.19.0.md](v0.19.0.md) 合订本（顶部目录可跳转各版）。

## 架构演进 Q&A

围绕 PagedAttention 退役与 MRv2 接管的几个核心问题，详见 [FAQ.md](FAQ.md)：

- **Q1**：v0.25.0 到底删没删 PagedAttention？（删了；删的是原始 CUDA kernel，源码逐版本查证）
- **Q2**：删的只是 kernel，PagedAttention 的"思想"删了吗？（没删，分页管理 KV Cache 的思想由 MRv2 继承）
- **Q3**：MRv2 是什么？全称？跟 PagedAttention 什么关系？（Model Runner V2，V1 引擎内部执行器的第二代重写；注意 V1 engine ≠ MRv2，二者层级和时间都不同）
- **Q4**：MRv2 什么时候加的？（V1 engine v0.7.0 引入、v0.8.0 默认开；V0 LLMEngine v0.11.0 被掏空成垫片；MRv2 从 v0.15/v0.16 起开发，v0.18 已成熟；有开关 `VLLM_USE_V2_MODEL_RUNNER`，按模型/场景条件启用而非全局默认。三层概念与完整时间线见 [V1-vs-MRv2.md](V1-vs-MRv2.md)）
- **Q5**：MRv2 全面接管对使用有什么影响？（多数用户无感；引用 PagedAttention 内部 API / 自定义 attention 后端需迁移）
- **Q6**：MRv2 的 block tables 管理（block tables 从 CPU 移到 GPU，采用"只传差异"策略，预分配 + 行锁定 + 关闭时统一释放）
- **Q7**：MRv2 整体架构还有哪些关键变化？（持久化批处理解耦、异步优先设计、Triton 原生采样器、代码复杂度降低、PagedAttention "思想保留实现换代"）

## 各版头条亮点

> 一条贯穿的主线：vLLM 这几个版本在做一件事——**把投机解码（MTP）从"能用"打磨到"好用"，同时把执行引擎从老 PagedAttention 架构换到 MRv2**。0.19 零气泡调度+MRV2 成熟打底、0.20 大换血（MRV2 重构 + DSV4 原生 + FA4）、0.21 投机解码起跑、0.22-0.23 成熟、0.24 全面铺开、0.25 完成换代、0.26 在新底盘上深挖性能。

### v0.19.0 — 零气泡调度 + MRV2 成熟，新硬件 B300 就位
- **零气泡异步调度 + 推测解码**（#32951）：异步执行与投机解码深度融合，消除计算气泡，高并发吞吐显著提升——本版最核心的性能突破
- **Gemma 4 全支持**（#38826）：MoE / 多模态 / 推理 / 工具调用，开箱即用
- **通用 CPU KV 缓存卸载**（#37160/#37874）：可插拔 CachePolicy，内存受限下能跑更大模型
- **MRV2 成熟**：流水线并行分段 CUDA 图、推测解码拒绝采样器、多模态嵌入推测解码、流式输入、EPLB
- **B300/GB300（SM 10.3）支持**（#37756），默认启用调优的 AllReduce 融合通信器

### v0.21.0 — 投机解码起跑，新硬件后端就位
- **MTP 投机解码首次完整落地**（Gemma4），vLLM 正式拥抱"一次猜多个 token"的加速路线
- Blackwell 新硬件拿到专属 MLA 注意力后端（TOKENSPEED_MLA），P/D 分离架构支持 KV 双向传输
- NVFP4 量化从 kernel 跑通到推理，显存省一截

### v0.22.0 — 投机解码开始"不挑场景"
- 投机解码与 thinking 模式兼容了——以前开 MTP 和结构化输出会打架，这版开始和解
- draft 模型能用独立 attention 后端，DSV4 的 MTP 跨硬件铺开
- 一句话：MTP 从"demo 能跑"走向"各种场景都能跑"

### v0.23.0 — 架构重构打底
- KV-Cache 来了一次大重构（可插拔 KVCacheSpec），为后面多模型/多后端铺路
- MoE 路由全面迁到 oracle 架构，DSA MTP 引入 index share
- 这版没多少"看得见的提速"，但为 0.24/0.25 的全面成熟埋了地基

### v0.24.0 — 投机解码全面成熟 + GLM-5 全栈适配
- **投机解码集体到位**：MTP / EAGLE / DFlash / 动态投机解码四线并进，还能跟 thinking budget 叠加
- **GLM-5 系列拿到原生高性能支持**：从 MLA prefill kernel 到 MoE router GEMM 到 GLM5.1/5.2 流式 tool parser，全栈适配
- Streaming Parser Engine 上线，统一各模型的工具调用/reasoning 解析
- DeepSeek-V4 持续被喂优化（TTFT -2~4%、吞吐 +4%）
- 不再内部设 `CUDA_VISIBLE_DEVICES`，改 `device_ids` 参数——多卡部署要注意

### v0.25.0 ⭐ 架构级里程碑 — 换底盘
- **PagedAttention 正式退役**：删掉 2023 年让 vLLM 一战成名的原始 CUDA kernel（commit `d715b3aa1`，约 1472 行），MRv2 全面接管。思想（分页管 KV Cache）没删，删的是那套老实现
- **MRv2 成为所有稠密模型唯一默认路径**：V1 engine 自 v0.7.0 起步、v0.8.0 默认开；MRv2（V1 内部执行器重写）自 v0.16.0 起步、v0.18 已成熟，到这版正式独挑大梁。动态投机解码终于兼容完整 CUDA Graphs（以前二选一）。三层概念辨析见 [V1-vs-MRv2.md](V1-vs-MRv2.md)
- 投机解码打破"草稿模型和主模型必须同词表"的限制（TLI 异构词表方案）
- Transformers 后端性能追平原生 vLLM——450+ HF 模型无需移植也能享受 fused kernels + CUDA graphs
- 详见 [FAQ.md](FAQ.md)

### v0.26.0 — 在新底盘上深挖性能
- **DeepSeek-V4 端到端延迟再压榨**：专用路由内核 TPOT -2.94%、fused_topk_bias 提速 1.5~2x、去冗余再降 1.8%；DSv4 DSpark 投机解码登陆 AMD 与 XPU
- **Inkling 模型家族完整合入**，自带 MTP=1 投机解码——新模型从设计就内建投机解码
- 多模型（Olmo/MistralLarge3/HunyuanVL）迁到 Transformers 建模后端，配套 Transformers 5.13.0
- 引擎核心补硬骨头：fp32 生成头、attention 后端按 KV-cache 组选择、KV 二级存储成熟
- 安全加固（移除 diskcache 消除 pickle 风险），移除 TeleChat/Persimmon/Fuyu 等旧模型

## 盘点方法

各版本段由独立 agent 提取 commit、按标签筛选、归类总结。详见各版本文档。

> 注：按 commit 标签（`[Feature]/[Perf]/[Core]/...`）自动筛选会遗漏"无标签但重大"的架构级改动（如 v0.25.0 删除 PagedAttention 的 commit 标题是 `Delete PagedAttention` 无标签）。此类条目已手工补正，并对照官方发布说明逐项核实。
