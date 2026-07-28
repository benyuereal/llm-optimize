# vLLM 重大更新盘点（v0.20.0 → v0.26.0）

本目录盘点 vLLM 从 v0.20.0 到 v0.26.0 共 6 个 minor 版本的重大更新，聚焦新特性、性能、核心架构、模型支持、Kernel/Attention/量化/MoE 等方向，已过滤 Bugfix/CI/Docs 等日常维护提交。

> 数据来源：vLLM 官方仓库 git tag 间的 commit（v0.20.0→v0.26.0 共约 2820 个 commit），按 `[Feature]/[Perf]/[Core]/[Model]/[Kernel]/[Attention]/[Quantization]/[MoE]/[SpecDecode]/[MTP]` 等标签筛选，每版约 67-95 条重大更新。

## 版本索引

| 版本 | 文档 | commit 数 | 重大更新数 |
|---|---|---|---|
| v0.21.0 | [v0.21.0.md](v0.21.0.md) | ~395 | 67 |
| v0.22.0 | [v0.22.0.md](v0.22.0.md) | ~474 | 75 |
| v0.23.0 | [v0.23.0.md](v0.23.0.md) | ~427 | 70 |
| v0.24.0 | [v0.24.0.md](v0.24.0.md) | ~576 | 70 |
| v0.25.0 | [v0.25.0.md](v0.25.0.md) | ~576 | 75 |
| v0.26.0 | [v0.26.0.md](v0.26.0.md) | ~429 | 95 |

## 架构演进 Q&A

围绕 PagedAttention 退役与 MRv2 接管的几个核心问题，详见 [FAQ.md](FAQ.md)：

- **Q1**：v0.25.0 到底删没删 PagedAttention？（删了；删的是原始 CUDA kernel，源码逐版本查证）
- **Q2**：删的只是 kernel，PagedAttention 的"思想"删了吗？（没删，分页管理 KV Cache 的思想由 MRv2 继承）
- **Q3**：MRv2 是什么？全称？跟 PagedAttention 什么关系？（Model Runner V2，新一代推理执行引擎，对应代码里的 V1 engine）
- **Q4**：MRv2 什么时候加的？（v0.7.0 引入，v0.24.0 成默认，v0.25.0 唯一路径）
- **Q5**：MRv2 全面接管对使用有什么影响？（多数用户无感；引用 PagedAttention 内部 API / 自定义 attention 后端需迁移）

## 各版头条亮点

### v0.21.0（相对 0.20.0）
- **Gemma4 MTP 投机解码** — MTP（Multi-Token Prediction）投机解码首次完整落地
- **TOKENSPEED_MLA Blackwell 后端** — MLA 注意力在 Blackwell 硬件的高性能后端
- **P/D 双向 KV 传输** — Prefill/Decode 分离架构的 KV cache 双向传输
- **NVFP4 量化全链路** — NVFP4 量化从 kernel 到推理的完整支持

### v0.22.0（相对 0.21.0）
- **Gemma4 MTP 完整实现** — MTP 投机解码成熟化
- **带 thinking budget 的投机解码** — 投机解码与 thinking 模式兼容（关键：之前 MTP 与结构化输出冲突，这里开始改善）
- **drafter 独立 attention 后端** — 投机解码的 draft 模型可用独立 attention 后端
- **MRV2 投机解码成熟化** + DSV4 MTP 跨硬件铺开

### v0.23.0（相对 0.22.0）
- **DSA MTP index share** — MTP 的 index 共享机制
- **KV-Cache Layout 重构 + 可插拔 KVCacheSpec** — KV cache 架构大重构，为多模型/多后端打基础
- **MoE 全面迁移到 oracle 架构** — MoE 路由的 oracle 架构
- **DSV4 栈打磨 + TOKENSPEED_MLA/TRTLLM kernel**

### v0.24.0（相对 0.23.0）
- **投机解码全面成熟** — MTP/EAGLE/DFlash/Dynamic SD + thinking budget 叠加，稳定性与兼容性大幅提升
- **MLA 注意力后端重构 + GLM-5 全栈适配** — GLM 系列模型原生高性能支持
- **P/D 分离 + KV 传输生态成熟** — Mooncake/DeepEP v2/NIXL/HMA/cross-layer
- **NVFP4/MXFP4 量化全链路深化**

### v0.25.0（相对 0.24.0）⭐ 架构级里程碑
- **PagedAttention 正式退役** — 删除 2023 年让 vLLM 一战成名的原始 PagedAttention CUDA kernel（commit `d715b3aa1`，删除约 1472 行：`paged_attention_v1.cu`/`v2.cu` + `attention_kernels.cuh`），由 MRv2 全面接管
- **MRv2 成为所有稠密模型默认执行路径** — 零配置启用，动态推测解码兼容完整 CUDA Graphs。MRv2（Model Runner V2，对应代码里的 V1 engine）从 v0.7.0 引入，经约 18 个版本打磨，v0.24.0 成默认、v0.25.0 成唯一路径。详见 [FAQ.md](FAQ.md)
- **投机解码异构词表通用方案（TLI）** — 打破 Draft/Target 必须同词表的限制
- **Transformers 后端性能追平原生 vLLM** — 450+ HF 架构无需移植即可获 fused kernels + torch.compile + CUDA graphs 加速

### v0.26.0（相对 0.25.0）
- **DeepSeek-V4 端到端延迟持续打磨** — 专用路由内核 TPOT -2.94%、fused_topk_bias 路由 kernel 提速 1.5~2x、去冗余 repeat/copy TPOT -1.8%；DSv4 DSpark 投机解码登陆 AMD 与 XPU
- **Inkling 模型家族完整合入** — 含 MTP=1 投机解码、Piecewise CUDA Graph、Hopper FA4 相对注意力、LoRA、NVFP4 量化
- **多模型迁移到 Transformers 建模后端** — Olmo/Olmo2、MistralLarge3、HunyuanVL，配套 Transformers 5.13.0
- **引擎核心增强** — fp32 生成头（head_dtype，扩展到 LoRA）、attention 后端按 KV-cache 组选择、KV 二级存储成熟
- **安全加固** — 移除 diskcache 消除 pickle 反序列化风险、并发稀疏不变量竞态修复；移除 TeleChat/Persimmon/Fuyu 模型

## 盘点方法

5 个版本段并行分析，每段由独立 agent 提取 commit、按标签筛选、归类总结。详见各版本文档。

> 注：按 commit 标签（`[Feature]/[Perf]/[Core]/...`）自动筛选会遗漏"无标签但重大"的架构级改动（如 v0.25.0 删除 PagedAttention 的 commit 标题是 `Delete PagedAttention` 无标签）。此类条目已手工补正。
