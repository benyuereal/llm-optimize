# vLLM 重大更新盘点（v0.20.0 → v0.25.0）

本目录盘点 vLLM 从 v0.20.0 到 v0.25.0 共 5 个 minor 版本的重大更新，聚焦新特性、性能、核心架构、模型支持、Kernel/Attention/量化/MoE 等方向，已过滤 Bugfix/CI/Docs 等日常维护提交。

> 数据来源：vLLM 官方仓库 git tag 间的 commit（v0.20.0→v0.25.0 共约 2391 个 commit），按 `[Feature]/[Perf]/[Core]/[Model]/[Kernel]/[Attention]/[Quantization]/[MoE]/[SpecDecode]/[MTP]` 等标签筛选，每版约 67-75 条重大更新。

## 版本索引

| 版本 | 文档 | commit 数 | 重大更新数 |
|---|---|---|---|
| v0.21.0 | [v0.21.0.md](v0.21.0.md) | ~395 | 67 |
| v0.22.0 | [v0.22.0.md](v0.22.0.md) | ~474 | 75 |
| v0.23.0 | [v0.23.0.md](v0.23.0.md) | ~427 | 70 |
| v0.24.0 | [v0.24.0.md](v0.24.0.md) | ~576 | 70 |
| v0.25.0 | [v0.25.0.md](v0.25.0.md) | ~576 | 75 |

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
- **MRv2 成为所有稠密模型默认执行路径** — 零配置启用，动态推测解码兼容完整 CUDA Graphs
- **投机解码异构词表通用方案（TLI）** — 打破 Draft/Target 必须同词表的限制
- **Transformers 后端性能追平原生 vLLM** — 450+ HF 架构无需移植即可获 fused kernels + torch.compile + CUDA graphs 加速

## 盘点方法

5 个版本段并行分析，每段由独立 agent 提取 commit、按标签筛选、归类总结。详见各版本文档。

> 注：按 commit 标签（`[Feature]/[Perf]/[Core]/...`）自动筛选会遗漏"无标签但重大"的架构级改动（如 v0.25.0 删除 PagedAttention 的 commit 标题是 `Delete PagedAttention` 无标签）。此类条目已手工补正。
