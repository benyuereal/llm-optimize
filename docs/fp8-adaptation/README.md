# FP8 适配文档（Hygon DCU / BW1100, gfx938）

Qwen 系列模型在 Hygon DCU 上的 **FP8 rowwise (W8A8) 量化 + vLLM decode 性能优化** 文档汇总。
源目录：`/workspace/llm-adaptation/`（本目录为归档副本）。

## 目录

### [qwen3-32b-fp8-rowwise/](qwen3-32b-fp8-rowwise/)

Qwen3-32B（dense, 64 层, hidden 5120）FP8 量化与性能优化主文档。

| 文件 | 内容 |
|---|---|
| [OPTIMIZE.md](qwen3-32b-fp8-rowwise/OPTIMIZE.md) | 主报告：量化配方、W8A16 死路、激活量化根因与修复、DP4 vs TP4、fused topk+topp、DP1 batch 对比、算子流程图、GEMM 带宽利用率 vs M 与 split-K 验证（§6.9 为最新实测） |

### [qwen3.8-27b-fp8-rowwise/](qwen3.8-27b-fp8-rowwise/)

Qwen3.8-27B（qwen3_5 架构, 64 层, GQA 24:4, head_dim 256）适配实例，量化配方与 32B 完全一致。

| 文件 | 内容 |
|---|---|
| [README.md](qwen3.8-27b-fp8-rowwise/README.md) | 量化方案 + 启动方式 + 性能数据 |
| [quantize_qwen3_8_27b_fp8_rowwise.py](qwen3.8-27b-fp8-rowwise/quantize_qwen3_8_27b_fp8_rowwise.py) | llm-compressor `model_free_ptq` 量化脚本 |
| [start.sh](qwen3.8-27b-fp8-rowwise/start.sh) / [start_dp1.sh](qwen3.8-27b-fp8-rowwise/start_dp1.sh) | vLLM 启动脚本（TP / DP1） |

## 关键结论速览（Qwen3-32B, 2026-09-09 实测, card 4, 峰值 1.8 TB/s）

- 生产 GEMM 路径 = `rocblas_gemm_ex`（OCP e4m3），经 `lightop.hipblaslt_w8a8_channelwise_gemm`。
- FP8 算力峰值 **447 TFLOPS**，脊点 AI = **248 FLOP/byte**。
- 带宽利用率随 M 下降：M≤16 ~75-83%（BW-bound），M=128 46.5%（混合区），M=512 23%（compute-bound）。
- **M=17~64 平台卡在 68%**（M=16 能到 83%，同权重量慢 22%）——kernel tile 配置问题，非算力受限；tile 切换台阶在 M=17 / M=80。
- 库内 64 个 algo（含 split-K）枚举无改善（M=32/64 1.00x）→ 已提算子优化单。
- 复现脚本：`/workspace/_src/hipblaslt_splitk_probe/`（`bw_real.py` / `ridge_probe.py` / `ab_m64.py`）。
