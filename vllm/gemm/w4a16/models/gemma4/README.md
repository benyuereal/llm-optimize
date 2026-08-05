# gemma-4-31B-it-AWQ-4bit · 调优 config

gemma-4-31B-it-AWQ-4bit 在 Hygon DCU BW10 (gfx936) 上的 aiter w4a16 调优产物。

> 通用的 patch 机制(kernel、vllm 接入、安装脚本)见上级 [`../../README.md`](../../README.md)。
> 本文件只讲 gemma4 专属的内容:模型 shape、调优 config、性能数据。

## 模型信息

- 模型: gemma-4-31B-it-AWQ-4bit(compressed-tensors 格式)
- 量化: AWQ,uint4,**group_size=32**,asymmetric(symmetric=false),zp_dtype=int8
- 结构: 60 层,hidden=5376,GQA 32:16,head_dim=256,sliding_window=1024
- intermediate: 14336

## 性能对比 (端到端, vllm bench serve, 4 prompts × 5120 in / 1024 out)

| 配置 | duration | TPOT | TTFT | 吞吐 (tok/s) | acceptance |
|------|----------|------|------|-------------|-----------|
| baseline (vllm 原生 triton w4a16) | 94.45s | 90.72ms | 1.10s | 43.37 | 96.85% |
| **aiter patch (gs=32, tuned)** | **~77s** | **~74ms** | 1.29s | **~52 tok/s** | **~97%** |

- TPOT: 90.72ms → ~74ms(**-18.5%**)
- 吞吐: 43.37 → ~52 tok/s(**+20%**)
- TTFT 基本持平(1.10s vs 1.29s)
- 精度: 离线 verify `cos_sim = 1.000000`(完全一致);端到端 speculative acceptance
  ~97% vs 96.85%,**无下降**;HumanEval pass@1 = **97.56%**(164 题,evalscope 评测)

> 实测稳态 TPOT 73-75ms(3 轮以上 bench 稳定),提升幅度因系统负载略有波动。
> 最低观测值 68ms(清晨系统空闲时)。

环境: TP=4, attention-backend TRITON_ATTN, kv-cache-dtype fp8, optimization-level 3,
MTP speculative decoding (num_speculative_tokens=3), max-num-batched-tokens 16384。

> 首次 bench 含 warmup(编译/图捕获)偏慢,**对比取稳态第二轮**。

## 调优 config

`configs/awq_w4a16/` 下 10 个 config(均 device_name=BW200, group_size=32),按层权重 shape 命名:

| 文件 (N,K) | 对应层 | 说明 |
|-----------|--------|------|
| N=1344, K=5376 | q_proj | GQA, N = 16 heads × 256 head_dim ÷ 4 (Q 头数减半) |
| N=3584, K=5376 | gate/up_proj | |
| N=5376, K=5376 | o_proj | |
| N=5376, K=14336 | down_proj | 大 K, 用 BK=64 |
| N=10752, K=5376 | merged gate/up | |
| N=4096, K=5376 | | 运行时日志出现 |
| N=5120, K=5376 | | 运行时日志出现 |
| N=5376, K=2048 | | 运行时日志出现 |
| N=5376, K=4096 | | 运行时日志出现 |
| N=1344, K=14336 | | 运行时日志出现 |

config 文件格式:`{M_int: {BLOCK_SIZE_M/N/K, SPLITK, num_warps, NUM_CUS, D_SHAPE, ...}}`。
运行时 aiter 据当前层 `(N,K)` 找文件,再据 `M`(batch)选文件里的具体 config。

### 关键调优参数

- `BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32`(down 大 K 用 BK=64)
- **`SPLITK=2`** 对小 M(M=4,decode 批)是关键 —— BW10 有 80 CUs,小 M 时把 K 维 split
  给多个 CU 做 reduce,否则 CU 利用率太低
- `num_warps=4, NUM_CUS=80`

未调优的 aiter 默认 config 比 vllm 还慢 10-28%;调优后比 vllm 快 1.6-2.33x(kernel 级, M=4)。

## 安装

```bash
cd gemm/w4a16
./patch.sh install gemma4    # 装 patch + gemma4 的 config
./patch.sh status gemma4     # 查看 gemma4 config 是否就位
```
