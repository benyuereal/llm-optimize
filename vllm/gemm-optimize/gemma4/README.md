# gemma-4-31B-it-AWQ-4bit · aiter w4a16 GEMM 加速 patch

Hygon DCU BW10 (gfx936) 上,用 **aiter triton w4a16 kernel** 替换 vllm 自带 triton w4a16 kernel,
针对 AWQ **group_size=32** 调优,端到端 TPOT 降低 ~25%,精度无损。

## 性能对比 (端到端, vllm bench serve, 4 prompts × 5120 in / 1024 out)

| 配置 | duration | TPOT | TTFT | acceptance |
|------|----------|------|------|-----------|
| baseline (vllm 原生 triton w4a16) | 94.45s | 90.72ms | 1.10s | 97.48% |
| **aiter patch (gs=32, tuned)** | **71.03s** | **68.11ms** | 1.07s | 97.19% |

- duration: 94.45s → 71.03s(**-24.8%**)
- TPOT: 90.72ms → 68.11ms(**-24.9%**)
- TTFT 基本持平
- 精度: 离线 verify `cos_sim = 1.000000`;端到端 speculative acceptance 97.48% vs 97.19%,无下降

环境: TP=4, attention-backend TRITON_ATTN, kv-cache-dtype fp8, optimization-level 3,
MTP speculative decoding (num_speculative_tokens=3), max-num-batched-tokens 16384。

---

## 文件清单

```
gemm-optimize/gemma4/
├── README.md                         # 本文件
├── start.sh                          # vllm serve 启动脚本 (含 patch 开关)
├── triton_w4a16.py.original          # vllm 原始 triton_w4a16.py (回退用)
├── triton_w4a16.py.aiter_patch       # 打了 aiter patch 的 triton_w4a16.py
├── aiter_gemm_a16w4.py               # aiter triton w4a16 kernel (已修 triton3.5 兼容)
├── configs/awq_w4a16/                # 调优后的 aiter config (gs=32, BW200, 10 个 shape)
└── tests/                            # 验证 / 性能 / 调优脚本
    ├── verify_aiter_v2.py            # 精度验证 (aiter vs vllm, cos_sim)
    ├── verify_aiter_self.py          # aiter 自洽性验证
    ├── bench_aiter_vs_vllm.py        # aiter vs vllm kernel 级性能对比
    ├── tune_aiter_config.py          # config 参数扫描
    ├── gen_aiter_configs.py          # 生成 config json
    └── test_custom_op.py             # custom_op + torch.compile 兼容性测试
```

---

## 如何切换 (对比 baseline vs patch)

patch 的启停由环境变量 `VLLM_AITER_W4A16_PATCH` 控制,**不需要换文件**:
- `=1` 启用 aiter patch(用 aiter kernel)
- `=0` 走 vllm 原生 triton w4a16(baseline)

`start.sh` 第 38 行:
```bash
export VLLM_AITER_W4A16_PATCH=1   # 1=aiter, 0=baseline
```

> 当前提交里 `start.sh` 写的是 `=0`(测 baseline 时改的)。要跑 aiter 版请改回 `=1`。

如果想**物理替换文件**对比(而非环境变量),把对应版本拷到 vllm 安装目录:
```bash
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/kernels/linear/mixed_precision
# baseline
cp triton_w4a16.py.original  $VLLM_DIR/triton_w4a16.py
# aiter patch
cp triton_w4a16.py.aiter_patch $VLLM_DIR/triton_w4a16.py
```

---

## 部署步骤 (从零启用 aiter patch)

### 1. 替换 vllm 的 triton_w4a16.py
```bash
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/kernels/linear/mixed_precision
cp triton_w4a16.py.aiter_patch $VLLM_DIR/triton_w4a16.py
```

### 2. 放置 aiter kernel 和 config
```bash
# aiter kernel (已修 triton 3.5 兼容: @triton.jit, 去掉 key=)
cp aiter_gemm_a16w4.py /public/home/weishb/aiter/aiter/ops/triton/gemm_a16w4.py

# 调优 config (gs=32, BW200)
cp configs/awq_w4a16/*.json \
   /public/home/weishb/aiter/aiter/ops/triton/configs/gemm/awq_w4a16/
```

### 3. 开启 patch 并启动
```bash
export VLLM_AITER_W4A16_PATCH=1
export AITER_ROOT=/public/home/weishb/aiter
bash start.sh
```

---

## 技术要点

### 1. GPTQ → AWQ 列重排 (精度关键)
vllm 的 AWQ 权重是 GPTQ int32 打包([K, N//8],每 int32 装 8 个 N 值,顺序 [0,1,...7])。
aiter 的 `reverse_awq_order` 用 `AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]` 解包。
要让 aiter 解出正确的列序,送进 aiter 前必须按 **`AWQ_INV=[0,2,4,6,1,3,5,7]`** 重排列:

```python
_AWQ_INV = [0, 2, 4, 6, 1, 3, 5, 7]
def _gptq_to_awq_packed(b_q_gptq, qzeros_gptq):
    # b_q_gptq: [K, N//8] int32 (GPTQ)
    N8 = b_q_gptq.shape[1]
    inv = torch.tensor(_AWQ_INV, device=b_q_gptq.device, dtype=torch.int64)
    idx = inv * (N8 // 8) + torch.arange(N8 // 8, device=b_q_gptq.device).unsqueeze(0)
    b_q_awq = b_q_gptq[:, idx.reshape(-1)]
    qzeros_awq = qzeros_gptq[:, idx.reshape(-1)] if qzeros_gptq is not None else None
    return b_q_awq.contiguous(), (qzeros_awq.contiguous() if qzeros_awq is not None else None)
```

不做这步 → `cos_sim=0.25`(完全错);做了 → `cos_sim=1.000000`。

### 2. torch.compile / cudagraph 兼容 (性能关键)
aiter 的 triton driver 调用 inductor 编译不了。解法:用 `torch.library.custom_op` +
`register_fake` 把 aiter kernel 包成注册算子,对 dynamo/inductor/cudagraph 全透明:

```python
torch.library.define(f"aiter::{op_name}", "(Tensor input, Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor")
torch.library.impl(f"aiter::{op_name}", "CUDA")(_aiter_impl)
@torch.library.register_fake(f"aiter::{op_name}")
def _fake(input, qweight, scales, qzeros):
    return torch.empty((input.shape[0], qweight.shape[0]), dtype=scales.dtype, device=input.device)
```

这样 vllm 的 torch.compile (opt-level 3) + cudagraph 能正常捕获(49 张图全捕获成功),
aiter kernel 在图内执行,无 graph break。

### 3. aiter triton 3.5 兼容修复
`aiter_gemm_a16w4.py` 相对 aiter 原版改了两处:
- `@triton.utils.jit` → `@triton.jit`(5 处,triton 3.5 无 `triton.utils.jit`)
- `@triton.jit(key=[...])` → `@triton.jit`(去掉 `key=`,triton 3.5 已移除)

### 4. config 调优 (小 M 性能关键)
BW10 有 80 CUs。调优后关键参数:
- `BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32`(down 大 K 用 BK=64)
- **`SPLITK=2`** 对小 M(M=4,decode 批)是关键 —— 把 K 维 split 给多个 CU 做 reduce,
  否则小 M 时 CU 利用率太低
- `num_warps=4, NUM_CUS=80`

未调优的 aiter 默认 config 比 vllm 还慢 10-28%;调优后比 vllm 快 1.6-2.33x(kernel 级, M=4)。

---

## 注意事项

- **group_size=32**: 模型是 AWQ gs=32。aiter asm 路径(awq_gemm_asm)硬编码 gs=64,
  改 gs=32 需重写汇编,且 gs=32→64 合并精度损失大(cos_sim 0.7-0.84),故走 triton 路径。
- **AITER_ROOT**: patch 用 importlib 绕过 aiter `__init__.py` 的 JIT 编译(ck_tile/core.hpp 找不到),
  只加载 `aiter.ops.triton` 子模块。`AITER_ROOT` 指向 aiter 仓库根目录。
- **torch compile 缓存**: 切换 patch 开关后,若遇 `'_OpNamespace' 'aiter' object has no attribute ...`
  报错,清缓存:
  ```bash
  rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root
  ```
