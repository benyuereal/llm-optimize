# aiter w4a16 GEMM patch (通用)

Hygon DCU BW10 (gfx936) 上,用 **aiter triton w4a16 kernel** 替换 vllm 自带 triton w4a16 kernel,
提升 AWQ w4a16 推理性能,精度无损。

这是一套**通用机制**:kernel、vllm 接入、安装脚本与模型无关,适用于任何 AWQ w4a16 模型
(group_size ∈ {32, 64, 128, -1})。**调优 config 按模型区分**,放在 `models/<model>/` 下。

> 目前已调优的模型见 `models/`。首个也是唯一一个:**gemma4**(gemma-4-31B-it-AWQ-4bit, gs=32),
> 端到端 TPOT -25%。详见 [`models/gemma4/README.md`](models/gemma4/README.md)。

---

## 目录结构

```
gemm/w4a16/
├── README.md                         # 本文件 (通用说明)
├── patch.sh                          # 一键 install / revert / status / models
├── triton_w4a16.py                   # vllm 原始 triton_w4a16.py (回退用)
├── triton_w4a16.py.patch             # 打了 aiter patch 的 triton_w4a16.py
├── aiter_gemm_a16w4.py               # aiter triton w4a16 kernel (已修 triton3.5 兼容)
├── tests/                            # 验证 / 性能 / 调优脚本 (通用)
│   ├── verify_aiter_v2.py            # 精度验证 (aiter vs vllm, cos_sim)
│   ├── verify_aiter_self.py          # aiter 自洽性验证
│   ├── bench_aiter_vs_vllm.py        # aiter vs vllm kernel 级性能对比
│   ├── tune_aiter_config.py          # config 参数扫描
│   ├── gen_aiter_configs.py          # 生成 config json
│   └── test_custom_op.py             # custom_op + torch.compile 兼容性测试
└── models/                           # 按模型区分的调优 config
    └── gemma4/
        ├── README.md                 # gemma4 专属: shape / 性能数据 / 调优细节
        └── configs/awq_w4a16/        # gemma4 的 10 个调优 config (gs=32, BW200)
```

**通用 vs 模型专属**:
- 通用(本目录根):`aiter_gemm_a16w4.py`(kernel)、`triton_w4a16.py.patch`(vllm 接入)、`patch.sh`、`tests/`
  —— 换模型不用动
- 模型专属(`models/<model>/`):`configs/`(按模型权重 shape 调优的 json)—— 换模型要重新调优

---

## 一键安装 / 回退

```bash
cd gemm/w4a16

./patch.sh install [model]   # 安装 patch (默认 model=gemma4)
./patch.sh revert            # 回退到 vllm 原始 triton_w4a16.py
./patch.sh status [model]    # 查看当前安装状态 (默认 model=gemma4)
./patch.sh models            # 列出可选的模型
```

`install` 会:
1. 备份当前 vllm `triton_w4a16.py` → `triton_w4a16.py.bak`(只备份一次,且不把 patch 版当原始版备份)
2. 把 `triton_w4a16.py.patch` 拷到 vllm 安装目录
3. 把 `aiter_gemm_a16w4.py` 放到 `$AITER_ROOT/aiter/ops/triton/`
4. 把 `models/<model>/configs/awq_w4a16/*.json` 放到 `$AITER_ROOT/aiter/ops/triton/configs/gemm/awq_w4a16/`
5. 清空 torch.compile 缓存(切换后必做)

> 脚本顶部的 `VLLM_DIR` 和 `AITER_ROOT` 可按环境修改。

### 启动 vllm

`patch.sh install` 只装文件,不启动 vllm。启动时需设置环境变量:
```bash
export VLLM_AITER_W4A16_PATCH=1   # 1=启用 aiter kernel; 0 或 unset=走 vllm 原生 triton w4a16
export AITER_ROOT=/public/home/weishb/aiter
# 然后用你自己的 vllm serve 启动命令
```

### 对比 baseline vs patch

**方式 A(推荐,不换文件)**:都 `./patch.sh install`,靠环境变量切换:
- `VLLM_AITER_W4A16_PATCH=1` → aiter patch
- `VLLM_AITER_W4A16_PATCH=0` → vllm 原生 triton w4a16(文件是 patch 版,但运行时不走 aiter 分支)

切换环境变量后**必须清缓存再重启**:
```bash
rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root
```

**方式 B(物理换文件)**:
```bash
./patch.sh install    # vllm 文件 = patch 版
./patch.sh revert     # vllm 文件 = 原始版
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

> 各模型的 shape 和具体 config 见 `models/<model>/README.md`。

---

## 为新模型添加调优

1. 用 `tests/gen_aiter_configs.py` 生成新模型各层 shape 的 config 骨架
2. 用 `tests/tune_aiter_config.py` 扫描调优(或参考 gemma4 的 config 手动设 SPLITK=2 等)
3. 把调好的 config 放到 `models/<新模型>/configs/awq_w4a16/`
4. `./patch.sh install <新模型>`

---

## 注意事项

- **group_size=32**: aiter asm 路径(awq_gemm_asm)硬编码 gs=64,改 gs=32 需重写汇编,
  且 gs=32→64 合并精度损失大(cos_sim 0.7-0.84),故走 triton 路径。
- **AITER_ROOT**: patch 用 importlib 绕过 aiter `__init__.py` 的 JIT 编译(ck_tile/core.hpp 找不到),
  只加载 `aiter.ops.triton` 子模块。`AITER_ROOT` 指向 aiter 仓库根目录。
- **torch compile 缓存**: 切换 patch 开关或换文件后,若遇 `'_OpNamespace' 'aiter' object has no attribute ...`
  报错,是旧的编译缓存被复用了。`patch.sh install/revert` 已自动清缓存;手动切换环境变量时需自己清:
  ```bash
  rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root
  ```
