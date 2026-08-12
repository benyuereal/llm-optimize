# aiter w4a16 GEMM patch (通用)

Hygon DCU BW10 (gfx936) 上,用 **aiter triton w4a16 kernel** 替换 vllm 自带 triton w4a16 kernel,
提升 AWQ w4a16 推理性能,精度无损。

这是一套**通用机制**:kernel、vllm 接入、安装脚本与模型无关,适用于任何 AWQ w4a16 模型
(group_size ∈ {32, 64, 128, -1}),**同时兼容对称量化 (uint4b8) 与非对称量化**。
**调优 config 按模型区分**,放在 `models/<model>/` 下。

> 目前已调优的模型见 `models/`。首个也是唯一一个:**gemma4**(gemma-4-31B-it-AWQ-4bit, gs=32),
> 端到端 TPOT -25%。详见 [`models/gemma4/README.md`](models/gemma4/README.md)。

**兼容性**:
- **对称量化 (uint4b8, 无 zp 张量)**:patch 自动造全 8 zp,等价 `(w-8)*scale` ✅
- **非对称量化 (有 zp 张量)**:用真实 zp ✅
- **bf16 模型**:启动加 `--dtype float16` 对齐 aiter kernel 的 fp16 输出,精度/性能无损
  (HumanEval pass@1 = 96.95%)。详见 [技术要点 2.2](#22-bf16-模型---dtype-float16-兼容关键)

**前置条件**:
- vllm 0.23.0 DCU 定制版(`0.23.0+das.dtk2604`)
- aiter DCU 定制版(`pip install aiter`,版本 `0.1.3+das.dtk2604` 即可,无需 aiter 源码仓库)
- Hygon DCU BW10 (gfx936)

---

## 目录结构

```
gemm/w4a16/
├── README.md                         # 本文件 (通用说明)
├── patch.sh                          # 一键 install / revert / status / models
├── aiter.patch                       # 源码 patch (3 段合一: vllm triton_w4a16 + aiter kernel + configs)
├── tests/                            # 验证 / 性能 / 调优脚本 (通用)
│   ├── verify_aiter_v2.py            # 精度验证 (aiter vs vllm, cos_sim)
│   ├── verify_aiter_self.py          # aiter 自洽性验证
│   ├── verify_symmetric_zp.py        # 对称量化 zp 兼容验证
│   ├── bench_aiter_vs_vllm.py        # aiter vs vllm kernel 级性能对比
│   ├── bench_bf16_kernel.py          # bf16 vs fp16 kernel 精度+性能对比
│   ├── isolate_bf16_slow.py          # 定位 bf16 慢的根因 (隔离实验)
│   ├── isolate_scales_dtype.py       # scales dtype 对性能影响
│   ├── tune_aiter_config.py          # config 参数扫描
│   ├── gen_aiter_configs.py          # 生成 config json
│   └── test_custom_op.py             # custom_op + torch.compile 兼容性测试
└── models/                           # 按模型区分的调优 config
    └── gemma4/
        ├── README.md                 # gemma4 专属: shape / 性能数据 / 调优细节
        └── configs/awq_w4a16/        # gemma4 的 10 个调优 config (gs=32, BW200)
```

**通用 vs 模型专属**:
- 通用(本目录根):`aiter.patch`(3 段合一 patch)、`patch.sh`、`tests/`
  —— 换模型不用动 (config 段用 gemma4 的, 非 gemma4 模型 install 时覆盖 config)
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

`install` 会(用 `patch -p0` 打 `aiter.patch` 到 dist-packages):
1. 1/3 段: patch vllm `triton_w4a16.py`(GPTQ→AWQ 重排 / 对称 zp 兼容 / 原始 qweight 释放 / custom_op 接入)
2. 2/3 段: patch aiter `gemm_a16w4.py`(`@triton.utils.jit` → `@triton.jit`,triton3.5 兼容)
3. 3/3 段: 新增 10 个 config json 到 `aiter/ops/triton/configs/gemm/awq_w4a16/`(gemma4 调优配置)
4. 清空 torch.compile 缓存(切换后必做)

`revert` 用 `patch -p0 -R` 原样回退,恢复 vllm/aiter 原始文件并删除新增的 config。

> patch 基于 dist-packages 里 vllm+aiter 的**当前状态**打。若你之前手动改过这些文件,先 `revert` 回干净态再 `install`。
> 脚本顶部的 `DIST_DIR` 可按环境修改(默认 `/usr/local/lib/python3.10/dist-packages`)。

### 启动 vllm

`patch.sh install` 只装文件,不启动 vllm。**装了 patch 后默认就启用 aiter kernel**,
直接用你原来的 `vllm serve` 命令启动即可,无需额外环境变量:
```bash
# AITER_ROOT 默认指向 pip aiter 安装位置 (/usr/local/lib/python3.10/dist-packages), 无需设置
vllm serve /data/zq/models/gemma-4-31B-it-AWQ-4bit/ ...   # 你的启动命令
```

> **bf16 模型必须加 `--dtype float16`**:
> 对称量化模型通常是 bf16 权重,而 aiter kernel 硬编码 fp16 输出。不加会 `Half != BFloat16` 崩溃
> (torch.compile / cudagraph 下尤甚)。加 `--dtype float16` 把全链路对齐 fp16 即可:
> ```bash
> vllm serve <bf16对称模型> --dtype float16 ...
> ```
> 该参数对 fp16 模型无影响(本来就是 fp16),所以**两种模型都加 `--dtype float16` 最安全**。
> 生产脚本 `start.sh` 已内置此参数。

### 对比 baseline vs patch

**方式 A(推荐,不换文件)**:都 `./patch.sh install`,靠环境变量切换:
- 不设 / 设 `VLLM_AITER_W4A16_PATCH=1` → aiter patch(默认)
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

### 性能验证 (端到端 bench)

用 `vllm bench serve` 对比 baseline 和 patch。启动 vllm 后,在另一终端跑:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8001 \
  --model gemma4 \
  --tokenizer /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
  --dataset-name random \
  --random-input-len 5120 \
  --random-output-len 1024 \
  --num-prompts 4 \
  --seed 42
```

参数说明:
- `--num-prompts 4`:小批量,聚焦 decode 阶段(w4a16 GEMM 的主战场)
- `--seed 42`:固定随机输入,保证 baseline 和 patch 对比时输入完全一致
- `--random-input-len 5120 --random-output-len 1024`:长输入长输出场景
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`:避免 bench 工具连 HuggingFace

> 第一次跑含 warmup(编译/图捕获),偏慢。**对比时取第二次稳态结果**,或连跑两轮看第二轮。
> 关键指标:`Mean TPOT (ms)`(per-output-token 延迟)、`Benchmark duration (s)`。

对比流程:
1. `./patch.sh install` → 直接启动 → bench → 记 patch 数据(默认启用)
2. `VLLM_AITER_W4A16_PATCH=0` 重启(清缓存)→ bench → 记 baseline 数据
3. 比 TPOT / duration(精度另用 `tests/verify_aiter_v2.py` 验 cos_sim)

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
    return torch.empty((input.shape[0], qweight.shape[0]), dtype=input.dtype, device=input.device)
```

这样 vllm 的 torch.compile (opt-level 3) + cudagraph 能正常捕获(49 张图全捕获成功),
aiter kernel 在图内执行,无 graph break。

> `register_fake` 的输出 dtype 跟随 `input.dtype`。配合 `--dtype float16` 启动(见下节),
> input 与 aiter kernel 输出同为 fp16,fake 声明与实际一致,无 dtype 不匹配。

### 2.1 对称量化 (uint4b8) 兼容 (精度关键)

aiter kernel 的 dequant 算式是 `(w - zeros) * scales`,**必须有 zeros 张量**。
但对称量化 (uint4b8) 没有显式 zp 张量(零点恒为 8)。解法:模型加载时
(`process_weights_after_loading`)对对称量化**造一个全 8 的 zp 张量**,使其等价于
原生对称量化 `(w - 8) * scale`:

```python
# _aiter_preprocess_layer: 对称量化无 zp 时, 造全 zp_bias(8) 的 zp
if w_zp is None:
    # 8 个 4bit zp_bias pack 进一个 int32 = 0x88888888 (有符号 int32 = -2004318072)
    packed_bias = zp_bias & 0xF
    packed_bias = packed_bias | (packed_bias << 4) | ... | (packed_bias << 28)
    w_zp = torch.full((K//G, N//8), packed_bias, dtype=torch.int32, device=...)
```

- 对称模型(无 zp):`zp_bias = weight_type.bias`(=8),造全 8 zp ✅
- 非对称模型(有 zp):`zp_bias = 0`,用真实 zp ✅

### 2.1.1 原始 qweight 释放 (省 ~3.5GB/卡, 解 TP1 单卡 OOM)

`process_weights_after_loading` 把权重 repack 成 aiter 格式 (`aq`/`az`) 后,原始
`qweight [K, N//8] int32` + `qzeros`(~3.5GB/卡)就不再需要了 —— aiter 路径只读
`_aiter_w4a16_cache`(aq/az/scales)。但 repack 后原始权重和 aiter 权重**两份同样大的数据并存**,
TP1 单卡 32GB 装 31B 模型会 OOM,故 repack 完立即释放原始那份:

```python
# _aiter_preprocess_layer 末尾: aiter 路径只读 _aiter_w4a16_cache, 不再读 w_q/w_zp
del w_q_data, b_q_awq, qzeros_awq
replace_parameter(layer, w_q_name, None)        # 置空原始 qweight
if w_zp_name is not None and ... is not None:
    replace_parameter(layer, w_zp_name, None)   # 置空原始 qzeros
torch.cuda.empty_cache()
```

实测(gemma4-31B-AWQ, TP1 单卡全部 60 层):

| 张量 | 不释放 | 释放后 |
|------|--------|--------|
| 原始 qweight (int32) | 3.41 GB | 置空 |
| 原始 qzeros (int32) | 0.11 GB | 置空 |
| aiter aq (int8, repack后) | 3.41 GB | 3.41 GB(保留) |
| aiter az (int8, repack后) | 0.11 GB | 0.11 GB(保留) |
| scales (fp16) | 0.85 GB | 0.85 GB(保留) |
| **量化权重合计** | **7.89 GB** | **4.37 GB** |
| **净省** | | **3.52 GB/卡** |

TP4 每卡只装 1/4 权重,本来不紧张,省的 ~0.88GB/卡 属锦上添花;**TP1 单卡**这 3.5GB 是
OOM 与不 OOM 的区别(31B 权重 + KV cache + activation 易顶满 32GB)。

### 2.2 bf16 模型 + `--dtype float16` (兼容关键)

aiter kernel 硬编码 fp16 输出。模型权重若是 bf16(如对称量化模型),不加处理会
`Half != BFloat16` 崩溃(尤其在 torch.compile / cudagraph 下)。解法:启动加
`--dtype float16`,把全链路强制 fp16,正好对上 aiter kernel 的原生 fp16 输出:

```bash
vllm serve <bf16模型> --dtype float16 ...
```

- 零 kernel 改动(用原始 fp16 kernel)
- 绕过 `Half != BFloat16` 崩溃
- 精度无损:该模型 scale max=2.1875、权重 max=53760(远低于 fp16 上限 65504),bf16→fp16 无溢出,
  且 fp16 尾数(10位)比 bf16(7位)更细。HumanEval pass@1 = 96.95%(实测)
- 性能无损:aiter fp16 全速,MTP 接受率 98%

### 3. aiter triton 3.5 兼容修复
`aiter_gemm_a16w4.py` 相对 aiter 原版改了两处:
- `@triton.utils.jit` → `@triton.jit`(5 处,triton 3.5 无 `triton.utils.jit`)
- `@triton.jit(key=[...])` → `@triton.jit`(去掉 `key=`,triton 3.5 已移除)

### 4. config 调优 (小 M 性能关键)
BW10 有 48 CUs。调优后关键参数:
- `BLOCK_SIZE_M=16, BLOCK_SIZE_K=32, NG(NUM_GROUPS)=1`(BK=32/NG=1 是安全配置,
  BK=64/NG=2 在 BW10 上会触发 VMFault 崩溃)
- **小形状**(N≤3584 等): `BLOCK_SIZE_N=64, SPLITK=2`
- **大形状**(N=4096/5120/10752, K=4096/5376): `BLOCK_SIZE_N=128` + **自适应 SPLITK**
  (按 M 取 3~16,M 越小 SPLITK 越大,把 K 维 split 给更多 CU 做 reduce)
- `num_warps=4, NUM_CUS=48`

未调优的 aiter 默认 config 比 vllm 还慢 10-28%;调优后比 vllm 快 1.6-2.33x(kernel 级, M=4)。
SPLITK 离线调优对 kernel 级提速 ~1.5x(大形状),但 decode 端到端 GEMM 时间波动 2-3x 会淹没收益,
故 SPLITK 主要保证大形状不慢、不崩,端到端 TPOT 收益来自 w4a16 kernel 本身。

> 各模型的 shape 和具体 config 见 `models/<model>/README.md`。

---

## 为新模型添加调优

1. 用 `tests/gen_aiter_configs.py` 生成新模型各层 shape 的 config 骨架
2. 用 `tests/tune_aiter_config.py` 扫描调优(或参考 gemma4 的 config;大形状用 BN=128+自适应 SPLITK,小形状用 BN=64/SK=2)
3. 把调好的 config 放到 `models/<新模型>/configs/awq_w4a16/`
4. `./patch.sh install <新模型>`

---

## 注意事项

- **group_size=32**: aiter asm 路径(awq_gemm_asm)硬编码 gs=64,改 gs=32 需重写汇编,
  且 gs=32→64 合并精度损失大(cos_sim 0.7-0.84),故走 triton 路径。
- **aiter 依赖**: 用环境里 `pip install aiter` 装的预编译版(DCU 定制版 `0.1.3+das.dtk2604`)即可,
  **不需要 aiter 源码仓库**。`aiter.patch` 的 2/3 段会改 pip aiter 的 `gemm_a16w4.py`(改成
  triton3.5 兼容版,原版 `@triton.utils.jit` 在 triton 3.5 下会报错),3/3 段放置调优 config。
  patch 代码用 importlib 只加载 `aiter.ops.triton` 子模块,不触发 aiter `__init__.py` 的其他依赖。
- **DIST_DIR**: patch 打到 `DIST_DIR=/usr/local/lib/python3.10/dist-packages`(vllm + aiter pip 安装位置)。
  如装在别处(conda/venv),改 `patch.sh` 顶部的 `DIST_DIR` 指向实际的 site-packages 目录。
- **torch compile 缓存**: 切换 patch 开关或换文件后,若遇 `'_OpNamespace' 'aiter' object has no attribute ...`
  报错,是旧的编译缓存被复用了。`patch.sh install/revert` 已自动清缓存;手动切换环境变量时需自己清:
  ```bash
  rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root
  ```
