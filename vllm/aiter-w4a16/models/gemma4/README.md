# gemma-4-31B-it-AWQ-4bit · 部署 & 调优

在 Hygon DCU BW10 (gfx936) 上加速 gemma-4-31B-it-AWQ-4bit 推理。两阶段叠加：
**阶段一** aiter w4a16 GEMM（本目录）+ **阶段二** flash attn fp8 KV（`vllm/flash-attn/`），
端到端 **TPOT -60%、吞吐 +147%**，精度无损。

> 通用的 patch 机制（kernel、vllm 接入、安装脚本）见上级 [`../../README.md`](../../README.md)。
> 本文件讲 gemma4 专属的全部内容：模型 shape、两阶段部署、调优 config、性能与精度。

---

## 更新记录

### 2026-08-14 · SPLITK 全 batch 调优（仅改 config，未动 kernel/源码）

**背景**：之前 SPLITK 只在小 batch 调过（M=4 用 SK=2），且一度怀疑 SK≥2 有精度/稳定性问题。
本次系统性验证后放开 SK，对 6 个 TP4 sharded 形状做全 batch 调优。

**关键结论**：

1. **SPLITK 精度无损**：用真实模型权重对比数学真值，SK 1~16 全部 `cos=1.0000`。
   之前"SK≥2 导致 cos<0.99"是测量 bug（拿 SK 配置输出对比 SK1 配置输出，两个近似值相减放大了误差）。
   SK>1 走 fp32 累加（`D_DTYPE=32`），数值上比 SK=1 更精确。
2. **VMFault 根因不是 SPLITK**：是 `BLOCK_SIZE_K=64` 或 `NUM_GROUPS=2`。
   `BK=32 / NG=1` + 任意 SK（6/8/12/16）生产稳定运行，无 VMFault。
3. **小 batch（M≤4）维持原 patch 配置**：5/6 个形状原配置已最优，仅 o_sw 差 1.04x，不值得动。
4. **中 batch（M≥8）用 SK≠1 最优**：单卡 GEMM 比 SK=1 快 1.4~2.7x → **decode 吞吐提升**。
5. **大 batch（M≥128）用 SK=1**：CU 利用率已饱和，split-K 无收益反而增加 reduce 开销。
6. **TTFT 优化 = prefill 段大 M config 调优**：TTFT 由 prefill 时间决定，prefill 阶段 M = 输入 token 数
   （1024~4096+），是 GEMM 密集型。本次为大 M（M=256~4096）精测出 `BM256 BN128 SK1 DD16` 配置
   （tile 数少、compute 密度高），比上次的 BM128 版本更优 → **prefill 加速 → TTFT 降低**。
   注：此前版本 config 只覆盖到 M=128，M>128 回退 BM16 导致 prefill 慢 2.5x，是 TTFT 高的根因。

**config 覆盖范围**：6 个 sharded 形状的 config 从 8 个 M key（1/2/4/8/16/32/64/128）
扩展到覆盖**全部 cudagraph 捕获尺寸**（1/2/4/8/16/24/32/.../512）+ prefill 尺寸（1024/2048/4096），
共 54 个 M key，确保任何 batch 都有精确匹配的调优 config，不回退默认值。

**6 个 sharded 形状 SPLITK 策略一览**（M: 各 batch 段 SK 值）：

| 形状 (N,K) | 层 | M=1 | M=4 | M=8 | M=16 | M=32 | M=64 | M≥128 |
|------------|-----|-----|-----|-----|------|------|------|-------|
| 4096,5376 | qkv_sw | 16 | 12 | 12 | 6 | 6 | 3 | 1 |
| 5120,5376 | qkv_fa | 12 | 12 | 12 | 6 | 3 | 2 | 1 |
| 5376,2048 | o_sw | 2 | 2 | 4 | 3 | 4 | 1 | 1 |
| 5376,4096 | o_fa | 16 | 8 | 8 | 8 | 4 | 2 | 1 |
| 10752,5376 | gate_up | 12 | 8 | 8 | 4 | 2 | 1 | 1 |
| 5376,5376 | down | 16 | 8 | 8 | 8 | 4 | 2 | 1 |

> 所有配置统一 `BK=32 / NG=1`（VMFault 安全），`BN=128`（大形状）/ `BN=64`（小形状）。
> SK>1 时 `D_DTYPE=32`（fp32 累加）；SK=1 时 `D_DTYPE=16`（fp16，省显存）。

**改动文件**（只改 config，未碰 kernel/源码/启动脚本）：
- `models/gemma4/configs/awq_w4a16/` 下 6 个 sharded JSON（重写，扩展 M 覆盖 + SK 调优）
- `aiter.patch` 第 3 段重新生成（gemma4 config 固化在 patch 里），段 1/2 代码未动
- 4 个非 sharded JSON（1344/3584/5376-14336）同步为已部署的较新版本（13 M key）

**端到端性能**（vllm bench serve，5120 in / 1024 out，两阶段叠加，TP4）：

| batch | TPOT | TTFT | output 吞吐 (tok/s) | engine gen (tok/s) | acceptance |
|-------|------|------|---------------------|--------------------|-----------|
| 4 | 34.98ms | 1.20s | 109.93 | ~110 | 97.54% |
| 8 | 46.90ms | — | 131.82 | ~200 | 94.24% |
| 16 | 59.94ms | — | 181.10 | ~330 | 92.54% |
| 32 | — | — | — | **468~489** | 91~94% |

**精度**：HumanEval pass@1 = **98.17%**（164 题全量，比上次的 97.56% 更高，精度无损）。

> 注：`vllm bench serve` 的 output 吞吐含 TTFT/排队（端到端），大 batch 下显著低于
> engine 日志的 generation throughput（纯 decode）。评估真实 decode 吞吐看 engine gen 列。

---

## 模型信息

- 模型：gemma-4-31B-it-AWQ-4bit（compressed-tensors 格式）
- 量化：AWQ，uint4，**group_size=32**，asymmetric（symmetric=false），zp_dtype=int8
- 结构：60 层，hidden=5376，GQA 32:16，head_dim=256，sliding_window=1024
- intermediate：14336

---

## 〇. 前置条件

确认环境已具备（一般 DCU 镜像里已装好）：

| 组件 | 版本 | 查看命令 |
|------|------|---------|
| vllm | 0.23.0 DCU 定制版 (`0.23.0+das.dtk2604`) | `pip show vllm \| grep Version` |
| aiter | DCU 定制版 (`0.1.3+das.dtk2604`，pip 安装) | `pip show aiter \| grep Version` |
| 硬件 | Hygon DCU BW10 (gfx936) | `rocminfo \| grep gfx` |

> aiter 用 `pip install aiter` 装的预编译版即可，**不需要 aiter 源码仓库**。
> 模型权重：`gemma-4-31B-it-AWQ-4bit`（AWQ，uint4，group_size=32）+ `gemma-4-31B-it-assistant`（MTP draft）。

---

## 新容器完整部署（阶段一 + 阶段二）

新镜像容器里从零部署，两个阶段叠加（aiter w4a16 GEMM + flash attn fp8 KV）。
按顺序执行，每步都给可复制命令。

### 0. 拉取本仓库

```bash
git clone git@github.com:benyuereal/llm-optimize.git
cd llm-optimize
```

### 1. 阶段一 · 安装 aiter w4a16 patch

```bash
cd vllm/aiter-w4a16
./patch.sh install          # 打 aiter.patch (vllm triton_w4a16 + aiter kernel + 10 个 config)
./patch.sh status           # 确认: "aiter patch 已打" + "gemma4 config : 10 / 10 个已放置"
cd ../..                    # 回到 llm-optimize 根
```

### 2. 阶段二 · 安装 flash attn whl

whl 体积大（200M，超 GitHub 单文件 100M 限制）不入仓库，放在 GitHub Release 附件。

**whl 文件名**：`flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl`

```bash
# 2.1 下载 whl 放到 vllm/flash-attn/dist/ (三选一)

#   方式 A: 浏览器到本仓库 GitHub Release 页面下载, 手动放到 dist/
#   方式 B: gh CLI
gh release download <release-tag> \
    -R benyuereal/llm-optimize \
    -p "flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl" \
    -D vllm/flash-attn/dist/

#   方式 C: 若 whl 已在别处, 直接拷贝/指定路径
cp /path/to/flash_attn-*.whl vllm/flash-attn/dist/

# 2.2 安装 (脚本会: 卸载旧 flash_attn → pip 装新 whl → 验证 import + 512 prefill 符号
#              → 打 vllm 侧 fp8_e5m2 patch (3 文件) → 打 aiter 侧 fp8_e5m2 patch (1 文件))
cd vllm/flash-attn
bash patch.sh install
bash patch.sh status        # 确认: whl "版本: 2.8.3+das.opt1..." + vllm/aiter "fp8_e5m2 patch 已打"
cd ../..                    # 回到 llm-optimize 根
```

> `patch.sh install` 做三件事:
> ```bash
> # 1. 替换 flash_attn python 包 (whl, 200M)
> pip3 uninstall -y flash_attn
> pip3 install --force-reinstall --no-deps vllm/flash-attn/dist/flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl
>
> # 2. 打 vllm 侧 fp8_e5m2 patch (patch -p0, 改 3 个 vllm 源码文件)
> cd /usr/local/lib/python3.10/dist-packages && patch -p0 < vllm/flash-attn/flash_fp8e5m2.patch
>
> # 3. 打 aiter 侧 fp8_e5m2 patch (patch -p0, 改 1 个 aiter 源码文件)
> cd /usr/local/lib/python3.10/dist-packages && patch -p0 < vllm/flash-attn/flash_aiter_fp8e5m2.patch
> ```
> 若 whl 不在 `dist/`，可指定路径：`WHL=/path/to/flash_attn-*.whl bash patch.sh install`

> `--no-deps` 很关键：避免 pip 顺带升级依赖（如 torch）破坏 DCU 环境。

> **为什么必须改 vllm 源码？** 上游 vllm 对 compressed-tensors 模型一律禁用 `fp8_e5m2` KV cache
> （在 `attention.py` 里无条件报错 `fp8_e5m2 kv-cache is not supported with fp8 checkpoints.`）。
> 但我们的 AWQ-4bit 模型 checkpoint 里并没有 fp8 KV scale（`kv_cache_scheme=None`），
> 只是运行时想把 KV cache 存成 e5m2。所以需要改 3 个文件放行 e5m2 并让读/写路径走 triton
> （C++ `reshape_and_cache_flash` op 不支持 e5m2）。不打这个 vllm patch，新容器启动会直接报上面的错。

> **为什么必须改 aiter 源码？** aiter 的 `unified_attention.py` 把 decode/prefill attention 路由到
> flash `varlen_fwd_unified`，但原版在 fp8 KV 时直接把 fp16 的 Q 和 e5m2 的 K 传给 flash，
> flash 的 prefix decode kernel 要求 Q/K 同 dtype，会报
> `RuntimeError: For prefix decode, query and key must have the same dtype`。
> 改动：fp8 KV 时把 Q cast 成 KV dtype 再传，并附 unit descale（纯 bit-cast，数值无影响）；
> 同时扩展 flash 路由条件（MTP q_len≤8 / prefill / head_dim=512）。不打这个 aiter patch，
> 新容器 CUDA graph 捕获阶段就会崩溃。

### 3. 启动服务（两阶段叠加）

用阶段二提供的启动脚本（它已同时启用两阶段）：

```bash
cd vllm/flash-attn/models/gemma4
bash start_flash.sh
```

该脚本相比阶段一的 `start.sh` 关键差异（两个阶段叠加所需）：

| 参数 | 阶段一 (start.sh) | 阶段二 (start_flash.sh) |
|------|-------------------|----------------------------------|
| `--attention-backend` | TRITON_ATTN | **ROCM_AITER_UNIFIED_ATTN** |
| `--kv-cache-dtype` | fp8 | **fp8_e5m2** |
| `--dtype` | (未显式) | **float16** |
| `HIP_VISIBLE_DEVICES` | 0,1,2,3 | **0,4,2,3** |
| 环境变量 | VLLM_AITER_W4A16_PATCH=1 | +ATTN_FLASH_PREFILL=1 +ATTN_FLASH_HEAD512=1 |

> 脚本里模型路径默认 `/data/zq/models/...`，若你的模型路径不同，用环境变量覆盖:
> ```bash
> # 主模型 + draft 自动推导为主模型同父目录下的 gemma-4-31B-it-assistant
> MODEL_DIR=/your/path/to/gemma-4-31B-it-AWQ-4bit bash start_flash.sh
>
> # 或单独指定 draft 模型路径
> MODEL_DIR=/your/main DRAFT_MODEL_DIR=/your/draft bash start_flash.sh
> ```
> 也可直接改脚本里的默认值。

等到日志出现 `Application startup complete`、`Uvicorn running on ...` 表示就绪。
启动成功应看到 draft 模型 head512 的 `full_attention` 走 flash kernel（不再是 aiter 2D）、
CUDA graph capture 通过。

---

## 调优 config

`configs/awq_w4a16/` 下 10 个 config（均 device_name=BW200，group_size=32），按层权重 shape 命名：

| 文件 (N,K) | 对应层 | TP4 sharded? | 说明 |
|-----------|--------|:---:|------|
| N=4096, K=5376 | qkv_proj (sliding) | ✓ | fused qkv, sliding 层 TP4 分片 |
| N=5120, K=5376 | qkv_proj (full attn) | ✓ | fused qkv, full_attention 层 TP4 分片 |
| N=5376, K=2048 | o_proj (sliding) | ✓ | sliding 层 TP4 分片 |
| N=5376, K=4096 | o_proj (full attn) | ✓ | full_attention 层 TP4 分片 |
| N=10752, K=5376 | merged gate/up | ✓ | gate_up 合并后 TP4 分片 |
| N=5376, K=5376 | down_proj | ✓ | down_proj TP4 分片 |
| N=1344, K=5376 | q_proj (单) | | 未合并的 q_proj |
| N=3584, K=5376 | gate/up_proj (单) | | 未合并的 gate/up |
| N=5376, K=14336 | down_proj (单) | | 未分片的 down（大 K） |
| N=1344, K=14336 | | | 未分片大 K 形状 |

> 6 个打 ✓ 的 sharded 形状是 TP4 生产路径实际命中的，已做全 batch SPLITK 调优（见"更新记录"）。
> 4 个未打 ✓ 的是非合并/非分片路径，用通用安全配置。

config 文件格式：`{M_int: {BLOCK_SIZE_M/N/K, SPLITK, num_warps, NUM_CUS, D_SHAPE, ...}}`。
运行时 aiter 据当前层 `(N,K)` 找文件，再据 `M`（batch）选文件里的具体 config。

### 关键调优参数

- **`BLOCK_SIZE_K=32, NUM_GROUPS=1`**：VMFault 安全配置的硬约束。
  BK=64 或 NG=2 会触发 VMFault（已确认根因），BK=32/NG=1 + 任意 SPLITK 生产稳定。
- **`SPLITK` 自适应**：核心调优旋钮。小/中 batch（M≤64）CU 利用率不足，用 SK≠1（2~16）
  把 K 维 split 给多个 CU 并行 reduce；大 batch（M≥128）CU 已饱和，SK=1 省 reduce 开销。
  SK>1 时 `D_DTYPE=32`（fp32 累加，精度无损实测 cos=1.0000），SK=1 时 `D_DTYPE=16`。
- `BLOCK_SIZE_M=16`（小/中 M decode）/ `BM=256`（大 M prefill，TTFT 关键），`BLOCK_SIZE_N=128`（大形状 N≥4096）/ `64`（小形状）
- `num_warps=4, NUM_CUS=48`（大形状）/ `NUM_CUS=80`（小形状）

> **TTFT 与 decode 分别由不同 M 段的 config 决定**：
> - **TTFT**（prefill）：大 M（256~4096）config → `BM256 BN128 SK1 DD16`，prefill GEMM 加速
> - **decode 吞吐**：小/中 M（1~64）config → `SK≠1`（2~12）增并行，GEMM 快 1.4~2.7x
> 两者都在本次 aiter.patch 的 config 里，一个 patch 同时覆盖。

未调优的 aiter 默认 config 比 vllm 还慢 10-28%；调优后比 vllm 快 1.6-2.33x（kernel 级，M=4）。

> 6 个 TP4 sharded 形状（4096/5120/5376-2048/5376-4096/10752/5376-5376）已做全 batch
> SPLITK 调优，覆盖全部 cudagraph M（1~512）+ prefill（1024/2048/4096），详见"更新记录"。
> 其余 4 个非 sharded 形状（1344/3584/5376-14336/1344-14336）用通用安全配置（BK=32, SK=1~2）。

---

## 性能数据

### 两阶段叠加参考性能（batch 4）

| 阶段 | TPOT | 说明 |
|------|------|------|
| baseline (vllm 原生 triton w4a16) | 90.72ms | 无优化 |
| 仅阶段一 (aiter w4a16) | **67.76ms** | GEMM 加速, attention 仍走 triton |
| 阶段一+二 (叠加 flash) | **35.02ms** | draft full_attention 走 flash, ~2.25x, 接受率 98.08% |

### 阶段一详细对比（vllm bench serve，4 prompts × 5120 in / 1024 out）

| 配置 | duration | TPOT | TTFT | 吞吐 (tok/s) | acceptance |
|------|----------|------|------|-------------|-----------|
| baseline (vllm 原生) | 94.45s | 90.72ms | 1.10s | 43.37 | 96.85% |
| **aiter patch (gs=32, tuned)** | **71.12s** | **67.76ms** | 1.06s | **57.59** | **98.30%** |
| 提升 | **-24.7%** | **-25.3%** | 持平 | **+32.8%** | +1.45pp |

- TPOT：90.72ms → 67.76ms（-25.3%）→ 叠加 flash 后 35.02ms（vs baseline **-61.4%**）
- 吞吐：43.37 → 57.59 tok/s（阶段一 +32.8%）→ 叠加 flash 后 107.02 tok/s（vs baseline **+147%**）
- TTFT 基本持平（1.10s vs 1.06s）
- Acceptance rate：96.85% → 98.30%（阶段一）→ 98.08%（叠加 flash，持平）
- 精度：离线 verify `cos_sim = 1.000000`（完全一致）；HumanEval pass@1 = **97.56%**（164 题）

> 关闭 `--enable-log-requests` 后性能进一步提升（约 5-7ms TPOT 差异），生产环境建议关闭。
> 该参数会占用 CPU I/O 资源，影响 decode 阶段性能。

环境：TP=4，attention-backend TRITON_ATTN（阶段一）/ ROCM_AITER_UNIFIED_ATTN（叠加），
kv-cache-dtype fp8 / fp8_e5m2，optimization-level 3，MTP speculative decoding (num_speculative_tokens=3)，
max-num-batched-tokens 16384。

> 首次 bench 含 warmup（编译/图捕获）偏慢，**对比取稳态第二轮**。

### 性能验证命令

另开终端，连跑两轮取第二轮稳态：

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

> 第一次跑含 warmup（编译/图捕获），偏慢。**对比时取第二次稳态结果**，或连跑两轮看第二轮。
> 关键指标：`Mean TPOT (ms)`（per-output-token 延迟）、`Benchmark duration (s)`。

---

## 锁频检测（性能关键）

DCU 默认按负载动态调频，decode 阶段负载低时频率会掉下来，导致 TPOT 偏高 5~7ms、
吞吐掉 7% 左右。**必须把用的几张卡锁到 sclk level 6（760MHz）** 才能拿到上面的参考性能。

`start_flash.sh` 启动时已自动锁频并打印确认，但某些容器环境下 rocm-smi 锁频会静默失败
（权限不足、设备接口差异等），打印"锁频完成"实际却没锁上。所以装好后**务必检测一次**。

### 检测锁频是否生效

vllm 跑着的时候，另开一个终端：

```bash
rocm-smi --showclocks 2>&1 | grep sclk
```

正常输出（4 张卡都 level 6 / 760Mhz）：
```
HCU[0] : sclk clock level: 6 (760Mhz)
HCU[1] : sclk clock level: 6 (760Mhz)
HCU[2] : sclk clock level: 6 (760Mhz)
HCU[3] : sclk clock level: 6 (760Mhz)
```

异常表现（说明没锁上）：
- 没有 `sclk` 这一行，或显示 `level: 0 (300Mhz)`
- `rocm-smi` 主表里 Perf 列是 `auto` 而非 `manual`

### 手动锁频

如果检测发现没锁上，手动锁（不需要重启 vllm，锁完直接 bench 即可）：

```bash
for i in 0 4 2 3; do
  rocm-smi -d $i --setperflevel manual
  rocm-smi -d $i --setsclk 6
done

# 确认
rocm-smi --showclocks 2>&1 | grep sclk
```

看到 4 张卡都 `Successfully set sclk frequency mask to Level 6` 即成功。
锁完**不用重启 vllm**，直接再跑 bench，TPOT 应回落到正常水平。

> 若手动锁频也报错（如 `Permission denied`、`Failed to set`），说明容器没有
> rocm-smi 写权限。需在宿主机层面锁频（所有容器共享），或给容器加设备权限。

---

## 精度验证（HumanEval）

启动 vllm 后，用 evalscope 跑 HumanEval 代码生成评测，确认精度无下降：

```bash
evalscope eval \
  --model gemma4 \
  --api-url http://127.0.0.1:8001/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --eval-batch-size 16 \
  --generation-config '{"temperature": 0.2, "top_p": 0.95, "repetition_penalty": 1.05, "max_tokens": 8192, "extra_body": {"chat_template_kwargs": {"thinking_mode": "disabled"}}}' \
  --timeout 100000 \
  --work-dir ./outputs/
```

结果在 `./outputs/` 下，查看 pass@1 是否达到该模型正常水平即可。

### 实测精度结果

SPLITK 全 batch 调优后（2026-08-14）：

```
┌─────────┬───────────┬─────────────────┬──────────────────┬───────┬─────────┐
│ Model   │ Dataset   │ Metric          │ Subset           │   Num │   Score │
├─────────┼───────────┼─────────────────┼──────────────────┼───────┼─────────┤
│ gemma4  │ humaneval │ mean_acc_pass@1 │ openai_humaneval │   164 │  0.9817 │
└─────────┴───────────┴─────────────────┴──────────────────┴───────┴─────────┘
```

- **HumanEval pass@1 = 0.9817（98.17%）**，164 题全量评测
- 平均延迟 11.23s，平均吞吐 15.72 tok/s，平均输入 186 token / 输出 176 token
- 上一次（SK=2 小 batch 配置）测得 97.56%，本次 SK 放开后 98.17%，**精度无损且有波动提升**

pass@1 达 98.17%，与该模型未打 patch 时的正常水平一致，确认 **SPLITK≠1 精度无损**。

> 另有离线精度验证 `tests/verify_aiter_v2.py`，直接对比 aiter 与 vllm kernel 输出的 `cos_sim`，
> 实测 **= 1.000000**（完全一致），从算子层确认精度无损。

---

## 回退

两阶段都要回退（顺序与安装相反）：

```bash
# 先回退阶段二 (卸载 flash whl, 重装官方 flash-attn)
cd vllm/flash-attn && bash patch.sh revert

# 再回退阶段一 (恢复 vllm+aiter 原始文件)
cd ../aiter-w4a16 && bash patch.sh revert
```

> 若只部署了阶段一，单独 `cd vllm/aiter-w4a16 && ./patch.sh revert` 即可，
> 恢复原始 `triton_w4a16.py` 和 aiter 原始 kernel，清缓存。

---

## 附：常见问题

- **启动报 `'_OpNamespace' 'aiter' object has no attribute ...`**：
  旧的 torch.compile 缓存被复用了。执行
  `rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root` 后重启。
  `patch.sh install/revert` 已自动清缓存，只在手动切换环境变量时需要手动清。

- **排查问题时想关掉 aiter**：`VLLM_AITER_W4A16_PATCH=0 vllm serve ...`（装了 patch 但运行时回退到 vllm 原生 triton，用于定位是否 aiter 引入的问题）。

- **TPOT 比参考值偏高 5~7ms**：大概率是 DCU 没锁频。按上面"锁频检测"一节
  检查 `rocm-smi --showclocks | grep sclk`，没锁上就手动锁。

- **换到新容器/新机器**：只要 `pip install aiter`（DCU 定制版）和 vllm 已装好，
  重新 `git clone` 本仓库 + 按上面"新容器完整部署"两步 install 即可，无需拉 aiter 源码。

- **换别的 AWQ w4a16 模型**：kernel 和安装脚本是通用的，只需为该模型调优 config
  （放 `models/<新模型>/configs/awq_w4a16/`），再 `./patch.sh install <新模型>`。
  详见 [`../../README.md`](../../README.md)。
