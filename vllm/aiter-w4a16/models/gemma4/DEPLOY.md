# aiter w4a16 GEMM 加速 · 部署方案

在 Hygon DCU BW10 (gfx936) 上,用 aiter triton w4a16 kernel 替换 vllm 自带 kernel,
加速 gemma-4-31B-it-AWQ-4bit 推理,**精度无损**,端到端 **TPOT -25%、吞吐 +33%**。

本方案面向部署人员,按下面 4 步操作即可,无需理解内部实现。

> **新容器从零部署?** 直接看下面的 [§新容器完整部署(阶段一+阶段二)](#新容器完整部署阶段一阶段二),
> 一节走完两个阶段。本节后续是阶段一的单阶段细节。

---

## 新容器完整部署(阶段一+阶段二)

新镜像容器里从零部署 gemma-4-31B-it-AWQ-4bit,两个阶段叠加(aiter w4a16 GEMM + flash attn fp8 KV)。
按顺序执行,每步都给可复制命令。**前提:容器里已装好 vllm + aiter DCU 定制版**(见 [§前置条件](#〇-前置条件))。

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

whl 体积大(200M,超 GitHub 单文件 100M 限制)不入仓库,放在 GitHub Release 附件。

**whl 文件名**:`flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl`

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
#              → 打 vllm 侧 fp8_e5m2 patch, 改 3 个 vllm 源码文件)
cd vllm/flash-attn
bash patch.sh install
bash patch.sh status        # 确认: whl "版本: 2.8.3+das.opt1..." + "fp8_e5m2 patch 已打"
cd ../..                    # 回到 llm-optimize 根
```

> `patch.sh install` 做两件事:
> ```bash
> # 1. 替换 flash_attn python 包 (whl, 200M)
> pip3 uninstall -y flash_attn
> pip3 install --force-reinstall --no-deps vllm/flash-attn/dist/flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl
>
> # 2. 打 vllm 侧 fp8_e5m2 patch (patch -p0, 改 3 个 vllm 源码文件)
> cd /usr/local/lib/python3.10/dist-packages && patch -p0 < vllm/flash-attn/flash_fp8e5m2.patch
> ```
> 若 whl 不在 `dist/`,可指定路径:`WHL=/path/to/flash_attn-*.whl bash patch.sh install`

> `--no-deps` 很关键:避免 pip 顺带升级依赖(如 torch)破坏 DCU 环境。

> **为什么必须改 vllm 源码?** 上游 vllm 对 compressed-tensors 模型一律禁用 `fp8_e5m2` KV cache
> (在 `attention.py` 里无条件报错 `fp8_e5m2 kv-cache is not supported with fp8 checkpoints.`)。
> 但我们的 AWQ-4bit 模型 checkpoint 里并没有 fp8 KV scale (`kv_cache_scheme=None`),
> 只是运行时想把 KV cache 存成 e5m2。所以需要改 3 个文件放行 e5m2 并让读/写路径走 triton
> (C++ `reshape_and_cache_flash` op 不支持 e5m2)。不打这个 vllm patch,新容器启动会直接报上面的错。

### 3. 启动服务(两阶段叠加)

用阶段二提供的启动脚本(它已同时启用两阶段):

```bash
cd vllm/flash-attn/models/gemma4
bash start_flash.sh
```

该脚本相比阶段一的 `start.sh` 关键差异(两个阶段叠加所需):

| 参数 | 阶段一 (start.sh) | 阶段二 (start_flash.sh) |
|------|-------------------|----------------------------------|
| `--attention-backend` | TRITON_ATTN | **ROCM_AITER_UNIFIED_ATTN** |
| `--kv-cache-dtype` | fp8 | **fp8_e5m2** |
| `--dtype` | (未显式) | **float16** |
| `HIP_VISIBLE_DEVICES` | 0,1,2,3 | **0,4,2,3** |
| 环境变量 | VLLM_AITER_W4A16_PATCH=1 | +ATTN_FLASH_PREFILL=1 +ATTN_FLASH_HEAD512=1 |

> 脚本里 `MODEL_DIR` 和 `--speculative-config` 的模型路径默认 `/data/zq/models/...`,
> 若你的模型路径不同,设环境变量 `MODEL_DIR=/your/path` 再启动,或直接改脚本。

等到日志出现 `Application startup complete`、`Uvicorn running on ...` 表示就绪。
启动成功应看到 draft 模型 head512 的 `full_attention` 走 flash kernel(不再是 aiter 2D)、
CUDA graph capture 通过。

### 4. 锁频检测(性能关键)

DCU 默认动态调频,decode 阶段频率会掉,导致 TPOT 偏高 5~7ms。脚本启动时已自动锁频,
但容器环境下可能静默失败,**务必检测一次**:

```bash
rocm-smi --showclocks 2>&1 | grep sclk
```

4 张卡都应显示 `sclk clock level: 6 (760Mhz)`。若没锁上,手动锁(不用重启 vllm):

```bash
for i in 0 4 2 3; do
  rocm-smi -d $i --setperflevel manual
  rocm-smi -d $i --setsclk 6
done
rocm-smi --showclocks 2>&1 | grep sclk
```

### 5. 性能验证

另开终端,连跑两轮取第二轮稳态:

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

两阶段叠加的参考性能(batch 4):

| 阶段 | TPOT | 说明 |
|------|------|------|
| 仅阶段一 (aiter w4a16) | ~67.76ms | GEMM 加速, attention 仍走 triton |
| 阶段一+二 (叠加 flash) | **~35.63ms** | draft full_attention 走 flash, 1.68~2.2x |

> 阶段二主要加速 MTP draft 的 full_attention(head_dim=512),对长输入长输出 + MTP 场景收益最大。

### 6. 精度验证

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

两阶段叠加 HumanEval pass@1 = **97.56%**(164 题全量),与未优化前一致,精度无损。

### 回退(两阶段都要回退)

```bash
# 先回退阶段二 (卸载 flash whl, 重装官方 flash-attn)
cd vllm/flash-attn && bash patch.sh revert

# 再回退阶段一 (恢复 vllm+aiter 原始文件)
cd ../aiter-w4a16 && bash patch.sh revert
```

> 回退顺序:先 flash(阶段二)再 aiter(阶段一),与安装顺序相反。

---

## 〇. 前置条件

确认环境已具备(一般 DCU 镜像里已装好):

| 组件 | 版本 | 查看命令 |
|------|------|---------|
| vllm | 0.23.0 DCU 定制版 (`0.23.0+das.dtk2604`) | `pip show vllm \| grep Version` |
| aiter | DCU 定制版 (`0.1.3+das.dtk2604`,pip 安装) | `pip show aiter \| grep Version` |
| 硬件 | Hygon DCU BW10 (gfx936) | `rocminfo \| grep gfx` |

> aiter 用 `pip install aiter` 装的预编译版即可,**不需要 aiter 源码仓库**。
> 模型权重:`gemma-4-31B-it-AWQ-4bit`(AWQ,uint4,group_size=32)。

---

## 一. 下载代码

```bash
git clone git@github.com:benyuereal/llm-optimize.git
cd llm-optimize/vllm/aiter-w4a16
```

---

## 二. 一键安装

```bash
./patch.sh install
```

> `install` 后面可跟模型名(如 `./patch.sh install gemma4`),指定用哪套调优 config。
> 目前只有 gemma4 一个模型,也是默认值,所以可省略。

这一条命令会用 `patch -p0` 把 `aiter.patch` 打到 dist-packages,自动完成:
1. patch vllm 的 `triton_w4a16.py`(GPTQ→AWQ 重排 / 对称 zp 兼容 / 原始 qweight 释放 / custom_op 接入)
2. patch aiter 的 w4a16 kernel(改成 triton 3.5 兼容版)
3. 新增 gemma4 的调优 config(10 个 json)
4. 清空 torch.compile 缓存

装好后查看状态确认:

```bash
./patch.sh status
```

看到 `vllm triton_w4a16.py : aiter patch 已打` 和 `gemma4 config : 10 / 10 个已放置` 即成功。

> **默认启用**:装完 patch 后,直接用你原来的 `vllm serve` 命令启动即可,aiter kernel 自动生效,
> 无需设置任何环境变量。

---

## 三. 启动 vllm 并做性能验证

> **只部署阶段一?** 若只要 aiter w4a16 不叠加 flash,可用阶段二的启动脚本
> `vllm/flash-attn/models/gemma4/start_flash.sh` 作参考,把
> `--attention-backend` 改回 `TRITON_ATTN`、`--kv-cache-dtype` 改回 `fp8`、
> 去掉 `--dtype float16` 和 `ATTN_FLASH_*` 环境变量即可。下面的 `start.sh`
> 是阶段一的基准启动脚本(本地产物,未入仓库),内容供参考。

### 1. 启动 vllm

推荐使用项目提供的启动脚本 `start.sh`(位于 [`models/gemma4/start.sh`](start.sh)):

```bash
cd /public/home/weishb
bash start.sh
```

脚本内容(已包含锁频、环境变量、优化参数):

```bash
#!/bin/bash
# ============================================================
# gemma-4-31B-it-AWQ-4bit vllm serve 启动脚本
# 配置: TP=4 + aiter w4a16 gemm 加速 (group_size=32)
# ============================================================
set -e

export PATH=/opt/hyhal/bin:/opt/dtk/bin:$PATH

# ---- 锁定 4 卡高频 (sclk level 6 = 760MHz, 性能关键) ----
echo "[start.sh] 锁定 DCU 高性能模式..."
for i in 0 1 2 3; do
  rocm-smi -d $i --setperflevel manual >/dev/null 2>&1 || true
  rocm-smi -d $i --setsclk 6 >/dev/null 2>&1 || true
done
echo "[start.sh] 锁频结果确认:"
for i in 0 1 2 3; do
  sclk=$(rocm-smi -d $i --showclocks 2>/dev/null | grep -oE "sclk clock level: [0-9]+ \([0-9]+Mhz\)" | head -1)
  echo "  HCU[$i]: ${sclk:-未获取到频率}"
done

# ---- 环境变量 ----
export HIP_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_AITER_W4A16_PATCH=1

# ---- 启动 vllm serve (TP=4) ----
echo "[start.sh] 启动 vllm serve (TP=4, aiter w4a16 patch 启用)..."
vllm serve /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
    --host 0.0.0.0 --port 8001 \
    --served-model-name gemma4 \
    --max-model-len 32768 --max-num-seqs 256 \
    --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
    --optimization-level 3 --trust-remote-code \
    --enable-prefix-caching --enable-chunked-prefill \
    --language-model-only --async-scheduling \
    --performance-mode throughput \
    --max-num-batched-tokens 16384 \
    --speculative-config '{"method": "mtp", "model": "/data/zq/models/gemma-4-31B-it-assistant", "num_speculative_tokens": 3}'
```

> **注意**:脚本中**没有** `--enable-log-requests`(该参数会占用 CPU I/O 资源,导致 TPOT 偏高 5-7ms,生产环境建议关闭)。

等到日志出现 `Application startup complete`、`Uvicorn running on ...` 表示就绪。

### 2. 跑性能压测(aiter patch)

另开一个终端:

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

> **第一轮含预热(编译/图捕获),偏慢,请连跑两轮,取第二轮稳态结果。**
> 关键看 `Mean TPOT (ms)`(每 token 延迟)和 `Benchmark duration (s)`(总时长)。

### 参考性能数据(本方案实测)

| 配置 | duration | TPOT | TTFT | 吞吐 (tok/s) |
|------|----------|------|------|-------------|
| baseline (vllm 原生) | 94.45s | 90.72ms | 1.10s | 43.37 |
| **aiter patch** | **71.12s** | **67.76ms** | 1.06s | **57.59 tok/s** |
| 提升 | **-24.7%** | **-25.3%** | 持平 | **+32.8%** |

> 关闭 `--enable-log-requests` 后性能进一步提升(约 5-7ms TPOT 差异),实测稳态 TPOT 67-68ms。
> 该参数会占用 CPU I/O 资源,影响 decode 阶段性能,生产环境建议关闭。

环境:TP=4,MTP 投机解码(num_speculative_tokens=3),optimization-level 3,
**关闭 `--enable-log-requests`**(该参数可导致 TPOT 偏高 5-7ms)。
**前提:DCU 已锁频到 sclk 760MHz**(见下节,锁频不生效会导致 TPOT 偏高 5~7ms)。

---

## 三·5. 锁频检测与手动锁频(性能关键)

DCU 默认按负载动态调频,decode 阶段负载低时频率会掉下来,导致 TPOT 偏高 5~7ms、
吞吐掉 7% 左右。**必须把用的几张卡锁到 sclk level 6(760MHz)** 才能拿到上面参考表的性能。

`start.sh` 启动时已自动锁频并打印确认,但某些容器环境下 rocm-smi 锁频会静默失败
(权限不足、设备接口差异等),打印"锁频完成"实际却没锁上。所以装好后**务必检测一次**。

### 检测锁频是否生效

vllm 跑着的时候,另开一个终端:

```bash
rocm-smi --showclocks 2>&1 | grep sclk
```

正常输出(4 张卡都 level 6 / 760Mhz):
```
HCU[0] : sclk clock level: 6 (760Mhz)
HCU[1] : sclk clock level: 6 (760Mhz)
HCU[2] : sclk clock level: 6 (760Mhz)
HCU[3] : sclk clock level: 6 (760Mhz)
```

异常表现(说明没锁上):
- 没有 `sclk` 这一行,或显示 `level: 0 (300Mhz)`
- `rocm-smi` 主表里 Perf 列是 `auto` 而非 `manual`

### 手动锁频

如果检测发现没锁上,手动锁(不需要重启 vllm,锁完直接 bench 即可):

```bash
for i in 0 1 2 3; do
  rocm-smi -d $i --setperflevel manual
  rocm-smi -d $i --setsclk 6
done

# 确认
rocm-smi --showclocks 2>&1 | grep sclk
```

看到 4 张卡都 `Successfully set sclk frequency mask to Level 6` 即成功。
锁完**不用重启 vllm**,直接再跑 bench,TPOT 应回落到正常水平(~68ms)。

> 若手动锁频也报错(如 `Permission denied`、`Failed to set`),说明容器没有
> rocm-smi 写权限。需在宿主机层面锁频(所有容器共享),或给容器加设备权限。

---

## 四. 精度验证(HumanEval)

启动 vllm(aiter patch 默认启用)后,用 evalscope 跑 HumanEval 代码生成评测,确认精度无下降:

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

结果在 `./outputs/` 下,查看 pass@1 是否达到该模型正常水平即可。

### 实测精度结果(本方案)

```
┌─────────┬───────────┬─────────────────┬──────────────────┬───────┬─────────┐
│ Model   │ Dataset   │ Metric          │ Subset           │   Num │   Score │
├─────────┼───────────┼─────────────────┼──────────────────┼───────┼─────────┤
│ gemma4  │ humaneval │ mean_acc_pass@1 │ openai_humaneval │   164 │  0.9756 │
└─────────┴───────────┴─────────────────┴──────────────────┴───────┴─────────┘
```

- **HumanEval pass@1 = 0.9756(97.56%)**,164 题全量评测
- 平均延迟 15.91s,平均吞吐 11.89 tok/s,平均输入 186 token / 输出 189 token

pass@1 达 97.56%,与该模型未打 patch 时的正常水平一致,确认 **aiter patch 精度无损**。

> 另有离线精度验证 `tests/verify_aiter_v2.py`,直接对比 aiter 与 vllm kernel 输出的 `cos_sim`,
> 实测 **= 1.000000**(完全一致),从算子层确认精度无损。

---

## 五. 回退

如需回到 vllm 原始状态:

```bash
./patch.sh revert
```

恢复原始 `triton_w4a16.py` 和 aiter 原始 kernel,清缓存。之后直接启动 vllm 即走 baseline。

---

## 附:常见问题

- **启动报 `'_OpNamespace' 'aiter' object has no attribute ...`**:
  旧的 torch.compile 缓存被复用了。执行
  `rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root` 后重启。
  `patch.sh install/revert` 已自动清缓存,只在手动切换环境变量时需要手动清。

- **排查问题时想关掉 aiter**:`VLLM_AITER_W4A16_PATCH=0 vllm serve ...`(装了 patch 但运行时回退到 vllm 原生 triton,用于定位是否 aiter 引入的问题)。

- **TPOT 比参考值(68ms)偏高 5~7ms**:大概率是 DCU 没锁频。按"三·5. 锁频检测"一节
  检查 `rocm-smi --showclocks | grep sclk`,没锁上就手动锁。

- **换到新容器/新机器**:只要 `pip install aiter`(DCU 定制版)和 vllm 已装好,
  重新 `git clone` 本仓库 + `./patch.sh install` 即可,无需拉 aiter 源码。

- **换别的 AWQ w4a16 模型**:kernel 和安装脚本是通用的,只需为该模型调优 config
  (放 `models/<新模型>/configs/awq_w4a16/`),再 `./patch.sh install <新模型>`。
  详见 [`../../README.md`](../../README.md)。
