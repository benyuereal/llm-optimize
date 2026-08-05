# aiter w4a16 GEMM 加速 · 部署方案

在 Hygon DCU BW10 (gfx936) 上,用 aiter triton w4a16 kernel 替换 vllm 自带 kernel,
加速 gemma-4-31B-it-AWQ-4bit 推理,**精度无损**,端到端 **TPOT -25%、吞吐 +31%**。

本方案面向部署人员,按下面 4 步操作即可,无需理解内部实现。

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
cd llm-optimize/vllm/gemm/w4a16
```

---

## 二. 一键安装

```bash
./patch.sh install gemma4
```

这一条命令会自动完成:
1. 备份 vllm 原始文件(用于回退)
2. 替换 vllm 的 `triton_w4a16.py` 为 aiter patch 版
3. 覆盖 aiter 的 w4a16 kernel(改成 triton 3.5 兼容版)
4. 放置 gemma4 的调优 config(10 个 json)
5. 清空编译缓存

装好后查看状态确认:

```bash
./patch.sh status gemma4
```

看到 `vllm triton_w4a16.py : aiter patch 版` 和 `gemma4 config : 10 / 10 个已放置` 即成功。

> **默认启用**:装完 patch 后,直接用你原来的 `vllm serve` 命令启动即可,aiter kernel 自动生效,
> 无需设置任何环境变量。

---

## 三. 启动 vllm 并做性能验证

### 1. 启动 vllm

用你原来的启动命令即可(示例):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
  --tensor-parallel-size 4 \
  --served-model-name gemma4 \
  ... (你原有的其他参数)
```

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

### 3. 跑 baseline 对比(可选)

想确认加速效果,可关掉 aiter 再测一遍 baseline:

```bash
# 关掉 aiter, 走 vllm 原生 triton w4a16
rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root   # 必须清缓存
VLLM_AITER_W4A16_PATCH=0 vllm serve /data/zq/models/gemma-4-31B-it-AWQ-4bit/ ...
```

启动后用同样的 bench 命令再测一次,对比两轮的 TPOT / duration。

### 参考性能数据(本方案实测,稳态第二轮)

| 配置 | duration | TPOT | TTFT | 吞吐 (tok/s) |
|------|----------|------|------|-------------|
| baseline (vllm 原生) | 94.45s | 90.72ms | 1.10s | 43.37 |
| **aiter patch** | **72.11s** | **68.46ms** | 1.19s | **56.80** |
| 提升 | -23.6% | **-24.6%** | 持平 | **+31%** |

环境:TP=4,MTP 投机解码(num_speculative_tokens=3),optimization-level 3。

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

结果在 `./outputs/` 下。对比 baseline 和 patch 的 pass@1,应基本一致。

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

- **想临时关掉 aiter 做对比**:`VLLM_AITER_W4A16_PATCH=0 vllm serve ...`(装了 patch 但运行时不走 aiter)。

- **换到新容器/新机器**:只要 `pip install aiter`(DCU 定制版)和 vllm 已装好,
  重新 `git clone` 本仓库 + `./patch.sh install gemma4` 即可,无需拉 aiter 源码。

- **换别的 AWQ w4a16 模型**:kernel 和安装脚本是通用的,只需为该模型调优 config
  (放 `models/<新模型>/configs/awq_w4a16/`),再 `./patch.sh install <新模型>`。
  详见 [`vllm/gemm/w4a16/README.md`](vllm/gemm/w4a16/README.md)。
