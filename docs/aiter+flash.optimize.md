# gemma-4-31B-it-AWQ-4bit · BW10 DCU 性能优化技术总结

> 本文档总结在 Hygon DCU BW10 (gfx936) 上优化 gemma-4-31B-it-AWQ-4bit 推理性能的完整技术方案,
> 供团队总结回顾与后续参阅。部署操作手册见各子目录 README,本文聚焦**技术原理、关键决策、踩过的坑**。

---

## 1. 项目概况

### 1.1 目标

在 Hygon DCU BW10 (gfx936, 48 CUs, 32GB/卡) 上,基于 vllm 0.23.0 DCU 定制版,
将 gemma-4-31B-it-AWQ-4bit 的推理延迟降到最低,生产 TP4 部署,精度无损。

### 1.2 环境

| 组件 | 版本 |
|------|------|
| 硬件 | Hygon DCU BW10 (gfx936), 48 CUs, 32GB/卡, 5 卡 (生产用 0,4,2,3) |
| vllm | 0.23.0 DCU 定制版 (`0.23.0+das.48722a8.dtk2604`) |
| aiter | DCU 定制版 (`0.1.3+das.dtk2604`, pip 安装) |
| torch | 2.10.0 |
| python | 3.10 |
| 主模型 | gemma-4-31B-it-AWQ-4bit (compressed-tensors, AWQ uint4, group_size=32, asymmetric) |
| draft 模型 | gemma-4-31B-it-assistant (MTP, model_type=gemma4_assistant) |
| 投机解码 | MTP, num_speculative_tokens=3 (生产) / 5 (试验) |

### 1.3 模型结构关键参数

主模型 gemma4:60 层,hidden=5376,GQA 32:16,head_dim=256,sliding_window=1024,intermediate=14336。
MTP draft 模型有一层 `full_attention`:**global_head_dim=512**(注意是 512,不是主模型的 256),
num_global_key_value_heads=4 —— 这一层是阶段二的优化重点。

### 1.4 最终成果

| 配置 | TPOT | 相对 baseline | 吞吐 (tok/s) | MTP 接受率 |
|------|------|--------------|-------------|-----------|
| baseline (vllm 原生 triton w4a16) | 90.72ms | 1.00x | 43.37 | 96.85% |
| 仅阶段一 (aiter w4a16 GEMM) | 67.76ms | 1.34x | 57.59 | 98.30% |
| **阶段一+二 (叠加 flash)** | **35.02ms** | **2.59x** | **107.02** | 98.08% |

- TPOT 降低 **61.4%**,吞吐提升 **147%**
- 精度无损:HumanEval pass@1 = 97.56%,离线 cos_sim = 1.000000
- 测试条件:TP4,batch 4,input 5120 / output 1024,MTP num_spec=3

---

## 2. 优化路线总览

两阶段叠加,互不冲突,分别攻破两个瓶颈:

| 阶段 | 目录 | 优化对象 | 攻克的瓶颈 | 效果 |
|------|------|---------|-----------|------|
| 阶段一 | `vllm/aiter-w4a16/` | aiter triton w4a16 GEMM 替换 vllm 自带 kernel | AWQ 反量化 GEMM (decode 主战场) | TPOT -25% |
| 阶段二 | `vllm/flash-attn/` | flash fp8 KV + head_dim=512 替换 aiter 2D attention | draft full_attention (head_dim=512) | TPOT 再 -48% |

### 2.1 瓶颈定位方法

用 vllm profiler 抓 decode step 的算子耗时分布,发现:
- **awq_gemm 占 52.5%** —— 真正的第一瓶颈,在阶段一解决
- attention 占比小,但 draft 侧的 head_dim=512 full_attention 走 aiter 2D kernel **无 tuned config**,
  单次 14.7ms/call,是 draft 侧的性能洼地 —— 阶段二解决
- 改动前先确认"改的算子确实是瓶颈",避免无效优化

---

## 3. 阶段一:aiter w4a16 GEMM

### 3.1 问题

vllm 自带的 triton w4a16 kernel 在 BW10 上性能不佳。aiter (AMD/Hygon DCU 专用算子库)
有更快的 triton w4a16 kernel,但接入 vllm 需要解决权重重排、量化兼容、torch.compile 兼容等问题。

### 3.2 方案

用 aiter 的 `gemm_a16w4` triton kernel 替换 vllm 的 `triton_w4a16.py`,通过
`torch.library.custom_op` 包装成注册算子,对 torch.compile / cudagraph 全透明。

改动三层(patch 三段合一):
1. **vllm 侧** (`triton_w4a16.py`):GPTQ→AWQ 列重排、对称量化 zp 兼容、原始 qweight 释放、custom_op 接入
2. **aiter 侧** (`gemm_a16w4.py`):triton 3.5 兼容修复(`@triton.utils.jit`→`@triton.jit`)
3. **config**:10 个按 gemma4 各层 shape 调优的 config json

### 3.3 关键技术点

#### 3.3.1 GPTQ → AWQ 列重排(精度关键)

vllm 的 AWQ 权重是 GPTQ int32 打包 `[K, N//8]`,顺序 `[0,1,...7]`。
aiter 的 `reverse_awq_order` 用 `AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]` 解包。
送进 aiter 前必须按 **`AWQ_INV=[0,2,4,6,1,3,5,7]`** 重排列:
- 不做 → cos_sim=0.25(完全错)
- 做了 → cos_sim=1.000000

```python
_AWQ_INV = [0, 2, 4, 6, 1, 3, 5, 7]
# b_q_gptq[:, idx.reshape(-1)] 重排, idx 由 _AWQ_INV 构造
```

#### 3.3.2 对称量化 (uint4b8) 兼容(精度关键)

aiter kernel 的 dequant 算式 `(w - zeros) * scales` **必须有 zeros 张量**,
但对称量化 (uint4b8) 没有显式 zp(零点恒为 8)。解法:模型加载时造一个全 8 的 zp 张量,
等价于 `(w - 8) * scale`。8 个 4bit zp pack 进一个 int32 = `0x88888888`。

gemma4 是 asymmetric(有真实 zp),用真实 zp。

#### 3.3.3 torch.compile / cudagraph 兼容(性能关键)

aiter triton driver 调用 inductor 编译不了。用 `torch.library.custom_op` +
`register_fake` 把 aiter kernel 包成注册算子,对 dynamo/inductor/cudagraph 全透明:
- vllm 的 torch.compile (opt-level 3) + cudagraph 49 张图全捕获成功
- aiter kernel 在图内执行,无 graph break

#### 3.3.4 原始 qweight 释放(省 3.5GB/卡,解 TP1 OOM)

repack 成 aiter 格式 (aq/az) 后,原始 `qweight [K,N//8] int32` + `qzeros`(~3.5GB/卡)
不再需要。repack 完立即 `replace_parameter(layer, w_q_name, None)` 释放:
- TP4 每卡省 0.88GB(锦上添花)
- **TP1 单卡 32GB 装 31B 模型,这 3.5GB 是 OOM 与不 OOM 的区别**

#### 3.3.5 bf16 模型 + `--dtype float16`(兼容关键)

aiter kernel 硬编码 fp16 输出。bf16 权重不加处理会 `Half != BFloat16` 崩溃
(尤其在 torch.compile/cudagraph 下)。解法:启动加 `--dtype float16` 全链路对齐 fp16。
- 零 kernel 改动
- 精度无损:gemma4 scale max=2.1875、权重 max=53760(远低于 fp16 上限 65504),bf16→fp16 无溢出,
  fp16 尾数(10位)比 bf16(7位)更细
- 两种模型(对称/非对称)都加 `--dtype float16` 最安全

#### 3.3.6 config 调优(BW10 专属)

BW10 有 48 CUs。关键参数:
- `BLOCK_SIZE_M=16, BLOCK_SIZE_K=32, NG(NUM_GROUPS)=1`(BK=32/NG=1 是安全配置,
  **BK=64/NG=2 在 BW10 上触发 VMFault 崩溃**)
- 小形状(N≤3584):`BLOCK_SIZE_N=64, SPLITK=2`
- 大形状(N=4096/5120/10752, K=4096/5376):`BLOCK_SIZE_N=128` + 自适应 SPLITK(按 M 取 3~16)
- `num_warps=4, NUM_CUS=48`

未调优的 aiter 默认 config 比 vllm 还慢 10-28%;调优后比 vllm 快 1.6-2.33x(kernel 级,M=4)。

> **踩坑:SPLITK 离线调优对 kernel 级提速 ~1.5x,但 decode 端到端 GEMM 时间波动 2-3x 会淹没收益。
> SPLITK 主要保证大形状不慢、不崩,端到端 TPOT 收益来自 w4a16 kernel 本身。**

### 3.4 阶段一性能

| 配置 | TPOT | 吞吐 | 接受率 | TTFT |
|------|------|------|--------|------|
| baseline | 90.72ms | 43.37 tok/s | 96.85% | 1.10s |
| aiter patch (tuned) | 67.76ms | 57.59 tok/s | 98.30% | 1.06s |
| 提升 | **-25.3%** | +32.8% | +1.45pp | 持平 |

---

## 4. 阶段二:flash attention fp8 KV + head_dim=512

### 4.1 问题

MTP draft 模型的 `full_attention` 层 global_head_dim=512,原来走 aiter 的 2D kernel:
- aiter 2D 对 head_dim=512 **无 tuned config**,单次 14.7ms/call
- 是 draft 侧的性能洼地

同时,主模型 attention 走 aiter unified_attention 的 triton 2D 路径,也想换成更快的 flash mixed kernel。

### 4.2 方案

扩展 `flash-attention-cutlass` 源码,使其 fp8 mixed kernel 支持 head_dim=512,
通过 aiter 的 unified_attention 路由到 flash。改三层:

1. **flash-attention-cutlass 源码**(编译成 whl):fp8 e5m2/e4m3 mixed kernel + head_dim=512 支持
2. **vllm 源码**(3 文件):让 vllm 支持 `fp8_e5m2` KV cache
3. **aiter 源码**(1 文件 `unified_attention.py`):把 decode/prefill attention 路由到 flash

### 4.3 关键技术点

#### 4.3.1 fp8_e5m2 KV cache 放行(vllm 侧)

上游 vllm 对 compressed-tensors 模型**一律禁用** `fp8_e5m2` KV cache(在 `attention.py` 无条件报错
`fp8_e5m2 kv-cache is not supported with fp8 checkpoints.`)。但我们的 AWQ-4bit 模型 checkpoint
里并没有 fp8 KV scale(`kv_cache_scheme=None`),只是运行时想把 KV 存成 e5m2。

改 3 个文件:
- `attention.py` — 放行 e5m2(去掉对 AWQ 模型的无条件报错)
- `rocm_aiter_unified_attn.py` — 读侧 view 用 e5m2 + 写侧走 triton(C++ `reshape_and_cache_flash` op 不支持 e5m2)
- `triton_reshape_and_cache_flash.py` — 写侧按字符串选 e5m2 dtype

> **选 e5m2 而非 e4m3 的原因**:fp8 KV 三路径(e4m3 triton / e5m2 triton / e5m2 flash)全测,
> e5m2 flash 最快且精度可接受。e4m3 reshape 和 e5m2 性能相同(85us),但 e5m2 精度差 2 倍
> (MAE 0.036 vs 0.018);最终 HumanEval 验证 e5m2 路径精度无损(97.56% vs fp16 96.95%)。

#### 4.3.2 Q-cast 解决 dtype 不匹配(aiter 侧)

flash 的 prefix decode kernel 要求 Q/K 同 dtype。fp8 KV 时 K 是 e5m2,Q 是 fp16,直接传会报
`RuntimeError: For prefix decode, query and key must have the same dtype`。

解法:fp8 KV 时把 Q cast 成 KV dtype 再传,并附 unit descale(纯 bit-cast,数值无影响):
```python
_q_flash = q.to(k.dtype)              # fp16 Q → e5m2
_unit_descale = _get_unit_descale(q.device)  # 全 1 descale, 数值无影响
varlen_fwd_unified(q=_q_flash, k=k, v=v, ..., descale_q=_unit_descale, ...)
```

#### 4.3.3 flash 路由条件(aiter 侧)

`unified_attention.py` 加三个环境变量控制路由:
- `ATTN_FLASH_PREFILL=1`:prefill 走 flash(`_flash_q_len_limit=999999`)
- `ATTN_FLASH_HEAD512=1`:draft head_dim=512 的 decode 走 flash
- MTP q_len≤8:decode 走 flash(自动)

> **关键坑:`ATTN_FLASH_PREFILL=0` 会崩!** 设 0 想让 prefill 回退 aiter triton 2D,
> 但 aiter triton 2D 在 fp8 KV prefill 时有编译 bug
> (`'constexpr_type' object has no attribute 'is_ptr'`,e4m3 和 e5m2 都崩)。
> **phase2 (ROCM_AITER_UNIFIED_ATTN + fp8 KV) 的 prefill 必须走 flash,没有 triton 退路。**

#### 4.3.4 head_dim=512 flash kernel 实现细节

flash-attention-cutlass 源码改 19 个文件(17 改 + 2 新建),分两层:

**Layer A(fp8 e4m3 mixed kernel 支持,10 文件)**:让 gfx936 的 fp8 mixed kernel 同时支持
e5m2 和 e4m3 KV(e4m3 走软件 `__e4m32float` dequant,复用 e5m2 的 compact-LDS pipeline)。
核心是给 cast 函数加 `InputElement` 模板参数,if-else 选 e5m2/e4m3。

**Layer B(head_dim=512 入口,3 改 + 2 新建)**:解除 flash 入口的 TORCH_CHECK 限制,
让 fp8 + 512 在 decode/prefill 双路径走 flash mixed kernel。新建 2 个 target 文件
实例化 `run_fp8_mha_fwd_prefix_prefill_<Float16/BFloat16, 512, 512>`
(FP16_SWITCH 运行时分发需要 fp16+bf16 两个符号都在,漏 bf16 会链接报 undefined symbol)。

> **编译只编译 gfx936,不编译 gfx938**:gfx938 的 fp8 GEMM utils 不支持 head_dim=512,会 static_assert。

#### 4.3.5 路由链路

draft head_dim=512 验证通过的完整路由:
- aiter `unified_attention()` + `ATTN_FLASH_HEAD512=1` → `varlen_fwd_unified`
- **decode**:num_kv_heads=1 不满足 paged fast path → `hg_prefix_decode_varlen_fwd` →
  `run_mha_fwd_splitkv_dispatch<512,512>` → `use_gfx936_fp8_bf16(512)=true` →
  `run_flash_mixed_splitkv_fwd_tile16x32`
- **prefill**:fp8+512 不满足 fast path → `hg_prefix_prefill_varlen_fwd` →
  `run_mha_fwd` → flash_c_api.h 512 分支 → `run_fp8_mha_fwd_prefix_prefill_<512,512>` →
  `run_flash_mixed_fwd_prefix_prefill_launcher`

### 4.4 阶段二性能

| 指标 | triton 基准 | flash (叠加) | 提升 |
|------|-----------|-------------|------|
| Mean TPOT (batch4) | ~79ms | 35.02ms | ~2.25x |
| MTP 接受率 | ~96% | 98.08% | 持平 |
| 接受长度 | ~3.8 | 3.94 | 持平 |
| HumanEval pass@1 | 96.95% | 96.95% | 精度无损 |

kernel 级对比(prefill q=512):flash_fp16=0.338ms,flash_fp8=1.001ms,triton=1.465ms ——
flash 比 triton 快 1.46-4.33x。

> **已知回退:TTFT**。phase2 用 ROCM_AITER_UNIFIED_ATTN 后端,prefill 必须走 flash
> (triton 2D 在 fp8 KV 下崩)。flash prefill 虽然 kernel 级更快,但端到端 TTFT 从
> phase1 的 1.06s 涨到 ~2.3s(Q-cast、路由开销)。这是该路径的固有代价,无 env 可绕,
> 因为 `ATTN_FLASH_PREFILL=0` 的 triton 退路有编译 bug。生产以 TPOT 为主要指标,TTFT 回退可接受。

---

## 5. 生产启动配置

`vllm/flash-attn/models/gemma4/start_flash.sh` 关键配置(两阶段叠加):

```bash
export HIP_VISIBLE_DEVICES=0,4,2,3          # TP4 用的 4 块卡 (锁高性能模式)
export VLLM_AITER_W4A16_PATCH=1             # 阶段一: aiter w4a16 GEMM
export ATTN_FLASH_PREFILL=1                 # 阶段二: prefill 走 flash (必须=1, =0 会崩)
export ATTN_FLASH_HEAD512=1                 # 阶段二: draft head512 decode 走 flash

vllm serve "$MODEL_DIR" \
    --served-model-name gemma4 \
    --dtype float16 \                       # 对齐 aiter kernel fp16 输出
    --kv-cache-dtype fp8_e5m2 \             # 阶段二: fp8 KV
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \  # 阶段二: aiter unified (路由到 flash)
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --optimization-level 3 \                # torch.compile + cudagraph
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --language-model-only \
    --async-scheduling \                    # 零气泡调度
    --performance-mode throughput \
    --max-num-batched-tokens 16384 \
    --speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL_DIR\", \"num_speculative_tokens\": 3}"
```

阶段一 → 阶段二的关键参数差异:

| 参数 | 阶段一 (start.sh) | 阶段二 (start_flash.sh) |
|------|-------------------|------------------------|
| `--attention-backend` | TRITON_ATTN | **ROCM_AITER_UNIFIED_ATTN** |
| `--kv-cache-dtype` | fp8 | **fp8_e5m2** |
| `--dtype` | (未显式) | **float16** |
| 环境变量 | VLLM_AITER_W4A16_PATCH=1 | +ATTN_FLASH_PREFILL=1 +ATTN_FLASH_HEAD512=1 |

---

## 6. 锁频(性能关键)

DCU 默认按负载动态调频,decode 阶段负载低时频率掉下来,导致 TPOT 偏高 5~7ms、吞吐掉 ~7%。
**必须把用的 4 块卡锁到 sclk level 6 (760MHz)**。`start_flash.sh` 启动时自动锁频并打印确认,
但某些容器下 rocm-smi 锁频会静默失败(权限不足),需手动检测:

```bash
rocm-smi --showclocks 2>&1 | grep sclk
# 正常: 4 张卡都 level 6 (760Mhz)
# 异常: level 0 (300Mhz) 或 Perf 列是 auto
```

手动锁频(不用重启 vllm):
```bash
for i in 0 4 2 3; do
  rocm-smi -d $i --setperflevel manual
  rocm-smi -d $i --setsclk 6
done
```

---

## 7. 投机解码 (MTP) 调参分析

### 7.1 num_speculative_tokens 评估

基于实测 per-position 接受率(num_spec=3 时 98.56/97.98/97.69%,每位置衰减 ~0.44%):

| num_spec | 接受长度 | 预估 TPOT | 预估提升 | 位置N接受率 |
|----------|---------|----------|---------|------------|
| 3 (生产) | 3.94 | 35.02ms | 基准 | pos2=97.7% |
| 4 | 4.91 | 30~32ms | 3~14% | pos3≈97.3% |
| 5 (试验) | 5.88 | 27~33ms | 6~23% | pos4≈96.8% |

- 接受率衰减平缓,第 4/5 个 draft token 仍有 96-97% 接受率,提 num_spec 有正收益
- num_spec=5 实测整体接受率 90.2%(位置3/4 衰减到 82-83%),batch 大时 step 变重
- **建议:生产先试 num_spec=4**(零成本,改一个数字),看实测再决定是否上 5

### 7.2 dspark 评估结论(不采用)

调研了 dspark(并行起草,半自回归)是否值得替换 MTP,结论 **不采用**:
- 生产 vllm 0.23.0 无 dspark 代码(需升级到未发布 dev 版,丢失 DCU 定制)
- 无 gemma4 dspark checkpoint(官方未发布,dspark 需专门训练的 Markov head + confidence head)
- dspark 需 non-causal attention,当前 ROCM_AITER_UNIFIED_ATTN 不支持(supports_non_causal=False)
- MTP 接受率已 98%,瓶颈在 GEMM 非 attention,dspark 收益不匹配成本

---

## 8. 稳定性问题(进行中)

### 8.1 现象

生产压测(batch_eval.py,默认 concurrency=5、max_tokens=4096、中prompt)跑 **30 分钟左右**
服务卡死:rocm-smi 显示 VRAM 96%、HCLI 100% 四卡全满,但不接受新请求。

### 8.2 分析

- **不是死锁,是慢速资源泄漏**:HCLI=100% 说明 GPU 在持续算(没挂起),30 分钟累积后某资源耗尽
- **已排除**:KV cache 满(容量 16707×64=106万 token,5 并发只占 3%)、即时死锁(30 分钟才卡)、num_spec=5 本身(短时正常)
- **最可能**:GPU VRAM 缓慢泄漏 / prefix cache 无限累积 / host 内存泄漏 / async-scheduling 长时 bug
- 32 分钟内未在本地复现(done 稳步增长到 106,err=0,VRAM/RSS 恒定),需更长压测或更多触发条件

### 8.3 复现监控

复现脚本 `test/reproduce_stuck.py`(短时)和 `test/leak_monitor.py`(长时后台+资源监控),
每 30s 采样 VRAM/RSS/队列/drafts/HCU%。关键观察指标:
- done 停止增长但 pending 持续(卡死信号)
- VRAM/RSS 持续上涨(泄漏信号)
- run 高但 drafts 停止增长(调度卡住)

---

## 9. 关键踩坑记录

| 坑 | 现象 | 根因 | 解法 |
|----|------|------|------|
| GPTQ/AWQ 列序 | cos_sim=0.25 | GPTQ 打包顺序与 aiter 解包顺序不一致 | 按 `AWQ_INV=[0,2,4,6,1,3,5,7]` 重排 |
| VMFault 崩溃 | TP2 崩溃 | BK=64/NG=2 在 BW10 触发 VMFault | BK=32/NG=1 安全配置 |
| Half != BFloat16 | bf16 模型崩溃 | aiter kernel 硬编码 fp16 输出 | `--dtype float16` |
| is_ptr 编译错误 | ATTN_FLASH_PREFILL=0 崩 | aiter triton 2D 在 fp8 KV prefill 有编译 bug | prefill 必须走 flash(=1) |
| Q/K dtype 不匹配 | prefix decode 崩 | fp8 KV 时 Q(fp16)≠K(e5m2) | Q cast 成 KV dtype + unit descale |
| 旧 compile 缓存 | `'_OpNamespace' 'aiter' object has no attribute` | 切换 patch 后旧缓存被复用 | 清 `torch_compile_cache` + `torchinductor_root` |
| DCU 降频 | TPOT 偏高 5~7ms | decode 负载低时动态降频 | 锁 sclk level 6 |
| TP1 OOM | 单卡装不下 31B | repack 后原始权重双份 | repack 完释放原始 qweight(省 3.5GB) |
| TTFT 回退 | 1.06s→2.3s | phase2 prefill 必走 flash 无 triton 退路 | 接受(TPOT 为主要指标) |

---

## 10. 文件索引

### 10.1 优化代码

| 路径 | 说明 |
|------|------|
| `vllm/aiter-w4a16/aiter.patch` | 阶段一 patch(vllm triton_w4a16 + aiter kernel + 10 config,三段合一) |
| `vllm/aiter-w4a16/patch.sh` | 阶段一安装脚本(install/revert/status) |
| `vllm/aiter-w4a16/models/gemma4/configs/awq_w4a16/` | gemma4 调优 config(10 个 json) |
| `vllm/flash-attn/flash_fp8e5m2.patch` | 阶段二 vllm 侧 patch(3 文件) |
| `vllm/flash-attn/flash_aiter_fp8e5m2.patch` | 阶段二 aiter 侧 patch(1 文件 unified_attention.py) |
| `vllm/flash-attn/flash-attn.patch` | flash-attention-cutlass 源码 patch(19 文件,编译 whl 用) |
| `vllm/flash-attn/patch.sh` | 阶段二安装脚本(装 whl + 打 vllm/aiter patch) |
| `vllm/flash-attn/dist/flash_attn-2.8.3+das.opt1.dtk2604-*.whl` | 编译好的 flash whl(GitHub Release) |
| `flash-attention-cutlass/` | flash-attention-cutlass 源码视角(改动详情、编译方法) |

### 10.2 文档

| 路径 | 说明 |
|------|------|
| `vllm/aiter-w4a16/README.md` | 阶段一通用说明(kernel 机制、技术要点) |
| `vllm/aiter-w4a16/models/gemma4/README.md` | gemma4 部署(两阶段完整流程、性能、精度、调优) |
| `vllm/flash-attn/README.md` | 阶段二说明(三层改动、快速开始) |
| `vllm/flash-attn/models/gemma4/DEPLOY.md` | 阶段二部署指南 + bench 命令 |
| `flash-attention-cutlass/README.md` | flash 源码改动详情(Layer A/B 逐文件) |
| **本文档** `PERF-OPTIMIZATION.md` | **技术总结(本文)** |

### 10.3 启动脚本

| 路径 | 说明 |
|------|------|
| `vllm/aiter-w4a16/models/gemma4/start.sh` | 阶段一启动脚本(TRITON_ATTN) |
| `vllm/flash-attn/models/gemma4/start_flash.sh` | **生产启动脚本(两阶段叠加)** |
| `flash-attention-cutlass/scripts/start_flash.sh` | flash 侧启动脚本(同上) |

---

## 11. 性能验证方法

### 11.1 端到端 bench

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm bench serve \
    --backend vllm --base-url http://localhost:8001 \
    --model gemma4 --tokenizer /data/zq/models/gemma-4-31B-it-AWQ-4bit/ \
    --dataset-name random --random-input-len 5120 --random-output-len 1024 \
    --num-prompts 4 --seed 42
```

- 连跑两轮取第二轮稳态(第一轮含 warmup)
- 关键指标:Mean TPOT、Mean TTFT、Benchmark duration
- **不要加 `--request-rate 1`**(排队时间算进 TTFT 造成虚高)

### 11.2 精度验证

```bash
# HumanEval
evalscope eval --model gemma4 --api-url http://127.0.0.1:8001/v1/chat/completions \
    --api-key EMPTY --eval-type openai_api --datasets humaneval \
    --eval-batch-size 16 --generation-config '{"temperature":0.2,"top_p":0.95,...}'

# 离线算子级
python vllm/aiter-w4a16/tests/verify_aiter_v2.py   # cos_sim 应 = 1.000000
```

### 11.3 实时监控

```bash
# 接受率 per-position
curl -s http://127.0.0.1:8001/metrics | grep "per_pos"

# 队列状态
curl -s http://127.0.0.1:8001/metrics | grep "num_requests"

# GPU
rocm-smi --showuse --showmeminfo vram
```
