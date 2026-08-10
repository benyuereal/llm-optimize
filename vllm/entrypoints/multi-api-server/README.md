# 多 API server 进程并行化 (ASR CPU 前端优化)

CITIC 信用卡中心 ASR 性能优化。**Qwen3-ASR-1.7B** 跑在海光 DCU **BW150 (gfx936)** 单卡,
tp=1, bf16 无损。优化 vLLM 的 **`entrypoints/` 模块**（HTTP 前端 + 启动编排层）,
与 [`gemm/w4a16/`](../../gemm/w4a16/)（算子层）是不同层、正交的两条优化轴。

## 目标与战果

| | 值 |
|---|---|
| **目标** | 70 并发, 10s 音频, 端到端 < 500ms |
| **起点** | 1012ms |
| **当前** | **~440ms p50** ✅ 达标 |
| **总降幅** | -57%, 无损 |

正确性已验证: 70 并发请求输出**逐字一致** (temperature=0 确定性), 35 chunk + `[DONE]` 完整,
无丢字/重复/乱码。多 API server 只是 HTTP 层加进程, 推理走同一 EngineCore 同一权重, 逻辑零改动。

## 核心方案: `--api-server-count N`

**零源码改动**。vLLM 原生参数 (`vllm/entrypoints/cli/serve.py:run_multi_api_server`) 启动 N 个
API server 进程, 共享**同一个 EngineCore** (单卡单引擎, dp=1), 通过 ZMQ 通信,
**同一端口 SO_REUSEPORT** (内核负载均衡连接)。

```
                端口 8001 (SO_REUSEPORT, 内核负载均衡)
                          │
        ┌────────┬────────┼────────┬────────┐
   ApiServer_0  _1     _2      _3     (各持独立 GIL + event loop)
        └────────┴────────┼────────┴────────┘
                          │  ZMQ (共享同一个引擎)
                     EngineCore (1个, 单卡 tp=1)
                          │
                       BW150 GPU
```

**为什么有效**: 单 uvicorn 进程 = 1 event loop = 1 GIL。70 并发的 multipart body 解析
(~155ms, python-multipart 纯 Python、GIL-bound) + body 接收 (~130ms) 全在单线程串行。
N 个进程 = N 个独立 GIL + event loop → multipart 解析真正并行 (绕过 GIL)。
等价于 vDCU 多实例的并行化效果, 但**不切分卡、不需 nginx、单条 `vllm serve` 命令、单端口**。

**为什么不用线程池**: python-multipart 纯 Python、无 C 扩展、完全 GIL-bound, 线程池无法
真并行, 70 线程争 GIL 反而 context-switch 开销 → 引入超时。多进程才是正解 (已验证)。

## 饱和点定位 (不能无限堆进程)

c=70, 10s 音频, requests 客户端 (127.0.0.1), 稳态 p50ms:

| 进程数 | upload(CPU) | server | decode(GPU) | total | 边际收益 |
|--------|------------|--------|-------------|-------|---------|
| 1 (baseline) | 302 | 61 | 366 | 764 | — |
| 2 | 238 | 57 | 310 | 595 | -169 |
| 3 | 135 | 57 | 314 | 513 | -82 |
| **4** | **50** | **52** | **328** | **440** | **-73 ← 甜点** |
| 6 | 37 | 64 | 331 | 435 | -5 ← 饱和 |

- **upload (CPU multipart 解析+body 接收)**: 4 进程压到 50ms 基本榨干。这是 `--api-server-count`
  能压的部分。
- **decode (GPU 推理)**: 302→328→331 纹丝不动, 占 total 75%。GPU 带宽硬底线,
  加 API server 无济于事。
- **甜点 = 4**: CPU 瓶颈消除后, 剩下全是不受 API server 控制的 GPU decode。
  再堆进程只浪费内存 (每进程 ~4GB RSS)。

> decode 的 330ms 是下一阶段战场, 需从 GPU 侧入手 (FP8 权重量化 / GEMM skinny 优化),
> 与本方案正交, 见 [`gemm/`](../../gemm/)。

## 目录结构

```
entrypoints/multi-api-server/
├── README.md                       # 本文件
├── patch.sh                        # 一键 install / revert / status (speech_to_text.py)
├── speech_to_text.py.orig          # vllm 原始版 (CRLF, 对齐海光定制版行尾)
├── speech_to_text.py.patch         # 标准 diff patch (3 hunks)
├── speech_to_text.py.patched       # 改后版
├── deploy/                         # 启动配置谱系 (按优化阶段编号)
│   ├── 01_baseline.sh              #   原始 (带 profiler)
│   ├── 02_maxnumseqs128.sh         #   --max-num-seqs 128
│   ├── 03_rocm_transpose_weight.sh #   + VLLM_ROCM_TRANSPOSE_WEIGHT=1
│   ├── 04_fp8kv_rejected.sh        #   FP8 KV cache (实测变慢 7%, 已否决, 留存对照)
│   └── 05_multi_api_server.sh      #   ★ 甜点: --api-server-count 4 (当前生产)
├── middleware/
│   └── asr_timing_mw.py            # 计时中间件 (量 mw_total, 受 VLLM_ASR_PROFILE 控制)
└── bench/                          # 基准与正确性验证
    ├── bench_decompose.py          #   c=70 拆分 upload/server/decode/total (requests 客户端)
    ├── verify_concurrent.py        #   70 并发输出一致性验证
    ├── verify_correctness.py       #   串行输出正确性验证
    └── cer.py                      #   字错误率
```

## speech_to_text.py patch (辅助改动)

主优化 `--api-server-count` 零源码改动, 但探索中对 `speech_to_text.py` 做了两处辅助改动:

1. **音频解码 `asyncio.to_thread`**: `librosa.load` 用 `asyncio.to_thread` 移出 event loop,
   避免阻塞单线程 event loop, 与多进程方案配合。
2. **profile 埋点**: 受 `VLLM_ASR_PROFILE=1` 控制 (默认关, 零开销), 写
   `/tmp/asr_route_prof.log`, 量 `preprocess` / `first_output` 耗时。

```bash
./patch.sh install    # 安装
./patch.sh revert     # 回退
./patch.sh status     # 查看状态
```

**兼容性**: vllm 0.18.1 DCU 定制版 (`0.18.1+das.dtk2604`)。海光定制版 `speech_to_text.py`
与官方 0.18.1 完全一致 (除 CRLF 行尾), `.orig` 取自官方 v0.18.1 tag 并转 CRLF 对齐。

## 部署

```bash
# 1. 打 speech_to_text patch (可选, 主优化不依赖它)
cd entrypoints/multi-api-server && ./patch.sh install

# 2. 用甜点配置启动 (1 引擎 + 4 API server, 单端口 8001)
bash deploy/05_multi_api_server.sh

# 想看耗时分 解:
VLLM_ASR_PROFILE=1 bash deploy/05_multi_api_server.sh
```

**前置条件**:
- vllm 0.18.1 DCU 定制版 (`0.18.1+das.dtk2604`)
- Hygon DCU BW150 (gfx936), 单卡
- 模型 `/data/Qwen3-ASR-1.7B`

## 探索中被否决的方案

| 方案 | 结论 | 原因 |
|------|------|------|
| 线程池并行 multipart 解析 | ✗ | python-multipart 纯 Python GIL-bound, 线程池无法并行, 反引入超时 |
| nginx 软多实例 (同卡多端口) | ✗ | 验证思路 OK 但不能交付 (要外部组件), `--api-server-count` 更优雅 |
| data parallel (DP) | ✗ | 只有一张卡 |
| FP8 KV cache (`--kv-cache-dtype fp8`) | ✗ | 实测变慢 7%, dequant 开销 > 带宽收益 |
| GPU 做音频解码 | ✗ | WAV 是解析非压缩解码, CPU 5ms 足够; GPU ffmpeg 场景不适用 |
