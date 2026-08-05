# llm-optimize

自建大模型推理部署的优化实践：针对 vLLM + LiteLLM + Claude Code 这套自建链路的实际问题与解决方案。

## 背景

在自建 GPU 推理服务上，用 **vLLM** 部署 `glm-5.2` 模型，通过 **LiteLLM** 做协议转换，让 **Claude Code**（Anthropic 官方 CLI）以 Anthropic 协议接入使用。

```
Claude Code → LiteLLM(协议转换) → vLLM(推理) → glm-5.2
```

为了提升推理吞吐，vLLM 开启了 **MTP（Multi-Token Prediction，投机解码）**。但开启 MTP 后，Claude Code 的工具调用频繁失败。本仓库记录这个问题的诊断与解决过程。

## 目录结构

```
llm-optimize/
├── README.md            # 本文件
├── .gitignore
├── proxy/               # 转发修复代理
│   ├── README.md        # proxy 模块详细文档（问题分析、原理、使用）
│   ├── llm_proxy.py     # 混合流式 + 修复代理（正式版）
│   ├── llm_proxy_v1.py  # 缓冲式修复代理（备用版）
│   ├── probe_proxy.py   # 探测版（调试用，抓取畸形样本）
│   └── requirements.txt # Python 依赖
└── vllm/gemm/w4a16/     # aiter w4a16 GEMM 加速 (Hygon DCU)
    ├── README.md        # 通用 patch 机制说明
    ├── patch.sh         # 一键安装 / 回退
    └── models/gemma4/   # gemma-4-31B-it-AWQ-4bit 调优 + 部署方案
        └── DEPLOY.md    # ← 部署方案 (面向部署人员, 傻瓜式)
```

## vLLM w4a16 GEMM 加速 (Hygon DCU)

另一项优化:在 Hygon DCU BW10 (gfx936) 上,用 aiter triton w4a16 kernel 替换 vllm 自带 kernel,
加速 gemma-4-31B-it-AWQ-4bit 推理,精度无损,端到端 TPOT -25%、吞吐 +31%。

部署人员请直接看 [`vllm/gemm/w4a16/models/gemma4/DEPLOY.md`](vllm/gemm/w4a16/models/gemma4/DEPLOY.md)
(下载 → 一键安装 → 性能验证 → 精度验证,四步完成)。
通用机制说明见 [`vllm/gemm/w4a16/README.md`](vllm/gemm/w4a16/README.md)。

## 核心问题：MTP 导致工具调用参数结尾漂移

### 现象

开启 MTP（`num_speculative_tokens: 2`）后，Claude Code 约 10% 的工具调用报 `Invalid tool parameters`。关闭 MTP 即恢复正常，但 MTP 的推理加速不可放弃。

### 根因：工具调用参数结尾多出 `{}`

问题的本质是 **tool call 的参数 JSON 结尾多出来一对 `{}`，把本该收尾的 `"}` 顶替掉了**，导致 JSON 字符串未闭合。

一个合法的 tool call 参数长这样（结尾是引号 `"` 闭合字符串 + 大括号 `}` 闭合对象）：

```
{"command": "ls", "description": "测试 10"}
                                       ↑ 这两个字符 " } 是收尾
```

MTP 投机解码一次预测 2 个 token，在生成到这种**需要精确结构的结尾处**时，draft 头偶发性地多吐了一对 `{}`，把收尾的 `"}` 挤掉，变成：

```
{"command": "ls", "description": "测试 10{}
                                       ↑ 多了 {}，原本的 " } 没了
```

把畸形和正常对比，差异只在结尾 3 个字符：

```
正常: ...测试 10"}      ← 引号 + 大括号，合法闭合
畸形: ...测试 10{}      ← 多了 {}，引号丢失，字符串未终止
```

Claude Code 收到后按工具 schema 校验，`json.loads` 报 `Unterminated string starting at ...` → 显示 `Invalid tool parameters`，拒绝执行该工具调用。

**关键诊断结论：**
- 错误发生在 **Claude Code 客户端的本地 schema 校验**环节，不是网络或服务端。
- LiteLLM 日志全是 `200 OK`——它只负责转发，不校验参数合法性。
- vLLM 也认为生成成功——它不知道结尾多吐了 `{}`。
- 漂移是**概率性**的（约 10%），且对**较长的参数串**更敏感（结尾 token 越靠后，MTP 猜错概率越高）。
- 漂移形态高度规律：**永远是结尾 `"}` 被 `{}` 取代**，这让它可以被确定性修复。
- **关闭 MTP 即恢复正常**，但 MTP 的推理加速不可放弃。

### 解决方案

在 Claude Code 和 LiteLLM 之间加一层**转发修复代理**：检测畸形的 tool call、原地修复结尾、只把合法响应返回给 Claude Code。MTP 照开，Claude Code 零改动。

正式版（`llm_proxy.py`）采用**混合流式**：文字内容实时流式转发（打字机效果），工具调用参数缓冲到块结束再修复——兼顾流式体感与修复能力。

详见 [proxy/README.md](proxy/README.md)。

## 快速开始

```bash
# 1. 安装依赖
pip install -r llm-optimize/proxy/requirements.txt

# 2. 启动代理（监听 :4001，上游 LiteLLM :4000）
nohup python llm-optimize/proxy/llm_proxy.py > /tmp/llm_proxy.out 2>&1 &

# 3. 让 Claude Code 走代理
export ANTHROPIC_BASE_URL=http://127.0.0.1:4001
claude
```

## 验证效果

- **工具调用成功率**：开启代理后 100%，MTP 漂移被自动修复。
- **流式体感**：文字逐字流出（首字延迟 ~0.16s），非缓冲式（首字延迟 ~1.2s）。
- **MTP 加速保留**：无需关闭 MTP，推理吞吐不受影响。

## 环境信息

- vLLM: 0.18.1（HCU 定制版）
- LiteLLM: 1.93.0
- 模型: glm-5.2
- MTP: `deepseek_mtp`, `num_speculative_tokens: 2`

## License

MIT
