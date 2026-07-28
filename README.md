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
└── proxy/               # 转发修复代理
    ├── README.md        # proxy 模块详细文档（问题分析、原理、使用）
    ├── llm_proxy.py     # 混合流式 + 修复代理（正式版）
    ├── llm_proxy_v1.py  # 缓冲式修复代理（备用版）
    ├── probe_proxy.py   # 探测版（调试用，抓取畸形样本）
    └── start.sh         # vLLM 启动脚本参考
```

## 核心问题：MTP 导致工具调用参数结尾漂移

### 现象

开启 MTP（`num_speculative_tokens: 2`）后，Claude Code 约 10% 的工具调用报 `Invalid tool parameters`。关闭 MTP 即恢复正常，但 MTP 的推理加速不可放弃。

### 根因

MTP 一次投机预测 2 个 token，在生成 **tool call 参数 JSON** 时，结尾本该是 `"}`，偶发被替换成 `{}`，导致 JSON 字符串未闭合：

```
正常: {"command": "ls", "description": "测试 10"}
畸形: {"command": "ls", "description": "测试 10{}
```

Claude Code 校验 `json.loads` 报 `Unterminated string` → `Invalid tool parameters`。

关键诊断结论：
- LiteLLM 日志全是 `200 OK`（只转发，不校验）。
- vLLM 也认为生成成功（不知道结尾漂移）。
- 错误发生在 Claude Code 客户端的本地 schema 校验环节。
- 漂移是概率性的（~10%），对长参数串更敏感。

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
