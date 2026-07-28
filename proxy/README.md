# LLM 转发修复代理（proxy）

一个夹在 **Claude Code** 和 **LiteLLM** 之间的轻量代理，用于解决自建 vLLM + MTP（投机解码）场景下，工具调用（tool call）参数结尾漂移导致 Claude Code 报 `Invalid tool parameters` 的问题。

## 背景与问题

### 部署架构

```
Claude Code → LiteLLM(协议转换) → vLLM(推理, MTP 开启) → glm-5.2
```

- vLLM 用 `glm-5.2` 模型，开启了 MTP 投机解码（`num_speculative_tokens: 2`）以加速推理。
- LiteLLM 把 Anthropic 协议转成 OpenAI 协议再转发给 vLLM。
- Claude Code 通过 `ANTHROPIC_BASE_URL` 指向 LiteLLM 使用。

### 问题现象

开启 MTP 后，Claude Code 频繁报 `Invalid tool parameters`，约 10% 的工具调用失败。

### 根因

MTP 投机解码一次预测多个 token，在生成 **tool call 的参数 JSON** 时，结尾本该是 `"}`（引号闭合 + 大括号闭合），偶发性地被替换/截断成 `{}`，导致 JSON 字符串未闭合。

实测抓到的畸形样本：

```
正常: {"command": "ls", "description": "测试 10"}
畸形: {"command": "ls", "description": "测试 10{}   ← 结尾 "} 变成了 {}
```

Claude Code 收到后按 schema 校验，`json.loads` 报 `Unterminated string`，于是显示 `Invalid tool parameters` 并拒绝执行。

关键点：
- LiteLLM 日志全是 `200 OK`（它只转发，不校验参数）。
- vLLM 也认为生成成功（它不知道结尾漂移了）。
- 只有 Claude Code 做了 schema 校验，才暴露问题。
- 漂移是概率性的（约 10%），且对较长的参数串更敏感。
- **关闭 MTP 即恢复正常**，但 MTP 带来的推理加速不可放弃。

## 解决方案

在 Claude Code 和 LiteLLM 之间加一层 **转发修复代理**：缓冲响应、检测畸形的 tool call、原地修复结尾、只把合法的响应返回给 Claude Code。MTP 照开，Claude Code 零改动。

```
Claude Code → 转发修复代理(:4001) → LiteLLM(:4000) → vLLM(:8001, MTP 开)
                  ↑
           检测畸形 tool call → 原地修复结尾 → 返回合法响应
```

### 混合流式策略（llm_proxy.py，正式版）

为了在修复的同时保留流式体感（打字机效果），采用混合流式：

- **文字内容（`text_delta`）**：实时流式转发，低延迟，打字机效果。
- **工具调用参数（`input_json_delta`）**：缓冲到该 tool_use 块结束（`content_block_stop`），校验/修复结尾后，一次性发携带完整修复后 JSON 的 delta，再转发 stop。
- **修复策略**：结尾模式修复（`{}` → `"}` 等若干候选），命中即修；修不了的极端情况原样发出。

流式协议的关键细节：每条 SSE 事件必须是两行 `event: <type>\ndata: <json>\n\n`，缺了 `event:` 行会导致 Claude Code 解析 `malformed`。

## 文件说明

| 文件 | 说明 |
|---|---|
| `llm_proxy.py` | **正式版**，混合流式 + 修复。文字实时流，工具参数缓冲修复。默认用这个。 |
| `llm_proxy_v1.py` | 缓冲版（备用）。收完整个响应再修复返回，更稳但无打字机效果。 |
| `probe_proxy.py` | 探测版（调试用）。只透传并记录所有 tool_use 到日志，不修复、不重试。用于抓取畸形样本。 |
| `start.sh` | vLLM 启动脚本参考（含 MTP 配置）。 |

## 安装与使用

### 依赖

```bash
pip install -r requirements.txt
```

依赖包（见 `requirements.txt`）：`fastapi`、`uvicorn`、`httpx`。

### 启动代理

```bash
nohup python /workspace/llm-optimize/proxy/llm_proxy.py > /tmp/llm_proxy.out 2>&1 &
```

默认监听 `:4001`，上游指向 `http://127.0.0.1:4000`（LiteLLM）。

### 让 Claude Code 走代理

在启动 Claude Code 前设置环境变量（**必须在启动前设置，运行中改不生效**）：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4001
claude
```

### 日志

- `/tmp/llm_proxy.log` — 修复记录（触发修复时才写）
- `/tmp/llm_proxy.out` — 启动与报错日志

### 确认代理存活

```bash
python -c "import socket; s=socket.socket(); s.settimeout(2)
try:
    s.connect(('127.0.0.1',4001)); print('4001 OPEN')
except Exception as e: print('4001 FREE:', e)"
```

## 验证流式

```python
import time, httpx
t0 = time.time(); chunks = []
with httpx.Client(timeout=None) as c:
    with c.stream("POST", "http://127.0.0.1:4001/v1/messages", json={
        "model": "glm5.2", "max_tokens": 200, "stream": True,
        "messages": [{"role": "user", "content": "从1数到50，每个数字单独一行"}],
    }) as resp:
        for chunk in resp.iter_bytes():
            if chunk: chunks.append(time.time() - t0)
print(f"块数={len(chunks)} 首块={chunks[0]:.3f}s 末块={chunks[-1]:.3f}s 跨度={chunks[-1]-chunks[0]:.3f}s")
# 真流式：块数多、跨度>1s、首块<0.3s
# 缓冲式：块数=1、跨度≈0、首块>1s
```

## 回滚

如混合流式正式版出问题，切回复冲版：

```bash
# 停掉混合流式
pkill -f llm_proxy.py
# 用缓冲版启动（注意改文件名/端口）
nohup python /workspace/llm-optimize/proxy/llm_proxy_v1.py > /tmp/llm_proxy_v1.out 2>&1 &
# Claude Code 指向缓冲版端口（需先在 llm_proxy_v1.py 里改 LISTEN_PORT）
```

## 配置项

在 `llm_proxy.py` 顶部：

```python
UPSTREAM = "http://127.0.0.1:4000"   # LiteLLM 地址
LISTEN_PORT = 4001                    # 代理监听端口
LOG_PATH = "/tmp/llm_proxy.log"       # 日志路径
MAX_RETRIES = 3                       # 非流式重试次数（仅缓冲版/非流式分支用）
```

## 已知限制

- **不支持 WebSearch 工具**：`WebSearch` 依赖 Anthropic 服务端搜索能力，自建 vLLM 无此能力，且 LiteLLM 把 `web_search_options` 转成 vLLM 不认的裸 `tool_choice` 字段会触发 400。这是协议/能力问题，非本代理能修复。
- **WebFetch 受 Claude Code 域名校验限制**：与代理无关。
- 修复针对的是结尾 `{}` 漂移这一种主要形态；其它罕见形态靠重试兜底（缓冲版）或原样发出（混合流式版）。
