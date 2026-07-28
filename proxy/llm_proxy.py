"""
转发修复代理（混合流式版 v2）：Claude Code -> LiteLLM。

解决 MTP(num_speculative_tokens>1) 导致的 tool call 参数结尾漂移问题。
模型生成 tool_use 的 input JSON 时，结尾本该是 "}, 偶发被换成 {}, 导致
Claude Code 校验失败报 "Invalid tool parameters"。

混合流式策略：
  - 文字内容(text_delta 等)：实时流式转发（打字机效果，低延迟）
  - tool_use 的参数(input_json_delta)：缓冲到该 tool_use 块结束(content_block_stop),
    校验/修复结尾后，发携带完整修复后 JSON 的 delta, 再转发 stop。
  - 修复策略：结尾模式修复({} -> "} 等)，命中即修；修不了的极端情况原样发出。

关键格式（v1 malformed 的修复）：
  每条 SSE 事件必须是两行：event: <type>\ndata: <json>\n\n
  缺了 event: 行会导致 Claude Code 解析 malformed。

用法：
  python /workspace/llm_proxy.py
让 Claude Code 的 ANTHROPIC_BASE_URL 指向 http://127.0.0.1:4001
"""
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

UPSTREAM = "http://127.0.0.1:4000"   # LiteLLM
LISTEN_PORT = 4001
LOG_PATH = "/tmp/llm_proxy.log"
MAX_RETRIES = 3

app = FastAPI()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sse_event(ev: dict) -> str:
    """构造规范的 SSE 事件：event: <type>\ndata: <json>\n\n"""
    t = ev.get("type", "")
    return f"event: {t}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


# ---------- 畸形修复 ----------
_REPAIR_TAIL_CANDIDATES = [
    ('{}',    '"}'),
    ("{}'",  '"}'),
    ('{}"',   '"}'),
    ('{',     '"}'),
    ('}',     '"}'),
]


def _find_embedded_brace(raw: str):
    """定位'第一个对象未闭合就嵌入下一个 {'的多余 { 位置。找不到返回 None。

    跟踪字符串内外、转义、括号深度。第一个对象开始后(depth>=1)，
    若在 depth==1 且非字符串内遇到 {，视为多余嵌入。
    注：合法嵌套({\"a\":{\"b\":1}})也可能命中，但调用方会用 json.loads 验证兜住误判。
    """
    depth = 0
    in_str = False
    esc = False
    started = False
    for i, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if not started:
                started = True
                depth = 1
            else:
                if depth == 1:
                    return i
                depth += 1
        elif ch in "[":  # noqa: E741
            if started:
                depth += 1
        elif ch in "}]":
            if started:
                depth -= 1
    return None


def try_repair_input(raw: str):
    """尝试修复畸形 input JSON。成功返回修复后字符串，失败/已合法返回 None。"""
    if not raw or not raw.strip():
        return None
    try:
        json.loads(raw)
        return None  # 本就合法
    except Exception:
        pass
    for bad_tail, good_tail in _REPAIR_TAIL_CANDIDATES:
        if raw.endswith(bad_tail):
            cand = raw[: -len(bad_tail)] + good_tail
            try:
                json.loads(cand)
                return cand
            except Exception:
                continue
    for suffix in ['"}', '"}', '"]}', '"]', '}']:
        cand = raw + suffix
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    stripped = raw.rstrip("\"'{}[]:, \t\r\n")
    for suffix in ['"}', '"}', '}']:
        cand = stripped + suffix
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    # 4) 对象未闭合就嵌入下一个 { 的形态（如 {"a":1{"a":1}，第一个对象没闭合就塞进第二个）。
    #    找到那个多余的 {（第一个对象内、depth==1、非字符串内出现的 {），删掉它及之后内容，补 } 闭合。
    #    用字符串/括号状态机定位，但最终用 json.loads 验证，误判会被验证兜住。
    embed_pos = _find_embedded_brace(raw)
    if embed_pos is not None:
        cand = raw[:embed_pos] + "}"
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass
    # 5) 提取第一个完整的合法 JSON 对象（对付"整段重复"等漂移：{...}{...} 或 {...}垃圾）
    #    MTP 偶发会把已生成的参数 JSON 又吐一遍，或尾部掺入垃圾。
    #    可靠做法：从前往后在每个可能的结束位置(} 或 ])尝试 json.loads，
    #    第一个能独立解析为合法 JSON 的前缀即为模型真正想发的参数，丢弃其后内容。
    #    不手写状态机，避免引号/数字配对 bug；直接复用 json 库的解析能力。
    s = raw.lstrip()
    if s.startswith("{") or s.startswith("["):
        for i in range(1, len(raw)):
            if raw[i] in "}]":
                cand = raw[: i + 1]
                try:
                    json.loads(cand)
                    # 仅当后面还有非空白内容时才视为"需要截断"（避免误伤本就合法的完整 JSON）
                    if raw[i + 1:].strip():
                        return cand
                except Exception:
                    continue
    return None


# ---------- 混合流式 ----------
async def hybrid_stream(body, headers):
    """混合流式：文字实时转发，tool_use 参数缓冲到块结束再修复。

    关键：每条事件用 sse_event() 构造，带 event: 前缀行（v1 malformed 的修复）。
    """
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{UPSTREAM}/v1/messages", content=body, headers=headers) as resp:
            tool_input_parts = {}   # index -> [partial_json...]
            tool_meta = {}          # index -> name

            async for line in resp.aiter_lines():
                if not line:
                    continue
                # 上游的 event: 行和 data: 行是分开的。我们重新构造规范格式。
                # 只处理 data: 行（解析事件）；event: 行忽略（我们按 type 重建）
                if line.startswith("event: "):
                    continue  # 跳过上游的 event 行，下面用 data 里的 type 重建
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue

                t = ev.get("type")
                idx = ev.get("index")

                # tool_use 块开始：记录 index，转发（带 event 行）
                if t == "content_block_start":
                    cb = ev.get("content_block", {}) or {}
                    if cb.get("type") == "tool_use":
                        tool_input_parts[idx] = []
                        tool_meta[idx] = cb.get("name", "?")
                    yield sse_event(ev)
                    continue

                # tool_use 的 input 分片：缓冲，不转发
                if t == "content_block_delta" and idx in tool_input_parts:
                    delta = ev.get("delta", {}) or {}
                    if delta.get("type") == "input_json_delta":
                        tool_input_parts[idx].append(delta.get("partial_json", ""))
                        continue  # 不 yield
                    yield sse_event(ev)
                    continue

                # tool_use 块结束：修复缓冲 input，补发完整 delta，再转发 stop
                if t == "content_block_stop" and idx in tool_input_parts:
                    raw_input = "".join(tool_input_parts.pop(idx))
                    name = tool_meta.pop(idx, "?")
                    final_input = raw_input
                    if raw_input.strip():
                        try:
                            json.loads(raw_input)
                        except Exception:
                            fixed = try_repair_input(raw_input)
                            if fixed is not None:
                                final_input = fixed
                                log(f"REPAIR(hybrid) name={name} OK")
                                log(f"  before = {raw_input!r}")
                                log(f"  after  = {fixed!r}")
                            else:
                                log(f"REPAIR(hybrid) name={name} FAILED (原样发出)")
                                log(f"  input  = {raw_input!r}")
                    # 补发携带完整(修复后) input 的 delta
                    fixed_delta_ev = {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": final_input},
                    }
                    yield sse_event(fixed_delta_ev)
                    # 再转发 stop
                    yield sse_event(ev)
                    continue

                # 其它所有事件(文字 text_delta、message_start/stop、message_delta 等)：立即转发
                yield sse_event(ev)


# ---------- 非流式 ----------
def collect_stream_events(resp_lines):
    events = []
    for line in resp_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except Exception:
                pass
    return events


async def fetch_once(client, body, headers, is_stream):
    resp = await client.post(f"{UPSTREAM}/v1/messages", content=body, headers=headers)
    if is_stream:
        raw_lines = []
        async for line in resp.aiter_lines():
            raw_lines.append(line)
        events = collect_stream_events(raw_lines)
        return resp.status_code, dict(resp.headers), raw_lines, events
    else:
        data = resp.content
        try:
            events = [json.loads(data)] if data else []
        except Exception:
            events = []
        return resp.status_code, dict(resp.headers), data, events


def non_stream_repair(data_bytes):
    try:
        data = json.loads(data_bytes)
    except Exception:
        return data_bytes, None
    bad_name = None
    for block in data.get("content", []) or []:
        if block.get("type") == "tool_use":
            inp = block.get("input")
            if isinstance(inp, str):
                try:
                    json.loads(inp)
                except Exception:
                    fixed = try_repair_input(inp)
                    if fixed is not None:
                        block["input"] = json.loads(fixed)
                    else:
                        bad_name = block.get("name")
    return json.dumps(data, ensure_ascii=False).encode("utf-8"), bad_name


# ---------- 路由 ----------
@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    try:
        is_stream = json.loads(body).get("stream", False)
    except Exception:
        is_stream = False

    if is_stream:
        return StreamingResponse(
            hybrid_stream(body, headers),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=None) as client:
        last_raw = None
        last_status = 500
        last_headers = {}
        for attempt in range(1, MAX_RETRIES + 1):
            status, resp_headers, raw, events = await fetch_once(client, body, headers, is_stream)
            fixed_bytes, bad_name = non_stream_repair(raw)
            if bad_name is None:
                log(f"attempt {attempt}: OK (non-stream)")
                return JSONResponse(content=json.loads(fixed_bytes), status_code=status, headers=resp_headers)
            else:
                log(f"attempt {attempt}: BAD (non-stream, unrepairable) name={bad_name}")
                last_raw = raw
                last_status = status
                last_headers = resp_headers
                continue
        log("retries exhausted, returning last response")
        return JSONResponse(content=json.loads(last_raw), status_code=last_status, headers=last_headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(request.method, f"{UPSTREAM}/{path}", content=body, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    open(LOG_PATH, "a", encoding="utf-8").close()
    log(f"转发修复代理(混合流式)启动，监听 :{LISTEN_PORT}，上游 {UPSTREAM}，日志 {LOG_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")