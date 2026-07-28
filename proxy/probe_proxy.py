"""
探测版中间层：透传 Claude Code -> LiteLLM，记录所有 tool_use 块到日志。
不做重试，仅用于抓取畸形 tool call 的真面目。

用法：
  python /workspace/probe_proxy.py
然后让 Claude Code 的 ANTHROPIC_BASE_URL 指向 http://127.0.0.1:4001
撞到 Invalid tool parameters 后，看 /tmp/toolcall_probe.log
"""
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

UPSTREAM = "http://127.0.0.1:4000"   # LiteLLM
LISTEN_PORT = 4001
LOG_PATH = "/tmp/toolcall_probe.log"

app = FastAPI()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_tool_uses(events: list[dict]) -> list[dict]:
    """从已收集的 SSE 事件里，重组出完整的 tool_use 块。

    Claude/Anthropic 流式协议里，一个 tool_use 的 input 是分片到达的：
      message_start
      content_block_start  { index, content_block: { type: tool_use, id, name, input: {} } }
      content_block_delta  { delta: { type: input_json_delta, partial_json: "..." } }  (多次)
      content_block_stop   { index }
    我们按 index 把 partial_json 拼起来，得到完整 input。
    """
    starts: dict[int, dict] = {}
    input_parts: dict[int, list[str]] = {}
    for ev in events:
        t = ev.get("type")
        if t == "content_block_start":
            idx = ev.get("index")
            cb = ev.get("content_block", {}) or {}
            if cb.get("type") == "tool_use":
                starts[idx] = {"id": cb.get("id"), "name": cb.get("name"), "input_raw": cb.get("input", {})}
                input_parts[idx] = []
        elif t == "content_block_delta":
            idx = ev.get("index")
            delta = ev.get("delta", {}) or {}
            if delta.get("type") == "input_json_delta" and idx in input_parts:
                input_parts[idx].append(delta.get("partial_json", ""))
    results = []
    for idx, meta in starts.items():
        raw_input = "".join(input_parts.get(idx, []))
        results.append({
            "index": idx,
            "id": meta["id"],
            "name": meta["name"],
            "input_str": raw_input,
            "input_initial": meta["input_raw"],
        })
    return results


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    is_stream = json.loads(body).get("stream", False)

    async def stream_and_collect():
        collected: list[dict] = []
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{UPSTREAM}/v1/messages", content=body, headers=headers) as resp:
                # 透传状态码与响应头
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    yield line + "\n"
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                            collected.append(ev)
                        except Exception:
                            pass
        # 流结束后分析
        tool_uses = extract_tool_uses(collected)
        for tu in tool_uses:
            ok = True
            err = ""
            try:
                if tu["input_str"].strip():
                    json.loads(tu["input_str"])
            except Exception as e:
                ok = False
                err = f"INVALID_JSON: {e}"
            flag = "OK" if ok else f"BAD {err}"
            log(f"tool_use name={tu['name']} {flag}")
            log(f"  input_str = {tu['input_str']!r}")
            if not ok:
                log(f"  >>> 畸形 tool call，长度={len(tu['input_str'])}")

    if is_stream:
        return StreamingResponse(stream_and_collect(), media_type="text/event-stream")
    else:
        # 非流式：透传并解析
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{UPSTREAM}/v1/messages", content=body, headers=headers)
            try:
                data = resp.json()
                for block in data.get("content", []):
                    if block.get("type") == "tool_use":
                        inp = block.get("input", {})
                        log(f"tool_use(non-stream) name={block.get('name')} input={json.dumps(inp, ensure_ascii=False)[:500]}")
            except Exception:
                pass
        return JSONResponse(content=resp.json(), status_code=resp.status_code, headers=dict(resp.headers))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(request.method, f"{UPSTREAM}/{path}", content=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    open(LOG_PATH, "a", encoding="utf-8").close()
    log(f"探测中间层启动，监听 :{LISTEN_PORT}，上游 {UPSTREAM}，日志 {LOG_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
