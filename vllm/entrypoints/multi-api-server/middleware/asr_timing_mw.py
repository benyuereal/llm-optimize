"""ASR 请求计时中间件 v2: 量 body 接收 vs multipart 解析。"""
import os
import time

_PROF = os.getenv("VLLM_ASR_PROFILE", "0") == "1"
_LOG = "/tmp/asr_mw_prof.log"


async def asr_timing_middleware(request, call_next):
    if not _PROF:
        return await call_next(request)
    path = request.url.path
    if "transcription" not in path and "translation" not in path:
        return await call_next(request)

    t0 = time.perf_counter()
    # 包一层 receive, 量 body 接收时间
    receive = request._receive
    body_done = [None]
    chunk_count = [0]
    total_bytes = [0]
    first_chunk = [None]

    async def timed_receive():
        msg = await receive()
        if msg.get("type") == "http.request":
            body = msg.get("body", b"")
            if first_chunk[0] is None and body:
                first_chunk[0] = time.perf_counter()
            chunk_count[0] += 1
            total_bytes[0] += len(body)
            if not msg.get("more_body", False):
                body_done[0] = time.perf_counter()
        return msg

    request._receive = timed_receive

    response = await call_next(request)
    t1 = time.perf_counter()

    with open(_LOG, "a") as f:
        fc = first_chunk[0]
        bd = body_done[0]
        first_chunk_ms = (fc - t0) * 1000 if fc else -1
        body_done_ms = (bd - t0) * 1000 if bd else -1
        total_ms = (t1 - t0) * 1000
        f.write(f"first_chunk={first_chunk_ms:.1f} body_done={body_done_ms:.1f} "
                f"parse_plus_handler={total_ms - body_done_ms:.1f} "
                f"total={total_ms:.1f} bytes={total_bytes[0]} chunks={chunk_count[0]}\n")
    return response
