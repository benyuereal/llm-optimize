#!/usr/bin/env python3
"""用 requests 拆分(和 benchmark.py 同栈),量上传/服务端/decode。
requests.post(stream=True) 会在 body 发送完毕后返回,所以:
  t0: post 调用前
  t1: with 块进入(body 发送完毕)
  t2: 第一个 data 行
  t3: 结束标记
"""
import time
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

test_model = "/data/Qwen3-ASR-1.7B"
BASE_URL = "http://127.0.0.1:8001/v1"
TEST_AUDIO_PATH = "/data/asr_en10.wav"
CONCURRENT_WORKERS = 70
AUDIO_BYTES = open(TEST_AUDIO_PATH, "rb").read()
DONE_MARKER = "[DONE]"


def one_request(wid):
    t0 = time.perf_counter()
    files = {
        "file": ("audio.wav", AUDIO_BYTES, "audio/wav"),
        "model": (None, test_model),
        "stream": (None, "true"),
        "language": (None, "en"),
        "temperature": (None, "0.0"),
    }
    upload_done = None
    first_byte = None
    done_time = None
    try:
        with requests.post(f"{BASE_URL}/audio/transcriptions",
                           files=files, stream=True, timeout=60) as resp:
            upload_done = time.perf_counter()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                if first_byte is None:
                    first_byte = time.perf_counter()
                data = line[6:]
                if data == DONE_MARKER:
                    done_time = time.perf_counter()
                    break
        if done_time is None:
            done_time = time.perf_counter()
        upload = (upload_done - t0) * 1000
        server = (first_byte - upload_done) * 1000 if first_byte else -1
        decode = (done_time - first_byte) * 1000 if first_byte else -1
        total = (done_time - t0) * 1000
        return {"wid": wid, "upload": upload, "server": server,
                "decode": decode, "total": total}
    except Exception as e:
        return {"wid": wid, "error": str(e)}


def main():
    print(f"c={CONCURRENT_WORKERS}, audio={len(AUDIO_BYTES)} bytes (requests)")
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as ex:
        futs = {ex.submit(one_request, i): i for i in range(CONCURRENT_WORKERS)}
        for f in as_completed(futs):
            results.append(f.result())
    ok = [r for r in results if "error" not in r]
    print(f"success {len(ok)}/{len(results)}")
    for key in ["upload", "server", "decode", "total"]:
        vals = sorted(r[key] for r in ok if r[key] > 0)
        if not vals:
            print(f"  {key:8s}: n=0")
            continue
        n = len(vals)
        print(f"  {key:8s}: p50={vals[n//2]:.0f} p90={vals[int(n*0.9)]:.0f} "
              f"max={vals[-1]:.0f} mean={st.mean(vals):.0f} ms")


if __name__ == "__main__":
    main()
