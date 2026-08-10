#!/usr/bin/env python3
"""验证 --api-server-count=4 下,70 并发请求的输出是否完全一致。
关键点:70 个请求会被内核 SO_REUSEPORT 分散到 4 个 API server 进程,
要确认不同进程处理出来的结果一字不差。
"""
import requests, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

URL = "http://127.0.0.1:8001/v1/audio/transcriptions"
AUDIO = open("/data/asr_en10.wav", "rb").read()
N = 70
files_tmpl = {"file": ("audio.wav", AUDIO, "audio/wav"),
              "model": (None, "/data/Qwen3-ASR-1.7B"),
              "stream": (None, "true"),
              "language": (None, "en"),
              "temperature": (None, "0.0")}


def one(wid):
    files = {k: (v[0], v[1] if isinstance(v[1], bytes) else v[1], v[2])
             if isinstance(v, tuple) and len(v) == 3 else v
             for k, v in files_tmpl.items()}
    # 重新构造 files (bytes 不可复用)
    files = {"file": ("audio.wav", AUDIO, "audio/wav"),
             "model": (None, "/data/Qwen3-ASR-1.7B"),
             "stream": (None, "true"),
             "language": (None, "en"),
             "temperature": (None, "0.0")}
    chunks = []
    done = False
    err = None
    try:
        with requests.post(URL, files=files, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return wid, False, [], f"HTTP {r.status_code}"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    done = True
                    break
                obj = json.loads(d)
                chunks.append(obj["choices"][0]["delta"].get("content", ""))
    except Exception as e:
        return wid, False, [], str(e)
    return wid, done, chunks, None


results = {}
with ThreadPoolExecutor(max_workers=N) as ex:
    futs = {ex.submit(one, i): i for i in range(N)}
    for f in as_completed(futs):
        wid, done, chunks, err = f.result()
        results[wid] = (done, chunks, err)

# 汇总
oks = [w for w in results if results[w][0] and results[w][2] is None]
errs = [w for w in results if results[w][2] is not None]
no_done = [w for w in results if not results[w][0] and results[w][2] is None]
print(f"成功 {len(oks)}/{N}, 错误 {len(errs)}, 无DONE {len(no_done)}")
if errs:
    for w in errs[:5]:
        print(f"  wid{w} err: {results[w][2]}")

# 比较所有成功请求的文本
texts = {}
for w in oks:
    texts[w] = "".join(results[w][1])

uniq = Counter(texts.values())
print(f"\n唯一文本数: {len(uniq)} (期望=1)")
if len(uniq) == 1:
    print("✓ 70 并发请求输出完全一致")
    txt = list(uniq.keys())[0]
    print(f"  chunks 数: {len(results[oks[0]][1])}")
    print(f"  文本: {txt!r}")
else:
    print("⚠ 输出有差异:")
    for t, cnt in uniq.most_common():
        print(f"  出现{cnt}次: {t!r}")

# 检查 chunk 数是否一致
chunk_counts = Counter(len(results[w][1]) for w in oks)
print(f"\nchunk 数分布: {dict(chunk_counts)} (期望全部相同)")
