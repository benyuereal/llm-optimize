#!/usr/bin/env python3
"""验证多 API server 下输出正确性：流式完整性 + 多次结果一致性 + 无重复/丢字。"""
import requests, json

URL = "http://127.0.0.1:8001/v1/audio/transcriptions"
AUDIO = open("/data/asr_en10.wav", "rb").read()
files = {"file": ("audio.wav", AUDIO, "audio/wav"),
         "model": (None, "/data/Qwen3-ASR-1.7B"),
         "stream": (None, "true"),
         "language": (None, "en"),
         "temperature": (None, "0.0")}

results = []
for i in range(5):
    chunks = []
    done = False
    with requests.post(URL, files=files, stream=True, timeout=30) as r:
        assert r.status_code == 200, r.status_code
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                done = True
                break
            obj = json.loads(d)
            c = obj["choices"][0]["delta"].get("content", "")
            chunks.append(c)
    text = "".join(chunks)
    results.append((len(chunks), done, text))
    print(f"req{i}: chunks={len(chunks)} done={done}")
    print(f"  text: {text!r}")

print("\n=== 一致性检查 ===")
texts = [t for _, _, t in results]
uniq = set(texts)
print(f"5 次请求, 唯一文本数: {len(uniq)}")
if len(uniq) == 1:
    print("✓ 完全一致 (deterministic, temperature=0)")
else:
    print("⚠ 结果有差异:")
    for i, t in enumerate(texts):
        print(f"  req{i}: {t!r}")
        if i > 0 and t != texts[0]:
            print(f"    diff vs req0: 长度 {len(t)} vs {len(texts[0])}")

# 检查是否有明显异常：重复片段、乱码
base = texts[0]
print(f"\n最终文本长度: {len(base)} 字符")
print(f"文本: {base!r}")
# 检查是否有连续重复 token（简单启发式）
words = base.split()
dup_runs = sum(1 for i in range(1, len(words)) if words[i] == words[i-1] and words[i])
print(f"连续重复词数: {dup_runs}")
