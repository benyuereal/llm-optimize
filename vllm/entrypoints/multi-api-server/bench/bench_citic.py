#!/usr/bin/env python3
"""CITIC 用户视角并发压测脚本 (已对齐本机环境)。

与 bench_decompose.py 互补:
  - 本脚本: 用户视角, 关心 TTFT / RTF / QPS / 识别结果, 混用 OpenAI SDK + requests
  - bench_decompose.py: 工程视角, 拆解 upload/server/decode/total 分位数

改动 (相对 CITIC 原脚本):
  - test_model: /models/... -> /data/Qwen3-ASR-1.7B
  - BASE_URL: 0.0.0.0 -> 127.0.0.1 (0.0.0.0 作客户端目标地址在部分环境连不上)
  - TEST_AUDIO_PATH: /data1/kongmx/... -> /data/asr_en10.wav (10s 音频, 与压测目标一致)
  - CONCURRENT_WORKERS: 5 -> 70
  - 音频读一次到内存复用, 避免 70 线程同时读磁盘污染延迟数据
  - 统计加 p50/p90/max 分位数 (原脚本只有 mean)

用法:
  python3 bench_citic.py                    # 默认 streaming, 70 并发, 10s 音频
  python3 bench_citic.py 70 streaming       # 显式指定
  python3 bench_citic.py 5 basic            # 5 并发 basic 模式
"""
import requests
import time
import librosa
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ==================== 配置 ====================
test_model = "/data/Qwen3-ASR-1.7B"
BASE_URL = "http://127.0.0.1:8001/v1"
TEST_AUDIO_PATH = "/data/asr_en10.wav"   # 10s 音频

# ==================== 并发参数 (命令行可覆盖) ====================
CONCURRENT_WORKERS = 70
TEST_TYPE = "streaming"   # "basic" 或 "streaming"

# 音频读一次到内存, 70 线程复用, 避免同时读磁盘
AUDIO_BYTES = open(TEST_AUDIO_PATH, "rb").read()
AUDIO_DURATION = librosa.get_duration(
    y=librosa.load(TEST_AUDIO_PATH, sr=16000)[0], sr=16000)


def run_basic_transcription(worker_id):
    """单次基础转录 (OpenAI SDK), 返回耗时和 RTF"""
    client = OpenAI(base_url=BASE_URL, api_key="dummy-key")
    start_time = time.time()
    try:
        # 用内存 bytes 模拟文件上传
        import io
        transcription = client.audio.transcriptions.create(
            model=test_model,
            file=("audio.wav", AUDIO_BYTES, "audio/wav"),
            temperature=0.0,
            response_format="text",
        )
        total_time = time.time() - start_time
        rtf = total_time / AUDIO_DURATION
        print(f"[Worker {worker_id}] 识别结果: {str(transcription)[:30]}... "
              f"(耗时 {total_time:.2f}s, RTF {rtf:.4f})")
        return {"worker": worker_id, "time": total_time, "rtf": rtf,
                "text": str(transcription)}
    except Exception as e:
        print(f"[Worker {worker_id}] 请求失败: {str(e)}")
        return {"worker": worker_id, "error": str(e)}


def run_streaming_transcription(worker_id):
    """单次流式转录 (requests), 返回 TTFT / 耗时 / RTF / 文本"""
    audio_start_time = time.time()
    first_token_time = None
    all_text = ""
    url = f"{BASE_URL}/audio/transcriptions"

    try:
        files = {
            "file": ("audio.wav", AUDIO_BYTES, "audio/wav"),
            "model": (None, test_model),
            "stream": (None, "true"),
            "language": (None, "en"),
            "temperature": (None, "0.0"),
        }
        with requests.post(url, files=files, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(data)
                        choices = json_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                all_text += content
                    except json.JSONDecodeError:
                        continue

        process_end_time = time.time()
        total_time = process_end_time - audio_start_time
        rtf = total_time / AUDIO_DURATION
        ttft_ms = (first_token_time - audio_start_time) * 1000 \
            if first_token_time else -1

        print(f"[Worker {worker_id}] 识别结果: {all_text[:30]}... "
              f"(TTFT {ttft_ms:.0f}ms, 总耗时 {total_time:.2f}s, RTF {rtf:.4f})")
        return {
            "worker": worker_id,
            "ttft_ms": ttft_ms,
            "time": total_time,
            "rtf": rtf,
            "text": all_text,
        }
    except Exception as e:
        print(f"[Worker {worker_id}] 请求失败: {str(e)}")
        return {"worker": worker_id, "error": str(e)}


def percentile(vals, p):
    if not vals:
        return -1
    vals = sorted(vals)
    return vals[min(int(len(vals) * p), len(vals) - 1)]


def run_concurrent_tests():
    print(f"=== CITIC 用户视角并发压测 ===")
    print(f"并发数: {CONCURRENT_WORKERS}")
    print(f"测试类型: {TEST_TYPE}")
    print(f"音频文件: {TEST_AUDIO_PATH} ({AUDIO_DURATION:.2f}s, "
          f"{len(AUDIO_BYTES)} bytes)")
    print(f"模型: {test_model}")
    print("-" * 60)

    overall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        if TEST_TYPE == "basic":
            futs = {executor.submit(run_basic_transcription, i): i
                    for i in range(CONCURRENT_WORKERS)}
        else:
            futs = {executor.submit(run_streaming_transcription, i): i
                    for i in range(CONCURRENT_WORKERS)}
        for future in as_completed(futs):
            res = future.result()
            if res:
                results.append(res)

    total_wall_time = time.time() - overall_start

    # ---------- 汇总统计 ----------
    print("\n" + "=" * 60)
    print("=== 压测汇总报告 ===")
    print(f"总并发数: {CONCURRENT_WORKERS}")
    print(f"总耗时 (所有并发完成): {total_wall_time:.2f} s")

    success_results = [r for r in results if "error" not in r]
    error_count = len(results) - len(success_results)

    if success_results:
        times = [r["time"] * 1000 for r in success_results]   # ms
        rtfs = [r["rtf"] for r in success_results]
        print(f"成功任务数: {len(success_results)}")
        print(f"失败任务数: {error_count}")
        print(f"单请求耗时 (ms): p50={percentile(times,0.5):.0f} "
              f"p90={percentile(times,0.9):.0f} max={max(times):.0f} "
              f"mean={st.mean(times):.0f}")
        print(f"平均 RTF: {st.mean(rtfs):.4f}")

        if TEST_TYPE == "streaming":
            ttfts = [r["ttft_ms"] for r in success_results
                     if r.get("ttft_ms", -1) > 0]
            if ttfts:
                print(f"首字延迟 TTFT (ms): p50={percentile(ttfts,0.5):.0f} "
                      f"p90={percentile(ttfts,0.9):.0f} max={max(ttfts):.0f} "
                      f"mean={st.mean(ttfts):.0f}")

        qps = len(success_results) / total_wall_time
        print(f"吞吐量 (QPS): {qps:.4f} 请求/秒")

        # 输出一致性检查 (所有成功请求的文本是否一致)
        texts = [r.get("text", "") for r in success_results]
        uniq = set(texts)
        print(f"\n输出一致性: {len(uniq)} 种唯一结果 "
              f"({'✓ 完全一致' if len(uniq) == 1 else '⚠ 有差异'})")
        if len(uniq) == 1:
            print(f"  识别结果: {texts[0]!r}")
    else:
        print("所有请求均失败，请检查服务状态。")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        CONCURRENT_WORKERS = int(sys.argv[1])
    if len(sys.argv) >= 3:
        TEST_TYPE = sys.argv[2]
    run_concurrent_tests()
