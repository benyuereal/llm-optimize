#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反量化开销隔离测试 (决定 flash+fp8KV 能否拿到 2x).

目的: 量出 "fp8 KV 反量化" 这一步本身贵不贵.
方法: 让 fp16 和 fp8 KV 走同一条 splitkv 路径, 差值 = 纯反量化+fp8读取开销.

源码事实 (已核对):
  - vllm_mha_varlen_fwd (fp16, paged KV): force_split_kernel=paged_KV=true
    -> run_mha_fwd -> num_splits<=1 && !force_split_kernel=false -> 走 splitkv
    (flash_api.cpp:4209, flash_c_api.cpp:91)
  - vllm_mha_varlen_fwd_kv_fp8 (fp8 e5m2): 强制 splitkv
    -> run_mha_fwd_prefix_kv_fp8 -> run_mha_fwd_splitkv_dispatch_kv_fp8
    (flash_api.cpp:3497, :906-921)
  - 两者都走 splitkv, 路径一致, 差值即反量化开销.

判定:
  B/A 接近 1.0 (B 慢 <15%) -> 反量化几乎零开销, flash+fp8KV 有望 2x, 值得改 kernel
  B/A 明显 > 1.0 (B 慢 >40%) -> 反量化贵, 2x 悬, 需另想方案

gemma4: heads=32/16, head_size=256, block=64 (fp8 kernel 硬性要求 page_block_size==64)

用法:
  HIP_VISIBLE_DEVICES=1 python bench_dequant_overhead_isolation.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)
FP8_KV_DTYPE = torch.float8_e5m2  # vllm_mha_varlen_fwd_kv_fp8 硬性要求 e5m2

import flash_attn_2_cuda as fa


def make_inputs(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    """paged KV cache. Q 固定 fp16.
    k layout: [num_blocks, num_kv_heads, page_block_size, head_size]
    v layout: [num_blocks, num_kv_heads, head_size, page_block_size]  (v 转置)
    """
    num_query_tokens = num_seqs * q_len
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    num_blks = num_seqs * num_blks_per_seq

    q = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                    dtype=torch.float16, device=device)
    k16 = torch.randn(num_blks, NUM_KV_HEADS, block_size, HEAD_SIZE,
                      dtype=torch.float16, device=device)
    v16 = torch.randn(num_blks, NUM_KV_HEADS, HEAD_SIZE, block_size,
                      dtype=torch.float16, device=device)
    k = k16 if kv_dtype == torch.float16 else k16.to(kv_dtype)
    v = v16 if kv_dtype == torch.float16 else v16.to(kv_dtype)
    out = torch.empty(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                      dtype=torch.float16, device=device)

    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, num_seqs * kv_len + 1, kv_len,
                                dtype=torch.int32, device=device)
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    block_table = torch.arange(0, num_blks, dtype=torch.int32,
                               device=device).reshape(num_seqs, num_blks_per_seq)
    return q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, block_table


def make_descale(device="cuda"):
    return torch.ones(1, dtype=torch.float32, device=device)


def call_fp16(q, k, v, out, csq, csk, suk, bt, q_len, kv_len, window_left=SLIDING_WINDOW - 1):
    """vllm_mha_varlen_fwd (fp16 KV, paged -> splitkv). flash_api.cpp:3860."""
    fa.vllm_mha_varlen_fwd(
        q, k, v, out,
        csq, csk, suk,
        None,       # leftpad_k_
        bt,         # block_table_
        None,       # alibi_slopes_
        q_len, kv_len,
        0.0,        # p_dropout
        SOFTMAX_SCALE,
        False,      # zero_tensors
        True,       # is_causal
        window_left,  # window_size_left
        0,          # window_size_right
        0.0,        # softcap
        False,      # return_softmax
        None, None, None,  # q/k/v_descale (fp16 不需要)
        None,       # gen_
        None,       # s_aux_ (gemma4 无 sinks)
    )
    return out


def call_fp8kv(q, k, v, out, csq, csk, suk, bt, q_len, kv_len, window_left=SLIDING_WINDOW - 1):
    """vllm_mha_varlen_fwd_kv_fp8 (fp8 e5m2 KV, 强制 splitkv). flash_api.cpp:3497."""
    kd = make_descale(q.device)
    vd = make_descale(q.device)
    fa.vllm_mha_varlen_fwd_kv_fp8(
        q, k, v, out,
        csq, csk, suk,
        None,       # leftpad_k_
        bt,         # block_table_
        None,       # alibi_slopes_
        q_len, kv_len,
        0.0,        # p_dropout
        SOFTMAX_SCALE,
        False,      # zero_tensors
        True,       # is_causal
        window_left,  # window_size_left
        0,          # window_size_right
        0.0,        # softcap
        False,      # return_softmax
        None,       # q_descale_ (Q fp16)
        kd,         # k_descale_
        vd,         # v_descale_
        None,       # gen_
        None,       # s_aux_ (gemma4 无 sinks)
    )
    return out


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="只跑一个场景快速验证")
    args = ap.parse_args()

    print(f"反量化开销隔离测试 (fp16 splitkv vs fp8KV splitkv, 同路径)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, sw={SLIDING_WINDOW}")
    print(f"  A=vllm_mha_varlen_fwd(fp16 KV, paged->splitkv)")
    print(f"  B=vllm_mha_varlen_fwd_kv_fp8(e5m2 KV, 强制 splitkv)")
    print(f"  block=64 (fp8 kernel 硬性要求), 两者都走 splitkv, 差值=反量化开销")
    print(f"  device: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 110)

    BLOCK = 64  # vllm_mha_varlen_fwd_kv_fp8 要求 page_block_size==64
    if args.smoke:
        scenarios = [("decode kv=2048", args.seqs, 1, 2048)]
    else:
        scenarios = [
            ("decode  kv=512",  args.seqs, 1,   512),
            ("decode  kv=1024", args.seqs, 1,   1024),
            ("decode  kv=2048", args.seqs, 1,   2048),
            ("decode  kv=4096", args.seqs, 1,   4096),
            ("decode  kv=8192", args.seqs, 1,   8192),
            ("prefill q=512",   1,         512, 512),
            ("prefill q=1024",  1,         1024,1024),
            ("prefill q=2048",  1,         2048,2048),
        ]

    # ============ 性能对比 ============
    print(f"{'场景':<18} {'A:fp16 splitkv':>15} {'B:fp8KV splitkv':>16} {'B/A(慢的倍数)':>16} {'反量化开销':>12}")
    print("-" * 110)

    results = []
    for name, ns, ql, kl in scenarios:
        t_a = t_b = float("nan")
        try:
            q, k, v, out, csq, csk, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            t_a = bench(lambda: call_fp16(q, k, v, out, csq, csk, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [A fp16 {name} ERR] {e}")
        try:
            q, k, v, out, csq, csk, suk, bt = make_inputs(ns, ql, kl, BLOCK, FP8_KV_DTYPE)
            t_b = bench(lambda: call_fp8kv(q, k, v, out, csq, csk, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [B fp8 {name} ERR] {e}")

        bva = (t_b / t_a) if (t_a == t_a and t_b == t_b and t_a > 0) else float("nan")
        overhead = (bva - 1.0) * 100 if bva == bva else float("nan")
        results.append((name, t_a, t_b, bva))
        print(f"{name:<18} {t_a:>14.3f}ms {t_b:>15.3f}ms {bva:>15.2f}x {overhead:>10.1f}%")

    # ============ 判定 ============
    print()
    print("=" * 110)
    print("判定 (基于 decode 场景 B/A):")
    print("=" * 110)
    decode_results = [r for r in results if "decode" in r[0] and r[3] == r[3]]
    if decode_results:
        avg_bva = sum(r[3] for r in decode_results) / len(decode_results)
        if avg_bva < 1.15:
            verdict = "✅ 反量化几乎零开销 (<15%), flash+fp8KV 有望拿到 2x, 值得改 kernel"
        elif avg_bva < 1.40:
            verdict = "⚠️ 反量化有开销 (15-40%), flash+fp8KV 可能拿到 1.3-1.7x, 改 kernel 收益打折"
        else:
            verdict = "❌ 反量化很贵 (>40%), 2x 拿不到, 别白改, 需另想方案 (如只 V 用 fp8/K 保 fp16)"
        print(f"  decode 平均 B/A = {avg_bva:.2f}x")
        print(f"  {verdict}")
    else:
        print("  无有效 decode 结果, 无法判定")

    # ============ 精度: B(fp8 KV) vs A(fp16 KV) ============
    print()
    print("=" * 110)
    print("精度对比: B(fp8 e5m2 KV) vs A(fp16 KV) 基准 (Q fp16, 无 sinks)")
    print("=" * 110)
    print(f"{'场景':<18} {'max_abs':>12} {'mean_abs':>12} {'mean_rel':>12}  判定")
    print("-" * 110)

    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("prefill q=512", 1, 512, 512),
                              ("prefill q=2048", 1, 2048, 2048)]:
        try:
            torch.manual_seed(42)
            q, k16, v16, out_a, csq, csk, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            call_fp16(q, k16, v16, out_a, csq, csk, suk, bt, ql, kl)
            k8 = k16.to(FP8_KV_DTYPE)
            v8 = v16.to(FP8_KV_DTYPE)
            out_b = torch.empty_like(out_a)
            call_fp8kv(q, k8, v8, out_b, csq, csk, suk, bt, ql, kl)

            diff = (out_a.float() - out_b.float()).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            eps = 1e-3
            rel = diff / torch.clamp(out_a.float().abs(), min=eps)
            mean_rel = rel.mean().item()
            ok = max_abs < 0.5
            verdict = "✅ 一致" if ok else "⚠️ 偏大"
            print(f"{name:<18} {max_abs:>12.5f} {mean_abs:>12.6f} {mean_rel*100:>11.4f}%  {verdict}")
        except Exception as e:
            print(f"{name:<18} ERROR: {e}")

    print("-" * 110)
    print("说明: B/A 接近 1.0 = 反量化便宜 (省带宽还没算进来, 真实 flash+fp8KV 还会再快一截)")


if __name__ == "__main__":
    main()
