#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接调 kernel 对比三条路径 (绕过 wrapper), 回答两个问题:
  Q1. 切 flash_attn 值不值?  -> A(flash fp16+sinks) vs C(triton fp16+sinks, 当前生产)
  Q2. fp8 KV 值不值?         -> B(flash fp8+sinks) vs A(flash fp16+sinks)

路径 (都 Q=fp16, 都带 sinks 模拟 gemma4):
  A. flash_attn vllm_mha_varlen_fwd       fp16 KV + sinks  (flash_api.cpp:3860)
  B. flash_attn vllm_mha_varlen_fwd_kv_fp8 e5m2 KV + sinks (flash_api.cpp:3497)
  C. aiter triton unified_attention        fp16 KV + sinks  (强制 triton 分支)

gemma4: heads=32/16, head_size=256, sliding_window=1024
block=64 (flash fp8 kernel 硬性要求 page_block_size==64, 源码行3574; 三路统一用 64 公平对比)

注意 layout 差异:
  flash_attn: k=[nb,nh_k,blk,hd], v=[nb,nh_k,hd,blk]  (v 转置)
  triton:     k=[nb,blk,nh_k,hd], v=[nb,blk,nh_k,hd]  (k/v 同 layout)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_flash_vs_triton_vs_fp8.py
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


def make_sinks(device="cuda"):
    """sinks [num_query_heads]. flash 要求 fp16 (==Q dtype); triton 接 fp32."""
    return torch.zeros(NUM_QUERY_HEADS, dtype=torch.float16, device=device)


def make_descale(device="cuda"):
    return torch.ones(1, dtype=torch.float32, device=device)


# ---------------- flash_attn 路径 (k=[nb,nh_k,blk,hd], v=[nb,nh_k,hd,blk]) ----------------
def make_inputs_flash(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    num_q_tokens = num_seqs * q_len
    nbps = (kv_len + block_size - 1) // block_size
    nb = num_seqs * nbps
    q = torch.randn(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    k16 = torch.randn(nb, NUM_KV_HEADS, block_size, HEAD_SIZE, dtype=torch.float16, device=device)
    v16 = torch.randn(nb, NUM_KV_HEADS, HEAD_SIZE, block_size, dtype=torch.float16, device=device)
    k = k16 if kv_dtype == torch.float16 else k16.to(kv_dtype)
    v = v16 if kv_dtype == torch.float16 else v16.to(kv_dtype)
    out = torch.empty(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    csq = torch.arange(0, num_q_tokens + 1, q_len, dtype=torch.int32, device=device)
    csk = torch.arange(0, num_seqs * kv_len + 1, kv_len, dtype=torch.int32, device=device)
    suk = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    bt = torch.arange(0, nb, dtype=torch.int32, device=device).reshape(num_seqs, nbps)
    return q, k, v, out, csq, csk, suk, bt


def call_flash_fp16(q, k, v, out, csq, csk, suk, bt, ql, kl):
    sinks = make_sinks(q.device)
    fa.vllm_mha_varlen_fwd(
        q, k, v, out, csq, csk, suk, None, bt, None, ql, kl,
        0.0, SOFTMAX_SCALE, False, True, SLIDING_WINDOW - 1, 0, 0.0, False,
        None, None, None, None, sinks)


def call_flash_fp8(q, k, v, out, csq, csk, suk, bt, ql, kl):
    sinks = make_sinks(q.device)
    kd = vd = make_descale(q.device)
    fa.vllm_mha_varlen_fwd_kv_fp8(
        q, k, v, out, csq, csk, suk, None, bt, None, ql, kl,
        0.0, SOFTMAX_SCALE, False, True, SLIDING_WINDOW - 1, 0, 0.0, False,
        None, kd, vd, None, sinks)


# ---------------- triton 路径 (k=v=[nb,blk,nh_k,hd]) ----------------
def make_inputs_triton(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    num_q_tokens = num_seqs * q_len
    nbps = (kv_len + block_size - 1) // block_size
    nb = num_seqs * nbps
    q = torch.randn(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    k16 = torch.randn(nb, block_size, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    v16 = torch.randn(nb, block_size, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    k = k16 if kv_dtype == torch.float16 else k16.to(kv_dtype)
    v = v16 if kv_dtype == torch.float16 else v16.to(kv_dtype)
    out = torch.empty(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    csq = torch.arange(0, num_q_tokens + 1, q_len, dtype=torch.int32, device=device)
    suk = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    bt = torch.arange(0, nb, dtype=torch.int32, device=device).reshape(num_seqs, nbps)
    return q, k, v, out, csq, suk, bt


def call_triton(q, k, v, out, csq, suk, bt, ql, kl):
    """aiter triton unified_attention. 强制走 triton 分支 (patch varlen_fwd_unified=None)."""
    import aiter.ops.triton.unified_attention as ua
    sinks = make_sinks(q.device)
    saved = ua.varlen_fwd_unified
    ua.varlen_fwd_unified = None  # 强制 use_fa_unified_2d=False -> triton kernel
    try:
        ua.unified_attention(
            q, k, v, out, csq, ql, suk, kl, SOFTMAX_SCALE, True,
            (SLIDING_WINDOW - 1, 0), bt, 0.0,
            None,           # q_descale
            None, None,     # k/v_descale (fp16)
            output_scale=None,
            sinks=sinks,
        )
    finally:
        ua.varlen_fwd_unified = saved


def bench(fn, iters=50, warmup=10):
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
    args = ap.parse_args()

    print(f"三路直接调 kernel 对比 (Q=fp16, sinks=gemma4, block=64)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, sw={SLIDING_WINDOW}")
    print(f"  A=flash fp16+sinks  B=flash fp8(e5m2)+sinks  C=triton fp16+sinks(当前生产)")
    print(f"  device: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 110)

    BLOCK = 64
    scenarios = [
        ("decode  kv=512",  args.seqs, 1,   512),
        ("decode  kv=1024", args.seqs, 1,   1024),
        ("decode  kv=2048", args.seqs, 1,   2048),
        ("decode  kv=4096", args.seqs, 1,   4096),
        ("prefill q=512",   1,         512, 512),
        ("prefill q=1024",  1,         1024,1024),
        ("prefill q=2048",  1,         2048,2048),
    ]

    print(f"{'场景':<18} {'A:flash_fp16':>13} {'B:flash_fp8':>12} {'C:triton_fp16':>14} "
          f"{'A vs C':>9} {'B vs C':>9}")
    print("-" * 110)

    for name, ns, ql, kl in scenarios:
        t_a = t_b = t_c = float("nan")
        # A: flash fp16
        try:
            q, k, v, out, csq, csk, suk, bt = make_inputs_flash(ns, ql, kl, BLOCK, torch.float16)
            t_a = bench(lambda: call_flash_fp16(q, k, v, out, csq, csk, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [A {name} ERR] {e}")
        # B: flash fp8
        try:
            q, k, v, out, csq, csk, suk, bt = make_inputs_flash(ns, ql, kl, BLOCK, FP8_KV_DTYPE)
            t_b = bench(lambda: call_flash_fp8(q, k, v, out, csq, csk, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [B {name} ERR] {e}")
        # C: triton fp16
        try:
            q, k, v, out, csq, suk, bt = make_inputs_triton(ns, ql, kl, BLOCK, torch.float16)
            t_c = bench(lambda: call_triton(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [C {name} ERR] {e}")

        avc = (t_c / t_a) if (t_a == t_a and t_c > 0) else float("nan")
        bvc = (t_c / t_b) if (t_b == t_b and t_c > 0) else float("nan")
        print(f"{name:<18} {t_a:>12.3f}ms {t_b:>11.3f}ms {t_c:>13.3f}ms "
              f"{avc:>8.2f}x {bvc:>8.2f}x")

    print("-" * 110)
    print("A vs C > 1.0 = flash_attn 比 triton 快 (切 flash_attn 的收益)")
    print("B vs C > 1.0 = flash fp8 比 triton fp16 快 (方案D 的收益)")
    print("B vs A        = fp8 相对 fp16 在 flash 内部的得失 (之前测 ~0.33x, fp8 splitkv 劣化)")


if __name__ == "__main__":
    main()
