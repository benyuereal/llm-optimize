#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e4m3 fp8 KV 性能三路对比 (都走 aiter.unified_attention 真实入口, Q=fp16).

路径:
  A. fp16flash : Q fp16 + KV fp16       -> flash fp16 (varlen_fwd_unified 非 fp8 分支)
  B. fp8triton : Q fp16 + KV e4m3       -> 强制 triton 2D (patch varlen_fwd_unified=None)
  C. fp8gate   : Q fp16 + KV e4m3       -> 生产 gate (decode->flash e4m3, prefill->triton)

对比维度: decode (max_seqlen_q=1) 和 prefill (max_seqlen_q>=512), 多个 batch/seq.
gemma4: heads=32/16, head_size=256, sliding_window=1024, block=128, bshd.

用法: HIP_VISIBLE_DEVICES=4 python3 bench_e4m3_three_way.py
"""
import torch
import time
import argparse
import contextlib

H_Q = 32
H_KV = 16
D = 256
BLOCK = 128
SLIDING = 1024
SCALE = 1.0 / (D ** 0.5)
DEV = "cuda:0"


def make_inputs(num_seqs, q_len, kv_len, kv_dtype):
    nbps = (kv_len + BLOCK - 1) // BLOCK
    nb = num_seqs * nbps
    nq = num_seqs * q_len
    q = torch.randn(nq, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    k16 = torch.randn(nb, BLOCK, H_KV, D, device=DEV, dtype=torch.float16) * 0.1
    v16 = torch.randn(nb, BLOCK, H_KV, D, device=DEV, dtype=torch.float16) * 0.1
    k = k16 if kv_dtype == torch.float16 else k16.to(kv_dtype)
    v = v16 if kv_dtype == torch.float16 else v16.to(kv_dtype)
    out = torch.empty(nq, H_Q, D, device=DEV, dtype=torch.float16)
    csq = torch.arange(0, (num_seqs + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    suk = torch.full((num_seqs,), kv_len, device=DEV, dtype=torch.int32)
    bt = torch.arange(0, nb, device=DEV, dtype=torch.int32).reshape(num_seqs, nbps)
    # fp8 走 triton 需要 descale (unit)
    if kv_dtype != torch.float16:
        kd = vd = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    else:
        kd = vd = None
    return q, k, v, out, csq, suk, bt, kd, vd


def call_unified(q, k, v, out, csq, suk, bt, ql, kl, kd, vd, force_triton=False):
    """走 aiter.unified_attention 真实入口. force_triton=True 时 patch 掉 flash 走 triton."""
    from aiter.ops.triton import unified_attention as ua
    if force_triton:
        saved = ua.varlen_fwd_unified
        ua.varlen_fwd_unified = None
        try:
            ua.unified_attention(
                q, k, v, out, csq, ql, suk, kl, SCALE, True,
                (SLIDING, 0), bt, 0.0, None, kd, vd,
            )
        finally:
            ua.varlen_fwd_unified = saved
    else:
        ua.unified_attention(
            q, k, v, out, csq, ql, suk, kl, SCALE, True,
            (SLIDING, 0), bt, 0.0, None, kd, vd,
        )


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
    args = ap.parse_args()

    print(f"e4m3 fp8 KV 性能三路对比 (Q=fp16, 走 aiter.unified_attention 真实入口)")
    print(f"  heads={H_Q}/{H_KV}, head_size={D}, sw={SLIDING}, block={BLOCK}, bshd")
    print(f"  A=fp16flash  B=fp8triton(强制)  C=fp8gate(decode->flash, prefill->triton)")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    # (name, num_seqs, q_len, kv_len)
    scenarios = [
        # decode: 多 batch
        ("decode bs=4  kv=2048",  4,   1, 2048),
        ("decode bs=16 kv=2048",  16,  1, 2048),
        ("decode bs=64 kv=2048",  64,  1, 2048),
        ("decode bs=4  kv=4096",  4,   1, 4096),
        ("decode bs=64 kv=4096",  64,  1, 4096),
        ("decode bs=64 kv=8192",  64,  1, 8192),
        # prefill: 单 seq 长上下文
        ("prefill q=512  kv=512",  1, 512, 512),
        ("prefill q=1024 kv=1024", 1, 1024, 1024),
        ("prefill q=2048 kv=2048", 1, 2048, 2048),
        ("prefill q=4096 kv=4096", 1, 4096, 4096),
    ]

    print(f"{'场景':<24} {'A:fp16flash':>12} {'B:fp8triton':>12} {'C:fp8gate':>12} "
          f"{'C vs A':>9} {'C vs B':>9} {'A vs B':>9}")
    print("-" * 100)

    for name, ns, ql, kl in scenarios:
        ta = tb = tc = float("nan")
        # A: fp16 flash
        try:
            q, k, v, out, csq, suk, bt, kd, vd = make_inputs(ns, ql, kl, torch.float16)
            ta = bench(lambda: call_unified(q, k, v, out, csq, suk, bt, ql, kl, kd, vd, force_triton=False), args.iters)
        except Exception as e:
            print(f"  [A {name} ERR] {type(e).__name__}: {e}")
        # B: fp8 triton (强制)
        try:
            q, k, v, out, csq, suk, bt, kd, vd = make_inputs(ns, ql, kl, torch.float8_e4m3fn)
            tb = bench(lambda: call_unified(q, k, v, out, csq, suk, bt, ql, kl, kd, vd, force_triton=True), args.iters)
        except Exception as e:
            print(f"  [B {name} ERR] {type(e).__name__}: {e}")
        # C: fp8 gate (decode->flash, prefill->triton)
        try:
            q, k, v, out, csq, suk, bt, kd, vd = make_inputs(ns, ql, kl, torch.float8_e4m3fn)
            tc = bench(lambda: call_unified(q, k, v, out, csq, suk, bt, ql, kl, kd, vd, force_triton=False), args.iters)
        except Exception as e:
            print(f"  [C {name} ERR] {type(e).__name__}: {e}")

        def ratio(x, y):
            return f"{y/x:.2f}x" if (x == x and y == y and x > 0) else "  -  "
        print(f"{name:<24} {ta:>11.3f}ms {tb:>11.3f}ms {tc:>11.3f}ms "
              f"{ratio(ta,tc):>9} {ratio(tb,tc):>9} {ratio(tb,ta):>9}")

    print("-" * 100)
    print("C vs A > 1.0 = fp8gate 比 fp16flash 快 (decode 应明显快, prefill=triton 看情况)")
    print("C vs B > 1.0 = gate(混合) 比纯 triton 快 (decode 走 flash 的收益)")
    print("A vs B > 1.0 = fp16flash 比 fp8triton 快")


if __name__ == "__main__":
    main()
