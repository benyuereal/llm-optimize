#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B 验证: KV cache 用 fp8 存 (省显存), 调 flash_attn 前反量化成 fp16 计算.

关键事实:
  - KV cache 显存占用取决于"存在显存里的 dtype", 不取决于计算时 dtype.
    fp8 存 -> 显存省一半; 调用前 .half() 转换 -> 临时张量 fp16, 多一次拷贝.
  - flash_attn 拿到 fp16 KV -> 走已验证 2.5-6x 加速的快速 kernel (decode hg_prefix_decode, prefill 快速路径).
  - 所以 B 的性能 = A 的性能 - 反量化拷贝开销.
  - 只要拷贝开销 < flash 加速, B 净赚 (显存省一半 + 仍有加速).

对比 (gemma4, 不带 sinks 因 gemma4 不用 sinks; sliding_window 通过 window_size 生效):
  A. 纯 fp16 KV cache          -> fp16 存, fp16 算 (基准, flash_attn 快速 kernel)
  B. fp8 KV cache + 调前反量化   -> fp8 存, 调前 .half() 转 fp16, flash_attn 算 (方案B 目标)
  C. 纯 fp16 triton             -> 当前生产基准 (triton kernel)

反量化方式 (B 内部两种):
  B1. .to(fp16) 无 scale (近似无损, 数据已是 fp8 范围内)
  B2. .to(fp16) * descale (带 scale, 模拟真实 fp8 kv cache 反量化)

layout: flash_attn k=[nb,nh_k,blk,hd], v=[nb,nh_k,hd,blk] (v 转置)
        triton     k=v=[nb,blk,nh_k,hd]
block=128 (flash_attn decode 硬性要求)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_fp8cache_dequant_vs_fp16.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)
FP8_DTYPE = torch.float8_e4m3fnuz  # ROCm 默认 fp8

import flash_attn_2_cuda as fa


def make_inputs_flash(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    """varlen_fwd_unified wrapper 实际期望 triton layout: k=v=[nb,blk,nh_k,hd].
    (flash_attn wrapper 内部处理 layout, 传入用 triton layout 即可, 与之前能跑通的脚本一致).
    Q fp16. 不返回 cu_seqlens_k (wrapper 用 seqused_k 驱动 paged 路径)."""
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


def make_inputs_triton(num_seqs, q_len, kv_len, block_size, device="cuda"):
    """triton layout: k=v=[nb,blk,nh_k,hd]. Q fp16, KV fp16."""
    num_q_tokens = num_seqs * q_len
    nbps = (kv_len + block_size - 1) // block_size
    nb = num_seqs * nbps
    q = torch.randn(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    k = torch.randn(nb, block_size, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    v = torch.randn(nb, block_size, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    out = torch.empty(num_q_tokens, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
    csq = torch.arange(0, num_q_tokens + 1, q_len, dtype=torch.int32, device=device)
    suk = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    bt = torch.arange(0, nb, dtype=torch.int32, device=device).reshape(num_seqs, nbps)
    return q, k, v, out, csq, suk, bt


def call_flash_varlen_unified(q, k, v, out, csq, suk, bt, ql, kl):
    """走 Python wrapper varlen_fwd_unified (生产实际入口), 不带 sinks (gemma4 不用).
    decode -> hg_prefix_decode_varlen_fwd, prefill -> 快速路径. triton layout 输入."""
    from flash_attn import varlen_fwd_unified
    varlen_fwd_unified(q=q, k=k, v=v, cu_seqlens_q=csq, seqused_k=suk, block_table=bt,
        max_seqlen_q=ql, max_seqlen_k=kl, softmax_scale=SOFTMAX_SCALE, causal=True,
        softcap=0.0, window_size=(SLIDING_WINDOW - 1, 0), out=out)


def call_triton(q, k, v, out, csq, suk, bt, ql, kl):
    """aiter triton unified_attention (当前生产). block=16 走 triton 分支."""
    import aiter.ops.triton.unified_attention as ua
    saved = ua.varlen_fwd_unified
    ua.varlen_fwd_unified = None  # 强制 triton
    try:
        ua.unified_attention(q, k, v, out, csq, ql, suk, kl, SOFTMAX_SCALE, True,
            (SLIDING_WINDOW - 1, 0), bt, 0.0, None, None, None, output_scale=None)
    finally:
        ua.varlen_fwd_unified = saved


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
    args = ap.parse_args()

    print(f"方案B: fp8 KV cache + 调前反量化 vs 纯 fp16 (gemma4, 无 sinks, block=128)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, sw={SLIDING_WINDOW}")
    print(f"  A=flash fp16(cache fp16)  B=flash fp8cache->fp16  C=triton fp16(当前生产)")
    print(f"  flash 入口: varlen_fwd_unified (生产 wrapper, decode hg_prefix_decode / prefill 快速路径)")
    print(f"  device: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 110)

    BLOCK = 128
    scenarios = [
        ("decode  kv=512",  args.seqs, 1,   512),
        ("decode  kv=1024", args.seqs, 1,   1024),
        ("decode  kv=2048", args.seqs, 1,   2048),
        ("decode  kv=4096", args.seqs, 1,   4096),
        ("prefill q=512",   1,         512, 512),
        ("prefill q=1024",  1,         1024,1024),
        ("prefill q=2048",  1,         2048,2048),
    ]

    def run_flash(q, k, v, out, csq, suk, bt, ql, kl):
        call_flash_varlen_unified(q, k, v, out, csq, suk, bt, ql, kl)

    print(f"{'场景':<18} {'A:fp16':>10} {'B:fp8->fp16':>12} {'C:triton':>10} "
          f"{'B vs A':>8} {'A vs C':>8} {'B vs C':>8}")
    print("-" * 110)

    for name, ns, ql, kl in scenarios:
        t_a = t_b = t_c = float("nan")
        # A: flash 纯 fp16
        try:
            q, k, v, out, csq, suk, bt = make_inputs_flash(ns, ql, kl, BLOCK, torch.float16)
            t_a = bench(lambda: run_flash(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [A {name} ERR] {e}")
        # B: fp8 cache + 调前反量化成 fp16
        try:
            q, k8, v8, out, csq, suk, bt = make_inputs_flash(ns, ql, kl, BLOCK, FP8_DTYPE)
            k16_buf = torch.empty_like(k8, dtype=torch.float16)
            v16_buf = torch.empty_like(v8, dtype=torch.float16)
            def run_b():
                k16_buf.copy_(k8)      # fp8 -> fp16 (反量化, 无 scale)
                v16_buf.copy_(v8)
                run_flash(q, k16_buf, v16_buf, out, csq, suk, bt, ql, kl)
            t_b = bench(run_b, args.iters)
        except Exception as e:
            print(f"  [B {name} ERR] {e}")
        # C: triton fp16 (block=16 当前生产)
        try:
            q, k, v, out, csq, suk, bt = make_inputs_triton(ns, ql, kl, 16)
            t_c = bench(lambda: call_triton(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [C {name} ERR] {e}")

        bva = (t_a / t_b) if (t_a == t_a and t_b == t_b and t_b > 0) else float("nan")
        avc = (t_c / t_a) if (t_a == t_a and t_c == t_c and t_c > 0) else float("nan")
        bvc = (t_c / t_b) if (t_b == t_b and t_c == t_c and t_c > 0) else float("nan")
        print(f"{name:<18} {t_a:>9.3f}ms {t_b:>11.3f}ms {t_c:>9.3f}ms "
              f"{bva:>7.2f}x {avc:>7.2f}x {bvc:>7.2f}x")

    print("-" * 110)
    print("B vs A < 1.0 = 反量化拷贝有开销 (B 比 A 慢); 差距 = 反量化代价")
    print("A vs C > 1.0 = flash_attn 比 triton 快 (切换收益)")
    print("B vs C > 1.0 = 方案B 仍比当前生产 triton 快 (净收益, 还省一半 KV 显存)")

    # ============ 精度: B vs A ============
    print()
    print("=" * 110)
    print("精度: B(fp8 cache 反量化) vs A(纯 fp16) (无 scale, 数据 randn 在 fp8 范围内)")
    print("=" * 110)
    print(f"{'场景':<18} {'max_abs':>12} {'mean_abs':>12} {'mean_rel':>12}  判定")
    print("-" * 110)
    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("prefill q=512", 1, 512, 512),
                              ("prefill q=2048", 1, 2048, 2048)]:
        try:
            torch.manual_seed(42)
            q, k16, v16, out_a, csq, suk, bt = make_inputs_flash(ns, ql, kl, BLOCK, torch.float16)
            run_flash(q, k16, v16, out_a, csq, suk, bt, ql, kl)
            k8 = k16.to(FP8_DTYPE); v8 = v16.to(FP8_DTYPE)
            k16r = k8.to(torch.float16); v16r = v8.to(torch.float16)
            out_b = torch.empty_like(out_a)
            run_flash(q, k16r, v16r, out_b, csq, suk, bt, ql, kl)
            diff = (out_a.float() - out_b.float()).abs()
            max_abs = diff.max().item(); mean_abs = diff.mean().item()
            rel = diff / torch.clamp(out_a.float().abs(), min=1e-3)
            mean_rel = rel.mean().item()
            ok = max_abs < 0.5
            print(f"{name:<18} {max_abs:>12.5f} {mean_abs:>12.6f} {mean_rel*100:>11.4f}%  "
                  f"{'✅ 一致' if ok else '⚠️ 偏大'}")
        except Exception as e:
            print(f"{name:<18} ERROR: {e}")


if __name__ == "__main__":
    main()
