#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案D 验证: flash_attn HIP 内核 (fp8 KV + fp16 Q) vs aiter triton (fp16 KV)

目标: 确认 hg_prefix_prefill_varlen_fwd / paged_attention 这两个 HIP 内核
      在 fp8 KV + fp16 Q 下能正确反量化, 且性能优于 triton 路径.

gemma4 真实 shape:
  num_query_heads=32, num_kv_heads=16, head_size=256, sliding_window=1024

三条对比路径 (都用 fp16 Q):
  A. flash_attn fp16 KV   -> varlen_fwd_unified 快速路径 (基准, 之前压测 4-6x)
  B. flash_attn fp8  KV   -> varlen_fwd_unified HG 路径 (带 descale, 方案D 目标)
  C. aiter triton fp16 KV -> unified_attention triton 分支 (当前生产能跑的)

正确性: B vs A (fp8 反量化后应接近 fp16 基准)
性能:   B vs C (方案D 的实际收益: fp8 KV 能否比 triton fp16 快)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_attn_fp8kv_hip_vs_triton.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
NUM_QUERIES_PER_KV = NUM_QUERY_HEADS // NUM_KV_HEADS  # 2
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)
FP8_DTYPE = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn


def make_inputs(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    """构造 paged KV cache 输入. Q 固定 fp16, KV 按 kv_dtype."""
    num_query_tokens = num_seqs * q_len
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    num_blks = num_seqs * num_blks_per_seq

    q = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                    dtype=torch.float16, device=device)
    k = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                    dtype=kv_dtype, device=device)
    v = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                    dtype=kv_dtype, device=device)
    out = torch.empty(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                      dtype=torch.float16, device=device)

    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=device)
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    block_table = torch.arange(0, num_blks, dtype=torch.int32,
                               device=device).reshape(num_seqs, num_blks_per_seq)
    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


def make_sinks(device="cuda"):
    """gemma4 有 sliding_window -> vllm 传 sinks [num_query_heads].
    传 s_aux 会让 flash_attn 走 hg_prefix_prefill_varlen_fwd (生产实际路径),
    而非 s_aux=None 的快速路径 flash_attn_cuda.varlen_fwd_unified."""
    return torch.zeros(NUM_QUERY_HEADS, dtype=torch.float32, device=device)


def make_descale(num_seqs, device="cuda"):
    """per-tensor descale: 标量 (numel=1). 模拟 fp8 kv cache 的反量化 scale."""
    # 用 1.0 作为 descale (randn 数据 fp8 化再反量化近似无损, scale=1 简化)
    k_descale = torch.ones(1, dtype=torch.float32, device=device)
    v_descale = torch.ones(1, dtype=torch.float32, device=device)
    return k_descale, v_descale


def call_flash_attn(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                    q_len, kv_len, k_descale=None, v_descale=None, q_descale=None,
                    use_sinks=True):
    """调 flash_attn.varlen_fwd_unified, 内部自动路由到 HIP 内核.
    传 descale 时走 fp8 路径, 不传时走 fp16 快速路径.
    use_sinks=True 模拟 gemma4 (有 sliding_window -> sinks), 走生产实际的 HG 内核."""
    from flash_attn import varlen_fwd_unified
    window = (SLIDING_WINDOW - 1, 0)
    kwargs = dict(
        q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        block_table=block_table, max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=SOFTMAX_SCALE, causal=True, softcap=0.0,
        window_size=window, out=out,
    )
    if use_sinks:
        kwargs["s_aux"] = make_sinks(q.device)
    # 传 descale 才会走 fp8 反量化路径 (wrapper 内部按 k.dtype 选择分支)
    if k_descale is not None:
        kwargs["k_descale"] = k_descale
    if v_descale is not None:
        kwargs["v_descale"] = v_descale
    if q_descale is not None:
        kwargs["q_descale"] = q_descale
    varlen_fwd_unified(**kwargs)
    return out


def call_triton(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                q_len, kv_len):
    """调 aiter triton unified_attention (fp16 KV 路径)."""
    from aiter.ops.triton.unified_attention import unified_attention
    window = (SLIDING_WINDOW - 1, 0)
    unified_attention(
        q, k, v, out, cu_seqlens_q, q_len, seqused_k, kv_len,
        SOFTMAX_SCALE, True, window, block_table, 0.0,
        None, None, None,  # q/k/v descale = None
    )
    return out


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seqs", type=int, default=4)
    args = ap.parse_args()

    print(f"gemma4 方案D验证: flash_attn HIP (fp8 KV) vs triton (fp16 KV)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, "
          f"sw={SLIDING_WINDOW}, Q=fp16")
    print(f"  device: {torch.cuda.get_device_name(0)}, "
          f"cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 100)

    BLOCK = 128  # flash_attn 要求 block_size%64==0, decode 要求 128
    scenarios = [
        ("decode  kv=512",  args.seqs, 1,   512),
        ("decode  kv=1024", args.seqs, 1,   1024),
        ("decode  kv=2048", args.seqs, 1,   2048),
        ("decode  kv=4096", args.seqs, 1,   4096),
        ("prefill q=512",   1,         512, 512),
        ("prefill q=1024",  1,         1024,1024),
        ("prefill q=2048",  1,         2048,2048),
    ]

    # ============ 性能对比 ============
    print(f"{'场景':<18} {'A:fa_fp16':>11} {'B:fa_fp8':>11} {'C:triton_fp16':>14} "
          f"{'B vs C':>9} {'A vs C':>9}")
    print("-" * 100)

    for name, ns, ql, kl in scenarios:
        t_a = t_b = t_c = float("nan")
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            t_a = bench(lambda: call_flash_attn(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [A fa_fp16 {name} ERR] {e}")
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, FP8_DTYPE)
            kd, vd = make_descale(ns)
            t_b = bench(lambda: call_flash_attn(q, k, v, out, csq, suk, bt, ql, kl, kd, vd), args.iters)
        except Exception as e:
            print(f"  [B fa_fp8 {name} ERR] {e}")
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, 16, torch.float16)
            t_c = bench(lambda: call_triton(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [C triton {name} ERR] {e}")

        bvc = (t_c / t_b) if (t_b == t_b and t_c > 0) else float("nan")
        avc = (t_c / t_a) if (t_a == t_a and t_c > 0) else float("nan")
        flag = " ⚡" if (bvc == bvc and bvc >= 1.0) else ""
        print(f"{name:<18} {t_a:>10.3f}ms {t_b:>10.3f}ms {t_c:>13.3f}ms "
              f"{bvc:>8.2f}x{flag} {avc:>8.2f}x")

    # ============ 精度对比: B(fp8 KV) vs A(fp16 KV) ============
    print()
    print("=" * 100)
    print("精度对比: B(flash_attn fp8 KV) vs A(flash_attn fp16 KV) 基准")
    print("(fp8 反量化后应接近 fp16 基准; Q 都是 fp16)")
    print("=" * 100)
    print(f"{'场景':<18} {'max_abs':>12} {'mean_abs':>12} {'mean_rel':>12}  判定")
    print("-" * 100)

    torch.manual_seed(42)
    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("prefill q=512", 1, 512, 512),
                              ("prefill q=2048", 1, 2048, 2048)]:
        try:
            # A: fp16 KV 基准
            torch.manual_seed(42)
            q, k16, v16, out_a, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            call_flash_attn(q, k16, v16, out_a, csq, suk, bt, ql, kl)
            # B: fp8 KV (同一份数据量化成 fp8)
            k8 = k16.to(FP8_DTYPE)
            v8 = v16.to(FP8_DTYPE)
            out_b = torch.empty_like(out_a)
            kd, vd = make_descale(ns)
            call_flash_attn(q, k8, v8, out_b, csq, suk, bt, ql, kl, kd, vd)

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

    print("-" * 100)
    print("列说明:")
    print("  A:fa_fp16     = flash_attn HIP, fp16 KV (基准, 之前压测的快速路径)")
    print("  B:fa_fp8      = flash_attn HIP, fp8 KV + descale (方案D 目标)")
    print("  C:triton_fp16 = aiter triton, fp16 KV (当前生产能跑的)")
    print("  B vs C > 1.0  = 方案D 比 triton 快 (方案D 的实际收益)")
    print("  A vs C        = fp16 快速路径相对 triton 的加速 (参考)")


if __name__ == "__main__":
    main()
