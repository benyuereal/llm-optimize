#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案D 真实验证 (修正版): triton 路径 fp8 KV + sinks vs triton fp16 KV + sinks

关键发现 (纠正之前认知):
  - 生产入口 flash_attn.varlen_fwd_unified 不支持 fp8 KV (签名无 descale,
    prefill 直接调 flash_attn_cuda.varlen_fwd_unified 传 None descale,
    只认 fp16). 之前以为它支持 fp8 是基于错误版本.
  - aiter 在 block_size%64==0 且 head_size==256 时 (gemma4: 128/256) 会
    走 flash_attn 路径 (use_fa_unified_2d=True), 而 flash_attn 不支持 fp8 KV+sinks.
  - 但 aiter 的 triton 路径 (kernel_unified_attention_2d) 原生支持 fp8 KV + sinks:
      * 行291-295: K_load.dtype.is_fp8() -> K = K_load.to(f32)*k_scale -> Q.dtype(fp16)
      * 行306-310: V 同理
      * 行185-189: USE_SINKS 处理 sinks
      * KV 反量化由 k.dtype 决定, 不依赖 output_scale
  - 所以真正的方案D = 让 aiter 在 fp8 KV 时走 triton 路径 (而非 flash_attn),
    KV cache 用 fp8 存 (省内存/带宽), attention 用 triton 反量化.
    不需要改任何 C++ 内核.

对比路径 (Q 都是 fp16, 都带 sinks 模拟 gemma4):
  A. triton fp16 KV + sinks        -> 当前生产能跑的 (基准)
  B. triton fp8  KV + sinks+descale -> 方案D 目标 (fp8 KV cache)
  C. flash_attn fp16 KV (无 sinks)  -> 快速路径速度上限参考

正确性: B vs A (fp8 反量化后应接近 fp16 基准)
性能:   A vs C (triton vs flash_attn, 都 fp16), B vs A (fp8 是否拖慢/加快 triton)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_attn_fp8kv_triton_vs_fp16.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
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
    triton kernel_unified_attention_2d 的 USE_SINKS 分支接收 fp32 sinks."""
    return torch.zeros(NUM_QUERY_HEADS, dtype=torch.float32, device=device)


def make_descale(device="cuda"):
    """per-tensor descale 标量. scale=1.0 模拟近似无损反量化."""
    k_descale = torch.ones(1, dtype=torch.float32, device=device)
    v_descale = torch.ones(1, dtype=torch.float32, device=device)
    return k_descale, v_descale


def call_triton(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                q_len, kv_len, k_descale=None, v_descale=None,
                use_sinks=True, output_scale=None):
    """调 aiter triton unified_attention.
    fp8 KV: 传 k_descale/v_descale, k/v 是 fp8 tensor, 内核按 dtype 自动反量化.
    fp16 KV: k_descale/v_descale 传 None (vllm 生产 fp16 时 _k_scale 仍存在但不用)."""
    from aiter.ops.triton.unified_attention import unified_attention
    window = (SLIDING_WINDOW - 1, 0)
    sinks = make_sinks(q.device) if use_sinks else None
    unified_attention(
        q, k, v, out, cu_seqlens_q, q_len, seqused_k, kv_len,
        SOFTMAX_SCALE, True, window, block_table, 0.0,
        None,              # q_descale (Q 是 fp16, 不量化)
        k_descale,         # fp8 KV 时传, fp16 时 None
        v_descale,         # 同上
        output_scale=output_scale,  # fp8 输出量化 scale; 方案D 输出 fp16 -> None
        sinks=sinks,
    )
    return out


def call_flash_attn(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                    q_len, kv_len, use_sinks=False):
    """调 flash_attn.varlen_fwd_unified (快速路径, 无 sinks).
    生产入口不支持 fp8 KV, 也不支持 sinks+decode, 这里只作 fp16 速度上限参考."""
    from flash_attn import varlen_fwd_unified
    window = (SLIDING_WINDOW - 1, 0)
    kwargs = dict(
        q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        block_table=block_table, max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=SOFTMAX_SCALE, causal=True, softcap=0.0,
        window_size=window, out=out,
    )
    # flash_attn decode 路径 assert s_aux is None -> 参考路径不传 sinks
    if use_sinks:
        kwargs["s_aux"] = make_sinks(q.device)
    varlen_fwd_unified(**kwargs)
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

    print(f"方案D 真实验证: triton fp8 KV+sinks vs triton fp16 KV+sinks (Q=fp16)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, "
          f"sw={SLIDING_WINDOW}")
    print(f"  A=triton fp16 KV+sinks (基准)  B=triton fp8 KV+sinks (方案D)  "
          f"C=flash_attn fp16 (速度上限参考, 无sinks)")
    print(f"  device: {torch.cuda.get_device_name(0)}, "
          f"cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 108)

    BLOCK = 128  # gemma4 生产 block_size
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
    print(f"{'场景':<18} {'A:tri_fp16':>12} {'B:tri_fp8':>12} {'C:fa_fp16':>12} "
          f"{'B vs A':>9} {'A vs C':>9}")
    print("-" * 108)

    for name, ns, ql, kl in scenarios:
        t_a = t_b = t_c = float("nan")
        # A: triton fp16 KV + sinks
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            t_a = bench(lambda: call_triton(q, k, v, out, csq, suk, bt, ql, kl), args.iters)
        except Exception as e:
            print(f"  [A tri_fp16 {name} ERR] {e}")
        # B: triton fp8 KV + sinks + descale
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, FP8_DTYPE)
            kd, vd = make_descale(ns)
            t_b = bench(lambda: call_triton(q, k, v, out, csq, suk, bt, ql, kl, kd, vd),
                        args.iters)
        except Exception as e:
            print(f"  [B tri_fp8 {name} ERR] {e}")
        # C: flash_attn fp16 (无 sinks, 速度上限参考)
        try:
            q, k, v, out, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            t_c = bench(lambda: call_flash_attn(q, k, v, out, csq, suk, bt, ql, kl),
                        args.iters)
        except Exception as e:
            print(f"  [C fa_fp16 {name} ERR] {e}")

        bva = (t_a / t_b) if (t_b == t_b and t_a > 0) else float("nan")
        avc = (t_c / t_a) if (t_a == t_a and t_c > 0) else float("nan")
        print(f"{name:<18} {t_a:>11.3f}ms {t_b:>11.3f}ms {t_c:>11.3f}ms "
              f"{bva:>8.2f}x {avc:>8.2f}x")

    # ============ 精度对比: B(fp8 KV) vs A(fp16 KV) ============
    print()
    print("=" * 108)
    print("精度对比: B(triton fp8 KV) vs A(triton fp16 KV) 基准")
    print("(fp8 反量化后应接近 fp16 基准; Q 都是 fp16, 都带 sinks)")
    print("=" * 108)
    print(f"{'场景':<18} {'max_abs':>12} {'mean_abs':>12} {'mean_rel':>12}  判定")
    print("-" * 108)

    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("prefill q=512", 1, 512, 512),
                              ("prefill q=2048", 1, 2048, 2048)]:
        try:
            torch.manual_seed(42)
            # A: fp16 KV 基准
            q, k16, v16, out_a, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            call_triton(q, k16, v16, out_a, csq, suk, bt, ql, kl)
            # B: fp8 KV (同一份数据量化成 fp8)
            k8 = k16.to(FP8_DTYPE)
            v8 = v16.to(FP8_DTYPE)
            out_b = torch.empty_like(out_a)
            kd, vd = make_descale(ns)
            call_triton(q, k8, v8, out_b, csq, suk, bt, ql, kl, kd, vd)

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

    print("-" * 108)
    print("列说明:")
    print("  A:tri_fp16 = triton fp16 KV + sinks (当前生产能跑的, 基准)")
    print("  B:tri_fp8  = triton fp8  KV + sinks + descale (方案D 目标)")
    print("  C:fa_fp16  = flash_attn fp16 无 sinks (速度上限参考)")
    print("  B vs A > 1.0 = fp8 比 fp16 快 (带宽受益); < 1.0 = 反量化开销")
    print("  A vs C > 1.0 = triton 比 flash_attn 慢 (flash_attn 上限参考)")


if __name__ == "__main__":
    main()
