#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixed kernel 真实性能测试 (决定值不值得改 Q-fp16+KV-fp8).

目的: 量出现成 mixed kernel (fp8存储/bf16计算) 在 decode 时的真实性能,
      对比 fp16 decode, 判断 "改 Q 加载回 fp16" 这条路有没有戏.

现成入口: hg_prefix_decode_varlen_fwd (生产 decode 入口)
  - gfx936 + Q e5m2 + hdim256 + block128 -> 走 mixed kernel:
    Flash_fp8_bf16_fwd_kernel_traits + flash_mixed_fwd_splitkv_tile16x32_kernel
    (flash_fwd_launch_template_pa.h:696-710)
  - 硬性要求: bshd layout (layout=1), use_bf16_output, q/k/v descale 都要有
  - output 是 bf16 (mixed kernel 用 bf16 计算)

对比:
  A. 全 fp16 (Q/KV fp16)  -> fp16 splitkv (生产 decode 路径)         [output fp16]
  B. 全 fp8 e5m2 (Q/KV)   -> mixed kernel (fp8存/bf16算) [要评估的]  [output bf16]
  C. 全 fp8 e4m3 (Q/KV)   -> 看 gfx936 走哪条 (参考)                 [output bf16/fp16]

判定:
  B vs A >= 1.0 (B 更快或持平) -> mixed kernel 省带宽赢过反量化, 改 Q 回 fp16 后只会更快, 值得改
  B vs A 明显 < 1.0 (B 慢)    -> mixed kernel 本身慢 (splitkv tile16x32 开销大), 改了也没用

gemma4: heads=32/16, head_size=256, sliding_window=1024, block=128 (生产配置)
bshd layout: q=[total_q,nh,hd], k=[nb,blk,nh_k,hd], v=[nb,blk,nh_k,hd] (v 不转置)

用法:
  HIP_VISIBLE_DEVICES=1 python bench_mixed_kernel_perf.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)

import flash_attn_2_cuda as fa


def make_inputs(num_seqs, q_len, kv_len, block_size, dtype, device="cuda"):
    """bshd layout (layout=1).
    q: [total_q, num_query_heads, head_size]
    k: [num_blocks, block_size, num_kv_heads, head_size]
    v: [num_blocks, block_size, num_kv_heads, head_size]  (v 不转置, bshd)
    """
    num_query_tokens = num_seqs * q_len
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    num_blks = num_seqs * num_blks_per_seq

    # randn 不支持 fp8 -> 先 fp16 再量化
    q16 = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                      dtype=torch.float16, device=device)
    k16 = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                      dtype=torch.float16, device=device)
    v16 = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                      dtype=torch.float16, device=device)
    q = q16 if dtype == torch.float16 else q16.to(dtype)
    k = k16 if dtype == torch.float16 else k16.to(dtype)
    v = v16 if dtype == torch.float16 else v16.to(dtype)

    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=device)
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    block_table = torch.arange(0, num_blks, dtype=torch.int32,
                               device=device).reshape(num_seqs, num_blks_per_seq)
    return q, k, v, cu_seqlens_q, seqused_k, block_table


def make_descale(device="cuda"):
    """标量 descale fp32, scale=1.0 近似无损."""
    return torch.ones(1, dtype=torch.float32, device=device)


def call_decode(q, k, v, out, csq, suk, bt, q_len, kv_len, dtype):
    """hg_prefix_decode_varlen_fwd.
    源码版签名 (24 参数, 支持 fp8 descale):
      q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes_,
      block_table, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale,
      zero_tensors, is_causal, window_size_left, window_size_right, softcap,
      return_softmax, layout, scales_q_, scales_k_, scales_v_, s_aux_, is_bf16_output
    gfx936 fp8 mixed kernel 要求: bshd layout, use_bf16_output, q/k/v descale 都有
    """
    is_fp8 = (dtype == torch.float8_e5m2 or dtype == torch.float8_e4m3fn)
    if is_fp8:
        sq = make_descale(q.device)
        sk = make_descale(q.device)
        sv = make_descale(q.device)
        is_bf16_output = True  # gfx936 fp8 mixed 要求 bf16 output
    else:
        sq = sk = sv = None
        is_bf16_output = False

    fa.hg_prefix_decode_varlen_fwd(
        q, k, v, out,
        csq,
        None,       # cu_seqlens_k (bshd 用 seqused_k)
        suk,
        None,       # alibi_slopes_
        bt,
        q_len, kv_len,
        0.0,        # p_dropout
        SOFTMAX_SCALE,
        False,      # zero_tensors
        True,       # is_causal
        SLIDING_WINDOW - 1,  # window_size_left
        0,          # window_size_right
        0.0,        # softcap
        False,      # return_softmax
        1,          # layout (bshd)
        sq, sk, sv,  # scales_q/k/v
        None,       # s_aux_ (gemma4 无 sinks)
        is_bf16_output,
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

    print(f"mixed kernel 真实性能测试 (hg_prefix_decode_varlen_fwd)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, sw={SLIDING_WINDOW}, block=128")
    print(f"  A=全fp16 (fp16 splitkv, 生产decode路径)        [out fp16]")
    print(f"  B=全fp8 e5m2 (mixed kernel fp8存/bf16算) [目标] [out bf16]")
    print(f"  注: e4m3 在 gfx936 mixed kernel 触发 VMFault, 已排除")
    print(f"  bshd layout, device: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 115)

    BLOCK = 128  # 生产配置
    if args.smoke:
        scenarios = [("decode kv=2048", args.seqs, 1, 2048)]
    else:
        scenarios = [
            ("decode  kv=512",  args.seqs, 1,   512),
            ("decode  kv=1024", args.seqs, 1,   1024),
            ("decode  kv=2048", args.seqs, 1,   2048),
            ("decode  kv=4096", args.seqs, 1,   4096),
            ("decode  kv=8192", args.seqs, 1,   8192),
        ]

    # ============ 性能对比 ============
    print(f"{'场景':<18} {'A:全fp16':>12} {'B:fp8e5m2':>12} {'B/A':>8}")
    print("-" * 115)

    results = []
    for name, ns, ql, kl in scenarios:
        t_a = t_b = float("nan")
        # A: fp16
        try:
            q, k, v, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            out = torch.empty(ns * ql, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=q.device)
            t_a = bench(lambda: call_decode(q, k, v, out, csq, suk, bt, ql, kl, torch.float16), args.iters)
        except Exception as e:
            print(f"  [A fp16 {name} ERR] {e}")
        # B: fp8 e5m2
        try:
            q, k, v, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float8_e5m2)
            out = torch.empty(ns * ql, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=q.device)
            t_b = bench(lambda: call_decode(q, k, v, out, csq, suk, bt, ql, kl, torch.float8_e5m2), args.iters)
        except Exception as e:
            print(f"  [B e5m2 {name} ERR] {e}")

        ba = (t_a / t_b) if (t_a == t_a and t_b == t_b and t_b > 0) else float("nan")
        results.append((name, t_a, t_b, ba))
        print(f"{name:<18} {t_a:>11.3f}ms {t_b:>11.3f}ms {ba:>7.2f}x")

    # ============ 判定 ============
    print()
    print("=" * 115)
    print("判定 (基于 B/A, 即 mixed kernel e5m2 vs fp16):")
    print("=" * 115)
    valid = [r for r in results if r[3] == r[3]]
    if valid:
        avg_ba = sum(r[3] for r in valid) / len(valid)
        if avg_ba >= 1.0:
            verdict = "✅ mixed kernel 比 fp16 快或持平 (省带宽赢过反量化), 改 Q 回 fp16 后只会更快, 值得改"
        elif avg_ba >= 0.7:
            verdict = f"⚠️ mixed kernel 比 fp16 慢 ({avg_ba:.2f}x), 但不算太糟; 改 Q 回 fp16 可能补回部分, 需试"
        else:
            verdict = f"❌ mixed kernel 明显慢 ({avg_ba:.2f}x), splitkv tile16x32 本身开销大, 改 Q 也救不回, 别改"
        print(f"  decode 平均 B/A = {avg_ba:.2f}x")
        print(f"  {verdict}")
    else:
        print("  无有效结果")

    # ============ 精度: B (fp8 e5m2) vs A (fp16) ============
    print()
    print("=" * 115)
    print("精度对比: B(e5m2) vs A(fp16) 基准")
    print("  注意: fp8 output 是 bf16, A output 是 fp16, 比 .float() 后的数值差")
    print("=" * 115)
    print(f"{'场景':<18} {'B e5m2 max_abs':>16}  判定")
    print("-" * 115)

    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("decode kv=4096", args.seqs, 1, 4096)]:
        try:
            torch.manual_seed(42)
            # A: fp16 基准
            q, k, v, csq, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            out_a = torch.empty(ns * ql, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float16, device=q.device)
            call_decode(q, k, v, out_a, csq, suk, bt, ql, kl, torch.float16)
            # B: e5m2
            q8, k8, v8, _, _, _ = make_inputs(ns, ql, kl, BLOCK, torch.float8_e5m2)
            out_b = torch.empty(ns * ql, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=q8.device)
            call_decode(q8, k8, v8, out_b, csq, suk, bt, ql, kl, torch.float8_e5m2)

            max_b = (out_a.float() - out_b.float()).abs().max().item()
            ok = max_b < 1.0
            verdict = "✅ 一致" if ok else "⚠️ 偏大"
            print(f"{name:<18} {max_b:>16.5f}  {verdict}")
        except Exception as e:
            print(f"{name:<18} ERROR: {e}")

    print("-" * 115)
    print("说明: B/A>=1.0 = mixed kernel 省带宽赢反量化, 改 Q 回 fp16 值得; B/A<0.7 = kernel 本身慢, 别改")


if __name__ == "__main__":
    main()
