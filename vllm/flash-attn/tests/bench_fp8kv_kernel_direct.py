#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接调 flash_attn kernel 验证 fp8 KV (方案D), 绕过 Python wrapper.

源码 /public/home/weishb/flash-attention-cutlass/csrc/flash_attn_cutlass/flash_api.cpp
预装版 flash_attn_2_cuda 已含这些符号 (2.8.3+das.opt1).

目标算子: vllm_mha_varlen_fwd_kv_fp8  (flash_api.cpp:3497)
  - Q: fp16,  KV: float8_e5m2 (源码行3538 硬性要求 e5m2)
  - 支持 sinks (s_aux, 源码行3810), 要求 s_aux dtype == Q dtype (fp16), heads<=64
  - paged KV: page_block_size 必须 == 64 (源码行3574)
  - descale: q/k/v_descale (标量 fp32 tensor)

对比:
  A. vllm_mha_varlen_fwd (fp16 KV)         -> fp16 基准 (同算子族, KV fp16)
  B. vllm_mha_varlen_fwd_kv_fp8 (fp8 KV)   -> 方案D (Q fp16 + KV fp8 + sinks)
  C. vllm_mha_varlen_fwd (fp16, 无 sinks)  -> 参考 (看 sinks 开销)

gemma4: heads=32/16, head_size=256, sliding_window=1024, block=64 (fp8 kernel 要求)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_fp8kv_kernel_direct.py
"""
import torch
import time
import argparse

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)
FP8_KV_DTYPE = torch.float8_e5m2  # 源码 vllm_mha_varlen_fwd_kv_fp8 硬性要求 e5m2

import flash_attn_2_cuda as fa


def make_inputs(num_seqs, q_len, kv_len, block_size, kv_dtype, device="cuda"):
    """paged KV cache. Q 固定 fp16.
    k layout (paged): [num_blocks, num_kv_heads, page_block_size, head_size]
    v layout (paged): [num_blocks, num_kv_heads, head_size, page_block_size]  (注意 v 转置)
    源码行3616-3617: CHECK_SHAPE(k, num_blocks, num_heads_k, page_block_size, head_size_og)
                     CHECK_SHAPE(v, num_blocks, num_heads_k, head_size_value, page_block_size)
    """
    num_query_tokens = num_seqs * q_len
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    num_blks = num_seqs * num_blks_per_seq

    q = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                    dtype=torch.float16, device=device)
    # randn 不支持 fp8 -> 先 fp16 再量化. fp16 路径 .to(fp16) 是 no-op.
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


def make_sinks(device="cuda"):
    """sinks [num_query_heads], dtype 必须 == Q dtype (fp16), heads<=64."""
    return torch.zeros(NUM_QUERY_HEADS, dtype=torch.float16, device=device)


def make_descale(device="cuda"):
    """标量 descale fp32. scale=1.0 近似无损."""
    return torch.ones(1, dtype=torch.float32, device=device)


def call_fp16(q, k, v, out, csq, csk, suk, bt, q_len, kv_len, use_sinks=True):
    """vllm_mha_varlen_fwd (fp16 KV). 参数顺序见 flash_api.cpp:3860."""
    sinks = make_sinks(q.device) if use_sinks else None
    # 签名: q,k,v,out_,cu_seqlens_q,cu_seqlens_k,seqused_k,leftpad_k_,block_table_,
    #       alibi_slopes_,max_seqlen_q,max_seqlen_k,p_dropout,softmax_scale,zero_tensors,
    #       is_causal,window_size_left,window_size_right,softcap,return_softmax,
    #       q_descale_,k_descale_,v_descale_,gen_,s_aux_
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
        SLIDING_WINDOW - 1,  # window_size_left
        0,          # window_size_right
        0.0,        # softcap
        False,      # return_softmax
        None, None, None,  # q/k/v_descale (fp16 不需要)
        None,       # gen_
        sinks,      # s_aux_
    )
    return out


def call_fp8kv(q, k, v, out, csq, csk, suk, bt, q_len, kv_len, use_sinks=True):
    """vllm_mha_varlen_fwd_kv_fp8 (fp8 KV e5m2). 参数顺序见 flash_api.cpp:3497."""
    sinks = make_sinks(q.device) if use_sinks else None
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
        SLIDING_WINDOW - 1,  # window_size_left
        0,          # window_size_right
        0.0,        # softcap
        False,      # return_softmax
        None,       # q_descale_ (Q fp16, 不量化)
        kd,         # k_descale_
        vd,         # v_descale_
        None,       # gen_
        sinks,      # s_aux_
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
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="只跑一个场景快速验证")
    args = ap.parse_args()

    print(f"方案D 直接调 kernel 验证 (绕过 wrapper)")
    print(f"  heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, head_size={HEAD_SIZE}, sw={SLIDING_WINDOW}")
    print(f"  A=vllm_mha_varlen_fwd(fp16 KV+sinks)  B=vllm_mha_varlen_fwd_kv_fp8(e5m2 KV+sinks)")
    print(f"  block=64 (fp8 kernel 硬性要求), KV fp8 dtype=e5m2")
    print(f"  device: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 100)

    BLOCK = 64  # vllm_mha_varlen_fwd_kv_fp8 要求 page_block_size==64
    if args.smoke:
        scenarios = [("decode kv=2048", args.seqs, 1, 2048),
                     ("prefill q=512", 1, 512, 512)]
    else:
        scenarios = [
            ("decode  kv=512",  args.seqs, 1,   512),
            ("decode  kv=1024", args.seqs, 1,   1024),
            ("decode  kv=2048", args.seqs, 1,   2048),
            ("decode  kv=4096", args.seqs, 1,   4096),
            ("prefill q=512",   1,         512, 512),
            ("prefill q=1024",  1,         1024,1024),
            ("prefill q=2048",  1,         2048,2048),
        ]

    # ============ 能跑通 + 性能 ============
    print(f"{'场景':<18} {'A:fp16+sinks':>13} {'B:fp8+sinks':>13} {'B vs A':>9}")
    print("-" * 100)

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

        bva = (t_a / t_b) if (t_b == t_b and t_a > 0) else float("nan")
        print(f"{name:<18} {t_a:>12.3f}ms {t_b:>12.3f}ms {bva:>8.2f}x")

    # ============ 精度: B(fp8 KV) vs A(fp16 KV) ============
    print()
    print("=" * 100)
    print("精度对比: B(fp8 e5m2 KV) vs A(fp16 KV) 基准 (都带 sinks, Q fp16)")
    print("=" * 100)
    print(f"{'场景':<18} {'max_abs':>12} {'mean_abs':>12} {'mean_rel':>12}  判定")
    print("-" * 100)

    for name, ns, ql, kl in [("decode kv=2048", args.seqs, 1, 2048),
                              ("prefill q=512", 1, 512, 512),
                              ("prefill q=2048", 1, 2048, 2048)]:
        try:
            torch.manual_seed(42)
            # A: fp16 基准
            q, k16, v16, out_a, csq, csk, suk, bt = make_inputs(ns, ql, kl, BLOCK, torch.float16)
            call_fp16(q, k16, v16, out_a, csq, csk, suk, bt, ql, kl)
            # B: fp8 KV (同一份数据量化)
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

    print("-" * 100)
    print("说明: B vs A > 1.0 = fp8 KV 比 fp16 快 (省带宽); A 走 vllm_mha_varlen_fwd 同算子族保证公平")


if __name__ == "__main__":
    main()
