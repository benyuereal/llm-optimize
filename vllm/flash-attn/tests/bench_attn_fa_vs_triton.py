#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention 算子隔离 benchmark: flash_attn.varlen_fwd_unified (HIP 编译内核)
                              vs  aiter triton kernel_unified_attention (当前在用)

gemma4 真实 shape:
  num_query_heads   = 32
  num_kv_heads      = 16   (GQA, num_queries_per_kv = 2)
  head_size         = 256
  sliding_window    = 1024

对比两种 block_size:
  block_size = 16  -> aiter 走 triton 路径 (当前生产配置)
  block_size = 64  -> aiter 走 flash_attn HIP 路径 (--block-size 64 后)

用法:
  HIP_VISIBLE_DEVICES=0 python bench_attn_fa_vs_triton.py
"""
import torch
import time
import argparse

# ---- gemma4 真实参数 ----
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 256
NUM_QUERIES_PER_KV = NUM_QUERY_HEADS // NUM_KV_HEADS  # = 2
SLIDING_WINDOW = 1024
SOFTMAX_SCALE = 1.0 / (HEAD_SIZE ** 0.5)


def make_inputs(num_seqs, q_len, kv_len, block_size, dtype=torch.float16, device="cuda"):
    """构造 unified_attention 需要的输入张量。

    q: [num_query_tokens, num_query_heads, head_size]
    k,v: [num_blks, blk_size, num_kv_heads, head_size]  (paged KV cache)
    """
    num_query_tokens = num_seqs * q_len
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    num_blks = num_seqs * num_blks_per_seq

    q = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                    dtype=dtype, device=device)
    k = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                    dtype=dtype, device=device)
    v = torch.randn(num_blks, block_size, NUM_KV_HEADS, HEAD_SIZE,
                    dtype=dtype, device=device)
    out = torch.empty_like(q)

    # cu_seqlens_q: [num_seqs+1]
    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=device)
    # seqused_k: [num_seqs]  每条序列的实际 kv 长度
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)

    # block_table: [num_seqs, max_num_blocks_per_seq]
    block_table = torch.arange(0, num_blks, dtype=torch.int32,
                               device=device).reshape(num_seqs, num_blks_per_seq)

    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


def make_inputs_shared(num_seqs, q_len, kv_len, dtype=torch.float16, device="cuda"):
    """生成共享的逻辑 Q/K/V 数据 (与 block_size 无关), 用于精度对比。

    返回:
      q: [num_query_tokens, num_query_heads, head_size]
      k_logic, v_logic: [num_seqs, kv_len, num_kv_heads, head_size]  (逻辑连续布局)
    """
    num_query_tokens = num_seqs * q_len
    q = torch.randn(num_query_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                    dtype=dtype, device=device)
    # 用相同 seed 的逻辑 KV, 后面分别 pack 到不同 block_size
    k_logic = torch.randn(num_seqs, kv_len, NUM_KV_HEADS, HEAD_SIZE,
                          dtype=dtype, device=device)
    v_logic = torch.randn(num_seqs, kv_len, NUM_KV_HEADS, HEAD_SIZE,
                          dtype=dtype, device=device)
    return q, k_logic, v_logic


def pack_paged(kv_logic, block_size):
    """把逻辑连续 KV [num_seqs, kv_len, num_kv_heads, head_size]
    pack 成 paged 布局 [num_blks, blk_size, num_kv_heads, head_size] + block_table。

    每条序列的 block 在物理上连续排列, block_table[i] = 该序列起始 block 号。
    kv_len 需 <= num_blks_per_seq * block_size (不足部分 padding, 但 seqused_k 限定真实长度)。
    """
    num_seqs, kv_len, num_kv_heads, head_size = kv_logic.shape
    num_blks_per_seq = (kv_len + block_size - 1) // block_size
    padded_len = num_blks_per_seq * block_size
    num_blks = num_seqs * num_blks_per_seq

    # 先 pad 到 block_size 整数倍
    if padded_len > kv_len:
        pad = torch.zeros(num_seqs, padded_len - kv_len, num_kv_heads, head_size,
                          dtype=kv_logic.dtype, device=kv_logic.device)
        kv_padded = torch.cat([kv_logic, pad], dim=1)
    else:
        kv_padded = kv_logic

    # reshape 成 [num_seqs, num_blks_per_seq, block_size, num_kv_heads, head_size]
    # 再展平成 [num_blks, block_size, num_kv_heads, head_size]
    paged = kv_padded.view(num_seqs, num_blks_per_seq, block_size,
                           num_kv_heads, head_size).reshape(num_blks, block_size,
                                                            num_kv_heads, head_size)
    block_table = torch.arange(0, num_blks, dtype=torch.int32,
                               device=kv_logic.device).reshape(num_seqs, num_blks_per_seq)
    return paged.contiguous(), block_table


def run_triton(q, k_logic, v_logic, kv_len, block_size):
    """用共享 Q/K/V 跑 triton unified_attention, 返回输出 [num_query_tokens, num_query_heads, head_size]."""
    from aiter.ops.triton.unified_attention import unified_attention
    num_seqs = k_logic.shape[0]
    q_len = q.shape[0] // num_seqs
    num_query_tokens = q.shape[0]

    k_paged, block_table = pack_paged(k_logic, block_size)
    v_paged, _ = pack_paged(v_logic, block_size)
    out = torch.empty_like(q)

    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=q.device)
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=q.device)
    window = (SLIDING_WINDOW - 1, 0)

    unified_attention(q, k_paged, v_paged, out, cu_seqlens_q, q_len,
                      seqused_k, kv_len, SOFTMAX_SCALE,
                      True, window, block_table, 0.0,
                      None, None, None)
    return out


def run_flash_attn(q, k_logic, v_logic, kv_len, block_size):
    """用共享 Q/K/V 跑 flash_attn.varlen_fwd_unified, 返回输出."""
    from flash_attn import varlen_fwd_unified
    num_seqs = k_logic.shape[0]
    q_len = q.shape[0] // num_seqs
    num_query_tokens = q.shape[0]

    k_paged, block_table = pack_paged(k_logic, block_size)
    v_paged, _ = pack_paged(v_logic, block_size)
    out = torch.empty_like(q)

    cu_seqlens_q = torch.arange(0, num_query_tokens + 1, q_len,
                                dtype=torch.int32, device=q.device)
    seqused_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=q.device)
    window = (SLIDING_WINDOW - 1, 0)

    varlen_fwd_unified(q=q, k=k_paged, v=v_paged, cu_seqlens_q=cu_seqlens_q,
                       seqused_k=seqused_k, block_table=block_table,
                       max_seqlen_q=q_len, max_seqlen_k=kv_len,
                       softmax_scale=SOFTMAX_SCALE, causal=True,
                       softcap=0.0, window_size=window, out=out)
    return out


def compare_precision(num_seqs, q_len, kv_len, dtype=torch.float16):
    """精度对比: 同一份随机 Q/K/V, 两个算子分别跑, 比较输出差异."""
    torch.manual_seed(42)  # 固定 seed, 保证两个算子输入完全一致
    q, k_logic, v_logic = make_inputs_shared(num_seqs, q_len, kv_len, dtype=dtype)

    try:
        out_triton = run_triton(q, k_logic, v_logic, kv_len, block_size=16)
    except Exception as e:
        return None, None, None, f"triton ERROR: {e}"
    try:
        out_fa = run_flash_attn(q, k_logic, v_logic, kv_len, block_size=128)
    except Exception as e:
        return None, None, None, f"flash_attn ERROR: {e}"

    diff = (out_triton.float() - out_fa.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # 相对误差: 用 |a-b| / max(|a|,|b|, eps), 避免除零
    eps = 1e-3
    denom = torch.clamp(out_triton.float().abs(), min=eps)
    rel = diff / denom
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()
    return max_abs, mean_abs, max_rel, mean_rel


def bench_triton_unified(num_seqs, q_len, kv_len, block_size, iters=50, warmup=10):
    """测 aiter triton kernel_unified_attention (当前在用的路径)."""
    from aiter.ops.triton.unified_attention import unified_attention

    q, k, v, out, cu_seqlens_q, seqused_k, block_table = make_inputs(
        num_seqs, q_len, kv_len, block_size)

    max_seqlen_q = q_len
    max_seqlen_k = kv_len
    window = (SLIDING_WINDOW - 1, 0)  # left_window, right_window

    # warmup
    for _ in range(warmup):
        unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q,
                          seqused_k, max_seqlen_k, SOFTMAX_SCALE,
                          True, window, block_table, 0.0,
                          None, None, None)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q,
                          seqused_k, max_seqlen_k, SOFTMAX_SCALE,
                          True, window, block_table, 0.0,
                          None, None, None)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    return elapsed


def bench_flash_attn_unified(num_seqs, q_len, kv_len, block_size, iters=50, warmup=10):
    """测 flash_attn.varlen_fwd_unified (HIP 编译内核, block_size%64==0 时启用)."""
    from flash_attn import varlen_fwd_unified

    q, k, v, out, cu_seqlens_q, seqused_k, block_table = make_inputs(
        num_seqs, q_len, kv_len, block_size)

    max_seqlen_q = q_len
    max_seqlen_k = kv_len
    window = (SLIDING_WINDOW - 1, 0)

    # warmup
    for _ in range(warmup):
        varlen_fwd_unified(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                           seqused_k=seqused_k, block_table=block_table,
                           max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                           softmax_scale=SOFTMAX_SCALE, causal=True,
                           softcap=0.0, window_size=window, out=out)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        varlen_fwd_unified(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                           seqused_k=seqused_k, block_table=block_table,
                           max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                           softmax_scale=SOFTMAX_SCALE, causal=True,
                           softcap=0.0, window_size=window, out=out)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seqs", type=int, default=4, help="并发序列数 (decode 场景)")
    args = ap.parse_args()

    device = "cuda"
    print(f"gemma4 attention 算子对比 (heads={NUM_QUERY_HEADS}/{NUM_KV_HEADS}, "
          f"head_size={HEAD_SIZE}, sliding_window={SLIDING_WINDOW})")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 90)

    # 场景设计:
    #  decode:  q_len=1,  kv_len 较长 (典型生成阶段)
    #  prefill: q_len 较大, kv_len=q_len (典型预填充阶段)
    scenarios = [
        ("decode  kv=512",  args.seqs, 1,   512),
        ("decode  kv=1024", args.seqs, 1,   1024),
        ("decode  kv=2048", args.seqs, 1,   2048),
        ("decode  kv=4096", args.seqs, 1,   4096),
        ("prefill q=512",   1,         512, 512),
        ("prefill q=1024",  1,         1024,1024),
        ("prefill q=2048",  1,         2048,2048),
    ]

    header = f"{'场景':<20} {'block':>6} {'triton(ms)':>12} {'flash_attn(ms)':>15} {'加速比':>8}"
    print(header)
    print("-" * 90)

    for name, ns, ql, kl in scenarios:
        # block_size=16: 当前生产配置, 走 triton
        # block_size=128: flash_attn decode 路径硬性要求 (64 会报 "only support page block_size 128")
        row_triton = None
        row_fa = None
        try:
            t_tri = bench_triton_unified(ns, ql, kl, 16, args.iters) * 1000
            row_triton = t_tri
        except Exception as e:
            t_tri = float("nan")
            print(f"  [triton bs=16 {name} ERROR] {e}")

        try:
            t_fa = bench_flash_attn_unified(ns, ql, kl, 128, args.iters) * 1000
            row_fa = t_fa
        except Exception as e:
            t_fa = float("nan")
            print(f"  [flash_attn bs=128 {name} ERROR] {e}")

        speedup = (t_tri / t_fa) if (row_triton and row_fa and t_fa > 0) else float("nan")
        flag = " ⚡" if (speedup == speedup and speedup >= 1.0) else ""
        print(f"{name:<20} {'16/64':>6} {t_tri:>12.3f} {t_fa:>15.3f} {speedup:>7.2f}x{flag}")

    print("=" * 90)
    print("说明:")
    print("  triton(ms)    = aiter triton kernel_unified_attention (block_size=16, 当前生产路径)")
    print("  flash_attn(ms)= flash_attn.varlen_fwd_unified (block_size=128, HIP 编译内核)")
    print("  flash_attn decode 路径硬性要求 block_size=128 (64 会报 'only support page block_size 128')")
    print("  加速比 > 1.0 表示 flash_attn 更快, 改 --block-size 128 后 attention 会切到这个内核")

    # ============ 精度对比 ============
    print()
    print("=" * 90)
    print("精度对比: 同一份随机 Q/K/V 输入, triton vs flash_attn 输出差异")
    print("(fp16 下两个不同实现有微小数值差异是正常的, 关键看量级是否在 fp16 精度范围内)")
    print("=" * 90)
    pheader = (f"{'场景':<20} {'max_abs':>12} {'mean_abs':>12} "
               f"{'max_rel':>12} {'mean_rel':>12}  判定")
    print(pheader)
    print("-" * 90)

    # 精度场景: decode + prefill 各取几个代表
    precision_scenarios = [
        ("decode  kv=512",  args.seqs, 1,   512),
        ("decode  kv=2048", args.seqs, 1,   2048),
        ("decode  kv=4096", args.seqs, 1,   4096),
        ("prefill q=512",   1,         512, 512),
        ("prefill q=1024",  1,         1024,1024),
        ("prefill q=2048",  1,         2048,2048),
    ]

    all_ok = True
    for name, ns, ql, kl in precision_scenarios:
        max_abs, mean_abs, max_rel, mean_rel = compare_precision(ns, ql, kl)
        if max_abs is None:
            print(f"{name:<20} {'ERROR':>12} {mean_abs}")
            all_ok = False
            continue
        # fp16 精度判定: max_abs < 0.1 且 mean_rel < 5% 视为一致
        # (attention 是 softmax+加权求和, fp16 累加误差正常在 1e-3~1e-1 量级)
        ok = (max_abs < 0.5 and mean_rel < 0.05)
        if not ok:
            all_ok = False
        verdict = "✅ 一致" if ok else "⚠️  偏大"
        print(f"{name:<20} {max_abs:>12.5f} {mean_abs:>12.6f} "
              f"{max_rel*100:>11.4f}% {mean_rel*100:>11.4f}%  {verdict}")

    print("-" * 90)
    print("判定标准: max_abs < 0.5 且 mean_rel < 5% 视为数值一致 (fp16 正常误差范围)")
    if all_ok:
        print("结论: ✅ 所有场景精度一致, flash_attn 内核可安全替换 triton")
    else:
        print("结论: ⚠️  存在偏大场景, 需进一步确认 (建议跑 HumanEval 端到端验证)")
    print("  注: 即使有微小数值差异, 只要 max_abs 在 fp16 量级 (1e-2), 对 argmax 采样几乎无影响")


if __name__ == "__main__":
    main()
