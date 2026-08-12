"""性能对比: 三个 attention 链路 (gemma4 形状).

链路:
  A. fp16 flash  (Q/KV fp16, flash)  -- 基线
  B. fp8 triton  (Q fp16, KV e4m3, triton chunked_pa_prefill / unified)  -- 路径1现状
  C. fp8 flash   (Q->e5m2, KV e5m2, flash)  -- 新方案

测 decode (max_seqlen_q=1) 和 prefill (max_seqlen_q>=16) 的 latency.
扫 batch 和 seq_len.

gemma4: head_size=256, GQA 32/16, block_size=128, sliding_window=1024, bshd.
"""
import torch
import time
import statistics

DEV = "cuda:0"
H_Q, H_KV, D, BLOCK, SLIDING = 32, 16, 256, 128, 1024
LAYOUT = "bshd"
WARMUP = 10
ITERS = 50

def make_kv(num_blocks, dtype):
    return torch.randn(num_blocks, BLOCK, H_KV, D, device=DEV, dtype=dtype) * 0.1

def bench(fn, warmup=WARMUP, iters=ITERS):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000  # ms

def setup(bs, seq_k, q_len):
    """返回 (q_fp16, q_e5, k_fp16, k_e5, v_fp16, v_e5, cu, suk, bt, ds, is_decode)."""
    nbps = (seq_k + BLOCK - 1) // BLOCK
    nb = bs * nbps
    is_decode = q_len == 1
    if is_decode:
        cu = torch.arange(0, bs + 1, device=DEV, dtype=torch.int32)
    else:
        cu = torch.arange(0, (bs + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    suk = torch.full((bs,), seq_k, device=DEV, dtype=torch.int32)
    bt = torch.arange(0, nb, device=DEV, dtype=torch.int32).reshape(bs, nbps)
    torch.manual_seed(42)
    q_fp16 = torch.randn(bs * q_len, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    q_e5 = q_fp16.to(torch.float8_e5m2)
    k_fp16 = make_kv(nb, torch.float16)
    v_fp16 = make_kv(nb, torch.float16)
    k_e5 = k_fp16.to(torch.float8_e5m2)
    v_e5 = v_fp16.to(torch.float8_e5m2)
    ds = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    return q_fp16, q_e5, k_fp16, k_e5, v_fp16, v_e5, cu, suk, bt, ds, is_decode

def run_flash_fp16(args):
    from flash_attn import varlen_fwd_unified
    q_fp16, _, k_fp16, _, v_fp16, _, cu, suk, bt, ds, is_decode = args
    ql = 1 if is_decode else q_fp16.shape[0] // (cu.numel() - 1)
    sk = suk[0].item()
    varlen_fwd_unified(q_fp16, k_fp16, v_fp16, cu, suk, bt,
                       max_seqlen_q=ql, max_seqlen_k=sk,
                       causal=True, window_size=(SLIDING, 0), layout=LAYOUT)

def run_flash_fp8(args):
    from flash_attn import varlen_fwd_unified
    _, q_e5, _, k_e5, _, v_e5, cu, suk, bt, ds, is_decode = args
    ql = 1 if is_decode else q_e5.shape[0] // (cu.numel() - 1)
    sk = suk[0].item()
    varlen_fwd_unified(q_e5, k_e5, v_e5, cu, suk, bt,
                       max_seqlen_q=ql, max_seqlen_k=sk,
                       causal=True, window_size=(SLIDING, 0), layout=LAYOUT,
                       q_descale=ds, k_descale=ds, v_descale=ds)

def run_triton_fp8(args):
    """fp8 triton: Q fp16 + KV e4m3, 走 aiter unified_attention triton 路径."""
    from aiter.ops.triton.unified_attention import unified_attention
    q_fp16, _, _, _, v_fp16, _, cu, suk, bt, ds, is_decode = args
    ql = 1 if is_decode else q_fp16.shape[0] // (cu.numel() - 1)
    sk = suk[0].item()
    nbps = bt.shape[1]
    nb = bt.shape[0] * nbps
    # triton 路径 KV 用 e4m3 (vllm 默认)
    torch.manual_seed(42)
    k_e4 = torch.randn(nb, BLOCK, H_KV, D, device=DEV, dtype=torch.float16).to(torch.float8_e4m3fn)
    v_e4 = torch.randn(nb, BLOCK, H_KV, D, device=DEV, dtype=torch.float16).to(torch.float8_e4m3fn)
    out = torch.empty(q_fp16.shape[0], H_Q, D, device=DEV, dtype=torch.float16)
    k_scale = torch.ones(H_KV, device=DEV, dtype=torch.float32)
    v_scale = torch.ones(H_KV, device=DEV, dtype=torch.float32)
    unified_attention(
        q=q_fp16, k=k_e4, v=v_e4, out=out,
        cu_seqlens_q=cu, max_seqlen_q=ql,
        seqused_k=suk, max_seqlen_k=sk,
        softmax_scale=D ** (-0.5),
        causal=True, window_size=(SLIDING, 0),
        block_table=bt, softcap=0.0,
        q_descale=None, k_descale=k_scale, v_descale=v_scale,
    )

def main():
    print(f"GPU: {torch.cuda.get_device_name(DEV)}")
    print(f"gemma4: H_Q={H_Q} H_KV={H_KV} D={D} BLOCK={BLOCK} SW={SLIDING}")
    print(f"warmup={WARMUP} iters={ITERS}\n")

    # ---- DECODE ----
    print("=" * 80)
    print("DECODE (max_seqlen_q=1)")
    print("=" * 80)
    print(f"{'bs':>4} {'seq_k':>6} | {'A:fp16flash':>12} {'B:fp8triton':>12} {'C:fp8flash':>12} | {'B/A':>6} {'C/A':>6} {'C/B':>6}")
    print("-" * 80)
    for bs in [1, 4, 8, 16, 32, 64, 128]:
        for seq_k in [2048, 8192]:
            try:
                args = setup(bs, seq_k, 1)
                t_a = bench(lambda: run_flash_fp16(args))
                t_c = bench(lambda: run_flash_fp8(args))
                t_b = bench(lambda: run_triton_fp8(args))
                print(f"{bs:>4} {seq_k:>6} | {t_a:>10.3f}ms {t_b:>10.3f}ms {t_c:>10.3f}ms | {t_b/t_a:>5.2f}x {t_c/t_a:>5.2f}x {t_c/t_b:>5.2f}x")
            except Exception as e:
                print(f"{bs:>4} {seq_k:>6} | ERR: {type(e).__name__}: {str(e)[:60]}")

    # ---- PREFILL ----
    print()
    print("=" * 80)
    print("PREFILL (max_seqlen_q>=16)")
    print("=" * 80)
    print(f"{'bs':>4} {'q_len':>6} | {'A:fp16flash':>12} {'B:fp8triton':>12} {'C:fp8flash':>12} | {'B/A':>6} {'C/A':>6} {'C/B':>6}")
    print("-" * 80)
    for bs, q_len in [(1, 512), (1, 2048), (1, 4096), (2, 2048), (4, 1024)]:
        try:
            args = setup(bs, q_len, q_len)
            t_a = bench(lambda: run_flash_fp16(args))
            t_c = bench(lambda: run_flash_fp8(args))
            t_b = bench(lambda: run_triton_fp8(args))
            print(f"{bs:>4} {q_len:>6} | {t_a:>10.3f}ms {t_b:>10.3f}ms {t_c:>10.3f}ms | {t_b/t_a:>5.2f}x {t_c/t_a:>5.2f}x {t_c/t_b:>5.2f}x")
        except Exception as e:
            print(f"{bs:>4} {q_len:>6} | ERR: {type(e).__name__}: {str(e)[:60]}")

if __name__ == "__main__":
    main()
