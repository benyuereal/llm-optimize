"""性能对比: e4m3->e5m2 转换 + flash 的开销.

A: fp16 flash (基线)
B: fp8 triton (Q-fp16 + KV-e4m3)
C: e4m3->e5m2 转换 + flash (Q cast e5m2, KV e4m3->e5m2)
D: 原生 e5m2 flash (理想, 无转换)

重点看 C 的转换开销 vs D, 以及 C vs A/B 的整体收益.
gemma4: head_size=256, GQA 32/16, block_size=128, sliding_window=1024, bshd.
"""
import torch
import time
import statistics

DEV = "cuda:0"
H_Q, H_KV, D, BLOCK, SLIDING = 32, 16, 256, 128, 1024
LAYOUT = "bshd"
WARMUP, ITERS = 10, 50

def make_kv(nb, dtype):
    return torch.randn(nb, BLOCK, H_KV, D, device=DEV, dtype=dtype) * 0.1

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
    return statistics.median(ts) * 1000

def setup(bs, seq_k, q_len):
    nbps = (seq_k + BLOCK - 1) // BLOCK
    nb = bs * nbps
    if q_len == 1:
        cu = torch.arange(0, bs + 1, device=DEV, dtype=torch.int32)
    else:
        cu = torch.arange(0, (bs + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    suk = torch.full((bs,), seq_k, device=DEV, dtype=torch.int32)
    bt = torch.arange(0, nb, device=DEV, dtype=torch.int32).reshape(bs, nbps)
    ds = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    torch.manual_seed(42)
    q_fp16 = torch.randn(bs * q_len, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    k_fp16 = make_kv(nb, torch.float16)
    v_fp16 = make_kv(nb, torch.float16)
    # e4m3 (vllm 存的)
    k_e4 = k_fp16.to(torch.float8_e4m3fn)
    v_e4 = v_fp16.to(torch.float8_e4m3fn)
    # e5m2 (原生)
    k_e5 = k_fp16.to(torch.float8_e5m2)
    v_e5 = v_fp16.to(torch.float8_e5m2)
    return dict(q_fp16=q_fp16, k_fp16=k_fp16, v_fp16=v_fp16,
                k_e4=k_e4, v_e4=v_e4, k_e5=k_e5, v_e5=v_e5,
                cu=cu, suk=suk, bt=bt, ds=ds, q_len=q_len, seq_k=seq_k)

def call_flash(a, q, k, v, descale=None):
    from flash_attn import varlen_fwd_unified
    kw = dict(max_seqlen_q=a["q_len"], max_seqlen_k=a["seq_k"], causal=True,
              window_size=(SLIDING, 0), layout=LAYOUT)
    if descale is not None:
        kw.update(q_descale=descale, k_descale=descale, v_descale=descale)
    varlen_fwd_unified(q, k, v, a["cu"], a["suk"], a["bt"], **kw)

def A_fp16_flash(a):
    call_flash(a, a["q_fp16"], a["k_fp16"], a["v_fp16"])

def D_e5m2_flash(a):
    q = a["q_fp16"].to(torch.float8_e5m2)
    call_flash(a, q, a["k_e5"], a["v_e5"], a["ds"])

def C_e4m2convert_flash(a):
    # 模拟实际: vllm 存 e4m3, 每次调 attention 时转 e5m2
    q = a["q_fp16"].to(torch.float8_e5m2)
    k = a["k_e4"].to(torch.float8_e5m2)   # 转换开销
    v = a["v_e4"].to(torch.float8_e5m2)   # 转换开销
    call_flash(a, q, k, v, a["ds"])

def B_fp8_triton(a):
    from aiter.ops.triton.unified_attention import unified_attention
    out = torch.empty(a["q_fp16"].shape[0], H_Q, D, device=DEV, dtype=torch.float16)
    k_scale = torch.ones(H_KV, device=DEV, dtype=torch.float32)
    unified_attention(
        q=a["q_fp16"], k=a["k_e4"], v=a["v_e4"], out=out,
        cu_seqlens_q=a["cu"], max_seqlen_q=a["q_len"],
        seqused_k=a["suk"], max_seqlen_k=a["seq_k"],
        softmax_scale=D ** (-0.5), causal=True, window_size=(SLIDING, 0),
        block_table=a["bt"], softcap=0.0,
        q_descale=None, k_descale=k_scale, v_descale=k_scale,
    )

def main():
    print(f"GPU: {torch.cuda.get_device_name(DEV)}  gemma4 SW={SLIDING}\n")
    print("=" * 95)
    print("DECODE (q_len=1)")
    print("=" * 95)
    print(f"{'bs':>4} {'seq_k':>6} | {'A:fp16fl':>9} {'B:fp8tri':>9} {'C:e4->e5fl':>10} {'D:e5m2fl':>9} | {'C/A':>5} {'C/B':>5} {'C/D':>5}")
    print("-" * 95)
    for bs in [1, 4, 8, 16, 32, 64, 128]:
        for seq_k in [2048, 8192]:
            try:
                a = setup(bs, seq_k, 1)
                t_a = bench(lambda: A_fp16_flash(a))
                t_b = bench(lambda: B_fp8_triton(a))
                t_c = bench(lambda: C_e4m2convert_flash(a))
                t_d = bench(lambda: D_e5m2_flash(a))
                print(f"{bs:>4} {seq_k:>6} | {t_a:>7.3f}ms {t_b:>7.3f}ms {t_c:>8.3f}ms {t_d:>7.3f}ms | "
                      f"{t_c/t_a:>4.2f}x {t_c/t_b:>4.2f}x {t_c/t_d:>4.2f}x")
            except Exception as e:
                print(f"{bs:>4} {seq_k:>6} | ERR: {type(e).__name__}: {str(e)[:50]}")

    print()
    print("=" * 95)
    print("PREFILL (q_len>=16)")
    print("=" * 95)
    print(f"{'bs':>4} {'q_len':>6} | {'A:fp16fl':>9} {'B:fp8tri':>9} {'C:e4->e5fl':>10} {'D:e5m2fl':>9} | {'C/A':>5} {'C/B':>5} {'C/D':>5}")
    print("-" * 95)
    for bs, ql in [(1, 512), (1, 2048), (1, 4096), (2, 2048), (4, 1024)]:
        try:
            a = setup(bs, ql, ql)
            t_a = bench(lambda: A_fp16_flash(a))
            t_b = bench(lambda: B_fp8_triton(a))
            t_c = bench(lambda: C_e4m2convert_flash(a))
            t_d = bench(lambda: D_e5m2_flash(a))
            print(f"{bs:>4} {ql:>6} | {t_a:>7.3f}ms {t_b:>7.3f}ms {t_c:>8.3f}ms {t_d:>7.3f}ms | "
                  f"{t_c/t_a:>4.2f}x {t_c/t_b:>4.2f}x {t_c/t_d:>4.2f}x")
        except Exception as e:
            print(f"{bs:>4} {ql:>6} | ERR: {type(e).__name__}: {str(e)[:50]}")

if __name__ == "__main__":
    main()
