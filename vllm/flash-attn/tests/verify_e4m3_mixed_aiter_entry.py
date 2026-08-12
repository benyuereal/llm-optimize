"""验证生产真实路径: Q fp16 + KV e4m3, 走 aiter.unified_attention 入口 (不是直接调 flash).

这模拟 vllm 实际调用: vllm 传 fp16 Q + e4m3 KV cache 给 unified_attention,
aiter 内部把 Q cast 成 e4m3 再调 flash mixed kernel.

gemma4 形状: head_size=256, GQA 32q/16kv, block_size=128, sliding_window=1024.
bshd layout. 测 decode (max_seqlen_q=1) 和 prefill (max_seqlen_q=512) 两种.

对比基准:
  - ref: fp16 Q + fp16 KV, 走同一个 unified_attention (非 fp8 分支, flash fp16)
  - fp8: fp16 Q + e4m3 KV, 走 fp8 分支 (flash e4m3 mixed kernel)
算 cos_sim / MAE.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda:0"

# gemma4 形状
H_Q = 32          # num_query_heads
H_KV = 16         # num_kv_heads (GQA)
D = 256           # head_size
BLOCK = 128       # paged block_size
BS = 4            # batch (sequences)
SEQ_K = 2048
SLIDING = 1024
LAYOUT = "bshd"


def make_kv(num_blocks, dtype):
    # bshd: [num_blocks, BLOCK, H_KV, D]. fp8 不支持 randn, 先 fp16 再 cast.
    t = torch.randn(num_blocks, BLOCK, H_KV, D, device=DEV, dtype=torch.float16) * 0.1
    return t.to(dtype)


def run_case(label, q_dtype, kv_dtype, max_seqlen_q, seq_k, tol_cos=0.99):
    """通过 aiter.unified_attention 入口跑, 返回 out."""
    from aiter.ops.triton.unified_attention import unified_attention

    num_blocks_per_seq = (seq_k + BLOCK - 1) // BLOCK
    num_blocks = BS * num_blocks_per_seq
    cu_seqlens_q = torch.arange(0, BS + 1, device=DEV, dtype=torch.int32)
    # decode: 每条 seq q_len=1, cu_seqlens = [0,1,2,3,4]; total q = BS
    # prefill: 每条 seq q_len=max_seqlen_q
    if max_seqlen_q == 1:
        cu_seqlens_q = torch.arange(0, BS + 1, device=DEV, dtype=torch.int32)
        num_q_tokens = BS
    else:
        cu_seqlens_q = torch.arange(0, (BS + 1) * max_seqlen_q, max_seqlen_q,
                                    device=DEV, dtype=torch.int32)
        num_q_tokens = BS * max_seqlen_q
    seqused_k = torch.full((BS,), seq_k, device=DEV, dtype=torch.int32)
    block_table = torch.arange(0, num_blocks, device=DEV,
                               dtype=torch.int32).reshape(BS, num_blocks_per_seq)

    torch.manual_seed(42)
    q = torch.randn(num_q_tokens, H_Q, D, device=DEV, dtype=q_dtype) * 0.1
    k = make_kv(num_blocks, kv_dtype)
    v = make_kv(num_blocks, kv_dtype)
    out = torch.empty(num_q_tokens, H_Q, D, device=DEV, dtype=torch.float16)

    # fp8 KV 走 triton 时需要 k_descale/v_descale (非 None), 用 unit scale 模拟生产.
    # 生产 vllm 传 layer._k_scale (标量 1.0 tensor for 无校准 fp8).
    if kv_dtype != q_dtype:
        k_descale = torch.tensor([1.0], device=DEV, dtype=torch.float32)
        v_descale = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    else:
        k_descale = None
        v_descale = None
    try:
        unified_attention(
            q=q, k=k, v=v, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k, max_seqlen_k=seq_k,
            softmax_scale=1.0 / (D ** 0.5),
            causal=True,
            window_size=(SLIDING, 0),
            block_table=block_table,
            softcap=0.0,
            q_descale=None, k_descale=k_descale, v_descale=v_descale,
        )
    except Exception as e:
        import traceback
        print(f"  [{label}] CRASH: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    print(f"  [{label}] out shape={tuple(out.shape)} dtype={out.dtype} "
          f"abs_mean={out.abs().mean().item():.6f} "
          f"nan={torch.isnan(out).any().item()} inf={torch.isinf(out).any().item()}")
    return out


def main():
    print("=" * 70)
    print("生产路径测试: Q fp16 + KV e4m3, 走 aiter.unified_attention 入口")
    print("=" * 70)

    # ---- DECODE ----
    print("\n### DECODE (max_seqlen_q=1) ###")
    ref = run_case("ref fp16Q+fp16KV", torch.float16, torch.float16,
                   max_seqlen_q=1, seq_k=SEQ_K)
    fp8 = run_case("fp8 fp16Q+e4m3KV", torch.float16, torch.float8_e4m3fn,
                   max_seqlen_q=1, seq_k=SEQ_K)
    if ref is not None and fp8 is not None:
        r, f = ref.float(), fp8.float()
        cos = F.cosine_similarity(r.flatten(), f.flatten(), dim=0).item()
        mae = (r - f).abs().mean().item()
        maxerr = (r - f).abs().max().item()
        print(f"  cos_sim  = {cos:.6f}")
        print(f"  MAE      = {mae:.6f}")
        print(f"  max_err  = {maxerr:.6f}")
        print(f"  ref abs_mean = {r.abs().mean().item():.6f}")
        print(f"  fp8 abs_mean = {f.abs().mean().item():.6f}")
        ok = cos > tol_cos if (tol_cos := 0.99) else False
        print(f"  DECODE 结论: {'PASS' if cos > 0.99 else 'FAIL'} (cos>0.99)")

    # ---- PREFILL ----
    print("\n### PREFILL (max_seqlen_q=512) ###")
    SEQ_K_P = 2048
    refp = run_case("ref fp16Q+fp16KV", torch.float16, torch.float16,
                    max_seqlen_q=512, seq_k=SEQ_K_P)
    fp8p = run_case("fp8 fp16Q+e4m3KV", torch.float16, torch.float8_e4m3fn,
                    max_seqlen_q=512, seq_k=SEQ_K_P)
    if refp is not None and fp8p is not None:
        r, f = refp.float(), fp8p.float()
        cos = F.cosine_similarity(r.flatten(), f.flatten(), dim=0).item()
        mae = (r - f).abs().mean().item()
        maxerr = (r - f).abs().max().item()
        print(f"  cos_sim  = {cos:.6f}")
        print(f"  MAE      = {mae:.6f}")
        print(f"  max_err  = {maxerr:.6f}")
        print(f"  ref abs_mean = {r.abs().mean().item():.6f}")
        print(f"  fp8 abs_mean = {f.abs().mean().item():.6f}")
        print(f"  PREFILL 结论: {'PASS' if cos > 0.99 else 'FAIL'} (cos>0.99)")


if __name__ == "__main__":
    main()
