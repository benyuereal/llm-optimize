"""验证: prefill 也能走 flash + e5m2 KV 吗?

prefill 形状: max_seqlen_q=512 (>=16), Q cast e5m2 + KV e5m2.
对比 fp16 参考.

gemma4: head_size=256, GQA 32/16, block_size=128, sliding_window=1024, bshd.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda:0"

H_Q, H_KV, D, BLOCK = 32, 16, 256, 128
BS = 2
Q_LEN = 512       # prefill q length (>=16 走 prefill 分支)
SEQ_K = 512       # prefill 时 kv 长度 = q 长度 (causal)
SLIDING = 1024
LAYOUT = "bshd"

def make_kv(num_blocks, dtype):
    return torch.randn(num_blocks, BLOCK, H_KV, D, device=DEV, dtype=dtype) * 0.1

def main():
    from flash_attn import varlen_fwd_unified

    nbps = (SEQ_K + BLOCK - 1) // BLOCK
    nb = BS * nbps
    # prefill: 每个 seq 有 Q_LEN 个 query token
    cu = torch.tensor([0, Q_LEN, 2*Q_LEN], device=DEV, dtype=torch.int32)
    suk = torch.full((BS,), SEQ_K, device=DEV, dtype=torch.int32)
    bt = torch.arange(0, nb, device=DEV, dtype=torch.int32).reshape(BS, nbps)
    ds = torch.tensor([1.0], device=DEV, dtype=torch.float32)

    torch.manual_seed(42)
    q_fp16 = torch.randn(BS * Q_LEN, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    k_fp16 = make_kv(nb, torch.float16)
    v_fp16 = make_kv(nb, torch.float16)

    # fp16 参考 (prefill)
    print("=== fp16 参考 (prefill, Q_LEN=512) ===")
    try:
        out_ref = varlen_fwd_unified(
            q_fp16, k_fp16, v_fp16, cu, suk, bt,
            max_seqlen_q=Q_LEN, max_seqlen_k=SEQ_K,
            causal=True, window_size=(SLIDING, 0), layout=LAYOUT,
        )
        print(f"  out={tuple(out_ref.shape)} dtype={out_ref.dtype} abs_mean={out_ref.abs().mean().item():.6f} "
              f"nan={torch.isnan(out_ref).any().item()}")
    except Exception as e:
        import traceback
        print(f"  fp16 prefill CRASH: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return

    # e5m2: Q cast e5m2 + KV e5m2 (prefill)
    print("\n=== Q-fp16->e5m2 + KV-e5m2, flash prefill ===")
    q_e5 = q_fp16.to(torch.float8_e5m2)
    k_e5 = k_fp16.to(torch.float8_e5m2)
    v_e5 = v_fp16.to(torch.float8_e5m2)
    try:
        out_e5 = varlen_fwd_unified(
            q_e5, k_e5, v_e5, cu, suk, bt,
            max_seqlen_q=Q_LEN, max_seqlen_k=SEQ_K,
            causal=True, window_size=(SLIDING, 0), layout=LAYOUT,
            q_descale=ds, k_descale=ds, v_descale=ds,
        )
        print(f"  out={tuple(out_e5.shape)} dtype={out_e5.dtype} abs_mean={out_e5.abs().mean().item():.6f} "
              f"nan={torch.isnan(out_e5).any().item()} inf={torch.isinf(out_e5).any().item()}")
    except Exception as e:
        import traceback
        print(f"  e5m2 prefill CRASH: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return

    # 精度
    print("\n=== 精度: fp16 ref vs (Q->e5m2 + KV-e5m2) prefill ===")
    r, f = out_ref.float(), out_e5.float()
    cos = F.cosine_similarity(r.flatten(), f.flatten(), dim=0).item()
    mae = (r - f).abs().mean().item()
    maxerr = (r - f).abs().max().item()
    print(f"  cos_sim  = {cos:.6f}")
    print(f"  MAE      = {mae:.6f}")
    print(f"  max_err  = {maxerr:.6f}")
    print(f"  ref abs_mean = {r.abs().mean().item():.6f}")
    print(f"  e5m2 abs_mean = {f.abs().mean().item():.6f}")
    ok = cos > 0.99 and not torch.isnan(out_e5).any()
    print(f"\n=== 结论: {'PASS — prefill 也能 flash + e5m2' if ok else '需分析'} ===")

if __name__ == "__main__":
    main()
