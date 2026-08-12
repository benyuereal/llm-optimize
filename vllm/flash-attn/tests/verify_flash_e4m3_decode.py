"""验证路径2 flash e4m3 mixed kernel (decode) 是否 crash + 精度.

全 fp8 e4m3 (Q/K/V 都是 e4m3) 调 varlen_fwd_unified, max_seqlen_q=1 走 decode.
gemma4 形状: head_size=256, GQA 32q/16kv, block_size=128, sliding_window=1024.
对比 fp16 参考算 cos_sim / MAE.

bshd layout: K/V cache = [num_blocks, BLOCK, H_KV, D]
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
    # bshd: [num_blocks, BLOCK, H_KV, D]
    return torch.randn(num_blocks, BLOCK, H_KV, D, device=DEV, dtype=dtype) * 0.1

def to_fp8(x):
    return x.to(torch.float8_e4m3fn)

def main():
    from flash_attn import varlen_fwd_unified

    num_blocks_per_seq = (SEQ_K + BLOCK - 1) // BLOCK
    num_blocks = BS * num_blocks_per_seq
    cu_seqlens_q = torch.arange(0, BS + 1, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((BS,), SEQ_K, device=DEV, dtype=torch.int32)
    block_table = torch.arange(0, num_blocks, device=DEV, dtype=torch.int32).reshape(BS, num_blocks_per_seq)
    q_descale = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    k_descale = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    v_descale = torch.tensor([1.0], device=DEV, dtype=torch.float32)

    # ---- 同源数据 ----
    torch.manual_seed(42)
    q_fp16 = torch.randn(BS, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    k_fp16 = make_kv(num_blocks, torch.float16)
    v_fp16 = make_kv(num_blocks, torch.float16)

    print("=== fp16 参考 (decode) ===")
    try:
        out_ref = varlen_fwd_unified(
            q_fp16, k_fp16, v_fp16, cu_seqlens_q, seqused_k, block_table,
            max_seqlen_q=1, max_seqlen_k=SEQ_K,
            causal=True, window_size=(SLIDING, 0), layout=LAYOUT,
        )
        print(f"  out shape={tuple(out_ref.shape)} dtype={out_ref.dtype} "
              f"abs_mean={out_ref.abs().mean().item():.6f} "
              f"nan={torch.isnan(out_ref).any().item()} inf={torch.isinf(out_ref).any().item()}")
    except Exception as e:
        import traceback
        print(f"  fp16 参考 CRASH: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    print("\n=== 全 fp8 e4m3 (path2 kernel, decode) ===")
    q_fp8 = to_fp8(q_fp16)
    k_fp8 = to_fp8(k_fp16)
    v_fp8 = to_fp8(v_fp16)
    try:
        out_fp8 = varlen_fwd_unified(
            q_fp8, k_fp8, v_fp8, cu_seqlens_q, seqused_k, block_table,
            max_seqlen_q=1, max_seqlen_k=SEQ_K,
            causal=True, window_size=(SLIDING, 0), layout=LAYOUT,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        )
        print(f"  out shape={tuple(out_fp8.shape)} dtype={out_fp8.dtype} "
              f"abs_mean={out_fp8.abs().mean().item():.6f} "
              f"nan={torch.isnan(out_fp8).any().item()} inf={torch.isinf(out_fp8).any().item()}")
    except Exception as e:
        import traceback
        print(f"  全 fp8 e4m3 CRASH: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    print("\n=== 同源精度对比 (fp16 ref vs fp8 e4m3) ===")
    ref = out_ref.float()
    f8 = out_fp8.float()
    cos = F.cosine_similarity(ref.flatten(), f8.flatten(), dim=0).item()
    mae = (ref - f8).abs().mean().item()
    maxerr = (ref - f8).abs().max().item()
    print(f"  cos_sim  = {cos:.6f}")
    print(f"  MAE      = {mae:.6f}")
    print(f"  max_err  = {maxerr:.6f}")
    print(f"  ref abs_mean = {ref.abs().mean().item():.6f}")
    print(f"  fp8 abs_mean = {f8.abs().mean().item():.6f}")

    ok = cos > 0.99 and not torch.isnan(out_fp8).any() and not torch.isinf(out_fp8).any()
    print(f"\n=== 结论: {'PASS (不crash, cos>0.99)' if ok else '需进一步分析'} ===")

if __name__ == "__main__":
    main()
