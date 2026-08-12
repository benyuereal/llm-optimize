"""验证 e4m3→e5m2 转换 + flash 的精度.

C: Q cast e5m2 + KV(e4m3→e5m2 转换) + flash  -- 简化方案
D: Q cast e5m2 + KV(原生 e5m2) + flash       -- 理想上限

对比 fp16 参考, 看 C 的精度损失 vs D.
gemma4: head_size=256, GQA 32/16, block_size=128, sliding_window=1024, bshd.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda:0"

H_Q, H_KV, D, BLOCK, SLIDING = 32, 16, 256, 128, 1024
LAYOUT = "bshd"

def make_kv(num_blocks, dtype):
    return torch.randn(num_blocks, BLOCK, H_KV, D, device=DEV, dtype=dtype) * 0.1

def run_flash(q, k, v, bs, seq_k, q_len, descale=None):
    from flash_attn import varlen_fwd_unified
    nbps = (seq_k + BLOCK - 1) // BLOCK
    nb = bs * nbps
    if q_len == 1:
        cu = torch.arange(0, bs + 1, device=DEV, dtype=torch.int32)
    else:
        cu = torch.arange(0, (bs + 1) * q_len, q_len, device=DEV, dtype=torch.int32)
    suk = torch.full((bs,), seq_k, device=DEV, dtype=torch.int32)
    bt = torch.arange(0, nb, device=DEV, dtype=torch.int32).reshape(bs, nbps)
    kw = dict(max_seqlen_q=q_len, max_seqlen_k=seq_k, causal=True,
              window_size=(SLIDING, 0), layout=LAYOUT)
    if descale is not None:
        kw.update(q_descale=descale, k_descale=descale, v_descale=descale)
    return varlen_fwd_unified(q, k, v, cu, suk, bt, **kw)

def main():
    bs, seq_k = 4, 2048
    nbps = (seq_k + BLOCK - 1) // BLOCK
    nb = bs * nbps

    torch.manual_seed(42)
    q_fp16 = torch.randn(bs, H_Q, D, device=DEV, dtype=torch.float16) * 0.1
    k_fp16 = make_kv(nb, torch.float16)
    v_fp16 = make_kv(nb, torch.float16)

    # A: fp16 参考
    out_ref = run_flash(q_fp16, k_fp16, v_fp16, bs, seq_k, 1)
    print(f"A fp16参考:    abs_mean={out_ref.abs().mean().item():.6f}")

    # D: 原生 e5m2 (Q cast e5m2, KV e5m2) -- 理想上限
    ds = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    out_d = run_flash(q_fp16.to(torch.float8_e5m2),
                      k_fp16.to(torch.float8_e5m2),
                      v_fp16.to(torch.float8_e5m2), bs, seq_k, 1, ds)
    print(f"D 原生e5m2:    abs_mean={out_d.abs().mean().item():.6f}")

    # C: e4m3→e5m2 转换 (Q cast e5m2, KV 先 e4m3 再转 e5m2)
    q_e5 = q_fp16.to(torch.float8_e5m2)
    k_c = k_fp16.to(torch.float8_e4m3fn).to(torch.float8_e5m2)   # e4m3 存, 用时转 e5m2
    v_c = v_fp16.to(torch.float8_e4m3fn).to(torch.float8_e5m2)
    out_c = run_flash(q_e5, k_c, v_c, bs, seq_k, 1, ds)
    print(f"C e4m3->e5m2:  abs_mean={out_c.abs().mean().item():.6f}")

    # 精度
    r = out_ref.float()
    print("\n=== 精度 (vs fp16 参考) ===")
    for name, o in [("D 原生e5m2", out_d), ("C e4m3->e5m2", out_c)]:
        f = o.float()
        cos = F.cosine_similarity(r.flatten(), f.flatten(), dim=0).item()
        mae = (r - f).abs().mean().item()
        print(f"  {name}: cos={cos:.6f}  MAE={mae:.6f}  abs_mean={f.abs().mean().item():.6f}")

    # C vs D 差异 (转换本身的损失)
    cd = F.cosine_similarity(out_d.float().flatten(), out_c.float().flatten(), dim=0).item()
    print(f"\n  C vs D (转换损失): cos={cd:.6f}")

if __name__ == "__main__":
    main()
