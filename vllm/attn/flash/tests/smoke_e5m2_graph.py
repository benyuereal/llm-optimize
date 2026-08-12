"""e5m2 flash dispatch + CUDA graph capture 冒烟测试.
模拟 gemma4: head_size=256, block_size=128, bshd, KV=e5m2, Q=fp16.
验证:
 1. 普通 forward 能跑通 (走 hg_prefix_decode / hg_prefix_prefill)
 2. CUDA graph capture 期间不报 hipErrorStreamCaptureUnsupported
"""
import torch
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

torch.manual_seed(0)
dev = torch.device('cuda:0')
torch.cuda.set_device(dev)

from flash_attn import varlen_fwd_unified

# gemma4 参数
num_q_heads = 32
num_kv_heads = 16
head_size = 256
block_size = 128
num_blocks = 4
bs = 2  # num_seqs

def make_kv(num_blocks, block_size, nh, hs, dtype, dev):
    # bshd: [num_blocks, block_size, num_kv_heads, head_size]
    t = torch.randn(num_blocks, block_size, nh, hs, dtype=torch.float16, device=dev) * 0.1
    return t.to(dtype)

def run_decode():
    max_seqlen_q = 1
    q = torch.randn(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev) * 0.1
    k = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    v = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([block_size, block_size], dtype=torch.int32, device=dev)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
    out = torch.empty(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev)

    # 模拟 unified_attention.py 的 fp8 分支: cast Q + unit descale
    _q_flash = q.to(k.dtype)
    from aiter.ops.triton.unified_attention import _get_unit_descale
    _ud = _get_unit_descale(dev)

    ret = varlen_fwd_unified(
        q=_q_flash, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=block_table,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=block_size,
        softmax_scale=head_size ** -0.5, causal=True,
        window_size=(1024, 0),
        out=out, q_descale=_ud, k_descale=_ud, v_descale=_ud,
    )
    o = ret[0] if isinstance(ret, tuple) else ret
    assert o.shape == (bs * max_seqlen_q, num_q_heads, head_size), o.shape
    assert not torch.isnan(o).any(), "decode out has NaN"
    print(f'PASS decode: out shape={tuple(o.shape)} mean={o.float().mean().item():.4f} no NaN')

def run_prefill():
    max_seqlen_q = 64
    q = torch.randn(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev) * 0.1
    k = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    v = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    cu_seqlens_q = torch.tensor([0, 64, 128], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([block_size, block_size], dtype=torch.int32, device=dev)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
    out = torch.empty(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev)

    _q_flash = q.to(k.dtype)
    from aiter.ops.triton.unified_attention import _get_unit_descale
    _ud = _get_unit_descale(dev)

    ret = varlen_fwd_unified(
        q=_q_flash, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=block_table,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=block_size,
        softmax_scale=head_size ** -0.5, causal=True,
        window_size=(1024, 0),
        out=out, q_descale=_ud, k_descale=_ud, v_descale=_ud,
    )
    o = ret[0] if isinstance(ret, tuple) else ret
    assert o.shape == (bs * max_seqlen_q, num_q_heads, head_size), o.shape
    assert not torch.isnan(o).any(), "prefill out has NaN"
    print(f'PASS prefill: out shape={tuple(o.shape)} mean={o.float().mean().item():.4f} no NaN')

def run_decode_graph_capture():
    """模拟 vllm cudagraph capture: 在 capture 期间跑 decode forward."""
    max_seqlen_q = 1
    q = torch.randn(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev) * 0.1
    k = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    v = make_kv(num_blocks, block_size, num_kv_heads, head_size, torch.float8_e5m2, dev)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([block_size, block_size], dtype=torch.int32, device=dev)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
    out = torch.empty(bs * max_seqlen_q, num_q_heads, head_size, dtype=torch.float16, device=dev)

    from aiter.ops.triton.unified_attention import _get_unit_descale
    _ud = _get_unit_descale(dev)  # capture 前确保已缓存

    # warmup (非 capture)
    for _ in range(3):
        _q_flash = q.to(k.dtype)
        varlen_fwd_unified(
            q=_q_flash, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=block_table,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=block_size,
            softmax_scale=head_size ** -0.5, causal=True, window_size=(1024, 0),
            out=out, q_descale=_ud, k_descale=_ud, v_descale=_ud,
        )
    torch.cuda.synchronize()

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _q_flash = q.to(k.dtype)
        _ud2 = _get_unit_descale(dev)  # capture 期间只读 dict, 不创建
        varlen_fwd_unified(
            q=_q_flash, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=block_table,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=block_size,
            softmax_scale=head_size ** -0.5, causal=True, window_size=(1024, 0),
            out=out, q_descale=_ud2, k_descale=_ud2, v_descale=_ud2,
        )
    print('PASS decode CUDA graph capture: 未报 hipErrorStreamCaptureUnsupported')
    # replay 一次
    q.copy_(torch.randn_like(q) * 0.1)
    g.replay()
    torch.cuda.synchronize()
    assert not torch.isnan(out).any(), "graph replay out has NaN"
    print(f'PASS decode graph replay: out mean={out.float().mean().item():.4f} no NaN')

if __name__ == '__main__':
    run_decode()
    run_prefill()
    run_decode_graph_capture()
    print('\n全部通过 ✅')
