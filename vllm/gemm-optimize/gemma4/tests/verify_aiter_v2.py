#!/usr/bin/env python3
"""
精度验证 v2: 关键修复 — GPTQ 输入需先转成 AWQ 列顺序再喂 awq_reorder_and_repack。

AWQ 打包列顺序: [0,4,1,5,2,6,3,7] (每8列)
GPTQ 打包列顺序: [0,1,2,3,4,5,6,7]
reverse_awq_order 把 AWQ顺序 -> [0,1,2,3,4,5,6,7]
所以喂 AWQ 格式才对。我们的 compressed-tensors 是 GPTQ,需要先把列重排成 AWQ 顺序。

做法: unpack GPTQ b_q -> [K,N] (GPTQ列序) -> 列重排成 AWQ序 -> 重新打包成 [K,N//8] int32 (AWQ)
然后 awq_reorder_and_repack 会 reverse 回 GPTQ序, 与 kernel 内部解包自洽。
"""
import os, sys, types, importlib.util, logging
pkg = types.ModuleType("aiter")
pkg.__path__ = ["/public/home/weishb/aiter/aiter"]
pkg.__spec__ = importlib.util.spec_from_file_location("aiter", "/public/home/weishb/aiter/aiter/__init__.py",
    submodule_search_locations=["/public/home/weishb/aiter/aiter"])
pkg.logger = logging.getLogger("aiter"); sys.modules["aiter"] = pkg
for sub in ["aiter.ops","aiter.ops.triton","aiter.ops.triton.utils"]:
    m = types.ModuleType(sub); m.__path__=["/public/home/weishb/aiter/"+sub.replace(".","/")]; sys.modules[sub]=m
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
_load("aiter.ops.triton.utils.core","/public/home/weishb/aiter/aiter/ops/triton/utils/core.py")
_load("aiter.ops.triton.utils.arch_info","/public/home/weishb/aiter/aiter/ops/triton/utils/arch_info.py")
s=importlib.util.spec_from_file_location("aiter.ops.triton.gemm_a16w4","/public/home/weishb/aiter/aiter/ops/triton/gemm_a16w4.py")
_g=importlib.util.module_from_spec(s); sys.modules["aiter.ops.triton.gemm_a16w4"]=_g; s.loader.exec_module(_g)

from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
import torch

# AWQ reverse order: reverse_awq_order 把 [c0..c7] -> [c0,c4,c1,c5,c2,c6,c3,c7]
# 要让 reverse 后 = GPTQ真值 [g0..g7], 需 c=[g0,g2,g4,g6,g1,g3,g5,g7]
# 即 AWQ 打包位 j 存 GPTQ 列 AWQ_INV[j], AWQ_INV = [0,2,4,6,1,3,5,7]
AWQ_INV = [0,2,4,6,1,3,5,7]
def gptq_to_awq_packed(b_q_gptq, qzeros_gptq):
    """b_q_gptq [K,N//8] int32 (GPTQ), 返回 [K,N//8] int32 (AWQ原始)"""
    K, N8 = b_q_gptq.shape
    N = N8 * 8
    shifts = torch.arange(8, device=b_q_gptq.device, dtype=torch.int32) * 4
    w = ((b_q_gptq.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N)  # GPTQ真值列序
    # AWQ 位 j <- GPTQ 列 AWQ_INV[j]
    perm = (torch.arange(N//8, device=b_q_gptq.device)[:, None] * 8 + torch.tensor(AWQ_INV, device=b_q_gptq.device)).reshape(-1)
    w_awq = w[:, perm].contiguous()
    b_q_awq = torch.sum((w_awq.view(K, N8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32).contiguous()
    KG = qzeros_gptq.shape[0]
    z = ((qzeros_gptq.unsqueeze(-1) >> shifts) & 0xF).reshape(KG, N)
    z_awq = z[:, perm].contiguous()
    qzeros_awq = torch.sum((z_awq.view(KG, N8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32).contiguous()
    return b_q_awq, qzeros_awq

def run_test(M, K, N, G, seed=0):
    torch.manual_seed(seed); device="cuda"
    w_int = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
    zp_int = torch.randint(0, 16, (K//G, N), dtype=torch.int32, device=device)
    scales = torch.rand(K//G, N, dtype=torch.float16, device=device)*0.1+0.01
    a = torch.randn(M, K, dtype=torch.float16, device=device)*0.5
    w_fp = (w_int.float()-zp_int.repeat_interleave(G,0).float())*scales.repeat_interleave(G,0).float()
    ref = (a.float() @ w_fp.float()).to(torch.float16)

    shifts = torch.arange(8, device=device, dtype=torch.int32)*4
    N8 = N//8
    b_q = torch.sum((w_int.view(K,N8,8)&0xF)<<shifts, dim=2, dtype=torch.int32).contiguous()
    qzeros = torch.sum((zp_int.view(K//G,N8,8)&0xF)<<shifts, dim=2, dtype=torch.int32).contiguous()
    scales = scales.contiguous()

    out_vllm = triton_w4a16_gemm(a=a, b_q=b_q, scales=scales, qzeros=qzeros, group_size=G, zp_bias=0)

    # GPTQ -> AWQ 打包, 再 aiter repack
    b_q_awq, qzeros_awq = gptq_to_awq_packed(b_q, qzeros)
    aq, az = _g.awq_reorder_and_repack(b_q_awq, qzeros_awq)
    out_aiter = _g.gemm_a16w4(a, aq, scales, az)

    def stats(name, out):
        diff=(out.float()-ref.float()).abs()
        cos=torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        print(f"  {name:18s} max_abs={diff.max():.4f} mean_abs={diff.mean():.6f} cos_sim={cos:.6f}")
        return cos
    print(f"\n[M={M},K={K},N={N},G={G}]")
    stats("vllm_triton", out_vllm)
    ca=stats("aiter_v2", out_aiter)
    diff=(out_aiter.float()-out_vllm.float()).abs()
    cosva=torch.nn.functional.cosine_similarity(out_aiter.float().flatten(), out_vllm.float().flatten(), dim=0).item()
    print(f"  {'aiter_vs_vllm':18s} max_abs={diff.max():.4f} cos_sim={cosva:.6f}")
    return cosva

if __name__=="__main__":
    print("="*70); print("精度验证 v2: GPTQ->AWQ 列重排"); print("="*70)
    run_test(4, 256, 512, 32)
    run_test(16, 1024, 1024, 32)
    run_test(4, 5376, 1344, 32)
    run_test(4, 5376, 3584, 32)
    run_test(1, 5376, 14336, 32)
