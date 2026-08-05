#!/usr/bin/env python3
"""验证 aiter 内部自洽: awq_reorder_and_repack+gemm_a16w4 vs awq_dequantize_torch+matmul
直接用 aiter 测试的同款构造,确认 kernel 本身正确,再定位 GPTQ->AWQ 转换问题。"""
import os, sys, types, importlib.util, logging
pkg = types.ModuleType("aiter")
pkg.__path__=["/public/home/weishb/aiter/aiter"]
pkg.__spec__=importlib.util.spec_from_file_location("aiter","/public/home/weishb/aiter/aiter/__init__.py",submodule_search_locations=["/public/home/weishb/aiter/aiter"])
pkg.logger=logging.getLogger("aiter"); sys.modules["aiter"]=pkg
for sub in ["aiter.ops","aiter.ops.triton","aiter.ops.triton.utils"]:
    m=types.ModuleType(sub); m.__path__=["/public/home/weishb/aiter/"+sub.replace(".","/")]; sys.modules[sub]=m
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
_load("aiter.ops.triton.utils.core","/public/home/weishb/aiter/aiter/ops/triton/utils/core.py")
_load("aiter.ops.triton.utils.arch_info","/public/home/weishb/aiter/aiter/ops/triton/utils/arch_info.py")
s=importlib.util.spec_from_file_location("aiter.ops.triton.gemm_a16w4","/public/home/weishb/aiter/aiter/ops/triton/gemm_a16w4.py")
_g=importlib.util.module_from_spec(s); sys.modules["aiter.ops.triton.gemm_a16w4"]=_g; s.loader.exec_module(_g)
import torch

def awq_dequantize_torch(qweight, scales, qzeros, group_size):
    """aiter 测试里的参考实现 — qweight [K,N//8] int32 (AWQ原始), scales [K//G,N], qzeros [K//G,N//8] int32"""
    if group_size==-1: group_size=qweight.shape[0]
    bits=4
    shifts=torch.arange(0,32,bits,device=qzeros.device)
    iweights=torch.bitwise_right_shift(qweight[:,:,None],shifts[None,None,:]).to(torch.int8)
    iweights=iweights.view(iweights.shape[0],-1)
    zeros=torch.bitwise_right_shift(qzeros[:,:,None],shifts[None,None,:]).to(torch.int8)
    zeros=zeros.view(qzeros.shape[0],-1)
    zeros=_g.reverse_awq_order(zeros)
    iweights=_g.reverse_awq_order(iweights)
    iweights=torch.bitwise_and(iweights,(2**bits)-1)
    zeros=torch.bitwise_and(zeros,(2**bits)-1)
    scales=scales.repeat_interleave(group_size,dim=0)
    zeros=zeros.repeat_interleave(group_size,dim=0)
    return (iweights-zeros)*scales

def run(M,K,N,G,seed=0):
    torch.manual_seed(seed); device="cuda"
    input=torch.rand(M,K,dtype=torch.float16,device=device)
    # AWQ 原始格式输入 (随机 int32)
    qweight=torch.randint(0,torch.iinfo(torch.int32).max,(K,N//8),device=device)
    qzeros=torch.randint(0,torch.iinfo(torch.int32).max,(K//G,N//8),device=device)
    scales=torch.rand(K//G,N,dtype=torch.float16,device=device)
    # aiter gemm
    aq,az=_g.awq_reorder_and_repack(qweight,qzeros)
    out_triton=_g.gemm_a16w4(input,aq,scales,az)
    # torch 参考
    deq=awq_dequantize_torch(qweight,scales,qzeros,G)
    out_torch=torch.matmul(input,deq.to(torch.float16))
    diff=(out_triton.float()-out_torch.float()).abs()
    cos=torch.nn.functional.cosine_similarity(out_triton.float().flatten(),out_torch.float().flatten(),dim=0).item()
    print(f"[M={M},K={K},N={N},G={G}] aiter自洽: max_abs={diff.max():.4f} mean_abs={diff.mean():.6f} cos_sim={cos:.6f}")
    return cos

print("="*60); print("aiter 内部自洽验证 (AWQ原始输入)"); print("="*60)
run(4,256,512,32)
run(16,1024,1024,32)
run(4,5376,1344,32)
run(4,5376,3584,32)
