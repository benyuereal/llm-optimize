#!/usr/bin/env python3
"""性能测试: aiter gemm_a16w4 vs vllm triton_w4a16 (group_size=32)
真实形状 (gemma-4-31B, TP=4):
  q/k/v_proj: K=5376, N=5376/4=1344 (但GQA k/v: N=5376/16*2=... 实际q=1344,kv=...)
  o_proj:     K=5376, N=5376
  gate/up:    K=5376, N=14336/4=3584
  down:       K=14336, N=5376
测试 M (token数) 覆盖 bs=4 + MTP(accept~3.68): M=1,4,8,16,32
"""
import os, sys, types, importlib.util, logging, json
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
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
import torch

AWQ_INV=[0,2,4,6,1,3,5,7]
def gptq_to_awq_packed(b_q_gptq, qzeros_gptq):
    K,N8=b_q_gptq.shape; N=N8*8
    shifts=torch.arange(8,device=b_q_gptq.device,dtype=torch.int32)*4
    w=((b_q_gptq.unsqueeze(-1)>>shifts)&0xF).reshape(K,N)
    perm=(torch.arange(N//8,device=b_q_gptq.device)[:,None]*8+torch.tensor(AWQ_INV,device=b_q_gptq.device)).reshape(-1)
    w_awq=w[:,perm].contiguous()
    b_q_awq=torch.sum((w_awq.view(K,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
    KG=qzeros_gptq.shape[0]
    z=((qzeros_gptq.unsqueeze(-1)>>shifts)&0xF).reshape(KG,N)
    z_awq=z[:,perm].contiguous()
    qzeros_awq=torch.sum((z_awq.view(KG,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
    return b_q_awq,qzeros_awq

def bench(fn, args, warmup=20, iters=100):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(*args)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters  # ms

def make_inputs(K,N,G,M,device="cuda",seed=0):
    torch.manual_seed(seed)
    w_int=torch.randint(0,16,(K,N),dtype=torch.int32,device=device)
    zp_int=torch.randint(0,16,(K//G,N),dtype=torch.int32,device=device)
    scales=torch.rand(K//G,N,dtype=torch.float16,device=device)*0.1+0.01
    a=torch.randn(M,K,dtype=torch.float16,device=device)*0.5
    shifts=torch.arange(8,device=device,dtype=torch.int32)*4; N8=N//8
    b_q=torch.sum((w_int.view(K,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
    qzeros=torch.sum((zp_int.view(K//G,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
    scales=scales.contiguous()
    # aiter 预处理 (一次性, 不计入 GEMM 时间)
    b_q_awq,qzeros_awq=gptq_to_awq_packed(b_q,qzeros)
    aq,az=_g.awq_reorder_and_repack(b_q_awq,qzeros_awq)
    return a,b_q,scales,qzeros,aq,az

SHAPES=[
    ("q_proj",  5376, 1344),   # K, N (TP4)
    ("o_proj",  5376, 5376),
    ("gate",    5376, 3584),
    ("down",    14336,5376),
]
G=32
M_LIST=[1,4,8,16,32]

results=[]
print(f"{'shape':12s} {'M':>4s} {'vllm_us':>10s} {'aiter_us':>10s} {'speedup':>8s}")
print("-"*50)
for name,K,N in SHAPES:
    for M in M_LIST:
        a,b_q,scales,qzeros,aq,az=make_inputs(K,N,G,M)
        # vllm
        t_v=bench(triton_w4a16_gemm,(a,b_q,scales,qzeros,G,0))
        # aiter
        t_a=bench(_g.gemm_a16w4,(a,aq,scales,az))
        sp=t_v/t_a
        print(f"{name:12s} {M:4d} {t_v*1000:10.2f} {t_a*1000:10.2f} {sp:7.2f}x")
        results.append({"shape":name,"M":M,"K":K,"N":N,"vllm_us":t_v*1000,"aiter_us":t_a*1000,"speedup":sp})

with open("/public/home/weishb/test/bench_aiter_vs_vllm.json","w") as f:
    json.dump(results,f,indent=2)
print("\n结果已存 bench_aiter_vs_vllm.json")
