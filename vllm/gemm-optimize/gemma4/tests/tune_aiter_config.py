#!/usr/bin/env python3
"""为 aiter gemm_a16w4 在 BW10 (gfx936) 上调优 tile config。
手动 sweep BLOCK_SIZE_M/N/K + SPLITK + num_warps, 找最优, 生成 json config。"""
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
    b_q_awq,qzeros_awq=gptq_to_awq_packed(b_q,qzeros)
    aq,az=_g.awq_reorder_and_repack(b_q_awq,qzeros_awq)
    return a,aq,scales,az

def run_with_config(input, qweight, scales, qzeros, M, N, K, G, cfg):
    """手动构造完整 config 调 awq_gemm_triton_impl"""
    group_size=G
    config={
        "BLOCK_SIZE_M": cfg["BM"],
        "BLOCK_SIZE_N": cfg["BN"],
        "BLOCK_SIZE_K": cfg["BK"],
        "SCHEDULER": 0,
        "SPLITK": cfg["SPLITK"],
        "D_SHAPE": (M, N),
        "D_DTYPE": 16,
        "DP_TILES": 0,
        "DANGLING_TILES": 0,
        "NUM_CUS": 0,
        "NUM_CUS_STREAMK": 0,
        "NUM_GROUPS": (cfg["BK"] + group_size - 1) // group_size,
        "USE_REDUCE_KERNEL": False,
        "num_warps": cfg["NW"],
        "num_stages": cfg["NS"],
    }
    return _g.awq_gemm_triton_impl(input, qweight, scales, qzeros, config, config.copy(), _g.awq_gemm_kernel)

def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

# 搜索空间 (针对小 M, BW10)
G=32
SHAPES=[("q_proj",5376,1344),("gate",5376,3584),("down",14336,5376),("o_proj",5376,5376)]
M_LIST=[1,4,8,16]
GRID=[
    {"BM":16,"BN":64,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":16,"BN":128,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":16,"BN":128,"BK":64,"SPLITK":1,"NW":4,"NS":1},
    {"BM":16,"BN":64,"BK":64,"SPLITK":1,"NW":4,"NS":1},
    {"BM":16,"BN":256,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":16,"BN":128,"BK":32,"SPLITK":1,"NW":8,"NS":1},
    {"BM":16,"BN":128,"BK":64,"SPLITK":1,"NW":8,"NS":1},
    {"BM":16,"BN":64,"BK":32,"SPLITK":2,"NW":4,"NS":1},
    {"BM":16,"BN":128,"BK":32,"SPLITK":2,"NW":4,"NS":1},
    {"BM":32,"BN":64,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":32,"BN":128,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":32,"BN":128,"BK":64,"SPLITK":1,"NW":4,"NS":1},
    {"BM":32,"BN":64,"BK":64,"SPLITK":1,"NW":8,"NS":1},
    {"BM":8,"BN":128,"BK":32,"SPLITK":1,"NW":4,"NS":1},
    {"BM":8,"BN":64,"BK":32,"SPLITK":1,"NW":4,"NS":1},
]

best={}
print("调优 aiter gemm_a16w4 (BW10, group_size=32)...")
for name,K,N in SHAPES:
    for M in M_LIST:
        a,aq,scales,az=make_inputs(K,N,G,M)
        best_t=1e9; best_cfg=None
        for cfg in GRID:
            try:
                t=bench(lambda: run_with_config(a,aq,scales,az,M,N,K,G,cfg))
                if t<best_t: best_t=t; best_cfg=cfg
            except Exception as e:
                pass
        key=f"{name}_M{M}"
        best[key]={"K":K,"N":N,"M":M,"best_us":best_t*1000,"cfg":best_cfg}
        print(f"{key:16s} K={K:5d} N={N:5d} best={best_t*1000:7.2f}us  cfg={best_cfg}")

with open("/public/home/weishb/test/aiter_tuned_best.json","w") as f:
    json.dump(best,f,indent=2)
print("\n存 aiter_tuned_best.json")
