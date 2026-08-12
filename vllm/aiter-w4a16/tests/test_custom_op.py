#!/usr/bin/env python3
"""测试 custom_op 注册的 aiter gemm 能否被 torch.compile fullgraph 处理"""
import os
os.environ["VLLM_AITER_W4A16_PATCH"] = "1"
os.environ["AITER_ROOT"] = "/public/home/weishb/aiter"
import torch
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    _make_aiter_launch_fn, _AITER_MOD)

K,N,G,M=256,64,32,4
device='cuda'
w_int=torch.randint(0,16,(K,N),dtype=torch.int32,device=device)
zp_int=torch.randint(0,16,(K//G,N),dtype=torch.int32,device=device)
scales=torch.rand(K//G,N,dtype=torch.float16,device=device)*0.1+0.01
a=torch.randn(M,K,dtype=torch.float16,device=device)*0.5
shifts=torch.arange(8,device=device,dtype=torch.int32)*4; N8=N//8
b_q=torch.sum((w_int.view(K,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
qzeros=torch.sum((zp_int.view(K//G,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
AWQ_INV=[0,2,4,6,1,3,5,7]
w=((b_q.unsqueeze(-1)>>shifts)&0xF).reshape(K,N)
perm=(torch.arange(N//8,device=device)[:,None]*8+torch.tensor(AWQ_INV,device=device)).reshape(-1)
w_awq=w[:,perm].contiguous()
b_q_awq=torch.sum((w_awq.view(K,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
z=((qzeros.unsqueeze(-1)>>shifts)&0xF).reshape(K//G,N)
z_awq=z[:,perm].contiguous()
qzeros_awq=torch.sum((z_awq.view(K//G,N8,8)&0xF)<<shifts,dim=2,dtype=torch.int32).contiguous()
aq,az=_AITER_MOD.awq_reorder_and_repack(b_q_awq,qzeros_awq)
fn=_make_aiter_launch_fn(N,K,G)
out=fn(a,aq,scales,az)
print('custom_op launch OK:', out.shape, out.dtype)

def f(x,aq,sc,az):
    return fn(x,aq,sc,az)
fc=torch.compile(f,fullgraph=True)
out2=fc(a,aq,scales,az)
print('torch.compile fullgraph OK:', out2.shape)
print('diff:', (out.float()-out2.float()).abs().max().item())
