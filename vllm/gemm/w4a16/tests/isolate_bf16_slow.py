#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔离实验: 定位路B (aiter bf16) 慢 25-35% 的根因
卡1, 实际 layer 形状 (N=10752, K=5376, G=32, 有 splitk config)

4 个实验:
  ① fp16 in -> 原始 kernel (fp16 out)        基准
  ② bf16 in -> 改造版 kernel (bf16 out)       路B现状
  ③ bf16 in -> 转 fp16 喂 kernel -> 改造版 bf16 out   隔离 input load
  ④ fp16 in -> 改造版 kernel (fp16 out)       隔离输出 cast
"""
import os
import sys
import time
import json
import importlib.util

os.environ["HIP_VISIBLE_DEVICES"] = "1"  # 卡1

import torch

W4A16_DIR = "/public/home/weishb/llm-optimize/vllm/gemm/w4a16"
CONFIG_DIR = "/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs/gemm/awq_w4a16"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


orig = load_mod(os.path.join(W4A16_DIR, "aiter_gemm_a16w4.py"), "aiter_orig")
bf16 = load_mod(os.path.join(W4A16_DIR, "aiter_gemm_a16w4_bf16.py"), "aiter_bf16")


def load_configs(N, K, G):
    f = os.path.join(CONFIG_DIR, f"awq_gemm_N={N},K={K},device_name=BW200,dtype=w4a16,group_size={G}.json")
    if os.path.exists(f):
        with open(f) as fh:
            return {int(k): v for k, v in json.load(fh).items()}
    return None


def make_sym_quant_weights(N, K, G, device, seed=0):
    torch.manual_seed(seed)
    w_fp = torch.randn(K, N, dtype=torch.float32, device=device) * 0.1
    w_groups = w_fp.view(K // G, G, N)
    scale = (w_groups.abs().amax(dim=1) / 7.0).clamp(min=1e-6)
    w_int = (torch.round(w_groups / scale.unsqueeze(1)).to(torch.int32) + 8).clamp(0, 15)
    w_int_flat = w_int.reshape(K, N)
    qweight_gptq = torch.zeros(K, N // 8, dtype=torch.int32, device=device)
    for j in range(8):
        qweight_gptq |= (w_int_flat[:, j::8] & 0xF) << (4 * j)
    qzeros_gptq = torch.full((K // G, N // 8), -2004318072, dtype=torch.int32, device=device)
    aq, az = orig.awq_reorder_and_repack(qweight_gptq, qzeros_gptq)
    return aq.contiguous(), az.contiguous(), scale  # scale fp32, 调用方自己转 dtype


def bench(fn, args, warmup=15, iters=80):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device: {torch.cuda.get_device_name(0)}")

    N, K, G = 10752, 5376, 32
    configs = load_configs(N, K, G)
    aq, az, scale_fp32 = make_sym_quant_weights(N, K, G, device)
    scales_fp16 = scale_fp32.to(torch.float16)
    scales_bf16 = scale_fp32.to(torch.bfloat16)

    Ms = [1, 16, 128, 512, 1024, 2048]

    # 预生成输入
    xs = {}
    for M in Ms:
        x_fp32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
        xs[M] = {"fp16": x_fp32.to(torch.float16), "bf16": x_fp32.to(torch.bfloat16)}

    print(f"\n形状 N={N} K={K} G={G}  路径: splitk(sk=2)")
    print(f"\n{'M':>6} | {'①fp16->fp16':>12} | {'②bf16->bf16':>12} | {'③bf16->fp16->bf16':>16} | {'④fp16->fp16':>12} | 诊断")
    print("-" * 95)

    for M in Ms:
        x_fp16 = xs[M]["fp16"]
        x_bf16 = xs[M]["bf16"]

        # ① fp16 in -> orig kernel (fp16 out)
        def run1():
            return orig.gemm_a16w4(x_fp16, aq, scales_fp16, az, configs=configs)
        t1 = bench(run1, ())

        # ② bf16 in -> bf16 kernel (bf16 out)
        def run2():
            return bf16.gemm_a16w4(x_bf16, aq, scales_bf16, az, configs=configs)
        t2 = bench(run2, ())

        # ③ bf16 in -> 转 fp16 喂 kernel -> bf16 kernel (bf16 out, 但 kernel 内部全 fp16)
        #    用 bf16 改造版, 但 input 和 scales 都转 fp16, out_dtype=input.dtype=fp16,
        #    然后 .to(bf16) 模拟最终输出 bf16. 这样 kernel 内部纯 fp16, 只多一次外层 cast
        def run3():
            x = x_bf16.to(torch.float16)
            out = bf16.gemm_a16w4(x, aq, scales_fp16, az, configs=configs)  # fp16 out
            return out.to(torch.bfloat16)
        t3 = bench(run3, ())

        # ④ fp16 in -> bf16 kernel (fp16 out, out_dtype=fp16)
        def run4():
            return bf16.gemm_a16w4(x_fp16, aq, scales_fp16, az, configs=configs)
        t4 = bench(run4, ())

        # 诊断
        r2 = t2 / t1
        r3 = t3 / t1
        r4 = t4 / t1
        if r4 > 1.1:
            diag = "kernel编译路径慢(④也慢)"
        elif r3 < r2 - 0.05:
            diag = "input load慢(③比②快)"
        elif r2 > 1.1 and abs(r3 - r2) < 0.05:
            diag = "输出cast/store慢(③≈②都慢)"
        else:
            diag = "无明显差异"

        print(f"{M:>6} | {t1:>12.4f} | {t2:>12.4f} | {t3:>16.4f} | {t4:>12.4f} | {diag} (②/①={r2:.2f} ④/①={r4:.2f})")

    print("\n=== 诊断判据 ===")
    print("④/① > 1.1   : 慢在 kernel 编译路径本身 (triton 对 bf16 input 生成的代码差) -> 难优化")
    print("③ < ② 且 ③≈①: 慢在 bf16 input load -> 可考虑 input 保持 fp16")
    print("③≈② 都慢    : 慢在输出 cast/store (bf16 写出比 fp16 慢) -> 优化输出路径")


if __name__ == "__main__":
    main()
