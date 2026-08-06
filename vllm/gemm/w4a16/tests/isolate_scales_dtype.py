#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
确认: input fp16 是性能关键, 还是 scales dtype 也影响?
四组对比 (都用改造版 kernel):
  A: input fp16 + scales fp16  (全 fp16, =基准)
  B: input fp16 + scales bf16  (input fp16, scales bf16)
  C: input bf16 + scales fp16  (input bf16, scales fp16)
  D: input bf16 + scales bf16  (全 bf16, =路B现状)
看 B 是否还快 (scales bf16 不影响性能) -> 决定方案③能否保留 bf16 scales
"""
import os
import sys
import time
import json
import importlib.util

os.environ["HIP_VISIBLE_DEVICES"] = "1"

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
    return aq.contiguous(), az.contiguous(), scale


def bench(fn, warmup=15, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    N, K, G = 10752, 5376, 32
    configs = load_configs(N, K, G)
    aq, az, scale_fp32 = make_sym_quant_weights(N, K, G, device)
    scales_fp16 = scale_fp32.to(torch.float16)
    scales_bf16 = scale_fp32.to(torch.bfloat16)

    print(f"形状 N={N} K={K} G={G}  scales dtype 影响测试")
    print(f"\n{'M':>6} | {'A:in fp16+sc fp16':>16} | {'B:in fp16+sc bf16':>16} | {'C:in bf16+sc fp16':>16} | {'D:in bf16+sc bf16':>16}")
    print("-" * 90)

    for M in [1, 128, 1024, 2048]:
        x_fp32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
        x_fp16 = x_fp32.to(torch.float16)
        x_bf16 = x_fp32.to(torch.bfloat16)

        tA = bench(lambda: bf16.gemm_a16w4(x_fp16, aq, scales_fp16, az, configs=configs))
        tB = bench(lambda: bf16.gemm_a16w4(x_fp16, aq, scales_bf16, az, configs=configs))
        tC = bench(lambda: bf16.gemm_a16w4(x_bf16, aq, scales_fp16, az, configs=configs))
        tD = bench(lambda: bf16.gemm_a16w4(x_bf16, aq, scales_bf16, az, configs=configs))

        print(f"{M:>6} | {tA:>16.4f} | {tB:>16.4f} | {tC:>16.4f} | {tD:>16.4f}")

    print("\n=== 判读 ===")
    print("B ≈ A : scales bf16 不影响性能 -> 方案③可保留 bf16 scales (权重精度保 bf16)")
    print("B >> A : scales bf16 也慢 -> scales 必须 fp16, 方案③权重也变 fp16 (=路A)")


if __name__ == "__main__":
    main()
