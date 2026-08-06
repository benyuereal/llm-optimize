#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比测试: 原始 aiter w4a16 kernel (fp16 输出) vs bf16 改造版 (out_dtype 跟随输入)
用卡 1, 实际 gemma layer 形状 + 现成 BW200 config (覆盖 dp/splitk/splitk+reduce 三条路径)

三方面对比:
  1. 精度: 原始版(fp16 中转) vs 改造版(直接 out_dtype) vs fp32 参考
  2. 输出 dtype: 改造版应输出 input.dtype (bf16/fp16), 原始版恒 fp16
  3. 性能: 改造版 vs 原始版 延时 (多条路径)
"""
import os
import sys
import time
import json
import importlib.util

os.environ["HIP_VISIBLE_DEVICES"] = "1"  # 用卡 1

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


def make_sym_quant_weights(N, K, G, device, dtype, seed=0):
    """对称量化 (uint4b8) 权重 -> GPTQ pack -> awq_reorder_and_repack -> kernel 输入格式"""
    torch.manual_seed(seed)
    w_fp = torch.randn(K, N, dtype=torch.float32, device=device) * 0.1
    w_groups = w_fp.view(K // G, G, N)
    scale = (w_groups.abs().amax(dim=1) / 7.0).clamp(min=1e-6)  # [K//G, N]
    w_int = (torch.round(w_groups / scale.unsqueeze(1)).to(torch.int32) + 8).clamp(0, 15)
    w_ref = ((w_int.float() - 8.0) * scale.unsqueeze(1)).reshape(K, N)  # fp32 参考
    w_int_flat = w_int.reshape(K, N)
    qweight_gptq = torch.zeros(K, N // 8, dtype=torch.int32, device=device)
    for j in range(8):
        qweight_gptq |= (w_int_flat[:, j::8] & 0xF) << (4 * j)
    qzeros_gptq = torch.full((K // G, N // 8), -2004318072, dtype=torch.int32, device=device)  # 0x88888888
    aq, az = orig.awq_reorder_and_repack(qweight_gptq, qzeros_gptq)
    return aq.contiguous(), scale.to(dtype).contiguous(), az.contiguous(), w_ref


def run_kernel(mod, input, aq, scales, az, N, K, G, configs):
    return mod.gemm_a16w4(input, aq, scales, az, configs=configs)


def path_of(configs, M):
    if not configs:
        return "default(dp)"
    c = configs[min(configs.keys(), key=lambda v: abs(v - M))]
    sk = c["SPLITK"]
    r = "+reduce" if c.get("USE_REDUCE_KERNEL") else ""
    return f"{'splitk' if sk > 1 else 'dp'}{r}(sk={sk})"


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
    return ts[len(ts) // 2], ts[int(len(ts) * 0.9)]


def test_shape(N, K, G, device, Ms):
    configs = load_configs(N, K, G)
    print(f"\n{'='*70}")
    print(f"形状 N={N} K={K} G={G}  config: {'有(BW200)' if configs else '无(default dp)'}")
    print(f"{'='*70}")
    aq, scales_fp16, az, w_ref = make_sym_quant_weights(N, K, G, device, torch.float16)
    scales_bf16 = scales_fp16.to(torch.bfloat16)

    # 精度表
    print(f"\n{'M':>6} | {'路径':>16} | {'orig.dtype':>10} | {'bf16.dtype':>10} | "
          f"{'orig-vs-ref':>11} | {'bf16-vs-ref':>11} | {'bf16vsorig':>10}")
    print("-" * 95)
    for M in Ms:
        x_fp32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
        ref = x_fp32 @ w_ref
        x_fp16 = x_fp32.to(torch.float16)
        x_bf16 = x_fp32.to(torch.bfloat16)
        out_orig = run_kernel(orig, x_fp16, aq, scales_fp16, az, N, K, G, configs)
        out_bf16 = run_kernel(bf16, x_bf16, aq, scales_bf16, az, N, K, G, configs)
        out_bf16_fp16 = run_kernel(bf16, x_fp16, aq, scales_fp16, az, N, K, G, configs)
        def rel(a, b):
            return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-6)).mean().item()
        print(f"{M:>6} | {path_of(configs, M):>16} | {str(out_orig.dtype):>10} | {str(out_bf16.dtype):>10} | "
              f"{rel(out_orig, ref):>11.6f} | {rel(out_bf16, ref):>11.6f} | {rel(out_bf16_fp16, out_orig):>10.6f}")

    # 性能表
    print(f"\n--- 性能 (ms, 中位数/p90) ---")
    print(f"{'M':>6} | {'路径':>16} | {'orig(fp16)':>16} | {'bf16(bf16)':>16} | {'提速':>7}")
    print("-" * 75)
    for M in Ms:
        x_fp16 = (torch.randn(M, K, device=device) * 0.5).to(torch.float16)
        x_bf16 = x_fp16.to(torch.bfloat16)
        a_o = (x_fp16, aq, scales_fp16, az, N, K, G, configs)
        a_b = (x_bf16, aq, scales_bf16, az, N, K, G, configs)
        med_o, p90_o = bench(lambda *a: run_kernel(orig, *a), a_o)
        med_b, p90_b = bench(lambda *a: run_kernel(bf16, *a), a_b)
        sp = med_o / med_b if med_b > 0 else 0
        print(f"{M:>6} | {path_of(configs, M):>16} | {med_o:>7.4f}/{p90_o:<7.4f} | {med_b:>7.4f}/{p90_b:<7.4f} | {sp:>6.3f}x")


def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device: {torch.cuda.get_device_name(0)}")

    # 形状1: 实际 gemma layer (有 config, 覆盖 splitk 路径)
    test_shape(10752, 5376, 32, device, [1, 4, 16, 64, 128, 256, 512, 1024, 2048])

    # 形状2: 另一个实际 layer (有 config)
    test_shape(5376, 5376, 32, device, [1, 16, 128, 512, 2048])


if __name__ == "__main__":
    main()
