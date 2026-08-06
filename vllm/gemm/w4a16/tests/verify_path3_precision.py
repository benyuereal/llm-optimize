#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证方案③ (input 转 fp16 喂 kernel, 输出 bf16) 的精度
对比: ① 纯fp16  ② 纯bf16  ③ bf16-input转fp16-输出bf16  vs fp32参考
看 ③ 的精度是否可接受 (相对 ② 纯bf16 损失多少)
"""
import os
import sys
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
    w_ref = ((w_int.float() - 8.0) * scale.unsqueeze(1)).reshape(K, N)
    w_int_flat = w_int.reshape(K, N)
    qweight_gptq = torch.zeros(K, N // 8, dtype=torch.int32, device=device)
    for j in range(8):
        qweight_gptq |= (w_int_flat[:, j::8] & 0xF) << (4 * j)
    qzeros_gptq = torch.full((K // G, N // 8), -2004318072, dtype=torch.int32, device=device)
    aq, az = orig.awq_reorder_and_repack(qweight_gptq, qzeros_gptq)
    return aq.contiguous(), az.contiguous(), scale, w_ref


def rel_err(a, b):
    return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-6)).mean().item()


def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    N, K, G = 10752, 5376, 32
    configs = load_configs(N, K, G)
    aq, az, scale_fp32, w_ref = make_sym_quant_weights(N, K, G, device)
    scales_fp16 = scale_fp32.to(torch.float16)
    scales_bf16 = scale_fp32.to(torch.bfloat16)

    print(f"形状 N={N} K={K} G={G}  精度对比 (vs fp32 参考)")
    print(f"\n{'M':>6} | {'①纯fp16':>10} | {'②纯bf16':>10} | {'③bf16in转fp16':>14} | {'③-②差值':>10}")
    print("-" * 65)

    for M in [1, 128, 1024, 2048]:
        x_fp32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
        ref = x_fp32 @ w_ref
        x_fp16 = x_fp32.to(torch.float16)
        x_bf16 = x_fp32.to(torch.bfloat16)

        # ① 纯 fp16 (原始 kernel)
        o1 = orig.gemm_a16w4(x_fp16, aq, scales_fp16, az, configs=configs)
        # ② 纯 bf16 (改造版, input/scales bf16)
        o2 = bf16.gemm_a16w4(x_bf16, aq, scales_bf16, az, configs=configs)
        # ③ bf16 input 转 fp16 喂 kernel, 输出 bf16 (scales 也 fp16)
        o3 = bf16.gemm_a16w4(x_bf16.to(torch.float16), aq, scales_fp16, az, configs=configs).to(torch.bfloat16)

        e1, e2, e3 = rel_err(o1, ref), rel_err(o2, ref), rel_err(o3, ref)
        print(f"{M:>6} | {e1:>10.6f} | {e2:>10.6f} | {e3:>14.6f} | {e3-e2:>+10.6f}")

    print("\n=== 判读 ===")
    print("③-② 差值 ~0   : 方案③精度等同纯bf16, 可用 (且性能=fp16基准)")
    print("③-② 明显为正  : 方案③比纯bf16精度差, input fp16 化引入了精度损失")


if __name__ == "__main__":
    main()
