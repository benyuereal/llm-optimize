#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证全 fp8 mixed kernel (Q/K/V 全 e5m2, scale=1.0) 的精度,
作为路径 A (Q-fp16+KV-fp8) 的回退备选评估。

关键发现 (已验证):
  1. mixed kernel (tile16x32) 完全忽略 descale 参数 (传 1.0/2.0/0.5 输出相同).
     - e5m2->bf16 是纯位转换 (e5m2x2_to_bf16x2), 不乘 descale.
     - 因此只能用 scale=1.0 直接量化, 不能用动态 scale (kernel 不会乘回).
  2. gfx936 mixed kernel 只支持 e5m2 (e4m3 会 VMFault).
     - e5m2: 2位尾数, 本征量化误差 ~4.5% rel.
     - 生产 vllm 用 e4m3 存 KV cache (本征误差 ~2.5%), 但 mixed kernel 不支持.
  3. fp8 路径输出 bf16, fp16 路径输出 fp16 (kernel 要求 out_dtype==in_dtype).
  4. rel_err 指标被近零分母放大 (输出 abs_mean~0.013, 很多元素接近0).
     更可靠指标: cos_sim (输出方向) 和 MAE (绝对误差).

对比基准: fp16 KV (全 fp16 splitkv) vs fp32 参考 = cos 1.00000, MAE 4e-6 (近乎完美).
"""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
import statistics as st
import flash_attn_2_cuda as fa

NUM_QH, NUM_KVH, HD = 32, 16, 256
SCALE = 1.0 / (HD ** 0.5)
BLK = 128
NS = 8
QL = 1


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def mae(a, b):
    return (a.float() - b.float()).abs().mean().item()


def rel(a, b):
    return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-6)).mean().item()


def fp32_ref(q, k, v, kl, sw):
    """逐 head 手算 fp32 参考 attention. q:[NS,QH,HD] k/v:[nblk,BLK,KVH,HD]."""
    kflat = k.reshape(NS, kl // BLK, BLK, NUM_KVH, HD).reshape(NS, kl, NUM_KVH, HD).float()
    vflat = v.reshape(NS, kl // BLK, BLK, NUM_KVH, HD).reshape(NS, kl, NUM_KVH, HD).float()
    out = torch.empty(NS, NUM_QH, HD, dtype=torch.float32, device='cuda')
    for s in range(NS):
        for h in range(NUM_QH):
            kh = h // 2  # QH/KVH = 2
            sc = q[s, h].float() @ kflat[s, :, kh].T * SCALE
            if sw >= 0:  # sliding window
                stt = max(0, kl - 1 - sw)
                sc = sc[stt:]; vv = vflat[s, stt:, kh]
            else:
                vv = vflat[s, :, kh]
            out[s, h] = F.softmax(sc, dim=-1) @ vv
    return out


def run_fp16(q, k, v, kl, sw):
    out = torch.empty(NS, NUM_QH, HD, dtype=torch.float16, device='cuda')
    csq = torch.arange(0, NS + 1, dtype=torch.int32, device='cuda')
    suk = torch.full((NS,), kl, dtype=torch.int32, device='cuda')
    nblk = NS * ((kl + BLK - 1) // BLK)
    bt = torch.arange(0, nblk, dtype=torch.int32, device='cuda').reshape(NS, nblk // NS)
    fa.hg_prefix_decode_varlen_fwd(
        q, k, v, out, csq, None, suk, None, bt,
        QL, kl, 0.0, SCALE, False, True, sw, 0, 0.0, False, 1,
        None, None, None, None, False)
    return out


def run_fp8(q, k, v, kl, sw):
    """全 fp8 e5m2, scale=1.0 (mixed kernel 忽略 descale, 只能 scale=1)."""
    q8 = q.to(torch.float8_e5m2); k8 = k.to(torch.float8_e5m2); v8 = v.to(torch.float8_e5m2)
    out = torch.empty(NS, NUM_QH, HD, dtype=torch.bfloat16, device='cuda')
    csq = torch.arange(0, NS + 1, dtype=torch.int32, device='cuda')
    suk = torch.full((NS,), kl, dtype=torch.int32, device='cuda')
    nblk = NS * ((kl + BLK - 1) // BLK)
    bt = torch.arange(0, nblk, dtype=torch.int32, device='cuda').reshape(NS, nblk // NS)
    s = torch.ones(1, dtype=torch.float32, device='cuda')
    fa.hg_prefix_decode_varlen_fwd(
        q8, k8, v8, out, csq, None, suk, None, bt,
        QL, kl, 0.0, SCALE, False, True, sw, 0, 0.0, False, 1,
        s, s, s, None, True)
    return out


def test(kl, sw, std, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    nblk = NS * ((kl + BLK - 1) // BLK)
    q = torch.randn(NS, NUM_QH, HD, dtype=torch.float16, device='cuda', generator=g) * std
    k = torch.randn(nblk, BLK, NUM_KVH, HD, dtype=torch.float16, device='cuda', generator=g) * std
    v = torch.randn(nblk, BLK, NUM_KVH, HD, dtype=torch.float16, device='cuda', generator=g) * std
    ref = fp32_ref(q, k, v, kl, sw)
    o1 = run_fp16(q, k, v, kl, sw)
    o3 = run_fp8(q, k, v, kl, sw)
    return cos(ref, o1), mae(ref, o1), cos(ref, o3), mae(ref, o3), ref.float().abs().mean().item()


def main():
    print("=" * 90)
    print("全 fp8 mixed kernel (e5m2, scale=1.0) 精度评估  vs  fp32 参考")
    print(f"配置: NS={NS}, QH={NUM_QH}, KVH={NUM_KVH}, HD={HD}, QL={QL}")
    print("基准: fp16 KV vs fp32 → cos 1.00000, MAE 4e-6 (近乎完美)")
    print("注: rel_err 被近零分母放大 (输出 abs_mean~0.013), 看 cos_sim 和 MAE 更可靠")
    print("=" * 90)

    cases = [
        (1024, 1023, 0.5, "生产 SW=1024 N(0,0.5)"),
        (1024, 1023, 0.3, "生产 SW=1024 N(0,0.3) [小量级]"),
        (1024, 1023, 1.0, "生产 SW=1024 N(0,1.0) [大量级]"),
        (8192, -1, 0.5, "长KV 无窗 kl=8192"),
        (4096, -1, 0.5, "长KV 无窗 kl=4096"),
    ]

    hdr = f"{'case':<32}{'fp16_cos':>10}{'fp16_MAE':>11}{'fp8_cos':>10}{'fp8_MAE':>11}{'out|mean':>10}"
    print(hdr)
    print("-" * len(hdr))
    for kl, sw, std, desc in cases:
        # 3 seeds 平均
        cs1, ms1, cs3, ms3, oms = [], [], [], [], []
        for seed in [42, 123, 7]:
            c1, m1, c3, m3, om = test(kl, sw, std, seed)
            cs1.append(c1); ms1.append(m1); cs3.append(c3); ms3.append(m3); oms.append(om)
        print(f"{desc:<32}{st.mean(cs1):>10.5f}{st.mean(ms1):>11.6f}"
              f"{st.mean(cs3):>10.5f}{st.mean(ms3):>11.6f}{st.mean(oms):>10.4f}")

    print("=" * 90)
    print("结论:")
    print("  - fp16 路径: cos 1.00000, MAE ~4e-6 → 近乎完美, 作为黄金参考")
    print("  - fp8 e5m2 路径: cos 0.9958-0.9986, MAE 0.0003-0.0037")
    print("  - cos_sim ~0.998 对 KV cache 偏低 (行业 e4m3 通常 >0.9999)")
    print("  - 误差来源: e5m2 仅 2 位尾数 (本征 rel 4.5%), 且 kernel 不支持 descale 缩放")
    print("  - 大量级 N(0,1) 时 cos 降到 0.996 (e5m2 截断更严重)")
    print()
    print("评估:")
    print("  ✓ 输出方向一致 (cos>0.995), 语义保持, 不至于生成乱码")
    print("  ✗ 精度损失明显 (cos 0.998 vs fp16 1.0), 长序列/多层累积可能影响质量")
    print("  ✗ 无法用 descale 改善 (kernel 硬编码忽略)")
    print("  → 作为路径 A 回退: 勉强可用但非理想. 路径 A (Q-fp16+KV-fp8, e4m3 若支持) 会好得多")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
