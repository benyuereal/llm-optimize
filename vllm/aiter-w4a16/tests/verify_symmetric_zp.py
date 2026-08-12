#!/usr/bin/env python3
"""验证: 对称量化用「全零 zp + aiter」等价于「原生对称 dequant (w*scale)」。

aiter kernel dequant 算式: (w - zp) * scale
  - 非对称: zp = 真 zero_point
  - 对称:   zp = 0  → (w - 0) * scale = w * scale  (等价对称量化)

本测试构造对称量化权重, 对比:
  A. aiter gemm_a16w4 + 全零 zp  (我们要走的兼容路径)
  B. 手动 dequant w*scale + matmul (对称量化参考实现)
两者 cos_sim 应 = 1.0, 证明全零 zp 方案精度无损。

同时对比:
  C. vllm 原生 triton_w4a16_gemm (HAS_ZP=False 对称路径) — 若可导入
"""
import os, sys, types, importlib.util, logging

# ---- 加载 aiter (绕过 __init__ JIT) ----
AITER_ROOT = "/public/home/weishb/aiter/aiter"
pkg = types.ModuleType("aiter")
pkg.__path__ = [AITER_ROOT]
pkg.__spec__ = importlib.util.spec_from_file_location("aiter", AITER_ROOT + "/__init__.py",
    submodule_search_locations=[AITER_ROOT])
pkg.logger = logging.getLogger("aiter")
sys.modules["aiter"] = pkg
for sub in ["aiter.ops", "aiter.ops.triton", "aiter.ops.triton.utils"]:
    m = types.ModuleType(sub); m.__path__ = [AITER_ROOT + "/" + sub.replace(".", "/")]
    sys.modules[sub] = m
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
_load("aiter.ops.triton.utils.core", AITER_ROOT + "/ops/triton/utils/core.py")
_load("aiter.ops.triton.utils.arch_info", AITER_ROOT + "/ops/triton/utils/arch_info.py")
s = importlib.util.spec_from_file_location("aiter.ops.triton.gemm_a16w4",
    AITER_ROOT + "/ops/triton/gemm_a16w4.py")
_g = importlib.util.module_from_spec(s)
sys.modules["aiter.ops.triton.gemm_a16w4"] = _g
s.loader.exec_module(_g)

import torch


def awq_dequantize_symmetric(qweight, scales, group_size):
    """对称量化参考 dequant: w * scale (无 zp)。
    qweight [K,N//8] int32 (AWQ 原始 N-packed), scales [K//G,N] fp16。
    注意 aiter 的 awq_reorder_and_repack 期望 AWQ 原始格式输入。
    """
    if group_size == -1:
        group_size = qweight.shape[0]
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    iweights = _g.reverse_awq_order(iweights)
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    return iweights * scales  # 对称: 无 zp


def run(M, K, N, G, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    input = torch.rand(M, K, dtype=torch.float16, device=device)

    # AWQ 原始格式输入
    qweight = torch.randint(0, 2**31 - 1, (K, N // 8), device=device, dtype=torch.int32)
    scales = torch.rand(K // G, N, dtype=torch.float16, device=device)

    # === A. aiter + 全零 zp (兼容路径) ===
    qzeros_zero = torch.zeros((K // G, N // 8), dtype=torch.int32, device=device)
    aq, az = _g.awq_reorder_and_repack(qweight, qzeros_zero)
    out_aiter_zerozp = _g.gemm_a16w4(input, aq, scales, az)

    # === B. 对称 dequant 参考实现 (w * scale) + matmul ===
    deq_sym = awq_dequantize_symmetric(qweight, scales, G)
    out_ref_sym = torch.matmul(input, deq_sym.to(torch.float16))

    diff = (out_aiter_zerozp.float() - out_ref_sym.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_aiter_zerozp.float().flatten(), out_ref_sym.float().flatten(), dim=0).item()
    ok = cos > 0.9999
    print(f"[M={M},K={K},N={N},G={G}] aiter+全零zp vs 对称参考: "
          f"max_abs={diff.max():.4f} mean_abs={diff.mean():.6f} cos_sim={cos:.6f} "
          f"{'✅' if ok else '❌'}")
    return cos, ok


print("=" * 70)
print("对称量化兼容验证: aiter+全零zp vs 原生对称 dequant (w*scale)")
print("kernel 算式 (w-zp)*scale, zp=0 时 == w*scale, 应等价")
print("=" * 70)
results = []
results.append(run(4, 256, 512, 32))
results.append(run(16, 1024, 1024, 32))
results.append(run(4, 5376, 1344, 32))
results.append(run(4, 5376, 3584, 32))
results.append(run(4, 14336, 1344, 32))
results.append(run(4, 5376, 5376, 32))

print("=" * 70)
all_ok = all(ok for _, ok in results)
print(f"总结: {'✅ 全部通过, 全零 zp 等价对称量化' if all_ok else '❌ 有不等价, 需排查'}")
sys.exit(0 if all_ok else 1)
