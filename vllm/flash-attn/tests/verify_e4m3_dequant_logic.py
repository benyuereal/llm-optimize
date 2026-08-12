"""Python 复现 __e4m32float 逻辑, 对比 torch 的 e4m3->float32 转换.

目的: 不编译, 快速验证 e4m3 dequant 的位操作逻辑是否正确.
如果 Python 复现和 torch.to(float32) 一致, 说明 __e4m32float 逻辑对,
那 cos 0.88 的根因在别处 (字节序/加载宽度/pack 布局).
如果不一致, 就找到了 __e4m32float 的 bug.
"""
import torch
import numpy as np

def py_e4m32float(src_byte):
    """精确复现 intrinsic.h 的 __e4m32float. src_byte: 0-255."""
    src = int(src_byte) & 0xff
    sign = src & 0x80
    exp = (src & 0x78) >> 3   # e4m3_mant_bits=3
    mant = src & 0x7
    # denorm or zero (exp==0)
    if exp == 0x0:
        result = 0.0078125 * ((mant & 0x4) >> 2) + 0.00390625 * ((mant & 0x2) >> 1) + 0.001953125 * (mant & 0x1)
        result = -result if sign > 0 else result
        return np.float32(result)
    else:
        e4m3_bias = 7
        fp32_bias = 127
        fp32_mant_bits = 23
        e4m3_bits = 8
        # NaN (exp==0xf and mant==0x7) -> 0x7fffffff
        if exp == 0xf and mant == 0x7:
            tmp = 0x7fffffff
        else:
            tmp = ((sign << (32 - e4m3_bits)) +
                   ((exp - e4m3_bias + fp32_bias) << fp32_mant_bits) +
                   (mant << (fp32_mant_bits - 3)))
        return np.float32(np.frombuffer(np.uint32(tmp).tobytes(), dtype=np.float32)[0])

def main():
    # 测所有 256 个 e4m3 字节
    bytes_all = np.arange(256, dtype=np.uint8)
    # torch 参考: e4m3 -> float32
    t = torch.tensor(bytes_all, dtype=torch.uint8).view(torch.float8_e4m3fn)
    torch_ref = t.to(torch.float32).numpy()

    # python 复现
    py_res = np.array([py_e4m32float(b) for b in bytes_all], dtype=np.float32)

    print("=== __e4m32float Python 复现 vs torch.to(float32) ===")
    # 比较 (NaN/inf 特殊处理)
    finite = np.isfinite(torch_ref) & np.isfinite(py_res)
    match = np.sum(finite & (torch_ref == py_res))
    close = np.sum(finite & np.isclose(torch_ref, py_res, atol=0, rtol=0))
    print(f"  完全相等的字节: {match}/256")
    print(f"  有限值且相等: {close}/256 (finite={np.sum(finite)})")

    # 找不一致的
    diff = finite & (torch_ref != py_res)
    if np.any(diff):
        print("\n  不一致的字节 (前20个):")
        idx = np.where(diff)[0][:20]
        for i in idx:
            print(f"    byte=0x{i:02x} ({i:3d}): torch={torch_ref[i]:.8g}  py={py_res[i]:.8g}  "
                  f"sign={i&0x80:#x} exp={(i&0x78)>>3} mant={i&0x7}")
    else:
        print("  ✓ 所有有限值完全一致 — __e4m32float 逻辑正确")

    # 看 NaN/inf 字节
    print(f"\n  torch NaN/inf 字节: {np.where(~np.isfinite(torch_ref))[0]}")
    print(f"  py    NaN/inf 字节: {np.where(~np.isfinite(py_res))[0]}")

    # 单独看 e4m3 的边界值
    print("\n=== 关键字节检查 ===")
    for b in [0x00, 0x40, 0x7f, 0x80, 0xff, 0x7b, 0x7c, 0x7d, 0x7e]:
        print(f"  0x{b:02x}: torch={torch_ref[b]:.8g}  py={py_res[b]:.8g}")

if __name__ == "__main__":
    main()
