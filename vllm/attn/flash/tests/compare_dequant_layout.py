"""对比 e5m2x2_to_bf16x2 和 e4m3x2_to_bf16x2 的输出布局.

e5m2: bit-trick, 把2字节e5m2当fp16高字节, v_cvt_f32_f16 转 fp32, 取 bf16.
  return (value0 >> 16) | (value1 & 0xffff0000)
e4m3: 软件 dequant, byte0->b0, byte1->b1, return (b0) | (b1<<16)

关键: 两者对同一个 uint16 输入, "低 bf16" 和 "高 bf16" 对应的字节是否一致?
如果不一致, 就是 e4m3 pack 布局反了 -> cos 0.88.
"""
import torch
import numpy as np
import struct

def bf16_to_fp32(bf16_bits):
    """bf16 (16位) -> fp32: 左移16位补0."""
    return struct.unpack('<f', struct.pack('<I', bf16_bits << 16))[0]

def e5m2x2_to_bf16x2_py(input16):
    """精确复现 e5m2x2_to_bf16x2 (pv_gemm_utils.h:17).
    input: uint16. 输出 uint32 (两个 bf16 打包).
    """
    # half2_bits = (low_byte << 8) | (high_byte << 16)
    low = input16 & 0xff
    high = (input16 >> 8) & 0xff
    half2_bits = (low << 8) | (high << 16)  # uint32
    # v_cvt_f32_f16 取低16位 fp16 -> value0
    # v_cvt_f32_f16 sdwa 取高16位 fp16 -> value1
    h0 = half2_bits & 0xffff
    h1 = (half2_bits >> 16) & 0xffff
    value0 = np.float16(np.frombuffer(np.uint16(h0).tobytes(), dtype=np.float16)[0]).astype(np.float32)
    value1 = np.float16(np.frombuffer(np.uint16(h1).tobytes(), dtype=np.float16)[0]).astype(np.float32)
    # bf16 = fp32 的高16位
    v0_bits = struct.unpack('<I', struct.pack('<f', value0))[0]
    v1_bits = struct.unpack('<I', struct.pack('<f', value1))[0]
    return (v0_bits >> 16) | (v1_bits & 0xffff0000)

def e4m3x2_to_bf16x2_py(input16):
    """精确复现 e4m3x2_to_bf16x2 (pv_gemm_utils.h:28).
    byte0=低字节, byte1=高字节. b0,bf16(f0); b1,bf16(f1).
    return (b0) | (b1 << 16)
    """
    byte0 = input16 & 0xff
    byte1 = (input16 >> 8) & 0xff
    # e4m3 -> fp32 (用 torch)
    f0 = torch.tensor([byte0], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).item()
    f1 = torch.tensor([byte1], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).item()
    # fp32 -> bf16 (round to nearest even, 取高16位)
    b0 = np.float32(f0).astype(np.dtype('bfloat16')) if hasattr(np, 'float8') else bf16_round(f0)
    b1 = bf16_round(f1)
    return b0 | (b1 << 16)

def bf16_round(f):
    """fp32 -> bf16 bits (round to nearest even)."""
    u = struct.unpack('<I', struct.pack('<f', np.float32(f)))[0]
    # bf16 = 取高16位, 加 round-half-even
    lsb = (u >> 16) & 1
    rounding_bias = 0x7fff + lsb
    return ((u + rounding_bias) >> 16) & 0xffff

def main():
    print("=== e5m2 vs e4m3 dequant 输出布局对比 ===")
    print("(看: 同一 uint16 输入, 低 bf16 和高 bf16 各对应哪个字节)\n")
    # 用几个有代表性的输入 (低字节和高字节不同值)
    # 注意: e5m2 和 e4m3 对同一字节的解释不同, 所以不能直接比数值,
    # 要比"低bf16对应低字节, 高bf16对应高字节"这个布局是否一致.
    test_inputs = [0x0102, 0x0a0b, 0x7f80, 0x40c0, 0xff7f]
    print(f"{'input':>8} | {'e5m2 低bf16':>12} {'e5m2 高bf16':>12} | {'e4m3 低bf16':>12} {'e4m3 高bf16':>12}")
    print("-" * 75)
    for inp in test_inputs:
        e5 = e5m2x2_to_bf16x2_py(inp)
        e4 = e4m3x2_to_bf16x2_py(inp)
        e5_lo = bf16_to_fp32(e5 & 0xffff)
        e5_hi = bf16_to_fp32((e5 >> 16) & 0xffff)
        e4_lo = bf16_to_fp32(e4 & 0xffff)
        e4_hi = bf16_to_fp32((e4 >> 16) & 0xffff)
        print(f"  0x{inp:04x} | {e5_lo:>12.6g} {e5_hi:>12.6g} | {e4_lo:>12.6g} {e4_hi:>12.6g}")

    print("\n=== 关键: 低 bf16 应对应低字节, 高 bf16 应对应高字节 ===")
    print("e5m2: 低字节=input&0xff, 高字节=(input>>8)&0xff")
    print("e4m3: byte0=低字节, byte1=高字节")
    # 验证: 输入 0x0102 (低字节=0x02, 高字节=0x01)
    # e5m2 低bf16 应该由 0x02(e5m2) 决定, 高bf16 由 0x01(e5m2) 决定
    # e4m3 低bf16 应该由 0x02(e4m3) 决定, 高bf16 由 0x01(e4m3) 决定
    inp = 0x0102
    lo_byte_val_e5m2 = torch.tensor([0x02], dtype=torch.uint8).view(torch.float8_e5m2).to(torch.float32).item()
    hi_byte_val_e5m2 = torch.tensor([0x01], dtype=torch.uint8).view(torch.float8_e5m2).to(torch.float32).item()
    lo_byte_val_e4m3 = torch.tensor([0x02], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).item()
    hi_byte_val_e4m3 = torch.tensor([0x01], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).item()
    print(f"\n输入 0x0102: 低字节=0x02, 高字节=0x01")
    print(f"  e5m2: 低字节0x02={lo_byte_val_e5m2}, 高字节0x01={hi_byte_val_e5m2}")
    print(f"  e4m3: 低字节0x02={lo_byte_val_e4m3}, 高字节0x01={hi_byte_val_e4m3}")
    e5 = e5m2x2_to_bf16x2_py(inp)
    e4 = e4m3x2_to_bf16x2_py(inp)
    print(f"  e5m2 输出: 低bf16={bf16_to_fp32(e5&0xffff):.6g} (应={lo_byte_val_e5m2:.6g}?), "
          f"高bf16={bf16_to_fp32((e5>>16)&0xffff):.6g} (应={hi_byte_val_e5m2:.6g}?)")
    print(f"  e4m3 输出: 低bf16={bf16_to_fp32(e4&0xffff):.6g} (应={lo_byte_val_e4m3:.6g}?), "
          f"高bf16={bf16_to_fp32((e4>>16)&0xffff):.6g} (应={hi_byte_val_e4m3:.6g}?)")

if __name__ == "__main__":
    main()
