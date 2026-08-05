# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import json
import logging
import functools
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
import aiter.ops.triton.utils.arch_info as arch_info
from aiter import logger

import torch
import triton
import triton.language as tl

from triton.language.extra.hip import libdevice

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

def reverse_awq_order(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the AWQ order of the given tensor.

    Args:
        tensor: Input tensor to reorder

    Returns:
        Reordered tensor with bits masked to 4 bits
    """
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        tensor.shape[-1],
        dtype=torch.int32,
        device=tensor.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    tensor = tensor[:, reverse_order_tensor] & 0xF
    return tensor

def awq_reorder_and_repack(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder and pack weights and zeros using AWQ order.

    This function unpacks the 4-bit quantized weights and zeros from int32,
    applies reverse_awq_order to reorder them, and then packs them.
    For weight, repack to [N, K//2]
    For zeros, repack to [K//G, N//2]
    Args:
        qweight: Quantized weight tensor of shape [K, N // 8] with dtype int32
        qzeros: Quantized zero points tensor of shape [K // G, N // 8] with dtype int32

    Returns:
        Tuple of (reordered_qweight, reordered_qzeros) both with dtype int8
    """
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    G = K // qzeros.shape[0]

    # Unpack weights: [K, N//8] -> [K, N//8, 8] -> [K, N]
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    iweights = iweights.view(K, -1)

    # Unpack zeros: [K//G, N//8] -> [K//G, N//8, 8] -> [K//G, N]
    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    zeros = zeros.view(K//G, -1)

    # Apply reverse AWQ order to both tensors
    iweights = reverse_awq_order(iweights)
    zeros = reverse_awq_order(zeros)

    # Mask to 4 bits
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    # Repack weight to int32 and pack along the K direction
    # [K, N] -> [N, K]
    iweights = iweights.transpose(1, 0).contiguous()
    # Reshape to [N, K//2, 2] for weights
    iweights_packed = iweights.view(N, -1, 2)

    # Repack zeros to int8 and pack along the N direction
    # Reshape to [K//G, N//2, 2] for zeros
    zeros_packed = zeros.view(K//G, -1, 2)

    # Pack 2 int4 values into int8 using bit shifts
    # Direct packing: pack in the order they appear after reordering
    packed_weights = torch.zeros([N, K//2], dtype=torch.int8, device=qweight.device)
    packed_zeros = torch.zeros([K//G, N//2], dtype=torch.int8, device=zeros.device)

    for i in range(2):
        packed_weights |= (iweights_packed[:, :, i].to(torch.int8) << (i * bits))
        packed_zeros |= (zeros_packed[:, :, i].to(torch.int8) << (i * bits))

    return packed_weights, packed_zeros

'''
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK
        }, num_warps=num_warps, num_stages=num_stages)
        for BN in [16, 32, 64, 128, 256]
        for BK in [16, 32, 64, 128, 256]
        for num_warps in [1, 2, 4, 8, 16] for num_stages in [1, 2]
    ],
    key=["K", "N"],
    perf_debug=True,
)
'''
@triton.heuristics(values={
    "NUM_GROUPS": lambda args: triton.cdiv(args["BLOCK_SIZE_K"], args["group_size"]),
    "BLOCK_SIZE_K2": lambda args: args["BLOCK_SIZE_K"] // 2
})
@triton.jit
def awq_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per group
        zeros_ptr,  # zeros, per group
        result_ptr,  # Output matrix
        N,
        N2,
        K,
        K2,
        group_size: tl.constexpr,  # Should always be one of the supported group sizes
        NUM_GROUPS: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_K2: tl.constexpr):

    # Setup the pids.
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    tl.assume(N2 > 0)
    tl.assume(K2 > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(BLOCK_SIZE_K2 > 0)
    tl.assume(group_size > 0)

    # Compute offsets and masks for qweight_ptr.
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_n = tl.max_contiguous(tl.multiple_of(offsets_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offsets_k = pid_k * BLOCK_SIZE_K2 + tl.arange(0, BLOCK_SIZE_K2)
    offsets_k = tl.max_contiguous(tl.multiple_of(offsets_k, BLOCK_SIZE_K2), BLOCK_SIZE_K2)
    offsets = K2 * offsets_n[:, None] + offsets_k[None, :]

    masks_n = offsets_n < N
    masks_k = offsets_k < K2

    masks = masks_n[:, None] & masks_k[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    result_offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    result_offsets = (N * result_offsets_k[:, None] + result_offsets_n[None, :]) # [K, N]

    result_masks_n = result_offsets_n < N
    result_masks_k = result_offsets_k < K
    result_masks = result_masks_k[:, None] & result_masks_n[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0) #[BLOCK_SIZE_N, BLOCK_SIZE_K//2]
    iweights = tl.interleave(iweights, iweights) # [BLOCK_SIZE_N, BLOCK_SIZE_K]

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = tl.arange(0, 2) * 4
    bshifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_N * BLOCK_SIZE_K2, 2))
    bshifts = tl.reshape(bshifts, (BLOCK_SIZE_N, BLOCK_SIZE_K))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> bshifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_k = pid_k * BLOCK_SIZE_K // group_size + tl.arange(0, NUM_GROUPS)
    zero_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    zero_offsets_n2 = zero_offsets_n // 2
    zero_offsets = N2 * zero_offsets_k[:, None] + zero_offsets_n2[None, :]

    zero_masks_k = zero_offsets_k < K//group_size
    zero_masks_n = zero_offsets_n < N
    zero_masks = zero_masks_k[:, None] & zero_masks_n[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0) # [NUM_GROUPS, BLOCK_SIZE_N]

    # Compute scale offsets and masks.
    scale_offsets_k = pid_k * BLOCK_SIZE_K // group_size + tl.arange(0, NUM_GROUPS)
    scale_offsets_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    scale_offsets = N * scale_offsets_k[:, None] + scale_offsets_n[None, :]
    scale_masks_k = scale_offsets_k < K//group_size
    scale_masks_n = scale_offsets_n < N
    scale_masks = scale_masks_k[:, None] & scale_masks_n[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0) # [NUM_GROUPS, BLOCK_SIZE_N]

    if NUM_GROUPS == 1:
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N)) # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N)) # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    else:
        zeros = tl.broadcast_to(zeros[:, None, :], (NUM_GROUPS, group_size, BLOCK_SIZE_N))
        scales = tl.broadcast_to(scales[:, None, :], (NUM_GROUPS, group_size, BLOCK_SIZE_N))
        zeros = tl.reshape(zeros, [BLOCK_SIZE_K, BLOCK_SIZE_N])
        scales = tl.reshape(scales, [BLOCK_SIZE_K, BLOCK_SIZE_N])

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zshifts = (zero_offsets_n[None, :] % 2) * 4 # [1, BLOCK_SIZE_N]
    zeros = (zeros >> zshifts) & 0xF # [BLOCK_SIZE_K, BLOCK_SIZE_N]

    # Dequantize.
    iweights = (iweights.T - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)

@triton.jit
def awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, k_idx, iter_begin, iter_end,
                          M, N, N2, K, K2, not_reduce, GROUP_SIZE: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          NUM_GROUPS: tl.constexpr, USE_REDUCE_KERNEL: tl.constexpr = 0):

    if not USE_REDUCE_KERNEL:
        k_idx = 0

    tl.assume(tile_idx >= 0)
    tl.assume(k_idx >= 0)
    tl.assume(iter_begin >= 0)
    tl.assume(iter_end >= 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    tl.assume(K2 > 0)
    tl.assume(N2 > 0)

    num_tile_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tile_n = tl.cdiv(N, BLOCK_SIZE_N)

    tile_idx_m = tile_idx // num_tile_n
    tile_idx_n = tile_idx % num_tile_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    #reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)

    # Create the necessary shifts to use to unpack.
    #shifts = reverse_awq_order_tensor * 4
    shifts = tl.arange(0, 2) * 4

    #zshifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 2), 2))
    #zshifts = tl.reshape(zshifts, (BLOCK_SIZE_K, BLOCK_SIZE_N)).T

    #bshifts = tl.broadcast_to(shifts[:, None], (8, (BLOCK_SIZE_K // 8) * BLOCK_SIZE_N))
    bshifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_N * (BLOCK_SIZE_K // 2), 2))
    bshifts = tl.reshape(bshifts, (BLOCK_SIZE_N, BLOCK_SIZE_K))

    # Offsets and masks.
    offsets_am = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    #offsets_zn = tile_idx_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    offsets_zn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_zn = offsets_zn // 2
    #offsets_zn = tl.max_contiguous(tl.multiple_of(offsets_zn, BLOCK_SIZE_N // 2), BLOCK_SIZE_N // 2)
    masks_zn = offsets_zn < N2

    offsets_bn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #offsets_bn = tl.max_contiguous(tl.multiple_of(offsets_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    masks_bn = offsets_bn < N

    offsets_sn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #offsets_sn = tl.max_contiguous(tl.multiple_of(offsets_sn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_ak = iter_begin * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    #offsets_ak = tl.max_contiguous(tl.multiple_of(offsets_ak, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_ak[None, :]

    offsets_bk = iter_begin * (BLOCK_SIZE_K // 2) + tl.arange(0, BLOCK_SIZE_K // 2)
    #offsets_bk = tl.max_contiguous(tl.multiple_of(offsets_bk, BLOCK_SIZE_K // 2), BLOCK_SIZE_K // 2)
    #offsets_b = offsets_bk[:, None] + K // 2 * offsets_bn[None, :]
    #offsets_b = K // 2 * offsets_bn[:, None] + offsets_bk[None, :]
    offsets_b = K2 * offsets_bn[:, None] + offsets_bk[None, :]
    zshifts = (offsets_bn[:, None] % 2) * 4 # [N, 1]
    zshifts = zshifts.T

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    for k in range(iter_end - iter_begin):
        masks_ak = offsets_ak < K
        masks_bk = offsets_bk < K2
        masks_a = masks_am[:, None] & masks_ak[None, :]
        masks_b = masks_bn[:, None] & masks_bk[None, :]
        other_bzs = 0.0
        a = tl.load(a_ptrs, mask=masks_a, other=0.)
        b = tl.load(b_ptrs, masks_b, other_bzs) #[N, K//2]
        b = tl.interleave(b, b) # [N, K]

        # Dequantize b.
        offsets_szk = ((BLOCK_SIZE_K * k + iter_begin * BLOCK_SIZE_K) // GROUP_SIZE + tl.arange(0, NUM_GROUPS))
        masks_szk = offsets_szk < K // GROUP_SIZE
        masks_z = masks_szk[:, None] & masks_zn[None, :]
        masks_s = masks_szk[:, None] & masks_sn[None, :]
        #masks_z = masks_zn[:, None] & masks_szk[None, :]
        #masks_s = masks_sn[:, None] & masks_szk[None, :]

        offsets_z = N2 * offsets_szk[:, None] + offsets_zn[None, :]
        #offsets_z = K // GROUP_SIZE * offsets_zn[:, None] + offsets_szk[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=other_bzs) # [K//G, N]
        #zshifts = (offsets_bn[:, None] % 2) * 4 # [N, 1]
        #zeros = (zeros >> _zshifts) & 0xF # [N, K//G]

        '''
        zeros = zeros.T # [K//G, N//2]
        zeros = tl.interleave(zeros, zeros) # [K//G, N]
        zeros = zeros.T # [N, K//G]
        '''

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        #offsets_s = K // GROUP_SIZE * offsets_sn[:, None] + offsets_szk[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=other_bzs) # [K//G, N]

        if NUM_GROUPS == 1:
            # Original efficient implementation for single group
            zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            #zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_N, BLOCK_SIZE_K))
            #scales = tl.broadcast_to(scales, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        else:
            # Reshape to (NUM_GROUPS, 1, N) then broadcast to (NUM_GROUPS, group_size_in_block, N)
            # Reshape to (K//G, 1, N) then broadcast to (K//G, group_size_in_block, N)
            zeros = tl.broadcast_to(zeros[:, None, :], (NUM_GROUPS, GROUP_SIZE, BLOCK_SIZE_N))
            scales = tl.broadcast_to(scales[:, None, :], (NUM_GROUPS, GROUP_SIZE, BLOCK_SIZE_N))
            ## Reshape back to (BLOCK_SIZE_K, N)
            zeros = tl.reshape(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            scales = tl.reshape(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

            # Reshape to (N, K//G, 1) then broadcast to (N, K//G, group_size_in_block)
            #zeros = tl.broadcast_to(zeros[:, :, None], (BLOCK_SIZE_N, NUM_GROUPS, GROUP_SIZE))
            #scales = tl.broadcast_to(scales[:, :, None], (BLOCK_SIZE_N, NUM_GROUPS, GROUP_SIZE))
            #zeros = tl.reshape(zeros, (BLOCK_SIZE_N, BLOCK_SIZE_K))
            #scales = tl.reshape(scales, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        b = (b >> bshifts) & 0xF
        b = b.T
        zeros = (zeros >> zshifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(a_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        offsets_ak += BLOCK_SIZE_K
        offsets_bk += BLOCK_SIZE_K // 2
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // 2

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # compiler hints
    offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_c = M * N * k_idx + N * offs_cm[:, None] + offs_cn[None, :]
    c_ptrs = c_ptr + offs_c
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if USE_REDUCE_KERNEL:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        if c_ptr.type.element_ty == tl.float16:
            tl.store(c_ptrs, c, mask=c_mask)
        elif not_reduce:
            tl.store(c_ptrs, c, mask=c_mask)
        else:
            tl.atomic_add(c_ptrs, c, mask=c_mask)

@triton.jit
def awq_gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    DP_TILES: tl.constexpr, DANGLING_TILES: tl.constexpr,
                    NUM_GROUPS: tl.constexpr, USE_REDUCE_KERNEL: tl.constexpr):

    pid = tl.program_id(axis=0)

    if pid < DP_TILES:
        iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K)
        awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, pid, 0, 0, iters_per_cta,
                              M, N, N2, K, K2, True, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                              NUM_GROUPS, USE_REDUCE_KERNEL)
    else:
        iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
        total_iters = iters_per_tile * DANGLING_TILES
        iters_per_cta = tl.cdiv(total_iters, NUM_CUS)

        iter_begin = (pid - DP_TILES) * iters_per_cta
        iter_end = tl.minimum(iter_begin + iters_per_cta, total_iters)

        while iter_begin < iter_end:
            tile_idx = iter_begin // iters_per_tile + DP_TILES
            tile_iter_begin = (tile_idx - DP_TILES) * iters_per_tile
            tile_iter_end = tile_iter_begin + iters_per_tile
            local_iter_begin = iter_begin - tile_iter_begin
            local_iter_end = tl.minimum(iter_end, tile_iter_end) - tile_iter_begin
            k_idx = tl.cdiv(local_iter_begin, iters_per_cta)
            awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, k_idx, local_iter_begin, local_iter_end,
                                M, N, N2, K, K2, False, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                NUM_GROUPS, USE_REDUCE_KERNEL)
            iter_begin = tile_iter_end

@triton.jit
def awq_gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    SPLITK: tl.constexpr, NUM_GROUPS: tl.constexpr, USE_REDUCE_KERNEL: tl.constexpr):

    pid = tl.program_id(axis=0)

    tiles_M = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_N = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tiles_M * tiles_N
    tile_idx = pid % total_tiles

    iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K * SPLITK)
    iter_begin = pid // total_tiles * iters_per_cta
    iter_end = iter_begin + iters_per_cta
    k_idx = tl.cdiv(iter_begin, iters_per_cta)

    awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, k_idx, iter_begin, iter_end,
                          M, N, N2, K, K2, False, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                          NUM_GROUPS, USE_REDUCE_KERNEL)

@triton.jit
def awq_gemm_kernel_splitk_fused(
    a_ptr, b_ptr, c_ptr,
    zeros_ptr, scales_ptr,
    out_ptr, barrier_ptr,
    M, N, N2, K, K2,
    GROUP_SIZE: tl.constexpr,
    NUM_CUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLITK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    USE_REDUCE_KERNEL: tl.constexpr):

    pid = tl.program_id(axis=0)
    tiles_M = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_N = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tiles_M * tiles_N

    if pid < total_tiles * SPLITK:
        tile_idx = pid % total_tiles
        iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K * SPLITK)
        iter_begin = pid // total_tiles * iters_per_cta
        iter_end = iter_begin + iters_per_cta
        k_idx = tl.cdiv(iter_begin, iters_per_cta)
        awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, k_idx, iter_begin, iter_end,
                            M, N, N2, K, K2, False, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                            NUM_GROUPS, USE_REDUCE_KERNEL)
        # set barriers
        tile_idx_m = tile_idx // tiles_N
        tile_idx_n = tile_idx % tiles_N
        offset = total_tiles * k_idx + tiles_N * tile_idx_m + tile_idx_n
        tl.store(barrier_ptr + offset, 1, cache_modifier=".wt")
    else:
        pid = pid - total_tiles * SPLITK
        # reduce kernel
        pid_m = pid // tiles_N
        pid_n = pid % tiles_N

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        # compiler hints
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # reduce split k
        for batch_idx in range(SPLITK):
            batch_offset = batch_idx * M * N
            input_offsets = batch_offset + offs_m[:, None] * N + offs_n[None, :]
            # wait barrier
            offset = total_tiles * batch_idx + pid_m * tiles_N + pid_n
            while tl.load(barrier_ptr + offset, cache_modifier=".cv", volatile=True) != 1:
                pass
            input_data = tl.load(c_ptr + input_offsets, mask=mask, other=0.0)
            acc += input_data

        output_offsets = offs_m[:, None] * N + offs_n[None, :]
        acc_f16 = acc.to(tl.float16)
        tl.store(out_ptr + output_offsets, acc_f16, mask=mask)

'''
@triton.autotune(
    configs=[
        triton.Config({
        }, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16] for num_stages in [1, 2]
    ],
    key=["M", "K", "N", "GROUP_SIZE", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "SCHEDULER", "SPLITK"],
    perf_debug=True,
    #enable=int(os.getenv("TRITON_DO_AUTOTUNING", 0)) == 1,
    #prune_configs_by={
    #    "early_config_prune": lambda configs, nargs, **kwargs: [
    #        config for config in configs
    #        # SCHEDULE=1 代表 STREAMK，不需要遍历那么多 SPLITK 的值
    #        if config.all_kwargs()["SCHEDULER"] == 0 or (config.all_kwargs()["SCHEDULER"] == 1 and config.all_kwargs()["SPLITK"] == 1)
    #    ]
    #}
)
@triton.heuristics(values={
    "NUM_GROUPS": lambda args: triton.cdiv(args["BLOCK_SIZE_K"], args["GROUP_SIZE"])
})
'''
@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                    NUM_GROUPS: tl.constexpr,
                    DP_TILES: tl.constexpr, DANGLING_TILES: tl.constexpr,
                    SPLITK: tl.constexpr, SCHEDULER: tl.constexpr, USE_REDUCE_KERNEL: tl.constexpr):
        if SCHEDULER == 0:
            return awq_gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2, GROUP_SIZE, NUM_CUS, BLOCK_SIZE_M,
                                          BLOCK_SIZE_N, BLOCK_SIZE_K,
                                          SPLITK, NUM_GROUPS, USE_REDUCE_KERNEL)
        else:
            return awq_gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2, GROUP_SIZE, NUM_CUS, BLOCK_SIZE_M,
                                           BLOCK_SIZE_N, BLOCK_SIZE_K,
                                           DP_TILES, DANGLING_TILES, NUM_GROUPS, USE_REDUCE_KERNEL)

@triton.jit
def awq_gemm_kernel_fused(
    a_ptr, b_ptr, c_ptr,
    zeros_ptr,
    scales_ptr,
    out_ptr, barrier_ptr,
    M, N, N2, K, K2,
    GROUP_SIZE: tl.constexpr,
    NUM_CUS: tl.constexpr,
    # NUM_GEMM_CUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    DP_TILES: tl.constexpr, DANGLING_TILES: tl.constexpr,
    SPLITK: tl.constexpr, SCHEDULER: tl.constexpr,
    USE_REDUCE_KERNEL: tl.constexpr):
        if SCHEDULER == 0:
            return awq_gemm_kernel_splitk_fused(
                a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr,
                out_ptr, barrier_ptr,
                M, N, N2, K, K2,
                GROUP_SIZE,
                NUM_CUS,
                # NUM_GEMM_CUS,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N, BLOCK_SIZE_K,
                SPLITK, NUM_GROUPS, USE_REDUCE_KERNEL)
        else:
            # TODO: to be supported
            assert False
            return awq_gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, N2, K, K2, GROUP_SIZE, NUM_CUS, BLOCK_SIZE_M,
                                           BLOCK_SIZE_N, BLOCK_SIZE_K,
                                           DP_TILES, DANGLING_TILES, NUM_GROUPS, USE_REDUCE_KERNEL)

# qweights - [N     , K // 2], int8
# scales   - [K // G, N     ], float16
# zeros    - [K // G, N // 2], int8
# result   - [K, N], float16
def awq_dequantize_triton(qweight: torch.Tensor,
                          scales: torch.Tensor,
                          zeros: torch.Tensor,
                          **kwargs) -> torch.Tensor:
    N = qweight.shape[0]
    K = qweight.shape[1] * 2
    group_size = K // scales.shape[0]

    assert K > 0 and N > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == N // 2
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    configs = {
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "num_warps": 4,
        "num_stages": 1
    }

    result = torch.empty(K, N, device=qweight.device, dtype=scales.dtype)

    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    awq_dequantize_kernel[grid](qweight,
                                scales,
                                zeros,
                                result,
                                N,
                                N//2,
                                K,
                                K//2,
                                group_size,
                                **configs
                                )

    return result


@functools.lru_cache
def get_w4a16_awq_gemm_config_filepath(N: int, K: int, GROUP_SIZE: int, **kwargs) -> str:
    device_name = arch_info.get_device()
    if device_name.lower().startswith("bw"):
        device_name = "BW200"
    if "k100" in device_name.lower():
        device_name = "K100_AI"
    json_file_name = f"awq_gemm_N={N},K={K},device_name={device_name},dtype=w4a16,group_size={GROUP_SIZE}.json"

    config_file_path = os.path.join(
        f"{AITER_TRITON_CONFIGS_PATH}", "gemm/awq_w4a16", json_file_name
    )
    return config_file_path

@functools.lru_cache
def get_w4a16_awq_gemm_configs(
    N: int, K: int, GROUP_SIZE: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    config_file_path = get_w4a16_awq_gemm_config_filepath(N, K, GROUP_SIZE)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
            f"\nUsing default W4A16 AWQ GEMM kernel config. Performance might "
            f"be sub-optimal! Config file not found at {config_file_path}")
    return None

# The inference function
# input   - [m, k]
# qweight - [n, k // 2]
# qzeros  - [k//g, n//2]
# scales  - [k//g, n]
def gemm_a16w4(input: torch.tensor,
               qweight: torch.tensor,
               scales: torch.tensor,
               qzeros: torch.tensor,
               use_fused_kernel: int = 0,
               configs: Optional[Dict] = None) -> torch.tensor:
            #    not_used_placeholder: int = 0) -> torch.tensor:

    M, K = input.shape
    N = qweight.shape[0] # (N, K//2)
    group_size = K // qzeros.shape[0]

    default_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "SCHEDULER": 0,
        "SPLITK": 1,
        "D_SHAPE": (M, N),
        "D_DTYPE": 16,
        "DP_TILES": 0,
        "DANGLING_TILES": 0,
        "NUM_CUS": 0,
        "NUM_CUS_STREAMK": 0,
        "NUM_GROUPS":(32 + group_size - 1) // group_size,
        "USE_REDUCE_KERNEL": False
    }
    if configs is None:
        configs = get_w4a16_awq_gemm_configs(N, K, group_size)
    config = configs[min(configs.keys(), key=lambda x: abs(x - M))] if configs else default_config
    # Make sure not getting this wrong from other configs
    d_shape = list(config["D_SHAPE"])
    d_shape[-2] = M
    config["D_SHAPE"] = d_shape
    # if use_fused_kernel == 1 and config["SPLITK"] > 1 and config["USE_REDUCE_KERNEL"]:
    #     return awq_gemm_triton_fused_impl(
    #         input, qweight, scales, qzeros, config, config.copy(), awq_gemm_kernel_fused)

    return awq_gemm_triton_impl(input, qweight, scales, qzeros, config, config.copy(), awq_gemm_kernel)

def awq_gemm_triton_fused_impl(
    input: torch.tensor,
    qweight: torch.tensor,
    scales: torch.tensor,
    qzeros: torch.tensor,
    config: Dict,
    cfg4kernel: Dict,
    func) -> torch.tensor:
    M, K = input.shape
    N = qweight.shape[0] # (N, K//2)
    assert(qweight.is_contiguous())
    group_size = qweight.shape[1] * 2 // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[1] == K // 2 and qweight.shape[0] == N
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 2
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    num_cus = config["NUM_CUS"]
    d_shape = config["D_SHAPE"]
    d_dtype = config["D_DTYPE"]
    d_dtype = torch.float16 if d_dtype == 16 else torch.float32

    def grid(META):
        # tiles_M = (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"]
        # tiles_N = (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"]
        tiles_M = triton.cdiv(M, META["BLOCK_SIZE_M"])
        tiles_N = triton.cdiv(N, META["BLOCK_SIZE_N"])
        total_tiles = tiles_M * tiles_N
        if META["SCHEDULER"] == 0:
            # dp or splitk
            # add extra total_tiles for reduction
            return (total_tiles * META["SPLITK"] + total_tiles,)
        else:
            # TODO: not supported yet
            # streamk
            return (META["DP_TILES"] + config["NUM_CUS_STREAMK"],)

    result = torch.zeros(d_shape, dtype=d_dtype, device=input.device)

    cfg4kernel.pop("D_SHAPE", None)
    cfg4kernel.pop("D_DTYPE", None)
    cfg4kernel.pop("NUM_CUS_STREAMK", None)

    fn = func[grid]

    if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
        fn = partial(func.warmup, grid=grid)

    total_tiles_splitk = \
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]) * config["SPLITK"]
    final_result = torch.zeros((M, N), dtype=torch.float16, device=input.device)
    barrier = torch.zeros((total_tiles_splitk, ), dtype=torch.float16, device=input.device)

    fn(input,
       qweight,
       result,
       qzeros,
       scales,
       final_result, # new added
       barrier,      # new added
       M,
       N,
       N//2,
       K,
       K//2,
       group_size,
       **cfg4kernel)

    if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
        return

    return final_result

def awq_gemm_triton_impl(input: torch.tensor,
                        qweight: torch.tensor,
                        scales: torch.tensor,
                        qzeros: torch.tensor,
                        config: Dict,
                        cfg4kernel: Dict,
                        func) -> torch.tensor:
    M, K = input.shape
    N = qweight.shape[0] # (N, K//2)
    assert(qweight.is_contiguous())
    group_size = qweight.shape[1] * 2 // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[1] == K // 2 and qweight.shape[0] == N
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 2
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    num_cus = config["NUM_CUS"]
    d_shape = config["D_SHAPE"]
    d_dtype = config["D_DTYPE"]
    d_dtype = torch.float16 if d_dtype == 16 else torch.float32

    #curr_num_cus = torch.cuda.get_device_properties("cuda").multi_processor_count
    #if num_cus > 0 and num_cus != curr_num_cus:
    #    print("AWQ_GEMM config tuned based on num_cus={num_cus}, but now running on num_cus={curr_num_cus}, may lead to bad performance!")

    def grid(META):
        tiles_M = (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"]
        tiles_N = (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        if META["SCHEDULER"] == 0:
            # dp or splitk
            return (total_tiles * META["SPLITK"],)
        else:
            # streamk
            return (META["DP_TILES"] + config["NUM_CUS_STREAMK"],)

    if cfg4kernel["USE_REDUCE_KERNEL"]:
        result = torch.zeros(d_shape, dtype=d_dtype, device=input.device)
    elif cfg4kernel["DP_TILES"] > 0 or cfg4kernel["SPLITK"] > 1:
        result = torch.zeros(d_shape, dtype=d_dtype, device=input.device)
    else:
        result = torch.empty(d_shape, dtype=d_dtype, device=input.device)

    cfg4kernel.pop("D_SHAPE", None)
    cfg4kernel.pop("D_DTYPE", None)
    cfg4kernel.pop("NUM_CUS_STREAMK", None)

    fn = func[grid]

    if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
        fn = partial(func.warmup, grid=grid)

    fn(input,
       qweight,
       result,
       qzeros,
       scales,
       M,
       N,
       N//2,
       K,
       K//2,
       group_size,
       **cfg4kernel)

    if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
        return

    if result.ndim == 3:
        batch_size = result.shape[0]
        final_result = torch.empty((M, N), dtype=torch.float16, device=input.device)
        awq_reduce_and_convert_triton(result, final_result, M, N, batch_size)
        return final_result
    else:
        result = result.to(torch.float16)
        return result

# The tuning functions below
def prune_configs(configs, nargs, **kwargs):

    def _ceil_div(x, y):
        return (x + y - 1) // y

    def _prune(config):
        _config = config.all_kwargs()
        all_kwargs = {**_config, **kwargs, **nargs}

        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = all_kwargs["BLOCK_SIZE_M"], all_kwargs["BLOCK_SIZE_N"], all_kwargs["BLOCK_SIZE_K"]
        num_stages = all_kwargs["num_stages"]

        if num_stages > 1 and (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_K * BLOCK_SIZE_N) * 2 > 16384:
            return True

    remained = [c for c in configs if not _prune(c)]
    return remained

def update_config(M, K, N, G, cfg):
    config = cfg.copy()
    # 根据基本的配置计算其他参数，一则用于 launch，一则避免 kernel 内重复计算
    config["NUM_GROUPS"] = (config["BLOCK_SIZE_K"] + G - 1) // G
    if config["SCHEDULER"] == 0 and config["SPLITK"] == 1:
        # dp
        config["DP_TILES"] = 0
        config["DANGLING_TILES"] = 0
        config["D_SHAPE"] = (M, N)
        config["D_DTYPE"] = 16
    elif config["SCHEDULER"] == 0 and config["SPLITK"] > 1:
        # splitk
        config["DP_TILES"] = 0
        config["DANGLING_TILES"] = 0
        config["D_DTYPE"] = 32
        config["D_SHAPE"] = (config["SPLITK"], M, N) if config["USE_REDUCE_KERNEL"] else (M, N)
    else:
        # streamk
        tiles_M = (M + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]
        tiles_N = (N + config["BLOCK_SIZE_N"] - 1) // config["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        dangling_tiles = max(0, total_tiles - config["NUM_CUS"]) % config["NUM_CUS"]
        dp_tiles = total_tiles - dangling_tiles
        if dangling_tiles == 0:
            # redirect to dp
            config["SCHEDULER"] = 0
            config["SPLITK"] = 1
            config["USE_REDUCE_KERNEL"] = 0
            config["DP_TILES"] = 0
            config["DANGLING_TILES"] = 0
            config["D_SHAPE"] = (M, N)
            config["D_DTYPE"] = 16
        else:
            # still streamk
            config["DP_TILES"] = dp_tiles
            config["DANGLING_TILES"] = dangling_tiles
            iters_per_tile = (K + config["BLOCK_SIZE_K"] - 1) // config["BLOCK_SIZE_K"]
            dangling_iters = iters_per_tile * config["DANGLING_TILES"]
            dangling_iters_per_cu = (dangling_iters + config["NUM_CUS"] - 1) // config["NUM_CUS"]
            num_cus_streamk = (dangling_iters + dangling_iters_per_cu - 1) // dangling_iters_per_cu
            config["NUM_CUS_STREAMK"] = num_cus_streamk
            num_cus_per_dangling_tile = (iters_per_tile + dangling_iters_per_cu - 1) // dangling_iters_per_cu + 1
            config["D_DTYPE"] = 32
            config["D_SHAPE"] = (num_cus_per_dangling_tile, M, N) if config["USE_REDUCE_KERNEL"] else (M, N)

    return config

'''
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": M,
            "BLOCK_SIZE_N": N,
        }, num_warps=num_warps, num_stages=num_stages)
        for M in [128, 64, 32,  16] for N in [512, 128, 64, 32, 16]\
        for num_warps in [1, 2, 4, 8, 16] for num_stages in [1, 2]
    ],
    key=["M", "N", "batch_size"],
    perf_debug=True
)
'''

@triton.jit
def awq_reduce_and_convert_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    tl.assume(M >= 0)
    tl.assume(N >= 0)
    tl.assume(batch_size >= 0)
    tl.assume(BLOCK_SIZE_M >= 0)
    tl.assume(BLOCK_SIZE_N >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for batch_idx in range(batch_size):
        batch_offset = batch_idx * M * N
        input_offsets = batch_offset + offs_m[:, None] * N + offs_n[None, :]
        input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        acc += input_data

    output_offsets = offs_m[:, None] * N + offs_n[None, :]
    acc_f16 = acc.to(tl.float16)
    tl.store(output_ptr + output_offsets, acc_f16, mask=mask)

def awq_reduce_and_convert_triton(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    M: int,
    N: int,
    batch_size: int = 1
) -> None:

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 16

    awq_reduce_and_convert_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        batch_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps
    )
