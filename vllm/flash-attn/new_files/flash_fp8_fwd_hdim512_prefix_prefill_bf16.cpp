// Copyright (c) 2025, Xin Zhou.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
//
// gemma4 MTP draft 模型 full_attention 层 (global_head_dim=512) + fp8 KV prefill.
// 实例化 run_fp8_mha_fwd_prefix_prefill_<BFloat16, 512, 512>, 复用 gfx936 mixed kernel.

#include "../flash_fwd_launch_template.h"

template<>
void run_fp8_mha_fwd_prefix_prefill_<BFloat16, 512, 512>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_fp8_flash_fwd_prefix_prefill<BFloat16, 512, 512>(params, stream);
#endif
}
