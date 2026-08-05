#!/usr/bin/env python3
"""生成 aiter gemm_a16w4 的 BW10 调优 config json 文件。
基于 tune_aiter_config.py 的调优结果, 覆盖 M=1..128。"""
import os, json

CONFIG_DIR="/public/home/weishb/aiter/aiter/ops/triton/configs/gemm/awq_w4a16"
os.makedirs(CONFIG_DIR, exist_ok=True)

# 调优结果: 每个形状的最优 tile (来自 tune_aiter_config.py)
# q_proj/gate/o_proj: BM16 BN64 BK32 SPLITK2 NW4
# down(K大): BM16 BN64 BK64 SPLITK1 NW4 (M>=8 时 BK32 SPLITK2)
# 统一用 SPLITK2 + BK32 对小M更稳, down 用 BK64 SPLITK1
TUNED={
    (1344,5376): {"BM":16,"BN":64,"BK":32,"SPLITK":2,"NW":4},  # q_proj
    (3584,5376): {"BM":16,"BN":64,"BK":32,"SPLITK":2,"NW":4},  # gate/up
    (5376,5376): {"BM":16,"BN":64,"BK":32,"SPLITK":2,"NW":4},  # o_proj
    (5376,14336):{"BM":16,"BN":64,"BK":64,"SPLITK":1,"NW":4},  # down (K大)
}

def make_config(M, N, K, G, tuned):
    BM,BN,BK,SK,NW = tuned["BM"],tuned["BN"],tuned["BK"],tuned["SPLITK"],tuned["NW"]
    num_groups = (BK + G - 1) // G
    if SK > 1:
        D_DTYPE = 32
        D_SHAPE = [M, N]
    else:
        D_DTYPE = 16
        D_SHAPE = [M, N]
    return {
        "USE_REDUCE_KERNEL": False,
        "NUM_CUS": 80,  # BW10 CU数 (gfx936)
        "SCHEDULER": 0,
        "BLOCK_SIZE_M": BM,
        "BLOCK_SIZE_N": BN,
        "BLOCK_SIZE_K": BK,
        "SPLITK": SK,
        "num_warps": NW,
        "num_ctas": 1,
        "num_stages": 1,
        "NUM_GROUPS": num_groups,
        "DP_TILES": 0,
        "DANGLING_TILES": 0,
        "D_DTYPE": D_DTYPE,
        "D_SHAPE": D_SHAPE,
    }

G=32
for (N,K), tuned in TUNED.items():
    cfgs={}
    for M in [1,2,4,8,16,32,64,128]:
        cfgs[M]=make_config(M,N,K,G,tuned)
    fname=f"awq_gemm_N={N},K={K},device_name=BW200,dtype=w4a16,group_size={G}.json"
    path=os.path.join(CONFIG_DIR, fname)
    with open(path,"w") as f:
        json.dump(cfgs, f, indent=2)
    print(f"写入 {fname}  (M4: BM={tuned['BM']} BN={tuned['BN']} BK={tuned['BK']} SK={tuned['SPLITK']})")

print("\n所有 config 已生成")
