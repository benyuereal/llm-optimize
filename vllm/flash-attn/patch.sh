#!/bin/bash
# ============================================================
# flash attention fp8 KV + head_dim=512 patch · 一键安装 / 回退脚本
#
# 让 gemma-4 MTP draft 模型的 full_attention 层 (head_dim=512) 走 flash
# mixed kernel, 替代慢的 aiter 2D kernel。TPOT -40% (1.68x 加速), 精度无损。
#
# 本 patch 改两层:
#   1. flash_attn python 包 (pip whl, 200M, 替换官方版) —— 新增 fp8_e5m2
#      mixed kernel + head_dim=512 prefill 符号。
#   2. vllm 源码 (3 个文件, patch -p0) —— 让 vllm 支持 fp8_e5m2 KV cache:
#        attention.py                       放行 e5m2 (原本对 compressed-tensors 模型一律报错)
#        rocm_aiter_unified_attn.py         读侧 view 用 e5m2 + 写侧走 triton (C++ op 不支持 e5m2)
#        triton_reshape_and_cache_flash.py  写侧按字符串选 e5m2 dtype
#      不打这层 vllm patch, 新容器启动会报:
#        ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.
#
# 用法:
#   ./patch.sh install   # 安装: 卸载旧 flash_attn, 装本目录 whl, 打 vllm patch
#   ./patch.sh revert    # 回退: 反向打 vllm patch + 卸载本 whl
#   ./patch.sh status    # 查看当前安装状态 (whl + vllm patch)
# ============================================================
set -e

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHL_DIR="$SCRIPT_DIR/dist"

# vllm 安装根目录 (pip 装的位置; vllm 侧 patch 在此目录下 -p0 执行)
DIST_DIR=/usr/local/lib/python3.10/dist-packages

# vllm 侧 patch (3 个文件: attention.py + rocm_aiter_unified_attn.py + triton_reshape_and_cache_flash.py)
VLLM_PATCH="$SCRIPT_DIR/flash_fp8e5m2.patch"

# vllm 侧被改的 3 个文件 (用于 status 检测)
VLLM_ATTN="$DIST_DIR/vllm/model_executor/layers/attention/attention.py"
VLLM_UNIFIED="$DIST_DIR/vllm/v1/attention/backends/rocm_aiter_unified_attn.py"
VLLM_RESHAPE="$DIST_DIR/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"

# 优先用环境变量 WHL 指定的 whl, 其次 dist/ 目录
if [ -n "$WHL" ] && [ -f "$WHL" ]; then
    : # 用环境变量
else
    WHL=$(ls "$WHL_DIR"/flash_attn-*.whl 2>/dev/null | head -1)
fi

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

ACTION="${1:-}"

usage() {
    echo "用法: $0 {install|revert|status}"
    echo "  install  装 flash_attn whl + 打 vllm 侧 fp8_e5m2 patch"
    echo "  revert   反向打 vllm patch + 卸载本 whl"
    echo "  status   查看当前安装状态"
    exit 1
}

[ -z "$ACTION" ] && usage

# 判断 vllm 侧 patch 是否已打 (检查 attention.py 是否含我们的标记)
vllm_patch_state() {
    if [ ! -f "$VLLM_ATTN" ]; then
        echo "missing"
    elif grep -q "_ckpt_kv_scheme" "$VLLM_ATTN" 2>/dev/null; then
        echo "patch"
    else
        echo "original"
    fi
}

# 清 torch.compile 缓存 (vllm 源码改动后必做)
clear_compile_cache() {
    info "清空 torch.compile 缓存..."
    rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root 2>/dev/null || true
    rm -rf ~/.cache/vllm/torch_compile_cache 2>/dev/null || true
}

case "$ACTION" in
install)
    # ---------- 1. 装 flash_attn whl ----------
    if [ -z "$WHL" ]; then
        err "未找到 whl 文件: $WHL_DIR/flash_attn-*.whl"
        echo ""
        echo "  whl 体积较大 (200M), 不随仓库分发。请先下载放到 dist/ 目录:"
        echo "    1. 到本仓库 GitHub Release 页面下载 flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl"
        echo "    2. 放到 $WHL_DIR/"
        echo "    3. 重新执行: bash patch.sh install"
        echo ""
        echo "  或若已有 whl, 指定路径: WHL=/path/to/flash_attn-*.whl bash patch.sh install"
        exit 1
    fi
    info "==== 1/2 安装 flash_attn whl: $(basename "$WHL") ===="
    info "卸载旧版本..."
    pip3 uninstall -y flash_attn >/dev/null 2>&1 || true
    info "安装新 whl (--no-deps, 避免升级 torch 破坏 DCU 环境)..."
    pip3 install --force-reinstall --no-deps "$WHL"
    info "验证 import..."
    python3 -c "import flash_attn; print('  flash_attn version:', flash_attn.__version__)"
    info "验证 512 prefill 符号 (fp16 + bf16)..."
    SO=$(python3 -c "import flash_attn,os;print(os.path.join(os.path.dirname(flash_attn.__file__),'lib','libflash_attention.so'))" 2>/dev/null)
    if [ -n "$SO" ] && [ -f "$SO" ]; then
        N=$(nm -D "$SO" 2>/dev/null | grep -c "run_fp8_mha_fwd_prefix_prefill.*512")
        if [ "$N" -ge 2 ]; then
            info "符号验证通过: fp16 + bf16 的 512 prefill 符号均存在 ($N 个)"
        else
            warn "符号验证: 只找到 $N 个 512 prefill 符号 (预期 2 个)"
        fi
    else
        warn "未找到 libflash_attention.so, 跳过符号验证"
    fi

    # ---------- 2. 打 vllm 侧 fp8_e5m2 patch ----------
    echo ""
    info "==== 2/2 打 vllm 侧 fp8_e5m2 patch ===="
    if [ ! -f "$VLLM_PATCH" ]; then
        err "找不到 vllm 侧 patch: $VLLM_PATCH"
        exit 1
    fi
    if [ ! -d "$DIST_DIR/vllm" ]; then
        err "vllm 安装目录不存在: $DIST_DIR/vllm"
        exit 1
    fi

    local_vllm_state=$(vllm_patch_state)
    if [ "$local_vllm_state" = "patch" ]; then
        warn "vllm 侧 patch 已打, 跳过 (如需重打请先 ./patch.sh revert)"
    elif [ "$local_vllm_state" = "missing" ]; then
        err "vllm 文件不存在: $VLLM_ATTN"
        exit 1
    else
        info "打 patch: $VLLM_PATCH"
        info "  (attention.py + rocm_aiter_unified_attn.py + triton_reshape_and_cache_flash.py)"
        (cd "$DIST_DIR" && patch -p0 --no-backup-if-mismatch < "$VLLM_PATCH") 2>&1 | sed 's/^/  /'
        info "vllm 侧 fp8_e5m2 patch 已应用"
        clear_compile_cache
    fi

    echo ""
    info "==== 安装完成! ===="
    echo "  启动服务请用:"
    echo "    bash $SCRIPT_DIR/models/gemma4/start_flash.sh"
    echo ""
    echo "  回退: $0 revert"
    ;;

revert)
    # ---------- 1. 反向打 vllm 侧 patch ----------
    info "==== 1/2 反向打 vllm 侧 fp8_e5m2 patch ===="
    local_vllm_state=$(vllm_patch_state)
    if [ "$local_vllm_state" = "original" ]; then
        warn "vllm 侧未打 patch, 跳过"
    elif [ "$local_vllm_state" = "missing" ]; then
        err "vllm 文件不存在: $VLLM_ATTN"
        exit 1
    else
        info "反向打 patch: $VLLM_PATCH"
        (cd "$DIST_DIR" && patch -p0 -R --no-backup-if-mismatch < "$VLLM_PATCH") 2>&1 | sed 's/^/  /'
        info "已回退 vllm 侧到原始状态"
        clear_compile_cache
    fi

    # ---------- 2. 卸载 whl ----------
    echo ""
    info "==== 2/2 卸载 flash_attn whl ===="
    pip3 uninstall -y flash_attn
    warn "已卸载。如需恢复官方版, 请自行 pip3 install flash-attn 或重装 DCU 定制版。"
    ;;

status)
    info "==== flash_attn whl 状态 ===="
    VER=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    PATH_INFO=$(python3 -c "import flash_attn; print(flash_attn.__file__)" 2>/dev/null)
    if [ -n "$VER" ]; then
        echo "  版本: $VER"
        echo "  路径: $PATH_INFO"
        if [ -n "$WHL" ]; then
            case "$PATH_INFO" in
                *dist-packages/flash_attn*) echo "  本包 whl: $(basename "$WHL")" ;;
            esac
        fi
    else
        echo "  未安装 flash_attn"
    fi

    echo ""
    info "==== vllm 侧 fp8_e5m2 patch 状态 ===="
    local_vllm_state=$(vllm_patch_state)
    case "$local_vllm_state" in
        patch)    info "attention.py : ${GREEN}fp8_e5m2 patch 已打${NC}" ;;
        original) warn "attention.py : 原始版 (未打 patch)" ;;
        missing)  err  "attention.py : 不存在 ($VLLM_ATTN)" ;;
    esac
    # 其余 2 个文件
    if [ -f "$VLLM_UNIFIED" ]; then
        if grep -q "triton_reshape_and_cache_flash" "$VLLM_UNIFIED" 2>/dev/null && \
           grep -q "torch.float8_e5m2" "$VLLM_UNIFIED" 2>/dev/null; then
            info "rocm_aiter_unified_attn.py : 已改 (e5m2 view + triton 写路径)"
        else
            warn "rocm_aiter_unified_attn.py : 原始版"
        fi
    fi
    if [ -f "$VLLM_RESHAPE" ]; then
        if grep -q "torch.float8_e5m2 if kv_cache_dtype" "$VLLM_RESHAPE" 2>/dev/null; then
            info "triton_reshape_and_cache_flash.py : 已改 (e5m2 dtype 选择)"
        else
            warn "triton_reshape_and_cache_flash.py : 原始版"
        fi
    fi
    ;;

*)
    usage
    ;;
esac
