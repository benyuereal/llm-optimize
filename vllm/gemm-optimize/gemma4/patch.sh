#!/bin/bash
# ============================================================
# gemma-4-31B-it-AWQ-4bit · aiter w4a16 GEMM patch
# 一键安装 / 回退脚本
#
# 用法:
#   ./patch.sh install   # 安装 aiter patch (替换 vllm triton_w4a16.py + 放置 aiter kernel/config)
#   ./patch.sh revert    # 回退到 vllm 原始 triton_w4a16.py
#   ./patch.sh status    # 查看当前安装状态
# ============================================================
set -e

# ---------- 路径配置 (按需修改) ----------
# vllm 安装目录里的 triton_w4a16.py 所在路径
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/kernels/linear/mixed_precision

# aiter 仓库根目录 (AITER_ROOT)
AITER_ROOT=/public/home/weishb/aiter

# 本脚本所在目录 (patch 文件就在这里)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ----------------------------------------

VLLM_FILE="$VLLM_DIR/triton_w4a16.py"
ORIG="$SCRIPT_DIR/triton_w4a16.py"          # vllm 原始版
PATCH="$SCRIPT_DIR/triton_w4a16.py.patch"   # aiter patch 版
AITER_KERNEL="$SCRIPT_DIR/aiter_gemm_a16w4.py"
CONFIG_DIR="$SCRIPT_DIR/configs/awq_w4a16"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# 判断当前 vllm 装的是哪一版
which_version() {
    if [ ! -f "$VLLM_FILE" ]; then
        echo "missing"
    elif diff -q "$VLLM_FILE" "$PATCH" >/dev/null 2>&1; then
        echo "patch"
    elif diff -q "$VLLM_FILE" "$ORIG" >/dev/null 2>&1; then
        echo "original"
    else
        echo "unknown"
    fi
}

# 清 torch.compile 缓存 (切换后必做, 否则可能报 aiter namespace 找不到)
clear_compile_cache() {
    info "清空 torch.compile 缓存..."
    rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root 2>/dev/null || true
    rm -rf ~/.cache/vllm/torch_compile_cache 2>/dev/null || true
}

# ---------- install ----------
do_install() {
    info "==== 安装 aiter w4a16 patch ===="

    # 0. 检查源文件
    for f in "$PATCH" "$AITER_KERNEL" "$CONFIG_DIR"; do
        if [ ! -e "$f" ]; then err "找不到 $f"; exit 1; fi
    done
    if [ ! -d "$VLLM_DIR" ]; then err "vllm 目录不存在: $VLLM_DIR"; exit 1; fi

    # 1. 备份当前 vllm 原始 triton_w4a16.py (只备份一次, 且不备份 patch 版)
    #    若 .bak 已存在, 或当前文件已经是 patch 版, 则跳过 (避免把 patch 版当原始版备份)
    if [ ! -f "$VLLM_FILE.bak" ]; then
        if diff -q "$VLLM_FILE" "$PATCH" >/dev/null 2>&1; then
            warn "当前 vllm 文件已是 patch 版, 跳过备份 (.bak 用 repo 的原始版替代)"
        else
            cp "$VLLM_FILE" "$VLLM_FILE.bak"
            info "已备份原始文件 -> $VLLM_FILE.bak"
        fi
    fi

    # 2. 替换 vllm triton_w4a16.py
    cp "$PATCH" "$VLLM_FILE"
    info "已替换 vllm triton_w4a16.py -> aiter patch 版"

    # 3. 放置 aiter kernel (修过 triton 3.5 兼容)
    mkdir -p "$AITER_ROOT/aiter/ops/triton"
    cp "$AITER_KERNEL" "$AITER_ROOT/aiter/ops/triton/gemm_a16w4.py"
    info "已放置 aiter kernel -> $AITER_ROOT/aiter/ops/triton/gemm_a16w4.py"

    # 4. 放置调优 config (gs=32, BW200)
    AITER_CFG="$AITER_ROOT/aiter/ops/triton/configs/gemm/awq_w4a16"
    mkdir -p "$AITER_CFG"
    cp "$CONFIG_DIR"/*.json "$AITER_CFG/" 2>/dev/null || true
    info "已放置 aiter config ($(ls "$CONFIG_DIR"/*.json 2>/dev/null | wc -l) 个) -> $AITER_CFG"

    # 5. 清缓存
    clear_compile_cache

    # 6. 提示环境变量
    echo
    info "==== 安装完成 ===="
    echo
    echo "启动 vllm 前请设置环境变量 (或在 start 脚本里加):"
    echo -e "  ${GREEN}export VLLM_AITER_W4A16_PATCH=1${NC}   # 启用 aiter patch"
    echo -e "  ${GREEN}export AITER_ROOT=$AITER_ROOT${NC}"
    echo
    echo "回退:  ./patch.sh revert"
}

# ---------- revert ----------
do_revert() {
    info "==== 回退到 vllm 原始 triton_w4a16.py ===="

    # 优先用 .bak 恢复; 但若 .bak 是错误备份 (内容 == patch 版), 改用 repo 的原始版
    restored=0
    if [ -f "$VLLM_FILE.bak" ] && ! diff -q "$VLLM_FILE.bak" "$PATCH" >/dev/null 2>&1; then
        cp "$VLLM_FILE.bak" "$VLLM_FILE"
        info "已从 .bak 恢复原始 triton_w4a16.py"
        restored=1
    fi
    if [ "$restored" -eq 0 ] && [ -f "$ORIG" ]; then
        cp "$ORIG" "$VLLM_FILE"
        info "已用 repo 的 triton_w4a16.py 恢复原始版"
        restored=1
    fi
    if [ "$restored" -eq 0 ]; then
        err "找不到原始版文件, 无法回退"
        exit 1
    fi

    clear_compile_cache

    echo
    info "==== 回退完成 ===="
    echo
    echo "若 start 脚本里有 ${YELLOW}VLLM_AITER_W4A16_PATCH=1${NC}, 请改为 ${YELLOW}=0${NC} (或 unset) 再启动 vllm。"
}

# ---------- status ----------
do_status() {
    echo "==== 当前状态 ===="
    local v; v=$(which_version)
    case "$v" in
        patch)    info "vllm triton_w4a16.py : ${GREEN}aiter patch 版${NC}" ;;
        original) warn "vllm triton_w4a16.py : 原始版 (未打 patch)" ;;
        missing)  err  "vllm triton_w4a16.py : 不存在 ($VLLM_FILE)" ;;
        *)        warn "vllm triton_w4a16.py : 未知版本 (既非原始也非 patch, 可能被手动改过)" ;;
    esac

    if [ -f "$VLLM_FILE.bak" ]; then
        info "原始备份存在 : $VLLM_FILE.bak"
    else
        warn "原始备份不存在 (.bak)"
    fi

    if [ -f "$AITER_ROOT/aiter/ops/triton/gemm_a16w4.py" ]; then
        info "aiter kernel  : 已放置 ($AITER_ROOT/aiter/ops/triton/gemm_a16w4.py)"
    else
        warn "aiter kernel  : 未放置"
    fi

    local cfg_n; cfg_n=$(ls "$AITER_ROOT/aiter/ops/triton/configs/gemm/awq_w4a16/" 2>/dev/null | grep "group_size=32" | grep "BW200" | wc -l)
    if [ "$cfg_n" -gt 0 ]; then
        info "aiter config  : $cfg_n 个 (gs=32, BW200)"
    else
        warn "aiter config  : 未放置 (gs=32, BW200)"
    fi

    echo
    echo "环境变量 (当前 shell):"
    echo "  VLLM_AITER_W4A16_PATCH = ${VLLM_AITER_W4A16_PATCH:-(未设置)}"
    echo "  AITER_ROOT             = ${AITER_ROOT:-(未设置)}"
}

# ---------- main ----------
case "${1:-}" in
    install) do_install ;;
    revert)  do_revert ;;
    status)  do_status ;;
    *)
        echo "用法: $0 {install|revert|status}"
        echo
        echo "  install  安装 aiter w4a16 patch (替换 vllm kernel + 放置 aiter kernel/config)"
        echo "  revert   回退到 vllm 原始 triton_w4a16.py"
        echo "  status   查看当前安装状态"
        exit 1
        ;;
esac
