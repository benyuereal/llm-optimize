#!/bin/bash
# ============================================================
# aiter w4a16 GEMM patch (通用) · 一键安装 / 回退脚本
#
# 用 aiter triton w4a16 kernel 替换 vllm 自带 triton w4a16 kernel。
# kernel 和 vllm 接入是通用的; 调优 config 按模型区分 (models/<model>/)。
#
# 用法:
#   ./patch.sh install [model]   # 安装 patch (默认 model=gemma4)
#   ./patch.sh revert            # 回退 (patch -R)
#   ./patch.sh status [model]    # 查看当前安装状态 (默认 model=gemma4)
#   ./patch.sh models            # 列出可选的模型
#
# 示例:
#   ./patch.sh install           # 装 patch, 用 gemma4 的 config
#   ./patch.sh install gemma4    # 同上 (显式指定)
#   ./patch.sh status gemma4
#
# 原理: 用 patch 命令把 aiter.patch 打到 vllm + aiter 安装目录。
#   install = patch -p0 < aiter.patch
#   revert  = patch -p0 -R < aiter.patch
# ============================================================
set -e

# ---------- 路径配置 (按需修改) ----------
# vllm + aiter 安装根目录 (pip 装的位置; patch 在此目录下 -p0 执行)
DIST_DIR=/usr/local/lib/python3.10/dist-packages

# vllm 安装目录里的 triton_w4a16.py 所在路径
VLLM_DIR=$DIST_DIR/vllm/model_executor/kernels/linear/mixed_precision

# aiter 包安装路径 (pip install aiter 装的位置)
AITER_PKG_DIR=$DIST_DIR/aiter

# 本脚本所在目录 (aiter.patch + models/ 就在这里)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ----------------------------------------

DEFAULT_MODEL=gemma4
MODELS_DIR="$SCRIPT_DIR/models"

VLLM_FILE="$VLLM_DIR/triton_w4a16.py"
AITER_KERNEL_DST="$AITER_PKG_DIR/ops/triton/gemm_a16w4.py"
AITER_CFG_DIR="$AITER_PKG_DIR/ops/triton/configs/gemm/awq_w4a16"

# patch 文件 (3 段合一: vllm triton_w4a16 + aiter kernel + configs)
PATCH_FILE="$SCRIPT_DIR/aiter.patch"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# 解析模型参数: $1=动作, $2=模型(可选)
ACTION="${1:-}"
MODEL="${2:-$DEFAULT_MODEL}"
CONFIG_DIR="$MODELS_DIR/$MODEL/configs/awq_w4a16"

# 判断当前是否已打 patch (检查 triton_w4a16.py 是否含 aiter 标记)
which_version() {
    if [ ! -f "$VLLM_FILE" ]; then
        echo "missing"
    elif grep -q "_aiter_preprocess_layer" "$VLLM_FILE" 2>/dev/null; then
        echo "patch"
    else
        echo "original"
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
    info "==== 安装 aiter w4a16 patch (模型: $MODEL) ===="

    # 0. 检查
    if [ ! -f "$PATCH_FILE" ]; then err "找不到 patch 文件: $PATCH_FILE"; exit 1; fi
    if [ ! -d "$VLLM_DIR" ]; then err "vllm 目录不存在: $VLLM_DIR"; exit 1; fi
    if [ ! -d "$AITER_PKG_DIR" ]; then
        err "aiter 包不存在: $AITER_PKG_DIR"
        echo "请先 pip install aiter (DCU 定制版), 或修改脚本顶部的 AITER_PKG_DIR"; exit 1
    fi

    # 已打 patch 则跳过
    local v; v=$(which_version)
    if [ "$v" = "patch" ]; then
        warn "当前已打 patch, 跳过 (如需重打请先 ./patch.sh revert)"
        return 0
    fi

    # 1. 在 dist-packages 目录下打 patch (3 段: vllm triton_w4a16 + aiter kernel + configs)
    #    configs 段是新增文件, 用本模型的 config; 先把本模型 config 段从 patch 里分离
    #    aiter.patch 的 3/3 段已含 gemma4 的 10 个 config, 直接打即可
    info "打 patch: $PATCH_FILE"
    (cd "$DIST_DIR" && patch -p0 --no-backup-if-mismatch < "$PATCH_FILE") 2>&1 | sed 's/^/  /'
    info "patch 已应用 (triton_w4a16 + aiter kernel + configs)"

    # 2. 若模型不是 gemma4, 覆盖 config 为该模型的 (patch 里固化的是 gemma4 config)
    if [ "$MODEL" != "gemma4" ] && [ -d "$CONFIG_DIR" ]; then
        mkdir -p "$AITER_CFG_DIR"
        cp "$CONFIG_DIR"/*.json "$AITER_CFG_DIR/" 2>/dev/null || true
        info "已覆盖为 $MODEL 的 config ($(ls "$CONFIG_DIR"/*.json 2>/dev/null | wc -l) 个)"
    fi

    # 3. 清缓存
    clear_compile_cache

    # 4. 提示
    echo
    info "==== 安装完成 (模型: $MODEL) ===="
    echo
    echo "patch 默认启用 aiter w4a16 kernel, 直接用你原来的 vllm serve 命令启动即可。"
    echo "  (AITER_ROOT 默认指向 $AITER_PKG_DIR, 无需设置)"
    echo
    echo "想跑 baseline 对比时, 设环境变量再启动:"
    echo -e "  ${YELLOW}VLLM_AITER_W4A16_PATCH=0${NC}  vllm serve ...   # 关掉 aiter, 走 vllm 原生 triton"
    echo
    echo "回退:  ./patch.sh revert"
}

# ---------- revert ----------
do_revert() {
    info "==== 回退 (patch -R) ===="

    local v; v=$(which_version)
    if [ "$v" = "original" ]; then
        warn "当前未打 patch, 无需回退"
        return 0
    fi
    if [ "$v" = "missing" ]; then
        err "vllm 文件不存在: $VLLM_FILE"; exit 1
    fi

    # 在 dist-packages 目录下反向打 patch (恢复原始文件 + 删除新增的 configs)
    info "反向打 patch: $PATCH_FILE"
    (cd "$DIST_DIR" && patch -p0 -R --no-backup-if-mismatch < "$PATCH_FILE") 2>&1 | sed 's/^/  /'
    info "已回退到 vllm 原始状态"

    clear_compile_cache

    echo
    info "==== 回退完成 ===="
    echo
    echo "已恢复 vllm 原始 triton_w4a16.py + aiter 原始 kernel, 删除新增 configs。"
    echo "直接启动 vllm 即走 baseline。"
}

# ---------- status ----------
do_status() {
    echo "==== 当前状态 ===="
    local v; v=$(which_version)
    case "$v" in
        patch)    info "vllm triton_w4a16.py : ${GREEN}aiter patch 已打${NC}" ;;
        original) warn "vllm triton_w4a16.py : 原始版 (未打 patch)" ;;
        missing)  err  "vllm triton_w4a16.py : 不存在 ($VLLM_FILE)" ;;
    esac

    # aiter kernel 是否已改 (patch 版: @triton.jit 而非 @triton.utils.jit)
    if [ -f "$AITER_KERNEL_DST" ]; then
        if grep -q "@triton.utils.jit" "$AITER_KERNEL_DST" 2>/dev/null; then
            warn "aiter kernel  : 仍是原版 (未打 patch)"
        else
            info "aiter kernel  : 已改为 triton3.5 兼容版"
        fi
    else
        warn "aiter kernel  : aiter 包未安装 ($AITER_PKG_DIR)"
    fi

    # config 计数
    if [ -d "$CONFIG_DIR" ]; then
        local cfg_n
        cfg_n=$(ls "$AITER_CFG_DIR/" 2>/dev/null | grep -Ff <(ls "$CONFIG_DIR") | wc -l)
        if [ "$cfg_n" -gt 0 ]; then
            info "$MODEL config   : $cfg_n / $(ls "$CONFIG_DIR"/*.json 2>/dev/null | wc -l) 个已放置"
        else
            warn "$MODEL config   : 未放置"
        fi
    fi

    echo
    echo "环境变量 (当前 shell):"
    echo "  VLLM_AITER_W4A16_PATCH = ${VLLM_AITER_W4A16_PATCH:-(未设置)}"
    echo "  AITER_ROOT             = ${AITER_ROOT:-(未设置)}"
}

# ---------- models ----------
do_models() {
    echo "可选模型 (models/ 下):"
    if [ ! -d "$MODELS_DIR" ]; then echo "  (无)"; return; fi
    for d in "$MODELS_DIR"/*/; do
        [ -d "$d" ] || continue
        m=$(basename "$d")
        n=$(ls "$d/configs/awq_w4a16"/*.json 2>/dev/null | wc -l)
        echo "  $m  ($n 个 config)"
    done
}

# ---------- main ----------
case "$ACTION" in
    install) do_install ;;
    revert)  do_revert ;;
    status)  do_status ;;
    models)  do_models ;;
    *)
        echo "用法: $0 {install|revert|status|models} [model]"
        echo
        echo "  install [model]  安装 aiter patch (默认 model=$DEFAULT_MODEL)"
        echo "  revert           回退 (patch -R)"
        echo "  status [model]   查看当前安装状态 (默认 model=$DEFAULT_MODEL)"
        echo "  models           列出可选的模型"
        exit 1
        ;;
esac
