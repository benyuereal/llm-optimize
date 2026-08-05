#!/bin/bash
# ============================================================
# aiter w4a16 GEMM patch (通用) · 一键安装 / 回退脚本
#
# 用 aiter triton w4a16 kernel 替换 vllm 自带 triton w4a16 kernel。
# kernel 和 vllm 接入是通用的; 调优 config 按模型区分 (models/<model>/)。
#
# 用法:
#   ./patch.sh install [model]   # 安装 patch (默认 model=gemma4)
#   ./patch.sh revert            # 回退到 vllm 原始 triton_w4a16.py
#   ./patch.sh status [model]    # 查看当前安装状态 (默认 model=gemma4)
#   ./patch.sh models            # 列出可选的模型
#
# 示例:
#   ./patch.sh install           # 装 patch, 用 gemma4 的 config
#   ./patch.sh install gemma4    # 同上 (显式指定)
#   ./patch.sh status gemma4
# ============================================================
set -e

# ---------- 路径配置 (按需修改) ----------
# vllm 安装目录里的 triton_w4a16.py 所在路径
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/kernels/linear/mixed_precision

# aiter 包安装路径 (pip install aiter 装的位置; patch 会覆盖其中的 kernel)
# 注: 不需要完整 aiter 源码仓库, 用 pip 装的预编译 aiter 即可
AITER_PKG_DIR=/usr/local/lib/python3.10/dist-packages/aiter

# 本脚本所在目录 (patch 文件 + models/ 就在这里)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ----------------------------------------

DEFAULT_MODEL=gemma4
MODELS_DIR="$SCRIPT_DIR/models"

VLLM_FILE="$VLLM_DIR/triton_w4a16.py"
ORIG="$SCRIPT_DIR/triton_w4a16.py"          # vllm 原始版
PATCH="$SCRIPT_DIR/triton_w4a16.py.patch"   # aiter patch 版
AITER_KERNEL="$SCRIPT_DIR/aiter_gemm_a16w4.py"  # 改过 triton3.5 兼容的 kernel
AITER_KERNEL_DST="$AITER_PKG_DIR/ops/triton/gemm_a16w4.py"
AITER_CFG_DIR="$AITER_PKG_DIR/ops/triton/configs/gemm/awq_w4a16"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# 解析模型参数: $1=动作, $2=模型(可选)
ACTION="${1:-}"
MODEL="${2:-$DEFAULT_MODEL}"
CONFIG_DIR="$MODELS_DIR/$MODEL/configs/awq_w4a16"

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
    info "==== 安装 aiter w4a16 patch (模型: $MODEL) ===="

    # 0. 检查源文件
    for f in "$PATCH" "$AITER_KERNEL"; do
        if [ ! -e "$f" ]; then err "找不到 $f"; exit 1; fi
    done
    if [ ! -d "$VLLM_DIR" ]; then err "vllm 目录不存在: $VLLM_DIR"; exit 1; fi
    if [ ! -d "$CONFIG_DIR" ]; then
        err "找不到模型 config 目录: $CONFIG_DIR"
        echo "可用模型:"; do_models; exit 1
    fi
    if [ ! -d "$AITER_PKG_DIR" ]; then
        err "aiter 包不存在: $AITER_PKG_DIR"
        echo "请先 pip install aiter (DCU 定制版), 或修改脚本顶部的 AITER_PKG_DIR"; exit 1
    fi

    # 1. 备份当前 vllm 原始 triton_w4a16.py (只备份一次, 且不备份 patch 版)
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

    # 3. 覆盖 pip aiter 的 kernel (改成 triton 3.5 兼容版; 原版 @triton.utils.jit 会报错)
    if [ ! -f "$AITER_KERNEL_DST.bak" ]; then
        cp "$AITER_KERNEL_DST" "$AITER_KERNEL_DST.bak"
        info "已备份 aiter 原始 kernel -> $AITER_KERNEL_DST.bak"
    fi
    cp "$AITER_KERNEL" "$AITER_KERNEL_DST"
    info "已覆盖 aiter kernel -> $AITER_KERNEL_DST"

    # 4. 放置该模型的调优 config
    mkdir -p "$AITER_CFG_DIR"
    cp "$CONFIG_DIR"/*.json "$AITER_CFG_DIR/" 2>/dev/null || true
    info "已放置 $MODEL 的 aiter config ($(ls "$CONFIG_DIR"/*.json 2>/dev/null | wc -l) 个) -> $AITER_CFG_DIR"

    # 5. 清缓存
    clear_compile_cache

    # 6. 提示
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
    info "==== 回退到 vllm 原始 triton_w4a16.py ===="

    # 恢复 vllm triton_w4a16.py
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

    # 恢复 aiter 原始 kernel (可选; 不恢复也不影响 baseline, 因为 baseline 不走 aiter)
    if [ -f "$AITER_KERNEL_DST.bak" ]; then
        cp "$AITER_KERNEL_DST.bak" "$AITER_KERNEL_DST"
        info "已恢复 aiter 原始 kernel"
    fi

    clear_compile_cache

    echo
    info "==== 回退完成 ===="
    echo
    echo "已恢复 vllm 原始 triton_w4a16.py, 直接启动 vllm 即走 baseline。"
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

    # aiter kernel 是否已覆盖 (patch 版)
    if [ -f "$AITER_KERNEL_DST" ] && diff -q "$AITER_KERNEL_DST" "$AITER_KERNEL" >/dev/null 2>&1; then
        info "aiter kernel  : 已覆盖为 patch 版 ($AITER_KERNEL_DST)"
    elif [ -f "$AITER_KERNEL_DST" ]; then
        warn "aiter kernel  : 仍是原版 (未覆盖)"
    else
        warn "aiter kernel  : aiter 包未安装 ($AITER_PKG_DIR)"
    fi

    # config 计数
    local cfg_n
    if [ -d "$CONFIG_DIR" ]; then
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
        echo "  revert           回退到 vllm 原始 triton_w4a16.py"
        echo "  status [model]   查看当前安装状态 (默认 model=$DEFAULT_MODEL)"
        echo "  models           列出可选的模型"
        exit 1
        ;;
esac
