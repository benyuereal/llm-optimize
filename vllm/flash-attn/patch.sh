#!/bin/bash
# ============================================================
# flash attention fp8 KV + head_dim=512 patch · 一键安装 / 回退脚本
#
# 让 gemma-4 MTP draft 模型的 full_attention 层 (head_dim=512) 走 flash
# mixed kernel, 替代慢的 aiter 2D kernel。TPOT -40% (1.68x 加速), 精度无损。
#
# 用法:
#   ./patch.sh install   # 安装: 卸载旧 flash_attn, 装本目录 whl, 验证
#   ./patch.sh revert    # 回退: 卸载本 whl (需自行重装官方版)
#   ./patch.sh status    # 查看当前 flash_attn 安装状态
#
# 注: 本 patch 是替换 flash_attn python 包 (pip whl), 不改 vllm 源码。
#     vllm 侧只需启动时设置环境变量 (见 models/gemma4/start_tp4_flash_e5m2.sh)。
# ============================================================
set -e

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHL_DIR="$SCRIPT_DIR/dist"
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
    echo "  install  安装本目录的 flash_attn whl"
    echo "  revert   卸载本 whl (需自行重装官方版)"
    echo "  status   查看当前安装状态"
    exit 1
}

[ -z "$ACTION" ] && usage

case "$ACTION" in
install)
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
    info "安装 flash_attn: $(basename "$WHL")"
    info "卸载旧版本..."
    pip3 uninstall -y flash_attn >/dev/null 2>&1 || true
    info "安装新 whl..."
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
    echo ""
    info "安装完成! 启动服务请用:"
    echo "    bash $SCRIPT_DIR/models/gemma4/start_tp4_flash_e5m2.sh"
    echo ""
    info "回退: $0 revert"
    ;;

revert)
    info "卸载本 flash_attn whl..."
    pip3 uninstall -y flash_attn
    warn "已卸载。如需恢复官方版, 请自行 pip3 install flash-attn 或重装 DCU 定制版。"
    ;;

status)
    info "当前 flash_attn 状态:"
    VER=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    PATH_INFO=$(python3 -c "import flash_attn; print(flash_attn.__file__)" 2>/dev/null)
    if [ -n "$VER" ]; then
        echo "  版本: $VER"
        echo "  路径: $PATH_INFO"
        if [ -n "$WHL" ]; then
            WHL_VER=$(echo "$WHL" | grep -oE "flash_attn-[0-9.]+\+[a-z0-9.]+")
            case "$PATH_INFO" in
                *dist-packages/flash_attn*) echo "  本包 whl: $(basename "$WHL")" ;;
            esac
        fi
    else
        echo "  未安装 flash_attn"
    fi
    ;;

*)
    usage
    ;;
esac
