#!/bin/bash
# ============================================================
# ASR speech_to_text.py patch · 一键安装 / 回退脚本
#
# 对 vllm 的 speech_to_text.py 做两处改动:
#   1. 音频解码 (librosa.load) 用 asyncio.to_thread 移出 event loop
#      —— 避免阻塞单线程 event loop, 与 --api-server-count 多进程配合
#   2. 加 profile 埋点 (受 VLLM_ASR_PROFILE=1 控制, 默认关, 零开销)
#      —— 写 /tmp/asr_route_prof.log, 量 preprocess / first_output 耗时
#
# 注意: 本 patch 只是 entrypoints 层的辅助改动。
#       本项目的主优化是启动参数 --api-server-count (多 API server 进程,
#       共享单引擎, SO_REUSEPORT 单端口), 那个零源码改动, 见 deploy/。
#
# 用法:
#   ./patch.sh install    # 安装 patch
#   ./patch.sh revert     # 回退到 vllm 原始 speech_to_text.py
#   ./patch.sh status     # 查看当前安装状态
# ============================================================
set -e

# ---------- 路径配置 (按需修改) ----------
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/speech_to_text
VLLM_FILE="$VLLM_DIR/speech_to_text.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="$SCRIPT_DIR/speech_to_text.py.orig"        # vllm 原始版 (CRLF, 对齐海光定制版)
PATCHED="$SCRIPT_DIR/speech_to_text.py.patched"  # 改后版
PATCH="$SCRIPT_DIR/speech_to_text.py.patch"      # 标准 diff patch
# ----------------------------------------

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# 判断当前 vllm 装的是哪一版
which_version() {
    if [ ! -f "$VLLM_FILE" ]; then
        echo "missing"
    elif diff -q "$VLLM_FILE" "$PATCHED" >/dev/null 2>&1; then
        echo "patched"
    elif diff -q "$VLLM_FILE" "$ORIG" >/dev/null 2>&1; then
        echo "original"
    else
        echo "unknown"
    fi
}

# ---------- install ----------
do_install() {
    info "==== 安装 ASR speech_to_text patch ===="

    for f in "$PATCHED" "$ORIG"; do
        if [ ! -e "$f" ]; then err "找不到 $f"; exit 1; fi
    done
    if [ ! -d "$VLLM_DIR" ]; then err "vllm 目录不存在: $VLLM_DIR"; exit 1; fi

    # 1. 备份当前 vllm 原始文件 (只备份一次, 且不备份 patched 版)
    if [ ! -f "$VLLM_FILE.bak" ]; then
        if diff -q "$VLLM_FILE" "$PATCHED" >/dev/null 2>&1; then
            warn "当前已是 patched 版, 跳过备份 (.bak 用 repo 的 orig 版替代)"
        else
            cp "$VLLM_FILE" "$VLLM_FILE.bak"
            info "已备份原始文件 -> $VLLM_FILE.bak"
        fi
    fi

    # 2. 替换
    cp "$PATCHED" "$VLLM_FILE"
    info "已替换 speech_to_text.py -> patched 版"

    # 3. 清 torch.compile 缓存
    rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root 2>/dev/null || true

    echo
    info "==== 安装完成 ===="
    echo
    echo "patch 已生效。配合 --api-server-count 启动 (见 deploy/05_multi_api_server.sh)。"
    echo "profile 埋点默认关; 想看耗时分 解时设 VLLM_ASR_PROFILE=1 再启动。"
    echo
    echo "回退:  ./patch.sh revert"
}

# ---------- revert ----------
do_revert() {
    info "==== 回退到 vllm 原始 speech_to_text.py ===="

    restored=0
    if [ -f "$VLLM_FILE.bak" ] && ! diff -q "$VLLM_FILE.bak" "$PATCHED" >/dev/null 2>&1; then
        cp "$VLLM_FILE.bak" "$VLLM_FILE"
        info "已从 .bak 恢复原始 speech_to_text.py"
        restored=1
    fi
    if [ "$restored" -eq 0 ] && [ -f "$ORIG" ]; then
        cp "$ORIG" "$VLLM_FILE"
        info "已用 repo 的 orig 版恢复"
        restored=1
    fi
    if [ "$restored" -eq 0 ]; then
        err "找不到原始版文件, 无法回退"; exit 1
    fi

    rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root 2>/dev/null || true
    echo
    info "==== 回退完成 ===="
}

# ---------- status ----------
do_status() {
    echo "==== 当前状态 ===="
    local v; v=$(which_version)
    case "$v" in
        patched)  info "speech_to_text.py : ${GREEN}patched 版${NC} (音频解码 to_thread + profile 埋点)" ;;
        original) warn "speech_to_text.py : 原始版 (未打 patch)" ;;
        missing)  err  "speech_to_text.py : 不存在 ($VLLM_FILE)" ;;
        *)        warn "speech_to_text.py : 未知版本 (既非原始也非 patched, 可能被手动改过)" ;;
    esac

    if [ -f "$VLLM_FILE.bak" ]; then
        info "原始备份存在 : $VLLM_FILE.bak"
    else
        warn "原始备份不存在 (.bak)"
    fi

    echo
    echo "环境变量 (当前 shell):"
    echo "  VLLM_ASR_PROFILE = ${VLLM_ASR_PROFILE:-(未设置, profile 关)}"
}

# ---------- main ----------
ACTION="${1:-}"
case "$ACTION" in
    install) do_install ;;
    revert)  do_revert ;;
    status)  do_status ;;
    *)
        echo "用法: $0 {install|revert|status}"
        echo
        echo "  install  安装 patch (音频解码 to_thread + profile 埋点)"
        echo "  revert   回退到 vllm 原始 speech_to_text.py"
        echo "  status   查看当前安装状态"
        exit 1
        ;;
esac
