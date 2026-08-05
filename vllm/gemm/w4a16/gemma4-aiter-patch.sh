#!/bin/bash
# ============================================================
# gemma4-aiter-patch.sh — aiter w4a16 GEMM 加速 · 一键安装脚本
#
# 自包含脚本，无需外部依赖。执行后自动完成:
#   1. 备份 vllm 原始 triton_w4a16.py
#   2. 替换为 aiter patch 版 (支持 group_size=32, AWQ)
#   3. 覆盖 aiter 的 w4a16 kernel (triton 3.5 兼容版)
#   4. 放置 gemma-4-31B-it-AWQ-4bit 调优 config (10 个)
#   5. 清空 torch.compile 缓存
#   6. 可选: 生成生产环境优化版 start.sh
#
# 用法:
#   bash gemma4-aiter-patch.sh               # 安装 patch
#   bash gemma4-aiter-patch.sh --status      # 查看当前状态
#   bash gemma4-aiter-patch.sh --revert      # 回退到 vllm 原始版
#   bash gemma4-aiter-patch.sh --gen-start   # 仅生成 start.sh
# ============================================================
set -e

# ---------- 路径配置 ----------
VLLM_DIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/kernels/linear/mixed_precision
AITER_PKG_DIR=/usr/local/lib/python3.10/dist-packages/aiter
VLLM_FILE="$VLLM_DIR/triton_w4a16.py"
AITER_KERNEL_DST="$AITER_PKG_DIR/ops/triton/gemm_a16w4.py"
AITER_CFG_DIR="$AITER_PKG_DIR/ops/triton/configs/gemm/awq_w4a16"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    echo "用法: $0 [--status|--revert|--gen-start]"
    echo
    echo "  (无参数)    安装 aiter w4a16 patch"
    echo "  --status    查看当前安装状态"
    echo "  --revert    回退到 vllm 原始版"
    echo "  --gen-start 仅生成生产环境优化版 start.sh"
    exit 0
}

SELF="$0"
EXTRACT_DIR=$(mktemp -d /tmp/aiter_patch_XXXXXX)
trap "rm -rf '$EXTRACT_DIR'" EXIT

extract_archive() {
    mkdir -p "$EXTRACT_DIR"
    local line
    line=$(grep -n "^__ARCHIVE_START__$" "$SELF" | tail -1 | cut -d: -f1)
    if [ -z "$line" ]; then
        err "脚本内部错误: 找不到嵌入的归档数据"
        exit 1
    fi
    local b64_data
    b64_data=$(tail -n +$((line + 1)) "$SELF" | head -1)
    echo "$b64_data" | base64 -d | tar xzf - -C "$EXTRACT_DIR"
    info "已解压嵌入文件 ($(ls "$EXTRACT_DIR" | wc -l) 项)"
}

clear_compile_cache() {
    info "清空 torch.compile 缓存..."
    rm -rf /root/.cache/vllm/torch_compile_cache /tmp/torchinductor_root 2>/dev/null || true
    rm -rf ~/.cache/vllm/torch_compile_cache 2>/dev/null || true
}

check_env() {
    if [ ! -d "$VLLM_DIR" ]; then
        err "vllm 目录不存在: $VLLM_DIR"
        echo "请确认 vllm DCU 定制版已安装 (pip show vllm)"
        exit 1
    fi
    if [ ! -d "$AITER_PKG_DIR" ]; then
        err "aiter 包不存在: $AITER_PKG_DIR"
        echo "请先 pip install aiter (DCU 定制版)"
        exit 1
    fi
    info "环境检查通过"
    info "  vllm: $(pip show vllm 2>/dev/null | grep Version | head -1)"
    info "  aiter: $(pip show aiter 2>/dev/null | grep Version | head -1)"
}

which_version() {
    local patch_file="$EXTRACT_DIR/triton_w4a16.py"
    if [ ! -f "$VLLM_FILE" ]; then
        echo "missing"
    elif [ -f "$patch_file" ] && diff -q "$VLLM_FILE" "$patch_file" >/dev/null 2>&1; then
        echo "patch"
    elif [ -f "$VLLM_FILE.bak" ]; then
        if diff -q "$VLLM_FILE" "$VLLM_FILE.bak" >/dev/null 2>&1; then
            echo "original"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

do_install() {
    echo "========================================="
    info "aiter w4a16 GEMM 加速 · 一键安装"
    echo "========================================="
    echo
    check_env
    extract_archive

    local patch_file="$EXTRACT_DIR/triton_w4a16.py"
    local kernel_file="$EXTRACT_DIR/aiter_gemm_a16w4.py"
    local config_dir="$EXTRACT_DIR/configs"

    # 1. 备份
    if [ ! -f "$VLLM_FILE.bak" ]; then
        cp "$VLLM_FILE" "$VLLM_FILE.bak"
        info "已备份原始文件 -> $VLLM_FILE.bak"
    fi

    # 2. 替换 vllm triton_w4a16.py
    cp "$patch_file" "$VLLM_FILE"
    info "已替换 vllm triton_w4a16.py -> aiter patch 版"

    # 3. 覆盖 aiter kernel
    if [ ! -f "$AITER_KERNEL_DST.bak" ]; then
        cp "$AITER_KERNEL_DST" "$AITER_KERNEL_DST.bak"
        info "已备份 aiter 原始 kernel -> $AITER_KERNEL_DST.bak"
    fi
    cp "$kernel_file" "$AITER_KERNEL_DST"
    info "已覆盖 aiter kernel -> $AITER_KERNEL_DST"

    # 4. 放置 config
    mkdir -p "$AITER_CFG_DIR"
    local cfg_count=0
    if [ -d "$config_dir" ]; then
        cfg_count=$(ls "$config_dir"/*.json 2>/dev/null | wc -l)
        cp "$config_dir"/*.json "$AITER_CFG_DIR/" 2>/dev/null || true
        info "已放置 $cfg_count 个调优 config -> $AITER_CFG_DIR"
    fi

    clear_compile_cache

    echo
    echo "========================================="
    info "安装完成!"
    echo "========================================="
    echo
    echo "  aiter w4a16 kernel 已启用, 直接启动 vllm 即可生效。"
    echo "  (环境变量 VLLM_AITER_W4A16_PATCH=1 默认开启)"
    echo
    echo "  想跑 baseline 对比:"
    echo "    VLLM_AITER_W4A16_PATCH=0 vllm serve ..."
    echo
    echo "  回退命令:  $0 --revert"
    echo

    read -r -p "是否生成生产环境优化版 start.sh? [Y/n] " yn
    yn=${yn:-Y}
    if [[ "$yn" =~ ^[Yy] ]]; then
        gen_start_script
    fi
}

do_revert() {
    echo "========================================="
    info "回退到 vllm 原始版"
    echo "========================================="
    echo
    restored=0
    if [ -f "$VLLM_FILE.bak" ]; then
        cp "$VLLM_FILE.bak" "$VLLM_FILE"
        info "已从 .bak 恢复原始 triton_w4a16.py"
        restored=1
    fi
    if [ "$restored" -eq 0 ]; then
        err "找不到原始备份文件 $VLLM_FILE.bak, 无法回退"
        exit 1
    fi
    if [ -f "$AITER_KERNEL_DST.bak" ]; then
        cp "$AITER_KERNEL_DST.bak" "$AITER_KERNEL_DST"
        info "已恢复 aiter 原始 kernel"
    fi
    clear_compile_cache
    echo
    info "回退完成, 直接启动 vllm 即走 baseline。"
}

do_status() {
    extract_archive
    echo "==== 当前状态 ===="
    local v
    v=$(which_version)
    case "$v" in
        patch)    info "vllm triton_w4a16.py : ${GREEN}aiter patch 版${NC}" ;;
        original) warn "vllm triton_w4a16.py : 原始版 (未打 patch)" ;;
        missing)  err  "vllm triton_w4a16.py : 不存在" ;;
        *)        warn "vllm triton_w4a16.py : 未知版本" ;;
    esac
    [ -f "$VLLM_FILE.bak" ] && info "原始备份存在" || warn "原始备份不存在"
    [ -f "$AITER_KERNEL_DST.bak" ] && info "aiter kernel 备份存在" || warn "aiter kernel 备份不存在"
    local cfg_n=0
    [ -d "$AITER_CFG_DIR" ] && cfg_n=$(ls "$AITER_CFG_DIR"/*.json 2>/dev/null | wc -l)
    info "aiter config 文件: $cfg_n 个"
}

gen_start_script() {
    local output="${1:-./start.sh}"
    local start_file="$EXTRACT_DIR/start.sh"
    if [ -f "$start_file" ]; then
        cp "$start_file" "$output"
        chmod +x "$output"
        info "已生成 start.sh -> $output"
        echo
        echo "使用方法: bash $output"
        echo
        echo "注意: 请根据实际环境修改模型路径"
    else
        err "嵌入的 start.sh 不存在"
        exit 1
    fi
}

case "${1:-}" in
    --help|-h) usage ;;
    --status)  do_status ;;
    --revert)  do_revert ;;
    --gen-start) extract_archive; gen_start_script "${2:-./start.sh}" ;;
    "")
        extract_archive
        v=$(which_version)
        if [ "$v" = "patch" ]; then
            warn "检测到 aiter patch 已安装, 跳过安装"
            read -r -p "是否重新安装? [y/N] " yn
            yn=${yn:-N}
            [[ "$yn" =~ ^[Yy] ]] && do_install
        else
            do_install
        fi
        ;;
    *)
        echo "未知参数: $1"
        usage
        ;;
esac
exit 0
__ARCHIVE_START__
H4sIAAAAAAAAA+Q7a2/bVpb5rF9xV8EOyIamLMdOvJqoqGO7rte24trOZLeGQVDSlcyaIhmS8kPZAOn2kW47fU2fmNk2ndkGyMximhbd7XbaSftfZiI//sWccx/kpSjbSYECu1gBiUXec84973PuQ2bpzE/+GYXPxYkJ9hc+g3/Z9/LE2PnxiQvny+Pn4f3Fi+WLZ8jET8/amTPdKLZDQs6Evh+fBHfa+P/Rj1mKQyf2PWtn3C5fMIO9n2AONPCF8fFj7D9xcQLGMvYvj01MlM+Q0Z+Al9zn/7n9z5LV5Zl/Gll0GtSL6Mh8k3qx03JoWCFTgd3YpCNj5mhBQD3tuHTaD/ZCp70Zr9HduEKSR9LwPXClejf2w4jEPok3KdleXFwiQeg/TxtxoVgsFtaYs43U7Yg2ybXxqfIFMje7tES2aOhRl7T8kKxcme6Qpfnzo6NmoTDfCVzaAZ4i0uoijuPF4yM7lM3YpNe7NrDbs2PH98g50gokPccjNokcr+1SQdsodPGZzC2vPUsiwERBbZeAkFv4XptktMm27XZpRAIa4vP5MYNEm04L5l8fNcYN0zSNsckN3Swsu912hCBc0qXlRcejdrjABYmoCyIjV9FeFNMOsT3gHciGtEXDEATxt2lYWLJD1/FUzNLsruvaHTtDDcigVkAf17jkrr3nd2NCdwOYBYjV94AJIC/UqAV+FI+A3hs0iiyurciyWzENLde3myCvXikQcp0PVcj6gkFqpdLkBuFCE/KXW++R0N+JqgtEc7ygG+sGWNiNqghGtBoTBlRHmzoQihq2S6MKAUKl0hzQAkJojFK9Vb6AE/Vo6Kvjmbk0P0BV2e7PSc33KPOCaK/ToeBPDdJFs9Qn9UJhepM2tgIfnqUCWqHfAbY6oNYIvMOKwYnB/aydGiQz0gipHVMpPwrMv1qccWAHPus1gywMsNONQWCr6XSqowZh0rOHsiFE5k96SpDJz+lJgnNCBeQkggoJ1JDFhANzoH4SKpytASIKI6M6C62C0wn8MCZ+JL+B68mv8V5Akwf+x3XqZjd2XPnW9dttcIyEDsRxY7NQYCredt2O2fGb1LXoLm1gjJtgAhpGphqCjF4k6JOQBq7doKDt0AZbgrsfSysBkbiXIUFg8liWAwZGZKcbC3IWdwBLIQmTxeA5nWT+RhcCzQNriwEFFu1lhxZqRUIrryIFUlTHjGCxaxD+XqjHHIh+AZd9ayTPi6i5ad9rOaDttZX5tSs1i+VCa/Xq8vKVlbXZGWtu5crVZWt1/rnZVVIl6yPge5iLLowbpDwGvjE2cWHjONxnr07V1qy1f17muAX0SlU8U4SUQchZJdBYZtTqjh1VJ/VjkBiKneLsOPEmpiHXaTgxYVFe2CgUoGCMwIfYDtq0TTsdCwJyZ5z0X/v06Nadw/+53//+JaJdvlYGV26HfjewIqdHq+fHdIYI+Ieff3/4+W8r5MQsRrSjjz85+PMHh/ff4TR1cvS7l/qfvXTw9iv7H794dPsNqArwZv/OXaxPoHADSO9/8tLRrU8ffnPr4Td/OPjunf7nv3747Sv933wLXwQUOfj1S5Bjul5jk/RvP9h//wui2a7r71iOZ7VDO9jUSf+PH/Vf/YKwMPjrrX8FunYQuHuSUXL4xYsH7907vH83S8gg/ft/4sFlYuaCkkqae57d8Un/rdf7dx8Qbf/DTwmbhNQhg23pSN2aml+bXbGWrsyASTFLwptrz1rztV+giUGL4BxgnQvgHuApBpkwyMUNNBZadeRJgCX9Vz8Ehey/+SuY4f5bk/AIObXQpC3C1GkxY1nMWBCeGqsQhEBqAaMdPniQNyaPgaToHHz3/uEPtwWYBZpyYssi/zi/hgJghkJybdevQ9VNxWFvnZbyBkuL58dMSs4DfkIad0NvEJHzjL0ZqMGPTOptOyEkojaNtSIHXblyZa1okGKpG4Ul1weXLkHqKwV78abvnTfLo6Wmg9USEqrdplGRu36w1QaK3POX/GbXpWvwXSuyCVMY04J8FG+CnGAFhZlzMB8H3VBAIyjXDDSbgU32HjOJ1QJ3sJBJTKdaIryY1iDDpihJXUP3XjQSnKhb7zDGrQhyTmMzIRtVj+E0lQprARixKosC6nORvctqACqMySeJ1sXABmABCTbMCnm3jt2YGDb9IEJjJA8iwQ57x6tJcSP1gc4wk8AEegpxkkGK8D9Am6IwaUWT+UVR30iVpggEoChMp8BGkzjRPKhFespU9Oj2ZJgZaQYQhcEYKhLRIn0oax5nLBkykTFQHhZUiwNpHX0wdjgCF+IYbUNKCqniRGS4xwGaWDeWGFoJ0dD99EeYgjmj47X8HzFPgptO9hj6H4gnlbU0rSlcncZTipSy8xhWzUfPMRyl1j7J0pkSwcGDEIq2VlxX1/gbQ6syJPj9V9/uv3YHQgKVVkXmohimCjn1xIdEyWgH8XUr9i1757rop7W6dZ29NkS/zx7SMsL6C1hp8OZfdLW1EbGKICNPEqxSgwBJ1dJlCcG1yiSIKKczo007oGyohsVxkjxBJrm2+MqtKsotNI5em2rQ9zTpNix4qwkF/gzvMbtUOTSbXwda44zWDpDREhHNrhfBEpL2qDZS1smTT4q5dPIzMrr7tA45hnGlIbMir0L/ijQyvNRIqUSOZUhfrxisDG4kIsnPOSETX/Foshk4llLK0UiZ87ODpgOGdnASZG7DxCW80+763UjjMEiFQ/HZom5H0xiiue3QHSYdcD8pxSaXLglFGBlu1Q8uWcaGqTo//cIczKz4Erf0+ijP1z1mEXX4kY0yl1ilJ+TrnaAFMUVOET1FEXM/QhOPpwwRgMIkhsKVDMmOvUVFF8dbTqvlabgaNchcGoYPv/l2/5e3odftP/jdwWtfP2IznOaMbDuLvSkShlZXqAbSXmiHe2ajG8WQ8PyA7H91r//KLyG9PPzmjaNbb+1/9Cbxg2F9sMYb4XOO1+w2YEgXLbHSP0LKAyMMdIGcTwx0GMZehWc6lptYmhMAiTIYluhbOVICyt/yrrQT5IZFJsUhpS/gSseXfLPEkFsrhtgakeZS+oYlLBQInPVqNnFLilTJOE+dRrGFSaTjeJqAMLfoHriIAeLsVV27U2/aZLdC7Hqk7ZIRsqTrGRLCpFXwvEYsaaxzwhspKHUjWhmOeCPnz8XLi1emF9hi1VoqVkgZViHquxq+w0Wr+nIBXsKKNk9sdfqZ2Zmri7MrAADrmuLq8uL8GkLDwqY4Y60+M7U8C0/aEsYwezWDS10+b57czLK1Nr84uyqozUzV5hbna3OZl7WrS9b0Vf6UpyBGrdW1ldmppYUToNiiHcloULnOkTnQP+QhyPCQHYpXV2etFZBsGmSfXanNLgLc0zboOUvsZvLUtJhjYBcOCxRhq/VEBYq1BOT6yBh2CkvJ+xwK2t1KyyV+RFp5BN815BJa/IWwDfa452HI6DwezqbhzqOb8EQAIU+0JAEYYs1bkqFOYAgWlNCWk/4rLx+9eI9L5wcWgKEOWsW0YbFu1G5aNxbg39xNnhbicC9112weggB1PKq1eIdVqdwQNG8WT8zMYFJtjRVXItQinhLtiGepJDnM4xzbGf6mqB/DGFP5MLZIcfrqzFRR15S8khKRCiYtSPc8R2mHn/9ebiIc3Ll79MK7h9+/27/9LWGmLrHqYpCjF37ov/xGZmchIfpUlrWQtsHlYGqcYxiPiu9h/mNgj5z5FLfj09JOEO9pWjYbJpSSN/opBiOijvKJTSG2aIg4cdENMTp0t0GDmMyyP7A8SBkM7CgS7geeB/XEjuNQNG7YoTN1GNI59Vwh4BXy0RUilOEHp6EU1E4gM5nsAfi7AJa4Yt+M7U9p7H8Der7rjGX8FiXfekKOpEXYv/fb/ievi1XBh18T7eSjhHTPTbQJfOut/6evD/787uA2GU4xsGkHrcBb9w9v/+HgvS8fPvgBOon+a/f2b71wdOu7w+/fydf+oZtVoqW9rthrQGY9iZ/0tEOsMdjKJFmJCFLRMFJRjpQ878C9foHZC4ahSi0PYJ7Kx3Wracc2NuoWdPP4FdqDTTti1OEdpAt8W9RZ0UaohI0UsxcMQe0FOdxeoPb9Rrb1Hbbmk/wZ6YSccRuw7Z7SPoXUD2E5adleE74jujZkGl0WkTfv7X/wRyIatmRxZw+sARbUd2VcJo2x97hwWMDKa/eyCEljjPKc2DALcLCeKWB4U9nAU9FMH1S0r0MtBzbUnt1Is9JZeSY0xiw9qSD2ELF3AmLiJTlcnhqw2QC3HGLdKGfcKLuqSLNpkUsPtBI18MGbbCsf08HHHybp4PD1u/2372PDH9hxY9MgB//5ev+NrzK7xFr/3++pId7/7AMdKB1999Hh558JbO3wP16GtQanQvpffHnw3p3991/Vf457/+QXi4tLos3n5xvLU2vTz1RHMVf0X/5q/81/O/zvL9gZDem/eQdQ5ZEMKGFwI3g4LSy0ZdDO31VJcbRYyfcRJ2SawdpB7IgoDTPfd2kN33jhiup/9uXhf92F1PebT45u3cIUyWThCBVyY0h/ktuXKRSeEptFzzsxy//qfOp6RhNhtYwHjDTk1c22gjhkJzrr0EwvqGe2xG7EzjbfLhYZIYEdPCvmmYCMj9SdmMjTD35CjCfjYhhWuumhEo0UcvkTYwHDF+gsMwxCD5mfARKt67F7Ajub1CPPTK1azy1XWZfNJ29kRM5Oyg9YhaJmnA50b4n4S9wWNUPsQAmoVdB3UzAase+W3TEyj1uZx/rAo8fP4NgTbpPXie36XltVHDtwB+2peI3sJA1PMvSseiEiOVXlDKbnbBL62iaFicLsAR5hh9ARsUOKh2/bMEGTwXNtVkjsYhqB9nA3CCWl5xARjw7JoPbRDZgBiEbNtkkm2alAcqiP6M8tW5fnp1aHU77s+o0tglyDceOuB9SRALsgQnbsbdqCMI+rF8Y5Lb7CXBpGiw/Vjh9aGBxKOyL29+lu9tZKhUyDF9VwVTUFXyCAnpJ3UrTLOm5kbvCG7TIqARZBeO8DVDTYf5x4MUUmFQplR3H4CBSJnOIBurNNoWsQF1fsmGAc+q1WROWtlUmjPGbA6nhs1Bgbx/srJudrhjNbgdrQwp5Fw/PNeByWregIuPfJAlFwIKJpLeziVQfmKiAW249u8psYSayy6y5dj3twFp35QoWjV6XpmSgxsDLMSdIjaj1jjsBpss0QsBn4aRtvBThNbVRPBr3cYDlZpK74O6WG7yZ6wvnYFZrYEQLjCKPP53lCuhZuvbpy/3bUkK/1FMkTSF6CVBuOVEvYqZd4+sKgE47CjtZrRN5VYWmA2x/47nY8fkdpAW/ppFPX1bk1OTnuMevHsMAHJR+DTih20FEp4LoNYDFK4kigEOlj4m4U3vugkIDApVGpdJuGezz3AItgWemppgzwruM2SXlGzGWVmwjnUq8N8icxy2Umz2NfHXFQoj1P/h4Fe4KMm8p+vwUa4ZZPheVQLO9Pbqiw5aY11uTQ9RBcuWFHMbS4WkpqHTfgDVKBBWlGoUhUz5LidOQuszpDimvoOmNDPG5ILcjJ8eqcGFxIjLTBVCmuvfG4wqs66eTHS1Bu5gVICeuJ5VdZyZV6FmFRIa2u64LJdpwmWEPD21ksJTDf4wWTeLQLKVjx/+ixA8BuNLqdrmvjRhCThEWDJthdSsGTrXLXbEHmwZ1yTgH1s2XhXc4Y62gyDyb0prOtJSIv6MrCm7G7BVNKVMnvwnB+F5TTWzvaYqiCxiWyUFD2aNj1l0Uwh9pJkamKtG0iE1QNduNForKmDO3JvgAbPA+pJ0FJpyFHtxITK6NbWVZxHagN0rpElvDAgsuSEEl54cZgp7mcL4PBVjlFg/jYPlRHzVF9uOyihZFN4eVK3rWVOpjRQ13qoa7qYWuIHupbcrTuDVFE3csqoo5rUi5wQutnQjUqgUtE5kaFJS5PqpV6Tiv1RCt5nVxlkcvvmUqd/OWVXw2L94wuzuJ8rHF3KXQ82q5BdnXS9Lt1DFnsFV3MHNAk4k3QBBAvbig0VtHDWVrGWj3UEBAvmyGlKQkg34b2QqGS59XAdg/qltKhQB3FSxZOw8bbqZNQPzoy4TNFcg0qEknVGomS9ZOgAezRx8/CMg13lmOmqYYfhhS+8+WK59RBh7xQIfsipfEsmJlBq+cOFHP2nfY7gC1SJKvoovBhRXK8Jt3N2rUNPcku0s6lH3ZqkDbsw6NL3rpNaglzpvoplSTDASNh8WyPm1aMoSdITUZU5A2Aoo/LpAd5HoIkCxApwZGu9PDijTKTiJeUooyZsjmqD6WXrWxs4FHK2qDGSuyyMJqIt1sZXTgtucTJrLvPZlIZxwN7pmpPM1gGj13ozWlWS7sxkXHySKqO616SiLKA+VyUdt/n1MmFshPKSn7KiimyU+o14Cx8aRLhEUwaX+d4+xXYMbwZ4D8Xi70ksnu5yD4GA0B/HIzWS2MUGh8ZpkNIZfypd4orSdz8oaighk2Slsc05PJG6Vr46X7OO2fkTxko7ucNLsAGKyNfrNURTDdBAJufdSTwUY7+lGywBmipjdc5JkvTjzXbYJMYmA6tYxouQXgVFysybU7n2xs14TSwq0knRM7ZfoyJM5iU/9DDiveUrRoM/8ZpjVAjaYSG1f8G91BWnBsnNEHaIIVL8o4IyM7WZJxbPAdVC35D3qLN7b3xTTe7Ik65+HGgut+m/kSCbboPA81uFihbaEOB1eOIlLT8/YWKQP6FcFGHHkaA7UN+zxjx00pUwWG+adILLNzxYW9Ar/w6OdsDUhfvbCuIz49tASPJTkdVXk7eZ1E2SMTZSKaFEpsZU6F6XcKupFE6lXTgYDaw0q7Qv0GYS+PWX0jq4rvSpaA9xGdZbWQTKtI0RuYAJ10+6ymx5BcySIyGI7wvECUuMZvBUy0LOdwpslMCyU9oMgSUqiT37U6z5eAn+2ubzC+qNDBEJO0MtKKY2k1FKNUtMjuPQjoYIBosh313W24RjZSxwCxgnwqyuzRUyCUOBZ9puSHEREs3FbOeRDQIPLvrxgPbisIlVthBqeIVV3ieUrzgWJ1LZ7SjiEJnZptOZGWPaYp5v+p0oe2qU5ICZoiARw0hcy3jVKeQEGfbeSpi/T4MXexg88OygduSyJJyejZZGORX3EGBvInezpsXKJOJSltFgOIXDUjHiTp4mFMhNxLUm2Q7ItqNhZsGucHRb+pcIn2IXMps2Q6YX/ZRpxWxk5tZpSQnz9JirGS5gN4v9az8jw8EkxzkRCbzCuLcCuo5blWS4i4L5yqpmJkrEuLSE6/J9sAVBzu53sCQN+2IH0XnJRMVfJlv78M6v9ntdPZ4BqFh/vCEaPyHm7Cqu951Qtxmxh9KOtAWx+J+cNp9phO2JA/sABJcoiCVPfjbLHTo0G90NGVzZshPu0yEkT+vglrbbu2Wd9NmB4/+xEtNH+zi15Kzg5WZ2hQ5b04QDUHLE2WDjI+S6asR/sYKt7poerAQKflJTLFELlUBsJLLpoOtT7LFQ6rsx1v8B1wZNOpKghfGH48g/hIM/w2sOvI96mmE2M28AdbyVFT98bMXVN4/jKNYo+NceRfGf0LlDZX1f4fyhAYLeRLHCfyYwh4v6GMKmRfwkYUTKWNtU/6gmp38RCJxi+ML9I7kd9eiQ4F8IkkmZytn+VYyXxSzn1r/rb3vb27jOBL13/wUa9CJAAmACJCiGdpwPVmSbRVFyCfRSaVoFmoJgCJCEICwgChSYZWdi8+5vJPte+dLfD7fxU6cO9f9kPPe5e6c2HE+zBMp6a/3Fd709PzomZ1dgCQgk8qOqyxiZ6Znpqenp7tnuoexHb6TZvLe5TVV4QWDuYI/MpxvgRmn5zeaAlR7Db9wTzqlR+H5CWRsdeFElQM6FagbVQEr3uo1twUQMAv1O9w8BnVQocl7F5r+Zkf3v036A7826vWOp/kpKmt63NClvGR5pObz6sBxwpwHwLpt9bnBtBjuN4An/dyareZrMSNdUTGjHDKBRN0IWAa4K3qb8vVtA8aoiRsZ4ksLaeo8Xn2q6j8X9Z9l/ecCcanJo1qWnso4PhYyRi9cZclnWrrqKlt1lSTupvaJOCSx7eHGpT8LTb4kZNSsNXGLJbmQrIxySS4se6pL0l5AxBBxs6/KNMpqkwkdHm67XBWivsRp07XYOqceNshCGhk5nNW26ltcFOf1r/c7sMEGqHHBxuv3TKN1OnQ0m0XZm/gGi/trsAo5nllfDF9xT8i9wmefH332un4r6LQDqS2wpYhykHLxz8ni1VAwABGJAUd5ih6zYwEt1+MoI/ylBztVY/3/wWeIjXW9jUfcYAMA34PNRqtS9Tv+aqPZYDJbtYm3gcGzn5hlhEC1zrDPz91UBRwZk556Qb25pmoI0piKabzqo2MCt6NAu1mvOu/yOkf9u89KLq+2280smEqEOWDF2LxAXkxHy2qgV7pyq/2an864b/zilXcvFUHZWsKEO9CeINWUIeNV8yJmAlfZoI8NtmaaQd45W85emOI5JMdVfEhrKa7YcrdT747R8i5vOsDFUq8956UctVU2k/Uju2hf9cpYw/Wr5ljFVWRhscgKFUFaMOLxfrXV3JZ2j7OyinFIye8gyG4TzIO6WIVQDb0GFKzI2BdCe6RdLsPBPFysmzoq9oXGvlb3WU24AgSam9Jy2ZbXCBpwgrO6zRoMT0DKDuuC9hIV3SUzAPOwDXAp5WjjoKYCcQ+WRy/goDNSM1NIh9E4xhKxZKKHwF2iqnkiUZDBmZ2/EUjiGhyQwqgIuwer/DQ0BHZwkyyIU5ObLA9ACi9ro9KdG8GBV9/gce06YTAWxC9BLDDacw9xN4aKFuIWDcFOfW2twqfrBtebEaO5AmrOC3TeFtjaEqWjF5hY7Wupyy1j+dxZQLwZC4dY6+4gZHDwsHceuPeFJnPt7xDrF5CG/SuLV6il6bnVEq77fAsyjSxSfIHE9qlbYHEZduOXKonY6DUcXTgdGywobwYLMoh1qKhBZpygyGA9hTDc+OBB0XDDkCJjCIk+RocQIgSLmGzV6zXD+zAmVhToJkJuM3FFT+iiQkR5zvq8NK3ujiAlzO6h6hoRGYO8iPzFT6+WoCJILWR4amb5zYwFjkV5tC7PRxUKrGsazhhFcF1DT+KUMYmFDFtefpdfM2K6KUiwfo8ABIGW100vZOB+B95Jhg2DieSd9e2gwRDFL36gHI+g0+UM5MPnAr0BQlbDOjfoD0WxYObU0b4ovPOw3OWgGe5B/CUoY/XEfis6MwUGUtkxAxIxK2z5eIohUcyg0OomRKbYAda26ibOWlqTmNcdyi9BUBA9v2+8z9F9Crc+HsWO9ImOs9rr+00mN7EhwsIAGqiuw5x533vl/JLGDhXwPe8qA9lokau8nkIIjxu3wEOpzXkLOTvoXMaAI5akguJpNHA4ZQGnHAuHoZgtECFeez5uauiIk+UI+AEIVb7GnBsX17mGAbGOum0fTNh4HZGvD70OgW55AxzduHqwLeMCtEkCDW6jroFl6MKrr+UCbprucTsTv0HEjUY5/le1HfQIAcFehOArW5Wb6dvz4QBifMMJfbWNnLeFI01A6EScoam5w1M+sg2hhmxfp7jeq3e8wry3ykU94BoKIleKQ0t1ygAwBCuJXbVmd+BCrBibdkdSZ8pG0TLUZ/kQYmMrbzoLQ1qAfHCvmgsFpRCDLiqikKNWnPpWHZx9maRh3QIZFKlja5gQHWqwFXn1nEeI2Bo6LoQcekU5q1gjm56nNC42Atfln63KQpk73sme5HtpR0wHC/zMvFwkZbKyEXViaeOmpHhjFD55gBQ+nPDVoS3hfoeXh1Q4i5Dom4ZBiPAWiJihI1xgWIvwZ3vubLGZ/hLkWiL9jUagkFHJWZDBEoJRsoSX8XZFlqxsKdIccQVHIAD/iKegMAJA+s5XelLCwW5Id1Seqb2BNe/MHKh+YNcPTAVaFBOur+6zVUguj1mzLmoe5phZA6xiJFCcPi3zOGXitC1RTGG+FjAcMA0Z2ZRIQajgG6niE6H6obiY4aUHCdHgzLJQ4yyj9C1F2GnhBBwipKwyukFsw5rwlwuDDV9bsyNJzXuW67gVjVG5hNuC895Hnx040uPePQgykfXu/+GPZuhFCvfdf3n00RsPP/9SeKJ/9fGjt9/1zojgjGYEw4GRGSllH8bFlEvowy6JAZ781gI216OTIOKJiAQxMHCZNvhBlDafhZAz5m00vMnjuHDmlfQVhIiLYJDg6gPvYlUESpUZt9Fh5zYJaMWaF5dYcoWVCCYJzFfGU5Gl51lxRgzpGKtmdjDRW0FNBaX/9/8ZTNNqdWSsW5l86qWjucUXU2Fv9JTNHRmBUhiRRCbubZZo6WXpDb6Svs09l4w8/2ZqZTj6Co9lWfqsr9hAd2g0GzEAvHMWx9zFIa1fq1V42FrnjigKSWJRZEDmlQdR4A7zGBOhIkmvwkNJSYsGY6MBToJuaFjzHjn8NS2yaGYmvw2zHyG9l4xrejwKpjqGYxy8L5QuDNnLkfccQ524r46mLgVN3ugrmUcpeUS5dcDCbeCQkxaBBIixSNFPxN1XRQQlTkrGt9XKzdIWPWmGhARSgskwvqP5p7RlnMmaiHWe5pLRho9uTVklnt7CtGbbR6MJ7ZuO8R+X8mdJfAMZynLEbcS+/1AozBYK9vsP09PTU8n7D48jPcb3H0Lx6X8QQLgOM/q8+LnGdqBeu90UodjVT3n3j3Ncv4m5jEvBbiqyzre2s97FRrWX9a40Avb/q+J1gay3BEfeWCcm/K0CxKU5cWx04Wr5pcsvXwex7hXZyUHxbfltYvmDNOuRMde7VsB9+UMEM6G/8k2/daPv3+DXlHtNEXzeysvXwdktv97oqGYaq2g8mZiAKKFiQMOGml9B54Yu+HEHdR76h59fpnFPMUW86Iv917A+pwoI88phyKtb4GLYEpuU8za/bAtPs/AX0Jg4S4247X0Nc8FeizW43W210Qu4+wbacWf4B9lN/i8vURJGJUDZtUvfvXTt+qXK1WsXL13zMN76DI+0fo4HXp/lIdefxe1eYgqjHYmWLcOWNTAtvOrNKcZWImxioq4wjKE8HdcD12c07+CUg6UIhn5gIOBKE0JTLDLiuiLvrSnEqTZctVaIj5eMKYfCDifbiMBTvLg68jEUlwktbtg55NqMqevQXyuE5JHIQevjRj15ewo+oFyGji1qPQja55ZwzvLgrActiOjmi06r8opjzQGRswRCL6DMNepBeP2S5QOgN7McCD87Ue1tooX9JRV7QFp6iDm7uKLKiCiRpAiJXuVY1WoC/swakJx8xh5QZVtWvgfCeI7eGqbdVU6ahkbdYsIgRWDOSLhupsL3EACT7kr2UrmpsaM+iTCQq20G1QQ9F81s3GbwKWTIUFIZxGVsRBrQEJQRO2YizyiHMgo6orQKlUZdEaAmz1VuofIVHXr4+YL6kfXIb2y0IYlTjob1f6vBSLArVCQ21LRNC7DMpWMeuYPK0SI885SvJnI88CJUXFI465Om5Z8qXnVOx4ARIxNkY5jx+FDUbz06aufFBTfc6HjZ0QxONouzhcPi3SIjOw8mHLnmyX7L1iQnSHH1wUZWeJOXmWbT4XI0POWktwgOzObeGkkNjOuoVrJeunj6NN+DeOjcGDRDNcFvrDqiD9fqhGyhM2jvVcxYx9la8GqNLr7WJeqKYx2YcX56FUlUysibZjvolMPyBP1AfkM4JpMZaEAXE0Har9sk3TJMMKtpD1Ao+Hx8c67hlUPDo13SHFr1Cl/xUajXHXJQHO0Q+Eh6ReO6HT/E4h3DnQ52L6R3Ueci75o+2hbxMfAOvtybtmEXqzMtjVvPyAU6DkQcetr0hRFk1CYVOjici2WlAqhJewIk2dRioCKyDIcsQG5DR6cpkktG1iB+WPLSFkEI5tFYsXgCHMOlG4yb8yWQsSHiCAAencxhgVGxyuxi1mhgYuLUqVMqGKHPdECI1Ib8T8QYLy1roReL4V3ktBlP3I4b/mI5G5kPobhf1HfjdrNeq79Z2fK7naCk/sKPTMm/Ucev+KdGFMzKi2X+Rg0c6oVf2TIKLgxZULWP5eX7TOBQM7uiSmBfVBEEIZg/xHNfTi3AEUJZ2lw79e5apVZf7d/gYdiYSGogfr3e7zKVt1EN0rgAS4hcMzi5CBLvgyxmOI/Al2UTwWCp5V+1cS0l4zAbJYs2YBcwkDKKE7uZUNhK2Ei0644RsxLSTWmDFbEbtQCMTrN6LzVjS0pfH7gPwwegLaBGVEmxkYSLdetBv6nbNdyHVSFCoWWipRGnlwXymfpMG9EG+S2A9Xa/WWPMe8vfDuCCM5yXCCVZ3w7W9zX1UZeeYVd8Q0h0YQ1RJhQM0VWmaBZiDE3w9euMZ3QwimWjJiL/uEPy+bcbQYkE7dtwl1BqYTPvB0F/s55GcC+UvKmMI2fDmVP2Xgh/XHB9LBedRZ1fKWYH5DvbMjDqKkCOB3iuQLIMNyRjGMLmD4YN9MMjiwbxL4o5whOKrjtDnmFeJgSBFd30b1MvdPjCVkujw+P8pFVhC1Ic4A3RtQ2zawvFmL4tFMMwhu7chg0qDjQoSkXWMzUyHUPkjG5eh1HjdfmEcIxp7MmoSZi3QfJ4HLkiqSmClRmNfVtWtJoaRBHIzeShjaILweSOTh4WoKjJjJtLFyC4xFVmMGzwFPv2GBRmMkqcn6Cg9ayERi8nxyi5ES6JIf/skrqcnCMybWbjodnjcaa4x6oMbALfqeYkIj2Rpa0pD2PSsH8gHJ83uUznx0SyMseYoEl4Ja2QKQVQh35zgdTWAm7jbXCNRMba8j2IgcUtLxhNdN3vgbMabHA8sgcrKy4MMoIVgIhtSmsTusfK4kUjf9pRP8ndwNUBcTLDIaAE1VuMCIBmwhDlqZ/4aAGxaNw2QMBYxHDnRWhT7tYQilbH8aAWdYg81LTBFcdVR6g6zSC4ZSzEJfJK6xu8iI3AF/aS1tJIJgzy8AzGBFMUKikBzOVLuyRcOAG2bY6Lcg8LrrUwVeQ0xIaFnudBv7Q8rEiNVriTksHoUrKMg2VQUNEMg6wEpbLGR4VjJKpBS57hLeuJM/Ef2mHQ1T2Cgmisv1GRkAkT0Jo+MBE5IAFxyLBpbuKwGjajpOoQhjhOe+Qu6qB1WqE6mj5IOVXKQSEGuGgSEWFxNOShIjUGWQp/aDoBF1BVAiLlFLSpgxCoGYVPWPMoidjzZ21CC3bjso0BoSNH0IoZSSN2THKyxO5CUUejWzgodYihHA046bfcwcQ0xCHA3TkV9lqgOB7CCHfAHbURp01Oq0P5fUuIAjCzBdd0SkTg8GH/3DH2z0EkIc2ZyoYRvTnnl0R4x4DEa7QKawNzO61tEOE4iaLZl8DLqLmdxdDxeak9YtBCXT8kJ2vxLmvIppnwwyLW+4iVRqtV76bFCyLicRDxuAaxrVB7DARDAXfnrLeB//B7R6v1G42W+LveqsVdLYTYV1kwscD7PCCJtdq9Srde64NbiL7EYJtVyETZz0IMZRMJp2GsJOEUaaHxQi8DmgU8MGTMK8YK99PCNVTDGyKm8JRtL5H4dxpGNiJz9CxFZ7OJc2YuOi0rQ9tg3MYQaZnhn8GIygcmHoIwo+PgpGfMki1SsmxLmdiWwBQClWhj0oqGYZZr0XLfosV4ufjo9tjJkJEkMuzqBe6o7Djc8wN1Tyb2gsokKmBbjWaTcdGOuH3TaNUa1bq6uSXZrriXwOuFjv30hY502qF4ZTQDtqSyGf2ysfa+mrOHCP1o1eH+tN/dVspjG++6SpVR9E3tAZGdPB06Y3eoighs53DK4oJ+c0MIo6CPZLTKaAJWO65LZwwJJfkl0bmBmqw6YvbAgS5ti9z8KQyHPHAEDTnUBh30CPTkq241Q270Pl2n5KkWXFkxGoFgDSg7cygE5PPeomhfftyh65y+sEJnO04BwRJG38MwD2eS3aEWxh2iCVu9H9YwutNydT1yOIhCuxNMkxG2TPlx9YjDnTQBDTua1WGM0DiGVToGjL5uDiEY1RCCAw0hGH4IAR1C4BiCD6oq2dgPap+dNAANPQTfsrZnXLBVYX4t6TRZkS5ru2/bwNXMhYboYFJxIyXEPWkAHZ7oNlwQh2iFEqBhisDiBCmrljnCgOEq7UDhqvXwiwHi4NWJNuaqSDUxxu8LdiXxl9ztnK/h8InHvcX1SAxvkmfzR4H0hQslq+YIZZBLGGIfoOcxvrbyExaxYcwQHtlYQNS5DaXbb6smwm/tCNCq3qrjwMfGNZ8ueMyhsroDWJjK6xgCsW/3BPTxntBLKqGXbQL1qg00BIcLxt1PUtV+hEUGG1ihrkxaS/ZWdQwHxbB2NrhAaQlW8MhPJMvCR9aVChhjS8xYCMfWaNvP48LR0KwaO2qKWOnwHO20ouY2iK8YOCpO2k3uOKgCoEXVJG0OqhmaiB1pOCfYcXEA14gnKRALnwTijpOlOAekrArq7qO15HdcVi3TBG4ugR25BChlO3z5Jw/F0rBmyKxUCduVZFQBPQNwp8ceDJr4l7zJ8K1qa8z0mRQ05glrk2twDvCqP7Q74WUqLOcDiMNJ1RRGNHEEwxOHtjMpU7lNHoFV2Gn9NgkkGEAgCmKcuZsPWIcCqq+tNaqNeqvnqUidGBaQP8NAAicbAI5iMzcAHc0wbg7qAJ0Ka3YGpAP1KhqWK2o5uetqGMcLEEgf3R2MB7MiLOiVRquyCi/RZuWjMO4mkDYigWP2MGAPe6KgV9JR6ODoDUxqvMioaRZpRQ/YOoo4HHXbxxHxUOKoBtkhm1U3vchsx6weZMEYzgE2wbsn4IDLaAQtDDdJB1jtcbMUYQ6CpB/ks6450AKrTImwqSvqfMeG7D6h0ZDh0a2YIxmkIvLoFp6ukCcQQzZh+e5WlmbFvsAFiegpZ0oGukJFVkNFtJGI9wi3zygwq85sBMELHex1L/7YVvWIhjsEclR7jCeCvXS9dfAUs7s3jLoPD7Kb/RvQ3eGADnt1kr9rtsjfbTwtDoDOKOGM9c2WoxhkU4Zyv7VWFXniPcJ0CJrxclrVfjpNHljFHFZFvqyGjeIIQ09qOMkKxC65RAqzphgwVDPYFB6wiWPFQ0Ox5RAgwV57s1Gt+LVaZP2Bh65Br1v3NzcOdOxqn5oa/ZIp+vQU9oULr4WOLEd5qHrQ49SLr1aWLl+5FOrTxfPll69cLr/szHVCOvSRrDyO7TTEA5yOC+2S/KHM87rTmpCZlsbmCKLg9HxyJrngsopCOvoRPOtKFgKcTWXN1gcF9AmdvGMs6AhR0DrLNIczqCkqh4QmwckP1ED4AzBD4LHX7vkMe1BP2IhJ/dMWGQ2cLgJNrRYaWEabrPDGGhMtJC2AdGGANWuBpRK3ikarsdnfpEfxZ6wJpIOikY3WYUyk3vMKtMWg5Om1YTRnm7uFnjOq847qxlAVxKjxqseAwkDEyC2gZ+IqM3FbjF91gfzI2cCi6rpxDhdTzO5lBoCUtzAkldjdsxagKSGP4a5NuH1z0INDeoV4gAgxPw4mMBQbkMmYc2OShthQO81G7099P73+6pXLS3bJx7A3wlwFTMwfcHEHi5Xjbu3wYpwB8sKCDgD2aVldgRILE3r3LVpJ7NfDbcoMLmIto2opEgTIjHHS/rjYPGE30Xydl7R5SQwXwTOzY3hVb0zMYgCbGHL9V9bACyWtDxwtNOERpRNVqIL1hYvoqt/tNiBop8xxMotI5jAhhxRmEBPWwraZhJ3vZBSDGIOTEUyYeHbljoAhPH5+YIrn5moViNASYxzfkIUOxzoOyz4OxkIOyEZklWPASR6n4DGk0DHJ3dnEWg9CRBK6NUp3IKOgdW3ULoeGOyhj0IK078h1cJq2fIZAD1tZCH9SJ2P8tILHYa1stms89F8pld/qpVzKFi5h1GJcS4bgCC0oIrQyBcDxI6ncGrJ0ERerjC5ViRFZ/RDmQgXi8B5oODaHwdDo3nDmvWFMhkaXhwM7yGio4ILRqSIvPVbwwqORJ6+REZ8kmSeuFVRCl1tCLkiQ/GrVO8LF5xBR8W3b21AZcHa6Cg9q86Wh7vsIPmtqulhOrS1d7TRaUE3dgkfq165iRmVhJSUoAAOs+Oo49sYhbPkNxTyMrIjlrjt4RpG9XPr4xdQ8UeVXt3icKz603Ku3UlnvVrvpA2QeWoTHLC+EYy53/CBwoEi8VCCblWZkA3/kbF1fPrLePABKOVMiQO1Yw2QyDoZ9BrmyVpjFQwoe9kYZizNhTilEOoBnNJuVcMhgMqOJfnOE8DWjCjazCMFmRMQZ9j+9w8Ivumit32XrNwdx/cIrbP+8cuka/8GXYmQYG7406i1/lVEfY6npdgBR9eutW+mUCFd68Wrl/GtLV5deK18uv5yCYFsZftFC1O10GcYrAtmV1W0R+AZeT/JSdb/b3BaZFV5SB6sRVRiaIWBN1jt9emMLA9gsawh6IjkyxZ/w6CjWNktOenLopYJ3/8tfPfzkM+/60rVL5xcX/t9Xf3X/i7vwIsE/vfnozbt77/zFox/96v7vfrL36YdiB/UefPjjvTe+MiHCmQdvKe83mxXsYDqzTFC8AsiYgudS08OULPCr9O6SYqZ4sYzuh7hgujuReRzhhgjprWQgYhBdYRHa3PE24wwy1dga24HNOIMjANE0ukMUt/lIEdzBrUeE6hUUTt7Opy7HYNIzVAsx05aOMdCA6Z7jg5pBbSxnh9VRwkeQUegawZHiaPE1IoQpCrdpeiAOB1qNDmYu4j+/KZPRJA720uI4jEoDWVQ8Szo8CzoYyxnSRiXp5ihsh5IGTQdYXaG60aRC08D9jKzRUJ5cs6EMk3zC+fFr/CAL+cj8bdJbunrx6jwPcUuC5ZnaRRDAo8LcgJQwx8Mxx0kZWC7wct5ymUPLCl+jlSzGt5aRF1lWjgbahsTKCLVrQgRe9OxiJigRtQzLwL1rDcAROhJZdzomtrs7yeeJhy3vDBAfXVyqFREPJIQDdIvI3eGQ3jxyN97aMx4V4ugTQV7MEN6C6rmLPZf7uQc+zbKqAdezo/Pwp8tosQIvVqZgzPjhUVBoKQFEX0QUkEiV50vC/yqcxRSwgQ9atLsGmkryUSWhuTHEaQ3cjj0LkV2dmRB4dnaG5CkFnGXY31HtBnD8+y6NQ+fJ0ML1zU5vG44KyhFhiaVdTMwB/yFseTe63Cor9KzFS0vn5z3ywgTRuBhwyF4+RQd6aoU89UdLLzhKL6jSyIudgVuXoUcrcg0O5kvi8u/AcnjVeGAxRO0QdwyGKAJxsQeWGoL1LgwFKOoFK1c6fZoaH+KSGcYZsTPBpFz1qE6+2e3jU2yco/JHz/hDXmo/FLaTtUaz3vF762kmCTLuzNi++JfKp/yDwe/YNopsDkkZHz4s6adxwNZTwTwRNp3JYKRsvtneqnfTmTw81dQL4DWFdGp1K0WENhNy6sXvFaemUhJUaqPAfgG7cACNhLHAKlXOX0Yo8FQRH77MXksp5JRLd8q72YXSnYXdLIFRukN+sBy+fDles+TRsjsadbt5aCVF+RM2CSgHe2eQh7/yP2g3Wnp9r6XuRD5UtAumN+jjWegsb5t9McdCFrOgELvpw5CKeDdyAJ3giyryfaZleLRpmZc739pe0Q+p8H+viVfWWOFNHvoZG+p3ufsUBjjlsTPn/DmPe194a505cf6kHlapy0FiICkeA2UVIlTWGjxCPgQa4dFRNv0OxJvwGt1u/UYfntTjbLa9xgFxozxGYMZAl0ZXRMBmd0+8pbZXh9ZFhBNhfuUPyraZCCrCsegWvNUgiyFZmu2gHvRoloiUz7sW8NjR2D0Iu4yP2uKbLrB5tqsNv2fjzauuM6AtfD/gVnsDg65IpFH8OwhyGEbBxWDiUCLXpCTm+u1G0AvSNnSyLPGx9E69FS4EIW/WnOrZHbAib9S3M/Mw0+gIXt/O8h8MabAE8KRiLZNv9OqbQTqzK6ONXIYr4FGEBi8E+rf8RhNs1VlvS9CQfH6RLQof2CtCMmryb/jkV57xxlajdcNUFNdSr7de488kCCgef0mWR/gBFUxSiTDbeq/Wu/zl+Va17m3yVy5SFjyuCK3m+FD85tPeBWHDhhMiCCO11u63+MPHd2zU7qYMlsAfbGWyOCwgxrHr3To0Kp8oYhn8wMbjMvpm1ttY0WoCfGqxTyjVw3ch9LPvG2fP3sh6Le7IqnUGnbEimIx8mjDNm5FSd88tdVuif0QpU96PAmVI+RGF2NSj2i+kH87ZwBk/VFKwxnmT4614+CAuUQ161lu4SE0sgWsCb4q/IL3ebvKAeaI9BwBpUQI1guMO5e5InYO1IhzaxLFrWMVwvhIkCFYs/0ihehGk4Fm3TD1Q4KYvoJHzjHmKaHl6MS8PhvjHi5Xrr5x/9RL7CjeKyhkj6+LS93mW0S+pHlvQTV3ZyhS6vftrRRwAOXLlQUl6GqKcE4TDazam/kRqhlRzBlnbNXYlm5VqDuNa5pumWv+J2cORe+v2M2QnAGchLLa82ZC8OcgzLstfFIcDRaGR3J73/NUgDTfBFzOZFdot/p6sSTyCdS76G8C8usioWB97Df6eDRvIVheetuEPnfFTZOMUrqYeeW6yjUV0a1mRgIhdKUot54qw+hbJsEhRliPKiT6xjtuL3T5MI+dnL1gZ4RkTEg4e89kmKPGiLjYGvuJpUpjPLqznrKfeHpMulDeFRya2K//NV9udbZgYpzHbVBHsPvDWR9NcZsK0qEeMMpLPxzD3aI4ezcaxs/P4XCh+WbsxI/m4/go7XQR/Pgp3RZOGVJTzjYDe7ck4OLDLFBTJlIXBpKzsPtoCtGhbgEKASyp+EJQPjcQ2/dwczvZzcxjjz4isUeO3IYF1p9oPFCdcVnvAisWLQrxF5OObgDQftyM7H8lOmD25diyzGPuZRTZKysjHC7n0xHSDNNhxiEg9Se7apuHiHLfzmDv1itp/XJkhUEDv6bILVDkOVJmAIvd/iRVqMevsQMaqVraqlR3Vyka1IfwFxCaKgOzLFrZ4VusAoeA5kJXl12oef5bXaBWf5Kj19dNsMgk+nDZvgoluyP3lDAWWHeZoBHZS/YrQdr1nlRLHHc6uYONKLoLmLaJXIs5KNuO0ceLlP7EopD1TELKyeiIfNZ5N0yw532l30moR4b2vTEQZXEhxZex+y7LI8eHyI7B9NGaq6+Lu20kXri6+yvBSuVq+8n1yOUlPAwcnHq1OA1hQATf7nSxfnyX4n/SK0ZMqDhVZzdedBtrFrJoDa3EAHZuLwVGwjAVtwQWHD8FqKs4JFDK08fKd4EwRswgAxeGlDSs82qx3INhisoR4ItFkm54tk/BN05Js2Z/p2LNsWbTqW7CC9VmiGEtWrptQAX2kp23MhilZW4wNw7BD0j99WpNt5qhUiIvZkPXoYN2ymZb+BijCKg2ngas0lCqugQ+jk6sUFvCcpZxCnyslgmAiCJ5AQXCy2u92K7qbWKTar/nk7KXS6bY7DCuNepBOQV4qg94FkAFRuNtdRpb9lrBuQqQMAVBSk/z9NBsIaZCouZ0usK4UYJqbNIUtAe5L1xhfhceg2i0Jp3RH/LGb9Vb7PSa/bHndfgssp0Yp2tYuXMze9pp1nz8rtcr+6Wg76dOpTLxkPDK5eHRS8WOTU4eROgfImaOWIMFUpHhzjAkF2zmqtImjstokHX4Bb1jTXGrtGX1XKIKdFwYSYTpemB6tYHgSxDoGBceUb9Uam1BwWhdEXyIhQohixnUjjjOX6I/kdjjRHxIIlei8BS+3g3Rzi+008raWlLVNyRtv0uk+Z6wBm7JrzIoRI1VPXYfcfqyrEXjOxfYk2GjkIRc8x9tsb3ER2fA2kabvkAuJ9HuGGpVqvdGsgC52O+ttZ+y589Lg3rWtdoRtUpM3JhohFfVZi8ONQ+NdfYRDmdOnK9JMKzsJf/F+72pPq2G9aeEOh4Jv74TZqLxyTN4C2fiIy5LZDrlUZcZgJTWk8d1wNQzHubccsw1PRC5SMziz03MzzjNmcF2SutSm3wAJpuQtV9FFiHgHyUd75ExmVqgCJuui8tXv1PyeJC1Ybngl7OUsMFwx+3ri16SdXQhY+x//bv/uvb1f/H7/o3978OGPH71198Ef7j2898mDez/fe+u/7n/5s713f7T/t7/hXkhv7P3k7x68/9n937/jNX1G4+vq66M3/7j31l152rz3F289evvu3qd3EQ7pwTI9voLTkrTDyICWqpcVZb+s9nVpdrB9mJyHKSbDAyFG/VAis96oS95UONs8u4soRM5+kNk5igjJ3AOJm8gMBxrPC+ZwLGlstEPS/Z0uxo/Y7iby4AydLocIhloHQZfJhsNSoSVku21YimLc2VGCttvOFQfsMMJ2zW/daLLdQRXd9G+DwzitnwsrgxnvWxEaIgfa0S0bcMzWKMez++EQ99mu2+AvOTFtqGZGlXaSLCEho4xaieIeq13ARRhRwCLoemjajiRgc8maxULLNkyqiLGgBzd6XKqMs/9y2oYdRsRs8hm1Y9mlF9wkvRBP0nQbhaSajAx6F9XfGDAYLqQPvbTghxQ7R2/D5A9JKPHS8cIJ290LCd+Z65yakNLJWrPad/YNQBpzCL0Mhck7RC9ZLffSimDhkasgsqeH4OdUVBEXNEbi7Q7JvhpkecLYt4PIfe0j+MovogN8cS4LV/jhVpGnvOTLPO9cgX0zCrD81w0gI3W4597zWs2JcJKfQB9op3OiQ7MSLtEobfFoCtT30Pgt0C7Qqzty9KBTQmCVQV/M0E8i8oMM+WJmFuRpGH1L0vHCZNn5lei5rmxDJxhQwN0AjigqR76Uidg+WsyaI8Sr4fWjgrxEBXg5WHCXIwZ2mZDryRG8hej9tgFjiOAtIw3c4gxxohbVQcOcRIQ4OUJ4k4GhTWhYk87BIpvEsBhhvCEsRj4/GvaKE21FF1hEDwHkRORvTQbynmsBvQbwOiMvM7wD1KLDpWkx0gEq3l1KkK+xnNX+bKzSEuwm/LPeN7gUirMXx8CFK5VFivapp4Fd/Znsp2TvtBm8PQjnV1LfsfEKfDyVpBOX8me5U1U+WB9fG1MsPXvuHP+XJfvfmUJx5qnCueL0zLnZ6cLM9FNThcJMYeYpb2p8XdKpD+P3vKe67XYvrtyg/BOaJp8+u9ponV31g/WJSa90hMSqw7URPzeTmy68mGv0cue/92e5mdVGz7vVbG56Qb17q+7tvff53k8/e/jjD/c/+jdWA02U897Sq6UZtiX5oA95/DY6B+bt/fTjR2/8wksTjzkQHI7YUxACcvWJCSaltrs9D1zlSmfbnd7Z9e11vwkImec/a70N/uMZKAHnAjmWvEfvv7l370Nvxtu7+8mjf/3g0S//2ksH1eaG16zfqjc92IefnZ1afGUn6+2/8c8P//wPe2/9x6P372V47Yl6db3tpZblqluR4C5eeM1j0LDG/mef7H31bj6fT02AeNQAsWjKK3hFb/o5r9Zm/Lbbrm7mgs2Gl6t5z7D/59iYQGPAPmz6rb7f9F44W6vfOtvqN5te8YVvF7wf/pDtb/16RHU+htnISjXwAZr0Hnxy7+G9T1mn2bj3P/h8771/evD+L/b/9ide+v4Xd/fe+8eHX//N3tu/Z0P/2e/uf/3H/Xf+1/4vfv0cDJJl3//ip/e/+vDB3/92/51f73/w9YNPf8+K7v/DrzNOrLAGHnz5N/v/+BE2OR+JC+h46Zl0eEzr7a0q+N8FbCR6VD9kAkO94+Xal7wUHzMvg7MHT6fnvrNyxns9jX8sru+8nkmxOutwoyDHI2JhZz3vlQuvLT/TWJn3nrkDcOZz+x/9y8N3/nvv3Z/t/eQ30Pt33vbSOJC9dz8HQvj0fz/87a+z3sN/ftO7eOnVK1e/n9+siaHu/+qN/f/8n/e/eOPhT3+U2U0huiXJPXjn871f/vneux88evtdQUZIuq9cfrXy3cvXL7945VLl4qXvXr5w6XppKlvIFrPTqshLlVdee7Fy9aWXrlwuXyoV5Pela+fL11+6em3x0rXrJFc2yRbqg/c/i16R+3//x/27vxTyEhYQFD7pdUDEYNPIporNKEP53nvvPPryAzaJe3f/AwE/5+3/+6d7n/6YkQGj9If3vvYgVtpb9//w0f7Hbz+895v/+8aP4OOUxzB3/4vf7v/8v9gq2n/nLx/+52/4PZFmo1VnZVhj6P167erVJQ/b2P+rt/fe+2umMHVE7/fu/eXDXzHQwGoYYf78YwjAdu9r9gsgCHR898qVxQrC4m534EB74RUTIYxzUV6WBq4VtawjimcNjHJECVRn+HIn5c+CcnB25+bZzXat3gzORnDXs+J2aC633g568AYs/Kc+8sHNTU0V1BcOvZbjUHPcnZlDnlEFNv3bIrdZbzGJ9tnZOSOPyX0MyE22qs7NqoyNWznuF5zD+0jg8Sqz/B6TTOEMNQdPwUHAYnGyfX5pqaxKofSa6zB1ttlkbXP1XXfqRqef26xvtrvbuX6v0WzsoDvmVP47eqzCZ5Pn5JATTmv4XbbV57oMRK+eq7LhqRwMPJjrdOtrjdt8FHDwa+VW1/utDYY2XoqRtJXdbN9gsG/26wFT3mRek2myff9GXWCz3Wpua6QE261qLmAYq/WbtD1yXYnX83rrbPO7sQ4OlvYscFmedarXZmgN8KxSz3OnXoW3uBq3YLz8zPDUndRmvbferqXmvdRmrwOWJ943+D2A3vwgaDDybvWgEjdvafgV7AB46e2eSvSAE5TyZ4XN9uz42hgg//O/Lfl/eubZp7xz4+uSTn/i8r+efxLPosCmq5hdKJ2bfnbWCGnB42pEhbKYLvIAFuE2YIJnZ2ai5v/cuWlL/yueK85OJ/rf40hwIpMqMMYtQre6/IrXeMB9zNdeznPCkdnlix3l9G0f6cggSlG+3tqfW35wRFri36o9P9Bu31bgJdJ3FZZWfgy7eke6eRNvcdVB7Vsuz7yU5zlfRXjmMwFHVoDpYoLp0WG6GIfpmQTTo8P0TBym5xJMjw7Tc3GYLswmqB4ho56Nw/V0wqpHiOvpWF49mzDrEeJ6NpZbF4oJvx4lEylGceyJ3W9ath8mufW/6ZkZpv4xXWz6Meh/z05PhfW/c4VE/3scSTACrQHGLei4Re1a2GY8rwPF7HJH0xrAryKWeOwyp60NEaXLCO8bZgeCE2TNn2w5qQ8rRn1n9LG4aMIm51OfFfdDBrQr+FUxmdYRTmvxuEzrTDKtI5zWmeMyrXPJtI5wWueOy7QSrT2Z1xFsrrPHZWKnk+11lBM7fWz219lkgx3lxM4emx2WGj9cM2vEGE5mduDMGuP+hqb2ZNhZjmuKs/+M6Ph/8Pm/ff+7eK7w7LOJ/edxpOT8f3zn/5IXJsf/Y0F0MQbRyYHSWE7/w4hODpPGcvgfRnRy9j+ms/8wqpOj/zEd/YdRnZz8j+nk38FAkoP/cR38G8g+WfqoU/+bPjf3Det/xalE/3ssKdH/xqb/wSqyeHAiWIxD/3MgOhErxqH/ORCdyBTj0P8ciE70v/Hofw5UJ/rfePQ/B6oT/W88+p+LgST635j0PxPZT4D+NzP1ndlvWv+bTfS/x5ES/W9s+h+sIosHJ4LFOPQ/B6ITsWIc+p8D0YlMMQ79z4HoRP8bj/7nQHWi/41H/3OgOtH/xqP/uRhIov+NSf8zkf0E6H/nCsWpb/r+ZxL/6bGkRP8bm/4Hq8jiwYlgMQ79z4HoRKwYh/7nQHQiU4xD/3MgOtH/xqP/OVCd6H/j0f8cqE70v/Hofy4Gkuh/Y9L/TGQ/CfofKH6PLf6T8/xvNnn/5bGk46n/qe+2y/ToGEPxSIxBDShO/2OryOLBx06wOMmILsYg+tiJFScZ0TMxiD52MsVJRvRcDKKPn/53kjFN9L8wqo+f/neSUT0dx6ePn/53klE9G8epj6H+d5JxTfU/A9lPjv5XnJqZ+8bufybxfx9LOp7634k2DJ0k/e8kI/pE6X8nGdEnSv87yYg+WfrfScb0CdP/TjKqT5j+d5JRfdL0v5OM6z8B/Y+7ASb+f5gS/S9hDIn+d5wQneh/if73ZCE60f8S/e+JQ3Wi/yX63xOI6z8B/e+bjP9SmEr0v8eREv0v0f9OKKIT/S/R/54sRCf6X6L/PXGoTvS/RP97AnH9hOh/SUpSkpKUpCQlKUlJSlKSkpSkJCUpSUlKUpKSlKQnM/1/2na6iwC4AQA=