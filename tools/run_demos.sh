#!/usr/bin/env bash
#一键跑通并回归 KLite 的三个 demo，附 PaddleOCR-VL 的逐阶段数值校验。
#
# 用法:
#   bash tools/run_demos.sh            # 跑全部
#   bash tools/run_demos.sh llama      # 只跑指定 demo（llama / qwen3 / paddleocr / test）
#
# 前置条件见 README：权重需先下载并（Qwen3 / PaddleOCR-VL）用tools/export_*.py 转换。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
MODELS="${MODELS_ROOT:-/root/KuiperLLama/models}"
REF="${MODELS}/paddleocr-vl/ref"

WANT="${1:-all}"
FAILED=()

hr() { printf '\n=============== %s ===============\n' "$1"; }

run_case() {
    local name="$1"; shift
    hr "$name"
    if "$@"; then
        echo "[PASS] $name"
    else
        echo "[FAIL] $name"
        FAILED+=("$name")
    fi
}

hr "编译"
cmake -S "$ROOT" -B "$BUILD" > /dev/null || { echo "cmake 配置失败"; exit 1; }
make -C "$BUILD" -j"$(nproc)" > /dev/null || { echo "编译失败"; exit 1; }
echo "编译完成: ${BUILD}/demo"

if [[ "$WANT" == "all" || "$WANT" == "llama" ]]; then
    run_case "llama (stories110M)" "${BUILD}/demo" llama
fi

if [[ "$WANT" == "all" || "$WANT" == "qwen3" ]]; then
    run_case "qwen3-0.6B" "${BUILD}/demo" qwen3
fi

if [[ "$WANT" == "all" || "$WANT" == "paddleocr" ]]; then
    # 清掉上一轮的中间张量，避免拿旧结果去比对
    rm -f "${REF}"/klite_*.bin
    run_case "paddleocr-vl (OCR)" "${BUILD}/demo" paddleocr
    run_case "paddleocr-vl 逐阶段数值校验" \
        python3 "${ROOT}/tools/compare_paddleocr.py" "$REF"
fi

if [[ "$WANT" == "all" || "$WANT" == "test" ]]; then
    hr "单元测试"
    TBUILD="${ROOT}/build_test"
    if cmake -S "$ROOT" -B "$TBUILD" -DKLITE_BUILD_TESTS=ON > /dev/null &&
       make -C "$TBUILD" -j"$(nproc)" > /dev/null; then
        run_case "gtest (kernels/tensor)" "${TBUILD}/test/test_llm"
    else
        echo "[FAIL] 单元测试编译失败"
        FAILED+=("单元测试编译")
    fi
fi

hr "汇总"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "全部通过 ✅"
    exit 0
fi
printf '失败项:\n'
printf '  - %s\n' "${FAILED[@]}"
exit 1
