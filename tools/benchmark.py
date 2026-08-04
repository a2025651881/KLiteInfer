#!/usr/bin/env python3
"""KLite 基准测试：CPU / CUDA × 三个模型，输出可直接粘贴的 Markdown 表格。

用法:
    python3 tools/benchmark.py [--repeat3] [--device cpu,cuda]

注意 use_cuda 是编译期常量（src/config/config.cpp），切换后端需重新编译，
脚本会自动改配置、编译、跑完后还原。
"""
import argparse
import pathlib
import re
import shutil
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
CONFIG = ROOT / "src" / "config" / "config.cpp"

MODELS = ["llama", "qwen3", "paddleocr"]
LABEL = {
    "llama": "stories110M (110M)",
    "qwen3": "Qwen3-0.6B",
    "paddleocr": "PaddleOCR-VL (0.9B)",
}

PATTERNS = {
    "prompt": re.compile(r"Prompt tokens\s*:\s*(\d+)"),
    "generated": re.compile(r"Generated\s*:\s*(\d+)"),
    "ttft_ms": re.compile(r"TTFT\s*:\s*([\d.]+)\s*ms"),
    "decode_tps": re.compile(r"Decode\s*:.*\(([\d.]+)\s*tok/s\)"),
    "total_s": re.compile(r"Total\s*:\s*([\d.]+)\s*s"),
    "vision_ms": re.compile(r"Vision encode\s*:\s*([\d.]+)\s*ms"),
    "img_per_s": re.compile(r"\(([\d.]+)\s*img/s\)"),
}


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def build_for(device):
    """把 use_cuda 改成目标后端并重新编译。"""
    src = CONFIG.read_text()
    want = "true" if device == "cuda" else "false"
    new = re.sub(r"const bool use_cuda = (true|false);",
                 f"const bool use_cuda = {want};", src)
    CONFIG.write_text(new)
    r = run(f"cmake -S {ROOT} -B {BUILD} && make -C {BUILD} -j$(nproc)")
    if r.returncode != 0:
        sys.exit(f"编译失败:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def measure(model):
    r = run(f"{BUILD}/demo {model}")
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(f"  [warn] {model} 退出码 {r.returncode}")
    got = {}
    for key, pat in PATTERNS.items():
        m = pat.search(out)
        if m:
            got[key] = float(m.group(1))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cpu,cuda")
    args = ap.parse_args()

    backup = CONFIG.read_text()
    results = {}
    try:
        for device in args.device.split(","):
            device = device.strip()
            print(f"=== 编译 {device.upper()} 后端 ===", flush=True)
            build_for(device)
            for model in MODELS:
                runs = []
                for i in range(args.repeat):
                    got = measure(model)
                    if got:
                        runs.append(got)
                    print(f"  {device}/{model} run {i + 1}/{args.repeat}: "
                          f"{got.get('total_s', float('nan')):.2f}s", flush=True)
                if not runs:
                    continue
                agg = {}
                for key in PATTERNS:
                    vals = [r[key] for r in runs if key in r]
                    if vals:
                        agg[key] = statistics.median(vals)
                results[(device, model)] = agg
    finally:
        CONFIG.write_text(backup)
        run(f"cmake -S {ROOT} -B {BUILD} && make -C {BUILD} -j$(nproc)")

    # ----------------汇总 ----------------
    print("\n\n### 文本生成（取 %d 次中位数）\n" % args.repeat)
    print("| 模型 | 后端 | prompt | 生成 | TTFT | decode 吞吐 | 总时长 |")
    print("| --- | --- | --: | --: | --: | --: | --: |")
    for model in ["llama", "qwen3"]:
        for device in ["cpu", "cuda"]:
            a = results.get((device, model))
            if not a:
                continue
            print(f"| {LABEL[model]} | {device.upper()} | {a.get('prompt', 0):.0f} | "
                  f"{a.get('generated', 0):.0f} | {a.get('ttft_ms', 0):.0f} ms | "
                  f"**{a.get('decode_tps', 0):.1f} tok/s** | {a.get('total_s', 0):.2f} s |")

    print("\n### PaddleOCR-VL 端到端（828 patches / 224 prompt tokens）\n")
    print("| 后端 | 视觉编码 | TTFT | decode 吞吐 | 端到端 | 吞吐 |")
    print("| --- | --: | --: | --: | --: | --: |")
    for device in ["cpu", "cuda"]:
        a = results.get((device, "paddleocr"))
        if not a:
            continue
        print(f"| {device.upper()} | {a.get('vision_ms', 0):.0f} ms | "
              f"{a.get('ttft_ms', 0):.0f} ms | {a.get('decode_tps', 0):.1f} tok/s | "
              f"{a.get('total_s', 0):.2f} s | **{a.get('img_per_s', 0):.2f} img/s** |")

    cpu = results.get(("cpu", "paddleocr"))
    cuda = results.get(("cuda", "paddleocr"))
    if cpu and cuda and cuda.get("vision_ms"):
        print(f"\n视觉编码加速比: {cpu['vision_ms'] / cuda['vision_ms']:.0f}x，"
              f"端到端加速比: {cpu['total_s'] / cuda['total_s']:.0f}x")


if __name__ == "__main__":
    main()
