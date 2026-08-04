#!/usr/bin/env python3
"""PaddleOCR-VL 多样本 OCR 对比：KLite (C++/CUDA) vs HuggingFace 参考实现。

对目录下每张图片：
  1. 用 HF processor 做预处理，落盘 pixel_values.bin + meta.json（供 C++ 侧读取）
  2. 用 HF 模型贪心解码，作为 ground truth
  3. 调 ./build/demo paddleocr 得到 KLite 输出
  4. 计算字符错误率 CER（Levenshtein / len(ref)）并汇总成 Markdown

用法:
    python3 tools/ocr_eval.py <hf_model_dir> <image_dir> <out_dir> [--max-new-tokens 256]

HF 参考实现默认放到 GPU 上跑（fp32，与 C++ 侧同精度），否则 9 张图要跑很久。
"""
import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO = ROOT / "build" / "demo"
SEP = "-" * 40


def levenshtein(a, b):
    """标准 DP 编辑距离，滚动数组。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def normalize(s):
    """比较前统一空白，避免换行/空格差异淹没真实错误。"""
    return re.sub(r"\s+", " ", s).strip()


def run_klite(ref_dir, max_new_tokens):
    """跑 C++ 实现。必须传相同的生成上限，否则长度差异会淹没真实误差。"""
    r = subprocess.run([str(DEMO), "paddleocr", "", "", str(ref_dir), str(max_new_tokens)],
                       capture_output=True, text=True)
    # OCR 文本在 stdout 的分隔线之前（glog 日志走 stderr）
    text = r.stdout.split(SEP)[0].strip()
    stats = {}
    for key, pat in [("vision_ms", r"Vision encode\s*:\s*([\d.]+)"),
                     ("total_s", r"Total\s*:\s*([\d.]+)"),
                     ("generated", r"Generated\s*:\s*(\d+)")]:
        m = re.search(pat, r.stdout)
        if m:
            stats[key] = float(m.group(1))
    return text, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prompt", default="请识别图中的文字内容。")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    images = sorted(
        p for p in glob.glob(os.path.join(args.image_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        sys.exit(f"{args.image_dir} 下没有图片")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoProcessor, AutoModelForCausalLM

    print(f"加载 HF 参考实现到 {args.device}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32).eval().to(args.device)

    rows = []
    # 支持分批跑：已有结果按图片名合并，最终汇总所有样本
    result_path = os.path.join(args.out_dir, "ocr_eval.json")
    merged = {}
    if os.path.exists(result_path):
        try:
            for r in json.load(open(result_path)):
                merged[r["name"]] = r
        except Exception:
            merged = {}
    for path in images:
        stem = pathlib.Path(path).stem
        case_dir = os.path.join(args.out_dir, stem)
        os.makedirs(case_dir, exist_ok=True)
        image = Image.open(path).convert("RGB")

        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        grid = inputs["image_grid_thw"].tolist()
        ids = inputs["input_ids"][0].tolist()
        n_patches = int(inputs["pixel_values"].shape[0])

        # C++ 侧读取的预处理产物
        inputs["pixel_values"].detach().float().cpu().numpy().astype(np.float32).tofile(
            os.path.join(case_dir, "pixel_values.bin"))
        json.dump({"image": path, "grid_thw": grid, "num_patches": n_patches,
                   "input_ids": ids, "seq_len": len(ids),
                   "image_token_id": model.config.image_token_id},
                  open(os.path.join(case_dir, "meta.json"), "w"))

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**{k: v.to(args.device) for k, v in inputs.items()},
                                 max_new_tokens=args.max_new_tokens, do_sample=False)
        hf_text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0].strip()
        hf_s = time.time() - t0
        open(os.path.join(case_dir, "ref_generated.txt"), "w").write(hf_text)

        klite_text, stats = run_klite(case_dir, args.max_new_tokens)

        a, b = normalize(hf_text), normalize(klite_text)
        dist = levenshtein(a, b)
        cer = dist / max(len(a), 1)
        # 一方是另一方前缀 → 数值一致，只是终止时机差一点
        prefix = a.startswith(b) or b.startswith(a)
        rows.append({
            "name": pathlib.Path(path).name, "size": f"{image.size[0]}x{image.size[1]}",
            "patches": n_patches, "prompt_tokens": len(ids),
            "hf_chars": len(a), "klite_chars": len(b), "edit": dist, "cer": cer,
            "prefix_match": prefix,
            "hf_s": hf_s, "klite_s": stats.get("total_s", float("nan")),
            "vision_ms": stats.get("vision_ms", float("nan")),
            "hf_text": hf_text, "klite_text": klite_text,
        })
        flag = "逐字一致" if dist == 0 else (f"前缀一致 (差 {abs(len(a) - len(b))} 字符)"
                                            if prefix else f"CER {cer:.2%}")
        print(f"  {pathlib.Path(path).name:24s} {n_patches:5d} patches  "
              f"HF {hf_s:6.2f}s / KLite {rows[-1]['klite_s']:5.2f}s   {flag}", flush=True)

    # 与历史结果合并后一并汇总（便于分批跑完9 张图）
    for r in rows:
        merged[r["name"]] = r
    rows = sorted(merged.values(), key=lambda x: x["name"])
    json.dump(rows, open(result_path, "w"), ensure_ascii=False, indent=2)

    # ---------------- 汇总 ----------------
    exact = sum(1 for r in rows if r["edit"] == 0)
    total_chars = sum(r["hf_chars"] for r in rows)
    total_edit = sum(r["edit"] for r in rows)

    print(f"\n\n### OCR 结果对比（{len(rows)} 张 PaddleOCR 官方测试图，"
          f"生成上限 {args.max_new_tokens} tokens）\n")
    print("| 图片 | 分辨率 | patches | 输出字符 | 与 HF 对比 | 视觉编码 | KLite 端到端 |")
    print("| --- | --- | --: | --: | :--: | --: | --: |")
    for r in rows:
        if r["edit"] == 0:
            ok = "✅ 逐字一致"
        elif r["prefix_match"]:
            ok = "前缀一致"
        else:
            ok = f"CER {r['cer']:.1%}"
        print(f"| {r['name']} | {r['size']} | {r['patches']} | {r['hf_chars']} | {ok} | "
              f"{r['vision_ms']:.0f} ms | {r['klite_s']:.2f} s |")
    print(f"\n逐字一致: **{exact}/{len(rows)}**，整体字符错误率: "
          f"**{total_edit / max(total_chars, 1):.4%}**（{total_edit}/{total_chars} 字符）")

    speed = [r for r in rows if r["klite_s"] == r["klite_s"]]
    if speed:
        avg = sum(r["klite_s"] for r in speed) / len(speed)
        print(f"KLite 平均端到端 {avg:.2f} s/图 ≈ {1 / avg:.2f} img/s")


if __name__ == "__main__":
    main()
