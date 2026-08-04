#!/usr/bin/env python3
"""生成 README 用的 OCR 结果表格素材。

读取 tools/ocr_eval.py 产出的 ocr_eval.json：
  1. 为每个展示样本生成缩略图到<out_dir>；
  2. 在标准输出打印可直接粘进 README 的 Markdown 表格。

用法:
    python3 tools/make_showcase.py <image_dir> <ocr_eval.json> <out_dir>
"""
import argparse
import json
import os
import pathlib

from PIL import Image

# 展示样本：(文件名, 说明)。覆盖不同版式与语种
SHOWCASE = [
    ("medal_table.png", "中文表格"),
    ("formula.png", "数学公式"),
    ("book.jpg", "英文书页"),
    ("doc_with_formula.png", "学术文档· 公式混排"),
    ("textline.png", "中文文本行"),
]

THUMB_W = 240# 缩略图宽度
THUMB_H = 190   # 缩略图高度上限
MAX_LINES = 4   # 识别结果在表格里最多显示几行
MAX_CHARS = 46  # 每行最多字符数，超出截断（表格列宽有限）


def thumb(image, w, h):
    r = min(w / image.width, h / image.height)
    return image.resize((max(1, int(image.width * r)), max(1, int(image.height * r))),
                        Image.LANCZOS)


def brief(text):
    """把 OCR 输出压成适合表格单元格的几行"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = []
    for ln in lines[:MAX_LINES]:
        if len(ln) > MAX_CHARS:
            ln = ln[:MAX_CHARS] + "…"
        # 竖线会破坏 Markdown 表格；反引号避免与代码块冲突
        out.append(ln.replace("|", "\\|").replace("`", "'"))
    if len(lines) > MAX_LINES:
        out.append("…")
    return "<br>".join(out)  # 单元格内换行


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("result_json")
    ap.add_argument("out_dir", help="缩略图输出目录，如 docs/images/ocr")
    ap.add_argument("--url-prefix", default="docs/images/ocr",
                    help="README 中引用图片的路径前缀")
    args = ap.parse_args()

    rows = {r["name"]: r for r in json.load(open(args.result_json))}
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("| 输入图片 | 类型 | KLite 识别结果 | 与 HF 参考实现 | 端到端 |")
    print("| :--: | :--: | --- | :--: | --: |")
    for name, desc in SHOWCASE:
        r = rows.get(name)
        if r is None:
            continue
        stem = pathlib.Path(name).stem
        img = thumb(Image.open(os.path.join(args.image_dir, name)).convert("RGB"),
                    THUMB_W, THUMB_H)
        img.save(os.path.join(args.out_dir, f"{stem}.png"))

        verdict = "✅ 逐字一致" if r["edit"] == 0 else f"CER {r['cer']:.1%}"
        print(f'| <img src="{args.url_prefix}/{stem}.png" width="{THUMB_W}"> '
              f'| {desc}<br><sub>{r["patches"]} patches</sub> '
              f'| {brief(r["klite_text"])} '
              f'| {verdict} '
              f'| **{r["klite_s"]:.1f} s**<br><sub>HF {r["hf_s"]:.1f} s</sub> |')


if __name__ == "__main__":
    main()
