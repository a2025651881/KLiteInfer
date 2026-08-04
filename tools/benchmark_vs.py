#!/usr/bin/env python3
"""KLiteInfer vs HuggingFace Transformers 单请求推理基准。

两者条件严格对齐：同一份输入、fp32 全精度、贪心解码、batch=1、无量化，
并顺带校验输出是否逐字一致（性能数字只有在结果相同时才有意义）。

    # 多模态（PaddleOCR-VL）：比端到端延迟与图像吞吐
    python3 tools/benchmark_vs.py ocr  <hf_dir> <image> --ref-dir <klite_ref_dir>

    # 纯文本（Qwen3）：比 TTFT 与 decode 吞吐
    python3 tools/benchmark_vs.py text <hf_dir>
"""
import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO = ROOT / "build" / "demo"
TEXT_PROMPT = "What is AI?"
OCR_PROMPT = "请识别图中的文字内容。"
SEP = "-" * 40
FRAMEWORKS = ["KLiteInfer", "HF Transformers"]


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def timed(fn, repeat):
    """跑 repeat 次取中位数耗时（秒）"""
    return median([_one(fn) for _ in range(repeat)])


def _one(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# =============================================================== 纯文本
def text_klite(model_bin, tokenizer, repeat):
    """demo 自己就打印分段耗时，直接解析"""
    ttft, tps = [], []
    for _ in range(repeat):
        r = subprocess.run([str(DEMO), "qwen3", model_bin, tokenizer, TEXT_PROMPT],
                           capture_output=True, text=True)
        m_ttft = re.search(r"TTFT\s*:\s*([\d.]+)\s*ms", r.stdout)
        m_tps = re.search(r"Decode\s*:\s*[\d.]+\s*s\s*\(([\d.]+)\s*tok/s\)", r.stdout)
        if not (m_ttft and m_tps):
            raise RuntimeError("无法解析 KLite 输出")
        ttft.append(float(m_ttft.group(1)))
        tps.append(float(m_tps.group(1)))
    return {"ttft_ms": median(ttft), "decode_tps": median(tps)}


def text_hf(model_dir, n_tokens, repeat):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval().cuda()
    text = tok.apply_chat_template([{"role": "user", "content": TEXT_PROMPT}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")

    def run(max_new):
        with torch.inference_mode():
            model.generate(**ids, max_new_tokens=max_new, min_new_tokens=max_new,
                           do_sample=False, use_cache=True)
        torch.cuda.synchronize()

    run(4)  # warmup
    # decode 吞吐用两次运行的差值算，避免把 prefill 混进稳态速率
    ttft = timed(lambda: run(1), repeat)
    total = timed(lambda: run(n_tokens), repeat)
    return {"ttft_ms": ttft * 1000, "decode_tps": (n_tokens - 1) / max(total - ttft, 1e-9)}


# =============================================================== 多模态
def ocr_klite(ref_dir, n_tokens, repeat):
    """KLite 读取 ocr_eval.py 产出的预处理结果（pixel_values.bin + meta.json）"""
    vision, inner, text = [], [], None
    for _ in range(repeat):
        r = subprocess.run([str(DEMO), "paddleocr", "", "", str(ref_dir), str(n_tokens)],
                           capture_output=True, text=True)
        m_v = re.search(r"Vision encode\s*:\s*([\d.]+)\s*ms", r.stdout)
        m_t = re.search(r"Total\s*:\s*([\d.]+)\s*s", r.stdout)
        if m_v:
            vision.append(float(m_v.group(1)))
        if m_t:
            inner.append(float(m_t.group(1)))
        text = r.stdout.split(SEP)[0].strip()
    # 取 demo 自报的 Total（不含进程启动与权重 mmap），与 HF侧的纯推理耗时可比
    return {"e2e_s": median(inner), "vision_ms": median(vision), "text": text}


def ocr_hf(model_dir, image_path, n_tokens, repeat):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32).eval().cuda()
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    dev = {k: v.to("cuda") for k, v in inputs.items()}
    out = {}

    def run():
        with torch.inference_mode():
            gen = model.generate(**dev, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        out["text"] = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0].strip()

    e2e = timed(run, repeat)
    return {"e2e_s": e2e, "vision_ms": float("nan"), "text": out.get("text", "")}


# =============================================================== 汇总
def report(mode, results, n_tokens):
    if mode == "text":
        print(f"\n\n### 文本生成（fp32, 贪心, batch=1, {n_tokens} tokens）\n")
        print("| 框架 | TTFT | decode 吞吐 |")
        print("| --- | --: | --: |")
        for k in FRAMEWORKS:
            r = results.get(k)
            if r:
                print(f"| {k} | {r['ttft_ms']:.0f} ms | {r['decode_tps']:.1f} tok/s |")
        return

    print(f"\n\n### 多模态 OCR（fp32, 贪心, batch=1, 生成 {n_tokens} tokens）\n")
    print("| 框架 | 视觉编码 | 端到端 | 图像吞吐 |")
    print("| --- | --: | --: | --: |")
    for k in FRAMEWORKS:
        r = results.get(k)
        if not r:
            continue
        vis = "—" if r["vision_ms"] != r["vision_ms"] else f"{r['vision_ms']:.0f} ms"
        print(f"| {k} | {vis} | {r['e2e_s']:.2f} s | {60.0 / r['e2e_s']:.1f} img/min |")

    a, b = results.get(FRAMEWORKS[0]), results.get(FRAMEWORKS[1])
    if a and b:
        print(f"\n加速比 **{b['e2e_s'] / a['e2e_s']:.1f}×**；输出"
              f"{'逐字一致' if a.get('text') == b.get('text') else '存在差异'}"
              f"（{len(a.get('text', ''))} 字符）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "ocr"])
    ap.add_argument("hf_model_dir")
    ap.add_argument("image", nargs="?", default="", help="ocr 模式的输入图片")
    ap.add_argument("--ref-dir", default="", help="ocr 模式下 KLite 的预处理产物目录")
    ap.add_argument("--klite-bin", default="")
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", default="", help="klite / hf")
    ap.add_argument("--out", default="", help="结果 json，便于分批跑后合并")
    args = ap.parse_args()

    n_tokens = args.tokens or (128 if args.mode == "ocr" else 256)
    d = args.hf_model_dir
    results = {}
    if args.out and os.path.exists(args.out):
        results = json.load(open(args.out))

    if args.mode == "text":
        klite_bin = args.klite_bin or os.path.join(d, "klite_qwen3_0.6b.bin")
        jobs = {
            "KLiteInfer": lambda: text_klite(klite_bin, os.path.join(d, "tokenizer.json"),
                                             args.repeat),
            "HF Transformers": lambda: text_hf(d, n_tokens, args.repeat),
        }
    else:
        if not args.ref_dir:
            sys.exit("ocr 模式需要 --ref-dir（tools/ocr_eval.py 的输出目录）")
        jobs = {
            "KLiteInfer": lambda: ocr_klite(args.ref_dir, n_tokens, args.repeat),
            "HF Transformers": lambda: ocr_hf(d, args.image, n_tokens, args.repeat),
        }

    keys = {"klite": "KLiteInfer", "hf": "HF Transformers"}
    if args.only:
        jobs = {keys[args.only]: jobs[keys[args.only]]}

    for name, fn in jobs.items():
        print(f"\n>>> {name}", flush=True)
        results[name] = fn()
        brief = {k: v for k, v in results[name].items() if k != "text"}
        print(f"    {brief}", flush=True)

    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
    report(args.mode, results, n_tokens)


if __name__ == "__main__":
    main()
