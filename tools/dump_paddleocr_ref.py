#!/usr/bin/env python3
"""用 HuggingFace 参考实现跑一遍 PaddleOCR-VL，逐阶段 dump 中间张量，
作为 KLite C++ 实现的数值对齐基准。

用法:
    python3 tools/dump_paddleocr_ref.py <hf_model_dir> <out_dir> [--image xxx.png]

不指定 --image 时会自动生成一张带文字的测试图。

产出（out_dir 下）:
    meta.json                 形状/参数/token序列/grid_thw 等元信息
    pixel_values.bin          预处理后的 fp32 像素 [N, 3, 14, 14]（KLite 直接读这个）
    input_ids.bin             int32 token 序列
    ref_patch_embed.npy       patch embedding 输出（未加位置编码）[N, 1152]
    ref_pos_embed.npy         插值后的位置编码 [N, 1152]
    ref_vis_embed.npy         patch + pos [N, 1152]
    ref_vis_layer{0,1,-1}.npy 指定视觉层的输出
    ref_vis_post_ln.npy       post_layernorm 输出 [N, 1152]
    ref_projector.npy         projector 输出 [N/4, 1024]
    ref_inputs_embeds.npy     注入视觉特征后的文本 embedding [S, 1024]
    ref_position_ids.npy      3D-MRoPE 位置 [3, S]
    ref_logits_last.npy       最后一个位置的 logits [vocab]
    ref_generated.txt         贪心解码结果（对齐最终行为）
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw


def make_test_image(path):
    """生成一张带文字的测试图，避免依赖外部素材。"""
    img = Image.new("RGB", (448, 168), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 30), "Hello KLiteInfer", fill=(0, 0, 0))
    d.text((20, 70), "PaddleOCR-VL 2026", fill=(0, 0, 0))
    d.text((20, 110), "align check12345", fill=(0, 0, 0))
    img.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="请识别图中的文字内容。")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_path = args.image or make_test_image(os.path.join(args.out_dir, "test_image.png"))

    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32).eval()

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    print(f"pixel_values={tuple(pixel_values.shape)} grid_thw={grid_thw.tolist()} "
          f"input_ids={tuple(input_ids.shape)}")

    captured = {}

    visual = model.visual
    vision_model = visual.vision_model if hasattr(visual, "vision_model") else visual
    embeddings = vision_model.embeddings

    #---- hook: patch embedding（Conv2d）----
    def hook_patch(mod, inp, out):
        captured["patch_embed"] = out.detach().reshape(out.shape[0], -1).float().cpu().numpy()
    embeddings.patch_embedding.register_forward_hook(hook_patch)

    # ---- hook: embeddings 总输出（patch + pos）----
    def hook_emb(mod, inp, out):
        captured["vis_embed"] = out.detach().squeeze(0).float().cpu().numpy()
    embeddings.register_forward_hook(hook_emb)

    # ---- hook: 视觉 encoder 各层 ----
    layers = vision_model.encoder.layers
    watch = {0: "vis_layer0", 1: "vis_layer1", len(layers) - 1: "vis_layer_last"}
    for idx, key in watch.items():
        def make(key):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[key] = h.detach().squeeze(0).float().cpu().numpy()
            return hook
        layers[idx].register_forward_hook(make(key))

    # ---- hook: post_layernorm ----
    def hook_post(mod, inp, out):
        captured["vis_post_ln"] = out.detach().squeeze(0).float().cpu().numpy()
    vision_model.post_layernorm.register_forward_hook(hook_post)

    # ---- hook: projector ----
    def hook_proj(mod, inp, out):
        o = out[0] if isinstance(out, (list, tuple)) else out
        captured["projector"] = o.detach().float().cpu().numpy()
    model.mlp_AR.register_forward_hook(hook_proj)

    # ---- hook: 文本 embedding 注入结果 ----
    def hook_txt(mod, inp, out):
        captured["inputs_embeds"] = out.detach().squeeze(0).float().cpu().numpy()
    model.model.embed_tokens.register_forward_hook(hook_txt)

    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits.detach().squeeze(0).float().cpu().numpy()

    # 3D-MRoPE 位置
    with torch.no_grad():
        pos_ids, deltas = model.get_rope_index(
            input_ids, grid_thw, None, attention_mask=inputs.get("attention_mask"))
    pos_np = pos_ids.detach().squeeze(1).cpu().numpy().astype(np.int32)

    # 贪心解码，作为端到端行为基准
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_text = processor.batch_decode(gen[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("HF 贪心输出:", gen_text)

    #---------------- 落盘 ----------------
    pixel_values.detach().float().cpu().numpy().astype(np.float32).tofile(
        os.path.join(args.out_dir, "pixel_values.bin"))
    input_ids.detach().cpu().numpy().astype(np.int32).tofile(
        os.path.join(args.out_dir, "input_ids.bin"))

    for key, arr in captured.items():
        np.save(os.path.join(args.out_dir, f"ref_{key}.npy"), arr)
    np.save(os.path.join(args.out_dir, "ref_position_ids.npy"), pos_np)
    np.save(os.path.join(args.out_dir, "ref_logits_last.npy"), logits[-1])
    with open(os.path.join(args.out_dir, "ref_generated.txt"), "w") as f:
        f.write(gen_text)

    ids = input_ids[0].tolist()
    meta = {
        "image": image_path,
        "prompt": args.prompt,
        "grid_thw": grid_thw.tolist(),
        "num_patches": int(pixel_values.shape[0]),
        "pixel_values_shape": list(pixel_values.shape),
        "seq_len": len(ids),
        "input_ids": ids,
        "image_token_id": model.config.image_token_id,
        "vision_start_token_id": model.config.vision_start_token_id,
        "num_image_tokens": sum(1 for t in ids if t == model.config.image_token_id),
        "rope_deltas": int(deltas.flatten()[0]) if deltas is not None else 0,
        "dumped": sorted(captured.keys()),
        "generated": gen_text,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("已dump:", sorted(captured.keys()))
    print("输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
