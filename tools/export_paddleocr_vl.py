#!/usr/bin/env python3
"""把 HuggingFace 的 PaddleOCR-VL 权重导出为 KLite 的扁平 fp32 权重文件。

用法:
    python3 tools/export_paddleocr_vl.py <hf_model_dir> <output.bin> [--seq-len 4096]

设计依据（对照 modeling_paddleocr_vl.py 的实际推理路径）:
  * patch_embedding 是 Conv2d(3,1152,k=14,s=14) 且带 bias，因输入恰为14x14，
    等价于对 588=3*14*14 维做Linear，故这里把 [1152,3,14,14] 展平成 [1152,588]。
  * 推理时 interpolate_pos_encoding=True，只用 position_embedding(729,1152)
    做 bilinear 插值；packing_position_embedding(32768,1152) 是死权重，不导出。
  * return_pooler_output=False，MAP pooling head（head.probe / head.attention /
    head.layernorm / head.mlp）不参与推理，是死权重，不导出。
  * 视觉 attention 的 q/k/v/out_proj 与 mlp.fc1/fc2 全部带 bias。
  * 文本 decoder 无 q/k norm、无 bias，lm_head 不与 embedding 共享。

文件结构（小端，头部为 int32 数组，其后为紧排 fp32 权重）:
  头部24 个 int32:
     0 magic("KLVL") 1 version
     2 vis_hidden 3 vis_layers 4 vis_heads 5 vis_inter
     6 vis_patch7 vis_merge  8 vis_pos_grid
     9 txt_dim10 txt_inter 11 txt_layers 12 txt_heads
    13 txt_kv_heads 14 txt_head_size 15 txt_vocab 16 txt_seq_len
    17 image_token_id 18 vision_start_token_id
    19 mrope_t 20 mrope_h 21 mrope_w
    22 shared_lm_head(0/1) 23 reserved
  权重区顺序见 write_all() 中的注释，必须与
  PaddleOCRVLModel::create_param_layers() 严格一致。
"""
import argparse
import json
import os
import struct
import sys

import torch
from safetensors.torch import load_file

KLVL_MAGIC = 0x4C564C4B  # "KLVL"
KLVL_VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("output")
    ap.add_argument("--seq-len", type=int, default=4096, help="KV-Cache 容量")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    vcfg = cfg["vision_config"]

    vis_hidden = vcfg["hidden_size"]
    vis_layers = vcfg["num_hidden_layers"]
    vis_heads = vcfg["num_attention_heads"]
    vis_inter = vcfg["intermediate_size"]
    vis_patch = vcfg["patch_size"]
    vis_merge = vcfg["spatial_merge_size"]
    vis_pos_grid = vcfg["image_size"] // vis_patch  # 384 // 14 = 27

    txt_dim = cfg["hidden_size"]
    txt_inter = cfg["intermediate_size"]
    txt_layers = cfg["num_hidden_layers"]
    txt_heads = cfg["num_attention_heads"]
    txt_kv_heads = cfg["num_key_value_heads"]
    txt_head_size = cfg["head_dim"]
    txt_vocab = cfg["vocab_size"]
    seq_len = min(args.seq_len, cfg.get("max_position_embeddings", args.seq_len))
    shared = 1 if cfg.get("tie_word_embeddings", False) else 0

    mrope = cfg.get("rope_scaling", {}).get("mrope_section", [16, 24, 24])
    merged = vis_hidden * vis_merge * vis_merge  # 4608

    print(f"vision: hidden={vis_hidden} layers={vis_layers} heads={vis_heads} "
          f"inter={vis_inter} patch={vis_patch} merge={vis_merge} pos_grid={vis_pos_grid}")
    print(f"text  : dim={txt_dim} inter={txt_inter} layers={txt_layers} heads={txt_heads} "
          f"kv={txt_kv_heads} head_size={txt_head_size} vocab={txt_vocab} seq_len={seq_len}")
    print(f"tokens: image={cfg['image_token_id']} vision_start={cfg['vision_start_token_id']} "
          f"mrope={mrope} shared_lm_head={bool(shared)}")

    state = load_file(os.path.join(args.model_dir, "model.safetensors"))

    written = [0]

    def emit(name, expect_shape, reshape=None):
        if name not in state:
            sys.exit(f"权重中缺少 {name}")
        t = state[name]
        if reshape is not None:
            t = t.reshape(reshape)
        if tuple(t.shape) != tuple(expect_shape):
            sys.exit(f"{name} 形状不符: 期望 {tuple(expect_shape)}, 实际 {tuple(t.shape)}")
        fout.write(t.detach().to(torch.float32).contiguous().numpy().tobytes())
        written[0] += t.numel()

    with open(args.output, "wb") as fout:
        header = [
            KLVL_MAGIC, KLVL_VERSION,
            vis_hidden, vis_layers, vis_heads, vis_inter, vis_patch, vis_merge, vis_pos_grid,
            txt_dim, txt_inter, txt_layers, txt_heads, txt_kv_heads, txt_head_size,
            txt_vocab, seq_len,
            cfg["image_token_id"], cfg["vision_start_token_id"],
            mrope[0], mrope[1], mrope[2],
            shared, 0,
        ]
        fout.write(struct.pack("<24i", *header))

        # ---------------- 视觉编码器 ----------------
        # 1. patch embed: Conv2d 权重展平为 [hidden, C*p*p]
        emit("visual.vision_model.embeddings.patch_embedding.weight",
             (vis_hidden,3 * vis_patch * vis_patch),
             reshape=(vis_hidden, 3 * vis_patch * vis_patch))
        emit("visual.vision_model.embeddings.patch_embedding.bias", (vis_hidden,))
        # 2. 可学习位置编码（27x27），运行期bilinear 插值到 (h, w)
        emit("visual.vision_model.embeddings.position_embedding.weight",
             (vis_pos_grid * vis_pos_grid, vis_hidden))
        # 3. 27 层 encoder，逐层分组：ln1, q, k, v, o, ln2, fc1, fc2（权重后紧跟 bias）
        for i in range(vis_layers):
            p = f"visual.vision_model.encoder.layers.{i}"
            emit(f"{p}.layer_norm1.weight", (vis_hidden,))
            emit(f"{p}.layer_norm1.bias", (vis_hidden,))
            emit(f"{p}.self_attn.q_proj.weight", (vis_hidden, vis_hidden))
            emit(f"{p}.self_attn.q_proj.bias", (vis_hidden,))
            emit(f"{p}.self_attn.k_proj.weight", (vis_hidden, vis_hidden))
            emit(f"{p}.self_attn.k_proj.bias", (vis_hidden,))
            emit(f"{p}.self_attn.v_proj.weight", (vis_hidden, vis_hidden))
            emit(f"{p}.self_attn.v_proj.bias", (vis_hidden,))
            emit(f"{p}.self_attn.out_proj.weight", (vis_hidden, vis_hidden))
            emit(f"{p}.self_attn.out_proj.bias", (vis_hidden,))
            emit(f"{p}.layer_norm2.weight", (vis_hidden,))
            emit(f"{p}.layer_norm2.bias", (vis_hidden,))
            emit(f"{p}.mlp.fc1.weight", (vis_inter, vis_hidden))
            emit(f"{p}.mlp.fc1.bias", (vis_inter,))
            emit(f"{p}.mlp.fc2.weight", (vis_hidden, vis_inter))
            emit(f"{p}.mlp.fc2.bias", (vis_hidden,))
        # 4. post_layernorm
        emit("visual.vision_model.post_layernorm.weight", (vis_hidden,))
        emit("visual.vision_model.post_layernorm.bias", (vis_hidden,))

        # ---------------- Projector (mlp_AR) ----------------
        # pre_norm 作用在 2x2 merge 之前，eps=1e-5
        emit("mlp_AR.pre_norm.weight", (vis_hidden,))
        emit("mlp_AR.pre_norm.bias", (vis_hidden,))
        emit("mlp_AR.linear_1.weight", (merged, merged))
        emit("mlp_AR.linear_1.bias", (merged,))
        emit("mlp_AR.linear_2.weight", (txt_dim, merged))
        emit("mlp_AR.linear_2.bias", (txt_dim,))

        # ---------------- 文本解码器 ----------------
        emit("model.embed_tokens.weight", (txt_vocab, txt_dim))
        q_dim = txt_heads * txt_head_size
        kv_dim = txt_kv_heads * txt_head_size
        for i in range(txt_layers):
            p = f"model.layers.{i}"
            emit(f"{p}.input_layernorm.weight", (txt_dim,))
            emit(f"{p}.self_attn.q_proj.weight", (q_dim, txt_dim))
            emit(f"{p}.self_attn.k_proj.weight", (kv_dim, txt_dim))
            emit(f"{p}.self_attn.v_proj.weight", (kv_dim, txt_dim))
            emit(f"{p}.self_attn.o_proj.weight", (txt_dim, q_dim))
            emit(f"{p}.post_attention_layernorm.weight", (txt_dim,))
            emit(f"{p}.mlp.gate_proj.weight", (txt_inter, txt_dim))
            emit(f"{p}.mlp.down_proj.weight", (txt_dim, txt_inter))
            emit(f"{p}.mlp.up_proj.weight", (txt_inter, txt_dim))
        emit("model.norm.weight", (txt_dim,))
        if not shared:
            emit("lm_head.weight", (txt_vocab, txt_dim))

    size_gb = os.path.getsize(args.output) / (1 << 30)
    print(f"导出完成: {args.output}  参数量={written[0]/1e6:.1f}M  文件={size_gb:.2f} GiB")

    skipped = [k for k in state
               if "packing_position_embedding" in k or ".head." in k or k.startswith("visual.vision_model.head")]
    if skipped:
        print(f"已跳过 {len(skipped)} 个推理不使用的死权重（packing_position_embedding / MAP head）")


if __name__ == "__main__":
    main()
