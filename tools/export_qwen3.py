#!/usr/bin/env python3
"""把 HuggingFace 格式的 Qwen3 权重导出为 KLite 的扁平 fp32 权重文件。

用法:
    python3 tools/export_qwen3.py<hf_model_dir> <output.bin> [--seq-len 2048]

输出文件结构（小端）:
    [扩展头 10 x int32]
        magic("KLIT"), version, dim, hidden_dim(FFN中间维), layer_num,
        head_num, kv_head_num, head_size, vocab_size, seq_len
    [fp32 权重区，紧排]
        1. token embedding      [vocab, dim]
        2. attention rmsnorm    L x [dim]
        3. wq                   L x [q_dim, dim]
        4. wk                   L x [kv_dim, dim]
        5. wv                   L x [kv_dim, dim]
        6. wo                   L x [dim, q_dim]
        7. ffn rmsnorm          L x [dim]
        8. w1 (gate_proj)       L x [inter, dim]
        9. w2 (down_proj)       L x [dim, inter]
       10. w3 (up_proj)         L x [inter, dim]
       11. final rmsnorm[dim]
       12. q rmsnorm            L x [head_size]
       13. k rmsnorm            L x [head_size]
       14. lm_head              [vocab, dim]（tie_word_embeddings 时省略）

    vocab_size 取正值表示 lm_head 与 embedding 共享权重，取负值表示不共享。
    该布局必须与 Qwen3Model::create_param_layers() 保持一致。
"""
import argparse
import json
import os
import struct
import sys

import torch
from safetensors.torch import load_file

KLITE_MAGIC = 0x54494C4B  # "KLIT"
KLITE_VERSION = 1


def load_state_dict(model_dir):
    """加载 safetensors 权重，支持单文件与分片。"""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            state.update(load_file(os.path.join(model_dir, shard)))
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            sys.exit(f"未找到权重文件: {single}")
        state = load_file(single)
    return state


def write_tensor(fout, tensor, expect_shape, name):
    if tuple(tensor.shape) != tuple(expect_shape):
        sys.exit(f"{name} 形状不符: 期望 {tuple(expect_shape)}，实际 {tuple(tensor.shape)}")
    fout.write(tensor.detach().to(torch.float32).contiguous().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="HuggingFace模型目录")
    parser.add_argument("output", help="输出的 .bin 路径")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="KV-Cache 容量；HF 的 max_position_embeddings 通常过大")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)

    dim = cfg["hidden_size"]
    inter_dim = cfg["intermediate_size"]
    layer_num = cfg["num_hidden_layers"]
    head_num = cfg["num_attention_heads"]
    kv_head_num = cfg["num_key_value_heads"]
    head_size = cfg.get("head_dim") or dim // head_num
    vocab_size = cfg["vocab_size"]
    shared = bool(cfg.get("tie_word_embeddings", False))
    seq_len = min(args.seq_len, cfg.get("max_position_embeddings", args.seq_len))

    q_dim = head_num * head_size
    kv_dim = kv_head_num * head_size

    print(f"dim={dim} inter={inter_dim} layers={layer_num} heads={head_num} "
          f"kv_heads={kv_head_num} head_size={head_size} vocab={vocab_size} "
          f"seq_len={seq_len} shared_lm_head={shared}")

    state = load_state_dict(args.model_dir)

    def get(name):
        if name not in state:
            sys.exit(f"权重中缺少 {name}")
        return state[name]

    with open(args.output, "wb") as fout:
        fout.write(struct.pack(
            "<10i", KLITE_MAGIC, KLITE_VERSION, dim, inter_dim, layer_num, head_num,
            kv_head_num, head_size, vocab_size if shared else -vocab_size, seq_len))

        # 1. token embedding
        write_tensor(fout, get("model.embed_tokens.weight"), (vocab_size, dim), "embed_tokens")

        # 2. attention rmsnorm
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.input_layernorm.weight"), (dim,),
                         f"layers.{i}.input_layernorm")

        # 3~6. attention 投影
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.q_proj.weight"), (q_dim, dim),
                         f"layers.{i}.q_proj")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.k_proj.weight"), (kv_dim, dim),
                         f"layers.{i}.k_proj")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.v_proj.weight"), (kv_dim, dim),
                         f"layers.{i}.v_proj")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.o_proj.weight"), (dim, q_dim),
                         f"layers.{i}.o_proj")

        # 7. ffn rmsnorm
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.post_attention_layernorm.weight"), (dim,),
                         f"layers.{i}.post_attention_layernorm")

        # 8~10. feed forward
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.mlp.gate_proj.weight"), (inter_dim, dim),
                         f"layers.{i}.gate_proj")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.mlp.down_proj.weight"), (dim, inter_dim),
                         f"layers.{i}.down_proj")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.mlp.up_proj.weight"), (inter_dim, dim),
                         f"layers.{i}.up_proj")

        # 11. final rmsnorm
        write_tensor(fout, get("model.norm.weight"), (dim,), "model.norm")

        # 12~13. q/k rmsnorm（Qwen3 独有，逐 head 归一化）
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.q_norm.weight"), (head_size,),
                         f"layers.{i}.q_norm")
        for i in range(layer_num):
            write_tensor(fout, get(f"model.layers.{i}.self_attn.k_norm.weight"), (head_size,),
                         f"layers.{i}.k_norm")

        # 14. lm_head
        if not shared:
            write_tensor(fout, get("lm_head.weight"), (vocab_size, dim), "lm_head")

    size_gb = os.path.getsize(args.output) / (1 << 30)
    print(f"导出完成: {args.output} ({size_gb:.2f} GiB)")


if __name__ == "__main__":
    main()
