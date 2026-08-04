#!/usr/bin/env python3
"""比对 KLite C++ 的中间张量与 HF 参考实现的输出，定位数值偏差发生在哪一阶段。

用法:
    python3 tools/compare_paddleocr.py <ref_dir>

ref_dir 需同时包含:
    ref_*.npy    —— tools/dump_paddleocr_ref.py 产出的HF 参考
    klite_*.bin  —— ./demo paddleocr ... <ref_dir> 产出的 C++ 中间结果（fp32 紧排）
"""
import os
import sys

import numpy as np

# (阶段名, 参考文件, C++ 文件, rel_rms 上限, rel_max 上限)
#
# 误差随层数累积：视觉 encoder 有 27 层，BLAS 与 torch 的求和顺序不同，
# fp32 下越深的阶段相对误差越大，故按阶段分别给阈值。
STAGES = [
    ("patch embedding",      "ref_patch_embed.npy",   "klite_patch_embed.bin",   1e-6, 1e-4),
    ("+ position embedding", "ref_vis_embed.npy",     "klite_vis_embed.bin",     1e-6, 1e-4),
    ("vision layer 0",       "ref_vis_layer0.npy",    "klite_vis_layer0.bin",    1e-5, 1e-3),
    ("vision layer 1",       "ref_vis_layer1.npy",    "klite_vis_layer1.bin",    1e-5, 1e-3),
    ("vision layer last",    "ref_vis_layer_last.npy","klite_vis_layer_last.bin",5e-4, 1e-1),
    ("post_layernorm",       "ref_vis_post_ln.npy",   "klite_vis_post_ln.bin",   1e-4, 1e-2),
    ("projector",            "ref_projector.npy",     "klite_projector.bin",     1e-4, 1e-2),
    ("inputs_embeds",        "ref_inputs_embeds.npy", "klite_inputs_embeds.bin", 1e-4, 1e-2),
]


def build_ref_inputs_embeds(d, ref, n_rows):
    """重建 scatter 之后的参考 inputs_embeds。

    HF 侧的 hook 抓在视觉特征 scatter **之前**（image_token 行仍是 embed_tokens
    的原始查表值，207 行彼此完全相同），且因 use_cache=False 每步重算全序列，
    行数是最后一个 decode 步的长度。这里截到 prefill 长度并把 projector 输出
    按image_token 位置填回，才与 C++ 侧 dump 的语义一致。
    """
    import json
    meta_path = os.path.join(d, "meta.json")
    proj_path = os.path.join(d, "ref_projector.npy")
    if not (os.path.exists(meta_path) and os.path.exists(proj_path)):
        return ref
    meta = json.load(open(meta_path))
    ids = meta["input_ids"]
    # image_token_id 取自 config.json，缺省用 PaddleOCR-VL 的 100295
    image_token = meta.get("image_token_id", 100295)
    pos = [i for i, t in enumerate(ids) if t == image_token and i < n_rows]
    proj = np.load(proj_path).reshape(-1, ref.shape[-1])
    if len(pos) != proj.shape[0]:
        return ref
    out = ref.reshape(-1, ref.shape[-1])[:n_rows].copy()
    out[pos] = proj
    return out


def compare(name, ref, got, tol_rms, tol_max):
    ref = ref.astype(np.float64).ravel()
    got = got.astype(np.float64).ravel()
    n = min(ref.size, got.size)
    note = "" if ref.size == got.size else f"  [ref={ref.size} klite={got.size}，比较前 {n} 个]"
    a, b = ref[:n], got[:n]
    diff = np.abs(a - b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    # 各阶段数值量级差异巨大（1e-2 ~ 1e+2），用相对于该阶段 RMS 的误差判据
    scale = float(np.sqrt(np.mean(a * a))) + 1e-12
    rel_rms = float(np.sqrt(np.mean(diff * diff))) / scale
    rel_max = float(diff.max()) / scale
    ok = rel_rms < tol_rms and rel_max < tol_max and cos > 0.999999
    flag = "OK  " if ok else "FAIL"
    print(f"{flag} {name:22s} scale={scale:9.3e}  rel_rms={rel_rms:.3e}  "
          f"rel_max={rel_max:.3e}  cos={cos:.8f}{note}")
    return ok


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]

    print(f"参考目录: {d}\n")
    all_ok = True
    first_fail = None
    for name, ref_f, got_f, tol_rms, tol_max in STAGES:
        rp, gp = os.path.join(d, ref_f), os.path.join(d, got_f)
        if not os.path.exists(rp):
            print(f"SKIP {name:22s} 缺少参考 {ref_f}")
            continue
        if not os.path.exists(gp):
            print(f"SKIP {name:22s} 缺少 C++ 输出 {got_f}")
            continue
        ref = np.load(rp)
        got = np.fromfile(gp, dtype=np.float32)
        if name == "inputs_embeds":
            n_rows = got.size // ref.shape[-1]
            ref = build_ref_inputs_embeds(d, ref, n_rows)
        ok = compare(name, ref, got, tol_rms, tol_max)
        all_ok &= ok
        if not ok and first_fail is None:
            first_fail = name

    print()
    if all_ok:
        print("全部阶段数值一致 ✅")
    else:
        print(f"首个发散阶段: {first_fail} ← 从这里开始排查")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
