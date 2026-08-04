#ifndef LLAMA_INFER_ROPE_KERNEL_H
#define LLAMA_INFER_ROPE_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
/**
 * @param theta RoPE 底数：llama2 系为 1e4，llama3 为 5e5，Qwen2/Qwen3 为 1e6
 */
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                            float theta);

/**
 * @param interleaved true  -> 相邻配对 (2j, 2j+1)，llama2.c / GGML 约定
 *                    false -> 半分割配对 (j, j + head_size/2)，HF / Qwen3 约定
 */
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream,
                     bool interleaved);

/**
 * @brief 按「逐 token 给定的 cos/sin 向量」做半分割配对旋转 (j, j + head_size/2)
 *
 * 与 rope_kernel_* 的区别：后者用标量 pos 去索引预计算的 cache，而 3D-MRoPE
 * 每个 token 的频率按 (t,h,w) 分段取值，无法用单一pos 表达，只能整段传入。
 *
 * @param vec       [heads, head_size]，原地旋转
 * @param cos_vec   [head_size/2]
 */
void rope_half_split_cpu(float* vec, int32_t heads, int32_t head_size, const float* cos_vec,
                         const float* sin_vec, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_ROPE_KERNEL_H
