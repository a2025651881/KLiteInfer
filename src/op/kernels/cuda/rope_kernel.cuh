#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream,
                    bool interleaved);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, float theta, cudaStream_t stream);

/// 见 rope_half_split_cpu；用于 3D-MRoPE 的逐 token cos/sin
void rope_half_split_cu(float* vec, int32_t heads, int32_t head_size, const float* cos_vec,
                        const float* sin_vec, void* stream = nullptr);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
