#ifndef KLITE_CUDA_VISION_KERNEL_CUH
#define KLITE_CUDA_VISION_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include "../cpu/vision_kernel.h"  // 复用 GeluKind

namespace kernel {

/// 与 vision_gemm_nt_cpu 语义一致：Y[N,M] = X[N,K] * W[M,K]^T + b[M]
void vision_gemm_nt_cu(const float* X, const float* W, const float* b, int32_t N, int32_t K,
                       int32_t M, float* Y, void* stream = nullptr);

void vision_layernorm_cu(const float* x, int32_t rows, int32_t dim, const float* gamma,
                         const float* beta, float eps, float* out, void* stream = nullptr);

void vision_gelu_cu(float* x, size_t n, GeluKind kind, void* stream = nullptr);

void vision_rope2d_cu(float* q, float* k, const float* cos_tab, const float* sin_tab, int32_t n,
                      int32_t dim, int32_t heads, int32_t head_dim, void* stream = nullptr);

void vision_attention_cu(const float* q, const float* k, const float* v, int32_t n, int32_t dim,
                         int32_t heads, int32_t head_dim, float* out, float* score_buf,
                         int32_t score_rows, void* stream = nullptr);

/**
 * @brief CUDA 版所需的 score 行数（行长 n）= chunk * n
 *
 * chunk 为一次 batched GEMM 处理的 head 数。整块 [heads, n, n] 在大图上会突破
 * int32 元素上限（13200 patches 时 16*13200^2 = 2.8e9 > 2^31-1），因此按 head 分块。
 */
int32_t vision_attention_score_rows_cu(int32_t n, int32_t heads);

void vision_residual_cu(float* y, const float* x, size_t n, void* stream = nullptr);

void vision_pos_embed_cu(const float* table, int32_t g, int32_t h, int32_t w, int32_t dim,
                         int32_t t, float* out, void* stream = nullptr);

void vision_spatial_merge_cu(const float* in, int32_t t, int32_t h, int32_t w, int32_t dim,
                             int32_t m, float* out, void* stream = nullptr);

}  // namespace kernel

#endif  // KLITE_CUDA_VISION_KERNEL_CUH
