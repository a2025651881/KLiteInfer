#ifndef KLITE_CPU_VISION_KERNEL_H
#define KLITE_CPU_VISION_KERNEL_H
#include <cstddef>
#include <cstdint>

namespace kernel {

/// GELU 变体：视觉 MLP 用 tanh 近似（gelu_pytorch_tanh），Projector 用精确 erf，
/// 两者不可混用，否则与参考实现对不齐。
enum class GeluKind : int32_t {
  kTanh = 0,
  kErf = 1,
};

// -----------------------------------------------------------------------------
//  以下 kernel 服务于视觉 encoder / projector，特点是「批量 token」而非单token
//  解码，因此都以行主序 [rows, dim] 布局工作，与面向自回归解码的 matmul kernel
//  （要求 [features, N] 转置布局）不同。
// -----------------------------------------------------------------------------

/**
 * @brief Y[N,M] = X[N,K] * W[M,K]^T + b[M]，全部行主序；b 可为 nullptr
 *
 * 对应 PyTorch 的 nn.Linear(weight=[M,K], bias=[M])。
 */
void vision_gemm_nt_cpu(const float* X, const float* W, const float* b, int32_t N, int32_t K,
                        int32_t M, float* Y, void* stream = nullptr);

///逐行 LayerNorm：y = (x - mean) / sqrt(var + eps) * gamma + beta（beta 可为 nullptr）
void vision_layernorm_cpu(const float* x, int32_t rows, int32_t dim, const float* gamma,
                          const float* beta, float eps, float* out, void* stream = nullptr);

/// in-place GELU
void vision_gelu_cpu(float* x, size_t n, GeluKind kind, void* stream = nullptr);

/**
 * @brief 视觉 2D RoPE：对 q、k 同时做半分割配对旋转 (d, d + head_dim/2)
 *
 * cos/sin 表布局为 [n, head_dim/2]，其中前后各半分别对应 h、w 方向的频率。
 */
void vision_rope2d_cpu(float* q, float* k, const float* cos_tab, const float* sin_tab, int32_t n,
                       int32_t dim, int32_t heads, int32_t head_dim, void* stream = nullptr);

/**
 * @brief 全可见（双向、无 mask、无 KV-Cache）多头注意力
 *
 * q/k/v/out 均为行主序 [n, dim]，第 h 个 head 占据每行的 [h*head_dim, (h+1)*head_dim)。
 *
 * @param score_buf  中间打分缓冲，容量为 score_rows 行 × n 列
 * @param score_rows 由vision_attention_score_rows() 给出。CPU 版每行独立复用，
 *                   只需 heads 行；CUDA 版一次算整块 [chunk, n, n]，需 chunk*n 行。
 *                   n 较大时（如 13200）必须分块，否则 heads*n*n 会超过 int32 上限。
 */
void vision_attention_cpu(const float* q, const float* k, const float* v, int32_t n, int32_t dim,
                          int32_t heads, int32_t head_dim, float* out, float* score_buf,
                          int32_t score_rows, void* stream = nullptr);

/// CPU 版所需的 score 行数：每个 head 逐行复用，只需 heads 行
int32_t vision_attention_score_rows_cpu(int32_t n, int32_t heads);

/// 残差累加：y[i] += x[i]
void vision_residual_cpu(float* y, const float* x, size_t n, void* stream = nullptr);

/**
 * @brief 位置编码 bilinear 插值并按帧累加到 out
 *
 * 把 [g, g, dim] 的学习表插值到 [h, w, dim]（align_corners=false，与
 * F.interpolate(mode="bilinear") 一致），再对 t 个帧逐一累加。
 */
void vision_pos_embed_cpu(const float* table, int32_t g, int32_t h, int32_t w, int32_t dim,
                          int32_t t, float* out, void* stream = nullptr);

/**
 * @brief 2x2 spatial merge：[t*h*w, dim] -> [t*(h/m)*(w/m), m*m*dim]
 *
 * 每个输出 token 内部按 (i, j) 行优先拼接 m*m 个 patch 的 dim 维整块。
 */
void vision_spatial_merge_cpu(const float* in, int32_t t, int32_t h, int32_t w, int32_t dim,
                              int32_t m, float* out, void* stream = nullptr);

}  // namespace kernel

#endif  // KLITE_CPU_VISION_KERNEL_H
