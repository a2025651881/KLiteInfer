#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include <base/cuda_config.h>
#include "cpu/vision_kernel.h"  // GeluKind
#include "tensor/tensor.h"
namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, const CudaConfig* config);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                          CudaConfig*);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernelDim)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, int32_t dim, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream, bool interleaved);

typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

/// 3D-MRoPE：按逐 token 的 cos/sin 向量做半分割配对旋转
typedef void (*RopeHalfSplitKernel)(float* vec, int32_t heads, int32_t head_size,
                                    const float* cos_vec, const float* sin_vec, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value, const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size, int stride,
                               void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

// -----------------------------------------------------------------------------
//  视觉 encoder / projector 用的批量 kernel（行主序 [rows, dim]）
// -----------------------------------------------------------------------------
typedef void (*VisionGemmNTKernel)(const float* X, const float* W, const float* b, int32_t N,
                                   int32_t K, int32_t M, float* Y, void* stream);

typedef void (*VisionLayerNormKernel)(const float* x, int32_t rows, int32_t dim,
                                      const float* gamma, const float* beta, float eps,
                                      float* out, void* stream);

typedef void (*VisionGeluKernel)(float* x, size_t n, GeluKind kind, void* stream);

typedef void (*VisionRope2dKernel)(float* q, float* k, const float* cos_tab, const float* sin_tab,
                                   int32_t n, int32_t dim, int32_t heads, int32_t head_dim,
                                   void* stream);

typedef void (*VisionAttentionKernel)(const float* q, const float* k, const float* v, int32_t n,
                                int32_t dim, int32_t heads, int32_t head_dim, float* out,
                                      float* score_buf, int32_t score_rows, void* stream);

/// score 缓冲需要的行数（行长 n）；大图上 CUDA 版会按 head 分块
int32_t vision_attention_score_rows(base::DeviceType device_type, int32_t n, int32_t heads);

typedef void (*VisionResidualKernel)(float* y, const float* x, size_t n, void* stream);

typedef void (*VisionPosEmbedKernel)(const float* table, int32_t g, int32_t h, int32_t w,
                                     int32_t dim, int32_t t, float* out, void* stream);

typedef void (*VisionSpatialMergeKernel)(const float* in, int32_t t, int32_t h, int32_t w,
                                         int32_t dim, int32_t m, float* out, void* stream);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

RopeHalfSplitKernel get_rope_half_split_kernel(base::DeviceType device_type);

ScaleKernel get_scale_kernel(base::DeviceType device_type);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type);

VisionGemmNTKernel get_vision_gemm_nt_kernel(base::DeviceType device_type);

VisionLayerNormKernel get_vision_layernorm_kernel(base::DeviceType device_type);

VisionGeluKernel get_vision_gelu_kernel(base::DeviceType device_type);

VisionRope2dKernel get_vision_rope2d_kernel(base::DeviceType device_type);

VisionAttentionKernel get_vision_attention_kernel(base::DeviceType device_type);

VisionResidualKernel get_vision_residual_kernel(base::DeviceType device_type);

VisionPosEmbedKernel get_vision_pos_embed_kernel(base::DeviceType device_type);

VisionSpatialMergeKernel get_vision_spatial_merge_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // KERNELS_INTERFACE_H