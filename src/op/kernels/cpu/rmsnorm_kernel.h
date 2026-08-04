#ifndef LLAMA_INFER_RMSNORM_KERNEL_H
#define LLAMA_INFER_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);

/**
 * @brief 逐行 RMSNorm：input 形如 [rows, dim]，在最后一维上归一化
 *        用于 Qwen3 的 q_norm / k_norm（逐 head 归一化）
 */
void rmsnorm_kernel_cpu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t dim, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_RMSNORM_KERNEL_H