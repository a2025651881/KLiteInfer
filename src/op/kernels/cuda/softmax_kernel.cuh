#ifndef KLITE_CUDA_SOFTMAX_KERNEL_CUH
#define KLITE_CUDA_SOFTMAX_KERNEL_CUH
#include "tensor/tensor.h"

namespace kernel {
/**
 * @brief 一维 in-place softmax（CUDA）
 *
 * 与 softmax_inplace_cpu 语义一致：对整个张量做一次 softmax，结果原地写回。
 * 采用「减最大值」的稳定形式，避免 exp 溢出。
 */
void softmax_inplace_cu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel

#endif  // KLITE_CUDA_SOFTMAX_KERNEL_CUH
