#ifndef KLITE_CUDA_SCALE_KERNEL_CUH
#define KLITE_CUDA_SCALE_KERNEL_CUH
#include "tensor/tensor.h"

namespace kernel {
/// in-place 数乘（CUDA），语义与 scale_inplace_cpu 一致
void scale_inplace_cu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}  // namespace kernel

#endif  // KLITE_CUDA_SCALE_KERNEL_CUH
