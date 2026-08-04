#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "scale_kernel.cuh"

namespace kernel {
namespace {
constexpr int kThreadPerBlock = 256;

__global__ void scale_inplace_kernel(float scale, float* data, int size) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < size) {
    data[idx] *= scale;
  }
}
}  // namespace

void scale_inplace_cu(float scale, const tensor::Tensor& tensor, void* stream) {
  CHECK(!tensor.is_empty());
  CHECK(tensor.device_type() == base::DeviceType::kDeviceCUDA);
  const int size = static_cast<int>(tensor.size());
  float* data = const_cast<float*>(tensor.ptr<float>());
  const int blocks = (size + kThreadPerBlock - 1) / kThreadPerBlock;
  if (stream != nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    scale_inplace_kernel<<<blocks, kThreadPerBlock, 0, cu_stream>>>(scale, data, size);
  } else {
    scale_inplace_kernel<<<blocks, kThreadPerBlock>>>(scale, data, size);
  }
}
}  // namespace kernel
