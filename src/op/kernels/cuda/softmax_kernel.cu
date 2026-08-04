#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cfloat>
#include "softmax_kernel.cuh"

namespace kernel {
namespace {
constexpr int kThreadPerBlock = 512;

/**
 * 单 block 完成整个向量的 softmax：
 *   1. grid-stride 求最大值 -> block内归约
 *   2. 原地写入 exp(x - max) 并求和 -> block 内归约
 *   3. 乘以 1 / sum
 * 用一个 block 是为了让三步共享同一份归约结果，无需额外的全局同步。
 */
__global__ void softmax_inplace_kernel(float* data, int size) {
  __shared__ float shared[kThreadPerBlock];
  const int tid = static_cast<int>(threadIdx.x);

  float local_max = -FLT_MAX;
  for (int i = tid; i < size; i += kThreadPerBlock) {
    local_max = fmaxf(local_max, data[i]);
  }
  shared[tid] = local_max;
  __syncthreads();
  for (int s = kThreadPerBlock / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }
  const float max_val = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = tid; i < size; i += kThreadPerBlock) {
    const float e = expf(data[i] - max_val);
    data[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (int s = kThreadPerBlock / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }
  const float sum_val = shared[0];
  __syncthreads();

  const float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < size; i += kThreadPerBlock) {
    data[i] *= inv_sum;
  }
}
}  // namespace

void softmax_inplace_cu(const tensor::Tensor& input, void* stream) {
  CHECK(!input.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  const int size = static_cast<int>(input.size());
  float* data = const_cast<float*>(input.ptr<float>());
  if (stream != nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    softmax_inplace_kernel<<<1, kThreadPerBlock, 0, cu_stream>>>(data, size);
  } else {
    softmax_inplace_kernel<<<1, kThreadPerBlock>>>(data, size);
  }
}
}  // namespace kernel
