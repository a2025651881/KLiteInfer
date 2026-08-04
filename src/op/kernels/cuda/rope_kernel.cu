#include "rope_kernel.cuh"
namespace kernel {

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache,
                                    bool interleaved) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  const int head_pair_count = head_size / 2;
  const int num_heads = dim / head_size;
  const int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  const int head_idx = idx / head_pair_count;
  const int j = idx % head_pair_count;
  const int i = head_idx * head_size;

  // 两种配对方式共用同一份 cache：第 j 对的频率存放在槽位 2j
  const float fci = sin_cache[pos * head_size + j * 2];
  const float fcr = cos_cache[pos * head_size + j * 2];

  const int v0_idx = interleaved ? i + j * 2 : i + j;
  const int v1_idx = interleaved ? i + j * 2 + 1 : i + j + head_pair_count;

  const int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; ++v) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);
    const float v0 = vec[v0_idx];
    const float v1 = vec[v1_idx];
    vec[v0_idx] = v0 * fcr - v1 * fci;
    vec[v1_idx] = v0 * fci + v1 * fcr;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                             float theta) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= head_size) {
    return;
  }
  const int head_dim = idx;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / powf(theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    *(sin_cache + pos * head_size + head_dim) = sinf(val);
    *(cos_cache + pos * head_size + head_dim) = cosf(val);
  }
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, float theta, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()), theta);
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()), theta);
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream,
                    bool interleaved) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  const int32_t total_pairs = dim / 2;
  int threads = 128;
  int blocks = (total_pairs + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>(), interleaved);
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>(), sin_cache.ptr<float>(),
                                             cos_cache.ptr<float>(), interleaved);
  }
}
namespace {
constexpr int kHalfSplitBlock = 128;

/// 一个线程处理一对 (j, j + half)，覆盖 heads * half
__global__ void rope_half_split_kernel(float* vec, int32_t heads, int32_t head_size,
                                       const float* cos_vec, const float* sin_vec) {
  const int32_t half = head_size / 2;
  const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= heads * half) {
    return;
  }
  const int32_t j = idx % half;
  const int32_t h = idx / half;
  float* p = vec + static_cast<size_t>(h) * head_size;
  const float c = cos_vec[j];
  const float s = sin_vec[j];
  const float x0 = p[j];
  const float x1 = p[j + half];
  p[j] = x0 * c - x1 * s;
  p[j + half] = x1 * c + x0 * s;
}
}  // namespace

void rope_half_split_cu(float* vec, int32_t heads, int32_t head_size, const float* cos_vec,
                        const float* sin_vec, void* stream) {
  const int32_t total = heads * (head_size / 2);
  const int32_t blocks = (total + kHalfSplitBlock - 1) / kHalfSplitBlock;
  auto cu_stream = static_cast<cudaStream_t>(stream);
  rope_half_split_kernel<<<blocks, kHalfSplitBlock, 0, cu_stream>>>(vec, heads, head_size, cos_vec,
                                                                   sin_vec);
}
}  // namespace kernel
