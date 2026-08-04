#include "rope_kernel.h"
#include <cmath>

namespace kernel {

void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                            float theta) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      *(sin_cache + pos * head_size + head_dim) = sinf(val);
      *(cos_cache + pos * head_size + head_dim) = cosf(val);
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream,
                     bool interleaved) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  const int32_t half = head_size / 2;
  const float* sin_ptr = sin_cache.ptr<float>();
  const float* cos_ptr = cos_cache.ptr<float>();

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t j = 0; j < half; ++j) {
      // 两种配对方式共用同一份cache：第 j 对的频率存放在槽位 2j
      const float fci = *(sin_ptr + pos * head_size + j * 2);
      const float fcr = *(cos_ptr + pos * head_size + j * 2);

      const int32_t v0_idx = interleaved ? i + j * 2 : i + j;
      const int32_t v1_idx = interleaved ? i + j * 2 + 1 : i + j + half;

      const int32_t rotn = i < kv_dim ? 2 : 1;  // 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; ++v) {
        float* vec = const_cast<float*>(v == 0 ? input_q.ptr<float>() : input_k.ptr<float>());
        const float v0 = vec[v0_idx];
        const float v1 = vec[v1_idx];
        vec[v0_idx] = v0 * fcr - v1 * fci;
        vec[v1_idx] = v0 * fci + v1 * fcr;
      }
    }
  }
}
void rope_half_split_cpu(float* vec, int32_t heads, int32_t head_size, const float* cos_vec,
                         const float* sin_vec, void* stream) {
  (void)stream;
  const int32_t half = head_size / 2;
  for (int32_t h = 0; h < heads; ++h) {
    float* p = vec + static_cast<size_t>(h) * head_size;
    for (int32_t j = 0; j < half; ++j) {
      const float x0 = p[j];
      const float x1 = p[j + half];
      p[j] = x0 * cos_vec[j] - x1 * sin_vec[j];
      p[j + half] = x1 * cos_vec[j] + x0 * sin_vec[j];
    }
  }
}
}  // namespace kernel
