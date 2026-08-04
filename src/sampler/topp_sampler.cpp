#include "sampler/topp_sampler.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <algorithm>
#include <cmath>

namespace sampler {

TopPSampler::TopPSampler(base::DeviceType device_type, float temperature, float topp, int32_t topk,
                         uint64_t seed)
    : Sampler(device_type),
      temperature_(temperature),
      topp_(topp),
      topk_(topk),
      rng_(static_cast<std::mt19937::result_type>(seed)) {}

const float* TopPSampler::fetch_logits(const float* logits, size_t size, void* stream) {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return logits;
  }
  host_logits_.resize(size);
  const size_t byte_size = size * sizeof(float);
  if (stream != nullptr) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(host_logits_.data(), logits, byte_size, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
  } else {
    cudaMemcpy(host_logits_.data(), logits, byte_size, cudaMemcpyDeviceToHost);
  }
  return host_logits_.data();
}

size_t TopPSampler::sample(const float* logits, size_t size, void* stream) {
  CHECK(logits != nullptr);
  CHECK_GT(size, 0u);
  const float* data = fetch_logits(logits, size, stream);

  // temperature <= 0：退化为贪心
  if (temperature_ <= 0.0f) {
    return static_cast<size_t>(std::distance(data, std::max_element(data, data + size)));
  }

  // 1. temperature 缩放 + 数值稳定的 softmax
  const float inv_temp = 1.0f / temperature_;
  float max_logit = data[0];
  for (size_t i = 1; i < size; ++i) {
    max_logit = std::max(max_logit, data[i]);
  }

  candidates_.resize(size);
  float sum_exp = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    const float p = std::exp((data[i] - max_logit) * inv_temp);
    candidates_[i] = {p, static_cast<int32_t>(i)};
    sum_exp += p;
  }
  if (!(sum_exp > 0.0f)) {
    return static_cast<size_t>(std::distance(data, std::max_element(data, data + size)));
  }
  const float inv_sum = 1.0f / sum_exp;
  for (auto& item : candidates_) {
    item.first *= inv_sum;
  }

  // 2. top-k 截断（未指定则全量排序）
  size_t keep = size;
  if (topk_ > 0 && static_cast<size_t>(topk_) < size) {
    keep = static_cast<size_t>(topk_);
    std::partial_sort(candidates_.begin(), candidates_.begin() + keep, candidates_.end(),
                      [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                        return a.first > b.first;
                      });
  } else {
    std::sort(candidates_.begin(), candidates_.end(),
              [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                return a.first > b.first;
              });
  }

  // 3. top-p(nucleus) 截断：保留累积概率刚好达到 topp 的最小集合
  float cumulative = 0.0f;
  size_t nucleus = keep;
  if (topp_ > 0.0f && topp_ < 1.0f) {
    for (size_t i = 0; i < keep; ++i) {
      cumulative += candidates_[i].first;
      if (cumulative >= topp_) {
        nucleus = i + 1;
        break;
      }
    }
  } else {
    for (size_t i = 0; i < keep; ++i) {
      cumulative += candidates_[i].first;
    }
  }

  // 4. 在截断后的分布上按概率采样
  std::uniform_real_distribution<float> dist(0.0f, cumulative);
  const float target = dist(rng_);
  float acc = 0.0f;
  for (size_t i = 0; i < nucleus; ++i) {
    acc += candidates_[i].first;
    if (acc >= target) {
      return static_cast<size_t>(candidates_[i].second);
    }
  }
  return static_cast<size_t>(candidates_[nucleus - 1].second);
}

}  // namespace sampler
