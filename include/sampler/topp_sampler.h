#ifndef KLITE_INCLUDE_SAMPLER_TOPP_SAMPLER_H
#define KLITE_INCLUDE_SAMPLER_TOPP_SAMPLER_H
#include <base/base.h>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include "sampler/sampler.h"

namespace sampler {

/**
 * @brief temperature + top-k + top-p(nucleus) 采样
 *
 * 采样在 CPU 上完成：当 logits 位于 CUDA 时会先拷回主机。
 * 对stories110M 这类小模型，纯 argmax 贪心解码会陷入重复，
 * 需要引入随机性。
 */
class TopPSampler : public Sampler {
 public:
  /**
   * @param temperature <=0 时退化为 argmax
   * @param topp        累积概率阈值，(0,1]；取 1 表示不做nucleus 截断
   * @param topk        候选数上限，<=0 表示不限制
   */
  explicit TopPSampler(base::DeviceType device_type, float temperature = 1.0f, float topp = 0.9f,
                       int32_t topk = 0, uint64_t seed = 42);

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  /// 把 logits 取到主机内存，返回可读指针
  const float* fetch_logits(const float* logits, size_t size, void* stream);

  float temperature_ = 1.0f;
  float topp_ = 0.9f;
  int32_t topk_ = 0;
  std::mt19937 rng_;
  std::vector<float> host_logits_;
  std::vector<std::pair<float, int32_t>> candidates_;
};

}  // namespace sampler
#endif  // KLITE_INCLUDE_SAMPLER_TOPP_SAMPLER_H
