#ifndef KUIPER_INCLUDE_OP_VISION_H_
#define KUIPER_INCLUDE_OP_VISION_H_
#include "base/base.h"
#include "layer.h"
#include <cstdint>

namespace op {

// =============================================================================
//  LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
//  weight[0] = gamma, weight[1] = beta (可选, has_bias=true 时存在)
// =============================================================================
class LayerNormLayer : public LayerParam {
 public:
  explicit LayerNormLayer(base::DeviceType device_type, int32_t dim, float eps = 1e-6f,
                          bool has_bias = true);

  base::Status check() const override;

  base::Status forward() override;

  float eps() const { return eps_; }

 private:
  int32_t dim_     = 0;
  float   eps_     = 1e-6f;
  bool    has_bias_ = true;
};

// =============================================================================
//  GELU 激活 (按 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) 近似计算)
//  无参，输入输出形状一致；支持 in-place（input == output）
// =============================================================================
class GELULayer : public Layer {
 public:
  explicit GELULayer(base::DeviceType device_type);

  base::Status check() const override;

  base::Status forward() override;
};

// =============================================================================
//  Patch Embedding (Conv2d, kernel=patch, stride=patch, padding=0)
//  实现等价于：把图像按 [c, p, p] 展开为每个 patch 的 c*p*p 维向量后做 matmul
//
//  输入  : pixel_values  shape = [num_patches, in_channels * patch_size * patch_size]
//          (调用方负责把原图 unfold 成该形状)
//  权重  : weight        shape = [hidden_size, in_channels * patch_size * patch_size]
//  输出  : output        shape = [num_patches, hidden_size]
// =============================================================================
class PatchEmbedLayer : public LayerParam {
 public:
  explicit PatchEmbedLayer(base::DeviceType device_type, int32_t in_channels,
                           int32_t hidden_size, int32_t patch_size);

  base::Status check() const override;

  base::Status forward() override;

  int32_t in_channels() const { return in_channels_; }
  int32_t hidden_size() const { return hidden_size_; }
  int32_t patch_size()  const { return patch_size_; }

 private:
  int32_t in_channels_ = 0;
  int32_t hidden_size_ = 0;
  int32_t patch_size_  = 0;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_VISION_H_
