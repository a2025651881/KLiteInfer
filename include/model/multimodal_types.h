#ifndef KUIPER_INCLUDE_MODEL_MULTIMODAL_TYPES_H_
#define KUIPER_INCLUDE_MODEL_MULTIMODAL_TYPES_H_

#include <cstdint>
#include "tensor/tensor.h"

namespace model {

/**
 * @brief 图像 patch 网格 (t, h, w)，单位为 patch
 */
struct ImageGridTHW {
  int32_t t = 1;
  int32_t h = 0;
  int32_t w = 0;

  int32_t num_patches() const { return t * h * w; }

  int32_t num_img_tokens(int32_t merge) const {
    return t * (h / merge) * (w / merge);
  }
};

/**
 * @brief 经过预处理 / 归一化后的图像
 */
struct ProcessedImage {
  tensor::Tensor pixel_values;  // [t, c, h, w] / 已 normalize
  ImageGridTHW   grid_thw;      // patch grid
};

/**
 * @brief 3D-MRoPE 位置输出
 *  positions shape: [3, seq_len], int32 (CPU)
 *  3 维分别表示 (t, h, w)
 */
struct MRoPEPositions {
  tensor::Tensor positions;
  int32_t        mrope_position_delta = 0;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_MULTIMODAL_TYPES_H_
