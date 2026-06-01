#include "op/vision.h"
#include <glog/logging.h>
#include <cmath>
#include <cstring>

namespace op {

// =============================================================================
//  LayerNormLayer
// =============================================================================
LayerNormLayer::LayerNormLayer(base::DeviceType device_type, int32_t dim, float eps,
                               bool has_bias)
    : LayerParam(device_type, LayerType::kLayerRMSNorm /*复用 enum*/, false, "LayerNorm"),
      dim_(dim),
      eps_(eps),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(has_bias ? 2 : 1);  // gamma (+ beta)
}

base::Status LayerNormLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, base::DataType::kDataTypeFp32,
                                      dim_);
  if (!status) {
    LOG(ERROR) << "LayerNormLayer input check failed.";
    return status;
  }
  status = check_tensor_with_dim(get_output(0), device_type_, base::DataType::kDataTypeFp32,
                                 dim_);
  if (!status) {
    LOG(ERROR) << "LayerNormLayer output check failed.";
    return status;
  }
  status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeFp32,
                                 dim_);
  if (!status) {
    LOG(ERROR) << "LayerNormLayer weight(gamma) check failed.";
    return status;
  }
  if (has_bias_) {
    status = check_tensor_with_dim(get_weight(1), device_type_, base::DataType::kDataTypeFp32,
                                   dim_);
    if (!status) {
      LOG(ERROR) << "LayerNormLayer weight(beta) check failed.";
      return status;
    }
  }
  return base::error::Success();
}

base::Status LayerNormLayer::forward() {
  // 仅 CPU 实现；CUDA 实现可在后续接入对应 kernel
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    LOG(WARNING) << "LayerNormLayer CUDA path not yet implemented, fallback to CPU compute.";
  }

  const auto& input  = get_input(0);
  const auto& output = get_output(0);
  const auto& gamma  = get_weight(0);
  const float* x_ptr = input.ptr<float>();
  float*       y_ptr = const_cast<float*>(output.ptr<float>());
  const float* g_ptr = gamma.ptr<float>();
  const float* b_ptr = has_bias_ ? get_weight(1).ptr<float>() : nullptr;

  const size_t total = input.size();
  const int32_t D = dim_;
  CHECK_EQ(total % D, 0u);
  const size_t rows = total / D;

  for (size_t r = 0; r < rows; ++r) {
    const float* xr = x_ptr + r * D;
    float* yr = y_ptr + r * D;
    // mean
    double mean = 0.0;
    for (int32_t i = 0; i < D; ++i) mean += xr[i];
    mean /= D;
    // var
    double var = 0.0;
    for (int32_t i = 0; i < D; ++i) {
      double d = xr[i] - mean;
      var += d * d;
    }
    var /= D;
    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps_);
    for (int32_t i = 0; i < D; ++i) {
      float n = (xr[i] - static_cast<float>(mean)) * inv;
      n = n * g_ptr[i];
      if (b_ptr) n += b_ptr[i];
      yr[i] = n;
    }
  }
  return base::error::Success();
}

// =============================================================================
//  GELULayer  (近似实现: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715 x^3))))
// =============================================================================
GELULayer::GELULayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerSwiGLU /*复用 enum*/, "GELU") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status GELULayer::check() const {
  if (get_input(0).is_empty()) {
    return base::error::InvalidArgument("GELU: input is empty");
  }
  if (get_output(0).is_empty()) {
    return base::error::InvalidArgument("GELU: output is empty");
  }
  if (get_input(0).size() != get_output(0).size()) {
    return base::error::InvalidArgument("GELU: in/out size mismatch");
  }
  return base::error::Success();
}

base::Status GELULayer::forward() {
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    LOG(WARNING) << "GELULayer CUDA path not yet implemented, fallback to CPU compute.";
  }
  const float* x = get_input(0).ptr<float>();
  float* y = const_cast<float*>(get_output(0).ptr<float>());
  const size_t n = get_input(0).size();
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float kBeta  = 0.044715f;
  for (size_t i = 0; i < n; ++i) {
    float xv = x[i];
    float t  = kAlpha * (xv + kBeta * xv * xv * xv);
    y[i] = 0.5f * xv * (1.0f + std::tanh(t));
  }
  return base::error::Success();
}

// =============================================================================
//  PatchEmbedLayer
//  输入要求已经被 unfold 成 [num_patches, c*p*p]，权重为 [hidden, c*p*p]
//  实现：output[i, j] = sum_k input[i, k] * weight[j, k]
// =============================================================================
PatchEmbedLayer::PatchEmbedLayer(base::DeviceType device_type, int32_t in_channels,
                                 int32_t hidden_size, int32_t patch_size)
    : LayerParam(device_type, LayerType::kLayerMatmul /*复用 enum*/, false, "PatchEmbed"),
      in_channels_(in_channels),
      hidden_size_(hidden_size),
      patch_size_(patch_size) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status PatchEmbedLayer::check() const {
  if (get_input(0).is_empty() || get_output(0).is_empty() || get_weight(0).is_empty()) {
    return base::error::InvalidArgument("PatchEmbed: input/output/weight empty");
  }
  return base::error::Success();
}

base::Status PatchEmbedLayer::forward() {
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    LOG(WARNING) << "PatchEmbedLayer CUDA path not yet implemented, fallback to CPU compute.";
  }
  const auto& in  = get_input(0);
  const auto& wgt = get_weight(0);
  const auto& out = get_output(0);

  const int32_t patch_dim = in_channels_ * patch_size_ * patch_size_;
  const size_t  num_patches = in.size() / patch_dim;
  CHECK_EQ(in.size() % patch_dim, 0u);
  CHECK_EQ(wgt.size(), static_cast<size_t>(hidden_size_) * patch_dim);
  CHECK_EQ(out.size(), num_patches * hidden_size_);

  const float* x = in.ptr<float>();
  const float* w = wgt.ptr<float>();
  float*       y = const_cast<float*>(out.ptr<float>());

  // 朴素 GEMM：output = input @ weight^T
  for (size_t i = 0; i < num_patches; ++i) {
    for (int32_t j = 0; j < hidden_size_; ++j) {
      const float* xr = x + i * patch_dim;
      const float* wr = w + j * patch_dim;
      float s = 0.0f;
      for (int32_t k = 0; k < patch_dim; ++k) s += xr[k] * wr[k];
      y[i * hidden_size_ + j] = s;
    }
  }
  return base::error::Success();
}

}  // namespace op
