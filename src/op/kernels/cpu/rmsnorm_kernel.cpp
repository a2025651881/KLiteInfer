
#include "rmsnorm_kernel.h"

namespace kernel {
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
static constexpr float kRmsNormEps = 1e-6f;
#else
static constexpr float kRmsNormEps = 1e-5f;
#endif

void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + kRmsNormEps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}

void rmsnorm_kernel_cpu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t dim, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK_GT(dim, 0);

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);
  CHECK_EQ(static_cast<int32_t>(weight.size()), dim);
  CHECK_EQ(input.size() % static_cast<size_t>(dim), 0u);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t rows = static_cast<int32_t>(input.size()) / dim;

  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);
  for (int32_t r = 0; r < rows; ++r) {
    arma::fvec in_row(const_cast<float*>(in_ptr) + static_cast<size_t>(r) * dim, dim, false, true);
    arma::fvec out_row(const_cast<float*>(out_ptr) + static_cast<size_t>(r) * dim, dim, false, true);

    const float mean = arma::as_scalar(arma::mean(arma::pow(in_row, 2))) + kRmsNormEps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_row = wei_tensor % (rsqrt * in_row);
  }
}
}  // namespace kernel
