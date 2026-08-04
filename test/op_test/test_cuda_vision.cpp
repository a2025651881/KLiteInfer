#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "../src/op/kernels/kernels_interface.h"
#include "base/alloc.h"

namespace {

/// 视觉 kernel 的 CPU/CUDA 一致性对比：两条实现是数值对齐的唯一保证
class VisionKernelTest : public ::testing::Test {
 protected:
  std::vector<float> rand_vec(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
  }

  /// 把主机数据拷到显存，返回需手动 cudaFree 的指针
  float* to_device(const std::vector<float>& host) {
    float* dev = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev), sizeof(float) * host.size());
    cudaMemcpy(dev, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice);
    return dev;
  }

  std::vector<float> to_host(const float* dev, size_t n) {
    std::vector<float> host(n);
    cudaMemcpy(host.data(), dev, sizeof(float) * n, cudaMemcpyDeviceToHost);
    return host;
  }

  void expect_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      ASSERT_NEAR(a[i], b[i], tol) << "index " << i;
    }
  }
};

TEST_F(VisionKernelTest, gemm_nt_with_bias) {
  const int32_t N = 37, K = 64, M = 48;
  auto X = rand_vec(static_cast<size_t>(N) * K, 1);
  auto W = rand_vec(static_cast<size_t>(M) * K, 2);
  auto b = rand_vec(M, 3);

  std::vector<float> ref(static_cast<size_t>(N) * M, 0.f);
  kernel::get_vision_gemm_nt_kernel(base::DeviceType::kDeviceCPU)(X.data(), W.data(), b.data(), N,
                                                                  K, M, ref.data(), nullptr);

  float* dX = to_device(X);
  float* dW = to_device(W);
  float* db = to_device(b);
  float* dY = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dY), sizeof(float) *ref.size());
  kernel::get_vision_gemm_nt_kernel(base::DeviceType::kDeviceCUDA)(dX, dW, db, N, K, M, dY,
                                                                   nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dY, ref.size()), 1e-4f);
  cudaFree(dX); cudaFree(dW); cudaFree(db); cudaFree(dY);
}

TEST_F(VisionKernelTest, layernorm) {
  const int32_t rows = 20, dim = 96;
  auto x = rand_vec(static_cast<size_t>(rows) * dim, 4);
  auto gamma = rand_vec(dim, 5);
  auto beta = rand_vec(dim, 6);

  std::vector<float> ref(x.size());
  kernel::get_vision_layernorm_kernel(base::DeviceType::kDeviceCPU)(
      x.data(), rows, dim, gamma.data(), beta.data(), 1e-6f, ref.data(), nullptr);

  float* dx = to_device(x);
  float* dg = to_device(gamma);
  float* db = to_device(beta);
  float* dout = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dout), sizeof(float) * x.size());
  kernel::get_vision_layernorm_kernel(base::DeviceType::kDeviceCUDA)(dx, rows, dim, dg, db, 1e-6f,
                                                                     dout, nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dout, x.size()), 1e-5f);
  cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dout);
}

TEST_F(VisionKernelTest, gelu_tanh_and_erf) {
  for (auto kind : {kernel::GeluKind::kTanh, kernel::GeluKind::kErf}) {
    auto x = rand_vec(1000, 7);
    std::vector<float> ref = x;
    kernel::get_vision_gelu_kernel(base::DeviceType::kDeviceCPU)(ref.data(), ref.size(), kind,
                                                                 nullptr);
    float* dx = to_device(x);
    kernel::get_vision_gelu_kernel(base::DeviceType::kDeviceCUDA)(dx, x.size(), kind, nullptr);
    cudaDeviceSynchronize();
    expect_close(ref, to_host(dx, x.size()), 1e-6f);
    cudaFree(dx);
  }
}

TEST_F(VisionKernelTest, rope2d) {
  const int32_t n = 15, heads = 4, head_dim = 72, dim = heads * head_dim;
  const int32_t half = head_dim / 2;
  auto q = rand_vec(static_cast<size_t>(n) * dim, 8);
  auto k = rand_vec(static_cast<size_t>(n) * dim, 9);
  auto cos_tab = rand_vec(static_cast<size_t>(n) * half, 10);
  auto sin_tab = rand_vec(static_cast<size_t>(n) * half, 11);

  std::vector<float> q_ref = q, k_ref = k;
  kernel::get_vision_rope2d_kernel(base::DeviceType::kDeviceCPU)(
      q_ref.data(), k_ref.data(), cos_tab.data(), sin_tab.data(), n, dim, heads, head_dim, nullptr);

  float* dq = to_device(q);
  float* dk = to_device(k);
  float* dc = to_device(cos_tab);
  float* ds = to_device(sin_tab);
  kernel::get_vision_rope2d_kernel(base::DeviceType::kDeviceCUDA)(dq, dk, dc, ds, n, dim, heads,
                head_dim, nullptr);
  cudaDeviceSynchronize();
  expect_close(q_ref, to_host(dq, q.size()), 1e-5f);
  expect_close(k_ref, to_host(dk, k.size()), 1e-5f);
  cudaFree(dq); cudaFree(dk); cudaFree(dc); cudaFree(ds);
}

TEST_F(VisionKernelTest, bidirectional_attention) {
  const int32_t n = 33, heads = 3, head_dim = 24, dim = heads * head_dim;
  auto q = rand_vec(static_cast<size_t>(n) * dim, 12);
  auto k = rand_vec(static_cast<size_t>(n) * dim, 13);
  auto v = rand_vec(static_cast<size_t>(n) * dim, 14);

  std::vector<float> ref(static_cast<size_t>(n) * dim, 0.f);
  std::vector<float> score_cpu(static_cast<size_t>(heads) * n);
  kernel::get_vision_attention_kernel(base::DeviceType::kDeviceCPU)(
      q.data(), k.data(), v.data(), n, dim, heads, head_dim, ref.data(), score_cpu.data(),
      kernel::vision_attention_score_rows(base::DeviceType::kDeviceCPU, n, heads), nullptr);

  float* dq = to_device(q);
  float* dk = to_device(k);
  float* dv = to_device(v);
  float* dout = nullptr;
  float* dscore = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dout), sizeof(float) * ref.size());
  cudaMalloc(reinterpret_cast<void**>(&dscore), sizeof(float) * heads * n * n);
  const int32_t score_rows =
      kernel::vision_attention_score_rows(base::DeviceType::kDeviceCUDA, n, heads);
  kernel::get_vision_attention_kernel(base::DeviceType::kDeviceCUDA)(
      dq, dk, dv, n, dim, heads, head_dim, dout, dscore, score_rows, nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dout, ref.size()), 1e-5f);
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dscore);
}

TEST_F(VisionKernelTest, pos_embed_interpolate) {
  const int32_t g = 27, h = 8, w = 12, dim = 16, t = 2;
  auto table = rand_vec(static_cast<size_t>(g) * g * dim, 15);
  const size_t out_size = static_cast<size_t>(t) * h * w * dim;

  std::vector<float> ref(out_size, 0.f);
  kernel::get_vision_pos_embed_kernel(base::DeviceType::kDeviceCPU)(table.data(), g, h, w, dim, t,
                                                                    ref.data(), nullptr);

  float* dtable = to_device(table);
  float* dout = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dout), sizeof(float) * out_size);
  cudaMemset(dout, 0, sizeof(float) * out_size);
  kernel::get_vision_pos_embed_kernel(base::DeviceType::kDeviceCUDA)(dtable, g, h, w, dim, t, dout,
                                                                     nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dout, out_size), 1e-6f);
  cudaFree(dtable); cudaFree(dout);
}

TEST_F(VisionKernelTest, spatial_merge) {
  const int32_t t = 2, h = 6, w = 8, dim = 5, m = 2;
  auto in = rand_vec(static_cast<size_t>(t) * h * w * dim, 16);
  const size_t out_size = static_cast<size_t>(t) * (h / m) * (w / m) * m * m * dim;

  std::vector<float> ref(out_size, 0.f);
  kernel::get_vision_spatial_merge_kernel(base::DeviceType::kDeviceCPU)(in.data(), t, h, w, dim, m,
                                                                        ref.data(), nullptr);

  float* din = to_device(in);
  float* dout = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dout), sizeof(float) * out_size);
  kernel::get_vision_spatial_merge_kernel(base::DeviceType::kDeviceCUDA)(din, t, h, w, dim, m,
                                                                         dout, nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dout, out_size), 0.f);
  cudaFree(din); cudaFree(dout);
}

TEST_F(VisionKernelTest, mrope_half_split) {
  const int32_t heads = 16, head_size = 128, half = head_size / 2;
  auto vec = rand_vec(static_cast<size_t>(heads) * head_size, 17);
  auto cos_v = rand_vec(half, 18);
  auto sin_v = rand_vec(half, 19);

  std::vector<float> ref = vec;
  kernel::get_rope_half_split_kernel(base::DeviceType::kDeviceCPU)(
      ref.data(), heads, head_size, cos_v.data(), sin_v.data(), nullptr);

  float* dvec = to_device(vec);
  float* dc = to_device(cos_v);
  float* ds = to_device(sin_v);
  kernel::get_rope_half_split_kernel(base::DeviceType::kDeviceCUDA)(dvec, heads, head_size, dc, ds,
                                                                    nullptr);
  cudaDeviceSynchronize();
  expect_close(ref, to_host(dvec, vec.size()), 1e-5f);
  cudaFree(dvec); cudaFree(dc); cudaFree(ds);
}

}  // namespace
