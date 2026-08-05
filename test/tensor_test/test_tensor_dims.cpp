// Tensor 维度计算的回归测试。
//
// 起因是一个真实 bug：13200patches 的大图上，视觉注意力打分张量形状为
// [heads*n, n] = [211200, 13200]，元素数 2.79e9 已越过 INT32_MAX，
// 而当时 size_ = dim0 * dim1 在 int32 域内相乘，回绕成负数，
// allocator 收到一个荒谬的 EB 级请求后直接失败。
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "base/alloc.h"
#include "tensor/tensor.h"

namespace {
// 只验证维度计算，不真的分配这么大内存：need_alloc=false + 外部指针
float* fake_ptr() {
  static float dummy = 0.f;
  return &dummy;
}

tensor::Tensor make_view(const std::vector<int32_t>& dims) {
  return tensor::Tensor(base::DataType::kDataTypeFp32, dims, false, nullptr, fake_ptr());
}
}  // namespace

TEST(test_tensor_dims, two_dim_product_exceeds_int32) {
  // 视觉注意力在n=13200 时的真实形状
  const int32_t rows = 211200, cols = 13200;
  tensor::Tensor t(base::DataType::kDataTypeFp32, rows, cols, false, nullptr, fake_ptr());

  const size_t expect = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  EXPECT_GT(expect, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  EXPECT_EQ(t.size(), expect);            // 2787840000
  EXPECT_EQ(t.byte_size(), expect * 4);   // 不应回绕成负数或小值
}

TEST(test_tensor_dims, three_and_four_dim_product_exceeds_int32) {
  const int32_t a = 16, b = 13200, c = 13200;
  tensor::Tensor t3(base::DataType::kDataTypeFp32, a, b, c, false, nullptr, fake_ptr());
  EXPECT_EQ(t3.size(), static_cast<size_t>(a) * b * c);

  tensor::Tensor t4(base::DataType::kDataTypeFp32, 2, a, b, c, false, nullptr, fake_ptr());
  EXPECT_EQ(t4.size(), static_cast<size_t>(2) * a * b * c);
}

// vector 版构造走reduce_dimension，累加器类型也必须是 size_t
TEST(test_tensor_dims, vector_ctor_product_exceeds_int32) {
  auto t = make_view({211200, 13200});
  EXPECT_EQ(t.size(), static_cast<size_t>(211200) * 13200);

  auto t3 = make_view({16, 13200, 13200});
  EXPECT_EQ(t3.size(), static_cast<size_t>(16) * 13200 * 13200);
}

TEST(test_tensor_dims, small_shapes_unaffected) {
  auto t = make_view({32, 151});
  EXPECT_EQ(t.size(), 32u * 151u);
  EXPECT_EQ(t.get_dim(0), 32);
  EXPECT_EQ(t.get_dim(1), 151);
  EXPECT_EQ(t.dims_size(), 2);
}
