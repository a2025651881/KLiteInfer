// 视觉注意力打分缓冲的分块策略测试。
//
// 整块 score 形状是 [heads, n, n]。n 大到 13200 时元素数 16*13200^2 = 2.79e9
// 已越过 INT32_MAX，必须按 head 分块，否则既会 int32 溢出也会一次要 11GB 显存。
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../src/op/kernels/kernels_interface.h"

namespace {
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
}

TEST(test_vision_chunk, cpu_needs_only_one_row_per_head) {
  // CPU 版逐行复用打分缓冲，只需 heads 行
  EXPECT_EQ(kernel::vision_attention_score_rows(base::DeviceType::kDeviceCPU, 828, 16), 16);
  EXPECT_EQ(kernel::vision_attention_score_rows(base::DeviceType::kDeviceCPU, 13200, 16), 16);
}

TEST(test_vision_chunk, cuda_small_image_uses_all_heads) {
  // 小图（n=828）放得下整块，chunk 应等于 heads
  const int32_t rows = kernel::vision_attention_score_rows(base::DeviceType::kDeviceCUDA, 828, 16);
  EXPECT_EQ(rows, 16 * 828);
}

TEST(test_vision_chunk, cuda_large_image_splits_and_avoids_overflow) {
  const int32_t n = 13200, heads = 16;
  const int32_t rows = kernel::vision_attention_score_rows(base::DeviceType::kDeviceCUDA, n, heads);

  ASSERT_GT(rows, 0);
  // 分块后必须真的小于整块
  EXPECT_LT(static_cast<int64_t>(rows), static_cast<int64_t>(heads) * n);
  // 缓冲元素数（rows * n）不能越过 int32 上限
  EXPECT_LE(static_cast<int64_t>(rows) * n, kInt32Max);
  // 行数应是 n 的整数倍（rows = chunk * n）
  EXPECT_EQ(rows % n, 0);
  const int32_t chunk = rows / n;
  EXPECT_GE(chunk, 1);
  EXPECT_LE(chunk, heads);
}

TEST(test_vision_chunk, cuda_chunk_never_zero_on_extreme_n) {
  // 极端大图上单个 head 都放不下时，也必须至少给 1 个 head，不能返回 0
  const int32_t n = 40000, heads = 16;
  const int32_t rows = kernel::vision_attention_score_rows(base::DeviceType::kDeviceCUDA, n, heads);
  EXPECT_EQ(rows, n);  // chunk == 1
}
