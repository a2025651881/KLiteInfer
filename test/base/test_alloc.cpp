#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>
#include <cuda_runtime.h>
#include "base/buffer.h"
using namespace base;
// Test case for CPUAllocator
TEST(AllocTest, CpuAllocatorTest) {
    auto cpuAllocator =CPUAllocatorInstance().getInstance();
    size_t size = 1024; // 1 KB
    void* ptr = cpuAllocator.allocate(size);
    ASSERT_NE(ptr, nullptr) << "CPU allocation failed for size: " << size;
}
//  Test case for GPUAllocator
TEST(AllocTest, GpuAllocatorTest) {
    auto gpuAllocator =GPUAllocatorInstance().getInstance();
    size_t size = 1024; // 1 KB
    void* ptr = gpuAllocator.allocate(size);
    ASSERT_NE(ptr, nullptr) << "GPU allocation failed for size: " << size;
}
// Test case for Buffer class
TEST(test_buffer, use_external1) {
  auto alloc = GPUAllocatorInstance::getInstance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  // Check that the buffer is using external memory and has the correct properties
  CHECK_EQ(buffer.is_external(), true);
  std::shared_ptr<GPUAllocator> alloc2 =std::make_shared<GPUAllocator>();
  Buffer buffer2(32, alloc2);
  // Check that the buffer is not using external memory and has the correct properties
  CHECK_EQ(buffer2.is_external(), false);
  buffer2.allocate();
  // Check that the buffer was allocated successfully and has the correct properties
  CHECK_NE(buffer2.ptr(), nullptr);
  buffer2.copy_from(buffer);
  // Check that the data was copied correctly
  CHECK_EQ(buffer2.device_type(), DeviceType::GPU);
  CHECK_EQ(buffer2.byte_size(), 32);
  // Change the device type and check that it was updated correctly
  buffer2.set_device_type(DeviceType::CPU);
  CHECK_EQ(buffer2.device_type(), DeviceType::CPU);
  delete[] ptr;
  cudaFree(buffer.ptr());
}