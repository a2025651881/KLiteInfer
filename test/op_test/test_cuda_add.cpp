#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../src/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  LOG(INFO) << "========================================";
  LOG(INFO) << "Start add kernel test (NO stream)";

  int32_t size = 32 * 151;
  LOG(INFO) << "Tensor size: " << size;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  LOG(INFO) << "Allocate GPU tensors success";

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);
  LOG(INFO) << "Set value t1=2.0, t2=3.0 success";

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  LOG(INFO) << "Add kernel executed and synchronized success";

  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  LOG(INFO) << "Copy result from device to host success";

  for (int32_t i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  LOG(INFO) << "All elements check passed! Result = 5.0";

  delete[] output;
  LOG(INFO) << "Test add1_nostream finished successfully!";
  LOG(INFO) << "========================================";
}

TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  LOG(INFO) << "========================================";
  LOG(INFO) << "Start add kernel test (WITH stream)";

  int32_t size = 32 * 151;
  LOG(INFO) << "Tensor size: " << size;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  LOG(INFO) << "Allocate GPU tensors success";

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);
  LOG(INFO) << "Set value t1=2.0, t2=3.0 success";

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  LOG(INFO) << "CUDA stream created success";

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  LOG(INFO) << "Add kernel executed with stream and synchronized success";

  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  LOG(INFO) << "Copy result from device to host success";

  for (int32_t i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  LOG(INFO) << "All elements check passed! Result = 5.0";

  cudaStreamDestroy(stream);
  LOG(INFO) << "CUDA stream destroyed success";
  delete[] output;

  LOG(INFO) << "Test add1_stream finished successfully!";
  LOG(INFO) << "========================================";
}

TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  LOG(INFO) << "========================================";
  LOG(INFO) << "Start add kernel test (large size & float align)";

  int32_t size = 32 * 151 * 13;
  LOG(INFO) << "Tensor size: " << size;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  LOG(INFO) << "Allocate GPU tensors success";

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);
  LOG(INFO) << "Set value t1=2.1, t2=3.3 success";

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)( t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  LOG(INFO) << "Add kernel executed and synchronized success";

  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  LOG(INFO) << "Copy result from device to host success";

  for (int32_t i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);
  }
  LOG(INFO) << "All elements check passed! Result = 5.4 (epsilon=0.1)";

  delete[] output;
  LOG(INFO) << "Test add_align1 finished successfully!";
  LOG(INFO) << "========================================";
}