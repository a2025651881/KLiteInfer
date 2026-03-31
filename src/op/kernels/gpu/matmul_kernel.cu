#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
#include "base/cuda_config.h"

namespace kernel {

template<int THREADS_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input,const float* weight,float* output,size_t M,size_t K){
    __shared__ float sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + 1;

    int packet_size = 4;
    int packet_num = M / packet_size;
    int packet_off = packet_num * packet_size;

#pragma unroll
    for(int p= start_row;p<end_row;p++){
        sdata[tid] = 0;
        int row_offset = p * M;
        float4* input_float4_ptr = (float4*)input;
        float4* weight_float4_ptr = (float4*)weight + row_offset;
    #pragma unroll
        for(int i=tid;i<packet_num;i+=blockDim.x){
            float4 input_float4 = *(input_float4_ptr + i);
            float4 weight_float4 = *(weight_float4_ptr + i);
            float tmp = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y+
                        input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
            sdata[tid] += tmp;
        }

        for(int i=tid + packet_off;i<M;i+=blockDim.x){
            sdata[tid] +=input[i]* weight[row_offset + i];
        }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float,THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if(tid == 0){
        output[p] = part_sum;
    }
    __syncthreads();
    }

}

// 修复：模板参数统一为 int
template<int BLOCK_PER_THREADS,int BLOCK_PER_ROW>
__global__  void matmul_kernel_cu_fp32int8(const float* input,const int8_t* weight,float* output,float* scales,const int32_t group_size,int M,int K){
    __shared__ float sdata[BLOCK_PER_THREADS];
    int tid = threadIdx.x;

    int packet_size = 4;
    int start_row = blockIdx.x * BLOCK_PER_ROW;
    int end_row = start_row+1;
    for(int p=start_row;p<end_row;p++){
        sdata[tid] =0;
        for(int i=tid;i<M;i+=blockDim.x){
            const int weight_idx = p * M +i;
            const int group_idx = weight_idx / group_size;
            // 修复语法错误 + = → +=
            sdata[tid] += input[i] * scales[group_idx]*static_cast<float>(weight[weight_idx]);
        }
            __syncthreads();

    using BlockReduce = cub::BlockReduce<float, BLOCK_PER_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

// 修复：所有函数都放在 kernel 命名空间内！
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::GPU);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::GPU);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  int packet_size = 4;

  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    // 修复：使用命名空间 kernel::
    kernel::matmul_kernel_cu_fp32<128, 1> <<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    kernel::matmul_kernel_cu_fp32<128, 1> <<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::GPU);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::GPU);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
        matmul_kernel_cu_fp32int8<128, 1> <<<K, 128, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<int8_t>(), const_cast<float*>(output.ptr<float>()), scale.ptr<float>(), group_size, M, K);
  } else {
        matmul_kernel_cu_fp32int8<128, 1> <<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  const_cast<float*>(output.ptr<float>()),scale.ptr<float>(), group_size, M, K);
  }
}

} // namespace kernel