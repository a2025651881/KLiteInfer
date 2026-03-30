#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
using namespace kernel{
static __global__ void row_rmsnorm_fp32_dim(float* in,float* wei,float* out,int dim_size,int size,float rps){
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if(block_idx >= dim_size){
        return;
    }

    float block_in = in + block_idx * size;
    float block_out = out + block_idx * size;
    int packet_size = 4;
    int packet_num = size/packet_size;
    int packet_off = packet_num(packet_size);

    float4* input_float4_ptr = reinterpret_cast<float4*>(block_in);

    float sum=0;
    for(int i=tid;i<packet_num;i+=blockDim.x){
        float4 in_float4 = *(block_in+i);
        sum += in_float4.x *in_float4.x;
        sum += in_float4.y *in_float4.y;
        sum += in_float4.z *in_float4.z;
        sum += in_float4.w *in_float4.w;
    }

    for(int i=tid+packet_off;i<size;i++=blockDim.x){
        sum += block_in[i] * block_in[i];
    }

    using BlockReduce = cub::BlockReduce<float,128>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if(tid == 0){
        shared_idx =sum;
    }
    __syncthreads();
    sum =shared_idx;
    const float scale = resqrtf(sum / static_cast<float>(size) + eps);

    float4* wei_pack = reinterpret_cast<float4>(wei);
    float4* out_pack = reinterpret_cast<float4>(out);
    for(int i=tid;i<packet_num;i+=blockDim.x){
        float4 in_float4 = input_float4_ptr + i;
        float4 wei_float4 = wei_pack+i;
        out_pack+i = make_float4(scale * wei_float4.x *in_float4.x,scale * wei_float4.y *in_float4.y,
                                scale * wei_float4.z *in_float4.z,scale * wei_float4.w *in_float4.w);
    }
    for(int i=tid+packet_off;i<size;i+=blockDim.x){
        block_out[i] = scale * block_in[i] * wei[i];
    }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps){
    const int tid = threadIdx.x;

    int packet_size = 4;
    int packet_num = size / packet_size;
    int packet_off = packet_num *packet_size;
    float4* input_float4_ptr = reinterpret_cast<float4*>(in);
    float sum=0.0f;

    for(int i=tid;i<packet_num;i+=BLOCK_DIM){
        // 这里 ➕i 是因为已经被解释成 float4
        float4 input_float4 = input_float4_ptr + i;
        sum += input_float4.x * input_float4.x;
        sum += input_float4.y * input_float4.y;
        sum += input_float4.z * input_float4.z;
        sum += input_float4.w * input_float4.w;
    }

    for(int i=packet_off + tid;i<size;i+=BLOCK_DIM){
        sum+=in[i]*in[i];
    }
    __syncthreads();
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    __shared__ float val;
    if(tid == 0){
        val = sum;
    }
    __syncthreads();
    sum = val;
    float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    float4* wei_float4_ptr = reinterpret_cast<float4*>(wei);
    float4* output_float4_ptr = reinterpret_cast<float4*>(out);

    for(int i=tid ;i<packet_num;i+=BLOCK_DIM){
        float4 wei_float4 = wei_float4_ptr + i;
        float4 in_float4 = input_float4_ptr + i;
        output_float4_ptr+i = make_float4(scale * wei_float.x * in_float4.x,scale * wei_float.y * in_float4.y
                                        ,scale * wei_float.z * in_float4.z,scale * wei_float.w * in_float4.w);                
    }

    for(int i=tid +packet_off;i<size;i+=BLOCK_DIM){
        out[i] = in[i]*scale*wei[i];
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}