#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
int32_t thread_num = 128;

__device__ void softmax_gpu(float* x,int size){
    int tid = threadIdx.x;
    int step = blockDim.x;

    float max_val = tid < size ? x[tid] : -FLT_MAX;
    for(int i=tid;i<size;i+=step){
        if(x[i]>max_val){
            max_val = x[i];
        }
    }
    __shared__ float shared_tmp[step];
    __shared__ float val;
    shared_tmp[tid] = x[tid];

    for(int i=step/2;i>32;i>>1){
        if(tid < i){
            shared_tmp[tid] = fmax(shared_tmp[tid],shared_tmp[tid+i]);
        }
    }

    for(int i=16;i>0;i>>1){
        float max_val = shared_tmp[i];
        if(tid < i){
            float tmp = __shfl_down_sync(0xffffffff,max_val,i);
            max_val = tmp>max_val?tmp:max_val;
        }
    }

    if(tid == 0){
        val = max_val;
    }
    __syncthreads();

    max_val = val;

    float sum = 0.0f;
    for(int i=tid;i<size;i+=step){
        x[i] =expf(x[tid]-max_val);
        sum +=x[i];
    }
    using BlockReduce = cub::BlockReduce<float, thread_num>;
     __shared__ BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Sum(sum);

    if(threadIdx.x ==0){
        val =sum;
    }
    // 保证所有线程都可以同步到 sum 值
    __syncthreads();
    sum = val;

    for(int i=tid ;i < size;i+=step){
        x[i]=x[i]/sum;
    }
}

__global__ void multi_head_attention_kernel(int32_t pos,int32_t seq_len,float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset){
    int head = blockIdx.x;
    if(head >= head_num){
        return;
    }
    extern __shared__ float s_query_head[];
    float scale = 1.f / sqrtf(float(head_size));
    float* query_head = query + head * head_size;

    for(int i=threadIdx.x;i< head_size;i+=blockDim.x){
        s_query_head[i]=query_head[i];
    }
    __syncthreads();

    float* score_head = score_ptr + head * seq_len;
    int head_offset = (head / kv_mul) * head_size;

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // 定位到历史第 t 个位置、当前头的 key 向量
        float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

        // 计算 Q 和 K 的点积（使用 float4 向量加载加速）
        float score = 0.0f;
        for (int i = 0; i < head_size; i += 4) {
        // 一次性加载 4 个 float
            float4 key_val = *reinterpret_cast<float4*>(key_head + i);
            float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
            // 计算 4 组乘加：x*x + y*y + z*z + w*w
            score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
        }

        // 点积结果缩放
        score *= scale;
        // 保存当前位置的注意力分数
        score_head[t] = score;
    }

    __syncthreads();
    
    softmax_gpu(score_head,pos+1);
    
    __syncthreads();

    float* output_head = output + head * head_size;
    // 每个线程负责计算输出向量中的一个维度
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float value = 0.0f;
        // 加权求和：sum(score_t * V_t[i])
        for (int t = 0; t <= pos; t++) {
            // 定位到第 t 个位置、当前头的 value 向量
            float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
            // 取出当前位置的注意力权重
            float score = score_head[t];
            // 累加加权值
            value += score * value_head[i];
        }
    // 写回当前头的输出结果
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(mha_out.ptr<float>());

    float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
    cudaStream_t stream = config->stream;
    multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
            pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
            head_size, layer_offset);
}
}