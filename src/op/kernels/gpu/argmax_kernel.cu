#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
namespace kernel{
__device__ argmax_wrap(float* input_ptr,size_t idx){
    for(size_t i = 16;i>0;i>>1){
        size_t value = __shlf_down_sync(0xffffffff,input_ptr,i);
        size_t index = __shlf_down_sync(0xffffffff,idx,i);
        if(value > input_ptr){
            input_ptr = value;
            idx = index;
        }
    }
}

__device__  argmax_block(float* max_vlaue,const size_t index,float* shared_ptr,float* shared_idx){
    size_t wrap_size = 32;
    size_t wrap_num = threadIdx.x / wrap_size;
    size_t wrap_idx = threadIdx.x % 32;
    argmax_wrap(max_vlaue,index);
    __syncthreads();
    if(wrap_idx == 0){
        shared_ptr[wrap_num] = max_vlaue;
        shared_idx[wrap_num] = index;
    }
    __syncthreads();
    if(threadIdx.x < 32){
        max_vlaue = shared_ptr[wrap_num];
        index = shared_idx[wrap_num];
    }
    __syncthreads();
    if(threadIdx.x < 32){
        argmax_wrap(max_vlaue,index);
    }
}

__global__void argmax_kernel_fp32(float* output,const float* input_ptr,size_t size,size_t* output_idx){
    __shared__ size_t shared_idx[32];
    __shared__ float shared_value[32];
    
    int tid = threadIdx.x;
    if(tid > size) return;

    float value = input_ptr[tid];
    float idx = tid; 

    for(int i=tid;i<size;i+=threadDim.x){
        if(input_ptr[i]> value){
            value = input_ptr[i];
            idx = i;
        }
    }

    argmax_block(value,idx,shared_ptr,shared_idx);
    __syncthreads();

    if(tid == 0){
        *output = value;
    }
    
}


size_t argmax_kernel_cu(const float* input_ptr,size_t size,void* stream){
    std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();;
    size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
    size_t output_index = 0;
    if(!stream){
        argmax_kernel_fp32<<<1,512>>>(input_ptr,size,index);
        cudaMemcpy(&output_inddex,index,sizeof(size_t),cudaMemecpyDeviceToHost);
    } else {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        argmax_kernel_fp<<<1,512,0,stream_>>>(input_ptr,size,index);
        cudaMemcpyAsync(&output_index,index,sizeof(size_t),cudaMemcpyDeviceToHost,stream_);
    }
    return output_index;
}
}
#endif