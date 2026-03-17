#include "../kernels_interfaxe.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
namespace kernel{

__global__void argmax_kernel_fp32(const float* input_ptr,size_t size,size_t* output_idx){
    __shared__ size_t shared_max_ptr[32];
    
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
        argmaxc_kernel_fp<<<1,512,0,stream_>>>(input_ptr,size,index);
        cudaMemcpyAsync(&output_index,index,sizeof(size_t),cudaMemcpyDeviceToHost,stream_);
    }
    return output_index;
}
}
#endif