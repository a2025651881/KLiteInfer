#include <base/base.h>
#include <algorithm>
#include "sampler/argmax_sampler.h"
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler{
    size_t ArgmaxSampler::sample(const float* logits,size_t size,void* stream){
        if(device_type_ == base::DeviceType::kDeviceCPU){
            return std::distance(logits,std::max_element(logits,logits+size));
        }else{
            return kernel::argmax_kernel_cu(logits,size,stream);
        }     
    }
}