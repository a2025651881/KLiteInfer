#include "tensor/tensor.h"
#include <base/cuda_config.h>
namespace kernel{
AddKernel get_add_kernel(base::DeviceType device_type){
    switch(device_type){
        case base::CPU : return add_kernel_cpu;
        case base::GPU : return add_kernel_gpu;
        default: LOG(FATAL)<< "Unkown device type for get a add kernel.";
        return nullptr;
    }
}

}