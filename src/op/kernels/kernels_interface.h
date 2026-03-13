#ifndef KELI_INTERFACE_H
#define KELI_INTERFACE_H
#include "tensor/tensor.h"
namespace kernel{
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output,void stream);



AddKernel get_add_kernel(base::DeviceType device_type);
}
#endif