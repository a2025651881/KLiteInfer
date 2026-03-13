#ifndef KELIN_ADD_KERNEL_H
#define KELIN_ADD_KERNEL_H
#include "tensor/Tensor"
namespace add_kernel{
void add_kernel_cu(const tensor::Tensor& input1,const tensor::Tensor& input2,
                       const tensor::Tensor& output,void* stream);
}
#endif