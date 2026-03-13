#ifndef KELI_INCLUDE_OP_KERNEL
#define KELI_INCLUDE_OP_KERNEL
namespace kernel{
void add_kernel_cpu(const tensor::Tensor& input1,const tensor::Tensor& inupt2,
                    const tensor::Tensor& output,void* stream =nullptr);
}
#endif