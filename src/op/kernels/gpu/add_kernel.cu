#include "add_kernel.cuh"
namespace add_kernel{
    __global__ add_kernel_cu_fp32(int32_t size,const float* input0,const float* input1,float* output){
        int32_t tid = blockDim.x*blockIdx.x+threadIdx.x;
        if(tid < size){
            output[tid] = input0[tid]+input1[tid];
        }
        return;
    }

    void add_kernel_cu(const tensor::Tensor& input1,const tensor::Tensor& input2,
                       const tensor::Tensor& output,void* stream){
        CHECK_EQ(input1.is_empty(),false);
        CHECK_EQ(input2.is_empty(),false);
        CHECK_EQ(output.is_empty(),false);
        int32_t size = static_cast<int32_t>(input1.size());
        CHECK_EQ(size,input2.size());
        CHECK_EQ(size,output.size());
        int32_t thread_num = 512;
        int32_t block_num = (size+thread_num-1)/thread_num;
        if(stream){
            cudaStream_t stream_ = static_cast<cudaStream_t*>(stream);
            add_kernel_cu_fp32<<block_num,thread_num,0,stream_>>(size,input1.ptr<float>(),input2.ptr<float>()
                                ,const_cast<float*>(output.ptr<float>()));
        }else{
            add_kernel_cu_fp32<<block_num,thread_num>>(size,input1.ptr<float>(),input2.ptr<float>()
            ,const_cast<float*>(output.ptr<float>()));
        }
    }
}