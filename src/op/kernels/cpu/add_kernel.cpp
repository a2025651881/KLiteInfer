#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"
namespace kernel{
void add_kernel_cpu(const tensor::Tensor& input1,const tensor::Tensor& inupt2,
                    const tensor::Tensor& output,void* stream){
        UNUSED(stream);
        CHECK_NE(input1.size(),0);
        CHECK_NE(input2.size(),0);
        CHECK_NE(output.size(),0);
        
        CHECK_EQ(input1.size(),input2.size());
        CHECK_EQ(input1.size(),output.size());
        arma::fvex input_vec1(const_cast<float*>(input1.ptr<float>()),input1.size(),false,true);
        arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()),input2.size(),false,true);
        arma::fvec output_vec(const_cast<float*>(output.ptr<float>()),output.size(),false,true);
        output_vec = input_vec1+input_vec2;
    }
}