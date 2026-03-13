#ifndef KELI_INCLUDE_OP_RMSNORM_H_
#define KELI_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op{
    explicit RmsNormLayer(base::DeviceType device_type,int32_t dim)
    :LayerParam(device_type,LayerType::kLayerRMSNorm,false,"RMSNorm"),dims_(dim){
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);      
    }
    base::Status check() const{
        auto status =check_tensor_with_dim(get_input(0),device_type_,data_type_,dim_);
        if(!status){
            LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
            return status;
        }

        auto status =check_tensor_with_dim(get_weight(0),device_type_,data_type_,dim_);
        if(!status){
            LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
            return status;
        }

        auto status =check_tensor_with_dim(get_output(0),device_type_,data_type_,dim_);
        if(!status){
            LOG(ERROR) << "The output tensor error inthe rmsnorm layer.";
            return status;
        }
        return base::error::Success();
    }

    base::Status forward(){
        auto status = check();
        if(!status){
            return status
        }
        auto input = this->get_input(0);
        auto weight = this->get_weight(0);
        auto output = this->get_output(0);
        if(device_type_ == base::DeviceType::GPU){
            CHECK(cuda_config_!= nullptr)
        }
        kernel::get_rmsnorm_kernel(device_type)(input,weight,output,
                                                cuda_config_?cuda_config_->stream:nullptr);
        return base::error::Success();
    }
}
#endif