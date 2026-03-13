#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op{
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type,int32_t dim,int32_t seq_len,
                                int32_t vocab_size)
: dim_(dim),
  seq_len_(seq_len),
  vocab_size_(vocab_size),
  LayerParam(device_type,LayerType::kLayerEmbedding,false,"Embedding"){
    reset_weight_size(1);
    reset_input_size(2);
    reset_output_size(1);
}

base::Status EmbeddingLayer::check() const{
    const auto& input_tensor = get_input(0);
    const auto& token_size = get_input(1).size();
    if(token_size > input_tensor.size()){
        return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
    }

    
}


base::Satus EmbeddingLayer::forward(){
    base::Status status = check();
    if(!status) return status;
    if(device_type_ == base::DeviceType::kDeviceCUDA){
        CHECK(cuda_config_ !=nullptr);
    }
    kernel::get_emb_kernel(device_type_)(get_input(0),get_weight(0),get_output(0),
                                        vocab_size_,cuda_config_?cuda_config_->stream:nullptr);

    return base::StatusCode::kSuccess;
}

}