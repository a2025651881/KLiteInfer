#ifndef KELI_INCLUDE_OP_ENCODE_H_
#define KELI_INCLUDE_OP_ENCODE_H_

namespace op{
class EncodeLayerBase : public Layer{
    public:
        explicit EncodeLayerBase(std::string token_model_path,bool has_bos,bool has_eos)
        : Layer(base::DeviceType::kDeviceCPU,LayerType::kLayerEncode,"Encode"),
          has_bos_(has_bos),
          has_eos_(has_eos),
          token_model_path(std::move(token_model_path)){}

        
}
}