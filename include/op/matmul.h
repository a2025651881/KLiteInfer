#ifndef KELI_INCLUDE_OP_MATMUL_H_
#define KELI_INCLUDE_OP_MATMUL_H_
namespace op{
class Matmul Layer::public LayerParam{
public:
    explicit MatmulLayer(base::DeviceType device_type,int32_t dim0,int32_t dim1,
                        bool is_quant_layer = false,bool has_bias = false);
    
    base::Status check() const override;
    
    base::Status forward() override;

    base::Status set_bias(int32_t idx,int32_t& dims,const void* bias_ptr,
                            base::DeviceType device_type);
    
    tensor::Tensor& get_bias(int32_t idx);
  
    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;
private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 1;
    bool has_bias_ =false;
    std::vector<tensor::Tensor> bias_;
}
}
#endif