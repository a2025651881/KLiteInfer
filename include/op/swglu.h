#ifndef LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#define LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op{
    public:
        explicit SwiGLULayer(base::DeviceType device_type,int32_t hidden_dim);

        base::Status check() const override;

        base::Status forward() override;
    private:
        int32_t hidden_dim_=0;
}
#endif