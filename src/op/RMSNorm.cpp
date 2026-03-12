#ifndef KELI_INCLUDE_OP_RMSNORM_H_
#define KELI_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op{
    class RmsNormLayer:public LayerParam{
        public:
            explicit RmsNormLayer(base::DeviceType device_type,int32_t dim);

            base::Status check() const override;

            base::Status forward overrid;
        private:
            int32_t dim_ =0;
    }
}
#endif