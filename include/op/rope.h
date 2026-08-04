#ifndef KELI_INCLUDE_OP_ROPE_H_
#define KELI_INCLUDE_OP_ROPE_H_
#include "layer.h"
#include "base/base.h"
#include <cstdint>
namespace op{
    class RoPELayer: public Layer{
        public:
            /**
             * @param interleaved true-> 相邻配对 (2j, 2j+1)，llama2.c 约定
             *                    false -> 半分割配对 (j, j+head_size/2)，HF / Qwen3 约定
             */
            explicit RoPELayer(base::DeviceType device_type,int32_t dim,int32_t kv_dim,int32_t head_size,
                               bool interleaved = false);

            base::Status check() const override;

            base::Status forward() override;
        private:
            int32_t dim_ = 0;
            int32_t kv_dim_ = 0;
            int32_t head_size_ =0;
            bool interleaved_ = false;
    };
}
#endif
