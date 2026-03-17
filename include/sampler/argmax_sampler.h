#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sample.h"
namespace sampler{
    class ArgmaxSampler : public Sampler{
        public:
            explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type){}

            size_t sample(const float* logits,size_t size,void* stream) override;};
}
#endif