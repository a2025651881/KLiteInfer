#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
namespace sampler{
class Sampler{
    public:
        explicit Sampler(base::DeviceType device_type):device_type_(device_type){}

        virtual size_t sample(const float* logits,size_t size,void* stream =nullptr)=0;

    protected:
        base::DeviceType device_type_;
};
}
#endif