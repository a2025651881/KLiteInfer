#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"

namespace model {
void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConig> config){
    if (add_layer_) {
        add_layer_->set_cuda_config(config);
        add_layer_->to_cuda();
    } 

    if (rope_layer_) {
        rope_layer_->set_cuda_config(config);
        rope_layer_->to_cuda();
    }

    if (swiglu_layer_) {
        swiglu_layer_->set_cuda_config(config);
        swiglu_layer_->to_cuda();
    }

    if (cls_layer_) {
        cls_layer_->set_cuda_config(config);
        cls_layer_->to_cuda();
    }

    if (embedding_layer_) {
        embedding_layer_->set_cuda_config(config);
        embedding_layer_->to_cuda();
    }

    if (mha_layer_) {
        mha_layer_->set_cuda_config(config);
        mha_layer_->to_cuda();
    }

    for (auto& weight_layer : wq_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wk_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wv_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wo_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w1_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w2_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w3_layers_) {
        if (weight_layer) {
            weight_layer->set_cuda_config(config);
            weight_layer->to_cuda();
        }
    }

    for (auto& rms_norm_layer : rmsnorm_layers_) {
        if (rms_norm_layer) {
            rms_norm_layer->to_cuda();
            rms_norm_layer->set_cuda_config(config);
        }
    }
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model)
    :Model(tokenizer_type,base::ModelType::kModelTypeLLama2,std::move(token_path),
            std::move(model_path),is_quant_model){} 

base::Status Qwen3Model::init(base::DeviceType device_type){
      using namespace base;
    if (token_path_.empty()) {
        return error::PathNotValid(token_path_);
    }
    if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
        return error::InternalError("The cpu device do not support int8 quant model.");
    }

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA) {
        cudaSetDevice(0);
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cudaStreamCreate(&cuda_config_->stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return error::InternalError("The cuda hanle create failed.");
        }
    }

    Status read_status = gen_model_from_file();
    if (!read_status) {
        return read_status;
    }
    init_mem();
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
    } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
    }

    sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
    return error::Success();
}

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
}  // namespace model

#endif