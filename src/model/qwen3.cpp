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
                       bool is_prompt, int& next) const{
        auto status = forward(input, pos_tensor, next);
        if (!status) {
            return status;
        }
        next = post_processing(pos_tensor, is_prompt);
        return base::error::Success();
    }

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const{
        if (input.is_empty()) {
            return base::error::InvalidArgument("The input tensor is empty.");
        }
        if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
            return base::error::InternalError("Unsupported int8 quant in the cpu device");
        }
    
        for(int32_t layer_idx=0;layer_idx<config_->layer_num_;++layer_idx){
            attention_rms(layer_idx,input);
            attention_qkv(layer_idx,pos_tensor);
            attention_mha(layer_idx,pos_tensor);
            feed_forward(layer_idx,input);
        }
        cls_logits(input);
        return base::error::Success();
    }

    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const{
        auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
        auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
        if(input_tokens.size() != tokens.size()){
            input_tokens.reshape({static_cast<int32_t>(tokens.size())});
            input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->hidden_dim_});
        }
        for(int i=0;i < tokens.size();i++){
            input_tokens.index<int32_t>(i) = tokens.at(i);
        }
        auto input_token_num =
        tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
        LOG_IF(FATAL, !qwen_layers_->embedding_layer_)<< "The embedding layer in the llama2 model is null pointer.";
        STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

        op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
        return output;
    }

private:
    void init_mem(){
        std::shared_ptr<base::DeviceAllocator> alloc;
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            alloc = base::CPUDeviceAllocatorFactory::get_instance();
        }else {
            alloc = base::CUDADeviceAllocatorFactory::get_instance();
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK_NE(cuda_config_, nullptr);
            qwen_layers_->to_cuda(cuda_config_);
        }

        std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

        tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
        tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->hidden_dim_, true , alloc);
        tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);
        tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);

        CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
        CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

        CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
        CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

        tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
        tensor::Tensor out_mha(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
        CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
        CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
        CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

        tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);
        tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
        CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

        // kv cache
        tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                            config_->kv_dim_, true, alloc);
        tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                            config_->kv_dim_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
        CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

        // Wq query output
        tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kQuery, query));

        // Pos tensor
        tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

        // Attention output
        tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
        CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
        tensor::Tensor attn_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

        // final forward output
        tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
            CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
        }
        CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
    }

    base::Status create_layers() override{

    }

    void create_param_layers(){
        using namespace base;
        if(!qwen_layers_){
            qwen_layers_ = std::make_unique<Qwen3Layers>();
        }

        if(!is_quant_model_){
            create_param_layers();
        }else{
            return error::FunctionNotImplement("");
        }
        create_nonparam_layers();

        if (!qwen_layers_->embedding_layer_) {
            return error::InternalError("Create the embedding layer for the llama model failed!");
        }

        if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
            // input norm
            return error::InternalError("Create the rmsnorm layers for the llama model failed!");
        }

        if(qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
            qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
            qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
            qwen_layers_->wo_layers_.size() != config_->layer_num_) {
            return error::InternalError(
             "Create the matmul layer in the attention and ffn attention layers for "
            "the llama model "
            "failed.");
        }

        if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
            qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
            qwen_layers_->w3_layers_.size() != config_->layer_num_) {
            return error::InternalError(
            "Create the matmul layer in the feedforward layers for the llama model "
            "failed.");
        
        for (int32_t i = 0; i < config_->layer_num_; ++i) {
            if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
                !qwen_layers_->w3_layers_.at(i)) {
            return error::InternalError(
                "Create the matmul layer in the feedforward layers for the llama model "
                "failed.");
            }
        }

        if (!qwen_layers_->rope_layer_) {
            return error::InternalError("Create the rope layer for the llama model failed!");
        }

        if (!qwen_layers_->add_layer_) {
            return error::InternalError("Create the add layer for the llama model failed!");
        }

        if (!qwen_layers_->mha_layer_) {
            return error::InternalError("Create the mha layer for the llama model failed!");
        }

        if (!qwen_layers_->swiglu_layer_) {
            return error::InternalError("Create the SwiGLU layer for the llama model failed!");
        }
        return error::Success();
    }
}

    void create_nonparam_layers(){
        CHECK(qwen_layers_ != nullptr);
        qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
            device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

        qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
            device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
            config_->head_size_);

        qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

        qwen_layers_->swiglu_layer_ =std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);
    }

    void create_param_quant_layers() {}

    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const{
        CHECK(qwen_layers_ != nullptr);
        tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
        tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

        tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
        tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
        tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

        const auto& mha_layer = qwen_layers_->mha_layer_;
        CHECK_NE(mha_layer,nullptr) << ""The multi head attention layer is null pointer."";
        
        int pos = pos_tensor.index<int32_t>(0);
        std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
        std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);

        STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
        
        tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
        const auto& wo_layer =qwen_layers_->wo_layers_.at(layer_idx);
        CHECK_NE(wo_layer,nullptr) << "The weight output layer is null pointer.";
        STATUS_CHECK(wo_layer->forward(mha_output,attn_output));
    }

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const{
    CHECK(qwen_layers_ != nullptr);
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op:Layer> rmsnorm_layers = qwen_layers_->rmsnorm_layers_.at(layer_idx);
    if(!rms_norm_layer){
        LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
    }
    STATUS_CHECK(rms_norm_layer->forward(input,rmsnorm_output));
  }

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const{
    CHECK(qwen_layers_ != nullptr);
    CHECK_NE(qwen_layers_->add_layer_,nullptr) << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(qwen_layers_->add_layer_->forward(input,get_buffer(ModelBufferType::kAttnOutput),input));

    // RMSNorm
    tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& fnn_rmsnorm = qwen_layers_->rms_norm_layer.at(layer_idx+ config_->layer_num_);
    CHECK_NE(ffn_rmsnorm,nullptr) << "The final rmsnorm layer in the feedforward block si null pointer";
    STATUS_CHECK(ffn_rmsnorm->forward(input,ffn_norm_output));

    // w1
    tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer,nullptr) << "The w1 layer inthe feedforward block ix null pointer";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output,w1_output));

    // w3
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
     const auto& w1_layer = qwen_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer,nullptr) << "The w3 layer inthe feedforward block ix null pointer";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output,w1_output));

    // SwiGLU
    CHECK_NE(qwen_layers_->swiglu_layer_,nullptr) << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

    // w2
    tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer =qwen_layers_->w2_layers_.at(layer_idx);
    STATUS_CHECK(w2_layer -> forward(w1_output,w2_output));
    
    // residual add
    CHECK_NE(qwen_layers_->add_layer_,nullptr) << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const{
        CHECK(qwen_layers_ != nullptr);
        // kv cache
        tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
        int32_t pos = pos_tensor.index<int32_t>(0);
        auto [key, val] = slice_kv_cache(layer_idx, pos);

        // query
        const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
        CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

        auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
        STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

        // query norm
        auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
        query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
        query_norm->forward(query, query);
        query.reshape({(int32_t)query.size()});

        // key
        const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
        CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
        STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

        // key norm
        auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
        key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
        key_norm->forward(key, key);
        key.reshape({(int32_t)key.size()});

        // value
        const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
        CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
        STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

        // rope
        CHECK_NE(qwen_layers_->rope_layer_, nullptr)
            << "The RoPE layer in the attention block is null pointer.";
        STATUS_CHECK(qwen_layers_->rope_layer_->forward(
            query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
            get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
    }

    void cls_logits(const tensor::Tensor& input) const{
        CHECK(qwen_layers_ != nullptr);
        const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
        CHECK_NE(norm, nullptr);
        STATUS_CHECK(norm->forward(input, input));

        tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        CHECK_NE(qwen_layers_->cls_layer_, nullptr);
        STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
    }

    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const{
        tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        const float* forward_logits = forward_output.ptr<float>();

        int32_t next = 0;
        if(is_prompt){
            next = -1;
        }else{
            next = static_cast<int32_t>(sampler_->sample(forward_logits,forward_output.size(),
                                                        cuda_config_ ? cuda_config_->stream : nullptr));
        }
        return next;
    }

}  // namespace model
#endif