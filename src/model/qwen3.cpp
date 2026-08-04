#include "model/qwen3.h"
#include <base/alloc.h>
#include <base/cuda_config.h>
#include <cmath>
#include <utility>
#include <vector>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"

namespace model {

void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
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
      rms_norm_layer->set_cuda_config(config);
      rms_norm_layer->to_cuda();
    }
  }
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model, WeightLayout weight_layout)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model),
      weight_layout_(weight_layout) {}

base::Status Qwen3Model::init(base::DeviceType device_type) {
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
    if (cudaGetLastError() != cudaSuccess) {
      return error::InternalError("The cuda handle create failed.");
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
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>(),
                                   rope_theta());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), rope_theta(),
                                  cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

base::Status Qwen3Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

op::EmbeddingOutput Qwen3Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the qwen3 model is null pointer.";
  STATUS_CHECK(
      qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

void Qwen3Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    qwen_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32,
                           config_->head_size_ * config_->seq_len_, true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32,
                           config_->head_size_ * config_->seq_len_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  // MHA 输出维度为 head_num * head_size，Qwen3 中它可能与 hidden size 不同
  tensor::Tensor out_mha(base::DataType::kDataTypeFp32, config_->q_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));

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

  // wq输出
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->q_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // attention 中间结果
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  tensor::Tensor attn_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // logits
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status Qwen3Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the qwen3 model failed!");
  }

  // rmsnorm 布局：attn norm(L) + ffn norm(L) + final norm(1) [+ q norm(L) + k norm(L)]
  const size_t expect_rmsnorm =
      use_qk_norm() ? static_cast<size_t>(4 * config_->layer_num_ + 1)
                    : static_cast<size_t>(2 * config_->layer_num_ + 1);
  if (qwen_layers_->rmsnorm_layers_.size() != expect_rmsnorm) {
    return error::InternalError("Create the rmsnorm layers for the qwen3 model failed!");
  }

  const size_t layer_num = static_cast<size_t>(config_->layer_num_);
  if (qwen_layers_->wq_layers_.size() != layer_num || qwen_layers_->wk_layers_.size() != layer_num ||
      qwen_layers_->wv_layers_.size() != layer_num || qwen_layers_->wo_layers_.size() != layer_num) {
    return error::InternalError(
        "Create the matmul layer in the attention layers for the qwen3 model failed.");
  }

  if (qwen_layers_->w1_layers_.size() != layer_num || qwen_layers_->w2_layers_.size() != layer_num ||
      qwen_layers_->w3_layers_.size() != layer_num) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the qwen3 model failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the qwen3 model failed.");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the qwen3 model failed!");
  }
  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the qwen3 model failed!");
  }
  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the qwen3 model failed!");
  }
  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the qwen3 model failed!");
  }
  if (!qwen_layers_->cls_layer_) {
    return error::InternalError("Create the cls layer for the qwen3 model failed!");
  }
  return error::Success();
}

void Qwen3Model::create_param_layers() {
  CHECK(qwen_layers_ != nullptr);
  CHECK(raw_model_data_ != nullptr);

  const int32_t layer_num = config_->layer_num_;
  const int32_t dim = config_->dim_;
  const int32_t q_dim = config_->q_dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t inter_dim = config_->immediate_dim_;
  const int32_t head_size = config_->head_size_;
  const int32_t vocab_size = config_->vocab_size_;

  // ---------------------------------------------------------------------
  // 权重区布局（fp32，紧排，元素为单位）：
  //   1. token embedding      [vocab, dim]
  //   2. attention rmsnormL x [dim]
  //   3. wqL x [q_dim, dim]
  //   4. wk                   L x [kv_dim, dim]
  //   5. wv                   L x [kv_dim, dim]
  //   6. wo                   L x [dim, q_dim]
  //   7. ffn rmsnorm          L x [dim]
  //   8. w1 (gate)            L x [inter_dim, dim]
  //   9. w2 (down)            L x [dim, inter_dim]
  //  10. w3 (up)              L x [inter_dim, dim]
  //  11. final rmsnorm        [dim]
  //  12. kQwen3  : q rmsnormL x [head_size]
  //      kLlama2C: freq_cis_real/imag [seq_len, head_size/2] x 2（跳过，RoPE
  //                缓存由sin_cos_cache_calc 自行计算）
  //  13. kQwen3  : k rmsnorm  L x [head_size]
  //  14. lm_head             [vocab, dim]（is_shared_weight_ 时复用 1）
  // ---------------------------------------------------------------------
  const bool qk_norm = use_qk_norm();
  size_t cursor = 0;
  const size_t off_embedding = cursor;
  cursor += static_cast<size_t>(vocab_size) * dim;
  const size_t off_attn_norm = cursor;
  cursor += static_cast<size_t>(layer_num) * dim;
  const size_t off_wq = cursor;
  cursor += static_cast<size_t>(layer_num) * q_dim * dim;
  const size_t off_wk = cursor;
  cursor += static_cast<size_t>(layer_num) * kv_dim * dim;
  const size_t off_wv = cursor;
  cursor += static_cast<size_t>(layer_num) * kv_dim * dim;
  const size_t off_wo = cursor;
  cursor += static_cast<size_t>(layer_num) * dim * q_dim;
  const size_t off_ffn_norm = cursor;
  cursor += static_cast<size_t>(layer_num) * dim;
  const size_t off_w1 = cursor;
  cursor += static_cast<size_t>(layer_num) * inter_dim * dim;
  const size_t off_w2 = cursor;
  cursor += static_cast<size_t>(layer_num) * dim * inter_dim;
  const size_t off_w3 = cursor;
  cursor += static_cast<size_t>(layer_num) * inter_dim * dim;
  const size_t off_final_norm = cursor;
  cursor += static_cast<size_t>(dim);
  size_t off_q_norm = 0;
  size_t off_k_norm = 0;
  if (qk_norm) {
    off_q_norm = cursor;
    cursor += static_cast<size_t>(layer_num) * head_size;
    off_k_norm = cursor;
    cursor += static_cast<size_t>(layer_num) * head_size;
  } else {
    // freq_cis_real + freq_cis_imag，各 [seq_len, head_size / 2]
    cursor += static_cast<size_t>(config_->seq_len_) * head_size;
  }
  size_t off_cls = off_embedding;
  if (!config_->is_shared_weight_) {
    off_cls = cursor;
    cursor += static_cast<size_t>(vocab_size) * dim;
  }

  const size_t expect_size = cursor * sizeof(float) + raw_model_data_->header_size;
  CHECK_GE(raw_model_data_->file_size, expect_size)
      << "The weight file is too small: expect at least " << expect_size << " bytes but got "
      << raw_model_data_->file_size
      << " bytes. 请确认导出脚本与 Qwen3Model::create_param_layers 中的权重布局一致。";

  const auto cpu = base::DeviceType::kDeviceCPU;

  // 1. embedding
  auto embedding_layer =
      std::make_shared<op::EmbeddingLayer>(device_type_, dim, config_->seq_len_, vocab_size);
  embedding_layer->set_weight(0, {vocab_size, dim}, raw_model_data_->weight(off_embedding), cpu);
  qwen_layers_->embedding_layer_ = embedding_layer;

  // 2. attention rmsnorm
  for (int32_t i = 0; i < layer_num; ++i) {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_layer->set_weight(0, {dim},
                          raw_model_data_->weight(off_attn_norm + static_cast<size_t>(i) * dim),
                          cpu);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
  }

  // 7. ffn rmsnorm
  for (int32_t i = 0; i < layer_num; ++i) {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_layer->set_weight(0, {dim},
                          raw_model_data_->weight(off_ffn_norm + static_cast<size_t>(i) * dim),
                          cpu);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
  }

  // 11. final rmsnorm
  {
    auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    rms_layer->set_weight(0, {dim}, raw_model_data_->weight(off_final_norm), cpu);
    qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
  }

  // 12. query rmsnorm（作用在每个 head 的 head_size 维度上），仅 Qwen3
  if (qk_norm) {
    for (int32_t i = 0; i < layer_num; ++i) {
      auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, head_size);
      rms_layer->set_weight(
          0, {head_size}, raw_model_data_->weight(off_q_norm + static_cast<size_t>(i) * head_size),
          cpu);
      qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
    }

    // 13. key rmsnorm
    for (int32_t i = 0; i < layer_num; ++i) {
      auto rms_layer = std::make_shared<op::RmsNormLayer>(device_type_, head_size);
      rms_layer->set_weight(
          0, {head_size}, raw_model_data_->weight(off_k_norm + static_cast<size_t>(i) * head_size),
          cpu);
      qwen_layers_->rmsnorm_layers_.push_back(rms_layer);
    }
  }

  // 3~6. attention 投影
  const size_t wq_stride = static_cast<size_t>(q_dim) * dim;
  const size_t wk_stride = static_cast<size_t>(kv_dim) * dim;
  const size_t wo_stride = static_cast<size_t>(dim) * q_dim;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, q_dim, dim);
    wq->set_weight(0, {q_dim, dim}, raw_model_data_->weight(off_wq + i * wq_stride), cpu);
    qwen_layers_->wq_layers_.push_back(wq);

    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim);
    wk->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(off_wk + i * wk_stride), cpu);
    qwen_layers_->wk_layers_.push_back(wk);

    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim);
    wv->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(off_wv + i * wk_stride), cpu);
    qwen_layers_->wv_layers_.push_back(wv);

    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, q_dim);
    wo->set_weight(0, {dim, q_dim}, raw_model_data_->weight(off_wo + i * wo_stride), cpu);
    qwen_layers_->wo_layers_.push_back(wo);
  }

  // 8~10. feed forward
  const size_t w1_stride = static_cast<size_t>(inter_dim) * dim;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, inter_dim, dim);
    w1->set_weight(0, {inter_dim, dim}, raw_model_data_->weight(off_w1 + i * w1_stride), cpu);
    qwen_layers_->w1_layers_.push_back(w1);

    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, inter_dim);
    w2->set_weight(0, {dim, inter_dim}, raw_model_data_->weight(off_w2 + i * w1_stride), cpu);
    qwen_layers_->w2_layers_.push_back(w2);

    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, inter_dim, dim);
    w3->set_weight(0, {inter_dim, dim}, raw_model_data_->weight(off_w3 + i * w1_stride), cpu);
    qwen_layers_->w3_layers_.push_back(w3);
  }

  // 14. lm_head
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, vocab_size, dim);
  cls_layer->set_weight(0, {vocab_size, dim}, raw_model_data_->weight(off_cls), cpu);
  qwen_layers_->cls_layer_ = cls_layer;
}

void Qwen3Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->q_dim_, config_->kv_dim_, config_->head_size_, rope_interleaved());

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);
}

void Qwen3Model::create_param_quant_layers() {
  LOG(FATAL) << "The int8 quantized qwen3 model is not supported yet.";
}

void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& rms_norm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK_NE(rms_norm_layer, nullptr)
      << "The attention rmsnorm layer is a null pointer in the qwen3 model";
  STATUS_CHECK(rms_norm_layer->forward(input, rmsnorm_output));
}

void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // query norm（逐 head 归一化），仅 Qwen3 有
  if (use_qk_norm()) {
    const auto& query_norm =
        qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
    CHECK_NE(query_norm, nullptr) << "The query norm layer in the attention block is null pointer.";
    query.reshape({static_cast<int32_t>(query.size()) / config_->head_size_, config_->head_size_});
    STATUS_CHECK(query_norm->forward(query, query));
    query.reshape({static_cast<int32_t>(query.size())});
  }

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // key norm，仅 Qwen3 有
  if (use_qk_norm()) {
    const auto& key_norm =
        qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
    CHECK_NE(key_norm, nullptr) << "The key norm layer in the attention block is null pointer.";
    key.reshape({static_cast<int32_t>(key.size()) / config_->head_size_, config_->head_size_});
    STATUS_CHECK(key_norm->forward(key, key));
    key.reshape({static_cast<int32_t>(key.size())});
  }

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

void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = qwen_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";

  const int32_t pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attention 残差
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput),
                 input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm =
      qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // ffn 残差
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen3Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model
