#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

/**
 * @brief 权重区布局。两者前半部分一致，差异在final rmsnorm 之后：
 *   kQwen3   : ... final_norm, q_norm(L), k_norm(L), [lm_head]
 *   kLlama2C : ... final_norm, freq_cis_real/imag(需跳过), [lm_head]
 */
enum class WeightLayout : uint8_t {
  kQwen3 = 0,
  kLlama2C = 1,
};

struct QWen3TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t immediate_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
};

struct Qwen3Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Qwen3Model : public Model {
 public:
  explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model,
                      WeightLayout weight_layout = WeightLayout::kQwen3);

  base::Status init(base::DeviceType device_type) override;

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
  /// Qwen3 才有逐 head 的 q/k rmsnorm，llama2 系没有
  bool use_qk_norm() const { return weight_layout_ == WeightLayout::kQwen3; }

  /// RoPE 底数：llama2 系为 1e4，Qwen2/Qwen3 为 1e6
  float rope_theta() const {
    return weight_layout_ == WeightLayout::kLlama2C ? 10000.0f : 1000000.0f;
  }

  /// llama2.c 用相邻配对，Qwen3(HF) 用半分割配对
  bool rope_interleaved() const { return weight_layout_ == WeightLayout::kLlama2C; }

  WeightLayout weight_layout_ = WeightLayout::kQwen3;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
}  // namespace model

#endif