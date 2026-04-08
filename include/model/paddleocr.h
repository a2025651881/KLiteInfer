#ifndef KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_
#define KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_

#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "op/layer.h"
#include "tensor/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace model {

/**
 * @brief Vision Transformer 配置 (SigLIP-like)
 */
struct VisionTransformerConfig {
  int32_t hidden_size_        = 1152;
  int32_t num_hidden_layers_   = 27;
  int32_t num_attention_heads_ = 16;
  int32_t intermediate_size_   = 4304;
  int32_t patch_size_          = 14;
  int32_t spatial_merge_size_ = 2;    // 2x2 patch merge
  int32_t num_channels_        = 3;
  float   layer_norm_eps_      = 1e-6f;

  int32_t head_dim() const {
    return hidden_size_ / num_attention_heads_;
  }

  int32_t merged_hidden() const {
    return hidden_size_ * spatial_merge_size_ * spatial_merge_size_;
  }
};

/**
 * @brief PaddleOCR-VL 整体配置 (继承 Qwen 配置)
 */
struct PaddleOCRVLTransformerConfig {
  // Text LLM (ERNIE 4.5 / Qwen 类)
  int32_t text_hidden_size_    = 896;
  int32_t text_num_layers_     = 24;
  int32_t text_num_heads_      = 14;
  int32_t text_inter_size_     = 3584;
  float   text_norm_eps_       = 1e-6f;

  // Multimodal special tokens
  int32_t image_token_id_        = 100017;
  int32_t vision_start_token_id_ = 100016;
  int32_t vision_max_tokens_     = 1280;

  VisionTransformerConfig vision;
};

/**
 * @brief MRoPE 位置输出 (3D rope: t, h, w)
 */
struct MRoPEPositions {
  tensor::Tensor positions;          // shape: [3, seq_len] (int32, CPU)
  int32_t        mrope_position_delta = 0;
};

/**
 * @brief 图像网格信息 (t, h, w) in patches
 */
struct ImageGridTHW {
  int32_t t = 1;
  int32_t h = 0;
  int32_t w = 0;

  int32_t num_patches() const {
    return t * h * w;
  }

  int32_t num_img_tokens(int32_t merge) const {
    return t * (h / merge) * (w / merge);
  }
};

/**
 * @brief 预处理后图像
 */
struct ProcessedImage {
  tensor::Tensor pixel_values;   // normalized image patches
  ImageGridTHW   grid_thw;       // t, h, w (in patches)
};

// ----------------------------------------------------------------------------
// Layer 结构定义
// ----------------------------------------------------------------------------

struct SiglipVisionLayers {
  std::shared_ptr<op::Layer> patch_embedding;       // Conv2d patch embed
  std::vector<std::shared_ptr<op::Layer>> attn_norm;
  std::vector<std::shared_ptr<op::Layer>> qkv_proj;
  std::vector<std::shared_ptr<op::Layer>> out_proj;
  std::vector<std::shared_ptr<op::Layer>> ffn_norm;
  std::vector<std::shared_ptr<op::Layer>> fc1;
  std::vector<std::shared_ptr<op::Layer>> fc2;
  std::shared_ptr<op::Layer> post_layernorm;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

struct PaddleOCRVLProjectorLayers {
  std::shared_ptr<op::Layer> pre_norm;
  std::shared_ptr<op::Layer> linear_1;
  std::shared_ptr<op::Layer> linear_2;
  std::shared_ptr<op::Layer> act;  // GELU

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

// Forward declaration
class Qwen3Layers;

// ----------------------------------------------------------------------------
// PaddleOCR-VL 模型主类
// ----------------------------------------------------------------------------

class PaddleOCRVLModel : public Model {
 public:
  explicit PaddleOCRVLModel(base::TokenizerType tokenizer_type,
                             std::string token_path,
                             std::string model_path,
                             bool is_quant_model = false);

  // --------------------------------------------------------------------------
  // 基类重载
  // --------------------------------------------------------------------------
  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input,
                       const tensor::Tensor& pos_tensor,
                       bool is_prompt,
                       int& next) const override;

  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // --------------------------------------------------------------------------
  // 多模态扩展接口 (核心)
  // --------------------------------------------------------------------------

  /**
   * @brief 多模态预测入口 (文本 + 图像)
   */
  virtual base::Status predict_multimodal(const std::vector<int>& tokens,
                                          const std::vector<ProcessedImage>& images,
                                          bool is_prompt,
                                          int& next_token) const;

  /**
   * @brief 图像编码：pixel_values → vision hidden
   */
  tensor::Tensor encode_image(const tensor::Tensor& pixel_values,
                               const ImageGridTHW& grid_thw) const;

  /**
   * @brief 计算 3D-MRoPE 位置编码
   */
  MRoPEPositions compute_mrope_positions(const std::vector<int>& tokens,
                                          const std::vector<ProcessedImage>& images) const;

 protected:
  // --------------------------------------------------------------------------
  // Model 基类纯虚函数实现
  // --------------------------------------------------------------------------
  void           init_mem() override;
  base::Status   create_layers() override;
  void           create_param_layers() override;
  void           create_nonparam_layers() override;
  void           create_param_quant_layers() override;
  int32_t        post_processing(const tensor::Tensor& pos,
                                  bool is_prompt) const override;

 private:
  // --------------------------------------------------------------------------
  // 内部核心函数
  // --------------------------------------------------------------------------

  /**
   * @brief 多模态 embedding：文本 + 图像特征
   */
  op::EmbeddingOutput embedding_multimodal(const std::vector<int>& tokens,
                                           const std::vector<ProcessedImage>& images) const;

  /**
   * @brief Patch embedding: image → patch tokens
   */
  tensor::Tensor _patch_embed(const tensor::Tensor& pixel_values) const;

  /**
   * @brief 构建视觉 Rotory Position Embedding
   */
  void _build_vision_rope(const ImageGridTHW& grid_thw,
                           tensor::Tensor& rot_cos,
                           tensor::Tensor& rot_sin) const;

  /**
   * @brief 单层 Vision Transformer 推理
   */
  void _encoder_layer(int32_t layer_i,
                       const tensor::Tensor& rot_cos,
                       const tensor::Tensor& rot_sin,
                       tensor::Tensor& hidden) const;

  /**
   * @brief 视觉特征 → 文本隐层映射 (Projector)
   */
  tensor::Tensor _project(const tensor::Tensor& vision_hidden,
                           const ImageGridTHW& grid_thw) const;

  /**
   * @brief 空间融合：2x2 patch merge
   */
  tensor::Tensor _spatial_merge(const tensor::Tensor& hidden,
                                 const ImageGridTHW& grid_thw) const;

  // ------------------------------
  // LLM 内部调用 (可选)
  // ------------------------------
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& mrope_pos) const;
  void attention_mha(int32_t layer_idx, const tensor::Tensor& mrope_pos) const;
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  void cls_logits(const tensor::Tensor& input) const;

 private:
  std::shared_ptr<kernel::CudaConfig>          cuda_config_ = nullptr;
  std::unique_ptr<Qwen3Layers>                 qwen_layers_;
  std::unique_ptr<SiglipVisionLayers>          siglip_layers_;
  std::unique_ptr<PaddleOCRVLProjectorLayers>  projector_layers_;
  const PaddleOCRVLTransformerConfig*          vl_config_ = nullptr;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_