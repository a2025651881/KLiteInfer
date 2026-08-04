#ifndef KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_
#define KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_

#include <base/cuda_config.h>
#include "model.h"
#include "model/multimodal_types.h"
#include "model/qwen3.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/layer.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "tensor/tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace model {

/**
 * @brief Vision Transformer 配置（SigLIP-like）
 *
 * 默认值取自 PaddleOCR-VL 的 config.json 的 vision_config
 */
struct VisionTransformerConfig {
  int32_t hidden_size_         = 1152;
  int32_t num_hidden_layers_   = 27;
  int32_t num_attention_heads_ = 16;
  int32_t intermediate_size_   = 4304;
  int32_t patch_size_          = 14;
  int32_t spatial_merge_size_  = 2;
  int32_t num_channels_        = 3;
  int32_t pos_grid_            = 27;  // image_size / patch_size = 384 / 14
  float   layer_norm_eps_      = 1e-6f;
  // 视觉 2D RoPE 的底数（SigLIPRotaryEmbedding 固定 1e4）
  float   rope_theta_          = 10000.0f;

  int32_t head_dim() const { return hidden_size_ / num_attention_heads_; }  // 72

  int32_t merged_hidden() const {
    return hidden_size_ * spatial_merge_size_ * spatial_merge_size_;  // 4608
  }
};

/**
 * @brief PaddleOCR-VL 整体配置
 *
 * 默认值取自 PaddlePaddle/PaddleOCR-VL 的 config.json
 */
struct PaddleOCRVLTransformerConfig {
  // Text LLM（ERNIE 4.5 类，GQA + 3D-MRoPE，无 q/k norm，无 bias）
  int32_t text_hidden_size_    = 1024;
  int32_t text_num_layers_     = 18;
  int32_t text_num_heads_      = 16;
  int32_t text_num_kv_heads_   = 2;
  int32_t text_head_dim_       = 128;
  int32_t text_inter_size_= 3072;
  int32_t text_vocab_size_     = 103424;
  float   text_norm_eps_       = 1e-5f;
  float   text_rope_theta_     = 500000.0f;
  // 3D-MRoPE 在 head_dim/2 = 64 个频率上的 (t, h, w) 划分
  int32_t mrope_section_t_     = 16;
  int32_t mrope_section_h_     = 24;
  int32_t mrope_section_w_     = 24;

  // Projector 的 pre_norm 用的是 1e-5，与 vision 的 1e-6 不同
  float   projector_norm_eps_  = 1e-5f;

  // Multimodal special tokens
  int32_t image_token_id_        = 100295;
  int32_t vision_start_token_id_ = 101305;
  int32_t vision_max_tokens_     = 1280;

  VisionTransformerConfig vision;
};

// ----------------------------------------------------------------------------
//  视觉 / Projector 权重
//
//  视觉侧需要严格控制数值细节（bias、LayerNorm、2D RoPE、bilinear 位置编码
//  插值），且是「批量 token」而非单token 解码，因此这里直接持有权重张量，
//  由 kernels/{cpu,cuda}/vision_kernel 提供双后端实现，而不套用面向自回归
//  解码的算子层。CPU 后端下为 mmap 零拷贝视图；CUDA 后端下在 init 时上传显存。
// ----------------------------------------------------------------------------

struct SiglipEncoderLayerWeights {
  tensor::Tensor ln1_w, ln1_b;                  // [hidden]
  tensor::Tensor q_w, q_b, k_w, k_b, v_w, v_b;  // [hidden, hidden] / [hidden]
  tensor::Tensor o_w, o_b;
  tensor::Tensor ln2_w, ln2_b;                  // [hidden]
  tensor::Tensor fc1_w, fc1_b;                  // [inter, hidden] / [inter]
  tensor::Tensor fc2_w, fc2_b;                  // [hidden, inter] / [hidden]
};

struct SiglipVisionWeights {
  tensor::Tensor patch_w, patch_b;  // [hidden, C*p*p] / [hidden]
  tensor::Tensor pos_embed;         // [pos_grid*pos_grid, hidden]
  std::vector<SiglipEncoderLayerWeights> layers;
  tensor::Tensor post_ln_w, post_ln_b;  // [hidden]
};

struct PaddleOCRVLProjectorWeights {
  tensor::Tensor pre_norm_w, pre_norm_b;  // [hidden] LayerNorm(eps=1e-5)，作用在 merge 之前
  tensor::Tensor linear1_w, linear1_b;    // [merged, merged] / [merged]
  tensor::Tensor linear2_w, linear2_b;    // [text_hidden, merged] / [text_hidden]
};

/**
 * @brief 视觉 encoder 逐层复用的中间缓冲
 *
 * 27 层共用同一份，避免每层反复分配（n=828 时 score单独就有 44MB）。
 */
struct VisionWorkspace {
  tensor::Tensor normed;  // [n, hidden]
  tensor::Tensor q, k, v, attn;  // [n, hidden]
  tensor::Tensor ff;      // [n, inter]
  tensor::Tensor score;   // [heads, n, n]
};

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
  // 多模态扩展接口
  // --------------------------------------------------------------------------

  ///多模态预测入口（文本 + 图像），内部完成 prefill 与单步 decode
  base::Status predict_multimodal(const std::vector<int>& tokens,
                                  const std::vector<ProcessedImage>& images,
                                  bool is_prompt,
                                  int& next_token) const override;

  /**
   * @brief 图像编码：pixel_values → 已投影到文本隐层的视觉特征
   * @param pixel_values [num_patches, C*patch*patch]，CPU/fp32/已 normalize
   *                     （与 image processor 输出的 [N,3,14,14] 内存布局一致）
   * @return [num_img_tokens, text_hidden]
   */
  tensor::Tensor encode_image(const tensor::Tensor& pixel_values,
                              const ImageGridTHW& grid_thw) const;

  /// 计算 3D-MRoPE 位置，等价于参考实现的 get_rope_index
  MRoPEPositions compute_mrope_positions(const std::vector<int>& tokens,
                                         const std::vector<ProcessedImage>& images) const;

  const PaddleOCRVLTransformerConfig* vl_config() const { return vl_config_.get(); }

  /// 最近一次 encode_image 的耗时（毫秒），用于基准测试
  double last_vision_ms() const { return last_vision_ms_; }

  /// 调试用：把各阶段中间张量 dump 到指定目录，供与 HF 参考结果比对
  void set_dump_dir(const std::string& dir) const { dump_dir_ = dir; }

 protected:
  base::Status read_model_file() override;
  void           init_mem() override;
  base::Status   create_layers() override;
  void           create_param_layers() override;
  void           create_nonparam_layers() override;
  void           create_param_quant_layers() override;
  int32_t        post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  op::EmbeddingOutput embedding_multimodal(const std::vector<int>& tokens,
                                           const std::vector<ProcessedImage>& images) const;

  // ------------------------------ 视觉路径 ------------------------------
  //  以下均在 vision_device_ 上执行，指针为该设备上的地址

  /// patch embedding + bilinear 插值的位置编码
  void _vision_embeddings(const float* pixel_dev, const ImageGridTHW& grid, float* out) const;

  /// 构造视觉 2D RoPE 的 cos/sin 表（布局 [n, 36]：前 18 为 h 方向、后 18 为 w 方向）
  /// 表在 CPU 生成后按 vision_device_ 就位
  void _build_vision_rope(const ImageGridTHW& grid, tensor::Tensor& cos_tab,
                          tensor::Tensor& sin_tab) const;

  /// 单层 vision encoder（pre-norm + 2D RoPE + 全可见双向注意力 + tanh GELU MLP）
  void _vision_encoder_layer(int32_t layer_i, int32_t num_tokens, const float* cos_tab,
                             const float* sin_tab, float* hidden, VisionWorkspace& ws) const;

  /// pre_norm → 2x2 spatial merge → linear_1 → erf GELU → linear_2，返回 CPU 张量
  tensor::Tensor _project(const float* vision_hidden, const ImageGridTHW& grid) const;

  /// 在 vision_device_ 上分配 [rows, cols] 的 fp32 张量
  tensor::Tensor _vision_alloc(int32_t rows, int32_t cols) const;

  /// vision_device_ 为 CUDA 时同步 stream
  void _vision_sync() const;

  /// 把视觉 / projector 权重从 mmap 视图上传到显存（仅 CUDA 后端）
  void _upload_vision_weights();

  // ------------------------------ 文本 decoder ------------------------------
  /// 单个 token 走完 18 层，logits 写入 kForwardOutput
  base::Status _llm_forward_token(int32_t token_pos, const tensor::Tensor& mrope_cos,
                                  const tensor::Tensor& mrope_sin,
                                  const tensor::Tensor& input) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
  void attention_qkv(int32_t layer_idx, int32_t token_pos, const tensor::Tensor& mrope_cos,
                     const tensor::Tensor& mrope_sin) const;
  void attention_mha(int32_t layer_idx, int32_t token_pos) const;
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  void cls_logits(const tensor::Tensor& input) const;

  /// 由 3D 位置 (t,h,w) 生成该 token 的 cos/sin（长度 head_dim）
  void _build_mrope_cos_sin(int32_t pos_t, int32_t pos_h, int32_t pos_w, float* cos_out,
                            float* sin_out) const;

  /// dump 中间张量；数据在vision_device_ 上时先拷回主机
  void _dump(const std::string& name, const float* data, size_t count) const;

 private:
  std::shared_ptr<kernel::CudaConfig>            cuda_config_ = nullptr;
  std::unique_ptr<Qwen3Layers>                   qwen_layers_;
  std::unique_ptr<SiglipVisionWeights>           siglip_;
  std::unique_ptr<PaddleOCRVLProjectorWeights>   projector_;
  std::unique_ptr<PaddleOCRVLTransformerConfig>  vl_config_;

  /**
   * 视觉 encoder / projector 的执行设备。
   *
   * 与device_type_（文本 decoder 的设备）分开：视觉侧是批量 token 的大GEMM，
   * 能吃满 GPU；而文本 decoder 的 3D-MRoPE 目前仍是 CPU 实现。两者之间通过
   * 一次 device→host 拷贝衔接（视觉特征只有 [207, 1024]，开销可忽略）。
   */
  base::DeviceType vision_device_ = base::DeviceType::kDeviceCPU;

  /// 多模态 decode 阶段跨 step 维护的 KV-cache 写入位置
  mutable int32_t mm_decode_step_ = 0;
  /// 参考实现中 position id 数量远少于 token 数，decode 时需要用它续算
  mutable int32_t mm_rope_pos_ = 0;
  mutable std::string dump_dir_;
  /// 每个阶段只 dump 第一次（prefill）的结果，避免被 decode 步覆盖
  mutable std::vector<std::string> dumped_;
  /// 最近一次视觉编码耗时（ms）
  mutable double last_vision_ms_ = 0.0;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_PADDLEOCR_VL_H_
