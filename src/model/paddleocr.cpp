// =============================================================================
//  PaddleOCR-VL Model
//
//  结构：
//    1. SigLIP-like Vision Transformer (patch_embed + N x [Attn + FFN] + post_ln)
//    2. Projector MLP (linear_1 -> GELU -> linear_2) + 2x2 spatial merge
//    3. ERNIE / Qwen 系文本解码器 (复用 Qwen3Layers，由权重加载阶段挂载)
//    4. 3D-MRoPE (t, h, w) 位置编码
//
//  说明：
//    - 视觉路径目前为 CPU 主路径 (LayerNorm/GELU/PatchEmbed 仅 CPU)。
//      对外接口对设备透明：CUDA 模式下视觉 hidden 在 CPU 上算完后，
//      通过 DeviceAllocator::memcpy 注入到 LLM 的 (可能位于 CUDA 的) embedding 张量。
// =============================================================================
#include "model/paddleocr.h"
#include "model/qwen3.h"
#include "op/matmul.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "op/mha.h"
#include "op/add.h"
#include "op/swiglu.h"
#include "op/embedding.h"
#include "op/vision.h"
#include "base/alloc.h"
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace model {

// =============================================================================
//  to_cuda helpers
// =============================================================================
void SiglipVisionLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  auto move = [&](std::shared_ptr<op::Layer>& l) {
    if (l) {
      l->set_cuda_config(config);
      l->to_cuda();
    }
  };
  move(patch_embedding);
  for (auto& l : attn_norm) move(l);
  for (auto& l : qkv_proj) move(l);
  for (auto& l : out_proj) move(l);
  for (auto& l : ffn_norm) move(l);
  for (auto& l : fc1) move(l);
  for (auto& l : fc2) move(l);
  move(post_layernorm);
}

void PaddleOCRVLProjectorLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  auto move = [&](std::shared_ptr<op::Layer>& l) {
    if (l) {
      l->set_cuda_config(config);
      l->to_cuda();
    }
  };
  move(pre_norm);
  move(linear_1);
  move(linear_2);
  move(act);
}

// =============================================================================
//  Constructor
// =============================================================================
PaddleOCRVLModel::PaddleOCRVLModel(base::TokenizerType tokenizer_type, std::string token_path,
                                   std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypePaddleOCRVL,
            std::move(token_path), std::move(model_path), is_quant_model) {}

// =============================================================================
//  init
// =============================================================================
base::Status PaddleOCRVLModel::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("CPU device does not support int8 quant for PaddleOCR-VL.");
  }
  device_type_ = device_type;

  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    if (cudaGetLastError() != cudaSuccess) {
      return error::InternalError("PaddleOCR-VL: cuda stream create failed.");
    }
  }

  // 读权重 / 解析 config
  Status read_status = gen_model_from_file();
  if (!read_status) return read_status;

  // PaddleOCR-VL 自身的多模态配置；后续可由权重元信息覆盖
  static const PaddleOCRVLTransformerConfig default_vl_cfg{};
  vl_config_ = &default_vl_cfg;

  STATUS_CHECK(create_layers());
  init_mem();

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  mm_decode_step_ = 0;
  return error::Success();
}

// =============================================================================
//  create_layers
// =============================================================================
base::Status PaddleOCRVLModel::create_layers() {
  create_param_layers();
  create_nonparam_layers();
  if (is_quant_model_) {
    create_param_quant_layers();
  }
  return base::error::Success();
}

// ---------- 参数层 ----------
void PaddleOCRVLModel::create_param_layers() {
  CHECK(vl_config_ != nullptr);
  const auto& vc = vl_config_->vision;

  // 视觉算子目前只有 CPU 实现，强制使用 CPU device 创建权重张量；
  // 这样在 CUDA 模式下也不会触发未实现的 CUDA 路径。
  const base::DeviceType vis_dev = base::DeviceType::kDeviceCPU;

  // ----------- SigLIP Vision Encoder -----------
  siglip_layers_ = std::make_unique<SiglipVisionLayers>();

  siglip_layers_->patch_embedding = std::make_shared<op::PatchEmbedLayer>(
      vis_dev, vc.num_channels_, vc.hidden_size_, vc.patch_size_);

  siglip_layers_->attn_norm.resize(vc.num_hidden_layers_);
  siglip_layers_->qkv_proj.resize(vc.num_hidden_layers_);
  siglip_layers_->out_proj.resize(vc.num_hidden_layers_);
  siglip_layers_->ffn_norm.resize(vc.num_hidden_layers_);
  siglip_layers_->fc1.resize(vc.num_hidden_layers_);
  siglip_layers_->fc2.resize(vc.num_hidden_layers_);

  for (int32_t i = 0; i < vc.num_hidden_layers_; ++i) {
    siglip_layers_->attn_norm[i] =
        std::make_shared<op::LayerNormLayer>(vis_dev, vc.hidden_size_, vc.layer_norm_eps_);
    siglip_layers_->qkv_proj[i] =
        std::make_shared<op::MatmulLayer>(vis_dev, 3 * vc.hidden_size_, vc.hidden_size_,
                                          /*is_quant_layer=*/false, /*has_bias=*/true);
    siglip_layers_->out_proj[i] =
        std::make_shared<op::MatmulLayer>(vis_dev, vc.hidden_size_, vc.hidden_size_,
                                          false, true);
    siglip_layers_->ffn_norm[i] =
        std::make_shared<op::LayerNormLayer>(vis_dev, vc.hidden_size_, vc.layer_norm_eps_);
    siglip_layers_->fc1[i] =
        std::make_shared<op::MatmulLayer>(vis_dev, vc.intermediate_size_, vc.hidden_size_,
                                          false, true);
    siglip_layers_->fc2[i] =
        std::make_shared<op::MatmulLayer>(vis_dev, vc.hidden_size_, vc.intermediate_size_,
                                          false, true);
  }
  siglip_layers_->post_layernorm =
      std::make_shared<op::LayerNormLayer>(vis_dev, vc.hidden_size_, vc.layer_norm_eps_);

  // ----------- Projector MLP -----------
  projector_layers_ = std::make_unique<PaddleOCRVLProjectorLayers>();
  const int32_t merged = vc.merged_hidden();          // 1152 * 2 * 2 = 4608
  // 与 LLM 文本侧 hidden 严格对齐：优先使用 config_->dim_（来自权重），
  // 兜底再走 vl_config_->text_hidden_size_ 默认值。
  const int32_t text_h =
      (config_ && config_->dim_ > 0) ? config_->dim_ : vl_config_->text_hidden_size_;

  projector_layers_->pre_norm =
      std::make_shared<op::LayerNormLayer>(vis_dev, vc.hidden_size_, vc.layer_norm_eps_);
  projector_layers_->linear_1 =
      std::make_shared<op::MatmulLayer>(vis_dev, merged, merged, false, true);
  projector_layers_->act      = std::make_shared<op::GELULayer>(vis_dev);
  projector_layers_->linear_2 =
      std::make_shared<op::MatmulLayer>(vis_dev, text_h, merged, false, true);

  // ----------- Text LLM (复用 Qwen3Layers) -----------
  // Qwen3Layers 是只持有共享指针的容器，由权重加载阶段（gen_model_from_file
  // 的 PaddleOCR-VL 适配）按 config_ 填充。这里只构造空容器。
  qwen_layers_ = std::make_unique<Qwen3Layers>();
}

void PaddleOCRVLModel::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  if (!config_) {
    LOG(WARNING) << "PaddleOCR-VL: TransformerConfig not loaded; skip non-param layer creation.";
    return;
  }

  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_,
      config_->head_num_, config_->head_size_);
  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void PaddleOCRVLModel::create_param_quant_layers() {
  // PaddleOCR-VL 暂不支持量化分支
  LOG(WARNING) << "PaddleOCR-VL: quant layers not implemented yet.";
}

// =============================================================================
//  init_mem
// =============================================================================
void PaddleOCRVLModel::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc =
      device_type_ == base::DeviceType::kDeviceCPU
          ? std::static_pointer_cast<base::DeviceAllocator>(
                base::CPUDeviceAllocatorFactory::get_instance())
          : std::static_pointer_cast<base::DeviceAllocator>(
                base::CUDADeviceAllocatorFactory::get_instance());
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    // 视觉权重一律放 CPU；只把 LLM 部分搬到 CUDA。
    qwen_layers_->to_cuda(cuda_config_);
  }

  // ---------------- LLM 文本侧 hidden 维 ----------------
  // 必须与 Qwen3 embedding/cls layer 输出维度一致；优先取 config_->dim_。
  const int32_t text_hidden =
      (config_ && config_->dim_ > 0) ? config_->dim_ : vl_config_->text_hidden_size_;

  // ---------------- Vision intermediate buffers (CPU) ----------------
  // 视觉算子只有 CPU 实现 → 中间张量统一 CPU 分配。
  const auto& vc = vl_config_->vision;
  const int32_t max_patches =
      vl_config_->vision_max_tokens_ * vc.spatial_merge_size_ * vc.spatial_merge_size_;

  tensor::Tensor vision_hidden(base::DataType::kDataTypeFp32,
                               max_patches, vc.hidden_size_, true, alloc_cpu);
  tensor::Tensor projector_out(base::DataType::kDataTypeFp32,
                               vl_config_->vision_max_tokens_, text_hidden,
                               true, alloc_cpu);
  CHECK(insert_buffer(model::ModelBufferType::kVisionHidden,    vision_hidden));
  CHECK(insert_buffer(model::ModelBufferType::kProjectorOutput, projector_out));

  // ---------------- MRoPE positions (CPU, int32, [3, seq_len]) ----------------
  const int32_t seq_len = config_ ? config_->seq_len_ : 4096;
  tensor::Tensor mrope_pos(base::DataType::kDataTypeInt32, 3, seq_len, true, alloc_cpu);
  CHECK(insert_buffer(model::ModelBufferType::kMRoPEPositions, mrope_pos));

  // ---------------- 文本侧 token / embedding ----------------
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, text_hidden, true, alloc);
  CHECK(insert_buffer(model::ModelBufferType::kInputTokens,     input_tokens));
  CHECK(insert_buffer(model::ModelBufferType::kInputEmbeddings, input_embeddings));

  // 1D pos 张量（与 Qwen3Model::init_mem 行为一致），用作 KV cache 索引
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(model::ModelBufferType::kInputPos, pos_tensor));

  if (config_) {
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32,
                             config_->head_size_ * config_->seq_len_, true, alloc);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32,
                             config_->head_size_ * config_->seq_len_, true, alloc);
    CHECK(insert_buffer(model::ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(model::ModelBufferType::kCosCache, cos_cache));

    tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                             config_->seq_len_, config_->kv_dim_, true, alloc);
    tensor::Tensor val_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                             config_->seq_len_, config_->kv_dim_, true, alloc);
    CHECK(insert_buffer(model::ModelBufferType::kKeyCache,   key_cache));
    CHECK(insert_buffer(model::ModelBufferType::kValueCache, val_cache));

    tensor::Tensor forward_output(base::DataType::kDataTypeFp32,
                                  config_->vocab_size_, true, alloc);
    CHECK(insert_buffer(model::ModelBufferType::kForwardOutput, forward_output));
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32,
                                        config_->vocab_size_, true, alloc_cpu);
      CHECK(insert_buffer(model::ModelBufferType::kForwardOutputCPU, forward_output_cpu));
    }
  }
}

// =============================================================================
//  predict_multimodal: 多模态推理入口
//
//  Bug 修复要点：
//   - 不再用基类 fill_input(pos_tensor, ...) —— 它要求 pos_tensor 为 1D，而
//     mrope.positions 是 [3, L]，语义不符。
//   - 显式维护跨 step 的 decode 位置 mm_decode_step_，保证 KV-cache 索引连续。
//   - 视觉特征已在 embedding_multimodal 内通过 device-aware memcpy 注入；
//     这里直接把 input_embeddings 当成 LLM 的 token-level 输入即可。
// =============================================================================
base::Status PaddleOCRVLModel::predict_multimodal(const std::vector<int>& tokens,
                                                  const std::vector<ProcessedImage>& images,
                                                  bool is_prompt,
                                                  int& next_token) const {
  if (tokens.empty()) {
    return base::error::InvalidArgument("PaddleOCR-VL: tokens is empty.");
  }
  if (!config_) {
    return base::error::InternalError("PaddleOCR-VL: text config not loaded.");
  }

  // 1) 文本 + 图像融合后的 embedding (input_embeddings buffer 已被原地填充)
  auto embed_out = embedding_multimodal(tokens, images);

  // 2) 计算 3D-MRoPE 位置 (写到 kMRoPEPositions buffer)
  MRoPEPositions mrope = compute_mrope_positions(tokens, images);

  // 3) 构造 KV-cache 用的 1D pos_tensor
  //    - prompt 阶段 pos = 0；这里对 forward 提供起始位置；
  //    - decode 阶段 pos = mm_decode_step_（由历史步数累加而来）
  auto pos_tensor = get_buffer(model::ModelBufferType::kInputPos);
  if (is_prompt) {
    mm_decode_step_ = static_cast<int32_t>(tokens.size());
    pos_tensor.index<int32_t>(0) = 0;
  } else {
    pos_tensor.index<int32_t>(0) = mm_decode_step_;
    ++mm_decode_step_;
  }

  // 4) 喂入文本 LLM 主体
  int next = -1;
  STATUS_CHECK(forward(embed_out.input_embeddings, pos_tensor, next));

  // 5) prompt 阶段不出 token；decode 阶段从 logits 采样
  next_token = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

// =============================================================================
//  embedding / embedding_multimodal
// =============================================================================
op::EmbeddingOutput PaddleOCRVLModel::embedding(const std::vector<int>& tokens) const {
  CHECK(qwen_layers_ != nullptr);

  const int32_t text_hidden =
      (config_ && config_->dim_ > 0) ? config_->dim_ : vl_config_->text_hidden_size_;

  auto input_tokens     = get_buffer(model::ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(model::ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), text_hidden});
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens[i];
  }
  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));

  if (qwen_layers_->embedding_layer_) {
    STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num,
                                                         input_embeddings));
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "PaddleOCR-VL: embedding layer not loaded; input_embeddings left zero-filled.";
    if (auto buf = input_embeddings.get_buffer()) {
      auto* a = buf->allocator().get();
      if (a) a->memset_zero(buf->ptr(), input_embeddings.byte_size(), nullptr, true);
    }
  }
  return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

op::EmbeddingOutput PaddleOCRVLModel::embedding_multimodal(
    const std::vector<int>& tokens, const std::vector<ProcessedImage>& images) const {
  // 先按文本走一遍 embedding，得到完整序列的 embedding tensor
  auto out = embedding(tokens);
  if (images.empty()) return out;

  const int32_t img_id = vl_config_->image_token_id_;
  const int32_t merge  = vl_config_->vision.spatial_merge_size_;
  const int32_t hidden =
      (config_ && config_->dim_ > 0) ? config_->dim_ : vl_config_->text_hidden_size_;

  // 找到所有 image_token_id 占位符的位置
  std::vector<int32_t> img_pos;
  img_pos.reserve(images.size() * 64);
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == img_id) img_pos.push_back(static_cast<int32_t>(i));
  }
  if (img_pos.empty()) {
    LOG(WARNING) << "PaddleOCR-VL: tokens 中未发现 image_token_id="
                 << img_id << "，跳过图像注入。";
    return out;
  }

  // device-aware memcpy：input_embeddings 可能在 CUDA 上，
  // 而 vision feature 始终位于 CPU。
  auto dst_buf = out.input_embeddings.get_buffer();
  CHECK(dst_buf != nullptr);
  auto dst_alloc = dst_buf->allocator();
  CHECK(dst_alloc != nullptr);
  const bool dst_on_cuda = (out.input_embeddings.device_type() == base::DeviceType::kDeviceCUDA);
  const auto memcpy_kind = dst_on_cuda ? base::MemcpyKind::kMemcpyCPU2CUDA
                                       : base::MemcpyKind::kMemcpyCPU2CPU;
  void* stream = (dst_on_cuda && cuda_config_) ? (void*)cuda_config_->stream : nullptr;

  size_t img_token_idx = 0;
  for (const auto& img : images) {
    tensor::Tensor feat = encode_image(img.pixel_values, img.grid_thw);  // CPU, [n_img_tok, hidden]
    const int32_t n_img_tok = img.grid_thw.num_img_tokens(merge);
    CHECK_EQ(static_cast<int32_t>(feat.size()), n_img_tok * hidden)
        << "vision feature size mismatch with grid_thw";

    for (int32_t k = 0; k < n_img_tok && img_token_idx < img_pos.size();
         ++k, ++img_token_idx) {
      const int32_t dst_row = img_pos[img_token_idx];
      void* dst = static_cast<float*>(dst_buf->ptr()) + dst_row * hidden;
      const void* src = feat.ptr<float>() + k * hidden;
      dst_alloc->memcpy(src, dst, sizeof(float) * hidden, memcpy_kind, stream,
                        /*need_sync=*/!dst_on_cuda);
    }
  }
  if (dst_on_cuda && cuda_config_) {
    cudaStreamSynchronize(cuda_config_->stream);
  }

  return out;
}

// =============================================================================
//  encode_image: pixel_values -> projector_output (text_hidden 维度，CPU)
// =============================================================================
tensor::Tensor PaddleOCRVLModel::encode_image(const tensor::Tensor& pixel_values,
                                              const ImageGridTHW& grid_thw) const {
  const auto& vc = vl_config_->vision;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  const int32_t num_patches = grid_thw.num_patches();
  CHECK_GT(num_patches, 0) << "encode_image: grid_thw 必须为正，得到 t=" << grid_thw.t
                           << " h=" << grid_thw.h << " w=" << grid_thw.w;

  // 0) 像素 unfold: [T, C, H_pix, W_pix] → [num_patches, C*p*p]
  tensor::Tensor unfolded = _unfold_pixels(pixel_values, grid_thw);

  // 1) Patch embed → [num_patches, hidden_size]
  tensor::Tensor hidden(base::DataType::kDataTypeFp32, num_patches, vc.hidden_size_,
                        /*need_alloc=*/true, alloc_cpu);
  STATUS_CHECK(siglip_layers_->patch_embedding->forward(unfolded, hidden));

  // 2) Vision RoPE 预计算 (cos / sin)
  tensor::Tensor rot_cos, rot_sin;
  _build_vision_rope(grid_thw, rot_cos, rot_sin);

  // 3) N 层 encoder
  for (int32_t i = 0; i < vc.num_hidden_layers_; ++i) {
    _encoder_layer(i, rot_cos, rot_sin, hidden);
  }

  // 4) post layernorm
  tensor::Tensor post(base::DataType::kDataTypeFp32, num_patches, vc.hidden_size_,
                      true, alloc_cpu);
  STATUS_CHECK(siglip_layers_->post_layernorm->forward(hidden, post));

  // 5) projector + 2x2 spatial merge
  tensor::Tensor merged = _spatial_merge(post, grid_thw);   // [num_img_tok, merged_hidden]
  return _project(merged, grid_thw);                        // [num_img_tok, text_hidden]
}

// =============================================================================
//  _unfold_pixels:
//    输入  pixel_values shape = [T, C, H_pix, W_pix] (CPU, fp32)
//    输出  shape          = [T*Hg*Wg, C*p*p]
//
//  Conv2d(kernel=p, stride=p, padding=0) 等价：每个 patch 取 [c, dh, dw]
//  扁平为 c*p*p 维向量。
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_unfold_pixels(const tensor::Tensor& pixel_values,
                                                const ImageGridTHW& grid_thw) const {
  const auto& vc = vl_config_->vision;
  const int32_t Tg = grid_thw.t;
  const int32_t Hg = grid_thw.h;
  const int32_t Wg = grid_thw.w;
  const int32_t p  = vc.patch_size_;
  const int32_t C  = vc.num_channels_;
  const int32_t H_pix = Hg * p;
  const int32_t W_pix = Wg * p;
  const int32_t patch_dim = C * p * p;

  CHECK_EQ(pixel_values.size(),
           static_cast<size_t>(Tg) * C * H_pix * W_pix)
      << "_unfold_pixels: pixel_values size 与 grid_thw 不一致";

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor out(base::DataType::kDataTypeFp32,
                     Tg * Hg * Wg, patch_dim, true, alloc_cpu);

  const float* src = pixel_values.ptr<float>();
  float*       dst = out.ptr<float>();

  // src layout: [T, C, H_pix, W_pix]
  // 每帧大小 = C * H_pix * W_pix
  const size_t per_frame = static_cast<size_t>(C) * H_pix * W_pix;
  const size_t per_chan  = static_cast<size_t>(H_pix) * W_pix;

  for (int32_t ti = 0; ti < Tg; ++ti) {
    const float* frame = src + ti * per_frame;
    for (int32_t hi = 0; hi < Hg; ++hi) {
      for (int32_t wi = 0; wi < Wg; ++wi) {
        float* d = dst + ((ti * Hg + hi) * Wg + wi) * patch_dim;
        // 输出顺序：channel-major（与一般权重 [hidden, C*p*p] 约定一致）
        for (int32_t c = 0; c < C; ++c) {
          for (int32_t dh = 0; dh < p; ++dh) {
            const float* srow =
                frame + c * per_chan + (hi * p + dh) * W_pix + wi * p;
            std::memcpy(d + c * p * p + dh * p, srow, sizeof(float) * p);
          }
        }
      }
    }
  }
  return out;
}

// =============================================================================
//  _build_vision_rope
//   生成 [num_patches, head_dim/2] 的 cos / sin 表
// =============================================================================
void PaddleOCRVLModel::_build_vision_rope(const ImageGridTHW& grid_thw,
                                          tensor::Tensor& rot_cos,
                                          tensor::Tensor& rot_sin) const {
  const auto& vc = vl_config_->vision;
  const int32_t hd_half = vc.head_dim() / 2;
  const int32_t N = grid_thw.t * grid_thw.h * grid_thw.w;

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  rot_cos = tensor::Tensor(base::DataType::kDataTypeFp32, N, hd_half, true, alloc_cpu);
  rot_sin = tensor::Tensor(base::DataType::kDataTypeFp32, N, hd_half, true, alloc_cpu);

  float* c = rot_cos.ptr<float>();
  float* s = rot_sin.ptr<float>();

  const float base_freq = 10000.0f;
  for (int32_t idx = 0; idx < N; ++idx) {
    const int32_t hw = grid_thw.h * grid_thw.w;
    const int32_t ti = idx / hw;
    const int32_t hi = (idx % hw) / grid_thw.w;
    const int32_t wi = idx % grid_thw.w;
    // 简化策略：以 (h_idx + w_idx + t_idx*max_hw) 为 1D 等价位置；
    // 完整 2D-RoPE 在精度对齐阶段再补。
    const int32_t pos = ti * hw + hi * grid_thw.w + wi;
    for (int32_t k = 0; k < hd_half; ++k) {
      const float inv_freq = 1.0f / std::pow(base_freq,
                                             static_cast<float>(2 * k) /
                                                 static_cast<float>(vc.head_dim()));
      const float angle = pos * inv_freq;
      c[idx * hd_half + k] = std::cos(angle);
      s[idx * hd_half + k] = std::sin(angle);
    }
  }
}

// =============================================================================
//  _encoder_layer: 单层 ViT (Pre-LN, qkv 融合)
//
//  Bug 修复要点：
//   - 旧版按 [3, T, hidden] 切片是错的；fused QKV matmul 输出 layout 实际是
//     [T, 3*hidden]，每行内是 [q | k | v] 三段拼接。
//   - 应用 RoPE 到 q/k 的偶/奇维度。
//   - 不再借用 projector_layers_->act 做 GELU（避免与 _project 共享 IO 状态）。
// =============================================================================
void PaddleOCRVLModel::_encoder_layer(int32_t layer_i,
                                      const tensor::Tensor& rot_cos,
                                      const tensor::Tensor& rot_sin,
                                      tensor::Tensor& hidden) const {
  const auto& vc = vl_config_->vision;
  auto& attn_norm = siglip_layers_->attn_norm[layer_i];
  auto& qkv_proj  = siglip_layers_->qkv_proj[layer_i];
  auto& out_proj  = siglip_layers_->out_proj[layer_i];
  auto& ffn_norm  = siglip_layers_->ffn_norm[layer_i];
  auto& fc1       = siglip_layers_->fc1[layer_i];
  auto& fc2       = siglip_layers_->fc2[layer_i];

  const int32_t T  = static_cast<int32_t>(hidden.size()) / vc.hidden_size_;
  const int32_t H  = vc.num_attention_heads_;
  const int32_t HD = vc.head_dim();
  const int32_t HD_half = HD / 2;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // --- Self-Attention 子块 ---
  tensor::Tensor normed(base::DataType::kDataTypeFp32, T, vc.hidden_size_, true, alloc_cpu);
  STATUS_CHECK(attn_norm->forward(hidden, normed));

  tensor::Tensor qkv(base::DataType::kDataTypeFp32, T, 3 * vc.hidden_size_, true, alloc_cpu);
  STATUS_CHECK(qkv_proj->forward(normed, qkv));

  // 应用 RoPE 到 q / k 的 (偶, 偶+1) 维度对（GPT-NeoX/SigLIP 风格的成对旋转）
  // 注：rot_cos/rot_sin 的有效行数应 ≥ T；当上游 grid 与 token 数不一致时跳过。
  float* qkv_data = qkv.ptr<float>();
  const float* cos_p = rot_cos.ptr<float>();
  const float* sin_p = rot_sin.ptr<float>();
  const int32_t rope_rows = static_cast<int32_t>(rot_cos.size()) / std::max(HD_half, 1);
  if (rope_rows >= T && HD_half > 0) {
    auto apply_rope = [&](float* base) {
      for (int32_t i = 0; i < T; ++i) {
        for (int32_t h = 0; h < H; ++h) {
          float* x = base + i * 3 * vc.hidden_size_ + h * HD;
          const float* c = cos_p + i * HD_half;
          const float* s = sin_p + i * HD_half;
          for (int32_t k = 0; k < HD_half; ++k) {
            const float a = x[2 * k], b = x[2 * k + 1];
            x[2 * k]     = a * c[k] - b * s[k];
            x[2 * k + 1] = a * s[k] + b * c[k];
          }
        }
      }
    };
    // q 起始偏移 0，k 起始偏移 hidden
    apply_rope(qkv_data + 0);
    apply_rope(qkv_data + vc.hidden_size_);
  }

  // 朴素 multi-head self-attention (CPU 路径)
  // QKV layout: 每行 [q (hidden) | k (hidden) | v (hidden)]
  tensor::Tensor attn_out(base::DataType::kDataTypeFp32, T, vc.hidden_size_, true, alloc_cpu);
  {
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    float* o = attn_out.ptr<float>();
    std::memset(o, 0, sizeof(float) * T * vc.hidden_size_);
    auto Qptr = [&](int32_t i, int32_t h) {
      return qkv_data + i * 3 * vc.hidden_size_ + 0 * vc.hidden_size_ + h * HD;
    };
    auto Kptr = [&](int32_t j, int32_t h) {
      return qkv_data + j * 3 * vc.hidden_size_ + 1 * vc.hidden_size_ + h * HD;
    };
    auto Vptr = [&](int32_t j, int32_t h) {
      return qkv_data + j * 3 * vc.hidden_size_ + 2 * vc.hidden_size_ + h * HD;
    };
    std::vector<float> scores(T);
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t i = 0; i < T; ++i) {
        const float* qi = Qptr(i, h);
        for (int32_t j = 0; j < T; ++j) {
          const float* kj = Kptr(j, h);
          float ss = 0.f;
          for (int32_t d = 0; d < HD; ++d) ss += qi[d] * kj[d];
          scores[j] = ss * scale;
        }
        // softmax
        float m = scores[0];
        for (int32_t j = 1; j < T; ++j) m = std::max(m, scores[j]);
        float sum = 0.f;
        for (int32_t j = 0; j < T; ++j) {
          scores[j] = std::exp(scores[j] - m);
          sum += scores[j];
        }
        const float inv_sum = 1.f / sum;
        for (int32_t j = 0; j < T; ++j) scores[j] *= inv_sum;
        // out_i += sum_j scores[j] * v_j
        float* oi = o + i * vc.hidden_size_ + h * HD;
        for (int32_t j = 0; j < T; ++j) {
          const float* vj = Vptr(j, h);
          for (int32_t d = 0; d < HD; ++d) oi[d] += scores[j] * vj[d];
        }
      }
    }
  }

  tensor::Tensor proj_out(base::DataType::kDataTypeFp32, T, vc.hidden_size_, true, alloc_cpu);
  STATUS_CHECK(out_proj->forward(attn_out, proj_out));

  // residual: hidden = hidden + proj_out （视觉路径全在 CPU，可直接用 add）
  CHECK(qwen_layers_->add_layer_ != nullptr) << "VecAddLayer 未创建（config_ 是否已加载？）";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(hidden, proj_out, hidden));

  // --- FFN 子块 ---
  STATUS_CHECK(ffn_norm->forward(hidden, normed));

  tensor::Tensor h1(base::DataType::kDataTypeFp32, T, vc.intermediate_size_, true, alloc_cpu);
  STATUS_CHECK(fc1->forward(normed, h1));

  // GELU 原地激活（直接计算，避免与 projector 共用 layer 实例）
  {
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kBeta  = 0.044715f;
    float* x = h1.ptr<float>();
    const size_t n = h1.size();
    for (size_t i = 0; i < n; ++i) {
      float v = x[i];
      x[i] = 0.5f * v * (1.0f + std::tanh(kAlpha * (v + kBeta * v * v * v)));
    }
  }

  tensor::Tensor h2(base::DataType::kDataTypeFp32, T, vc.hidden_size_, true, alloc_cpu);
  STATUS_CHECK(fc2->forward(h1, h2));

  // residual
  STATUS_CHECK(qwen_layers_->add_layer_->forward(hidden, h2, hidden));
}

// =============================================================================
//  _spatial_merge: 2x2 patch merge -> [Tg * Hm * Wm, hidden * merge^2]
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_spatial_merge(const tensor::Tensor& hidden,
                                                const ImageGridTHW& grid_thw) const {
  const auto& vc = vl_config_->vision;
  const int32_t merge = vc.spatial_merge_size_;
  const int32_t Tg = grid_thw.t;
  const int32_t H  = grid_thw.h;
  const int32_t W  = grid_thw.w;
  CHECK_EQ(H % merge, 0) << "_spatial_merge: grid h 必须能被 merge 整除";
  CHECK_EQ(W % merge, 0) << "_spatial_merge: grid w 必须能被 merge 整除";

  const int32_t Hm = H / merge;
  const int32_t Wm = W / merge;
  const int32_t hidden_size = vc.hidden_size_;
  const int32_t merged_dim  = hidden_size * merge * merge;
  const int32_t num_img_tok = Tg * Hm * Wm;

  // 视觉路径全在 CPU，hidden tensor 也是 CPU 分配 → memcpy 安全
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor out(base::DataType::kDataTypeFp32, num_img_tok, merged_dim, true, alloc_cpu);
  const float* src = hidden.ptr<float>();
  float*       dst = out.ptr<float>();

  for (int32_t t = 0; t < Tg; ++t) {
    const float* src_t = src + t * (H * W) * hidden_size;
    float*       dst_t = dst + t * (Hm * Wm) * merged_dim;
    for (int32_t hm = 0; hm < Hm; ++hm) {
      for (int32_t wm = 0; wm < Wm; ++wm) {
        float* d = dst_t + (hm * Wm + wm) * merged_dim;
        int32_t off = 0;
        for (int32_t dh = 0; dh < merge; ++dh) {
          for (int32_t dw = 0; dw < merge; ++dw) {
            const int32_t r = (hm * merge + dh) * W + (wm * merge + dw);
            std::memcpy(d + off, src_t + r * hidden_size, sizeof(float) * hidden_size);
            off += hidden_size;
          }
        }
      }
    }
  }
  return out;
}

// =============================================================================
//  _project: linear_1 -> GELU -> linear_2 (输入已 spatial-merge)
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_project(const tensor::Tensor& vision_hidden,
                                          const ImageGridTHW& /*grid_thw*/) const {
  const auto& vc   = vl_config_->vision;
  const int32_t merged = vc.merged_hidden();
  CHECK_EQ(vision_hidden.size() % merged, 0u)
      << "_project: vision_hidden 与 merged_hidden 不对齐";
  const int32_t N  = static_cast<int32_t>(vision_hidden.size()) / merged;

  const int32_t text_hidden =
      (config_ && config_->dim_ > 0) ? config_->dim_ : vl_config_->text_hidden_size_;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor h1(base::DataType::kDataTypeFp32, N, merged, true, alloc_cpu);
  STATUS_CHECK(projector_layers_->linear_1->forward(vision_hidden, h1));
  STATUS_CHECK(projector_layers_->act->forward(h1, h1));

  tensor::Tensor out(base::DataType::kDataTypeFp32, N, text_hidden, true, alloc_cpu);
  STATUS_CHECK(projector_layers_->linear_2->forward(h1, out));
  return out;
}

// =============================================================================
//  _patch_embed: 单独暴露，用于权重检查 / 单测
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_patch_embed(const tensor::Tensor& pixel_values) const {
  const auto& vc = vl_config_->vision;
  const int32_t num_patches = static_cast<int32_t>(pixel_values.size()) /
                              (vc.num_channels_ * vc.patch_size_ * vc.patch_size_);
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor out(base::DataType::kDataTypeFp32, num_patches, vc.hidden_size_, true, alloc);
  STATUS_CHECK(siglip_layers_->patch_embedding->forward(pixel_values, out));
  return out;
}

// =============================================================================
//  compute_mrope_positions: 3D-MRoPE 位置编码 (t, h, w)
//
//  规则（简化自 Qwen2-VL / PaddleOCR-VL）：
//    - 文本 token：(p, p, p)，p 单调递增
//    - 图像 token：以 vision_start_token_id 之后插入图像；按 (t, h, w) 三个维度
//                  分别给予该 patch 的 (t_idx, h_idx, w_idx) + 起始 p
//    - 图像段长度 = grid.t * grid.h/merge * grid.w/merge
//    - 离开图像段后，p 重新回到「最大 (t,h,w)」 + 1 继续往后递增
//
//  Bug 修复：
//    - 复用 kMRoPEPositions 预分配 buffer，避免每步重新分配
//    - 当 token 序列中 image_token_id 数量不足 span 时，img_idx 不再误推进
// =============================================================================
MRoPEPositions PaddleOCRVLModel::compute_mrope_positions(
    const std::vector<int>& tokens, const std::vector<ProcessedImage>& images) const {
  MRoPEPositions ret;
  const int32_t L     = static_cast<int32_t>(tokens.size());
  const int32_t merge = vl_config_->vision.spatial_merge_size_;

  // 优先复用 buffer；如果 buffer 容量不够（极少数情况），再回落分配
  tensor::Tensor buf = get_buffer(model::ModelBufferType::kMRoPEPositions);
  const int32_t cap  = static_cast<int32_t>(buf.size()) / 3;
  if (cap >= L && L > 0) {
    buf.reshape({3, L});
    ret.positions = buf;
  } else {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    ret.positions = tensor::Tensor(base::DataType::kDataTypeInt32, 3, L, true, alloc_cpu);
  }

  int32_t* pos_t = ret.positions.ptr<int32_t>(0);
  int32_t* pos_h = pos_t + L;
  int32_t* pos_w = pos_h + L;

  int32_t p = 0;
  size_t img_idx = 0;
  for (int32_t i = 0; i < L; ++i) {
    const bool is_image_token = (tokens[i] == vl_config_->image_token_id_);

    if (is_image_token && img_idx < images.size()) {
      const auto& g = images[img_idx].grid_thw;
      const int32_t Tg = g.t;
      const int32_t Hg = g.h / merge;
      const int32_t Wg = g.w / merge;
      const int32_t span = Tg * Hg * Wg;
      const int32_t end  = std::min(L, i + span);
      const int32_t actual = end - i;

      for (int32_t k = 0; k < actual; ++k) {
        const int32_t ti = (k / (Hg * Wg));
        const int32_t hi = (k / Wg) % Hg;
        const int32_t wi = k % Wg;
        pos_t[i + k] = p + ti;
        pos_h[i + k] = p + hi;
        pos_w[i + k] = p + wi;
      }
      const int32_t max_off = std::max({Tg, Hg, Wg});
      p += max_off;
      i = end - 1;          // for 循环还会 ++i

      // 仅当整个 span 都在序列内时才推进 img_idx；
      // 否则保留给下一次（流式扩展场景）
      if (actual == span) ++img_idx;
    } else {
      pos_t[i] = p;
      pos_h[i] = p;
      pos_w[i] = p;
      ++p;
    }
  }

  ret.mrope_position_delta = p - L;
  return ret;
}

// =============================================================================
//  predict / forward / post_processing
// =============================================================================
base::Status PaddleOCRVLModel::predict(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor,
                                       bool is_prompt, int& next) const {
  STATUS_CHECK(forward(input, pos_tensor, next));
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

base::Status PaddleOCRVLModel::forward(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor,
                                       int& /*next*/) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("PaddleOCR-VL: input is empty.");
  }
  if (!config_) {
    return base::error::InternalError("PaddleOCR-VL: text config not loaded.");
  }
  // LLM 主体复用 Qwen3 的 transformer block；当 wq/wk/wv/wo/w1/w2/w3 等权重
  // 尚未加载时（典型于权重文件适配未完成阶段），不能继续往下走，避免读取
  // 未初始化的 forward_output buffer 误导上层采样。
  CHECK(qwen_layers_ != nullptr);
  const bool llm_loaded = qwen_layers_->wq_layers_.size() ==
                              static_cast<size_t>(config_->layer_num_) &&
                          qwen_layers_->wk_layers_.size() ==
                              static_cast<size_t>(config_->layer_num_) &&
                          qwen_layers_->cls_layer_ != nullptr;
  if (!llm_loaded) {
    LOG_FIRST_N(WARNING, 1)
        << "PaddleOCR-VL forward: LLM 权重尚未挂载到 qwen_layers_，"
        << "请先在 gen_model_from_file 中适配 PaddleOCR-VL 权重映射。";
    return base::error::NotImplemented(
        "PaddleOCR-VL LLM forward not yet wired with weights.");
  }

  // TODO: 调用 attention_rms / attention_qkv / attention_mha / feed_forward / cls_logits
  //       —— 建议把 Qwen3Model 内部对应函数抽成可复用的工具函数后在此调用。
  (void)input;
  (void)pos_tensor;
  return base::error::Success();
}

int32_t PaddleOCRVLModel::post_processing(const tensor::Tensor& /*pos*/, bool is_prompt) const {
  if (is_prompt || !sampler_) return -1;
  // 仅当 forward 真正写入了 logits（即 LLM 已挂载并跑过）才采样。
  if (!qwen_layers_ || !qwen_layers_->cls_layer_) {
    return -1;
  }
  const tensor::Tensor& fwd = get_buffer(model::ModelBufferType::kForwardOutput);
  return static_cast<int32_t>(
      sampler_->sample(fwd.ptr<float>(), fwd.size(),
                       cuda_config_ ? cuda_config_->stream : nullptr));
}

}  // namespace model
