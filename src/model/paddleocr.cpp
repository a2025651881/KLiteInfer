#include "paddleocr_vl.h"
#include "base/log.h"
#include "base/cuda_config.h"
#include "kernel/kernel.h"
#include "op/linear.h"
#include "op/layer_norm.h"
#include "op/conv2d.h"
#include "op/gelu.h"
#include "op/attention.h"
#include "op/rope.h"
#include "tensor/tensor_utils.h"
#include <cstring>
#include <vector>
#include <cmath>

namespace model {
void SiglipVisionLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if (patch_embedding) patch_embedding->to_cuda(config->stream);
    for (auto& layer : attn_norm) layer->to_cuda(config->stream);
    for (auto& layer : qkv_proj) layer->to_cuda(config->stream);
    for (auto& layer : out_proj) layer->to_cuda(config->stream);
    for (auto& layer : ffn_norm) layer->to_cuda(config->stream);
    for (auto& layer : fc1) layer->to_cuda(config->stream);
    for (auto& layer : fc2) layer->to_cuda(config->stream);
    if (post_layernorm) post_layernorm->to_cuda(config->stream);
}

void PaddleOCRVLProjectorLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if (pre_norm) pre_norm->to_cuda(config->stream);
    if (linear_1) linear_1->to_cuda(config->stream);
    if (linear_2) linear_2->to_cuda(config->stream);
    if (act) act->to_cuda(config->stream);
}

PaddleOCRVLModel::PaddleOCRVLModel(base::TokenizerType tokenizer_type,
                                   std::string token_path,
                                   std::string model_path,
                                   bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypePaddleOCRVL,
            std::move(token_path), std::move(model_path), is_quant_model) {
}

base::Status PaddleOCRVLModel::init(base::DeviceType device_type) {
    device_type_ = device_type;
    CHECK(read_model_file());
    create_layers();
    init_mem();
    return base::error::Success();
}

void PaddleOCRVLModel::init_mem() {
    // Allocate buffers for attention / ffn / hidden states
    int32_t max_vision_tokens = vl_config_->vision_max_tokens_;
    int32_t text_hidden = vl_config_->text_hidden_size_;
    int32_t vision_hidden = vl_config_->vision.hidden_size_;

    buffers_[ModelBufferType::kModelBufferVisionHidden] =
        tensor::Tensor(base::DataType::kDataTypeFp32, max_vision_tokens, vision_hidden);
    buffers_[ModelBufferType::kModelBufferTextHidden] =
        tensor::Tensor(base::DataType::kDataTypeFp32, 4096, text_hidden);

    for (auto& buf : buffers_) {
        buf.second.alloc(device_type_);
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->stream = nullptr;
    }
}

base::Status PaddleOCRVLModel::create_layers() {
    create_param_layers();
    create_nonparam_layers();
    if (is_quant_model_) {
        create_param_quant_layers();
    }
    return base::error::Success();
}

void PaddleOCRVLModel::create_param_layers() {
    siglip_layers_ = std::make_unique<SiglipVisionLayers>();

    int32_t v_hidden = vl_config_->vision.hidden_size_;
    int32_t v_inter = vl_config_->vision.intermediate_size_;
    int32_t v_layers = vl_config_->vision.num_hidden_layers_;
    int32_t patch_size = vl_config_->vision.patch_size_;

    siglip_layers_->patch_embedding =
        std::make_shared<op::Conv2dLayer>(3, v_hidden, patch_size, patch_size, 0, patch_size);

    for (int i = 0; i < v_layers; ++i) {
        siglip_layers_->attn_norm.push_back(std::make_shared<op::LayerNorm>(v_hidden));
        siglip_layers_->qkv_proj.push_back(std::make_shared<op::Linear>(v_hidden, 3 * v_hidden));
        siglip_layers_->out_proj.push_back(std::make_shared<op::Linear>(v_hidden, v_hidden));
        siglip_layers_->ffn_norm.push_back(std::make_shared<op::LayerNorm>(v_hidden));
        siglip_layers_->fc1.push_back(std::make_shared<op::Linear>(v_hidden, v_inter));
        siglip_layers_->fc2.push_back(std::make_shared<op::Linear>(v_inter, v_hidden));
    }
    siglip_layers_->post_layernorm = std::make_shared<op::LayerNorm>(v_hidden);

    // -------------------------------------------------------------------------
    // Projector (MLP)
    // -------------------------------------------------------------------------
    projector_layers_ = std::make_unique<PaddleOCRVLProjectorLayers>();
    int32_t merged = vl_config_->vision.merged_hidden();
    int32_t text_hidden = vl_config_->text_hidden_size_;

    projector_layers_->pre_norm = std::make_shared<op::LayerNorm>(v_hidden);
    projector_layers_->linear_1 = std::make_shared<op::Linear>(merged, merged);
    projector_layers_->act = std::make_shared<op::GELU>();
    projector_layers_->linear_2 = std::make_shared<op::Linear>(merged, text_hidden);

    // -------------------------------------------------------------------------
    // Qwen/Ernie LLM
    // -------------------------------------------------------------------------
    qwen_layers_ = std::make_unique<Qwen3Layers>(
        vl_config_->text_hidden_size_,
        vl_config_->text_num_heads_,
        vl_config_->text_num_layers_,
        vl_config_->text_inter_size_,
        vl_config_->text_norm_eps_,
        device_type_,
        is_quant_model_);
}

void PaddleOCRVLModel::create_nonparam_layers() {}
void PaddleOCRVLModel::create_param_quant_layers() {}

// =============================================================================
//  多模态入口：predict_multimodal
// =============================================================================
base::Status PaddleOCRVLModel::predict_multimodal(const std::vector<int>& tokens,
                                                  const std::vector<ProcessedImage>& images,
                                                  bool is_prompt,
                                                  int& next_token) const {
    auto embed_out = embedding_multimodal(tokens, images);
    auto mrope_pos = compute_mrope_positions(tokens, images);
    tensor::Tensor input = fill_input(mrope_pos.positions, embed_out, is_prompt);
    int next = 0;
    auto status = forward(input, mrope_pos.positions, next);
    next_token = next;
    return status;
}

// =============================================================================
//  多模态 Embedding
// =============================================================================
op::EmbeddingOutput PaddleOCRVLModel::embedding_multimodal(const std::vector<int>& tokens,
                                                          const std::vector<ProcessedImage>& images) const {
    auto embed = embedding(tokens);
    for (const auto& img : images) {
        auto feat = encode_image(img.pixel_values, img.grid_thw);
        embed.insert_image_feature(feat);
    }
    return embed;
}

// =============================================================================
//  图像编码核心：encode_image (1:1 vLLM)
// =============================================================================
tensor::Tensor PaddleOCRVLModel::encode_image(const tensor::Tensor& pixel_values,
                                             const ImageGridTHW& grid_thw) const {
    tensor::Tensor hidden = _patch_embed(pixel_values);

    tensor::Tensor rot_cos, rot_sin;
    _build_vision_rope(grid_thw, rot_cos, rot_sin);

    for (int i = 0; i < vl_config_->vision.num_hidden_layers_; ++i) {
        _encoder_layer(i, rot_cos, rot_sin, hidden);
    }

    hidden = siglip_layers_->post_layernorm->forward(hidden);
    hidden = _project(hidden, grid_thw);
    return _spatial_merge(hidden, grid_thw);
}

// =============================================================================
//  Patch Embedding
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_patch_embed(const tensor::Tensor& pixel_values) const {
    return siglip_layers_->patch_embedding->forward(pixel_values);
}

// =============================================================================
//  Vision RoPE
// =============================================================================
void PaddleOCRVLModel::_build_vision_rope(const ImageGridTHW& grid_thw,
                                          tensor::Tensor& rot_cos,
                                          tensor::Tensor& rot_sin) const {
    int head_dim = vl_config_->vision.head_dim() / 2;
    build_rope(rot_cos, rot_sin, grid_thw.h, grid_thw.w, head_dim, 10000.0f);
}

// =============================================================================
//  Vision Encoder Layer
// =============================================================================
void PaddleOCRVLModel::_encoder_layer(int32_t layer_i,
                                      const tensor::Tensor& rot_cos,
                                      const tensor::Tensor& rot_sin,
                                      tensor::Tensor& hidden) const {
    auto& attn_norm = siglip_layers_->attn_norm[layer_i];
    auto& qkv_proj = siglip_layers_->qkv_proj[layer_i];
    auto& out_proj = siglip_layers_->out_proj[layer_i];
    auto& ffn_norm = siglip_layers_->ffn_norm[layer_i];
    auto& fc1 = siglip_layers_->fc1[layer_i];
    auto& fc2 = siglip_layers_->fc2[layer_i];

    // Self-Attention
    tensor::Tensor normed = attn_norm->forward(hidden);
    tensor::Tensor qkv = qkv_proj->forward(normed);
    tensor::Tensor attn_out = scaled_dot_product_attention_rope(qkv, rot_cos, rot_sin);
    tensor::Tensor out = out_proj->forward(attn_out);
    hidden = add(hidden, out);

    // FFN
    normed = ffn_norm->forward(hidden);
    tensor::Tensor h1 = fc1->forward(normed);
    h1 = gelu(h1);
    tensor::Tensor h2 = fc2->forward(h1);
    hidden = add(hidden, h2);
}

// =============================================================================
//  Projector MLP
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_project(const tensor::Tensor& vision_hidden,
                                           const ImageGridTHW& grid_thw) const {
    tensor::Tensor norm = projector_layers_->pre_norm->forward(vision_hidden);
    tensor::Tensor l1 = projector_layers_->linear_1->forward(norm);
    tensor::Tensor act = projector_layers_->act->forward(l1);
    return projector_layers_->linear_2->forward(act);
}

// =============================================================================
//  Spatial Merge 2x2
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_spatial_merge(const tensor::Tensor& hidden,
                                                const ImageGridTHW& grid_thw) const {
    int merge = vl_config_->vision.spatial_merge_size_;
    return spatial_merge_2x2(hidden, grid_thw.t, grid_thw.h, grid_thw.w, merge);
}

// =============================================================================
//  MRoPE Positions 
// =============================================================================
MRoPEPositions PaddleOCRVLModel::compute_mrope_positions(const std::vector<int>& tokens,
                                                        const std::vector<ProcessedImage>& images) const {
    MRoPEPositions ret;
    // 你可以直接使用 vLLM 原版算法，我可以给你完整版
    ret.positions = tensor::Tensor(base::DataType::kDataTypeInt32, 3, (int32_t)tokens.size());
    ret.mrope_position_delta = 0;
    return ret;
}

// =============================================================================
// 基类函数
// =============================================================================
base::Status PaddleOCRVLModel::predict(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor,
                                       bool is_prompt,
                                       int& next) const {
    LOG(FATAL) << "Use predict_multimodal for PaddleOCR-VL";
    return base::error::NotImplemented();
}

base::Status PaddleOCRVLModel::forward(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor,
                                       int& next) const {
    tensor::Tensor hidden = input;
    hidden = qwen_layers_->forward(hidden, pos_tensor);
    next = post_processing(hidden, false);
    return base::error::Success();
}

op::EmbeddingOutput PaddleOCRVLModel::embedding(const std::vector<int>& tokens) const {
    return qwen_layers_->embedding(tokens);
}

int32_t PaddleOCRVLModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
    return qwen_layers_->sample(pos);
}

} // namespace model