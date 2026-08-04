// =============================================================================
//  PaddleOCR-VL
//
//  实现严格对照 HF 参考实现 modeling_paddleocr_vl.py 的推理路径：
//
//  视觉 (SigLIP-like, 27 层, CPU 显式计算):
//    pixel_values [N,3*14*14]
//      -> patch_embedding (等价 Conv2d(k=14,s=14) + bias) -> [N, 1152]
//      -> + bilinear 插值后的 position_embedding(27x27 -> h x w)
//      -> 27 x { LN1 -> QKV(+bias) -> 2D RoPE -> 全可见双向注意力 -> out_proj(+bias) -> +res
//                LN2 -> fc1(+bias) -> tanh-GELU -> fc2(+bias) -> +res }
//      -> post_layernorm
//
//  Projector (mlp_AR):
//      -> pre_norm(LayerNorm eps=1e-5，作用在 merge 之前)
//      -> 2x2 spatial merge，拼接顺序 (p1,p2,d)
//      -> linear_1(+bias) -> erf-GELU -> linear_2(+bias) -> [N/4, 1024]
//
//  文本 (ERNIE4.5 类, 18 层, GQA 16/2, head_dim 128, RMSNorm eps=1e-5):
//      3D-MRoPE，频率轴划分 j<16 -> t, 16<=j<40 -> h, j>=40 -> w，半分割配对(j, j+64)
//
//  注意：视觉算子与MRoPE 目前只有 CPU 实现，因此本模型强制在 CPU 上运行，
//        优先保证与参考实现的数值一致性。
// =============================================================================
#include "model/paddleocr.h"
#include <glog/logging.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>
#include "base/alloc.h"
#include "../op/kernels/kernels_interface.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"

namespace model {
namespace {

constexpr int32_t kKlvlMagic = 0x4C564C4B;  // "KLVL"
constexpr int32_t kKlvlVersion = 1;
constexpr int32_t kKlvlHeaderInts = 24;

/// 以 mmap 内存为后端创建只读权重视图（零拷贝）
tensor::Tensor view_weight(const void* ptr, const std::vector<int32_t>& dims) {
  tensor::Tensor t(base::DataType::kDataTypeFp32, dims, false, nullptr,
                   const_cast<void*>(ptr));
  t.set_device_type(base::DeviceType::kDeviceCPU);
  return t;
}

}  // namespace

// =============================================================================
//  构造 / 初始化
// =============================================================================
PaddleOCRVLModel::PaddleOCRVLModel(base::TokenizerType tokenizer_type, std::string token_path,
                                   std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypePaddleOCRVL, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status PaddleOCRVLModel::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  device_type_ = device_type;
  vision_device_ = device_type;

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

  if (device_type == DeviceType::kDeviceCUDA) {
    // 视觉权重是mmap 视图，文本权重挂在算子层上，两者分别上传
    _upload_vision_weights();
    qwen_layers_->to_cuda(cuda_config_);
    LOG(INFO) << "PaddleOCR-VL: vision encoder / projector 与文本 decoder 均走 CUDA。";
  }

  init_mem();
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  mm_decode_step_ = 0;
  mm_rope_pos_ = 0;
  return error::Success();
}

void PaddleOCRVLModel::_upload_vision_weights() {
  CHECK(cuda_config_ != nullptr);
  auto up = [this](tensor::Tensor& t) {
    if (!t.is_empty()) {
      t.to_cuda(cuda_config_->stream);
    }
  };
  up(siglip_->patch_w);
  up(siglip_->patch_b);
  up(siglip_->pos_embed);
  for (auto& L : siglip_->layers) {
    up(L.ln1_w); up(L.ln1_b);
    up(L.q_w); up(L.q_b);
    up(L.k_w); up(L.k_b);
    up(L.v_w); up(L.v_b);
    up(L.o_w); up(L.o_b);
    up(L.ln2_w); up(L.ln2_b);
    up(L.fc1_w); up(L.fc1_b);
    up(L.fc2_w); up(L.fc2_b);
  }
  up(siglip_->post_ln_w);
  up(siglip_->post_ln_b);
  up(projector_->pre_norm_w);
  up(projector_->pre_norm_b);
  up(projector_->linear1_w);
  up(projector_->linear1_b);
  up(projector_->linear2_w);
  up(projector_->linear2_b);
  cudaStreamSynchronize(cuda_config_->stream);
}

tensor::Tensor PaddleOCRVLModel::_vision_alloc(int32_t rows, int32_t cols) const {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (vision_device_ == base::DeviceType::kDeviceCUDA) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }
  return tensor::Tensor(base::DataType::kDataTypeFp32, rows, cols, true, alloc);
}

void PaddleOCRVLModel::_vision_sync() const {
  if (vision_device_ == base::DeviceType::kDeviceCUDA && cuda_config_ != nullptr) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

// =============================================================================
//  权重文件解析（KLVL 扩展头）
// =============================================================================
base::Status PaddleOCRVLModel::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("PaddleOCR-VL: the model path is empty.");
  }

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_);
  }
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the weight file " + model_path_);
  }

  int32_t hdr[kKlvlHeaderInts] = {0};
  if (fread(hdr, sizeof(int32_t), kKlvlHeaderInts, file) != kKlvlHeaderInts) {
    fclose(file);
    close(fd);
    return error::ModelParseError("PaddleOCR-VL: failed to read the KLVL header.");
  }
  if (hdr[0] != kKlvlMagic) {
    fclose(file);
    close(fd);
    return error::ModelParseError(
        "PaddleOCR-VL: bad magic, 请用 tools/export_paddleocr_vl.py 导出权重。");
  }
  if (hdr[1] != kKlvlVersion) {
    fclose(file);
    close(fd);
    return error::ModelParseError("PaddleOCR-VL: unsupported KLVL version " +
                                  std::to_string(hdr[1]));
  }

  vl_config_ = std::make_unique<PaddleOCRVLTransformerConfig>();
  auto& vc = vl_config_->vision;
  vc.hidden_size_ = hdr[2];
  vc.num_hidden_layers_ = hdr[3];
  vc.num_attention_heads_ = hdr[4];
  vc.intermediate_size_ = hdr[5];
  vc.patch_size_ = hdr[6];
  vc.spatial_merge_size_ = hdr[7];
  vc.pos_grid_ = hdr[8];

  vl_config_->text_hidden_size_ = hdr[9];
  vl_config_->text_inter_size_ = hdr[10];
  vl_config_->text_num_layers_ = hdr[11];
  vl_config_->text_num_heads_ = hdr[12];
  vl_config_->text_num_kv_heads_ = hdr[13];
  vl_config_->text_head_dim_ = hdr[14];
  vl_config_->text_vocab_size_ = hdr[15];
  vl_config_->image_token_id_ = hdr[17];
  vl_config_->vision_start_token_id_ = hdr[18];
  vl_config_->mrope_section_t_ = hdr[19];
  vl_config_->mrope_section_h_ = hdr[20];
  vl_config_->mrope_section_w_ = hdr[21];

  // 文本 decoder 的运行期配置
  config_->dim_ = vl_config_->text_hidden_size_;
  config_->hidden_dim_ = vl_config_->text_hidden_size_;
  config_->immediate_dim_ = vl_config_->text_inter_size_;
  config_->layer_num_ = vl_config_->text_num_layers_;
  config_->head_num_ = vl_config_->text_num_heads_;
  config_->kv_head_num_ = vl_config_->text_num_kv_heads_;
  config_->head_size_ = vl_config_->text_head_dim_;
  config_->q_dim_ = config_->head_num_ * config_->head_size_;
  config_->kv_dim_ = config_->kv_head_num_ * config_->head_size_;
  config_->kv_mul_ = config_->head_num_ / config_->kv_head_num_;
  config_->vocab_size_ = vl_config_->text_vocab_size_;
  config_->seq_len_ = hdr[16];
  config_->is_shared_weight_ = hdr[22] != 0;

  raw_model_data_ = std::make_shared<RawModelDataFp32>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = static_cast<size_t>(ftell(file));
  fclose(file);

  raw_model_data_->fd = fd;
  raw_model_data_->data = mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("PaddleOCR-VL: mmap failed for " + model_path_);
  }
  raw_model_data_->header_size = sizeof(int32_t) * kKlvlHeaderInts;
  raw_model_data_->weight_data =
      static_cast<int8_t*>(raw_model_data_->data) + raw_model_data_->header_size;

  LOG(INFO) << "PaddleOCR-VL: vision " << vc.hidden_size_ << "d x " << vc.num_hidden_layers_
            << "L, text " << config_->dim_ << "d x " << config_->layer_num_ << "L, GQA "
            << config_->head_num_ << "/" << config_->kv_head_num_ << ", head_dim "
            << config_->head_size_ << ", vocab " << config_->vocab_size_;
  return error::Success();
}

base::Status PaddleOCRVLModel::create_layers() {
  create_param_layers();
  create_nonparam_layers();
  if (!siglip_ || !projector_ || !qwen_layers_ || !qwen_layers_->cls_layer_) {
    return base::error::InternalError("PaddleOCR-VL: create layers failed.");
  }
  return base::error::Success();
}

// =============================================================================
//  权重加载：顺序必须与 tools/export_paddleocr_vl.py 严格一致
// =============================================================================
void PaddleOCRVLModel::create_param_layers() {
  CHECK(raw_model_data_ != nullptr);
  CHECK(vl_config_ != nullptr);
  const auto& vc = vl_config_->vision;

  const int32_t vh = vc.hidden_size_;
  const int32_t vi = vc.intermediate_size_;
  const int32_t patch_in = vc.num_channels_ * vc.patch_size_ * vc.patch_size_;  // 588
  const int32_t merged = vc.merged_hidden();// 4608
  const int32_t td = config_->dim_;
  const int32_t ti = config_->immediate_dim_;
  const int32_t q_dim = config_->q_dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t vocab = config_->vocab_size_;

  size_t cursor = 0;
  auto take = [&](size_t count) {
    const void* p = raw_model_data_->weight(cursor);
    cursor += count;
    return p;
  };

  // ---------------- 视觉编码器 ----------------
  siglip_ = std::make_unique<SiglipVisionWeights>();
  siglip_->patch_w = view_weight(take(static_cast<size_t>(vh) * patch_in), {vh, patch_in});
  siglip_->patch_b = view_weight(take(vh), {vh});
  siglip_->pos_embed = view_weight(take(static_cast<size_t>(vc.pos_grid_) * vc.pos_grid_ * vh),
                                   {vc.pos_grid_ * vc.pos_grid_, vh});

  siglip_->layers.resize(vc.num_hidden_layers_);
  for (int32_t i = 0; i < vc.num_hidden_layers_; ++i) {
    auto& L = siglip_->layers[i];
    L.ln1_w = view_weight(take(vh), {vh});
    L.ln1_b = view_weight(take(vh), {vh});
    L.q_w = view_weight(take(static_cast<size_t>(vh) * vh), {vh, vh});
    L.q_b = view_weight(take(vh), {vh});
    L.k_w = view_weight(take(static_cast<size_t>(vh) * vh), {vh, vh});
    L.k_b = view_weight(take(vh), {vh});
    L.v_w = view_weight(take(static_cast<size_t>(vh) * vh), {vh, vh});
    L.v_b = view_weight(take(vh), {vh});
    L.o_w = view_weight(take(static_cast<size_t>(vh) * vh), {vh, vh});
    L.o_b = view_weight(take(vh), {vh});
    L.ln2_w = view_weight(take(vh), {vh});
    L.ln2_b = view_weight(take(vh), {vh});
    L.fc1_w = view_weight(take(static_cast<size_t>(vi) * vh), {vi, vh});
    L.fc1_b = view_weight(take(vi), {vi});
    L.fc2_w = view_weight(take(static_cast<size_t>(vh) * vi), {vh, vi});
    L.fc2_b = view_weight(take(vh), {vh});
  }
  siglip_->post_ln_w = view_weight(take(vh), {vh});
  siglip_->post_ln_b = view_weight(take(vh), {vh});

  // ---------------- Projector ----------------
  projector_ = std::make_unique<PaddleOCRVLProjectorWeights>();
  projector_->pre_norm_w = view_weight(take(vh), {vh});
  projector_->pre_norm_b = view_weight(take(vh), {vh});
  projector_->linear1_w =
      view_weight(take(static_cast<size_t>(merged) * merged), {merged, merged});
  projector_->linear1_b = view_weight(take(merged), {merged});
  projector_->linear2_w = view_weight(take(static_cast<size_t>(td) * merged), {td, merged});
  projector_->linear2_b = view_weight(take(td), {td});

  // ---------------- 文本解码器 ----------------
  const auto cpu = base::DeviceType::kDeviceCPU;
  qwen_layers_ = std::make_unique<Qwen3Layers>();

  auto embedding_layer =
      std::make_shared<op::EmbeddingLayer>(device_type_, td, config_->seq_len_, vocab);
  embedding_layer->set_weight(0, {vocab, td}, take(static_cast<size_t>(vocab) * td), cpu);
  qwen_layers_->embedding_layer_ = embedding_layer;

  // 每层顺序：input_ln, q, k, v, o, post_ln, gate, down, up
  std::vector<const void*> attn_norm(config_->layer_num_), ffn_norm(config_->layer_num_);
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    attn_norm[i] = take(td);

    auto wq = std::make_shared<op::MatmulLayer>(device_type_, q_dim, td);
    wq->set_weight(0, {q_dim, td}, take(static_cast<size_t>(q_dim) * td), cpu);
    qwen_layers_->wq_layers_.push_back(wq);

    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, td);
    wk->set_weight(0, {kv_dim, td}, take(static_cast<size_t>(kv_dim) * td), cpu);
    qwen_layers_->wk_layers_.push_back(wk);

    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, td);
    wv->set_weight(0, {kv_dim, td}, take(static_cast<size_t>(kv_dim) * td), cpu);
    qwen_layers_->wv_layers_.push_back(wv);

    auto wo = std::make_shared<op::MatmulLayer>(device_type_, td, q_dim);
    wo->set_weight(0, {td, q_dim}, take(static_cast<size_t>(td) * q_dim), cpu);
    qwen_layers_->wo_layers_.push_back(wo);

    ffn_norm[i] = take(td);

    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, ti, td);
    w1->set_weight(0, {ti, td}, take(static_cast<size_t>(ti) * td), cpu);
    qwen_layers_->w1_layers_.push_back(w1);

    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, td, ti);
    w2->set_weight(0, {td, ti}, take(static_cast<size_t>(td) * ti), cpu);
    qwen_layers_->w2_layers_.push_back(w2);

    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, ti, td);
    w3->set_weight(0, {ti, td}, take(static_cast<size_t>(ti) * td), cpu);
    qwen_layers_->w3_layers_.push_back(w3);
  }
  const void* final_norm = take(td);

  // rmsnorm 顺序：attn norm(L) + ffn norm(L) + final norm(1)
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto n = std::make_shared<op::RmsNormLayer>(device_type_, td);
    n->set_weight(0, {td}, attn_norm[i], cpu);
    qwen_layers_->rmsnorm_layers_.push_back(n);
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto n = std::make_shared<op::RmsNormLayer>(device_type_, td);
    n->set_weight(0, {td}, ffn_norm[i], cpu);
    qwen_layers_->rmsnorm_layers_.push_back(n);
  }
  {
    auto n = std::make_shared<op::RmsNormLayer>(device_type_, td);
    n->set_weight(0, {td}, final_norm, cpu);
    qwen_layers_->rmsnorm_layers_.push_back(n);
  }

  auto cls = std::make_shared<op::MatmulLayer>(device_type_, vocab, td);
  const void* cls_ptr = config_->is_shared_weight_
                            ? raw_model_data_->weight(0)  // 共享 embedding（本模型为 false）
                            : take(static_cast<size_t>(vocab) * td);
  cls->set_weight(0, {vocab, td}, cls_ptr, cpu);
  qwen_layers_->cls_layer_ = cls;

  const size_t expect = cursor * sizeof(float) + raw_model_data_->header_size;
  CHECK_EQ(raw_model_data_->file_size, expect)
      << "PaddleOCR-VL权重文件大小与布局不符：期望 " << expect << " 实际 "
      << raw_model_data_->file_size << "，请确认导出脚本与 create_param_layers 一致。";
}

void PaddleOCRVLModel::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);
  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);
  // RoPE 不走通用层：3D-MRoPE需要逐 token 的 cos/sin，见 _build_mrope_cos_sin
}

void PaddleOCRVLModel::create_param_quant_layers() {
  LOG(FATAL) << "PaddleOCR-VL: 量化推理尚未实现。";
}

void PaddleOCRVLModel::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }
  const int32_t td = config_->dim_;

  // 输入 token / pos 由主机侧写入，固定放CPU
  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, cpu_alloc);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, td, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, cpu_alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, td, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor out_mha(base::DataType::kDataTypeFp32, config_->q_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->q_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  tensor::Tensor attn_output(base::DataType::kDataTypeFp32, td, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
  tensor::Tensor score(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                       alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, score));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor val_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, val_cache));

  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

// =============================================================================
//  视觉：patch embedding + 位置编码（bilinear 插值，align_corners=false）
// =============================================================================
void PaddleOCRVLModel::_vision_embeddings(const float* pixel_dev, const ImageGridTHW& grid,
                                          float* out) const {
  const auto& vc = vl_config_->vision;
  const int32_t dim = vc.hidden_size_;
  const int32_t patch_in = vc.num_channels_ * vc.patch_size_ * vc.patch_size_;
  const int32_t num_patches = grid.num_patches();
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;

  // patch embedding：输入每个 patch 内部是连续的 C*p*p，等价于对 588 维做 Linear
  kernel::get_vision_gemm_nt_kernel(vision_device_)(
      pixel_dev, siglip_->patch_w.ptr<float>(), siglip_->patch_b.ptr<float>(), num_patches,
      patch_in, dim, out, stream);
  _dump("patch_embed", out, static_cast<size_t>(num_patches) * dim);

  // 位置编码：[27,27] 插值到 (h, w) 后按 t 逐帧累加
  kernel::get_vision_pos_embed_kernel(vision_device_)(siglip_->pos_embed.ptr<float>(),
                                                      vc.pos_grid_, grid.h, grid.w, dim, grid.t,
                                                      out, stream);
  _dump("vis_embed", out, static_cast<size_t>(num_patches) * dim);
}

// =============================================================================
//  视觉 2D RoPE：head_dim=72 -> 36 对，布局 [h(18) | w(18)]，配对偏移 36
// =============================================================================
void PaddleOCRVLModel::_build_vision_rope(const ImageGridTHW& grid, tensor::Tensor& cos_tab,
                                tensor::Tensor& sin_tab) const {
  const auto& vc = vl_config_->vision;
  const int32_t head_dim = vc.head_dim();  // 72
  const int32_t half = head_dim / 2;       // 36
  const int32_t nfreq = half / 2;          // 18
  const int32_t n = grid.num_patches();

  std::vector<float> inv_freq(nfreq);
  for (int32_t k = 0; k < nfreq; ++k) {
    inv_freq[k] = 1.f / std::pow(vc.rope_theta_,
                                 static_cast<float>(2 * k) / static_cast<float>(half));
  }

  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  cos_tab = tensor::Tensor(base::DataType::kDataTypeFp32, n, half, true, cpu_alloc);
  sin_tab = tensor::Tensor(base::DataType::kDataTypeFp32, n, half, true, cpu_alloc);
  float* cos_p = cos_tab.ptr<float>();
  float* sin_p = sin_tab.ptr<float>();

  for (int32_t p = 0; p < n; ++p) {
    const int32_t pid = p % (grid.h * grid.w);  // 每帧独立从 0 开始
    const int32_t h_id = pid / grid.w;
    const int32_t w_id = pid % grid.w;
    float* c = cos_p + static_cast<size_t>(p) * half;
    float* s = sin_p + static_cast<size_t>(p) * half;
    for (int32_t k = 0; k < nfreq; ++k) {
      const float ah = static_cast<float>(h_id) * inv_freq[k];
      const float aw = static_cast<float>(w_id) * inv_freq[k];
      c[k] = std::cos(ah);
      s[k] = std::sin(ah);
      c[nfreq + k] = std::cos(aw);
      s[nfreq + k] = std::sin(aw);
    }
  }

  if (vision_device_ == base::DeviceType::kDeviceCUDA) {
    cos_tab.to_cuda(cuda_config_->stream);
    sin_tab.to_cuda(cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

// =============================================================================
//  单层 vision encoder
// =============================================================================
void PaddleOCRVLModel::_vision_encoder_layer(int32_t layer_i, int32_t n, const float* cos_tab,
                                             const float* sin_tab, float* hidden,
                                             VisionWorkspace& ws) const {
  const auto& vc = vl_config_->vision;
  const int32_t dim = vc.hidden_size_;
  const int32_t inter = vc.intermediate_size_;
  const int32_t heads = vc.num_attention_heads_;
  const int32_t hd = vc.head_dim();  // 72
  const auto& L = siglip_->layers[layer_i];
  const size_t total = static_cast<size_t>(n) * dim;
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;

  auto gemm = kernel::get_vision_gemm_nt_kernel(vision_device_);
  auto layernorm = kernel::get_vision_layernorm_kernel(vision_device_);
  auto residual = kernel::get_vision_residual_kernel(vision_device_);

  float* normed = ws.normed.ptr<float>();
  float* q = ws.q.ptr<float>();
  float* k = ws.k.ptr<float>();
  float* v = ws.v.ptr<float>();
  float* attn = ws.attn.ptr<float>();

  // ---- pre-norm + QKV(+bias) ----
  layernorm(hidden, n, dim, L.ln1_w.ptr<float>(), L.ln1_b.ptr<float>(), vc.layer_norm_eps_, normed,
            stream);
  gemm(normed, L.q_w.ptr<float>(), L.q_b.ptr<float>(), n, dim, dim, q, stream);
  gemm(normed, L.k_w.ptr<float>(), L.k_b.ptr<float>(), n, dim, dim, k, stream);
  gemm(normed, L.v_w.ptr<float>(), L.v_b.ptr<float>(), n, dim, dim, v, stream);

  // ---- 2D RoPE：半分割配对 (d, d+36)，q/k 同时旋转 ----
  kernel::get_vision_rope2d_kernel(vision_device_)(q, k, cos_tab, sin_tab, n, dim, heads, hd,
                                                   stream);

  // ---- 全可见双向注意力（单图无 mask、非因果）----
  kernel::get_vision_attention_kernel(vision_device_)(q, k, v, n, dim, heads, hd, attn,
                                                      ws.score.ptr<float>(),
                                                      ws.score.get_dim(0), stream);

  // ---- out_proj(+bias) + 残差 ----
  gemm(attn, L.o_w.ptr<float>(), L.o_b.ptr<float>(), n, dim, dim, normed, stream);
  residual(hidden, normed, total, stream);

  // ---- pre-norm + MLP(tanh GELU) + 残差 ----
  layernorm(hidden, n, dim, L.ln2_w.ptr<float>(), L.ln2_b.ptr<float>(), vc.layer_norm_eps_, normed,
            stream);
  float* ff = ws.ff.ptr<float>();
  gemm(normed, L.fc1_w.ptr<float>(), L.fc1_b.ptr<float>(), n, dim, inter, ff, stream);
  kernel::get_vision_gelu_kernel(vision_device_)(ff, static_cast<size_t>(n) * inter,
                                                 kernel::GeluKind::kTanh, stream);
  gemm(ff, L.fc2_w.ptr<float>(), L.fc2_b.ptr<float>(), n, inter, dim, normed, stream);
  residual(hidden, normed, total, stream);
}

// =============================================================================
//  Projector：pre_norm -> 2x2 merge -> linear_1 -> erf GELU -> linear_2
// =============================================================================
tensor::Tensor PaddleOCRVLModel::_project(const float* vision_hidden,
                                          const ImageGridTHW& grid) const {
  const auto& vc = vl_config_->vision;
  const int32_t dim = vc.hidden_size_;
  const int32_t m = vc.spatial_merge_size_;
  const int32_t merged = vc.merged_hidden();
  const int32_t td = config_->dim_;
  const int32_t n = grid.num_patches();
  const int32_t n_tok = grid.t * (grid.h / m) * (grid.w / m);
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;

  auto gemm = kernel::get_vision_gemm_nt_kernel(vision_device_);

  // pre_norm 作用在 merge 之前，eps=1e-5（与 vision 的 1e-6 不同）
  tensor::Tensor normed = _vision_alloc(n, dim);
  kernel::get_vision_layernorm_kernel(vision_device_)(
      vision_hidden, n, dim, projector_->pre_norm_w.ptr<float>(),
      projector_->pre_norm_b.ptr<float>(), vl_config_->projector_norm_eps_, normed.ptr<float>(),
      stream);

  // 2x2 merge：拼接顺序 (p1, p2, d)，每个 patch 的 dim 维整块连续
  tensor::Tensor merged_buf = _vision_alloc(n_tok, merged);
  kernel::get_vision_spatial_merge_kernel(vision_device_)(
      normed.ptr<float>(), grid.t, grid.h, grid.w, dim, m, merged_buf.ptr<float>(), stream);

  tensor::Tensor h1 = _vision_alloc(n_tok, merged);
  gemm(merged_buf.ptr<float>(), projector_->linear1_w.ptr<float>(),
       projector_->linear1_b.ptr<float>(), n_tok, merged, merged, h1.ptr<float>(), stream);
  // Projector 用精确 erf GELU，与视觉 MLP 的 tanh 近似不同
  kernel::get_vision_gelu_kernel(vision_device_)(
      h1.ptr<float>(), static_cast<size_t>(n_tok) * merged, kernel::GeluKind::kErf, stream);

  tensor::Tensor out = _vision_alloc(n_tok, td);
  gemm(h1.ptr<float>(), projector_->linear2_w.ptr<float>(), projector_->linear2_b.ptr<float>(),
       n_tok, merged, td, out.ptr<float>(), stream);
  _vision_sync();
  // 视觉特征留在原设备上，由 embedding_multimodal 直接 scatter（同设备拷贝）
  _dump("projector", out.ptr<float>(), static_cast<size_t>(n_tok) * td);
  return out;
}

// =============================================================================
//  图像编码入口
// =============================================================================
tensor::Tensor PaddleOCRVLModel::encode_image(const tensor::Tensor& pixel_values,
                                              const ImageGridTHW& grid_thw) const {
  CHECK(siglip_ != nullptr && projector_ != nullptr);
  const auto& vc = vl_config_->vision;
  const int32_t n = grid_thw.num_patches();
  CHECK_GT(n, 0);
  CHECK_EQ(grid_thw.h % vc.spatial_merge_size_, 0);
  CHECK_EQ(grid_thw.w % vc.spatial_merge_size_, 0);

  const int32_t dim = vc.hidden_size_;
  const int32_t heads = vc.num_attention_heads_;
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;
  const auto t_begin = std::chrono::steady_clock::now();

  // 输入像素按需上传显存
  tensor::Tensor pixel = pixel_values;
  if (vision_device_ == base::DeviceType::kDeviceCUDA) {
    pixel = pixel_values.clone();
    pixel.to_cuda(cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  }

  tensor::Tensor hidden = _vision_alloc(n, dim);
  _vision_embeddings(pixel.ptr<float>(), grid_thw, hidden.ptr<float>());

  tensor::Tensor cos_tab, sin_tab;
  _build_vision_rope(grid_thw, cos_tab, sin_tab);

  // 27 层复用同一份中间缓冲。CPU 版注意力逐行复用 score，只需 heads 行；
  // CUDA 版用 batched GEMM 一次算出全部 [heads, n, n]。
  VisionWorkspace ws;
  ws.normed = _vision_alloc(n, dim);
  ws.q = _vision_alloc(n, dim);
  ws.k = _vision_alloc(n, dim);
  ws.v = _vision_alloc(n, dim);
  ws.attn = _vision_alloc(n, dim);
  ws.ff = _vision_alloc(n, vc.intermediate_size_);
  // score 行数由kernel 决定：CPU 逐行复用只需 heads 行；CUDA 一次算整块，
  // 大图上会按 head 分块（否则 heads*n*n 超过 int32 上限）
  ws.score = _vision_alloc(kernel::vision_attention_score_rows(vision_device_, n, heads), n);

  for (int32_t i = 0; i < vc.num_hidden_layers_; ++i) {
    _vision_encoder_layer(i, n, cos_tab.ptr<float>(), sin_tab.ptr<float>(), hidden.ptr<float>(),
                          ws);
    if (i == 0 || i == 1 || i == vc.num_hidden_layers_ - 1) {
      const std::string tag = (i == vc.num_hidden_layers_ - 1) ? "vis_layer_last"
                                                               : "vis_layer" + std::to_string(i);
      _dump(tag, hidden.ptr<float>(), static_cast<size_t>(n) * dim);
    }
  }

  // post_layernorm
  tensor::Tensor post = _vision_alloc(n, dim);
  kernel::get_vision_layernorm_kernel(vision_device_)(
      hidden.ptr<float>(), n, dim, siglip_->post_ln_w.ptr<float>(),
      siglip_->post_ln_b.ptr<float>(), vc.layer_norm_eps_, post.ptr<float>(), stream);
  _dump("vis_post_ln", post.ptr<float>(), static_cast<size_t>(n) * dim);

  auto out = _project(post.ptr<float>(), grid_thw);
  _vision_sync();
  last_vision_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_begin)
                        .count();
  LOG(INFO) << "PaddleOCR-VL: vision encode 耗时 " << last_vision_ms_ << " ms（" << n
            << " patches, "
            << (vision_device_ == base::DeviceType::kDeviceCUDA ? "CUDA" : "CPU") << "）";
  return out;
}

void PaddleOCRVLModel::_dump(const std::string& name, const float* data, size_t count) const {
  if (dump_dir_.empty()) {
    return;
  }
  // 只保留第一次（prefill）的结果，decode 阶段的同名张量不覆盖
  if (std::find(dumped_.begin(), dumped_.end(), name) != dumped_.end()) {
    return;
  }
  dumped_.push_back(name);

  // 视觉路径的中间结果可能在显存里，需要先同步并拷回主机
  const float* host = data;
  std::vector<float> staging;
  cudaPointerAttributes attr{};
  if (cudaPointerGetAttributes(&attr, data) == cudaSuccess &&
      attr.type == cudaMemoryTypeDevice) {
    _vision_sync();
    staging.resize(count);
    cudaMemcpy(staging.data(), data, sizeof(float) * count, cudaMemcpyDeviceToHost);
    host = staging.data();
  }
  cudaGetLastError();  // 清掉指针属性查询可能留下的错误状态

  const std::string path = dump_dir_ + "/klite_" + name + ".bin";
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    LOG(WARNING) << "dump 失败: " << path;
    return;
  }
  f.write(reinterpret_cast<const char*>(host), sizeof(float) * count);
}

// =============================================================================
//  3D-MRoPE
//
//  inv_freq[j] = theta^(-j/64)，j = 0..63
//  频率轴划分：j < 16 -> t，16 <= j < 40 -> h，j >= 40 -> w
//  旋转为半分割配对 (j, j + 64)
// =============================================================================
void PaddleOCRVLModel::_build_mrope_cos_sin(int32_t pos_t, int32_t pos_h, int32_t pos_w,
                                float* cos_out, float* sin_out) const {
  const int32_t half = config_->head_size_ / 2;  // 64
  const int32_t sec_t = vl_config_->mrope_section_t_;
  const int32_t sec_h = vl_config_->mrope_section_h_;
  for (int32_t j = 0; j < half; ++j) {
    const float inv_freq = std::pow(vl_config_->text_rope_theta_,
                                    -static_cast<float>(j) / static_cast<float>(half));
    int32_t pos = pos_w;
    if (j < sec_t) {
      pos = pos_t;
    } else if (j < sec_t + sec_h) {
      pos = pos_h;
    }
    const float angle = static_cast<float>(pos) * inv_freq;
    cos_out[j] = std::cos(angle);
    sin_out[j] = std::sin(angle);
  }
}

MRoPEPositions PaddleOCRVLModel::compute_mrope_positions(
    const std::vector<int>& tokens, const std::vector<ProcessedImage>& images) const {
  const int32_t merge = vl_config_->vision.spatial_merge_size_;
  const int32_t image_token = vl_config_->image_token_id_;

  std::vector<int32_t> pt, ph, pw;
  pt.reserve(tokens.size());
  ph.reserve(tokens.size());
  pw.reserve(tokens.size());

  int32_t next_pos = 0;
  size_t i = 0;
  size_t img_idx = 0;
  while (i < tokens.size()) {
    if (tokens[i] == image_token && img_idx < images.size()) {
      const auto& g = images[img_idx].grid_thw;
      const int32_t hb = g.h / merge;
      const int32_t wb = g.w / merge;
      const int32_t count = g.t * hb * wb;
      const int32_t base = next_pos;
      for (int32_t t = 0; t < g.t; ++t) {
        for (int32_t a = 0; a < hb; ++a) {
          for (int32_t b = 0; b < wb; ++b) {
            // 图片的 second_per_grid_t = 0，故 t轴恒为 base
            pt.push_back(base);
            ph.push_back(base + a);
            pw.push_back(base + b);
          }
        }
      }
      i += static_cast<size_t>(count);
      // 下一段文本从上一段最大值 + 1 续接
      next_pos = base + std::max(hb, wb);
      ++img_idx;
    } else {
      pt.push_back(next_pos);
      ph.push_back(next_pos);
      pw.push_back(next_pos);
      ++next_pos;
      ++i;
    }
  }

  const int32_t len = static_cast<int32_t>(pt.size());
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor positions(base::DataType::kDataTypeInt32, 3, len, true, alloc);
  int32_t* p = positions.ptr<int32_t>();
  std::memcpy(p, pt.data(), sizeof(int32_t) * len);
  std::memcpy(p + len, ph.data(), sizeof(int32_t) * len);
  std::memcpy(p + 2 * len, pw.data(), sizeof(int32_t) * len);

  MRoPEPositions out;
  out.positions = positions;
  // 参考实现的 rope_delta：位置数远少于 token 数
  out.mrope_position_delta = next_pos - static_cast<int32_t>(tokens.size());
  return out;
}

// =============================================================================
//  文本 decoder：逐 token 前向（GQA + MRoPE + KV-Cache）
// =============================================================================
void PaddleOCRVLModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  tensor::Tensor rms_out = get_buffer(ModelBufferType::kOutputRMSNorm);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  STATUS_CHECK(norm->forward(input, rms_out));
}

void PaddleOCRVLModel::attention_qkv(int32_t layer_idx, int32_t token_pos,
                                     const tensor::Tensor& mrope_cos,
                                     const tensor::Tensor& mrope_sin) const {
  tensor::Tensor rms_out = get_buffer(ModelBufferType::kOutputRMSNorm);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  auto [key, val] = slice_kv_cache(layer_idx, token_pos);

  STATUS_CHECK(qwen_layers_->wq_layers_.at(layer_idx)->forward(rms_out, query));
  STATUS_CHECK(qwen_layers_->wk_layers_.at(layer_idx)->forward(rms_out, key));
  STATUS_CHECK(qwen_layers_->wv_layers_.at(layer_idx)->forward(rms_out, val));

  // MRoPE：逐 head 半分割配对 (j, j + 64)
  const int32_t hs = config_->head_size_;
  auto rope = kernel::get_rope_half_split_kernel(device_type_);
  void* stream = cuda_config_ ? cuda_config_->stream : nullptr;
  rope(query.ptr<float>(), config_->head_num_, hs, mrope_cos.ptr<float>(), mrope_sin.ptr<float>(),
       stream);
  rope(key.ptr<float>(), config_->kv_head_num_, hs, mrope_cos.ptr<float>(),
       mrope_sin.ptr<float>(), stream);
}

void PaddleOCRVLModel::attention_mha(int32_t layer_idx, int32_t token_pos) const {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_out = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  const auto& mha = qwen_layers_->mha_layer_;
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha)->set_pos(token_pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha->forward(query, score, key_cache, val_cache, mha_out));

  tensor::Tensor attn_out = get_buffer(ModelBufferType::kAttnOutput);
  STATUS_CHECK(qwen_layers_->wo_layers_.at(layer_idx)->forward(mha_out, attn_out));
}

void PaddleOCRVLModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  STATUS_CHECK(qwen_layers_->add_layer_->forward(
      input, get_buffer(ModelBufferType::kAttnOutput), input));

  tensor::Tensor ffn_norm_out = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  STATUS_CHECK(ffn_norm->forward(input, ffn_norm_out));

  tensor::Tensor w1_out = get_buffer(ModelBufferType::kW1Output);
  tensor::Tensor w3_out = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(qwen_layers_->w1_layers_.at(layer_idx)->forward(ffn_norm_out, w1_out));
  STATUS_CHECK(qwen_layers_->w3_layers_.at(layer_idx)->forward(ffn_norm_out, w3_out));
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

  tensor::Tensor w2_out = get_buffer(ModelBufferType::kW2Output);
  STATUS_CHECK(qwen_layers_->w2_layers_.at(layer_idx)->forward(w1_out, w2_out));
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_out, input));
}

void PaddleOCRVLModel::cls_logits(const tensor::Tensor& input) const {
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  STATUS_CHECK(norm->forward(input, input));
  tensor::Tensor logits = get_buffer(ModelBufferType::kForwardOutput);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, logits));
}

base::Status PaddleOCRVLModel::_llm_forward_token(int32_t token_pos,
                                                 const tensor::Tensor& mrope_cos,
                                                 const tensor::Tensor& mrope_sin,
                                                 const tensor::Tensor& input) const {
  if (token_pos >= config_->seq_len_) {
    return base::error::InternalError("PaddleOCR-VL: token position exceeds seq_len.");
  }
  for (int32_t l = 0; l < config_->layer_num_; ++l) {
    attention_rms(l, input);
    attention_qkv(l, token_pos, mrope_cos, mrope_sin);
    attention_mha(l, token_pos);
    feed_forward(l, input);
  }
  cls_logits(input);
  return base::error::Success();
}

// =============================================================================
//  embedding：文本查表 + 视觉特征按image_token 占位符注入
// =============================================================================
op::EmbeddingOutput PaddleOCRVLModel::embedding(const std::vector<int>& tokens) const {
  return embedding_multimodal(tokens, {});
}

op::EmbeddingOutput PaddleOCRVLModel::embedding_multimodal(
    const std::vector<int>& tokens, const std::vector<ProcessedImage>& images) const {
  const int32_t dim = config_->dim_;
  const int32_t len = static_cast<int32_t>(tokens.size());
  const bool on_cuda = device_type_ == base::DeviceType::kDeviceCUDA;
  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc =
      on_cuda ? std::static_pointer_cast<base::DeviceAllocator>(
                    base::CUDADeviceAllocatorFactory::get_instance())
              : std::static_pointer_cast<base::DeviceAllocator>(cpu_alloc);
  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, len, true, cpu_alloc);
  tensor::Tensor embeddings(base::DataType::kDataTypeFp32, len, dim, true, alloc);

  auto emb_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->embedding_layer_);
  CHECK(emb_param != nullptr);
  const float* table = emb_param->get_weight(0).ptr<float>();
  float* out = embeddings.ptr<float>();

  // 按 token 从 embedding 表取行；CUDA 下 table 与 out 同在显存，走 D2D 拷贝
  const size_t row_bytes = sizeof(float) * dim;
  auto copy_row = [&](float* dst, const float* src) {
    if (on_cuda) {
      cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, stream);
    } else {
      std::memcpy(dst, src, row_bytes);
    }
  };

  for (int32_t i = 0; i < len; ++i) {
    const int32_t tok = tokens[i];
    CHECK_GE(tok, 0);
    CHECK_LT(tok, config_->vocab_size_);
    input_tokens.index<int32_t>(i) = tok;
    copy_row(out + static_cast<size_t>(i) * dim, table + static_cast<size_t>(tok) * dim);
  }

  // 视觉特征注入：按出现顺序填入 image_token 占位行
  if (!images.empty()) {
    const int32_t image_token = vl_config_->image_token_id_;
    int32_t filled = 0;
    for (const auto& img : images) {
      tensor::Tensor feat = encode_image(img.pixel_values, img.grid_thw);
      const int32_t n_tok = feat.get_dim(0);
      const float* src = feat.ptr<float>();
      int32_t used = 0;
      for (int32_t i = filled; i < len && used < n_tok; ++i) {
        if (tokens[i] != image_token) {
          continue;
        }
        copy_row(out + static_cast<size_t>(i) * dim, src + static_cast<size_t>(used) * dim);
        ++used;
        filled = i + 1;
      }
      CHECK_EQ(used, n_tok) << "PaddleOCR-VL: image token 占位符数量(" << used
                            << ") 与视觉特征数(" << n_tok << ") 不一致。";
      if (on_cuda) {
        // feat 在本轮结束后释放，拷贝必须先落地
        cudaStreamSynchronize(stream);
      }
    }
  }
  if (on_cuda) {
    cudaStreamSynchronize(stream);
  }

  tensor::Tensor token_num(base::DataType::kDataTypeInt32, len);
  _dump("inputs_embeds", out, static_cast<size_t>(len) * dim);
  return op::EmbeddingOutput(input_tokens, embeddings, token_num);
}

// =============================================================================
//  多模态推理入口
// =============================================================================
base::Status PaddleOCRVLModel::predict_multimodal(const std::vector<int>& tokens,
                                                  const std::vector<ProcessedImage>& images,
                                                  bool is_prompt, int& next_token) const {
  if (tokens.empty()) {
    return base::error::InvalidArgument("PaddleOCR-VL: tokens is empty.");
  }
  CHECK(config_ != nullptr && vl_config_ != nullptr);

  const int32_t dim = config_->dim_;
  const int32_t half = config_->head_size_ / 2;
  const bool on_cuda = device_type_ == base::DeviceType::kDeviceCUDA;
  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (on_cuda) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }
  tensor::Tensor mrope_cos(base::DataType::kDataTypeFp32, half, true, alloc);
  tensor::Tensor mrope_sin(base::DataType::kDataTypeFp32, half, true, alloc);
  // MRoPE 的频率按 (t,h,w) 分段取值，只能在主机侧逐 token 算好再送上去
  std::vector<float> cos_host(half), sin_host(half);

  auto emb = embedding_multimodal(tokens, images);
  float* emb_ptr = emb.input_embeddings.ptr<float>();
  const int32_t len = static_cast<int32_t>(tokens.size());

  if (is_prompt) {
    mm_decode_step_ = 0;
    mm_rope_pos_ = 0;
  }

  MRoPEPositions mrope;
  const int32_t* pos_data = nullptr;
  if (is_prompt) {
    mrope = compute_mrope_positions(tokens, images);
    pos_data = mrope.positions.ptr<int32_t>();
  }

  for (int32_t i = 0; i < len; ++i) {
    int32_t pt = 0, ph = 0, pw = 0;
    if (is_prompt) {
      pt = pos_data[i];
      ph = pos_data[len + i];
      pw = pos_data[2 * len + i];
    } else {
      // decode 阶段三轴同值，从prompt 结束处续算
      pt = ph = pw = mm_rope_pos_;
    }
    _build_mrope_cos_sin(pt, ph, pw, cos_host.data(), sin_host.data());
    const size_t half_bytes = sizeof(float) * half;
    if (on_cuda) {
      cudaMemcpyAsync(mrope_cos.ptr<float>(), cos_host.data(), half_bytes, cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(mrope_sin.ptr<float>(), sin_host.data(), half_bytes, cudaMemcpyHostToDevice,
                      stream);
    } else {
      std::memcpy(mrope_cos.ptr<float>(), cos_host.data(), half_bytes);
      std::memcpy(mrope_sin.ptr<float>(), sin_host.data(), half_bytes);
    }

    // 该token 的 hidden 行视图，残差在其上原地累加
    tensor::Tensor input(base::DataType::kDataTypeFp32, dim, false, nullptr,
                         emb_ptr + static_cast<size_t>(i) * dim);
    input.set_device_type(device_type_);

    auto status = _llm_forward_token(mm_decode_step_, mrope_cos, mrope_sin, input);
    if (!status) {
      return status;
    }
    ++mm_decode_step_;
    if (is_prompt) {
      // prompt 阶段位置由 mrope 给出，结束后从最大值 + 1 续接
      mm_rope_pos_ = std::max(mm_rope_pos_, std::max(pt, std::max(ph, pw)) + 1);
    } else {
      ++mm_rope_pos_;
    }
  }

  next_token = post_processing(get_buffer(ModelBufferType::kInputPos), false);
  return base::error::Success();
}

base::Status PaddleOCRVLModel::predict(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor, bool is_prompt,
                                       int& next) const {
  UNUSED(input);
  UNUSED(pos_tensor);
  UNUSED(is_prompt);
  UNUSED(next);
  return base::error::FunctionNotImplement(
      "PaddleOCR-VL 请使用 predict_multimodal 作为推理入口。");
}

base::Status PaddleOCRVLModel::forward(const tensor::Tensor& input,
                                       const tensor::Tensor& pos_tensor, int& next) const {
  UNUSED(input);
  UNUSED(pos_tensor);
  UNUSED(next);
  return base::error::FunctionNotImplement(
      "PaddleOCR-VL 请使用 predict_multimodal 作为推理入口。");
}

int32_t PaddleOCRVLModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  UNUSED(pos);
  if (is_prompt) {
    return -1;
  }
  tensor::Tensor logits = get_buffer(ModelBufferType::kForwardOutput);
  return static_cast<int32_t>(sampler_->sample(logits.ptr<float>(), logits.size(),
                                               cuda_config_ ? cuda_config_->stream : nullptr));
}

}  // namespace model
