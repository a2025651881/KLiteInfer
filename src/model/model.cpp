#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace model {

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : is_quant_model_(is_quant_model),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      model_type_(model_type),
      tokenizer_type_(tokenizer_type) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

int32_t Model::seq_len() const { return config_ ? config_->seq_len_ : 0; }

void Model::set_sampler(std::unique_ptr<sampler::Sampler> sampler) {
  CHECK(sampler != nullptr);
  sampler_ = std::move(sampler);
}

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << "The buffer is not exist: " << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0) << "The buffer is not exist: " << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  size_t header_size = 0;
  {
    // 先读第一个 int32 判断是 KLite 扩展头还是 llama2.c 原始头
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError(
          "Failed to retrieve the configuration information from the model file.");
    }
    if (magic == kKliteModelMagic) {
      int32_t buf[9] = {0};  // version + 8 个字段
      if (fread(buf, sizeof(int32_t), 9, file) != 9) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Failed to read the KLite extended model header.");
      }
      if (buf[0] != kKliteModelVersion) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Unsupported KLite model header version: " +
                                      std::to_string(buf[0]));
      }
      config.dim = buf[1];
      config.hidden_dim = buf[2];
      config.layer_num = buf[3];
      config.head_num = buf[4];
      config.kv_head_num = buf[5];
      config.head_size = buf[6];
      config.vocab_size = buf[7];
      config.seq_len = buf[8];
      header_size = sizeof(int32_t) * 10;
    } else {
      int32_t buf[6] = {0};
      if (fread(buf, sizeof(int32_t), 6, file) != 6) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Failed to read the llama2.c model header.");
      }
      config.dim = magic;  // 原始格式第一个字段就是 dim
      config.hidden_dim = buf[0];
      config.layer_num = buf[1];
      config.head_num = buf[2];
      config.kv_head_num = buf[3];
      config.vocab_size = buf[4];
      config.seq_len = buf[5];
      config.head_size = 0;  // 由 dim / head_num 推导
      header_size = sizeof(int32_t) * 7;
    }
  }

  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError("Failed to retrieve the group size from the model file.");
    }
    header_size += sizeof(int32_t);
  }

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    fclose(file);
    close(fd);
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = static_cast<size_t>(ftell(file));
  fclose(file);

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }

  raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + header_size;
  raw_model_data_->header_size = header_size;
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.dim;
  config_->immediate_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  if (config.head_num <= 0 || config.kv_head_num <= 0) {
    return base::error::ModelParseError("The head num in the model file is invalid.");
  }
  // llama2.c 的文件头不含 head_size（此时为 0），按 dim / head_num 推导；
  // KLite 扩展头会显式给出（Qwen3 的 head_size != dim / head_num）
  config_->head_size_ =
      config.head_size > 0 ? config.head_size : config.dim / config.head_num;
  config_->q_dim_ = config_->head_num_ * config_->head_size_;
  config_->kv_dim_ = config_->kv_head_num_ * config_->head_size_;
  config_->kv_mul_ = config.head_num / config.kv_head_num;

  // vocab_size 为负数时表示 lm_head 与 embedding 不共享权重
  config_->is_shared_weight_ = config.vocab_size > 0;
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#else
    return error::FunctionNotImplement(
        "The BPE tokenizer requires the QWEN3_SUPPORT macro to be enabled at compile time.");
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed! " << create_encode_status.get_err_msg();
    return create_encode_status;
  }

  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Read model file " << model_path_ << " failed! " << mmap_status.get_err_msg();
    return mmap_status;
  }

  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed! "
               << layer_create_status.get_err_msg();
    return layer_create_status;
  }
  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  const int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  const int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  auto key_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                key_cache_ptr, true);
  auto val_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                                  val_cache_ptr, true);
  key_cache->set_device_type(device_type_);
  val_cache->set_device_type(device_type_);

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_);
  key.assign(key_cache);
  val.assign(val_cache);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
  const int32_t dim = config_->dim_;
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      dim * sizeof(float), nullptr, input_embeddings.ptr<float>(index * dim), true);

  tensor::Tensor input(base::DataType::kDataTypeFp32, dim);
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model
