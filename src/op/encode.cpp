#include "op/encode.h"
#include <glog/logging.h>
#include <fstream>
#include "base/unicode.h"
#include "nlohmann/json.hpp"
namespace op {
namespace {
/// 取路径所在目录，用于在 tokenizer.model 旁边找配套的 json
std::string dir_of(const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}
}  // namespace

std::string SpeEncodeLayer::decode(int32_t token_id) const {
  return decode(std::vector<int32_t>{token_id});
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(spe != nullptr);
  if (added_tokens_.empty() && skip_tokens_.empty()) {
    return this->spe->DecodeIds(token_ids);
  }
  // 扩展 token 的 id 超出 sentencepiece 词表，必须挑出来单独还原，
  // 其余连续片段仍交给 sentencepiece（否则会丢失 subword 的拼接规则）
  std::string out;
  std::vector<int32_t> chunk;
  auto flush = [&]() {
    if (!chunk.empty()) {
      out += this->spe->DecodeIds(chunk);
      chunk.clear();
    }
  };
  for (const int32_t id : token_ids) {
    if (skip_tokens_.count(id) != 0) {
      flush();
      continue;
    }
    const auto it = added_tokens_.find(id);
    if (it != added_tokens_.end()) {
      flush();
      out += it->second;
    } else {
      chunk.push_back(id);
    }
  }
  flush();
  return out;
}

void SpeEncodeLayer::load_added_tokens() {
  const std::string cfg_path = dir_of(token_model_path_) + "/tokenizer_config.json";
  std::ifstream f(cfg_path);
  if (!f.is_open()) {
    return;  // llama2.c 系只有 tokenizer.model，保持原行为
  }
  nlohmann::json cfg;
  try {
    cfg = nlohmann::json::parse(f);
  } catch (const nlohmann::json::parse_error&) {
    LOG(WARNING) << "无法解析 " << cfg_path << "，扩展 token 将无法还原。";
    return;
  }
  const auto it = cfg.find("added_tokens_decoder");
  if (it == cfg.end() || !it->is_object()) {
    return;
  }

  const int32_t spe_size = spe->GetPieceSize();
  for (const auto& [key, value] : it->items()) {
    const int32_t id = std::stoi(key);
    if (id < spe_size) {
      continue;  // sentencepiece 自己能解码
    }
    const bool special = value.value("special", false);
    if (special) {
      skip_tokens_.insert(id);
    } else {
      added_tokens_.emplace(id, value.value("content", std::string()));
    }
  }
  if (!added_tokens_.empty() || !skip_tokens_.empty()) {
    // 有明确 eos 的现代 tokenizer，BOS 不再兼作结束符
    bos_as_ending_ = false;
    LOG(INFO) << "扩展词表: " << added_tokens_.size() << " 个可见token, "
              << skip_tokens_.size() << " 个特殊 token（解码时跳过）";
  }
}

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  using namespace sentencepiece::util;
  spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto rc = spe->Load(token_model_path_);
  if (!rc.ok()) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }
  load_added_tokens();
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(spe != nullptr);
  // sentencepiece
  std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
  if (has_bos_) {
    input_ids.insert(input_ids.begin(), spe->bos_id());
  }
  if (has_eos_) {
    input_ids.push_back(spe->eos_id());
  }
  return input_ids;
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  CHECK(this->spe != nullptr);
  if (token_id == this->spe->eos_id()) {
    return true;
  }
  // llama2.c / stories 系用 BOS 标记一段故事的结束；有扩展词表的模型有明确 eos，
  // 若也把 BOS 当结束会提前截断输出
  return bos_as_ending_ && token_id == this->spe->bos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
  CHECK(spe != nullptr);
  return spe->GetPieceSize();
}

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);
  CHECK(f.is_open())
      << "The token model path is not valid, please check the path and type of token model.";
  json data;
  try {
    data = json::parse(f);
  } catch (json::parse_error&) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }

  const auto& datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }
  bos_id_ = special_tokens["<|begin_of_text|>"];
  eos_id_ = special_tokens["<|end_of_text|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|eot_id|>"];

  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  std::map<std::string, std::string> replacements;
  replacements[" "] = "Ġ";
  std::string s = absl::StrReplaceAll(sentence, replacements);
  auto input_ids = this->tiktoken_->encode(s);

  if (has_bos_) {
    input_ids.insert(input_ids.begin(), bos_id_);
  }
  if (has_eos_) {
    input_ids.push_back(eos_id_);
  }
  return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const { return ""; }

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  auto s = tiktoken_->decode(token_ids);
  std::map<std::string, std::string> reverse_replacements;
  reverse_replacements["Ġ"] = " ";
  const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
  return sentence;
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  if (token_id == stop_token1_ || token_id == stop_token2_) {
    return true;
  } else {
    return false;
  }
}

int32_t BpeEncodeLayer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);

  json data = json::parse(f);
  const auto& datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }
  bos_id_ = special_tokens["<|im_start|>"];
  eos_id_ = special_tokens["<|im_end|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|endoftext|>"];

  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

#endif
}  // namespace op