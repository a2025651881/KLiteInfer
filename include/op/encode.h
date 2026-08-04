#ifndef KUIPER_INCLUDE_OP_ENCODE_H_
#define KUIPER_INCLUDE_OP_ENCODE_H_
#include <sentencepiece_processor.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "layer.h"
#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#include "base/base.h"
#include <cstdint>
#endif
namespace op {

class EncodeLayerBase : public Layer {
 public:
  explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
      : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
        has_bos_(has_bos),
        has_eos_(has_eos),
        token_model_path_(std::move(token_model_path)) {}

  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  virtual std::string decode(int32_t token_id) const = 0;

  virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

  virtual bool is_sentence_ending(int32_t token_id) const = 0;

  virtual int32_t vocab_size() const = 0;

 protected:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::string token_model_path_;
};

class SpeEncodeLayer : public EncodeLayerBase {
 public:
  explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::string decode(int32_t token_id) const override;

  std::string decode(const std::vector<int32_t>& token_ids) const override;

  bool is_sentence_ending(int32_t token_id) const override;

  int32_t vocab_size() const override;

 private:
  /**
   * @brief 加载 sentencepiece 词表之外的扩展 token
   *
   * HF 导出的模型常把新增 token 放在 tokenizer_config.json 的 added_tokens_decoder 里，
   * 这些 id 超出 tokenizer.model 的词表范围，直接交给 sentencepiece 解码会被静默丢弃。
   * PaddleOCR-VL 的 OTSL 表格标记（<lcel> / <fcel> / <nl> …）就属于这一类。
   */
  void load_added_tokens();

  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
  /// id -> 字面量，仅含 special=false 的 token（需原样输出）
  std::unordered_map<int32_t, std::string> added_tokens_;
  /// special=true 的 token，解码时跳过（与 HF skip_special_tokens=True 语义一致）
  std::unordered_set<int32_t> skip_tokens_;
  /**
   * 是否把 BOS 也当作句子结束。
   *
   * llama2.c / stories 系用 BOS 标记一段故事的结束，必须开启；而 HF 导出的模型有
   * 明确的 eos，把 BOS 当结束会导致提前截断，故检测到扩展词表时自动关闭。
   */
  bool bos_as_ending_ = true;
};

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
class BpeEncodeLayer : public EncodeLayerBase {
public:
  explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::string decode(int32_t token_id) const override;

  std::string decode(const std::vector<int32_t>& token_ids) const override;

  bool is_sentence_ending(int32_t token_id) const override;

  int32_t vocab_size() const override;

 protected:
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t stop_token1_ = -1;
  int32_t stop_token2_ = -1;
  int32_t num_token_ = 0;
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenEncodeLayer : public BpeEncodeLayer {
public:
  explicit QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
};
#endif

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ENCODE_H_