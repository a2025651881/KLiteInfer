// SpeEncodeLayer 的扩展词表回归测试。
//
// 起因是一个真实 bug：表格图上模型生成了 70 个 token，但只解码出 9 个字符。
// PaddleOCR-VL 有 1019 个 added token（OTSL 表格标记 <lcel>/<fcel>/<nl>、
// <|LOC_n|> 定位符），id 超出 tokenizer.model 的词表范围，
// sentencepiece 遇到就静默丢弃，一个字都不报错。
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "config/config.h"
#include "op/encode.h"

namespace {
bool exists(const std::string& p) { return std::ifstream(p).good(); }
}  // namespace

// 扩展词表里的可见 token（special=false）必须原样还原
TEST(test_encode, spe_restores_added_tokens) {
  const std::string tok = config::paddleocr_tokenizer_path;
  if (!exists(tok)) {
    GTEST_SKIP() << "缺少 " << tok << "，跳过（需要 PaddleOCR-VL 分词器）";
  }
  op::SpeEncodeLayer enc(tok, true, false);
  const int32_t spe_vocab = enc.vocab_size();

  // OTSL 表格标记，id 均超出 sentencepiece 词表
  const int32_t kFcel = 101309, kLcel = 101311, kNl = 101313;
  ASSERT_GT(kFcel, spe_vocab - 1) << "前提变了：该 token 已在 sentencepiece 词表内";

  EXPECT_EQ(enc.decode(std::vector<int32_t>{kLcel}), "<lcel>");
  EXPECT_EQ(enc.decode(std::vector<int32_t>{kFcel}), "<fcel>");
  EXPECT_EQ(enc.decode(std::vector<int32_t>{kNl}), "<nl>");
}

// 普通 token 与扩展 token 混排时，普通片段仍须整段交给 sentencepiece，
// 否则会丢掉 subword 的拼接规则
TEST(test_encode, spe_mixes_normal_and_added_tokens) {
  const std::string tok = config::paddleocr_tokenizer_path;
  if (!exists(tok)) {
    GTEST_SKIP() << "缺少 " << tok;
  }
  op::SpeEncodeLayer enc(tok, true, false);

  const std::string plain = "CRuncover";
  std::vector<int32_t> ids = enc.encode(plain);
  ASSERT_FALSE(ids.empty());
  // encode 会在开头插入 BOS，解码时应被跳过
  const std::string decoded_plain = enc.decode(ids);

  ids.push_back(101311);  // <lcel>
  ids.push_back(101313);  // <nl>
  const std::string decoded = enc.decode(ids);

  EXPECT_EQ(decoded, decoded_plain + "<lcel><nl>");
  EXPECT_NE(decoded.find("<lcel>"), std::string::npos);
}

// special=true 的 token（如 <|IMAGE_PLACEHOLDER|>）解码时应被跳过，
// 与 HF 的 skip_special_tokens=True 语义保持一致
TEST(test_encode, spe_skips_special_tokens) {
  const std::string tok = config::paddleocr_tokenizer_path;
  if (!exists(tok)) {
    GTEST_SKIP() << "缺少 " << tok;
  }
  op::SpeEncodeLayer enc(tok, true, false);

  const int32_t kImagePlaceholder = 100295;
  EXPECT_EQ(enc.decode(std::vector<int32_t>{kImagePlaceholder}), "");
  // 夹在可见 token 中间也只跳过它自己
  EXPECT_EQ(enc.decode(std::vector<int32_t>{101311, kImagePlaceholder, 101313}), "<lcel><nl>");
}

// 有明确 eos 的现代 tokenizer 不应把 BOS 当结束符，否则输出会被提前截断
TEST(test_encode, spe_bos_not_treated_as_ending_when_extended) {
  const std::string tok = config::paddleocr_tokenizer_path;
  if (!exists(tok)) {
    GTEST_SKIP() << "缺少 " << tok;
  }
  op::SpeEncodeLayer enc(tok, true, false);
  EXPECT_TRUE(enc.is_sentence_ending(2));    // </s>
  EXPECT_FALSE(enc.is_sentence_ending(1));   // <s>，不应终止
  EXPECT_FALSE(enc.is_sentence_ending(101311));
}

// llama2.c / stories 系只有 tokenizer.model，靠 BOS 标记故事结束，
// 这条旧约定必须保留
TEST(test_encode, spe_bos_is_ending_for_llama2c) {
  const std::string tok = config::llama_tokenizer_path;
  if (!exists(tok)) {
    GTEST_SKIP() << "缺少 " << tok;
  }
  op::SpeEncodeLayer enc(tok, true, false);
  EXPECT_TRUE(enc.is_sentence_ending(1));  // BOS 也算结束
  EXPECT_TRUE(enc.is_sentence_ending(2));

  // 无扩展词表时解码路径不变
  const auto ids = enc.encode("Once upon a time");
  EXPECT_FALSE(enc.decode(ids).empty());
}
