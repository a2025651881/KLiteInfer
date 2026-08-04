#include "serve/generator.h"
#include <glog/logging.h>
#include <chrono>

namespace serve {
namespace {
// Qwen3 chat 模板的控制 token，不应出现在输出文本里
constexpr int32_t kImStart = 151644;
constexpr int32_t kImEnd = 151645;

/// 重新解码整段并返回相对 prev 的增量（逐 token 解码会切坏多字节字符）
std::string emit_delta(const std::string& full, std::string& prev, const TokenCallback& cb) {
  if (full.size() <= prev.size()) {
    return {};
  }
  std::string delta = full.substr(prev.size());
  prev = full;
  if (cb) {
    cb(delta);
  }
  return delta;
}
}  // namespace

std::string fill_chat_template(const std::string& content) {
  return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
}

GenStats generate_text(const model::Qwen3Model& model, const std::string& prompt, int max_steps,
                       const TokenCallback& on_token) {
  GenStats st;
  auto tokens = model.encode(prompt);
  if (tokens.empty()) {
    LOG(ERROR) << "prompt 编码结果为空。";
    return st;
  }
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  st.prompt_tokens = prompt_len;
  // prompt 本身就占满可用步数时，循环会全程停在 prefill，一个 token 也产不出来
  if (prompt_len >= max_steps) {
    LOG(ERROR) << "Prompt 长度 (" << prompt_len << ") 已达到可用步数上限 (" << max_steps
               << ")，无法生成任何内容。";
    return st;
  }

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  const auto t_start = std::chrono::steady_clock::now();
  auto t_first = t_start;
  bool got_first = false;
  std::vector<int32_t> words;
  std::string emitted;

  while (pos < max_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (!got_first) {
        t_first = std::chrono::steady_clock::now();
        got_first = true;
      }
      if (next != kImEnd && next != kImStart) {
        words.push_back(next);
        if (on_token) {
          emit_delta(model.decode(words), emitted, on_token);
        }
      }
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }

  const auto t_end = std::chrono::steady_clock::now();
  st.text = model.decode(words);
  st.generated = static_cast<int32_t>(words.size());
  st.ttft_s = std::chrono::duration<double>(t_first - t_start).count();
  st.decode_s = std::chrono::duration<double>(t_end - t_first).count();
  return st;
}

GenStats generate_ocr(model::PaddleOCRVLModel& model, const std::vector<int>& tokens,
                      const std::vector<model::ProcessedImage>& images, int max_tokens,
                      const TokenCallback& on_token) {
  GenStats st;
  st.prompt_tokens = static_cast<int32_t>(tokens.size());

  // 生成步数上限 = KV-Cache 剩余容量（seq_len 已被 prompt 占去一部分）
  const int budget = model.seq_len() - static_cast<int>(tokens.size());
  if (budget <= 0) {
    LOG(ERROR) << "prompt 长度 (" << tokens.size() << ") 已占满 seq_len (" << model.seq_len()
               << ")，无法生成。";
    return st;
  }
  const int total_steps = std::min(max_tokens, budget);

  std::vector<int32_t> generated;
  std::vector<int> step_tokens = tokens;
  std::vector<model::ProcessedImage> step_images = images;
  int32_t next = 0;
  bool is_prompt = true;
  std::string emitted;

  const auto t_start = std::chrono::steady_clock::now();
  auto t_first = t_start;
  for (int step = 0; step < total_steps; ++step) {
    auto status = model.predict_multimodal(step_tokens, step_images, is_prompt, next);
    if (!status) {
      LOG(ERROR) << "predict_multimodal 失败: " << status.get_err_msg();
      break;
    }
    if (is_prompt) {
      // prefill 含视觉编码 + 全部 prompt token，到这里产出第一个字
      t_first = std::chrono::steady_clock::now();
    }
    is_prompt = false;
    step_images.clear();  // 后续 decode 不再传图
    if (model.is_sentence_ending(next)) {
      break;
    }
    generated.push_back(next);
    if (on_token) {
      emit_delta(model.decode(generated), emitted, on_token);
    }
    step_tokens = {next};
  }

  const auto t_end = std::chrono::steady_clock::now();
  st.text = model.decode(generated);
  st.generated = static_cast<int32_t>(generated.size());
  st.ttft_s = std::chrono::duration<double>(t_first - t_start).count();
  st.decode_s = std::chrono::duration<double>(t_end - t_first).count();
  return st;
}

}  // namespace serve
