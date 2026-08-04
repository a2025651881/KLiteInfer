#ifndef KLITE_INCLUDE_SERVE_GENERATOR_H_
#define KLITE_INCLUDE_SERVE_GENERATOR_H_
#include <functional>
#include <string>
#include "model/paddleocr.h"
#include "model/qwen3.h"

namespace serve {

/// 一次生成的分段耗时统计（prefill 与 decode 分开，便于对比不同后端）
struct GenStats {
  int32_t prompt_tokens = 0;
  int32_t generated = 0;
  double ttft_s = 0.0;    // time-to-first-token：吃完 prompt 并产出首个 token
  double decode_s = 0.0;  // 其余 token 的累计耗时
  std::string text;       // 完整输出

  double total_s() const { return ttft_s + decode_s; }
  /// decode 阶段吞吐（不含 prefill，反映稳态生成速度）
  double decode_tps() const {
    return (generated > 1 && decode_s > 0) ? (generated - 1) / decode_s : 0.0;
  }
};

/**
 * @brief 流式回调，参数是本步新产出的文本增量
 *
 * 注意增量是「重新解码整段后取差值」得到的，不能逐 token 单独 decode：
 * 一个汉字往往跨多个 token，单独解码会得到乱码。
 */
using TokenCallback = std::function<void(const std::string&)>;

/**
 * @brief 文本模型（Llama2 / Qwen3）自回归生成
 *
 * @param max_steps  位置上限，含 prompt；受 KV-Cache 容量 seq_len 约束
 * @param on_token   非空时每产出一段文本就回调一次，用于流式输出
 */
GenStats generate_text(const model::Qwen3Model& model, const std::string& prompt, int max_steps,
                       const TokenCallback& on_token = nullptr);

/**
 * @brief PaddleOCR-VL 多模态生成
 *
 * @param tokens     由 image processor 产出的 input_ids（含 image_token 占位）
 * @param max_tokens 生成上限，会按 KV-Cache 余量再夹一次
 */
GenStats generate_ocr(model::PaddleOCRVLModel& model, const std::vector<int>& tokens,
                      const std::vector<model::ProcessedImage>& images, int max_tokens,
                      const TokenCallback& on_token = nullptr);

/// 套上 Qwen3 的 chat 模板
std::string fill_chat_template(const std::string& content);

}  // namespace serve
#endif  // KLITE_INCLUDE_SERVE_GENERATOR_H_
