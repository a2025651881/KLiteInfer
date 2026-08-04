#ifndef CONFIG_CFG_H
#define CONFIG_CFG_H

#include <string>

namespace config {

// ===================== 统一模型路径配置 =====================
// Qwen3 模型
extern const std::string qwen3_model_path;
extern const std::string qwen3_tokenizer_path;

// Llama 模型（可扩展）
extern const std::string llama_model_path;
extern const std::string llama_tokenizer_path;

// Qwen 模型（可扩展）
extern const std::string qwen_model_path;
extern const std::string qwen_tokenizer_path;

// PaddleOCR-VL 多模态模型
extern const std::string paddleocr_model_path;
extern const std::string paddleocr_tokenizer_path;
// 参考数据目录（tools/dump_paddleocr_ref.py 的输出），含 pixel_values.bin 与 meta.json
extern const std::string paddleocr_image_path;
extern const int paddleocr_image_t;
extern const int paddleocr_image_h;  // patch 单位
extern const int paddleocr_image_w;  // patch 单位

// ===================== 推理参数配置 =====================
extern const int max_generate_steps;
extern const bool use_cuda;

// ===================== 采样参数配置 =====================
// false 时使用贪心 argmax；小模型贪心容易陷入重复，建议开启
extern const bool use_sampling;
extern const float temperature;  // <=0 退化为argmax
extern const float top_p;        // 取 1 表示不做 nucleus 截断
extern const int top_k;          // <=0 表示不限制候选数
extern const unsigned long sampler_seed;

} 

#endif