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

// ===================== 推理参数配置 =====================
extern const int max_generate_steps;
extern const bool use_cuda;

} 

#endif