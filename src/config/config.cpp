#include "config/config.h"
namespace config {
// Qwen3 0.6B
const std::string qwen3_model_path    = "./qwen3-0.6b/model.safetensors";
const std::string qwen3_tokenizer_path = "./qwen3-0.6b/tokenizer.json";

// Llama
const std::string llama_model_path     = "./llama/model.bin";
const std::string llama_tokenizer_path = "./llama/tokenizer.model";

// Qwen
const std::string qwen_model_path      = "./qwen/model.safetensors";
const std::string qwen_tokenizer_path = "./qwen/tokenizer.json";

// 推理参数
const int max_generate_steps = 2048;
const bool use_cuda = true;

}