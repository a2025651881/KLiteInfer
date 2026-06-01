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

// PaddleOCR-VL
const std::string paddleocr_model_path     = "./paddleocr-vl/model.safetensors";
const std::string paddleocr_tokenizer_path = "./paddleocr-vl/tokenizer.json";
const std::string paddleocr_image_path     = "./paddleocr-vl/sample_image.bin";
const int paddleocr_image_t = 1;
const int paddleocr_image_h = 16;  // 16 * 14 = 224 像素
const int paddleocr_image_w = 16;

// 推理参数
const int max_generate_steps = 2048;
const bool use_cuda = true;

}