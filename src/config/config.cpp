#include "config/config.h"
namespace config {

// 模型统一存放目录；命令行参数可覆盖下面所有路径
static const std::string kModelsRoot = "/root/KuiperLLama/models";

// Qwen3 0.6B
// 权重需先用 tools/export_qwen3.py 从 HF safetensors 转成 KLite 扁平格式
const std::string qwen3_model_path     = kModelsRoot + "/qwen3-0.6b/klite_qwen3_0.6b.bin";
const std::string qwen3_tokenizer_path = kModelsRoot + "/qwen3-0.6b/tokenizer.json";

// Llama（llama2.c / karpathy stories 系原始格式，无需转换）
const std::string llama_model_path     = kModelsRoot + "/stories110M.bin";
const std::string llama_tokenizer_path = kModelsRoot + "/tokenizer.model";

// Qwen（暂未提供导出脚本）
const std::string qwen_model_path      = kModelsRoot + "/qwen/model.safetensors";
const std::string qwen_tokenizer_path  = kModelsRoot + "/qwen/tokenizer.json";

// PaddleOCR-VL
const std::string paddleocr_model_path     = kModelsRoot + "/paddleocr-vl/klite_paddleocr_vl.bin";
const std::string paddleocr_tokenizer_path = kModelsRoot + "/paddleocr-vl/tokenizer.model";
const std::string paddleocr_image_path     = kModelsRoot + "/paddleocr-vl/ref";
const int paddleocr_image_t = 1;
const int paddleocr_image_h = 16;  // 16 * 14 = 224 像素
const int paddleocr_image_w = 16;

// 推理参数
const int max_generate_steps = 2048;
const bool use_cuda = true;

// 采样参数
const bool use_sampling = true;
const float temperature = 0.9f;
const float top_p = 0.9f;
const int top_k = 0;
const unsigned long sampler_seed = 42;

}