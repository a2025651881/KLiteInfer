#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <fstream>
#include <string>
#include <algorithm>
#include "model/qwen3.h"
#include "model/paddleocr.h"
#include "config/cfg.h"

int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    while (pos < total_steps) {
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
            if (next != 151645 && next != 151644) {
                words.push_back(next);
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
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    return std::min(pos, total_steps);
}

std::string fill_template(const std::string& content) {
    const std::string format =
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    std::string result = format;
    size_t pos = result.find("%s");
    if (pos != std::string::npos) {
        result.replace(pos, 2, content);
    }
    return result;
}

// 转小写（用于匹配模型名）
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        LOG(INFO) << "Usage: ./demo [model_name]";
        LOG(INFO) << "Example: ./demo qwen3";
        LOG(INFO) << "Example: ./demo llama";
        LOG(INFO) << "Example: ./demo qwen";
        LOG(INFO) << "Example: ./demo paddleocr [image_path]";
        return -1;
    }

    std::string model_name = to_lower(argv[1]);

    // ============== PaddleOCR-VL 多模态分支 ==============
    if (model_name == "paddleocr") {
        const std::string ckpt = config::paddleocr_model_path;
        const std::string tok  = config::paddleocr_tokenizer_path;
        const std::string img_path =
            (argc >= 3) ? std::string(argv[2]) : config::paddleocr_image_path;
        LOG(INFO) << "Using model: PaddleOCR-VL";

        CHECK(!ckpt.empty()) << "PaddleOCR-VL checkpoint path is empty!";
        CHECK(!tok.empty())  << "PaddleOCR-VL tokenizer path is empty!";

        model::PaddleOCRVLModel model(base::TokenizerType::kEncodeBpe, tok.c_str(),
                                      ckpt.c_str(), /*is_quant=*/false);
        base::DeviceType device =
            config::use_cuda ? base::DeviceType::kDeviceCUDA : base::DeviceType::kDeviceCPU;
        auto status = model.init(device);
        if (!status) {
            LOG(FATAL) << "PaddleOCR-VL init failed, err: " << status.get_err_code();
        }

        // ---------- 构造图像 (从 .bin 读取已归一化的 fp32 像素) ----------
        const int Tg = config::paddleocr_image_t;
        const int Hg = config::paddleocr_image_h;  // patch 单位
        const int Wg = config::paddleocr_image_w;
        const int patch = 14;
        const int channels = 3;
        const size_t pixel_count =
            static_cast<size_t>(Tg) * channels * (Hg * patch) * (Wg * patch);

        auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor pixel(base::DataType::kDataTypeFp32,
                             static_cast<int32_t>(pixel_count), true, alloc_cpu);
        std::ifstream fin(img_path, std::ios::binary);
        if (!fin) {
            LOG(WARNING) << "Image file not found: " << img_path
                         << " — using zero-tensor placeholder.";
            std::memset(pixel.ptr<float>(), 0, sizeof(float) * pixel_count);
        } else {
            fin.read(reinterpret_cast<char*>(pixel.ptr<float>()),
                     sizeof(float) * pixel_count);
        }

        model::ProcessedImage img;
        img.pixel_values = pixel;
        // grid_thw 单位为 patch（与 paddleocr_image_h/w 的单位保持一致）
        img.grid_thw = model::ImageGridTHW{Tg, Hg, Wg};
        std::vector<model::ProcessedImage> images{img};

        // ---------- 构造多模态 prompt ----------
        std::string question =
            "<|im_start|>user\n<image>请识别图中的文字内容。<|im_end|>\n<|im_start|>assistant\n";
        auto tokens = model.encode(question);

        // 在 prompt 末尾插入 image_token_id × num_img_tokens 占位符；
        // 真正项目里应在 tokenizer 中显式处理 <|image_pad|> 这类 special token。
        // 这里取 PaddleOCRVLTransformerConfig 的默认 image_token_id，避免硬编码。
        constexpr int32_t kImageTokenId =
            model::PaddleOCRVLTransformerConfig{}.image_token_id_;
        constexpr int32_t kVisionStartId =
            model::PaddleOCRVLTransformerConfig{}.vision_start_token_id_;
        const int merge = model::PaddleOCRVLTransformerConfig{}.vision.spatial_merge_size_;
        const int n_img_tok = (Tg) * (Hg / merge) * (Wg / merge);

        std::vector<int32_t> mm_tokens;
        mm_tokens.reserve(tokens.size() + n_img_tok + 1);
        for (auto t : tokens) mm_tokens.push_back(t);
        mm_tokens.push_back(kVisionStartId);
        for (int i = 0; i < n_img_tok; ++i) mm_tokens.push_back(kImageTokenId);

        // ---------- 推理 ----------
        std::vector<int32_t> generated;
        int32_t next = 0;
        const int total_steps = config::max_generate_steps;
        bool is_prompt = true;
        auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < total_steps; ++step) {
            STATUS_CHECK(model.predict_multimodal(mm_tokens, images, is_prompt, next));
            is_prompt = false;
            if (model.is_sentence_ending(next)) break;
            generated.push_back(next);
            mm_tokens = {next};
            images.clear();  // 后续 decode 不再传图
        }
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();

        printf("\n%s\n", model.decode(generated).c_str());
        printf("----------------------------------------\n");
        printf("Generate steps: %d\n", static_cast<int>(generated.size()));
        printf("Duration: %.2lf s\n", duration);
        printf("----------------------------------------\n");
        return 0;
    }

    // ============== 文本 LLM 分支 ==============
    std::string checkpoint_path;
    std::string tokenizer_path;

    if (model_name == "qwen3") {
        checkpoint_path = config::qwen3_model_path;
        tokenizer_path = config::qwen3_tokenizer_path;
        LOG(INFO) << "Using model: Qwen3";
    } else if (model_name == "llama") {
        checkpoint_path = config::llama_model_path;
        tokenizer_path = config::llama_tokenizer_path;
        LOG(INFO) << "Using model: Llama";
    } else if (model_name == "qwen") {
        checkpoint_path = config::qwen_model_path;
        tokenizer_path = config::qwen_tokenizer_path;
        LOG(INFO) << "Using model: Qwen";
    } else {
        LOG(ERROR) << "Unsupported model: " << model_name;
        return -1;
    }

    CHECK(!checkpoint_path.empty()) << "Checkpoint path is empty!";
    CHECK(!tokenizer_path.empty()) << "Tokenizer path is empty!";

    // 初始化模型
    model::Qwen3Model model(
        base::TokenizerType::kEncodeBpe,
        tokenizer_path.c_str(),
        checkpoint_path.c_str(),
        false
    );

    base::DeviceType device = config::use_cuda ? base::DeviceType::kDeviceCUDA : base::DeviceType::kDeviceCPU;
    auto init_status = model.init(device);
    if (!init_status) {
        LOG(FATAL) << "Model init failed, err: " << init_status.get_err_code();
    }

    // 推理
    std::string question = "What is AI?";
    std::string prompt = fill_template(question);

    auto start = std::chrono::steady_clock::now();
    int steps = generate(model, prompt, config::max_generate_steps, true);
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    printf("\n----------------------------------------\n");
    printf("Generate steps: %d\n", steps);
    printf("Duration: %.2lf s\n", duration);
    printf("Speed: %.2lf steps/s\n", steps / duration);
    printf("----------------------------------------\n");

    return 0;
}