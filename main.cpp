#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <string>
#include <algorithm>
#include "model/qwen3.h"
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
    if (argc != 2) {
        LOG(INFO) << "Usage: ./demo [model_name]";
        LOG(INFO) << "Example: ./demo qwen3";
        LOG(INFO) << "Example: ./demo llama";
        LOG(INFO) << "Example: ./demo qwen";
        return -1;
    }

    std::string model_name = to_lower(argv[1]);
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