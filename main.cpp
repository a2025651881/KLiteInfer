#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "model/qwen3.h"
#include "model/paddleocr.h"
#include "sampler/topp_sampler.h"
#include "serve/generator.h"
#include "config/config.h"

// 转小写（用于匹配模型名）
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf(
            "KLite demo\n"
            "\n"
            "用法:\n"
            "  ./demo llama     [model.bin] [tokenizer.model] [prompt]\n"
            "  ./demo qwen3     [model.bin] [tokenizer.json]  [prompt]\n"
            "  ./demo paddleocr [model.bin] [tokenizer.model] [ref_dir] [max_new_tokens]\n"
            "\n"
            "所有路径参数均可省略，省略时取 src/config/config.cpp 中的默认值:\n"
            "  llama     : %s\n"
            "              %s\n"
            "  qwen3     : %s\n"
            "              %s\n"
            "  paddleocr : %s\n"
            "              %s\n"
            "              %s\n"
            "\n"
            "示例:\n"
            "  ./demo llama\n"
            "  ./demo qwen3 \"\" \"\" \"用一句话解释什么是张量\"\n"
            "  ./demo paddleocr\n",
            config::llama_model_path.c_str(), config::llama_tokenizer_path.c_str(),
            config::qwen3_model_path.c_str(), config::qwen3_tokenizer_path.c_str(),
            config::paddleocr_model_path.c_str(), config::paddleocr_tokenizer_path.c_str(),
            config::paddleocr_image_path.c_str());
        return -1;
    }

    std::string model_name = to_lower(argv[1]);

    // 位置参数为空串时视为「使用默认值」，方便只覆盖后面的参数
    auto arg_or = [argc, argv](int idx, const std::string& fallback) {
        if (argc > idx && argv[idx][0] != '\0') {
            return std::string(argv[idx]);
        }
        return fallback;
    };

    // ============== PaddleOCR-VL 多模态分支 ==============
    // 用法: ./demo paddleocr <model.bin> <tokenizer.model> <ref_dir> [max_new_tokens]
    //   ref_dir 由 tools/dump_paddleocr_ref.py 或 tools/ocr_eval.py 产出，
    //   含 pixel_values.bin / meta.json（meta 里的 input_ids 即 HF processor 的分词结果，
    //   这样能排除分词与图像预处理差异，专注验证模型数值）
    if (model_name == "paddleocr") {
        const std::string ckpt    = arg_or(2, config::paddleocr_model_path);
        const std::string tok     = arg_or(3, config::paddleocr_tokenizer_path);
        const std::string ref_dir = arg_or(4, config::paddleocr_image_path);
        // 与参考实现对比时需要相同的生成上限，否则长度差异会淹没真实误差
        const int cli_max_steps = (argc >= 6) ? std::atoi(argv[5]) : 0;
        LOG(INFO) << "Using model: PaddleOCR-VL";

        CHECK(!ckpt.empty()) << "PaddleOCR-VL checkpoint path is empty!";
        CHECK(!tok.empty())  << "PaddleOCR-VL tokenizer path is empty!";
        CHECK(!ref_dir.empty())
            << "需要提供参考数据目录（tools/dump_paddleocr_ref.py 的输出）作为第4 个参数。";

        // ---------- 读取参考元信息 ----------
        std::ifstream meta_f(ref_dir + "/meta.json");
        CHECK(meta_f.is_open()) << "无法打开 " << ref_dir << "/meta.json";
        nlohmann::json meta;
        meta_f >> meta;

        const int Tg = meta["grid_thw"][0][0].get<int>();
        const int Hg = meta["grid_thw"][0][1].get<int>();
        const int Wg = meta["grid_thw"][0][2].get<int>();
        const int num_patches = meta["num_patches"].get<int>();
        std::vector<int32_t> mm_tokens = meta["input_ids"].get<std::vector<int32_t>>();
        LOG(INFO) << "grid_thw = (" << Tg << ", " << Hg << ", " << Wg << "), patches = "
                  << num_patches << ", tokens = " << mm_tokens.size();

        // PaddleOCR-VL 的分词器是 SentencePiece 系（tokenizer.model），不是 tiktoken BPE
        model::PaddleOCRVLModel model(base::TokenizerType::kEncodeSpe, tok.c_str(),
                                      ckpt.c_str(), /*is_quant=*/false);
        // 视觉 encoder / projector 支持 CUDA；文本 decoder 内部仍走 CPU
        base::DeviceType ocr_device =
            config::use_cuda ? base::DeviceType::kDeviceCUDA : base::DeviceType::kDeviceCPU;
        auto status = model.init(ocr_device);
        if (!status) {
            LOG(FATAL) << "PaddleOCR-VL init failed, err: " << status.get_err_code()
                       << " msg: " << status.get_err_msg();
        }
        model.set_dump_dir(ref_dir);

        // ---------- 读取预处理好的像素 [N, 3*14*14] ----------
        const int patch = 14;
        const int channels = 3;
        const size_t pixel_count =
            static_cast<size_t>(num_patches) * channels * patch * patch;
        auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor pixel(base::DataType::kDataTypeFp32, num_patches,
                             channels * patch * patch, true, alloc_cpu);
        std::ifstream fin(ref_dir + "/pixel_values.bin", std::ios::binary);
        CHECK(fin.is_open()) << "无法打开 " << ref_dir << "/pixel_values.bin";
        fin.read(reinterpret_cast<char*>(pixel.ptr<float>()), sizeof(float) * pixel_count);
        CHECK_EQ(static_cast<size_t>(fin.gcount()), sizeof(float) * pixel_count)
            << "pixel_values.bin 大小与 grid_thw 不符";

        model::ProcessedImage img;
        img.pixel_values = pixel;
        img.grid_thw = model::ImageGridTHW{Tg, Hg, Wg};
        std::vector<model::ProcessedImage> images{img};

        // ---------- 推理 ----------
        // OCR 属于确定性任务，固定使用默认的贪心 argmax（不套用 config::use_sampling，
        // 采样会显著降低识别准确率）
        LOG(INFO) << "Sampler: argmax (OCR 使用确定性解码)";

        int total_steps = cli_max_steps > 0 ? cli_max_steps : config::max_generate_steps;
        auto start = std::chrono::steady_clock::now();
        // 流式打印，与 klite_server 复用同一套生成逻辑
        printf("\n");
        serve::GenStats st = serve::generate_ocr(model, mm_tokens, images, total_steps,
                                                 [](const std::string& delta) {
                                                     fwrite(delta.data(), 1, delta.size(), stdout);
                                                     fflush(stdout);
                                                 });
        auto end = std::chrono::steady_clock::now();
        const double total_s = std::chrono::duration<double>(end - start).count();
        const double ttft_s = st.ttft_s;
        const double decode_s = st.decode_s;
        const int n_gen = st.generated;

        printf("\n");
        printf("----------------------------------------\n");
        printf("Device        : %s\n", config::use_cuda ? "CUDA" : "CPU");
        printf("Image         : %d patches (grid %dx%dx%d)\n", num_patches, Tg, Hg, Wg);
        printf("Prompt tokens : %zu\n", meta["input_ids"].size());
        printf("Generated     : %d tokens\n", n_gen);
        printf("Vision encode : %.1f ms\n", model.last_vision_ms());
        printf("TTFT          : %.0f ms  (含视觉编码)\n", ttft_s * 1000.0);
        if (n_gen > 1) {
            printf("Decode        : %.2f s  (%.1f tok/s)\n", decode_s, (n_gen - 1) / decode_s);
        }
        printf("Total         : %.2f s  (%.2f img/s)\n", total_s, 1.0 / total_s);
        printf("----------------------------------------\n");
        return n_gen > 0 ? 0 : -1;
    }

    // ============== 文本 LLM 分支 ==============
    std::string checkpoint_path;
    std::string tokenizer_path;
    // 分词器类型与权重布局随模型而变
    base::TokenizerType tokenizer_type = base::TokenizerType::kEncodeBpe;
    model::WeightLayout weight_layout = model::WeightLayout::kQwen3;

    if (model_name == "qwen3") {
        checkpoint_path = config::qwen3_model_path;
        tokenizer_path = config::qwen3_tokenizer_path;
        LOG(INFO) << "Using model: Qwen3";
    } else if (model_name == "llama") {
        checkpoint_path = config::llama_model_path;
        tokenizer_path = config::llama_tokenizer_path;
        // llama2.c / stories系列：SentencePiece 分词器 + llama2 权重布局
        tokenizer_type = base::TokenizerType::kEncodeSpe;
        weight_layout = model::WeightLayout::kLlama2C;
        LOG(INFO) << "Using model: Llama";
    } else if (model_name == "qwen") {
        checkpoint_path = config::qwen_model_path;
        tokenizer_path = config::qwen_tokenizer_path;
        LOG(INFO) << "Using model: Qwen";
    } else {
        LOG(ERROR) << "Unsupported model: " << model_name;
        return -1;
    }

    // 命令行可覆盖 config.cpp 中的默认路径：./demo <name> [model] [tokenizer] [prompt]
    checkpoint_path = arg_or(2, checkpoint_path);
    tokenizer_path = arg_or(3, tokenizer_path);

    CHECK(!checkpoint_path.empty()) << "Checkpoint path is empty!";
    CHECK(!tokenizer_path.empty()) << "Tokenizer path is empty!";
    LOG(INFO) << "Checkpoint: " << checkpoint_path;
    LOG(INFO) << "Tokenizer:  " << tokenizer_path;

    // 初始化模型
    model::Qwen3Model model(
        tokenizer_type,
        tokenizer_path.c_str(),
        checkpoint_path.c_str(),
        false,
        weight_layout
    );

    base::DeviceType device = config::use_cuda ? base::DeviceType::kDeviceCUDA : base::DeviceType::kDeviceCPU;
    auto init_status = model.init(device);
    if (!init_status) {
        LOG(FATAL) << "Model init failed, err: " << init_status.get_err_code()
                   << " msg: " << init_status.get_err_msg();
    }

    // 替换默认的贪心 argmax：小模型贪心解码会陷入重复
    if (config::use_sampling) {
        model.set_sampler(std::make_unique<sampler::TopPSampler>(
            device, config::temperature, config::top_p, config::top_k, config::sampler_seed));
        LOG(INFO) << "Sampler: top-p, temperature=" << config::temperature
                  << " top_p=" << config::top_p << " top_k=" << config::top_k;
    }

    // 推理
    // SPE（llama2 / stories 系）没有 chat 模板，直接用裸 prompt
    const bool use_chat_template = (tokenizer_type == base::TokenizerType::kEncodeBpe);
    const std::string default_question = use_chat_template ? "What is AI?" : "Once upon a time";
    const std::string question = arg_or(4, default_question);
    std::string prompt = use_chat_template ? serve::fill_chat_template(question) : question;
    LOG(INFO) << "Prompt: " << question;

    // 生成步数不能超过模型的 seq_len，否则 KV-Cache 会写越界
    int total_steps = config::max_generate_steps;
    if (model.seq_len() > 0 && total_steps > model.seq_len()) {
        LOG(WARNING) << "max_generate_steps(" << total_steps << ") > seq_len(" << model.seq_len()
                     << "), clamp to seq_len.";
        total_steps = model.seq_len();
    }

    auto start = std::chrono::steady_clock::now();
    // 流式打印：边生成边输出，与 klite_server 走同一套生成逻辑
    serve::GenStats st = serve::generate_text(model, prompt, total_steps,
                                              [](const std::string& delta) {
                                                  fwrite(delta.data(), 1, delta.size(), stdout);
                                                  fflush(stdout);
                                              });
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    printf("\n----------------------------------------\n");
    printf("Device        : %s\n", config::use_cuda ? "CUDA" : "CPU");
    printf("Prompt tokens : %d\n", st.prompt_tokens);
    printf("Generated     : %d tokens\n", st.generated);
    printf("TTFT          : %.0f ms\n", st.ttft_s * 1000.0);
    if (st.generated > 1) {
        printf("Decode        : %.2f s  (%.1f tok/s)\n", st.decode_s, st.decode_tps());
    }
    printf("Total         : %.2f s\n", duration);
    printf("----------------------------------------\n");

    return st.generated > 0 ? 0 : -1;
}