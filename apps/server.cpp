// KLiteInfer 推理服务：进程启动时把模型加载进内存/显存并常驻，
// 之后每个请求只跑推理，省掉反复加载权重的开销。
//
// 单线程串行处理请求：KV-Cache 是单请求独占的，并发进来会互相踩踏。
// 要提高吞吐需要 continuous batching，见 README 的 Roadmap。
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "config/config.h"
#include "http_util.h"
#include "serve/generator.h"

using json = nlohmann::json;

namespace {

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

std::string to_lower(std::string s) {
  for (auto& c : s) {
    c = static_cast<char>(::tolower(c));
  }
  return s;
}

void usage() {
  printf(
      "KLiteInfer 推理服务（模型常驻）\n"
      "\n"
      "用法:\n"
      "  klite_server [选项]\n"
      "\n"
      "选项:\n"
      "  --model<name>      llama | qwen3 | paddleocr（默认 qwen3）\n"
      "  --ckpt <path>       权重路径，默认取 src/config/config.cpp\n"
      "  --tokenizer <path>  分词器路径，同上\n"
      "  --host <ip>         监听地址（默认 127.0.0.1）\n"
      "  --port <n>          监听端口（默认 8080）\n"
      "  --device <cpu|cuda> 推理设备（默认按 config::use_cuda）\n"
      "\n"
      "接口:\n"
      "  GET  /health    服务与模型状态\n"
      "  POST /generate  {\"prompt\":\"...\",\"max_tokens\":256,\"stream\":false}\n"
      "  POST /ocr       {\"ref_dir\":\"...\",\"max_tokens\":128,\"stream\":false}\n");
}

/// 读取 tools/ocr_eval.py 产出的预处理结果（pixel_values.bin + meta.json）
bool load_ocr_input(const std::string& ref_dir, std::vector<int>* tokens,
                    std::vector<model::ProcessedImage>* images, std::string* err) {
  std::ifstream mf(ref_dir + "/meta.json");
  if (!mf.is_open()) {
    *err = "打不开 " + ref_dir + "/meta.json";
    return false;
  }
  json meta;
  try {
    meta = json::parse(mf);
  } catch (const json::parse_error& e) {
    *err = std::string("meta.json 解析失败: ") + e.what();
    return false;
  }

  const auto grid = meta["grid_thw"][0];
  const int32_t tg = grid[0], hg = grid[1], wg = grid[2];
  const int num_patches = meta["num_patches"];
  *tokens = meta["input_ids"].get<std::vector<int>>();

  const std::string pv_path = ref_dir + "/pixel_values.bin";
  std::ifstream pf(pv_path, std::ios::binary | std::ios::ate);
  if (!pf.is_open()) {
    *err = "打不开 " + pv_path;
    return false;
  }
  const size_t bytes = static_cast<size_t>(pf.tellg());
  pf.seekg(0);
  const int32_t patch_dim = static_cast<int32_t>(bytes / sizeof(float) / num_patches);

  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor pixel(base::DataType::kDataTypeFp32, num_patches, patch_dim, true, alloc);
  pf.read(reinterpret_cast<char*>(pixel.ptr<float>()), static_cast<std::streamsize>(bytes));

  model::ProcessedImage img;
  img.pixel_values = pixel;
  img.grid_thw = model::ImageGridTHW{tg, hg, wg};
  images->clear();
  images->push_back(img);
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string model_name = "qwen3";
  std::string ckpt, tokenizer, host = "127.0.0.1";
  int port = 8080;
  bool use_cuda = config::use_cuda;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        LOG(FATAL) << a << " 需要一个参数（" << what << "）";
      }
      return argv[++i];
    };
    if (a == "--model") {
      model_name = to_lower(next("模型名"));
    } else if (a == "--ckpt") {
      ckpt = next("权重路径");
    } else if (a == "--tokenizer") {
      tokenizer = next("分词器路径");
    } else if (a == "--host") {
      host = next("监听地址");
    } else if (a == "--port") {
      port = std::atoi(next("端口").c_str());
    } else if (a == "--device") {
      use_cuda = to_lower(next("设备")) == "cuda";
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      printf("未知参数: %s\n\n", a.c_str());
      usage();
      return -1;
    }
  }

  const base::DeviceType device =
      use_cuda ? base::DeviceType::kDeviceCUDA : base::DeviceType::kDeviceCPU;

  // ---------------- 加载模型（一次，之后常驻）----------------
  std::unique_ptr<model::Qwen3Model> text_model;
  std::unique_ptr<model::PaddleOCRVLModel> ocr_model;
  const auto t_load = std::chrono::steady_clock::now();

  if (model_name == "paddleocr") {
    if (ckpt.empty()) ckpt = config::paddleocr_model_path;
    if (tokenizer.empty()) tokenizer = config::paddleocr_tokenizer_path;
    ocr_model = std::make_unique<model::PaddleOCRVLModel>(base::TokenizerType::kEncodeSpe,
                                                         tokenizer, ckpt, /*is_quant=*/false);
    auto st = ocr_model->init(device);
    if (!st) {
      LOG(ERROR) << "模型加载失败: " << st.get_err_msg();
      return -1;
    }
  } else if (model_name == "llama" || model_name == "qwen3") {
    const bool is_llama = model_name == "llama";
    if (ckpt.empty()) ckpt = is_llama ? config::llama_model_path : config::qwen3_model_path;
    if (tokenizer.empty()) {
      tokenizer = is_llama ? config::llama_tokenizer_path : config::qwen3_tokenizer_path;
    }
    const auto tok_type =
        is_llama ? base::TokenizerType::kEncodeSpe : base::TokenizerType::kEncodeBpe;
    text_model = std::make_unique<model::Qwen3Model>(tok_type, tokenizer, ckpt, false);
    auto st = text_model->init(device);
    if (!st) {
      LOG(ERROR) << "模型加载失败: " << st.get_err_msg();
      return -1;
    }
  } else {
    LOG(ERROR) << "不支持的模型: " << model_name << "（可选 llama / qwen3 / paddleocr）";
    return -1;
  }

  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
  // llama2.c / stories 系没有 chat 模板，直接喂裸 prompt
  const bool use_chat_template = model_name == "qwen3";

  const int fd_listen = http::listen_on(host.c_str(), port);
  if (fd_listen < 0) {
    // 用 return 而不是 LOG(FATAL)：后者会 abort 并产生 core dump，
    // 端口被占用属于正常的启动失败，给个退出码就够了
    LOG(ERROR) << "监听 " << host << ":" << port << " 失败（端口被占用？）";
    return -1;
  }
  ::signal(SIGPIPE, SIG_IGN);  // 客户端提前断开时不要杀进程
  // 用 sigaction 且不设 SA_RESTART：否则 accept 会被自动重启，
  // 收到 Ctrl-C / SIGTERM 后仍卡在阻塞里退不出来
  struct sigaction sa{};
  sa.sa_handler = on_signal;
  ::sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);

  printf("\n  KLiteInfer server·  model=%s  device=%s  加载耗时 %.2f s\n", model_name.c_str(),
         use_cuda ? "cuda" : "cpu", load_s);
  printf("  监听 http://%s:%d   （Ctrl-C 退出）\n\n", host.c_str(), port);
  fflush(stdout);

  while (g_stop == 0) {
    const int fd = ::accept(fd_listen, nullptr, nullptr);
    if (fd < 0) {
      if (g_stop) break;
      continue;
    }
    const auto req = http::read_request(fd);
    if (!req.ok) {
      ::close(fd);
      continue;
    }

    if (req.method == "GET" && req.path == "/health") {
      json resp = {{"status", "ok"},
                   {"model", model_name},
                   {"device", use_cuda ? "cuda" : "cpu"},
                   {"load_seconds", load_s}};
      http::send_json(fd, resp.dump());
      ::close(fd);
      continue;
    }

    if (req.method != "POST" || (req.path != "/generate" && req.path != "/ocr")) {
      http::send_json(fd, R"({"error":"未知接口，见 --help"})", 404);
      ::close(fd);
      continue;
    }

    json body;
    try {
      body = json::parse(req.body);
    } catch (const json::parse_error& e) {
      http::send_json(fd, json{{"error", std::string("JSON 解析失败: ") + e.what()}}.dump(), 400);
      ::close(fd);
      continue;
    }
    const bool stream = body.value("stream", false);
    if (stream) {
      http::send_stream_header(fd);
    }
    // 流式下客户端可能提前断开，此时 send 失败就停止生成
    bool alive = true;
    serve::TokenCallback cb = nullptr;
    if (stream) {
      cb = [&](const std::string& delta) {
        if (alive) {
          alive = http::send_chunk(fd, delta);
        }
      };
    }

    serve::GenStats st;
    std::string err;
    if (req.path == "/generate") {
      if (text_model == nullptr) {
        err = "当前服务加载的是 paddleocr，请用 /ocr";
      } else {
        const std::string q = body.value("prompt", std::string("Once upon a time"));
        const int max_tokens = body.value("max_tokens", config::max_generate_steps);
        const std::string prompt = use_chat_template ? serve::fill_chat_template(q) : q;
        st = serve::generate_text(*text_model, prompt,
                                  std::min(max_tokens, text_model->seq_len()), cb);
      }
    } else {
      if (ocr_model == nullptr) {
        err = "当前服务加载的是文本模型，请用 /generate";
      } else {
        const std::string ref_dir = body.value("ref_dir", config::paddleocr_image_path);
        std::vector<int> tokens;
        std::vector<model::ProcessedImage> images;
        if (!load_ocr_input(ref_dir, &tokens, &images, &err)) {
          // err 已填
        } else {
          const int max_tokens = body.value("max_tokens", 128);
          st = serve::generate_ocr(*ocr_model, tokens, images, max_tokens, cb);
        }
      }
    }

    if (stream) {
      if (!err.empty() && alive) {
        http::send_chunk(fd, "[error] " + err);
      }
      http::send_last_chunk(fd);
    } else if (!err.empty()) {
      http::send_json(fd, json{{"error", err}}.dump(), 400);
    } else {
      json resp = {{"text", st.text},
                   {"prompt_tokens", st.prompt_tokens},
                   {"generated", st.generated},
                   {"ttft_ms", st.ttft_s * 1000.0},
                   {"decode_tps", st.decode_tps()},
                   {"total_seconds", st.total_s()}};
      http::send_json(fd, resp.dump(2));
    }
    ::close(fd);

    if (err.empty()) {
      LOG(INFO) << req.path << "  prompt=" << st.prompt_tokens << " gen=" << st.generated
                << " ttft=" << static_cast<int>(st.ttft_s * 1000) << "ms"
                << " " << st.decode_tps() << " tok/s";
    } else {
      LOG(WARNING) << req.path << " 失败: " << err;
    }
  }

  ::close(fd_listen);
  printf("\n服务已停止。\n");
  return 0;
}
