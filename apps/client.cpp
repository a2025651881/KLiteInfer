// KLiteInfer 命令行客户端：把请求发给常驻的 klite_server。
// 只依赖标准 socket，不需要链接 klite 库。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "http_util.h"

namespace {

void usage() {
  printf(
      "KLiteInfer 客户端\n"
      "\n"
      "用法:\n"
      "  klite_client [选项] \"你的提问\"\n"
      "\n"
      "选项:\n"
      "  --host <ip>      服务地址（默认 127.0.0.1）\n"
      "  --port <n>       服务端口（默认 8080）\n"
      "  --max-tokens <n> 生成上限（默认 256）\n"
      "  --stream         流式输出，边生成边打印\n"
      "  --ocr <ref_dir>  走 /ocr 接口，识别该目录下的预处理图像\n"
      "  --health         只查询服务状态\n");
}

/// 极简 JSON 字符串转义（prompt 里可能有引号和换行）
std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

/// 收响应并打印 body。chunked 时边收边打（流式效果）
void print_response(int fd, bool stream) {
  std::string buf;
  char chunk[4096];
  bool head_done = false;
  size_t pos = 0;  // chunked 解析游标

  while (true) {
    const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      break;
    }
    buf.append(chunk, static_cast<size_t>(n));

    if (!head_done) {
      const size_t end = buf.find("\r\n\r\n");
      if (end == std::string::npos) {
        continue;
      }
      head_done = true;
      buf = buf.substr(end + 4);
      pos = 0;
    }
    if (!stream) {
      continue;  // 非流式等收完再一次性打印
    }
    // chunked: <hex size>\r\n<data>\r\n ... 0\r\n\r\n
    while (true) {
      const size_t crlf = buf.find("\r\n", pos);
      if (crlf == std::string::npos) {
        break;
      }
      const size_t size = std::strtoul(buf.substr(pos, crlf - pos).c_str(), nullptr, 16);
      if (size == 0) {
        printf("\n");
        return;
      }
      if (buf.size() < crlf + 2 + size + 2) {
        break;  // 这一块还没收完
      }
      fwrite(buf.data() + crlf + 2, 1, size, stdout);
      fflush(stdout);
      pos = crlf + 2 + size + 2;
    }
    if (pos > 0) {
      buf.erase(0, pos);
      pos = 0;
    }
  }
  if (!stream) {
    printf("%s\n", buf.c_str());
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string host = "127.0.0.1", prompt, ref_dir;
  int port = 8080, max_tokens = 256;
  bool stream = false, health = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s 需要一个参数\n", a.c_str());
        std::exit(-1);
      }
      return argv[++i];
    };
    if (a == "--host") {
      host = next();
    } else if (a == "--port") {
      port = std::atoi(next().c_str());
    } else if (a == "--max-tokens") {
      max_tokens = std::atoi(next().c_str());
    } else if (a == "--stream") {
      stream = true;
    } else if (a == "--ocr") {
      ref_dir = next();
    } else if (a == "--health") {
      health = true;
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      prompt = a;
    }
  }
  if (!health && prompt.empty() && ref_dir.empty()) {
    usage();
    return -1;
  }

  const int fd = http::connect_to(host.c_str(), port);
  if (fd < 0) {
    fprintf(stderr, "连不上 %s:%d，服务起了吗？\n", host.c_str(), port);
    return -1;
  }

  std::string req;
  if (health) {
    req = "GET /health HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
  } else {
    const bool is_ocr = !ref_dir.empty();
    const std::string path = is_ocr ? "/ocr" : "/generate";
    const std::string body =
        is_ocr ? "{\"ref_dir\":\"" + escape(ref_dir) + "\",\"max_tokens\":" +
                std::to_string(max_tokens) + ",\"stream\":" + (stream ? "true" : "false") + "}"
               : "{\"prompt\":\"" + escape(prompt) + "\",\"max_tokens\":" +
                     std::to_string(max_tokens) + ",\"stream\":" + (stream ? "true" : "false") + "}";
    req = "POST " + path + " HTTP/1.1\r\nHost: " + host +
          "\r\nContent-Type: application/json\r\nContent-Length: " +
          std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
  }

  if (!http::send_all(fd, req)) {
    fprintf(stderr, "发送请求失败\n");
    ::close(fd);
    return -1;
  }
  print_response(fd, stream && !health);
  ::close(fd);
  return 0;
}
