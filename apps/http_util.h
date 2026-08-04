#ifndef KLITE_APPS_HTTP_UTIL_H_
#define KLITE_APPS_HTTP_UTIL_H_
// 极简 HTTP/1.1 工具：只覆盖本项目需要的部分（POST + JSON，可选 chunked 流式），
// 不引入额外第三方依赖。
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace http {

/// 阻塞地写完整个缓冲区（write可能只写一部分）
inline bool send_all(int fd, const char* data, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    const ssize_t n = ::send(fd, data + sent, len - sent, 0);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

inline bool send_all(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

/// 从 header 中取出 Content-Length（大小写不敏感）
inline size_t content_length(const std::string& header) {
  static const char* kKey = "content-length:";
  std::string lower;
  lower.reserve(header.size());
  for (char c : header) {
    lower.push_back(static_cast<char>(::tolower(c)));
  }
  const size_t pos = lower.find(kKey);
  if (pos == std::string::npos) {
    return 0;
  }
  return static_cast<size_t>(::strtoul(header.c_str() + pos + std::strlen(kKey), nullptr, 10));
}

/// 一个已解析的请求
struct Request {
  std::string method;
  std::string path;
  std::string body;
  bool ok = false;
};

/// 读一个完整请求：先读到空行拿到 header，再按 Content-Length 收body
inline Request read_request(int fd) {
  Request req;
  std::string buf;
  char chunk[4096];
  size_t header_end = std::string::npos;

  while (header_end == std::string::npos) {
    const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      return req;
    }
    buf.append(chunk, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
    if (buf.size() > (8u << 20)) {  // 8MB 护栏，防止畸形请求打满内存
      return req;
    }
  }

  const std::string header = buf.substr(0, header_end);
  req.body = buf.substr(header_end + 4);
  const size_t need = content_length(header);
  while (req.body.size() < need) {
    const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      return req;
    }
    req.body.append(chunk, static_cast<size_t>(n));
  }

  // 请求行： "<METHOD> <PATH> HTTP/1.1"
  const size_t sp1 = header.find(' ');
  const size_t sp2 = header.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) {
    return req;
  }
  req.method = header.substr(0, sp1);
  req.path = header.substr(sp1 + 1, sp2 - sp1 - 1);
  req.ok = true;
  return req;
}

inline void send_json(int fd, const std::string& json, int code = 200) {
  const char* reason = code == 200 ? "OK" : (code == 404 ? "Not Found" : "Bad Request");
  char head[256];
  const int n = std::snprintf(head, sizeof(head),
                              "HTTP/1.1 %d %s\r\n"
                              "Content-Type: application/json; charset=utf-8\r\n"
                              "Content-Length: %zu\r\n"
                              "Connection: close\r\n\r\n",
                              code, reason, json.size());
  send_all(fd, head, static_cast<size_t>(n));
  send_all(fd, json);
}

/// 流式响应：Transfer-Encoding: chunked，之后用 send_chunk 逐块发
inline void send_stream_header(int fd) {
  static const char* kHead =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/plain; charset=utf-8\r\n"
      "Transfer-Encoding: chunked\r\n"
      "Connection: close\r\n\r\n";
  send_all(fd, kHead, std::strlen(kHead));
}

inline bool send_chunk(int fd, const std::string& data) {
  if (data.empty()) {
    return true;
  }
  char head[32];
  const int n = std::snprintf(head, sizeof(head), "%zx\r\n", data.size());
  return send_all(fd, head, static_cast<size_t>(n)) && send_all(fd, data) &&
         send_all(fd, "\r\n", 2);
}

inline void send_last_chunk(int fd) { send_all(fd, "0\r\n\r\n", 5); }

/// 监听端口，返回 listen fd（失败返回 -1）
inline int listen_on(const char* host, int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  int on = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = ::htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = ::inet_addr(host);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(fd, 16) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

/// 连接服务端，返回 fd（失败返回 -1）
inline int connect_to(const char* host, int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  int on = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = ::htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = ::inet_addr(host);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

}  // namespace http
#endif  // KLITE_APPS_HTTP_UTIL_H_
