// apps/http_util.h 的单元测试：用 socketpair 模拟真实的 socket 收发，
// 覆盖请求解析、Content-Length、chunked 编码等服务端依赖的逻辑。
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <thread>

#include "http_util.h"

namespace {

/// 建一对已连接的 socket，返回 {读端, 写端}
struct SocketPair {
  int a = -1, b = -1;
  SocketPair() { EXPECT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, &a), 0); }
  ~SocketPair() {
    if (a >= 0) ::close(a);
    if (b >= 0) ::close(b);
  }
};

/// 把 fd 里能读到的内容全部收上来（对端关闭后返回）
std::string drain(int fd) {
  std::string out;
  char buf[1024];
  while (true) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  return out;
}

}  // namespace

TEST(test_http, content_length_case_insensitive) {
  EXPECT_EQ(http::content_length("POST / HTTP/1.1\r\nContent-Length: 42"), 42u);
  EXPECT_EQ(http::content_length("POST / HTTP/1.1\r\ncontent-length: 7"), 7u);
  EXPECT_EQ(http::content_length("POST / HTTP/1.1\r\nCONTENT-LENGTH:123"), 123u);
  // 缺失该头时按 0 处理（GET 请求没有 body）
  EXPECT_EQ(http::content_length("GET /health HTTP/1.1\r\nHost: x"), 0u);
}

TEST(test_http, parse_post_request) {
  SocketPair sp;
  const std::string body = R"({"prompt":"hi","max_tokens":8})";
  const std::string req = "POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  ASSERT_TRUE(http::send_all(sp.b, req));

  const auto parsed = http::read_request(sp.a);
  ASSERT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.method, "POST");
  EXPECT_EQ(parsed.path, "/generate");
  EXPECT_EQ(parsed.body, body);
}

TEST(test_http, parse_get_request_without_body) {
  SocketPair sp;
  ASSERT_TRUE(http::send_all(sp.b, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"));

  const auto parsed = http::read_request(sp.a);
  ASSERT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.method, "GET");
  EXPECT_EQ(parsed.path, "/health");
  EXPECT_TRUE(parsed.body.empty());
}

// body 分多个 TCP 段到达时也要能拼完整（真实网络里很常见）
TEST(test_http, body_arrives_in_multiple_packets) {
  SocketPair sp;
  const std::string body = R"({"prompt":"分段到达的中文 body"})";
  const std::string header =
      "POST /generate HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";

  std::thread writer([&] {
    http::send_all(sp.b, header);
    // 故意切成两半，中间留间隔
    const size_t half = body.size() / 2;
    http::send_all(sp.b, body.substr(0, half));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    http::send_all(sp.b, body.substr(half));
  });

  const auto parsed = http::read_request(sp.a);
  writer.join();
  ASSERT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.body, body);
}

TEST(test_http, malformed_request_line_rejected) {
  SocketPair sp;
  // 没有空格分隔的请求行
  ASSERT_TRUE(http::send_all(sp.b, "GARBAGE\r\n\r\n"));
  const auto parsed = http::read_request(sp.a);
  EXPECT_FALSE(parsed.ok);
}

TEST(test_http, peer_closed_returns_not_ok) {
  SocketPair sp;
  ::close(sp.b);
  sp.b = -1;
  const auto parsed = http::read_request(sp.a);
  EXPECT_FALSE(parsed.ok);
}

TEST(test_http, send_json_has_correct_length_and_status) {
  SocketPair sp;
  const std::string payload = R"({"status":"ok"})";
  http::send_json(sp.a, payload);
  ::close(sp.a);
  sp.a = -1;

  const std::string resp = drain(sp.b);
  EXPECT_NE(resp.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(resp.find("Content-Length: " + std::to_string(payload.size())), std::string::npos);
  EXPECT_NE(resp.find("application/json"), std::string::npos);
  // body 在空行之后，且与原文一致
  const size_t sep = resp.find("\r\n\r\n");
  ASSERT_NE(sep, std::string::npos);
  EXPECT_EQ(resp.substr(sep + 4), payload);
}

TEST(test_http, send_json_error_code) {
  SocketPair sp;
  http::send_json(sp.a, R"({"error":"x"})", 400);
  ::close(sp.a);
  sp.a = -1;
  EXPECT_NE(drain(sp.b).find("HTTP/1.1 400 Bad Request"), std::string::npos);
}

// chunked 编码：每块是「十六进制长度 CRLF 数据 CRLF」，以0 块结尾
TEST(test_http, chunked_stream_encoding) {
  SocketPair sp;
  http::send_stream_header(sp.a);
  ASSERT_TRUE(http::send_chunk(sp.a, "hello"));   // 5 -> "5"
  ASSERT_TRUE(http::send_chunk(sp.a, "中文"));     // 6 字节 -> "6"
  ASSERT_TRUE(http::send_chunk(sp.a, ""));        // 空块应被忽略，不能提前终止流
  http::send_last_chunk(sp.a);
  ::close(sp.a);
  sp.a = -1;

  const std::string resp = drain(sp.b);
  EXPECT_NE(resp.find("Transfer-Encoding: chunked"), std::string::npos);
  const size_t sep = resp.find("\r\n\r\n");
  ASSERT_NE(sep, std::string::npos);
  EXPECT_EQ(resp.substr(sep + 4), "5\r\nhello\r\n6\r\n中文\r\n0\r\n\r\n");
}

TEST(test_http, chunk_size_is_hex) {
  SocketPair sp;
  // 20 字节应编码成十六进制 "14"，写成十进制就会让客户端解析错位
  http::send_chunk(sp.a, std::string(20, 'x'));
  http::send_last_chunk(sp.a);
  ::close(sp.a);
  sp.a = -1;
  EXPECT_EQ(drain(sp.b), "14\r\n" + std::string(20, 'x') + "\r\n0\r\n\r\n");
}

TEST(test_http, connect_to_unused_port_fails) {
  //挑一个几乎不可能被监听的端口，connect 应快速失败而不是挂住
  EXPECT_LT(http::connect_to("127.0.0.1", 1), 0);
}

TEST(test_http, listen_then_connect_roundtrip) {
  const int port = 18923;
  const int fd_listen = http::listen_on("127.0.0.1", port);
  ASSERT_GE(fd_listen, 0);
  // 同一端口再监听应失败（服务端据此提示"端口被占用"）
  EXPECT_LT(http::listen_on("127.0.0.1", port), 0);

  std::thread client([port] {
    const int fd = http::connect_to("127.0.0.1", port);
    ASSERT_GE(fd, 0);
    http::send_all(fd, "GET /health HTTP/1.1\r\n\r\n");
    ::close(fd);
  });

  const int fd_conn = ::accept(fd_listen, nullptr, nullptr);
  ASSERT_GE(fd_conn, 0);
  const auto parsed = http::read_request(fd_conn);
  EXPECT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.path, "/health");

  client.join();
  ::close(fd_conn);
  ::close(fd_listen);
}
