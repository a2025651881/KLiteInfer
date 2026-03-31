#ifndef KELI_INCLUDE_BASE_BASE_H_
#define KELI_INCLUDE_BASE_BASE_H_

#include <glog/logging.h>
#include <cstdint>
#include <string>
#include <cstdio>

namespace base {

enum class DeviceType : uint8_t {
  UNKNOWN = 0,
  CPU = 1,
  GPU = 2
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1
};

inline size_t data_type_size(DataType data_type) {
  switch (data_type) {
    case DataType::kDataTypeFp32:
      return sizeof(float);
    case DataType::kDataTypeInt32:
      return sizeof(int32_t);
    case DataType::kDataTypeInt8:
      return sizeof(int8_t);
    default:
      return 0;
  }
}

enum class StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionNotImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

enum class TokenizerType {
  kEncodeUnknown = 0,
  kEncodeSpe = 0,
  kEncodeBpe = 1,
};

class Status {
 public:
  Status(int code = static_cast<int>(StatusCode::kSuccess), std::string err_message = "");

  Status(const Status& other) = default;
  Status& operator=(const Status& other) = default;
  Status& operator=(int code);

  bool operator==(int code) const;
  bool operator!=(int code) const;

  operator int() const;
  operator bool() const;

  int32_t get_err_code() const;
  const std::string& get_err_msg() const;
  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = static_cast<int>(StatusCode::kSuccess);
  std::string message_;
};

}  // namespace base

namespace error {

#define STATUS_CHECK(call)                                                              \
  do {                                                                                  \
    const base::Status& status = call;                                                  \
    if (!status) {                                                                      \
      const size_t buf_size = 512;                                                      \
      char buf[buf_size];                                                               \
      snprintf(buf, buf_size - 1,                                                       \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n",        \
               __FILE__, __LINE__, static_cast<int>(status),                            \
               status.get_err_msg().c_str());                                           \
      LOG(FATAL) << buf;                                                                \
    }                                                                                   \
  } while (0)

base::Status Success(const std::string& err_msg = "");
base::Status FunctionNotImplement(const std::string& err_msg = "");
base::Status PathNotValid(const std::string& err_msg = "");
base::Status ModelParseError(const std::string& err_msg = "");
base::Status InternalError(const std::string& err_msg = "");
base::Status KeyValueHasExist(const std::string& err_msg = "");
base::Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

namespace model {

enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kForwardOutput = 15,
  kForwardOutputCPU = 16,
  kSinCache = 17,
  kCosCache = 18,
};

}  // namespace model

#endif  // KELI_INCLUDE_BASE_BASE_H_