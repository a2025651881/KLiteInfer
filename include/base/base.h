#ifndef KELI_INCLUDE_BASE_BASE_H_
#define KELI_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

namespace base{
enum class DeviceType : uint8_t{
    UNKNOWN =0,
    CPU = 1,
    GPU = 2
};

enum class DataType:uint8_t{
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3
};

enum class ModelType:uint8_t{
    kModelTypeUnkonwn = 0,
    kModelTypeLLama2 = 1
};

inline size_t DataTypeSize(DataType data_type){
    switch(data_type){
        case kDataTypeFp32: return sizeof(float); break;
        case kDataTypeInt32: return sizeof(int32_t); break;
        case kDataTypeInt8: return sizeof(int8_t); break;
        default: return 0;
    }
}

enum StatusCode :uint8_t{
    kSuccess =0,
    kFunctionUnImplement = 1,
    kPatheNotValid =2,
    kModelParseError =3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

enum class TokenizerType{
    kEncodeUnknown = 0,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
}

class Status{
    public:
        Status(int code =StatusCode::kSuccess, std::string err_message = "");
        
        Status(const Status& other) =default;

        Status& operator=(const Status& other) = default;

        Status& operator=(int code);
        
        bool operator==(int code) const;

        bool operator!=(int) const;

        operator int() const;

        operator bool() const;

        int32_t get_err_code() const;

        const std::string& get_err_msg() const;

        void set_err_msg(const std::string& err_msg);
    private:
        int code_ = StatusCode::kSuccess;
        std::string message_;
}
};
namespace error{

#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg="");

Status ModelParseError(const std::string& err_msg="");

Status kInternalError(const std::string& err_msg="");

Status kKeyValueHasExist(const std::string& err_msg="");

Status kInvalidArgument(const std::string& err_msg="");
};