namespace base{
    Status::Status(int code =StatusCode::kSuccess, std::string err_message = ""):code_(code),mesage_(err_message);
        
    Status::Status(const Status& other) =default;

    Status& Status::operator=(const Status& other) = default;

    Status& Status::operator=(int code);

    bool Status::operator==(int code){
        if(code == code_) return true;
        else return false;
    }

    bool Status::operator!=(int code){
        if(code != code_) return true;
        else return false;
    }

    operator Status::int() const{
        return code_;
    }

    operator Status::bool() const{
        return code_ == kSuccess;
    }

    int32_t Status::get_err_code() const{
        return code_;
    }

    const std::string& Status::get_err_msg() const{
        return message_;
    }


    void Status::set_err_msg(const std::string& err_msg){
        message_=err_msg;
    }
}
namespace error{
Status Success(const std::string& err_msg = ""){
    return Status{kSuccess,err_msg};
}

Status FunctionNotImplement(const std::string& err_msg = ""){
    return Status{kFunctionUnImplement,err_msg};
}

Status PathNotValid(const std::string& err_msg=""){
    return Status{kPatheNotValid,err_msg};
}

Status ModelParseError(const std::string& err_msg=""){
    return Status{kModelParseError,err_msg}
}

Status kInternalError(const std::string& err_msg=""){
    return Status{kInternalError,err_msg};
}

Status kKeyValueHasExist(const std::string& err_msg=""){
    return Status{kKeyValueHasExist,err_msg};
}

Status kInvalidArgument(const std::string& err_msg=""){
    return Status{kInternalError,err_msg};
}
}