#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdint>
#include <numeric>
#include <utility>
#include <cstring>
#include "tensor/tensor.h"
#include "base/base.h"
#include "base/alloc.h"

namespace op {

// ========================== BaseLayer ==========================
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::LayerType BaseLayer::layer_type() const { return layer_type_; }

base::DataType BaseLayer::data_type() const { return data_type_; }

base::Status BaseLayer::init() {
    return base::error::Success();
}

base::Status BaseLayer::forward() {
    return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
    return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
    return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::layer_name() const {
    return layer_name_;
}

void BaseLayer::set_layer_name(const std::string& layer_name) {
    layer_name_ = layer_name;
}

base::DeviceType BaseLayer::device_type() const {
    return device_type_;
}

void BaseLayer::set_device_type(base::DeviceType device_type) {
    device_type_ = device_type;
}

// ========================== Layer ==========================
Layer::Layer(base::DeviceType device_type, LayerType layer_type,
             base::DataType data_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, data_type, std::move(layer_name)) {}

base::Status Layer::init() {
    return base::error::Success();
}

base::Status Layer::forward() {
    return base::error::FunctionNotImplement("Layer::forward");
}

base::Status Layer::check() const {
    return base::error::Success();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output) {
    this->set_input(0, input1);
    this->set_output(0, output);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                           const tensor::Tensor& output) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_output(0, output);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_output(0, output);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);
    this->set_output(0, output);
    return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output) {
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);
    this->set_input(4, input5);
    this->set_output(0, output);
    return this->forward();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    inputs_[idx] = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    outputs_[idx] = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

size_t Layer::input_size() const {
    return inputs_.size();
}

size_t Layer::output_size() const {
    return outputs_.size();
}

void Layer::reset_input_size(size_t size) {
    inputs_.resize(size);
}

void Layer::reset_output_size(size_t size) {
    outputs_.resize(size);
}

void Layer::to_cuda() {
    for (auto& input : inputs_) {
        if (!input.is_empty()) {
            input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    for (auto& output : outputs_) {
        if (!output.is_empty()) {
            output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
    if (!config) return;
    cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const {
    return cuda_config_;
}

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }
    return base::error::Success();
}

// ========================== LayerParam ==========================
LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type,
                       bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)),
      is_quant_layer_(is_quant_layer) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32 ||
          weight.data_type() == base::DataType::kDataTypeInt8);
    if (!weight.is_empty()) {
        CHECK(weight.device_type() == device_type());
    }
    weights_.at(idx) = weight;
    return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

void LayerParam::to_cuda() {
    Layer::to_cuda();
    for (auto& weight : weights_) {
        if (!weight.is_empty()) {
            weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    if (!scales_.is_empty()) {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK_NE(weight_ptr, nullptr);

    size_t elem_size = is_quant_layer_ ? sizeof(int8_t) : sizeof(float);
    size_t buffer_size = std::accumulate(dims.begin(), dims.end(), elem_size, std::multiplies<>());

    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
        buffer_size, nullptr, const_cast<void*>(weight_ptr), true
    );

    if (device_type != base::DeviceType::Unknown) {
        buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_) {
        tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
        weight.set_buffer(buffer);
        weights_.at(idx) = weight;
    } else {
        tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
        weight.set_buffer(buffer);
        weights_.at(idx) = weight;

        int32_t weight_size = static_cast<int32_t>(weight.size());
        CHECK_EQ(weight_size % group_size_, 0);

        int32_t scale_num = weight_size / group_size_;
        const void* scale_ptr = static_cast<const int8_t*>(weight_ptr) + weight_size;
        scales_ = tensor::Tensor(base::DataType::kDataTypeFp32, {scale_num});
        scales_.set_buffer(std::make_shared<base::Buffer>(
            scale_num * sizeof(float), nullptr, const_cast<void*>(scale_ptr), true
        ));
        scales_.set_device_type(device_type);
    }
    return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
    CHECK(!scales.is_empty());
    scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) {
    group_size_ = group_size;
}

int32_t LayerParam::get_scale_num() const {
    CHECK(!scales_.is_empty());
    return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) {
    weights_.resize(size);
}

size_t LayerParam::weight_size() const {
    return weights_.size();
}

}  // namespace op