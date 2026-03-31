#ifndef KELI_INCLUDE_TENSOR_H_
#define KELI_INCLUDE_TENSOR_H_

#include <driver_types.h>
#include <vector>
#include <memory>
#include <numeric>
#include <armadillo>
#include <glog/logging.h>
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return static_cast<size_t>(init);
  }
  return std::accumulate(begin, end, init, std::multiplies<>());
}

class Tensor {
 public:
  Tensor() = default;

  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, const std::vector<int32_t>& dims, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  bool is_empty() const;

  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                   bool need_alloc, void* ptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<base::Buffer> get_buffer() const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type);

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc = false);

  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  tensor::Tensor clone() const;

 private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

// ========================== 模板实现 ==========================
template <typename T>
T* Tensor::ptr() {
  if (!buffer_ || !buffer_->ptr()) return nullptr;
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_ || !buffer_->ptr()) return nullptr;
  return reinterpret_cast<const T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK_GE(index, 0);
  CHECK_LT(index, size_);
  return ptr<T>() + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, size_);
  return ptr<T>() + index;
}

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, size_);
  return *ptr<T>(offset);
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, size_);
  return *ptr<T>(offset);
}

// ========================== 函数实现 ==========================
inline Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
                      std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  dims_.push_back(dim0);
  data_type_ = data_type;
  size_ = static_cast<size_t>(dim0);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(!need_alloc) << "The need_alloc is true when ptr parameter is not a null pointer";
    }
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

inline Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                      std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  data_type_ = data_type;
  size_ = static_cast<size_t>(dim0) * static_cast<size_t>(dim1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(!need_alloc) << "The need_alloc is true when ptr parameter is not a null pointer";
    }
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

inline Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
                      std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  data_type_ = data_type;
  size_ = static_cast<size_t>(dim0) * static_cast<size_t>(dim1) * static_cast<size_t>(dim2);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(!need_alloc) << "The need_alloc is true when ptr parameter is not a null pointer";
    }
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

inline Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool need_alloc,
                      std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  data_type_ = data_type;
  size_ = static_cast<size_t>(dim0) * static_cast<size_t>(dim1) * static_cast<size_t>(dim2) * static_cast<size_t>(dim3);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(!need_alloc) << "The need_alloc is true when ptr parameter is not a null pointer";
    }
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

inline Tensor::Tensor(base::DataType data_type, const std::vector<int32_t>& dims, bool need_alloc,
                      std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) {
  data_type_ = data_type;
  dims_ = dims;
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(!need_alloc) << "The need_alloc is true when ptr parameter is not a null pointer";
    }
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

inline void Tensor::to_cpu() {
  CHECK(buffer_);
  base::DeviceType device = device_type();
  if (device == base::DeviceType::GPU) {
    auto cpu_alloc = base::CPUAllocatorInstance();
    auto cpu_buf = std::make_shared<base::Buffer>(byte_size(), cpu_alloc);
    cpu_buf->copy_from(buffer_.get());
    buffer_ = cpu_buf;
  }
}

inline void Tensor::to_cuda(cudaStream_t stream) {
  CHECK(buffer_);
  base::DeviceType device = device_type();
  if (device == base::DeviceType::CPU) {
    auto gpu_alloc = base::GPUAllocatorInstance();
    auto gpu_buf = std::make_shared<base::Buffer>(byte_size(), gpu_alloc);
    gpu_buf->copy_from(buffer_.get());
    buffer_ = gpu_buf;
  }
}

inline bool Tensor::is_empty() const {
  return size_ == 0 || !buffer_ || !buffer_->ptr();
}

inline void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                                bool need_alloc, void* ptr) {
  if (!alloc && !need_alloc) {
    buffer_ = std::make_shared<base::Buffer>(base::data_type_size(data_type) * size_, nullptr, ptr, true);
  } else {
    allocate(alloc, true);
  }
}

inline void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t new_size = reduce_dimension(dims.begin(), dims.end(), 1);
  dims_ = dims;
  size_ = new_size;
}

inline std::shared_ptr<base::Buffer> Tensor::get_buffer() const {
  return buffer_;
}

inline size_t Tensor::size() const {
  return size_;
}

inline size_t Tensor::byte_size() const {
  return size_ * base::data_type_size(data_type_);
}

inline int32_t Tensor::dims_size() const {
  return static_cast<int32_t>(dims_.size());
}

inline base::DataType Tensor::data_type() const {
  return data_type_;
}

inline int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, dims_.size());
  return dims_[idx];
}

inline const std::vector<int32_t>& Tensor::dims() const {
  return dims_;
}

inline std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  size_t stride = 1;
  for (int i = static_cast<int>(dims_.size()) - 1; i >= 0; --i) {
    strides.push_back(stride);
    stride *= dims_[i];
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

inline bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) return false;
  if (buffer->byte_size() < byte_size()) return false;
  buffer_ = buffer;
  return true;
}

inline void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  data_type_ = data_type;
  dims_ = dims;
  size_ = reduce_dimension(dims.begin(), dims.end(), 1);
  buffer_.reset();
}

inline void Tensor::set_device_type(base::DeviceType device_type) {
  if (buffer_) buffer_->set_device_type(device_type);
}

inline base::DeviceType Tensor::device_type() const {
  return buffer_ ? buffer_->device_type() : base::DeviceType::kDeviceUnknown;
}

inline bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc) {
  if (!allocator) return false;
  size_t bytes = byte_size();
  if (bytes == 0) return false;
  buffer_ = std::make_shared<base::Buffer>(bytes, allocator);
  return buffer_ && buffer_->ptr();
}

inline Tensor Tensor::clone() const {
  Tensor out(data_type_, dims_);
  out.buffer_ = std::make_shared<base::Buffer>(byte_size(), buffer_->allocator());
  out.buffer_->copy_from(buffer_.get());
  return out;
}

}  // namespace tensor

#endif  // KELI_INCLUDE_TENSOR_H_