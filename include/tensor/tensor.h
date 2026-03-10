#ifndef KELI_INCLUDE_TENSOR_H_
#define KELI_INCLUDE_TENSOR_H_
#include <driver_types.h>
#include <vector>
#include <memory>
#include <armadillo>
#include <glog/logging.h>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor{
class Tensor{
    public:
        explicit Tensor() =default;
        explicit Tensor(base::DataType data_type,int32_t dim0,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr);
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr);
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,int32_t dim2,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr);
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,,int32_t dim2,int32_t dim3,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr);
        explicit Tensor(base::DataTyope data_type,vector<int32_t> dims,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr);
        
        void to_cpu();

        void to_cuda(cudaStream_t stream = nullptr);

        bool is_empty() const;

        void init_buffer(std:::shared_ptr<base::DeviceAllocator> alloc,base::DataType data_type,
                        bool need_alloc,void* ptr);
        
        template<typename T>
        T* ptr();

        template<typename T>
        const T* ptr() const;

        void reshape(const std::vector<int32_t>& dims);

        std::shared_ptr<base::Buffer> get_buffer() const;

        size_t size() cons;

        size_t byte_size() const;

        int32_t dims_size() const;

        base::DataType data_type() const;

        int32_t get_dim(int32_t idx) const;

        const std::vector<int32_t>& dims() const;

        std::vector<size_t> strides() const;

        bool assign(std::shared_ptr<base::Buffer> buffer);

        void reset(base::DataType data_type,const std::vector<int32_t>& dims);

        void set_device_type();

        base::DeviceType device_type() const;

        bool allocate(std::shared_ptr<base::DeviceAllocator> allocater, bool need_alloc =false);

        template<typename T>
        T* ptr(int64_t index);

        template<typename T>
        const T* ptr(int64_t index) const;

        template<typename T>
        T& index(int64_t offset);

        template<typename T>
        const T& index(int64_t offset) const;

        tensor::Tensor clone() const;
    private:    
        size_t size_ =0;
        std::vector<int32_t> dims_;
        std::shared_ptr<base::Buffer> buffer_;
        base::DataTyope data_type_ = base::DataType::kDataTypeUnknown;
}
}