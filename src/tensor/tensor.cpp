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
        template<typename T,typename Tp>
        static size_t reduce_dimension(T begin,T end,Tp init){
            if(begin > end){
                return 0;
            }
            return std::accumulate(begin,end,inti,std::multiplies<>());
        }

        explicit Tensor::Tensor() =default;
        explicit Tensor::Tensor(base::DataType data_type,int32_t dim0,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr){
            this->dims_.push_back(dim0);
            this->data_type_ = data_type;
            if(need_alloc && alloc){
                allocate(alloc);
            }else{
                if(ptr !=nullptr){
                    CHECK(need_alloc ==false) << "The need_alloc is true when ptr parameter is not a null pointer";
                }
                init_buffer(alloc,data_type_,need_alloc,ptr);
            }
        }
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr){
                        this->dims_.push_back(dim0);
            this->dims_.push_back(dim0);
            this->dims_.push_back(dim1);
            this->data_type_ = data_type;
            if(need_alloc && alloc){
                allocate(alloc);
            }else{
                if(ptr !=nullptr){
                    CHECK(need_alloc ==false) << "The need_alloc is true when ptr parameter is not a null pointer";
                }
                init_buffer(alloc,data_type_,need_alloc,ptr);
            }
        }
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,int32_t dim2,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr){
            this->dims_.push_back(dim0);
            this->dims_.push_back(dim1);
            this->dims_.push_back(dim2);
            this->size = dim0 * dim1 * dim2;
            this->data_type_=data_type;
            if(need_alloc && alloc){
                allocate(alloc);
            }else{
                if(ptr!=nullptr){
                    CHECK(need_alloc ==false) << "The need_alloc is true when ptr parameter is not a null pointer";
                }
                init_buffer(alloc,data_type_,need_alloc,ptr);
            }
        }
        
        explicit Tensor(base::DataType data_type,int32_t dim0,int32_t dim1,,int32_t dim2,int32_t dim3,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr){
            this->dims_.push_back(dim0);
            this->dims_.push_back(dim1);
            this->dims_.push_back(dim2);
            this->dims_.push_back(dim3);
            this->data_type_=data_type;
            this->size_ = dim0 * dim1 * dim2 * dim3;
            if(need_alloc && alloc){
                allocate(alloc);
            }else{
                if(ptr!=nullptr){
                    CHECK(need_alloc ==false) << "The need_alloc is true when ptr parameter is not a null pointer";
                }
                init_buffer(alloc,data_type_,need_alloc,ptr);
            }
        }

        explicit Tensor(base::DataTyope data_type,vector<int32_t> dims,bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,void* ptr = nullptr){
            this->data_type_ = data_type;
            this->dims_ = std::move(dims);
            this->size_ = reduce_dimension(dims_.begin(),dims_.end(),1);
            if(need_alloc && alloc){
                allocate(alloc);
            }else{
                if(ptr!=nullptr){
                    CHECK(need_alloc ==false) << "The need_alloc is true when ptr parameter is not a null pointer";
                }
                init_buffer(alloc,data_type_,need_alloc,ptr);
            }
        }
        
        void to_cpu(){
            CHECK_NE(buffer_,nullptr);
            const base::DeviceType device_type = this->device_type();
            if(device_type == base::DeviceType::UNKNOWN){
                LOG(ERROR) << "The device type of the tensor is unknow."
            }else if(device_type == base::DeviceType::GPU){
                size_t size = this->byte_size();
                auto cpu_alloc = base::CPUAllocatorInstance();
                auto cpu_buffer = std::make_shared<base::Buffer>();
                cpu_alloc->memcpy(buffer_->ptr(),cpu_buffer->ptr(),base::MemcpyKind::DeviceToHost);
                this->buffer_ = cpu_buffer;
            }else{
                LOG(INFO) << "The device type of the tensor is already cuda."
            }
        }

        void to_cuda(cudaStream_t stream = nullptr){
            CHECK_NE(buffer_,nullptr);
            const base::DeviceType device_type = this->device_type();
            if(device_type == base::DeviceType::UNKNOWN){
                LOG(INFO) << "The device type of the tensor is unkown";
            }else if(device_type == base::DeviceType::CPU){
                size_t byte_size = this->byte_size();
                auto gpu_alloc = base::GPUAllocatorInstance();
                auto gpu_buffer= std::make_shared<base::Buffer>();
                gpu_buffer->memcpy(this->buffer_->ptr(),gpu_buffer->ptr(),byte_size,base::MemcpyKind::HostToDevice);
                this->buffer_ = gpu_buffer;
            }else{
                LOG(INFO) << "The device type of the tensor is already cpu.";
            }
        }

        bool Tensor::is_empty() const{
            return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
        }

        void Tensor::init_buffer(std:::shared_ptr<base::DeviceAllocator> alloc,base::DataType data_type,
                        bool need_alloc,void* ptr){
            if(!alloc && !need_alloc){
                std::shared_ptr<base::Buffer> buffer = std::make_shared<bse:::Buffer>(data_type_size(data_type) * size_,nullptr,ptr,true);
                this->buffer_ = buffer;
            }else{
                allocate(alloc,true);
            }
        }
        
        template<typename T>
        T* ptr(){
            if(!buffer_) return nullptr;
            return reinterpret_cast<T*>(buffer_->ptr());
        }

        template<typename T>
        const T* ptr() const{
            CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr) << "The data area buffer of this tensor is empty or it points to a null pointer";
            return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr()));
        }

        void reshape(const std::vector<int32_t>& dims){
            size_t size = reduce_dimension(dims.begin(),dims.end(),1);
            if(!buffer_){
                this->dims_ = dims;
                this->size_ = size;
                return;
            }
            if(size > size_){
                auto new_buffer = std::make_shared<base Buffer>(size * base::DataTypeSIze(this->data_type_),buffer_->allocator());
                new_buffer->copy_from(buffer_.get());
                this->buffer_ = new_buffer;
            }
            dims_ = dims;
            size_ = size;
        }

        std::shared_ptr<base::Buffer> get_buffer() const{
            return this->buffer_;
        }

        size_t size() const{
            return size_;
        }

        size_t byte_size() const{
            return size_ * base::
        }

        int32_t dims_size() const{
            return static_cast<int32_t>(dims_.size());
        }

        base::DataType data_type() const{
            return this->data_type_;
        }

        int32_t get_dim(int32_t idx) const{
            CHECK_LE(idx,dims_.size()-1);
            return dims_[idx];
        }

        const std::vector<int32_t>& dims() const{
            return this->dims_;
        }

        std::vector<size_t> Tensor::strides() const{
            vector<size_t> strides;
            if(size_t i=0;i<dims_.size()-1;i++){
                size_t stride = reduce_dimension(dims_.begin()+1+i,dims_.end(),1);
                strides.push_back(stride);
            }
            strides.push_back(1);
            return strides;
        }

        bool assign(std::shared_ptr<base::Buffer> buffer){
            if(!buffer){
                LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
                return false;
            }
            if(buffer_){
                if(buffer_->device_type()!=buffer_->device_type())
                    LOG(ERROR) << "The device type of the new buffer is different from the original one.";
            }
            size_t byte_size =this->byte_size();
            if(byte_size > buffer->byte_size()){
                LOG(ERROR) << "The size of buffer is too small for the tensor!";
                return false;
            }
            buffer_ = buffer;
            return true;
        }

        void Tensor::reset(base::DataType data_type,const std::vector<int32_t>& dims){
            this->data_type_ = data_type;
            this->dims_ = dims;
            this->size_ = reduce_dimension(dims.begin(),dims.end(),1);
            this->buffer_ = nullptr;
        }

        void Tensor::set_device_type(base::DataType data_type){
            this->data_type_ = data_type;
            return;
        }

        base::DeviceType Tensor::device_type() const{
            return this->data_type_;
        }

        bool allocate(std::shared_ptr<base::DeviceAllocator> allocater, bool need_alloc =false){
            if(!allocater){
                LOG(ERROR) << "The allocator parameter in the allocate function is null";
                return false;
            }
            size_t byte_size = this->byte_size();
            if(!byte_size){
                LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
                return false;
            }
            if(buffer_ && byte_size <=buffer_->byte_size()){
                if(!need_alloc) return true;
            }
            buffer_ = std::make_shared<base::Buffer>(byte_size,alloc,nullptr);
            if(!buffer_->ptr()){
                LOG(ERROR) << "The memory allocated is a null pointer!";
                return false;
            }
            return true;
        }

        template<typename T>
        T* ptr(int64_t index){
            CHECK_GE(index,0);
            CHECK_LT(index,this->size());
            return reinterpret_cast<T*>(buffer_->ptr());
        }

        template<typename T>
        const T* ptr(int64_t index) const{
            CHECK_GE(index,0);
            CHECK_LT(index,this->size());
            return const_cast<const T*>(reinterpret_cast<const T*>(buffer_->ptr())+index);
        }

        template<typename T>
        T& index(int64_t offset){
            CHECK_GE(index,0);
            CHECK_LT(index,this->size());
            return *(reinterpret_cast<T*>(buffer_->ptr())+offset);
        }

        template<typename T>
        const T& index(int64_t offset) const{
            CHECK_GE(index,0);
            CHECK_LT(index,this->size());
            return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
        }

        tensor::Tensor Tensor::clone() const{
            Tensor new_tensor = *this;
            size_t byte_size = this->byte_size();
            auto allocate = buffer_->allocator();
            new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size,allocate);
            new_tensor.buffer_ =->copy_from(buffer_.get());
            return new_tensor;
        }
}
