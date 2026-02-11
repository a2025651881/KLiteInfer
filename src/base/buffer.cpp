#include "base/buffer.h"
#include <glob/logging.h>
#include "base/alloc.h"

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool use_external)
            : byte_size_(byte_size), allocator_(allocator), use_external_(use_external), ptr_(ptr) {
        if(!ptr && allocator){
            device_type_ = allocator->device_type();
            use_external_ = false;
            byte_size_ = byte_size;
            ptr_ = allocator->allocate(byte_size);
        }
    }
Buffer::~Buffer() {
        if (!use_external_) {
            allocator_->release(ptr_);
        }
    }

bool Buffer::allocate() {
    if (allocator_ && byte_size_ > 0) {
        us_external_ = false; // Ensure we are not using external memory
        ptr_ = allocator_->allocate(byte_size_);
        if(ptr_ == nullptr){
            LOG(ERROR) << "Buffer allocation failed for size: " << byte_size_;
            return false;
        }
        return true; // Already allocated
    }else {
        return false; // No allocator or invalid size
    }
}

void copy_from(const Buffer& buffer) const{
    if (allocator_ && buffer.allocator_) {
        DeviceType src_device = buffer.device_type();
        DeviceType dst_device = device_type_;
        switch (src_device) {
            case DeviceType::CPU:
                switch (dst_device) {
                    case DeviceType::CPU:
                        allocator_->memcpy(ptr_, buffer.ptr(), byte_size_, MemcpyKind::HostToHost);
                        break;
                    case DeviceType::GPU:
                        allocator_->memcpy(ptr_, buffer.ptr(), byte_size_, MemcpyKind::HostToDevice);
                        break;
                    default:
                        LOG(ERROR) << "Unsupported destination device type for copy_from";
                        return;
                }
                break;
            case DeviceType::GPU:
                switch (dst_device) {
                    case DeviceType::CPU:
                        allocator_->memcpy(ptr_, buffer.ptr(), byte_size_, MemcpyKind::DeviceToHost);
                        break;
                    case DeviceType::GPU:
                        allocator_->memcpy(ptr_, buffer.ptr(), byte_size_, MemcpyKind::DeviceToDevice);
                        break;
                    default:
                        LOG(ERROR) << "Unsupported destination device type for copy_from";
                        return;
                }
                break;
            default:
                LOG(ERROR) << "Unsupported source device type for copy_from";
                return;
        }
    } else {
        LOG(ERROR) << "Cannot copy from buffer: allocator is null";
    }    
}

  void Buffer::copy_from(const Buffer* buffer) const{
    if (allocator_ && buffer.allocator_) {
        DeviceType src_device = buffer->device_type();
        DeviceType dst_device = device_type_;
        switch (src_device) {
            case DeviceType::CPU:
                switch (dst_device) {
                    case DeviceType::CPU:
                        allocator_->memcpy(ptr_, buffer->ptr(), byte_size_, MemcpyKind::HostToHost);
                        break;
                    case DeviceType::GPU:
                        allocator_->memcpy(ptr_, buffer->ptr(), byte_size_, MemcpyKind::HostToDevice);
                        break;
                    default:
                        LOG(ERROR) << "Unsupported destination device type for copy_from";
                        return;
                }
                break;
            case DeviceType::GPU:
                switch (dst_device) {
                    case DeviceType::CPU:   
                        allocator_->memcpy(ptr_, buffer->ptr(), byte_size_, MemcpyKind::DeviceToHost);
                        break;
                    case DeviceType::GPU:
                        allocator_->memcpy(ptr_, buffer->ptr(), byte_size_, MemcpyKind::DeviceToDevice);
                        break;
                    default:
                        LOG(ERROR) << "Unsupported destination device type for copy_from";
                        return;
                }
                break;
            default:
                LOG(ERROR) << "Unsupported source device type for copy_from";
                return;
        }

        allocator_->memcpy(ptr_, buffer->ptr(), byte_size_, MemcpyKind::DeviceToDevice);
    } else {
        LOG(ERROR) << "Cannot copy from buffer: allocator is null";
    }   
  }

  void* Buffer::ptr() {
        return ptr_;
  }

  const void* Buffer::ptr() const {
        return ptr_;
  }

  size_t Buffer::byte_size() const {
        return byte_size_;
  }

  std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
        return allocator_;
  }

  DeviceType Buffer::device_type() const {
        return device_type_;
  }
  void Buffer::set_device_type(DeviceType device_type){
        device_type_ = device_type;
  }

  std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
    return shared_from_this();
  }

  bool Buffer::is_external() const {
    return is_external_;
  }

}