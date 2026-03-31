#ifndef BASE_ALLOC_H
#define BASE_ALLOC_H
#include "base/cuda_config.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <ostream>
#include "base/base.h"
using namespace std;
namespace base {
inline std::ostream& operator<<(std::ostream& os, const DeviceType& type) {
    switch (type) {
        case DeviceType::CPU:
            os << "CPU";
            break;
        case DeviceType::GPU:
            os << "GPU";
            break;
        default:
            os << "UnknownDeviceType(" << static_cast<int>(type) << ")";
            break;
    }
    return os;
}

enum class MemcpyKind {
    HostToHost=0,
    HostToDevice=1,
    DeviceToHost=2,
    DeviceToDevice=3
};

class DeviceAllocator {
    public:
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){};
        virtual ~DeviceAllocator() = default; // 只加虚析构，不改名
        virtual void* allocate(size_t size) = 0;
        virtual bool release(void* ptr) = 0;
        virtual bool memcpy(void* dest, const void* src, size_t count, MemcpyKind kind,cudaStream_t stream=nullptr,bool async=false);
        virtual bool memsetZero(void* dest, size_t count,DeviceType deviceType,cudaStream_t stream=nullptr,bool async=false);
        DeviceType device_type() const {
            return device_type_;
        }
    protected:
        DeviceType device_type_ = DeviceType::UNKNOWN;
};

class CPUAllocator: public DeviceAllocator {
    public:
        explicit CPUAllocator():DeviceAllocator(DeviceType::CPU){}
        void* allocate(size_t size) override;
        bool release(void* ptr) override;
};

class GPUBuffer{
    public:
        void* ptr;
        size_t size;
        bool isBusy;
        GPUBuffer(void* ptr,size_t size):ptr(ptr),size(size),isBusy(false){
        }
        ~GPUBuffer(){
            if(ptr != nullptr){
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
};

class GPUAllocator: public DeviceAllocator {
    public:
        GPUAllocator():DeviceAllocator(DeviceType::GPU){}
        void* allocate(size_t size) override;
        bool release(void* ptr) override;
    private:
        void* allocate_from_BigBuffer(size_t size);
        void* allocate_from_SmallBuffer(size_t size);
        bool release_from_BigBuffer(void* ptr);
        bool release_from_SmallBuffer(void* ptr);
        
        map<int,size_t> no_use_size;
        map<int,vector<GPUBuffer*>> Big_Buffers;
        map<int,vector<GPUBuffer*>> Small_Buffers;
};

class GPUAllocatorInstance {
    public:
        static GPUAllocator& getInstance() {
            static GPUAllocator instance;
            return instance;
        }
};

class CPUAllocatorInstance {
    public:
        static CPUAllocator& getInstance() {
            static CPUAllocator instance;
            return instance;
        }
};

}

#endif