#ifndef BASE_ALLOC_H
#define BASE_ALLOC_H
#include <cstddef>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <ostream>
using namespace std;
namespace base {
// DeviceType and MemcpyKind enums define the types of devices and memory copy operations
enum class DeviceType {
    CPU=0,
    GPU=1,
    UNKNOWN=2
};

// Overload the output stream operator for DeviceType to provide a human-readable representation
inline std::ostream& operator<<(std::ostream& os, const DeviceType& type) {
    switch (type) {
        case DeviceType::CPU:
            os << "CPU"; // 输出可读的字符串
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


// MemcpyKind enum defines the direction of memory copy operations
enum class MemcpyKind {
    HostToHost=0,
    HostToDevice=1,
    DeviceToHost=2,
    DeviceToDevice=3
};

// cudastream_t is a placeholder for CUDA stream type
class DeviceAllocator {
    public:
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){};
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

// CPUAllocator and GPUAllocator are derived classes for CPU and GPU memory management
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

// Singleton instances for GPU allocators
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