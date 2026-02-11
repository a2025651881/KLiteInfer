#include <cstddef>

// DeviceType and MemcpyKind enums define the types of devices and memory copy operations
enum class DeviceType {
    CPU=0,
    GPU=1,
    UNKNOWN=2
};

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
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {};
        virtual void* allocate(size_t size);
        virtual void release(void* ptr);
        virtual bool memcpy(void* dest, const void* src, size_t count, MemcpyKind kind);
        virtual bool memsetZero(void* dest,size_t count,);
        DeviceType device_type() const {
            return device_type_;
        }
    protected:
        DeviceType device_type_ = DeviceType::UNKNOWN;
};

// CPUAllocator and GPUAllocator are derived classes for CPU and GPU memory management
class CPUAllocator: public DeviceAllocator {
    public:
        void* allocate(size_t size) override;
        void release(void* ptr) override;
        bool memcpy(void* dest, const void* src, size_t count, MemcpyKind kind) override;
        bool memsetZero(void* dest,size_t count) override;
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
}

class GPUAllocator: public DeviceAllocator {
    public:
        void* allocate(size_t size) override;
        void release(void* ptr) override;
        bool memcpy(void* dest, const void* src, size_t count) override;
        bool memsetZero(void* dest) override;
    private:
        void* allocate_from_BigBuffer(size_t size);
        void* allocate_from_SmallBuffer(size_t size);
        bool relase_from_BigBuffer(void* ptr);
        bool relase_from_SmallBuffer(void* ptr);
        
        map<int,size_t> no_use_size;
        map<int,vector<GPUBuffer*>> Big_buffers;
        map<int,vector<GPUBuffer*>> Small_buffers;
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
        static CpuAllocator& getInstance() {
            static CpuAllocator instance;
            return instance;
        }
};
