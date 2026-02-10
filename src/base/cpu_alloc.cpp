#include "base/alloc.h"
#include <glob/logging.h>

class CpuAllocator: public DeviceAllocator {
    public:
        void* allocate(size_t size) override {
            // Implementation of CPU memory allocation
            if(size <= 0){
                LOG(ERROR) << "Invalid size for CPU memory allocation: " << size;
                return nullptr;
            }
            ptr_=malloc(size);
            CHECK_EQ(ptr_, nullptr) << "CPU memory allocation failed for size: " << size;
            return ptr_;
        }
        bool release(void* ptr) override {
            // Implementation of CPU memory release
            if (!ptr)
            {
                return false;
            }
            free(ptr);
            return true;
        }
};