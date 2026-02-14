#include "base/alloc.h"
#include <glog/logging.h>
namespace base {
void* CPUAllocator::allocate(size_t size){
    // Implementation of CPU memory allocation
    if(size <= 0){
        LOG(ERROR) << "Invalid size for CPU memory allocation: " << size;
        return nullptr;
    }
    void* ptr = malloc(size);
    CHECK_EQ(ptr, nullptr) << "CPU memory allocation failed for size: " << size;
    return ptr;
}
bool CPUAllocator::release(void* ptr) {
    // Implementation of CPU memory release
    if (!ptr)
    {
        return false;
    }
    free(ptr);
    return true;
}
}