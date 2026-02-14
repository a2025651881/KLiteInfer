#include "alloc.h"
#include <logging.h>
#include <cuda_runtime.h>
using namespace std;
namespace base {
bool DeviceAllocator::memcpy(void* dest, const void* src, size_t count, MemcpyKind kind,cudaStream_t stream,bool async) {
    // Base implementation of memory copy, can be overridden by derived classes
    if(dest == nullptr || src == nullptr || count < 0){
        return false;
    }
    if(count == 0){
        return true; // No data to copy, consider it a success
    }


    switch (kind)
    {
        case MemcpyKind::HostToHost:
            std::memcpy(dest, src, count);
            return true;
        case MemcpyKind::HostToDevice:
        case MemcpyKind::DeviceToHost:
        case MemcpyKind::DeviceToDevice:{
            cudaMemcpyKind cudaKind;
            switch (kind)
            {
                case MemcpyKind::HostToDevice:
                    cudaKind = cudaMemcpyHostToDevice;
                    break;
                case MemcpyKind::DeviceToHost:
                    cudaKind = cudaMemcpyDeviceToHost;
                    break;
                case MemcpyKind::DeviceToDevice:
                    cudaKind = cudaMemcpyDeviceToDevice;
                    break;
                default:
                    return false; // Should never reach here
            }
            // 如果stream不为空则使用stream异步拷贝，否则使用同步拷贝
            cudaError_t errn=cudaSuccess;
            if(stream != nullptr){
                errn = cudaMemcpyAsync(dest, src, count, cudaKind, stream);
            }else {
                errn = cudaMemcpy(dest, src, count, cudaKind);
            }
            if (errn != cudaSuccess) {
                LOG(ERROR) << "CUDA memcpy failed: " << cudaGetErrorString(errn);
                return false;
            }
            break;
        }
        default:
            // Other kinds of memory copy not supported in base allocator
            LOG(ERROR) << "Unsupported MemcpyKind: " << static_cast<int>(kind);
            return false;
        }

    // If async is true, we need to synchronize to ensure the copy is complete before returning
    if(async){
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG(ERROR) << "CUDA async memcpy synchronization failed: " << cudaGetErrorString(err);
            return false;
        }
    return true;
    }
    return false;
}

bool DeviceAllocator::memsetZero(void* dest, size_t count,DeviceType deviceType,cudaStream_t stream,bool async) {
    // Base implementation of memory set to zero, can be overridden by derived classes
    if(dest == nullptr || count <= 0){
        LOG(ERROR) << "Invalid parameters for memsetZero: dest=" << dest << ", count=" << count;
        return false;
    }
    switch (deviceType)
    {
        case DeviceType::CPU:
            memset(dest, 0, count);
            return true;
        case DeviceType::GPU:{
            cudaError_t errn=cudaSuccess;
            if(stream != nullptr){
                errn = cudaMemsetAsync(dest, 0, count, stream);
            }else {
                errn = cudaMemset(dest, 0, count);
            }
            if (errn != cudaSuccess) {
                LOG(ERROR) << "CUDA async memset failed: " << cudaGetErrorString(errn);
                return false;
            }
            if(async){
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    LOG(ERROR) << "CUDA async memset synchronization failed: " << cudaGetErrorString(err);
                    return false;
                }
            }
        return true;
        }
    default:
        LOG(ERROR) << "Unsupported DeviceType for memsetZero: " << static_cast<int>(deviceType);
        return false;
    }
    return false;
}
}

