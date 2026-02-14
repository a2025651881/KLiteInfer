#include "base/alloc.h"
#include <logging.h>
#include <cuda_runtime.h>
namespace base {
void* GPUAllocator::allocate_from_BigBuffer(size_t size) {
    // Implementation to allocate from big buffer pool
    int id;
    cudaGetDevice(&id);
    int index=-1;
    for(int i=0;i< this->Big_Buffers[id].size();i++){
        if(this->Big_Buffers[id][i]->size - size < 1024*1024 && !this->Big_Buffers[id][i]->isBusy){
            index = i;
        }
    }
    if(index != -1){
        this->Big_Buffers[id][index]->isBusy = true;
        return this->Big_Buffers[id][index]->ptr;  
    }
    // If no suitable big buffer is available, allocate a new one
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    CHECK_EQ(err, cudaSuccess) << "CUDA malloc failed for big buffer: " << cudaGetErrorString(err);
    GPUBuffer* newBuffer = new GPUBuffer(ptr, size);
    newBuffer->isBusy = true;
    this->Big_Buffers[id].emplace_back(newBuffer);
    return ptr;
}
void* GPUAllocator::allocate_from_SmallBuffer(size_t size) {
    // Implementation to allocate from small buffer pool
    // Check if there are available small buffers of the requested size
    int id;
    cudaGetDevice(&id);
    if(size > no_use_size[id]) goto allocate_new;
    for(int i=0;i<Small_Buffers[id].size();i++){
        if(Small_Buffers[id][i]->size >= size && !Small_Buffers[id][i]->isBusy){
            Small_Buffers[id][i]->isBusy = true;
            no_use_size[id]-=size;
            return Small_Buffers[id][i]->ptr;
        }
    }
    allocate_new:
        // If no suitable small buffer is available, allocate a new one
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        CHECK_EQ(err, cudaSuccess) << "CUDA malloc failed for small buffer: " << cudaGetErrorString(err);
        // Add the new buffer to the small buffer pool for future reuse
        // smallBuffers[size].emplace_back(ptr);
        GPUBuffer* newBuffer = new GPUBuffer(ptr, size);
        newBuffer->isBusy = true;
        this->no_use_size[id]+=size;
        this->Small_Buffers[id].emplace_back(newBuffer);
        return ptr;
}
bool GPUAllocator::release_from_BigBuffer(void* ptr) {
    // Implementation to release a buffer back to the big buffer pool
    for(int i=0;i<Big_Buffers.size();i++){
        for(int j=0;j<Big_Buffers[i].size();j++){
            if(Big_Buffers[i][j]->ptr == ptr){
                Big_Buffers[i][j]->isBusy = false;
                return true;
            }
        }
    }
    return false; // Pointer not found in big buffer pool
}
bool GPUAllocator::release_from_SmallBuffer(void* ptr) {
    // Implementation to release a buffer back to the small buffer pool
    for(int i=0;i<Small_Buffers.size();i++){
        for(int j=0;j<Small_Buffers[i].size();j++){
            if(Small_Buffers[i][j]->ptr == ptr){
                Small_Buffers[i][j]->isBusy = false;
                no_use_size[i]+=Small_Buffers[i][j]->size;
                return true;
            }
        }
    }
    return false; // Pointer not found in small buffer pool
}
void* GPUAllocator::allocate(size_t size){
    if(size <= 0) return nullptr;
    
    if(size > 1024 * 1024){ // If requested size is greater than 1MB, allocate a new buffer
        return allocate_from_BigBuffer(size);
    } else { // Try to reuse existing buffers for smaller sizes
        return allocate_from_SmallBuffer(size);
    }
}

bool GPUAllocator::release(void* ptr){
    /* call cudaGetDevice is within the release function is unreliable 
     *because it may be called from a different thread than the one that allocated the memory, 
     *leading to incorrect device context. 
     */
    for(auto now_map:Small_Buffers){
        int id=now_map.first;
        vector<GPUBuffer*>& buffer_list = now_map.second;
        vector<GPUBuffer*> tmp;
        if(no_use_size[id] > 1024*1024*1024) {
            for(int j=0;j<buffer_list.size();j++){
                if(!buffer_list[j]->isBusy){
                    // switch to the correct device context before freeing the memory
                    cudaError_t state=cudaSetDevice(id);
                    state=cudaFree(buffer_list[j]->ptr);
                    CHECK_EQ(state, cudaSuccess) << "CUDA free failed: " << cudaGetErrorString(state);
                }else{
                    tmp.push_back(buffer_list[j]);
                }
            }
        }
        buffer_list.clear();
        this->Small_Buffers[id] = tmp;
        this->no_use_size[id] = 0;
    }
    if(release_from_SmallBuffer(ptr)) return true;
    if(release_from_BigBuffer(ptr)) return true;
    LOG(ERROR) << "Attempted to release a pointer that was not allocated by this allocator: " << ptr;
    return false;
}
}