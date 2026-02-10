#include "base/alloc.h"
#include <glob/logging.h>
#include <cuda_runtime.h>

class GpuAllocator: public DeviceAllocator {
    private:
        void* allocate_from_BigBuffer(size_t size) {
            // Implementation to allocate from big buffer pool
            int id=cudaGetDevice();
            int index=-1;
            for(int i=0;i<Big_Buffers[id].size();i++){
                if(Big_Buffers[id][i].size - size < 1024*1024 && !Big_Buffers[id][i].isBusy){
                    index = i;
                }
            }
            if(index != -1){
                Big_Buffers[id][index].isBusy = true;
                return Big_Buffers[id][index].ptr;
            }
            // If no suitable big buffer is available, allocate a new one
            void* ptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            CHECK_EQ(err, cudaSuccess) << "CUDA malloc failed for big buffer: " << cudaGetErrorString(err);
            GPUBuffer* newBuffer = new GPUBuffer(ptr, size);
            newBuffer->isBusy = true;
            Big_Buffers[id].emplace_back(newBuffer);
            return ptr;
        }

        void* allocate_from_SmallBuffer(size_t size) {
            // Implementation to allocate from small buffer pool
            // Check if there are available small buffers of the requested size
            int id=cudaGetDevice();
            if(size > no_use_size[id]) goto allocate_new;

            for(int i=0;i<Small_Buffers[id].size();i++){
                if(Small_Buffers[id][i].size >= size && !Small_Buffers[id][i].isBusy){
                    Small_Buffers[id][i].isBusy = true;
                    no_use_size[id]-=size;
                    return Small_Buffers[id][i].ptr;
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
                no_use_size[id]+=size;
                Small_Buffers[id].emplace_back(newBuffer);
                return ptr;
        }

        bool release_from_BigBuffer(void* ptr) {
            // Implementation to release a buffer back to the big buffer pool
            for(int i=0;i<Big_Buffers.size();i++){
                int id=Big_Buffers[i].first;
                for(int j=0;j<Big_Buffers[id].size();j++){
                    if(Big_Buffers[id][j].ptr == ptr){
                        Big_Buffers[id][j].isBusy = false;
                        return true;
                    }
                }
            }
            return false; // Pointer not found in big buffer pool
        }

        bool release_from_SmallBuffer(void* ptr) {
            // Implementation to release a buffer back to the small buffer pool
            for(int i=0;i<Small_Buffers.size();i++){
                for(int j=0;j<Small_Buffers[id].size();j++){
                    if(Small_Buffers[id][j].ptr == ptr){
                        Small_Buffers[id][j].isBusy = false;
                        no_use_size[id]+=Small_Buffers[id][j].size;
                        return true;
                    }
                }
            }
            return false; // Pointer not found in small buffer pool
        }
    public:
        void* allocate(size_t size) override {
            if(size <= 0) return nullptr;
            
            if(size > 1024 * 1024){ // If requested size is greater than 1MB, allocate a new buffer
                return allocate_from_BigBuffer(size);
            } else { // Try to reuse existing buffers for smaller sizes
                return allocate_from_SmallBuffer(size);
            }

        }
        
        bool release(void* ptr) override {
            /* call cudaGetDevice is within the release function is unreliable 
             *because it may be called from a different thread than the one that allocated the memory, 
             *leading to incorrect device context. 
             */
            for(int i=0;i<Small_Buffers.size();i++){
                int id=Small_Buffers[i].first;
                Buffer* buffer = Small_Buffers[id].second;
                vector<Buffer*> tmp;
                if(no_use_size[id] > 1024*1024*1024) {
                    for(int j=0;j<Small_Buffers[id].size();j++){
                        if(!Small_Buffers[id][j].isBusy){
                            // switch to the correct device context before freeing the memory
                            state=cudaSetDevice(id);
                            state=cudaFree(Small_Buffers[id][j].ptr);
                            CHECK_EQ(state, cudaSuccess) << "CUDA free failed: " << cudaGetErrorString(state);
                        }else{
                            tmp.emplace_back(Small_Buffers[id][j]);
                        }
                    }
                }
                buffer.clear();
                Small_Buffers[id] = tmp;
                no_use_size[id] = 0;
            }

            if(release_from_SmallBuffer(ptr)) return true;
            if(release_from_BigBuffer(ptr)) return true;
            LOG(ERROR) << "Attempted to release a pointer that was not allocated by this allocator: " << ptr;
            return false;
        }
};
