#include "model/raw_model_data.h"
#include <sys/mman.h>
#include <unistd.h>

namespace model {

RawModelData::~RawModelData() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);
    data = nullptr;
  }
  if (fd >= 0) {
    close(fd);
    fd = -1;
  }
}

const void* RawModelDataFp32::weight(size_t offset) const {
  return static_cast<const float*>(weight_data) + offset;
}

const void* RawModelDataInt8::weight(size_t offset) const {
  return static_cast<const int8_t*>(weight_data) + offset;
}

}  // namespace model
