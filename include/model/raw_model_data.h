#ifndef KLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#define KLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
#include <cstddef>
#include <cstdint>

namespace model {

/**
 * @brief mmap 映射的权重文件原始数据
 *
 * data 指向整个文件起始位置，weight_data 指向跳过文件头之后的权重区。
 */
struct RawModelData {
  virtual ~RawModelData();

  int32_t fd = -1;
  size_t file_size = 0;
  size_t header_size = 0;  // 文件头字节数，权重区从这里开始
  void* data = nullptr;
  void* weight_data = nullptr;

  /// @param offset 以元素（而非字节）为单位的偏移
  virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};

}  // namespace model
#endif  // KLITE_INCLUDE_MODEL_RAW_MODEL_DATA_H_
