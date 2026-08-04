#ifndef KLITE_INCLUDE_MODEL_CONFIG_H_
#define KLITE_INCLUDE_MODEL_CONFIG_H_
#include <cstdint>

namespace model {

/**
 * @brief KLite 扩展权重头的 magic（"KLIT"），用于与 llama2.c 原始格式区分
 *
 * 扩展头共 10 个 int32：
 *   magic, version, dim, hidden_dim, layer_num, head_num, kv_head_num,
 *   head_size, vocab_size, seq_len
 * 之所以需要扩展头，是因为 Qwen3 的 head_size 不等于 dim / head_num
 * （0.6B: head_dim=128，而 1024/16=64），无法从其它字段推导。
 */
constexpr int32_t kKliteModelMagic = 0x54494C4B;
constexpr int32_t kKliteModelVersion = 1;

/**
 * @brief 解析后的权重文件头信息
 *
 * llama2.c 原始格式为紧排的 7 个 int32（无 head_size，此时该字段为 0，
 * 由dim / head_num 推导）。
 */
struct ModelConfig {
  int32_t dim = 0;          // hidden size
  int32_t hidden_dim = 0;   // FFN intermediate size
  int32_t layer_num = 0;    // transformer block 数
  int32_t head_num = 0;     // query 头数
  int32_t kv_head_num = 0;  // key/value 头数（GQA）
  int32_t vocab_size = 0;   // 取负值表示 lm_head 不与 embedding 共享权重
  int32_t seq_len = 0;      // 最大序列长度（KV-Cache 容量）
  int32_t head_size = 0;    // 单头维度，仅扩展头提供；0 表示需推导
};

/**
 * @brief 运行期使用的模型结构信息，由 ModelConfig 推导而来
 */
struct TransformerConfig {
  virtual ~TransformerConfig() = default;

  int32_t dim_ = 0;            // hidden size
  int32_t hidden_dim_ = 0;     // 与 dim_ 相同，保留以兼容既有调用
  int32_t immediate_dim_ = 0;  // FFN intermediate size

  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t head_size_ = 0;

  int32_t q_dim_ = 0;   // head_num_ * head_size_
  int32_t kv_dim_ = 0;  // kv_head_num_ * head_size_
  int32_t kv_mul_ = 0;  // head_num_ / kv_head_num_

  int32_t vocab_size_ = 0;
  int32_t seq_len_ = 0;

  bool is_shared_weight_ = false;  // lm_head 是否复用 embedding 权重
};

}  // namespace model
#endif  // KLITE_INCLUDE_MODEL_CONFIG_H_
