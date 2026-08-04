#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/mha_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/rope_kernel.cuh"
#include "cuda/scale_kernel.cuh"
#include "cuda/softmax_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "cuda/vision_kernel.cuh"
#include "kernels_interface.h"
namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return emb_kernel_normal;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return emb_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
  }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu_qint8;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return mha_kernel;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return mha_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rope_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_inplace_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return scale_inplace_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a scale kernel.";
    return nullptr;
  }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return softmax_inplace_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return softmax_inplace_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an softmax kernel.";
    return nullptr;
  }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return swiglu_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return swiglu_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
    return nullptr;
  }
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu_dim;
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu_dim;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm dim kernel.";
    return nullptr;
  }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_sum_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
    return nullptr;
  }
}

// -----------------------------------------------------------------------------
//  视觉 kernel 与 MRoPE 分发
// -----------------------------------------------------------------------------
#define KLITE_DISPATCH_KERNEL(ret_type, getter, cpu_impl, cu_impl, what)  \
  ret_type getter(base::DeviceType device_type) {                         \
    if (device_type == base::DeviceType::kDeviceCPU) {                    \
      return cpu_impl;                                                    \
    } else if (device_type == base::DeviceType::kDeviceCUDA) {            \
      return cu_impl;                                                     \
    } else {                                \
      LOG(FATAL) << "Unknown device type for get a " what " kernel.";     \
      return nullptr;                                                     \
    }                                                                     \
  }

KLITE_DISPATCH_KERNEL(RopeHalfSplitKernel, get_rope_half_split_kernel, rope_half_split_cpu,
                      rope_half_split_cu, "mrope half split")
KLITE_DISPATCH_KERNEL(VisionGemmNTKernel, get_vision_gemm_nt_kernel, vision_gemm_nt_cpu,
                      vision_gemm_nt_cu, "vision gemm")
KLITE_DISPATCH_KERNEL(VisionLayerNormKernel, get_vision_layernorm_kernel, vision_layernorm_cpu,
                      vision_layernorm_cu, "vision layernorm")
KLITE_DISPATCH_KERNEL(VisionGeluKernel, get_vision_gelu_kernel, vision_gelu_cpu, vision_gelu_cu,
                      "vision gelu")
KLITE_DISPATCH_KERNEL(VisionRope2dKernel, get_vision_rope2d_kernel, vision_rope2d_cpu,
                      vision_rope2d_cu, "vision rope2d")
KLITE_DISPATCH_KERNEL(VisionAttentionKernel, get_vision_attention_kernel, vision_attention_cpu,
                      vision_attention_cu, "vision attention")
KLITE_DISPATCH_KERNEL(VisionResidualKernel, get_vision_residual_kernel, vision_residual_cpu,
                      vision_residual_cu, "vision residual")
KLITE_DISPATCH_KERNEL(VisionPosEmbedKernel, get_vision_pos_embed_kernel, vision_pos_embed_cpu,
                      vision_pos_embed_cu, "vision pos embed")
KLITE_DISPATCH_KERNEL(VisionSpatialMergeKernel, get_vision_spatial_merge_kernel,
                      vision_spatial_merge_cpu, vision_spatial_merge_cu, "vision spatial merge")

#undef KLITE_DISPATCH_KERNEL

int32_t vision_attention_score_rows(base::DeviceType device_type, int32_t n, int32_t heads) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return vision_attention_score_rows_cpu(n, heads);
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return vision_attention_score_rows_cu(n, heads);
  }
  LOG(FATAL) << "Unknown device type for vision attention score rows.";
  return 0;
}

}  // namespace kernel