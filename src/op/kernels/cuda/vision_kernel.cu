#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cfloat>
#include <mutex>
#include "vision_kernel.cuh"

namespace kernel {
namespace {

constexpr int kBlock = 256;
/// LayerNorm / softmax 走block-per-row，行内归约用固定线程数
constexpr int kRowBlock = 256;

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t st = (expr);                                               \
    CHECK_EQ(st, CUBLAS_STATUS_SUCCESS) << "cuBLAS call failed: " << #expr;    \
  } while (0)

/**
 * 进程内共享一个 cuBLAS handle。
 *
 * kernel 的分发签名只带 void* stream（拿不到 CudaConfig），因此 handle 在这里
 * 懒创建；每次调用前用 cublasSetStream 绑定到传入的 stream。
 *
 * 显式设为 PEDANTIC_MATH：Ampere/Hopper 上一旦启用 TF32，尾数只有 10 bit，
 * 与 HF 参考实现的逐阶段数值对齐会直接失败。
 */
cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  });
  return handle;
}

/// 把 bias 广播成列主序 Y[M, N] 的初值（每列一份 b[M]）
__global__ void fill_bias_kernel(float* Y, const float* b, int32_t M, int32_t N) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(M) * N;
  if (idx < total) {
    Y[idx] = b[idx % static_cast<size_t>(M)];
  }
}

__global__ void zero_kernel(float* Y, size_t total) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    Y[idx] = 0.0f;
  }
}

__global__ void layernorm_kernel(const float* x, int32_t dim, const float* gamma,
                                 const float* beta, float eps, float* out) {
  __shared__ float shared[kRowBlock];
  const int tid = static_cast<int>(threadIdx.x);
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const float* src = x + row;
  float* dst = out + row;

  float sum = 0.0f;
  for (int i = tid; i < dim; i += kRowBlock) {
    sum += src[i];
  }
  shared[tid] = sum;
  __syncthreads();
  for (int s = kRowBlock / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(dim);
  __syncthreads();

  float var = 0.0f;
  for (int i = tid; i < dim; i += kRowBlock) {
    const float d = src[i] - mean;
    var += d * d;
  }
  shared[tid] = var;
  __syncthreads();
  for (int s = kRowBlock / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }
  const float inv = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  __syncthreads();

  for (int i = tid; i < dim; i += kRowBlock) {
    float v = (src[i] - mean) * inv * gamma[i];
    if (beta != nullptr) v += beta[i];
    dst[i] = v;
  }
}

__global__ void gelu_kernel(float* x, size_t n, int32_t kind) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float v = x[idx];
  if (kind == 0) {  // tanh 近似
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    x[idx] = 0.5f * v * (1.0f + tanhf(kSqrt2OverPi * (v + 0.044715f * v * v * v)));
  } else {  // 精确 erf
    x[idx] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
  }
}

/// 一个线程处理一对 (d, d + half)，覆盖 n * heads * half
__global__ void rope2d_kernel(float* q, float* k, const float* cos_tab, const float* sin_tab,
                              int32_t n, int32_t dim, int32_t heads, int32_t head_dim) {
  const int32_t half = head_dim / 2;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * heads * half;
  if (idx >= total) {
    return;
  }
  const int32_t d = static_cast<int32_t>(idx % half);
  const int32_t h = static_cast<int32_t>((idx / half) % heads);
  const int32_t p = static_cast<int32_t>(idx / (static_cast<size_t>(half) * heads));

  const float c = cos_tab[static_cast<size_t>(p) * half + d];
  const float s = sin_tab[static_cast<size_t>(p) * half + d];
  const size_t base = static_cast<size_t>(p) * dim + static_cast<size_t>(h) * head_dim;

  float* qh = q + base;
  const float q0 = qh[d], q1 = qh[d + half];
  qh[d] = q0 * c - q1 * s;
  qh[d + half] = q1 * c + q0 * s;

  float* kh = k + base;
  const float k0 = kh[d], k1 = kh[d + half];
  kh[d] = k0 * c - k1 * s;
  kh[d + half] = k1 * c + k0 * s;
}

/// 对 score 的每一行（共 rows 行，每行 n 个）做 softmax，block per row
__global__ void softmax_rows_kernel(float* score, int32_t n) {
  __shared__ float shared[kRowBlock];
  const int tid = static_cast<int>(threadIdx.x);
  float* row = score + static_cast<size_t>(blockIdx.x) * n;

  float local_max = -FLT_MAX;
  for (int i = tid; i < n; i += kRowBlock) {
    local_max = fmaxf(local_max, row[i]);
  }
  shared[tid] = local_max;
  __syncthreads();
  for (int s = kRowBlock / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    __syncthreads();
  }
  const float max_val = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += kRowBlock) {
    const float e = __expf(row[i] - max_val);
    row[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (int s = kRowBlock / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }
  const float inv =1.0f / shared[0];
  __syncthreads();

  for (int i = tid; i < n; i += kRowBlock) {
    row[i] *= inv;
  }
}

__global__ void residual_kernel(float* y, const float* x, size_t n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += x[idx];
  }
}

/// 每个 block 负责一个目标像素 (i, j)，bilinear 插值后累加到 t 个帧
__global__ void pos_embed_kernel(const float* table, int32_t g, int32_t h, int32_t w, int32_t dim,
                                 int32_t t, float* out) {
  const int32_t pix = static_cast<int32_t>(blockIdx.x);
  const int32_t i = pix / w;
  const int32_t j = pix % w;

  const float scale_y = static_cast<float>(g) / static_cast<float>(h);
  const float scale_x = static_cast<float>(g) / static_cast<float>(w);

  float sy = (static_cast<float>(i) + 0.5f) * scale_y - 0.5f;
  sy = fmaxf(sy, 0.0f);
  const int32_t y0 = min(static_cast<int32_t>(floorf(sy)), g - 1);
  const int32_t y1 = min(y0 + 1, g - 1);
  const float fy = sy - static_cast<float>(y0);

  float sx = (static_cast<float>(j) + 0.5f) * scale_x - 0.5f;
  sx = fmaxf(sx, 0.0f);
  const int32_t x0 = min(static_cast<int32_t>(floorf(sx)), g - 1);
  const int32_t x1 = min(x0 + 1, g - 1);
  const float fx = sx - static_cast<float>(x0);

  const float* p00 = table + (static_cast<size_t>(y0) * g + x0) * dim;
  const float* p01 = table + (static_cast<size_t>(y0) * g + x1) * dim;
  const float* p10 = table + (static_cast<size_t>(y1) * g + x0) * dim;
  const float* p11 = table + (static_cast<size_t>(y1) * g + x1) * dim;

  const float w00 = (1.0f - fy) * (1.0f - fx);
  const float w01 = (1.0f - fy) * fx;
  const float w10 = fy * (1.0f - fx);
  const float w11 = fy * fx;

  const size_t per_frame = static_cast<size_t>(h) * w * dim;
  const size_t row = static_cast<size_t>(pix) * dim;
  for (int32_t d = static_cast<int32_t>(threadIdx.x); d < dim;
       d += static_cast<int32_t>(blockDim.x)) {
    const float v = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
    for (int32_t f = 0; f < t; ++f) {
      out[static_cast<size_t>(f) * per_frame + row + d] += v;
    }
  }
}

__global__ void spatial_merge_kernel(const float* in, int32_t h, int32_t w, int32_t dim, int32_t m,
                int32_t hb_num, int32_t wb_num, size_t total, float* out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int32_t merged = m * m * dim;
  const int32_t d = static_cast<int32_t>(idx % dim);
  const int32_t sub = static_cast<int32_t>((idx / dim) % (m * m));
  const int32_t o = static_cast<int32_t>(idx / merged);

  const int32_t i = sub / m;
  const int32_t j = sub % m;
  const int32_t wb = o % wb_num;
  const int32_t hb = (o / wb_num) % hb_num;
  const int32_t f = o / (wb_num * hb_num);

  const size_t src_row = static_cast<size_t>(f) * h * w +
                         static_cast<size_t>(hb * m + i) * w + (wb * m + j);
  out[idx] = in[src_row * dim + d];
}

inline int grid_of(size_t total, int block) {
  return static_cast<int>((total + block - 1) / block);
}

}  // namespace

void vision_gemm_nt_cu(const float* X, const float* W, const float* b, int32_t N, int32_t K,
                       int32_t M, float* Y, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  cublasHandle_t handle = get_cublas_handle();
  CUBLAS_CHECK(cublasSetStream(handle, cu_stream));

  // 行主序 Y[N,M] 等价于列主序 Yc[M,N]；Yc = Wc^T * Xc，
  // 其中 Wc 为列主序 [K,M]（即 W 行主序 [M,K]），Xc 为列主序 [K,N]。
  const size_t total = static_cast<size_t>(M) * N;
  if (b != nullptr) {
    fill_bias_kernel<<<grid_of(total, kBlock), kBlock, 0, cu_stream>>>(Y, b, M, N);
  } else {
    zero_kernel<<<grid_of(total, kBlock), kBlock, 0, cu_stream>>>(Y, total);
  }

  const float alpha = 1.0f;
  const float beta = 1.0f;  // 累加到已填好的 bias 上
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, K, X, K, &beta, Y,
                           M));
}

void vision_layernorm_cu(const float* x, int32_t rows, int32_t dim, const float* gamma,
                         const float* beta, float eps, float* out, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  layernorm_kernel<<<rows, kRowBlock, 0, cu_stream>>>(x, dim, gamma, beta, eps, out);
}

void vision_gelu_cu(float* x, size_t n, GeluKind kind, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  gelu_kernel<<<grid_of(n, kBlock), kBlock, 0, cu_stream>>>(x, n, static_cast<int32_t>(kind));
}

void vision_rope2d_cu(float* q, float* k, const float* cos_tab, const float* sin_tab, int32_t n,
                      int32_t dim, int32_t heads, int32_t head_dim, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  const size_t total = static_cast<size_t>(n) * heads * (head_dim / 2);
  rope2d_kernel<<<grid_of(total, kBlock), kBlock, 0, cu_stream>>>(q, k, cos_tab, sin_tab, n, dim,
                                                                 heads, head_dim);
}

int32_t vision_attention_score_rows_cu(int32_t n, int32_t heads) {
  // 单块 score 的元素数上限（512M float = 2 GB）。既控制显存峰值，也避免 int32
  // 溢出：n=13200 时整块 heads*n*n = 2.8e9 已超过 2^31-1。
  constexpr long long kMaxElems = 512LL << 20;
  const long long per_head = static_cast<long long>(n) * n;
  long long chunk = per_head > 0 ? kMaxElems / per_head : heads;
  if (chunk < 1) chunk = 1;
  if (chunk > heads) chunk = heads;
  return static_cast<int32_t>(chunk * n);
}

void vision_attention_cu(const float* q, const float* k, const float* v, int32_t n, int32_t dim,
                         int32_t heads, int32_t head_dim, float* out, float* score_buf,
                         int32_t score_rows, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  cublasHandle_t handle = get_cublas_handle();
  CUBLAS_CHECK(cublasSetStream(handle, cu_stream));

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  const long long int stride_qkv = head_dim;      // head 在行内的偏移
  const long long int stride_score = static_cast<long long int>(n) * n;
  int32_t chunk = n > 0 ? score_rows / n : heads;
  if (chunk < 1) chunk = 1;

  // 按 head 分块：整块 [heads, n, n] 在大图上会超出 int32 元素上限
  for (int32_t h0 = 0; h0 < heads; h0 += chunk) {
    const int32_t batch = (heads - h0) < chunk ? (heads - h0) : chunk;
    const size_t head_off = static_cast<size_t>(h0) * head_dim;

    // score[b][i][j] = scale * dot(q_i, k_j)
    // 列主序看作 C[j, i]（ldc = n），C = Kc^T * Qc，Kc/Qc 的leading dim 均为 dim
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, head_dim, &scale,
                                           k + head_off, dim, stride_qkv, q + head_off, dim,
                                           stride_qkv, &zero, score_buf, n, stride_score, batch));

    softmax_rows_kernel<<<batch * n, kRowBlock, 0, cu_stream>>>(score_buf, n);

    // out[i][h*hd + d] = sum_j score[b][i][j] * v[j][h*hd + d]
    // 列主序 Oc[d, i]（ldc = dim），Oc = Vc * Pc，Pc 为列主序 [n, n]（ldb = n）
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, n, n, &one,
                                           v + head_off, dim, stride_qkv, score_buf, n,
                                           stride_score, &zero, out + head_off, dim, stride_qkv,
                                           batch));
  }
}

void vision_residual_cu(float* y, const float* x, size_t n, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  residual_kernel<<<grid_of(n, kBlock), kBlock, 0, cu_stream>>>(y, x, n);
}

void vision_pos_embed_cu(const float* table, int32_t g, int32_t h, int32_t w, int32_t dim,
                         int32_t t, float* out, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  pos_embed_kernel<<<h * w, kBlock, 0, cu_stream>>>(table, g, h, w, dim, t, out);
}

void vision_spatial_merge_cu(const float* in, int32_t t, int32_t h, int32_t w, int32_t dim,
                             int32_t m, float* out, void* stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  const int32_t hb_num = h / m;
  const int32_t wb_num = w / m;
  const size_t total =
      static_cast<size_t>(t) * hb_num * wb_num * static_cast<size_t>(m) * m * dim;
  spatial_merge_kernel<<<grid_of(total, kBlock), kBlock, 0, cu_stream>>>(in, h, w, dim, m, hb_num,
                                                                        wb_num, total, out);
}

}  // namespace kernel
