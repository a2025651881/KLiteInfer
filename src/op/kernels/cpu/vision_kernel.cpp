#include "vision_kernel.h"
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace kernel {
namespace {

inline float gelu_tanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return 0.5f * x * (1.f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

inline float gelu_erf(float x) {
  return 0.5f * x * (1.f + std::erf(x * 0.7071067811865476f));
}

void softmax_row(float* x, int32_t n) {
  float m = x[0];
  for (int32_t i = 1; i < n; ++i) m = std::max(m, x[i]);
  float sum = 0.f;
  for (int32_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - m);
    sum += x[i];
  }
  const float inv = 1.f / sum;
  for (int32_t i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace

void vision_gemm_nt_cpu(const float* X, const float* W, const float* b, int32_t N, int32_t K,
                        int32_t M, float* Y, void* stream) {
  (void)stream;
  // 行主序 [N,K] 用 armadillo 列主序 (K,N) 包装即为 X^T，故 Y^T = W * X^T
  arma::fmat Xm(const_cast<float*>(X), K, N, false, true);
  arma::fmat Wm(const_cast<float*>(W), K, M, false, true);
  arma::fmat Ym(Y, M, N, false, true);
  Ym = Wm.t() * Xm;
  if (b != nullptr) {
    arma::fvec bv(const_cast<float*>(b), M, false, true);
    Ym.each_col() += bv;
  }
}

void vision_layernorm_cpu(const float* x, int32_t rows, int32_t dim, const float* gamma,
                          const float* beta, float eps, float* out, void* stream) {
  (void)stream;
  for (int32_t r = 0; r < rows; ++r) {
    const float* src = x + static_cast<size_t>(r) * dim;
    float* dst = out + static_cast<size_t>(r) * dim;
    float mean = 0.f;
    for (int32_t i = 0; i < dim; ++i) mean += src[i];
    mean /= static_cast<float>(dim);
    float var = 0.f;
    for (int32_t i = 0; i < dim; ++i) {
      const float d = src[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(dim);
    const float inv = 1.f / std::sqrt(var + eps);
    for (int32_t i = 0; i < dim; ++i) {
      dst[i] = (src[i] - mean) * inv * gamma[i] + (beta ? beta[i] : 0.f);
    }
  }
}

void vision_gelu_cpu(float* x, size_t n, GeluKind kind, void* stream) {
  (void)stream;
  if (kind == GeluKind::kTanh) {
    for (size_t i = 0; i < n; ++i) x[i] = gelu_tanh(x[i]);
  } else {
    for (size_t i = 0; i < n; ++i) x[i] = gelu_erf(x[i]);
  }
}

void vision_rope2d_cpu(float* q, float* k, const float* cos_tab, const float* sin_tab, int32_t n,
                       int32_t dim, int32_t heads, int32_t head_dim, void* stream) {
  (void)stream;
  const int32_t half = head_dim / 2;
  for (int32_t p = 0; p < n; ++p) {
    const float* c = cos_tab + static_cast<size_t>(p) * half;
    const float* s = sin_tab + static_cast<size_t>(p) * half;
    for (int32_t h = 0; h < heads; ++h) {
      float* qh = q + static_cast<size_t>(p) * dim + static_cast<size_t>(h) * head_dim;
      float* kh = k + static_cast<size_t>(p) * dim + static_cast<size_t>(h) * head_dim;
      for (int32_t d = 0; d < half; ++d) {
        const float q0 = qh[d], q1 = qh[d + half];
        qh[d] = q0 * c[d] - q1 * s[d];
        qh[d + half] = q1 * c[d] + q0 * s[d];
        const float k0 = kh[d], k1 = kh[d + half];
        kh[d] = k0 * c[d] - k1 * s[d];
        kh[d + half] = k1 * c[d] + k0 * s[d];
      }
    }
  }
}

int32_t vision_attention_score_rows_cpu(int32_t n, int32_t heads) {
  (void)n;
  return heads;
}

void vision_attention_cpu(const float* q, const float* k, const float* v, int32_t n, int32_t dim,
                          int32_t heads, int32_t head_dim, float* out, float* score_buf,
                          int32_t score_rows, void* stream) {
  (void)stream;
  (void)score_rows;
  const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
  for (int32_t h = 0; h < heads; ++h) {
    const size_t off = static_cast<size_t>(h) * head_dim;
    float* scores = score_buf + static_cast<size_t>(h) * n;
    for (int32_t i = 0; i < n; ++i) {
      const float* qi = q + static_cast<size_t>(i) * dim + off;
      for (int32_t j = 0; j < n; ++j) {
        const float* kj = k + static_cast<size_t>(j) * dim + off;
        float dot = 0.f;
        for (int32_t d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
        scores[j] = dot * scale;
      }
      softmax_row(scores, n);
      float* dst = out + static_cast<size_t>(i) * dim + off;
      std::memset(dst, 0, sizeof(float) * head_dim);
      for (int32_t j = 0; j < n; ++j) {
        const float pw = scores[j];
        if (pw == 0.f) continue;
        const float* vj = v + static_cast<size_t>(j) * dim + off;
        for (int32_t d = 0; d < head_dim; ++d) dst[d] += pw * vj[d];
      }
    }
  }
}

void vision_residual_cpu(float* y, const float* x, size_t n, void* stream) {
  (void)stream;
  for (size_t i = 0; i < n; ++i) y[i] += x[i];
}

void vision_pos_embed_cpu(const float* table, int32_t g, int32_t h, int32_t w, int32_t dim,
                          int32_t t, float* out, void* stream) {
  (void)stream;
  const float scale_y = static_cast<float>(g) / static_cast<float>(h);
  const float scale_x = static_cast<float>(g) / static_cast<float>(w);
  const size_t per_frame = static_cast<size_t>(h) * w * dim;

  for (int32_t i = 0; i < h; ++i) {
    float sy = (static_cast<float>(i) + 0.5f) * scale_y - 0.5f;
    sy = std::max(sy, 0.f);
    const int32_t y0 = std::min(static_cast<int32_t>(std::floor(sy)), g - 1);
    const int32_t y1 = std::min(y0 + 1, g - 1);
    const float fy = sy - static_cast<float>(y0);

    for (int32_t j = 0; j < w; ++j) {
      float sx = (static_cast<float>(j) + 0.5f) * scale_x - 0.5f;
      sx = std::max(sx, 0.f);
      const int32_t x0 = std::min(static_cast<int32_t>(std::floor(sx)), g - 1);
      const int32_t x1 = std::min(x0 + 1, g - 1);
      const float fx = sx - static_cast<float>(x0);

      const float* p00 = table + (static_cast<size_t>(y0) * g + x0) * dim;
      const float* p01 = table + (static_cast<size_t>(y0) * g + x1) * dim;
      const float* p10 = table + (static_cast<size_t>(y1) * g + x0) * dim;
      const float* p11 = table + (static_cast<size_t>(y1) * g + x1) * dim;

      const float w00 = (1.f - fy) * (1.f - fx);
      const float w01 = (1.f - fy) * fx;
      const float w10 = fy * (1.f - fx);
      const float w11 = fy * fx;

      const size_t row = (static_cast<size_t>(i) * w + j) * dim;
      for (int32_t f = 0; f < t; ++f) {
        float* dst = out + static_cast<size_t>(f) * per_frame + row;
        for (int32_t d = 0; d < dim; ++d) {
          dst[d] += w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
        }
      }
    }
  }
}

void vision_spatial_merge_cpu(const float* in, int32_t t, int32_t h, int32_t w, int32_t dim,
                              int32_t m, float* out, void* stream) {
  (void)stream;
  const int32_t hb_num = h / m;
  const int32_t wb_num = w / m;
  const int32_t merged = m * m * dim;
  for (int32_t f = 0; f < t; ++f) {
    for (int32_t hb = 0; hb < hb_num; ++hb) {
      for (int32_t wb = 0; wb < wb_num; ++wb) {
        const int32_t o = (f * hb_num + hb) * wb_num + wb;
        for (int32_t i = 0; i < m; ++i) {
          for (int32_t j = 0; j < m; ++j) {
            const size_t src_row = static_cast<size_t>(f) * h * w +
                                   static_cast<size_t>(hb * m + i) * w + (wb * m + j);
            std::memcpy(out + static_cast<size_t>(o) * merged +
                            static_cast<size_t>(i * m + j) * dim,
                        in + src_row * dim, sizeof(float) * dim);
          }
        }
      }
    }
  }
}

}  // namespace kernel
