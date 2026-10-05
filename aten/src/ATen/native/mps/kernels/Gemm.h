#pragma once
#include <c10/metal/common.h>

struct GemmParams {
  int32_t M, N, K, batch;
  int64_t lda, ldb, ldc;
  int64_t batch_a, batch_b, batch_c;
  int64_t bias_b, bias_r, bias_c;
  float alpha, beta;
  int32_t splits, k_chunk, simd_m, simd_n, simd_k, slots;
  int32_t uber_m, uber_n, ubers_m, ubers_n, barrier;
  int32_t gx, gy, groups, full_groups;
  bool has_bias, raster_m, linear, swap;
};
