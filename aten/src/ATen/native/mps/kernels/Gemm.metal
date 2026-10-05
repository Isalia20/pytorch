#include <ATen/native/mps/kernels/Gemm.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

template <typename T>
inline T gemm_epilogue(
    float acc,
    device const T* bias,
    long i,
    constant GemmParams& p) {
  float v = acc * p.alpha;
  if (p.has_bias) {
    v += float(bias[i]) * p.beta;
  }
  return T(v);
}

template <
    typename T_,
    int BM_,
    int BN_,
    int BK_,
    int SM_,
    int SN_,
    bool SA_,
    bool SB_,
    bool TA_,
    bool TB_>
struct GemmTile {
  using T = T_;
  static constexpr constant int BM = BM_, BN = BN_, BK = BK_, SM = SM_,
                                SN = SN_, TM = BM_ / (8 * SM_),
                                TN = BN_ / (8 * SN_), NT = 32 * SM_ * SN_;
  static constexpr constant bool SA = SA_, SB = SB_, TA = TA_, TB = TB_;
};

template <int W>
inline long gemm_stage_origin(long ld, uint tid) {
  return tid / (W / 4) * ld + tid % (W / 4) * 4;
}

template <typename T, int D0, int D1, int NT, bool TR, bool E0, bool E1>
inline void gemm_stage(
    threadgroup T* tile,
    device const T* src,
    long ld,
    uint n0,
    uint n1,
    uint tid) {
  constexpr int H = TR ? D1 : D0, W = TR ? D0 : D1;
  constexpr bool ER = TR ? E1 : E0, EC = TR ? E0 : E1;
  const uint rows = TR ? n1 : n0, cols = TR ? n0 : n1;
  constexpr uint vecs = W / 4, row_step = NT / vecs;
  if (tid >= row_step * vecs) {
    return;
  }
  const ushort first_row = tid / vecs, col = (tid % vecs) * 4;
#pragma clang loop unroll(full)
  for (ushort row_base = 0; row_base < H; row_base += row_step) {
    const ushort row = first_row + row_base;
    if (H % row_step != 0 && row >= H) {
      break;
    }
    vec<T, 4> v;
    if ((!ER || row < rows) && (!EC || col + 3 < cols)) {
      v = *reinterpret_cast<device const vec<T, 4>*>(src);
    } else {
#pragma clang loop unroll(full)
      for (ushort i = 0; i < 4; ++i) {
        v[i] = (!ER || row < rows) && (!EC || col + i < cols) ? src[i] : T(0);
      }
    }
    *reinterpret_cast<threadgroup vec<T, 4>*>(tile + row * W + col) = v;
    src += row_step * ld;
  }
}

template <typename C>
struct GemmSimdPtrs {
  device const typename C::T *A, *B, *ap, *bp, *sa, *sb;
};

template <typename C, bool EM, bool EN, bool EK>
inline void gemm_simd_step(
    thread const GemmSimdPtrs<C>& q,
    threadgroup typename C::T* tile_a,
    threadgroup typename C::T* tile_b,
    thread simdgroup_matrix<float, 8, 8> (&acc)[C::TN][C::TM],
    constant GemmParams& p,
    uint rows,
    uint cols,
    uint count_k,
    ushort sg,
    ushort lane) {
  using T = typename C::T;
  const uint sm = sg / C::SN, sn = sg % C::SN;
  const uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
  const uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
  const uint tid = sg * 32 + lane;
  if (C::SA) {
    gemm_stage<T, C::BM, C::BK, C::NT, C::TA, EM, EK>(
        tile_a, q.sa, p.lda, rows, count_k, tid);
  }
  if (C::SB) {
    gemm_stage<T, C::BK, C::BN, C::NT, C::TB, EK, EN>(
        tile_b, q.sb, p.ldb, count_k, cols, tid);
  }
  if (C::SA || C::SB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma clang loop unroll(full)
  for (uint kk = 0; kk < C::BK; kk += 8) {
    if (EK && kk >= count_k) {
      break;
    }
    simdgroup_matrix<T, 8, 8> af[C::TM], bf[C::TN];
#pragma clang loop unroll(full)
    for (ushort j = 0; j < C::TN; ++j) {
      const uint n = (j * C::SN + sn) * 8 + fn, k = kk + fm;
      thread auto& b = bf[j].thread_elements();
      if (C::SB) {
        const uint off = C::TB ? n * C::BK + k : k * C::BN + n;
        b[0] = tile_b[off];
        b[1] = tile_b[off + (C::TB ? C::BK : 1)];
      } else if (!EN && !EK) {
        if (!C::TB) {
          const auto v = *reinterpret_cast<device const vec<T, 2>*>(
              q.bp + kk * p.ldb + j * C::SN * 8);
          b[0] = v[0];
          b[1] = v[1];
        } else {
          const auto r = q.bp + j * C::SN * 8 * p.ldb + kk;
          b[0] = r[0];
          b[1] = r[p.ldb];
        }
      } else if (!C::TB && (!EN || n + 1 < cols) && (!EK || k < count_k)) {
        const auto v =
            *reinterpret_cast<device const vec<T, 2>*>(q.B + k * p.ldb + n);
        b[0] = v[0];
        b[1] = v[1];
      } else {
#pragma clang loop unroll(full)
        for (ushort e = 0; e < 2; ++e) {
          const bool valid = (!EN || n + e < cols) && (!EK || k < count_k);
          b[e] = valid ? q.B[C::TB ? (n + e) * p.ldb + k : k * p.ldb + n + e]
                       : T(0);
        }
      }
    }
#pragma clang loop unroll(full)
    for (ushort i = 0; i < C::TM; ++i) {
      const uint m = (i * C::SM + sm) * 8 + fm, k = kk + fn;
      thread auto& a = af[i].thread_elements();
      if (C::SA) {
        const uint off = C::TA ? k * C::BM + m : m * C::BK + k;
        a[0] = tile_a[off];
        a[1] = tile_a[off + (C::TA ? C::BM : 1)];
      } else if (!EM && !EK) {
        if (!C::TA) {
          const auto v = *reinterpret_cast<device const vec<T, 2>*>(
              q.ap + i * C::SM * 8 * p.lda + kk);
          a[0] = v[0];
          a[1] = v[1];
        } else {
          const auto r = q.ap + kk * p.lda + i * C::SM * 8;
          a[0] = r[0];
          a[1] = r[p.lda];
        }
      } else if (!C::TA && (!EM || m < rows) && (!EK || k + 1 < count_k)) {
        const auto v =
            *reinterpret_cast<device const vec<T, 2>*>(q.A + m * p.lda + k);
        a[0] = v[0];
        a[1] = v[1];
      } else {
#pragma clang loop unroll(full)
        for (ushort e = 0; e < 2; ++e) {
          const bool valid = (!EM || m < rows) && (!EK || k + e < count_k);
          a[e] = valid ? q.A[C::TA ? (k + e) * p.lda + m : m * p.lda + k + e]
                       : T(0);
        }
      }
    }
#pragma clang loop unroll(full)
    for (ushort j = 0; j < C::TN; ++j) {
#pragma clang loop unroll(full)
      for (ushort i = 0; i < C::TM; ++i) {
        simdgroup_multiply_accumulate(acc[j][i], af[i], bf[j], acc[j][i]);
      }
    }
  }
  if (C::SA || C::SB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename C, bool EM, bool EN, int MODE, typename O>
inline void gemm_simd_store(
    thread simdgroup_matrix<float, 8, 8> (&acc)[C::TN][C::TM],
    device O* out,
    device const typename C::T* bias,
    long ld,
    constant GemmParams& p,
    uint batch,
    uint m0,
    uint n0,
    ushort sg,
    ushort lane) {
  const uint sm = sg / C::SN, sn = sg % C::SN;
  const uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
  const uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
#pragma clang loop unroll(full)
  for (ushort j = 0; j < C::TN; ++j) {
#pragma clang loop unroll(full)
    for (ushort i = 0; i < C::TM; ++i) {
      const uint m = m0 + (i * C::SM + sm) * 8 + fm;
      const auto e = acc[j][i].thread_elements();
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 2; ++c) {
        const uint n = n0 + (j * C::SN + sn) * 8 + fn + c;
        if ((EM && m >= uint(p.M)) || (EN && n >= uint(p.N))) {
          continue;
        }
        float v = e[c];
        if (MODE == 1) {
          v = v * p.alpha +
              float(bias[batch * p.bias_b + m * p.bias_r + n * p.bias_c]) *
                  p.beta;
        } else if (MODE == 0) {
          v *= p.alpha;
        }
        out[m * ld + n] = O(v);
      }
    }
  }
}

template <typename C, bool EM, bool EN>
inline void gemm_simd_tile(
    device const typename C::T* A,
    device const typename C::T* B,
    device typename C::T* Cout,
    device const typename C::T* bias,
    device float* partials,
    threadgroup typename C::T* tile_a,
    threadgroup typename C::T* tile_b,
    constant GemmParams& p,
    uint3 group,
    ushort sg,
    ushort lane) {
  const uint batch = group.z / p.splits, split = group.z % p.splits;
  const uint m0 = group.y * C::BM, n0 = group.x * C::BN;
  const uint kstart = split * p.k_chunk;
  const uint kend = min(kstart + p.k_chunk, uint(p.K));
  A += batch * p.batch_a + (C::TA ? kstart * p.lda + m0 : m0 * p.lda + kstart);
  B += batch * p.batch_b + (C::TB ? n0 * p.ldb + kstart : kstart * p.ldb + n0);
  const uint rows = p.M - m0, cols = p.N - n0;
  const uint sm = sg / C::SN, sn = sg % C::SN;
  const uint fm = ((lane >> 1) & 3) | ((lane >> 2) & 4);
  const uint fn = (((lane >> 2) & 2) | (lane & 1)) * 2;
  const uint tid = sg * 32 + lane;
  GemmSimdPtrs<C> q = {
      A,
      B,
      A + (C::TA ? fn * p.lda + sm * 8 + fm : (sm * 8 + fm) * p.lda + fn),
      B + (C::TB ? (sn * 8 + fn) * p.ldb + fm : fm * p.ldb + sn * 8 + fn),
      A + gemm_stage_origin<C::TA ? C::BM : C::BK>(p.lda, tid),
      B + gemm_stage_origin<C::TB ? C::BK : C::BN>(p.ldb, tid)};
  simdgroup_matrix<float, 8, 8> acc[C::TN][C::TM];
#pragma clang loop unroll(full)
  for (ushort j = 0; j < C::TN; ++j) {
#pragma clang loop unroll(full)
    for (ushort i = 0; i < C::TM; ++i) {
      acc[j][i] = simdgroup_matrix<float, 8, 8>(0);
    }
  }
  const long a_step = C::TA ? C::BK * p.lda : C::BK;
  const long b_step = C::TB ? C::BK : C::BK * p.ldb;
  const uint full_end = kend - kend % C::BK;
  for (uint k0 = kstart; k0 < full_end; k0 += C::BK) {
    gemm_simd_step<C, EM, EN, false>(
        q, tile_a, tile_b, acc, p, rows, cols, C::BK, sg, lane);
    q.A += a_step;
    q.B += b_step;
    q.ap += a_step;
    q.bp += b_step;
    q.sa += a_step;
    q.sb += b_step;
  }
  if (full_end < kend) {
    gemm_simd_step<C, EM, EN, true>(
        q, tile_a, tile_b, acc, p, rows, cols, kend - full_end, sg, lane);
  }
  if (p.splits > 1) {
    const auto plane = (ulong(batch) * p.splits + split) * p.M * p.N;
    gemm_simd_store<C, EM, EN, 2>(
        acc, partials + plane, bias, p.N, p, batch, m0, n0, sg, lane);
  } else if (p.has_bias) {
    gemm_simd_store<C, EM, EN, 1>(
        acc, Cout + batch * p.batch_c, bias, p.ldc, p, batch, m0, n0, sg, lane);
  } else {
    gemm_simd_store<C, EM, EN, 0>(
        acc, Cout + batch * p.batch_c, bias, p.ldc, p, batch, m0, n0, sg, lane);
  }
}

template <typename C>
kernel void gemm_simd(
    device const typename C::T* A [[buffer(0)]],
    device const typename C::T* B [[buffer(1)]],
    device typename C::T* out [[buffer(2)]],
    device const typename C::T* bias [[buffer(3)]],
    constant GemmParams& p [[buffer(4)]],
    device float* partials [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup typename C::T tile_a[C::SA ? C::BM * C::BK : 1];
  threadgroup typename C::T tile_b[C::SB ? C::BN * C::BK : 1];
  if (p.linear) {
    const uint nx = (p.N + C::BN - 1) / C::BN, ny = (p.M + C::BM - 1) / C::BM;
    group = uint3(group.x % nx, group.x / nx % ny, group.x / nx / ny);
  }
  const bool em = (group.y + 1) * C::BM > uint(p.M);
  const bool en = (group.x + 1) * C::BN > uint(p.N);
  if (!em && !en) {
    gemm_simd_tile<C, false, false>(
        A, B, out, bias, partials, tile_a, tile_b, p, group, sg, lane);
  } else if (!em) {
    gemm_simd_tile<C, false, true>(
        A, B, out, bias, partials, tile_a, tile_b, p, group, sg, lane);
  } else if (!en) {
    gemm_simd_tile<C, true, false>(
        A, B, out, bias, partials, tile_a, tile_b, p, group, sg, lane);
  } else {
    gemm_simd_tile<C, true, true>(
        A, B, out, bias, partials, tile_a, tile_b, p, group, sg, lane);
  }
}

template <typename T, int VEC>
kernel void gemm_reduce(
    device const float* partials [[buffer(0)]],
    device T* C [[buffer(1)]],
    device const T* bias [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]) {
  const uint m = tid.y, n = tid.x * VEC;
  if (m >= uint(p.M) || n >= uint(p.N)) {
    return;
  }
  const ulong plane = ulong(p.M) * p.N;
  const ulong src = ulong(tid.z) * p.splits * plane + ulong(m) * p.N + n;
  float sum[VEC] = {};
  for (int s = 0; s < p.splits; ++s) {
#pragma clang loop unroll(full)
    for (int v = 0; v < VEC; ++v) {
      sum[v] += partials[src + s * plane + v];
    }
  }
#pragma clang loop unroll(full)
  for (int v = 0; v < VEC; ++v) {
    const long i = tid.z * p.bias_b + m * p.bias_r + (n + v) * p.bias_c;
    C[tid.z * p.batch_c + m * p.ldc + n + v] =
        gemm_epilogue(sum[v], bias, i, p);
  }
}

template <typename T>
struct GemmLoad4 {
  using type = vec<T, 4>;
};

template <>
struct GemmLoad4<half> {
  using type = packed_half4;
};

template <typename T>
inline vec<T, 4> gemm_load4(device const T* p) {
  return vec<T, 4>(
      *reinterpret_cast<device const typename GemmLoad4<T>::type*>(p));
}

template <typename T>
inline void vector_store(
    device T* C,
    device const T* bias,
    float value,
    uint batch,
    uint w,
    uint r,
    constant GemmParams& p) {
  const uint m = p.swap ? r : w, n = p.swap ? w : r;
  const long i = batch * p.bias_b + m * p.bias_r + n * p.bias_c;
  C[batch * p.batch_c + m * p.ldc + n] = gemm_epilogue(value, bias, i, p);
}

template <typename T, int R, bool TV, bool EDGE>
inline void vector_trans_a_step(
    device const T* x,
    device const T* v,
    thread float4 (&acc)[R],
    uint w,
    constant GemmParams& p) {
  float4 a = 0;
  if (!EDGE || w + 3 < uint(p.M)) {
    a = float4(gemm_load4(x));
  } else {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 4; ++c) {
      if (w + c < uint(p.M)) {
        a[c] = float(x[c]);
      }
    }
  }
  float b[R];
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R; ++r) {
    b[r] = float(v[TV ? r * p.ldb : r]);
  }
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R; ++r) {
    acc[r] += a * b[r];
  }
}

template <typename T, int R, bool TV, bool EDGE>
inline void vector_trans_a(
    device const T* x,
    device const T* v,
    thread float4 (&acc)[R],
    uint w,
    uint y,
    constant GemmParams& p) {
  if (EDGE && w >= uint(p.M)) {
    return;
  }
  x += y * p.lda + w;
  v += TV ? y : y * p.ldb;
  const long x_step = p.gy * p.lda, v_step = TV ? p.gy : p.gy * p.ldb;
  for (uint s = 0; s < uint(p.K / p.gy); ++s) {
    vector_trans_a_step<T, R, TV, EDGE>(x, v, acc, w, p);
    x += x_step;
    v += v_step;
  }
  if (y < uint(p.K % p.gy)) {
    vector_trans_a_step<T, R, TV, EDGE>(x, v, acc, w, p);
  }
}

template <typename T, int R, bool FLAT, bool TV, bool EK>
inline void vector_dot(
    device const T* xp,
    device const T* vp,
    thread float (&acc)[R],
    uint k,
    uint4 x_off,
    uint4 v_off,
    constant GemmParams& p) {
  const bool valid = !EK || k < (p.K & ~3u);
  const auto zero = vec<T, 4>(0);
  vec<T, 4> a = zero, b[R];
  if (valid) {
    a = FLAT ? vec<T, 4>(xp[x_off.x], xp[x_off.y], xp[x_off.z], xp[x_off.w])
             : gemm_load4(xp);
  }
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R; ++r) {
    b[r] = zero;
    if (valid) {
      const auto row = vp + (TV ? r * p.ldb : r);
      b[r] = TV
          ? gemm_load4(row)
          : vec<T, 4>(row[v_off.x], row[v_off.y], row[v_off.z], row[v_off.w]);
    }
  }
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R; ++r) {
    acc[r] += dot(float4(a), float4(b[r]));
  }
}

template <typename T, int R, int U, bool FLAT, bool TV>
kernel void gemm_vector(
    device const T* X [[buffer(0)]],
    device const T* V [[buffer(1)]],
    device T* C [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    constant GemmParams& p [[buffer(4)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 t [[thread_position_in_threadgroup]]) {
  X += group.z * p.batch_a;
  V += group.z * p.batch_b;
  const uint gx = p.gx, gy = p.gy, tid = t.y * gx + t.x;
  if (FLAT && R < 8) {
    const uint w = (group.x * gx + t.x) * 4;
    float4 acc[R] = {};
    if ((group.x + 1) * gx * 4 <= uint(p.M)) {
      vector_trans_a<T, R, TV, false>(X, V, acc, w, t.y, p);
    } else {
      vector_trans_a<T, R, TV, true>(X, V, acc, w, t.y, p);
    }
    for (ushort c = 0; c < 4; ++c) {
      for (ushort r = 0; r < R; ++r) {
        partial[((c * R + r) * gx + t.x) * gy + t.y] = acc[r][c];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gy >= 32) {
      const uint lane = tid % 32;
      for (uint q = tid / 32; q < 4 * R * gx; q += gx * gy / 32) {
        const uint col = group.x * gx * 4 + (q % gx) * 4 + q / (R * gx);
        if (col >= uint(p.M)) {
          continue;
        }
        float sum = 0;
        for (uint s = 0; s < gy; s += 32) {
          sum += simd_sum(partial[q * gy + s + lane]);
        }
        if (lane == 0) {
          vector_store(C, bias, sum, group.z, col, (q / gx) % R, p);
        }
      }
    } else {
      const uint y = tid % gy;
      for (uint s = gy / 2; s; s /= 2) {
        if (y < s) {
          for (uint q = tid / gy; q < 4 * R * gx; q += gx) {
            partial[q * gy + y] += partial[q * gy + y + s];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint q = tid; q < 4 * R * gx; q += gx * gy) {
        const uint col = group.x * gx * 4 + (q % gx) * 4 + q / (R * gx);
        if (col < uint(p.M)) {
          vector_store(C, bias, partial[q * gy], group.z, col, (q / gx) % R, p);
        }
      }
    }
    return;
  }
  const bool tiled = gy > 1;
  const uint k_threads = tiled ? gy : gx;
  const uint w = tiled ? group.y * gx + t.x : group.y;
  const uint wc = min(w, uint(p.M) - 1);
  const uint lane_k = tiled ? t.y : t.x;
  float acc[R] = {};
  auto xp = X + (FLAT ? wc + lane_k * 4 * p.lda : wc * p.lda + lane_k * 4);
  auto vp = V + (TV ? lane_k * 4 : lane_k * 4 * p.ldb);
  const uint lda = p.lda, ldb = p.ldb, step = 4 * k_threads;
  const uint x_adv = FLAT ? step * lda : 0, v_adv = TV ? 0 : step * ldb;
  uint4 x_off = uint4(0, 1, 2, 3) * lda, v_off = uint4(0, 1, 2, 3) * ldb;
  uint block = 0;
  for (; block < uint(p.full_groups); block += U) {
#pragma clang loop unroll(full)
    for (ushort u = 0; u < U; ++u) {
      vector_dot<T, R, FLAT, TV, false>(xp, vp, acc, 0, x_off, v_off, p);
      x_off += x_adv;
      v_off += v_adv;
      xp += FLAT ? 0 : step;
      vp += TV ? step : 0;
    }
  }
  for (; block < uint(p.groups); ++block) {
    const uint k = (block * k_threads + lane_k) * 4;
    vector_dot<T, R, FLAT, TV, true>(xp, vp, acc, k, x_off, v_off, p);
    x_off += x_adv;
    v_off += v_adv;
    xp += FLAT ? 0 : step;
    vp += TV ? step : 0;
  }
  if (tiled) {
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) {
      partial[(r * gx + t.x) * gy + t.y] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = gy / 2; s; s /= 2) {
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R; ++r) {
        if (t.y < s) {
          partial[(r * gx + t.x) * gy + t.y] +=
              partial[(r * gx + t.x) * gy + t.y + s];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) {
      acc[r] = partial[(r * gx + t.x) * gy];
    }
  } else {
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R; ++r) {
      acc[r] = simd_sum(acc[r]);
    }
  }
  if (lane_k != 0 || w >= uint(p.M)) {
    return;
  }
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R; ++r) {
    float sum = acc[r];
    for (uint tail = p.K % 4; tail > 0; --tail) {
      const long k = (p.K & ~3) + tail - 1;
      sum += float(X[FLAT ? k * p.lda + w : w * p.lda + k]) *
          float(V[TV ? r * p.ldb + k : k * p.ldb + r]);
    }
    vector_store(C, bias, sum, group.z, w, r, p);
  }
}

// clang-format off
#define INSTANTIATE_KERNEL(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define INSTANTIATE_SIMD(T, BM, BN, BK, SM, SN, SA, SB, TA, TB) \
  INSTANTIATE_KERNEL("gemm_simd_" #T "_" #BM "_" #BN "_" #BK "_" #SM "_" #SN "_" #SA #SB "_" #TA #TB, \
                     gemm_simd, GemmTile<T, BM, BN, BK, SM, SN, SA, SB, TA, TB>)
#define INSTANTIATE_SIMD_LAYOUTS(T, ...) \
  INSTANTIATE_SIMD(T, __VA_ARGS__, 0, 0) \
  INSTANTIATE_SIMD(T, __VA_ARGS__, 0, 1) \
  INSTANTIATE_SIMD(T, __VA_ARGS__, 1, 0) \
  INSTANTIATE_SIMD(T, __VA_ARGS__, 1, 1)
#define INSTANTIATE_SIMD_TYPES(...) \
  INSTANTIATE_SIMD_LAYOUTS(float, __VA_ARGS__) \
  INSTANTIATE_SIMD_LAYOUTS(half, __VA_ARGS__) \
  INSTANTIATE_SIMD_LAYOUTS(bfloat, __VA_ARGS__)
#define INSTANTIATE_SIMD_TILE(BM, BN, SM, SN) \
  INSTANTIATE_SIMD_TYPES(BM, BN, 32, 2, 2, 1, 0) \
  INSTANTIATE_SIMD_TYPES(BM, BN, 16, SM, SN, 1, 1) \
  INSTANTIATE_SIMD_TYPES(BM, BN, 32, SM, SN, 1, 1)

INSTANTIATE_SIMD_TILE(16, 16, 2, 2)
INSTANTIATE_SIMD_TILE(16, 32, 2, 2)
INSTANTIATE_SIMD_TILE(16, 64, 2, 2)
INSTANTIATE_SIMD_TILE(16, 128, 2, 2)
INSTANTIATE_SIMD_TILE(32, 16, 2, 2)
INSTANTIATE_SIMD_TILE(32, 32, 2, 2)
INSTANTIATE_SIMD_TILE(32, 64, 2, 2)
INSTANTIATE_SIMD_TILE(32, 128, 2, 4)
INSTANTIATE_SIMD_TILE(64, 16, 2, 2)
INSTANTIATE_SIMD_TILE(64, 32, 2, 2)
INSTANTIATE_SIMD_TILE(64, 64, 2, 2)
INSTANTIATE_SIMD_TILE(128, 32, 4, 2)
INSTANTIATE_SIMD_LAYOUTS(half, 32, 32, 16, 2, 2, 0, 0)
INSTANTIATE_SIMD_LAYOUTS(bfloat, 32, 32, 16, 2, 2, 0, 0)
INSTANTIATE_SIMD_LAYOUTS(half, 48, 32, 32, 2, 2, 1, 1)
INSTANTIATE_SIMD_LAYOUTS(half, 48, 48, 24, 2, 2, 1, 1)
INSTANTIATE_SIMD_LAYOUTS(float, 48, 48, 16, 2, 2, 1, 1)
INSTANTIATE_SIMD_LAYOUTS(bfloat, 48, 48, 16, 2, 2, 1, 1)
INSTANTIATE_SIMD_LAYOUTS(float, 64, 48, 24, 2, 2, 1, 1)
INSTANTIATE_SIMD_LAYOUTS(bfloat, 64, 48, 24, 2, 2, 1, 1)

#define INSTANTIATE_REDUCE(T) \
  INSTANTIATE_KERNEL("gemm_reduce_" #T "_1", gemm_reduce, T, 1) \
  INSTANTIATE_KERNEL("gemm_reduce_" #T "_4", gemm_reduce, T, 4)

INSTANTIATE_REDUCE(float)
INSTANTIATE_REDUCE(half)
INSTANTIATE_REDUCE(bfloat)

#define INSTANTIATE_VECTOR(T, R, U, FLAT, TV) \
  INSTANTIATE_KERNEL("gemm_vector_" #T "_" #R "_" #U "_" #FLAT #TV, gemm_vector, T, R, U, FLAT, TV)
#define INSTANTIATE_VECTOR_TYPES(...) \
  INSTANTIATE_VECTOR(float, __VA_ARGS__) \
  INSTANTIATE_VECTOR(half, __VA_ARGS__) \
  INSTANTIATE_VECTOR(bfloat, __VA_ARGS__)
#define INSTANTIATE_VECTOR_U(R, U) \
  INSTANTIATE_VECTOR_TYPES(R, U, 0, 0) \
  INSTANTIATE_VECTOR_TYPES(R, U, 0, 1)
#define INSTANTIATE_VECTOR_R(R) \
  INSTANTIATE_VECTOR_U(R, 1) \
  INSTANTIATE_VECTOR_TYPES(R, 1, 1, 0) \
  INSTANTIATE_VECTOR_TYPES(R, 1, 1, 1)

INSTANTIATE_VECTOR_R(1)
INSTANTIATE_VECTOR_R(2)
INSTANTIATE_VECTOR_R(3)
INSTANTIATE_VECTOR_R(4)
INSTANTIATE_VECTOR_R(5)
INSTANTIATE_VECTOR_R(6)
INSTANTIATE_VECTOR_R(7)
INSTANTIATE_VECTOR_R(8)
INSTANTIATE_VECTOR_R(9)
INSTANTIATE_VECTOR_R(10)
INSTANTIATE_VECTOR_R(11)
INSTANTIATE_VECTOR_R(12)
INSTANTIATE_VECTOR_R(13)
INSTANTIATE_VECTOR_R(14)
INSTANTIATE_VECTOR_R(15)
INSTANTIATE_VECTOR_R(16)
INSTANTIATE_VECTOR_U(1, 2)
INSTANTIATE_VECTOR_U(1, 4)
INSTANTIATE_VECTOR_U(1, 8)
INSTANTIATE_VECTOR_U(2, 2)
INSTANTIATE_VECTOR_U(2, 4)
INSTANTIATE_VECTOR_U(2, 8)
INSTANTIATE_VECTOR_U(3, 2)
INSTANTIATE_VECTOR_U(3, 4)
INSTANTIATE_VECTOR_U(3, 8)
INSTANTIATE_VECTOR_U(4, 2)
INSTANTIATE_VECTOR_U(4, 4)
INSTANTIATE_VECTOR_U(5, 2)
INSTANTIATE_VECTOR_U(5, 4)
INSTANTIATE_VECTOR_U(6, 2)
// clang-format on

#if C10_METAL_HAS_MPP
using namespace mpp::tensor_ops;

template <typename T, int UM, int UN, int UK, bool TA, bool TB>
kernel void gemm_mpp(
    device T* A [[buffer(0)]],
    device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    constant GemmParams& p [[buffer(4)]],
    device float* partials [[buffer(5)]],
    threadgroup float* red [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
  using tensor_t = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
  using ext_t = dextents<int32_t, 2>;
  constexpr int TSM = 16 * UM, TSN = 16 * UN, KC = 16 * UK;
  constexpr int KD = KC % 64 == 0 ? 64 : (KC % 32 == 0 ? 32 : 16);
  constexpr int NSUB = KC / KD;
  const int SM = p.simd_m, SN = p.simd_n, KS = p.simd_k, GS = p.splits;
  const int TILE_M = TSM * SM, TILE_N = TSN * SN, gM = p.M, gN = p.N;
  const uint sub_n = tgid.x % p.uber_n, sub_m = tgid.x / p.uber_n % p.uber_m;
  uint uber_n = tgid.y % p.ubers_n, uber_m = tgid.y / p.ubers_n, bidx = tgid.z;
  uint gsplit = 0;
  if (p.linear) {
    const uint t = tgid.x / (p.uber_n * p.uber_m);
    uber_m = p.raster_m ? t % p.ubers_m : t / p.ubers_n % p.ubers_m;
    uber_n = p.raster_m ? t / p.ubers_m % p.ubers_n : t % p.ubers_n;
    gsplit = t / (p.ubers_m * p.ubers_n) % GS;
    bidx = t / (p.ubers_m * p.ubers_n) / GS;
  }
  const int n_tile = uber_n * p.uber_n + sub_n;
  const int m_tile = uber_m * p.uber_m + sub_m;
  if (n_tile * TILE_N >= gN || m_tile * TILE_M >= gM) {
    return;
  }
  A += bidx * p.batch_a;
  B += bidx * p.batch_b;
  C += bidx * p.batch_c;
  const uint ks = sgid / (SM * SN), pos = sgid % (SM * SN);
  int m_off = m_tile * TILE_M + (pos % SM) * TSM;
  int n_off = n_tile * TILE_N + (pos / SM) * TSN;
  int gK = p.K;
  if (GS > 1) {
    const int k_start = gsplit * (p.K / GS);
    gK = gsplit == uint(GS - 1) ? p.K - k_start : p.K / GS;
    A += TA ? k_start * p.lda : k_start;
    B += TB ? k_start : k_start * p.ldb;
  }
  const bool active = m_off < gM && n_off < gN;
  if (GS == 1) {
    m_off = m_off + TSM > gM && gM >= TSM ? gM - TSM : m_off;
    n_off = n_off + TSN > gN && gN >= TSN ? gN - TSN : n_off;
  }
  const bool inside = m_off + TSM <= gM && n_off + TSN <= gN;
  const int lda = p.lda, ldb = p.ldb, ldc = p.ldc;
  const array<int32_t, 2> sA = {1, lda}, sB = {1, ldb}, sC = {1, ldc};
  const auto a_ext = [](int m, int k) {
    return TA ? ext_t(m, k) : ext_t(k, m);
  };
  const auto b_ext = [](int n, int k) {
    return TB ? ext_t(k, n) : ext_t(n, k);
  };
  tensor_t tA(A, a_ext(gM, gK), sA), tB(B, b_ext(gN, gK), sB);
  const auto a_sub = [&](int k0) {
    return tA.template slice<TA ? TSM : KD, TA ? KD : TSM>(
        TA ? m_off : k0, TA ? k0 : m_off);
  };
  const auto b_sub = [&](int k0) {
    return tB.template slice<TB ? KD : TSN, TB ? TSN : KD>(
        TB ? k0 : n_off, TB ? n_off : k0);
  };
  constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate;
  constexpr auto desc = matmul2d_descriptor(TSM, TSN, KD, TA, TB, true, mode);
  matmul2d<desc, execution_simdgroup> op;
  using a_t = decltype(a_sub(0));
  using b_t = decltype(b_sub(0));
  auto cT = op.template get_destination_cooperative_tensor<a_t, b_t, float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cT[i] = 0;
  }
  auto cA0 = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto cA1 = cA0, cA2 = cA0;
  const int k_iters = (gK + KC * KS - 1) / (KC * KS);
  const int chunk = p.barrier ? p.barrier : k_iters;
  for (int it0 = 0; it0 < k_iters; it0 += chunk) {
    if (p.barrier) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int it = it0; it < min(it0 + chunk, k_iters); ++it) {
      const int k0 = (it * KS + ks) * KC;
      if (active && inside && k0 + KC <= gK) {
        cA0.load(a_sub(k0));
        if constexpr (NSUB > 1) {
          cA1.load(a_sub(k0 + KD));
        }
        if constexpr (NSUB > 2) {
          cA2.load(a_sub(k0 + 2 * KD));
        }
        auto b0 = b_sub(k0);
        op.run(cA0, b0, cT);
        if constexpr (NSUB > 1) {
          auto b1 = b_sub(k0 + KD);
          op.run(cA1, b1, cT);
        }
        if constexpr (NSUB > 2) {
          auto b2 = b_sub(k0 + 2 * KD);
          op.run(cA2, b2, cT);
        }
      } else if (active && k0 < gK) {
        const int kend = min(k0 + KC, gK);
        for (int kb = k0; kb < kend; kb += KD) {
          const int ke = min(kb + KD, kend);
          if (inside) {
            tensor_t vA(A, a_ext(gM, ke), sA), vB(B, b_ext(gN, ke), sB);
            auto sa = vA.template slice<
                TA ? TSM : dynamic_extent,
                TA ? dynamic_extent : TSM>(TA ? m_off : kb, TA ? kb : m_off);
            auto sb = vB.template slice<
                TB ? dynamic_extent : TSN,
                TB ? TSN : dynamic_extent>(TB ? kb : n_off, TB ? n_off : kb);
            op.run(sa, sb, cT);
          } else {
            const long a_off =
                TA ? m_off + long(kb) * lda : kb + long(m_off) * lda;
            const long b_off =
                TB ? kb + long(n_off) * ldb : n_off + long(kb) * ldb;
            tensor_t vA(A + a_off, a_ext(gM - m_off, ke - kb), sA);
            tensor_t vB(B + b_off, b_ext(gN - n_off, ke - kb), sB);
            op.run(vA, vB, cT);
          }
        }
      }
    }
  }
  if (KS > 1) {
    for (int w = 0; w < KS - 1; w += p.slots) {
      const int g = int(ks) - 1 - w;
      if (ks != 0 && active && g >= 0 && g < p.slots) {
        const auto slot = red + (g * SM * SN + pos) * TSM * TSN;
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
          const auto idx = cT.get_multidimensional_index(i);
          slot[idx[1] * TSN + idx[0]] = cT[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks == 0 && active) {
        for (int s = 0; s < min(p.slots, KS - 1 - w); ++s) {
          const auto slot = red + (s * SM * SN + pos) * TSM * TSN;
          for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            const auto idx = cT.get_multidimensional_index(i);
            cT[i] += slot[idx[1] * TSN + idx[0]];
          }
        }
      }
      if (w + p.slots < KS - 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (ks != 0 || !active) {
      return;
    }
  } else if (!active) {
    return;
  }
  if (GS > 1) {
    const auto plane = partials + (ulong(bidx) * GS + gsplit) * gM * gN;
    tensor<device float, ext_t, tensor_inline> vP(
        plane + n_off + long(m_off) * gN,
        ext_t(gN - n_off, gM - m_off),
        array<int32_t, 2>{1, gN});
    cT.store(vP);
    return;
  }
  auto cO = op.template get_destination_cooperative_tensor<a_t, b_t, T>();
  if (p.has_bias) {
    for (uint16_t e = 0; e < cT.get_capacity(); ++e) {
      const auto idx = cT.get_multidimensional_index(e);
      const int r = m_off + idx[1], c = n_off + idx[0];
      float v = cT[e] * p.alpha;
      if (r < gM && c < gN) {
        v +=
            float(bias[bidx * p.bias_b + r * p.bias_r + c * p.bias_c]) * p.beta;
      }
      cO[e] = T(v);
    }
  } else {
    for (uint16_t e = 0; e < cT.get_capacity(); ++e) {
      cO[e] = T(cT[e] * p.alpha);
    }
  }
  if (inside) {
    tensor_t tC(C, ext_t(gN, gM), sC);
    auto mC = tC.template slice<TSN, TSM>(n_off, m_off);
    cO.store(mC);
  } else {
    tensor_t vC(
        C + n_off + long(m_off) * ldc, ext_t(gN - n_off, gM - m_off), sC);
    cO.store(vC);
  }
}
// clang-format off
#define INSTANTIATE_MPP(T, UM, UN, UK, TA, TB) \
  INSTANTIATE_KERNEL("gemm_mpp_" #T "_" #UM "_" #UN "_" #UK "_" #TA #TB, gemm_mpp, T, UM, UN, UK, TA, TB)
#define INSTANTIATE_MPP_LAYOUTS(T, ...) \
  INSTANTIATE_MPP(T, __VA_ARGS__, 0, 0) \
  INSTANTIATE_MPP(T, __VA_ARGS__, 0, 1) \
  INSTANTIATE_MPP(T, __VA_ARGS__, 1, 0) \
  INSTANTIATE_MPP(T, __VA_ARGS__, 1, 1)
#define INSTANTIATE_MPP_TILE(UM, UN) \
  INSTANTIATE_MPP_LAYOUTS(half, UM, UN, 1) \
  INSTANTIATE_MPP_LAYOUTS(half, UM, UN, 2) \
  INSTANTIATE_MPP_LAYOUTS(half, UM, UN, 3) \
  INSTANTIATE_MPP_LAYOUTS(half, UM, UN, 4) \
  INSTANTIATE_MPP_LAYOUTS(bfloat, UM, UN, 1) \
  INSTANTIATE_MPP_LAYOUTS(bfloat, UM, UN, 2) \
  INSTANTIATE_MPP_LAYOUTS(bfloat, UM, UN, 3) \
  INSTANTIATE_MPP_LAYOUTS(bfloat, UM, UN, 4)

INSTANTIATE_MPP_TILE(1, 1)
INSTANTIATE_MPP_TILE(1, 2)
INSTANTIATE_MPP_TILE(1, 4)
INSTANTIATE_MPP_TILE(2, 1)
INSTANTIATE_MPP_TILE(2, 2)
INSTANTIATE_MPP_TILE(2, 4)
// clang-format on
#endif
