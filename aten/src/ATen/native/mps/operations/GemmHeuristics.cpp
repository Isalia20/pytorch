#include <ATen/ceil_div.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/operations/GemmHeuristics.h>
#include <c10/util/llvmMathExtras.h>
#include <fmt/format.h>

#include <algorithm>
#include <tuple>

namespace at::native::mps {

GemvConfig GemvPolicy::clamp_vec(GemvConfig cfg, int64_t align) {
  while (cfg.vec > 1 && (align & (cfg.vec - 1))) {
    cfg.vec >>= 1;
  }
  return cfg;
}

namespace {

GemvConfig t2d(int nsimd, int kq) {
  GemvConfig cfg{nsimd, 1};
  cfg.kq = kq;
  cfg.kernel = GemvKernel::T2D;
  return cfg;
}

} // namespace

// One profile for all GPU generations: sweep-fitted on M5 Pro and biased
// toward oversubscription, since on unmeasured hardware extra simdgroups only
// add reduction overhead while missing ones leave memory latency exposed.
// nsimd_min and nsimd_max must match the built MB_GEMV_* kernels, since the
// snap assumes contiguous powers of two.
GemvTuning gemv_tuning(c10::ScalarType dt) {
  GemvTuning t{
      .vec = 2,
      .nsimd_min = 16,
      .nsimd_max = 32,
      .min_k_per_simd = 32,
      .waves = 896,
      .small_outlen = 1024,
      .t2d_kq = 8,
      .scalar_cols_k = 0,
      .nt_nsimd_lo = 4,
      .nt_nsimd_hi = 8,
      .nt_vec = 8};
  if (dt == at::kFloat) {
    // fp32 moves twice the bytes per element, so it saturates the bus with
    // far fewer waves and rewards backing off the K-split on wide outputs.
    t.nsimd_min = 4;
    t.waves = 56;
    t.t2d_kq = 4;
    t.scalar_cols_k = 16384; // long fp32 reductions prefer scalar columns
    t.nt_nsimd_hi = 16;
    t.nt_vec = 4;
  }
  return t;
}

GemvPolicy::GemvPolicy(uint32_t cores) : cores_(cores) {}

GemvPolicy GemvPolicy::current() {
  static const GemvPolicy policy(at::mps::MPSDevice::getInstance()->getCoreCount());
  return policy;
}

GemvConfig GemvPolicy::pick_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  const GemvTuning t = gemv_tuning(dt);

  // Small matrices sit in cache, so let t2d stream them.
  if (outlen <= t.small_outlen) {
    return t2d(16, t.t2d_kq);
  }
  // Very long fp32 reductions prefer scalar columns.
  if (t.scalar_cols_k && K >= t.scalar_cols_k) {
    return {32, 1};
  }

  // Aim for about waves simdgroups per core. The output gives blocks of them,
  // the rest comes from splitting K.
  const int64_t block_n = int64_t{32} * t.vec;
  const int64_t target = int64_t(cores_ > 0 ? cores_ : 10) * t.waves;
  const int64_t narrow = target * block_n / t.nsimd_max;
  const int64_t wide = target * block_n / t.nsimd_min;

  // Narrow output splits K the most, wide output the least.
  int nsimd = outlen <= narrow ? t.nsimd_max : outlen <= wide ? t.nsimd_max / 2 : t.nsimd_min;

  // Keep enough K on each simdgroup to be worth the split.
  int k_cap = static_cast<int>(K / t.min_k_per_simd);
  k_cap = k_cap < t.nsimd_min ? t.nsimd_min : (k_cap > t.nsimd_max ? t.nsimd_max : k_cap);
  if (nsimd > k_cap) {
    nsimd = k_cap;
  }

  // Round down to a built nsimd (power of two in range).
  int chosen = t.nsimd_min;
  while (chosen * 2 <= nsimd && chosen < t.nsimd_max) {
    chosen *= 2;
  }
  return clamp_vec({chosen, t.vec}, align);
}

// gemv_nt reduces one whole row per simdgroup, so occupancy is outlen
// simdgroups no matter what nsimd is; nsimd only sets threadgroup granularity
// and vec the K-loop load width.
GemvConfig GemvPolicy::pick_nt(c10::ScalarType dt, int64_t outlen, int64_t /*K*/, int64_t align) const {
  const GemvTuning t = gemv_tuning(dt);
  if (dt == at::kFloat) {
    return clamp_vec({outlen >= 2048 ? t.nt_nsimd_lo : t.nt_nsimd_hi, t.nt_vec}, align);
  }
  return clamp_vec({outlen >= 8192 ? t.nt_nsimd_hi : t.nt_nsimd_lo, t.nt_vec}, align);
}

namespace {

struct GemmDevice {
  int64_t generation, cores;
  bool macos27, mpp;
};

const GemmDevice& gemm_device() {
  using namespace at::mps;
  static const GemmDevice device = [] {
    const int64_t count = MPSDevice::getInstance()->getCoreCount();
    const int64_t cores = count > 0 ? count : 10;
    int64_t generation = 11;
    if (is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS)) {
      generation = cores < 11 ? 22 : 23;
    } else if (is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS)) {
      generation = 18;
    } else if (is_apple_family_or_newer(AppleGPUFamily::APPLE_8_PLUS)) {
      generation = cores < 10 ? 16 : 17;
    }
    return GemmDevice{generation, cores, is_macos_at_least(MacOSVersion::MACOS_27_0), has_mpp()};
  }();
  return device;
}

const char* gemm_type(c10::ScalarType dt) {
  return dt == at::kFloat ? "float" : dt == at::kHalf ? "half" : "bfloat";
}

std::tuple<int64_t, int64_t, int64_t, int64_t> mpp_ultra_tile(int64_t M, int64_t N, int64_t K, bool tb, int64_t cores) {
  const int64_t k_thr = tb ? 6000 : 12288;
  int64_t sm = 1, sn = 2;
  auto tiles_m = at::ceil_div(M, int64_t(32)), tiles_n = at::ceil_div(N, int64_t(64));
  while (tiles_m * tiles_n > cores * 4 && sm * sn * 2 <= 4 && std::max(tiles_m, tiles_n) >= 64) {
    (tiles_n < tiles_m ? sm : sn) *= 2;
    tiles_m = at::ceil_div(M, sm * 32);
    tiles_n = at::ceil_div(N, sn * 32);
  }
  const auto tiles = tiles_m * tiles_n;
  int64_t splits = K <= k_thr ? 1 : (K + 1) / 2 <= k_thr ? 2 : (K + 3) / 4 <= k_thr ? 4 : 8;
  if (splits < 8 && tiles * splits < cores && at::ceil_div(K, splits * 2) >= 512) {
    splits *= 2;
    while (splits <= 4 && tiles * splits < cores && at::ceil_div(K, splits * 2) >= 512) {
      splits *= 2;
    }
  }
  return {sm, sn, splits, at::ceil_div(K, splits)};
}

GemmPlan mpp_plan(c10::ScalarType dt, int64_t M, int64_t N, int64_t K, int64_t batch, bool ta, bool tb) {
  const auto cores = gemm_device().cores;
  int64_t uk = 0, um = 2, un = 2, sm = 1, sn = 2, ks = 1, gs = 1, uber_m = 2, uber_n = 2, barrier = 4;
  bool linear = true, raster_m = false;
  const auto shrink = [&](int64_t splits) {
    auto tiles_m = at::ceil_div(M, um * sm * 16), tiles_n = at::ceil_div(N, un * sn * 16);
    for (bool shrink_m = false; tiles_m * splits * tiles_n < cores; shrink_m = !shrink_m) {
      auto& unroll = shrink_m ? um : un;
      auto& simd = shrink_m ? sm : sn;
      if (unroll > 1) {
        --unroll;
      } else if (simd > 1) {
        --simd;
      }
      if (um == 1 && un == 1 && sm == 1 && sn == 1) {
        return std::pair{at::ceil_div(M, int64_t(16)), at::ceil_div(N, int64_t(16))};
      }
      tiles_m = at::ceil_div(M, um * sm * 16);
      tiles_n = at::ceil_div(N, un * sn * 16);
    }
    return std::pair{tiles_m, tiles_n};
  };
  const bool big = M > 16 && N > 16;
  if (cores <= 12 || (cores < 60 && big && ((K > 2048 && K % 32) || (ta && !tb && (K | M | N) % 32)))) {
    const auto k12 = K >> 12;
    const auto m_log = std::clamp<int64_t>(c10::llvm::Log2_64_Ceil(M), 4, k12 < 3 ? 7 : 8);
    const auto n_log = std::clamp<int64_t>(c10::llvm::Log2_64_Ceil(N), 4, k12 < 3 ? 6 : 7);
    const auto sm_log = std::min<int64_t>(k12 < 3 ? 2 : 3, m_log - 4);
    const auto sn_log = k12 < 3 ? 0 : std::min<int64_t>(n_log - 4, 2);
    uk = std::clamp<int64_t>((K + 15) >> 4, 1, K > 4096 ? 3 : 2);
    um = (int64_t(1) << (m_log - sm_log)) >> 4;
    un = (int64_t(1) << (n_log - sn_log)) >> 4;
    sm = int64_t(1) << sm_log;
    sn = int64_t(1) << sn_log;
    uber_n = 1;
    barrier = 0;
    linear = k12 > 2;
  } else if (!big) {
    const bool n_le_m = N <= M;
    const auto wide = std::max(M, N);
    uk = std::clamp<int64_t>((K + 15) >> 4, 1, 4);
    raster_m = M <= N;
    um = n_le_m ? 2 : 1;
    sn = un = n_le_m ? 1 : 2;
    if ((cores << 7) <= wide && (n_le_m ? ta : !tb)) {
      ks = std::min<int64_t>(K >= 16384 || wide > 16384 ? 2 : 1, at::ceil_div(K, uk * 16));
      if (M < N) {
        sn = std::clamp<int64_t>((N + 15) >> 4, 1, 4);
        un = at::ceil_div(N, sn * 16) > 1 ? 2 : 1;
        raster_m = true;
      } else {
        sm = std::clamp<int64_t>((M + 15) >> 4, 1, 4);
        um = at::ceil_div(M, sm * 16) > 1 ? 2 : 1;
      }
    } else {
      ks = std::clamp<int64_t>(at::ceil_div(K, uk * 16), 1, 8);
      if (M < N) {
        un = 1;
        raster_m = true;
      } else {
        sm = 2;
        um = 1;
      }
    }
    const auto [tiles_m, tiles_n] = shrink(1);
    uber_m = std::min<int64_t>(n_le_m ? 2 : 1, tiles_m);
    uber_n = std::min<int64_t>(n_le_m ? 1 : 2, tiles_n);
  } else {
    raster_m = M <= N || !ta;
    int64_t k_per = K;
    if (cores >= 60) {
      std::tie(sm, sn, gs, k_per) = mpp_ultra_tile(M, N, K, tb, cores);
    }
    uk = std::clamp<int64_t>((k_per + 15) >> 4, 1, 4);
    const auto [tiles_m, tiles_n] = shrink(gs);
    uber_m = std::min<int64_t>(2, tiles_m);
    uber_n = std::min<int64_t>(2, tiles_n);
  }
  const auto ubers_m = at::ceil_div(at::ceil_div(M, um * sm * 16), uber_m);
  const auto ubers_n = at::ceil_div(at::ceil_div(N, un * sn * 16), uber_n);
  const auto tile = sm * sn * um * un * 256;
  GemmPlan plan{};
  auto& p = plan.params;
  p.splits = gs;
  p.simd_m = sm;
  p.simd_n = sn;
  p.simd_k = ks;
  p.slots = ks > 1 ? std::clamp<int64_t>(4096 / tile, 1, ks - 1) : 0;
  p.uber_m = uber_m;
  p.uber_n = uber_n;
  p.ubers_m = ubers_m;
  p.ubers_n = ubers_n;
  p.barrier = barrier;
  p.raster_m = raster_m;
  p.linear = linear;
  plan.kernel = fmt::format("gemm_mpp_{}_{}_{}_{}_{}{}", gemm_type(dt), um, un, uk, int(ta), int(tb));
  const uint64_t subtiles = uber_m * uber_n;
  plan.groups = linear ? std::array<uint64_t, 3>{subtiles * ubers_m * ubers_n * batch * gs, 1, 1}
                       : std::array<uint64_t, 3>{subtiles, uint64_t(ubers_m * ubers_n), uint64_t(batch)};
  plan.threads = {uint64_t(32 * sm * sn * ks), 1, 1};
  plan.threadgroup_memory = std::max<int64_t>(16, p.slots * tile * 4);
  return plan;
}

std::tuple<int64_t, int64_t, int64_t> simd_square_tiles(int64_t size, int64_t cores, bool half) {
  const auto pad = [size](int64_t t) { return (t - size % t) % t; };
  const auto p32 = pad(32), p48 = pad(48), p64 = pad(64);
  if (cores > 64) {
    if (half) {
      return !p48 || p48 <= p32 ? std::tuple{48, 48, 24} : std::tuple{48, 32, 32};
    }
    return p64 + p32 <= 2 * p48 ? std::tuple{64, 32, 16} : std::tuple{48, 48, 16};
  }
  if (half) {
    const auto waste64 = ((p64 * (size + 31)) >> 5) + ((p32 * (size + 63)) >> 6);
    return waste64 <= 2 * ((size + 47) * p48 / 48) ? std::tuple{64, 32, 32} : std::tuple{48, 48, 24};
  }
  if (!p48 || p32 >= p48) {
    return {64, 48, 24};
  }
  const auto groups = at::ceil_div(size, int64_t(64));
  const bool narrow = float(groups * groups) / float(cores) <= 12.f ? p32 <= p64 : p64 > p32;
  return narrow ? std::tuple{64, 32, 32} : std::tuple{64, 64, 16};
}

GemmPlan simd_plan(c10::ScalarType dt, int64_t M, int64_t N, int64_t K, int64_t batch, bool ta, bool tb) {
  const auto [gen, cores, macos27, mpp] = gemm_device();
  const auto area = float(M * N);
  const auto over = [&](int64_t chunk) { return float(chunk) / area > 0.1f; };
  int64_t splits = 1, chunk = K;
  bool split = over(K) && K > 2048;
  if (!macos27) {
    while (split && over(chunk)) {
      chunk /= 2;
    }
  } else {
    while (split && over(chunk)) {
      chunk = at::ceil_div(K, splits *= 2);
    }
    if (chunk > 8192) {
      while (chunk > 8192) {
        chunk = at::ceil_div(K, splits *= 2);
      }
      split = true;
    } else if (gen == 17 && cores > 64 && N <= 256 && K == 2048) {
      chunk = N > 127 ? 512 : 256;
      split = true;
    }
  }
  if (split) {
    const int64_t quantum = gen >= 18 ? 32 : 16;
    chunk = at::ceil_div(std::max<int64_t>(chunk, 1), quantum) * quantum;
    splits = at::ceil_div(K, chunk);
  } else {
    splits = 1;
    chunk = K;
  }
  int64_t bm = M > 32 ? 64 : 32, bn = 64, bk = 16, sm = 2, sn = 2;
  if (M <= 32 && N >= 512) {
    bn = 128;
    sn = 4;
  } else if (N <= 32) {
    bm = M >= 512 ? 128 : bm;
    bn = 32;
    sm = M >= 512 ? 4 : 2;
  }
  const bool ultra = macos27 && gen == 17 && cores > 64;
  if (ultra) {
    bk = K <= 10240 ? 16 : 32;
    if (N <= 256 && K == 2048) {
      bm = N == 256 ? 64 : 32;
      bn = bk = 32;
      sm = sn = 2;
    }
  }
  const bool cubic = M == N && N == K;
  if (cubic) {
    std::tie(bm, bn, bk) = simd_square_tiles(M, cores, dt == at::kHalf);
  } else if (splits * splits * M * N * batch <= 4 * cores * bm * bn) {
    const auto work = splits * splits * M * N * batch;
    bm /= 2;
    for (bool shrink_n = true; work <= 4 * cores * bm * bn && (bm > 16 || bn > 16); shrink_n = !shrink_n) {
      if (bn > 16 && shrink_n) {
        bn /= 2;
      } else if (bm > 16) {
        bm /= 2;
      }
    }
    sm = sn = 2;
  }
  bool sa = true;
  if (gen >= 18) {
    sm = sn = 2;
    bk = 32;
    if (cubic) {
      bm = 32;
      bn = dt == at::kFloat ? 64 : 32;
      bk = dt == at::kFloat ? 32 : 16;
      sa = dt == at::kFloat;
    }
  }
  if (splits > 1) {
    chunk = at::ceil_div(chunk, bk) * bk;
    splits = at::ceil_div(K, chunk);
  }
  GemmPlan plan{};
  auto& p = plan.params;
  p.splits = splits;
  p.k_chunk = splits > 1 ? chunk : K;
  p.linear = ultra;
  plan.kernel = fmt::format("gemm_simd_{}_{}_{}_{}_{}_{}_{}{}_{}{}", gemm_type(dt), bm, bn, bk, sm, sn, int(sa),
                            int(gen < 18), int(ta), int(tb));
  const uint64_t gx = at::ceil_div(N, bn), gy = at::ceil_div(M, bm), gz = splits * batch;
  plan.groups = ultra ? std::array<uint64_t, 3>{gx * gy * gz, 1, 1} : std::array<uint64_t, 3>{gx, gy, gz};
  plan.threads = {uint64_t(32 * sm * sn), 1, 1};
  return plan;
}

GemmPlan vector_plan(c10::ScalarType dt, int64_t M, int64_t N, int64_t K, int64_t batch, bool ta, bool tb) {
  const auto [gen, cores, macos27, mpp] = gemm_device();
  const auto R = std::min(M, N), W = std::max(M, N);
  const bool swap = M < N, flat = swap ? !tb : ta, tv = swap ? !ta : tb;
  int64_t unroll = 1;
  if (K >= 2048) {
    unroll = R < 4 ? 8 : R < 6 ? 4 : R == 6 ? 2 : 1;
    if (K % 32 == 0 && K % (unroll * 128)) {
      const auto q = K >> 7;
      unroll = std::min(unroll, q & -q);
    }
  }
  int64_t gx = 32, gy = 1, groups_x = 1, groups_y = W, memory = 0;
  if (R < 8 && flat) {
    gx = W < cores * 8 ? 1 : W < cores * 16 ? 2 : W < cores * 32 ? 4 : 8;
    gy = 32 / gx;
    groups_x = at::ceil_div(W, gx * 4);
    groups_y = 1;
    const auto coverage = cores * (macos27 && K <= cores * 256 ? 32 : 16);
    while (groups_x * batch * (gx * gy / 32) < coverage && gx * gy * 2 / 32 <= std::min<int64_t>(16, 64 / R) &&
           gy * 2 < K) {
      gy *= 2;
    }
    memory = R * gx * gy * 16;
    unroll = 1;
  } else if (swap && flat) {
    gx = gy = 16;
    groups_y = at::ceil_div(W, int64_t(16));
    memory = R * 1024;
  }
  const auto k_threads = gy > 1 ? gy : gx;
  GemmPlan plan{};
  auto& p = plan.params;
  p.gx = gx;
  p.gy = gy;
  p.groups = at::ceil_div(K & ~int64_t(3), k_threads * 4);
  p.full_groups = K / (k_threads * 4 * unroll) * unroll;
  p.swap = swap;
  plan.kernel = fmt::format("gemm_vector_{}_{}_{}_{}{}", gemm_type(dt), R, unroll, int(flat), int(tv));
  plan.groups = {uint64_t(groups_x), uint64_t(groups_y), uint64_t(batch)};
  plan.threads = {uint64_t(gx), uint64_t(gy), 1};
  plan.threadgroup_memory = std::max<int64_t>(16, memory);
  return plan;
}

} // namespace

GemmPlan gemm_plan(c10::ScalarType dt, int64_t M, int64_t N, int64_t K, int64_t batch, bool ta, bool tb, int64_t lda,
                   int64_t ldb, int64_t ldc) {
  const auto [gen, cores, macos27, mpp] = gemm_device();
  const bool a18 = mpp && gen >= 22 && dt != at::kFloat;
  const auto R = std::min(M, N);
  const auto thr = M * N * (dt == at::kFloat ? 4 : 3) >= cores * 4096 ? 9 : 15;
  bool vector = std::max(M, N) <= 128 && R <= 16 && tb && !ta && (a18 || macos27);
  if (a18) {
    vector = vector || N <= thr || (M <= thr && N <= 128);
  } else if (!vector && !(gen >= 22 && cores > 10 && !macos27 && (M > 4 || K > 4096 || (M >= N ? ta : !tb)))) {
    vector = R <= thr;
  }
  if (vector && K * std::max(lda, ldb) < (int64_t(1) << 32)) {
    return vector_plan(dt, M, N, K, batch, ta, tb);
  }
  constexpr int64_t i32 = int64_t(1) << 31;
  if (a18 && std::max(M, K) * lda < i32 && std::max(K, N) * ldb < i32 && M * ldc < i32) {
    return mpp_plan(dt, M, N, K, batch, ta, tb);
  }
  return simd_plan(dt, M, N, K, batch, ta, tb);
}

} // namespace at::native::mps
