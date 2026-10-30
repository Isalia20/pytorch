// Native convolution kernels and their shared layout transforms.
#include <ATen/native/mps/kernels/Convolution.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define INSTANTIATE_CONV2D(NAME, ...) \
  template [[host_name(NAME)]] [[kernel]] decltype(__VA_ARGS__) __VA_ARGS__

// Tiled NCHW -> NHWC transpose, with optional padding and unfolded tap rows.
// Conv3d uses the same transform with its spatial dimensions flattened.
struct NchwToNhwcGeometry {
  int4 dims; // C, H, W, padded W
  int2 pad;
  int4 rows; // kH, sH, dH, output rows

  int input_positions() const {
    return dims.y * dims.z;
  }

  int output_positions() const {
    return rows.w * dims.w * rows.x;
  }

  int input_position(int position) const {
    const int kh = position % rows.x;
    const int w = position / rows.x % dims.w - pad.y;
    const int h = position / rows.x / dims.w * rows.y + kh * rows.z - pad.x;
    return h >= 0 && h < dims.y && w >= 0 && w < dims.z ? h * dims.z + w : -1;
  }
};

template <typename T, int TC, int TX, int NTH, bool WIDE>
void nchw_to_nhwc_load(
    device const T* s,
    threadgroup T (&tile)[TC][TX + 1],
    thread const NchwToNhwcGeometry& p,
    int2 origin,
    uint tid,
    true_type,
    bool_constant<WIDE>) {
  const int C = p.dims.x, X = p.input_positions();
  const int c0 = origin.x, x0 = origin.y;
  for (int i = tid; i < TC * (TX / 2); i += NTH) {
    const int x = (i % (TX / 2)) * 2, c = i / (TX / 2);
    const int gc = c0 + c, gx = x0 + x;
    if (gc >= C || gx >= X) {
      continue;
    }
    const int64_t off = (int64_t)gc * X + gx;
    if (gx + 1 < X) {
      vec<T, 2> v = *(device const vec<T, 2>*)(s + off);
      tile[c][x] = v.x;
      tile[c][x + 1] = v.y;
    } else {
      tile[c][x] = s[off];
    }
  }
}

template <typename T, int TC, int TX, int NTH>
void nchw_to_nhwc_load(
    device const T* s,
    threadgroup T (&tile)[TC][TX + 1],
    thread const NchwToNhwcGeometry& p,
    int2 origin,
    uint tid,
    false_type,
    true_type) {
  const int C = p.dims.x, X = p.input_positions(), Xp = p.output_positions();
  const int c0 = origin.x, x0 = origin.y;
  for (int x = tid; x < TX && x0 + x < Xp; x += NTH) {
    const int position = p.input_position(x0 + x);
    for (int c = 0; c < TC && c0 + c < C; ++c) {
      tile[c][x] = position >= 0 ? s[(int64_t)(c0 + c) * X + position] : T(0);
    }
  }
}

template <typename T, int TC, int TX, int NTH>
void nchw_to_nhwc_load(
    device const T* s,
    threadgroup T (&tile)[TC][TX + 1],
    thread const NchwToNhwcGeometry& p,
    int2 origin,
    uint tid,
    false_type,
    false_type) {
  const int C = p.dims.x, X = p.input_positions(), Xp = p.output_positions();
  const int c0 = origin.x, x0 = origin.y;
  for (int i = tid; i < TC * TX; i += NTH) {
    const int x = i % TX, c = i / TX;
    const int gc = c0 + c, gx = x0 + x;
    if (gc >= C || gx >= Xp) {
      continue;
    }
    const int position = p.input_position(gx);
    tile[c][x] = position >= 0 ? s[(int64_t)gc * X + position] : T(0);
  }
}

template <typename T, int TC, int TX, int NTH>
void nchw_to_nhwc_store(
    threadgroup const T (&tile)[TC][TX + 1],
    device T* d,
    thread const NchwToNhwcGeometry& p,
    int2 origin,
    uint tid,
    true_type) {
  const int C = p.dims.x, Xp = p.output_positions();
  const int c0 = origin.x, x0 = origin.y;
  for (int i = tid; i < (TC / 2) * TX; i += NTH) {
    const int c = (i % (TC / 2)) * 2, x = i / (TC / 2);
    const int gc = c0 + c, gx = x0 + x;
    if (gx >= Xp || gc >= C) {
      continue;
    }
    const int64_t off = (int64_t)gx * C + gc;
    if (gc + 1 < C) {
      *(device vec<T, 2>*)(d + off) = vec<T, 2>(tile[c][x], tile[c + 1][x]);
    } else {
      d[off] = tile[c][x];
    }
  }
}

template <typename T, int TC, int TX, int NTH>
void nchw_to_nhwc_store(
    threadgroup const T (&tile)[TC][TX + 1],
    device T* d,
    thread const NchwToNhwcGeometry& p,
    int2 origin,
    uint tid,
    false_type) {
  const int C = p.dims.x, Xp = p.output_positions();
  const int c0 = origin.x, x0 = origin.y;
  for (int i = tid; i < TC * TX; i += NTH) {
    const int c = i % TC, x = i / TC;
    const int gc = c0 + c, gx = x0 + x;
    if (gc < C && gx < Xp) {
      d[(int64_t)gx * C + gc] = tile[c][x];
    }
  }
}

template <typename T, int TC, int TX, int NTH, bool VECR, bool VECW>
[[max_total_threads_per_threadgroup(NTH)]]
kernel void nchw_to_nhwc(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant int4& dims [[buffer(2)]],
    constant int2& pad [[buffer(3)]],
    constant int4& rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup T tile[TC][TX + 1];
  const NchwToNhwcGeometry p{dims, pad, rows};
  const int2 origin(int(tgid.y) * TC, int(tgid.x) * TX);
  device const T* s = src + int64_t(tgid.z) * dims.x * p.input_positions();
  device T* d = dst + int64_t(tgid.z) * dims.x * p.output_positions();
  nchw_to_nhwc_load<T, TC, TX, NTH>(
      s,
      tile,
      p,
      origin,
      tid,
      bool_constant<VECR>{},
      bool_constant<TX * 2 >= NTH>{});
  threadgroup_barrier(mem_flags::mem_threadgroup);
  nchw_to_nhwc_store<T, TC, TX, NTH>(
      tile, d, p, origin, tid, bool_constant<VECW>{});
}

// Implicit-GEMM conv3d on simdgroup_matrix per (batch, out-depth) plane; k
// walks DHWIO order, geometry is runtime state, fp32 accumulation. IDX = long
// widens spatial indices and tap coordinates beyond int32.
template <typename T, typename IDX, int BM, int BN, int BK, int WM, int WN>
kernel void conv3d_simd(
    device T* act [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv3DParams& gP [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constant Conv2DParams& p = gP.conv2d;
  constexpr int TGP = WM * WN * c10::metal::simdgroup_size;
  constexpr int WT_M = BM / WM, WT_N = BN / WN;
  constexpr int TM = WT_M / 8, TN = WT_N / 8;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  static_assert(BM % (8 * WM) == 0 && BN % (8 * WN) == 0 && BK % 8 == 0, "");
  static_assert(TGP % BK == 0 && TGP % BN == 0, "cooperative load mappings");

  threadgroup T As[BM * LDA];
  threadgroup T Bs[BK * LDB];

  const int tid = int(sgid) * c10::metal::simdgroup_size + int(lane);
  const int n_tiles = p.C_out_per_group / BN + (p.C_out_per_group % BN != 0);
  const int g = int(tgid.x) / n_tiles;
  const int n_block = (int(tgid.x) % n_tiles) * BN;
  const IDX m_block = IDX(tgid.y) * BM;
  const int dd = int(tgid.z) % gP.outD;
  const int nb = int(tgid.z) / gP.outD;
  const int c0 = g * p.C_in_per_group;
  const int o0 = g * p.C_out_per_group;

  const IDX M = IDX(p.outH) * p.outW;
  const int K = gP.kD * p.kH * p.kW * p.C_in_per_group;
  const int64_t act_nb = (int64_t)nb * gP.D * p.H * p.W * p.C_in;

  simdgroup_matrix<float, 8, 8> Cfrag[TM][TN];
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      Cfrag[i][j] = simdgroup_matrix<float, 8, 8>(0);
    }
  }
  const int warp_m = (int(sgid) / WN) * WT_M;
  const int warp_n = (int(sgid) % WN) * WT_N;

  // fixed k-column / n-column per thread so the k -> (c, kw, kh, kd)
  // decomposition happens once per K-tile, not once per element
  const int a_k = tid % BK;
  const int a_r0 = tid / BK;
  const int b_n = tid % BN;
  const int b_r0 = tid / BN;

  for (uint32_t k0 = 0; k0 < uint32_t(K); k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int k = k0 + a_k;
    IDX di = 0, dh = 0, dw = 0;
    int cc = 0;
    bool k_ok = k < K;
    if (k_ok) {
      cc = k % p.C_in_per_group;
      int r = k / p.C_in_per_group;
      const int kw = r % p.kW;
      r /= p.kW;
      const int kh = r % p.kH;
      const int kd = r / p.kH;
      di = IDX(dd) * gP.sD - gP.padD + IDX(kd) * gP.dD;
      dh = IDX(kh) * p.dH - p.padH;
      dw = IDX(kw) * p.dW - p.padW;
      k_ok = di >= 0 && di < gP.D;
    }
    for (int rl = a_r0; rl < BM; rl += TGP / BK) {
      const IDX m = m_block + rl;
      T v = T(0);
      if (k_ok && m < M) {
        const IDX hi = (m / p.outW) * p.sH + dh;
        const IDX wi = (m % p.outW) * p.sW + dw;
        if (hi >= 0 && hi < p.H && wi >= 0 && wi < p.W) {
          v =
              act[act_nb + (((int64_t)di * p.H + hi) * p.W + wi) * p.C_in + c0 +
                  cc];
        }
      }
      As[rl * LDA + a_k] = v;
    }
    for (int rl = b_r0; rl < BK; rl += TGP / BN) {
      const int kb = k0 + rl;
      const int n = n_block + b_n;
      Bs[rl * LDB + b_n] = (kb < K && n < p.C_out_per_group)
          ? wts[(int64_t)kb * p.C_out + o0 + n]
          : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> Afrag[TM];
      simdgroup_matrix<T, 8, 8> Bfrag[TN];
      for (int i = 0; i < TM; ++i) {
        simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA + kk], LDA);
      }
      for (int j = 0; j < TN; ++j) {
        simdgroup_load(Bfrag[j], &Bs[kk * LDB + warp_n + j * 8], LDB);
      }
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
      }
    }
  }

  // per-lane (row, col) ownership within an 8x8 fragment; thread_elements()
  // holds (row, col) and (row, col + 1)
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      const IDX m = m_block + warp_m + i * 8 + fm;
      if (m >= M) {
        continue;
      }
      const int ho = int(m / p.outW), wo = int(m % p.outW);
      for (int e = 0; e < 2; ++e) {
        const int n = n_block + warp_n + j * 8 + fn + e;
        if (n >= p.C_out_per_group) {
          continue;
        }
        float v = Cfrag[i][j].thread_elements()[e];
        const int o = o0 + n;
        if (p.has_bias) {
          v += (float)bias[o];
        }
        int64_t idx;
        if (gP.out_ncdhw) {
          idx = ((((int64_t)nb * p.C_out + o) * gP.outD + dd) * p.outH + ho) *
                  p.outW +
              wo;
        } else {
          idx = ((((int64_t)nb * gP.outD + dd) * p.outH + ho) * p.outW + wo) *
                  p.C_out +
              o;
        }
        dst[idx] = (T)v;
      }
    }
  }
}

#define INSTANTIATE_CONV3D_SIMD(PFX, DT, IDX, BM, BN, BK, WM, WN)            \
  template                                                                   \
      [[host_name(PFX "_" #DT "_" #BM "_" #BN "_" #WM "_" #WN)]] kernel void \
      conv3d_simd<DT, IDX, BM, BN, BK, WM, WN>(                              \
          device DT*,                                                        \
          device DT*,                                                        \
          device DT*,                                                        \
          constant Conv3DParams&,                                            \
          device const DT*,                                                  \
          uint3,                                                             \
          uint,                                                              \
          uint);

// int covers bounded planes and coordinates; one long variant handles the rest
#define INSTANTIATE_CONV3D_SIMD_ALL(DT)                             \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 64, 64, 16, 2, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 32, 64, 16, 1, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd_long", DT, long, 64, 64, 16, 2, 2)

INSTANTIATE_CONV3D_SIMD_ALL(float)
INSTANTIATE_CONV3D_SIMD_ALL(half)
INSTANTIATE_CONV3D_SIMD_ALL(bfloat)

// Keep TC/TX pairs in sync with nchw_to_nhwc in operations/Convolution.mm.
#define INSTANTIATE_NCHW_TO_NHWC(DT, TC, TX, VECR, VECW)       \
  INSTANTIATE_CONV2D(                                          \
      "nchw_to_nhwc_" #DT "_" #TC "_" #TX "_" #VECR "_" #VECW, \
      nchw_to_nhwc<DT, TC, TX, kConv2dFlatThreads, VECR, VECW>);

static_assert(
    kNchwToNhwcMinChannelTile == 4 && kNchwToNhwcMaxChannelTile == 32 &&
        kNchwToNhwcVectorChannelTile == 16 && kNchwToNhwcTileElements == 1024,
    "Match the nchw_to_nhwc registrations below");

#define INSTANTIATE_NCHW_TO_NHWC_ALL(DT)             \
  INSTANTIATE_NCHW_TO_NHWC(DT, 4, 256, false, false) \
  INSTANTIATE_NCHW_TO_NHWC(DT, 4, 256, false, true)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 8, 128, false, false) \
  INSTANTIATE_NCHW_TO_NHWC(DT, 8, 128, false, true)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, false, false) \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, false, true)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, true, false)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, true, true)   \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, false, false) \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, false, true)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, true, false)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, true, true)

INSTANTIATE_NCHW_TO_NHWC_ALL(float)
INSTANTIATE_NCHW_TO_NHWC_ALL(half)
INSTANTIATE_NCHW_TO_NHWC_ALL(bfloat)

// DHWIO copy of an OIDHW weight view; ATen's generic permute copy runs well
// below memory bandwidth for this permutation. Each thread streams one kW row.
template <typename T>
kernel void conv_weight_to_dhwio(
    device const T* source [[buffer(0)]],
    device T* destination [[buffer(1)]],
    constant ConvWeightPermuteParams& params [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  const int output_channel = position.x;
  const int input_channel = position.y;
  const int kernel_plane_index = position.z;
  const int kernel_height_index = kernel_plane_index % params.kernel_height;
  const int kernel_depth_index = kernel_plane_index / params.kernel_height;
  device const T* source_row = source +
      output_channel * params.output_channel_stride +
      input_channel * params.input_channel_stride +
      kernel_depth_index * params.depth_stride +
      kernel_height_index * params.height_stride;
  device T* destination_row = destination +
      (kernel_plane_index * params.kernel_width *
           params.input_channels_per_group +
       input_channel) *
          params.output_channels +
      output_channel;
  bool bad = false;
  for (uint kernel_width_index = 0; kernel_width_index < params.kernel_width;
       ++kernel_width_index) {
    const T value = source_row[kernel_width_index * params.width_stride];
    bad |= params.emit_nonfinite_flags && !isfinite(float(value));
    destination_row
        [kernel_width_index * params.input_channels_per_group *
         params.output_channels] = value;
  }
  if (params.emit_nonfinite_flags) {
    const bool any_bad = simd_any(bad);
    if (simd_is_first()) {
      const int chunks =
          (params.output_channels + c10::metal::simdgroup_size - 1) /
          c10::metal::simdgroup_size;
      device uchar* flags = reinterpret_cast<device uchar*>(
          destination +
          int64_t(params.output_channels) * params.input_channels_per_group *
              params.kernel_height * params.kernel_width);
      const int64_t offset =
          (int64_t(kernel_height_index) * params.input_channels_per_group +
           input_channel) *
              chunks +
          output_channel / c10::metal::simdgroup_size;
      flags[offset] = any_bad;
    }
  }
}

#define INSTANTIATE_CONV_WEIGHT_TO_DHWIO(DT)                      \
  template [[host_name("conv_weight_to_dhwio_" #DT)]] kernel void \
  conv_weight_to_dhwio<DT>(                                       \
      device const DT*, device DT*, constant ConvWeightPermuteParams&, uint3);

INSTANTIATE_CONV_WEIGHT_TO_DHWIO(float)
INSTANTIATE_CONV_WEIGHT_TO_DHWIO(half)
INSTANTIATE_CONV_WEIGHT_TO_DHWIO(bfloat)

template <typename T, bool OUT_NLC>
kernel void conv1d_depthwise(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int x = int(pos.x) * kConv1dDepthwiseOutputsPerThread, o = int(pos.y);
  device const T* row =
      input + (int64_t(pos.z) * p.C_in + o / p.C_out_per_group) * p.W;
  float acc[kConv1dDepthwiseOutputsPerThread];
  for (int j = 0; j < kConv1dDepthwiseOutputsPerThread; ++j) {
    acc[j] = p.has_bias ? float(bias[o]) : 0.0f;
  }
  for (int tap = 0; tap < p.kW; ++tap) {
    const float w = float(weight[o * p.kW + tap]);
    for (int j = 0; j < kConv1dDepthwiseOutputsPerThread; ++j) {
      const int i = (x + j) * p.sW + tap * p.dW - p.padW;
      if (i >= 0 && i < p.W) {
        acc[j] += float(row[i]) * w;
      }
    }
  }
  for (int j = 0; j < kConv1dDepthwiseOutputsPerThread && x + j < p.outW; ++j) {
    output
        [OUT_NLC ? (int64_t(pos.z) * p.outW + x + j) * p.C_out + o
                 : (int64_t(pos.z) * p.C_out + o) * p.outW + x + j] = T(acc[j]);
  }
}

#define INSTANTIATE_CONV1D_DEPTHWISE(DT, OUT, OUT_NLC)                 \
  template [[host_name("conv1d_depthwise_" #OUT "_" #DT)]] kernel void \
  conv1d_depthwise<DT, OUT_NLC>(                                       \
      device const DT*,                                                \
      device const DT*,                                                \
      device DT*,                                                      \
      constant Conv2DParams&,                                          \
      device const DT*,                                                \
      uint3);

INSTANTIATE_CONV1D_DEPTHWISE(float, ncl, false)
INSTANTIATE_CONV1D_DEPTHWISE(half, ncl, false)
INSTANTIATE_CONV1D_DEPTHWISE(bfloat, ncl, false)
INSTANTIATE_CONV1D_DEPTHWISE(float, nlc, true)
INSTANTIATE_CONV1D_DEPTHWISE(half, nlc, true)
INSTANTIATE_CONV1D_DEPTHWISE(bfloat, nlc, true)

#if C10_METAL_HAS_MPP
#include <metal_cooperative_tensor>
#include <metal_simdgroup>

using namespace mpp::tensor_ops;

template <int X, int Y, int Z>
struct StaticInt3 {
  static constexpr constant int x = X;
  static constexpr constant int y = Y;
  static constexpr constant int z = Z;
};

// One 2D convolution per (batch, output-depth) slice; the KD loop accumulates
// kernel-depth taps into one cooperative tensor, skipping depth padding.
template <
    typename T,
    typename Block,
    int NSG,
    typename Kernel,
    typename Stride,
    typename Dilation,
    typename Source,
    bool RELAXED,
    bool HAS_BIAS,
    bool OUT_NCDHW,
    bool GROUPED>
kernel void conv3d_mpp(
    device T* act [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv3DParams& gP [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constant Conv2DParams& p = gP.conv2d;
  constexpr int BO = Block::x, BW = Block::y, BH = Block::z;
  constexpr int KW = Kernel::x, KH = Kernel::y, KD = Kernel::z;
  constexpr int SX = Stride::x, SY = Stride::y, SZ = Stride::z;
  constexpr int DX = Dilation::x, DY = Dilation::y, DZ = Dilation::z;
  constexpr int SRCC = Source::x, SRCW = Source::y, SRCH = Source::z;
  const int h_tiles = p.outH / BH + (p.outH % BH != 0);
  int o_off, o_end, c0;
  if constexpr (GROUPED) {
    const int o_tiles = p.C_out_per_group / BO + (p.C_out_per_group % BO != 0);
    const int g = int(tgid.x) / o_tiles;
    o_off = (int(tgid.x) % o_tiles) * BO + g * p.C_out_per_group;
    o_end = g * p.C_out_per_group + p.C_out_per_group;
    c0 = g * p.C_in_per_group;
  } else {
    o_off = int(tgid.x) * BO;
    o_end = p.C_out;
    c0 = 0;
  }
  const int wo_off = int(tgid.y) * BW;
  int zi = int(tgid.z);
  const int ho_off = (zi % h_tiles) * BH;
  zi /= h_tiles;
  const int dd = zi % gP.outD;
  const int nb = zi / gP.outD;

  // DHWIO weights viewed innermost-first; kernel-depth slice kd starts at
  // row kd * KH of the flattened (KH * KD) height axis.
  tensor<device T, dextents<int32_t, 4>, tensor_inline> tWt(
      wts, dextents<int32_t, 4>(p.C_out, p.C_in_per_group, KW, KH * KD));

  constexpr auto desc = convolution2d_descriptor(
      int4(BO, BW, BH, 1),
      int4(SRCC, SRCW, SRCH, 1),
      int2(KW, KH),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(SX, SY),
      int2(DX, DY),
      1,
      RELAXED,
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<desc, execution_simdgroups<NSG>> op;
  // MPP anchors the window at ceil((K - 1) * dilation / 2) (measured; for
  // even kernels with dilation > 1 this is NOT (K / 2) * dilation).
  op.set_offsets(int2(
      wo_off * SX - p.padW + ((KW - 1) * DX + 1) / 2,
      ho_off * SY - p.padH + ((KH - 1) * DY + 1) / 2));

  const int64_t plane = (int64_t)p.H * p.W * p.C_in;
  device T* actn = act + (int64_t)nb * gP.D * plane;
  auto act_view = [&](device T* ptr) {
    if constexpr (GROUPED) {
      return tensor<device T, dextents<int32_t, 4>, tensor_inline>(
          ptr,
          dextents<int32_t, 4>(c0 + p.C_in_per_group, p.W, p.H, 1),
          array<int32_t, 4>{1, p.C_in, p.C_in * p.W, p.C_in * p.W * p.H});
    } else {
      return tensor<device T, dextents<int32_t, 4>, tensor_inline>(
          ptr, dextents<int32_t, 4>(p.C_in, p.W, p.H, 1));
    }
  };
  auto tA0 = act_view(actn);
  auto mA0 = tA0.slice(c0, 0, 0, 0);
  auto mW0 = tWt.slice(o_off, 0, 0, 0);

  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mA0),
      decltype(mW0),
      float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cT[i] = 0.0f;
  }

  for (int kd = 0; kd < KD; ++kd) {
    const int di = dd * SZ - gP.padD + kd * DZ;
    if (di < 0 || di >= gP.D) {
      continue;
    }
    auto tA = act_view(actn + di * plane);
    auto mA = tA.slice(c0, 0, 0, 0);
    auto mW = tWt.slice(o_off, 0, 0, kd * KH);
    op.run(mA, mW, cT);
  }

  if constexpr (GROUPED || OUT_NCDHW) {
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      auto idx = cT.get_multidimensional_index(i);
      const int o = o_off + (int)idx[0];
      const int x = wo_off + (int)idx[1];
      const int y = ho_off + (int)idx[2];
      if (o >= o_end || x >= p.outW || y >= p.outH) {
        continue;
      }
      float v = cT[i];
      if constexpr (HAS_BIAS) {
        v += (float)bias[o];
      }
      int64_t di;
      if constexpr (OUT_NCDHW) {
        di = ((((int64_t)nb * p.C_out + o) * gP.outD + dd) * p.outH + y) *
                p.outW +
            x;
      } else {
        di = ((((int64_t)nb * gP.outD + dd) * p.outH + y) * p.outW + x) *
                p.C_out +
            o;
      }
      dst[di] = (T)v;
    }
  } else {
    device T* dstn =
        dst + ((int64_t)nb * gP.outD + dd) * p.outH * p.outW * p.C_out;
    tensor<device T, dextents<int32_t, 4>, tensor_inline> tD(
        dstn, dextents<int32_t, 4>(p.C_out, p.outW, p.outH, 1));
    auto mD = tD.slice(o_off, wo_off, ho_off, 0);
    auto cO = op.template get_destination_cooperative_tensor<
        decltype(mA0),
        decltype(mW0),
        T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      float v = cT[i];
      if constexpr (HAS_BIAS) {
        // clamp: lanes past O are masked by the bounds-aware store below
        const int o = o_off + (int)cT.get_multidimensional_index(i)[0];
        v += (float)bias[min(o, p.C_out - 1)];
      }
      cO[i] = (T)v;
    }
    cO.store(mD);
  }
}

#define INSTANTIATE_CONV3D_MPP(                                             \
    DT,                                                                     \
    DNAME,                                                                  \
    RELAXED,                                                                \
    BO,                                                                     \
    BW,                                                                     \
    BH,                                                                     \
    NSG,                                                                    \
    KD,                                                                     \
    KH,                                                                     \
    KW,                                                                     \
    SZ,                                                                     \
    SY,                                                                     \
    SX,                                                                     \
    DZ,                                                                     \
    DY,                                                                     \
    DX,                                                                     \
    CNAME,                                                                  \
    SRCC,                                                                   \
    BNAME,                                                                  \
    HAS_BIAS,                                                               \
    LNAME,                                                                  \
    OUT_NCDHW,                                                              \
    GNAME,                                                                  \
    GROUPED)                                                                \
  template                                                                  \
      [[host_name("conv3d_mpp_" #DNAME "_b" #BO "_w" #BW "_h" #BH "_s" #NSG \
                  "_k" #KD #KH #KW "_s" #SZ #SY #SX "_d" #DZ #DY #DX        \
                  "_c" #CNAME "_" #BNAME "_" #LNAME "_" #GNAME)]]           \
      kernel void conv3d_mpp<                                               \
          DT,                                                               \
          StaticInt3<BO, BW, BH>,                                           \
          NSG,                                                              \
          StaticInt3<KW, KH, KD>,                                           \
          StaticInt3<SX, SY, SZ>,                                           \
          StaticInt3<DX, DY, DZ>,                                           \
          StaticInt3<SRCC, 16384, 16384>,                                   \
          RELAXED,                                                          \
          HAS_BIAS,                                                         \
          OUT_NCDHW,                                                        \
          GROUPED>(                                                         \
          device DT*,                                                       \
          device DT*,                                                       \
          device DT*,                                                       \
          constant Conv3DParams&,                                           \
          device const DT*,                                                 \
          uint3);

#define INSTANTIATE_CONV3D_MPP_DTYPES( \
    BO,                                \
    BW,                                \
    BH,                                \
    NSG,                               \
    KD,                                \
    KH,                                \
    KW,                                \
    SZ,                                \
    SY,                                \
    SX,                                \
    DZ,                                \
    DY,                                \
    DX,                                \
    CNAME,                             \
    SRCC,                              \
    BNAME,                             \
    HAS_BIAS,                          \
    LNAME,                             \
    OUT_NCDHW,                         \
    GNAME,                             \
    GROUPED)                           \
  INSTANTIATE_CONV3D_MPP(              \
      float,                           \
      float,                           \
      false,                           \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)                         \
  INSTANTIATE_CONV3D_MPP(              \
      half,                            \
      half,                            \
      true,                            \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)                         \
  INSTANTIATE_CONV3D_MPP(              \
      bfloat,                          \
      bfloat,                          \
      true,                            \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)

#define INSTANTIATE_CONV3D_MPP_TILES( \
    KD,                               \
    KH,                               \
    KW,                               \
    SZ,                               \
    SY,                               \
    SX,                               \
    DZ,                               \
    DY,                               \
    DX,                               \
    CNAME,                            \
    SRCC,                             \
    BNAME,                            \
    HAS_BIAS,                         \
    LNAME,                            \
    OUT_NCDHW,                        \
    GNAME,                            \
    GROUPED)                          \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      32,                             \
      8,                              \
      8,                              \
      2,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      32,                             \
      16,                             \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      64,                             \
      8,                              \
      8,                              \
      2,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      64,                             \
      16,                             \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      128,                            \
      8,                              \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)

#define INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(          \
    KD, KH, KW, SZ, SY, SX, CNAME, SRCC, BNAME, HAS_BIAS) \
  INSTANTIATE_CONV3D_MPP_TILES(                           \
      KD,                                                 \
      KH,                                                 \
      KW,                                                 \
      SZ,                                                 \
      SY,                                                 \
      SX,                                                 \
      1,                                                  \
      1,                                                  \
      1,                                                  \
      CNAME,                                              \
      SRCC,                                               \
      BNAME,                                              \
      HAS_BIAS,                                           \
      ncdhw,                                              \
      true,                                               \
      ungrouped,                                          \
      false)

#define INSTANTIATE_CONV3D_MPP_STANDARD(KD, KH, KW, SZ, SY, SX, CNAME, SRCC) \
  INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(                                   \
      KD, KH, KW, SZ, SY, SX, CNAME, SRCC, bias, true)                       \
  INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(                                   \
      KD, KH, KW, SZ, SY, SX, CNAME, SRCC, nobias, false)

#define INSTANTIATE_CONV3D_MPP_GROUPED(KD, KH, KW, SZ, SY, SX) \
  INSTANTIATE_CONV3D_MPP_TILES(                                \
      KD,                                                      \
      KH,                                                      \
      KW,                                                      \
      SZ,                                                      \
      SY,                                                      \
      SX,                                                      \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      nobias,                                                  \
      false,                                                   \
      ncdhw,                                                   \
      true,                                                    \
      grouped,                                                 \
      true)

// Deduplicated direct Conv3d specs from the surveyed model shapes.
INSTANTIATE_CONV3D_MPP_GROUPED(1, 1, 2, 1, 1, 2)
INSTANTIATE_CONV3D_MPP_GROUPED(1, 2, 1, 1, 2, 1)
INSTANTIATE_CONV3D_MPP_GROUPED(2, 1, 1, 2, 1, 1)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 2, 2, 1, 2, 2, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 2, 2, 1, 2, 2, 36, 36)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 2, 2, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, 3, 3)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 2, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 3, 3)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 48, 48)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 64, 64)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 2, 2, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 2, 2, 2, dyn, -1)

template <typename T, bool NLC, bool OUT_NLC, int BO, int NSG>
kernel void conv1d_mpp(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    device T* head [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device T, extents, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(
      OUT_NLC ? 64 : BO,
      OUT_NLC ? BO : 64,
      static_cast<int>(dynamic_extent),
      OUT_NLC && !NLC,
      OUT_NLC || NLC,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      tensor2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  const int cg = p.C_in_per_group, og = p.C_out_per_group;
  const int group = int(tgid.x) / ((og + BO - 1) / BO);
  const int o_off = int(tgid.x) % ((og + BO - 1) / BO) * BO;
  const int x_off = int(tgid.y) * 64;
  const bool merged = NLC && p.dW == 1 && p.C_in == cg;
  const int K = merged ? p.kW * cg : cg;
  // Tiles are 64 output positions wide (the matmul2d N extent). `head` holds
  // the positions read by the tiles that start inside the left padding: up to
  // ceil(padW / 64) * 64 of them plus the taps reaching past the last one.
  const bool from_head = !NLC && x_off < p.padW;
  const int length = from_head
      ? min(p.W + p.padW, (p.padW + 63) / 64 * 64 + (p.kW - 1) * p.dW)
      : p.W;
  device T* in = (from_head ? head : input) +
      int64_t(tgid.z) * length * p.C_in + group * cg * (NLC ? 1 : length);
  for (int tap = 0; tap < (merged ? 1 : p.kW); ++tap) {
    const int start =
        NLC ? tap * p.dW : x_off + tap * p.dW - (from_head ? 0 : p.padW);
    if (!NLC && start >= length) {
      continue;
    }
    tensor2d wt(
        weight + (merged ? 0 : (int64_t(tap) * p.C_out + group * og) * cg),
        extents(K, og),
        strides{1, merged ? p.kW * cg : cg});
    tensor2d xt(
        in + start * (NLC ? p.C_in : 1),
        NLC ? extents(K, p.outW) : extents(length - start, cg),
        strides{1, NLC ? p.sW * p.C_in : length});
    auto w_tile = wt.slice(0, o_off);
    auto x_tile = xt.slice(0, NLC ? x_off : 0);
    if constexpr (OUT_NLC) {
      op.run(x_tile, w_tile, acc);
    } else {
      op.run(w_tile, x_tile, acc);
    }
  }
  auto out =
      op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o = group * og +
        min(o_off + int(acc.get_multidimensional_index(i)[OUT_NLC ? 0 : 1]),
            og - 1);
    out[i] =
        T(acc[i] +
          (p.has_bias && acc.is_valid_element(i) ? float(bias[o]) : 0.0f));
  }
  tensor2d dst(
      output + int64_t(tgid.z) * p.C_out * p.outW +
          group * og * (OUT_NLC ? 1 : p.outW),
      OUT_NLC ? extents(og, p.outW) : extents(p.outW, og),
      strides{1, OUT_NLC ? p.C_out : p.outW});
  auto dst_tile = OUT_NLC ? dst.slice(o_off, x_off) : dst.slice(x_off, o_off);
  out.store(dst_tile);
}

#define INSTANTIATE_CONV1D_MPP(DT, IN, NLC, OUT, OUT_NLC, BO, NSG)   \
  template [[host_name("conv1d_mpp_" #IN "_" #OUT "_b" #BO "_s" #NSG \
                       "_" #DT)]] kernel void                        \
  conv1d_mpp<DT, NLC, OUT_NLC, BO, NSG>(                             \
      device DT*,                                                    \
      device DT*,                                                    \
      device DT*,                                                    \
      constant Conv2DParams&,                                        \
      device const DT*,                                              \
      device DT*,                                                    \
      uint3);

#define INSTANTIATE_CONV1D_MPP_NCL(DT, BO, NSG)               \
  INSTANTIATE_CONV1D_MPP(DT, ncl, false, ncl, false, BO, NSG) \
  INSTANTIATE_CONV1D_MPP(DT, ncl, false, nlc, true, BO, NSG)

#define INSTANTIATE_CONV1D_MPP_LAYOUTS(DT, NSG)              \
  INSTANTIATE_CONV1D_MPP_NCL(DT, 64, NSG)                    \
  INSTANTIATE_CONV1D_MPP(DT, nlc, true, ncl, false, 64, NSG) \
  INSTANTIATE_CONV1D_MPP(DT, nlc, true, nlc, true, 64, NSG)

#define INSTANTIATE_CONV1D_MPP_ALL(DT)  \
  INSTANTIATE_CONV1D_MPP_LAYOUTS(DT, 2) \
  INSTANTIATE_CONV1D_MPP_LAYOUTS(DT, 4) \
  INSTANTIATE_CONV1D_MPP_NCL(DT, 16, 2) \
  INSTANTIATE_CONV1D_MPP_NCL(DT, 32, 2)

INSTANTIATE_CONV1D_MPP_ALL(float)
INSTANTIATE_CONV1D_MPP_ALL(half)
INSTANTIATE_CONV1D_MPP_ALL(bfloat)

// One-image bfloat16 1x1 convolution (Conv2dKernel::PointwisePlane): a 2x2
// grid of SIMD groups computes 64 output channels x 128 positions, reducing
// complete 32-channel input blocks with matmul2d. C = 128 or 256 fixes the
// reduction length; C = 0 reads it at runtime.
template <int C>
[[max_total_threads_per_threadgroup(4 * c10::metal::simdgroup_size)]]
kernel void conv2d_pointwise(
    device bfloat* input [[buffer(0)]],
    device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const bfloat* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
  // A 2x2 grid of simdgroups, each computing 32 channels by 64 positions.
  constexpr int rows = 32, columns = 64, reduction = 32;
  const int channels = C == 0 ? p.C_in : C;
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using matrix = tensor<device bfloat, extents, tensor_inline>;
  using Mode = matmul2d_descriptor::mode;
  constexpr auto desc = matmul2d_descriptor(
      rows, columns, reduction, false, false, true, Mode::multiply_accumulate);
  constexpr auto first = matmul2d_descriptor(
      rows, columns, reduction, false, false, true, Mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<first, execution_simdgroup> initial;
  auto b =
      op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using tile = remove_addrspace_t<decltype(b)>;
  auto acc =
      op.template get_destination_cooperative_tensor<matrix, tile, float>();
  const int plane = p.H * p.W;
  const int o0 = (int(gid.y) * 2 + int(sid) / 2) * rows;
  const int x0 = (int(gid.x) * 2 + int(sid) % 2) * columns;
  if (x0 >= plane || o0 >= p.C_out) {
    return;
  }
  input += int64_t(gid.z) * channels * plane;
  output += int64_t(gid.z) * p.C_out * plane;
  matrix w(weight, extents(channels, p.C_out), strides{1, channels});
  matrix x(
      input + x0,
      extents(min(columns, plane - x0), reduction),
      strides{1, plane});
  auto wt = w.slice(0, o0);
  b.load(x);
  initial.run(wt, b, acc);
#pragma unroll
  for (int c0 = reduction; c0 < channels; c0 += reduction) {
    matrix v(
        input + int64_t(c0) * plane + x0,
        extents(min(columns, plane - x0), reduction),
        strides{1, plane});
    auto wv = w.slice(c0, o0);
    b.load(v);
    op.run(wv, b, acc);
  }
  auto res =
      op.template get_destination_cooperative_tensor<matrix, tile, bfloat>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
    res[i] =
        bfloat(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
  }
  matrix out(output, extents(plane, p.C_out), strides{1, plane});
  res.store(out.slice(x0, o0));
}

INSTANTIATE_CONV2D("conv2d_pointwise_128", conv2d_pointwise<128>);
INSTANTIATE_CONV2D("conv2d_pointwise_256", conv2d_pointwise<256>);
INSTANTIATE_CONV2D("conv2d_pointwise_dynamic", conv2d_pointwise<0>);

// bfloat16 3x3 convolution over 64, 128 or 176 input channels
// (Conv2dKernel::Narrow): each SIMD group accumulates 64 output channels of a
// narrow position tile, reading its input fragments straight into
// cooperative tensors.
template <int C>
[[max_total_threads_per_threadgroup(
    (C == 128 ? 8 : 4) * c10::metal::simdgroup_size)]] kernel void
conv2d_narrow(
    device bfloat* input [[buffer(0)]],
    device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const bfloat* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 64, BN = C == 176 ? 32 : 16;
  constexpr int BK = C == 176 ? 16 : C == 64 ? 64 : 32;
  constexpr int SG = C == 128 ? 8 : 4;
  constexpr bool TRANS = C == 128;
  using Mode = matmul2d_descriptor::mode;
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, BK, TRANS, false, true, Mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  using E = dextents<int32_t, 2>;
  using S = array<int32_t, 2>;
  using matrix = tensor<device bfloat, E, tensor_inline>;
  using tile = remove_addrspace_t<
      decltype(op.template get_right_input_cooperative_tensor<
               bfloat,
               bfloat,
               float>())>;
  auto acc =
      op.template get_destination_cooperative_tensor<matrix, tile, float>();
#pragma unroll
  for (ushort i = 0; i < BM * BN / c10::metal::simdgroup_size; ++i) {
    acc[i] = 0;
  }
  const int plane = p.H * p.W, o0 = int(gid.x) * BM,
            t0 = (int(gid.y) * SG + int(sid)) * BN;
  if (t0 >= plane) {
    return;
  }
  input += int64_t(gid.z) * C * plane;
  output += int64_t(gid.z) * p.C_out * plane;
  {
    auto b =
        op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    uint left = 0, right = 0;
#pragma unroll
    for (ushort i = 0; i < BN * BK / c10::metal::simdgroup_size; ++i) {
      const int col = (t0 + int(b.get_multidimensional_index(i)[0])) % p.W;
      if (col == 0) {
        left |= 1u << i;
      }
      if (col == p.W - 1) {
        right |= 1u << i;
      }
    }
    // Channel chunks run outermost so that the nine taps of a chunk reuse its
    // input rows.
#pragma unroll
    for (int c0 = 0; c0 < C; c0 += BK) {
#pragma unroll
      for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
          const int pos = t0 + (kh - 1) * p.W + kw - 1;
          const uint mask = kw == 0 ? left : kw == 2 ? right : 0;
          matrix w(
              weight + int64_t(kh * 3 + kw) * C * p.C_out,
              E(TRANS ? p.C_out : C, TRANS ? C : p.C_out),
              S{1, TRANS ? p.C_out : C});
          auto wv = w.slice(TRANS ? o0 : c0, TRANS ? c0 : o0);
          if (pos >= 0 && pos + BN <= plane) {
            matrix xv(
                input + int64_t(c0 * plane) + pos,
                E(BN, min(BK, C - c0)),
                S{1, plane});
            b.load(xv);
          } else {
#pragma unroll
            for (ushort i = 0; i < BN * BK / c10::metal::simdgroup_size; ++i) {
              const auto ij = b.get_multidimensional_index(i);
              const int cc = c0 + int(ij[1]), sp = pos + int(ij[0]);
              // A wide final offset keeps boundary loads fast.
              b[i] = cc < C && sp >= 0 && sp < plane
                  ? input[int64_t(cc * plane) + sp]
                  : bfloat(0);
            }
          }
          if (mask) {
#pragma unroll
            for (ushort i = 0; i < BN * BK / c10::metal::simdgroup_size; ++i) {
              if (mask & (1u << i)) {
                b[i] = bfloat(0);
              }
            }
          }
          op.run(wv, b, acc);
        }
      }
    }
  }
  auto res =
      op.template get_destination_cooperative_tensor<matrix, tile, bfloat>();
  if (p.has_bias) {
#pragma unroll
    for (ushort i = 0; i < BM * BN / c10::metal::simdgroup_size; ++i) {
      const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
      res[i] = bfloat(acc[i] + (o < p.C_out ? float(bias[o]) : 0.0f));
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < BM * BN / c10::metal::simdgroup_size; ++i) {
      res[i] = bfloat(acc[i]);
    }
  }
  matrix out(output, E(plane, p.C_out), S{1, plane});
  res.store(out.slice(t0, o0));
}

INSTANTIATE_CONV2D("conv2d_narrow_64", conv2d_narrow<64>);
INSTANTIATE_CONV2D("conv2d_narrow_128", conv2d_narrow<128>);
INSTANTIATE_CONV2D("conv2d_narrow_176", conv2d_narrow<176>);

// Padded direct weights append one nonfinite flag per kernel row, output,
// and SIMD-width input-channel chunk; only flagged boundary tiles need a scan.
bool conv2d_mpp_boundary(
    thread const Conv2dMppParams& gp,
    int oh,
    int x,
    int width) {
  thread const Conv2DParams& p = gp.conv;
  const int step = gp.parity ? 2 : p.sW,
            logical_width = p.W - int(gp.parity != 0);
  return !gp.Wp &&
      (oh * p.sH < p.padH || oh * p.sH + (p.kH - 1) * p.dH - p.padH >= p.H ||
       x * step < p.padW ||
       (min(x + width, p.outW) - 1) * step + (p.kW - 1) * p.dW - p.padW >=
           logical_width);
}

template <typename T>
bool conv2d_mpp_padding_nonfinite(
    device const T* weight,
    thread const Conv2dMppParams& gp,
    int g,
    int o,
    int oh,
    int x,
    int k0,
    int K) {
  thread const Conv2DParams& p = gp.conv;
  const int step = gp.parity ? 2 : p.sW, width = p.W - int(gp.parity != 0);
  for (int kh = 0; kh < p.kH; ++kh) {
    const int ih = oh * p.sH + kh * p.dH - p.padH;
    for (int kw = 0; kw < p.kW; ++kw) {
      const int iw = x * step + kw * p.dW - p.padW;
      if (ih >= 0 && ih < p.H && iw >= 0 && iw < width)
        continue;
      const int64_t offset =
          (int64_t(kh * p.kW + kw) * p.C_out + g * p.C_out_per_group + o) *
              gp.K +
          k0;
      for (int c = 0; c < K; ++c) {
        if (!isfinite(float(weight[offset + c])))
          return true;
      }
    }
  }
  return false;
}

template <typename T, int BM>
bool conv2d_mpp_tile_nonfinite(
    device const T* weight,
    thread const Conv2dMppParams& gp,
    int g,
    int o0,
    uint tid) {
  thread const Conv2DParams& p = gp.conv;
  const int chunks = gp.K / c10::metal::simdgroup_size +
      (gp.K % c10::metal::simdgroup_size != 0);
  device const uchar* source = reinterpret_cast<device const uchar*>(
      weight + int64_t(p.kH) * p.kW * p.C_out * gp.K);
  const int count = min(BM, p.C_out_per_group - o0) * chunks;
  bool bad = false;
  for (int kh = 0; kh < p.kH; ++kh) {
    const int offset = (kh * p.C_out + g * p.C_out_per_group + o0) * chunks;
    for (uint j = tid % c10::metal::simdgroup_size; j < uint(count);
         j += c10::metal::simdgroup_size) {
      bad |= source[offset + j] != 0;
    }
  }
  return simd_any(bad);
}

// Single-tap GEMM; Winograd batches also select a separate weight matrix.
template <
    typename T,
    int BM,
    int BN,
    int NSG,
    bool BATCHED_WEIGHTS,
    bool OUT_NHWC = false>
[[max_total_threads_per_threadgroup(NSG * c10::metal::simdgroup_size)]]
kernel void conv2d_mpp_pointwise(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2dMppParams& geometry [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constant Conv2DParams& p = geometry.conv;
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device T, extents, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(
      OUT_NHWC ? BN : BM,
      OUT_NHWC ? BM : BN,
      static_cast<int>(dynamic_extent),
      OUT_NHWC,
      OUT_NHWC,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      tensor2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  const int plane = p.H * p.W, o0 = int(tgid.x) * BM;
  const int x0 = int(tgid.y) * BN, n = int(tgid.z);
  tensor2d x(
      input + int64_t(n) * p.C_in * plane,
      extents(plane, p.C_in),
      strides{1, plane});
  tensor2d w(
      weight + (BATCHED_WEIGHTS ? int64_t(n) * p.C_in * p.C_out : 0),
      extents(p.C_in, p.C_out),
      strides{1, p.C_in});
  auto xt = x.slice(x0, 0), wt = w.slice(0, o0);
  if IF_CONSTEXPR (OUT_NHWC)
    op.run(xt, wt, acc);
  else
    op.run(wt, xt, acc);
  auto res =
      op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o = o0 + int(acc.get_multidimensional_index(i)[OUT_NHWC ? 0 : 1]);
    res[i] = T(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
  }
  tensor2d out(
      output + int64_t(n) * p.C_out * plane,
      extents(OUT_NHWC ? p.C_out : plane, OUT_NHWC ? plane : p.C_out),
      strides{1, OUT_NHWC ? p.C_out : plane});
  res.store(out.slice(OUT_NHWC ? o0 : x0, OUT_NHWC ? x0 : o0));
}

Conv2dMppParams conv2d_mpp_params(
    constant Conv2dMppParams& geometry,
    false_type) {
  return geometry;
}

Conv2dMppParams conv2d_mpp_params(
    constant Conv2dMppParams& geometry,
    true_type) {
  auto gp = geometry;
  gp.conv.kH = gp.conv.kW = 3;
  gp.conv.sH = gp.conv.sW = gp.conv.dH = gp.conv.dW = gp.conv.padH =
      gp.conv.padW = 1;
  gp.conv.C_out_per_group = gp.conv.C_out;
  gp.K = gp.K_split = gp.conv.C_in;
  gp.splits = 1;
  gp.Wp = gp.parity = 0;
  return gp;
}

// Generic products have no column threadgroups.
template <typename T, int BM, int NSG>
void conv2d_mpp_fixed3_column(
    device const T*,
    device const T*,
    device T*,
    device const T*,
    thread const Conv2DParams&,
    int,
    int,
    uint,
    uint,
    bool,
    false_type) {}

// Output column 0 of the fixed 3x3 products, whose tiles start at column 1:
// BM output channels x kConv2dFixed3ColumnRows rows from 8x8 simdgroup
// matrices, each SIMD group owning BM / NSG channels over the whole
// reduction. Padding taps multiply zero, so non-finite weights propagate;
// the left padding column is skipped unless `nonfinite` flags such weights.
template <typename T, int BM, int NSG>
void conv2d_mpp_fixed3_column(
    device const T* input,
    device const T* weight,
    device T* output,
    device const T* bias,
    thread const Conv2DParams& p,
    int o0,
    int r0,
    uint sg,
    uint lane,
    bool nonfinite,
    true_type) {
  constexpr int frag = 8, taps = 3;
  constexpr int outputs = BM / NSG / frag,
                rows = kConv2dFixed3ColumnRows / frag;
  // Per-lane (row, col) ownership within an 8x8 fragment; thread_elements()
  // holds (row, col) and (row, col + 1).
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  o0 += int(sg) * (BM / NSG);
  const int plane = p.H * p.W;
  simdgroup_matrix<float, 8, 8> acc[outputs][rows];
  for (int i = 0; i < outputs; ++i) {
    for (int j = 0; j < rows; ++j) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int c0 = 0; c0 < p.C_in; c0 += frag) {
    const int c = c0 + fm;
#pragma unroll
    for (int kh = 0; kh < taps; ++kh) {
#pragma unroll
      for (int kw = 0; kw < taps; ++kw) {
        const int iw = kw - p.padW;
        if (iw < 0 && !nonfinite) {
          continue;
        }
        // Loads use clamped in-bounds addresses and select zero afterwards,
        // so they can all be in flight at once.
        simdgroup_matrix<T, 8, 8> x[rows];
        for (int j = 0; j < rows; ++j) {
          for (int e = 0; e < 2; ++e) {
            const int ih = r0 + j * frag + fn + e + kh - p.padH;
            const bool inside =
                c < p.C_in && iw >= 0 && iw < p.W && ih >= 0 && ih < p.H;
            const T value = input
                [min(c, p.C_in - 1) * plane + clamp(ih, 0, p.H - 1) * p.W +
                 clamp(iw, 0, p.W - 1)];
            x[j].thread_elements()[e] = inside ? value : T(0);
          }
        }
        device const T* w = weight + (kh * taps + kw) * p.C_out * p.C_in;
        for (int i = 0; i < outputs; ++i) {
          simdgroup_matrix<T, 8, 8> a;
          const int o = min(o0 + i * frag + fm, p.C_out - 1);
          for (int e = 0; e < 2; ++e) {
            const int k = c0 + fn + e;
            const T value = w[o * p.C_in + min(k, p.C_in - 1)];
            a.thread_elements()[e] = k < p.C_in ? value : T(0);
          }
          for (int j = 0; j < rows; ++j) {
            simdgroup_multiply_accumulate(acc[i][j], a, x[j], acc[i][j]);
          }
        }
      }
    }
  }
  for (int i = 0; i < outputs; ++i) {
    const int o = o0 + i * frag + fm;
    for (int j = 0; j < rows; ++j) {
      for (int e = 0; e < 2; ++e) {
        const int oh = r0 + j * frag + fn + e;
        if (o < p.C_out && oh < p.outH) {
          output[(o * p.outH + oh) * p.outW] =
              T(acc[i][j].thread_elements()[e] +
                (p.has_bias ? float(bias[o]) : 0.0f));
        }
      }
    }
  }
}

// Convolution as a GEMM (Conv2dKernel::Mpp): NSG SIMD groups compute BM output
// channels x BN output positions with matmul2d over K = taps x channels of one
// group, reading the input prepared per Conv2dMppInput. Split-K partials go to
// `partial` for conv2d_split_reduce.
template <
    typename T,
    bool IN_NHWC,
    bool OUT_NHWC,
    int BM,
    int BN,
    int NSG,
    bool FIXED3 = false>
[[max_total_threads_per_threadgroup(NSG * c10::metal::simdgroup_size)]]
kernel void conv2d_mpp(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2dMppParams& geometry [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    device float* partial [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const auto gp = conv2d_mpp_params(geometry, bool_constant<FIXED3>{});
  thread const Conv2DParams& p = gp.conv;
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device T, extents, tensor_inline>;
  using tensor2df = tensor<device float, extents, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(
      OUT_NHWC ? BN : BM,
      OUT_NHWC ? BM : BN,
      static_cast<int>(dynamic_extent),
      OUT_NHWC && !IN_NHWC,
      OUT_NHWC || IN_NHWC,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const bool flat = gp.Wp > 0;
  const int og = p.C_out_per_group;
  const int split = int(tgid.z) % gp.splits, k0 = split * gp.K_split;
  const int K = min(gp.K_split, gp.K - k0);
  const int o_tiles = og / BM + (og % BM != 0);
  const int g = int(tgid.x) / o_tiles;
  const int o_off = int(tgid.x) % o_tiles * BM;
  // Shallow fixed 3x3 tiles start at column 1 so that no tap reads the left
  // padding; the first threadgroups of the grid compute column 0 instead.
  const bool columns = FIXED3 && p.C_in <= kConv2dFixed3ColumnMaxInputs;
  const int x_begin = int(columns);
  const int column_blocks =
      columns ? c10::metal::ceil_div(p.outH, kConv2dFixed3ColumnRows) : 0;
  if (int(tgid.y) < column_blocks) {
    const int64_t image = int64_t(tgid.z) * p.C_in * p.H * p.W;
    conv2d_mpp_fixed3_column<T, BM, NSG>(
        input + image,
        weight,
        output + int64_t(tgid.z) * p.C_out * p.outH * p.outW,
        bias,
        p,
        o_off,
        int(tgid.y) * kConv2dFixed3ColumnRows,
        tid / c10::metal::simdgroup_size,
        tid % c10::metal::simdgroup_size,
        conv2d_mpp_tile_nonfinite<T, BM>(weight, gp, g, o_off, tid),
        bool_constant<FIXED3>{});
    return;
  }
  // With edge columns the row tiles stop at the last whole tile, and the
  // first threadgroups each run one trailing column down BN rows, so that
  // they do not trail the grid.
  const int h_tiles = c10::metal::ceil_div(p.outH, BN);
  const int edge_tiles = gp.edge_columns * h_tiles;
  const bool edge = int(tgid.y) < edge_tiles;
  const int tile = int(tgid.y) - column_blocks - edge_tiles;
  const int w_tiles = gp.edge_columns
      ? (p.outW - gp.edge_columns) / BN
      : c10::metal::ceil_div(p.outW - x_begin, BN);
  const int oh = flat ? 0 : edge ? int(tgid.y) % h_tiles * BN : tile / w_tiles;
  const int x_off = flat ? tile * BN
      : edge             ? p.outW - gp.edge_columns + int(tgid.y) / h_tiles
                         : x_begin + tile % w_tiles * BN;
  const int pitch = flat ? gp.Wp : p.W;
  const int plane = p.H * pitch;
  const int nb = int(tgid.z) / gp.splits;
  device T* in = input + int64_t(nb) * p.C_in * plane +
      int64_t(g) * gp.group_stride + k0 * (IN_NHWC ? 1 : plane);
  const int positions = flat ? (p.outH - 1) * gp.Wp + p.outW : p.outW;
  device T* out = output + int64_t(nb) * p.C_out * p.outH * p.outW +
      (OUT_NHWC ? int64_t(oh) * p.outW * p.C_out + g * og
                : (int64_t(g) * og * p.outH + oh) * p.outW);
  const bool partials = gp.splits > 1 || (flat && gp.Wp != p.outW);
  // The fixed 3x3 kernel seeds the bias into the accumulator; its short K
  // loop makes the epilogue gather measurable.
  const bool seeded = FIXED3 && p.has_bias;
  const bool boundary = conv2d_mpp_boundary(gp, oh, x_off, BN);
  bool exceptional = false;
  if (boundary) {
    exceptional = conv2d_mpp_tile_nonfinite<T, BM>(weight, gp, g, o_off, tid);
  }
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      tensor2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  if (seeded) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
      const int o =
          min(o_off + int(acc.get_multidimensional_index(i)[OUT_NHWC ? 0 : 1]),
              og - 1);
      acc[i] = float(bias[g * og + o]);
    }
  }
  const int out_pitch = flat ? gp.Wp : p.outW;
  const int scratch_pitch = partials ? out_pitch : gp.shift_width;
  device float* part = partial +
      (int64_t(split) * p.N + nb) * p.C_out * p.outH * scratch_pitch +
      (OUT_NHWC ? int64_t(oh) * scratch_pitch * p.C_out + g * og
                : (int64_t(g) * og * p.outH + oh) * scratch_pitch);
  auto dst_view = [&](int limit) {
    return tensor2d(
        out,
        OUT_NHWC ? extents(og, limit) : extents(limit, og),
        strides{1, OUT_NHWC ? p.C_out : p.outH * p.outW});
  };
  auto part_view = [&](int limit) {
    return tensor2df(
        part,
        OUT_NHWC ? extents(og, limit) : extents(limit, og),
        strides{1, OUT_NHWC ? p.C_out : p.outH * scratch_pitch});
  };
  auto dst_tile = [&](auto t, int n) {
    return OUT_NHWC ? t.slice(o_off, n) : t.slice(n, o_off);
  };
  auto res =
      op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
  if IF_CONSTEXPR (IN_NHWC) {
    if (edge) {
      // Positions step down the rows; channels keep the unit stride. The
      // host only sets edge columns for padded copies.
      const int rows = min(BN, p.outH - oh);
      for (int kh = 0; kh < p.kH; ++kh) {
        for (int kw = 0; kw < p.kW; ++kw) {
          tensor2d xt(
              in +
                  (int64_t(oh * p.sH + kh * p.dH) * p.W + x_off * p.sW +
                   kw * p.dW) *
                      p.C_in,
              extents(K, rows),
              strides{1, p.sH * p.W * p.C_in});
          tensor2d wt(
              weight + (int64_t(kh * p.kW + kw) * p.C_out + g * og) * gp.K + k0,
              extents(K, og),
              strides{1, gp.K});
          auto w_tile = wt.slice(0, o_off);
          if IF_CONSTEXPR (OUT_NHWC) {
            op.run(xt, w_tile, acc);
          } else {
            op.run(w_tile, xt, acc);
          }
        }
      }
      device T* column = output + int64_t(nb) * p.C_out * p.outH * p.outW +
          (OUT_NHWC ? (int64_t(oh) * p.outW + x_off) * p.C_out
                    : int64_t(oh) * p.outW + x_off);
      for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        const auto index = acc.get_multidimensional_index(i);
        const int o = g * og + o_off + int(index[OUT_NHWC ? 0 : 1]);
        const int r = int(index[OUT_NHWC ? 1 : 0]);
        if (o < (g + 1) * og && r < rows) {
          column
              [OUT_NHWC ? int64_t(r) * p.outW * p.C_out + o
                        : (int64_t(o) * p.outH + r) * p.outW] =
                  T(acc[i] + (p.has_bias ? float(bias[o]) : 0.0f));
        }
      }
      return;
    }
  }
  const auto run_tap = [&](int kh, int kw, int col, int n0) {
    const int ih = flat ? kh * p.dH : oh * p.sH + kh * p.dH - p.padH;
    const int length =
        min(positions - x_off - n0,
            flat ? plane - ih * pitch - col : (pitch - col + p.sW - 1) / p.sW);
    if (ih < 0 || ih >= p.H || length <= 0) {
      return;
    }
    device T* src = in + int64_t(ih * pitch + col) * (IN_NHWC ? p.C_in : 1);
    tensor2d xt(
        src,
        IN_NHWC ? extents(K, length) : extents(length, K),
        strides{1, IN_NHWC ? (flat ? p.C_in : p.sW * p.C_in) : plane});
    tensor2d wt(
        weight + (int64_t(kh * p.kW + kw) * p.C_out + g * og) * gp.K + k0,
        extents(K, og),
        strides{1, gp.K});
    auto w_tile = wt.slice(0, o_off);
    if IF_CONSTEXPR (OUT_NHWC) {
      op.run(xt, w_tile, acc);
    } else {
      op.run(w_tile, xt, acc);
    }
  };
  // Fixed 3x3 tiles past column 0 read every tap in bounds; they run the taps
  // row by row so that the three taps of an input row read it back to back.
  if (FIXED3 && x_off > 0) {
    for (int kh = 0; kh < p.kH; ++kh) {
      for (int kw = 0; kw < p.kW; ++kw) {
        run_tap(kh, kw, x_off + kw - p.padW, 0);
      }
    }
  } else {
    // A tap starting in the left padding runs from the first in-bounds
    // column, so slot s holds position s + n0 until the accumulator is
    // shifted.
    int shift = 0;
    for (int kw = 0; kw <= p.kW; ++kw) {
      const int tap = kw - p.padW;
      const int iw = flat ? kw * p.dW + x_off
          : gp.parity     ? x_off + (tap & 1) * gp.parity + (tap >> 1)
                          : x_off * p.sW + kw * p.dW - p.padW;
      const int n0 = kw < p.kW && iw < 0 ? (-iw + p.sW - 1) / p.sW : 0;
      if (n0 >= BN) {
        continue;
      }
      if (n0 < shift) {
        const int delta = shift - n0;
        acc.store(
            dst_tile(part_view(min(positions, x_off + BN)), x_off + delta));
        threadgroup_barrier(mem_flags::mem_device);
        acc.load(dst_tile(part_view(min(positions, x_off + BN)), x_off));
        // The next shift or final output store can reuse this scratch.
        threadgroup_barrier(mem_flags::mem_device);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          const auto index = acc.get_multidimensional_index(i);
          if (int(index[OUT_NHWC ? 1 : 0]) < delta) {
            const int o = min(o_off + int(index[OUT_NHWC ? 0 : 1]), og - 1);
            acc[i] = seeded ? float(bias[g * og + o]) : 0.0f;
          }
        }
      }
      shift = n0;
      if (kw == p.kW) {
        break;
      }
      for (int kh = 0; kh < p.kH; ++kh) {
        run_tap(kh, kw, iw + n0 * p.sW, n0);
      }
    }
  }
  if (exceptional) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
      const auto index = acc.get_multidimensional_index(i);
      const int o = o_off + int(index[OUT_NHWC ? 0 : 1]);
      const int x = x_off + int(index[OUT_NHWC ? 1 : 0]);
      if (o < og && x < p.outW && !isnan(acc[i]) &&
          conv2d_mpp_padding_nonfinite(weight, gp, g, o, oh, x, k0, K)) {
        acc[i] = NAN;
      }
    }
  }
  if (partials) {
    acc.store(dst_tile(part_view(positions), x_off));
    return;
  }
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o =
        min(o_off + int(acc.get_multidimensional_index(i)[OUT_NHWC ? 0 : 1]),
            og - 1);
    res[i] =
        T(acc[i] + (p.has_bias && !seeded ? float(bias[g * og + o]) : 0.0f));
  }
  res.store(dst_tile(dst_view(positions), x_off));
}

#define INSTANTIATE_CONV2D_MPP(DT, IN, IN_NHWC, OUT, OUT_NHWC, BM, BN, NSG) \
  INSTANTIATE_CONV2D(                                                       \
      "conv2d_mpp_" #IN "_" #OUT "_b" #BM "x" #BN "_s" #NSG "_" #DT,        \
      conv2d_mpp<DT, IN_NHWC, OUT_NHWC, BM, BN, NSG>);

#define INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, BM, BN, NSG)             \
  INSTANTIATE_CONV2D_MPP(DT, nchw, false, nchw, false, BM, BN, NSG) \
  INSTANTIATE_CONV2D_MPP(DT, nchw, false, nhwc, true, BM, BN, NSG)  \
  INSTANTIATE_CONV2D_MPP(DT, nhwc, true, nchw, false, BM, BN, NSG)  \
  INSTANTIATE_CONV2D_MPP(DT, nhwc, true, nhwc, true, BM, BN, NSG)

static_assert(
    kConv2dFewOutputTile == 8 && kConv2dMinTile == 16 &&
        kConv2dNarrowTile == 32 && kConv2dRowTile == 64 &&
        kConv2dWideTile == 128,
    "Match the MPP tile registration names below");

#define INSTANTIATE_CONV2D_MPP_ALL(DT)          \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 8, 64, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 64, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 64, 4) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 64, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 64, 4) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 64, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 64, 4) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 16, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 32, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 32, 2) \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 128, 2)

INSTANTIATE_CONV2D_MPP_ALL(float)
INSTANTIATE_CONV2D_MPP_ALL(half)
INSTANTIATE_CONV2D_MPP_ALL(bfloat)

#define INSTANTIATE_CONV2D_POINTWISE(BM, BN, NSG, PW)                     \
  INSTANTIATE_CONV2D(                                                     \
      "conv2d_mpp_nchw_nchw_b" #BM "x" #BN "_s" #NSG "_pw" #PW "_bfloat", \
      conv2d_mpp_pointwise<bfloat, BM, BN, NSG, PW == 2>);

INSTANTIATE_CONV2D_POINTWISE(160, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(128, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(128, 32, 2, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 2, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 4, 2)
INSTANTIATE_CONV2D_POINTWISE(64, 32, 4, 2)
INSTANTIATE_CONV2D(
    "conv2d_mpp_nchw_nchw_b32x64_s4_pw2_half",
    conv2d_mpp_pointwise<half, 32, 64, 4, true>);

INSTANTIATE_CONV2D(
    "conv2d_mpp_nchw_nchw_b64x16_s2_pw1_float",
    conv2d_mpp_pointwise<float, 64, 16, 2, false>);
INSTANTIATE_CONV2D(
    "conv2d_mpp_nchw_nchw_b32x32_s2_pw1_float",
    conv2d_mpp_pointwise<float, 32, 32, 2, false>);
INSTANTIATE_CONV2D(
    "conv2d_mpp_nchw_nhwc_b64x16_s2_pw1_float",
    conv2d_mpp_pointwise<float, 64, 16, 2, false, true>);

#define INSTANTIATE_CONV2D_FIXED3(BM, NSG)                    \
  INSTANTIATE_CONV2D(                                         \
      "conv2d_mpp_nchw_nchw_b" #BM "x64_s" #NSG "_k3_bfloat", \
      conv2d_mpp<bfloat, false, false, BM, 64, NSG, true>);

INSTANTIATE_CONV2D_FIXED3(64, 2)
INSTANTIATE_CONV2D_FIXED3(96, 4)

// Square stride-1 convolution through the MPP convolution2d op
// (Conv2dKernel::Tensor): NSG SIMD groups compute OUTPUTS channels x ROWS x
// COLUMNS positions of one group of one image from NHWC input and HWIO
// weights into NHWC output; a group reads its channel slice of each pixel. The
// op zero-fills reads outside its input slice and still multiplies those taps,
// so each tile slices the input at its receptive field clamped to the image
// and padding behaves as zero input. CHANNELS bounds the input channels the op
// specializes for; 0 takes any.
template <
    typename T,
    int CHANNELS,
    int OUTPUTS,
    int ROWS,
    int COLUMNS,
    int NSG,
    int TAPS>
[[max_total_threads_per_threadgroup(NSG * c10::metal::simdgroup_size)]]
kernel void conv2d_tensor(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  using extents = dextents<int32_t, 4>;
  using strides = array<int32_t, 4>;
  using tensor4d = tensor<device T, extents, tensor_inline>;
  constexpr auto desc = convolution2d_descriptor(
      int4(OUTPUTS, COLUMNS, ROWS, 1),
      int4(
          CHANNELS ? CHANNELS : static_cast<int>(dynamic_extent),
          COLUMNS + TAPS - 1,
          ROWS + TAPS - 1,
          1),
      int2(TAPS, TAPS),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1),
      int2(1, 1),
      1,
      true);
  convolution2d<desc, execution_simdgroups<NSG>> op;
  const int cg = p.C_in_per_group, og = p.C_out_per_group;
  const int o_tiles = c10::metal::ceil_div(og, OUTPUTS);
  const int g = int(tgid.x) / o_tiles, o0 = int(tgid.x) % o_tiles * OUTPUTS;
  const int x_tiles = c10::metal::ceil_div(p.outW, COLUMNS);
  const int x0 = int(tgid.y) % x_tiles * COLUMNS;
  const int y0 = int(tgid.y) / x_tiles * ROWS;
  const int n = int(tgid.z);
  const int ix = x0 - p.padW, iy = y0 - p.padH;
  tensor4d x(
      input + int64_t(n) * p.H * p.W * p.C_in + g * cg,
      extents(cg, p.W, p.H, 1),
      strides{1, p.C_in, p.C_in * p.W, p.C_in * p.W * p.H});
  auto xs = x.slice(0, max(ix, 0), max(iy, 0), 0);
  tensor4d w(
      weight + g * og,
      extents(og, cg, TAPS, TAPS),
      strides{1, p.C_out, p.C_out * cg, p.C_out * cg * TAPS});
  auto ws = w.slice(o0, 0, 0, 0);
  // The op centers the taps on each position; a clamped slice starts later.
  op.set_offsets(int2(TAPS / 2 + min(ix, 0), TAPS / 2 + min(iy, 0)));
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(xs),
      decltype(ws),
      float>();
  op.run(xs, ws, acc);
  auto res = op.template get_destination_cooperative_tensor<
      decltype(xs),
      decltype(ws),
      T>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o = min(o0 + int(acc.get_multidimensional_index(i)[0]), og - 1);
    res[i] = T(acc[i] + (p.has_bias ? float(bias[g * og + o]) : 0.0f));
  }
  tensor4d y(
      output + int64_t(n) * p.outH * p.outW * p.C_out + g * og,
      extents(og, p.outW, p.outH, 1),
      strides{1, p.C_out, p.C_out * p.outW, p.C_out * p.outW * p.outH});
  res.store(y.slice(o0, x0, y0, 0));
}

#define INSTANTIATE_CONV2D_TENSOR(                         \
    NAME, DT, CHANNELS, OUTPUTS, ROWS, COLUMNS, NSG, TAPS) \
  INSTANTIATE_CONV2D(                                      \
      "conv2d_tensor_" #NAME #DT,                          \
      conv2d_tensor<DT, CHANNELS, OUTPUTS, ROWS, COLUMNS, NSG, TAPS>);

#define INSTANTIATE_CONV2D_TENSOR_ALL(DT) \
  INSTANTIATE_CONV2D_TENSOR(              \
      ,                                   \
      DT,                                 \
      0,                                  \
      kConv2dTensorOutputs,               \
      kConv2dTensorRows,                  \
      kConv2dTensorColumns,               \
      kConv2dTensorSimdgroups,            \
      3)                                  \
  INSTANTIATE_CONV2D_TENSOR(              \
      stem_,                              \
      DT,                                 \
      kConv2dTensorStemChannels,          \
      kConv2dTensorOutputs,               \
      kConv2dTensorRows,                  \
      kConv2dTensorColumns,               \
      kConv2dTensorSimdgroups,            \
      3)                                  \
  INSTANTIATE_CONV2D_TENSOR(              \
      wide_,                              \
      DT,                                 \
      0,                                  \
      kConv2dTensorWideOutputs,           \
      kConv2dTensorWideSide,              \
      kConv2dTensorWideSide,              \
      kConv2dTensorWideSimdgroups,        \
      3)                                  \
  INSTANTIATE_CONV2D_TENSOR(              \
      k5_,                                \
      DT,                                 \
      0,                                  \
      kConv2dTensorOutputs,               \
      kConv2dTensorRows,                  \
      kConv2dTensorColumns,               \
      kConv2dTensorSimdgroups,            \
      5)

INSTANTIATE_CONV2D_TENSOR_ALL(half)
INSTANTIATE_CONV2D_TENSOR_ALL(bfloat)

template <typename T>
struct Conv2dPatchWindow {
  vec<T, 4> lo[3];
  vec<T, 4> hi[3];
};

void conv2d_patch_load8(
    device const half* src,
    thread half4& lo,
    thread half4& hi) {
  lo = half4(*reinterpret_cast<device const packed_half4*>(src));
  hi = half4(*reinterpret_cast<device const packed_half4*>(src + 4));
}

void conv2d_patch_load8(
    device const bfloat* src,
    thread vec<bfloat, 4>& lo,
    thread vec<bfloat, 4>& hi) {
  // Odd element offsets read element-aligned halves.
  if (reinterpret_cast<ulong>(src) % sizeof(uint) != 0) {
    using P4 = packed_vec<bfloat, 4>;
    lo = vec<bfloat, 4>(*reinterpret_cast<device const P4*>(src));
    hi = vec<bfloat, 4>(*reinterpret_cast<device const P4*>(src + 4));
    return;
  }
  const auto raw = *reinterpret_cast<device const packed_uint4*>(src);
  lo = as_type<vec<bfloat, 4>>(uint2(raw.x, raw.y));
  hi = as_type<vec<bfloat, 4>>(uint2(raw.z, raw.w));
}

template <typename T>
void conv2d_patch_mask(
    thread Conv2dPatchWindow<T>&,
    constant Conv2DParams&,
    int4,
    int,
    false_type) {}

template <typename T>
void conv2d_patch_mask(
    thread Conv2dPatchWindow<T>& window,
    constant Conv2DParams& p,
    int4 bounds,
    int column,
    true_type) {
  const int c = bounds.x, channels = bounds.y, center = bounds.z,
            offset = bounds.w;
  const int plane = p.H * p.W;
  const int col0 = column >= p.W ? column - p.W : column;
  if (c < channels && center + 8 <= plane && offset >= 0 &&
      offset + 8 <= plane && col0 + 8 <= p.W) {
    if (col0 == 0)
      window.lo[0].x = 0;
    if (col0 + 8 == p.W)
      window.hi[2].w = 0;
  } else {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int col = col0 + q >= p.W ? col0 + q - p.W : col0 + q;
      const bool valid = c < channels && center + q < plane &&
          offset + q >= 0 && offset + q < plane;
      if (q < 4) {
        if (!valid || col == 0)
          window.lo[0][q] = 0;
        if (!valid)
          window.lo[1][q] = 0;
        if (!valid || col + 1 == p.W)
          window.lo[2][q] = 0;
      } else {
        if (!valid || col == 0)
          window.hi[0][q - 4] = 0;
        if (!valid)
          window.hi[1][q - 4] = 0;
        if (!valid || col + 1 == p.W)
          window.hi[2][q - 4] = 0;
      }
    }
  }
}

template <typename T, int KS, bool FLAT>
Conv2dPatchWindow<T> conv2d_patch_load_window(
    device const T* src,
    constant Conv2DParams& p,
    int3 position,
    int2 tile,
    true_type) {
  const int c = position.x, ih = position.y, iw = position.z, channels = tile.x;
  const int plane = p.H * p.W;
  Conv2dPatchWindow<T> window = {};
  const int center = iw + 1, length = FLAT ? plane : p.W;
  const int offset = FLAT ? ih * p.W + center : center;
  ++src;
  vec<T, 4> lo(0), hi(0);
  T left = 0, right = 0;
  if (c < channels && (FLAT || (ih >= 0 && ih < p.H))) {
    if (offset >= 0 && offset + 8 <= length) {
      conv2d_patch_load8(src, lo, hi);
    } else {
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        if (offset + q >= 0 && offset + q < length) {
          if (q < 4)
            lo[q] = src[q];
          else
            hi[q - 4] = src[q];
        }
      }
    }
    if (offset > 0 && offset <= length)
      left = src[-1];
    if (offset + 8 >= 0 && offset + 8 < length)
      right = src[8];
  }
  window.lo[0] = vec<T, 4>(left, lo.x, lo.y, lo.z);
  window.hi[0] = vec<T, 4>(lo.w, hi.x, hi.y, hi.z);
  window.lo[1] = lo;
  window.hi[1] = hi;
  window.lo[2] = vec<T, 4>(lo.y, lo.z, lo.w, hi.x);
  window.hi[2] = vec<T, 4>(hi.y, hi.z, hi.w, right);
  conv2d_patch_mask(
      window,
      p,
      int4(c, channels, center, offset),
      tile.y,
      bool_constant<FLAT>{});
  return window;
}

template <typename T, int KS, bool FLAT>
Conv2dPatchWindow<T> conv2d_patch_load_window(
    device const T* src,
    constant Conv2DParams& p,
    int3 position,
    int2 tile,
    false_type) {
  const int c = position.x, ih = position.y, iw = position.z, channels = tile.x;
  vec<T, 4> a(0), b(0), z(0), d(0);
  if (KS != 3 && c < channels && iw + 16 <= p.W) {
    const auto r0 = *reinterpret_cast<device const packed_uint4*>(src);
    const auto r1 = *reinterpret_cast<device const packed_uint4*>(src + 8);
    a = as_type<vec<T, 4>>(uint2(r0.x, r0.y));
    b = as_type<vec<T, 4>>(uint2(r0.z, r0.w));
    z = as_type<vec<T, 4>>(uint2(r1.x, r1.y));
    d = as_type<vec<T, 4>>(uint2(r1.z, r1.w));
  } else if (c < channels && (KS != 3 || (ih >= 0 && ih < p.H))) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      vec<T, 4> v(0);
      if ((KS != 3 || iw + q * 4 >= 0) && iw + q * 4 + 4 <= p.W) {
        v = vec<T, 4>(
            *reinterpret_cast<device const packed_vec<T, 4>*>(src + q * 4));
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if ((KS != 3 || iw + q * 4 + k >= 0) && iw + q * 4 + k < p.W) {
            v[k] = src[q * 4 + k];
          }
        }
      }
      if (q == 0)
        a = v;
      else if (q == 1)
        b = v;
      else if (q == 2)
        z = v;
      else
        d = v;
    }
  }
  const T last = KS == 3 && c < channels && ih >= 0 && ih < p.H &&
          iw + 16 >= 0 && iw + 16 < p.W
      ? src[16]
      : T(0);
  return {
      {vec<T, 4>(a.x, a.z, b.x, b.z),
       vec<T, 4>(a.y, a.w, b.y, b.w),
       vec<T, 4>(a.z, b.x, b.z, z.x)},
      {vec<T, 4>(z.x, z.z, d.x, d.z),
       vec<T, 4>(z.y, z.w, d.y, d.w),
       vec<T, 4>(z.z, d.x, d.z, last)}};
}

// Staged patches (Conv2dKernel::Patch): a threadgroup stages CC input channels
// of KS x KS patches for BN positions in threadgroup memory and multiplies them
// by BM output channels with matmul2d.
template <typename T, int KS, int BM, int BN, int CC, int S, bool FLAT = false>
[[max_total_threads_per_threadgroup(4 * c10::metal::simdgroup_size)]]
kernel void conv2d_patch(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int NSG = 4, taps = KS * KS;
  const int channels = S == 1 && !FLAT ? 16 : p.C_in;
  threadgroup T stage[CC * taps * BN];
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device T, extents, tensor_inline>;
  using shared2d = tensor<threadgroup T, extents, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      false,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      shared2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  const int wt = p.outW / BN + (p.outW % BN != 0),
            oh = FLAT ? 0 : int(gid.y) / wt;
  const int ow0 = FLAT ? int(gid.y) * BN : int(gid.y) % wt * BN;
  const int n = int(gid.z), o0 = int(gid.x) * BM, start_col = ow0 % p.W;
  using idx_t = conditional_t<S == 1, int, int64_t>;
  const idx_t plane = idx_t(p.H) * p.W;
  if IF_CONSTEXPR (S == 1)
    input += int64_t(n) * channels * plane;
  for (int c0 = 0; c0 < channels; c0 += CC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid);
         e < (is_same_v<T, half> ? min(CC, channels - c0) : CC) * KS * (BN / 8);
         e += NSG * c10::metal::simdgroup_size) {
      const int cj = e / (BN / 8) / KS, kh = e / (BN / 8) % KS;
      const int j = e % (BN / 8) * 8, c = c0 + cj;
      const int iw = (ow0 + j) * S - (KS == 3 ? p.padW : 0);
      const int ih = oh * S + kh - (KS == 3 ? p.padH : 0);
      device const T* src = input +
          (S == 1 ? idx_t(c) : idx_t(n) * channels + c) * plane +
          idx_t(ih) * p.W + iw;
      const auto window = conv2d_patch_load_window<T, KS, FLAT>(
          src,
          p,
          int3(c, ih, iw),
          int2(channels, start_col + j),
          bool_constant < S == 1 > {});
#pragma unroll
      for (int kw = 0; kw < KS; ++kw) {
        const uint2 low = as_type<uint2>(window.lo[kw]);
        const uint2 high = as_type<uint2>(window.hi[kw]);
        *reinterpret_cast<threadgroup packed_uint4*>(
            stage + (cj * taps + kh * KS + kw) * BN + j) =
            packed_uint4(low.x, low.y, high.x, high.y);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int K = min(CC, channels - c0) * taps;
    tensor2d w(
        weight + int64_t(o0) * channels * taps + c0 * taps,
        extents(K, min(BM, p.C_out - o0)),
        strides{1, channels * taps});
    shared2d x(stage, extents(BN, K), strides{1, BN});
    op.run(w, x, acc);
  }
  auto res =
      op.template get_destination_cooperative_tensor<tensor2d, shared2d, T>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
    res[i] = T(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
  }
  tensor2d out(
      output + (int64_t(n) * p.C_out * p.outH + oh) * p.outW,
      extents(FLAT ? int(plane) : p.outW, p.C_out),
      strides{1, p.outH * p.outW});
  res.store(out.slice(ow0, o0));
}

#define INSTANTIATE_CONV2D_PATCH(T, KS, BM, BN, CC, S, FLAT) \
  INSTANTIATE_CONV2D(                                        \
      "conv2d_patch_" #KS "_" #BM "x" #BN "_s" #S "_" #T,    \
      conv2d_patch<T, KS, BM, BN, CC, S, FLAT>);

INSTANTIATE_CONV2D_PATCH(bfloat, 1, 128, 32, 128, 2, false)
INSTANTIATE_CONV2D_PATCH(bfloat, 2, 192, 16, 128, 2, false)
INSTANTIATE_CONV2D_PATCH(bfloat, 3, 32, 64, 16, 2, false)
INSTANTIATE_CONV2D_PATCH(bfloat, 3, 256, 16, 64, 2, false)
INSTANTIATE_CONV2D_PATCH(bfloat, 3, 64, 64, 16, 1, false)

INSTANTIATE_CONV2D_PATCH(bfloat, 3, 64, 32, 32, 1, true)
INSTANTIATE_CONV2D_PATCH(bfloat, 3, 128, 32, 32, 1, true)

INSTANTIATE_CONV2D_PATCH(half, 3, 96, 32, 32, 1, true)
INSTANTIATE_CONV2D_PATCH(half, 3, 128, 32, 32, 1, true)
INSTANTIATE_CONV2D_PATCH(half, 3, 192, 32, 32, 1, true)
INSTANTIATE_CONV2D(
    "conv2d_patch_3_128x32_s1_half_c16",
    conv2d_patch<half, 3, 128, 32, 16, 1, true>);

// Stride-2 rows read directly from NCHW input (Conv2dKernel::Stride2): a
// threadgroup computes BM output channels x BN positions of one output row.
template <typename T, int BM, int BN, int NSG>
[[max_total_threads_per_threadgroup(NSG * c10::metal::simdgroup_size)]]
kernel void conv2d_s2(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  using S = conditional_t<is_same_v<T, float>, float, ushort>;
  using P4 = packed_vec<S, 4>;
  // Two buffers, each with two input columns per output before the halo.
  constexpr int KC = kConv2dStride2BaseStageBytes / (2 * (2 * BN) * sizeof(T));
  constexpr int input_columns = 2 * (BN - 1) + kConv2dStride2MaxTapSpan + 1;
  constexpr int VEC = (input_columns + 7) / 8;
  constexpr int HALF = VEC * 4, ROW = 2 * HALF;
  constexpr int NT = NSG * c10::metal::simdgroup_size,
                PER = (KC * VEC + NT - 1) / NT;
  threadgroup T tile[2][KC * ROW];
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device T, extents, tensor_inline>;
  using tensor2dtg = tensor<threadgroup T, extents, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      false,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      tensor2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  const int og = p.C_out_per_group, cg = p.C_in_per_group;
  const int o_tiles = og / BM + (og % BM != 0);
  const int g = int(tgid.x) / o_tiles, o_off = int(tgid.x) % o_tiles * BM;
  const int w_tiles = p.outW / BN + (p.outW % BN != 0);
  const int oh = int(tgid.y) / w_tiles, x_off = int(tgid.y) % w_tiles * BN;
  const int nb = int(tgid.z), plane = p.H * p.W, x0 = x_off * 2 - p.padW;
  device T* in = input + (int64_t(nb) * p.C_in + g * cg) * plane;
  const int cchunks = (cg + KC - 1) / KC;
  const int nchunks = p.kH * cchunks;
  vec<S, 4> ev[PER], od[PER];
  auto fetch = [&](int chunk) {
    const int kh = chunk / cchunks, c0 = chunk % cchunks * KC;
    const int ih = oh * p.sH + kh * p.dH - p.padH;
    for (int i = 0; i < PER; ++i) {
      const int e = int(tid) + i * NT;
      const int k = e / VEC, j = e % VEC, x = x0 + j * 8;
      vec<S, 4> a = 0, b = 0;
      if (e < KC * VEC && c0 + k < cg && ih >= 0 && ih < p.H) {
        device const T* row = in + int64_t(c0) * plane + int64_t(ih) * p.W;
        device const T* src = row + int64_t(k) * plane + x;
        if (x >= 0 && x + 8 <= p.W) {
          a = vec<S, 4>(*reinterpret_cast<device const P4*>(src));
          b = vec<S, 4>(*reinterpret_cast<device const P4*>(src + 4));
        } else {
          for (int q = 0; q < 4; ++q) {
            if (x + q >= 0 && x + q < p.W) {
              a[q] = *reinterpret_cast<device const S*>(src + q);
            }
            if (x + q + 4 >= 0 && x + q + 4 < p.W) {
              b[q] = *reinterpret_cast<device const S*>(src + q + 4);
            }
          }
        }
      }
      ev[i] = vec<S, 4>(a.x, a.z, b.x, b.z);
      od[i] = vec<S, 4>(a.y, a.w, b.y, b.w);
    }
  };
  auto commit = [&](int buf) {
    threadgroup T* t = tile[buf];
    for (int i = 0; i < PER; ++i) {
      const int e = int(tid) + i * NT;
      if (e < KC * VEC) {
        const int k = e / VEC, j = e % VEC;
        *reinterpret_cast<threadgroup P4*>(t + k * ROW + j * 4) = P4(ev[i]);
        *reinterpret_cast<threadgroup P4*>(t + k * ROW + HALF + j * 4) =
            P4(od[i]);
      }
    }
  };
  if (nchunks > 0) {
    fetch(0);
    commit(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int buf = chunk & 1;
    if (chunk + 1 < nchunks) {
      fetch(chunk + 1);
    }
    const int kh = chunk / cchunks, c0 = chunk % cchunks * KC;
    const int kc = min(KC, cg - c0);
    for (int kw = 0; kw < p.kW; ++kw) {
      const int shift = kw * p.dW, off = (shift & 1) * HALF + (shift >> 1);
      tensor2dtg xt(tile[buf] + off, extents(BN, kc), strides{1, ROW});
      tensor2d wt(
          weight + (int64_t(kh * p.kW + kw) * p.C_out + g * og + o_off) * cg +
              c0,
          extents(kc, min(BM, og - o_off)),
          strides{1, cg});
      op.run(wt, xt, acc);
    }
    if (chunk + 1 < nchunks) {
      commit(buf ^ 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device T* out = output + int64_t(nb) * p.C_out * p.outH * p.outW +
      (int64_t(g) * og * p.outH + oh) * p.outW;
  auto res =
      op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o =
        min(o_off + int(acc.get_multidimensional_index(i)[1]), og - 1);
    res[i] = T(acc[i] + (p.has_bias ? float(bias[g * og + o]) : 0.0f));
  }
  tensor2d dst(out, extents(p.outW, og), strides{1, p.outH * p.outW});
  res.store(dst.slice(x_off, o_off));
}

#define INSTANTIATE_CONV2D_S2(DT, BM, BN, NSG)     \
  INSTANTIATE_CONV2D(                              \
      "conv2d_s2_b" #BM "x" #BN "_s" #NSG "_" #DT, \
      conv2d_s2<DT, BM, BN, NSG>);

#define INSTANTIATE_CONV2D_S2_ALL(DT)   \
  INSTANTIATE_CONV2D_S2(DT, 16, 32, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 16, 64, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 32, 32, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 32, 64, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 32, 128, 4) \
  INSTANTIATE_CONV2D_S2(DT, 64, 32, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 64, 64, 4)  \
  INSTANTIATE_CONV2D_S2(DT, 128, 32, 8) \
  INSTANTIATE_CONV2D_S2(DT, 128, 64, 8)

INSTANTIATE_CONV2D_S2_ALL(float)
INSTANTIATE_CONV2D_S2_ALL(half)
INSTANTIATE_CONV2D_S2_ALL(bfloat)
INSTANTIATE_CONV2D_S2(bfloat, 96, 32, 4)

#endif // C10_METAL_HAS_MPP

template <typename T, int KW, int S, int RH, int RW>
void conv2d_depthwise_accumulate(
    device const T* in,
    device const T* wt,
    constant Conv2DParams& p,
    int2 origin,
    thread float (&acc)[RH][RW],
    true_type) {
  const int ow0 = origin.x, oh0 = origin.y;
  for (int r = 0; r < RH; ++r) {
    for (int kh = 0; kh < p.kH; ++kh) {
      const int ih = (oh0 + r) * p.sH + kh * p.dH - p.padH;
      for (int kw = 0; kw < p.kW; ++kw) {
        const float w = float(wt[kh * p.kW + kw]);
        for (int j = 0; j < RW; ++j) {
          const int iw = (ow0 + j) * p.sW + kw * p.dW - p.padW;
          const float x = ih >= 0 && ih < p.H && iw >= 0 && iw < p.W
              ? float(in[int64_t(ih) * p.W + iw])
              : 0.0f;
          acc[r][j] += x * w;
        }
      }
    }
  }
}

template <typename T, int KW, int S, int RH, int RW>
void conv2d_depthwise_accumulate(
    device const T* in,
    device const T* wt,
    constant Conv2DParams& p,
    int2 origin,
    thread float (&acc)[RH][RW],
    false_type) {
  const int ow0 = origin.x, oh0 = origin.y;
  constexpr int LW = (RW - 1) * S + KW;
  for (int rr = 0; rr < (RH - 1) * S + KW; ++rr) {
    const int ih = oh0 * S - p.padH + rr;
    const bool valid_row = ih >= 0 && ih < p.H;
    device const T* row = valid_row ? in + int64_t(ih) * p.W : in;
    const int iw0 = ow0 * S - p.padW;
    float win[LW];
    for (int t = 0; t < LW; ++t) {
      win[t] = valid_row && iw0 + t >= 0 && iw0 + t < p.W ? float(row[iw0 + t])
                                                          : 0.0f;
    }
    for (int r = 0; r < RH; ++r) {
      const int kh = rr - r * S;
      if (kh < 0 || kh >= KW) {
        continue;
      }
      for (int kw = 0; kw < KW; ++kw) {
        const float w = float(wt[kh * KW + kw]);
        for (int j = 0; j < RW; ++j) {
          acc[r][j] += win[j * S + kw] * w;
        }
      }
    }
  }
}

// Depthwise convolution: each thread computes kConv2dDepthwiseRows x
// conv2d_depthwise_columns(KW) outputs of one (batch, channel) plane. KW = 0
// reads the filter shape at runtime; FULL tiles skip column bounds checks.
template <typename T, int KW, int S, bool FULL>
kernel void conv2d_depthwise(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr int RW = conv2d_depthwise_columns(KW);
  constexpr int RH = kConv2dDepthwiseRows;
  const int ow0 = int(pos.x) * RW, oh0 = int(pos.y) * RH, nc = int(pos.z);
  const int c = nc % p.C_in, mult = p.C_out / p.C_in;
  device const T* in = input + int64_t(nc) * p.H * p.W;
  for (int m = 0; m < mult; ++m) {
    const int o = c * mult + m;
    device const T* wt = weight + int64_t(o) * p.kH * p.kW;
    float acc[RH][RW];
    for (int r = 0; r < RH; ++r) {
      for (int j = 0; j < RW; ++j) {
        acc[r][j] = p.has_bias ? float(bias[o]) : 0.0f;
      }
    }
    conv2d_depthwise_accumulate<T, KW, S>(
        in, wt, p, int2(ow0, oh0), acc, bool_constant < KW == 0 > {});
    for (int r = 0; r < RH && oh0 + r < p.outH; ++r) {
      device T* out = output +
          ((int64_t(nc / p.C_in) * p.C_out + o) * p.outH + oh0 + r) * p.outW;
      for (int j = 0; j < RW && (FULL || ow0 + j < p.outW); ++j) {
        out[ow0 + j] = T(acc[r][j]);
      }
    }
  }
}

#define INSTANTIATE_CONV2D_DEPTHWISE(DT, KW, S, FULL)   \
  INSTANTIATE_CONV2D(                                   \
      "conv2d_depthwise_" #KW "_" #S "_" #DT "_" #FULL, \
      conv2d_depthwise<DT, KW, S, FULL>);

#define INSTANTIATE_CONV2D_DEPTHWISE_ALL(DT)    \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 0, 0, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 3, 1, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 3, 2, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 5, 1, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 5, 2, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 7, 1, false) \
  INSTANTIATE_CONV2D_DEPTHWISE(DT, 7, 2, false)

INSTANTIATE_CONV2D_DEPTHWISE_ALL(float)
INSTANTIATE_CONV2D_DEPTHWISE_ALL(half)
INSTANTIATE_CONV2D_DEPTHWISE_ALL(bfloat)

INSTANTIATE_CONV2D_DEPTHWISE(bfloat, 3, 1, true)

template <int S>
struct Conv2dTinyWindow {
  array<float4, 2 * S> values;
  float left, right;
};

template <typename T>
array<float4, 2> conv2d_tiny_load8(device const T* src) {
  using P4 = packed_vec<T, 4>;
  return {
      float4(*reinterpret_cast<device const P4*>(src)),
      float4(*reinterpret_cast<device const P4*>(src + 4))};
}

array<float4, 2> conv2d_tiny_load8(device const bfloat* src) {
  const auto raw = *reinterpret_cast<device const packed_uint4*>(src);
  return {
      float4(as_type<bfloat4>(uint2(raw.x, raw.y))),
      float4(as_type<bfloat4>(uint2(raw.z, raw.w)))};
}

template <int KW, typename T>
Conv2dTinyWindow<1> conv2d_tiny_load_window(
    device const T* src,
    int ow,
    int width,
    false_type) {
  return {
      conv2d_tiny_load8(src),
      KW == 3 && ow > 0 ? float(src[-1]) : 0.0f,
      KW == 3 && ow + 8 < width ? float(src[8]) : 0.0f};
}

template <int KW, typename T>
Conv2dTinyWindow<2> conv2d_tiny_load_window(
    device const T* src,
    int ow,
    int,
    true_type) {
  const auto first = conv2d_tiny_load8(src),
             second = conv2d_tiny_load8(src + 8);
  return {
      {first[0], first[1], second[0], second[1]},
      ow > 0 ? float(src[-1]) : 0.0f,
      0.0f};
}

template <int KW>
array<float4, 2> conv2d_tiny_gather(
    thread const Conv2dTinyWindow<1>& window,
    int kw) {
  const float4 a = window.values[0], z = window.values[1];
  return {
      KW == 1 || kw == 1 ? a
          : kw == 0      ? float4(window.left, a.x, a.y, a.z)
                         : float4(a.y, a.z, a.w, z.x),
      KW == 1 || kw == 1 ? z
          : kw == 0      ? float4(a.w, z.x, z.y, z.z)
                         : float4(z.y, z.z, z.w, window.right)};
}

template <int KW>
array<float4, 2> conv2d_tiny_gather(
    thread const Conv2dTinyWindow<2>& window,
    int kw) {
  const float4 a = window.values[0], z = window.values[1], b = window.values[2],
               d = window.values[3];
  return {
      kw == 0       ? float4(window.left, a.y, a.w, z.y)
          : kw == 1 ? float4(a.x, a.z, z.x, z.z)
                    : float4(a.y, a.w, z.y, z.w),
      kw == 0       ? float4(z.w, b.y, b.w, d.y)
          : kw == 1 ? float4(b.x, b.z, d.x, d.z)
                    : float4(b.y, b.w, d.y, d.w)};
}

template <typename T>
void conv2d_tiny_store8(device T* dst, float4 lo, float4 hi) {
  using P4 = packed_vec<T, 4>;
  *reinterpret_cast<device P4*>(dst) = P4(vec<T, 4>(lo));
  *reinterpret_cast<device P4*>(dst + 4) = P4(vec<T, 4>(hi));
}

void conv2d_tiny_store8(device bfloat* dst, float4 lo, float4 hi) {
  const uint2 a = as_type<uint2>(bfloat4(lo));
  const uint2 z = as_type<uint2>(bfloat4(hi));
  *reinterpret_cast<device packed_uint4*>(dst) =
      packed_uint4(a.x, a.y, z.x, z.y);
}

template <typename T, int C, int KW>
void conv2d_tiny_accumulate(
    device const T* weight,
    float4 low,
    float4 high,
    thread float4 (&lo)[4],
    thread float4 (&hi)[4]) {
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const float wt = float(weight[o * C * KW * KW]);
    lo[o] = fma(low, wt, lo[o]);
    hi[o] = fma(high, wt, hi[o]);
  }
}

// Stems with C input channels (Conv2dKernel::Tiny): each thread accumulates
// kConv2dTinyPositionsPerThread positions x kConv2dTinyChannelsPerThread output
// channels over all C x KW x KW taps with scalar FMAs.
template <typename T, int C, int KW, int S>
kernel void conv2d_tiny(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int ow = KW == 3 ? int(pos.x) * 8 : 0;
  const int oh = KW == 3 ? int(pos.y) : 0;
  const int o0 = KW == 3 ? int(pos.z) % (p.C_out / 4) * 4 : int(pos.y) * 4;
  const int n = KW == 3 ? int(pos.z) / (p.C_out / 4) : int(pos.z);
  const int64_t plane = int64_t(p.H) * p.W;
  const int64_t outplane = S == 1 ? plane : int64_t(p.outH) * p.outW;
  const int64_t position =
      KW == 3 ? int64_t(oh) * (S == 1 ? p.W : p.outW) + ow : int64_t(pos.x) * 8;
  float4 lo[4] = {}, hi[4] = {};
#pragma unroll
  for (int c = 0; c < C; ++c) {
#pragma unroll
    for (int kh = 0; kh < KW; ++kh) {
      const int ih = oh * S + kh - KW / 2;
      device const T* row_weight = weight + ((o0 * C + c) * KW + kh) * KW;
      if (KW == 3 && (ih < 0 || ih >= p.H)) {
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
          conv2d_tiny_accumulate<T, C, KW>(
              row_weight + kw, float4(0), float4(0), lo, hi);
        }
        continue;
      }
      const device T* src = input + (int64_t(n) * C + c) * plane +
          (KW == 3 ? int64_t(ih) * p.W + ow * S : position);
      const auto window = conv2d_tiny_load_window<KW>(
          src, ow, p.W, bool_constant < S == 2 > {});
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        const auto values = conv2d_tiny_gather<KW>(window, kw);
        conv2d_tiny_accumulate<T, C, KW>(
            row_weight + kw, values[0], values[1], lo, hi);
      }
    }
  }
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const float bv = p.has_bias ? float(bias[o0 + o]) : 0.0f;
    conv2d_tiny_store8(
        output + (int64_t(n) * p.C_out + o0 + o) * outplane + position,
        lo[o] + bv,
        hi[o] + bv);
  }
}

template <typename T, int C>
float4 conv2d_tiny_load4(device const T* source, int, false_type) {
  return float4(*reinterpret_cast<device const packed_vec<T, 4>*>(source));
}

template <typename T, int C>
float4 conv2d_tiny_load4(device const T* source, int channel, true_type) {
  array<float4, C> pixels;
#pragma unroll
  for (int i = 0; i < C; ++i) {
    pixels[i] = float4(
        *reinterpret_cast<device const packed_vec<T, 4>*>(source + i * 4));
  }
  float4 values;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int index = i * C + channel;
    values[i] = pixels[index / 4][index % 4];
  }
  return values;
}

template <typename T, int C, bool CHANNELS_LAST>
void conv2d_tiny_bounded_window(
    device const T* input,
    constant Conv2DParams& p,
    int n,
    int row,
    int channel,
    int column,
    thread array<float4, 3>& values,
    bool_constant<CHANNELS_LAST> layout) {
  const int64_t plane = int64_t(p.H) * p.W;
  const int stride = CHANNELS_LAST ? C : 1;
  const int c = CHANNELS_LAST ? channel : 0;
  device const T* source = input +
      (CHANNELS_LAST ? (int64_t(n) * plane + int64_t(row) * p.W) * C
                     : (int64_t(n) * C + channel) * plane + int64_t(row) * p.W);
#pragma unroll
  for (int block = 0; block < 3; ++block) {
    const int first = column - p.padW + block * 4;
    if (first >= 0 && first + 3 < p.W) {
      values[block] = conv2d_tiny_load4<T, C>(
          source + int64_t(first) * stride, channel, layout);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int x = first + i;
        values[block][i] =
            x >= 0 && x < p.W ? float(source[int64_t(x) * stride + c]) : 0.0f;
      }
    }
  }
}

float4 conv2d_tiny_bounded_gather(float4 a, float4 b, int tap) {
  return tap == 0 ? a
      : tap == 1  ? float4(a.y, a.z, a.w, b.x)
      : tap == 2  ? float4(a.z, a.w, b.x, b.y)
                  : float4(a.w, b.x, b.y, b.z);
}

template <typename T>
void conv2d_tiny_bounded_store(
    device T* output,
    constant Conv2DParams& p,
    int n,
    int o0,
    int row,
    int column,
    thread const float4 (&lo)[kConv2dTinyChannelsPerThread],
    thread const float4 (&hi)[kConv2dTinyChannelsPerThread],
    false_type) {
  const int64_t plane = int64_t(p.outH) * p.outW;
#pragma unroll
  for (int o = 0; o < kConv2dTinyChannelsPerThread; ++o) {
    device T* destination = output + (int64_t(n) * p.C_out + o0 + o) * plane +
        int64_t(row) * p.outW + column;
    if (column + kConv2dTinyPositionsPerThread <= p.outW) {
      conv2d_tiny_store8(destination, lo[o], hi[o]);
    } else {
#pragma unroll
      for (int i = 0; i < kConv2dTinyPositionsPerThread; ++i) {
        if (column + i < p.outW) {
          destination[i] = T(i < 4 ? lo[o][i] : hi[o][i - 4]);
        }
      }
    }
  }
}

template <typename T>
void conv2d_tiny_bounded_store(
    device T* output,
    constant Conv2DParams& p,
    int n,
    int o0,
    int row,
    int column,
    thread const float4 (&lo)[kConv2dTinyChannelsPerThread],
    thread const float4 (&hi)[kConv2dTinyChannelsPerThread],
    true_type) {
  device T* destination =
      output + ((int64_t(n) * p.outH + row) * p.outW + column) * p.C_out + o0;
#pragma unroll
  for (int i = 0; i < kConv2dTinyPositionsPerThread; ++i) {
    if (column + i < p.outW) {
      float4 values;
#pragma unroll
      for (int o = 0; o < kConv2dTinyChannelsPerThread; ++o) {
        values[o] = i < 4 ? lo[o][i] : hi[o][i - 4];
      }
      *reinterpret_cast<device packed_vec<T, 4>*>(destination + i * p.C_out) =
          packed_vec<T, 4>(vec<T, 4>(values));
    }
  }
}

// Same eight-position/four-output accumulator; row tails need bounded IO.
template <typename T, int C, int KW>
kernel void conv2d_tiny_bounded(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2dTinyParams& params [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  static_assert(
      KW == 3 || KW == 4, "Bounded Tiny requires a 3x3 or 4x4 filter");
  constant Conv2DParams& p = params.conv;
  const int columns = p.outW / kConv2dTinyPositionsPerThread +
      (p.outW % kConv2dTinyPositionsPerThread != 0);
  const int ow = int(pos.x) % columns * kConv2dTinyPositionsPerThread,
            oh = int(pos.x) / columns;
  const int o0 = int(pos.y) * kConv2dTinyChannelsPerThread, n = int(pos.z);
  const int channel_stride = params.weight_channels_last ? 1 : KW * KW;
  const int tap_stride = params.weight_channels_last ? C : 1;
  float4 lo[kConv2dTinyChannelsPerThread] = {},
         hi[kConv2dTinyChannelsPerThread] = {};
#pragma unroll
  for (int c = 0; c < C; ++c) {
#pragma unroll
    for (int kh = 0; kh < KW; ++kh) {
      const int ih = oh + kh - p.padH;
      device const T* row_weight =
          weight + o0 * C * KW * KW + c * channel_stride + kh * KW * tap_stride;
      if (ih < 0 || ih >= p.H) {
#pragma unroll
        for (int kw = 0; kw < KW; ++kw) {
          conv2d_tiny_accumulate<T, C, KW>(
              row_weight + kw * tap_stride, float4(0), float4(0), lo, hi);
        }
        continue;
      }
      array<float4, 3> window;
      if (params.input_channels_last) {
        conv2d_tiny_bounded_window<T, C>(
            input, p, n, ih, c, ow, window, true_type{});
      } else {
        conv2d_tiny_bounded_window<T, C>(
            input, p, n, ih, c, ow, window, false_type{});
      }
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        conv2d_tiny_accumulate<T, C, KW>(
            row_weight + kw * tap_stride,
            conv2d_tiny_bounded_gather(window[0], window[1], kw),
            conv2d_tiny_bounded_gather(window[1], window[2], kw),
            lo,
            hi);
      }
    }
  }
#pragma unroll
  for (int o = 0; o < kConv2dTinyChannelsPerThread; ++o) {
    const float value = p.has_bias ? float(bias[o0 + o]) : 0.0f;
    lo[o] += value;
    hi[o] += value;
  }
  if (params.output_channels_last) {
    conv2d_tiny_bounded_store(output, p, n, o0, oh, ow, lo, hi, true_type{});
  } else {
    conv2d_tiny_bounded_store(output, p, n, o0, oh, ow, lo, hi, false_type{});
  }
}

#define INSTANTIATE_CONV2D_TINY(DT)                                       \
  INSTANTIATE_CONV2D("conv2d_tiny_1_3_1_" #DT, conv2d_tiny<DT, 1, 3, 1>); \
  INSTANTIATE_CONV2D("conv2d_tiny_3_3_1_" #DT, conv2d_tiny<DT, 3, 3, 1>);

INSTANTIATE_CONV2D_TINY(float)
INSTANTIATE_CONV2D_TINY(half)
INSTANTIATE_CONV2D_TINY(bfloat)
INSTANTIATE_CONV2D("conv2d_tiny_4_1_1_bfloat", conv2d_tiny<bfloat, 4, 1, 1>);
INSTANTIATE_CONV2D("conv2d_tiny_4_3_1_float", conv2d_tiny<float, 4, 3, 1>);
INSTANTIATE_CONV2D("conv2d_tiny_5_3_1_float", conv2d_tiny<float, 5, 3, 1>);
INSTANTIATE_CONV2D("conv2d_tiny_6_3_1_float", conv2d_tiny<float, 6, 3, 1>);
INSTANTIATE_CONV2D("conv2d_tiny_3_3_2_bfloat", conv2d_tiny<bfloat, 3, 3, 2>);

INSTANTIATE_CONV2D(
    "conv2d_tiny_bounded_2_4_float",
    conv2d_tiny_bounded<float, 2, 4>);
INSTANTIATE_CONV2D(
    "conv2d_tiny_bounded_3_4_float",
    conv2d_tiny_bounded<float, 3, 4>);
INSTANTIATE_CONV2D(
    "conv2d_tiny_bounded_2_3_half",
    conv2d_tiny_bounded<half, 2, 3>);

template <typename T, int KW, int S, int RH, int RW>
void conv2d_o1_accumulate_row(
    device const T* row,
    device const T* wt,
    constant Conv2DParams& p,
    int3 position,
    bool valid_row,
    thread float (&acc)[RH][RW],
    true_type) {
  const int c = position.x, rr = position.y, ow0 = position.z;
  device const T* w = wt + (int64_t(c) * p.kH + rr) * p.kW;
  for (int kw = 0; kw < p.kW; ++kw) {
    const float wv = float(w[kw]);
    for (int j = 0; j < RW; ++j) {
      const int iw = (ow0 + j) * p.sW + kw * p.dW - p.padW;
      const float x = valid_row && iw >= 0 && iw < p.W ? float(row[iw]) : 0.0f;
      acc[0][j] += x * wv;
    }
  }
}

template <typename T, int KW, int S, int RH, int RW>
void conv2d_o1_accumulate_row(
    device const T* row,
    device const T* wt,
    constant Conv2DParams& p,
    int3 position,
    bool valid_row,
    thread float (&acc)[RH][RW],
    false_type) {
  const int c = position.x, rr = position.y, ow0 = position.z;
  using P4 = packed_vec<T, 4>;
  constexpr int LW = (RW - 1) * S + KW, LV = (LW + 3) / 4 * 4;
  const int iw0 = ow0 * S - p.padW;
  float win[LV];
  if (valid_row && iw0 >= 0 && iw0 <= p.W - LV) {
    for (int t = 0; t < LV; t += 4) {
      const P4 v = *reinterpret_cast<device const P4*>(row + iw0 + t);
      for (int q = 0; q < 4; ++q) {
        win[t + q] = float(v[q]);
      }
    }
  } else {
    for (int t = 0; t < LW; ++t) {
      const int iw = iw0 + t;
      win[t] = valid_row && iw >= 0 && iw < p.W ? float(row[iw]) : 0.0f;
    }
  }
  for (int r = 0; r < RH; ++r) {
    const int kh = rr - r;
    if (RH > 1 && (kh < 0 || kh >= KW)) {
      continue;
    }
    device const T* w =
        wt + (RH > 1 ? (c * KW + kh) * KW : (int64_t(c) * p.kH + kh) * p.kW);
    for (int kw = 0; kw < KW; ++kw) {
      const float wv = float(w[kw]);
      for (int j = 0; j < RW; ++j) {
        acc[r][j] += win[j * S + kw] * wv;
      }
    }
  }
}

template <typename T, int KW, int S, int RH, int RW>
void conv2d_o1_accumulate(
    device const T* in,
    device const T* wt,
    constant Conv2DParams& p,
    int2 position,
    uint first,
    uint step,
    thread float (&acc)[RH][RW]) {
  const int oh = position.x, ow0 = position.y;
  for (uint c = first; c < uint(p.C_in_per_group); c += step) {
    for (int rr = 0; rr < (RH == 1 ? p.kH : RH + KW - 1); ++rr) {
      const int ih = (RH > 1 ? oh + rr : oh * p.sH + rr * p.dH) - p.padH;
      const bool valid_row = ih >= 0 && ih < p.H;
      device const T* row = valid_row ? in + (int64_t(c) * p.H + ih) * p.W : in;
      conv2d_o1_accumulate_row<T, KW, S>(
          row,
          wt,
          p,
          int3(c, rr, ow0),
          valid_row,
          acc,
          bool_constant < KW == 0 > {});
    }
  }
}

template <int CP>
float conv2d_o1_replay_limit(constant Conv2DParams& p) {
  constexpr uint stages = __builtin_ctz(CP);
  constexpr uint round_limit = uint(1) << (numeric_limits<float>::digits - 1);
  const uint taps = uint(p.kH) * p.kW, terms = uint(p.C_in_per_group) * taps;
  if (terms >= round_limit)
    return INFINITY;
  const uint partial =
      (uint(p.C_in_per_group) / CP + (p.C_in_per_group % CP != 0)) * taps;
  if (2 * (terms - partial) < CP + 2 * stages)
    return INFINITY;
  // Above this magnitude, the direct rounding budget covers all result
  // flushing.
  return 4 * numeric_limits<float>::min() / numeric_limits<float>::epsilon();
}

// One output channel per group (Conv2dKernel::SingleOutput): each thread
// computes an RH x RW output patch over all input channels of its group; CP > 1
// splits the channels over CP lanes of a SIMD group and reduces with shuffles.
template <typename T, int KW, int S, int RH, int RW, int CP = 1>
[[max_total_threads_per_threadgroup(
    kConv2dO1Simdgroups * c10::metal::simdgroup_size)]]
kernel void conv2d_o1(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  static_assert(
      CP == 1 || CP == 2 || CP == 4 || CP == 8,
      "Expected one, two, four, or eight channel partitions");
  constexpr uint spatial_lanes = c10::metal::simdgroup_size / CP;
  const uint logical = CP == 1
      ? pos.x
      : pos.x / c10::metal::simdgroup_size * spatial_lanes +
          lane % spatial_lanes;
  const int cg = p.C_in_per_group, columns = p.outW / RW + (p.outW % RW != 0);
  const int ow0 = int(logical) % columns * RW, oh = int(logical) / columns * RH;
  if (oh >= p.outH)
    return;
  const int n = int(pos.y) / p.C_out, g = int(pos.y) % p.C_out;
  device const T* in = input + (int64_t(n) * p.C_in + g * cg) * p.H * p.W;
  device const T* wt =
      weight + int64_t(g) * cg * (RH > 1 ? KW * KW : int64_t(p.kH) * p.kW);
  float acc[RH][RW] = {};
  conv2d_o1_accumulate<T, KW, S>(
      in, wt, p, int2(oh, ow0), CP == 1 ? 0 : lane / spatial_lanes, CP, acc);
  for (uint distance = spatial_lanes; distance < c10::metal::simdgroup_size;
       distance *= 2) {
    for (int r = 0; r < RH; ++r) {
      for (int j = 0; j < RW; ++j) {
        acc[r][j] += simd_shuffle_xor(acc[r][j], distance);
      }
    }
  }
  if (CP > 1 && lane >= spatial_lanes)
    return;
  float replay_limit = -1.0f;
  if IF_CONSTEXPR (CP > 1 && is_same_v<T, bfloat>)
    replay_limit = conv2d_o1_replay_limit<CP>(p);
  bool replay = false;
  for (int r = 0; r < RH; ++r) {
    for (int j = 0; j < RW; ++j) {
      replay |= CP > 1 &&
          (!isfinite(acc[r][j]) ||
           (is_same_v<T, bfloat> && oh + r < p.outH && ow0 + j < p.outW &&
            abs(acc[r][j]) <= replay_limit));
    }
  }
  if (replay) {
    for (int r = 0; r < RH; ++r) {
      for (int j = 0; j < RW; ++j) {
        acc[r][j] = 0.0f;
      }
    }
    conv2d_o1_accumulate<T, KW, S>(in, wt, p, int2(oh, ow0), 0, 1, acc);
  }
  const float b = p.has_bias ? float(bias[g]) : 0.0f;
  for (int r = 0; r < RH && oh + r < p.outH; ++r) {
    device T* out =
        output + ((int64_t(n) * p.C_out + g) * p.outH + oh + r) * p.outW;
    for (int j = 0; j < RW && ow0 + j < p.outW; ++j) {
      out[ow0 + j] = T(acc[r][j] + b);
    }
  }
}

#define INSTANTIATE_CONV2D_O1(DT, KW, S, RH, RW)       \
  INSTANTIATE_CONV2D(                                  \
      "conv2d_o1_" #KW "_" #S "_" #RH "_" #RW "_" #DT, \
      conv2d_o1<DT, KW, S, RH, RW>);

// Generic taps, 3x3 rows, paired 2x4 patches and channel-partitioned rows for
// every dtype; the host appends _cp{kConv2dO1ChannelPartitions}.
#define INSTANTIATE_CONV2D_O1_ALL(DT)   \
  INSTANTIATE_CONV2D_O1(DT, 0, 0, 1, 8) \
  INSTANTIATE_CONV2D_O1(DT, 3, 1, 1, 8) \
  INSTANTIATE_CONV2D_O1(DT, 3, 1, 2, 4) \
  INSTANTIATE_CONV2D(                   \
      "conv2d_o1_3_1_1_4_" #DT "_cp8",  \
      conv2d_o1<DT, 3, 1, 1, 4, kConv2dO1ChannelPartitions>);

static_assert(kConv2dO1ChannelPartitions == 8, "Match the _cp8 names above");

INSTANTIATE_CONV2D_O1_ALL(float)
INSTANTIATE_CONV2D_O1_ALL(half)
INSTANTIATE_CONV2D_O1_ALL(bfloat)
INSTANTIATE_CONV2D_O1(float, 3, 2, 1, 1)
INSTANTIATE_CONV2D_O1(bfloat, 1, 1, 1, 16)
INSTANTIATE_CONV2D_O1(bfloat, 3, 1, 1, 16)

// Sums the float32 split-K partials of conv2d_mpp and adds bias.
template <typename T>
kernel void conv2d_split_reduce(
    device const float* partial [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2dSplitParams& p [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int C = p.channels, outH = p.height, outW = p.width, pitch = p.pitch;
  const int e = int(pos.x);
  const int o = p.channels_last ? e % C : e / (outH * outW);
  const int r = p.channels_last ? e / C / outW : e / outW % outH,
            c = p.channels_last ? e / C % outW : e % outW;
  const int64_t image = int64_t(C) * outH * pitch;
  const int64_t idx = p.channels_last ? (int64_t(r) * pitch + c) * C + o
                                      : (int64_t(o) * outH + r) * pitch + c;
  float v = p.has_bias ? float(bias[o]) : 0.0f;
  for (int s = 0; s < p.splits; ++s) {
    v += partial[(int64_t(s) * p.batch + pos.y) * image + idx];
  }
  output[pos.y * int64_t(C) * outH * outW + e] = T(v);
}

INSTANTIATE_CONV2D("conv2d_split_reduce_float", conv2d_split_reduce<float>);
INSTANTIATE_CONV2D("conv2d_split_reduce_half", conv2d_split_reduce<half>);
INSTANTIATE_CONV2D("conv2d_split_reduce_bfloat", conv2d_split_reduce<bfloat>);

// output[g][n][h][w][c] = input[n][h - pad][w - pad][g * group_channels + c]
// for channels-last input, zero in the padding. pos.x walks the channels of a
// padded row, pos.y its padded rows and pos.z the (group, image) pairs.
template <typename T>
kernel void conv2d_group_major(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant Conv2dGroupMajorParams& p [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr int V = kConv2dGroupMajorChannelsPerThread;
  const int vectors = p.group_channels / V + (p.group_channels % V != 0);
  const int padded_w = p.width + 2 * p.pad_width;
  const int wp = int(pos.x) / vectors, c = int(pos.x) % vectors * V;
  const int h = int(pos.y) - p.pad_height, w = wp - p.pad_width;
  const int g = int(pos.z) / p.batch, n = int(pos.z) % p.batch;
  const bool inside = h >= 0 && h < p.height && w >= 0 && w < p.width;
  device T* dst = output +
      ((int64_t(pos.z) * (p.height + 2 * p.pad_height) + pos.y) * padded_w +
       wp) *
          p.group_channels +
      c;
  device const T* src = input +
      ((int64_t(n) * p.height + h) * p.width + w) * p.channels +
      g * p.group_channels + c;
  using Vector = packed_vec<T, V>;
  if (c + V <= p.group_channels) {
    *reinterpret_cast<device Vector*>(dst) =
        inside ? *reinterpret_cast<device const Vector*>(src) : Vector(0);
    return;
  }
  for (int i = 0; c + i < p.group_channels; ++i) {
    dst[i] = inside ? src[i] : T(0);
  }
}

INSTANTIATE_CONV2D("conv2d_group_major_float", conv2d_group_major<float>);
INSTANTIATE_CONV2D("conv2d_group_major_half", conv2d_group_major<half>);
INSTANTIATE_CONV2D("conv2d_group_major_bfloat", conv2d_group_major<bfloat>);

// im2col columns [C_in * kH * kW, outH, outW] for Conv2dMppInput::Im2col and
// Columns; each thread writes kConv2dIm2colPositionsPerThread positions.
template <typename T>
kernel void conv2d_im2col(
    device const T* input [[buffer(0)]],
    device T* columns [[buffer(1)]],
    constant Conv2DParams& p [[buffer(2)]],
    constant long4& strides [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]) {
  using P4 = packed_vec<T, 4>;
  constexpr int PT = kConv2dIm2colPositionsPerThread;
  static_assert(PT == 8, "Stored as two packed four-element vectors");
  const int taps = p.kH * p.kW, positions = p.outH * p.outW;
  const int k = int(pos.y), c = k / taps, kh = k % taps / p.kW, kw = k % p.kW;
  const int p0 = int(pos.x) * PT;
  int oh = p0 / p.outW, ow = p0 % p.outW;
  device const T* in = input + pos.z * strides.x + c * strides.y;
  device T* out =
      columns + (int64_t(pos.z) * p.C_in * taps + k) * positions + p0;
  T v[PT];
  for (int j = 0; j < PT; ++j) {
    const int ih = oh * p.sH + kh * p.dH - p.padH;
    const int iw = ow * p.sW + kw * p.dW - p.padW;
    v[j] = p0 + j < positions && ih >= 0 && ih < p.H && iw >= 0 && iw < p.W
        ? in[ih * strides.z + iw * strides.w]
        : T(0);
    if (++ow == p.outW) {
      ow = 0;
      ++oh;
    }
  }
  if (p0 + PT <= positions) {
    *reinterpret_cast<device P4*>(out) = P4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<device P4*>(out + 4) = P4(v[4], v[5], v[6], v[7]);
  } else {
    for (int j = 0; p0 + j < positions; ++j) {
      out[j] = v[j];
    }
  }
}

// Preserve tiny half weight-transform coefficients before low-precision
// storage.
constant constexpr float kConv2dWinogradHalfWeightScale = 4.0f;

// Keep a nonzero BF16 subnormal nonzero when multiplying an original infinity.
// Metal may flush FMA operands, which would otherwise turn this product into
// 0*Inf.
template <typename T>
float conv2d_winograd_cold_fma(T x, T weight, float sum) {
  return fma(float(x), float(weight), sum);
}

template <>
float conv2d_winograd_cold_fma<bfloat>(bfloat x, bfloat weight, float sum) {
  constexpr int fraction_bits = numeric_limits<bfloat>::digits - 1;
  constexpr ushort min_normal = ushort(1) << fraction_bits;
  constexpr ushort sign_mask = ushort(1)
      << (numeric_limits<ushort>::digits - 1);
  constexpr ushort magnitude_mask = sign_mask - 1;
  constexpr ushort infinity = magnitude_mask & ~(min_normal - 1);
  const ushort x_bits = as_type<ushort>(x),
               weight_bits = as_type<ushort>(weight);
  const ushort x_magnitude = x_bits & magnitude_mask;
  const ushort weight_magnitude = weight_bits & magnitude_mask;
  float x_value = float(x), weight_value = float(weight);
  if (x_magnitude != 0 && x_magnitude < min_normal &&
      weight_magnitude == infinity) {
    x_value = x_bits & sign_mask ? -1.0f : 1.0f;
  }
  if (weight_magnitude != 0 && weight_magnitude < min_normal &&
      x_magnitude == infinity) {
    weight_value = weight_bits & sign_mask ? -1.0f : 1.0f;
  }
  return fma(x_value, weight_value, sum);
}

// Transform arithmetic can spread a nonfinite value beyond its receptive field.
// Element of tap `k` (row-major 3x3) for an OIHW or OHWI weight.
inline int64_t conv2d_winograd_weight_index(
    bool channels_last,
    int channels,
    int output_channel,
    int c,
    int k) {
  return channels_last ? (int64_t(output_channel) * 9 + k) * channels + c
                       : (int64_t(output_channel) * channels + c) * 9 + k;
}

template <typename T>
float conv2d_winograd_repair(
    float value,
    float bias,
    device const T* input,
    device const T* weight,
    thread const Conv2dWinogradRepairParams& p,
    int output_channel,
    int h,
    int w) {
  if (isfinite(float(T(value)))) {
    return value;
  }
  float sum = 0.0f;
  for (int c = 0; c < p.channels; ++c) {
    for (int kh = 0; kh < 3; ++kh) {
      const int ih = h + kh - p.pad_height;
      for (int kw = 0; kw < 3; ++kw) {
        const int iw = w + kw - p.pad_width;
        const T x = ih >= 0 && ih < p.height && iw >= 0 && iw < p.width
            ? input
                  [p.channels_last
                       ? int64_t(ih * p.width + iw) * p.channels + c
                       : (int64_t(c) * p.height + ih) * p.width + iw]
            : T(0);
        const T weight_value = weight[conv2d_winograd_weight_index(
            p.weight_channels_last,
            p.channels,
            output_channel,
            c,
            kh * 3 + kw)];
        sum = conv2d_winograd_cold_fma(x, weight_value, sum);
      }
    }
  }
  return sum + bias;
}

// BF16 sources >= 2^-41 lie on a 2^-48 grid. F4 weights retain a 2^-77
// grid, so products stay on 2^-125, above FP32's minimum normal 2^-126.
// Smaller nonzero sources reuse the nonfinite-output repair path.
constant constexpr ushort kConv2dWinogradBfloatMinMagnitude = 0x2b00;
constant constexpr ushort kConv2dWinogradBfloatMagnitudeMask =
    (ushort(1) << (numeric_limits<ushort>::digits - 1)) - 1;

template <typename T>
float conv2d_winograd_range_load(T value) {
  return float(value);
}

template <>
float conv2d_winograd_range_load<bfloat>(bfloat value) {
  const ushort magnitude =
      as_type<ushort>(value) & kConv2dWinogradBfloatMagnitudeMask;
  return magnitude && magnitude < kConv2dWinogradBfloatMinMagnitude
      ? NAN
      : float(value);
}

template <typename T, int N>
vec<float, N> conv2d_winograd_range_load_vector(
    packed_vec<T, N> values,
    false_type) {
  return vec<float, N>(values);
}

template <int N>
vec<float, N> conv2d_winograd_range_load_vector(
    packed_vec<bfloat, N> values,
    true_type) {
  vec<float, N> output;
  for (int i = 0; i < N; ++i) {
    output[i] = conv2d_winograd_range_load(values[i]);
  }
  return output;
}

// BF16 and FP32 share their exponent format. Keep range in an integer while
// accumulating exact BF16 products with a normalized FP32 significand.
struct BFRangeAccumulator {
  static constexpr constant int bfloat_fraction_bits =
      numeric_limits<bfloat>::digits - 1;
  static constexpr constant int float_fraction_bits =
      numeric_limits<float>::digits - 1;
  static constexpr constant int exponent_bits =
      numeric_limits<ushort>::digits - bfloat_fraction_bits - 1;
  static constexpr constant int exponent_bias = (1 << (exponent_bits - 1)) - 1;
  static constexpr constant uint exponent_mask = (1u << exponent_bits) - 1;
  static constexpr constant uint float_fraction_mask =
      (1u << float_fraction_bits) - 1;
  static constexpr constant uint float_sign_mask = 1u
      << (float_fraction_bits + exponent_bits);
  static constexpr constant ushort bfloat_sign_mask =
      ushort(1u << (bfloat_fraction_bits + exponent_bits));
  // An exponent gap above precision+1 cannot affect an FP32 RNE sum, even
  // immediately below a power of two where the adjacent spacing is smaller.
  static constexpr constant int max_exponent_gap =
      numeric_limits<float>::digits + 1;

  struct Term {
    int significand;
    int exponent;
  };

  float fraction = 0.0f;
  int exponent = 0;

  static Term decode_finite(ushort bits) {
    const int field = (bits >> bfloat_fraction_bits) & exponent_mask;
    const int magnitude = (bits & ((1u << bfloat_fraction_bits) - 1)) +
        (field ? 1u << bfloat_fraction_bits : 0);
    return {
        bits & bfloat_sign_mask ? -magnitude : magnitude,
        (field ? field : 1) - exponent_bias - bfloat_fraction_bits};
  }

  void add_product(bfloat x, bfloat weight) {
    const auto a = decode_finite(as_type<ushort>(x));
    const auto b = decode_finite(as_type<ushort>(weight));
    add({a.significand * b.significand, a.exponent + b.exponent});
  }

  void add(Term term) {
    if (!term.significand) {
      return;
    }
    const uint magnitude = uint(abs(term.significand));
    const int top = numeric_limits<uint>::digits - 1 - int(clz(magnitude));
    // Products have at most 16 significant bits, so both operations are exact.
    const float value = ldexp(float(term.significand), -top);
    const int value_exponent = term.exponent + top;
    if (fraction == 0.0f) {
      fraction = value;
      exponent = value_exponent;
      return;
    }
    const int common = max(exponent, value_exponent);
    const float left = exponent >= common - max_exponent_gap
        ? ldexp(fraction, exponent - common)
        : 0.0f;
    const float right = value_exponent >= common - max_exponent_gap
        ? ldexp(value, value_exponent - common)
        : 0.0f;
    const float sum = left + right;
    if (sum == 0.0f) {
      fraction = sum;
      return;
    }
    const uint bits = as_type<uint>(sum);
    exponent = common + int((bits >> float_fraction_bits) & exponent_mask) -
        exponent_bias;
    fraction = as_type<float>(
        (bits & (float_sign_mask | float_fraction_mask)) |
        (uint(exponent_bias) << float_fraction_bits));
  }

  ushort output_bits() const {
    constexpr int fraction_shift = float_fraction_bits - bfloat_fraction_bits;
    constexpr int min_normal_exponent = 1 - exponent_bias;
    constexpr int min_subnormal_exponent =
        min_normal_exponent - bfloat_fraction_bits;
    const uint bits = as_type<uint>(fraction);
    const ushort sign = ushort((bits >> fraction_shift) & bfloat_sign_mask);
    if (fraction == 0.0f || exponent < min_subnormal_exponent - 1) {
      return sign;
    }
    if (exponent > exponent_bias) {
      return sign | ushort(exponent_mask << bfloat_fraction_bits);
    }
    const uint mantissa =
        (bits & float_fraction_mask) | (1u << float_fraction_bits);
    const int shift = exponent < min_normal_exponent
        ? float_fraction_bits + min_subnormal_exponent - exponent
        : fraction_shift;
    uint rounded = mantissa >> shift;
    const uint residue = mantissa & ((1u << shift) - 1);
    const uint midpoint = 1u << (shift - 1);
    rounded += residue > midpoint || (residue == midpoint && (rounded & 1));
    if (exponent < min_normal_exponent) {
      return sign | ushort(rounded);
    }
    const int output_exponent =
        exponent + int(rounded == (1u << (bfloat_fraction_bits + 1)));
    if (output_exponent > exponent_bias) {
      return sign | ushort(exponent_mask << bfloat_fraction_bits);
    }
    return sign |
        ushort((output_exponent + exponent_bias) << bfloat_fraction_bits) |
        ushort(rounded & ((1u << bfloat_fraction_bits) - 1));
  }
};

template <typename T>
T conv2d_winograd_typed_repair(
    float value,
    device const T* bias,
    device const T* input,
    device const T* weight,
    thread const Conv2dWinogradRepairParams& p,
    int output_channel,
    int h,
    int w) {
  return T(conv2d_winograd_repair(
      value,
      bias ? float(bias[output_channel]) : 0.0f,
      input,
      weight,
      p,
      output_channel,
      h,
      w));
}

template <>
bfloat conv2d_winograd_typed_repair<bfloat>(
    float value,
    device const bfloat* bias,
    device const bfloat* input,
    device const bfloat* weight,
    thread const Conv2dWinogradRepairParams& p,
    int output_channel,
    int h,
    int w) {
  if (isfinite(float(bfloat(value)))) {
    return bfloat(value);
  }
  const float legacy = conv2d_winograd_repair(
      value,
      bias ? float(bias[output_channel]) : 0.0f,
      input,
      weight,
      p,
      output_channel,
      h,
      w);
  if (!isfinite(float(bfloat(legacy)))) {
    return bfloat(legacy);
  }
  BFRangeAccumulator sum;
  for (int c = 0; c < p.channels; ++c) {
    for (int kh = 0; kh < 3; ++kh) {
      const int ih = h + kh - p.pad_height;
      for (int kw = 0; kw < 3; ++kw) {
        const int iw = w + kw - p.pad_width;
        const bfloat x = ih >= 0 && ih < p.height && iw >= 0 && iw < p.width
            ? input
                  [p.channels_last
                       ? int64_t(ih * p.width + iw) * p.channels + c
                       : (int64_t(c) * p.height + ih) * p.width + iw]
            : bfloat(0);
        const bfloat weight_value = weight[conv2d_winograd_weight_index(
            p.weight_channels_last,
            p.channels,
            output_channel,
            c,
            kh * 3 + kw)];
        sum.add_product(x, weight_value);
      }
    }
  }
  if (bias) {
    sum.add(BFRangeAccumulator::decode_finite(
        as_type<ushort>(bias[output_channel])));
  }
  return as_type<bfloat>(sum.output_bits());
}

// Fused Winograd F(2x2, 3x3) for stride 1, dilation 1 and one group.
//
// A threadgroup owns BM output channels x BN tiles of 2x2 output pixels. Each
// chunk of KC = threads / BN input channels gives every thread one 4x4 input
// patch to transform,
// V = B^T d B, into threadgroup memory, and every SIMD group accumulates U V in
// FP32 for its two of the 16 coefficients over the whole block, so no
// accumulator is shared. U = G g G^T is transformed per chunk into threadgroup
// memory, or read from the conv2d_winograd_weights workspace, which also lets
// the input staging be double buffered. The output transform A^T M A goes
// through threadgroup memory. Nonfinite results, including those of bfloat16
// operands poisoned by conv2d_winograd_range_load, are recomputed directly.
constant constexpr int kConv2dWinogradInTile = 4;
constant constexpr int kConv2dWinogradOutTile = 2;
constant constexpr int kConv2dWinogradCoefficients =
    kConv2dWinogradInTile * kConv2dWinogradInTile;
constant constexpr int kConv2dWinogradTaps = 9;
constant constexpr int kConv2dWinogradFragment = 8;
constant constexpr int kConv2dWinogradMaxSharedFloats =
    32 * 1024 / sizeof(float);
// 4-byte threadgroup memory banks.
constant constexpr int kConv2dWinogradBanks = 32;

template <typename T>
inline float conv2d_winograd_weight_scale() {
  return is_same_v<T, half> ? kConv2dWinogradHalfWeightScale : 1.0f;
}

// Rows of G g for one column of a 3x3 filter.
inline float4 conv2d_winograd_filter_column(float g0, float g1, float g2) {
  return float4(g0, 0.5f * (g0 + g1 + g2), 0.5f * (g0 - g1 + g2), g2);
}

template <typename T>
inline __attribute__((always_inline)) void conv2d_winograd_load_filter(
    device const T* weight,
    constant Conv2dWinogradParams& p,
    int o,
    int c,
    thread float (&g)[kConv2dWinogradTaps]) {
  const bool valid = o < p.C_out && c < p.C_in;
  const int oc = min(o, p.C_out - 1), cc = min(c, p.C_in - 1);
#pragma unroll
  for (int k = 0; k < kConv2dWinogradTaps; ++k) {
    const float v =
        conv2d_winograd_range_load(weight[conv2d_winograd_weight_index(
            p.weight_channels_last, p.C_in, oc, cc, k)]);
    g[k] = valid ? v * conv2d_winograd_weight_scale<T>() : 0.0f;
  }
}

// U = G g G^T rounded to T, `stride` elements between coefficients.
template <typename T, typename Dst>
inline __attribute__((always_inline)) void conv2d_winograd_transform_filter(
    thread const float (&g)[kConv2dWinogradTaps],
    Dst dst,
    int stride) {
  float4 r[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    r[j] = conv2d_winograd_filter_column(g[j], g[3 + j], g[6 + j]);
  }
#pragma unroll
  for (int i = 0; i < kConv2dWinogradInTile; ++i) {
    const float4 u = conv2d_winograd_filter_column(r[0][i], r[1][i], r[2][i]);
#pragma unroll
    for (int j = 0; j < kConv2dWinogradInTile; ++j) {
      dst[(i * kConv2dWinogradInTile + j) * stride] = T(u[j]);
    }
  }
}

// Workspace [ceil(C_out / BM)][c_blocks][16][BM][KC], zero padded, so every
// chunk of an output-channel block is contiguous.
template <typename T, int BM, int KC>
kernel void conv2d_winograd_weights(
    device const T* weight [[buffer(0)]],
    device T* transformed [[buffer(1)]],
    constant Conv2dWinogradParams& p [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x), o = int(pos.y);
  float g[kConv2dWinogradTaps];
  conv2d_winograd_load_filter(weight, p, o, c, g);
  const int64_t block = int64_t(o / BM) * p.c_blocks + c / KC;
  conv2d_winograd_transform_filter<T>(
      g,
      transformed + block * kConv2dWinogradCoefficients * BM * KC +
          o % BM * KC + c % KC,
      BM * KC);
}

// Pixel offsets of one input patch; only the channel changes between chunks.
struct Conv2dWinogradPatch {
  int row[kConv2dWinogradInTile];
  int col[kConv2dWinogradInTile];
  bool row_ok[kConv2dWinogradInTile];
  bool col_ok[kConv2dWinogradInTile];
  bool interior;
};

inline Conv2dWinogradPatch conv2d_winograd_patch(
    constant Conv2dWinogradParams& p,
    int t) {
  Conv2dWinogradPatch patch;
  const bool valid = t < p.tiles;
  const int h0 = kConv2dWinogradOutTile * (t / p.tiles_w) - p.pad;
  const int w0 = kConv2dWinogradOutTile * (t % p.tiles_w) - p.pad;
#pragma unroll
  for (int i = 0; i < kConv2dWinogradInTile; ++i) {
    patch.row_ok[i] = valid && h0 + i >= 0 && h0 + i < p.H;
    patch.col_ok[i] = w0 + i >= 0 && w0 + i < p.W;
    patch.row[i] = clamp(h0 + i, 0, p.H - 1) * p.W;
    patch.col[i] = clamp(w0 + i, 0, p.W - 1);
  }
  patch.interior = valid && h0 >= 0 && h0 + kConv2dWinogradInTile <= p.H &&
      w0 >= 0 && w0 + kConv2dWinogradInTile <= p.W;
  return patch;
}

// x points at channel c: its plane for NCHW, its first pixel for NHWC.
template <typename T, bool IN_NHWC>
inline __attribute__((always_inline)) void conv2d_winograd_load_patch(
    device const T* x,
    thread const Conv2dWinogradPatch& patch,
    bool channel_ok,
    int pixel_stride,
    thread float4 (&d)[kConv2dWinogradInTile],
    bool_constant<IN_NHWC>) {
  if (channel_ok && patch.interior) {
#pragma unroll
    for (int i = 0; i < kConv2dWinogradInTile; ++i) {
      if (IN_NHWC) {
#pragma unroll
        for (int j = 0; j < kConv2dWinogradInTile; ++j) {
          d[i][j] = conv2d_winograd_range_load(
              x[(patch.row[i] + patch.col[j]) * pixel_stride]);
        }
      } else {
        d[i] = conv2d_winograd_range_load_vector(
            *reinterpret_cast<device const packed_vec<T, 4>*>(
                x + patch.row[i] + patch.col[0]),
            bool_constant<is_same_v<T, bfloat>>{});
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < kConv2dWinogradInTile; ++i) {
#pragma unroll
    for (int j = 0; j < kConv2dWinogradInTile; ++j) {
      const float v = conv2d_winograd_range_load(
          x[(patch.row[i] + patch.col[j]) * (IN_NHWC ? pixel_stride : 1)]);
      d[i][j] = channel_ok && patch.row_ok[i] && patch.col_ok[j] ? v : 0.0f;
    }
  }
}

// BF16 patches check their range once: a nonzero element below
// kConv2dWinogradBfloatMinMagnitude poisons the whole patch, which sends the
// outputs of its tile through the nonfinite repair.
template <bool IN_NHWC>
inline __attribute__((always_inline)) void conv2d_winograd_load_patch(
    device const bfloat* x,
    thread const Conv2dWinogradPatch& patch,
    bool channel_ok,
    int pixel_stride,
    thread float4 (&d)[kConv2dWinogradInTile],
    bool_constant<IN_NHWC>) {
  // Unsigned minimum of magnitude - 1, so zeros wrap to the maximum.
  ushort smallest = numeric_limits<ushort>::max();
  const bool interior = channel_ok && patch.interior;
  if (!IN_NHWC && interior) {
#pragma unroll
    for (int i = 0; i < kConv2dWinogradInTile; ++i) {
      const auto bits = *reinterpret_cast<device const packed_vec<ushort, 4>*>(
          x + patch.row[i] + patch.col[0]);
#pragma unroll
      for (int j = 0; j < kConv2dWinogradInTile; ++j) {
        smallest =
            min(smallest,
                ushort((bits[j] & kConv2dWinogradBfloatMagnitudeMask) - 1));
        d[i][j] = float(as_type<bfloat>(bits[j]));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kConv2dWinogradInTile; ++i) {
#pragma unroll
      for (int j = 0; j < kConv2dWinogradInTile; ++j) {
        const bool ok =
            interior || (channel_ok && patch.row_ok[i] && patch.col_ok[j]);
        const bfloat v =
            x[(patch.row[i] + patch.col[j]) * (IN_NHWC ? pixel_stride : 1)];
        const ushort bits = ok ? as_type<ushort>(v) : ushort(0);
        smallest = min(
            smallest, ushort((bits & kConv2dWinogradBfloatMagnitudeMask) - 1));
        d[i][j] = float(as_type<bfloat>(bits));
      }
    }
  }
  if (smallest < kConv2dWinogradBfloatMinMagnitude - 1) {
#pragma unroll
    for (int i = 0; i < kConv2dWinogradInTile; ++i) {
      d[i] = float4(NAN);
    }
  }
}

// V = B^T d B rounded to T, `stride` elements between coefficients.
template <typename T>
inline __attribute__((always_inline)) void conv2d_winograd_store_patch(
    thread const float4 (&d)[kConv2dWinogradInTile],
    threadgroup T* dst,
    int stride) {
  const float4 r[kConv2dWinogradInTile] = {
      d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3]};
#pragma unroll
  for (int i = 0; i < kConv2dWinogradInTile; ++i) {
    const float4 v(
        r[i].x - r[i].z, r[i].y + r[i].z, r[i].z - r[i].y, r[i].y - r[i].w);
#pragma unroll
    for (int j = 0; j < kConv2dWinogradInTile; ++j) {
      dst[(i * kConv2dWinogradInTile + j) * stride] = T(v[j]);
    }
  }
}

// Static helpers over plain arrays keep every per-thread value in registers.
// A staging slot e is one (channel, tile) pair; consecutive threads take
// consecutive tiles of NCHW input and consecutive channels of NHWC input, and
// each coefficient is staged as [KC][BN] or [BN][KC] accordingly.
template <typename T, int BM, int BN, bool PRETRANSFORMED, bool IN_NHWC>
struct Conv2dWinogradPipeline {
  static constant constexpr int threads =
      kConv2dWinogradSimdgroups * c10::metal::simdgroup_size;
  static constant constexpr int KC = threads / BN;
  static constant constexpr int coefficients_per_simdgroup =
      kConv2dWinogradCoefficients / kConv2dWinogradSimdgroups;
  static constant constexpr int TM = BM / kConv2dWinogradFragment;
  static constant constexpr int TN = BN / kConv2dWinogradFragment;
  static constant constexpr int patches = 1;
  static constant constexpr int filters =
      PRETRANSFORMED ? 1 : (BM * KC + threads - 1) / threads;
  static constant constexpr int stage_pitch = IN_NHWC ? KC : BN;
  static constant constexpr int stage_stride = KC * BN;
  // Rows between the U coefficients: the conv2d_winograd_weights blocks, or
  // this tile's own per-chunk staging.
  static constant constexpr int u_rows =
      PRETRANSFORMED ? kConv2dWinogradPretransformedRows : BM;
  static_assert(KC * BN == threads, "One input patch per thread and chunk");
  static_assert(
      PRETRANSFORMED || BM * KC % threads == 0 || filters == 1,
      "Weight staging must split evenly over threads");
  static_assert(
      kConv2dWinogradCoefficients % kConv2dWinogradSimdgroups == 0,
      "Coefficients must split evenly over SIMD groups");

  // NCHW slots fill 8-tile fragments a SIMD group at a time: lanes cover
  // lane_channels channels of 8 consecutive tiles.
  static constant constexpr int lane_channels =
      c10::metal::simdgroup_size / kConv2dWinogradFragment;
  static constant constexpr int tile_fragments = BN / kConv2dWinogradFragment;
  static inline __attribute__((always_inline)) int slot_channel(int e) {
    const int lane = e % c10::metal::simdgroup_size;
    const int group = e / c10::metal::simdgroup_size;
    return IN_NHWC ? e % KC
                   : group / tile_fragments * lane_channels +
            lane / kConv2dWinogradFragment;
  }
  static inline __attribute__((always_inline)) int slot_tile(int e) {
    const int lane = e % c10::metal::simdgroup_size;
    const int group = e / c10::metal::simdgroup_size;
    return IN_NHWC ? e / KC
                   : group % tile_fragments * kConv2dWinogradFragment +
            lane % kConv2dWinogradFragment;
  }

  static inline __attribute__((always_inline)) void load(
      thread float4 (&d)[patches][kConv2dWinogradInTile],
      thread float (&g)[filters][kConv2dWinogradTaps],
      thread const Conv2dWinogradPatch (&slots)[patches],
      device const T* input,
      device const T* weight,
      constant Conv2dWinogradParams& p,
      int o0,
      int c0,
      uint tid) {
#pragma unroll
    for (int i = 0; i < patches; ++i) {
      const int e = int(tid) + i * threads;
      const int c = c0 + slot_channel(e);
      if (KC * BN % threads == 0 || e < KC * BN) {
        device const T* x =
            input + min(c, p.C_in - 1) * (IN_NHWC ? 1 : p.H * p.W);
        conv2d_winograd_load_patch(
            x, slots[i], c < p.C_in, p.C_in, d[i], bool_constant<IN_NHWC>{});
      }
    }
    if (PRETRANSFORMED) {
      return;
    }
#pragma unroll
    for (int i = 0; i < filters; ++i) {
      const int e = int(tid) + i * threads;
      if (BM * KC % threads == 0 || e < BM * KC) {
        conv2d_winograd_load_filter(weight, p, o0 + e / KC, c0 + e % KC, g[i]);
      }
    }
  }

  static inline __attribute__((always_inline)) void store(
      thread const float4 (&d)[patches][kConv2dWinogradInTile],
      thread const float (&g)[filters][kConv2dWinogradTaps],
      threadgroup T* stage,
      threadgroup T* filter_stage,
      uint tid) {
#pragma unroll
    for (int i = 0; i < patches; ++i) {
      const int e = int(tid) + i * threads;
      if (KC * BN % threads == 0 || e < KC * BN) {
        const int c = slot_channel(e), t = slot_tile(e);
        // NCHW stages each 8-tile fragment column contiguously.
        conv2d_winograd_store_patch(
            d[i],
            stage +
                (IN_NHWC ? t * KC + c
                         : (t / kConv2dWinogradFragment * KC + c) *
                             kConv2dWinogradFragment +
                         t % kConv2dWinogradFragment),
            stage_stride);
      }
    }
    if (PRETRANSFORMED) {
      return;
    }
#pragma unroll
    for (int i = 0; i < filters; ++i) {
      const int e = int(tid) + i * threads;
      if (BM * KC % threads == 0 || e < BM * KC) {
        conv2d_winograd_transform_filter<T>(g[i], filter_stage + e, BM * KC);
      }
    }
  }

  static constant constexpr int KS = KC / kConv2dWinogradFragment;
  using Fragment =
      simdgroup_matrix<T, kConv2dWinogradFragment, kConv2dWinogradFragment>;

  // The SIMD group's U fragments of one chunk.
  template <typename FilterPtr>
  static inline __attribute__((always_inline)) void load_filters(
      thread Fragment (&a)[coefficients_per_simdgroup][KS][TM],
      FilterPtr u,
      uint sg) {
#pragma unroll
    for (int v = 0; v < coefficients_per_simdgroup; ++v) {
      const int q = int(sg) * coefficients_per_simdgroup + v;
#pragma unroll
      for (int s = 0; s < KS; ++s) {
#pragma unroll
        for (int m = 0; m < TM; ++m) {
          simdgroup_load(
              a[v][s][m],
              u + (q * u_rows + m * kConv2dWinogradFragment) * KC +
                  s * kConv2dWinogradFragment,
              KC);
        }
      }
    }
  }

  // acc[v] += U[q] V[q] over one chunk for the SIMD group's coefficients q,
  // with U fragments loaded by load_filters.
  static inline __attribute__((always_inline)) void multiply(
      thread float2 (&acc)[coefficients_per_simdgroup][TM][TN],
      thread const Fragment (&fa)[coefficients_per_simdgroup][KS][TM],
      threadgroup const T* stage,
      uint sg) {
#pragma unroll
    for (int v = 0; v < coefficients_per_simdgroup; ++v) {
#pragma unroll
      for (int k = 0; k < KC; k += kConv2dWinogradFragment) {
        const int q = int(sg) * coefficients_per_simdgroup + v;
        multiply_step(acc[v], fa[v][k / kConv2dWinogradFragment], stage, q, k);
      }
    }
  }

  // The same product loading each U fragment just before its use, which keeps
  // fewer fragments live for chunks of more than one fragment.
  template <typename FilterPtr>
  static inline __attribute__((always_inline)) void multiply(
      thread float2 (&acc)[coefficients_per_simdgroup][TM][TN],
      FilterPtr u,
      threadgroup const T* stage,
      uint sg) {
#pragma unroll
    for (int v = 0; v < coefficients_per_simdgroup; ++v) {
      const int q = int(sg) * coefficients_per_simdgroup + v;
#pragma unroll
      for (int k = 0; k < KC; k += kConv2dWinogradFragment) {
        Fragment a[TM];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
          simdgroup_load(
              a[m],
              u + (q * u_rows + m * kConv2dWinogradFragment) * KC + k,
              KC);
        }
        multiply_step(acc[v], a, stage, q, k);
      }
    }
  }

  // acc += U[q][:, k:k+8] V[q][k:k+8, :] for one 8-channel step.
  static inline __attribute__((always_inline)) void multiply_step(
      thread float2 (&acc)[TM][TN],
      thread const Fragment (&a)[TM],
      threadgroup const T* stage,
      int q,
      int k) {
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      Fragment b;
      if (IN_NHWC) {
        simdgroup_load(
            b,
            stage + q * stage_stride + n * kConv2dWinogradFragment * KC + k,
            KC,
            0,
            true);
      } else {
        simdgroup_load(
            b,
            stage + q * stage_stride + (n * KC + k) * kConv2dWinogradFragment,
            kConv2dWinogradFragment);
      }
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        simdgroup_matrix<
            float,
            kConv2dWinogradFragment,
            kConv2dWinogradFragment>
            c;
        c.thread_elements()[0] = acc[m][n].x;
        c.thread_elements()[1] = acc[m][n].y;
        simdgroup_multiply_accumulate(c, a[m], b, c);
        acc[m][n] = float2(c.thread_elements()[0], c.thread_elements()[1]);
      }
    }
  }

  // Pre-transformed weights are read from the workspace `u`, which frees the
  // room for a second input stage: one barrier per chunk of KC channels.
  static inline __attribute__((always_inline)) void accumulate(
      thread float2 (&acc)[coefficients_per_simdgroup][TM][TN],
      thread const Conv2dWinogradPatch (&slots)[patches],
      device const T* input,
      device const T* weight,
      device const T* u,
      constant Conv2dWinogradParams& p,
      threadgroup T* stage,
      threadgroup T* filter_stage,
      int o0,
      uint tid,
      uint sg,
      true_type) {
    constexpr int stage_elements = kConv2dWinogradCoefficients * KC * BN;
    constexpr int filter_elements = kConv2dWinogradCoefficients * u_rows * KC;
    // One-fragment chunks load U a chunk ahead so the products after each
    // barrier do not wait on device memory; deeper chunks would keep too many
    // fragments live.
    constexpr bool prefetch = KS == 1;
    float4 d[patches][kConv2dWinogradInTile];
    float g[filters][kConv2dWinogradTaps];
    Fragment fa[coefficients_per_simdgroup][KS][TM];
    const int chunks = (p.C_in + KC - 1) / KC;
    load(d, g, slots, input, weight, p, o0, 0, tid);
    store(d, g, stage, filter_stage, tid);
    if (prefetch) {
      load_filters(fa, u, sg);
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const bool more = chunk + 1 < chunks;
      if (more) {
        load(d, g, slots, input, weight, p, o0, (chunk + 1) * KC, tid);
      }
      threadgroup const T* current = stage + chunk % 2 * stage_elements;
      if (prefetch) {
        multiply(acc, fa, current, sg);
      } else {
        multiply(acc, u + chunk * filter_elements, current, sg);
      }
      if (more) {
        if (prefetch) {
          load_filters(fa, u + (chunk + 1) * filter_elements, sg);
        }
        threadgroup T* next = stage + (chunk + 1) % 2 * stage_elements;
        store(d, g, next, filter_stage, tid);
      }
    }
  }

  // Weights are transformed per chunk into threadgroup memory next to the
  // single input stage.
  static inline __attribute__((always_inline)) void accumulate(
      thread float2 (&acc)[coefficients_per_simdgroup][TM][TN],
      thread const Conv2dWinogradPatch (&slots)[patches],
      device const T* input,
      device const T* weight,
      device const T*,
      constant Conv2dWinogradParams& p,
      threadgroup T* stage,
      threadgroup T* filter_stage,
      int o0,
      uint tid,
      uint sg,
      false_type) {
    float4 d[patches][kConv2dWinogradInTile];
    float g[filters][kConv2dWinogradTaps];
    const int chunks = (p.C_in + KC - 1) / KC;
    load(d, g, slots, input, weight, p, o0, 0, tid);
    for (int chunk = 0; chunk < chunks; ++chunk) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      store(d, g, stage, filter_stage, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (chunk + 1 < chunks) {
        load(d, g, slots, input, weight, p, o0, (chunk + 1) * KC, tid);
      }
      multiply(acc, static_cast<threadgroup const T*>(filter_stage), stage, sg);
    }
  }
};

// NHWC output staging rows hold the channels of one tile. A fragment store
// writes four tile pairs; each pair takes its own 8-channel group of its row,
// which spreads the store over the threadgroup memory banks.
template <int ROW_GROUPS>
inline int conv2d_winograd_nhwc_swizzle(int t) {
  return t % kConv2dWinogradFragment / 2 % ROW_GROUPS * kConv2dWinogradFragment;
}

// Output transform A^T M A of one BM x BN block from the SIMD groups'
// coefficient accumulators, staged through `products` in passes of
// PASS_ROWS output channels, then the nonfinite repair.
template <
    typename T,
    int BM,
    int BN,
    int PASS_ROWS,
    bool IN_NHWC,
    bool OUT_NHWC>
inline __attribute__((always_inline)) void conv2d_winograd_store(
    thread const float2 (
        &acc)[kConv2dWinogradCoefficients / kConv2dWinogradSimdgroups]
             [BM / kConv2dWinogradFragment][BN / kConv2dWinogradFragment],
    threadgroup float* products,
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    constant Conv2dWinogradParams& p,
    int o0,
    int t0,
    uint tid,
    uint sg) {
  constexpr int threads =
      kConv2dWinogradSimdgroups * c10::metal::simdgroup_size;
  constexpr int CPS = kConv2dWinogradCoefficients / kConv2dWinogradSimdgroups;
  constexpr int TM = BM / kConv2dWinogradFragment;
  constexpr int TN = BN / kConv2dWinogradFragment;
  constexpr int pass_rows = PASS_ROWS;
  // `products` stages each coefficient as [pass_rows][BN] for NCHW output and
  // [BN][pass_rows] for NHWC output, so consecutive threads store
  // consecutive output addresses.
  // Element (fragment_row, fragment_col + {0, 1}) of an 8x8 fragment.
  const int lane = int(tid) % c10::metal::simdgroup_size, quad = lane / 4;
  const int fragment_row = (quad & 4) + (lane / 2) % 4;
  const int fragment_col = (quad & 2) * 2 + (lane % 2) * 2;
  const int pixel_stride = OUT_NHWC ? p.C_out : 1;
  // NCHW staging rows that start on the same threadgroup memory bank take
  // different 8-column groups, so a fragment store spreads over all banks.
  constexpr int bank_rows =
      BN < kConv2dWinogradBanks ? kConv2dWinogradBanks / BN : 1;
  constexpr int column_groups = BN / kConv2dWinogradFragment;
  constexpr int row_groups = pass_rows / kConv2dWinogradFragment;
  bool nonfinite = false;
#pragma unroll
  for (int pass = 0; pass < BM / pass_rows; ++pass) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Compile-time acc indices keep the accumulators in registers.
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      if (m * kConv2dWinogradFragment / pass_rows != pass) {
        continue;
      }
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        const int o = m * kConv2dWinogradFragment % pass_rows + fragment_row;
        const int t = n * kConv2dWinogradFragment + fragment_col;
#pragma unroll
        for (int v = 0; v < CPS; ++v) {
          const int q = int(sg) * CPS + v;
          if (OUT_NHWC) {
            const int channel = o ^ conv2d_winograd_nhwc_swizzle<row_groups>(t);
            products[(q * BN + t) * pass_rows + channel] = acc[v][m][n].x;
            products[(q * BN + t + 1) * pass_rows + channel] = acc[v][m][n].y;
          } else {
            const int swizzle =
                o / bank_rows % column_groups * kConv2dWinogradFragment;
            *reinterpret_cast<threadgroup float2*>(
                products + (q * pass_rows + o) * BN + (t ^ swizzle)) =
                acc[v][m][n];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < pass_rows * BN; e += threads) {
      const int o = o0 + pass * pass_rows + (OUT_NHWC ? e % pass_rows : e / BN);
      const int t = t0 + (OUT_NHWC ? e / pass_rows : e % BN);
      if (o >= p.C_out || t >= p.tiles) {
        continue;
      }
      const int slot = OUT_NHWC
          ? e ^ conv2d_winograd_nhwc_swizzle<row_groups>(e / pass_rows)
          : e ^ (e / BN / bank_rows % column_groups * kConv2dWinogradFragment);
      // Coefficient (i, j) of this (o, t) pair is at m[(4 i + j) * stride].
      threadgroup const float* m = products + slot;
      constexpr int stride = pass_rows * BN;
      float4 r[kConv2dWinogradOutTile];
#pragma unroll
      for (int j = 0; j < kConv2dWinogradInTile; ++j) {
        const float m0 = m[(0 * kConv2dWinogradInTile + j) * stride];
        const float m1 = m[(1 * kConv2dWinogradInTile + j) * stride];
        const float m2 = m[(2 * kConv2dWinogradInTile + j) * stride];
        const float m3 = m[(3 * kConv2dWinogradInTile + j) * stride];
        r[0][j] = m0 + m1 + m2;
        r[1][j] = m1 - m2 - m3;
      }
      const float b = p.has_bias ? float(bias[o]) : 0.0f;
      const int oh = kConv2dWinogradOutTile * (t / p.tiles_w);
      const int ow = kConv2dWinogradOutTile * (t % p.tiles_w);
#pragma unroll
      for (int i = 0; i < kConv2dWinogradOutTile; ++i) {
        if (oh + i >= p.outH) {
          continue;
        }
        const float2 y =
            float2(r[i].x + r[i].y + r[i].z, r[i].y - r[i].z - r[i].w) /
                conv2d_winograd_weight_scale<T>() +
            b;
        // The host keeps per-image extents within int32.
        device T* dst = output +
            (OUT_NHWC ? ((oh + i) * p.outW + ow) * p.C_out + o
                      : (o * p.outH + oh + i) * p.outW + ow);
        const T first = T(y.x), second = T(y.y);
        nonfinite |= !isfinite(float(first)) || !isfinite(float(second));
        if (!OUT_NHWC && ow + 1 < p.outW) {
          *reinterpret_cast<device packed_vec<T, 2>*>(dst) =
              packed_vec<T, 2>(first, second);
        } else {
          dst[0] = first;
          if (ow + 1 < p.outW) {
            dst[pixel_stride] = second;
          }
        }
      }
    }
  }
  if (!nonfinite) {
    return;
  }
  // Recompute this thread's nonfinite outputs directly. Keeping the repair out
  // of the unrolled store loop above keeps that loop fast.
  const Conv2dWinogradRepairParams repair{
      p.C_in, p.H, p.W, p.pad, p.pad, IN_NHWC, p.weight_channels_last};
  device const T* repair_bias = p.has_bias ? bias : nullptr;
  for (int pass = 0; pass < BM / pass_rows; ++pass) {
    for (int e = int(tid); e < pass_rows * BN; e += threads) {
      const int o = o0 + pass * pass_rows + (OUT_NHWC ? e % pass_rows : e / BN);
      const int t = t0 + (OUT_NHWC ? e / pass_rows : e % BN);
      if (o >= p.C_out || t >= p.tiles) {
        continue;
      }
      const int oh = kConv2dWinogradOutTile * (t / p.tiles_w);
      const int ow = kConv2dWinogradOutTile * (t % p.tiles_w);
      for (int i = 0; i < kConv2dWinogradOutTile && oh + i < p.outH; ++i) {
        for (int j = 0; j < kConv2dWinogradOutTile && ow + j < p.outW; ++j) {
          device T* dst = output +
              (OUT_NHWC ? ((oh + i) * p.outW + ow + j) * p.C_out + o
                        : (o * p.outH + oh + i) * p.outW + ow + j);
          if (!isfinite(float(*dst))) {
            *dst = conv2d_winograd_typed_repair(
                float(*dst),
                repair_bias,
                input,
                weight,
                repair,
                o,
                oh + i,
                ow + j);
          }
        }
      }
    }
  }
}

template <typename T, int BM, int BN, bool IN_NHWC, bool OUT_NHWC>
[[max_total_threads_per_threadgroup(
    kConv2dWinogradSimdgroups * c10::metal::simdgroup_size)]] kernel void
conv2d_winograd(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device const T* transformed [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant Conv2dWinogradParams& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr bool PRETRANSFORMED = conv2d_winograd_pretransformed(BM, BN);
  using P = Conv2dWinogradPipeline<T, BM, BN, PRETRANSFORMED, IN_NHWC>;
  constexpr int CPS = P::coefficients_per_simdgroup, TM = P::TM, TN = P::TN;
  constexpr int KC = P::KC;
  constexpr int stage_elements = kConv2dWinogradCoefficients * KC * BN;
  constexpr int filter_elements = kConv2dWinogradCoefficients * BM * KC;
  // Pre-transformed weights free the room for a second input stage.
  constexpr int stages = PRETRANSFORMED ? 2 : 1;
  constexpr int staged_floats =
      (stages * stage_elements + (PRETRANSFORMED ? 0 : filter_elements)) *
      sizeof(T) / sizeof(float);
  // The output transform needs all 16 coefficients of an (o, t) pair: one
  // pass when the block fits, else passes of whole fragment rows. Smaller
  // tiles keep less memory so more threadgroups stay resident.
  constexpr int block_floats = kConv2dWinogradCoefficients * BM * BN;
  constexpr int output_floats = block_floats < kConv2dWinogradMaxSharedFloats
      ? block_floats
      : kConv2dWinogradMaxSharedFloats;
  constexpr int shared_floats =
      staged_floats > output_floats ? staged_floats : output_floats;
  constexpr int fit = shared_floats / (kConv2dWinogradCoefficients * BN) /
      kConv2dWinogradFragment * kConv2dWinogradFragment;
  constexpr int pass_rows = fit < BM ? fit : BM;
  static_assert(
      shared_floats <= kConv2dWinogradMaxSharedFloats,
      "Threadgroup memory budget");
  static_assert(BM % pass_rows == 0, "Output passes must cover the block");
  threadgroup float shared[shared_floats];
  threadgroup T* stage = reinterpret_cast<threadgroup T*>(shared);
  threadgroup T* filter_stage = stage + stages * stage_elements;

  const int o0 = int(gid.x) * BM, t0 = int(gid.y) * BN;
  input += int64_t(gid.z) * p.C_in * p.H * p.W;
  output += int64_t(gid.z) * p.C_out * p.outH * p.outW;
  // This tile's rows of a conv2d_winograd_weights block.
  device const T* u = transformed +
      int64_t(o0 / kConv2dWinogradPretransformedRows) * p.c_blocks *
          kConv2dWinogradCoefficients * kConv2dWinogradPretransformedRows * KC +
      o0 % kConv2dWinogradPretransformedRows * KC;

  Conv2dWinogradPatch slots[P::patches];
#pragma unroll
  for (int i = 0; i < P::patches; ++i) {
    const int e = int(tid) + i * P::threads;
    slots[i] = conv2d_winograd_patch(p, t0 + P::slot_tile(e));
  }
  float2 acc[CPS][TM][TN];
#pragma unroll
  for (int v = 0; v < CPS; ++v) {
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        acc[v][m][n] = 0.0f;
      }
    }
  }
  P::accumulate(
      acc,
      slots,
      input,
      weight,
      u,
      p,
      stage,
      filter_stage,
      o0,
      tid,
      sg,
      bool_constant<PRETRANSFORMED>{});

  conv2d_winograd_store<T, BM, BN, pass_rows, IN_NHWC, OUT_NHWC>(
      acc, shared, input, weight, bias, output, p, o0, t0, tid, sg);
}

// Inputs within one channel chunk: the threadgroup transforms its input tiles
// once and runs every BM-output block over them with pre-transformed weights,
// instead of one threadgroup per output block repeating the input transform.
// The output staging follows the input stage in threadgroup memory.
template <typename T, int BM, int BN>
[[max_total_threads_per_threadgroup(
    kConv2dWinogradSimdgroups * c10::metal::simdgroup_size)]] kernel void
conv2d_winograd_loop(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device const T* transformed [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant Conv2dWinogradParams& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using P = Conv2dWinogradPipeline<T, BM, BN, true, false>;
  constexpr int CPS = P::coefficients_per_simdgroup, TM = P::TM, TN = P::TN;
  constexpr int KC = P::KC;
  constexpr int stage_floats =
      kConv2dWinogradCoefficients * KC * BN * sizeof(T) / sizeof(float);
  constexpr int fit = (kConv2dWinogradMaxSharedFloats - stage_floats) /
      (kConv2dWinogradCoefficients * BN) / kConv2dWinogradFragment *
      kConv2dWinogradFragment;
  constexpr int pass_rows = fit < BM ? fit : BM;
  static_assert(
      pass_rows > 0 && BM % pass_rows == 0,
      "Output passes must cover the block");
  threadgroup float
      shared[stage_floats + kConv2dWinogradCoefficients * pass_rows * BN];
  threadgroup T* stage = reinterpret_cast<threadgroup T*>(shared);

  const int t0 = int(gid.y) * BN;
  input += int64_t(gid.z) * p.C_in * p.H * p.W;
  output += int64_t(gid.z) * p.C_out * p.outH * p.outW;
  Conv2dWinogradPatch slots[P::patches];
#pragma unroll
  for (int i = 0; i < P::patches; ++i) {
    const int e = int(tid) + i * P::threads;
    slots[i] = conv2d_winograd_patch(p, t0 + P::slot_tile(e));
  }
  {
    float4 d[P::patches][kConv2dWinogradInTile];
    float g[P::filters][kConv2dWinogradTaps];
    P::load(d, g, slots, input, weight, p, 0, 0, tid);
    P::store(d, g, stage, stage, tid);
  }
  for (int o0 = 0; o0 < p.C_out; o0 += BM) {
    device const T* u = transformed +
        int64_t(o0 / kConv2dWinogradPretransformedRows) * p.c_blocks *
            kConv2dWinogradCoefficients * kConv2dWinogradPretransformedRows *
            KC +
        o0 % kConv2dWinogradPretransformedRows * KC;
    float2 acc[CPS][TM][TN];
#pragma unroll
    for (int v = 0; v < CPS; ++v) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          acc[v][m][n] = 0.0f;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    P::multiply(acc, u, stage, sg);
    conv2d_winograd_store<T, BM, BN, pass_rows, false, false>(
        acc,
        shared + stage_floats,
        input,
        weight,
        bias,
        output,
        p,
        o0,
        t0,
        tid,
        sg);
  }
}

// The 8- and 16-row tiles transform weights per chunk, serving few output
// channels and small problems in one dispatch; the 32-row tile reads the
// pre-transformed workspace.
#define INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, BM, BN) \
  INSTANTIATE_CONV2D(                                                      \
      "conv2d_winograd_" #LAYOUT "_" #BM "x" #BN "_" #DT,                  \
      conv2d_winograd<DT, BM, BN, IN_NHWC, OUT_NHWC>);

#define INSTANTIATE_CONV2D_WINOGRAD_TILES(DT, LAYOUT, IN_NHWC, OUT_NHWC) \
  INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, 8, 32)      \
  INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, 16, 16)     \
  INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, 16, 32)     \
  INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, 32, 16)     \
  INSTANTIATE_CONV2D_WINOGRAD(DT, LAYOUT, IN_NHWC, OUT_NHWC, 32, 32)

// Workspace chunks follow the input-channel chunk of the 32x16 and 32x32 tiles.
#define INSTANTIATE_CONV2D_WINOGRAD_WEIGHTS(DT, KC) \
  INSTANTIATE_CONV2D(                               \
      "conv2d_winograd_weights_" #KC "_" #DT,       \
      conv2d_winograd_weights<DT, kConv2dWinogradPretransformedRows, KC>);

#define INSTANTIATE_CONV2D_WINOGRAD_ALL(DT)                     \
  INSTANTIATE_CONV2D_WINOGRAD_WEIGHTS(DT, 8)                    \
  INSTANTIATE_CONV2D_WINOGRAD_WEIGHTS(DT, 16)                   \
  INSTANTIATE_CONV2D_WINOGRAD_TILES(DT, nchw, false, false)     \
  INSTANTIATE_CONV2D_WINOGRAD_TILES(DT, nchw_nhwc, false, true) \
  INSTANTIATE_CONV2D_WINOGRAD_TILES(DT, nhwc_nhwc, true, true)

INSTANTIATE_CONV2D_WINOGRAD_ALL(float)
INSTANTIATE_CONV2D_WINOGRAD_ALL(half)
INSTANTIATE_CONV2D_WINOGRAD_ALL(bfloat)
INSTANTIATE_CONV2D(
    "conv2d_winograd_loop_nchw_32x16_float",
    conv2d_winograd_loop<float, 32, 16>);

// Separate Winograd transforms (Conv2dKernel::Winograd2x2 and Winograd4x2):
// winograd_weights and winograd_input write the transformed operands of the
// coefficient GEMMs that conv2d_mpp runs, and winograd_output applies the
// output transform.
//
// Row transforms for Winograd F(2,3) and F(4,3). The column transform is
// F(2,3).
template <int TILE_H>
struct Conv2dWinogradTransform;

template <>
struct Conv2dWinogradTransform<2> {
  static void weights(
      float g0,
      float g1,
      float g2,
      thread float (&r)[4][3],
      int j) {
    r[0][j] = g0;
    r[1][j] = 0.5f * (g0 + g1 + g2);
    r[2][j] = 0.5f * (g0 - g1 + g2);
    r[3][j] = g2;
  }

  static void input(
      thread const float (&d)[4][4],
      thread float (&r)[4][4],
      int j) {
    r[0][j] = d[0][j] - d[2][j];
    r[1][j] = d[1][j] + d[2][j];
    r[2][j] = d[2][j] - d[1][j];
    r[3][j] = d[1][j] - d[3][j];
  }

  static void output(
      thread const float (&values)[4],
      thread float (&r)[2][4],
      int j) {
    r[0][j] = values[0] + values[1] + values[2];
    r[1][j] = values[1] - values[2] - values[3];
  }
};

template <>
struct Conv2dWinogradTransform<4> {
  static void weights(
      float g0,
      float g1,
      float g2,
      thread float (&r)[6][3],
      int j) {
    r[0][j] = 0.25f * g0;
    r[1][j] = -(g0 + g1 + g2) / 6.0f;
    r[2][j] = (-g0 + g1 - g2) / 6.0f;
    r[3][j] = g0 / 24.0f + g1 / 12.0f + g2 / 6.0f;
    r[4][j] = g0 / 24.0f - g1 / 12.0f + g2 / 6.0f;
    r[5][j] = g2;
  }

  static void input(
      thread const float (&d)[6][4],
      thread float (&r)[6][4],
      int j) {
    r[0][j] = 4 * d[0][j] - 5 * d[2][j] + d[4][j];
    r[1][j] = -4 * (d[1][j] + d[2][j]) + d[3][j] + d[4][j];
    r[2][j] = 4 * (d[1][j] - d[2][j]) - d[3][j] + d[4][j];
    r[3][j] = -2 * d[1][j] - d[2][j] + 2 * d[3][j] + d[4][j];
    r[4][j] = 2 * d[1][j] - d[2][j] - 2 * d[3][j] + d[4][j];
    r[5][j] = 4 * d[1][j] - 5 * d[3][j] + d[5][j];
  }

  static void output(
      thread const float (&values)[6],
      thread float (&r)[4][4],
      int j) {
    r[0][j] = values[0] + values[1] + values[2] + values[3] + values[4];
    r[1][j] = values[1] - values[2] + 2 * (values[3] - values[4]);
    r[2][j] = values[1] + values[2] + 4 * (values[3] + values[4]);
    r[3][j] = values[1] - values[2] + 8 * (values[3] - values[4]) + values[5];
  }
};

template <typename T, bool NHWC>
void winograd_load_row(
    device const T* x,
    thread float (&row)[4],
    int h,
    int w0,
    int height,
    int width,
    int channels,
    false_type) {
  for (int j = 0; j < 4; ++j) {
    const int w = w0 + j;
    row[j] = h >= 0 && h < height && w >= 0 && w < width
        ? conv2d_winograd_range_load(
              x[NHWC ? int64_t(h * width + w) * channels : h * width + w])
        : 0.0f;
  }
}

template <typename T, bool NHWC>
void winograd_load_row(
    device const T* x,
    thread float (&row)[4],
    int h,
    int w0,
    int height,
    int width,
    int channels,
    true_type) {
  if (h >= 0 && h < height && w0 >= 0 && w0 + 3 < width) {
    const float4 values = conv2d_winograd_range_load_vector(
        *reinterpret_cast<device const packed_vec<T, 4>*>(
            x + int64_t(h) * width + w0),
        bool_constant<is_same_v<T, bfloat>>{});
    for (int j = 0; j < 4; ++j) {
      row[j] = values[j];
    }
    return;
  }
  winograd_load_row<T, NHWC>(
      x, row, h, w0, height, width, channels, false_type{});
}

template <typename T, bool NHWC>
void winograd_store_row(
    device T* y,
    thread const float (&values)[2],
    int w0,
    int width,
    int channels,
    false_type) {
  for (int j = 0; j < 2; ++j) {
    const int w = w0 + j;
    if (w < width) {
      y[NHWC ? int64_t(w) * channels : w] = T(values[j]);
    }
  }
}

template <typename T, bool NHWC>
void winograd_store_row(
    device T* y,
    thread const float (&values)[2],
    int w0,
    int width,
    int channels,
    true_type) {
  if (w0 + 1 < width) {
    *reinterpret_cast<device packed_vec<T, 2>*>(y + w0) =
        packed_vec<T, 2>(T(values[0]), T(values[1]));
    return;
  }
  winograd_store_row<T, NHWC>(y, values, w0, width, channels, false_type{});
}

template <typename T, int TILE_H = 2>
kernel void winograd_weights(
    device const T* weight [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant int2& dims [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int O = dims.x, C = dims.y, o = int(pos.y), c = int(pos.x);
  device const T* g = weight + (int64_t(o) * C + c) * 9;
  float r[TILE_H + 2][3];
  for (int j = 0; j < 3; ++j) {
    float g0 = conv2d_winograd_range_load(g[j]);
    float g1 = conv2d_winograd_range_load(g[3 + j]);
    float g2 = conv2d_winograd_range_load(g[6 + j]);
    if IF_CONSTEXPR (is_same_v<T, half> && TILE_H == 2) {
      g0 *= kConv2dWinogradHalfWeightScale;
      g1 *= kConv2dWinogradHalfWeightScale;
      g2 *= kConv2dWinogradHalfWeightScale;
    }
    Conv2dWinogradTransform<TILE_H>::weights(g0, g1, g2, r, j);
  }
  for (int i = 0; i < TILE_H + 2; ++i) {
    const float x0 = r[i][0], x1 = r[i][1], x2 = r[i][2];
    const float u[4] = {x0, 0.5f * (x0 + x1 + x2), 0.5f * (x0 - x1 + x2), x2};
    for (int j = 0; j < 4; ++j) {
      out[(int64_t(i * 4 + j) * O + o) * C + c] = T(u[j]);
    }
  }
}

template <typename T, bool NHWC, int TILE_H = 2>
kernel void winograd_input(
    device const T* input [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant int4& dims [[buffer(2)]],
    constant int4& geom [[buffer(3)]],
    constant int& t0 [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int C = dims.x, tiles = dims.y, H = dims.z, W = dims.w;
  const int TH = geom.x, TW = geom.y, padH = geom.z, padW = geom.w;
  const int tile = t0 + int(pos.x), c = int(pos.y);
  const int n = tile / (TH * TW), ty = tile / TW % TH, tx = tile % TW;
  const int h0 = TILE_H * ty - padH, w0 = 2 * tx - padW;
  device const T* x = input +
      (NHWC ? int64_t(n) * H * W * C + c : (int64_t(n) * C + c) * H * W);
  float d[TILE_H + 2][4];
  for (int i = 0; i < TILE_H + 2; ++i) {
    const int h = h0 + i;
    winograd_load_row<T, NHWC>(
        x, d[i], h, w0, H, W, C, bool_constant < TILE_H == 4 && !NHWC > {});
  }
  float r[TILE_H + 2][4];
  for (int j = 0; j < 4; ++j) {
    Conv2dWinogradTransform<TILE_H>::input(d, r, j);
  }
  for (int i = 0; i < TILE_H + 2; ++i) {
    const float v[4] = {
        r[i][0] - r[i][2],
        r[i][1] + r[i][2],
        r[i][2] - r[i][1],
        r[i][1] - r[i][3]};
    for (int j = 0; j < 4; ++j) {
      out[(int64_t(i * 4 + j) * C + c) * tiles + int(pos.x)] = T(v[j]);
    }
  }
}

template <typename T, bool NHWC, int TILE_H = 2>
kernel void winograd_output(
    device const T* m [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant int4& dims [[buffer(3)]],
    constant int4& geom [[buffer(4)]],
    device const T* input [[buffer(5)]],
    device const T* weight [[buffer(6)]],
    constant Conv2dWinogradRepairParams& repair_geometry [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int O = dims.x, tiles = dims.y, outH = dims.z, outW = dims.w;
  const int TH = geom.x, TW = geom.y, t0 = geom.z;
  const int local = int(pos.x), tile = t0 + local, o = int(pos.y);
  const int n = tile / (TH * TW), ty = tile / TW % TH, tx = tile % TW;
  const auto repair = repair_geometry;
  input += int64_t(n) * repair.channels * repair.height * repair.width;
  float r[TILE_H][4];
  for (int j = 0; j < 4; ++j) {
    float values[TILE_H + 2];
#pragma unroll
    for (int q = 0; q < TILE_H + 2; ++q) {
      values[q] = float(m[(int64_t(q * 4 + j) * O + o) * tiles + local]);
    }
    Conv2dWinogradTransform<TILE_H>::output(values, r, j);
  }
  const float b = bias ? conv2d_winograd_range_load(bias[o]) : 0.0f;
  device T* y = output +
      (NHWC ? int64_t(n) * outH * outW * O + o
            : (int64_t(n) * O + o) * outH * outW);
  for (int i = 0; i < TILE_H; ++i) {
    const int h = TILE_H * ty + i;
    if (h >= outH) {
      continue;
    }
    float v[2] = {r[i][0] + r[i][1] + r[i][2], r[i][1] - r[i][2] - r[i][3]};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if IF_CONSTEXPR (is_same_v<T, half> && TILE_H == 2) {
        v[j] /= kConv2dWinogradHalfWeightScale;
      }
      v[j] += b;
    }
    if (!all(isfinite(float2(T(v[0]), T(v[1]))))) {
      device T* row = y + int64_t(h) * outW * (NHWC ? O : 1);
      for (int j = 0; j < 2 && 2 * tx + j < outW; ++j) {
        row[(2 * tx + j) * (NHWC ? O : 1)] = conv2d_winograd_typed_repair(
            v[j], bias, input, weight, repair, o, h, 2 * tx + j);
      }
      continue;
    }
    device T* row = y + int64_t(h) * outW * (NHWC ? O : 1);
    winograd_store_row<T, NHWC>(
        row, v, 2 * tx, outW, O, bool_constant < TILE_H == 4 && !NHWC > {});
  }
}

#define INSTANTIATE_WINOGRAD(DT)                                               \
  INSTANTIATE_CONV2D("winograd_weights_" #DT, winograd_weights<DT>);           \
  INSTANTIATE_CONV2D("winograd_input_nchw_" #DT, winograd_input<DT, false>);   \
  INSTANTIATE_CONV2D("winograd_input_nhwc_" #DT, winograd_input<DT, true>);    \
  INSTANTIATE_CONV2D("winograd_output_nchw_" #DT, winograd_output<DT, false>); \
  INSTANTIATE_CONV2D("winograd_output_nhwc_" #DT, winograd_output<DT, true>);

INSTANTIATE_WINOGRAD(float)
INSTANTIATE_WINOGRAD(half)
INSTANTIATE_WINOGRAD(bfloat)

INSTANTIATE_CONV2D("winograd_weights_4_bfloat", winograd_weights<bfloat, 4>);
INSTANTIATE_CONV2D("winograd_input_4_bfloat", winograd_input<bfloat, false, 4>);
INSTANTIATE_CONV2D(
    "winograd_output_4_bfloat",
    winograd_output<bfloat, false, 4>);

INSTANTIATE_CONV2D("conv2d_im2col_float", conv2d_im2col<float>);
INSTANTIATE_CONV2D("conv2d_im2col_half", conv2d_im2col<half>);
INSTANTIATE_CONV2D("conv2d_im2col_bfloat", conv2d_im2col<bfloat>);

// Reorders each input row as its even columns then its odd columns, so the
// stride-2 parity variant of conv2d_mpp reads contiguous taps.
template <typename T>
kernel void conv2d_deinterleave(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int2& dims [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int W = dims.x, odd = dims.y, x0 = int(pos.x) * 4;
  device const T* row = input + int64_t(pos.y) * W;
  device T* out = output + int64_t(pos.y) * (W + 1) + x0;
  for (int j = 0; j < 4 && x0 + j <= W; ++j) {
    const int x = x0 + j, w = x < odd ? 2 * x : 2 * (x - odd) + 1;
    out[j] = x == odd - 1 || w >= W ? T(0) : row[w];
  }
}

INSTANTIATE_CONV2D("conv2d_deinterleave_float", conv2d_deinterleave<float>);
INSTANTIATE_CONV2D("conv2d_deinterleave_half", conv2d_deinterleave<half>);
INSTANTIATE_CONV2D("conv2d_deinterleave_bfloat", conv2d_deinterleave<bfloat>);
