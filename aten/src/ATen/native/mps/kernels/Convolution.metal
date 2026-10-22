// Templated conv3d kernels. conv3d_mpp bakes popular geometries into bundled
// entry points; catalog misses use conv3d_simd.
#include <ATen/native/mps/kernels/Convolution.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define INSTANTIATE_CONV2D(NAME, ...) \
  template [[host_name(NAME)]] [[kernel]] decltype(__VA_ARGS__) __VA_ARGS__

// NCHW -> zero-padded NHWC whose rows are the KH filter rows of each output
// row; dims = {C, H, W, Wp}, pad = {padH, padW}, rows = {KH, SH, DH, Hout}.
template <typename T, int TC, int TX, int NTH, bool VECR, bool VECW>
kernel void nchw_to_nhwc(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant int4& dims [[buffer(2)]],
    constant int2& pad [[buffer(3)]],
    constant int4& rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup T tile[TC][TX + 1];
  const int C = dims.x, H = dims.y, W = dims.z, Wp = dims.w;
  const int KH = rows.x, SH = rows.y, DH = rows.z, Hout = rows.w;
  const int X = H * W, Xp = Hout * Wp * KH;
  const int c0 = int(tgid.y) * TC;
  const int x0 = int(tgid.x) * TX;
  device const T* s = src + (int64_t)tgid.z * C * X;
  device T* d = dst + (int64_t)tgid.z * C * Xp;

  if IF_CONSTEXPR (VECR) {
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
  } else if IF_CONSTEXPR (TX * 2 >= NTH) {
    for (int x = tid; x < TX && x0 + x < Xp; x += NTH) {
      const int gx = x0 + x;
      const int kh = gx % KH, w = gx / KH % Wp - pad.y;
      const int h = gx / KH / Wp * SH + kh * DH - pad.x;
      const bool inside = h >= 0 && h < H && w >= 0 && w < W;
      for (int c = 0; c < TC && c0 + c < C; ++c) {
        tile[c][x] = inside ? s[(int64_t)(c0 + c) * X + h * W + w] : T(0);
      }
    }
  } else {
    for (int i = tid; i < TC * TX; i += NTH) {
      const int x = i % TX, c = i / TX;
      const int gc = c0 + c, gx = x0 + x;
      if (gc >= C || gx >= Xp) {
        continue;
      }
      const int kh = gx % KH, w = gx / KH % Wp - pad.y;
      const int h = gx / KH / Wp * SH + kh * DH - pad.x;
      tile[c][x] = h >= 0 && h < H && w >= 0 && w < W
          ? s[(int64_t)gc * X + h * W + w]
          : T(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if IF_CONSTEXPR (VECW) {
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
  } else {
    for (int i = tid; i < TC * TX; i += NTH) {
      const int c = i % TC, x = i / TC;
      const int gc = c0 + c, gx = x0 + x;
      if (gc < C && gx < Xp) {
        d[(int64_t)gx * C + gc] = tile[c][x];
      }
    }
  }
}

// Implicit-GEMM conv3d on simdgroup_matrix per (batch, out-depth) plane; k
// walks DHWIO order, geometry is runtime state, fp32 accumulation. IDX = long
// widens the flat spatial index when a plane exceeds int32.
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
  constexpr int TGP = WM * WN * 32;
  constexpr int WT_M = BM / WM, WT_N = BN / WN;
  constexpr int TM = WT_M / 8, TN = WT_N / 8;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  static_assert(BM % (8 * WM) == 0 && BN % (8 * WN) == 0 && BK % 8 == 0, "");
  static_assert(TGP % BK == 0 && TGP % BN == 0, "cooperative load mappings");

  threadgroup T As[BM * LDA];
  threadgroup T Bs[BK * LDB];

  const int tid = int(sgid) * 32 + int(lane);
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

  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int k = k0 + a_k;
    int di = 0, dh = 0, dw = 0, cc = 0;
    bool k_ok = k < K;
    if (k_ok) {
      cc = k % p.C_in_per_group;
      int r = k / p.C_in_per_group;
      const int kw = r % p.kW;
      r /= p.kW;
      const int kh = r % p.kH;
      const int kd = r / p.kH;
      di = dd * gP.sD - gP.padD + kd * gP.dD;
      dh = kh * p.dH - p.padH;
      dw = kw * p.dW - p.padW;
      k_ok = di >= 0 && di < gP.D;
    }
    for (int rl = a_r0; rl < BM; rl += TGP / BK) {
      const IDX m = m_block + rl;
      T v = T(0);
      if (k_ok && m < M) {
        const int hi = int(m / p.outW) * p.sH + dh;
        const int wi = int(m % p.outW) * p.sW + dw;
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

// int covers planes within int32; one long variant handles the rest
#define INSTANTIATE_CONV3D_SIMD_ALL(DT)                             \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 64, 64, 16, 2, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 32, 64, 16, 1, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd_long", DT, long, 64, 64, 16, 2, 2)

INSTANTIATE_CONV3D_SIMD_ALL(float)
INSTANTIATE_CONV3D_SIMD_ALL(half)
INSTANTIATE_CONV3D_SIMD_ALL(bfloat)

#define INSTANTIATE_NCHW_TO_NHWC(DT, TC, TX, VECR, VECW)             \
  template [[host_name("nchw_to_nhwc_" #DT "_" #TC "_" #TX "_" #VECR \
                       "_" #VECW)]] kernel void                      \
  nchw_to_nhwc<DT, TC, TX, 256, VECR, VECW>(                         \
      device const DT*,                                              \
      device DT*,                                                    \
      constant int4&,                                                \
      constant int2&,                                                \
      constant int4&,                                                \
      uint3,                                                         \
      uint);

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
  for (uint kernel_width_index = 0; kernel_width_index < params.kernel_width;
       ++kernel_width_index) {
    destination_row
        [kernel_width_index * params.input_channels_per_group *
         params.output_channels] =
            source_row[kernel_width_index * params.width_stride];
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

template <int C>
kernel void conv2d_pointwise(
    device bfloat* input [[buffer(0)]],
    device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const bfloat* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using matrix = tensor<device bfloat, extents, tensor_inline>;
  using Mode = matmul2d_descriptor::mode;
  constexpr auto desc = matmul2d_descriptor(
      32, 64, 32, false, false, true, Mode::multiply_accumulate);
  constexpr auto first =
      matmul2d_descriptor(32, 64, 32, false, false, true, Mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<first, execution_simdgroup> initial;
  auto b =
      op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using tile = remove_addrspace_t<decltype(b)>;
  auto acc =
      op.template get_destination_cooperative_tensor<matrix, tile, float>();
  const int plane = p.H * p.W;
  const int o0 = (int(gid.y) * 2 + int(sid) / 2) * 32;
  const int x0 = (int(gid.x) * 2 + int(sid) % 2) * 64;
  if (x0 >= plane || o0 >= p.C_out) {
    return;
  }
  input += int64_t(gid.z) * C * plane;
  output += int64_t(gid.z) * p.C_out * plane;
  matrix w(weight, extents(C, p.C_out), strides{1, C});
  matrix x(input + x0, extents(min(64, plane - x0), 32), strides{1, plane});
  auto wt = w.slice(0, o0);
  b.load(x);
  initial.run(wt, b, acc);
#pragma unroll
  for (int c0 = 32; c0 < C; c0 += 32) {
    matrix v(
        input + int64_t(c0) * plane + x0,
        extents(min(64, plane - x0), 32),
        strides{1, plane});
    auto wv = w.slice(c0, o0);
    b.load(v);
    op.run(wv, b, acc);
  }
  auto res =
      op.template get_destination_cooperative_tensor<matrix, tile, bfloat>();
#pragma unroll
  for (ushort i = 0; i < 64; ++i) {
    const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
    res[i] =
        bfloat(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
  }
  matrix out(output, extents(plane, p.C_out), strides{1, plane});
  res.store(out.slice(x0, o0));
}

#define INSTANTIATE_CONV2D_POINTWISE_REGISTERS(C) \
  INSTANTIATE_CONV2D("conv2d_pointwise_" #C, conv2d_pointwise<C>);

INSTANTIATE_CONV2D_POINTWISE_REGISTERS(128)
INSTANTIATE_CONV2D_POINTWISE_REGISTERS(256)

template <int C>
kernel void conv2d_narrow(
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
  for (ushort i = 0; i < BM * BN / 32; ++i) {
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
    for (ushort i = 0; i < BN * BK / 32; ++i) {
      const int col = (t0 + int(b.get_multidimensional_index(i)[0])) % p.W;
      if (col == 0) {
        left |= 1u << i;
      }
      if (col == p.W - 1) {
        right |= 1u << i;
      }
    }
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int pos = t0 + (kh - 1) * p.W + kw - 1;
        const uint mask = kw == 0 ? left : kw == 2 ? right : 0;
        if (pos + BN <= 0 || pos >= plane) {
          continue;
        }
        matrix w(
            weight + int64_t(kh * 3 + kw) * C * p.C_out,
            E(TRANS ? p.C_out : C, TRANS ? C : p.C_out),
            S{1, TRANS ? p.C_out : C});
#pragma unroll
        for (int c0 = 0; c0 < C; c0 += BK) {
          auto wv = w.slice(TRANS ? o0 : c0, TRANS ? c0 : o0);
          if (pos >= 0 && pos + BN <= plane) {
            matrix xv(
                input + int64_t(c0 * plane) + pos,
                E(BN, min(BK, C - c0)),
                S{1, plane});
            b.load(xv);
          } else {
#pragma unroll
            for (ushort i = 0; i < BN * BK / 32; ++i) {
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
            for (ushort i = 0; i < BN * BK / 32; ++i) {
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
    for (ushort i = 0; i < BM * BN / 32; ++i) {
      const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
      res[i] = bfloat(acc[i] + (o < p.C_out ? float(bias[o]) : 0.0f));
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < BM * BN / 32; ++i) {
      res[i] = bfloat(acc[i]);
    }
  }
  matrix out(output, E(plane, p.C_out), S{1, plane});
  res.store(out.slice(t0, o0));
}

#define INSTANTIATE_CONV2D_NARROW(C) \
  INSTANTIATE_CONV2D("conv2d_narrow_" #C, conv2d_narrow<C>);

INSTANTIATE_CONV2D_NARROW(64)
INSTANTIATE_CONV2D_NARROW(128)
INSTANTIATE_CONV2D_NARROW(176)

// conv2d as one matmul2d per filter tap; ROW tiles cover BN positions of one
// output row, FLAT tiles BN consecutive positions of a plane with pitch Wp.
template <
    typename T,
    bool IN_NHWC,
    bool OUT_NHWC,
    int BM,
    int BN,
    int NSG,
    bool STAGED = false,
    int POINTWISE = 0,
    bool FIXED3 = false>
kernel void conv2d_mpp(
    device T* input [[buffer(0)]],
    device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2dMppParams& geometry [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    device float* partial [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  Conv2dMppParams gp = geometry;
  if constexpr (FIXED3) {
    gp.conv.kH = gp.conv.kW = 3;
    gp.conv.sH = gp.conv.sW = gp.conv.dH = gp.conv.dW = gp.conv.padH =
        gp.conv.padW = 1;
    gp.conv.C_out_per_group = gp.conv.C_out;
    gp.K = gp.K_split = gp.conv.C_in;
    gp.splits = 1;
    gp.Wp = gp.parity = 0;
  }
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
  auto acc = op.template get_destination_cooperative_tensor<
      tensor2d,
      tensor2d,
      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  if constexpr (POINTWISE) {
    const int plane = p.H * p.W, o0 = int(tgid.x) * BM;
    const int x0 = int(tgid.y) * BN, n = int(tgid.z);
    tensor2d x(
        input + int64_t(n) * p.C_in * plane,
        extents(plane, p.C_in),
        strides{1, plane});
    tensor2d w(
        weight + (POINTWISE == 2 ? int64_t(n) * p.C_in * p.C_out : 0),
        extents(p.C_in, p.C_out),
        strides{1, p.C_in});
    auto xt = x.slice(x0, 0), wt = w.slice(0, o0);
    op.run(wt, xt, acc);
    auto res =
        op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
      const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
      res[i] = T(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
    }
    tensor2d out(
        output + int64_t(n) * p.C_out * plane,
        extents(plane, p.C_out),
        strides{1, plane});
    res.store(out.slice(x0, o0));
    return;
  }
  const bool flat = gp.Wp > 0;
  const int og = p.C_out_per_group;
  const int split = int(tgid.z) % gp.splits, k0 = split * gp.K_split;
  const int K = min(gp.K_split, gp.K - k0);
  const int o_tiles = (og + BM - 1) / BM;
  const int g = int(tgid.x) / o_tiles;
  const int o_off = int(tgid.x) % o_tiles * BM;
  const int w_tiles = (p.outW + BN - 1) / BN;
  const int oh = flat ? 0 : int(tgid.y) / w_tiles;
  const int x_off = flat ? int(tgid.y) * BN : int(tgid.y) % w_tiles * BN;
  const int pitch = flat ? gp.Wp : p.W;
  const int plane = p.H * pitch;
  const int nb = int(tgid.z) / gp.splits;
  device T* in = input + int64_t(nb) * p.C_in * plane +
      (g * gp.K + k0) * (IN_NHWC ? 1 : plane);
  const int positions = flat ? (p.outH - 1) * gp.Wp + p.outW : p.outW;
  device T* out = output + int64_t(nb) * p.C_out * p.outH * p.outW +
      (OUT_NHWC ? int64_t(oh) * p.outW * p.C_out + g * og
                : (int64_t(g) * og * p.outH + oh) * p.outW);
  const bool partials = gp.splits > 1 || (flat && gp.Wp != p.outW);
  const int out_pitch = flat ? gp.Wp : p.outW;
  device float* part = partial +
      (int64_t(split) * p.N + nb) * p.C_out * p.outH * out_pitch +
      (OUT_NHWC ? int64_t(oh) * out_pitch * p.C_out + g * og
                : (int64_t(g) * og * p.outH + oh) * out_pitch);
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
        strides{1, OUT_NHWC ? p.C_out : p.outH * out_pitch});
  };
  auto dst_tile = [&](auto t, int n) {
    return OUT_NHWC ? t.slice(o_off, n) : t.slice(n, o_off);
  };
  auto res =
      op.template get_destination_cooperative_tensor<tensor2d, tensor2d, T>();
  // A tap starting in the left padding runs from the first in-bounds column,
  // so slot s holds position s + n0 until the accumulator is shifted.
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
      if (partials) {
        acc.store(
            dst_tile(part_view(min(positions, x_off + BN)), x_off + delta));
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          acc[i] = 0.0f;
        }
        acc.store(dst_tile(part_view(x_off + delta), x_off));
        threadgroup_barrier(mem_flags::mem_device);
        acc.load(dst_tile(part_view(min(positions, x_off + BN)), x_off));
      } else {
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          res[i] = T(acc[i]);
        }
        res.store(
            dst_tile(dst_view(min(positions, x_off + BN)), x_off + delta));
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          res[i] = T(0);
        }
        res.store(dst_tile(dst_view(x_off + delta), x_off));
        threadgroup_barrier(mem_flags::mem_device);
        res.load(dst_tile(dst_view(min(positions, x_off + BN)), x_off));
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
          acc[i] = float(res[i]);
        }
      }
    }
    shift = n0;
    if (kw == p.kW) {
      break;
    }
    const int col = iw + n0 * p.sW;
    for (int kh = 0; kh < p.kH; ++kh) {
      const int ih = flat ? kh * p.dH : oh * p.sH + kh * p.dH - p.padH;
      const int length =
          flat ? plane - ih * pitch - col : (pitch - col + p.sW - 1) / p.sW;
      if (ih < 0 || ih >= p.H || length <= 0) {
        continue;
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
      if constexpr (OUT_NHWC) {
        op.run(xt, w_tile, acc);
      } else {
        op.run(w_tile, xt, acc);
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
    res[i] = T(acc[i] + (p.has_bias ? float(bias[g * og + o]) : 0.0f));
  }
  if constexpr (!STAGED) {
    res.store(dst_tile(dst_view(positions), x_off));
  } else {
    // Small-K tiles are store bound: go through threadgroup memory so each
    // output row leaves as 16-byte vectors.
    using P4 = packed_vec<T, 4>;
    threadgroup T stage[BM * BN];
    tensor<threadgroup T, extents, tensor_inline> st(
        stage, extents(BN, BM), strides{1, BN});
    res.store(st);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int valid = min(BN, positions - x_off), rows = min(BM, og - o_off);
    for (int e = int(tid); e < BM * (BN / 8); e += NSG * 32) {
      const int o = e / (BN / 8), j = e % (BN / 8) * 8;
      if (o >= rows || j >= valid) {
        continue;
      }
      device T* dst = out + int64_t(o_off + o) * p.outH * p.outW + x_off + j;
      const threadgroup T* src = stage + o * BN + j;
      if (j + 8 <= valid) {
        *reinterpret_cast<device P4*>(dst) =
            *reinterpret_cast<const threadgroup P4*>(src);
        *reinterpret_cast<device P4*>(dst + 4) =
            *reinterpret_cast<const threadgroup P4*>(src + 4);
      } else {
        for (int q = 0; j + q < valid; ++q) {
          dst[q] = src[q];
        }
      }
    }
  }
}

#define INSTANTIATE_CONV2D_MPP(DT, IN, IN_NHWC, OUT, OUT_NHWC, BM, BN, NSG) \
  INSTANTIATE_CONV2D(                                                       \
      "conv2d_mpp_" #IN "_" #OUT "_b" #BM "x" #BN "_s" #NSG "_" #DT,        \
      conv2d_mpp<DT, IN_NHWC, OUT_NHWC, BM, BN, NSG>);

#define INSTANTIATE_CONV2D_MPP_STAGED(DT, NSG)         \
  INSTANTIATE_CONV2D(                                  \
      "conv2d_mpp_nhwc_nchw_b64x64_s" #NSG "_st_" #DT, \
      conv2d_mpp<DT, true, false, 64, 64, NSG, true>);

#define INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, BM, BN, NSG)             \
  INSTANTIATE_CONV2D_MPP(DT, nchw, false, nchw, false, BM, BN, NSG) \
  INSTANTIATE_CONV2D_MPP(DT, nchw, false, nhwc, true, BM, BN, NSG)  \
  INSTANTIATE_CONV2D_MPP(DT, nhwc, true, nchw, false, BM, BN, NSG)  \
  INSTANTIATE_CONV2D_MPP(DT, nhwc, true, nhwc, true, BM, BN, NSG)

#define INSTANTIATE_CONV2D_MPP_ALL(DT)           \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 64, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 64, 4)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 64, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 64, 4)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 64, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 64, 4)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 16, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 64, 32, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 32, 32, 2)  \
  INSTANTIATE_CONV2D_MPP_LAYOUTS(DT, 16, 128, 2) \
  INSTANTIATE_CONV2D_MPP_STAGED(DT, 2)           \
  INSTANTIATE_CONV2D_MPP_STAGED(DT, 4)

INSTANTIATE_CONV2D_MPP_ALL(float)
INSTANTIATE_CONV2D_MPP_ALL(half)
INSTANTIATE_CONV2D_MPP_ALL(bfloat)

#define INSTANTIATE_CONV2D_POINTWISE(BM, BN, NSG, PW)                     \
  INSTANTIATE_CONV2D(                                                     \
      "conv2d_mpp_nchw_nchw_b" #BM "x" #BN "_s" #NSG "_pw" #PW "_bfloat", \
      conv2d_mpp<bfloat, false, false, BM, BN, NSG, false, PW>);

INSTANTIATE_CONV2D_POINTWISE(160, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(128, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 128, 8, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 2, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 32, 4, 1)
INSTANTIATE_CONV2D_POINTWISE(64, 64, 4, 2)
INSTANTIATE_CONV2D_POINTWISE(64, 32, 4, 2)

#define INSTANTIATE_CONV2D_FIXED3(BM, NSG)                    \
  INSTANTIATE_CONV2D(                                         \
      "conv2d_mpp_nchw_nchw_b" #BM "x64_s" #NSG "_k3_bfloat", \
      conv2d_mpp<bfloat, false, false, BM, 64, NSG, false, 0, true>);

INSTANTIATE_CONV2D_FIXED3(64, 2)
INSTANTIATE_CONV2D_FIXED3(96, 4)

template <int KS, int BM, int BN, int CC, int S, bool FLAT = false>
kernel void conv2d_patch(
    device bfloat* input [[buffer(0)]],
    device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const bfloat* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int NSG = 4, taps = KS * KS;
  // The row tile stages the entire 16-channel reduction in one step.
  const int channels = S == 1 && !FLAT ? 16 : p.C_in;
  threadgroup bfloat stage[CC * taps * BN];
  using extents = dextents<int32_t, 2>;
  using strides = array<int32_t, 2>;
  using tensor2d = tensor<device bfloat, extents, tensor_inline>;
  using shared2d = tensor<threadgroup bfloat, extents, tensor_inline>;
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
  const int wt = (p.outW + BN - 1) / BN, oh = FLAT ? 0 : int(gid.y) / wt;
  const int ow0 = FLAT ? int(gid.y) * BN : int(gid.y) % wt * BN;
  const int n = int(gid.z), o0 = int(gid.x) * BM, start_col = ow0 % p.W;
  using idx_t = conditional_t<S == 1, int, int64_t>;
  const idx_t plane = idx_t(p.H) * p.W;
  if constexpr (S == 1)
    input += int64_t(n) * channels * plane;
  for (int c0 = 0; c0 < channels; c0 += CC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < CC * KS * (BN / 8); e += NSG * 32) {
      const int cj = e / (BN / 8) / KS, kh = e / (BN / 8) % KS;
      const int j = e % (BN / 8) * 8, c = c0 + cj;
      const int iw = (ow0 + j) * S - (KS == 3 ? p.padW : 0);
      const int ih = oh * S + kh - (KS == 3 ? p.padH : 0);
      device const bfloat* src = input +
          (S == 1 ? idx_t(c) : idx_t(n) * channels + c) * plane +
          idx_t(ih) * p.W + iw;
      bfloat4 a(0), b(0), z(0), d(0), right0(0), right1(0);
      if constexpr (S == 1) {
        const int center = iw + 1, length = FLAT ? plane : p.W;
        const int offset = FLAT ? ih * p.W + center : center;
        ++src;
        bfloat4 lo(0), hi(0);
        bfloat left = 0, right = 0;
        if (c < channels && (FLAT || (ih >= 0 && ih < p.H))) {
          if (offset >= 0 && offset + 8 <= length) {
            const auto raw = *reinterpret_cast<device const packed_uint4*>(src);
            lo = as_type<bfloat4>(uint2(raw.x, raw.y));
            hi = as_type<bfloat4>(uint2(raw.z, raw.w));
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
        a = bfloat4(left, lo.x, lo.y, lo.z);
        b = bfloat4(lo.w, hi.x, hi.y, hi.z);
        z = lo;
        d = hi;
        right0 = bfloat4(lo.y, lo.z, lo.w, hi.x);
        right1 = bfloat4(hi.y, hi.z, hi.w, right);
        if constexpr (FLAT) {
          const int col0 =
              start_col + j >= p.W ? start_col + j - p.W : start_col + j;
          if (c < channels && center + 8 <= plane && offset >= 0 &&
              offset + 8 <= plane && col0 + 8 <= p.W) {
            if (col0 == 0)
              a.x = 0;
            if (col0 + 8 == p.W)
              right1.w = 0;
          } else {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const int col = col0 + q >= p.W ? col0 + q - p.W : col0 + q;
              const bool valid = c < channels && center + q < plane &&
                  offset + q >= 0 && offset + q < plane;
              if (q < 4) {
                if (!valid || col == 0)
                  a[q] = 0;
                if (!valid)
                  z[q] = 0;
                if (!valid || col + 1 == p.W)
                  right0[q] = 0;
              } else {
                if (!valid || col == 0)
                  b[q - 4] = 0;
                if (!valid)
                  d[q - 4] = 0;
                if (!valid || col + 1 == p.W)
                  right1[q - 4] = 0;
              }
            }
          }
        }
      } else if (KS != 3 && c < channels && iw + 16 <= p.W) {
        const auto r0 = *reinterpret_cast<device const packed_uint4*>(src);
        const auto r1 = *reinterpret_cast<device const packed_uint4*>(src + 8);
        a = as_type<bfloat4>(uint2(r0.x, r0.y));
        b = as_type<bfloat4>(uint2(r0.z, r0.w));
        z = as_type<bfloat4>(uint2(r1.x, r1.y));
        d = as_type<bfloat4>(uint2(r1.z, r1.w));
      } else if (c < channels && (KS != 3 || (ih >= 0 && ih < p.H))) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          bfloat4 v(0);
          if ((KS != 3 || iw + q * 4 >= 0) && iw + q * 4 + 4 <= p.W) {
            v = bfloat4(*reinterpret_cast<device const packed_vec<bfloat, 4>*>(
                src + q * 4));
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
      const auto store = [&](int kw, bfloat4 lo, bfloat4 hi) {
        const uint2 low = as_type<uint2>(lo), high = as_type<uint2>(hi);
        *reinterpret_cast<threadgroup packed_uint4*>(
            stage + (cj * taps + kh * KS + kw) * BN + j) =
            packed_uint4(low.x, low.y, high.x, high.y);
      };
      store(
          0,
          S == 1 ? a : bfloat4(a.x, a.z, b.x, b.z),
          S == 1 ? b : bfloat4(z.x, z.z, d.x, d.z));
      if constexpr (KS >= 2) {
        store(
            1,
            S == 1 ? z : bfloat4(a.y, a.w, b.y, b.w),
            S == 1 ? d : bfloat4(z.y, z.w, d.y, d.w));
      }
      if constexpr (KS == 3) {
        const bfloat last = S == 2 && c < channels && ih >= 0 && ih < p.H &&
                iw + 16 >= 0 && iw + 16 < p.W
            ? src[16]
            : bfloat(0);
        store(
            2,
            S == 1 ? right0 : bfloat4(a.z, b.x, b.z, z.x),
            S == 1 ? right1 : bfloat4(z.z, d.x, d.z, last));
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
  auto res = op.template get_destination_cooperative_tensor<
      tensor2d,
      shared2d,
      bfloat>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    const int o = o0 + int(acc.get_multidimensional_index(i)[1]);
    res[i] =
        bfloat(acc[i] + (p.has_bias && o < p.C_out ? float(bias[o]) : 0.0f));
  }
  tensor2d out(
      output + (int64_t(n) * p.C_out * p.outH + oh) * p.outW,
      extents(FLAT ? int(plane) : p.outW, p.C_out),
      strides{1, p.outH * p.outW});
  res.store(out.slice(ow0, o0));
}

#define INSTANTIATE_CONV2D_PATCH(KS, BM, BN, CC, S, FLAT) \
  INSTANTIATE_CONV2D(                                     \
      "conv2d_patch_" #KS "_" #BM "_s" #S "_bfloat",      \
      conv2d_patch<KS, BM, BN, CC, S, FLAT>);

INSTANTIATE_CONV2D_PATCH(1, 128, 32, 128, 2, false)
INSTANTIATE_CONV2D_PATCH(2, 192, 16, 128, 2, false)
INSTANTIATE_CONV2D_PATCH(3, 32, 64, 16, 2, false)
INSTANTIATE_CONV2D_PATCH(3, 256, 16, 64, 2, false)
INSTANTIATE_CONV2D_PATCH(3, 64, 64, 16, 1, false)

INSTANTIATE_CONV2D_PATCH(3, 128, 32, 24, 1, true)

// Stride-2 NCHW conv without a de-interleave pass: per (kh, channel chunk) the
// contiguous input row segment is split into even/odd columns in threadgroup
// memory, so the kW taps read unit-stride views; loads of the next chunk are
// issued before the current chunk's matmuls.
template <typename T, int BM, int BN, int NSG>
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
  constexpr int KC = (BN == 32       ? 64
                          : BN == 64 ? 32
                                     : 16) /
      (sizeof(T) == 4 ? 2 : 1);
  constexpr int VEC = (2 * BN + 13) / 8, HALF = VEC * 4, ROW = 2 * HALF;
  constexpr int NT = NSG * 32, PER = (KC * VEC + NT - 1) / NT;
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
  const int o_tiles = (og + BM - 1) / BM;
  const int g = int(tgid.x) / o_tiles, o_off = int(tgid.x) % o_tiles * BM;
  const int w_tiles = (p.outW + BN - 1) / BN;
  const int oh = int(tgid.y) / w_tiles, x_off = int(tgid.y) % w_tiles * BN;
  const int nb = int(tgid.z), plane = p.H * p.W, x0 = x_off * 2 - p.padW;
  device T* in = input + (int64_t(nb) * p.C_in + g * cg) * plane;
  int kh_lo = 0, kh_hi = p.kH;
  while (kh_lo < p.kH && oh * p.sH + kh_lo * p.dH < p.padH) {
    ++kh_lo;
  }
  while (kh_hi > kh_lo && oh * p.sH + (kh_hi - 1) * p.dH - p.padH >= p.H) {
    --kh_hi;
  }
  const int cchunks = (cg + KC - 1) / KC;
  const int nchunks = (kh_hi - kh_lo) * cchunks;
  vec<S, 4> ev[PER], od[PER];
  auto fetch = [&](int chunk) {
    const int kh = kh_lo + chunk / cchunks, c0 = chunk % cchunks * KC;
    const int ih = oh * p.sH + kh * p.dH - p.padH;
    device const T* row = in + int64_t(c0) * plane + int64_t(ih) * p.W;
    for (int i = 0; i < PER; ++i) {
      const int e = int(tid) + i * NT;
      const int k = e / VEC, j = e % VEC, x = x0 + j * 8;
      vec<S, 4> a = 0, b = 0;
      if (e < KC * VEC && c0 + k < cg) {
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
    const int kh = kh_lo + chunk / cchunks, c0 = chunk % cchunks * KC;
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

// Depthwise conv2d, RH x RW outputs per thread; KW = S = 0 is the generic
// runtime-geometry variant.
template <typename T, int KW, int S, bool FULL>
kernel void conv2d_depthwise(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr int RW = KW == 7 ? 4 : 8, RH = 2;
  constexpr int LW = (RW - 1) * S + KW + (KW == 0);
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
    if IF_CONSTEXPR (KW == 0) {
      for (int r = 0; r < RH; ++r) {
        for (int kh = 0; kh < p.kH; ++kh) {
          const int ih = (oh0 + r) * p.sH + kh * p.dH - p.padH;
          if (ih < 0 || ih >= p.H) {
            continue;
          }
          for (int kw = 0; kw < p.kW; ++kw) {
            const float w = float(wt[kh * p.kW + kw]);
            for (int j = 0; j < RW; ++j) {
              const int iw = (ow0 + j) * p.sW + kw * p.dW - p.padW;
              if (iw >= 0 && iw < p.W) {
                acc[r][j] += float(in[int64_t(ih) * p.W + iw]) * w;
              }
            }
          }
        }
      }
    } else {
      for (int rr = 0; rr < (RH - 1) * S + KW; ++rr) {
        const int ih = oh0 * S - p.padH + rr;
        if (ih < 0 || ih >= p.H) {
          continue;
        }
        device const T* row = in + int64_t(ih) * p.W;
        const int iw0 = ow0 * S - p.padW;
        float win[LW];
        for (int t = 0; t < LW; ++t) {
          win[t] = iw0 + t >= 0 && iw0 + t < p.W ? float(row[iw0 + t]) : 0.0f;
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
    for (int r = 0; r < RH && oh0 + r < p.outH; ++r) {
      device T* out = output +
          ((int64_t(nc / p.C_in) * p.C_out + o) * p.outH + oh0 + r) * p.outW;
      if IF_CONSTEXPR (FULL) {
#pragma unroll
        for (int j = 0; j < RW; ++j) {
          out[ow0 + j] = T(acc[r][j]);
        }
      } else {
        for (int j = 0; j < RW && ow0 + j < p.outW; ++j) {
          out[ow0 + j] = T(acc[r][j]);
        }
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

template <int C, int KW, int S>
kernel void conv2d_tiny(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const bfloat* bias [[buffer(4)]],
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
      if (KW == 3 && (ih < 0 || ih >= p.H)) {
        continue;
      }
      const device bfloat* src = input + (int64_t(n) * C + c) * plane +
          (KW == 3 ? int64_t(ih) * p.W + ow * S : position);
      const packed_uint4 raw =
          *reinterpret_cast<device const packed_uint4*>(src);
      const float4 a = float4(as_type<bfloat4>(uint2(raw.x, raw.y)));
      const float4 z = float4(as_type<bfloat4>(uint2(raw.z, raw.w)));
      float4 b, d;
      if constexpr (S == 2) {
        const packed_uint4 next =
            *reinterpret_cast<device const packed_uint4*>(src + 8);
        b = float4(as_type<bfloat4>(uint2(next.x, next.y)));
        d = float4(as_type<bfloat4>(uint2(next.z, next.w)));
      }
      const float left = KW == 3 && ow > 0 ? float(src[-1]) : 0.0f;
      const float right =
          KW == 3 && S == 1 && ow + 8 < p.W ? float(src[8]) : 0.0f;
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        float4 l, h;
        if constexpr (S == 2) {
          l = kw == 0   ? float4(left, a.y, a.w, z.y)
              : kw == 1 ? float4(a.x, a.z, z.x, z.z)
                        : float4(a.y, a.w, z.y, z.w);
          h = kw == 0   ? float4(z.w, b.y, b.w, d.y)
              : kw == 1 ? float4(b.x, b.z, d.x, d.z)
                        : float4(b.y, b.w, d.y, d.w);
        } else {
          l = KW == 1 || kw == 1 ? a
              : kw == 0          ? float4(left, a.x, a.y, a.z)
                                 : float4(a.y, a.z, a.w, z.x);
          h = KW == 1 || kw == 1 ? z
              : kw == 0          ? float4(a.w, z.x, z.y, z.z)
                                 : float4(z.y, z.z, z.w, right);
        }
#pragma unroll
        for (int o = 0; o < 4; ++o) {
          const float wt =
              float(weight[((o0 + o) * C + c) * KW * KW + kh * KW + kw]);
          lo[o] = fma(l, wt, lo[o]);
          hi[o] = fma(h, wt, hi[o]);
        }
      }
    }
  }
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const float bv = p.has_bias ? float(bias[o0 + o]) : 0.0f;
    const uint2 a = as_type<uint2>(bfloat4(lo[o] + bv));
    const uint2 z = as_type<uint2>(bfloat4(hi[o] + bv));
    *reinterpret_cast<device packed_uint4*>(
        output + (int64_t(n) * p.C_out + o0 + o) * outplane + position) =
        packed_uint4(a.x, a.y, z.x, z.y);
  }
}

#define INSTANTIATE_CONV2D_TINY(C, KW, S) \
  INSTANTIATE_CONV2D(                     \
      "conv2d_tiny_" #C "_" #KW "_" #S "_bfloat", conv2d_tiny<C, KW, S>);

INSTANTIATE_CONV2D_TINY(4, 1, 1)
INSTANTIATE_CONV2D_TINY(1, 3, 1)
INSTANTIATE_CONV2D_TINY(3, 3, 1)
INSTANTIATE_CONV2D_TINY(3, 3, 2)

// One output channel per group: a thread computes consecutive outputs in
// registers instead of a 16-row MPP tile; KW = S = 0 is the generic
// runtime-geometry variant.
template <typename T, int KW, int S, int RH, int RW>
kernel void conv2d_o1(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  using P4 = packed_vec<T, 4>;
  constexpr int LW = KW ? (RW - 1) * S + KW : 1;
  constexpr int LV = (LW + 3) / 4 * 4;
  const int cg = p.C_in_per_group, columns = (p.outW + RW - 1) / RW;
  const int ow0 = int(pos.x) % columns * RW, oh = int(pos.x) / columns * RH;
  const int n = int(pos.y) / p.C_out, g = int(pos.y) % p.C_out;
  device const T* in = input + (int64_t(n) * p.C_in + g * cg) * p.H * p.W;
  device const T* wt =
      weight + int64_t(g) * cg * (RH > 1 ? KW * KW : int64_t(p.kH) * p.kW);
  float acc[RH][RW] = {};
  const int iw0 = ow0 * S - p.padW;
  for (int c = 0; c < cg; ++c) {
    for (int rr = 0; rr < (RH == 1 ? p.kH : RH + KW - 1); ++rr) {
      const int ih = (RH > 1 ? oh + rr : oh * p.sH + rr * p.dH) - p.padH;
      if (ih < 0 || ih >= p.H) {
        continue;
      }
      device const T* row = in + (int64_t(c) * p.H + ih) * p.W;
      if IF_CONSTEXPR (KW == 0) {
        device const T* w = wt + (int64_t(c) * p.kH + rr) * p.kW;
        for (int kw = 0; kw < p.kW; ++kw) {
          const float wv = float(w[kw]);
          for (int j = 0; j < RW; ++j) {
            const int iw = (ow0 + j) * p.sW + kw * p.dW - p.padW;
            if (iw >= 0 && iw < p.W) {
              acc[0][j] += float(row[iw]) * wv;
            }
          }
        }
      } else {
        float win[LV];
        if (iw0 >= 0 && iw0 + LV <= p.W) {
          for (int t = 0; t < LV; t += 4) {
            const P4 v = *reinterpret_cast<device const P4*>(row + iw0 + t);
            for (int q = 0; q < 4; ++q) {
              win[t + q] = float(v[q]);
            }
          }
        } else {
          for (int t = 0; t < LW; ++t) {
            const int iw = iw0 + t;
            win[t] = iw >= 0 && iw < p.W ? float(row[iw]) : 0.0f;
          }
        }
        for (int r = 0; r < RH; ++r) {
          const int kh = rr - r;
          if (RH > 1 && (kh < 0 || kh >= KW)) {
            continue;
          }
          // RH > 1 requires 3x3 filters and at most 128 input channels per
          // group.
          device const T* w = wt +
              (RH > 1 ? (c * KW + kh) * KW : (int64_t(c) * p.kH + kh) * p.kW);
          for (int kw = 0; kw < KW; ++kw) {
            const float wv = float(w[kw]);
            for (int j = 0; j < RW; ++j) {
              acc[r][j] += win[j * S + kw] * wv;
            }
          }
        }
      }
    }
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

#define INSTANTIATE_CONV2D_O1_ALL(DT)   \
  INSTANTIATE_CONV2D_O1(DT, 0, 0, 1, 8) \
  INSTANTIATE_CONV2D_O1(DT, 3, 1, 1, 8)

INSTANTIATE_CONV2D_O1_ALL(float)
INSTANTIATE_CONV2D_O1_ALL(half)
INSTANTIATE_CONV2D_O1_ALL(bfloat)
INSTANTIATE_CONV2D_O1(bfloat, 1, 1, 1, 16)
INSTANTIATE_CONV2D_O1(bfloat, 3, 1, 1, 16)
INSTANTIATE_CONV2D_O1(bfloat, 3, 1, 2, 4)

// geometry = {C_out, outH, outW, partial row pitch}, counts = {splits, N,
// has_bias, out_nhwc}.
template <typename T>
kernel void conv2d_split_reduce(
    device const float* partial [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant int4& geometry [[buffer(3)]],
    constant int4& counts [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int C = geometry.x, outH = geometry.y, outW = geometry.z,
            pitch = geometry.w;
  const int e = int(pos.x);
  const int o = counts.w ? e % C : e / (outH * outW);
  const int r = counts.w ? e / C / outW : e / outW % outH,
            c = counts.w ? e / C % outW : e % outW;
  const int64_t image = int64_t(C) * outH * pitch;
  const int64_t idx = counts.w ? (int64_t(r) * pitch + c) * C + o
                               : (int64_t(o) * outH + r) * pitch + c;
  float v = counts.z ? float(bias[o]) : 0.0f;
  for (int s = 0; s < counts.x; ++s) {
    v += partial[(int64_t(s) * counts.y + pos.y) * image + idx];
  }
  output[pos.y * int64_t(C) * outH * outW + e] = T(v);
}

#define INSTANTIATE_CONV2D_SPLIT_REDUCE(DT) \
  INSTANTIATE_CONV2D("conv2d_split_reduce_" #DT, conv2d_split_reduce<DT>);

INSTANTIATE_CONV2D_SPLIT_REDUCE(float)
INSTANTIATE_CONV2D_SPLIT_REDUCE(half)
INSTANTIATE_CONV2D_SPLIT_REDUCE(bfloat)

// im2col into [N][C * taps][outH * outW]: a thread fills 8 consecutive
// positions of one (c, kh, kw) column.
template <typename T>
kernel void conv2d_im2col(
    device const T* input [[buffer(0)]],
    device T* columns [[buffer(1)]],
    constant Conv2DParams& p [[buffer(2)]],
    constant long4& strides [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]) {
  using P4 = packed_vec<T, 4>;
  const int taps = p.kH * p.kW, positions = p.outH * p.outW;
  const int k = int(pos.y), c = k / taps, kh = k % taps / p.kW, kw = k % p.kW;
  const int p0 = int(pos.x) * 8;
  int oh = p0 / p.outW, ow = p0 % p.outW;
  device const T* in = input + pos.z * strides.x + c * strides.y;
  device T* out =
      columns + (int64_t(pos.z) * p.C_in * taps + k) * positions + p0;
  T v[8];
  for (int j = 0; j < 8; ++j) {
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
  if (p0 + 8 <= positions) {
    *reinterpret_cast<device P4*>(out) = P4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<device P4*>(out + 4) = P4(v[4], v[5], v[6], v[7]);
  } else {
    for (int j = 0; p0 + j < positions; ++j) {
      out[j] = v[j];
    }
  }
}

template <typename T, int BM, int BN, typename idx_t>
kernel void conv2d_winograd_fused(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv2DParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr bool REG_OUTPUT = BM == 32;
  constexpr int KC = REG_OUTPUT ? 16 : 8, NSG = REG_OUTPUT ? 8 : 4;
  union Shared {
    struct {
      T x[16 * KC * BN];
      T w[16 * BM * KC];
    } in;
    float m[8 * BM * BN];
  };
  threadgroup Shared sm;
  constexpr int TM = REG_OUTPUT ? 1 : BM / 8, TN = REG_OUTPUT ? 1 : BN / 8;
  constexpr int TP = REG_OUTPUT ? 16 : 4;
  const int mo = REG_OUTPUT ? int(sg) / 2 * 8 : 0;
  const int no = REG_OUTPUT ? int(sg) % 2 * 8 : 0;
  const int th = (p.outH + 1) / 2, tw = (p.outW + 1) / 2;
  const int t0 = int(gid.y) * BN, o0 = int(gid.x) * BM, n = int(gid.z);
  input += int64_t(n) * p.C_in * p.H * p.W;
  output += int64_t(n) * p.C_out * p.outH * p.outW;
  // Only two elements of each simdgroup matrix belong to a lane.
  float2 acc[TP][TM][TN] = {};
  for (int c0 = 0; c0 < p.C_in; c0 += KC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int e = int(tid); e < KC * BN; e += NSG * 32) {
      const int c = c0 + e / BN, t = t0 + e % BN;
      const int h0 = 2 * (t / tw) - p.padH, w0 = 2 * (t % tw) - p.padW;
      float4 d[4];
      if (c < p.C_in && t < th * tw && h0 >= 0 && h0 + 3 < p.H && w0 >= 0 &&
          w0 + 3 < p.W) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          device const T* src = input + (idx_t(c) * p.H + h0 + h) * p.W + w0;
          d[h] = float4(*reinterpret_cast<device const packed_vec<T, 4>*>(src));
        }
      } else {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            d[h][w] = c < p.C_in && t < th * tw && h0 + h >= 0 &&
                    h0 + h < p.H && w0 + w >= 0 && w0 + w < p.W
                ? float(input[(idx_t(c) * p.H + h0 + h) * p.W + w0 + w])
                : 0.0f;
          }
        }
      }
      float4 r[4] = {d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3]};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const float4 v(
            r[h].x - r[h].z, r[h].y + r[h].z, r[h].z - r[h].y, r[h].y - r[h].w);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          sm.in.x[(h * 4 + w) * KC * BN + e] = T(v[w]);
        }
      }
    }
#pragma unroll
    for (int e = int(tid); e < BM * KC; e += NSG * 32) {
      const int o = o0 + e / KC, c = c0 + e % KC;
      float3 d[3];
#pragma unroll
      for (int h = 0; h < 3; ++h) {
        d[h] = o < p.C_out && c < p.C_in
            ? float3(*reinterpret_cast<device const packed_vec<T, 3>*>(
                  weight + (idx_t(o) * p.C_in + c) * 9 + h * 3))
            : float3(0);
      }
      float3 r[4] = {
          d[0], 0.5f * (d[0] + d[1] + d[2]), 0.5f * (d[0] - d[1] + d[2]), d[2]};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const float4 v(
            r[h].x,
            0.5f * (r[h].x + r[h].y + r[h].z),
            0.5f * (r[h].x - r[h].y + r[h].z),
            r[h].z);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          sm.in.w[(h * 4 + w) * BM * KC + e] = T(v[w]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int v = 0; v < TP; ++v) {
      const int q = REG_OUTPUT ? v : int(sg) + v * NSG;
#pragma unroll
      for (int k = 0; k < KC; k += 8) {
        simdgroup_matrix<T, 8, 8> a[TM], b[TN];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
          simdgroup_load(
              a[m], sm.in.w + q * BM * KC + (mo + m * 8) * KC + k, KC);
        }
#pragma unroll
        for (int t = 0; t < TN; ++t) {
          simdgroup_load(b[t], sm.in.x + q * KC * BN + k * BN + no + t * 8, BN);
        }
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
          for (int t = 0; t < TN; ++t) {
            simdgroup_matrix<float, 8, 8> result;
            result.thread_elements()[0] = acc[v][m][t].x;
            result.thread_elements()[1] = acc[v][m][t].y;
            simdgroup_multiply_accumulate(result, a[m], b[t], result);
            acc[v][m][t] = float2(
                result.thread_elements()[0], result.thread_elements()[1]);
          }
        }
      }
    }
  }
  if IF_CONSTEXPR (REG_OUTPUT) {
    const int lane = tid % 32, quad = lane / 4;
    const int fm = (quad & 4) + (lane / 2) % 4;
    const int fn = (quad & 2) * 2 + (lane % 2) * 2;
    const int o = o0 + mo + fm, t = t0 + no + fn;
    if (o >= p.C_out)
      return;
    const float bv = p.has_bias ? float(bias[o]) : 0.0f;
    float2 r[2][4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      r[0][q] = acc[q][0][0] + acc[q + 4][0][0] + acc[q + 8][0][0];
      r[1][q] = acc[q + 4][0][0] - acc[q + 8][0][0] - acc[q + 12][0][0];
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 a = r[h][0] + r[h][1] + r[h][2] + bv;
      const float2 b = r[h][1] - r[h][2] - r[h][3] + bv;
      const int oh = 2 * (t / tw) + h, ow = 2 * (t % tw);
      if (t + 1 < th * tw && oh < p.outH && ow + 3 < p.outW) {
        device T* dst = output + (idx_t(o) * p.outH + oh) * p.outW + ow;
        *reinterpret_cast<device packed_vec<T, 4>*>(dst) =
            packed_vec<T, 4>(T(a.x), T(b.x), T(a.y), T(b.y));
      } else {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          const int hv = 2 * ((t + v) / tw) + h, wv = 2 * ((t + v) % tw);
          if (t + v < th * tw && hv < p.outH) {
            device T* dst = output + (idx_t(o) * p.outH + hv) * p.outW + wv;
            dst[0] = T(a[v]);
            if (wv + 1 < p.outW)
              dst[1] = T(b[v]);
          }
        }
      }
    }
  } else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each simdgroup owns one transform column; reduce its four rows first.
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const float2 a = acc[0][m][t] + acc[1][m][t] + acc[2][m][t];
        const float2 b = acc[1][m][t] - acc[2][m][t] - acc[3][m][t];
        const int lane = tid % 32, quad = lane / 4;
        const int fm = (quad & 4) + (lane / 2) % 4;
        const int fn = (quad & 2) * 2 + (lane % 2) * 2;
        const int e = (m * 8 + fm) * BN + t * 8 + fn;
        sm.m[int(sg) * BM * BN + e] = a.x;
        sm.m[int(sg) * BM * BN + e + 1] = a.y;
        sm.m[(4 + int(sg)) * BM * BN + e] = b.x;
        sm.m[(4 + int(sg)) * BM * BN + e + 1] = b.y;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int e = int(tid); e < BM * BN; e += NSG * 32) {
      const int o = o0 + e / BN, t = t0 + e % BN;
      if (o >= p.C_out || t >= th * tw)
        continue;
      float r[2][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        r[0][j] = sm.m[j * BM * BN + e];
        r[1][j] = sm.m[(4 + j) * BM * BN + e];
      }
      const float bv = p.has_bias ? float(bias[o]) : 0.0f;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int oh = 2 * (t / tw) + h, ow = 2 * (t % tw);
        if (oh < p.outH) {
          const float2 v(
              r[h][0] + r[h][1] + r[h][2] + bv,
              r[h][1] - r[h][2] - r[h][3] + bv);
          device T* dst = output + (idx_t(o) * p.outH + oh) * p.outW + ow;
          if (ow + 1 < p.outW) {
            *reinterpret_cast<device packed_vec<T, 2>*>(dst) =
                packed_vec<T, 2>(T(v.x), T(v.y));
          } else {
            dst[0] = T(v.x);
          }
        }
      }
    }
  }
}

#define INSTANTIATE_CONV2D_WINOGRAD_FUSED(BM, BN, IDX)      \
  INSTANTIATE_CONV2D(                                       \
      "conv2d_winograd_fused_" #BM "x" #BN "_bfloat_" #IDX, \
      conv2d_winograd_fused<bfloat, BM, BN, IDX>);

INSTANTIATE_CONV2D_WINOGRAD_FUSED(8, 32, int)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(8, 32, long)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(8, 8, int)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(8, 8, long)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(16, 16, int)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(16, 16, long)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(32, 16, int)
INSTANTIATE_CONV2D_WINOGRAD_FUSED(32, 16, long)

// Winograd F(TILE_H x 2, 3x3), with 4 * (TILE_H + 2) coefficients.
// Transforms are channel-major and chunked along the tile dimension.
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
    const float g0 = float(g[j]), g1 = float(g[3 + j]), g2 = float(g[6 + j]);
    if constexpr (TILE_H == 4) {
      r[0][j] = 0.25f * g0;
      r[1][j] = -(g0 + g1 + g2) / 6.0f;
      r[2][j] = (-g0 + g1 - g2) / 6.0f;
      r[3][j] = g0 / 24.0f + g1 / 12.0f + g2 / 6.0f;
      r[4][j] = g0 / 24.0f - g1 / 12.0f + g2 / 6.0f;
      r[5][j] = g2;
    } else {
      r[0][j] = g0;
      r[1][j] = 0.5f * (g0 + g1 + g2);
      r[2][j] = 0.5f * (g0 - g1 + g2);
      r[3][j] = g2;
    }
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
    if constexpr (TILE_H == 4 && !NHWC) {
      if (h >= 0 && h < H && w0 >= 0 && w0 + 3 < W) {
        const float4 v =
            float4(*reinterpret_cast<device const packed_vec<T, 4>*>(
                x + int64_t(h) * W + w0));
        for (int j = 0; j < 4; ++j) {
          d[i][j] = v[j];
        }
        continue;
      }
    }
    for (int j = 0; j < 4; ++j) {
      const int w = w0 + j;
      d[i][j] = h >= 0 && h < H && w >= 0 && w < W
          ? float(x[NHWC ? int64_t(h * W + w) * C : h * W + w])
          : 0.0f;
    }
  }
  float r[TILE_H + 2][4];
  for (int j = 0; j < 4; ++j) {
    if constexpr (TILE_H == 4) {
      r[0][j] = 4 * d[0][j] - 5 * d[2][j] + d[4][j];
      r[1][j] = -4 * (d[1][j] + d[2][j]) + d[3][j] + d[4][j];
      r[2][j] = 4 * (d[1][j] - d[2][j]) - d[3][j] + d[4][j];
      r[3][j] = -2 * d[1][j] - d[2][j] + 2 * d[3][j] + d[4][j];
      r[4][j] = 2 * d[1][j] - d[2][j] - 2 * d[3][j] + d[4][j];
      r[5][j] = 4 * d[1][j] - 5 * d[3][j] + d[5][j];
    } else {
      r[0][j] = d[0][j] - d[2][j];
      r[1][j] = d[1][j] + d[2][j];
      r[2][j] = d[2][j] - d[1][j];
      r[3][j] = d[1][j] - d[3][j];
    }
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
    uint2 pos [[thread_position_in_grid]]) {
  const int O = dims.x, tiles = dims.y, outH = dims.z, outW = dims.w;
  const int TH = geom.x, TW = geom.y, t0 = geom.z;
  const int local = int(pos.x), tile = t0 + local, o = int(pos.y);
  const int n = tile / (TH * TW), ty = tile / TW % TH, tx = tile % TW;
  float r[TILE_H][4];
  for (int j = 0; j < 4; ++j) {
    const float m0 = float(m[(int64_t(j) * O + o) * tiles + local]),
                m1 = float(m[(int64_t(4 + j) * O + o) * tiles + local]);
    const float m2 = float(m[(int64_t(8 + j) * O + o) * tiles + local]),
                m3 = float(m[(int64_t(12 + j) * O + o) * tiles + local]);
    if constexpr (TILE_H == 4) {
      const float m4 = float(m[(int64_t(16 + j) * O + o) * tiles + local]),
                  m5 = float(m[(int64_t(20 + j) * O + o) * tiles + local]);
      r[0][j] = m0 + m1 + m2 + m3 + m4;
      r[1][j] = m1 - m2 + 2 * (m3 - m4);
      r[2][j] = m1 + m2 + 4 * (m3 + m4);
      r[3][j] = m1 - m2 + 8 * (m3 - m4) + m5;
    } else {
      r[0][j] = m0 + m1 + m2;
      r[1][j] = m1 - m2 - m3;
    }
  }
  const float b = bias ? float(bias[o]) : 0.0f;
  device T* y = output +
      (NHWC ? int64_t(n) * outH * outW * O + o
            : (int64_t(n) * O + o) * outH * outW);
  for (int i = 0; i < TILE_H; ++i) {
    const int h = TILE_H * ty + i;
    if (h >= outH) {
      continue;
    }
    const float v[2] = {
        r[i][0] + r[i][1] + r[i][2], r[i][1] - r[i][2] - r[i][3]};
    if constexpr (TILE_H == 4 && !NHWC) {
      if (2 * tx + 1 < outW) {
        *reinterpret_cast<device packed_vec<T, 2>*>(
            y + int64_t(h) * outW + 2 * tx) =
            packed_vec<T, 2>(T(v[0] + b), T(v[1] + b));
        continue;
      }
    }
    for (int j = 0; j < 2; ++j) {
      const int w = 2 * tx + j;
      if (w < outW) {
        y[NHWC ? int64_t(h * outW + w) * O : h * outW + w] = T(v[j] + b);
      }
    }
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

#define INSTANTIATE_CONV2D_IM2COL(DT) \
  INSTANTIATE_CONV2D("conv2d_im2col_" #DT, conv2d_im2col<DT>);

INSTANTIATE_CONV2D_IM2COL(float)
INSTANTIATE_CONV2D_IM2COL(half)
INSTANTIATE_CONV2D_IM2COL(bfloat)

// Each row of dims.x columns becomes [even columns, 0, odd columns] with the
// odd columns starting at dims.y.
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

#define INSTANTIATE_CONV2D_DEINTERLEAVE(DT) \
  INSTANTIATE_CONV2D("conv2d_deinterleave_" #DT, conv2d_deinterleave<DT>);

INSTANTIATE_CONV2D_DEINTERLEAVE(float)
INSTANTIATE_CONV2D_DEINTERLEAVE(half)
INSTANTIATE_CONV2D_DEINTERLEAVE(bfloat)
