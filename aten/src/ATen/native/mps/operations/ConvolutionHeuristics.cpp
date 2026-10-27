#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/operations/ConvolutionHeuristics.h>
#include <c10/metal/common.h>

#include <algorithm>
#include <limits>

namespace at::native::mps {

// Estimate padded work adjusted for staging efficiency and device occupancy.
// Lower-cost tiles do less wasted work while keeping enough threadgroups active.
float Conv3dMppTile::cost(const Conv3DParams& params, int64_t groups, int64_t cores) const {
  constexpr float kOccupancyFill = 300.0f;
  constexpr float kSimdgroupOverhead = 8.0f;
  constexpr float kOutputChannelOverhead = 90.0f;
  constexpr float kWeightLoadCost = 0.15f;
  constexpr float kStagingOverhead = 0.4f;

  const auto& conv2d = params.conv2d;
  const auto output_channel_tiles =
      c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(output_channels));
  const auto width_tiles = c10::metal::ceil_div(int64_t(conv2d.outW), int64_t(output_width));
  const auto height_tiles = c10::metal::ceil_div(int64_t(conv2d.outH), int64_t(output_height));
  const auto threadgroups =
      int64_t(output_channel_tiles) * groups * width_tiles * height_tiles * conv2d.N * params.outD;
  const auto kernel_elements = params.kD * conv2d.kH * conv2d.kW;
  const auto padded_output = float(threadgroups) * output_channels * output_width * output_height;

  const auto outputs_per_simdgroup = float(output_width * output_height) / simdgroups;
  const auto input_tile_width =
      int64_t(output_width - 1) * conv2d.sW + (conv2d.kW - 1) * conv2d.dW + 1;
  const auto input_tile_height =
      int64_t(output_height - 1) * conv2d.sH + (conv2d.kH - 1) * conv2d.dH + 1;
  const auto staging_ratio = float(output_width * output_height) /
      (float(input_tile_width * input_tile_height) + kWeightLoadCost * output_channels * kernel_elements);
  const auto tile_efficiency = outputs_per_simdgroup / (outputs_per_simdgroup + kSimdgroupOverhead) *
      (output_channels / (output_channels + kOutputChannelOverhead)) *
      (staging_ratio / (staging_ratio + kStagingOverhead));
  const auto device_utilization = float(threadgroups) / (threadgroups + cores * kOccupancyFill);
  return padded_output / (tile_efficiency * device_utilization);
}

Conv3dMppTile select_conv3d_mpp_tile(const Conv3DParams& params, int64_t groups) {
  static const int64_t cores = []() {
    const unsigned device_cores = at::mps::MPSDevice::getInstance()->getCoreCount();
    return device_cores > 0 ? static_cast<int64_t>(device_cores) : 16;
  }();
  // Keep in sync with INSTANTIATE_CONV3D_MPP_TILES in Convolution.metal.
  const Conv3dMppTile candidates[] = {
      {.output_channels = 32, .output_width = 8, .output_height = 8, .simdgroups = 2},
      {.output_channels = 32, .output_width = 16, .output_height = 8, .simdgroups = 4},
      {.output_channels = 64, .output_width = 8, .output_height = 8, .simdgroups = 2},
      {.output_channels = 64, .output_width = 16, .output_height = 8, .simdgroups = 4},
      {.output_channels = 128, .output_width = 8, .output_height = 8, .simdgroups = 4},
  };
  auto best_tile = candidates[0];
  auto best_cost = std::numeric_limits<float>::max();
  for (const auto& candidate : candidates) {
    const auto candidate_cost = candidate.cost(params, groups, cores);
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_tile = candidate;
    }
  }
  return best_tile;
}

Conv3dSimdTile select_conv3d_simd_tile(int64_t output_height, int64_t output_width, bool use_long_index) {
  if (use_long_index) {
    return {.block_rows = 64, .block_columns = 64, .simdgroups_rows = 2, .simdgroups_columns = 2};
  }
  if (output_height * output_width < 48) {
    return {.block_rows = 32, .block_columns = 64, .simdgroups_rows = 1, .simdgroups_columns = 2};
  }
  return {.block_rows = 64, .block_columns = 64, .simdgroups_rows = 2, .simdgroups_columns = 2};
}

// Materializing im2col only wins for patch embeddings and occupancy-starved
// single-batch convolutions with a large reduction.
bool conv3d_prefer_im2col(const Tensor& input,
                          const Tensor& weight,
                          IntArrayRef stride,
                          IntArrayRef padding,
                          IntArrayRef dilation,
                          int64_t groups,
                          const Tensor& output) {
  if (groups != 1 || dilation[0] != 1 || dilation[1] != 1 || dilation[2] != 1) {
    return false;
  }
  const int64_t kernel_depth = weight.size(2);
  const int64_t kernel_height = weight.size(3);
  const int64_t kernel_width = weight.size(4);
  const int64_t reduction_size = weight.size(1) * kernel_depth * kernel_height * kernel_width;

  // The long-indexed direct kernel owns oversized planes, and the im2col
  // matrix must fit MPSGraph's int32 element limit.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t output_positions = output.size(0) * output.size(2) * output.size(3) * output.size(4);
  if (input.size(1) * input.size(3) * input.size(4) > kInt32Max ||
      output.size(1) * output.size(3) * output.size(4) > kInt32Max ||
      output_positions > kInt32Max / std::max<int64_t>(reduction_size, 1)) {
    return false;
  }
  const bool patch_embedding = padding[0] == 0 && padding[1] == 0 && padding[2] == 0 &&
      stride[0] == kernel_depth && stride[1] == kernel_height && stride[2] == kernel_width;
  if (patch_embedding) {
    return reduction_size >= 256;
  }
  const int64_t output_volume = output.size(2) * output.size(3) * output.size(4);
  return input.scalar_type() == kFloat && input.size(0) == 1 && output_volume <= 256 && reduction_size >= 4096;
}

const Conv2dDevice& conv2d_device() {
  static const Conv2dDevice device = [] {
    const bool apple10 = at::mps::is_apple_family_or_newer(at::mps::AppleGPUFamily::APPLE_10_PLUS);
    const int64_t cores = std::max<int64_t>(at::mps::MPSDevice::getInstance()->getCoreCount(), 8);
    return Conv2dDevice{.apple9 = at::mps::is_apple_family_or_newer(at::mps::AppleGPUFamily::APPLE_9_PLUS),
                        .apple10 = apple10,
                        .cores = cores,
                        .target = (apple10 ? 8 : 16) * cores};
  }();
  return device;
}

Conv2dPlan select_conv2d_kernel(const Conv2DParams& p,
                                const Tensor& input,
                                const Tensor& weight,
                                const Tensor& output,
                                int64_t groups,
                                const Conv2dDevice& dev) {
  using c10::metal::ceil_div;
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  constexpr int64_t kTile = 64;
  const int64_t N = p.N, C_in = p.C_in, C_out = p.C_out, H = p.H, W = p.W, outH = p.outH, outW = p.outW;
  const int64_t kH = p.kH, kW = p.kW, Cg = p.C_in_per_group, Og = p.C_out_per_group;
  const int64_t plane = H * W, positions = outH * outW, channels = std::min(C_in, C_out);
  const int64_t padded_h = H + 2 * p.padH, padded_w = W + 2 * p.padW;
  const bool apple10 = dev.apple10;
  const int64_t target = dev.target;
  const bool in_nhwc = !input.is_contiguous(), out_nhwc = !output.is_contiguous();
  const bool bf16 = input.scalar_type() == kBFloat16;
  const bool bfloat_simd = dev.apple9 && bf16;
  const bool bfloat_nchw = bfloat_simd && groups == 1 && !in_nhwc && !out_nhwc;
  const bool even_offset = input.storage_offset() % 2 == 0; // packed bfloat2 loads
  const bool s1 = p.sH == 1 && p.sW == 1, stride2 = p.sH == 2 && p.sW == 2, strided = p.sH > 1 || p.sW > 1;
  const bool d1 = p.dH == 1 && p.dW == 1, p1 = p.padH == 1 && p.padW == 1, p0 = p.padH == 0 && p.padW == 0;
  const bool k3 = kH == 3 && kW == 3;
  const bool pointwise = kH == 1 && kW == 1 && p0;
  const bool patch_embed = kH == p.sH && kW == p.sW && p0 && d1;
  const bool packed3 = apple10 && k3 && d1 && p1;

  // Two-pass Winograd: F(2x2) once both channel counts are large, F(4x2) for a single tall map or a
  // batch of small ones. Below ~128 channels Apple9 keeps the fused kernel (fused_mid) instead.
  const bool wino2x2 = apple10 ? channels >= 160 || (channels >= 128 && N * plane <= 16384)
                               : channels >= 96 || (channels >= 48 && C_in * C_out >= 11000);
  const bool fused_mid = !apple10 && bfloat_simd && !in_nhwc && !out_nhwc && C_in >= 16 && N * plane >= 16384 &&
      ((C_in <= 192 && C_out >= 32 && channels < 128) || (C_in >= 256 && C_in <= 512 && C_out == 32));
  const bool single_tall = N == 1 &&
      ((C_out == 128 && plane >= (apple10 ? 512 : 4096) && (plane <= 32768 || (apple10 ? C_in >= 256 : C_in <= 192))) ||
       (!apple10 && C_in <= 192 && C_out == 64 && plane >= 16384) ||
       (apple10 && C_in <= 128 && C_out > 128 && C_out <= 256 && plane >= 32768 && plane < 262144));
  const bool wino4x2 = bfloat_simd && !in_nhwc && !out_nhwc && C_in >= 128 &&
      (single_tall || (N >= 8 && C_out >= (apple10 ? 128 : 32) && C_out <= 128 && std::max(H, W) <= 80));
  const bool winograd = groups == 1 && k3 && s1 && d1 && input.scalar_type() != kHalf && (wino2x2 || wino4x2);
  if (winograd && (!fused_mid || wino4x2)) {
    return {wino4x2 ? Conv2dKernel::Winograd4x2 : Conv2dKernel::Winograd2x2};
  }

  const bool tiny_s2 = Cg == 3 && stride2 && W % 16 == 0 && Og <= (apple10 ? 16 : 32);
  const bool tiny_3x3 = k3 && p1 && d1 && W % 8 == 0 &&
      ((!strided && (Cg == 1 || (Cg == 3 && (!apple10 || Og <= 32)))) || tiny_s2);
  const bool tiny_1x1 = pointwise && !strided && Cg == 4 && plane % 8 == 0;
  if (bfloat_nchw && Og % 4 == 0 && even_offset && output.storage_offset() % 2 == 0 && (tiny_1x1 || tiny_3x3)) {
    return {.kernel = Conv2dKernel::Tiny, .simdgroups = 8};
  }

  const int64_t o1_threads = ceil_div(outW, int64_t(8)) * outH, o1_work = o1_threads * N * C_out;
  if (Og == 1 && !in_nhwc && !out_nhwc && o1_work >= 4096) {
    const bool wide = bfloat_simd && p.sW == 1 && p.dW == 1 &&
        ((kW == 1 && Cg <= 32) || (kW == 3 && Cg <= 16 && o1_work >= 65536));
    const bool paired = bfloat_simd && k3 && s1 && d1 && Cg <= 128 && o1_work >= 16384 &&
        (apple10 || (Cg <= 16 && o1_threads >= 4096));
    return {.kernel = Conv2dKernel::SingleOutput,
            .rows = paired ? 2 : 1,
            .columns = paired ? 4 : wide ? 16 : 8,
            .simdgroups = 8,
            .fixed_taps = (kW == 3 || wide) && p.sW == 1 && p.dW == 1};
  }

  const bool small_wino = Og > 1 && Og <= 4 && Cg >= 256 && Cg <= 384 && plane < 131072 && N * plane >= 16384;
  const bool mid_wino = Og > 1 && Og <= 16 && plane * (Og <= 8 ? 1 : N) >= 65536;
  if (!apple10 && bfloat_nchw && k3 && s1 && d1 && Cg >= 8 && (fused_mid || small_wino || mid_wino)) {
    const int64_t rows = fused_mid ? 32 : Og <= 8 ? 8 : 16;
    return {.kernel = Conv2dKernel::WinogradFused,
            .rows = rows,
            .columns = small_wino ? 8 : rows == 8 ? 32 : 16,
            .simdgroups = fused_mid ? 8 : 4};
  }

  const bool heavy_pointwise = C_out * C_in * positions >= (int64_t(1) << 29);
  const bool single_pointwise = apple10 && N == 1 && bf16 && heavy_pointwise &&
      ((Cg > 16 && Cg < 128 && Og >= 256) || (Cg > 128 && Cg <= 192 && Og > 32 && Og <= 64));
  if (pointwise && !strided && apple10 && bfloat_nchw && N == 1 && (Cg == 128 || Cg == 256) && Og >= 128 &&
      Og <= 256 && plane >= 262144 && plane <= kInt32Max / std::max(Cg, Og)) {
    return {.kernel = Conv2dKernel::PointwisePlane, .rows = 64, .columns = 128, .simdgroups = 4};
  }
  if (pointwise && !strided && apple10 && groups == 1 && N == 1 && !in_nhwc && !out_nhwc && !single_pointwise &&
      (Cg <= 16 || heavy_pointwise)) {
    return {Conv2dKernel::PointwiseMatmul};
  }

  const bool narrow_batch =
      N >= 8 && (Cg == 64 || Cg == 128) && W >= 32 && W <= 96 && positions >= 1024 && positions <= 16384;
  const bool narrow_large = N == 1 && Cg == 176 && W >= 128 && W <= 512 && positions >= 16384 && positions <= 131072;
  const int64_t narrow_columns = Cg == 64 ? 64 : 128;
  if (packed3 && !strided && bfloat_nchw && (narrow_batch || narrow_large) && Og > 32 && Og <= 64 &&
      N * ceil_div(positions, narrow_columns) >= target) {
    return {.kernel = Conv2dKernel::Narrow, .columns = narrow_columns, .simdgroups = Cg == 128 ? 8 : 4};
  }

  const bool sample_s2 = pointwise && stride2 && Cg >= 128 && Cg <= 256 && outW >= 24 &&
      ceil_div(Og, int64_t(128)) * outH * ceil_div(outW, int64_t(32)) * N >= target;
  const bool patch2 =
      patch_embed && kH == 2 && kW == 2 && N > 1 && Cg >= 96 && Cg <= 192 && Og % 192 == 0 && H % 2 == 0;
  const bool patch_flat = packed3 && !strided && Cg > 32 && Cg < 128 && Og > 128 && Og <= 256 && W >= 32 &&
      W % 2 == 0 && positions >= 4096 && positions <= 32768;
  const bool patch_s1 = packed3 && !strided && ((Cg == 16 && Og >= 256 && W % 8 == 0) || patch_flat) && even_offset &&
      C_in * plane <= kInt32Max && C_out * positions <= kInt32Max;
  const bool patch_s2 = packed3 && stride2 &&
      ((Cg > 8 && Cg <= 16 && Og > 16 && Og <= 32) ||
       (Cg >= 128 && Og > 128 && Og <= 256 && outW > 8 && outW <= 16 && N * outH >= target));
  const bool packed_patch = (W % 2 == 0 || (sample_s2 && H % 2 == 0)) && even_offset;
  if (bfloat_nchw && (patch_s1 || patch_s2 || (packed_patch && (sample_s2 || patch2)))) {
    Conv2dPlan plan{.kernel = Conv2dKernel::Patch, .simdgroups = 4, .flat = patch_flat};
    if (patch_flat) {
      plan.rows = 128;
      plan.columns = 32;
    } else if (patch_s1) {
      plan.rows = 64;
      plan.columns = 64;
    } else if (patch_s2) {
      plan.rows = Cg <= 16 ? 32 : 256;
      plan.columns = Cg <= 16 ? 64 : 16;
    } else if (sample_s2) {
      plan.rows = 128;
      plan.columns = 32;
    } else {
      plan.rows = 192;
      plan.columns = 16;
    }
    return plan;
  }

  const int64_t im2col_bytes = N * positions * C_in * kH * kW * input.element_size();
  const bool direct_columns = bfloat_nchw && stride2 && d1 && kH == kW && (kH == 2 || kH == 3) && Og >= 512 &&
      im2col_bytes <= (int64_t(128) << 20) &&
      (apple10 ? (N > 1 && Cg >= 384 && positions <= 256) ||
               (N == 1 && Cg >= 512 && positions >= 512 && (p.padH > 0 || p.padW > 0 || positions >= 8192))
               : N == 1 && Cg >= 512 && Og >= 2 * Cg && positions >= 512 && positions <= 4096);
  // Fraction of a 64- or 32-wide position tile past the end of an output row.
  const float waste64 = 1.0f - float(outW) / float(ceil_div(outW, kTile) * kTile);
  const float waste32 = 1.0f - float(outW) / float(ceil_div(outW, int64_t(32)) * 32);
  const bool flat = !strided && waste64 > 11.0f / float(Og) + 0.08f && waste64 * float(Cg * kH * kW) > 1000.0f;
  const bool narrow_better = waste64 - waste32 >= 0.14f;
  const int64_t bm = Og <= 16 ? 16 : Og <= 32 ? 32 : Og <= 64 ? 64 : Og <= 96 && bfloat_simd ? 96 : 128;
  const auto s2_threadgroups = [&](int64_t bn) { return ceil_div(Og, bm) * groups * outH * ceil_div(outW, bn) * N; };
  const int64_t bn = bm == 96 || outW < kTile || narrow_better || s2_threadgroups(kTile) < target ? 32
      : Cg <= 16 && Og <= 32 && outW >= 128 && s2_threadgroups(128) >= target                    ? 128
                                                                                                : kTile;
  if (!pointwise && !patch_embed && !in_nhwc && !out_nhwc && p.sW == 2 && (kW - 1) * p.dW <= 6 && Cg > 8 && Cg <= 128 &&
      outW >= 24 && std::min(waste64, waste32) < 0.35f && s2_threadgroups(bn) >= 4 * dev.cores) {
    return {.kernel = Conv2dKernel::Stride2, .rows = bm, .columns = bn, .simdgroups = bm == 128 ? 8 : 4};
  }

  // conv2d_mpp: how the input is read, then whether a fixed tile variant applies.
  const bool tiny = groups == 1 && p.dW == 1 && Cg <= 8;
  const bool unfolded = tiny && Og >= 64 && 5 * kH * Cg * padded_w <= 3 * Og * outW;
  const bool parity = !direct_columns && !in_nhwc && !tiny && !(groups > 1 && Cg <= 8) && p.sW == 2 && p.dW == 1 &&
      p.padW <= 2 && outW > 32;
  const bool narrow = strided && !tiny && im2col_bytes <= (int64_t(64) << 20) &&
      (outW <= 8 || (outW <= 20 && N * positions <= 1024));
  const bool merged = !unfolded && !narrow && groups == 1 && p.dW == 1 &&
      (tiny || (flat && positions <= 1024) || (p.sW > 1 && !in_nhwc && !parity) ||
       (Cg <= 16 && !parity && 2 * Cg * padded_h * padded_w <= Og * positions));
  const bool im2col = direct_columns || (patch_embed && !parity) || narrow ||
      (im2col_bytes <= (int64_t(64) << 20) &&
       ((groups > 1 && Cg <= 8) || (p.sW > 1 && !merged && !unfolded && !parity && Cg * kH * kW <= 1024)));
  Conv2dPlan plan{.kernel = Conv2dKernel::Mpp, .flat = flat, .parity = parity, .unfold_cheap = unfolded};
  plan.input = pointwise ? Conv2dMppInput::Pointwise
      : direct_columns   ? Conv2dMppInput::Columns
      : im2col           ? Conv2dMppInput::Im2col
      : unfolded         ? Conv2dMppInput::Unfolded
      : merged           ? Conv2dMppInput::Merged
                         : Conv2dMppInput::Direct;

  const bool wide_pointwise = apple10 && N > 1 && Cg >= 160 && Cg <= 960 && Cg % 32 == 0 && Og <= 640 &&
      Og % 160 == 0 && positions >= 4096;
  const bool compact_pointwise = N > 1 && Cg >= 256 && Og >= 128 && Og <= 256 && positions >= 512 &&
      positions <= 1024 && ceil_div(Og, int64_t(128)) * ceil_div(positions, int64_t(32)) * N >= target;
  const bool long_pointwise = apple10 && N > 1 && Cg >= 256 && Og > 32 && Og <= 64 && positions >= 4096;
  const bool batched_pointwise = apple10 && N > 1 && Cg > 16 && Cg <= 128 && Og >= 512 && positions >= 512;
  const bool fast_pointwise = direct_columns ||
      (bfloat_nchw && pointwise && !strided &&
       (wide_pointwise || compact_pointwise || long_pointwise || single_pointwise || batched_pointwise));
  const bool fixed3x3 = plan.input == Conv2dMppInput::Direct && !flat && apple10 && bfloat_nchw && k3 && s1 && d1 &&
      p1 &&
      ((N == 1 && Cg > 32 && Cg <= 64 && Og > 32 && Og <= 64 && positions >= 131072) ||
       (Cg > 16 && Og > 64 && Og <= 96 && outW >= kTile && N * outH * ceil_div(outW, kTile) >= target));
  if (fast_pointwise) {
    plan.variant = Conv2dMppVariant::Pointwise;
    plan.rows = wide_pointwise ? 160 : compact_pointwise ? 128 : 64;
    plan.columns = long_pointwise                                                 ? 128
        : wide_pointwise || compact_pointwise || (direct_columns && !apple10) ? 32
                                                                              : kTile;
    plan.simdgroups = long_pointwise ? 8 : direct_columns && apple10 ? 2 : 4;
  } else if (fixed3x3) {
    plan.variant = Conv2dMppVariant::Fixed3x3;
    plan.rows = Og > 64 ? 96 : 64;
    plan.columns = kTile;
    plan.simdgroups = Og > 64 ? 4 : 2;
  }
  return plan;
}

Conv2dMppTile select_conv2d_mpp_tile(const Conv2dMppParams& mpp,
                                     const Conv2dPlan& plan,
                                     int64_t groups,
                                     bool in_nhwc,
                                     bool out_nhwc,
                                     const Conv2dDevice& dev) {
  using c10::metal::ceil_div;
  constexpr int64_t kTile = 64;
  const auto& p = mpp.conv;
  const int64_t K = mpp.K, Wp = mpp.Wp, Og = p.C_out_per_group, outH = p.outH, outW = p.outW;
  const auto tiles_for = [&](int64_t width) {
    return Wp > 0 ? ceil_div((outH - 1) * Wp + outW, width) : outH * ceil_div(outW, width);
  };
  if (plan.variant != Conv2dMppVariant::Generic) {
    return {plan.rows, plan.columns, plan.simdgroups, tiles_for(plan.columns), K, 1, plan.variant};
  }
  const bool pointwise = plan.input == Conv2dMppInput::Pointwise;
  const bool unfolded = plan.unfold_cheap;
  const float waste64 = 1.0f - float(outW) / float(ceil_div(outW, kTile) * kTile);
  const float waste32 = 1.0f - float(outW) / float(ceil_div(outW, int64_t(32)) * 32);
  const int64_t rows = Og <= 16 ? 16 : Og <= 32 ? 32 : kTile;
  const auto threadgroups_for = [&](int64_t width) { return ceil_div(Og, rows) * groups * tiles_for(width) * p.N; };
  // Narrow the position tile until the grid fills the GPU; 16-row tiles widen to 128 for short K.
  int64_t width = kTile;
  if (rows == kTile) {
    if (!plan.flat && !pointwise && outW <= 32) {
      width = outW <= 16 ? 16 : 32;
    }
    while (width > 16 && threadgroups_for(width) < dev.target) {
      width /= 2;
    }
  } else if (rows == 16 && K <= 64 && (Wp > 0 ? tiles_for(1) : outW) >= 128) {
    width = 128;
  }
  if (!plan.flat && !pointwise && rows >= 32 && width == kTile && outW > 32 && waste64 - waste32 >= 0.14f) {
    width = 32;
  }
  const int64_t simdgroups = width == kTile && (!dev.apple10 || unfolded || (K > 16 && K <= 96)) ? 4 : 2;
  // Split K so the grid reaches `target` threadgroups, keeping >= 128 (16-row) or 256 elements per split.
  const int64_t k_cap = K / (rows == 16 ? 128 : 256);
  const int64_t k_split = c10::metal::round_up(
      ceil_div(K, std::max<int64_t>(std::min<int64_t>(dev.target / threadgroups_for(width), k_cap), 1)), int64_t(32));
  const int64_t splits = ceil_div(K, k_split);
  const bool partials = splits > 1 || (Wp > 0 && Wp != outW);
  const bool staged = unfolded && in_nhwc && !dev.apple10 && !out_nhwc && rows == kTile && width == kTile && !partials;
  const auto variant = staged ? Conv2dMppVariant::Staged : Conv2dMppVariant::Generic;
  return {rows, width, simdgroups, tiles_for(width), k_split, splits, variant};
}

} // namespace at::native::mps
