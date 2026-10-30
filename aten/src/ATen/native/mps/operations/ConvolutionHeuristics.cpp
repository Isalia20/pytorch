#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/operations/ConvolutionHeuristics.h>
#include <c10/metal/common.h>

#include <algorithm>
#include <limits>
#include <optional>

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

const Conv2dTuning& conv2d_tuning(bool apple10_or_newer) {
  static constexpr Conv2dTuning pre_apple10{
      .threadgroups_per_core = 16,
      .winograd_min_channels = 96,
      .winograd_alternative_min_channels = 48,
      .tall_winograd_min_positions = 4096,
      .batched_winograd_min_output_channels = 32,
      .tiny_stride2_max_output_channels = 32,
      .winograd_product_columns = 32,
      // Fewer transform chunks recover the half F2 crossover at C192/O128.
      .half_winograd_workspace_bytes = 2 * Conv2dTuning::workspace_bytes,
      .row_tile_simdgroups = 4,
      .im2col_product_columns = 32,
      .im2col_product_simdgroups = 4,
      // Scalar FMA loses to MPP above these measured 3x3 channel crossovers.
      .single_output_max_3x3_channels = 16,
      .single_output_half_max_3x3_channels = 8,
  };
  static constexpr Conv2dTuning apple10_plus{
      .threadgroups_per_core = 8,
      .winograd_min_channels = 160,
      .winograd_alternative_min_channels = 128,
      .tall_winograd_min_positions = 512,
      .batched_winograd_min_output_channels = 128,
      .tiny_stride2_max_output_channels = 64,
      .winograd_product_columns = 64,
      .half_winograd_workspace_bytes = Conv2dTuning::workspace_bytes,
      .row_tile_simdgroups = 2,
      .im2col_product_columns = 64,
      .im2col_product_simdgroups = 2,
      .single_output_max_3x3_channels = std::numeric_limits<int32_t>::max(),
      .single_output_half_max_3x3_channels = std::numeric_limits<int32_t>::max(),
  };
  return apple10_or_newer ? apple10_plus : pre_apple10;
}

const Conv2dDevice& conv2d_device() {
  static const Conv2dDevice device = [] {
    const bool apple10 = at::mps::is_apple_family_or_newer(at::mps::AppleGPUFamily::APPLE_10_PLUS);
    const bool apple9 = at::mps::is_apple_family_or_newer(at::mps::AppleGPUFamily::APPLE_9_PLUS);
    const int64_t cores =
        std::max<int64_t>(at::mps::MPSDevice::getInstance()->getCoreCount(), Conv2dTuning::min_occupancy_cores);
    const auto& tuning = conv2d_tuning(apple10);
    return Conv2dDevice{.supports_bfloat_simdgroup = apple9,
                        .supports_large_mpp_strides = apple9,
                        .apple10_or_newer = apple10,
                        .cores = cores,
                        .target = tuning.threadgroups_per_core * cores,
                        .tuning = tuning};
  }();
  return device;
}

Conv2dPlan select_conv2d_winograd(const Conv2DParams& p, bool in_nhwc, const Conv2dDevice& dev) {
  using c10::metal::ceil_div;
  using c10::metal::round_up;
  using Wino = Conv2dTuning::Winograd;
  const int64_t Cg = p.C_in_per_group, Og = p.C_out_per_group;
  const int64_t tiles = int64_t(p.N) * ceil_div(int64_t(p.outH), int64_t(2)) * ceil_div(int64_t(p.outW), int64_t(2));
  const bool small = tiles <= Wino::compact_tiles ||
      (tiles <= Wino::small_tiles && int64_t(p.C_in) * p.C_out <= Wino::small_channel_product);
  Conv2dPlan plan{.kernel = Conv2dKernel::WinogradFused, .simdgroups = kConv2dWinogradSimdgroups};
  if (Og <= kConv2dFewOutputTile) {
    plan.rows = kConv2dFewOutputTile;
    plan.columns = kConv2dNarrowTile;
    return plan;
  }
  if (small) {
    plan.rows = plan.columns = kConv2dMinTile;
    return plan;
  }
  // Shallow inputs prefer 16-row tiles when they pad fewer outputs than 32
  // rows.
  const bool padded_rows = Cg <= Wino::padded_rows_max_inputs &&
      round_up(Og, int64_t(kConv2dMinTile)) < round_up(Og, int64_t(kConv2dWinogradPretransformedRows));
  plan.rows = Og <= kConv2dMinTile || padded_rows ? kConv2dMinTile : kConv2dWinogradPretransformedRows;
  // A chunk holds threads / columns input channels. 16-row tiles and less
  // channel padding prefer the wider block, as does NCHW input on Apple10;
  // otherwise channels-last input reads 16 consecutive channels.
  constexpr int64_t threads = kConv2dWinogradSimdgroups * c10::metal::simdgroup_size;
  const auto padding = [&](int64_t columns) { return round_up(Cg, threads / columns) - Cg; };
  const bool wide = plan.rows == kConv2dMinTile || padding(kConv2dNarrowTile) < padding(kConv2dMinTile) ||
      (dev.apple10_or_newer && !in_nhwc);
  plan.columns = wide ? kConv2dNarrowTile : kConv2dMinTile;
  return plan;
}

namespace {

using Wino = Conv2dTuning::Winograd;
using HalfWino = Conv2dTuning::HalfWinograd;
using O1 = Conv2dTuning::SingleOutput;
using Narrow = Conv2dTuning::Narrow;
using Patch = Conv2dTuning::Patch;
using S2 = Conv2dTuning::Stride2;
using Copy = Conv2dTuning::Materialization;
using Pointwise = Conv2dTuning::Pointwise;
using Fixed3 = Conv2dTuning::Fixed3x3;
using c10::metal::ceil_div;
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int64_t kTile = kConv2dRowTile;

// Shape facts of one conv2d call. Each select_* method returns a plan when its
// kernel family runs the call; select_conv2d_kernel asks them in order.
struct Conv2dSelector {
  Conv2dSelector(const Conv2DParams& p,
                 const Tensor& input,
                 const Tensor& output,
                 int64_t groups,
                 const Conv2dDevice& dev)
      : p(p),
        dev(dev),
        tuning(dev.tuning),
        groups(groups),
        N(p.N),
        C_in(p.C_in),
        C_out(p.C_out),
        H(p.H),
        W(p.W),
        outH(p.outH),
        outW(p.outW),
        kH(p.kH),
        kW(p.kW),
        Cg(p.C_in_per_group),
        Og(p.C_out_per_group),
        plane(H * W),
        positions(outH * outW),
        channels(std::min(C_in, C_out)),
        padded_h(H + 2 * int64_t(p.padH)),
        padded_w(W + 2 * int64_t(p.padW)),
        element_size(input.element_size()),
        target(dev.target),
        apple10(dev.apple10_or_newer),
        in_nhwc(!input.is_contiguous()),
        out_nhwc(!output.is_contiguous()),
        fp32(input.scalar_type() == kFloat),
        half(input.scalar_type() == kHalf),
        bf16(input.scalar_type() == kBFloat16),
        bfloat_simd(dev.supports_bfloat_simdgroup && bf16),
        bfloat_nchw(bfloat_simd && groups == 1 && !in_nhwc && !out_nhwc),
        even_offset(input.storage_offset() % 2 == 0),
        even_output_offset(output.storage_offset() % 2 == 0),
        s1(p.sH == 1 && p.sW == 1),
        stride2(p.sH == 2 && p.sW == 2),
        strided(p.sH > 1 || p.sW > 1),
        d1(p.dH == 1 && p.dW == 1),
        p1(p.padH == 1 && p.padW == 1),
        p0(p.padH == 0 && p.padW == 0),
        k3(kH == 3 && kW == 3),
        pointwise(kH == 1 && kW == 1 && p0),
        patch_embed(kH == p.sH && kW == p.sW && p0 && d1),
        packed3(apple10 && k3 && d1 && p1) {}

  const Conv2DParams& p;
  const Conv2dDevice& dev;
  const Conv2dTuning& tuning;
  const int64_t groups, N, C_in, C_out, H, W, outH, outW, kH, kW, Cg, Og;
  const int64_t plane, positions, channels, padded_h, padded_w, element_size, target;
  const bool apple10, in_nhwc, out_nhwc, fp32, half, bf16, bfloat_simd, bfloat_nchw;
  // Packed two-element loads and stores.
  const bool even_offset, even_output_offset;
  const bool s1, stride2, strided, d1, p1, p0, k3, pointwise, patch_embed, packed3;

  bool heavy_pointwise() const {
    return C_out * C_in * positions >= Pointwise::min_multiply_adds;
  }

  // Host batch/per-image/weight checks bound the factors, but not their product.
  int64_t im2col_column_bytes() const {
    return C_in * kH * kW * element_size;
  }
  bool im2col_fits_workspace() const {
    return N * positions <= Conv2dTuning::workspace_bytes / im2col_column_bytes();
  }

  // Strided bfloat16 2x2/3x3 products over im2col columns.
  bool direct_columns() const {
    const bool wide = Og >= Copy::columns_min_outputs;
    const bool batched =
        wide && N > 1 && Cg >= Copy::columns_min_batched_inputs && positions <= Copy::columns_max_batched_positions;
    const bool single = wide && N == 1 && Cg >= Copy::columns_min_inputs &&
        positions >= Copy::columns_positions.first &&
        (p.padH > 0 || p.padW > 0 || positions >= Copy::columns_min_unpadded_positions);
    // Inputs deeper than conv2d_s2 takes, on small maps with outputs that do
    // not shrink the channels.
    const bool deep = Cg > S2::inputs.last && Og >= Cg && N * positions >= Copy::columns_positions.first &&
        N * positions < Copy::columns_positions.last;
    const bool expanding = wide && N == 1 && Cg >= Copy::columns_min_inputs &&
        Og >= Copy::columns_min_channel_expansion * Cg && Copy::columns_positions.contains(positions);
    return bfloat_nchw && stride2 && d1 && kH == kW && (kH == 2 || kH == 3) &&
        N * positions <= Copy::columns_workspace_bytes / im2col_column_bytes() &&
        (apple10 ? batched || single || deep : expanding);
  }

  // Fraction of a position tile of this width past the end of an output row.
  float row_waste(int64_t width) const {
    return 1.0f - float(outW) / float(ceil_div(outW, width) * width);
  }

  // Single bfloat16 images deep enough for 64-row fixed 3x3 MPP tiles, which
  // beat both F(4x2) and generic MPP tiles there.
  bool deep_fixed3() const {
    return apple10 && bfloat_nchw && k3 && s1 && d1 && p1 && N == 1 && Fixed3::deep_inputs.contains(Cg) &&
        Fixed3::deep_outputs.contains(Og) && positions >= Fixed3::min_deep_positions;
  }

  // Flattened rows trade a padded copy for full position tiles.
  bool flattens_rows() const {
    const float waste = row_waste(kTile);
    return !strided && waste > Conv2dTuning::flatten_channel_penalty / float(Og) + Conv2dTuning::flatten_copy_penalty &&
        waste * float(Cg * kH * kW) > Conv2dTuning::flatten_min_saved_reduction;
  }

  // conv2d_tensor: half/bfloat16 3x3 and 5x5 products with channels-last
  // output. The op's output-channel tile must not exceed the output channels;
  // dense 3x3 products take partial tiles, other shapes only whole ones, which
  // they would otherwise waste. Pre-Apple10 GPUs run the op slower than the
  // direct kernels except for channels-last stems.
  std::optional<Conv2dPlan> select_tensor() const {
    const bool copied = !in_nhwc && N * C_in * plane >= Conv2dTuning::tensor_max_copied_inputs;
    const bool k5 = kH == 5 && kW == 5;
    const bool dense3 = k3 && groups == 1;
    const bool stem = in_nhwc && dense3 && Cg <= kConv2dTensorStemChannels;
    const bool tiles = dense3 ? Og >= kConv2dTensorOutputs
                              : Og % kConv2dTensorOutputs == 0 && Cg > kConv2dTensorStemChannels;
    if (!(apple10 || stem) || fp32 || !(k3 || k5) || !s1 || !d1 || !out_nhwc || !tiles || copied) {
      return std::nullopt;
    }
    // Whole 64-channel tiles read each input tile once per 64 outputs while they still fill the GPU.
    const int64_t wide_side = kConv2dTensorWideSide;
    const bool wide = dense3 && Cg > kConv2dTensorStemChannels && Og % kConv2dTensorWideOutputs == 0 &&
        Og / kConv2dTensorWideOutputs * ceil_div(outH, wide_side) * ceil_div(outW, wide_side) * N >= target;
    return Conv2dPlan{.kernel = Conv2dKernel::Tensor,
                      .rows = wide ? kConv2dTensorWideOutputs : kConv2dTensorOutputs,
                      .simdgroups = wide ? kConv2dTensorWideSimdgroups : kConv2dTensorSimdgroups};
  }

  // conv2d_tiny: direct stems with a few input channels.
  std::optional<Conv2dPlan> select_tiny() const {
    const bool bounded_tiny_type = (fp32 && (Cg == 2 || Cg == 3) && kH == 4 && kW == 4) || (half && Cg == 2 && k3);
    if (bounded_tiny_type && groups == 1 && s1 && d1 && p.padH <= kH / 2 && p.padW <= kW / 2 &&
        Og % kConv2dTinyChannelsPerThread == 0 && Og <= Conv2dTuning::bounded_tiny_max_output_channels) {
      return Conv2dPlan{.kernel = Conv2dKernel::Tiny,
                        .simdgroups = kConv2dFlatThreads / c10::metal::simdgroup_size,
                        .bounded_tiny = true};
    }

    // Each thread loads whole rows of kConv2dTinyPositionsPerThread outputs.
    constexpr int64_t thread_positions = kConv2dTinyPositionsPerThread;
    const bool tiny_s2 =
        Cg == 3 && stride2 && W % (2 * thread_positions) == 0 && Og <= tuning.tiny_stride2_max_output_channels;
    // Apple10 float32 stems up to six channels prefer the direct tiny kernel.
    const bool float_stem = apple10 && fp32 && Conv2dTuning::tiny_float_inputs.contains(Cg);
    const bool tiny_3x3 = k3 && p1 && d1 && W % thread_positions == 0 &&
        ((!strided &&
          (Cg == 1 || (Cg == 3 && (!apple10 || fp32 || Og <= Conv2dTuning::tiny_stride1_max_output_channels)) ||
           float_stem)) ||
         tiny_s2);
    const bool tiny_1x1 = pointwise && !strided && Cg == 4 && plane % thread_positions == 0;
    const bool tiny_dtype = bfloat_simd || (k3 && s1 && (half || fp32));
    const bool tiny_nchw = tiny_dtype && groups == 1 && !in_nhwc && !out_nhwc;
    if (tiny_nchw && Og % kConv2dTinyChannelsPerThread == 0 && even_offset && even_output_offset &&
        (tiny_1x1 || tiny_3x3)) {
      return Conv2dPlan{.kernel = Conv2dKernel::Tiny, .simdgroups = kConv2dFlatThreads / c10::metal::simdgroup_size};
    }
    return std::nullopt;
  }

  std::optional<Conv2dPlan> select_winograd() const {
    // conv2d_winograd fuses both F(2x2, 3x3) transforms with the 16 coefficient
    // GEMMs; separate transforms around MPP GEMMs (F(2x2), or F(4x2) over tall
    // maps) amortize better over deep channels.
    const bool winograd_shape = groups == 1 && k3 && s1 && d1 && p.padH == p.padW && p.padH <= 1;
    const bool wino2x2 = channels >= tuning.winograd_min_channels ||
        (apple10 && channels >= tuning.winograd_alternative_min_channels && N * plane <= Wino::compact_positions);
    const int64_t tall_large_positions =
        C_in > Wino::tall_channels ? Wino::tall_large_deep_min_positions : Wino::tall_large_min_positions;
    const bool single_tall = N == 1 &&
        ((C_out == Wino::tall_channels && plane >= tuning.tall_winograd_min_positions &&
          (plane <= Wino::tall_compact_positions ||
           (apple10 ? C_in >= Wino::tall_min_deep_inputs : C_in <= Wino::tall_max_inputs))) ||
         (!apple10 && C_in <= Wino::tall_max_inputs && C_out == Wino::tall_wide_outputs &&
          plane >= Wino::tall_min_wide_positions) ||
         (apple10 && Wino::tall_large_outputs.contains(C_out) && plane >= tall_large_positions &&
          !(deep_fixed3() && C_out <= Fixed3::max_tall_outputs)));
    const bool wino4x2 = bfloat_nchw && C_in >= Wino::tall_channels &&
        (single_tall ||
         (N >= Wino::tall_min_batch && C_out >= tuning.batched_winograd_min_output_channels &&
          C_out <= Wino::tall_channels && std::max(H, W) <= Wino::tall_max_side));
    // Low-precision direct products are cheap enough that a mostly empty
    // input-channel chunk loses to them, unless one chunk holds all channels.
    constexpr int64_t winograd_chunk = kConv2dWinogradSimdgroups * c10::metal::simdgroup_size / kConv2dNarrowTile;
    const int64_t chunk_padding = c10::metal::round_up(Cg, winograd_chunk) - Cg;
    const bool full_chunks = fp32 || Cg <= winograd_chunk || chunk_padding * Wino::low_precision_padding_ratio <= Cg;
    const auto fused = select_conv2d_winograd(p, in_nhwc, dev);
    const int64_t fused_tiles = ceil_div(outH, int64_t(2)) * ceil_div(outW, int64_t(2));
    const bool starved = N * ceil_div(Og, fused.rows) * ceil_div(fused_tiles, fused.columns) < dev.cores &&
        Cg > Wino::starved_max_inputs;
    // NCHW single outputs keep the direct O1 kernels.
    const bool fused_winograd = winograd_shape && Cg >= Wino::fused_min_inputs && full_chunks && !starved &&
        (!bf16 || bfloat_simd) && (Og > 1 || in_nhwc || out_nhwc);
    if (apple10) {
      // Apple10 runs low-precision direct products on the matrix hardware, but
      // not float32 ones: fused F(2x2) wins up to the separate-transform crossover.
      if (fused_winograd && fp32 && channels < Wino::float_separate_min_channels) {
        auto plan = fused;
        // NCHW inputs that the 32-column tile splits into two chunks fit one
        // chunk of conv2d_winograd_loop, which transforms each input tile once
        // for all 32-output blocks, while its per-tile threadgroups fill the GPU.
        constexpr int64_t wide_chunk = kConv2dWinogradSimdgroups * c10::metal::simdgroup_size / kConv2dNarrowTile;
        const int64_t loop_tiles = ceil_div(outH, int64_t(2)) * ceil_div(outW, int64_t(2));
        plan.output_loop = !in_nhwc && !out_nhwc && plan.rows == kConv2dWinogradPretransformedRows &&
            Cg > wide_chunk && Cg <= kConv2dWinogradLoopChannels && Og > plan.rows &&
            N * ceil_div(loop_tiles, int64_t(kConv2dMinTile)) >= target;
        if (plan.output_loop) {
          plan.columns = kConv2dMinTile;
        }
        return plan;
      }
    } else if (fused_winograd && !wino4x2) {
      // Separate transforms cost O(C_in + C_out) per tile against the
      // O(C_in * C_out) product they share; for half they win once the harmonic
      // mean of the channel counts is large enough, for bfloat16 once both
      // channel counts are.
      const bool nchw = !in_nhwc && !out_nhwc;
      const bool separate = nchw &&
          (half ? N * positions >= HalfWino::min_positions &&
                   2 * Cg * Og > HalfWino::separate_min_harmonic_channels * (Cg + Og)
                : bf16 && channels >= Wino::bfloat_separate_min_channels);
      // Few low-precision outputs prefer 8- or 16-row MPP products unless the
      // maps are large enough to amortize the transforms.
      const bool few_outputs = !fp32 && Og <= kConv2dMinTile &&
          (Og <= kConv2dFewOutputTile ? plane < Wino::few_output_min_positions : N * plane < Wino::mid_min_positions);
      // Low-precision stems with wide outputs read im2col columns instead.
      const bool wide_stem = !fp32 && Cg <= Copy::tiny_max_inputs && Og >= Wino::stem_min_outputs;
      if (separate) {
        return Conv2dPlan{Conv2dKernel::Winograd2x2};
      }
      if (!few_outputs && !wide_stem) {
        return fused;
      }
    }
    // On Apple10, channels-last MPP folds the kW taps into contiguous channel
    // runs, and the direct half product keeps up with separate transforms;
    // both need deeper channels before the transforms pay off.
    const bool low_precision = !fp32;
    const int64_t winograd_channels = !apple10 || !low_precision ? 0
        : in_nhwc || out_nhwc                                    ? Wino::nhwc_min_channels
        : half                                                   ? Wino::half_min_channels
                                                                 : 0;
    if (groups == 1 && k3 && s1 && d1 && (!half || apple10) && channels >= winograd_channels && (wino2x2 || wino4x2)) {
      return Conv2dPlan{wino4x2 ? Conv2dKernel::Winograd4x2 : Conv2dKernel::Winograd2x2};
    }
    return std::nullopt;
  }

  // conv2d_o1: one output channel per group.
  std::optional<Conv2dPlan> select_single_output() const {
    // Scalar stride-2 outputs need at least one full O1 threadgroup per core.
    const auto o1_block = O1::simdgroups * c10::metal::simdgroup_size;
    if (apple10 && fp32 && groups == 1 && Og == 1 && !in_nhwc && !out_nhwc && k3 && stride2 && d1 &&
        O1::stride2_inputs.contains(Cg) && N * positions >= std::max(O1::min_threads, dev.cores * o1_block)) {
      return Conv2dPlan{.kernel = Conv2dKernel::SingleOutput,
                        .rows = 1,
                        .columns = 1,
                        .simdgroups = O1::simdgroups,
                        .fixed_taps = true};
    }

    // A paired thread computes a 2x4 output patch over all channels; its rows
    // share the loaded input rows. Short reductions (and float) fill the GPU
    // with these threads; longer ones split channels across SIMD lanes below.
    const int64_t paired_threads = N * C_out * ceil_div(outH, int64_t(2)) * ceil_div(outW, int64_t(4));
    const bool paired_fills =
        paired_threads >= dev.cores * O1::paired_threads_per_core && (Cg <= O1::short_max_inputs || fp32);
    if (Og == 1 && !in_nhwc && !out_nhwc && k3 && s1 && d1 && (bfloat_simd || !bf16) &&
        (paired_fills || paired_threads >= dev.cores * O1::paired_saturated_threads_per_core)) {
      return Conv2dPlan{.kernel = Conv2dKernel::SingleOutput,
                        .rows = 2,
                        .columns = 4,
                        .simdgroups = O1::simdgroups,
                        .fixed_taps = true};
    }

    constexpr int64_t channel_partitions = kConv2dO1ChannelPartitions;
    constexpr int64_t cooperative_columns = 4;
    constexpr int64_t spatial_lanes = c10::metal::simdgroup_size / channel_partitions;
    const int64_t cooperative_items = ceil_div(outW, cooperative_columns) * outH;
    const int64_t cooperative_groups = ceil_div(cooperative_items, spatial_lanes * O1::simdgroups) * N * C_out;
    // Split the reduction only when all channel residues and enough spatial
    // work are available; smaller planes use the per-thread kernels below.
    if ((bfloat_simd || !bf16) && Og == 1 && !in_nhwc && !out_nhwc && k3 && s1 && d1 && Cg >= channel_partitions &&
        cooperative_items * N * C_out >= O1::min_threads && cooperative_groups >= dev.cores &&
        ceil_div(cooperative_items, spatial_lanes) <=
            std::numeric_limits<uint32_t>::max() / c10::metal::simdgroup_size) {
      return Conv2dPlan{.kernel = Conv2dKernel::SingleOutput,
                        .rows = 1,
                        .columns = cooperative_columns,
                        .simdgroups = O1::simdgroups,
                        .channel_partitions = channel_partitions,
                        .fixed_taps = true};
    }

    const int64_t o1_threads = ceil_div(outW, int64_t(8)) * outH, o1_work = o1_threads * N * C_out;
    const auto o1_channels = half ? tuning.single_output_half_max_3x3_channels : tuning.single_output_max_3x3_channels;
    const bool o1_reduction = !(k3 && s1 && d1) || Cg <= o1_channels;
    if (Og == 1 && !in_nhwc && !out_nhwc && o1_reduction && o1_work >= O1::min_threads) {
      const bool wide = bfloat_simd && p.sW == 1 && p.dW == 1 &&
          ((kW == 1 && Cg <= O1::wide_max_inputs && outW % O1::wide_columns == 0) ||
           (kW == 3 && Cg <= O1::short_max_inputs && o1_work >= O1::wide_min_threads));
      const bool paired = bfloat_simd && k3 && s1 && d1 && Cg <= O1::paired_max_inputs &&
          o1_work >= O1::paired_min_threads &&
          (apple10 || (Cg <= O1::short_max_inputs && o1_threads >= O1::min_threads));
      const int64_t columns = paired ? 4 : wide ? O1::wide_columns : 8;
      return Conv2dPlan{.kernel = Conv2dKernel::SingleOutput,
                        .rows = paired ? 2 : 1,
                        .columns = columns,
                        .simdgroups = O1::simdgroups,
                        .fixed_taps = (kW == 3 || wide) && p.sW == 1 && p.dW == 1};
    }
    return std::nullopt;
  }

  // Staged direct tiles: half patches, bfloat16 pointwise planes, narrow
  // collective tiles and bfloat16 patches.
  std::optional<Conv2dPlan> select_staged() const {
    // Staging must save a collective product versus the direct tap loop to
    // pay for its explicit loads/barriers. These are performance limits. Up to
    // 64 outputs the patch tiles pad more than 64-row MPP tiles, which only
    // pays off when one channel chunk holds the whole reduction.
    const bool half_patch_rows = Og > kTile || Cg <= Patch::half_channel_chunk;
    if (apple10 && half && groups == 1 && !in_nhwc && !out_nhwc && k3 && s1 && p1 && d1 && W >= kConv2dNarrowTile &&
        Og >= Patch::half_min_outputs && half_patch_rows && ceil_div(Cg, Patch::half_channel_chunk) < kH * kW) {
      int64_t rows = Patch::half_rows[0];
      for (const auto candidate : Patch::half_rows) {
        // On equal padding, fewer output tiles reuse the staged input more.
        if (c10::metal::round_up(Og, candidate) <= c10::metal::round_up(Og, rows)) {
          rows = candidate;
        }
      }
      if (Cg <= Patch::half_compact_channels &&
          c10::metal::round_up(Og, int64_t(kConv2dWideTile)) == c10::metal::round_up(Og, rows)) {
        rows = kConv2dWideTile;
      }
      // The flat loader wraps at most one row within its 32-position tile.
      if (N * ceil_div(Og, rows) * ceil_div(positions, int64_t(kConv2dNarrowTile)) >= target) {
        return Conv2dPlan{
            .kernel = Conv2dKernel::Patch, .rows = rows, .columns = kConv2dNarrowTile, .simdgroups = 4, .flat = true};
      }
    }

    const bool fixed_plane = conv2d_pointwise_fixed_channels(Cg) && Pointwise::large_outputs.contains(Og);
    const bool dynamic_plane = Cg >= Pointwise::large_tile_min_inputs && Cg % kConv2dNarrowTile == 0 &&
        Og >= Pointwise::large_outputs.first && positions >= Pointwise::plane_positions_per_output_channel * Og &&
        c10::metal::round_up(Og, int64_t(kConv2dNarrowTile)) < c10::metal::round_up(Og, int64_t(128));
    if (pointwise && !strided && apple10 && bfloat_nchw && N == 1 && (fixed_plane || dynamic_plane) &&
        heavy_pointwise() && plane <= kInt32Max / std::max(Cg, Og)) {
      return Conv2dPlan{.kernel = Conv2dKernel::PointwisePlane, .rows = 64, .columns = 128, .simdgroups = 4};
    }
    const bool narrow_batch = N >= Narrow::min_batch && (Cg == 64 || Cg == 128) && Narrow::batch_width.contains(W) &&
        Narrow::batch_positions.contains(positions);
    const bool narrow_large =
        N == 1 && Cg == 176 && Narrow::large_width.contains(W) && Narrow::large_positions.contains(positions);
    const int64_t narrow_columns = Cg == 64 ? 64 : 128;
    if (packed3 && !strided && bfloat_nchw && (narrow_batch || narrow_large) && Narrow::outputs.contains(Og) &&
        N * ceil_div(positions, narrow_columns) >= target) {
      return Conv2dPlan{.kernel = Conv2dKernel::Narrow, .columns = narrow_columns, .simdgroups = Cg == 128 ? 8 : 4};
    }

    const bool sample_s2 = pointwise && stride2 && Patch::sample_inputs.contains(Cg) && outW >= S2::min_width &&
        ceil_div(Og, int64_t(128)) * outH * ceil_div(outW, int64_t(32)) * N >= target;
    constexpr int64_t patch2_rows = 192; // Prefer full output-channel tiles.
    const bool patch2 = patch_embed && kH == 2 && kW == 2 && N > 1 && Patch::embedding_inputs.contains(Cg) &&
        Og % patch2_rows == 0 && H % 2 == 0;
    const bool flat_rows = packed3 && !strided && Patch::flat_inputs.contains(Cg) && W >= Patch::flat_min_width &&
        W % 2 == 0;
    const bool patch_flat = flat_rows && Patch::flat_outputs.contains(Og) && Patch::flat_positions.contains(positions);
    const bool patch_flat64 = flat_rows && N == 1 && Og > kConv2dNarrowTile && Og <= kTile &&
        Cg <= Patch::flat_row_max_inputs &&
        (Cg <= Patch::flat_row_shallow_inputs || C_in * plane < Patch::flat_row_max_elements);
    // Only the 16-channel product needs an even offset; the flat loader reads
    // odd offsets with two-byte aligned loads.
    const bool patch_s1 = packed3 && !strided &&
        ((Cg == 16 && Og >= Patch::stride1_min_outputs && W % 8 == 0 && even_offset) || patch_flat ||
         patch_flat64) &&
        C_in * plane <= kInt32Max && C_out * positions <= kInt32Max;
    const bool patch_s2 = packed3 && stride2 &&
        ((Patch::stride2_small_inputs.contains(Cg) && Patch::stride2_small_outputs.contains(Og)) ||
         (Cg >= Patch::stride2_min_large_inputs && Patch::large_outputs.contains(Og) &&
          Patch::stride2_large_width.contains(outW) && N * outH >= target));
    const bool packed_patch = (W % 2 == 0 || (sample_s2 && H % 2 == 0)) && even_offset;
    if (bfloat_nchw && (patch_s1 || patch_s2 || (packed_patch && (sample_s2 || patch2)))) {
      Conv2dPlan plan{.kernel = Conv2dKernel::Patch, .simdgroups = 4, .flat = patch_flat || patch_flat64};
      if (patch_flat || patch_flat64) {
        plan.rows = patch_flat ? 128 : 64;
        plan.columns = 32;
      } else if (patch_s1) {
        plan.rows = 64;
        plan.columns = 64;
      } else if (patch_s2) {
        plan.rows = Cg <= Patch::stride2_small_inputs.last ? 32 : 256;
        plan.columns = Cg <= Patch::stride2_small_inputs.last ? 64 : 16;
      } else if (sample_s2) {
        plan.rows = 128;
        plan.columns = 32;
      } else {
        plan.rows = patch2_rows;
        plan.columns = 16;
      }
      return plan;
    }
    return std::nullopt;
  }

  // Strided products that bypass the generic MPP input selection.
  std::optional<Conv2dPlan> select_strided() const {
    // 5x5 stride-2 products: folding the kW taps into a padded NHWC copy beats
    // the stride-2 kernel, parity rows and im2col columns, except for Apple10
    // low-precision products.
    if ((fp32 || !apple10) && groups == 1 && stride2 && kH == 5 && kW == 5 && d1 && Cg <= Copy::merged_5x5_max_inputs) {
      return Conv2dPlan{.kernel = Conv2dKernel::Mpp, .input = Conv2dMppInput::Merged};
    }

    // Pre-Apple10 CL output also needs a temporary NCHW product.
    const int64_t float_column_bytes = im2col_column_bytes() + (!apple10 && out_nhwc ? C_out * element_size : 0);
    if (fp32 && groups == 1 && k3 && stride2 && d1 && Cg > Copy::tiny_max_inputs && Og >= kConv2dRowTile &&
        N * positions <= Conv2dTuning::workspace_bytes / float_column_bytes &&
        (apple10 || out_nhwc || Og >= Copy::float_columns_min_expansion * Cg)) {
      const bool narrow_rows =
          !out_nhwc && c10::metal::round_up(Og, int64_t(kConv2dNarrowTile)) < c10::metal::round_up(Og, kTile);
      const int64_t rows = narrow_rows ? kConv2dNarrowTile : kTile;
      const int64_t columns = narrow_rows ? kConv2dNarrowTile : kConv2dMinTile;
      return Conv2dPlan{.kernel = Conv2dKernel::Mpp,
                        .rows = rows,
                        .columns = columns,
                        .simdgroups = 2,
                        .input = Conv2dMppInput::Columns,
                        .variant = Conv2dMppVariant::Pointwise};
    }
    const float waste64 = row_waste(kTile), waste32 = row_waste(kConv2dNarrowTile);
    const bool narrow_better = waste64 - waste32 >= Conv2dTuning::narrow_min_padding_gain;
    // Smallest registered conv2d_s2 row tile holding the group's outputs; the
    // 96-row tile is bfloat16 only.
    constexpr int64_t s2_wide_rows = 96;
    int64_t bm = kConv2dWideTile;
    for (const int64_t rows : {s2_wide_rows, kTile, int64_t(kConv2dNarrowTile), int64_t(kConv2dMinTile)}) {
      if (Og <= rows && (rows != s2_wide_rows || bfloat_simd)) {
        bm = rows;
      }
    }
    const auto s2_threadgroups = [&](int64_t bn) { return ceil_div(Og, bm) * groups * outH * ceil_div(outW, bn) * N; };
    const bool narrow_columns = bm == s2_wide_rows || outW < kTile || narrow_better || s2_threadgroups(kTile) < target;
    const bool wide_columns = Cg <= S2::wide_max_inputs && bm == kConv2dNarrowTile && outW >= kConv2dWideTile &&
        s2_threadgroups(kConv2dWideTile) >= target;
    const int64_t bn = narrow_columns ? kConv2dNarrowTile : wide_columns ? kConv2dWideTile : kTile;
    if (!pointwise && !patch_embed && !in_nhwc && !out_nhwc && p.sW == 2 &&
        (kW - 1) * p.dW <= kConv2dStride2MaxTapSpan && S2::inputs.contains(Cg) && outW >= S2::min_width &&
        std::min(waste64, waste32) < Conv2dTuning::stride2_max_padding_waste &&
        s2_threadgroups(bn) >= Conv2dTuning::stride2_min_threadgroups_per_core * dev.cores) {
      return Conv2dPlan{.kernel = Conv2dKernel::Stride2, .rows = bm, .columns = bn, .simdgroups = bm == 128 ? 8 : 4};
    }
    return std::nullopt;
  }

  // conv2d_mpp: how the input is read, then whether a fixed tile variant applies.
  Conv2dPlan select_mpp() const {
    const bool tiny = groups == 1 && p.dW == 1 && Cg <= Copy::tiny_max_inputs;
    // Stems read im2col columns rather than the unfolded or merged copies,
    // except from NCHW input on Apple10.
    const bool im2col_stem = tiny && (in_nhwc || !apple10);
    const bool unfolded = tiny && !im2col_stem && Og >= Copy::unfolded_min_outputs &&
        Copy::unfolded_copy_cost * kH * Cg * padded_w <= Copy::unfolded_product_cost * Og * outW;
    const bool columns = direct_columns(), flat = flattens_rows();
    // Larger padding can address the other parity half of the copied row.
    const bool parity = !pointwise && !columns && !in_nhwc && Cg > Copy::tiny_max_inputs && p.sW == 2 && p.dW == 1 &&
        p.padW <= 2 && outW >= Copy::parity_min_width;
    // Few input channels read a K = Cg run per tap; when no copy folds the
    // taps (grouped, or dilated along W), im2col makes one long reduction.
    const bool short_taps = Cg <= Copy::tiny_max_inputs && !tiny;
    const bool narrow = strided && !tiny && im2col_fits_workspace() &&
        (outW <= Copy::narrow_max_width ||
         (outW <= Copy::narrow_max_small_width && N * positions <= Copy::narrow_max_positions));
    const bool grouped_merged = groups > 1 && kW >= Copy::grouped_merged_min_width && p.dW == 1 && !parity &&
        Cg > Copy::tiny_max_inputs && Cg <= Copy::grouped_merged_max_inputs &&
        N * C_in * padded_h * padded_w <= Conv2dTuning::workspace_bytes / element_size;
    // A tiny input's padded copy pays off when a direct product would shift
    // for left padding or copy for a W stride anyway.
    const bool tiny_merged = tiny && (p.padW > 0 || p.sW > 1);
    const bool merged = grouped_merged ||
        (!unfolded && !narrow && groups == 1 && p.dW == 1 &&
         (tiny_merged || (flat && positions <= Copy::merged_max_positions) || (p.sW > 1 && !in_nhwc && !parity) ||
          (Cg <= Copy::merged_max_inputs && !parity &&
           Copy::merged_copy_cost * Cg * padded_h <= Og * positions / padded_w)));
    const bool strided_columns =
        p.sW > 1 && !merged && !unfolded && !parity && Cg * kH * kW <= Copy::im2col_max_reduction;
    const bool im2col = columns || (patch_embed && !parity) || narrow ||
        (im2col_fits_workspace() && (short_taps || im2col_stem || strided_columns));
    Conv2dPlan plan{.kernel = Conv2dKernel::Mpp, .flat = flat, .parity = parity, .unfold_cheap = unfolded};
    plan.input = pointwise ? Conv2dMppInput::Pointwise
        : columns          ? Conv2dMppInput::Columns
        : im2col           ? Conv2dMppInput::Im2col
        : unfolded         ? Conv2dMppInput::Unfolded
        : merged           ? Conv2dMppInput::Merged
                           : Conv2dMppInput::Direct;

    // Single-tap tiles have no split-K. On Apple10 they win for batches, for
    // single images up to mid-size maps and reductions, and beyond those once
    // the product is heavy. Before Apple10 only batched compact maps filling
    // 128-row tiles take them.
    constexpr int64_t tall_rows = kConv2dWideTile, tall_columns = kConv2dNarrowTile, widest_rows = 160;
    // 128 rows unless 160 rows save enough padding; mid-size products also
    // keep 64x64 tiles when those pad less.
    const auto padded = c10::metal::round_up(Og, tall_rows), padded_widest = c10::metal::round_up(Og, widest_rows);
    const bool widest = Pointwise::large_tile_padding_cost * padded_widest < padded;
    const bool mid_tall = widest || padded <= c10::metal::round_up(Og, kTile);
    const int64_t mid_rows = widest ? widest_rows : mid_tall ? tall_rows : kTile;
    const int64_t mid_columns = mid_tall ? tall_columns : kTile;
    // Single images with too few single-tap threadgroups for a reduction the
    // generic tiles can split take those split-K tiles instead.
    const bool split_single = N == 1 && Cg >= 2 * Conv2dTuning::min_split_reduction &&
        ceil_div(Og, mid_rows) * ceil_div(positions, mid_columns) * Pointwise::split_inputs_per_threadgroup <= Cg;
    const bool mid_pointwise = apple10 &&
        (N > 1 || (positions <= Pointwise::mid_max_positions && Cg <= Pointwise::mid_max_inputs && !split_single));
    const bool compact_pointwise = !apple10 && N > 1 && Cg >= Pointwise::min_deep_inputs &&
        Pointwise::large_outputs.contains(Og) && Pointwise::compact_positions.contains(positions) &&
        ceil_div(Og, tall_rows) * ceil_div(positions, tall_columns) * N >= target;
    const bool single_pointwise = apple10 && N == 1 && bf16 && heavy_pointwise();
    const bool fast_pointwise =
        columns || (bfloat_nchw && pointwise && !strided && (mid_pointwise || compact_pointwise || single_pointwise));
    const bool large_fixed3 = N == 1 && Fixed3::large_channels.contains(Cg) && Fixed3::large_channels.contains(Og) &&
        positions >= Fixed3::min_large_positions;
    const bool mid_fixed3 = Cg >= Fixed3::min_inputs && Fixed3::mid_outputs.contains(Og) && outW >= kTile &&
        N * outH * ceil_div(outW, kTile) >= target;
    // Shallow single images whose rows fill whole 64-position tiles once
    // output column 0 moves to separate threadgroups (outW % kTile == 1), or
    // leave only the vacated position unused (outW % kTile == 0).
    const bool whole_fixed3 = N == 1 && Cg >= Fixed3::min_inputs && Cg <= kConv2dFixed3ColumnMaxInputs &&
        Og >= kTile && outW % kTile <= 1 && ceil_div(Og, kTile) * outH * ceil_div(outW - 1, kTile) >= target;
    const bool fixed3x3 = plan.input == Conv2dMppInput::Direct && !flat && apple10 && bfloat_nchw && k3 && s1 && d1 &&
        p1 && (large_fixed3 || deep_fixed3() || mid_fixed3 || whole_fixed3);
    if (fast_pointwise) {
      plan.variant = Conv2dMppVariant::Pointwise;
      plan.rows = plan.columns = kTile;
      plan.simdgroups = 4;
      if (columns) {
        plan.columns = tuning.im2col_product_columns;
        plan.simdgroups = tuning.im2col_product_simdgroups;
      } else if (mid_pointwise) {
        plan.rows = mid_rows;
        plan.columns = mid_columns;
      } else if (compact_pointwise) {
        plan.rows = tall_rows;
        plan.columns = tall_columns;
      } else if (Cg >= Pointwise::large_tile_min_inputs && Og >= Pointwise::large_tile_min_outputs) {
        plan.rows = widest ? widest_rows : tall_rows;
        plan.columns = tall_columns;
        plan.simdgroups = widest ? 4 : 2;
      }
    } else if (fixed3x3) {
      plan.variant = Conv2dMppVariant::Fixed3x3;
      // The 96-row tile only pays off when it pads fewer output channels.
      constexpr int64_t wide_rows = 96;
      plan.rows = c10::metal::round_up(Og, wide_rows) < c10::metal::round_up(Og, kTile) ? wide_rows : kTile;
      plan.columns = kTile;
      plan.simdgroups = plan.rows == wide_rows ? 4 : 2;
    }
    return plan;
  }
};

} // namespace

// Keep returned tiles and channel-specialized names in sync with the
// INSTANTIATE_CONV2D_* registrations in Convolution.metal.
Conv2dPlan select_conv2d_kernel(const Conv2DParams& p,
                                const Tensor& input,
                                const Tensor& output,
                                int64_t groups,
                                const Conv2dDevice& dev) {
  const Conv2dSelector selector(p, input, output, groups, dev);
  for (const auto select : {&Conv2dSelector::select_tensor,
                            &Conv2dSelector::select_tiny,
                            &Conv2dSelector::select_winograd,
                            &Conv2dSelector::select_single_output,
                            &Conv2dSelector::select_staged,
                            &Conv2dSelector::select_strided}) {
    if (auto plan = (selector.*select)()) {
      return *plan;
    }
  }
  return selector.select_mpp();
}

Conv2dMppTile select_conv2d_mpp_tile(const Conv2dMppParams& mpp,
                                     const Conv2dPlan& plan,
                                     int64_t groups,
                                     const Conv2dDevice& dev) {
  using c10::metal::ceil_div;
  constexpr int64_t kTile = kConv2dRowTile;
  const auto& tuning = dev.tuning;
  const auto& p = mpp.conv;
  const int64_t K = mpp.K, Wp = mpp.Wp, Og = p.C_out_per_group, outH = p.outH, outW = p.outW;
  const auto tiles_for = [&](int64_t width) {
    return Wp > 0 ? ceil_div((outH - 1) * Wp + outW, width) : outH * ceil_div(outW, width);
  };
  if (plan.variant == Conv2dMppVariant::Fixed3x3 && p.C_in <= kConv2dFixed3ColumnMaxInputs) {
    // Tiles start at column 1 behind the threadgroups that compute column 0.
    const int64_t tiles = ceil_div(outH, int64_t(kConv2dFixed3ColumnRows)) + outH * ceil_div(outW - 1, plan.columns);
    return {plan.rows, plan.columns, plan.simdgroups, tiles, K, 1, plan.variant};
  }
  if (plan.variant != Conv2dMppVariant::Generic) {
    return {plan.rows, plan.columns, plan.simdgroups, tiles_for(plan.columns), K, 1, plan.variant};
  }
  const bool pointwise = plan.input == Conv2dMppInput::Pointwise;
  const bool unfolded = plan.unfold_cheap;
  const float waste64 = 1.0f - float(outW) / float(ceil_div(outW, kTile) * kTile);
  const float waste32 = 1.0f - float(outW) / float(ceil_div(outW, int64_t(kConv2dNarrowTile)) * kConv2dNarrowTile);
  // Avoid padded output channels; retain the largest tile when work is equal.
  int64_t rows = kTile;
  for (const int64_t candidate : {kConv2dNarrowTile, kConv2dMinTile}) {
    if (c10::metal::round_up(Og, candidate) < c10::metal::round_up(Og, rows)) {
      rows = candidate;
    }
  }
  // The 8-row tile only serves few outputs; it loses reuse on larger groups.
  if (Og <= kConv2dFewOutputTile) {
    rows = kConv2dFewOutputTile;
  }
  // On Apple10, grouped convolutions reuse each input tile of their short
  // per-tap reductions over 32 rows, which outweighs padding a few outputs.
  // Merged kW taps make the reduction long enough that only padding counts.
  // Output rows narrower than a 32-position tile keep the narrowed tiles below.
  const bool grouped_rows = dev.apple10_or_newer && groups > 1 && Og > kConv2dMinTile && outW > kConv2dNarrowTile;
  if (grouped_rows) {
    const bool padded_narrow =
        c10::metal::round_up(Og, int64_t(kConv2dMinTile)) < c10::metal::round_up(Og, int64_t(kConv2dNarrowTile));
    rows = plan.input == Conv2dMppInput::Merged && padded_narrow ? kConv2dMinTile : kConv2dNarrowTile;
  }
  const auto threadgroups_for = [&](int64_t width) { return ceil_div(Og, rows) * groups * tiles_for(width) * p.N; };
  // Narrow the position tile until the grid fills the GPU; 16-row tiles widen to 128 for short K.
  int64_t width = kTile;
  if (rows == kTile) {
    if (!plan.flat && !pointwise && outW <= kConv2dNarrowTile) {
      width = outW <= kConv2dMinTile ? kConv2dMinTile : kConv2dNarrowTile;
    }
    while (width > kConv2dMinTile && threadgroups_for(width) < dev.target) {
      width /= 2;
    }
  } else if (rows == kConv2dMinTile && K <= Conv2dTuning::wide_tile_max_reduction &&
             (Wp > 0 ? tiles_for(1) : outW) >= kConv2dWideTile && threadgroups_for(kConv2dWideTile) >= dev.target) {
    width = kConv2dWideTile;
  }
  // A one-row kernel's short product costs about as much as shifting the
  // accumulator of each row's first tile past the left padding; on rows of
  // several 64-row tiles, narrow tiles halve the shifted tile and the share of
  // tiles that shift.
  const bool shifted_rows = plan.input == Conv2dMppInput::Direct && !plan.flat && p.kH == 1 && p.padW > 0 &&
      rows == kTile && outW >= 2 * kTile;
  // Merged inputs are padded NHWC copies, so a few trailing output columns can
  // take tiles running down the rows instead of a mostly empty tile per row.
  const auto edge_fits = [&](int64_t w) {
    return plan.input == Conv2dMppInput::Merged && outW > w && outW % w * Copy::merged_edge_fraction <= w &&
        threadgroups_for(w) >= dev.target;
  };
  if (!edge_fits(width) && !plan.flat && !pointwise && rows >= kConv2dNarrowTile && width == kTile &&
      outW > kConv2dNarrowTile && (shifted_rows || waste64 - waste32 >= Conv2dTuning::narrow_min_padding_gain)) {
    width = kConv2dNarrowTile;
  }
  // The 8-row tile is registered with two simdgroups only. Apple10 also keeps
  // 32-row tiles on two simdgroups; four would leave each a 32x16 product.
  const int64_t two_simdgroup_rows = dev.apple10_or_newer ? kConv2dNarrowTile : kConv2dMinTile;
  const bool two_simdgroups =
      width != kTile || rows == kConv2dFewOutputTile || grouped_rows || (rows <= two_simdgroup_rows && !unfolded);
  // On Apple10, direct products of a few taps also prefer four groups.
  const bool few_taps = dev.apple10_or_newer && (pointwise || plan.input == Conv2dMppInput::Direct) &&
      int64_t(p.kH) * p.kW <= Conv2dTuning::four_simdgroup_max_taps;
  const bool four_simdgroups = unfolded || few_taps || Conv2dTuning::four_simdgroup_reduction.contains(K);
  const int64_t simdgroups = two_simdgroups ? 2 : four_simdgroups ? 4 : tuning.row_tile_simdgroups;
  // Split K so the grid reaches `target` threadgroups, keeping >= 128 (16-row) or 256 elements per split.
  const int64_t k_cap =
      K / (rows <= kConv2dMinTile ? Conv2dTuning::small_tile_min_split_reduction : Conv2dTuning::min_split_reduction);
  const int64_t k_split = c10::metal::round_up(
      ceil_div(K, std::max<int64_t>(std::min<int64_t>(dev.target / threadgroups_for(width), k_cap), 1)),
      Conv2dTuning::split_reduction_alignment);
  const int64_t splits = ceil_div(K, k_split);
  const int64_t edge = splits == 1 && edge_fits(width) ? outW % width : 0;
  const int64_t tiles = edge ? outH * (outW / width) + edge * ceil_div(outH, width) : tiles_for(width);
  return {rows, width, simdgroups, tiles, k_split, splits, Conv2dMppVariant::Generic, edge};
}

} // namespace at::native::mps
