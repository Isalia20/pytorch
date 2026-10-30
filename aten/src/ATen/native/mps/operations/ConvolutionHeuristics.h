#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/kernels/Convolution.h>

#include <array>
#include <cstdint>
#include <string>

namespace at::native::mps {

struct Conv3dMppTile {
  int output_channels;
  int output_width;
  int output_height;
  int simdgroups;

  float cost(const Conv3DParams& params, int64_t groups, int64_t cores) const;
};

struct Conv3dSimdTile {
  int block_rows;
  int block_columns;
  int simdgroups_rows;
  int simdgroups_columns;

  MTLComputePipelineState_t pipeline_state(
      MetalShaderLibrary& library,
      const std::string& dtype,
      bool use_long_index) const;
};

Conv3dMppTile select_conv3d_mpp_tile(
    const Conv3DParams& params,
    int64_t groups);
Conv3dSimdTile select_conv3d_simd_tile(
    int64_t output_height,
    int64_t output_width,
    bool use_long_index);
bool conv3d_prefer_im2col(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    const Tensor& output);

// Native conv2d plans available with MPP (macOS 26.2+).
enum class Conv2dKernel {
  // winograd_* F(2x2,3x3) transforms around a 16-group pointwise conv2d_mpp.
  Winograd2x2,
  // winograd_*_4 F(4x2,3x3) transforms around 24 conv2d_mpp pw2 GEMMs.
  Winograd4x2,
  // conv2d_tiny: small-channel input (stem convs); each thread
  // computes eight spatial positions for four output channels.
  Tiny,
  // conv2d_o1: one output channel per group.
  SingleOutput,
  // conv2d_winograd: F(2x2,3x3) with both transforms and the 16 coefficient
  // GEMMs in one kernel; select_conv2d_winograd picks the tile.
  WinogradFused,
  // conv2d_pointwise: one-image 1x1 with complete 32-channel input blocks;
  // each simdgroup computes a 32x64 output tile.
  PointwisePlane,
  // conv2d_narrow: 3x3 on 64/128/176 channels, with input fragments held
  // in cooperative registers.
  Narrow,
  // conv2d_patch: sampled 1x1 or 2x2/3x3 patches staged per threadgroup.
  Patch,
  // conv2d_s2: stride-2 rows read directly from NCHW, <= 128 input channels.
  Stride2,
  // conv2d_mpp: GEMM over the input prepared per Conv2dMppInput, split-K when
  // the grid is small.
  Mpp,
  // conv2d_tensor: 3x3 stride-1 half/bfloat16 products through the MPP
  // convolution2d op, from NHWC input into NHWC output.
  Tensor,
};

// Input preparation for conv2d_mpp.
enum class Conv2dMppInput {
  // Original activation, with optional padding, parity split or NHWC copy.
  Direct,
  // 1x1: GEMM over [C_in, outH*outW] of the stride-sampled input.
  Pointwise,
  // conv2d_im2col columns, tiled over the whole image (Wp pitch).
  Im2col,
  // Im2col flattened to one row of outH*outW positions.
  Columns,
  // kH tap rows unfolded into the channels of a padded NHWC copy; the kW taps
  // are read along W.
  Unfolded,
  // Padded NHWC copy, group-major for grouped convolutions, with the kW taps
  // folded into K; the kH rows stay taps.
  Merged,
};

enum class Conv2dMppVariant {
  Generic,
  // k3_: 3x3 stride-1 taps unrolled, bias seeded into the accumulator.
  Fixed3x3,
  // pw1_: single tap.
  Pointwise,
};

// Family preferences, separate from shader capabilities. The >=Apple10
// profile also applies to future Apple GPU families; core count scales
// occupancy.
struct Conv2dTuning {
  int64_t threadgroups_per_core;
  int64_t winograd_min_channels;
  int64_t winograd_alternative_min_channels;
  int64_t tall_winograd_min_positions;
  int64_t batched_winograd_min_output_channels;
  int64_t tiny_stride2_max_output_channels;
  int64_t winograd_product_columns;
  int64_t half_winograd_workspace_bytes;
  int64_t row_tile_simdgroups;
  int64_t im2col_product_columns;
  int64_t im2col_product_simdgroups;
  int64_t single_output_max_3x3_channels;
  int64_t single_output_half_max_3x3_channels;

  // These bound temporary materialization, not the supported tensor size.
  static constexpr int64_t workspace_bytes = int64_t(64) << 20;
  // Keep the occupancy estimate conservative on small or unreported devices.
  static constexpr int64_t min_occupancy_cores = 8;

  // Inclusive empirical windows; exact channel specializations, packed-load
  // alignment and registered tile dimensions remain in the selector.
  struct Range {
    int64_t first, last;
    constexpr bool contains(int64_t value) const {
      return first <= value && value <= last;
    }
  };

  // Separate transforms lose on small channel counts. Fused products avoid
  // that traffic before Apple10; tall products reuse it over larger maps.
  struct Winograd {
    // Apple10 separate F(2x2) also covers compact maps from the alternative
    // channel count.
    static constexpr int64_t compact_positions = 16384;
    // F(4x2) over tall single images or batches of small maps.
    static constexpr int64_t tall_channels = 128;
    static constexpr int64_t tall_max_inputs = 192;
    static constexpr int64_t tall_wide_outputs = 64;
    static constexpr int64_t tall_compact_positions = 32768;
    static constexpr int64_t tall_min_deep_inputs = 256;
    static constexpr Range tall_large_outputs{129, 256};
    static constexpr int64_t tall_large_min_positions = 32768;
    // Inputs deeper than tall_channels amortize the transforms from smaller
    // maps.
    static constexpr int64_t tall_large_deep_min_positions = 16384;
    static constexpr int64_t tall_min_wide_positions = 16384;
    static constexpr int64_t tall_min_batch = 8;
    static constexpr int64_t tall_max_side = 80;
    // Before Apple10, few low-precision outputs need large maps to amortize
    // the fused transforms over the 8-row (Og <= 8) or 16-row MPP products.
    static constexpr int64_t few_output_min_positions = int64_t(512) * 512;
    static constexpr int64_t mid_min_positions = 65536;
    // Three-channel stems keep the tiny direct kernels.
    static constexpr int64_t fused_min_inputs = 4;
    // Each fused threadgroup reduces all input channels; deeper reductions
    // need at least one threadgroup per core.
    static constexpr int64_t starved_max_inputs = 256;
    // Deeper inputs repeat too many input transforms over 16-row tiles to
    // trade them for less output padding.
    static constexpr int64_t padded_rows_max_inputs = 64;
    // Before Apple10, low-precision stems with this many outputs take im2col
    // columns instead.
    static constexpr int64_t stem_min_outputs = 256;
    // Low-precision fused routes pad at most 1 / ratio of the input channels.
    static constexpr int64_t low_precision_padding_ratio = 4;
    // Crossovers to separate F(2x2): bfloat16 before Apple10, half/bfloat16
    // channels-last and half NCHW on Apple10, float32 on Apple10.
    static constexpr int64_t bfloat_separate_min_channels = 128;
    static constexpr int64_t nhwc_min_channels = 320;
    static constexpr int64_t half_min_channels = 256;
    static constexpr int64_t float_separate_min_channels = 256;
    // conv2d_winograd: 16-row tiles transform weights per chunk in a single
    // dispatch; they win while the 32-row grid would be small.
    static constexpr int64_t compact_tiles = 64;
    static constexpr int64_t small_tiles = 256;
    static constexpr int64_t small_channel_product = int64_t(128) * 128;
  };

  // Before Apple10, separate half F(2x2) transforms win once the harmonic mean
  // of the channel counts is large enough.
  struct HalfWinograd {
    static constexpr int64_t separate_min_harmonic_channels = 160;
    static constexpr int64_t min_positions = 4096;
  };

  // Direct output vectors avoid padded GEMMs for O/group=1; enough work and
  // short reductions are needed to amortize the wider/paired variants.
  struct SingleOutput {
    static constexpr int64_t min_threads = 4096;
    static constexpr int64_t simdgroups = kConv2dO1Simdgroups;
    // On Apple10+, one float output per thread wins for these stride-2
    // reductions.
    static constexpr Range stride2_inputs{24, 64};
    static constexpr int64_t wide_min_threads = 65536;
    // Output columns per thread of the wide variant; its 1x1 vector loads
    // need output rows that keep this alignment.
    static constexpr int64_t wide_columns = 16;
    static constexpr int64_t paired_min_threads = 16384;
    static constexpr int64_t wide_max_inputs = 32;
    static constexpr int64_t short_max_inputs = 16;
    static constexpr int64_t paired_max_inputs = 128;
    // Paired patches beat channel partitions once each core has this many
    // threads (short reductions or float), or regardless at saturation.
    static constexpr int64_t paired_threads_per_core = 1024;
    static constexpr int64_t paired_saturated_threads_per_core = 32768;
  };
  // On Apple10, 16-bit three-channel stems only win on the tiny kernel up to
  // this many outputs; float32 ones win at any count.
  static constexpr int64_t tiny_stride1_max_output_channels = 64;
  // Registered conv2d_tiny_{4,5,6}_3_1_float stems.
  static constexpr Range tiny_float_inputs{4, 6};
  // Empirical output-channel limit for the bounded Tiny preference.
  static constexpr int64_t bounded_tiny_max_output_channels = 24;

  // Collective loads won for batched narrow maps and the C176 large-map
  // specialization on Apple10; the grid must still fill the GPU.
  struct Narrow {
    static constexpr int64_t min_batch = 8;
    static constexpr Range batch_width{32, 96};
    static constexpr Range batch_positions{1024, 16384};
    static constexpr Range large_width{128, 512};
    static constexpr Range large_positions{16384, 131072};
    static constexpr Range outputs{33, 64};
  };

  // Packed gathers trade indexing/staging work for reuse across output
  // channels. The channel/map windows keep this reuse high enough.
  struct Patch {
    // Registered half output tiles minimize padded channel work. The
    // compact 128-row export halves staging when one 16-channel block fits.
    // Keep rows/chunks in sync with the half exports in Convolution.metal.
    static constexpr std::array<int64_t, 3> half_rows{96, 128, 192};
    static constexpr int64_t half_channel_chunk = kConv2dNarrowTile;
    static constexpr int64_t half_compact_channels = kConv2dMinTile;
    // The measured reuse sweep starts at 32 output channels.
    static constexpr int64_t half_min_outputs = 32;
    static constexpr Range sample_inputs{128, 256};
    static constexpr Range embedding_inputs{96, 192};
    static constexpr Range flat_inputs{32, 127};
    static constexpr Range large_outputs{129, 256};
    // The flat bfloat16 exports stage 32-channel chunks; 128-row tiles take
    // 128-256 outputs on mid-size maps.
    static constexpr Range flat_outputs{128, 256};
    static constexpr Range flat_positions{4096, 32768};
    // 64-row tiles take single images with 33-64 outputs; they lose to the
    // generic MPP tiles above three channel chunks and, unless the input is
    // shallow, to the direct fixed 3x3 tiles from 2^22 input elements.
    static constexpr int64_t flat_row_max_inputs = 96;
    static constexpr int64_t flat_row_shallow_inputs = 48;
    static constexpr int64_t flat_row_max_elements = int64_t(1) << 22;
    static constexpr int64_t flat_min_width = 32;
    static constexpr int64_t stride1_min_outputs = 256;
    static constexpr Range stride2_small_inputs{9, 16};
    static constexpr Range stride2_small_outputs{17, 32};
    static constexpr int64_t stride2_min_large_inputs = 128;
    static constexpr Range stride2_large_width{9, 16};
  };
  // Fused stride-two loses for long reductions or small grids without split-K.
  struct Stride2 {
    static constexpr Range inputs{9, 128};
    static constexpr int64_t min_width = 24;
    static constexpr int64_t wide_max_inputs = 16;
  };

  // Copies must pay for themselves in product reuse or avoided padded work.
  // Mid-channel merged copies and batched narrow maps lost in the sweeps;
  // keep the workspace, reduction and map limits together.
  struct Materialization {
    static constexpr int64_t columns_min_outputs = 512;
    static constexpr int64_t columns_min_inputs = 512;
    static constexpr int64_t columns_min_batched_inputs = 384;
    static constexpr int64_t columns_max_batched_positions = 256;
    static constexpr Range columns_positions{512, 4096};
    static constexpr int64_t columns_min_unpadded_positions = 8192;
    static constexpr int64_t columns_min_channel_expansion = 2;
    // Pre-Apple10 float copies need more channel reuse than the direct product.
    static constexpr int64_t float_columns_min_expansion = 4;
    static constexpr int64_t columns_workspace_bytes = 2 * workspace_bytes;
    static constexpr int64_t tiny_max_inputs = 8;
    static constexpr int64_t unfolded_min_outputs = 64;
    static constexpr int64_t unfolded_copy_cost = 5, unfolded_product_cost = 3;
    static constexpr int64_t parity_min_width = 33;
    static constexpr int64_t narrow_max_width = 8;
    static constexpr int64_t narrow_max_small_width = 20;
    static constexpr int64_t narrow_max_positions = 1024;
    static constexpr int64_t merged_max_positions = 1024;
    static constexpr int64_t merged_max_inputs = 16;
    static constexpr int64_t merged_5x5_max_inputs = 32;
    // Grouped products with wide kernels: a group-major copy folds the kW taps
    // into one reduction; deeper groups already fill each per-tap MPP step.
    static constexpr int64_t grouped_merged_min_width = 5;
    static constexpr int64_t grouped_merged_max_inputs = 128;
    static constexpr int64_t merged_copy_cost = 2;
    // A merged copy's trailing output columns, up to this fraction of a tile,
    // take tiles running down the rows; those tiles share no taps between
    // positions, so wider remainders keep narrower or flattened tiles.
    static constexpr int64_t merged_edge_fraction = 16;
    static constexpr int64_t im2col_max_reduction = 1024;
  };

  // Single-tap products have no split-K. On Apple10 batches take them, as do
  // single images up to mid-size maps and reductions or once the product is
  // heavy; before Apple10 only batched compact maps with deep reductions.
  struct Pointwise {
    static constexpr int64_t min_multiply_adds = int64_t(1) << 29;
    static constexpr int64_t mid_max_positions = 32768;
    static constexpr int64_t mid_max_inputs = 1024;
    // The 128-row product has higher throughput unless 160 rows save enough
    // padding. Register-input tiles avoid that padding on large spatial maps.
    static constexpr float large_tile_padding_cost = 1.125f;
    static constexpr int64_t plane_positions_per_output_channel = 256;
    static constexpr Range large_outputs{128, 256};
    static constexpr Range compact_positions{512, 1024};
    static constexpr int64_t min_deep_inputs = 256;
    static constexpr int64_t large_tile_min_inputs = 128;
    static constexpr int64_t large_tile_min_outputs = 65;
    // Split-K beats a single-tap grid of at most one threadgroup per this many
    // input channels.
    static constexpr int64_t split_inputs_per_threadgroup = 64;
  };
  // conv2d_tensor copies NCHW inputs to NHWC first; from this many input
  // elements the copy outweighs the op's gain over the direct kernels.
  static constexpr int64_t tensor_max_copied_inputs = int64_t(1) << 22;

  // Unrolling all nine taps pays off for large maps or well-filled 96-row
  // products; smaller cases retain the generic product.
  struct Fixed3x3 {
    static constexpr Range large_channels{33, 64};
    static constexpr int64_t min_large_positions = 131072;
    // Deeper single images keep 64-row tiles busy from about 220x220 maps;
    // above 192 outputs, 128-channel inputs still prefer F(4x2).
    static constexpr Range deep_inputs{32, 128};
    static constexpr Range deep_outputs{64, 256};
    static constexpr int64_t min_deep_positions = 49152;
    static constexpr int64_t max_tall_outputs = 192;
    static constexpr int64_t min_inputs = 17;
    static constexpr Range mid_outputs{65, 96};
  };

  // Four groups are preferred for short/mid reductions and, on Apple10, for
  // direct products of a few taps. K splitting uses 32-element reduction
  // blocks, independent of the hardware SIMD width.
  static constexpr Range four_simdgroup_reduction{17, 96};
  static constexpr int64_t four_simdgroup_max_taps = 3;
  static constexpr int64_t split_reduction_alignment = 32;
  static constexpr int64_t wide_tile_max_reduction = 64;

  // Flattened rows: channel-dependent and fixed waste penalties for the
  // padded copy, then a minimum saved reduction.
  static constexpr float flatten_channel_penalty = 11.0f;
  static constexpr float flatten_copy_penalty = 0.08f;
  static constexpr float flatten_min_saved_reduction = 1000.0f;
  // Required reduction in padded work before narrowing the spatial tile.
  // Shared by direct and stride-two tile selection.
  static constexpr float narrow_min_padding_gain = 0.14f;
  static constexpr float stride2_max_padding_waste = 0.35f;
  static constexpr int64_t stride2_min_threadgroups_per_core = 4;
  // Minimum K per split; smaller output-channel tiles permit twice as many
  // splits to fill the GPU.
  static constexpr int64_t min_split_reduction = 256;
  static constexpr int64_t small_tile_min_split_reduction =
      min_split_reduction / 2;
};

const Conv2dTuning& conv2d_tuning(bool apple10_or_newer);

struct Conv2dDevice {
  // Bfloat simdgroup matrices.
  bool supports_bfloat_simdgroup;
  bool supports_large_mpp_strides;
  // 8 threadgroups per core fill the GPU (16 before), and low-precision fused
  // Winograd loses.
  bool apple10_or_newer;
  int64_t cores;
  // Threadgroups that fill the GPU.
  int64_t target;
  const Conv2dTuning& tuning;
};

struct Conv2dPlan {
  Conv2dKernel kernel;
  // Threadgroup tile: output channels x output positions, and simdgroups per
  // threadgroup. SingleOutput instead uses spatial rows/columns per thread.
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t simdgroups = 0;
  // SingleOutput: channel residues reduced within each SIMD group.
  int64_t channel_partitions = 1;
  // SingleOutput: kernel width and horizontal stride baked into the kernel.
  bool fixed_taps = false;
  // Patch and Mpp: tiles run over the flattened image, not per output row.
  bool flat = false;
  // WinogradFused: one threadgroup transforms its input tiles once and runs
  // every 32-output block over them (conv2d_winograd_loop).
  bool output_loop = false;
  // Mpp only. Fixed3x3 and Pointwise fix the tile above; Generic leaves it to
  // select_conv2d_mpp_tile.
  Conv2dMppInput input = Conv2dMppInput::Direct;
  Conv2dMppVariant variant = Conv2dMppVariant::Generic;
  // Deinterleave even/odd columns so a stride-2 row reads with stride 1.
  bool parity = false;
  // <= 8 input channels feeding >= 64 outputs with short tap rows (Apple10
  // NCHW input): taken as Unfolded when no other layout claims the input;
  // either way the GEMM gets 4 simdgroups.
  bool unfold_cheap = false;
  // Tiny: bounded row tails and explicit input/weight/output layouts.
  bool bounded_tiny = false;
};

struct Conv2dMppTile {
  int64_t rows;
  int64_t columns;
  int64_t simdgroups;
  // Position tiles per image.
  int64_t tiles;
  // K elements per split; `splits` partial sums go through conv2d_split_reduce.
  int64_t k_split;
  int64_t splits;
  Conv2dMppVariant variant;
  // Trailing output columns tiled down the rows
  // (Conv2dMppParams::edge_columns).
  int64_t edge_columns = 0;
};

const Conv2dDevice& conv2d_device();
// conv2d_pointwise_{128,256} specializations; other channel counts use
// conv2d_pointwise_dynamic.
inline bool conv2d_pointwise_fixed_channels(int64_t channels) {
  return channels == 128 || channels == 256;
}

// conv2d_winograd tile for a 3x3, stride-1 convolution.
Conv2dPlan select_conv2d_winograd(
    const Conv2DParams& params,
    bool in_nhwc,
    const Conv2dDevice& device);
// `input` is contiguous in NCHW or channels-last.
Conv2dPlan select_conv2d_kernel(
    const Conv2DParams& params,
    const Tensor& input,
    const Tensor& output,
    int64_t groups,
    const Conv2dDevice& device);
// Tile for conv2d_mpp from the geometry after the input was prepared per
// plan.input.
Conv2dMppTile select_conv2d_mpp_tile(
    const Conv2dMppParams& params,
    const Conv2dPlan& plan,
    int64_t groups,
    const Conv2dDevice& device);

} // namespace at::native::mps
