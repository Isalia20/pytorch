#pragma once
#include <c10/metal/common.h>

// Outputs each conv1d_depthwise thread keeps in registers, so every weight
// tap is loaded once per 8 outputs.
C10_METAL_CONSTEXPR int32_t kConv1dDepthwiseOutputsPerThread = 8;

// nchw_to_nhwc stages channel_tile x (kNchwToNhwcTileElements / channel_tile)
// elements per threadgroup; channel tiles are powers of two in this range.
C10_METAL_CONSTEXPR int32_t kNchwToNhwcTileElements = 1024;
C10_METAL_CONSTEXPR int32_t kNchwToNhwcMinChannelTile = 4;
C10_METAL_CONSTEXPR int32_t kNchwToNhwcMaxChannelTile = 32;
// Paired (vec2) reads are registered for these channel tiles and wider.
C10_METAL_CONSTEXPR int32_t kNchwToNhwcVectorChannelTile = 16;

// MPP tile ladder instantiated by INSTANTIATE_CONV2D_MPP_ALL. Keep the
// registration list and these bounds in sync; dimensions are in elements.
C10_METAL_CONSTEXPR int32_t kConv2dMinTile = 16;
C10_METAL_CONSTEXPR int32_t kConv2dNarrowTile = 2 * kConv2dMinTile;
C10_METAL_CONSTEXPR int32_t kConv2dRowTile = 2 * kConv2dNarrowTile;
C10_METAL_CONSTEXPR int32_t kConv2dWideTile = 2 * kConv2dRowTile;
// Output-channel tile for groups of at most eight output channels; only the
// 64-position, two-SIMD-group product is registered.
C10_METAL_CONSTEXPR int32_t kConv2dFewOutputTile = kConv2dMinTile / 2;
// Fixed 3x3 MPP products over at most kConv2dFixed3ColumnMaxInputs channels
// start their tiles at column 1 and compute output column 0 in threadgroups of
// kConv2dFixed3ColumnRows rows; deeper reductions amortize the left-padding
// shift of their first tiles instead.
C10_METAL_CONSTEXPR int32_t kConv2dFixed3ColumnMaxInputs = 64;
C10_METAL_CONSTEXPR int32_t kConv2dFixed3ColumnRows = kConv2dMinTile;
// conv2d_tensor tiles: output channels x output rows x output columns, run by
// two SIMD groups; the wide tile covers 64 channels x 8 x 8 positions on four.
// Inputs with at most kConv2dTensorStemChannels channels use the variant that
// bakes that bound into the op.
C10_METAL_CONSTEXPR int32_t kConv2dTensorOutputs = kConv2dNarrowTile;
C10_METAL_CONSTEXPR int32_t kConv2dTensorRows = 2;
C10_METAL_CONSTEXPR int32_t kConv2dTensorColumns = kConv2dMinTile;
C10_METAL_CONSTEXPR int32_t kConv2dTensorSimdgroups = 2;
C10_METAL_CONSTEXPR int32_t kConv2dTensorWideOutputs = kConv2dRowTile;
C10_METAL_CONSTEXPR int32_t kConv2dTensorWideSide = 8;
C10_METAL_CONSTEXPR int32_t kConv2dTensorWideSimdgroups = 4;
C10_METAL_CONSTEXPR int32_t kConv2dTensorStemChannels = kConv2dMinTile;
// Threadgroup size of the flat launches (layout copies, im2col, split-K
// reduction) and of the bounded conv2d_tiny kernels.
C10_METAL_CONSTEXPR int32_t kConv2dFlatThreads = 8 * c10::metal::simdgroup_size;

// Single-output launch geometry; cooperative reductions share a SIMD group.
C10_METAL_CONSTEXPR int32_t kConv2dO1Simdgroups = 8;
C10_METAL_CONSTEXPR int32_t kConv2dO1ChannelPartitions = 8;

// Maximum horizontal tap offset (kW - 1) * dW supported by conv2d_s2.
C10_METAL_CONSTEXPR int32_t kConv2dStride2MaxTapSpan = 6;
// Target bytes for both input buffers before their alignment and tap halo.
C10_METAL_CONSTEXPR int32_t kConv2dStride2BaseStageBytes = 16 * 1024;

// Source element strides of the OIDHW weight view (may be non-contiguous).
struct ConvWeightPermuteParams {
  uint32_t output_channels;
  uint32_t input_channels_per_group;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t output_channel_stride;
  uint32_t input_channel_stride;
  uint32_t depth_stride;
  uint32_t height_stride;
  uint32_t width_stride;
  // Only the padded direct conv2d permutation appends nonfinite flags.
  bool emit_nonfinite_flags;
};

static_assert(
    sizeof(ConvWeightPermuteParams) == 40,
    "Shared host/Metal parameter layout");

struct Conv2DParams {
  int32_t N;
  int32_t C_in;
  int32_t C_out;
  int32_t H;
  int32_t W;
  int32_t outH;
  int32_t outW;
  int32_t kH;
  int32_t kW;
  int32_t sH;
  int32_t sW;
  int32_t padH;
  int32_t padW;
  int32_t dH;
  int32_t dW;
  int32_t C_in_per_group;
  int32_t C_out_per_group;
  bool has_bias;
};

// conv2d_depthwise computes kConv2dDepthwiseRows output rows and
// conv2d_depthwise_columns(kW) columns per thread; the 7x7 specialization
// halves the columns to bound its input window.
C10_METAL_CONSTEXPR int32_t kConv2dDepthwiseRows = 2;
C10_METAL_CONSTEXPR int32_t kConv2dDepthwiseColumns = 8;
inline constexpr int32_t conv2d_depthwise_columns(int32_t kernel_width) {
  return kernel_width == 7 ? kConv2dDepthwiseColumns / 2
                           : kConv2dDepthwiseColumns;
}

// conv2d_im2col writes this many consecutive output positions per thread as
// two packed four-element stores.
C10_METAL_CONSTEXPR int32_t kConv2dIm2colPositionsPerThread = 8;

// Tiny keeps two float4 spatial vectors for each of four output channels.
C10_METAL_CONSTEXPR int32_t kConv2dTinyPositionsPerThread = 8;
C10_METAL_CONSTEXPR int32_t kConv2dTinyChannelsPerThread = 4;

struct Conv2dTinyParams {
  Conv2DParams conv;
  bool input_channels_last;
  bool output_channels_last;
  bool weight_channels_last;
};

static_assert(
    sizeof(Conv2dTinyParams) == 76,
    "Shared host/Metal parameter layout");

struct Conv2dMppParams {
  Conv2DParams conv;
  int32_t K;
  int32_t K_split;
  int32_t splits;
  int32_t Wp;
  int32_t parity; // Even/odd plane stride; zero selects ordinary columns.
  // FP32 scratch columns per output row for left-padding shifts.
  int32_t shift_width;
  // Input elements between the first channels of consecutive groups.
  int32_t group_stride;
  // Trailing output columns that NHWC inputs tile down the rows instead of
  // with a mostly empty tile per row.
  int32_t edge_columns;
};

struct Conv2dWinogradRepairParams {
  int32_t channels;
  int32_t height;
  int32_t width;
  int32_t pad_height;
  int32_t pad_width;
  bool channels_last;
  // OHWI-ordered 3x3 weights instead of OIHW.
  bool weight_channels_last;
};

static_assert(
    sizeof(Conv2dWinogradRepairParams) == 24,
    "Shared host/Metal parameter layout");

// conv2d_winograd threadgroups: each SIMD group owns two of the 16 F(2x2, 3x3)
// coefficients of the whole output block.
C10_METAL_CONSTEXPR int32_t kConv2dWinogradSimdgroups = 8;
// conv2d_winograd_weights transforms weights once per call into blocks of this
// many output channels.
C10_METAL_CONSTEXPR int32_t kConv2dWinogradPretransformedRows = 32;

// conv2d_winograd_loop runs 32-output x 16-tile blocks over inputs of at most
// one chunk, this many channels.
C10_METAL_CONSTEXPR int32_t kConv2dWinogradLoopChannels =
    kConv2dWinogradSimdgroups * c10::metal::simdgroup_size / kConv2dMinTile;

// conv2d_winograd tiles that read those blocks; the 16x16 and 8-row tiles
// transform weights per input-channel chunk instead, in a single dispatch.
inline constexpr bool conv2d_winograd_pretransformed(
    int32_t rows,
    int32_t columns) {
  return rows == kConv2dWinogradPretransformedRows ||
      (rows == kConv2dMinTile && columns == kConv2dNarrowTile);
}

// conv2d_winograd geometry: outputs are tiles_w x ceil(outH / 2) tiles of 2x2
// pixels and the input channels are consumed in c_blocks chunks.
struct Conv2dWinogradParams {
  int32_t C_in;
  int32_t C_out;
  int32_t H;
  int32_t W;
  int32_t outH;
  int32_t outW;
  int32_t pad;
  int32_t tiles_w;
  int32_t tiles;
  int32_t c_blocks;
  bool has_bias;
  // OHWI-ordered 3x3 weights instead of OIHW.
  bool weight_channels_last;
};

static_assert(
    sizeof(Conv2dWinogradParams) == 44,
    "Shared host/Metal parameter layout");

struct Conv2dSplitParams {
  int32_t channels;
  int32_t height;
  int32_t width;
  int32_t pitch;
  int32_t splits;
  int32_t batch;
  bool has_bias;
  bool channels_last;
};

// conv2d_group_major: channels-last input copied to the padded group-major
// [groups][N][H + 2 pad][W + 2 pad][C / groups] layout of grouped merged
// products.
struct Conv2dGroupMajorParams {
  int32_t batch;
  int32_t channels;
  int32_t group_channels;
  int32_t height;
  int32_t width;
  int32_t pad_height;
  int32_t pad_width;
};

// Channels copied by one conv2d_group_major thread.
C10_METAL_CONSTEXPR int32_t kConv2dGroupMajorChannelsPerThread = 8;

static_assert(sizeof(Conv2DParams) == 72, "Shared host/Metal parameter layout");
static_assert(
    sizeof(Conv2dGroupMajorParams) == 28,
    "Shared host/Metal parameter layout");
static_assert(
    sizeof(Conv2dMppParams) == 104,
    "Shared host/Metal parameter layout");
static_assert(
    sizeof(Conv2dSplitParams) == 28,
    "Shared host/Metal parameter layout");

// Shared between Convolution.metal and operations/Convolution.mm. conv3d_mpp
// bakes kernel, stride, and dilation into template args; conv3d_simd reads
// them from here.
struct Conv3DParams {
  Conv2DParams conv2d;
  int32_t D;
  int32_t outD;
  int32_t kD;
  int32_t sD;
  int32_t padD;
  int32_t dD;
  bool out_ncdhw;
};
