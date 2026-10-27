#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/kernels/Convolution.h>

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

// conv2d on the MPP kernels (macOS 26.2+); select_conv2d_kernel picks one.
enum class Conv2dKernel {
  // winograd_* F(2x2,3x3) transforms around a 16-group pointwise conv2d_mpp.
  Winograd2x2,
  // winograd_*_4 F(4x2,3x3) transforms around 24 conv2d_mpp pw2 GEMMs.
  Winograd4x2,
  // conv2d_tiny: bfloat 1-, 3- or 4-channel input (stem convs), 8 outputs
  // per thread.
  Tiny,
  // conv2d_o1: one output channel per group.
  SingleOutput,
  // conv2d_winograd_fused: Apple9 F(2x2) with the tap GEMM in-kernel, up to
  // 32 output channels.
  WinogradFused,
  // conv2d_pointwise: 1x1 over one plane of >= 512x512 with 128 or 256 input
  // channels.
  PointwisePlane,
  // addmm over [C_out, C_in] x [C_in, H*W].
  PointwiseMatmul,
  // conv2d_narrow: 3x3 on 64/128/176 channels, the whole weight staged in
  // threadgroup memory.
  Narrow,
  // conv2d_patch: 2x2/3x3 tiles with the input patch staged per threadgroup.
  Patch,
  // conv2d_s2: stride-2 rows read directly from NCHW, <= 128 input channels.
  Stride2,
  // conv2d_mpp: GEMM over the input prepared per Conv2dMppInput, split-K when
  // the grid is small.
  Mpp,
};

// Layout conv2d_mpp reads. All but Direct and Pointwise copy the input.
enum class Conv2dMppInput {
  // As given; a stride-2 NCHW input is converted to NHWC first.
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
  // Padded NHWC copy with the kW taps folded into K; the kH rows stay taps.
  Merged,
};

enum class Conv2dMppVariant {
  Generic,
  // st_: output tile staged through threadgroup memory for 8-wide NCHW stores
  // (Apple9, NHWC input).
  Staged,
  // k3_: 3x3 stride-1 taps unrolled, bias seeded into the accumulator.
  Fixed3x3,
  // pw1_: single tap.
  Pointwise,
};

struct Conv2dDevice {
  // Bfloat simdgroup matrices.
  bool apple9;
  // 8 threadgroups per core fill the GPU (16 before); fused Winograd and
  // staged tiles lose.
  bool apple10;
  int64_t cores;
  // Threadgroups that fill the GPU.
  int64_t target;
};

struct Conv2dPlan {
  Conv2dKernel kernel;
  // Threadgroup tile: output channels x output positions, and simdgroups per
  // threadgroup.
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t simdgroups = 0;
  // SingleOutput: kernel width and unit stride baked into the kernel.
  bool fixed_taps = false;
  // Patch and Mpp: tiles run over the flattened image, not per output row.
  bool flat = false;
  // Mpp only. Fixed3x3 and Pointwise fix the tile above; Generic leaves it to
  // select_conv2d_mpp_tile.
  Conv2dMppInput input = Conv2dMppInput::Direct;
  Conv2dMppVariant variant = Conv2dMppVariant::Generic;
  // Deinterleave even/odd columns so a stride-2 row reads with stride 1.
  bool parity = false;
  // <= 8 input channels feeding >= 64 outputs with short tap rows: taken as
  // Unfolded when no other layout claims the input; either way the GEMM gets
  // 4 simdgroups and, on Apple9 with NHWC input, staged stores.
  bool unfold_cheap = false;
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
};

const Conv2dDevice& conv2d_device();
// `input` is contiguous in NCHW or channels-last.
Conv2dPlan select_conv2d_kernel(
    const Conv2DParams& params,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    int64_t groups,
    const Conv2dDevice& device);
// Tile for conv2d_mpp from the geometry after the input was prepared per
// plan.input.
Conv2dMppTile select_conv2d_mpp_tile(
    const Conv2dMppParams& params,
    const Conv2dPlan& plan,
    int64_t groups,
    bool in_nhwc,
    bool out_nhwc,
    const Conv2dDevice& device);

} // namespace at::native::mps
