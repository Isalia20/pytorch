//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ConvUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Convolution.h>
#include <ATen/native/mps/operations/ConvolutionHeuristics.h>
#include <ATen/ops/_mps_convolution_native.h>
#include <ATen/ops/_mps_convolution_transpose_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/im2col.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>
#include <c10/util/TypeCast.h>
#include <c10/util/safe_numerics.h>
#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <string_view>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Convolution_metallib.h>
#endif

// `memory_format` selects NDHWC vs NCDHW; `use_dhwio` selects DHWIO vs OIDHW
// (caller must insert the matching in-graph weight transpose).
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             c10::MemoryFormat memory_format,
                             bool use_dhwio,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::ChannelsLast3d) ? MPSGraphTensorNamedDataLayoutNDHWC
                                                                               : MPSGraphTensorNamedDataLayoutNCDHW;
  descriptor_.weightsLayout = use_dhwio ? MPSGraphTensorNamedDataLayoutDHWIO : MPSGraphTensorNamedDataLayoutOIDHW;

  descriptor_.groups = groups; // not yet tested in Xcode/C++
}

// Exact-stride match: a sliced view of CL3d has CL-like strides but isn't
// NHWC-packed; the raw-buffer NDHWC path would misread it (#180984).
static bool is_packed_channels_last_3d(const Tensor& t) {
  return t.dim() == 5 &&
      t.suggest_memory_format(/*channels_last_strides_exact_match=*/true) == at::MemoryFormat::ChannelsLast3d;
}

static bool is_packed_channels_last_2d(const Tensor& t) {
  return t.dim() == 4 &&
      t.suggest_memory_format(/*channels_last_strides_exact_match=*/true) == at::MemoryFormat::ChannelsLast;
}

// `a` supplies the rank, so it has to be an activation
static at::MemoryFormat conv_backward_memory_format(const Tensor& a, const Tensor& b) {
  if (!mps_conv_use_channels_last(a, b)) {
    return at::MemoryFormat::Contiguous;
  }
  return a.dim() == 5 ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
}

// DHWIO costs one in-graph weight transpose per call; only worth it when
// Cin/groups is large enough and the kernel is not factorized.
static bool conv3d_dhwio_is_beneficial(IntArrayRef weight_size) {
  constexpr int64_t kMinCinPerGroup = 4; // skip first-layer Cin=3, depthwise Cin/g=1.
  constexpr int64_t kMinKernelDim = 2; // skip 1x3x3, 3x1x1, 1x1x1.
  return weight_size.size() == 5 && weight_size[1] >= kMinCinPerGroup && weight_size[2] >= kMinKernelDim &&
      weight_size[3] >= kMinKernelDim && weight_size[4] >= kMinKernelDim;
}

// Force the tensor's stride pattern to match `desc_layout`; MPSGraph's 3D
// conv path takes a slow strided route otherwise. 4D NCHW tensors pass through.
static Tensor materialize_for_conv(const Tensor& t, c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::ChannelsLast3d) {
    return t.contiguous(at::MemoryFormat::ChannelsLast3d);
  }
  if (desc_layout == at::MemoryFormat::ChannelsLast) {
    return t.contiguous(at::MemoryFormat::ChannelsLast);
  }
  if (t.dim() == 5) {
    return t.contiguous();
  }
  return t;
}

// CL3d needs the NDArray path for explicit NDHWC ordering; NCDHW takes the
// tensor-direct Placeholder. Caller must materialize_for_conv first.
static at::native::mps::Placeholder make_conv_placeholder(MPSGraphTensor* graphTensor,
                                                          const at::Tensor& t,
                                                          c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::Contiguous) {
    return at::native::mps::Placeholder(graphTensor, t);
  }
  return at::native::mps::Placeholder(graphTensor,
                                      at::native::mps::getMPSNDArray(t, at::native::mps::getMPSShape(t, desc_layout)));
}

template <typename Encode>
static void conv_metal_launch(id<MTLComputePipelineState> pipeline,
                              const char* kernel_name,
                              ArrayRef<Tensor> inputs,
                              const Encode& encode) {
  using namespace mps;
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, kernel_name, inputs, stream);
      [encoder setComputePipelineState:pipeline];
      encode(encoder);
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

static void nchw_to_nhwc(const Tensor& source,
                         const Tensor& output,
                         int64_t channels,
                         int64_t height,
                         int64_t width,
                         IntArrayRef pad = {0, 0},
                         int64_t kH = 1,
                         int64_t sH = 1,
                         int64_t dH = 1) {
  using namespace mps;
  const int64_t batches = source.numel() / (channels * height * width);
  const int64_t rows = output.numel() / (batches * channels * (width + 2 * pad[1]) * kH);
  const int64_t padded = rows * (width + 2 * pad[1]) * kH;
  // The smallest registered channel tile covering `channels`; fewer channel rows widen the spatial tile.
  const int64_t channel_tile = std::clamp<int64_t>(
      std::bit_ceil(static_cast<uint64_t>(channels)), kNchwToNhwcMinChannelTile, kNchwToNhwcMaxChannelTile);
  const int64_t spatial_tile = kNchwToNhwcTileElements / channel_tile;
  // vec2 loads need the buffer base 2-element aligned too
  const bool vectorized_read = channel_tile >= kNchwToNhwcVectorChannelTile && pad[0] == 0 && pad[1] == 0 && kH == 1 &&
      sH == 1 && dH == 1 && padded % 2 == 0 && source.storage_offset() % 2 == 0;
  const bool vectorized_write = channels % 2 == 0;
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("nchw_to_nhwc_{}_{}_{}_{}_{}",
                                                          scalarToMetalTypeString(source),
                                                          channel_tile,
                                                          spatial_tile,
                                                          vectorized_read,
                                                          vectorized_write));
  const std::array<int32_t, 4> dims = {c10::checked_convert<int32_t>(channels, "int32_t"),
                                       c10::checked_convert<int32_t>(height, "int32_t"),
                                       c10::checked_convert<int32_t>(width, "int32_t"),
                                       c10::checked_convert<int32_t>(width + 2 * pad[1], "int32_t")};
  const std::array<int32_t, 2> pads = {c10::checked_convert<int32_t>(pad[0], "int32_t"),
                                       c10::checked_convert<int32_t>(pad[1], "int32_t")};
  const std::array<int32_t, 4> filter_rows = {c10::checked_convert<int32_t>(kH, "int32_t"),
                                              c10::checked_convert<int32_t>(sH, "int32_t"),
                                              c10::checked_convert<int32_t>(dH, "int32_t"),
                                              c10::checked_convert<int32_t>(rows, "int32_t")};
  const auto threadgroups =
      MTLSizeMake(c10::metal::ceil_div(padded, spatial_tile), c10::metal::ceil_div(channels, channel_tile), batches);
  conv_metal_launch(pipeline, "nchw_to_nhwc", {source}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, source, output, dims, pads, filter_rows);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:MTLSizeMake(kConv2dFlatThreads, 1, 1)];
  });
}

// NDHWC copy of a 5D tensor. Packed channels-last passes through; channels <= 8
// barely fills a transpose tile, so the plain strided copy wins there.
static Tensor conv3d_to_ndhwc(const Tensor& tensor) {
  if (tensor.is_contiguous(MemoryFormat::ChannelsLast3d)) {
    return tensor;
  }
  const auto channels = tensor.size(1);
  const int64_t spatial_size = tensor.size(2) * tensor.size(3) * tensor.size(4);
  if (channels <= 8 || spatial_size > std::numeric_limits<int32_t>::max()) {
    return tensor.contiguous(MemoryFormat::ChannelsLast3d);
  }
  auto output = at::empty(tensor.sizes(), tensor.options(), MemoryFormat::ChannelsLast3d);
  nchw_to_nhwc(tensor.contiguous(), output, channels, 1, spatial_size);
  return output;
}

// DHWIO weight copy; ATen's strided permute copy runs well below memory
// bandwidth for this permutation, 2-4x slower than the flat kernel.
static Tensor conv3d_weights_to_dhwio(const Tensor& weight, bool emit_nonfinite_flags = false) {
  using namespace mps;
  if (!offsetsFitIn<uint32_t>(weight)) {
    return emit_nonfinite_flags ? conv3d_weights_to_dhwio(weight.contiguous(), true)
                                : weight.permute({2, 3, 4, 1, 0}).contiguous();
  }
  const auto output_channels = weight.size(0);
  const auto input_channels_per_group = weight.size(1);
  const auto kernel_depth = weight.size(2);
  const auto kernel_height = weight.size(3);
  const auto kernel_width = weight.size(4);
  TORCH_INTERNAL_ASSERT(!emit_nonfinite_flags || kernel_depth == 1);
  const auto flag_bytes = emit_nonfinite_flags ? kernel_height * input_channels_per_group *
          c10::metal::ceil_div(output_channels, int64_t(c10::metal::simdgroup_size))
                                               : 0;
  auto output = emit_nonfinite_flags
      ? at::empty({weight.numel() + c10::metal::ceil_div(flag_bytes, int64_t(weight.element_size()))}, weight.options())
      : at::empty({kernel_depth, kernel_height, kernel_width, input_channels_per_group, output_channels},
                  weight.options());
  const ConvWeightPermuteParams params{
      .output_channels = static_cast<uint32_t>(output_channels),
      .input_channels_per_group = static_cast<uint32_t>(input_channels_per_group),
      .kernel_height = static_cast<uint32_t>(kernel_height),
      .kernel_width = static_cast<uint32_t>(kernel_width),
      .output_channel_stride = static_cast<uint32_t>(weight.stride(0)),
      .input_channel_stride = static_cast<uint32_t>(weight.stride(1)),
      .depth_stride = static_cast<uint32_t>(weight.stride(2)),
      .height_stride = static_cast<uint32_t>(weight.stride(3)),
      .width_stride = static_cast<uint32_t>(weight.stride(4)),
      .emit_nonfinite_flags = emit_nonfinite_flags,
  };
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv_weight_to_dhwio_{}", scalarToMetalTypeString(weight)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv_weight_to_dhwio", {weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, weight, output, params);
      [encoder dispatchThreads:MTLSizeMake(output_channels, input_channels_per_group, kernel_depth * kernel_height)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(output_channels, 256), 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
  return output;
}

struct Conv3dMppSpecialization {
  std::string dtype;
  int kernel_depth, kernel_height, kernel_width;
  int stride_depth, stride_height, stride_width;
  int dilation_depth, dilation_height, dilation_width;
  int src_channels, src_width, src_height;
  bool has_bias, out_ncdhw, grouped;
};

static id<MTLComputePipelineState> conv3d_mpp_pipeline(const Conv3dMppSpecialization& specialization,
                                                       const mps::Conv3dMppTile& tile) {
  if (specialization.src_width != 16384 || specialization.src_height != 16384) {
    return nil;
  }
  auto build_name = [&](const std::string& src_channels) {
    return fmt::format("conv3d_mpp_{}_b{}_w{}_h{}_s{}_k{}{}{}_s{}{}{}_d{}{}{}_c{}_{}_{}_{}",
                       specialization.dtype,
                       tile.output_channels,
                       tile.output_width,
                       tile.output_height,
                       tile.simdgroups,
                       specialization.kernel_depth,
                       specialization.kernel_height,
                       specialization.kernel_width,
                       specialization.stride_depth,
                       specialization.stride_height,
                       specialization.stride_width,
                       specialization.dilation_depth,
                       specialization.dilation_height,
                       specialization.dilation_width,
                       src_channels,
                       specialization.has_bias ? "bias" : "nobias",
                       specialization.out_ncdhw ? "ncdhw" : "ndhwc",
                       specialization.grouped ? "grouped" : "ungrouped");
  };
  if (specialization.src_channels >= 0) {
    const auto name = build_name(std::to_string(specialization.src_channels));
    if (lib.hasFunction(name)) {
      return lib.getPipelineStateForFunc(name);
    }
  }
  const auto name = build_name("dyn");
  return lib.hasFunction(name) ? lib.getPipelineStateForFunc(name) : nil;
}

MTLComputePipelineState_t mps::Conv3dSimdTile::pipeline_state(mps::MetalShaderLibrary& library,
                                                              const std::string& dtype,
                                                              bool use_long_index) const {
  const char* prefix = use_long_index ? "conv3d_simd_long" : "conv3d_simd";
  const auto name =
      fmt::format("{}_{}_{}_{}_{}_{}", prefix, dtype, block_rows, block_columns, simdgroups_rows, simdgroups_columns);
  return library.getPipelineStateForFunc(name);
}

// Coordinate expressions may overflow even when the input and output are small.
// Allow the widest spatial tile for lanes beyond the last valid output.
static bool conv_needs_long_coordinates(IntArrayRef input_size,
                                        IntArrayRef output_size,
                                        IntArrayRef kernel_size,
                                        IntArrayRef stride,
                                        IntArrayRef padding,
                                        IntArrayRef dilation) {
  constexpr int64_t max_coordinate = std::numeric_limits<int32_t>::max();
  constexpr int64_t output_tail = kConv2dWideTile;
  for (const auto dim : c10::irange(input_size.size())) {
    if (input_size[dim] > max_coordinate || output_size[dim] > max_coordinate - output_tail ||
        kernel_size[dim] > max_coordinate || stride[dim] > max_coordinate || padding[dim] > max_coordinate ||
        dilation[dim] > max_coordinate) {
      return true;
    }
    if ((output_size[dim] + output_tail) * stride[dim] + kernel_size[dim] * dilation[dim] > max_coordinate ||
        padding[dim] + stride[dim] - 1 > max_coordinate || input_size[dim] + stride[dim] - 1 > max_coordinate) {
      return true;
    }
  }
  return false;
}

// conv3d forward on the Metal kernels: MPP on macOS 26.2+, simdgroup otherwise;
// planes past int32 take the long-indexed simdgroup variant on any macOS.
static void conv3d_metal_forward(const Tensor& input_t,
                                 const Tensor& weight_t,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef padding,
                                 IntArrayRef stride,
                                 IntArrayRef dilation,
                                 int64_t groups,
                                 const Tensor& output_t,
                                 bool mpp_compatible = true) {
  using namespace mps;
  const auto dtype = input_t.scalar_type();
  const bool bias_defined = bias_opt && bias_opt->defined();
  const bool out_ncdhw = output_t.is_contiguous();
  TORCH_INTERNAL_ASSERT(out_ncdhw || is_packed_channels_last_3d(output_t));
  const int64_t input_channels = input_t.size(1);
  const int64_t input_depth = input_t.size(2);
  const int64_t input_height = input_t.size(3);
  const int64_t input_width = input_t.size(4);
  const int64_t output_channels = output_t.size(1);
  const int64_t output_depth = output_t.size(2);
  const int64_t output_height = output_t.size(3);
  const int64_t output_width = output_t.size(4);
  if (input_channels == 0) {
    // reduction over an empty channel set: bias (or zeros)
    if (bias_defined) {
      output_t.copy_(bias_opt->reshape({1, output_channels, 1, 1, 1}).expand_as(output_t));
    } else {
      output_t.zero_();
    }
    return;
  }
  // The MPP path addresses one (batch, depth) activation plane and one output
  // plane with int32 tensor extents; larger planes run the long-indexed
  // simdgroup kernel.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const bool use_long_index =
      input_channels * input_height * input_width > kInt32Max ||
      output_channels * output_height * output_width > kInt32Max ||
      conv_needs_long_coordinates(
          input_t.sizes().slice(2), output_t.sizes().slice(2), weight_t.sizes().slice(2), stride, padding, dilation);
  const int64_t input_channels_per_group = input_channels / groups;
  TORCH_CHECK(weight_t.size(2) * weight_t.size(3) * weight_t.size(4) * input_channels_per_group <= kInt32Max,
              "conv3d: kernel volume times channels per group exceeds int32");
  const bool use_mpp = mpp_compatible && !use_long_index && has_mpp();

  const auto activation = conv3d_to_ndhwc(input_t); // NDHWC
  const auto weights = conv3d_weights_to_dhwio(weight_t); // DHWIO
  std::optional<Tensor> bias;
  if (bias_defined) {
    bias = bias_opt->scalar_type() == dtype ? bias_opt->contiguous() : bias_opt->to(dtype).contiguous();
  }

  const int64_t output_channels_per_group = output_channels / groups;
  Conv3DParams params;
  params.conv2d.N = static_cast<int32_t>(input_t.size(0));
  params.conv2d.C_in = static_cast<int32_t>(input_channels);
  params.conv2d.C_out = static_cast<int32_t>(output_channels);
  params.conv2d.H = static_cast<int32_t>(input_height);
  params.conv2d.W = static_cast<int32_t>(input_width);
  params.conv2d.outH = static_cast<int32_t>(output_height);
  params.conv2d.outW = static_cast<int32_t>(output_width);
  params.conv2d.kH = static_cast<int32_t>(weight_t.size(3));
  params.conv2d.kW = static_cast<int32_t>(weight_t.size(4));
  params.conv2d.sH = static_cast<int32_t>(stride[1]);
  params.conv2d.sW = static_cast<int32_t>(stride[2]);
  params.conv2d.padH = static_cast<int32_t>(padding[1]);
  params.conv2d.padW = static_cast<int32_t>(padding[2]);
  params.conv2d.dH = static_cast<int32_t>(dilation[1]);
  params.conv2d.dW = static_cast<int32_t>(dilation[2]);
  params.conv2d.C_in_per_group = static_cast<int32_t>(input_channels_per_group);
  params.conv2d.C_out_per_group = static_cast<int32_t>(output_channels_per_group);
  params.conv2d.has_bias = bias_defined;
  params.D = static_cast<int32_t>(input_depth);
  params.outD = static_cast<int32_t>(output_depth);
  params.kD = static_cast<int32_t>(weight_t.size(2));
  params.sD = static_cast<int32_t>(stride[0]);
  params.padD = static_cast<int32_t>(padding[0]);
  params.dD = static_cast<int32_t>(dilation[0]);
  params.out_ncdhw = out_ncdhw;

  const auto dtype_name = scalarToMetalTypeString(input_t);
  Conv3dMppSpecialization specialization;
  specialization.dtype = dtype_name;
  specialization.kernel_depth = params.kD;
  specialization.kernel_height = params.conv2d.kH;
  specialization.kernel_width = params.conv2d.kW;
  specialization.stride_depth = params.sD;
  specialization.stride_height = params.conv2d.sH;
  specialization.stride_width = params.conv2d.sW;
  specialization.dilation_depth = params.dD;
  specialization.dilation_height = params.conv2d.dH;
  specialization.dilation_width = params.conv2d.dW;
  specialization.src_channels = input_channels_per_group <= 64 ? static_cast<int>(input_channels_per_group) : -1;
  specialization.src_width = static_cast<int>(std::max<int64_t>(input_width, 16384));
  specialization.src_height = static_cast<int>(std::max<int64_t>(input_height, 16384));
  specialization.has_bias = bias_defined;
  specialization.out_ncdhw = out_ncdhw;
  specialization.grouped = groups > 1;

  const auto& conv2d = params.conv2d;
  if (use_mpp) {
    const auto tile = select_conv3d_mpp_tile(params, groups);
    if (auto pipeline = conv3d_mpp_pipeline(specialization, tile)) {
      const auto output_channel_tiles =
          c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(tile.output_channels));
      const auto width_tiles = c10::metal::ceil_div(int64_t(conv2d.outW), int64_t(tile.output_width));
      const auto height_tiles = c10::metal::ceil_div(int64_t(conv2d.outH), int64_t(tile.output_height));
      const auto threadgroups = MTLSizeMake(static_cast<NSUInteger>(output_channel_tiles * groups),
                                            static_cast<NSUInteger>(width_tiles),
                                            static_cast<NSUInteger>(height_tiles) * conv2d.N * params.outD);
      const auto threads_per_threadgroup = MTLSizeMake(tile.simdgroups * c10::metal::simdgroup_size, 1, 1);
      conv_metal_launch(pipeline, "conv3d_mpp", {activation, weights}, [&](id<MTLComputeCommandEncoder> encoder) {
        mtl_setArgs(encoder, activation, weights, output_t, params, bias ? *bias : activation);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
      });
      return;
    }
  }

  const auto tile = select_conv3d_simd_tile(output_height, output_width, use_long_index);
  const auto pipeline = tile.pipeline_state(lib, dtype_name, use_long_index);
  const auto output_channel_tiles = c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(tile.block_columns));
  const auto output_position_tiles =
      c10::metal::ceil_div(int64_t(conv2d.outH) * conv2d.outW, static_cast<int64_t>(tile.block_rows));
  const auto threadgroups = MTLSizeMake(static_cast<NSUInteger>(output_channel_tiles * groups),
                                        static_cast<NSUInteger>(output_position_tiles),
                                        static_cast<NSUInteger>(conv2d.N) * params.outD);
  const auto threads_per_threadgroup = MTLSizeMake(
      static_cast<NSUInteger>(tile.simdgroups_rows) * tile.simdgroups_columns * c10::metal::simdgroup_size, 1, 1);
  conv_metal_launch(pipeline, "conv3d_simd", {activation, weights}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, activation, weights, output_t, params, bias ? *bias : activation);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
  });
}

static bool conv2d_fits_int32(IntArrayRef sizes) {
  uint64_t elements;
  return !c10::safe_multiplies_u64(sizes, &elements) && elements <= std::numeric_limits<int32_t>::max();
}

static void conv2d_simd_forward(const Tensor& input,
                                const Tensor& weight,
                                const std::optional<Tensor>& bias,
                                IntArrayRef padding,
                                IntArrayRef stride,
                                IntArrayRef dilation,
                                int64_t groups,
                                const Tensor& output) {
  const auto fits_extent = [](IntArrayRef values) {
    return std::ranges::all_of(values, [](int64_t value) { return value <= std::numeric_limits<int32_t>::max(); });
  };
  TORCH_CHECK_NOT_IMPLEMENTED(fits_extent(input.sizes()) && fits_extent(weight.sizes()) &&
                                  fits_extent(output.sizes()) && fits_extent(padding) && fits_extent(stride) &&
                                  fits_extent(dilation),
                              "conv2d: individual dimensions, stride, padding and dilation must fit int32");
  conv3d_metal_forward(input.unsqueeze(2),
                       weight.unsqueeze(2),
                       bias,
                       {0, padding[0], padding[1]},
                       {1, stride[0], stride[1]},
                       {1, dilation[0], dilation[1]},
                       groups,
                       output.unsqueeze(2),
                       false);
}

// A 1x1x1 stride-1 no-pad conv is a per-voxel GEMM over channels; both the
// direct kernel and im2col (which trivially matches the patch-embed gate here)
// lose to a plain matmul, so peel it off first.
static bool conv3d_is_pointwise(const Tensor& weight, IntArrayRef stride, IntArrayRef padding, int64_t groups) {
  return groups == 1 && weight.size(2) == 1 && weight.size(3) == 1 && weight.size(4) == 1 && stride[0] == 1 &&
      stride[1] == 1 && stride[2] == 1 && padding[0] == 0 && padding[1] == 0 && padding[2] == 0;
}

static void conv3d_pointwise_matmul(const Tensor& input,
                                    const Tensor& weight,
                                    const std::optional<Tensor>& bias_opt,
                                    const Tensor& output) {
  const int64_t batch_size = input.size(0);
  const int64_t input_channels = input.size(1);
  const int64_t output_channels = weight.size(0);
  const int64_t input_depth = input.size(2);
  const int64_t input_height = input.size(3);
  const int64_t input_width = input.size(4);
  const int64_t spatial_size = input_depth * input_height * input_width;
  // addmm fuses bias into each batch's output-channels x spatial-size matrix
  // in the GEMM epilogue (one kernel, no separate bias pass), so it beats a
  // batched baddbmm here. Written straight into the output buffer when possible.
  const auto input_matrix = input.contiguous().reshape({batch_size, input_channels, spatial_size});
  const auto weight_matrix = weight.reshape({output_channels, input_channels});
  const bool has_bias = bias_opt && bias_opt->defined();
  std::optional<Tensor> converted_bias;
  if (has_bias) {
    const auto& bias = *bias_opt;
    converted_bias = bias.scalar_type() == output.scalar_type() ? bias : bias.to(output.scalar_type());
  }
  if (output.is_contiguous()) {
    auto output_matrix = output.view({batch_size, output_channels, spatial_size});
    const auto bias_matrix = has_bias ? converted_bias->reshape({output_channels, 1}) : Tensor();
    for (const auto batch : c10::irange(batch_size)) {
      auto batch_output = output_matrix.select(0, batch);
      has_bias ? at::addmm_out(batch_output, bias_matrix, weight_matrix, input_matrix.select(0, batch))
               : at::mm_out(batch_output, weight_matrix, input_matrix.select(0, batch));
    }
  } else {
    // Channels-last output: matmul into a fresh buffer, then permute-copy.
    auto result = weight_matrix.unsqueeze(0).expand({batch_size, output_channels, input_channels}).bmm(input_matrix);
    if (has_bias) {
      result = result.add(converted_bias->reshape({1, output_channels, 1}));
    }
    output.copy_(result.reshape({batch_size, output_channels, input_depth, input_height, input_width}));
  }
}

// conv as unfold(im2col) + matmul; weight OIDHW flattens to output channels x reduction size.
static void conv3d_im2col_matmul(const Tensor& input,
                                 const Tensor& weight,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef stride,
                                 IntArrayRef padding,
                                 const Tensor& output) {
  const int64_t batch_size = input.size(0);
  const int64_t input_channels = input.size(1);
  const int64_t output_channels = weight.size(0);
  const int64_t kernel_depth = weight.size(2);
  const int64_t kernel_height = weight.size(3);
  const int64_t kernel_width = weight.size(4);
  const auto padded_input =
      at::constant_pad_nd(input.contiguous(), {padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]});
  const auto patches = padded_input.unfold(2, kernel_depth, stride[0])
                           .unfold(3, kernel_height, stride[1])
                           .unfold(4, kernel_width, stride[2]);
  const int64_t output_depth = patches.size(2);
  const int64_t output_height = patches.size(3);
  const int64_t output_width = patches.size(4);
  const int64_t output_positions = batch_size * output_depth * output_height * output_width;
  const int64_t reduction_size = input_channels * kernel_depth * kernel_height * kernel_width;
  const auto columns = patches.permute({0, 2, 3, 4, 1, 5, 6, 7}).reshape({output_positions, reduction_size});
  auto result = columns.matmul(weight.contiguous().reshape({output_channels, reduction_size}).t());
  if (bias_opt && bias_opt->defined()) {
    result =
        result.add(bias_opt->scalar_type() == result.scalar_type() ? *bias_opt : bias_opt->to(result.scalar_type()));
  }
  output.copy_(result.reshape({batch_size, output_depth, output_height, output_width, output_channels})
                   .permute({0, 4, 1, 2, 3}));
}

enum class Conv1dKernel {
  // kConv1dDepthwiseOutputsPerThread consecutive outputs per thread.
  Depthwise,
  // One matmul2d per tap over the NCL input.
  MppNcl,
  // One matmul2d per tap over a zero-padded NLC input.
  MppNlc,
  // One matmul2d over all taps of a zero-padded NLC input.
  MppNlcMerged,
  Im2colMatmul,
  // conv3d_metal_forward with unit depth and height.
  Conv3d,
};

static Conv1dKernel conv1d_pick_kernel(const Tensor& input,
                                       const Tensor& weight,
                                       const Tensor& output,
                                       int64_t padding,
                                       int64_t conv_stride,
                                       int64_t dilation,
                                       int64_t groups) {
  using namespace mps;
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  // matmul2d on M1/M2 misreads operands whose row stride exceeds this many elements.
  constexpr int64_t kPreApple9MppMaxStride = std::numeric_limits<uint16_t>::max();
  const auto channels = input.size(1);
  const auto length = input.size(3);
  const auto channels_per_group = weight.size(1);
  const auto kernel_size = weight.size(3);
  const auto out_channels = output.size(1);
  const auto out_length = output.size(3);
  // Conv2DParams sizes are int32, as are matmul2d offsets within a batch.
  if ((length + 2 * padding) * channels > kInt32Max || out_channels * out_length > kInt32Max) {
    return Conv1dKernel::Conv3d;
  }
  // Depthwise, or a single-channel conv whose output can't fill a 64x64 matmul2d tile.
  if (channels_per_group == 1 && (groups > 1 || std::min(out_channels, out_length) < 64)) {
    return Conv1dKernel::Depthwise;
  }
  // conv1d_mpp reduces over all of C_in * k inside each tile, so a short output with a large reduction leaves
  // most of the GPU idle, im2col + addmm splits the reduction. Measured cutoffs: the im2col copy pays off from a
  // 2048-element reduction and the tile shortage fades as L_out and N grow (N also adds addmm launches, hence N^2).
  const auto reduction = channels_per_group * kernel_size;
  if (groups == 1 && channels_per_group > 0 &&
      (out_length == 1 || (reduction >= 2048 && input.size(0) * input.size(0) * out_length * 32 <= reduction))) {
    return Conv1dKernel::Im2colMatmul;
  }
  // conv3d_metal_forward handles empty input channels.
  if (channels_per_group == 0 || !has_mpp()) {
    return Conv1dKernel::Conv3d;
  }
  // NLC is required for conv_stride > 1 and avoids a transpose for channels-last input.
  // Dilation > 1 or groups > 1 leave gaps between taps. merging into one matmul would need im2col.
  // With both 1, windows are contiguous and one matmul covers all taps, C_out >= 8 * C_in is the measured point
  // where that saves more than the NCL to NLC transpose costs.
  const bool nlc = conv_stride > 1 || input.is_contiguous(MemoryFormat::ChannelsLast) ||
      (dilation == 1 && groups == 1 && out_channels >= 8 * channels);
  const bool merged = nlc && dilation == 1 && groups == 1;
  // Largest operand row stride: input (NCL row or NLC position step) versus weight row.
  const auto operand_stride =
      std::max(nlc ? conv_stride * channels : length + padding, (merged ? kernel_size : 1) * channels_per_group);
  if (!is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS) && operand_stride > kPreApple9MppMaxStride) {
    return Conv1dKernel::Conv3d;
  }
  return merged ? Conv1dKernel::MppNlcMerged : nlc ? Conv1dKernel::MppNlc : Conv1dKernel::MppNcl;
}

static void conv1d_metal_forward(const Tensor& input_t,
                                 const Tensor& weight_t,
                                 const std::optional<Tensor>& bias_opt,
                                 int64_t padding,
                                 int64_t stride,
                                 int64_t dilation,
                                 int64_t groups,
                                 const Tensor& output_t,
                                 Conv1dKernel kernel) {
  using namespace mps;
  const int64_t length = input_t.size(3);
  const int64_t kernel_size = weight_t.size(3);
  const auto padded_nlc_input = [&] {
    return padding > 0 ? at::constant_pad_nd(input_t.squeeze(2).transpose(1, 2), {0, 0, padding, padding})
                       : conv3d_to_ndhwc(input_t.unsqueeze(2));
  };
  const auto tap_major_weight = [&] {
    const auto tap_major = weight_t.permute({3, 0, 1, 2});
    return tap_major.is_contiguous() ? tap_major : conv3d_weights_to_dhwio(weight_t.transpose(0, 1).unsqueeze(2));
  };
  const auto dtype = scalarToMetalTypeString(input_t);
  const auto out_layout = output_t.is_contiguous() ? "ncl" : "nlc";
  Tensor input, weight;
  std::optional<Tensor> head;
  auto input_length = length;
  std::string kernel_name;
  const auto out_channels_per_group = output_t.size(1) / groups;
  // The smallest tile that covers C_out per group, a 64-wide tile wastes most of the matmul on C_out <= 32.
  const auto small_tile = out_channels_per_group <= 16 ? 16 : 32;
  const auto out_tile = kernel == Conv1dKernel::MppNcl && out_channels_per_group <= 32 ? small_tile : 64;
  // Measured crossovers: a 16- or 32-channel tile is too little work to split 4 ways and from a 1536-element
  // reduction (24 K steps of 64) 2 simdgroups already keep a tile busy while 4 only add synchronization.
  const int simdgroups = out_tile < 64 || weight_t.size(1) * kernel_size >= 1536 ? 2 : 4;
  switch (kernel) {
    case Conv1dKernel::Depthwise:
      input = input_t.contiguous();
      weight = weight_t.contiguous();
      kernel_name = fmt::format("conv1d_depthwise_{}_{}", out_layout, dtype);
      break;
    case Conv1dKernel::MppNcl:
      input = input_t.contiguous();
      weight = tap_major_weight();
      if (padding > 0) {
        // Covers every 64-position tile that starts inside the left padding plus the taps reaching past it.
        const auto head_length =
            std::min(length + padding, c10::metal::ceil_div(padding, int64_t(64)) * 64 + (kernel_size - 1) * dilation);
        head = at::constant_pad_nd(input.narrow(3, 0, head_length - padding), {padding, 0});
      }
      kernel_name = fmt::format("conv1d_mpp_ncl_{}_b{}_s{}_{}", out_layout, out_tile, simdgroups, dtype);
      break;
    case Conv1dKernel::MppNlc:
      input = padded_nlc_input();
      input_length = length + 2 * padding;
      weight = tap_major_weight();
      kernel_name = fmt::format("conv1d_mpp_nlc_{}_b64_s{}_{}", out_layout, simdgroups, dtype);
      break;
    case Conv1dKernel::MppNlcMerged:
      input = padded_nlc_input();
      input_length = length + 2 * padding;
      weight = weight_t.squeeze(2).transpose(1, 2).contiguous();
      kernel_name = fmt::format("conv1d_mpp_nlc_{}_b64_s{}_{}", out_layout, simdgroups, dtype);
      break;
    case Conv1dKernel::Im2colMatmul: {
      const auto x = padding > 0 ? at::constant_pad_nd(input_t.select(2, 0), {padding, padding}) : input_t.select(2, 0);
      const auto batch = x.size(0);
      const auto out_length = output_t.size(3);
      const auto st = x.strides();
      const auto windows =
          x.as_strided({batch, out_length, x.size(1), kernel_size}, {st[0], stride * st[2], st[1], dilation * st[2]});
      const auto weights = weight_t.reshape({weight_t.size(0), -1});
      const bool has_bias = bias_opt && bias_opt->defined();
      const auto bias = has_bias ? bias_opt->to(input_t.scalar_type()) : Tensor();
      if (out_length == 1 || !output_t.is_contiguous()) {
        auto rows = output_t.select(2, 0).transpose(1, 2).view({-1, weights.size(0)});
        const auto columns = windows.reshape({rows.size(0), -1});
        has_bias ? at::addmm_out(rows, bias, columns, weights.t()) : at::mm_out(rows, columns, weights.t());
        return;
      }
      const auto columns = stride > 1 ? windows.flatten(2).mT() : windows.permute({0, 2, 3, 1}).flatten(1, 2);
      for (const auto n : c10::irange(batch)) {
        auto out = output_t.select(2, 0).select(0, n);
        has_bias ? at::addmm_out(out, bias.view({-1, 1}), weights, columns[n]) : at::mm_out(out, weights, columns[n]);
      }
      return;
    }
    case Conv1dKernel::Conv3d:
      conv3d_metal_forward(input_t.unsqueeze(2),
                           weight_t.unsqueeze(2),
                           bias_opt,
                           {0, 0, padding},
                           {1, 1, stride},
                           {1, 1, dilation},
                           groups,
                           output_t.unsqueeze(2));
      return;
  }
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto bias = has_bias ? bias_opt->to(input_t.scalar_type()).contiguous() : input;
  const Conv2DParams params{
      .C_in = static_cast<int32_t>(input_t.size(1)),
      .C_out = static_cast<int32_t>(output_t.size(1)),
      .W = static_cast<int32_t>(input_length),
      .outW = static_cast<int32_t>(output_t.size(3)),
      .kW = static_cast<int32_t>(kernel_size),
      .sW = static_cast<int32_t>(stride),
      .padW = static_cast<int32_t>(padding),
      .dW = static_cast<int32_t>(dilation),
      .C_in_per_group = static_cast<int32_t>(weight_t.size(1)),
      .C_out_per_group = static_cast<int32_t>(output_t.size(1) / groups),
      .has_bias = has_bias,
  };
  auto pipeline = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv1d", {input, weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, input, weight, output_t, params, bias, head);
      if (kernel == Conv1dKernel::Depthwise) {
        const auto threads = c10::metal::ceil_div(params.outW, kConv1dDepthwiseOutputsPerThread);
        // 256-wide threadgroups, like the other flat kernels in this file.
        [encoder dispatchThreads:MTLSizeMake(threads, params.C_out, input_t.size(0))
            threadsPerThreadgroup:MTLSizeMake(std::min(threads, 256), 1, 1)];
      } else {
        [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div(params.C_out_per_group, out_tile) * groups,
                                                  c10::metal::ceil_div(params.outW, 64),
                                                  input_t.size(0))
                threadsPerThreadgroup:MTLSizeMake(simdgroups * c10::metal::simdgroup_size, 1, 1)];
      }
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
}

static Conv2DParams conv2d_params(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& output,
                                  IntArrayRef padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  bool has_bias) {
  return {
      .N = c10::checked_convert<int32_t>(input.size(0), "int32_t"),
      .C_in = c10::checked_convert<int32_t>(input.size(1), "int32_t"),
      .C_out = c10::checked_convert<int32_t>(output.size(1), "int32_t"),
      .H = c10::checked_convert<int32_t>(input.size(2), "int32_t"),
      .W = c10::checked_convert<int32_t>(input.size(3), "int32_t"),
      .outH = c10::checked_convert<int32_t>(output.size(2), "int32_t"),
      .outW = c10::checked_convert<int32_t>(output.size(3), "int32_t"),
      .kH = c10::checked_convert<int32_t>(weight.size(2), "int32_t"),
      .kW = c10::checked_convert<int32_t>(weight.size(3), "int32_t"),
      .sH = c10::checked_convert<int32_t>(stride[0], "int32_t"),
      .sW = c10::checked_convert<int32_t>(stride[1], "int32_t"),
      .padH = c10::checked_convert<int32_t>(padding[0], "int32_t"),
      .padW = c10::checked_convert<int32_t>(padding[1], "int32_t"),
      .dH = c10::checked_convert<int32_t>(dilation[0], "int32_t"),
      .dW = c10::checked_convert<int32_t>(dilation[1], "int32_t"),
      .C_in_per_group = c10::checked_convert<int32_t>(weight.size(1), "int32_t"),
      .C_out_per_group = c10::checked_convert<int32_t>(output.size(1) / (input.size(1) / weight.size(1)), "int32_t"),
      .has_bias = has_bias};
}

static void conv2d_depthwise_forward(const Tensor& input_t,
                                     const Tensor& weight_t,
                                     const std::optional<Tensor>& bias_opt,
                                     const Conv2DParams& params,
                                     const Tensor& output_t) {
  using namespace mps;
  const auto input = input_t.contiguous();
  const auto weight = weight_t.contiguous();
  const auto output = output_t.is_contiguous() ? output_t : at::empty_like(output_t, MemoryFormat::Contiguous);
  // Registered fixed-tap specializations: square 3x3/5x5/7x7 with stride 1 or 2.
  const bool square = params.kH == params.kW && params.sH == params.sW && params.dH == 1 && params.dW == 1;
  const bool fast = square && (params.kW == 3 || params.kW == 5 || params.kW == 7) && params.sW <= 2;
  const int64_t rw = fast ? conv2d_depthwise_columns(params.kW) : kConv2dDepthwiseColumns;
  const bool full = fast && params.kW == 3 && params.sW == 1 && params.outW % kConv2dDepthwiseColumns == 0 &&
      input.scalar_type() == kBFloat16;
  const auto bias = params.has_bias ? bias_opt->to(input.scalar_type()).contiguous() : input;
  const auto name = fmt::format(
      "conv2d_depthwise_{}_{}_{}_{}", fast ? params.kW : 0, fast ? params.sW : 0, scalarToMetalTypeString(input), full);
  auto pipeline = lib.getPipelineStateForFunc(name);
  const auto nx = c10::metal::ceil_div(int64_t(params.outW), rw);
  const auto ny = c10::metal::ceil_div(int64_t(params.outH), int64_t(kConv2dDepthwiseRows));
  constexpr auto width = c10::metal::simdgroup_size;
  constexpr auto rows = kConv2dFlatThreads / width;
  conv_metal_launch(pipeline, "conv2d_depthwise", {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, weight, output, params, bias);
    [encoder dispatchThreads:MTLSizeMake(nx, ny, int64_t(params.N) * params.C_in)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(nx, width), std::min<int64_t>(ny, rows), 1)];
  });
  if (!output.is_same(output_t)) {
    output_t.copy_(output);
  }
}

// Padded group-major [groups, N, H + 2 pad, W + 2 pad, C / groups] copy for
// grouped merged products. Channels-last input copies whole channel runs; NCHW
// input takes the strided pad, made contiguous because an unpadded pad keeps
// the permuted strides.
static Tensor conv2d_group_major(const Tensor& input, int64_t groups, IntArrayRef pad) {
  using namespace mps;
  const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3), Cg = C / groups;
  if (input.is_contiguous()) {
    auto grouped = input.unflatten(1, {groups, Cg}).permute({1, 0, 3, 4, 2});
    return at::constant_pad_nd(grouped, {0, 0, pad[1], pad[1], pad[0], pad[0]}).contiguous();
  }
  const int64_t padded_h = H + 2 * pad[0], padded_w = W + 2 * pad[1];
  auto output = at::empty({groups, N, padded_h, padded_w, Cg}, input.options());
  const Conv2dGroupMajorParams params{.batch = c10::checked_convert<int32_t>(N, "int32_t"),
                                      .channels = c10::checked_convert<int32_t>(C, "int32_t"),
                                      .group_channels = c10::checked_convert<int32_t>(Cg, "int32_t"),
                                      .height = c10::checked_convert<int32_t>(H, "int32_t"),
                                      .width = c10::checked_convert<int32_t>(W, "int32_t"),
                                      .pad_height = c10::checked_convert<int32_t>(pad[0], "int32_t"),
                                      .pad_width = c10::checked_convert<int32_t>(pad[1], "int32_t")};
  const int64_t row_threads = padded_w * c10::metal::ceil_div(Cg, int64_t(kConv2dGroupMajorChannelsPerThread));
  auto pipeline = lib.getPipelineStateForFunc("conv2d_group_major_" + scalarToMetalTypeString(input));
  conv_metal_launch(pipeline, "conv2d_group_major", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, output, params);
    [encoder dispatchThreads:MTLSizeMake(row_threads, padded_h, groups * N)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(row_threads, kConv2dFlatThreads), 1, 1)];
  });
  return output;
}

static Tensor conv2d_im2col(const Tensor& input, const Conv2DParams& params) {
  using namespace mps;
  const int64_t positions = int64_t(params.outH) * params.outW, taps = int64_t(params.kH) * params.kW;
  auto columns = at::empty({params.N, params.C_in * taps, params.outH, params.outW}, input.options());
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv2d_im2col_{}", scalarToMetalTypeString(input)));
  const auto threads = c10::metal::ceil_div(positions, int64_t(kConv2dIm2colPositionsPerThread));
  conv_metal_launch(pipeline, "conv2d_im2col", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, columns, params, input.strides());
    [encoder dispatchThreads:MTLSizeMake(threads, params.C_in * taps, params.N)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(threads, kConv2dFlatThreads), 1, 1)];
  });
  return columns;
}

static Tensor conv2d_deinterleave(const Tensor& input) {
  using namespace mps;
  const int64_t W = input.size(3), rows = input.numel() / W;
  auto output = at::empty({input.size(0), input.size(1), input.size(2), W + 1}, input.options());
  const std::array<int32_t, 2> dims{c10::checked_convert<int32_t>(W, "int32_t"),
                                    c10::checked_convert<int32_t>((W + 1) / 2 + 1, "int32_t")};
  auto pipeline = lib.getPipelineStateForFunc("conv2d_deinterleave_" + scalarToMetalTypeString(input));
  // conv2d_deinterleave writes four consecutive row elements per thread.
  const auto columns = c10::metal::ceil_div(W + 1, int64_t(4));
  conv_metal_launch(pipeline, "conv2d_deinterleave", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, output, dims);
    [encoder dispatchThreads:MTLSizeMake(columns, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(columns, kConv2dFlatThreads), 1, 1)];
  });
  return output;
}

// conv2d_mpp over the input prepared per plan.input; split-K partials are summed by conv2d_split_reduce.
static void conv2d_mpp_forward(Tensor input,
                               const Tensor& weight_t,
                               const std::optional<Tensor>& bias_opt,
                               Conv2DParams params,
                               const mps::Conv2dPlan& plan,
                               int64_t groups,
                               const Tensor& output_t) {
  using namespace mps;
  using Input = Conv2dMppInput;
  const int64_t kH = params.kH, kW = params.kW, Cg = params.C_in_per_group, Og = params.C_out_per_group;
  const int64_t positions = int64_t(params.outH) * params.outW;
  const int64_t padded_h = int64_t(params.H) + 2 * int64_t(params.padH);
  const int64_t padded_w = int64_t(params.W) + 2 * int64_t(params.padW);
  const std::array<int64_t, 2> padding{params.padH, params.padW};
  const auto bias = params.has_bias ? bias_opt->to(input.scalar_type()).contiguous() : input;
  bool in_nhwc = !input.is_contiguous();
  const bool copy_output = !output_t.is_contiguous() && input.scalar_type() == kFloat && plan.input == Input::Columns &&
      !conv2d_device().apple10_or_newer;
  const bool out_nhwc = !output_t.is_contiguous() && !copy_output;
  // Check the materialized geometry before its copy kernels or MPP views use int32 offsets.
  std::array<int64_t, 3> input_shape{params.C_in, params.H, params.W};
  auto reduction = Cg;
  auto input_step = int64_t(params.sW);
  auto packed_input = in_nhwc;
  switch (plan.input) {
    case Input::Pointwise:
      input_shape = {params.C_in, 1, positions};
      input_step = 1;
      break;
    case Input::Im2col:
    case Input::Columns:
      reduction *= kH * kW;
      input_shape = {groups * reduction, params.outH, params.outW};
      input_step = 1;
      packed_input = false;
      break;
    case Input::Unfolded:
      reduction *= kH * kW;
      input_shape = {kH * params.C_in, params.outH, padded_w};
      packed_input = true;
      break;
    case Input::Merged:
      reduction *= kW;
      input_shape = {params.C_in, padded_h, padded_w};
      packed_input = true;
      break;
    case Input::Direct:
      if (plan.parity) {
        ++input_shape[2];
        input_step = 1;
      } else if (params.sW > 1) {
        packed_input = true;
      }
      if (plan.flat) {
        input_shape[1] = padded_h;
        input_shape[2] = padded_w;
      }
      break;
  }
  const bool linear = plan.input == Input::Pointwise || plan.input == Input::Columns;
  const auto output_height = linear ? int64_t(1) : params.outH;
  const auto output_width = linear ? positions : params.outW;
  const auto output_pitch =
      plan.flat && (plan.input == Input::Direct || plan.input == Input::Merged) ? input_shape[2] : output_width;
  const bool fits_int32 =
      conv2d_fits_int32(input_shape) && conv2d_fits_int32({params.C_out, output_height, output_pitch});
  // matmul2d on pre-Apple9 GPUs misreads operand row strides above uint16.
  constexpr int64_t max_legacy_mpp_stride = std::numeric_limits<uint16_t>::max();
  const bool strides_supported = conv2d_device().supports_large_mpp_strides ||
      (fits_int32 &&
       (packed_input ? input_step * input_shape[0] : input_shape[1] * input_shape[2]) <= max_legacy_mpp_stride &&
       reduction <= max_legacy_mpp_stride &&
       (out_nhwc ? int64_t(params.C_out) : output_height * output_pitch) <= max_legacy_mpp_stride);
  if (!fits_int32 || !strides_supported) {
    conv2d_simd_forward(
        input, weight_t, bias_opt, padding, {params.sH, params.sW}, {params.dH, params.dW}, groups, output_t);
    return;
  }
  const auto output = copy_output ? at::empty(output_t.sizes(), output_t.options()) : output_t;
  Tensor weight;
  int32_t K = params.C_in_per_group, Wp = 0;
  switch (plan.input) {
    case Input::Pointwise:
      if (params.sH > 1 || params.sW > 1) {
        input = input.slice(2, 0, std::nullopt, params.sH).slice(3, 0, std::nullopt, params.sW);
        input = input.contiguous(in_nhwc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
      }
      params.H = params.outH = 1;
      params.W = params.outW = c10::checked_convert<int32_t>(positions, "int32_t");
      params.sH = params.sW = 1;
      weight = weight_t.contiguous();
      break;
    case Input::Im2col:
    case Input::Columns:
      if (plan.input == Input::Columns && input.scalar_type() == kFloat && (out_nhwc || copy_output)) {
        auto columns = at::empty({params.N, groups * reduction, positions}, input.options());
        at::im2col_out(columns, input, {kH, kW}, {params.dH, params.dW}, padding, {params.sH, params.sW});
        input = std::move(columns);
      } else {
        input = conv2d_im2col(input, params);
      }
      in_nhwc = false;
      K = c10::checked_convert<int32_t>(Cg * kH * kW, "int32_t");
      params.C_in = c10::checked_convert<int32_t>(groups * K, "int32_t");
      params.C_in_per_group = K;
      params.kH = params.kW = params.sH = params.sW = params.dH = params.dW = 1;
      params.padH = params.padW = 0;
      if (plan.input == Input::Columns) {
        params.H = params.outH = 1;
        params.W = params.outW = c10::checked_convert<int32_t>(positions, "int32_t");
      } else {
        params.H = params.outH;
        params.W = Wp = params.outW;
      }
      weight = weight_t.contiguous();
      break;
    case Input::Unfolded: {
      auto rows =
          at::empty({params.N, kH * params.C_in, params.outH, padded_w}, input.options(), MemoryFormat::ChannelsLast);
      nchw_to_nhwc(input.contiguous(), rows, params.C_in, params.H, params.W, padding, kH, params.sH, params.dH);
      input = rows;
      in_nhwc = true;
      K = c10::checked_convert<int32_t>(kW * kH * Cg, "int32_t");
      params.C_in = c10::checked_convert<int32_t>(kH * Cg, "int32_t");
      params.C_in_per_group = K;
      params.H = params.outH;
      params.W = c10::checked_convert<int32_t>(padded_w, "int32_t");
      params.kH = params.kW = params.sH = params.dH = 1;
      params.padH = params.padW = 0;
      weight = conv3d_weights_to_dhwio(
          weight_t.as_strided({Cg, 1, weight_t.size(0), kW, kH},
                              {weight_t.stride(1), 0, weight_t.stride(0), weight_t.stride(3), weight_t.stride(2)}));
      break;
    }
    case Input::Merged: {
      if (groups > 1) {
        // A group's kW taps stay one contiguous channel run.
        input = conv2d_group_major(input, groups, padding);
        params.C_in = c10::checked_convert<int32_t>(Cg, "int32_t");
      } else if (in_nhwc) {
        input = at::constant_pad_nd(input, {padding[1], padding[1], padding[0], padding[0]});
      } else {
        auto padded =
            at::empty({params.N, params.C_in, padded_h, padded_w}, input.options(), MemoryFormat::ChannelsLast);
        nchw_to_nhwc(input, padded, params.C_in, params.H, params.W, padding);
        input = padded;
      }
      in_nhwc = true;
      K = c10::checked_convert<int32_t>(kW * Cg, "int32_t");
      params.C_in_per_group = K;
      params.H = c10::checked_convert<int32_t>(padded_h, "int32_t");
      params.W = c10::checked_convert<int32_t>(padded_w, "int32_t");
      params.kW = 1;
      params.padH = params.padW = 0;
      weight = conv3d_weights_to_dhwio(
          weight_t.as_strided({Cg, 1, kH, weight_t.size(0), kW},
                              {weight_t.stride(1), 0, weight_t.stride(2), weight_t.stride(0), weight_t.stride(3)}));
      break;
    }
    case Input::Direct:
      if (plan.parity) {
        input = conv2d_deinterleave(input);
        params.W = c10::checked_convert<int32_t>(input.size(3), "int32_t");
        params.sW = 1;
      } else if (params.sW > 1 && !in_nhwc) {
        input = conv3d_to_ndhwc(input.unsqueeze(2)).squeeze(2);
        in_nhwc = true;
      }
      if (plan.flat && (params.padH > 0 || params.padW > 0)) {
        input = at::constant_pad_nd(input, {padding[1], padding[1], padding[0], padding[0]});
        params.H = c10::checked_convert<int32_t>(input.size(2), "int32_t");
        params.W = c10::checked_convert<int32_t>(input.size(3), "int32_t");
        params.padH = params.padW = 0;
      }
      weight = conv3d_weights_to_dhwio(weight_t.transpose(0, 1).unsqueeze(2),
                                       !plan.flat && (params.padH > 0 || params.padW > 0));
      break;
  }
  if (plan.flat && (plan.input == Input::Direct || plan.input == Input::Merged)) {
    Wp = params.W;
  }
  const auto group_stride = plan.input == Input::Merged && groups > 1
      ? int64_t(params.N) * params.H * params.W * params.C_in
      : int64_t(K) * (in_nhwc ? 1 : int64_t(params.H) * params.W);
  Conv2dMppParams mpp{.conv = params,
                      .K = K,
                      .Wp = Wp,
                      .parity = plan.parity ? c10::checked_convert<int32_t>(input.size(3) / 2 + 1, "int32_t") : 0,
                      .group_stride = c10::checked_convert<int32_t>(group_stride, "int32_t")};
  const auto tile = select_conv2d_mpp_tile(mpp, plan, groups, conv2d_device());
  mpp.K_split = c10::checked_convert<int32_t>(tile.k_split, "int32_t");
  mpp.splits = c10::checked_convert<int32_t>(tile.splits, "int32_t");
  // Edge columns replace the flattened rows of a merged copy.
  if (tile.edge_columns > 0) {
    Wp = mpp.Wp = 0;
    mpp.edge_columns = c10::checked_convert<int32_t>(tile.edge_columns, "int32_t");
  }
  const int64_t pitch = Wp > 0 ? Wp : params.outW;
  const bool partials = tile.splits > 1 || pitch != params.outW;
  const bool fixed3x3 = tile.variant == Conv2dMppVariant::Fixed3x3;
  // Shallow fixed 3x3 products compute output column 0 separately instead of shifting.
  const bool column_threadgroups = fixed3x3 && params.C_in <= kConv2dFixed3ColumnMaxInputs;
  const bool shifts = !column_threadgroups && Wp == 0 && params.padW > 0 && (!mpp.parity || params.padW > 1);
  const bool shift_scratch = shifts && !partials && input.scalar_type() != kFloat;
  // Only boundary tiles shift accumulators. Keep their scratch in float32, and
  // reuse the output buffer for float32 inputs or the split-K buffer when present.
  mpp.shift_width = shift_scratch
      ? c10::checked_convert<int32_t>(
            std::min<int64_t>(
                params.outW,
                c10::metal::round_up(c10::metal::ceil_div(int64_t(params.padW), int64_t(params.sW)), tile.columns)),
            "int32_t")
      : params.outW;
  const auto scratch_pitch = partials ? pitch : mpp.shift_width;
  const auto partial = partials || shift_scratch
      ? at::empty({tile.splits * params.N * params.C_out * params.outH * scratch_pitch}, output.options().dtype(kFloat))
      : output;
  const char* variant = fixed3x3 ? "k3_" : tile.variant == Conv2dMppVariant::Pointwise ? "pw1_" : "";
  const auto dtype = scalarToMetalTypeString(input);
  const auto name = fmt::format("conv2d_mpp_{}_{}_b{}x{}_s{}_{}{}",
                                in_nhwc ? "nhwc" : "nchw",
                                out_nhwc ? "nhwc" : "nchw",
                                tile.rows,
                                tile.columns,
                                tile.simdgroups,
                                variant,
                                dtype);
  auto pipeline = lib.getPipelineStateForFunc(name);
  conv_metal_launch(pipeline, "conv2d_mpp", {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, weight, output, mpp, bias, partial);
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      c10::metal::ceil_div(Og, tile.rows) * groups, tile.tiles, params.N * tile.splits)
            threadsPerThreadgroup:MTLSizeMake(tile.simdgroups * c10::metal::simdgroup_size, 1, 1)];
    if (partials) {
      const auto image = int64_t(params.C_out) * params.outH * params.outW;
      const Conv2dSplitParams reduction{.channels = params.C_out,
                                        .height = params.outH,
                                        .width = params.outW,
                                        .pitch = c10::checked_convert<int32_t>(pitch, "int32_t"),
                                        .splits = mpp.splits,
                                        .batch = params.N,
                                        .has_bias = params.has_bias,
                                        .channels_last = out_nhwc};
      [encoder setComputePipelineState:lib.getPipelineStateForFunc("conv2d_split_reduce_" + dtype)];
      mtl_setArgs(encoder, partial, bias, output, reduction);
      [encoder dispatchThreads:MTLSizeMake(image, params.N, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(image, kConv2dFlatThreads), 1, 1)];
    }
  });
  if (copy_output) {
    output_t.copy_(output);
  }
}

// Fused F(2x2, 3x3) Winograd (conv2d_winograd) over plan.rows output channels x
// plan.columns 2x2 tiles. 32-row and 16x32 tiles read weights transformed once
// per call; the other tiles transform them per chunk in a single dispatch.
static void conv2d_winograd_fused_forward(const Tensor& input_t,
                                          const Tensor& weight_t,
                                          const std::optional<Tensor>& bias_opt,
                                          const Conv2DParams& params,
                                          const mps::Conv2dPlan& plan,
                                          const Tensor& output) {
  using namespace mps;
  using c10::metal::ceil_div;
  constexpr int64_t threads = kConv2dWinogradSimdgroups * c10::metal::simdgroup_size;
  // F(2x2, 3x3) has 4x4 transformed coefficients per 2x2 output tile.
  constexpr int64_t coefficients = 16;
  // There is no NHWC-input/NCHW-output kernel. An output with one channel or
  // one pixel is contiguous in both layouts, so channels-last input writes it
  // as NHWC instead of being copied to NCHW.
  const bool input_nhwc = !input_t.is_contiguous() && input_t.is_contiguous(MemoryFormat::ChannelsLast);
  const bool out_nhwc = !output.is_contiguous() || (input_nhwc && output.is_contiguous(MemoryFormat::ChannelsLast));
  const auto input = out_nhwc ? input_t : input_t.contiguous();
  const bool in_nhwc = !input.is_contiguous();
  // OHWI weights are read in place; other strided weights are copied.
  const bool weight_channels_last = !weight_t.is_contiguous() && weight_t.is_contiguous(MemoryFormat::ChannelsLast);
  const auto weight = weight_channels_last ? weight_t : weight_t.contiguous();
  const auto bias = params.has_bias ? bias_opt->to(input.scalar_type()).contiguous() : input;
  const int64_t chunk = threads / plan.columns;
  const int64_t tiles_w = ceil_div(int64_t(params.outW), int64_t(2));
  const int64_t tiles = ceil_div(int64_t(params.outH), int64_t(2)) * tiles_w;
  const int64_t output_blocks = ceil_div(int64_t(params.C_out), plan.rows);
  const int64_t chunks = ceil_div(int64_t(params.C_in), chunk);
  const Conv2dWinogradParams winograd{.C_in = params.C_in,
                                      .C_out = params.C_out,
                                      .H = params.H,
                                      .W = params.W,
                                      .outH = params.outH,
                                      .outW = params.outW,
                                      .pad = params.padH,
                                      .tiles_w = c10::checked_convert<int32_t>(tiles_w, "int32_t"),
                                      .tiles = c10::checked_convert<int32_t>(tiles, "int32_t"),
                                      .c_blocks = c10::checked_convert<int32_t>(chunks, "int32_t"),
                                      .has_bias = params.has_bias,
                                      .weight_channels_last = weight_channels_last};
  const bool pretransformed = conv2d_winograd_pretransformed(plan.rows, plan.columns);
  constexpr int64_t block_rows = kConv2dWinogradPretransformedRows;
  const int64_t weight_blocks = ceil_div(int64_t(params.C_out), block_rows);
  const auto transformed = pretransformed
      ? at::empty({weight_blocks * chunks * coefficients * block_rows * chunk}, input.options())
      : weight;
  const auto dtype = scalarToMetalTypeString(input);
  const std::string_view layout = !out_nhwc ? "nchw" : in_nhwc ? "nhwc_nhwc" : "nchw_nhwc";
  const auto name = fmt::format(
      "conv2d_winograd_{}{}_{}x{}_{}", plan.output_loop ? "loop_" : "", layout, plan.rows, plan.columns, dtype);
  auto pipeline = lib.getPipelineStateForFunc(name);
  conv_metal_launch(pipeline, "conv2d_winograd", {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    if (pretransformed) {
      [encoder setComputePipelineState:lib.getPipelineStateForFunc(
                                           fmt::format("conv2d_winograd_weights_{}_{}", chunk, dtype))];
      mtl_setArgs(encoder, weight, transformed, winograd);
      [encoder dispatchThreads:MTLSizeMake(chunks * chunk, weight_blocks * block_rows, 1)
          threadsPerThreadgroup:MTLSizeMake(chunk, threads / chunk, 1)];
      [encoder setComputePipelineState:pipeline];
    }
    mtl_setArgs(encoder, input, weight, transformed, bias, output, winograd);
    [encoder
         dispatchThreadgroups:MTLSizeMake(plan.output_loop ? 1 : output_blocks, ceil_div(tiles, plan.columns), params.N)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  });
}

static void conv2d_winograd_forward(const Tensor& input_t,
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    const Tensor& output_t,
                                    int64_t tile_h) {
  using namespace mps;
  const bool in_nhwc = !input_t.is_contiguous() && input_t.is_contiguous(MemoryFormat::ChannelsLast);
  const auto input = input_t.contiguous(in_nhwc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
  const bool out_nhwc = !output_t.is_contiguous();
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto bias = has_bias ? bias_opt->to(input.scalar_type()).contiguous() : Tensor();
  const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3), O = output_t.size(1);
  const int64_t outH = output_t.size(2), outW = output_t.size(3);
  const int64_t TH = c10::metal::ceil_div(outH, tile_h), TW = (outW + 1) / 2, tiles = N * TH * TW;
  // F(tile_h x 2, 3x3) has (tile_h + 2) x 4 transformed coefficients.
  const int64_t coefficients = 4 * (tile_h + 2);
  // Round the scratch budget to standard MPP position tiles; the final chunk may be shorter.
  auto workspace_bytes = input.scalar_type() == kHalf ? conv2d_device().tuning.half_winograd_workspace_bytes
                                                      : Conv2dTuning::workspace_bytes;
  const int64_t chunk =
      std::min(tiles,
               std::max<int64_t>(kConv2dRowTile,
                                 workspace_bytes / (coefficients * std::max(C, O) * input.element_size()) /
                                     kConv2dRowTile * kConv2dRowTile));
  const auto& device = conv2d_device();
  // Keep the 16-row generic product when a 32-row tile adds channel padding.
  const bool half_product = tile_h == 2 && input.scalar_type() == kHalf && !device.apple10_or_newer &&
      c10::metal::round_up(O, int64_t(kConv2dMinTile)) == c10::metal::round_up(O, int64_t(kConv2dNarrowTile)) &&
      conv2d_fits_int32({coefficients * C, 1, chunk}) && conv2d_fits_int32({coefficients * O, 1, chunk}) &&
      (device.supports_large_mpp_strides || std::max(C, chunk) <= std::numeric_limits<uint16_t>::max());
  const bool direct_product = tile_h == 4 || half_product;
  const auto dtype = scalarToMetalTypeString(input);
  const auto weight = weight_t.contiguous();
  auto weights = at::empty({coefficients, O, C}, input.options());
  auto transformed = at::empty({1, coefficients * C, 1, chunk}, input.options());
  auto products = at::empty({1, coefficients * O, 1, chunk}, input.options());
  const std::array<int32_t, 2> wdims{c10::checked_convert<int32_t>(O, "int32_t"),
                                     c10::checked_convert<int32_t>(C, "int32_t")};
  const std::array<int32_t, 4> igeom{c10::checked_convert<int32_t>(TH, "int32_t"),
                                     c10::checked_convert<int32_t>(TW, "int32_t"),
                                     c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                     c10::checked_convert<int32_t>(padding[1], "int32_t")};
  const Conv2dWinogradRepairParams repair{.channels = c10::checked_convert<int32_t>(C, "int32_t"),
                                          .height = c10::checked_convert<int32_t>(H, "int32_t"),
                                          .width = c10::checked_convert<int32_t>(W, "int32_t"),
                                          .pad_height = c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                          .pad_width = c10::checked_convert<int32_t>(padding[1], "int32_t"),
                                          .channels_last = in_nhwc};
  auto wk = lib.getPipelineStateForFunc(fmt::format("winograd_weights_{}{}", tile_h == 4 ? "4_" : "", dtype));
  conv_metal_launch(wk, "winograd_weights", {weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, weight, weights, wdims);
    [encoder dispatchThreads:MTLSizeMake(C, O, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(C, kConv2dFlatThreads), 1, 1)];
  });
  for (int64_t t0 = 0; t0 < tiles; t0 += chunk) {
    const int32_t count = c10::checked_convert<int32_t>(std::min(chunk, tiles - t0), "int32_t"),
                  start = c10::checked_convert<int32_t>(t0, "int32_t");
    const auto v = transformed.as_strided({1, coefficients * C, 1, count}, {coefficients * C * count, count, count, 1});
    const auto m = products.as_strided({1, coefficients * O, 1, count}, {coefficients * O * count, count, count, 1});
    const std::array<int32_t, 4> idims{c10::checked_convert<int32_t>(C, "int32_t"),
                                       count,
                                       c10::checked_convert<int32_t>(H, "int32_t"),
                                       c10::checked_convert<int32_t>(W, "int32_t")};
    const std::array<int32_t, 4> odims{c10::checked_convert<int32_t>(O, "int32_t"),
                                       count,
                                       c10::checked_convert<int32_t>(outH, "int32_t"),
                                       c10::checked_convert<int32_t>(outW, "int32_t")};
    const std::array<int32_t, 4> ogeom{
        c10::checked_convert<int32_t>(TH, "int32_t"), c10::checked_convert<int32_t>(TW, "int32_t"), start, 0};
    const char* input_kind = tile_h == 4 ? "4" : in_nhwc ? "nhwc" : "nchw";
    auto ik = lib.getPipelineStateForFunc(fmt::format("winograd_input_{}_{}", input_kind, dtype));
    conv_metal_launch(ik, "winograd_input", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
      mtl_setArgs(encoder, input, v, idims, igeom, start);
      [encoder dispatchThreads:MTLSizeMake(count, C, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(count, kConv2dFlatThreads), 1, 1)];
      if (direct_product) {
        // Keep these tiles in sync with the pw2 registrations.
        constexpr int64_t simdgroups = 4;
        const int64_t rows = half_product ? kConv2dNarrowTile : kConv2dRowTile;
        const int64_t columns = half_product ? kConv2dRowTile : device.tuning.winograd_product_columns;
        const Conv2dMppParams gemm_params{.conv = {.N = c10::checked_convert<int32_t>(coefficients, "int32_t"),
                                                   .C_in = c10::checked_convert<int32_t>(C, "int32_t"),
                                                   .C_out = c10::checked_convert<int32_t>(O, "int32_t"),
                                                   .H = 1,
                                                   .W = count}};
        const auto grid =
            MTLSizeMake(c10::metal::ceil_div(O, rows), c10::metal::ceil_div(int64_t(count), columns), coefficients);
        const auto products = fmt::format("conv2d_mpp_nchw_nchw_b{}x{}_s{}_pw2_{}", rows, columns, simdgroups, dtype);
        [encoder setComputePipelineState:lib.getPipelineStateForFunc(products)];
        mtl_setArgs(encoder, v, weights, m, gemm_params, input, m);
        [encoder dispatchThreadgroups:grid
                threadsPerThreadgroup:MTLSizeMake(simdgroups * c10::metal::simdgroup_size, 1, 1)];
      }
    });
    if (!direct_product) {
      const auto grouped_weights = weights.view({coefficients * O, C, 1, 1});
      const Conv2dPlan plan{.kernel = Conv2dKernel::Mpp, .input = Conv2dMppInput::Pointwise};
      const auto params = conv2d_params(v, grouped_weights, m, {0, 0}, {1, 1}, {1, 1}, false);
      conv2d_mpp_forward(v, grouped_weights, std::nullopt, params, plan, coefficients, m);
    }
    const char* output_kind = tile_h == 4 ? "4" : out_nhwc ? "nhwc" : "nchw";
    auto ok = lib.getPipelineStateForFunc(fmt::format("winograd_output_{}_{}", output_kind, dtype));
    conv_metal_launch(ok, "winograd_output", {m}, [&](id<MTLComputeCommandEncoder> encoder) {
      mtl_setArgs(encoder,
                  m,
                  has_bias ? std::optional<Tensor>(bias) : std::nullopt,
                  output_t,
                  odims,
                  ogeom,
                  input,
                  weight,
                  repair);
      [encoder dispatchThreads:MTLSizeMake(count, O, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(count, kConv2dFlatThreads), 1, 1)];
    });
  }
}

// Execute the native conv2d plan, including input and weight preparation.
static void conv2d_metal_forward(const Tensor& input_t,
                                 const Tensor& weight_t,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef padding,
                                 IntArrayRef stride,
                                 IntArrayRef dilation,
                                 int64_t groups,
                                 const Tensor& output_t) {
  using namespace mps;
  using c10::metal::ceil_div;
  const bool in_nhwc = !input_t.is_contiguous() && input_t.is_contiguous(MemoryFormat::ChannelsLast);
  auto input = input_t.contiguous(in_nhwc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto params = conv2d_params(input, weight_t, output_t, padding, stride, dilation, has_bias);
  const auto plan = select_conv2d_kernel(params, input, output_t, groups, conv2d_device());
  const int64_t N = params.N, Cg = params.C_in_per_group, Og = params.C_out_per_group;
  const int64_t outH = params.outH, outW = params.outW, positions = outH * outW;
  const auto dtype = scalarToMetalTypeString(input);
  Tensor weight;
  std::string name;
  MTLSize grid = MTLSizeMake(0, 0, 0);
  switch (plan.kernel) {
    case Conv2dKernel::Winograd2x2:
    case Conv2dKernel::Winograd4x2:
      conv2d_winograd_forward(
          input, weight_t, bias_opt, padding, output_t, plan.kernel == Conv2dKernel::Winograd4x2 ? 4 : 2);
      return;
    case Conv2dKernel::Mpp:
      conv2d_mpp_forward(input, weight_t, bias_opt, params, plan, groups, output_t);
      return;
    case Conv2dKernel::Tiny: {
      if (plan.bounded_tiny) {
        name = fmt::format("conv2d_tiny_bounded_{}_{}_{}", Cg, params.kW, dtype);
        grid = MTLSizeMake(
            outH * ceil_div(outW, int64_t(kConv2dTinyPositionsPerThread)), Og / kConv2dTinyChannelsPerThread, N);
        if (weight_t.is_contiguous(MemoryFormat::ChannelsLast)) {
          weight = weight_t;
        }
        break;
      }
      const bool pointwise = params.kW == 1;
      const int64_t columns = (pointwise ? int64_t(params.H) * params.W : outW) / kConv2dTinyPositionsPerThread;
      const int64_t channel_blocks = Og / kConv2dTinyChannelsPerThread;
      name = fmt::format("conv2d_tiny_{}_{}_{}_{}", Cg, params.kW, params.sW, dtype);
      grid = MTLSizeMake(columns, pointwise ? channel_blocks : outH, pointwise ? N : N * channel_blocks);
      break;
    }
    case Conv2dKernel::SingleOutput: {
      name = fmt::format("conv2d_o1_{}_{}_{}_{}_{}",
                         plan.fixed_taps ? params.kW : 0,
                         plan.fixed_taps ? params.sW : 0,
                         plan.rows,
                         plan.columns,
                         dtype);
      const int64_t items = ceil_div(outW, plan.columns) * ceil_div(outH, plan.rows);
      int64_t threads = items;
      if (plan.channel_partitions > 1) {
        name += fmt::format("_cp{}", plan.channel_partitions);
        const int64_t spatial_lanes = c10::metal::simdgroup_size / plan.channel_partitions;
        threads = ceil_div(items, spatial_lanes) * c10::metal::simdgroup_size;
      }
      grid = MTLSizeMake(threads, N * params.C_out, 1);
      break;
    }
    case Conv2dKernel::WinogradFused:
      conv2d_winograd_fused_forward(input, weight_t, bias_opt, params, plan, output_t);
      return;
    case Conv2dKernel::PointwisePlane:
      name = conv2d_pointwise_fixed_channels(Cg) ? fmt::format("conv2d_pointwise_{}", Cg) : "conv2d_pointwise_dynamic";
      grid = MTLSizeMake(ceil_div(int64_t(params.H) * params.W, plan.columns), ceil_div(Og, plan.rows), 1);
      break;
    case Conv2dKernel::Narrow:
      // Match the transposed weight operand of the C128 specialization.
      weight = conv3d_weights_to_dhwio((Cg == 128 ? weight_t : weight_t.transpose(0, 1)).unsqueeze(2));
      name = fmt::format("conv2d_narrow_{}", Cg);
      grid = MTLSizeMake(1, ceil_div(positions, plan.columns), N);
      break;
    case Conv2dKernel::Patch: {
      const int64_t tiles = plan.flat ? ceil_div(positions, plan.columns) : outH * ceil_div(outW, plan.columns);
      name = fmt::format("conv2d_patch_{}_{}x{}_s{}_{}", params.kH, plan.rows, plan.columns, params.sH, dtype);
      if (input.scalar_type() == kHalf && plan.rows == kConv2dWideTile &&
          Cg <= Conv2dTuning::Patch::half_compact_channels) {
        name += "_c16";
      }
      grid = MTLSizeMake(ceil_div(Og, plan.rows), tiles, N);
      break;
    }
    case Conv2dKernel::Stride2:
      weight = conv3d_weights_to_dhwio(weight_t.transpose(0, 1).unsqueeze(2));
      name = fmt::format("conv2d_s2_b{}x{}_s{}_{}", plan.rows, plan.columns, plan.simdgroups, dtype);
      grid = MTLSizeMake(ceil_div(Og, plan.rows) * groups, outH * ceil_div(outW, plan.columns), N);
      break;
    case Conv2dKernel::Tensor: {
      if (!in_nhwc) {
        input = conv3d_to_ndhwc(input.unsqueeze(2)).squeeze(2);
      }
      weight = conv3d_weights_to_dhwio(weight_t.unsqueeze(2));
      const bool wide = plan.rows == kConv2dTensorWideOutputs;
      const char* variant = params.kW == 5 ? "k5_" : wide ? "wide_" : Cg <= kConv2dTensorStemChannels ? "stem_" : "";
      name = fmt::format("conv2d_tensor_{}{}", variant, dtype);
      const int64_t rows = wide ? kConv2dTensorWideSide : kConv2dTensorRows;
      const int64_t columns = wide ? kConv2dTensorWideSide : kConv2dTensorColumns;
      grid = MTLSizeMake(ceil_div(Og, plan.rows) * groups, ceil_div(outW, columns) * ceil_div(outH, rows), N);
      break;
    }
  }
  // Tiny and SingleOutput size their grid in threads, the tiled kernels in threadgroups.
  if (!weight.defined()) {
    weight = weight_t.contiguous();
  }
  const auto bias = has_bias ? bias_opt->to(input.scalar_type()).contiguous() : input;
  const bool threads = plan.kernel == Conv2dKernel::Tiny || plan.kernel == Conv2dKernel::SingleOutput;
  const int64_t threadgroup = plan.simdgroups * c10::metal::simdgroup_size;
  auto pipeline = lib.getPipelineStateForFunc(name);
  conv_metal_launch(pipeline, name.c_str(), {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    if (plan.bounded_tiny) {
      const Conv2dTinyParams tiny_params{.conv = params,
                                         .input_channels_last = in_nhwc,
                                         .output_channels_last = !output_t.is_contiguous(),
                                         .weight_channels_last = !weight.is_contiguous()};
      mtl_setArgs(encoder, input, weight, output_t, tiny_params, bias);
    } else {
      mtl_setArgs(encoder, input, weight, output_t, params, bias);
    }
    const auto block = MTLSizeMake(threads ? std::min<int64_t>(grid.width, threadgroup) : threadgroup, 1, 1);
    if (threads) {
      [encoder dispatchThreads:grid threadsPerThreadgroup:block];
    } else {
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:block];
    }
  });
}

static void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                                     NSUInteger strideInX,
                                     NSUInteger strideInY,
                                     NSUInteger dilationRateInX,
                                     NSUInteger dilationRateInY,
                                     NSUInteger paddingHorizontal,
                                     NSUInteger paddingVertical) {
  descriptor_.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:strideInY], [[NSNumber alloc] initWithInteger:strideInX] ];
  descriptor_.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:dilationRateInY], [[NSNumber alloc] initWithInteger:dilationRateInX] ];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingHorizontal],
    [[NSNumber alloc] initWithInteger:paddingHorizontal]
  ];
  descriptor_.channelDimensionIndex = -3LL;
}

// Create convolution descriptor
static void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                           NSUInteger strideInX,
                           NSUInteger strideInY,
                           NSUInteger dilationRateInX,
                           NSUInteger dilationRateInY,
                           NSUInteger paddingHorizontal,
                           NSUInteger paddingVertical,
                           c10::MemoryFormat memory_format,
                           bool use_hwio,
                           NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW
                                                                           : MPSGraphTensorNamedDataLayoutNHWC;

  // HWIO only for the weights-gradient output; PyTorch weight tensors are always OIHW.
  descriptor_.weightsLayout = use_hwio ? MPSGraphTensorNamedDataLayoutHWIO : MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

static Tensor _mps_convolution_impl(const Tensor& input_t,
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups,
                                    std::optional<IntArrayRef> input_shape,
                                    const std::optional<Tensor>& output_opt = std::nullopt) {
  constexpr auto kChannelsLast = MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = MemoryFormat::Contiguous;

  const bool is3DConv = input_t.dim() == 5;
  const bool is1DConv = input_t.dim() == 4 && input_t.size(2) == 1 && weight_t.size(2) == 1 && padding[0] == 0;
  // Allocate output in the user-requested layout regardless of fast-path gate.
  const bool is_channels_last = mps_conv_use_channels_last(input_t, weight_t);
  const bool bias_defined = bias_opt ? bias_opt->defined() : false;

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto output_t = output_opt.has_value()
      ? *output_opt
      : at::empty(input_shape.has_value()
                      ? input_shape.value()
                      : conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation),
                  input->scalar_type(),
                  std::nullopt,
                  kMPS,
                  std::nullopt,
                  is_channels_last ? (is3DConv ? kChannelsLast3d : kChannelsLast) : kContiguous);
  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  if (is3DConv) {
    if (conv3d_is_pointwise(weight_t, stride, padding, groups)) {
      conv3d_pointwise_matmul(input_t, weight_t, bias_opt, output_t);
    } else if (conv3d_prefer_im2col(input_t, weight_t, stride, padding, dilation, groups, output_t)) {
      conv3d_im2col_matmul(input_t, weight_t, bias_opt, stride, padding, output_t);
    } else {
      conv3d_metal_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
    }
    return output_t;
  }
  const auto dtype = input_t.scalar_type();
  const bool gemv_dtype = dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
  const auto gemv_fits_indices = [&](int64_t reduction, int64_t outputs) {
    // Gemv.h unrolls four SIMD-wide loads of at most 16 bytes per lane.
    // Reserve that lookahead for its int32 reduction and output-tile arithmetic.
    constexpr int64_t gemv_unroll = 4;
    constexpr int64_t gemv_vector_bytes = 16;
    const auto margin = gemv_unroll * c10::metal::simdgroup_size * (gemv_vector_bytes / input_t.element_size());
    const auto limit = std::numeric_limits<int32_t>::max() - margin;
    return reduction <= limit && outputs <= limit;
  };
  // One full-input patch is a vector product; mm/addmm use native GEMV for M=1.
  if (gemv_dtype && input_t.size(0) == 1 && input_t.size(1) > 0 && groups == 1 && output_t.size(2) == 1 &&
      output_t.size(3) == 1 && input_t.sizes().slice(2) == weight_t.sizes().slice(2) && padding[0] == 0 &&
      padding[1] == 0 && dilation[0] == 1 && dilation[1] == 1) {
    if (!gemv_fits_indices(input_t.numel(), output_t.numel())) {
      conv2d_simd_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
      return output_t;
    }
    auto output_matrix = output_t.view({1, output_t.size(1)});
    const auto input_matrix = input_t.flatten(1);
    const auto weight_matrix = weight_t.flatten(1).t();
    if (bias_defined) {
      const auto bias = bias_opt->to(input_t.scalar_type()).contiguous();
      at::addmm_out(output_matrix, bias, input_matrix, weight_matrix);
    } else {
      at::mm_out(output_matrix, input_matrix, weight_matrix);
    }
    return output_t;
  }
  // Bound folded batch/channel axes, parity-copy rows, and Winograd tile origins.
  constexpr int64_t max_grid_axis = std::numeric_limits<int32_t>::max();
  const auto input_rows =
      conv2d_fits_int32({input_t.size(1), input_t.size(2)}) ? input_t.size(1) * input_t.size(2) : max_grid_axis;
  const auto output_positions =
      conv2d_fits_int32(output_t.sizes().slice(2)) ? output_t.size(2) * output_t.size(3) : max_grid_axis;
  const auto folded_extent = std::max({input_t.size(1), output_t.size(1), input_rows, output_positions});
  const auto batch_limit = std::max<int64_t>(1, max_grid_axis / folded_extent);
  if (input_t.size(0) > batch_limit) {
    for (int64_t start = 0; start < input_t.size(0);) {
      const auto count = std::min(batch_limit, input_t.size(0) - start);
      _mps_convolution_impl(input_t.narrow(0, start, count),
                            weight_t,
                            bias_opt,
                            padding,
                            stride,
                            dilation,
                            groups,
                            std::nullopt,
                            output_t.narrow(0, start, count));
      start += count;
    }
    return output_t;
  }
  if (conv_needs_long_coordinates(
          input_t.sizes().slice(2), output_t.sizes().slice(2), weight_t.sizes().slice(2), stride, padding, dilation)) {
    conv2d_simd_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
    return output_t;
  }
  if (is1DConv) {
    const auto kernel = conv1d_pick_kernel(input_t, weight_t, output_t, padding[1], stride[1], dilation[1], groups);
    // mm/addmm dispatch vector outputs to native GEMV; full matrices still use MPSGraph.
    const bool gemv = kernel == Conv1dKernel::Im2colMatmul && gemv_dtype &&
        (output_t.size(1) == 1 || (input_t.size(0) == 1 && output_t.size(3) == 1)) &&
        gemv_fits_indices(weight_t.numel() / output_t.size(1), output_t.numel());
    if (kernel != Conv1dKernel::Im2colMatmul || gemv) {
      conv1d_metal_forward(input_t, weight_t, bias_opt, padding[1], stride[1], dilation[1], groups, output_t, kernel);
      return output_t;
    }
  }
  // MPP uses int32 offsets within a batch, including grouped channel offsets.
  // The simd kernel retains 64-bit offsets for larger activation and output planes.
  if (!conv2d_fits_int32(input_t.sizes().slice(1)) || !conv2d_fits_int32(output_t.sizes().slice(1)) ||
      !conv2d_fits_int32(weight_t.sizes())) {
    conv2d_simd_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
    return output_t;
  }
  if (groups > 1 && groups == input_t.size(1) && weight_t.size(1) == 1) {
    const auto params = conv2d_params(input_t, weight_t, output_t, padding, stride, dilation, bias_defined);
    conv2d_depthwise_forward(input_t, weight_t, bias_opt, params, output_t);
    return output_t;
  }
  if (has_mpp()) {
    if (input_t.size(1) == 0) {
      bias_defined ? output_t.copy_(bias_opt->reshape({1, -1, 1, 1}).expand_as(output_t)) : output_t.zero_();
    } else {
      conv2d_metal_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
    }
    return output_t;
  }
  conv2d_simd_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t);
  return output_t;
}

Tensor _mps_convolution(const Tensor& input_t,
                        const Tensor& weight_t,
                        const std::optional<Tensor>& bias_opt,
                        IntArrayRef padding,
                        IntArrayRef stride,
                        IntArrayRef dilation,
                        int64_t groups) {
  return _mps_convolution_impl(input_t, weight_t, bias_opt, padding, stride, dilation, groups, std::nullopt);
}

static Tensor mps_convolution_backward_input(IntArrayRef input_size,
                                             const Tensor& grad_output_t,
                                             const Tensor& weight_t,
                                             IntArrayRef padding,
                                             IntArrayRef stride,
                                             IntArrayRef dilation,
                                             int64_t groups,
                                             bool bias_defined,
                                             at::MemoryFormat output_memory_format) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = grad_output_t.dim() == 5;
  if (!is_macos_at_least(MacOSVersion::MACOS_15_1)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : grad_output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  // Backward uses NDHWC+DHWIO only when the full fast path is beneficial; for
  // factorized kernels / small Cin / depthwise the NCDHW+OIDHW fallback wins.
  const bool use_dhwio =
      is3DConv && is_packed_channels_last_3d(grad_output_t) && conv3d_dhwio_is_beneficial(weight_t.sizes());
  const bool use_nhwc = !is3DConv && is_packed_channels_last_2d(grad_output_t);
  const auto desc_layout = use_dhwio ? kChannelsLast3d : use_nhwc ? kChannelsLast : kContiguous;
  // Allocate grad_input in the caller-supplied layout so it matches input.
  const bool is_channels_last = output_memory_format == kChannelsLast || output_memory_format == kChannelsLast3d;
  // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
  // The depthwise op is NCHW-only, hence the desc_layout condition.
  const bool isDepthwiseConv = groups > 1 && weight_t.dim() >= 4 && weight_t.size(1) == 1 &&
      grad_output_t.ndimension() >= 4 && desc_layout == kContiguous;
  // The graph computes in desc_layout, which is chosen from grad_output; transpose
  // back in-graph when the caller asked for a different layout.
  const bool transpose_grad_input = (use_nhwc || use_dhwio) && !is_channels_last;
  auto grad_input_t =
      at::empty(input_size,
                grad_output_t.options(),
                is_channels_last ? std::optional(is3DConv ? kChannelsLast3d : kChannelsLast) : std::nullopt);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // Contig scratch when graph emits NCDHW but grad_input is CL3d -- covers
  // the macOS-14 fallback and the 3D NCDHW fallback on macOS 15+.
  std::optional<Tensor> grad_input_c;
  const bool needs_contig_scratch = is_channels_last && (is3DConv && !use_dhwio);
  if (needs_contig_scratch) {
    grad_input_c = at::empty_like(grad_input_t, grad_input_t.options().memory_format(MemoryFormat::Contiguous));
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  // Add backward with input
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSShape* mps_input_shape = getMPSShape(input_size, desc_layout);
    std::string key = fmt::format("mps_{}_convolution_backward_input:{}:{}:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_channels_last,
                                  use_dhwio,
                                  use_nhwc,
                                  getArrayRefString(input_size),
                                  getTensorsStringKey({grad_output_t, weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      auto gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

      MPSGraphTensor* gradInputTensor;

      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        MPSGraphTensor* convWeightTensor = use_dhwio
            ? [mpsGraph transposeTensor:weightTensor permutation:@[ @2, @3, @4, @1, @0 ] name:nil]
            : weightTensor;
        gradInputTensor = [mpsGraph convolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:convWeightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                   name:nil];
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        gradInputTensor =
            [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                     weightsTensor:weightTransposeTensor
                                                                       outputShape:mps_input_shape
                                                                        descriptor:depthWiseConv3dDescriptor_
                                                                              name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       desc_layout,
                       false,
                       groups);

        gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                   name:nil];
      }

      if (transpose_grad_input) {
        MPSShape* to_nchw = is3DConv ? @[ @0, @4, @1, @2, @3 ] : @[ @0, @3, @1, @2 ];
        gradInputTensor = [mpsGraph transposeTensor:gradInputTensor permutation:to_nchw name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    const auto grad_out_for_graph =
        grad_input_c ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    // MPSGraph conv miscomputes for non-dense (offset/gapped) weight views; gather instead of using the strided API.
    auto weightsPlaceholder =
        Placeholder(cachedGraph->weightTensor_, weight_t, nil, true, MPSDataTypeInvalid, /*useMPSStridedAPI=*/false);
    const auto& grad_input_out = grad_input_c ? *grad_input_c : grad_input_t;
    auto outputPlaceholder = grad_input_c || transpose_grad_input
        ? Placeholder(cachedGraph->gradInputTensor_, grad_input_out)
        : make_conv_placeholder(cachedGraph->gradInputTensor_, grad_input_t, desc_layout);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, weightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  if (grad_input_c) {
    grad_input_t.copy_(*grad_input_c);
  }
  return grad_input_t;
}

static Tensor mps_convolution_backward_weights(IntArrayRef weight_size,
                                               const Tensor& grad_output_t,
                                               const Tensor& input_t,
                                               IntArrayRef padding,
                                               IntArrayRef stride,
                                               IntArrayRef dilation,
                                               int64_t groups,
                                               bool bias_defined,
                                               at::MemoryFormat output_memory_format) {
  using namespace at::native::mps;
  using namespace mps;
  const bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  // Half-precision WG regresses on NDHWC+DHWIO; force NCDHW+OIDHW.
  const bool half_precision_wg =
      grad_output_t.scalar_type() == at::kBFloat16 || grad_output_t.scalar_type() == at::kHalf;
  // Require BOTH inputs CL3d-packed; otherwise we'd permute the non-packed one each call.
  const bool use_dhwio = is3DConv && !half_precision_wg && is_packed_channels_last_3d(input_t) &&
      is_packed_channels_last_3d(grad_output_t) && conv3d_dhwio_is_beneficial(weight_size);
  const bool use_hwio = !is3DConv && (is_packed_channels_last_2d(input_t) || is_packed_channels_last_2d(grad_output_t));
  const auto desc_layout = use_dhwio ? kChannelsLast3d : use_hwio ? kChannelsLast : kContiguous;
  // grad_weight allocation: 2D follows the caller-supplied layout; 3D always
  // stays contiguous OIDHW (the graph already transposes DHWIO -> OIDHW).
  const bool allocate_grad_weight_cl = output_memory_format == kChannelsLast && !is3DConv;

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{grad_output_t, "grad_output", 1};
  TensorArg input{input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(
      weight_size, grad_output_t.options(), allocate_grad_weight_cl ? std::optional(kChannelsLast) : std::nullopt);

  TensorArg grad_weight{grad_weight_t, "result", 0};

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

  // Strided-API binding of a CL grad_weight is slow, and with packed NHWC feeds it asserts in
  // MPSNDArrayConvolutionA18.mm ("Weights tensor and ndArray input channel mismatch").
  std::optional<Tensor> grad_weight_c;
  if (allocate_grad_weight_cl) {
    grad_weight_c = at::empty_like(grad_weight_t, grad_weight_t.options().memory_format(MemoryFormat::Contiguous));
  }

  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    // Under DHWIO the graph emits weight grad in DHWIO order; the op output
    // shape must match, and we transpose back to OIDHW after.
    MPSShape* mps_weight_shape = use_dhwio
        ? @[ @(weight_size[2]), @(weight_size[3]), @(weight_size[4]), @(weight_size[1]), @(weight_size[0]) ]
        : use_hwio ? @[ @(weight_size[2]), @(weight_size[3]), @(weight_size[1]), @(weight_size[0]) ]
                   : getMPSShape(weight_size);
    std::string key = fmt::format("mps_{}convolution_backward_weights:{}:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  allocate_grad_weight_cl,
                                  use_dhwio,
                                  use_hwio,
                                  getTensorsStringKey({grad_output_t, input_t, grad_weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* inputShape = getMPSShape(input_t, desc_layout);
      MPSShape* gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      // For the non-CL path the depthwise heuristic inspects the OIHW weight shape.
      MPSShape* weight_shape_OIDHW = getMPSShape(weight_size);
      bool isDepthwiseConv = ((groups > 1 && (weight_shape_OIDHW[1].intValue == 1)) && inputShape.count >= 4 &&
                              weight_shape_OIDHW.count >= 4 && !use_hwio);

      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);

      MPSGraphTensor* gradWeightTensor;
      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        gradWeightTensor = [mpsGraph convolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                       name:nil];
        if (use_dhwio) {
          gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensor permutation:@[ @4, @3, @0, @1, @2 ] name:nil];
        }
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        NSNumber* outputFeatChannelDim = mps_weight_shape[0];
        MPSShape* weightShapeTranspose = @[ @1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3] ];
        MPSGraphTensor* gradWeightTensorTranspose =
            [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                         sourceTensor:inputTensor
                                                                          outputShape:weightShapeTranspose
                                                                           descriptor:depthWiseConv3dDescriptor_
                                                                                 name:nil];
        gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       desc_layout,
                       use_hwio,
                       groups);

        gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                       name:nil];
        if (use_hwio) {
          gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensor permutation:@[ @3, @2, @0, @1 ] name:nil];
        }
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
    });

    const bool contig_feeds = grad_weight_c.has_value() && desc_layout == kContiguous;
    const auto grad_out_for_graph =
        contig_feeds ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    const auto input_for_graph = contig_feeds ? input_t.contiguous() : materialize_for_conv(input_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    auto inputPlaceholder = make_conv_placeholder(cachedGraph->inputTensor_, input_for_graph, desc_layout);
    auto outputPlaceholder =
        Placeholder(cachedGraph->gradWeightTensor_, grad_weight_c ? *grad_weight_c : grad_weight_t);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (grad_weight_c) {
    grad_weight_t.copy_(*grad_weight_c);
  }
  return grad_weight_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_convolution_backward(const at::Tensor& input,
                                                                        const at::Tensor& grad_output,
                                                                        const at::Tensor& weight,
                                                                        IntArrayRef padding,
                                                                        IntArrayRef stride,
                                                                        IntArrayRef dilation,
                                                                        int64_t groups,
                                                                        std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    // Decide the layout once from input and weight; both gradients must share it.
    const auto memory_format = conv_backward_memory_format(input, weight);
    if (output_mask[0]) {
      grad_input = mps_convolution_backward_input(
          input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2], memory_format);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(
          weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2], memory_format);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

static Tensor mps_convolution_transpose_forward(const Tensor& grad_output,
                                                const Tensor& weight,
                                                IntArrayRef padding,
                                                IntArrayRef output_padding,
                                                IntArrayRef stride,
                                                IntArrayRef dilation,
                                                int64_t groups) {
  auto input_size =
      conv_input_size(grad_output.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
  const auto output_memory_format = conv_backward_memory_format(grad_output, weight);
  return mps_convolution_backward_input(
      input_size, grad_output, weight, padding, stride, dilation, groups, false, output_memory_format);
}

Tensor _mps_convolution_transpose(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  IntArrayRef padding,
                                  IntArrayRef output_padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  int64_t groups) {
  bool is_unsupported_3d_dtype =
      (input_t.dim() == 5 && (input_t.scalar_type() == kHalf || input_t.scalar_type() == kBFloat16));
  TORCH_CHECK(!is_unsupported_3d_dtype, "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS");

  auto output_t =
      mps_convolution_transpose_forward(input_t, weight_t, padding, output_padding, stride, dilation, groups);
  return output_t;
}

static Tensor mps_convolution_transpose_backward_input(const Tensor& grad_output_t,
                                                       const Tensor& weight_t,
                                                       IntArrayRef padding,
                                                       IntArrayRef stride,
                                                       IntArrayRef dilation,
                                                       int64_t groups,
                                                       IntArrayRef input_shape) {
  return _mps_convolution_impl(grad_output_t, weight_t, std::nullopt, padding, stride, dilation, groups, input_shape);
}

static Tensor mps_convolution_transpose_backward_weight(IntArrayRef weight_size,
                                                        const Tensor& grad_output_t,
                                                        const Tensor& input_t,
                                                        IntArrayRef padding,
                                                        IntArrayRef stride,
                                                        IntArrayRef dilation,
                                                        int64_t groups) {
  const auto output_memory_format = conv_backward_memory_format(input_t, grad_output_t);
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t, padding, stride, dilation, groups, false, output_memory_format);
}

std::tuple<Tensor, Tensor> mps_convolution_transpose_backward(const Tensor& input,
                                                              const Tensor& grad_output,
                                                              const Tensor& weight,
                                                              IntArrayRef padding,
                                                              IntArrayRef output_padding,
                                                              IntArrayRef stride,
                                                              IntArrayRef dilation,
                                                              int64_t groups,
                                                              std::array<bool, 2> output_mask) {
  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input =
        mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, input.sizes());
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

} // namespace at::native
