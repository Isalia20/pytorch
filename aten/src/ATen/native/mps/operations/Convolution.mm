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
#include <ATen/ops/mm.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>
#include <fmt/format.h>

#include <algorithm>
#include <limits>

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
                         int64_t padH,
                         int64_t padW,
                         int64_t kH = 1,
                         int64_t sH = 1,
                         int64_t dH = 1) {
  using namespace mps;
  const int64_t batches = source.numel() / (channels * height * width);
  const int64_t rows = output.numel() / (batches * channels * (width + 2 * padW) * kH);
  const int64_t padded = rows * (width + 2 * padW) * kH;
  const int channel_tile = channels <= 4 ? 4 : channels <= 8 ? 8 : channels <= 16 ? 16 : 32;
  const int spatial_tile = channels <= 4 ? 256 : channels <= 8 ? 128 : channels <= 16 ? 64 : 32;
  // vec2 loads need the buffer base 2-element aligned too
  const bool vectorized_read = channel_tile >= 16 && padH == 0 && padW == 0 && kH == 1 && sH == 1 && dH == 1 &&
      padded % 2 == 0 && source.storage_offset() % 2 == 0;
  const bool vectorized_write = channels % 2 == 0;
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("nchw_to_nhwc_{}_{}_{}_{}_{}",
                                                          scalarToMetalTypeString(source),
                                                          channel_tile,
                                                          spatial_tile,
                                                          vectorized_read,
                                                          vectorized_write));
  const std::array<int32_t, 4> dims = {static_cast<int32_t>(channels),
                                       static_cast<int32_t>(height),
                                       static_cast<int32_t>(width),
                                       static_cast<int32_t>(width + 2 * padW)};
  const std::array<int32_t, 2> pad = {static_cast<int32_t>(padH), static_cast<int32_t>(padW)};
  const std::array<int32_t, 4> filter_rows = {
      static_cast<int32_t>(kH), static_cast<int32_t>(sH), static_cast<int32_t>(dH), static_cast<int32_t>(rows)};
  const auto threadgroups = MTLSizeMake(c10::metal::ceil_div(padded, int64_t(spatial_tile)),
                                        c10::metal::ceil_div(channels, int64_t(channel_tile)),
                                        batches);
  conv_metal_launch(pipeline, "nchw_to_nhwc", {source}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, source, output, dims, pad, filter_rows);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  nchw_to_nhwc(tensor.contiguous(), output, channels, 1, spatial_size, 0, 0);
  return output;
}

// DHWIO weight copy; ATen's strided permute copy runs well below memory
// bandwidth for this permutation, 2-4x slower than the flat kernel.
static Tensor conv3d_weights_to_dhwio(const Tensor& weight) {
  using namespace mps;
  const auto output_channels = weight.size(0);
  const auto input_channels_per_group = weight.size(1);
  const auto kernel_depth = weight.size(2);
  const auto kernel_height = weight.size(3);
  const auto kernel_width = weight.size(4);
  auto output = at::empty({kernel_depth, kernel_height, kernel_width, input_channels_per_group, output_channels},
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

// conv3d forward on the Metal kernels: MPP on macOS 26.2+, simdgroup otherwise;
// planes past int32 take the long-indexed simdgroup variant on any macOS.
static void conv3d_metal_forward(const Tensor& input_t,
                                 const Tensor& weight_t,
                                 const std::optional<Tensor>& bias_opt,
                                 IntArrayRef padding,
                                 IntArrayRef stride,
                                 IntArrayRef dilation,
                                 int64_t groups,
                                 const Tensor& output_t) {
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
  const bool use_long_index = input_channels * input_height * input_width > kInt32Max ||
      output_channels * output_height * output_width > kInt32Max;
  const int64_t input_channels_per_group = input_channels / groups;
  TORCH_CHECK(weight_t.size(2) * weight_t.size(3) * weight_t.size(4) * input_channels_per_group <= kInt32Max,
              "conv3d: kernel volume times channels per group exceeds int32");
  const bool use_mpp = !use_long_index && has_mpp();

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
      const auto threads_per_threadgroup = MTLSizeMake(tile.simdgroups * 32, 1, 1);
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
  const auto threads_per_threadgroup =
      MTLSizeMake(static_cast<NSUInteger>(tile.simdgroups_rows) * tile.simdgroups_columns * 32, 1, 1);
  conv_metal_launch(pipeline, "conv3d_simd", {activation, weights}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, activation, weights, output_t, params, bias ? *bias : activation);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
  });
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
                                 const Tensor& output_t) {
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
  const auto kernel = conv1d_pick_kernel(input_t, weight_t, output_t, padding, stride, dilation, groups);
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
                threadsPerThreadgroup:MTLSizeMake(simdgroups * 32, 1, 1)];
      }
      getMPSProfiler().endProfileKernel(pipeline, stream);
    }
  });
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
  const bool square = params.kH == params.kW && params.sH == params.sW && params.dH == 1 && params.dW == 1;
  const bool fast = square && (params.kW == 3 || params.kW == 5 || params.kW == 7) && params.sW <= 2;
  const int64_t rw = fast && params.kW == 7 ? 4 : 8;
  const bool full =
      fast && params.kW == 3 && params.sW == 1 && params.outW % 8 == 0 && input.scalar_type() == kBFloat16;
  const auto bias = params.has_bias ? bias_opt->to(input.scalar_type()).contiguous() : input;
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv2d_depthwise_{}_{}_{}_{}",
                                                          fast ? params.kW : 0,
                                                          fast ? params.sW : 0,
                                                          scalarToMetalTypeString(input),
                                                          full));
  const auto nx = c10::metal::ceil_div(int64_t(params.outW), rw);
  const auto ny = c10::metal::ceil_div(int64_t(params.outH), int64_t(2));
  conv_metal_launch(pipeline, "conv2d_depthwise", {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, weight, output, params, bias);
    [encoder dispatchThreads:MTLSizeMake(nx, ny, params.N * params.C_in)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(nx, 32), std::min<int64_t>(ny, 8), 1)];
  });
  if (!output.is_same(output_t)) {
    output_t.copy_(output);
  }
}

static Tensor conv2d_im2col(const Tensor& input, const Conv2DParams& params) {
  using namespace mps;
  const int64_t positions = int64_t(params.outH) * params.outW, taps = int64_t(params.kH) * params.kW;
  auto columns = at::empty({params.N, params.C_in * taps, params.outH, params.outW}, input.options());
  const std::array<int64_t, 4> strides{input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv2d_im2col_{}", scalarToMetalTypeString(input)));
  const auto groups8 = c10::metal::ceil_div(positions, int64_t(8));
  conv_metal_launch(pipeline, "conv2d_im2col", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, columns, params, strides);
    [encoder dispatchThreads:MTLSizeMake(groups8, params.C_in * taps, params.N)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(groups8, 256), 1, 1)];
  });
  return columns;
}

static Tensor conv2d_deinterleave(const Tensor& input) {
  using namespace mps;
  const int64_t W = input.size(3), rows = input.numel() / W;
  auto output = at::empty({input.size(0), input.size(1), input.size(2), W + 1}, input.options());
  const std::array<int32_t, 2> dims{static_cast<int32_t>(W), static_cast<int32_t>((W + 1) / 2 + 1)};
  auto pipeline = lib.getPipelineStateForFunc("conv2d_deinterleave_" + scalarToMetalTypeString(input));
  const auto columns = c10::metal::ceil_div(W + 1, int64_t(4));
  conv_metal_launch(pipeline, "conv2d_deinterleave", {input}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, input, output, dims);
    [encoder dispatchThreads:MTLSizeMake(columns, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(columns, 256), 1, 1)];
  });
  return output;
}

static void conv2d_mpp_forward(const Tensor& input_t,
                               const Tensor& weight_t,
                               const std::optional<Tensor>& bias_opt,
                               IntArrayRef padding,
                               IntArrayRef stride,
                               IntArrayRef dilation,
                               int64_t groups,
                               const Tensor& output_t,
                               bool fused_mid = false) {
  using namespace mps;
  constexpr int64_t kTile = 64;
  const int64_t kH = weight_t.size(2), kW = weight_t.size(3), Cg = weight_t.size(1), Og = output_t.size(1) / groups;
  const auto dtype = mps::scalarToMetalTypeString(input_t);
  const bool out_nhwc = !output_t.is_contiguous();
  const bool has_bias = bias_opt && bias_opt->defined();
  const auto bias = has_bias ? bias_opt->to(input_t.scalar_type()).contiguous() : input_t;
  auto input = input_t;
  bool in_nhwc = !input.is_contiguous() && input.is_contiguous(MemoryFormat::ChannelsLast);
  input = input.contiguous(in_nhwc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
  Conv2DParams params{
      .N = static_cast<int32_t>(input.size(0)),
      .C_in = static_cast<int32_t>(input.size(1)),
      .C_out = static_cast<int32_t>(output_t.size(1)),
      .H = static_cast<int32_t>(input.size(2)),
      .W = static_cast<int32_t>(input.size(3)),
      .outH = static_cast<int32_t>(output_t.size(2)),
      .outW = static_cast<int32_t>(output_t.size(3)),
      .kH = static_cast<int32_t>(kH),
      .kW = static_cast<int32_t>(kW),
      .sH = static_cast<int32_t>(stride[0]),
      .sW = static_cast<int32_t>(stride[1]),
      .padH = static_cast<int32_t>(padding[0]),
      .padW = static_cast<int32_t>(padding[1]),
      .dH = static_cast<int32_t>(dilation[0]),
      .dW = static_cast<int32_t>(dilation[1]),
      .C_in_per_group = static_cast<int32_t>(Cg),
      .C_out_per_group = static_cast<int32_t>(Og),
      .has_bias = has_bias,
  };
  const auto launch =
      [&](const std::string& name, const Tensor& weight, MTLSize grid, MTLSize block, bool dispatch_threads = false) {
        auto pipeline = lib.getPipelineStateForFunc(name);
        conv_metal_launch(pipeline, name.c_str(), {input, weight}, [&](id<MTLComputeCommandEncoder> encoder) {
          mtl_setArgs(encoder, input, weight, output_t, params, bias);
          if (dispatch_threads) {
            [encoder dispatchThreads:grid threadsPerThreadgroup:block];
          } else {
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:block];
          }
        });
      };
  const bool pointwise = kH == 1 && kW == 1 && padding[0] == 0 && padding[1] == 0;
  const bool patch_embed =
      kH == stride[0] && kW == stride[1] && padding[0] == 0 && padding[1] == 0 && dilation[0] == 1 && dilation[1] == 1;
  const bool strided = stride[0] > 1 || stride[1] > 1;
  const bool apple10 = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
  const bool bfloat_simd = is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS) && input.scalar_type() == kBFloat16;
  const bool bfloat_nchw = bfloat_simd && groups == 1 && !in_nhwc && !out_nhwc;
  static const int64_t cores = std::max<int64_t>(at::mps::MPSDevice::getInstance()->getCoreCount(), 8);
  const int64_t target = (apple10 ? 8 : 16) * cores;
  const bool tiny_s2 = Cg == 3 && stride[0] == 2 && stride[1] == 2 && params.W % 16 == 0 && Og <= (apple10 ? 16 : 32);
  const bool tiny_3x3 = kH == 3 && kW == 3 && padding[0] == 1 && padding[1] == 1 && dilation[0] == 1 &&
      dilation[1] == 1 && params.W % 8 == 0 &&
      ((!strided && (Cg == 1 || (Cg == 3 && (!apple10 || Og <= 32)))) || tiny_s2);
  if (bfloat_nchw && Og % 4 == 0 && input.storage_offset() % 2 == 0 && output_t.storage_offset() % 2 == 0 &&
      ((pointwise && !strided && Cg == 4 && int64_t(params.H) * params.W % 8 == 0) || tiny_3x3)) {
    const auto w = weight_t.contiguous();
    const int64_t columns = pointwise ? int64_t(params.H) * params.W / 8 : params.outW / 8;
    launch(fmt::format("conv2d_tiny_{}_{}_{}_bfloat", Cg, kW, stride[1]),
           w,
           MTLSizeMake(columns, pointwise ? Og / 4 : params.outH, pointwise ? params.N : params.N * Og / 4),
           MTLSizeMake(std::min<int64_t>(columns, 256), 1, 1),
           true);
    return;
  }
  const int64_t o1_threads = c10::metal::ceil_div(int64_t(params.outW), int64_t(8)) * params.outH;
  if (Og == 1 && !in_nhwc && !out_nhwc && o1_threads * params.N * params.C_out >= 4096) {
    const bool wide = bfloat_simd && stride[1] == 1 && dilation[1] == 1 &&
        ((kW == 1 && Cg <= 32) || (kW == 3 && Cg <= 16 && o1_threads * params.N * params.C_out >= 65536));
    const bool paired = bfloat_simd && kH == 3 && kW == 3 && stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
        dilation[1] == 1 && Cg <= 128 && o1_threads * params.N * params.C_out >= 16384 &&
        (apple10 || (Cg <= 16 && o1_threads >= 4096));
    const bool fixed = (kW == 3 || wide) && stride[1] == 1 && dilation[1] == 1;
    const int64_t width = paired ? 4 : wide ? 16 : 8, height = paired ? 2 : 1;
    const int64_t threads =
        c10::metal::ceil_div(int64_t(params.outW), width) * c10::metal::ceil_div(int64_t(params.outH), height);
    const auto w = weight_t.contiguous();
    launch(fmt::format("conv2d_o1_{}_{}_{}_{}_{}", fixed ? kW : 0, fixed ? 1 : 0, height, width, dtype),
           w,
           MTLSizeMake(threads, params.N * params.C_out, 1),
           MTLSizeMake(std::min<int64_t>(threads, 256), 1, 1),
           true);
    return;
  }
  const bool small_wino = Og > 1 && Og <= 4 && Cg >= 256 && Cg <= 384 && int64_t(params.H) * params.W < 131072 &&
      int64_t(params.N) * params.H * params.W >= 16384;
  if (!apple10 && bfloat_nchw && kH == 3 && kW == 3 && stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
      dilation[1] == 1 && Cg >= 8 &&
      (fused_mid || small_wino ||
       (Og > 1 && Og <= 16 && int64_t(params.H) * params.W * (Og <= 8 ? 1 : params.N) >= (Og <= 8 ? 131072 : 65536)))) {
    const int64_t rows = fused_mid ? 32 : Og <= 8 ? 8 : 16, columns = small_wino ? 8 : rows == 8 ? 32 : 16;
    const int64_t tiles = ((int64_t(params.outH) + 1) / 2) * ((int64_t(params.outW) + 1) / 2);
    const auto w = weight_t.contiguous();
    constexpr int64_t max_int = std::numeric_limits<int32_t>::max();
    const bool index32 = int64_t(params.C_in) * params.H * params.W <= max_int &&
        int64_t(params.C_out) * params.outH * params.outW <= max_int && w.numel() <= max_int;
    launch(fmt::format("conv2d_winograd_fused_{}x{}_bfloat_{}", rows, columns, index32 ? "int" : "long"),
           w,
           MTLSizeMake(c10::metal::ceil_div(Og, rows), c10::metal::ceil_div(tiles, columns), params.N),
           MTLSizeMake(fused_mid ? 256 : 128, 1, 1));
    return;
  }
  const bool single_pointwise = apple10 && params.N == 1 && input.scalar_type() == kBFloat16 &&
      int64_t(params.C_out) * params.C_in * params.outH * params.outW >= (int64_t(1) << 29) &&
      ((Cg > 16 && Cg < 128 && Og >= 256) || (Cg > 128 && Cg <= 192 && Og > 32 && Og <= 64));
  const int64_t pointwise_plane = int64_t(params.H) * params.W;
  if (pointwise && apple10 && groups == 1 && params.N == 1 && input.scalar_type() == kBFloat16 && !out_nhwc &&
      !in_nhwc && !strided && (Cg == 128 || Cg == 256) && Og >= 128 && Og <= 256 && pointwise_plane >= 262144 &&
      pointwise_plane <= std::numeric_limits<int32_t>::max() / std::max(Cg, Og)) {
    const auto w = weight_t.contiguous();
    launch(fmt::format("conv2d_pointwise_{}", Cg),
           w,
           MTLSizeMake(c10::metal::ceil_div(pointwise_plane, int64_t(128)), c10::metal::ceil_div(Og, int64_t(64)), 1),
           MTLSizeMake(128, 1, 1));
    return;
  }
  if (pointwise && apple10 && groups == 1 && params.N == 1 && !out_nhwc && !in_nhwc && !strided && !single_pointwise &&
      (Cg <= 16 || int64_t(params.C_out) * params.C_in * params.outH * params.outW >= (int64_t(1) << 29))) {
    auto out = output_t.view({params.C_out, int64_t(params.outH) * params.outW});
    const auto x = input.view({params.C_in, int64_t(params.H) * params.W});
    const auto w = weight_t.reshape({params.C_out, params.C_in});
    has_bias ? at::addmm_out(out, bias.view({params.C_out, 1}), w, x) : at::mm_out(out, w, x);
    return;
  }
  const bool sample_s2 = pointwise && stride[0] == 2 && stride[1] == 2 && Cg >= 128 && Cg <= 256 && params.outW >= 24 &&
      c10::metal::ceil_div(Og, int64_t(128)) * params.outH * c10::metal::ceil_div(int64_t(params.outW), int64_t(32)) *
              params.N >=
          target;
  const bool patch2 =
      patch_embed && kH == 2 && kW == 2 && params.N > 1 && Cg >= 96 && Cg <= 192 && Og % 192 == 0 && params.H % 2 == 0;
  const bool packed3 =
      apple10 && kH == 3 && kW == 3 && dilation[0] == 1 && dilation[1] == 1 && padding[0] == 1 && padding[1] == 1;
  const int64_t positions = int64_t(params.outH) * params.outW;
  const bool narrow_batch = params.N >= 8 && (Cg == 64 || Cg == 128) && params.W >= 32 && params.W <= 96 &&
      positions >= 1024 && positions <= 16384;
  const bool narrow_large =
      params.N == 1 && Cg == 176 && params.W >= 128 && params.W <= 512 && positions >= 16384 && positions <= 131072;
  const int64_t narrow_columns = Cg == 64 ? 64 : 128;
  if (packed3 && !strided && bfloat_nchw && (narrow_batch || narrow_large) && Og > 32 && Og <= 64 &&
      params.N * c10::metal::ceil_div(positions, narrow_columns) >= target) {
    const auto w = conv3d_weights_to_dhwio((Cg == 128 ? weight_t : weight_t.transpose(0, 1)).unsqueeze(2));
    const int64_t simdgroups = Cg == 128 ? 8 : 4;
    launch(fmt::format("conv2d_narrow_{}", Cg),
           w,
           MTLSizeMake(1, c10::metal::ceil_div(positions, narrow_columns), params.N),
           MTLSizeMake(32 * simdgroups, 1, 1));
    return;
  }
  const bool patch_flat = packed3 && !strided && Cg > 32 && Cg < 128 && Og > 128 && Og <= 256 && params.W >= 32 &&
      params.W % 2 == 0 && positions >= 4096 && positions <= 32768;
  const bool patch_s1 = packed3 && !strided && ((Cg == 16 && Og >= 256 && params.W % 8 == 0) || patch_flat) &&
      input.storage_offset() % 2 == 0 &&
      int64_t(params.C_in) * params.H * params.W <= std::numeric_limits<int32_t>::max() &&
      int64_t(params.C_out) * params.outH * params.outW <= std::numeric_limits<int32_t>::max();
  const bool patch_s2 = packed3 && stride[0] == 2 && stride[1] == 2 &&
      ((Cg > 8 && Cg <= 16 && Og > 16 && Og <= 32) ||
       (Cg >= 128 && Og > 128 && Og <= 256 && params.outW > 8 && params.outW <= 16 &&
        int64_t(params.N) * params.outH >= target));
  const bool packed_patch = (params.W % 2 == 0 || (sample_s2 && params.H % 2 == 0)) && input.storage_offset() % 2 == 0;
  if (bfloat_nchw && (patch_s1 || patch_s2 || (packed_patch && (sample_s2 || patch2)))) {
    const auto w = weight_t.contiguous();
    const int64_t rows = patch_flat ? 128 : patch_s1 ? 64 : patch_s2 ? (Cg <= 16 ? 32 : 256) : sample_s2 ? 128 : 192;
    const int64_t columns = patch_flat ? 32 : patch_s1 || (patch_s2 && Cg <= 16) ? 64 : sample_s2 ? 32 : 16;
    launch(fmt::format("conv2d_patch_{}_{}_s{}_bfloat", kH, rows, params.sH),
           w,
           MTLSizeMake(c10::metal::ceil_div(Og, rows),
                       patch_flat ? c10::metal::ceil_div(positions, columns)
                                  : params.outH * c10::metal::ceil_div(int64_t(params.outW), columns),
                       params.N),
           MTLSizeMake(128, 1, 1));
    return;
  }
  const int64_t im2col_bytes = output_t.numel() / output_t.size(1) * Cg * kH * kW * groups * input.element_size();
  const bool direct_columns = bfloat_nchw && stride[0] == 2 && stride[1] == 2 && dilation[0] == 1 && dilation[1] == 1 &&
      kH == kW && (kH == 2 || kH == 3) && Og >= 512 && im2col_bytes <= (int64_t(128) << 20) &&
      (apple10 ? (params.N > 1 && Cg >= 384 && positions <= 256) ||
               (params.N == 1 && Cg >= 512 && positions >= 512 &&
                (padding[0] > 0 || padding[1] > 0 || positions >= 8192))
               : params.N == 1 && Cg >= 512 && Og >= 2 * Cg && positions >= 512 && positions <= 4096);
  const int64_t padded_h = params.H + 2 * padding[0], padded_w = params.W + 2 * padding[1];
  const int64_t row_tiles = c10::metal::ceil_div(int64_t(params.outW), kTile);
  const float tile_waste = 1.0f - float(params.outW) / float(row_tiles * kTile);
  const bool flat = !strided && tile_waste > 11.0f / float(Og) + 0.08f && tile_waste * float(Cg * kH * kW) > 1000.0f;
  const float waste32 = 1.0f - float(params.outW) / float(c10::metal::ceil_div(int64_t(params.outW), int64_t(32)) * 32);
  const bool narrow_better = tile_waste - waste32 >= 0.14f;
  const bool s2_mid = Og > 64 && Og <= 96 && bfloat_simd;
  const int64_t bm = Og <= 16 ? 16 : Og <= 32 ? 32 : Og <= 64 ? 64 : s2_mid ? 96 : 128;
  const auto s2_threadgroups = [&](int64_t bn) {
    return c10::metal::ceil_div(Og, bm) * groups * params.outH * c10::metal::ceil_div(int64_t(params.outW), bn) *
        params.N;
  };
  const int64_t bn = bm == 96 || params.outW < kTile || narrow_better || s2_threadgroups(kTile) < target ? 32
      : Cg <= 16 && Og <= 32 && params.outW >= 128 && s2_threadgroups(128) >= target                     ? 128
                                                                                                         : kTile;
  if (!pointwise && !patch_embed && !in_nhwc && !out_nhwc && stride[1] == 2 && (kW - 1) * dilation[1] <= 6 && Cg > 8 &&
      Cg <= 128 && params.outW >= 24 && std::min(tile_waste, waste32) < 0.35f && s2_threadgroups(bn) >= 4 * cores) {
    const int64_t simdgroups = bm == 128 ? 8 : 4;
    const auto w = conv3d_weights_to_dhwio(weight_t.transpose(0, 1).unsqueeze(2));
    launch(fmt::format("conv2d_s2_b{}x{}_s{}_{}", bm, bn, simdgroups, dtype),
           w,
           MTLSizeMake(c10::metal::ceil_div(Og, bm) * groups,
                       params.outH * c10::metal::ceil_div(int64_t(params.outW), bn),
                       params.N),
           MTLSizeMake(simdgroups * 32, 1, 1));
    return;
  }
  // Folding taps into K needs an NHWC staging copy (kH copies for every tap);
  // de-interleaving rows lets stride 2 read NCHW at unit stride.
  const bool tiny = groups == 1 && dilation[1] == 1 && Cg <= 8;
  const bool unfolded = tiny && Og >= 64 && 5 * kH * Cg * padded_w <= 3 * Og * params.outW;
  const bool parity = !direct_columns && !in_nhwc && !tiny && !(groups > 1 && Cg <= 8) && stride[1] == 2 &&
      dilation[1] == 1 && padding[1] <= 2 && params.outW > 32;
  const bool narrow = strided && !tiny && im2col_bytes <= (int64_t(64) << 20) &&
      (params.outW <= 8 || (params.outW <= 20 && int64_t(params.N) * params.outH * params.outW <= 1024));
  const bool merged = !unfolded && !narrow && groups == 1 && dilation[1] == 1 &&
      (tiny || (flat && params.outH * params.outW <= 1024) || (stride[1] > 1 && !in_nhwc && !parity) ||
       (Cg <= 16 && 2 * Cg * padded_h * padded_w <= Og * params.outH * params.outW));
  const bool im2col = direct_columns || (patch_embed && !parity) || narrow ||
      (im2col_bytes <= (int64_t(64) << 20) &&
       ((groups > 1 && Cg <= 8) || (stride[1] > 1 && !merged && !unfolded && !parity && Cg * kH * kW <= 1024)));
  Tensor weight;
  int32_t K = params.C_in_per_group, Wp = 0;
  if (pointwise) {
    if (stride[0] > 1 || stride[1] > 1) {
      input = input.slice(2, 0, std::nullopt, stride[0]).slice(3, 0, std::nullopt, stride[1]);
      input = input.contiguous(in_nhwc ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous);
    }
    params.H = params.outH = 1;
    params.W = params.outW = static_cast<int32_t>(output_t.size(2) * output_t.size(3));
    params.sH = params.sW = 1;
    weight = weight_t.contiguous();
  } else if (im2col) {
    input = conv2d_im2col(input, params);
    in_nhwc = false;
    K = static_cast<int32_t>(Cg * kH * kW);
    params.C_in = static_cast<int32_t>(groups * K);
    params.C_in_per_group = K;
    params.H = params.outH;
    params.W = Wp = params.outW;
    params.kH = params.kW = params.sH = params.sW = params.dH = params.dW = 1;
    params.padH = params.padW = 0;
    if (direct_columns) {
      params.H = params.outH = 1;
      params.W = params.outW = static_cast<int32_t>(positions);
      Wp = 0;
    }
    weight = weight_t.contiguous();
  } else {
    if (unfolded) {
      auto rows =
          at::empty({params.N, kH * params.C_in, params.outH, padded_w}, input.options(), MemoryFormat::ChannelsLast);
      nchw_to_nhwc(input.contiguous(),
                   rows,
                   params.C_in,
                   params.H,
                   params.W,
                   padding[0],
                   padding[1],
                   kH,
                   stride[0],
                   dilation[0]);
      input = rows;
      in_nhwc = true;
      K = static_cast<int32_t>(kW * kH * Cg);
      params.C_in = static_cast<int32_t>(kH * Cg);
      params.C_in_per_group = K;
      params.H = params.outH;
      params.W = static_cast<int32_t>(padded_w);
      params.kH = params.kW = params.sH = params.dH = 1;
      params.padH = params.padW = 0;
      weight = conv3d_weights_to_dhwio(
          weight_t.as_strided({Cg, 1, weight_t.size(0), kW, kH},
                              {weight_t.stride(1), 0, weight_t.stride(0), weight_t.stride(3), weight_t.stride(2)}));
    } else if (merged) {
      if (in_nhwc) {
        input = at::constant_pad_nd(input, {padding[1], padding[1], padding[0], padding[0]});
      } else {
        auto padded =
            at::empty({params.N, params.C_in, padded_h, padded_w}, input.options(), MemoryFormat::ChannelsLast);
        nchw_to_nhwc(input, padded, params.C_in, params.H, params.W, padding[0], padding[1]);
        input = padded;
      }
      in_nhwc = true;
      K = static_cast<int32_t>(kW * Cg);
      params.C_in_per_group = K;
      params.H = static_cast<int32_t>(padded_h);
      params.W = static_cast<int32_t>(padded_w);
      params.kW = 1;
      params.padH = params.padW = 0;
      weight = conv3d_weights_to_dhwio(
          weight_t.as_strided({Cg, 1, kH, weight_t.size(0), kW},
                              {weight_t.stride(1), 0, weight_t.stride(2), weight_t.stride(0), weight_t.stride(3)}));
    } else {
      weight = conv3d_weights_to_dhwio(weight_t.transpose(0, 1).unsqueeze(2));
    }
    if (parity) {
      input = conv2d_deinterleave(input);
      params.W = static_cast<int32_t>(input.size(3));
      params.sW = 1;
    } else if (stride[1] > 1 && !in_nhwc) {
      input = conv3d_to_ndhwc(input.unsqueeze(2)).squeeze(2);
      in_nhwc = true;
    }
    if (flat && !unfolded) {
      if (!merged && (padding[0] > 0 || padding[1] > 0)) {
        input = at::constant_pad_nd(input, {padding[1], padding[1], padding[0], padding[0]});
        params.H = static_cast<int32_t>(input.size(2));
        params.W = static_cast<int32_t>(input.size(3));
        params.padH = params.padW = 0;
      }
      Wp = params.W;
    }
  }
  const bool wide_pointwise = apple10 && params.N > 1 && Cg >= 160 && Cg <= 960 && Cg % 32 == 0 && Og <= 640 &&
      Og % 160 == 0 && params.W >= 4096;
  const bool compact_pointwise = params.N > 1 && Cg >= 256 && Og >= 128 && Og <= 256 && params.W >= 512 &&
      params.W <= 1024 &&
      c10::metal::ceil_div(Og, int64_t(128)) * c10::metal::ceil_div(int64_t(params.W), int64_t(32)) * params.N >=
          target;
  const bool long_pointwise = apple10 && params.N > 1 && Cg >= 256 && Og > 32 && Og <= 64 && params.W >= 4096;
  const bool fast_pointwise = direct_columns ||
      (bfloat_nchw && pointwise && !strided &&
       (wide_pointwise || compact_pointwise || long_pointwise || single_pointwise ||
        (apple10 && params.N > 1 && Cg > 16 && Cg <= 128 && Og >= 512 && params.W >= 512)));
  const bool fixed3x3 = apple10 && bfloat_nchw && Wp == 0 && params.kH == 3 && params.kW == 3 && params.sH == 1 &&
      params.sW == 1 && params.dH == 1 && params.dW == 1 && params.padH == 1 && params.padW == 1 &&
      ((params.N == 1 && Cg > 32 && Cg <= 64 && Og > 32 && Og <= 64 && positions >= 131072) ||
       (Cg > 16 && Og > 64 && Og <= 96 && params.outW >= 64 &&
        params.N * params.outH * c10::metal::ceil_div(int64_t(params.outW), int64_t(64)) >= target));
  const int64_t block = Og <= 16 ? 16 : Og <= 32 ? 32 : 64;
  const auto tiles_for = [&](int64_t width) {
    return Wp > 0 ? c10::metal::ceil_div(int64_t(params.outH - 1) * Wp + params.outW, width)
                  : params.outH * c10::metal::ceil_div(int64_t(params.outW), width);
  };
  const auto threadgroups_for = [&](int64_t width) {
    return c10::metal::ceil_div(Og, width >= kTile || (block == 32 && width == 32) ? block : kTile) * groups *
        tiles_for(width) * params.N;
  };
  int64_t width = kTile;
  if (block == kTile) {
    if (!flat && !pointwise && params.outW <= 32) {
      width = params.outW <= 16 ? 16 : 32;
    }
    while (width > 16 && threadgroups_for(width) < target) {
      width /= 2;
    }
  } else if (block == 16 && K <= 64 && (Wp > 0 ? tiles_for(1) : params.outW) >= 128) {
    width = 128;
  }
  if (!flat && !pointwise && block >= 32 && width == kTile && params.outW > 32 && narrow_better) {
    width = 32;
  }
  const int64_t pointwise_width = long_pointwise                            ? 128
      : wide_pointwise || compact_pointwise || (direct_columns && !apple10) ? 32
                                                                            : 64;
  const int64_t pointwise_simdgroups = long_pointwise ? 8 : direct_columns && apple10 ? 2 : 4;
  if (fast_pointwise) {
    width = pointwise_width;
  } else if (fixed3x3) {
    width = 64;
  }
  const int64_t simdgroups = fixed3x3                                   ? (Og > 64 ? 4 : 2)
      : fast_pointwise                                                  ? pointwise_simdgroups
      : width == kTile && (!apple10 || unfolded || (K > 16 && K <= 96)) ? 4
                                                                        : 2;
  const int64_t tiles = tiles_for(width), threadgroups = threadgroups_for(width);
  const int64_t k_split = fast_pointwise || fixed3x3
      ? K
      : c10::metal::round_up(
            c10::metal::ceil_div(int64_t(K), std::max<int64_t>(std::min<int64_t>(target / threadgroups, K / 256), 1)),
            int64_t(32));
  const int64_t splits = c10::metal::ceil_div(int64_t(K), k_split);
  const Conv2dMppParams mpp_params{.conv = params,
                                   .K = K,
                                   .K_split = static_cast<int32_t>(k_split),
                                   .splits = static_cast<int32_t>(splits),
                                   .Wp = Wp,
                                   .parity = parity ? static_cast<int32_t>(input.size(3) / 2 + 1) : 0};
  const int64_t pitch = Wp > 0 ? Wp : params.outW;
  const bool partials = splits > 1 || pitch != params.outW;
  const auto partial = partials
      ? at::empty({splits * params.N * params.C_out * params.outH * pitch}, output_t.options().dtype(kFloat))
      : output_t;
  const int64_t pointwise_rows = wide_pointwise ? 160 : compact_pointwise ? 128 : 64;
  const int64_t rows = fixed3x3 && Og > 64             ? 96
      : fast_pointwise                                 ? pointwise_rows
      : width >= kTile || (block == 32 && width == 32) ? block
                                                       : kTile;
  const bool staged = unfolded && in_nhwc && !apple10 && !out_nhwc && rows == kTile && width == kTile && !partials;
  auto pipeline = lib.getPipelineStateForFunc(fmt::format("conv2d_mpp_{}_{}_b{}x{}_s{}_{}{}",
                                                          in_nhwc ? "nhwc" : "nchw",
                                                          out_nhwc ? "nhwc" : "nchw",
                                                          rows,
                                                          width,
                                                          simdgroups,
                                                          fixed3x3             ? "k3_"
                                                              : fast_pointwise ? "pw1_"
                                                              : staged         ? "st_"
                                                                               : "",
                                                          dtype));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pipeline, "conv2d_mpp", {input, weight}, stream);
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, input, weight, output_t, mpp_params, bias, partial);
      [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div(Og, rows) * groups, tiles, params.N * splits)
              threadsPerThreadgroup:MTLSizeMake(simdgroups * 32, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline, stream);
      if (partials) {
        auto reduce = lib.getPipelineStateForFunc("conv2d_split_reduce_" + dtype);
        const auto image = int64_t(params.C_out) * params.outH * params.outW;
        const std::array<int32_t, 4> geometry{params.C_out, params.outH, params.outW, static_cast<int32_t>(pitch)};
        const std::array<int32_t, 4> counts{static_cast<int32_t>(splits), params.N, has_bias, out_nhwc};
        [encoder setComputePipelineState:reduce];
        mtl_setArgs(encoder, partial, bias, output_t, geometry, counts);
        [encoder dispatchThreads:MTLSizeMake(image, params.N, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(image, 256), 1, 1)];
      }
    }
  });
}

// Winograd products are independent GEMMs over chunks of transformed tiles.
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
  const int64_t coefficients = 4 * (tile_h + 2);
  const int64_t chunk = std::min(
      tiles,
      std::max<int64_t>(64, (int64_t(64) << 20) / (coefficients * std::max(C, O) * input.element_size()) / 64 * 64));
  const auto dtype = scalarToMetalTypeString(input);
  const auto weight = weight_t.contiguous();
  auto weights = at::empty({coefficients, O, C}, input.options());
  auto transformed = at::empty({1, coefficients * C, 1, chunk}, input.options());
  auto products = at::empty({1, coefficients * O, 1, chunk}, input.options());
  const std::array<int32_t, 2> wdims{static_cast<int32_t>(O), static_cast<int32_t>(C)};
  const std::array<int32_t, 4> igeom{static_cast<int32_t>(TH),
                                     static_cast<int32_t>(TW),
                                     static_cast<int32_t>(padding[0]),
                                     static_cast<int32_t>(padding[1])};
  auto wk = lib.getPipelineStateForFunc(fmt::format("winograd_weights_{}{}", tile_h == 4 ? "4_" : "", dtype));
  conv_metal_launch(wk, "winograd_weights", {weight}, [&](id<MTLComputeCommandEncoder> encoder) {
    mtl_setArgs(encoder, weight, weights, wdims);
    [encoder dispatchThreads:MTLSizeMake(C, O, 1) threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(C, 256), 1, 1)];
  });
  auto stream = getCurrentMPSStream();
  for (int64_t t0 = 0; t0 < tiles; t0 += chunk) {
    const int32_t count = static_cast<int32_t>(std::min(chunk, tiles - t0)), start = static_cast<int32_t>(t0);
    const auto v = transformed.as_strided({1, coefficients * C, 1, count}, {coefficients * C * count, count, count, 1});
    const auto m = products.as_strided({1, coefficients * O, 1, count}, {coefficients * O * count, count, count, 1});
    const std::array<int32_t, 4> idims{
        static_cast<int32_t>(C), count, static_cast<int32_t>(H), static_cast<int32_t>(W)};
    const std::array<int32_t, 4> odims{
        static_cast<int32_t>(O), count, static_cast<int32_t>(outH), static_cast<int32_t>(outW)};
    const std::array<int32_t, 4> ogeom{static_cast<int32_t>(TH), static_cast<int32_t>(TW), start, 0};
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        auto ik = lib.getPipelineStateForFunc(fmt::format("winograd_input_{}_{}",
                                                          tile_h == 4   ? "4"
                                                              : in_nhwc ? "nhwc"
                                                                        : "nchw",
                                                          dtype));
        getMPSProfiler().beginProfileKernel(ik, "winograd_input", {input}, stream);
        [encoder setComputePipelineState:ik];
        mtl_setArgs(encoder, input, v, idims, igeom, start);
        [encoder dispatchThreads:MTLSizeMake(count, C, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(count, 256), 1, 1)];
        getMPSProfiler().endProfileKernel(ik, stream);
        if (tile_h == 4) {
          const int64_t columns = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS) ? 64 : 32;
          const Conv2dMppParams gemm_params{
              .conv = {.N = 24, .C_in = static_cast<int32_t>(C), .C_out = static_cast<int32_t>(O), .H = 1, .W = count}};
          auto pk = lib.getPipelineStateForFunc(fmt::format("conv2d_mpp_nchw_nchw_b64x{}_s4_pw2_bfloat", columns));
          getMPSProfiler().beginProfileKernel(pk, "winograd_products", {v, weights}, stream);
          [encoder setComputePipelineState:pk];
          mtl_setArgs(encoder, v, weights, m, gemm_params, input, m);
          [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div(O, int64_t(64)),
                                                    c10::metal::ceil_div(int64_t(count), columns),
                                                    coefficients)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          getMPSProfiler().endProfileKernel(pk, stream);
        }
      }
    });
    if (tile_h == 2) {
      conv2d_mpp_forward(v, weights.view({16 * O, C, 1, 1}), std::nullopt, {0, 0}, {1, 1}, {1, 1}, 16, m);
    }
    auto ok = lib.getPipelineStateForFunc(fmt::format("winograd_output_{}_{}",
                                                      tile_h == 4    ? "4"
                                                          : out_nhwc ? "nhwc"
                                                                     : "nchw",
                                                      dtype));
    conv_metal_launch(ok, "winograd_output", {m}, [&](id<MTLComputeCommandEncoder> encoder) {
      mtl_setArgs(encoder, m, has_bias ? std::optional<Tensor>(bias) : std::nullopt, output_t, odims, ogeom);
      [encoder dispatchThreads:MTLSizeMake(count, O, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min<int64_t>(count, 256), 1, 1)];
    });
  }
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
                                    std::optional<IntArrayRef> input_shape) {
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

  auto output_t =
      at::empty(input_shape.has_value() ? input_shape.value()
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
  if (is1DConv) {
    conv1d_metal_forward(input_t, weight_t, bias_opt, padding[1], stride[1], dilation[1], groups, output_t);
    return output_t;
  }
  if (groups > 1 && groups == input_t.size(1) && weight_t.size(1) == 1) {
    const Conv2DParams params{
        .N = static_cast<int32_t>(input_t.size(0)),
        .C_in = static_cast<int32_t>(input_t.size(1)),
        .C_out = static_cast<int32_t>(output_t.size(1)),
        .H = static_cast<int32_t>(input_t.size(2)),
        .W = static_cast<int32_t>(input_t.size(3)),
        .outH = static_cast<int32_t>(output_t.size(2)),
        .outW = static_cast<int32_t>(output_t.size(3)),
        .kH = static_cast<int32_t>(weight_t.size(2)),
        .kW = static_cast<int32_t>(weight_t.size(3)),
        .sH = static_cast<int32_t>(stride[0]),
        .sW = static_cast<int32_t>(stride[1]),
        .padH = static_cast<int32_t>(padding[0]),
        .padW = static_cast<int32_t>(padding[1]),
        .dH = static_cast<int32_t>(dilation[0]),
        .dW = static_cast<int32_t>(dilation[1]),
        .C_in_per_group = 1,
        .C_out_per_group = static_cast<int32_t>(output_t.size(1) / groups),
        .has_bias = bias_defined,
    };
    conv2d_depthwise_forward(input_t, weight_t, bias_opt, params, output_t);
    return output_t;
  }
  if (has_mpp()) {
    if (input_t.size(1) == 0) {
      bias_defined ? output_t.copy_(bias_opt->reshape({1, -1, 1, 1}).expand_as(output_t)) : output_t.zero_();
    } else {
      // Winograd wins once both channel counts are large enough to amortize the
      // transformed-tensor traffic; the tensor-core GPUs need more work per byte.
      const int64_t C_in = input_t.size(1), C_out = weight_t.size(0), channels = std::min(C_in, C_out);
      const bool apple10 = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
      const bool worth = apple10 ? channels >= 160 || (channels >= 128 && input_t.numel() / C_in <= 16384)
                                 : channels >= 96 || (channels >= 48 && C_in * C_out >= 11000);
      const bool fused_mid = !apple10 && is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS) &&
          input_t.scalar_type() == kBFloat16 && input_t.is_contiguous() && output_t.is_contiguous() && C_in >= 16 &&
          input_t.numel() / C_in >= 16384 &&
          ((C_in <= 192 && C_out >= 32 && channels < 128) || (C_in >= 256 && C_in <= 512 && C_out == 32));
      const int64_t plane = input_t.size(2) * input_t.size(3);
      const bool single_tall = input_t.size(0) == 1 &&
          ((C_out == 128 && plane >= (apple10 ? 512 : 4096) && (plane <= 32768 || (!apple10 && C_in <= 192))) ||
           (!apple10 && C_in <= 192 && C_out == 64 && plane >= 16384) ||
           (apple10 && C_in <= 128 && C_out > 128 && C_out <= 256 && plane >= 32768 && plane < 262144));
      const bool tall_wino = is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS) &&
          input_t.scalar_type() == kBFloat16 && input_t.is_contiguous() && output_t.is_contiguous() && C_in >= 128 &&
          (single_tall ||
           (input_t.size(0) >= 8 && C_out >= (apple10 ? 128 : 32) && C_out <= 128 &&
            std::max(input_t.size(2), input_t.size(3)) <= 80));
      const bool winograd = groups == 1 && weight_t.size(2) == 3 && weight_t.size(3) == 3 && stride[0] == 1 &&
          stride[1] == 1 && dilation[0] == 1 && dilation[1] == 1 && input_t.scalar_type() != kHalf &&
          (worth || tall_wino);
      if (winograd && (!fused_mid || tall_wino)) {
        conv2d_winograd_forward(input_t, weight_t, bias_opt, padding, output_t, tall_wino ? 4 : 2);
      } else {
        conv2d_mpp_forward(input_t, weight_t, bias_opt, padding, stride, dilation, groups, output_t, fused_mid);
      }
    }
    return output_t;
  }
  // Conv2d is conv3d with unit depth; reuse its generic simdgroup fallback.
  conv3d_metal_forward(input_t.unsqueeze(2),
                       weight_t.unsqueeze(2),
                       bias_opt,
                       {0, padding[0], padding[1]},
                       {1, stride[0], stride[1]},
                       {1, dilation[0], dilation[1]},
                       groups,
                       output_t.unsqueeze(2));
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
