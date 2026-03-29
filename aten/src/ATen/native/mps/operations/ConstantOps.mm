//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fill_native.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zero_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ConstantOps_metallib.h>
#endif

static void packScalar(const Scalar& value, ScalarType dtype, void* out, size_t* size) {
  *size = c10::elementSize(dtype);
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, dtype, "fill_scalar_mps", [&] {
    *static_cast<scalar_t*>(out) = value.to<scalar_t>();
  });
}

static Tensor& fill_scalar_mps_impl(Tensor& self, const Scalar& value) {
  if (self.numel() == 0)
    return self;

  std::array<uint8_t, 8> scalarBytes{};
  size_t scalarSize = 0;
  packScalar(value, self.scalar_type(), scalarBytes.data(), &scalarSize);

  auto typeStr = scalarToMetalTypeString(self);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

  if (self.is_contiguous()) {
    bool useVec4 = self.element_size() <= 2;
    auto key = (useVec4 ? "fill_scalar_vec4_" : "fill_scalar_") + typeStr;
    id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);

    dispatch_sync(mpsStream->queue(), ^() {
      @autoreleasepool {
        getMPSProfiler().beginProfileKernel(pso, key, {self});

        [computeEncoder setComputePipelineState:pso];
        mtl_setBuffer(computeEncoder, self, 0);
        [computeEncoder setBytes:scalarBytes.data() length:scalarSize atIndex:1];
        if (useVec4) {
          uint32_t numel = static_cast<uint32_t>(self.numel());
          [computeEncoder setBytes:&numel length:sizeof(numel) atIndex:2];
          mtl_dispatch1DJob(computeEncoder, pso, static_cast<NSUInteger>((self.numel() + 3) / 4));
        } else {
          mtl_dispatch1DJob(computeEncoder, pso, self.numel());
        }

        getMPSProfiler().endProfileKernel(pso);
      }
    });
  } else if (self.dim() == 2) {
    auto strides_ref = self.strides();
    auto sizes_ref = self.sizes();
    // Map thread.x to the dimension with smallest stride for memory coalescing
    int fast_dim = (std::abs(strides_ref[0]) <= std::abs(strides_ref[1])) ? 0 : 1;
    int slow_dim = 1 - fast_dim;
    long stride_fast = strides_ref[fast_dim];
    long stride_slow = strides_ref[slow_dim];
    NSUInteger fast_size = sizes_ref[fast_dim];
    NSUInteger slow_size = sizes_ref[slow_dim];
    bool useVec4 = stride_fast == 1 && self.element_size() <= 2;

    auto key = (useVec4 ? "fill_scalar_2d_vec4_" : "fill_scalar_2d_") + typeStr;
    id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);

    dispatch_sync(mpsStream->queue(), ^() {
      @autoreleasepool {
        getMPSProfiler().beginProfileKernel(pso, key, {self});

        [computeEncoder setComputePipelineState:pso];
        mtl_setBuffer(computeEncoder, self, 0);
        [computeEncoder setBytes:scalarBytes.data() length:scalarSize atIndex:1];

        NSUInteger grid_x;
        if (useVec4) {
          [computeEncoder setBytes:&stride_slow length:sizeof(long) atIndex:2];
          uint32_t fs = static_cast<uint32_t>(fast_size);
          [computeEncoder setBytes:&fs length:sizeof(fs) atIndex:3];
          grid_x = (fast_size + 3) / 4;
        } else {
          [computeEncoder setBytes:&stride_fast length:sizeof(long) atIndex:2];
          [computeEncoder setBytes:&stride_slow length:sizeof(long) atIndex:3];
          grid_x = fast_size;
        }

        auto maxThreads = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger tgX = std::min(grid_x, maxThreads);
        [computeEncoder dispatchThreads:MTLSizeMake(grid_x, slow_size, 1)
                 threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];

        getMPSProfiler().endProfileKernel(pso);
      }
    });
  } else {
    auto key = "fill_scalar_strided_" + typeStr;
    id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);
    auto strides = self.strides();
    auto sizes = self.sizes();
    int32_t ndim = static_cast<int32_t>(self.dim());

    dispatch_sync(mpsStream->queue(), ^() {
      @autoreleasepool {
        getMPSProfiler().beginProfileKernel(pso, key, {self});

        [computeEncoder setComputePipelineState:pso];
        mtl_setBuffer(computeEncoder, self, 0);
        [computeEncoder setBytes:scalarBytes.data() length:scalarSize atIndex:1];
        mtl_setBytes(computeEncoder, strides, 2);
        mtl_setBytes(computeEncoder, sizes, 3);
        mtl_setBytes(computeEncoder, ndim, 4);
        mtl_dispatch1DJob(computeEncoder, pso, self.numel());

        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }

  return self;
}

static Tensor& fill_mps_tensor_(Tensor& self, uint8_t value) {
  TORCH_INTERNAL_ASSERT(self.is_contiguous());
  const auto stream = getCurrentMPSStream();
  auto storage_byte_offset = self.storage_offset() * self.itemsize();
  stream->fill(getMTLBufferStorage(self), value, self.nbytes(), storage_byte_offset);
  return self;
}

Tensor& fill_scalar_mps(Tensor& self, const Scalar& value) {
  if (isComplexType(self.scalar_type())) {
    auto self_as_real = at::view_as_real(self);
    auto self_as_real_real = self_as_real.select(self.dim(), 0);
    auto self_as_real_imag = self_as_real.select(self.dim(), 1);
    if (value.isComplex()) {
      auto value_cdouble = value.to<c10::complex<double>>();
      fill_scalar_mps_impl(self_as_real_real, value_cdouble.real());
      fill_scalar_mps_impl(self_as_real_imag, value_cdouble.imag());
      return self;
    }
    fill_scalar_mps_impl(self_as_real_real, value);
    fill_scalar_mps_impl(self_as_real_imag, 0.0f);
    return self;
  }
  // check if it's possible to use fillBuffer() to fill the Tensor's storage
  if (self.is_contiguous()) {
    if (value.toDouble() == 0.0) {
      return fill_mps_tensor_(self, 0);
    }
    if (self.scalar_type() == kBool) {
      return fill_mps_tensor_(self, value.toBool());
    }
    if (self.scalar_type() == kByte) {
      return fill_mps_tensor_(self, value.toByte());
    }
    if (self.scalar_type() == kChar) {
      return fill_mps_tensor_(self, value.toChar());
    }
  }

  return fill_scalar_mps_impl(self, value);
}

Tensor& fill_tensor_mps_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0,
              "fill_ only supports 0-dimension value tensor but got tensor with ",
              value.dim(),
              " dimensions.");
  Scalar scalar_value = value.item();
  return fill_scalar_mps(self, scalar_value);
}

Tensor& zero_mps_(Tensor& self) {
  return fill_scalar_mps(self, 0.0f);
}

} // namespace at::native
