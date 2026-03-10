#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/lerp_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Lerp_metallib.h>
#endif

TORCH_IMPL_FUNC(lerp_Tensor_mps)(const Tensor& self, const Tensor& end, const Tensor& weight, const Tensor& out) {
  TORCH_CHECK(out.is_mps());
  if (self.numel() == 0) {
    return;
  }
  using namespace mps;

  auto common_shape = at::infer_size_dimvector(at::infer_size_dimvector(self.sizes(), end.sizes()), weight.sizes());
  auto self_expanded = self.expand(common_shape).contiguous();
  auto end_expanded = end.expand(common_shape).contiguous();
  auto weight_expanded = weight.expand(common_shape).contiguous();

  auto pso = lib.getPipelineStateForFunc("lerp_tensor_kernel_" + scalarToMetalTypeString(out));
  auto numel = static_cast<uint32_t>(out.numel());

  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    auto computeEncoder = getCurrentMPSStream()->commandEncoder();
    [computeEncoder setComputePipelineState:pso];
    mtl_setArgs(computeEncoder, self_expanded, end_expanded, weight_expanded, out);
    mtl_dispatch1DJob(computeEncoder, pso, numel);
  });
}

} // namespace at::native
