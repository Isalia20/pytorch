#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <iostream>
#include <optional>

#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/empty_native.h>
#endif

namespace at {
namespace native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LinearAlgebra_metallib.h>
#endif

// expand potential 3d to 4d tensor
static inline std::tuple<Tensor, bool> ensure_4d(const Tensor& x) {
  if (x.dim() == 3) {
    return {x.unsqueeze(0), true};
  } else if (x.dim() > 4) {
    auto batchSize = c10::multiply_integers(x.sizes().begin(), x.sizes().end() - 3);
    return {x.view({batchSize, x.size(-3), x.size(-2), x.size(-1)}), true};
  } else {
    return {x, false};
  }
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                const Tensor& key,
                                                                const Tensor& value,
                                                                const std::optional<Tensor>& attn_mask,
                                                                double dropout_p,
                                                                bool is_causal,
                                                                const std::optional<Tensor>& dropout_mask,
                                                                std::optional<double> scale) {
  using namespace mps;
  TORCH_CHECK(query.is_mps() && key.is_mps() && value.is_mps(), 
              "_scaled_dot_product_attention_math_mps: All input tensors must be MPS tensors");
  
  if (is_causal) {
    TORCH_CHECK(!attn_mask.has_value(),
                "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
  }

  TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_mps: dropout_p != 0.0 is not supported");
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
              "_scaled_dot_product_attention_math_mps: query, key, and value must not be nested");

  // Ensure 4D tensors
  auto [q_, sq] = ensure_4d(query);
  auto [k_, sk] = ensure_4d(key);
  auto [v_, sv] = ensure_4d(value);
  // Prepare mask if needed
  std::optional<Tensor> mask_;
  if (attn_mask) {
    auto maskExpandedDims = query.sizes().vec();
    maskExpandedDims[maskExpandedDims.size() - 1] = k_.size(2);
    mask_ = attn_mask->expand(maskExpandedDims);
    std::tie(*mask_, std::ignore) = ensure_4d(*mask_);
  }

  // Get dimensions
  int64_t batchSize = q_.size(0);
  int64_t num_head = q_.size(1);
  int64_t seq_len_q = q_.size(2);
  int64_t head_dim = q_.size(-1);
  int64_t seq_len_k = k_.size(2);
  
  // Calculate scale factor
  auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
  
  // Create output tensors
  auto out = at::empty({batchSize, num_head, seq_len_q, seq_len_k}, query.options());
  auto attn = at::empty({batchSize, num_head, seq_len_q, seq_len_k}, query.options());
  
  // Empty tensors optimization
  if (q_.numel() == 0 || k_.numel() == 0 || v_.numel() == 0) {
    out.zero_();
    attn.zero_();
    // Reshape back to original dimension
    auto final_out = sq ? out.view_as(query) : out;
    auto final_attn = sq ? (query.dim() == 3 ? attn.squeeze(0) : [&]{
      std::vector<int64_t> shape(query.sizes().begin(), query.sizes().end() - 3);
      shape.insert(shape.end(), {attn.size(1), attn.size(2), attn.size(3)});
      return attn.view(shape);
    }()) : attn;
    return {std::move(final_out), std::move(final_attn)};
  }
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto attentionPSO = lib.getPipelineStateForFunc("attention");

  const uint matrix_dim = query.size(-1);
  uint32_t tileSize = 8;
  uint32_t gridY = seq_len_q / tileSize;
  uint32_t gridZ = batchSize * num_head;
  uint32_t out_blocks_per_simdgroup = (seq_len_k + 255) / 256;

  MTLSize threadGroupSize = MTLSizeMake(32, std::min<uint32_t>(32, seq_len_k / tileSize), 1);
  MTLSize gridSize = MTLSizeMake(1, gridY, gridZ);

  Tensor q_ref = q_;
  Tensor k_ref = k_.transpose(-2, -1);
  Tensor v_ref = v_;
  std::optional<Tensor> mask_ref = mask_;
  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:attentionPSO];
      mtl_setArgs(computeEncoder, q_ref, k_ref, v_ref, mask_ref, out, num_head, seq_len_q, seq_len_k, head_dim, scale_factor, is_causal, out_blocks_per_simdgroup);
      [computeEncoder setThreadgroupMemoryLength:head_dim * 8 * sizeof(float) atIndex:0];
      [computeEncoder setThreadgroupMemoryLength:seq_len_k * 8 * sizeof(float) atIndex:1];
      [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    });
  }

  // Reshape back to original dimension
  auto final_out = sq ? out.view_as(query) : out;
  auto final_attn = sq ? (query.dim() == 3 ? attn.squeeze(0) : [&]{
    std::vector<int64_t> shape(query.sizes().begin(), query.sizes().end() - 3);
    shape.insert(shape.end(), {attn.size(1), attn.size(2), attn.size(3)});
    return attn.view(shape);
  }()) : attn;

  return {std::move(final_out), std::move(final_attn)};
}

} // namespace native
} // namespace at
