//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#endif

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Sort_metallib.h>
#endif

static constexpr int TN = 4; // elements per thread

// When n_rows is small, large BN leaves the GPU underutilized because the
// block sort dispatches only n_rows * ceil(sort_size / NPB) threadgroups.
// Lower BN increases the block count, producing more concurrent TGs.
static int select_bn(int sort_size, size_t elem_size, int n_rows) {
  int potential_bn = at::ceil_div(sort_size, TN);

  // Aim for at least ~32 concurrent threadgroups across all rows.
  // Only kicks in above the single-block sweet spot (sort_size > NPB at BN=1024);
  // below that, single-block's lower dispatch overhead wins despite 1 TG.
  if (n_rows <= 2 && sort_size > 4096) {
    constexpr int target_total_tgs = 32;
    int target_blocks_per_row = std::max(1, target_total_tgs / std::max(n_rows, 1));
    int target_npb = std::max(512, at::ceil_div(sort_size, target_blocks_per_row));
    int target_bn = target_npb / TN;
    potential_bn = std::min(potential_bn, target_bn);
  }

  int bn;
  if (potential_bn > 512)
    bn = 1024;
  else if (potential_bn > 256)
    bn = 512;
  else if (potential_bn > 128)
    bn = 256;
  else if (potential_bn > 64)
    bn = 128;
  else if (potential_bn > 32)
    bn = 64;
  else
    bn = 32;
  // For 8-byte types (int64), cap at 256 to fit in threadgroup memory:
  // 256 * 4 * (8 + 4) = 12KB < 32KB (uint32 indices)
  if (elem_size > 4) {
    bn = std::min(bn, 256);
  }
  // BN=1024: NPB=4096, threadgroup = 4096*(elem_size+4)
  // For 4-byte types: 4096*8=32KB (limit). For 8-byte: doesn't fit.
  if (bn == 1024 && elem_size > 4)
    bn = 512;
  // Occupancy tweak: at BN=1024 the per-TG threadgroup memory (≈ NPB*(T+uint))
  // is 24KB (fp16) / 32KB (fp32), which pins 1 TG/core on Apple GPUs. Stepping
  // down to BN=512 halves threadgroup memory so 2 TGs run per core. Only do
  // this once the sort is multi-block and we have many rows — otherwise the
  // extra merge round costs more than occupancy saves.
  int bn1024_n_blocks = at::ceil_div(sort_size, 1024 * TN);
  if (bn == 1024 && bn1024_n_blocks > 1 && n_rows >= 8) {
    bn = 512;
  }
  return bn;
}

// Single-block sort: the whole sort dimension fits in one threadgroup.
static void sort_single_block(const Tensor& values,
                              const Tensor& indices,
                              int64_t dim,
                              bool descending,
                              int sort_size,
                              int bn) {
  int64_t inner_size = 1;
  for (int64_t i = dim + 1; i < values.ndimension(); i++)
    inner_size *= values.size(i);
  int n_rows = static_cast<int>(values.numel() / sort_size);
  int64_t stride_sorted = inner_size;

  // For the segment stride: distance between row 0 and row 1 of the sort.
  // For contiguous values sorted along a middle dim, this is the product of
  // all dims from 0..dim (exclusive) times the inner block.  But the simplest
  // formulation: stride_segment = values.stride(dim_before_sort_dim), i.e.
  // we need to jump by one unit in the "next outer" dimension.
  // Since values is contiguous and we iterate rows with tid.y, we set
  // stride_segment = sort_size * inner_size when dim > 0.
  // When dim == 0 and there are no outer dims, stride_segment is unused.
  // Actually, for the general case: the n_rows rows correspond to all
  // (outer_idx, inner_idx) pairs. We linearize them as:
  //   row = outer_idx * inner_size + inner_idx
  //   base = outer_idx * sort_size * inner_size + inner_idx
  //        = (row / inner_size) * sort_size * inner_size + (row % inner_size)
  // This can't be expressed as a single stride_segment. Let's fall back to
  // a layout where we handle the addressing ourselves.
  //
  // Simplification: for contiguous tensors, stride_segment is the stride for
  // the outer-most dimension that groups rows, but only works cleanly when
  // dim is the last dimension (stride_sorted=1, stride_segment=sort_size).
  // For general dims, we need inner_size.  The kernel computes:
  //   base = tid.y * stride_segment
  // So for last-dim sort: stride_segment = sort_size, stride_sorted = 1. ✓
  // For mid-dim sort of shape [A, sort, B]:
  //   row r → (outer=r/B, inner=r%B) → base = outer*sort*B + inner
  //   = (r/B)*sort*B + r%B
  //   This is NOT r * stride_segment for any single stride_segment.
  //
  // Solution: fall back to multi-block path for non-last-dim sorts with
  // sort_size > 1, OR compute base in kernel from row index. Let's use the
  // multi-block path for all sort dims (it handles contiguous intermediate
  // buffers) and reserve single-block for last-dim only.

  // This kernel only works when dim is the last dimension (stride_sorted = 1).
  // For other dims, we go through multi_block which uses contiguous intermediates.
  int64_t stride_segment = sort_size; // distance between consecutive rows

  const std::string kernel =
      "sort_block_" + scalarToMetalTypeString(values) + "_bn" + std::to_string(bn);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {values});

      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, values, indices,
                  sort_size, stride_sorted, stride_segment, descending);

      [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1)
           threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Multi-block sort: block sort → iterative partition+merge.
static void sort_multi_block(const Tensor& values,
                             const Tensor& indices,
                             int64_t dim,
                             bool descending,
                             int sort_size,
                             int bn) {
  int npb = bn * TN;
  int n_blocks = at::ceil_div(sort_size, npb);
  int n_rows = static_cast<int>(values.numel() / sort_size);

  int64_t stride_sorted = 1;
  int64_t stride_segment = sort_size;
  // We always work on contiguous intermediates, so stride = 1.

  // Allocate intermediate buffers (contiguous, [n_rows, sort_size]).
  // Use uint32 for indices internally to halve memory bandwidth;
  // convert to int64 only on final output.
  auto opts_val = values.options();
  auto opts_u32 = at::TensorOptions().dtype(at::kInt).device(values.device());

  Tensor dev_vals_0 = at::empty({n_rows, sort_size}, opts_val);
  Tensor dev_vals_1 = at::empty({n_rows, sort_size}, opts_val);
  Tensor dev_idxs_0 = at::empty({n_rows, sort_size}, opts_u32);
  Tensor dev_idxs_1 = at::empty({n_rows, sort_size}, opts_u32);
  Tensor block_parts = at::empty({n_rows, n_blocks + 1}, opts_u32);

  MPSStream* mpsStream = getCurrentMPSStream();
  const std::string type_str = scalarToMetalTypeString(values);
  const std::string bn_str = "_bn" + std::to_string(bn);

  // Compute strides for reading strided input
  int64_t inner_size = 1;
  for (int64_t i = dim + 1; i < values.ndimension(); i++)
    inner_size *= values.size(i);
  int64_t in_stride_sorted = inner_size;
  // For the segment stride of the INPUT (values): distance between rows.
  // row r → base = (r / inner_size) * sort_size * inner_size + (r % inner_size)
  // This isn't a single stride. We handle it by passing inner_size and computing
  // in the kernel.  BUT our kernel expects a simple stride_segment.
  // Workaround: permute the input so the sort dim is last, making it contiguous.
  // Since values is already a contiguous copy of self, we can reshape.

  // Actually: let's just handle the general case. For the mb_sort_block kernel,
  // we pass stride_sorted_axis and stride_segment. stride_segment should give us
  // the row offset. For last dim, stride_sorted=1, stride_segment=sort_size.
  // For other dims: not a simple stride.
  //
  // Fix: if dim != last dim, permute values so sort dim is last, sort, permute back.
  bool need_permute = (dim != values.ndimension() - 1);
  Tensor work_vals = values;
  if (need_permute) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < values.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    work_vals = values.permute(perm).contiguous();
    // Now work_vals has sort dim last, shape [..., sort_size], contiguous
    sort_size = work_vals.size(work_vals.ndimension() - 1);
    n_rows = static_cast<int>(work_vals.numel() / sort_size);
    in_stride_sorted = 1;

    dev_vals_0 = at::empty({n_rows, sort_size}, opts_val);
    dev_vals_1 = at::empty({n_rows, sort_size}, opts_val);
    dev_idxs_0 = at::empty({n_rows, sort_size}, opts_u32);
    dev_idxs_1 = at::empty({n_rows, sort_size}, opts_u32);
    n_blocks = at::ceil_div(sort_size, npb);
    block_parts = at::empty({n_rows, n_blocks + 1}, opts_u32);
  }

  // Phase 1: Block sort
  {
    const std::string kernel = "mb_sort_block_" + type_str + bn_str;
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(pso, kernel, {work_vals});

        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, work_vals, dev_vals_0, dev_idxs_0,
                    sort_size, in_stride_sorted,
                    static_cast<int64_t>(sort_size), // stride_segment (contiguous)
                    descending);

        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }

  // Phase 2: Iterative merge
  if (n_blocks > 1) {
    bool ping = false;
    int n_thr_partition = std::min(n_blocks + 1, 1024);

    const std::string part_kernel = "mb_partition_" + type_str + bn_str;
    const std::string merge_kernel = "mb_merge_" + type_str + bn_str;
    const std::string merge_final_kernel = "mb_merge_final_" + type_str + bn_str;

    // Count merge rounds to know which is last.
    int total_rounds = 0;
    for (int m = 2; (m / 2) < n_blocks; m *= 2) ++total_rounds;
    const bool direct_final_write = !need_permute;
    int cur_round = 0;

    for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2, ++cur_round) {
      const Tensor& v_in = ping ? dev_vals_1 : dev_vals_0;
      const Tensor& i_in = ping ? dev_idxs_1 : dev_idxs_0;
      const Tensor& v_out_buf = ping ? dev_vals_0 : dev_vals_1;
      const Tensor& i_out_buf = ping ? dev_idxs_0 : dev_idxs_1;
      const bool use_direct = direct_final_write && (cur_round == total_rounds - 1);
      ping = !ping;

      // Partition
      dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
          id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(part_kernel);
          [enc setComputePipelineState:pso];
          mtl_setArgs(enc, block_parts, v_in,
                      sort_size, merge_tiles, n_blocks, descending);
          [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1)
               threadsPerThreadgroup:MTLSizeMake(n_thr_partition, 1, 1)];
        }
      });

      // Merge: last round goes straight to values/indices (int64 output)
      // when we don't need to un-permute afterwards.
      dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
          id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(
              use_direct ? merge_final_kernel : merge_kernel);
          [enc setComputePipelineState:pso];
          if (use_direct) {
            mtl_setArgs(enc, block_parts, v_in, i_in, values, indices,
                        sort_size, merge_tiles, n_blocks, descending);
          } else {
            mtl_setArgs(enc, block_parts, v_in, i_in, v_out_buf, i_out_buf,
                        sort_size, merge_tiles, n_blocks, descending);
          }
          [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
               threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];
        }
      });
    }

    if (direct_final_write) {
      // Already written directly to values/indices in the last merge round.
      return;
    }
    // Final result is in the last output buffer
    // ping was flipped AFTER selecting v_out, so the last output is
    // in dev_vals_1 when ping=true, dev_vals_0 when ping=false.
    const Tensor& final_vals = ping ? dev_vals_1 : dev_vals_0;
    const Tensor& final_idxs = ping ? dev_idxs_1 : dev_idxs_0;

    // Only reached in the need_permute case (direct_final_write is false).
    auto final_v_view = final_vals.view(work_vals.sizes());
    auto final_i_view = final_idxs.view(work_vals.sizes());
    std::vector<int64_t> perm, inv_perm(values.ndimension());
    for (int64_t i = 0; i < values.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    for (int64_t i = 0; i < values.ndimension(); i++)
      inv_perm[perm[i]] = i;
    values.copy_(final_v_view.permute(inv_perm));
    indices.copy_(final_i_view.permute(inv_perm));
  } else {
    // Single block, result already in dev_vals_0/dev_idxs_0
    if (need_permute) {
      auto final_v_view = dev_vals_0.view(work_vals.sizes());
      auto final_i_view = dev_idxs_0.view(work_vals.sizes());
      std::vector<int64_t> perm, inv_perm(values.ndimension());
      for (int64_t i = 0; i < values.ndimension(); i++)
        if (i != dim) perm.push_back(i);
      perm.push_back(dim);
      for (int64_t i = 0; i < values.ndimension(); i++)
        inv_perm[perm[i]] = i;
      values.copy_(final_v_view.permute(inv_perm));
      indices.copy_(final_i_view.permute(inv_perm));
    } else {
      values.copy_(dev_vals_0.view(values.sizes()));
      indices.copy_(dev_idxs_0.view(indices.sizes()));
    }
  }
}

// Radix sort: 4-bit radix, O(n) per pass. Faster than merge sort for large arrays.
static void sort_radix(const Tensor& values,
                       const Tensor& indices,
                       int64_t dim,
                       bool descending,
                       int sort_size) {
  bool need_permute = (dim != values.ndimension() - 1);
  Tensor work_vals = values;
  if (need_permute) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < values.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    work_vals = values.permute(perm).contiguous();
    sort_size = work_vals.size(work_vals.ndimension() - 1);
  }

  int n_rows = static_cast<int>(work_vals.numel() / sort_size);
  // 16-bit and 32-bit types use 8-bit radix (256 bins) — halves global memory
  // traffic vs 4-bit radix while keeping the same total block-local work
  // (RBITS binary sub-passes × n_passes = total bits either way). 8-bit types
  // stay on 4-bit radix since 2 passes is already the minimum meaningful pass
  // count at that width.
  const size_t elem_size = values.element_size();
  const int radix_bits = (elem_size >= 2) ? 8 : 4;
  const int radix_size = 1 << radix_bits;
  int n_passes;
  if (elem_size == 1) n_passes = 2;       // 4-bit × 2 passes
  else if (elem_size == 2) n_passes = 2;  // 8-bit × 2 passes
  else if (elem_size == 4) n_passes = 4;  // 8-bit × 4 passes
  else TORCH_CHECK(false, "Radix sort not supported for element size ", elem_size);

  constexpr int RADIX_BN = 512;
  const int radix_ept = (elem_size >= 4) ? 4 : 8;
  const int RADIX_NPB = RADIX_BN * radix_ept;
  int n_blocks = at::ceil_div(sort_size, RADIX_NPB);
  int n_entries = radix_size * n_blocks;

  auto opts_val = work_vals.options();
  auto opts_u32 = at::TensorOptions().dtype(at::kInt).device(values.device());

  Tensor keys_0 = work_vals.reshape({n_rows, sort_size}).contiguous();
  Tensor keys_1 = at::empty({n_rows, sort_size}, opts_val);
  Tensor idxs_0 = at::empty({n_rows, sort_size}, opts_u32);
  Tensor idxs_1 = at::empty({n_rows, sort_size}, opts_u32);
  Tensor histograms = at::empty({n_rows, n_entries}, opts_u32);

  const std::string type_str = scalarToMetalTypeString(values);
  const std::string rbits_suffix = "_" + std::to_string(radix_bits) + "bit";
  const std::string count_kernel = "radix_count_" + type_str + rbits_suffix;
  const std::string scatter_kernel = "radix_scatter_" + type_str + rbits_suffix;
  const std::string scatter_final_kernel = "radix_scatter_final_" + type_str + rbits_suffix;

  // If we don't need a post-sort permute, the last scatter pass can write
  // directly into the caller's `values`/`indices` buffers (with int64 indices)
  // and we skip the final uint32→int64 + element-wise copy.
  const bool direct_final_write = !need_permute;

  MPSStream* mpsStream = getCurrentMPSStream();
  bool ping = false; // false: src=0, true: src=1

  for (int pass = 0; pass < n_passes; pass++) {
    int shift = pass * radix_bits;
    bool first_pass = (pass == 0);
    bool last_pass = (pass == n_passes - 1);
    bool use_direct = direct_final_write && last_pass;

    const Tensor& k_in = ping ? keys_1 : keys_0;
    const Tensor& i_in = ping ? idxs_1 : idxs_0;
    const Tensor& k_out_buf = ping ? keys_0 : keys_1;
    const Tensor& i_out_buf = ping ? idxs_0 : idxs_1;

    // Count
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(count_kernel);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, k_in, histograms, sort_size, n_blocks, shift, descending);
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(RADIX_BN, 1, 1)];
      }
    });

    // Scan (parallel, SCAN_BN=256 threads per row)
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc("radix_scan");
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, histograms, n_entries);
        [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
    });

    // Scatter: route the last pass through the int64-output kernel when
    // the destination is already the final output buffer.
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(
            use_direct ? scatter_final_kernel : scatter_kernel);
        [enc setComputePipelineState:pso];
        if (use_direct) {
          mtl_setArgs(enc, k_in, i_in, values, indices,
                      histograms, sort_size, n_blocks, shift, descending, first_pass);
        } else {
          mtl_setArgs(enc, k_in, i_in, k_out_buf, i_out_buf,
                      histograms, sort_size, n_blocks, shift, descending, first_pass);
        }
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(RADIX_BN, 1, 1)];
      }
    });

    ping = !ping;
  }

  if (direct_final_write) {
    // Already written directly to values/indices.
    return;
  }

  const Tensor& final_keys = ping ? keys_1 : keys_0;
  const Tensor& final_idxs = ping ? idxs_1 : idxs_0;

  auto fk_view = final_keys.view(work_vals.sizes());
  auto fi_view = final_idxs.view(work_vals.sizes());
  std::vector<int64_t> perm, inv_perm(values.ndimension());
  for (int64_t i = 0; i < values.ndimension(); i++)
    if (i != dim) perm.push_back(i);
  perm.push_back(dim);
  for (int64_t i = 0; i < values.ndimension(); i++)
    inv_perm[perm[i]] = i;
  values.copy_(fk_view.permute(inv_perm));
  indices.copy_(fi_view.permute(inv_perm));
}

// Threshold: use radix when its dispatch count isn't much worse than merge's.
// Radix has cheaper per-pass work (no log2(BN) internal merge rounds), so even
// with a few more dispatches the full-kernel cost is usually lower on MPS.
static bool should_use_radix(int sort_size, size_t elem_size, int n_rows) {
  if (elem_size > 4) return false; // int64: too many radix passes
  int n_blocks_merge = at::ceil_div(sort_size, 4096);
  int merge_rounds = 0;
  for (int m = 2; (m / 2) < n_blocks_merge; m *= 2) merge_rounds++;
  int merge_dispatches = 1 + 2 * merge_rounds;
  // 1 byte → 4-bit×2 = 2 passes; 2 byte → 8-bit×2 = 2 passes;
  // 4 byte → 8-bit×4 = 4 passes.
  int n_radix_passes = (elem_size <= 1) ? 2 : (elem_size <= 2) ? 2 : 4;
  int radix_dispatches = n_radix_passes * 3;
  return radix_dispatches <= merge_dispatches + 2;
}

void kthvalue_out_mps_impl(const Tensor& self, int64_t k, int64_t dim, Tensor& values, Tensor& indices) {
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }
  if (self.numel() == 0) {
    values.copy_(self);
    indices.copy_(values.toType(at::ScalarType::Long));
    return;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(self.scalar_type()), "kthvalue is not implemented for complex types");

  auto sorted = at::sort(self, dim, /*descending=*/false);
  auto sliced_vals = std::get<0>(sorted).select(dim, k - 1);
  auto sliced_inds = std::get<1>(sorted).select(dim, k - 1);
  values.copy_(sliced_vals.unsqueeze(dim));
  indices.copy_(sliced_inds.unsqueeze(dim));
}

} // anonymous namespace

// sort
TORCH_IMPL_FUNC(sort_stable_out_mps)
(const Tensor& self,
 std::optional<bool> stable,
 int64_t dim,
 bool descending,
 const Tensor& values,
 const Tensor& indices) {
  if (self.numel() == 0)
    return;

  dim = maybe_wrap_dim(dim, self.dim(), true);

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  int sort_size = static_cast<int>(self.size(dim));
  if (sort_size <= 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  TORCH_CHECK(!c10::isComplexType(self.scalar_type()),
              "Sort is not supported for complex types on MPS");

  // Prepare contiguous working buffers
  Tensor work_vals, work_inds;
  bool need_copy_back = false;
  if (values.is_contiguous() && indices.is_contiguous()) {
    values.copy_(self);
    work_vals = values;
    work_inds = indices;
  } else {
    work_vals = at::empty(self.sizes(), values.options());
    work_vals.copy_(self);
    work_inds = at::empty(self.sizes(), indices.options());
    need_copy_back = true;
  }

  int n_rows_for_bn = static_cast<int>(self.numel() / sort_size);
  int bn = select_bn(sort_size, self.element_size(), n_rows_for_bn);
  int npb = bn * TN;

  bool is_last_dim = (dim == self.ndimension() - 1);

  if (should_use_radix(sort_size, self.element_size(), n_rows_for_bn)) {
    sort_radix(work_vals, work_inds, dim, descending, sort_size);
  } else if (sort_size <= npb && is_last_dim) {
    sort_single_block(work_vals, work_inds, dim, descending, sort_size, bn);
  } else {
    sort_multi_block(work_vals, work_inds, dim, descending, sort_size, bn);
  }

  if (need_copy_back) {
    values.copy_(work_vals);
    indices.copy_(work_inds);
  }
}

std::tuple<Tensor&, Tensor&> kthvalue_out_mps(const Tensor& self,
                                              int64_t k,
                                              int64_t dim_,
                                              bool keepdim,
                                              Tensor& values,
                                              Tensor& indices) {
  at::globalContext().alertNotDeterministic("kthvalue MPS");

  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 1 && k <= slicesize, "kthvalue(): selected number k out of range for dimension ", dim);
  at::assert_no_overlap(self, values);
  _reduction_with_indices_allocate_or_resize_output(values, indices, self, dim, keepdim);

  kthvalue_out_mps_impl(self, k, dim, values, indices);

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }

  return std::forward_as_tuple(values, indices);
}
} // namespace at::native
