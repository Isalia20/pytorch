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

// Single-block sort (last-dim only). Reads `input` via (stride_sort, stride_seg),
// writes sorted values + int64 indices contiguously into values/indices.
static void sort_single_block(const Tensor& input,
                              const Tensor& values,
                              const Tensor& indices,
                              bool descending,
                              int sort_size,
                              int64_t stride_sort,
                              int64_t stride_seg,
                              int bn) {
  int n_rows = static_cast<int>(input.numel() / sort_size);
  const std::string kernel =
      "sort_block_" + scalarToMetalTypeString(input) + "_bn" + std::to_string(bn);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {input});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, input, values, indices,
                  sort_size, stride_sort, stride_seg, descending);
      [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1)
           threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Multi-block sort: block sort → iterative merge (partition fused into merge).
static void sort_multi_block(const Tensor& input,
                             const Tensor& values,
                             const Tensor& indices,
                             int64_t dim,
                             bool descending,
                             int sort_size,
                             int64_t in_stride_sort,
                             int64_t in_stride_seg,
                             int bn) {
  int npb = bn * TN;
  int n_blocks = at::ceil_div(sort_size, npb);
  int n_rows = static_cast<int>(input.numel() / sort_size);

  auto opts_val = values.options();
  auto opts_u32 = at::TensorOptions().dtype(at::kInt).device(values.device());
  // Use int16 storage reinterpreted as ushort in the kernel: halves the
  // intermediate index memory traffic. Only valid when global indices fit in
  // 16 bits (sort_size ≤ 65536). The kernel never treats these as signed.
  auto opts_u16 = at::TensorOptions().dtype(at::kShort).device(values.device());

  bool need_permute = (dim != input.ndimension() - 1);
  Tensor work_in = input;
  int64_t stride_sort = in_stride_sort;
  int64_t stride_seg = in_stride_seg;
  if (need_permute) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < input.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    work_in = input.permute(perm).contiguous();
    sort_size = work_in.size(work_in.ndimension() - 1);
    n_rows = static_cast<int>(work_in.numel() / sort_size);
    stride_sort = 1;
    stride_seg = sort_size;
    n_blocks = at::ceil_div(sort_size, npb);
  }

  // u16 index variant is only safe when the kernel can emit int64 indices on
  // the last merge round (direct_final_write). If we need a permute-back
  // copy, we'd have to interpret the u16 values as unsigned; since PyTorch's
  // kShort→kLong copy sign-extends, that would corrupt indices ≥ 32768. So
  // restrict u16 to the non-permute path.
  const bool direct_final_write = !need_permute;
  const bool use_u16 = direct_final_write && sort_size <= 65536;
  auto opts_idx = use_u16 ? opts_u16 : opts_u32;

  Tensor dev_vals_0 = at::empty({n_rows, sort_size}, opts_val);
  Tensor dev_idxs_0 = at::empty({n_rows, sort_size}, opts_idx);
  Tensor dev_vals_1, dev_idxs_1;
  if (n_blocks > 1) {
    dev_vals_1 = at::empty({n_rows, sort_size}, opts_val);
    dev_idxs_1 = at::empty({n_rows, sort_size}, opts_idx);
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  const std::string type_str = scalarToMetalTypeString(values);
  const std::string bn_str = "_bn" + std::to_string(bn);
  const std::string u16_suffix = use_u16 ? "_u16" : "";
  const std::string block_kernel = "mb_sort_block_" + type_str + bn_str + u16_suffix;
  const std::string merge_kernel = "mb_merge_" + type_str + bn_str + u16_suffix;
  const std::string merge_final_kernel = "mb_merge_final_" + type_str + bn_str + u16_suffix;

  int total_rounds = 0;
  for (int m = 2; (m / 2) < n_blocks; m *= 2) ++total_rounds;

  // Batch block sort + all merge rounds into one GCD dispatch_sync. The encoder
  // is shared across dispatches via MPSStream, so this only affects CPU-side
  // queue overhead (one round-trip instead of 1 + n_merge_rounds).
  bool ping = false;
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      id<MTLComputePipelineState> block_pso = lib.getPipelineStateForFunc(block_kernel);
      id<MTLComputePipelineState> merge_pso = nil;
      id<MTLComputePipelineState> merge_final_pso = nil;
      if (n_blocks > 1) {
        merge_pso = lib.getPipelineStateForFunc(merge_kernel);
        if (direct_final_write)
          merge_final_pso = lib.getPipelineStateForFunc(merge_final_kernel);
      }

      [enc setComputePipelineState:block_pso];
      mtl_setArgs(enc, work_in, dev_vals_0, dev_idxs_0,
                  sort_size, stride_sort, stride_seg, descending);
      [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
           threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];

      if (n_blocks > 1) {
        bool p = false;
        int cur_round = 0;
        for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2, ++cur_round) {
          const Tensor& v_in = p ? dev_vals_1 : dev_vals_0;
          const Tensor& i_in = p ? dev_idxs_1 : dev_idxs_0;
          const Tensor& v_out_buf = p ? dev_vals_0 : dev_vals_1;
          const Tensor& i_out_buf = p ? dev_idxs_0 : dev_idxs_1;
          const bool use_direct = direct_final_write && (cur_round == total_rounds - 1);
          p = !p;

          [enc setComputePipelineState:use_direct ? merge_final_pso : merge_pso];
          if (use_direct) {
            mtl_setArgs(enc, v_in, i_in, values, indices,
                        sort_size, merge_tiles, n_blocks, descending);
          } else {
            mtl_setArgs(enc, v_in, i_in, v_out_buf, i_out_buf,
                        sort_size, merge_tiles, n_blocks, descending);
          }
          [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
               threadsPerThreadgroup:MTLSizeMake(bn, 1, 1)];
        }
      }
    }
  });
  if (n_blocks > 1) ping = (total_rounds % 2 == 1);

  if (n_blocks > 1) {
    if (direct_final_write) return;

    const Tensor& final_vals = ping ? dev_vals_1 : dev_vals_0;
    const Tensor& final_idxs = ping ? dev_idxs_1 : dev_idxs_0;
    auto fv = final_vals.view(work_in.sizes());
    auto fi = final_idxs.view(work_in.sizes());
    std::vector<int64_t> perm, inv_perm(values.ndimension());
    for (int64_t i = 0; i < values.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    for (int64_t i = 0; i < values.ndimension(); i++)
      inv_perm[perm[i]] = i;
    values.copy_(fv.permute(inv_perm));
    indices.copy_(fi.permute(inv_perm));
  } else {
    // n_blocks == 1: result is in dev_vals_0/dev_idxs_0
    if (need_permute) {
      auto fv = dev_vals_0.view(work_in.sizes());
      auto fi = dev_idxs_0.view(work_in.sizes());
      std::vector<int64_t> perm, inv_perm(values.ndimension());
      for (int64_t i = 0; i < values.ndimension(); i++)
        if (i != dim) perm.push_back(i);
      perm.push_back(dim);
      for (int64_t i = 0; i < values.ndimension(); i++)
        inv_perm[perm[i]] = i;
      values.copy_(fv.permute(inv_perm));
      indices.copy_(fi.permute(inv_perm));
    } else {
      values.copy_(dev_vals_0.view(values.sizes()));
      indices.copy_(dev_idxs_0.view(indices.sizes()));
    }
  }
}

// Radix sort: 4-bit radix, O(n) per pass. Faster than merge sort for large arrays.
static void sort_radix(const Tensor& input,
                       const Tensor& values,
                       const Tensor& indices,
                       int64_t dim,
                       bool descending,
                       int sort_size) {
  bool need_permute = (dim != input.ndimension() - 1);
  Tensor work_in = input;
  if (need_permute) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < input.ndimension(); i++)
      if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    work_in = input.permute(perm).contiguous();
    sort_size = work_in.size(work_in.ndimension() - 1);
  }

  int n_rows = static_cast<int>(work_in.numel() / sort_size);
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

  // 2-byte types: default RBN=1024,EPT=4 (NPB=4096). For many-rows configs
  // (n_rows >= 32), drop to RBN=512,EPT=4 (NPB=2048) — halves per-TG tgmem so
  // ~2x TGs fit per core, yielding ~6% improvement on [256, 16384] f16.
  // 1-byte / 4-byte stay at RBN=512.
  // Env var PYTORCH_MPS_SORT_SMALL_TG=0 disables the auto heuristic.
  static const bool small_tg_env_disable = []() {
    const char* e = std::getenv("PYTORCH_MPS_SORT_SMALL_TG");
    return e && std::string(e) == "0";
  }();
  const bool small_tg = !small_tg_env_disable && elem_size == 2 && n_rows >= 32;
  const int RADIX_BN = small_tg ? 512
                                : ((elem_size == 2) ? 1024 : 512);
  const int radix_ept = (elem_size >= 4) ? 4 : ((elem_size == 2) ? 4 : 8);
  const int RADIX_NPB = RADIX_BN * radix_ept;
  int n_blocks = at::ceil_div(sort_size, RADIX_NPB);
  int n_entries = radix_size * n_blocks;

  auto opts_val = values.options();
  auto opts_u32 = at::TensorOptions().dtype(at::kInt).device(values.device());
  // u16 index variant is only safe when the final pass can write int64
  // directly (direct_final_write), since host-side kShort→kLong copy would
  // sign-extend and corrupt indices ≥ 32768. Restrict to non-permute path.
  const bool direct_final_write = !need_permute;
  const bool use_u16 = direct_final_write && sort_size <= 65536;
  auto opts_idx = use_u16
      ? at::TensorOptions().dtype(at::kShort).device(values.device())
      : opts_u32;

  Tensor keys_0 = work_in.reshape({n_rows, sort_size}).contiguous();
  Tensor keys_1 = at::empty({n_rows, sort_size}, opts_val);
  Tensor idxs_0 = at::empty({n_rows, sort_size}, opts_idx);
  Tensor idxs_1 = at::empty({n_rows, sort_size}, opts_idx);
  Tensor histograms = at::empty({n_rows, n_entries}, opts_u32);

  const std::string type_str = scalarToMetalTypeString(values);
  const std::string rbits_suffix = "_" + std::to_string(radix_bits) + "bit";
  const std::string bn_suffix = small_tg ? "_bn512" : "";
  const std::string u16_suffix = use_u16 ? "_u16" : "";
  const std::string count_kernel = "radix_count_" + type_str + rbits_suffix + bn_suffix;

  // Fuse scan into scatter when n_blocks is small (fits in scatter's tgmem
  // scratch). Saves one dispatch per pass. Skip for large n_blocks where the
  // per-TG redundant scan work would exceed the dispatch saved.
  // kMaxFusedBlocks must match the .metal constant (4).
  constexpr int kMaxFusedBlocks = 4;
  constexpr int kFusedWorkCap = 128; // n_blocks² × n_rows
  const bool use_fused_scan =
      (n_blocks <= kMaxFusedBlocks) &&
      (n_blocks * n_blocks * n_rows <= kFusedWorkCap);
  const std::string scatter_suffix = use_fused_scan ? "fused_" : "";
  const std::string scatter_kernel =
      "radix_scatter_" + scatter_suffix + type_str + rbits_suffix + bn_suffix + u16_suffix;
  const std::string scatter_final_kernel =
      "radix_scatter_" + scatter_suffix + "final_" + type_str + rbits_suffix + bn_suffix + u16_suffix;
  MPSStream* mpsStream = getCurrentMPSStream();

  // All radix passes coalesce into a single GCD dispatch_sync to avoid
  // n_passes × 3 queue round-trips (each ~1µs). The encoder stays open between
  // dispatches (MPSStream caches it), so GPU-side this is equivalent.
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      id<MTLComputePipelineState> count_pso = lib.getPipelineStateForFunc(count_kernel);
      id<MTLComputePipelineState> scan_pso = nil;
      if (!use_fused_scan) scan_pso = lib.getPipelineStateForFunc("radix_scan");
      id<MTLComputePipelineState> scatter_pso =
          lib.getPipelineStateForFunc(scatter_kernel);
      id<MTLComputePipelineState> scatter_final_pso =
          direct_final_write
              ? lib.getPipelineStateForFunc(scatter_final_kernel)
              : nil;

      bool ping = false;
      for (int pass = 0; pass < n_passes; pass++) {
        int shift = pass * radix_bits;
        bool first_pass = (pass == 0);
        bool last_pass = (pass == n_passes - 1);
        bool use_direct = direct_final_write && last_pass;

        const Tensor& k_in = ping ? keys_1 : keys_0;
        const Tensor& i_in = ping ? idxs_1 : idxs_0;
        const Tensor& k_out_buf = ping ? keys_0 : keys_1;
        const Tensor& i_out_buf = ping ? idxs_0 : idxs_1;

        [enc setComputePipelineState:count_pso];
        mtl_setArgs(enc, k_in, histograms, sort_size, n_blocks, shift, descending);
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(RADIX_BN, 1, 1)];

        if (!use_fused_scan) {
          [enc setComputePipelineState:scan_pso];
          mtl_setArgs(enc, histograms, n_entries);
          // SCAN_BN in Sort.metal is 1024 — keep in sync.
          [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc setComputePipelineState:use_direct ? scatter_final_pso : scatter_pso];
        if (use_direct) {
          mtl_setArgs(enc, k_in, i_in, values, indices,
                      histograms, sort_size, n_blocks, shift, descending, first_pass);
        } else {
          mtl_setArgs(enc, k_in, i_in, k_out_buf, i_out_buf,
                      histograms, sort_size, n_blocks, shift, descending, first_pass);
        }
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
             threadsPerThreadgroup:MTLSizeMake(RADIX_BN, 1, 1)];

        ping = !ping;
      }
    }
  });

  bool ping = (n_passes % 2 == 1);

  if (direct_final_write) {
    // Already written directly to values/indices.
    return;
  }

  const Tensor& final_keys = ping ? keys_1 : keys_0;
  const Tensor& final_idxs = ping ? idxs_1 : idxs_0;

  auto fk_view = final_keys.view(work_in.sizes());
  auto fi_view = final_idxs.view(work_in.sizes());
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
  if (elem_size > 4) return false;
  // Env-var overrides for A/B testing.
  static const int force_path = []() {
    const char* e = std::getenv("PYTORCH_MPS_SORT_FORCE");
    if (!e) return 0;
    std::string s(e);
    if (s == "merge") return -1;
    if (s == "radix") return 1;
    return 0;
  }();
  if (force_path == -1) return false;
  if (force_path == 1) return true;
  int n_blocks_merge = at::ceil_div(sort_size, 4096);
  int merge_rounds = 0;
  for (int m = 2; (m / 2) < n_blocks_merge; m *= 2) merge_rounds++;
  int merge_dispatches = 1 + merge_rounds;
  int n_radix_passes = (elem_size <= 1) ? 2 : (elem_size <= 2) ? 2 : 4;
  int radix_dispatches = n_radix_passes * 3;
  return radix_dispatches <= 2 * merge_dispatches + 2;
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

  // Input must be contiguous for the kernels' simple stride formulation
  // (noop when self is already contig).
  Tensor input = self.contiguous();

  // Write directly into values/indices when contiguous; otherwise use a
  // contig scratch buffer and copy back.
  Tensor out_vals, out_inds;
  bool need_copy_back = false;
  if (values.is_contiguous() && indices.is_contiguous()) {
    out_vals = values;
    out_inds = indices;
  } else {
    out_vals = at::empty(self.sizes(), values.options());
    out_inds = at::empty(self.sizes(), indices.options());
    need_copy_back = true;
  }

  int n_rows_for_bn = static_cast<int>(self.numel() / sort_size);
  int bn = select_bn(sort_size, self.element_size(), n_rows_for_bn);
  int npb = bn * TN;

  bool is_last_dim = (dim == self.ndimension() - 1);
  const int64_t stride_sort = 1;
  const int64_t stride_seg = sort_size;

  if (should_use_radix(sort_size, self.element_size(), n_rows_for_bn)) {
    sort_radix(input, out_vals, out_inds, dim, descending, sort_size);
  } else if (sort_size <= npb && is_last_dim) {
    sort_single_block(input, out_vals, out_inds, descending, sort_size,
                      stride_sort, stride_seg, bn);
  } else {
    sort_multi_block(input, out_vals, out_inds, dim, descending, sort_size,
                     stride_sort, stride_seg, bn);
  }

  if (need_copy_back) {
    values.copy_(out_vals);
    indices.copy_(out_inds);
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
