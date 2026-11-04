#pragma once
#include <c10/metal/common.h>
// Kernel options, fixed per instantiation because runtime mode flags around
// memory accesses cost up to 1.5x. They form the template parameter F and the
// number that ends a kernel's name.
C10_METAL_CONSTEXPR uint32_t fft_inverse = 1;
C10_METAL_CONSTEXPR uint32_t fft_packed_load =
    2; // Real data as a half-length complex transform
C10_METAL_CONSTEXPR uint32_t fft_four_step_twiddle = 4;
C10_METAL_CONSTEXPR uint32_t fft_extended_store =
    8; // Output mapping, chirp and mirrored outputs
C10_METAL_CONSTEXPR uint32_t fft_extended_load =
    16; // Input mapping, chirp, zero padding and mirrored inputs
C10_METAL_CONSTEXPR uint32_t fft_global_stage =
    32; // One stage through global memory
C10_METAL_CONSTEXPR uint32_t fft_packed_real_store = 64;
C10_METAL_CONSTEXPR uint32_t fft_several_pass_flags = fft_four_step_twiddle |
    fft_extended_store | fft_extended_load | fft_global_stage;
// Plan capacity: 4096 = 2^12 needs at most 12 stages.
C10_METAL_CONSTEXPR uint32_t fft_max_stages = 16;
// Largest transform of one pass: 4096 complex points fill the 32 KB of
// threadgroup memory Metal guarantees.
C10_METAL_CONSTEXPR uint32_t fft_max_points = 4096;
// Long-axis twiddles use W^m = W^(low bits) * W^(high bits): about 2^split + n
// / 2^split table entries for n points.
C10_METAL_CONSTEXPR uint32_t fft_twiddle_split_bits = 12;
C10_METAL_CONSTEXPR uint32_t fft_twiddle_low_entries = 1u
    << fft_twiddle_split_bits;
// Smem slot i of a padded row is at i + (i >> fft_pad_shift): a spare slot per
// 16 avoids bank conflicts at pow2 strides. fft_no_pad_shift shifts every index
// out, for rows with no room to spare.
C10_METAL_CONSTEXPR uint32_t fft_pad_shift = 4, fft_no_pad_shift = 31;
// Stage radices of threadgroup kernels, one radix per global-memory pass, and
// lengths with compile-time plans. Plans take radices in these orders.
#define FFT_STAGE_RADICES(_) _(16) _(8) _(4) _(2) _(3) _(5) _(7) _(11) _(13)
#define FFT_GLOBAL_RADICES(_) \
  _(28) _(25) _(21) _(16) _(15) _(12) _(9) _(8) _(7) _(5) _(4) _(3) _(2)
#define FFT_FIXED_POW2_LENGTHS(_) \
  _(64) _(128) _(256) _(512) _(1024) _(2048) _(4096)
#define FFT_FIXED_SMOOTH_LENGTHS(_) _(200) _(224) _(240)
// Hot lengths outside the pow2 and smooth kernels, single pass only: 11 and 13
// bloat a shared kernel's registers.
#define FFT_FIXED_SINGLE_PASS_LENGTHS(_) _(11) _(13) _(26)
// Lengths with a kernel for one contiguous transform per threadgroup.
#define FFT_CONTIGUOUS_LENGTHS(_) _(512) _(1024) _(2048) _(4096)
// Points per thread of a threadgroup kernel, chosen by measurement. 8 fits in
// registers and 32 always spills. 16 pays only on the lengths below, where it
// removes a stage. 26 points are radices 2 and 13: one butterfly per thread.
constexpr uint32_t fft_points_per_thread(
    uint32_t n,
    uint32_t flags,
    bool real,
    bool contiguous) {
  const bool single_pass = !(flags & fft_several_pass_flags);
  if (n == 26 && single_pass) {
    return 13;
  }
  const bool extended_512 = n == 512 &&
      ((flags & fft_extended_store) ||
       (real && (flags & fft_four_step_twiddle)));
  const bool drops_stage =
      single_pass && (n == 1024 || (n == 2048 && !contiguous));
  return extended_512 || drops_stage ? 16 : 8;
}
// batch_per_group transforms of n points per threadgroup, threads_per_transform
// threads each. n_sig: full signal (zero-padded beyond), n_in: inputs (c2r:
// n_sig/2+1, mirrored), n_out: outputs. Four-step: input e * ld_mul + i0,
// output k * out_k_mul + i0, mirror_out adds conj X[k] at n_sig - k, i0_dim1
// reads i0 from batch dim 1. A row has `row` smem slots; tiles interleave by
// slot_stride.
struct FFTParams {
  uint32_t n, n_sig, n_in, n_out, batch_per_group, threads_per_transform, ndim,
      nstages;
  int32_t in_stride, in_unit, out_stride; // along the axis: slice, full, out
  float scale;
  uint32_t ld_mul, out_k_mul, row, pad_shift, slot_stride;
  bool mirror_out, i0_dim1, via_smem;
  ::c10::metal::array<uint32_t, fft_max_stages> radix, ns,
      twstep; // ns: product so far
  ::c10::metal::array<float, fft_max_stages> inv_ns;
  ::c10::metal::array<uint32_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> in_strides, out_strides;
};
