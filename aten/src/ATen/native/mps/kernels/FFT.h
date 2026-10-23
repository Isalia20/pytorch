#pragma once
#include <c10/metal/common.h>
// tgb transforms of n points per group, jt threads each. n_sig: full signal
// (zero-padded beyond), n_in: inputs (c2r: n_sig/2+1, mirrored), n_out:
// outputs. Four-step: input e * ld_mul + i0, output k * out_k_mul + i0,
// mirror_out adds conj X[k] at n_sig - k, i0_dim1 reads i0 from batch dim 1.
// Smem slot i of a row of `row` slots is at i + (i >> padshift); tiles
// interleave by es.
struct FFTParams {
  uint32_t n, n_sig, n_in, n_out, tgb, jt, ndim, nstages;
  int32_t in_stride, in_unit, out_stride; // along the axis: slice, full, out
  float scale;
  uint32_t ld_mul, out_k_mul, mirror_out, i0_dim1, via_smem, row, padshift, es;
  ::c10::metal::array<uint32_t, 16> radix, ns, twstep; // ns: product so far
  ::c10::metal::array<float, 16> inv_ns;
  ::c10::metal::array<uint32_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> in_strides, out_strides;
};
