#include <ATen/native/mps/kernels/FFT.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;
// Stockham FFT with fused real packing, four-step twiddles and chirps. F holds
// the kernel option flags from FFT.h.
inline float2 mul_i(float2 a) {
  return float2(-a.y, a.x);
}
constant float W16[] = { // W_16^m for m < 10, as (re, im) pairs
    1.0f,          0.0f,          0.923879533f,  -0.382683432f, 0.707106781f,
    -0.707106781f, 0.382683432f,  -0.923879533f, 0.0f,          -1.0f,
    -0.382683432f, -0.923879533f, -0.707106781f, -0.707106781f, -0.923879533f,
    -0.382683432f, -1.0f,         0.0f,          -0.923879533f, 0.382683432f};
template <uint R> // First factor of a composite radix; 1 if coded directly
constexpr uint comp_r1() {
  return R > 4 && R % 4 == 0 ? 4
      : R > 3 && R % 3 == 0  ? 3
      : R > 5 && R % 5 == 0  ? 5
                             : 1;
}
// Radix-R butterfly over v[0], v[S], ... in place (composite R1 R2: X[k1 + R1
// k2] lands at v[(k2 + R2 k1) S]); tw holds W_R at stride ts
#define BFLY(COND)                                        \
  template <uint R, uint S, enable_if_t<(COND), int> = 0> \
  inline void bfly(thread float2* v, bool inv, constant float2* tw, uint ts)
BFLY(R == 1) {}
BFLY(R == 2) {
  const float2 a = v[0], b = v[S];
  v[0] = a + b;
  v[S] = a - b;
}
BFLY(R == 3) {
  const float2 s = v[S] + v[2 * S], d = v[S] - v[2 * S], t = v[0] - 0.5f * s;
  const float2 u =
      (inv ? 0.86602540378443865f : -0.86602540378443865f) * mul_i(d);
  v[0] = v[0] + s;
  v[S] = t + u;
  v[2 * S] = t - u;
}
BFLY(R == 4) {
  const float2 a0 = v[0] + v[2 * S], a1 = v[0] - v[2 * S], a2 = v[S] + v[3 * S];
  const float2 a3 = inv ? mul_i(v[S] - v[3 * S]) : -mul_i(v[S] - v[3 * S]);
  v[0] = a0 + a2;
  v[2 * S] = a0 - a2;
  v[S] = a1 + a3;
  v[3 * S] = a1 - a3;
}
BFLY(R == 5) {
  const float2 s1 = v[S] + v[4 * S], d1 = v[S] - v[4 * S],
               s2 = v[2 * S] + v[3 * S], d2 = v[2 * S] - v[3 * S];
  const float c1 = 0.30901699437494742f, c2 = -0.80901699437494742f,
              s = inv ? 1.0f : -1.0f;
  const float sn1 = s * 0.95105651629515357f, sn2 = s * 0.58778525229247313f;
  const float2 t1 = v[0] + c1 * s1 + c2 * s2, t2 = v[0] + c2 * s1 + c1 * s2;
  const float2 u1 = mul_i(sn1 * d1 + sn2 * d2), u2 = mul_i(sn2 * d1 - sn1 * d2);
  v[0] = v[0] + s1 + s2;
  v[S] = t1 + u1;
  v[4 * S] = t1 - u1;
  v[2 * S] = t2 + u2;
  v[3 * S] = t2 - u2;
}
BFLY(R == 7) {
  const float2 s1 = v[S] + v[6 * S], d1 = v[S] - v[6 * S],
               s2 = v[2 * S] + v[5 * S], d2 = v[2 * S] - v[5 * S];
  const float2 s3 = v[3 * S] + v[4 * S], d3 = v[3 * S] - v[4 * S];
  const float c1 = 0.62348980185873353f, c2 = -0.22252093395631440f,
              c3 = -0.90096886790241913f, sg = inv ? 1.0f : -1.0f;
  const float n1 = sg * 0.78183148246802980f, n2 = sg * 0.97492791218182360f,
              n3 = sg * 0.43388373911755812f;
  const float2 t1 = v[0] + c1 * s1 + c2 * s2 + c3 * s3,
               t2 = v[0] + c2 * s1 + c3 * s2 + c1 * s3,
               t3 = v[0] + c3 * s1 + c1 * s2 + c2 * s3;
  const float2 u1 = mul_i(n1 * d1 + n2 * d2 + n3 * d3),
               u2 = mul_i(n2 * d1 - n3 * d2 - n1 * d3),
               u3 = mul_i(n3 * d1 - n1 * d2 + n2 * d3);
  v[0] = v[0] + s1 + s2 + s3;
  v[S] = t1 + u1;
  v[6 * S] = t1 - u1;
  v[2 * S] = t2 + u2;
  v[5 * S] = t2 - u2;
  v[3 * S] = t3 + u3;
  v[4 * S] = t3 - u3;
}
BFLY(comp_r1<R>() > 1) {
  constexpr uint R1 = comp_r1<R>(), R2 = R / R1;
  for (uint n2 = 0; n2 < R2; n2++) {
    bfly<R1, S * R2>(v + n2 * S, inv, tw, ts);
  }
  for (uint n2 = 1; n2 < R2; n2++) {
    for (uint k1 = 1; k1 < R1; k1++) {
      const uint m = n2 * k1, i = 2 * 16 / R * m; // W16 holds pairs
      const float2 w = R % 8 == 0 ? float2(W16[i], W16[i + 1]) : tw[m * ts];
      v[(n2 + R2 * k1) * S] = mul(v[(n2 + R2 * k1) * S], inv ? conj(w) : w);
    }
  }
  for (uint k1 = 0; k1 < R1; k1++) {
    bfly<R2, S>(v + R2 * k1 * S, inv, tw, ts);
  }
}
BFLY(R == 11 || R == 13) {
  float2 o[R];
  o[0] = v[0];
  for (uint r = 1; r <= R / 2; r++) {
    o[0] += v[r * S] + v[(R - r) * S];
  }
  for (uint k = 1; k <= R / 2; k++) {
    float2 a = v[0], b = 0.0f;
    for (uint r = 1; r <= R / 2; r++) {
      const float2 w = tw[((r * k) % R) * ts];
      a += w.x * (v[r * S] + v[(R - r) * S]);
      b += (inv ? -w.y : w.y) * mul_i(v[r * S] - v[(R - r) * S]);
    }
    o[k] = a + b;
    o[R - k] = a - b;
  }
  for (uint k = 0; k < R; k++) {
    v[k * S] = o[k];
  }
}
template <uint R> // Composite butterflies leave X[k1 + R1 k2] at k2 + R2 k1
inline uint opos(uint r) {
  constexpr uint R1 = comp_r1<R>();
  return R1 == 1 ? r : (r % R1) * (R / R1) + r / R1;
}
inline int2 batch_off(constant FFTParams& p, uint y, uint z, uint d0) {
  int2 off = int(y) * int2(p.in_strides[d0], p.out_strides[d0]);
  for (uint d = d0 + 1; d < p.ndim; z /= p.sizes[d++]) {
    off += int(z % p.sizes[d]) * int2(p.in_strides[d], p.out_strides[d]);
  }
  return off;
}
inline float2 twid(device const float2* tt, uint m) {
  return mul(
      tt[m & (fft_twiddle_low_entries - 1)],
      tt[fft_twiddle_low_entries + (m >> fft_twiddle_split_bits)]);
}
inline uint pad(uint i, uint shift) { // Padding separates banks at pow2 strides
  return i + (i >> shift);
}
template <class I, class O>
struct Bufs {
  device const I* in;
  device O* out;
  device const float2 *cl, *cs, *tt;
  constant float2* tw;
  threadgroup float2* buf;
  uint i0; // four-step slice index
};
typedef constant FFTParams& Params;
template <bool GLOBAL_STAGE, class I, class O> // Stage twiddle W^m
inline float2 wget(thread const Bufs<I, O>& b, uint m) {
  return GLOBAL_STAGE ? twid(b.tt, m) : b.tw[m];
}
#define LOAD_PT(COND)               \
  template <                        \
      class I,                      \
      class O,                      \
      uint F,                       \
      uint N,                       \
      bool C,                       \
      enable_if_t<(COND), int> = 0> \
  inline float2 load_pt(thread const Bufs<I, O>& b, Params p, uint e)
LOAD_PT(!is_complex_v<I> && (F & fft_packed_load)) {
  const int s = C ? 1 : p.in_stride;
  const float2 x = cast_to<float2>(b.in[int(2 * e) * s]);
  return float2(x.x, cast_to<float2>(b.in[int(2 * e + 1) * s]).x);
}
LOAD_PT(is_complex_v<I> && (F & fft_packed_load)) {
  const uint n = N ? N : p.n;
  const int s = C ? 1 : p.in_stride;
  float2 a = cast_to<float2>(b.in[int(e) * s]),
         c = cast_to<float2>(b.in[int(n - e) * s]);
  // DC and Nyquist are real; discard imaginary parts like pocketfft.
  a.y = e ? a.y : 0.0f;
  c.y = e ? c.y : 0.0f;
  return (a + conj(c)) + mul_i(mul(a - conj(c), conj(b.tw[e])));
}
LOAD_PT(!(F & fft_packed_load) && !(F & fft_extended_load)) {
  const uint n = N ? N : p.n;
  const int s = C ? 1 : p.in_stride;
  const bool mirror = !(F & fft_global_stage) && is_complex_v<I> && e >= p.n_in;
  const float2 x = cast_to<float2>(b.in[int(mirror ? n - e : e) * s]);
  return mirror ? conj(x) : x;
}
LOAD_PT(!(F & fft_packed_load) && (F & fft_extended_load)) {
  const int s = C ? 1 : p.in_stride;
  const uint i0e = p.ld_mul > 1 ? b.i0 : 0, f = e * p.ld_mul + i0e;
  if (f >= p.n_sig) {
    return 0.0f;
  }
  const float2 x = is_complex_v<I> && p.n_in < p.n_sig && f >= p.n_in
      ? conj(cast_to<float2>(b.in[int(p.n_sig - f - i0e) * p.in_unit]))
      : cast_to<float2>(b.in[int(e) * s]);
  return b.cl ? mul(x, b.cl[f]) : x;
}
template <typename T>
using packed2 = conditional_t<is_same_v<T, half>, packed_half2, packed_float2>;
#define STORE_PT(COND)                                                      \
  template <class I, class O, uint F, bool C, enable_if_t<(COND), int> = 0> \
  inline void store_pt(                                                     \
      thread const Bufs<I, O>& b, Params p, uint k, float2 v, float2 w)
STORE_PT(bool(F& fft_packed_real_store)) {
  v *= p.scale;
  const int s = C ? 1 : p.out_stride;
  if (s == 1) {
    reinterpret_cast<device packed2<O>*>(b.out)[k] = packed2<O>(v);
  } else {
    b.out[int(2 * k) * s] = O(v.x);
    b.out[int(2 * k + 1) * s] = O(v.y);
  }
}
STORE_PT(!(F & fft_packed_real_store)) {
  constexpr bool extended = F & fft_extended_store;
  v *= p.scale;
  const int s = C ? 1 : p.out_stride;
  v = (F & fft_four_step_twiddle) ? mul(v, F & fft_inverse ? conj(w) : w) : v;
  const uint kf = extended && p.out_k_mul > 1 ? k * p.out_k_mul + b.i0 : k;
  if ((F & fft_global_stage) || kf < p.n_out) {
    b.out[int(k) * s] = cast_to<O>(extended && b.cs ? mul(v, b.cs[kf]) : v);
  }
  if (extended && p.mirror_out && b.i0 && b.i0 != p.out_k_mul / 2 &&
      p.n_sig - kf < p.n_out) {
    b.out[int(p.n_sig - kf - b.i0) * p.out_strides[p.i0_dim1]] =
        cast_to<O>(conj(v));
  }
}
// Unpack z = (Z[k], Z[n-k]): X[k] = E + T, X[n-k] = conj(E - T), w = W_2n^k
template <class I, class O, uint F, uint N, bool C>
inline void unpack(
    thread const Bufs<I, O>& b,
    Params p,
    uint k,
    float2 w,
    float4 z) {
  constexpr bool twiddled = F & fft_four_step_twiddle;
  const uint n = N ? N : p.n, os = C ? 1 : p.out_stride;
  const float2 a = z.xy, c = z.zw;
  const bool two = n - k != k;
  const float2 e = 0.5f * (a + conj(c)), t = mul(w, -0.5f * mul_i(a - conj(c)));
  float2 x = (e + t) * p.scale, y = conj(e - t) * p.scale;
  x = twiddled ? mul(x, twid(b.tt, k * b.i0)) : x;
  y = twiddled ? mul(y, twid(b.tt, (n - k) * b.i0)) : y;
  b.out[int(k) * os] = cast_to<O>(x);
  if (two && n - k < p.n_out) {
    b.out[int(n - k) * os] = cast_to<O>(y);
  }
  if (p.n_out > n + 1 && k) {
    b.out[int(2 * n - k) * os] = cast_to<O>(conj(x));
    if (two) {
      b.out[int(n + k) * os] = cast_to<O>(conj(y));
    }
  }
}
#define RADIX_BIT(R) (1u << (R))
#define MASK_P2 \
  (RADIX_BIT(1) | RADIX_BIT(2) | RADIX_BIT(4) | RADIX_BIT(8) | RADIX_BIT(16))
// j % ns through a reciprocal, much cheaper than an integer remainder on the
// GPU. Threadgroup stages have j below 2^13, exact in float. Global stages
// reach 2^31, where float(j) is inexact, so a second step on the small
// remainder corrects it.
template <bool POW2, bool GLOBAL_STAGE>
inline uint mod_ns(uint j, uint ns, float inv_ns) {
  if IF_CONSTEXPR (POW2) {
    return j & (ns - 1);
  }
  int k = int(j) - int(ns) * int(float(j) * inv_ns);
  if IF_CONSTEXPR (GLOBAL_STAGE) {
    k -= int(ns) * int(float(k) * inv_ns);
  }
  k = k < 0 ? k + int(ns) : k;
  return k >= int(ns) ? k - ns : k;
}
template <uint N, bool C>
inline uint row_pad_shift(Params p) {
  return C ? (N == fft_max_points ? fft_no_pad_shift : fft_pad_shift)
           : p.pad_shift;
}
// Butterflies per thread for stage s (Stockham stride Ns); smem exchanges
// stages and holds the final real unpack. A global stage goes through global
// memory.
template <uint R, bool P2, class I, class O, uint F, uint N, uint NS, bool C>
inline void stage(
    thread const Bufs<I, O>& b,
    Params p,
    uint s,
    uint lane,
    bool valid) {
  constexpr bool global_stage = F & fft_global_stage, inverse = F & fft_inverse;
  constexpr bool packed = F & fft_packed_load;
  constexpr bool unpacks = !global_stage && !is_complex_v<I> && packed;
  constexpr uint points =
      global_stage ? R : fft_points_per_thread(N, F, !is_complex_v<I>, C);
  constexpr uint butterflies = (points + R - 1) / R; // constexpr ceil_div
  const uint n = N ? N : p.n,
             threads = N ? N / points : p.threads_per_transform, nR = n / R,
             Ns = NS ? NS : p.ns[s],
             twstep = NS ? N / NS / R * (packed ? 2 : 1) : p.twstep[s],
             shift = row_pad_shift<N, C>(p);
  const bool first = NS ? NS == 1 : s == 0,
             last = global_stage || (NS ? NS * R == N : s + 1 == p.nstages);
  // Pow2 plans keep nR, Ns multiples of 16: padded smem slots are affine in r
  const uint nRv = valid ? nR : 0;
  const uint slot_stride = C ? 1 : p.slot_stride,
             read_stride = (nR + (nR >> shift)) * slot_stride,
             write_stride = (Ns + (Ns >> shift)) * slot_stride;
  const float inv_ns = NS ? 1.0f / float(NS) : p.inv_ns[s];
  float2 v[butterflies][R];
  for (uint m = 0; m < butterflies; m++) {
    const uint j = lane + m * threads;
    if (j < nR) {
      const uint k = mod_ns<P2, global_stage>(j, Ns, inv_ns);
      if (first && !global_stage) {
        for (uint r = 0; r < R; r++) {
          v[m][r] = load_pt<I, O, F, N, C>(b, p, j + r * nR);
        }
      } else {
        const uint bj = pad(j, shift) * slot_stride;
        v[m][0] = global_stage ? load_pt<I, O, F, N, C>(b, p, j) : b.buf[bj];
        // W^(r k) from gathered W^1, W^4, W^8 and W^4 steps: gathers cost more
        const float2 w1 = wget<global_stage>(b, k * twstep);
        float2 wa = 0.0f, wb = 0.0f;
        for (uint r = 1; r < R; r++) {
          wb = r == 4 || (r == 8 && !global_stage)
              ? wget<global_stage>(b, r * k * twstep)
              : (r & 3) ? wb
                        : mul(wb, wget<global_stage>(b, 4 * k * twstep));
          wa = (r & 3) == 0  ? float2(1.0f, 0.0f)
              : (r & 3) == 1 ? w1
                             : mul(wa, w1);
          const float2 w = r < 4 ? wa : (r & 3) ? mul(wa, wb) : wb;
          const float2 x = global_stage
              ? load_pt<I, O, F, N, C>(b, p, j + r * nR)
              : b.buf
                    [P2 ? bj + r * read_stride
                        : pad(j + r * nR, shift) * slot_stride];
          v[m][r] = mul(x, inverse ? conj(w) : w);
        }
      }
      bfly<R, 1>(v[m], inverse, b.tw, nR * (packed ? 2 : 1));
    }
  }
  if (last && !unpacks && (C || global_stage || !p.via_smem)) {
    // Four-step twiddles W^(k i0) of the outputs k = j + r nR step by W^(nR i0)
    const float2 twiddle_step =
        (F & fft_four_step_twiddle) ? twid(b.tt, nR * b.i0) : 0.0f;
    for (uint m = 0; m < butterflies; m++) {
      const uint j = lane + m * threads;
      if (j < nRv) {
        float2 w = (F & fft_four_step_twiddle) ? twid(b.tt, j * b.i0) : 0.0f;
        const uint k =
            global_stage ? mod_ns<P2, global_stage>(j, Ns, inv_ns) : 0;
        const uint base = global_stage ? (j - k) * R + k : j,
                   step = global_stage ? Ns : nR;
        for (uint r = 0; r < R; r++) {
          store_pt<I, O, F, C>(b, p, base + r * step, v[m][opos<R>(r)], w);
          w = mul(w, twiddle_step);
        }
      }
    }
    return;
  }
  if (!first) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint m = 0; m < butterflies; m++) {
    const uint j = lane + m * threads;
    if (j < nR) {
      const uint k = mod_ns<P2, global_stage>(j, Ns, inv_ns);
      const uint base = (j - k) * R + k, bb = pad(base, shift) * slot_stride;
      for (uint r = 0; r < R; r++) {
        b.buf
            [P2 ? bb + r * write_stride
                : pad(base + r * Ns, shift) * slot_stride] = v[m][opos<R>(r)];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// The real unpack as its own global-memory pass.
template <class I, class O, uint F>
inline void global_unpack(thread const Bufs<I, O>& b, Params p, uint lane) {
  if (lane < p.n / 2 + 1) {
    // The unpack twiddle W_2n^k from sincos, avoiding a large table read.
    float wr, wi = precise::sincos(-float(lane) * (M_PI_F / float(p.n)), wr);
    const float2 a = cast_to<float2>(b.in[int(lane) * p.in_stride]);
    const float2 c =
        cast_to<float2>(b.in[int(lane ? p.n - lane : 0) * p.in_stride]);
    unpack<I, O, F, 0, false>(b, p, lane, float2(wr, wi), float4(a, c));
  }
}
// Next stage of a fixed length: its pow2 part up to the largest radix, then 3,
// 5, 7
constexpr uint first_radix(uint n) {
  constexpr uint largest_pow2_radix = 16;
  const uint p2 = (n & -n) < largest_pow2_radix ? n & -n : largest_pow2_radix;
  return p2 > 1 ? p2 : n % 3 == 0 ? 3 : n % 5 == 0 ? 5 : n;
}
#define RUN_STAGES(COND)            \
  template <                        \
      class I,                      \
      class O,                      \
      uint MASK,                    \
      uint F,                       \
      uint N,                       \
      bool P2,                      \
      bool C,                       \
      enable_if_t<(COND), int> = 0> \
  inline void run_stages(           \
      thread const Bufs<I, O>& b, Params p, uint lane, bool valid)
RUN_STAGES((F & fft_global_stage) && MASK == RADIX_BIT(1)) {
  global_unpack<I, O, F>(b, p, lane);
}
RUN_STAGES((F & fft_global_stage) && MASK != RADIX_BIT(1)) {
  stage<__builtin_ctz(MASK), P2, I, O, F, 0, 0, C>(b, p, 0, lane, valid);
}
RUN_STAGES(!(F & fft_global_stage) && N != 0) {
  constexpr uint r0 = first_radix(N), r1 = first_radix(N / r0),
                 r2 = first_radix(N / r0 / r1);
  stage<r0, P2, I, O, F, N, 1, C>(b, p, 0, lane, valid);
  if IF_CONSTEXPR (r1 > 1) {
    stage<r1, P2, I, O, F, N, r0, C>(b, p, 1, lane, valid);
  }
  if IF_CONSTEXPR (r2 > 1) {
    stage<r2, P2, I, O, F, N, r0 * r1, C>(b, p, 2, lane, valid);
  }
}
RUN_STAGES(!(F & fft_global_stage) && N == 0) {
  for (uint s = 0; s < p.nstages; s++) {
#define STAGE(RR)                                            \
  case RR:                                                   \
    if IF_CONSTEXPR (MASK & RADIX_BIT(RR)) {                 \
      stage<RR, P2, I, O, F, 0, 0, C>(b, p, s, lane, valid); \
    }                                                        \
    break;
    switch (p.radix[s]) { STAGE(1) FFT_STAGE_RADICES(STAGE) }
#undef STAGE
  }
}
template <class I, class O, uint MASK, uint F, uint N = 0, bool C = false>
kernel void fft_kernel(
    device const I* in,
    device O* out,
    device const float2* cl,
    device const float2* cs,
    device const float2* tt,
    constant float2* tw,
    constant FFTParams& p,
    threadgroup float2* buf [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]]) {
  constexpr bool global_stage = F & fft_global_stage;
  constexpr bool P2 = !global_stage && MASK && (MASK & ~MASK_P2) == 0;
  constexpr bool unpacks =
      !global_stage && !is_complex_v<I> && (F & fft_packed_load);
  constexpr uint points = fft_points_per_thread(N, F, !is_complex_v<I>, C);
  const uint n = N ? N : p.n, batch = C || global_stage ? 1 : p.batch_per_group,
             threads = N ? N / points : p.threads_per_transform;
  // Consume the tail of freshly written inputs first.
  if ((C && !is_complex_v<I>) || ((F & fft_extended_store) && p.mirror_out)) {
    tg.x = groups.x - 1 - tg.x;
  }
  // Linear dispatch keeps neighboring tiles together across batch dimensions.
  if (!global_stage && p.ndim > 1) {
    const uint nx = ceil_div<uint>(p.sizes[0], batch), ny = p.sizes[1];
    tg = uint3(tg.x % nx, (tg.x / nx) % ny, tg.x / nx / ny);
  }
  const uint first_row = global_stage ? 0 : tg.x * batch,
             rows_here = min(batch, p.sizes[0] - first_row);
  const bool in_lanes_first = C || p.in_stride == 1;
  const uint row_index = global_stage ? 0
      : in_lanes_first                ? tid / threads
                                      : tid % batch;
  const uint lane = global_stage ? tg.x * threads + tid
      : in_lanes_first           ? tid % threads
                                 : tid / batch;
  const int2 off = batch_off(p, tg.y, tg.z, global_stage ? 0 : 1);
  const bool valid = global_stage || row_index < rows_here;
  const uint row = first_row + (valid ? row_index : 0),
             i0 = p.i0_dim1 ? tg.y : row;
  device const I* row_in = in + off.x + int(row) * p.in_strides[0];
  device O* row_out = out + off.y + int(row) * p.out_strides[0];
  const Bufs<I, O> bufs{
      row_in, row_out, cl, cs, tt, tw, buf + row_index * p.row, i0};
  run_stages<I, O, MASK, F, N, P2, C>(bufs, p, lane, valid);
  if (!unpacks && (C || global_stage || !p.via_smem)) {
    return;
  }
  // The last stage left its outputs in smem: this pass's real unpack or store.
  const bool out_lanes_first = C || p.out_stride == 1;
  const uint store_row = out_lanes_first ? tid / threads : tid % batch,
             store_lane = out_lanes_first ? tid % threads : tid / batch;
  threadgroup const float2* row_smem = buf + store_row * p.row;
  Bufs<I, O> store_bufs = bufs;
  store_bufs.out = out + off.y + int(first_row + store_row) * p.out_strides[0];
  store_bufs.i0 = p.i0_dim1 ? tg.y : first_row + store_row;
  const uint slot_stride = C ? 1 : p.slot_stride,
             shift = row_pad_shift<N, C>(p);
  float2 w = (F & fft_four_step_twiddle) ? twid(tt, store_lane * store_bufs.i0)
                                         : 0.0f,
         w_step = (F & fft_four_step_twiddle)
      ? twid(tt, threads * store_bufs.i0)
      : 0.0f;
  if (store_row >= rows_here) {
    return;
  }
  for (uint k = store_lane; k < (unpacks ? n / 2 + 1 : n); k += threads) {
    const float2 a = row_smem[pad(k, shift) * slot_stride],
                 c = row_smem[pad(k ? n - k : 0, shift) * slot_stride];
    if (unpacks) {
      unpack<I, O, F, N, C>(store_bufs, p, k, tw[k], float4(a, c));
    } else {
      store_pt<I, O, F, false>(store_bufs, p, k, a, w);
      w = mul(w, w_step);
    }
  }
}
// F is spelled twice: the number goes into the kernel name, which the host
// builds from the flags, and INST2 checks it against the named flags.
#define F_INV fft_inverse
#define F_PACKED fft_packed_load
#define F_TWIDDLE fft_four_step_twiddle
#define F_STORE fft_extended_store
#define F_LOAD fft_extended_load
#define F_GLOBAL fft_global_stage
#define F_REAL_STORE fft_packed_real_store
#define FFT_ARGS(IN, OUT)                                                    \
  device const IN*, device OUT*, device const float2*, device const float2*, \
      device const float2*, constant float2*, constant FFTParams&
#define INST2(IN, OUT, F, FLAGS, MASK, NAME, N, C)                    \
  static_assert((F) == (FLAGS), "kernel name must match its flags");  \
  template [[host_name("fft_" #IN "_" #OUT "_" NAME #F)]] kernel void \
  fft_kernel<IN, OUT, MASK, (FLAGS), N, C>(                           \
      FFT_ARGS(IN, OUT), threadgroup float2*, uint, uint3, uint3);
#define INST1(...) INST2(__VA_ARGS__)
#define UNPAREN(...) __VA_ARGS__
#define INSTF(IN, OUT, F, FLAGS, X) INST1(IN, OUT, F, FLAGS, UNPAREN X)
#define INST_FP32(IK, OK, F, FLAGS, X) INSTF(IK##_f, OK##_f, F, FLAGS, X)
#define INST(IK, OK, F, FLAGS, X) \
  INST_FP32(IK, OK, F, FLAGS, X) INSTF(IK##_h, OK##_h, F, FLAGS, X)
#define C_f float2
#define C_h half2
#define R_f float
#define R_h half
#define BASIC(_, X)       \
  _(C, C, 0, 0, X)        \
  _(C, C, 1, F_INV, X)    \
  _(R, C, 2, F_PACKED, X) \
  _(C, R, 67, F_INV | F_PACKED | F_REAL_STORE, X)
#define PLAIN(_, X) BASIC(_, X) _(R, C, 0, 0, X) _(C, R, 1, F_INV, X)
// Several-pass variants are fp32 only: half-precision transforms needing
// several passes run in fp32 on the host.
#define EXT(_, X)                            \
  _(C, C, 4, F_TWIDDLE, X)                   \
  _(C, C, 5, F_INV | F_TWIDDLE, X)           \
  _(C, C, 8, F_STORE, X)                     \
  _(C, C, 21, F_INV | F_LOAD | F_TWIDDLE, X) \
  _(R, C, 4, F_TWIDDLE, X)                   \
  _(R, C, 6, F_PACKED | F_TWIDDLE, X)        \
  _(C, R, 9, F_INV | F_STORE, X)
// Chirps exist only in Bluestein passes, whose lengths are powers of two.
#define CHIRP(_, X)                  \
  _(C, C, 9, F_INV | F_STORE, X)     \
  _(C, C, 20, F_LOAD | F_TWIDDLE, X) \
  _(C, C, 24, F_LOAD | F_STORE, X)   \
  _(R, C, 20, F_LOAD | F_TWIDDLE, X) \
  _(R, C, 24, F_LOAD | F_STORE, X)
#define MASK_P235 (MASK_P2 | RADIX_BIT(3) | RADIX_BIT(5))
#define MASK_P2357 (MASK_P235 | RADIX_BIT(7))
// The smallest family factoring n (radix branches cost registers); p16: 1, 16
#define FAMILY(MASK, NAME) \
  PLAIN(INST, (MASK, NAME, 0, false)) EXT(INST_FP32, (MASK, NAME, 0, false))
FAMILY(MASK_P2, "p2_")
CHIRP(INST_FP32, (MASK_P2, "p2_", 0, false))
FAMILY(MASK_P235, "p235_")
FAMILY(MASK_P2357, "p2357_")
FAMILY((MASK_P2357 | RADIX_BIT(11) | RADIX_BIT(13)), "all_")
PLAIN(INST, ((RADIX_BIT(16) | RADIX_BIT(1)), "p16_", 0, false))
#define FIXED_X(N) ((N & (N - 1) ? MASK_P235 : MASK_P2), #N "_", N, false)
#define FIXED(N) BASIC(INST, FIXED_X(N)) EXT(INST_FP32, FIXED_X(N))
#define FIXED_POW2(N) FIXED(N) CHIRP(INST_FP32, FIXED_X(N))
FFT_FIXED_POW2_LENGTHS(FIXED_POW2)
FFT_FIXED_SMOOTH_LENGTHS(FIXED)
// A real length of 11 or 13 is odd and stays plain; real 26 packs into 13.
PLAIN(INST, FIXED_X(11))
PLAIN(INST, FIXED_X(13))
BASIC(INST, FIXED_X(26))
#define CONTIG(N) BASIC(INST, (MASK_P2, #N "c_", N, true))
FFT_CONTIGUOUS_LENGTHS(CONTIG)
#define GLOBAL_X(R) (RADIX_BIT(R), "g" #R "_", 0, false)
#define GLOBAL_PASS(R)                                     \
  INSTF(float2, float2, 32, F_GLOBAL, GLOBAL_X(R))         \
  INSTF(float2, float2, 33, F_INV | F_GLOBAL, GLOBAL_X(R)) \
  INSTF(float2, float, 97, F_INV | F_GLOBAL | F_REAL_STORE, GLOBAL_X(R))
FFT_GLOBAL_RADICES(GLOBAL_PASS)
// Plans take the largest radix first, so above fft_max_points / 2 points only
// these can start a plan and read packed real data.
#define GLOBAL_FIRST_PASS(R)                                 \
  INSTF(float, float2, 34, F_PACKED | F_GLOBAL, GLOBAL_X(R)) \
  INSTF(float2, float2, 35, F_INV | F_PACKED | F_GLOBAL, GLOBAL_X(R))
GLOBAL_FIRST_PASS(7)
GLOBAL_FIRST_PASS(9)
GLOBAL_FIRST_PASS(12)
GLOBAL_FIRST_PASS(15)
GLOBAL_FIRST_PASS(16)
GLOBAL_FIRST_PASS(21)
GLOBAL_FIRST_PASS(25)
GLOBAL_FIRST_PASS(28)
INSTF(float2, float2, 32, F_GLOBAL, (RADIX_BIT(1), "g1_", 0, false))
