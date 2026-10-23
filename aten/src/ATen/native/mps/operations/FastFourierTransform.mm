#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/FFT.h>
#include <algorithm>
#include <numeric>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/polar.h>
#endif

namespace at::native {
namespace {
using namespace mps;
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FFT_metallib.h>
#endif
// Complex points per transform that fit in threadgroup memory; longer transforms take several passes
constexpr int64_t kMaxPoints = 4096;
// Stage radices of the threadgroup kernels, and of the global-memory passes (one radix per pass)
constexpr auto kRadices = {16, 8, 4, 2, 3, 5, 7}, kGlobalRadices = {28, 25, 21, 16, 15, 12, 9, 8, 7, 5, 4, 3, 2};
// Radix sequence of n over `rs` (largest first; {1} for n = 1), empty when n has other prime factors
c10::SmallVector<uint32_t, 16> radices(int64_t n, std::initializer_list<int> rs) {
  c10::SmallVector<uint32_t, 16> plan(n == 1, 1);
  for (const auto r : rs) {
    for (; n % r == 0; n /= r) {
      plan.push_back(r);
    }
  }
  return n == 1 ? plan : decltype(plan){};
}
bool is_smooth(int64_t n) {
  return !radices(n, kRadices).empty();
}
// W_n^k for k < n, or (two_level, for inter-pass twiddles of a long axis) W_n^b for b < 4096 then W_n^(4096 a)
Tensor twiddles(int64_t n, bool two_level = false) {
  static std::mutex mutex;
  static std::map<std::pair<int64_t, bool>, Tensor> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto& t = cache[{n, two_level}];
  if (!t.defined()) {
    const auto k = two_level ? at::cat({at::arange(4096, at::kDouble), at::arange(0, n, 4096, at::kDouble)})
                             : at::arange(n, at::kDouble);
    t = at::polar(at::ones_like(k), k.mul(-2 * M_PI / n)).to(at::device(kMPS).dtype(kComplexFloat));
  }
  return t;
}
// Bluestein chirp exp(-+ i pi k^2 / n) for k < n, and the M-point FFT of its conjugate wrapped around zero
const std::pair<Tensor, Tensor>& chirp_tables(int64_t n, bool inverse) {
  static std::mutex mutex;
  static std::map<std::pair<int64_t, bool>, std::pair<Tensor, Tensor>> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto& t = cache[{n, inverse}];
  if (!t.first.defined()) {
    const auto k = at::arange(n, at::kLong);
    const auto angle = k.mul(k).remainder(2 * n).to(at::kDouble).mul((inverse ? 1 : -1) * M_PI / n);
    const auto c = at::polar(at::ones_like(angle), angle);
    const auto M = static_cast<int64_t>(std::bit_ceil<uint64_t>(2 * n - 1));
    const auto b = at::cat({c.conj(), at::zeros({M - 2 * n + 1}, c.options()), c.narrow(0, 1, n - 1).conj().flip(0)});
    const auto opts = at::device(kMPS).dtype(kComplexFloat);
    t = {c.to(opts), at::fft_fft(b).to(opts)};
  }
  return t;
}
struct Dim {
  int64_t size, in_stride, out_stride;
};
struct Batch {
  c10::SmallVector<Dim, 16> dims; // Innermost first
  int64_t numel = 1, si = 0, so = 0; // Transforms, and the element strides along the axis
};
// Batch dims of a pass along `dim` (every other dim), innermost first, merged where both tensors allow it
Batch collapse(const Tensor& in, const Tensor& out, int64_t dim) {
  c10::SmallVector<int64_t, 16> dims;
  for (const auto d : c10::irange(in.dim())) {
    if (d != dim && in.size(d) != 1) {
      dims.push_back(d);
    }
  }
  std::sort(dims.begin(), dims.end(), [&](auto a, auto b) {
    return std::make_pair(std::abs(in.stride(a)), std::abs(out.stride(a))) <
        std::make_pair(std::abs(in.stride(b)), std::abs(out.stride(b)));
  });
  Batch b{{}, 1, in.stride(dim), out.stride(dim)};
  for (const auto d : dims) {
    b.numel *= in.size(d);
    const auto last = b.dims.empty() ? Dim{0, 0, 0} : b.dims.back();
    if (last.size && last.in_stride * last.size == in.stride(d) && last.out_stride * last.size == out.stride(d)) {
      b.dims.back().size *= in.size(d);
    } else {
      b.dims.push_back({in.size(d), in.stride(d), out.stride(d)});
    }
  }
  b.dims.resize(std::max<size_t>(b.dims.size(), 1), {1, 0, 0});
  return b;
}
// Per-pass options: four-step mapping and twiddles, Bluestein chirps (zero lengths = n), or one global stage
struct Pass {
  int64_t n_sig = 0, n_in = 0, n_out = 0, ld_mul = 1, out_k_mul = 1, global = 0, ns = 1;
  double scale = 1;
  std::optional<Tensor> cl, cs, tt;
  bool mirror_out = false, i0_dim1 = false;
};
struct Launch {
  Tensor in, out, tw;
  std::optional<Tensor> cl, cs, tt;
  FFTParams params;
  MTLComputePipelineState_t pso;
  MTLSize grid, group;
  NSUInteger smem;
};
Launch prepare(const Tensor& in, const Tensor& out, int64_t n, bool inv, bool half, const Batch& b, const Pass& x) {
  TORCH_CHECK(canUse32BitIndexMath(in) && canUse32BitIndexMath(out),
              "MPS FFT: tensors must have fewer than 2^31 elements");
  const auto radix = radices(x.global ? x.global : n, x.global ? kGlobalRadices : kRadices);
  TORCH_INTERNAL_ASSERT(!radix.empty(), "no radix plan for ", n);
  // Kernel family: a global radix, a fixed length, or the smallest radix set factoring n (p16: lengths 1 and 16)
  const bool fixed = (n >= 64 && (n & (n - 1)) == 0) || n == 200 || n == 240;
  const auto set = n % 7 == 0 ? "p2357" : n % 3 == 0 || n % 5 == 0 ? "p235" : n == 1 || n == 16 ? "p16" : "p2";
  const auto family = x.global ? "g" + std::to_string(x.global) : fixed ? std::to_string(n) : set;
  const auto n_sig = x.n_sig ? x.n_sig : n, n_in = x.n_in ? x.n_in : n, n_out = x.n_out ? x.n_out : n;
  const bool ld = x.cl || n_sig < n * x.ld_mul || (x.ld_mul > 1 && in.is_complex() && n_in < n_sig);
  const bool st = x.out_k_mul > 1 || x.cs, pairs = (half || x.global) && !out.is_complex(), tw4 = x.tt && !x.global;
  const uint32_t F =
      inv | (half ? 2 : 0) | (tw4 ? 4 : 0) | (st ? 8 : 0) | (ld ? 16 : 0) | (x.global ? 32 : 0) | (pairs ? 64 : 0);
  const int64_t ept = n == 1024 && !(F & 60) ? 16 : 8, rows = b.dims[0].size;
  const int64_t J = !x.global ? (n + ept - 1) / ept : x.global == 1 ? 256 : 64;
  // Batch small transforms (<= 5% idle lanes), tile strided axes; pad contiguous rows, interleave strided ones
  const bool strided = b.si != 1 || b.so != 1;
  const int64_t padded = n + n / 16 + 1;
  int64_t B = strided ? std::min<int64_t>({kMaxPoints / n, 1024 / J, std::max<int64_t>(4, 256 / J)}) : 64 / J;
  int64_t c = std::max<int64_t>(B, 1);
  for (; !strided && c <= 8 && c * padded <= kMaxPoints && ((c * J + 31) / 32 * 32 - c * J) * 20 > c * J; c++) {
  }
  B = std::clamp<int64_t>(!strided && c <= 8 && c * padded <= kMaxPoints ? c : B, 1, std::min<int64_t>(rows, 32));
  B = strided && rows > B && rows <= B + B / 2 && rows * n <= kMaxPoints && rows * J <= 1024 ? rows : B;
  const bool pad = !strided && B * (padded - (B == 1)) <= kMaxPoints;
  const int64_t row = strided ? 1 : pad ? padded - (B == 1) : n + (B > 1);
  TORCH_INTERNAL_ASSERT(b.dims.size() <= c10::metal::max_ndim);
  FFTParams params{};
  params.n = n, params.n_sig = n_sig, params.n_in = n_in, params.n_out = n_out, params.tgb = B, params.jt = J;
  params.ndim = b.dims.size(), params.nstages = radix.size(), params.in_stride = b.si, params.out_stride = b.so;
  params.in_unit = b.si / x.ld_mul, params.scale = x.scale, params.ld_mul = x.ld_mul, params.out_k_mul = x.out_k_mul;
  params.mirror_out = x.mirror_out, params.i0_dim1 = x.i0_dim1, params.via_smem = (b.si == 1) != (b.so == 1);
  params.row = row, params.padshift = pad ? 4 : 31, params.es = strided ? B : 1;
  for (uint32_t s = 0, ns = x.ns; s < radix.size(); ns *= radix[s++]) {
    params.radix[s] = radix[s];
    params.ns[s] = ns;
    params.twstep[s] = n / radix[s] / ns * (half ? 2 : 1);
    params.inv_ns[s] = 1.0f / ns;
  }
  for (const auto d : c10::irange(b.dims.size())) {
    params.sizes[d] = b.dims[d].size;
    params.in_strides[d] = b.dims[d].in_stride;
    params.out_strides[d] = b.dims[d].out_stride;
  }
  const bool contig = !strided && B == 1 && n >= 512 && (n & (n - 1)) == 0 && !(F & 60);
  const auto name = fmt::format(
      "fft_{}_{}_{}{}_{}", scalarToMetalTypeString(in), scalarToMetalTypeString(out), family, contig ? "c" : "", F);
  Launch l{in, out, twiddles(half ? 2 * n : n), x.cl, x.cs, x.tt, params, lib.getPipelineStateForFunc(name)};
  const int64_t ny = b.dims.size() > 1 ? b.dims[1].size : 1,
                threads = x.global == 1 ? n / 2 + 1 : n / std::max<int64_t>(x.global, 1);
  l.grid = x.global ? MTLSizeMake((threads + J - 1) / J, rows, b.numel / rows)
                    : MTLSizeMake((rows + B - 1) / B, ny, b.numel / rows / ny);
  l.group = MTLSizeMake(B * J, 1, 1);
  const int64_t bytes = B * (strided ? n : row) * (strided || B > 1 ? 2 : 3) * sizeof(float);
  l.smem = x.global ? 16 : std::min<int64_t>(32768, (bytes + 15) / 16 * 16);
  return l;
}
void run(const std::vector<Launch>& launches) {
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      for (const auto& l : launches) {
        getMPSProfiler().beginProfileKernel(l.pso, "fft", {l.in, l.out}, stream);
        [encoder setComputePipelineState:l.pso];
        mtl_setArgs(encoder, l.in, l.out, l.cl, l.cs, l.tt, l.tw, l.params);
        [encoder setThreadgroupMemoryLength:l.smem atIndex:0];
        [encoder dispatchThreadgroups:l.grid threadsPerThreadgroup:l.group];
        getMPSProfiler().endProfileKernel(l.pso, stream);
      }
    }
  });
}
// n-point transforms of the batch from `in` into `out`: x.n_in inputs (c2r: n/2+1, mirrored), x.n_out outputs
void fft_batch(const Tensor& in, const Tensor& out, const Batch& b, int64_t n, bool inv, const Pass& x) {
  const auto n_sig = x.n_sig ? x.n_sig : n, n_in = x.n_in ? x.n_in : n, n_out = x.n_out ? x.n_out : n;
  const bool real = !in.is_complex() || !out.is_complex();
  const int64_t m = real && n % 2 == 0 ? n / 2 : n;
  const bool half = !x.cl && !x.cs && n > 2 && m < n && m <= kMaxPoints;
  // Strided on both sides past 1024 points the tiles no longer coalesce: two tile passes move less
  const bool both = std::abs(b.si) != 1 && std::abs(b.so) != 1 && !x.cs && (half ? m : n) > 1024;
  if ((half ? m : n) <= kMaxPoints && !both) {
    run({prepare(in, out, half ? m : n, inv, half, b, x)});
    return;
  }
  // Long non-pow2 lengths in fp32: one global-memory stage per radix between two contiguous temps, then the unpack
  const bool plain = !x.cl && !x.cs && !x.n_sig;
  const bool fp32 = c10::toRealValueType(in.scalar_type()) == kFloat && (!real || n % 2 == 0);
  auto gplan = radices(m, kGlobalRadices);
  if (n >= 1 << 17 && (n & (n - 1)) && plain && fp32 && !gplan.empty()) {
    gplan.resize(gplan.size() + !in.is_complex(), 1);
    const auto tmp = at::empty({2, b.numel * m}, out.options().dtype(kComplexFloat));
    const auto tt = twiddles(m, true);
    std::vector<Launch> launches;
    for (size_t s = 0, ns = 1; s < gplan.size(); ns *= gplan[s++]) {
      const bool first = s == 0, last = s + 1 == gplan.size();
      Batch bp{{}, b.numel, first ? b.si : 1, last ? b.so : 1};
      for (int64_t d = 0, tstride = m; d < static_cast<int64_t>(b.dims.size()); tstride *= b.dims[d++].size) {
        bp.dims.push_back(
            {b.dims[d].size, first ? b.dims[d].in_stride : tstride, last ? b.dims[d].out_stride : tstride});
      }
      const Pass pass{
          .n_out = last ? n_out : m, .global = gplan[s], .ns = int64_t(ns), .scale = last ? x.scale : 1, .tt = tt};
      launches.push_back(
          prepare(first ? in : tmp[s % 2], last ? out : tmp[(s + 1) % 2], m, inv, first && real, bp, pass));
    }
    run(launches);
    return;
  }
  // Four-step: n = n1*n2, temp [outer batch][n2][n1][inner batch]; pass 1 transforms strided n1-point columns
  // times W_n^(k1*i0), pass 2 writes k1+n1*k2 (n1 <= 256 if it recurses). Real transforms halve one pass.
  const bool big = plain && b.numel * n >= 1 << 17;
  bool r2c = big && !in.is_complex(), c2r = big && !out.is_complex();
  const bool nest = !x.cs && n_out == n && out.is_complex();
  // n1: the largest smooth divisor <= sqrt(n) whose cofactor fits a pass; nr: with an even halved real pass
  int64_t n1 = 0, nr = 0;
  for (int64_t d = 1; d * d <= n; d++) {
    const int64_t q = n / d, h = r2c ? d : q;
    if (n % d || !is_smooth(d) || !is_smooth(q)) {
      continue;
    }
    nr = q <= kMaxPoints && h % 2 == 0 && h > 2 && h <= 2 * kMaxPoints ? d : nr;
    n1 = (q <= kMaxPoints || nest) && (!nest || d <= 256) ? d : n1;
  }
  r2c = r2c && nr, c2r = c2r && nr, n1 = r2c || c2r ? nr : n1;
  TORCH_CHECK(n1 > 0, "MPS FFT does not support transforms of length ", n);
  const auto n2 = n / n1, rows = c2r ? n2 / 2 + 1 : n2, rl = r2c ? n1 / 2 + 1 : n1;
  const bool inner = b.dims[0].size > 1 && std::abs(b.dims[0].in_stride) < std::abs(b.si);
  const int64_t ninner = inner ? b.dims[0].size : 1;
  Batch b1{{}, b.numel * rows, b.si * n2, ninner}, b2{{}, b.numel * rl, rl * ninner, b.so * n1};
  for (int64_t d = 0, t = 1; d < static_cast<int64_t>(b.dims.size()); t *= b.dims[d++].size) {
    const int64_t ts = d < inner ? t : t * rows * rl; // The temp's stride of batch dim d
    b1.dims.push_back({b.dims[d].size, b.dims[d].in_stride, ts});
    b2.dims.push_back({b.dims[d].size, ts, b.dims[d].out_stride});
  }
  b1.dims.insert(b1.dims.begin() + inner, Dim{rows, b.si, rl * ninner});
  b2.dims.insert(b2.dims.begin() + inner, Dim{rl, ninner, b.so});
  const bool inplace = !r2c && !c2r && n_out == n && out.is_complex() && n2 > 16 && b.so == ninner &&
      !in.is_alias_of(out) && std::all_of(b2.dims.begin(), b2.dims.end(), [](const Dim& d) {
        return d.size == 1 || d.in_stride == d.out_stride;
      });
  auto tmp =
      inplace ? out : at::empty({b.numel * rows * rl}, out.options().dtype(c10::toComplexType(out.scalar_type())));
  const Pass load{
      .n_sig = n_sig, .n_in = n_in, .n_out = rl, .ld_mul = n2, .cl = x.cl, .tt = twiddles(n, true), .i0_dim1 = inner};
  auto first = prepare(in, tmp, r2c ? n1 / 2 : n1, inv, r2c, b1, load);
  if (r2c || c2r || n2 <= 1024 || !nest) {
    Pass store{.n_sig = n, .n_out = c2r ? n2 / 2 : n_out, .out_k_mul = c2r ? 1 : n1, .scale = x.scale, .cs = x.cs};
    store.mirror_out = r2c, store.i0_dim1 = inner;
    run({first, prepare(tmp, out, c2r ? n2 / 2 : n2, inv, c2r, b2, store)});
    return;
  }
  run({first});
  fft_batch(tmp, out, b2, n2, inv, {.scale = x.scale});
}
// n-point transforms along `dim`; lengths with other prime factors use Bluestein's algorithm
void fft_pass(const Tensor& in, const Tensor& out, int64_t dim, int64_t n, bool inv, const Pass& x = {}) {
  if (is_smooth(n)) {
    return fft_batch(in, out, collapse(in, out, dim), n, inv, x);
  }
  // Bluestein: X = c * IFFT_M(FFT_M(x * c) * B)[:n], M a power of two >= 2n - 1, chirps folded into the kernels
  const auto& [c, B] = chirp_tables(n, inv);
  const auto M = B.numel();
  DimVector sizes(in.sizes());
  sizes[dim] = M;
  auto tmp = at::empty(sizes, in.options().dtype(c10::toComplexType(in.scalar_type())));
  fft_pass(in, tmp, dim, M, false, {.n_sig = n, .n_in = x.n_in ? x.n_in : n, .cl = c, .cs = B});
  fft_pass(tmp, out, dim, M, true, {.n_out = x.n_out ? x.n_out : n, .scale = x.scale / M, .cs = c});
}
double norm_scale(int64_t normalization, const Tensor& t, IntArrayRef dims) {
  const auto n = std::accumulate(dims.begin(), dims.end(), 1.0, [&](double a, int64_t d) { return a * t.size(d); });
  return std::pow(n, -0.5 * normalization); // fft_norm_mode: none, by_root_n, by_n
}
} // anonymous namespace

Tensor _fft_c2r_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto out = at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return _fft_c2r_mps_out(self, dim, normalization, last_dim_size, out);
}

Tensor _fft_r2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  auto out = at::empty({0}, self.options().dtype(c10::toComplexType(self.scalar_type())));
  return _fft_r2c_mps_out(self, dim, normalization, onesided, out);
}

Tensor _fft_c2c_mps(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }
  auto out = at::empty(self.sizes(), self.options());
  return _fft_c2c_mps_out(self, dim, normalization, forward, out);
}

Tensor& _fft_r2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided, Tensor& out) {
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kHalf, "Only float and half dtypes are supported");
  TORCH_CHECK(out.scalar_type() == c10::toComplexType(self.scalar_type()));
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  const auto input_sizes = self.sym_sizes();
  SymDimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }

  const Pass x{.n_out = out.size(last_dim), .scale = norm_scale(normalization, self, dim)};
  fft_pass(self, out, last_dim, self.size(last_dim), false, x);
  for (const auto i : c10::irange(dim.size() - 1)) {
    fft_pass(out, out, dim[i], self.size(dim[i]), false);
  }
  return out;
}

Tensor& _fft_c2r_mps_out(const Tensor& self,
                         IntArrayRef dim,
                         int64_t normalization,
                         int64_t last_dim_size,
                         Tensor& out) {
  TORCH_CHECK(self.is_complex(), "Input must be complex");
  TORCH_CHECK(out.scalar_type() == c10::toRealValueType(self.scalar_type()), "Unexpected output type");
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  TORCH_INTERNAL_ASSERT(self.size(dim.back()) == last_dim_size / 2 + 1,
                        "_fft_c2r: expected ",
                        last_dim_size / 2 + 1,
                        " elements in dim ",
                        dim.back(),
                        ", but got ",
                        self.size(dim.back()));
  const auto in_sizes = self.sym_sizes();
  SymDimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }

  const auto tmp = dim.size() > 1 ? at::empty(self.sizes(), self.options()) : self;
  for (const auto i : c10::irange(dim.size() - 1)) {
    fft_pass(i ? tmp : self, tmp, dim[i], self.size(dim[i]), true);
  }
  const Pass x{.n_in = self.size(dim.back()), .n_out = last_dim_size, .scale = norm_scale(normalization, out, dim)};
  fft_pass(tmp, out, dim.back(), last_dim_size, true, x);
  return out;
}

Tensor& _fft_c2c_mps_out(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward, Tensor& out) {
  TORCH_CHECK(self.is_complex());
  TORCH_CHECK(out.device() == self.device(), "Expected out tensor on ", self.device(), " but got ", out.device());
  at::native::resize_output_symint(out, self.sym_sizes());
  if (out.numel() == 0) {
    return out;
  }

  if (dim.empty()) {
    return out.copy_(self);
  }
  const auto scale = norm_scale(normalization, self, dim);
  for (const auto i : c10::irange(dim.size())) {
    fft_pass(i ? out : self, out, dim[i], self.size(dim[i]), !forward, {.scale = i ? 1.0 : scale});
  }
  return out;
}

} // namespace at::native
