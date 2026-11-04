#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/FFT.h>
#include <c10/util/safe_conv.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>

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
#include <ATen/ops/real.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {
using namespace mps;
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FFT_metallib.h>
#endif
// Longer transforms take several passes
constexpr int64_t kMaxPoints = fft_max_points;
// Chirps can only ride on two passes: longer Bluestein transforms multiply by them separately
constexpr int64_t kMaxTwoPassPoints = kMaxPoints * kMaxPoints;
// Metal's guarantees for a threadgroup: threads, memory and its alignment
constexpr int64_t kMaxGroupThreads = 1024, kMaxSmemBytes = 32768, kSmemAlign = 16;
constexpr int64_t kFloatsPerPoint = 2;
static_assert(kMaxPoints * kFloatsPerPoint * sizeof(float) == kMaxSmemBytes);
// Past this many bytes the least recently used twiddles and chirps go, so sweeping many lengths cannot grow memory
// without bound. The newest tables always stay, because one long transform can need more than the limit by itself.
constexpr int64_t kMaxCachedTableBytes = 1 << 28;
constexpr size_t kMinCachedTables = 32;
#define FFT_LIST_ITEM(value) value,
constexpr auto kRadices = {FFT_STAGE_RADICES(FFT_LIST_ITEM)}, kGlobalRadices = {FFT_GLOBAL_RADICES(FFT_LIST_ITEM)};
constexpr const char* kIndexLimitMessage = "MPS FFT: a single transform over 2^31 elements or more is not supported";
// Plan heuristics, measured on an M5 Pro (the only GPU tuned so far). Another GPU family would get its own instance.
struct FFTTuning {
  int64_t contiguous_group_threads = 64; // Target threads per group for contiguous rows
  int64_t max_contiguous_batch = 8; // Rows a group may take to fill SIMD lanes
  int64_t idle_lane_ratio = 20; // Accept at most 1 idle SIMD lane in this many
  int64_t max_group_batch = 32; // Rows per group of any kind
  int64_t tile_threads = 256, large_tile_threads = 512; // Threads per group of strided tiles
  int64_t large_tile_points = 1 << 22; // Total points from which tiles use large_tile_threads
  int64_t large_tile_length = 512,
          large_tile_batch = 1024; // The same for this many single-pass transforms of this length
  int64_t min_tile_batch = 4; // Rows per strided tile
  int64_t max_tile_axis = 1024; // Longer strided axes take two passes, unless the batch is large
  int64_t large_batch = 2048; // Transforms from which pow2 axes stay in one global-memory pass
  int64_t half_real_min_points = 1 << 17; // Below this long real transforms are latency-bound: halving gains nothing
  int64_t always_global_length = 1 << 17; // From here global-memory passes need no comparison with two-pass plans
  int64_t global_threads = 64, global_unpack_threads = 256; // Threads per group of global-memory passes
  int64_t runtime_stage_cost = 2; // Cost of a runtime-planned stage, a compile-time planned one being 1
  int64_t single_row_floats_per_point = 3; // 1.5x smem for one contiguous row: about 5% faster at 2048 points and up
  int64_t min_in_place_points = 16; // Second passes above this length may use the output as their input
};
constexpr FFTTuning kTuning;
bool is_one_of(int64_t n, std::initializer_list<int64_t> values) {
  return std::ranges::find(values, n) != values.end();
}
bool is_pow2(int64_t n) {
  return std::has_single_bit(static_cast<uint64_t>(n));
}
// Radix sequence of n over `rs` (largest first; {1} for n = 1), empty when n has other prime factors
c10::SmallVector<uint32_t, fft_max_stages> radices(int64_t n, std::initializer_list<int> rs) {
  c10::SmallVector<uint32_t, fft_max_stages> plan(n == 1, 1);
  for (const auto r : rs) {
    for (; n % r == 0; n /= r) {
      plan.push_back(r);
    }
  }
  return n == 1 ? plan : decltype(plan){};
}
// Lengths with compile-time planned kernels; longer pow2 lengths recurse into them
bool fixed_length(int64_t n) {
  return (n > kMaxPoints && is_pow2(n)) ||
      is_one_of(n, {FFT_FIXED_POW2_LENGTHS(FFT_LIST_ITEM) FFT_FIXED_SMOOTH_LENGTHS(FFT_LIST_ITEM)});
}
int split_cost(int64_t n) {
  return radices(n, kRadices).size() * (fixed_length(n) ? 1 : kTuning.runtime_stage_cost);
}
bool is_smooth(int64_t n) {
  return !radices(n, kRadices).empty();
}
enum class TableKind : uint8_t { twiddles, long_axis_twiddles, forward_chirp, inverse_chirp };
template <class Make>
std::pair<Tensor, Tensor> cached_tables(int64_t n, TableKind kind, Make make) {
  struct Entry {
    std::pair<Tensor, Tensor> tables;
    int64_t bytes;
    uint64_t last_use;
  };
  static std::mutex mutex;
  static std::map<std::pair<int64_t, TableKind>, Entry> cache;
  static int64_t cached_bytes = 0;
  static uint64_t clock = 0;
  std::lock_guard<std::mutex> guard(mutex);
  const auto found = cache.find({n, kind});
  if (found != cache.end()) {
    found->second.last_use = ++clock;
    return found->second.tables;
  }
  auto tables = make();
  const int64_t bytes = tables.first.nbytes() + (tables.second.defined() ? tables.second.nbytes() : 0);
  cached_bytes += bytes;
  cache.emplace(std::pair(n, kind), Entry{tables, bytes, ++clock});
  while (cached_bytes > kMaxCachedTableBytes && cache.size() > kMinCachedTables) {
    const auto oldest = std::ranges::min_element(cache, {}, [](const auto& item) { return item.second.last_use; });
    cached_bytes -= oldest->second.bytes;
    cache.erase(oldest);
  }
  return tables;
}
// W_n^k for k < n, or (long_axis, for inter-pass twiddles) W_n^b for low-entry b then W_n^(a * low entries)
Tensor twiddles(int64_t n, bool long_axis = false) {
  return cached_tables(n,
                       long_axis ? TableKind::long_axis_twiddles : TableKind::twiddles,
                       [&] {
                         const int64_t low_entries = fft_twiddle_low_entries;
                         const auto k = long_axis ? at::cat({at::arange(low_entries, at::kDouble),
                                                             at::arange(0, n, low_entries, at::kDouble)})
                                                  : at::arange(n, at::kDouble);
                         const auto table = at::polar(at::ones_like(k), k.mul(-2 * M_PI / n));
                         return std::pair(table.to(at::device(kMPS).dtype(kComplexFloat)), Tensor());
                       })
      .first;
}
// Bluestein chirp exp(-+ i pi k^2 / n) for k < n, and the M-point FFT of its conjugate wrapped around zero
std::pair<Tensor, Tensor> chirp_tables(int64_t n, bool inverse) {
  return cached_tables(n, inverse ? TableKind::inverse_chirp : TableKind::forward_chirp, [&] {
    const auto k = at::arange(n, at::kLong);
    const auto angle = k.mul(k).remainder(2 * n).to(at::kDouble).mul((inverse ? 1 : -1) * M_PI / n);
    const auto c = at::polar(at::ones_like(angle), angle);
    const auto M = static_cast<int64_t>(std::bit_ceil<uint64_t>(2 * n - 1));
    const auto b = at::cat({c.conj(), at::zeros({M - 2 * n + 1}, c.options()), c.narrow(0, 1, n - 1).conj().flip(0)});
    const auto opts = at::device(kMPS).dtype(kComplexFloat);
    return std::pair(c.to(opts), at::fft_fft(b).to(opts));
  });
}
struct Dim {
  int64_t size, in_stride, out_stride;
};
struct Batch {
  c10::SmallVector<Dim, c10::metal::max_ndim> dims; // Innermost first
  int64_t numel = 1, in_stride = 0, out_stride = 0; // Transforms, and the element strides along the axis
};
// Batch dims of a pass along `dim` (every other dim), innermost first, merged where both tensors allow it
Batch collapse(const Tensor& in, const Tensor& out, int64_t dim) {
  DimVector dims;
  for (const auto d : c10::irange(in.dim())) {
    if (d != dim && in.size(d) != 1) {
      dims.push_back(d);
    }
  }
  std::ranges::sort(dims, {}, [&](auto d) { return std::pair(std::abs(in.stride(d)), std::abs(out.stride(d))); });
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
  int64_t n_sig = 0, n_in = 0, n_out = 0, ld_mul = 1, out_k_mul = 1, global_radix = 0, start_ns = 1;
  double scale = 1;
  std::optional<Tensor> load_chirp, store_chirp, pass_twiddles;
  bool mirror_out = false, i0_dim1 = false;
};
struct Launch {
  Tensor in, out, tw;
  std::optional<Tensor> load_chirp, store_chirp, pass_twiddles;
  FFTParams params;
  MTLComputePipelineState_t pso;
  MTLSize grid, group;
  NSUInteger smem;
};
// Rows per threadgroup. Strided axes share tiles; contiguous rows batch until SIMD lanes are nearly all busy.
int64_t group_batch(int64_t n, int64_t threads, int64_t padded, const Batch& b, uint32_t flags, bool strided) {
  const auto rows = b.dims[0].size;
  const auto max_batch = std::min(rows, kTuning.max_group_batch);
  if (strided) {
    const bool single_pass = !(flags & fft_several_pass_flags);
    const bool large = b.numel * n >= kTuning.large_tile_points ||
        (n == kTuning.large_tile_length && single_pass && b.numel >= kTuning.large_tile_batch);
    const auto tile_threads = large ? kTuning.large_tile_threads : kTuning.tile_threads;
    const auto tile = std::clamp<int64_t>(
        std::min(
            {kMaxPoints / n, kMaxGroupThreads / threads, std::max(kTuning.min_tile_batch, tile_threads / threads)}),
        1,
        max_batch);
    // Absorb a tail of up to half a tile into one group
    const bool absorb_tail =
        rows > tile && rows <= tile + tile / 2 && rows * n <= kMaxPoints && rows * threads <= kMaxGroupThreads;
    return absorb_tail ? rows : tile;
  }
  const auto base = kTuning.contiguous_group_threads / threads;
  const auto fits = [&](int64_t batch) {
    return batch <= kTuning.max_contiguous_batch && batch * padded <= kMaxPoints;
  };
  const auto mostly_busy = [&](int64_t batch) {
    const auto lanes = batch * threads;
    return (at::round_up<int64_t>(lanes, c10::metal::simdgroup_size) - lanes) * kTuning.idle_lane_ratio <= lanes;
  };
  auto batch = std::max<int64_t>(base, 1);
  while (fits(batch) && !mostly_busy(batch)) {
    batch++;
  }
  return std::clamp<int64_t>(fits(batch) ? batch : base, 1, max_batch);
}
Launch prepare(const Tensor& in, const Tensor& out, int64_t n, bool inv, bool half, const Batch& b, const Pass& x) {
  TORCH_CHECK_NOT_IMPLEMENTED(canUse32BitIndexMath(in) && canUse32BitIndexMath(out), kIndexLimitMessage);
  const auto radix = radices(x.global_radix ? x.global_radix : n, x.global_radix ? kGlobalRadices : kRadices);
  TORCH_INTERNAL_ASSERT(!radix.empty(), "no radix plan for ", n);
  const auto n_sig = x.n_sig ? x.n_sig : n, n_in = x.n_in ? x.n_in : n, n_out = x.n_out ? x.n_out : n;
  const bool extended_load = x.load_chirp || n_sig < n * x.ld_mul || (x.ld_mul > 1 && in.is_complex() && n_in < n_sig);
  const bool extended_store =
      x.store_chirp || (x.out_k_mul > 1 && (!out.is_complex() || x.mirror_out || n_out < n_sig));
  const bool real_store = (half || x.global_radix) && !out.is_complex();
  const bool twiddled = x.pass_twiddles && !x.global_radix;
  const uint32_t flags = (inv ? fft_inverse : 0) | (half ? fft_packed_load : 0) |
      (twiddled ? fft_four_step_twiddle : 0) | (extended_store ? fft_extended_store : 0) |
      (extended_load ? fft_extended_load : 0) | (x.global_radix ? fft_global_stage : 0) |
      (real_store ? fft_packed_real_store : 0);
  const bool single_pass = !(flags & fft_several_pass_flags);
  const bool radix16_or_1 = n == 1 || n == 16; // The p16 kernels have only these two stage radices
  // Kernel family: a global radix, a fixed length, or the smallest radix set factoring n
  const std::string_view radix_set = n % 11 == 0 || n % 13 == 0 ? "all"
      : n % 7 == 0                                              ? "p2357"
      : n % 3 == 0 || n % 5 == 0                                ? "p235"
      : radix16_or_1 && single_pass                             ? "p16"
                                                                : "p2";
  const bool fixed = fixed_length(n) || (single_pass && is_one_of(n, {FFT_FIXED_SINGLE_PASS_LENGTHS(FFT_LIST_ITEM)}));
  const auto family = x.global_radix ? "g" + std::to_string(x.global_radix)
      : fixed                        ? std::to_string(n)
                                     : std::string(radix_set);
  const bool strided = b.in_stride != 1 || b.out_stride != 1;
  const int64_t points_per_thread = fft_points_per_thread(n, flags, !in.is_complex(), !strided);
  const int64_t rows = b.dims[0].size;
  // A single stage has one thread per butterfly
  const int64_t threads = !x.global_radix ? (radix.size() == 1 ? 1 : at::ceil_div(n, points_per_thread))
      : x.global_radix == 1               ? kTuning.global_unpack_threads
                                          : kTuning.global_threads;
  // Contiguous rows are padded when they fit, and strided ones interleave in smem
  const int64_t padded = n + (n >> fft_pad_shift) + 1;
  const int64_t batch = group_batch(n, threads, padded, b, flags, strided);
  const bool pad = !strided && batch * (padded - (batch == 1)) <= kMaxPoints;
  const int64_t row = strided ? 1 : pad ? padded - (batch == 1) : n + (batch > 1);
  TORCH_INTERNAL_ASSERT(b.dims.size() <= c10::metal::max_ndim);
  const auto u32 = [](int64_t value) { return c10::safe_conv<uint32_t>(value); };
  const auto i32 = [](int64_t value) { return c10::safe_conv<int32_t>(value); };
  FFTParams params{
      .n = u32(n),
      .n_sig = u32(n_sig),
      .n_in = u32(n_in),
      .n_out = u32(n_out),
      .batch_per_group = u32(batch),
      .threads_per_transform = u32(threads),
      .ndim = u32(b.dims.size()),
      .nstages = u32(radix.size()),
      .in_stride = i32(b.in_stride),
      .in_unit = i32(b.in_stride / x.ld_mul),
      .out_stride = i32(b.out_stride),
      .scale = static_cast<float>(x.scale),
      .ld_mul = u32(x.ld_mul),
      .out_k_mul = u32(x.out_k_mul),
      .row = u32(row),
      .pad_shift = pad ? fft_pad_shift : fft_no_pad_shift,
      .slot_stride = u32(strided ? batch : 1),
      .mirror_out = x.mirror_out,
      .i0_dim1 = x.i0_dim1,
      .via_smem = (b.in_stride == 1) != (b.out_stride == 1),
  };
  for (uint32_t s = 0, ns = u32(x.start_ns); s < radix.size(); ns *= radix[s++]) {
    params.radix[s] = radix[s];
    params.ns[s] = ns;
    params.twstep[s] = u32(n / radix[s] / ns * (half ? 2 : 1));
    params.inv_ns[s] = 1.0f / ns;
  }
  for (const auto d : c10::irange(b.dims.size())) {
    params.sizes[d] = u32(b.dims[d].size);
    params.in_strides[d] = i32(b.dims[d].in_stride);
    params.out_strides[d] = i32(b.dims[d].out_stride);
  }
  using namespace std::string_view_literals;
  const bool contiguous_kernel =
      !strided && batch == 1 && single_pass && is_one_of(n, {FFT_CONTIGUOUS_LENGTHS(FFT_LIST_ITEM)});
  const auto name = fmt::format("fft_{}_{}_{}{}_{}",
                                scalarToMetalTypeString(in),
                                scalarToMetalTypeString(out),
                                family,
                                contiguous_kernel ? "c"sv : ""sv,
                                flags);
  const int64_t axis_threads = x.global_radix == 1 ? n / 2 + 1 : n / std::max<int64_t>(x.global_radix, 1);
  const auto grid = x.global_radix ? MTLSizeMake(at::ceil_div(axis_threads, threads), rows, b.numel / rows)
                                   : MTLSizeMake(at::ceil_div(rows, batch) * (b.numel / rows), 1, 1);
  const int64_t floats_per_point = strided || batch > 1 ? kFloatsPerPoint : kTuning.single_row_floats_per_point;
  const int64_t smem_bytes = batch * (strided ? n : row) * floats_per_point * sizeof(float);
  return Launch{in,
                out,
                twiddles(half ? 2 * n : n),
                x.load_chirp,
                x.store_chirp,
                x.pass_twiddles,
                params,
                lib.getPipelineStateForFunc(name),
                grid,
                MTLSizeMake(batch * threads, 1, 1),
                static_cast<NSUInteger>(
                    x.global_radix ? kSmemAlign : std::min(kMaxSmemBytes, at::round_up(smem_bytes, kSmemAlign)))};
}
void run(c10::ArrayRef<Launch> launches) {
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      for (const auto& l : launches) {
        getMPSProfiler().beginProfileKernel(l.pso, "fft", {l.in, l.out}, stream);
        [encoder setComputePipelineState:l.pso];
        mtl_setArgs(encoder, l.in, l.out, l.load_chirp, l.store_chirp, l.pass_twiddles, l.tw, l.params);
        [encoder setThreadgroupMemoryLength:l.smem atIndex:0];
        [encoder dispatchThreadgroups:l.grid threadsPerThreadgroup:l.group];
        getMPSProfiler().endProfileKernel(l.pso, stream);
      }
    }
  });
}
// n-point transforms of the batch from `in` into `out`: x.n_in inputs (c2r: n/2+1, mirrored), x.n_out outputs.
// Returns false, having launched nothing, for what its passes cannot do: fft_pass then takes a longer way round.
bool fft_batch(const Tensor& in, const Tensor& out, const Batch& b, int64_t n, bool inv, const Pass& x) {
  const auto n_sig = x.n_sig ? x.n_sig : n, n_in = x.n_in ? x.n_in : n, n_out = x.n_out ? x.n_out : n;
  const bool real = !in.is_complex() || !out.is_complex();
  const int64_t half_length = real && n % 2 == 0 ? n / 2 : n;
  const bool half = !x.load_chirp && !x.store_chirp && n > 2 && half_length < n && half_length <= kMaxPoints;
  const int64_t length = half ? half_length : n;
  // Split long strided axes for small batches; large pow2 batches favor fewer global-memory passes.
  const bool split_axis = std::abs(b.in_stride) != 1 && std::abs(b.out_stride) != 1 && !x.store_chirp &&
      length > kTuning.max_tile_axis && (b.numel < kTuning.large_batch || !is_pow2(n));
  if (length <= kMaxPoints && !split_axis) {
    run(prepare(in, out, length, inv, half, b, x));
    return true;
  }
  if (c10::toRealValueType(in.scalar_type()) == kHalf) {
    return false;
  }
  const bool plain = !x.load_chirp && !x.store_chirp && !x.n_sig;
  auto global_plan = radices(half_length, kGlobalRadices);
  // Global passes read packed real data or every complex input: they cannot mirror a complex one
  const bool global_inputs = real ? n % 2 == 0 : n_in == n;
  const bool global = n > kMaxPoints && !is_pow2(n) && plain && global_inputs && !global_plan.empty();
  // Four-step: n = n1*n2, temp [outer batch][n2][n1][inner batch]; pass 1 transforms strided n1-point columns
  // times W_n^(k1*i0), pass 2 writes k1+n1*k2 (n1 <= max_tile_axis if it recurses). Real transforms halve one pass.
  const bool big = plain && b.numel * n >= kTuning.half_real_min_points;
  bool r2c = big && !in.is_complex(), c2r = big && !out.is_complex();
  const bool nest = !x.store_chirp && n_out == n && out.is_complex();
  // Prefer fixed-stage plans and fewer passes; n1_real balances the half-length real transform.
  int64_t n1 = 0, n1_real = 0;
  int best = std::numeric_limits<int>::max();
  int64_t best_real = best;
  for (int64_t d = 1; (!global || n < kTuning.always_global_length) && d * d <= n; d++) {
    const int64_t q = n / d;
    if (n % d || !is_smooth(d) || !is_smooth(q)) {
      continue;
    }
    const bool extra =
        nest && q > kTuning.max_tile_axis && (q > kMaxPoints || b.numel * d < kTuning.large_batch || !is_pow2(q));
    const auto cost = split_cost(d) + split_cost(q) + extra;
    if ((q <= kMaxPoints || nest) && (!nest || d <= kTuning.max_tile_axis) && cost <= best) {
      n1 = d;
      best = cost;
    }
    for (const auto first : {d, q}) {
      const auto second = n / first, halved = r2c ? first : second;
      const auto first_points = r2c ? first / 2 : first, second_points = c2r ? second / 2 : second;
      if (!(r2c || c2r) || halved % 2 || halved <= 2 || first_points > kMaxPoints || second_points > kMaxPoints) {
        continue;
      }
      const auto real_cost = std::max(first_points, second_points);
      if (real_cost < best_real || (real_cost == best_real && first > n1_real)) {
        n1_real = first;
        best_real = real_cost;
      }
    }
  }
  // Include the final unpack for real input when comparing global and threadgroup plans.
  const auto passes = global_plan.size() + !in.is_complex();
  if (global && (n >= kTuning.always_global_length || best >= 2 * static_cast<int>(passes))) {
    global_plan.resize(passes, 1);
    const auto tmp = at::empty({2, b.numel * half_length}, out.options().dtype(kComplexFloat));
    const auto long_twiddles = twiddles(half_length, true);
    std::vector<Launch> launches;
    launches.reserve(global_plan.size());
    for (size_t s = 0, ns = 1; s < global_plan.size(); ns *= global_plan[s++]) {
      const bool first = s == 0, last = s + 1 == global_plan.size();
      Batch bp{{}, b.numel, first ? b.in_stride : 1, last ? b.out_stride : 1};
      for (int64_t d = 0, tstride = half_length; d < static_cast<int64_t>(b.dims.size()); tstride *= b.dims[d++].size) {
        bp.dims.push_back(
            {b.dims[d].size, first ? b.dims[d].in_stride : tstride, last ? b.dims[d].out_stride : tstride});
      }
      const Pass pass{.n_out = last ? n_out : half_length,
                      .global_radix = global_plan[s],
                      .start_ns = static_cast<int64_t>(ns),
                      .scale = last ? x.scale : 1,
                      .pass_twiddles = long_twiddles};
      launches.push_back(
          prepare(first ? in : tmp[s % 2], last ? out : tmp[(s + 1) % 2], half_length, inv, first && real, bp, pass));
    }
    run(launches);
    return true;
  }
  r2c = r2c && n1_real;
  c2r = c2r && n1_real;
  n1 = r2c || c2r ? n1_real : n1;
  if (!n1) {
    return false;
  }
  const auto n2 = n / n1, rows = c2r ? n2 / 2 + 1 : n2, row_length = r2c ? n1 / 2 + 1 : n1;
  const bool inner = b.dims[0].size > 1 && std::abs(b.dims[0].in_stride) < std::abs(b.in_stride);
  const int64_t ninner = inner ? b.dims[0].size : 1;
  Batch b1{{}, b.numel * rows, b.in_stride * n2, ninner},
      b2{{}, b.numel * row_length, row_length * ninner, b.out_stride * n1};
  for (int64_t d = 0, t = 1; d < static_cast<int64_t>(b.dims.size()); t *= b.dims[d++].size) {
    const int64_t ts = d < inner ? t : t * rows * row_length; // The temp's stride of batch dim d
    b1.dims.push_back({b.dims[d].size, b.dims[d].in_stride, ts});
    b2.dims.push_back({b.dims[d].size, ts, b.dims[d].out_stride});
  }
  b1.dims.insert(b1.dims.begin() + inner, Dim{rows, b.in_stride, row_length * ninner});
  b2.dims.insert(b2.dims.begin() + inner, Dim{row_length, ninner, b.out_stride});
  const bool in_place = !r2c && !c2r && n_out == n && out.is_complex() && n2 > kTuning.min_in_place_points &&
      b.out_stride == ninner && !in.is_alias_of(out) &&
      std::ranges::all_of(b2.dims, [](const Dim& d) { return d.size == 1 || d.in_stride == d.out_stride; });
  auto tmp = in_place ? out : at::empty({b.numel * rows * row_length}, out.options().dtype(kComplexFloat));
  const Pass load{.n_sig = n_sig,
                  .n_in = n_in,
                  .n_out = row_length,
                  .ld_mul = n2,
                  .load_chirp = x.load_chirp,
                  .pass_twiddles = twiddles(n, true),
                  .i0_dim1 = inner};
  auto first = prepare(in, tmp, r2c ? n1 / 2 : n1, inv, r2c, b1, load);
  if (r2c || c2r || n2 <= kTuning.max_tile_axis || !nest) {
    const Pass store{.n_sig = n,
                     .n_out = c2r ? n2 / 2 : n_out,
                     .out_k_mul = c2r ? 1 : n1,
                     .scale = x.scale,
                     .store_chirp = x.store_chirp,
                     .mirror_out = r2c,
                     .i0_dim1 = inner};
    run({std::move(first), prepare(tmp, out, c2r ? n2 / 2 : n2, inv, c2r, b2, store)});
    return true;
  }
  run(first);
  return fft_batch(tmp, out, b2, n2, inv, {.scale = x.scale});
}
// Writes the first `length` points of a full complex result, real part only for a real output
void store_result(const Tensor& full, const Tensor& out, int64_t dim, int64_t length) {
  const auto part = full.narrow(dim, 0, length);
  out.copy_(out.is_complex() ? part : at::real(part));
}
// n-point transforms along `dim`; lengths with other prime factors use Bluestein's algorithm
void fft_pass(const Tensor& in, const Tensor& out, int64_t dim, int64_t n, bool inv, const Pass& x = {}) {
  dim = maybe_wrap_dim(dim, in.dim());
  if (!canUse32BitIndexMath(in) || !canUse32BitIndexMath(out)) {
    // Kernels index with 32 bits, so split the batch; one transform that large is not supported
    std::optional<int64_t> widest;
    for (const auto d : c10::irange(in.dim())) {
      if (d != dim && in.size(d) > 1 && (!widest || in.size(d) > in.size(*widest))) {
        widest = d;
      }
    }
    TORCH_CHECK_NOT_IMPLEMENTED(widest.has_value(), kIndexLimitMessage);
    const auto half_rows = in.size(*widest) / 2, other_rows = in.size(*widest) - half_rows;
    fft_pass(in.narrow(*widest, 0, half_rows), out.narrow(*widest, 0, half_rows), dim, n, inv, x);
    fft_pass(in.narrow(*widest, half_rows, other_rows), out.narrow(*widest, half_rows, other_rows), dim, n, inv, x);
    return;
  }
  if (is_smooth(n) && fft_batch(in, out, collapse(in, out, dim), n, inv, x)) {
    return;
  }
  const auto complex_options = in.options().dtype(kComplexFloat);
  DimVector sizes(in.sizes());
  sizes[dim] = n;
  const auto n_in = x.n_in ? x.n_in : n, n_out = x.n_out ? x.n_out : n;
  if (c10::toRealValueType(in.scalar_type()) == kHalf) {
    // Half-precision temps lose accuracy over several passes and can overflow, so these run in fp32
    const auto out_fp32 = at::empty(out.sizes(), out.options().dtype(out.is_complex() ? kComplexFloat : kFloat));
    fft_pass(in.to(in.is_complex() ? kComplexFloat : kFloat), out_fp32, dim, n, inv, x);
    out.copy_(out_fp32);
    return;
  }
  if (is_smooth(n)) {
    // Only complex outputs recurse beyond two passes, so go through a full complex transform
    const auto full = at::empty(sizes, complex_options);
    fft_pass(in, full, dim, n, inv, {.n_in = n_in, .scale = x.scale});
    store_result(full, out, dim, n_out);
    return;
  }
  // Bluestein: X = c * IFFT_M(FFT_M(x * c) * B)[:n], M a power of two >= 2n - 1, chirps folded into the kernels
  const auto [chirp, chirp_fft] = chirp_tables(n, inv);
  const auto M = chirp_fft.numel();
  sizes[dim] = M;
  const auto tmp = at::empty(sizes, complex_options);
  if (M <= kMaxTwoPassPoints) {
    fft_pass(in,
             tmp,
             dim,
             M,
             false,
             {.n_sig = n, .n_in = n_in, .scale = x.scale / M, .load_chirp = chirp, .store_chirp = chirp_fft});
    fft_pass(tmp, out, dim, M, true, {.n_out = n_out, .store_chirp = chirp});
    return;
  }
  DimVector axis_shape(in.dim(), 1);
  axis_shape[dim] = M;
  fft_pass(in, tmp, dim, M, false, {.n_sig = n, .n_in = n_in, .scale = x.scale / M, .load_chirp = chirp});
  tmp.mul_(chirp_fft.view(axis_shape));
  const auto full = at::empty(sizes, complex_options);
  fft_pass(tmp, full, dim, M, true);
  axis_shape[dim] = n_out;
  full.narrow(dim, 0, n_out).mul_(chirp.narrow(0, 0, n_out).view(axis_shape));
  store_result(full, out, dim, n_out);
}
// Normalize each axis before storing half-precision intermediates to avoid underflow and overflow.
double norm_scale(int64_t normalization, int64_t n) {
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
  const auto last_dim = maybe_wrap_dim(dim.back(), self.dim());
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }
  if (self.numel() == 0) { // An empty axis has one all-zero onesided output
    return out.zero_();
  }

  const Pass x{.n_out = out.size(last_dim), .scale = norm_scale(normalization, self.size(last_dim))};
  fft_pass(self, out, last_dim, self.size(last_dim), false, x);
  for (const auto i : c10::irange(dim.size() - 1)) {
    const auto n = self.size(dim[i]);
    fft_pass(out, out, dim[i], n, false, {.scale = norm_scale(normalization, n)});
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
  out_sizes[maybe_wrap_dim(dim.back(), self.dim())] = last_dim_size;
  at::native::resize_output_symint(out, out_sizes);
  if (out.numel() == 0) {
    return out;
  }

  const auto tmp = dim.size() > 1 ? at::empty(self.sizes(), self.options()) : self;
  for (const auto i : c10::irange(dim.size() - 1)) {
    const auto n = self.size(dim[i]);
    fft_pass(i ? tmp : self, tmp, dim[i], n, true, {.scale = norm_scale(normalization, n)});
  }
  const Pass x{
      .n_in = self.size(dim.back()), .n_out = last_dim_size, .scale = norm_scale(normalization, last_dim_size)};
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
  for (const auto i : c10::irange(dim.size())) {
    const auto n = self.size(dim[i]);
    fft_pass(i ? out : self, out, dim[i], n, !forward, {.scale = norm_scale(normalization, n)});
  }
  return out;
}

} // namespace at::native
