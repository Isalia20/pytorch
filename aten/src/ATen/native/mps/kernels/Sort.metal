// Merge sort for MPS tensors.
// Based on NVIDIA CUB's GPU merge sort, adapted from Apple MLX.
// Uses uint32 indices internally to minimize memory bandwidth;
// converts to int64 only on final output.

#include <metal_stdlib>
using namespace metal;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// ====================== Comparison ======================

template <typename T>
inline bool sort_lt(T a, T b) {
  return a < b;
}
template <>
inline bool sort_lt<float>(float a, float b) {
  if (a != a) return false;
  if (b != b) return true;
  return a < b;
}
template <>
inline bool sort_lt<half>(half a, half b) {
  if (a != a) return false;
  if (b != b) return true;
  return a < b;
}
template <>
inline bool sort_lt<bfloat>(bfloat a, bfloat b) {
  if (a != a) return false;
  if (b != b) return true;
  return a < b;
}

template <typename T>
inline bool sort_compare(T a, T b, bool desc) {
  return desc ? sort_lt(b, a) : sort_lt(a, b);
}

template <typename T>
inline T sort_init(bool desc);
template <> inline float sort_init<float>(bool d) { return d ? -__builtin_inff() : NAN; }
template <> inline half sort_init<half>(bool d) { return d ? half(-__builtin_inff()) : half(NAN); }
template <> inline bfloat sort_init<bfloat>(bool d) { return d ? bfloat(-__builtin_inff()) : bfloat(NAN); }
template <> inline int sort_init<int>(bool d) { return d ? int(0x80000000u) : 0x7FFFFFFF; }
template <> inline long sort_init<long>(bool d) { return d ? long(0x8000000000000000u) : 0x7FFFFFFFFFFFFFFF; }
template <> inline short sort_init<short>(bool d) { return d ? short(0x8000u) : short(0x7FFF); }
template <> inline char sort_init<char>(bool d) { return d ? char(-128) : char(127); }
template <> inline uchar sort_init<uchar>(bool d) { return d ? uchar(0) : uchar(255); }
template <> inline bool sort_init<bool>(bool d) { return !d; }

// ====================== Thread-level sort ======================

template <typename T, typename IdxT, short N>
inline void thread_sort(thread T (&v)[N], thread IdxT (&idx)[N], bool desc) {
  _Pragma("clang loop unroll(full)") for (short i = 0; i < N; ++i) {
    _Pragma("clang loop unroll(full)") for (short j = i & 1; j < N - 1; j += 2) {
      if (sort_compare(v[j + 1], v[j], desc)) {
        T tv = v[j]; v[j] = v[j + 1]; v[j + 1] = tv;
        IdxT ti = idx[j]; idx[j] = idx[j + 1]; idx[j + 1] = ti;
      }
    }
  }
}

// ====================== Merge utilities ======================

template <typename T>
inline int merge_partition(
    const threadgroup T* A, const threadgroup T* B,
    int a_sz, int b_sz, int diag, bool desc) {
  int lo = max(0, diag - b_sz), hi = min(diag, a_sz);
  while (lo < hi) {
    int m = lo + (hi - lo) / 2;
    if (sort_compare(B[diag - 1 - m], A[m], desc)) hi = m;
    else lo = m + 1;
  }
  return hi;
}

template <typename T, typename IdxT, short N>
inline void merge_step(
    const threadgroup T* A, const threadgroup T* B,
    const threadgroup IdxT* Ai, const threadgroup IdxT* Bi,
    int a_sz, int b_sz,
    thread T (&v)[N], thread IdxT (&idx)[N], bool desc) {
  T init = sort_init<T>(desc);
  int a = 0, b = 0;
  for (int i = 0; i < N; ++i) {
    T va = (a < a_sz) ? A[a] : init;
    T vb = (b < b_sz) ? B[b] : init;
    bool tb = (b < b_sz) && (a >= a_sz || sort_compare(vb, va, desc));
    v[i] = tb ? vb : va;
    idx[i] = tb ? Bi[b] : ((a < a_sz) ? Ai[a] : IdxT(0));
    b += int(tb); a += int(!tb);
  }
}

template <typename T>
inline int merge_partition_global(
    const device T* A, const device T* B,
    int a_sz, int b_sz, int diag, bool desc) {
  int lo = max(0, diag - b_sz), hi = min(diag, a_sz);
  while (lo < hi) {
    int m = lo + (hi - lo) / 2;
    if (sort_compare(B[diag - 1 - m], A[m], desc)) hi = m;
    else lo = m + 1;
  }
  return hi;
}

// ====================== SIMD bitonic sort ======================
// Sort 32*TN items across a 32-lane SIMD using in-register shuffles,
// no threadgroup memory. Each sub-phase is a constexpr offset so the
// compiler specializes the within-thread (offset < TN) vs cross-lane
// (offset ≥ TN) branch at compile time.

template <typename T>
inline T sort_shuffle_xor(T v, ushort delta) {
  if constexpr (is_same_v<T, bool>) {
    return bool(simd_shuffle_xor(uint(v), delta));
  } else if constexpr (sizeof(T) == 1) {
    uchar u = as_type<uchar>(v);
    return as_type<T>(uchar(simd_shuffle_xor(uint(u), delta)));
  } else if constexpr (sizeof(T) == 2) {
    ushort u = as_type<ushort>(v);
    return as_type<T>(ushort(simd_shuffle_xor(uint(u), delta)));
  } else if constexpr (sizeof(T) == 8) {
    ulong u = as_type<ulong>(v);
    uint lo = simd_shuffle_xor(uint(u), delta);
    uint hi = simd_shuffle_xor(uint(u >> 32), delta);
    return as_type<T>(ulong(lo) | (ulong(hi) << 32));
  } else {
    return simd_shuffle_xor(v, delta);
  }
}

// A single bitonic compare-swap substage at a known compile-time OFFSET
// within the global index (lane*TN + i). K is 1 << phase.
template <typename T, typename IdxT, short TN, int K, int OFFSET>
inline void bitonic_substage(
    thread T (&v)[TN], thread IdxT (&idx)[TN], uint lane, bool desc) {
  if constexpr (OFFSET < TN) {
    // Within-thread swap; partner index in same thread's registers.
    _Pragma("clang loop unroll(full)") for (short i = 0; i < TN; ++i) {
      short pi = i ^ OFFSET;
      if (pi > i) {
        int global_p = int(lane) * TN + i;
        bool ascending = (global_p & K) == 0;
        T vi = v[i], vp = v[pi];
        IdxT ii = idx[i], ip = idx[pi];
        bool vi_first = sort_compare(vi, vp, desc);
        bool do_swap = ascending ? !vi_first : vi_first;
        v[i] = do_swap ? vp : vi;
        v[pi] = do_swap ? vi : vp;
        idx[i] = do_swap ? ip : ii;
        idx[pi] = do_swap ? ii : ip;
      }
    }
  } else {
    // Cross-lane swap; partner in lane XOR (OFFSET/TN), same position i.
    constexpr ushort LANE_OFFSET = OFFSET / TN;
    bool i_am_low = (lane & uint(LANE_OFFSET)) == 0;
    _Pragma("clang loop unroll(full)") for (short i = 0; i < TN; ++i) {
      T vi = v[i];
      IdxT ii = idx[i];
      T vp = sort_shuffle_xor(vi, LANE_OFFSET);
      IdxT ip = sort_shuffle_xor(ii, LANE_OFFSET);
      int global_p = int(lane) * TN + i;
      bool ascending = (global_p & K) == 0;
      bool vi_first = sort_compare(vi, vp, desc);
      bool should_take = vi_first != (ascending == i_am_low);
      v[i] = should_take ? vp : vi;
      idx[i] = should_take ? ip : ii;
    }
  }
}

// Hand-unrolled SIMD bitonic sort for TN=4 (32*TN = 128 items per SIMD).
// Phases 1..7, each with phase substages. All offsets constexpr so the
// compiler emits straight-line code.
template <typename T, typename IdxT>
inline void simd_bitonic_sort4(
    thread T (&v)[4], thread IdxT (&idx)[4], uint lane, bool desc) {
  // Phase 1 (K=2): step 0
  bitonic_substage<T, IdxT, 4, 2, 1>(v, idx, lane, desc);
  // Phase 2 (K=4): steps 1, 0
  bitonic_substage<T, IdxT, 4, 4, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 4, 1>(v, idx, lane, desc);
  // Phase 3 (K=8): steps 2, 1, 0
  bitonic_substage<T, IdxT, 4, 8, 4>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 8, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 8, 1>(v, idx, lane, desc);
  // Phase 4 (K=16): steps 3, 2, 1, 0
  bitonic_substage<T, IdxT, 4, 16, 8>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 4>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 1>(v, idx, lane, desc);
  // Phase 5 (K=32): 4, 3, 2, 1, 0
  bitonic_substage<T, IdxT, 4, 32, 16>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 8>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 4>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 1>(v, idx, lane, desc);
  // Phase 6 (K=64): 5..0
  bitonic_substage<T, IdxT, 4, 64, 32>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 16>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 8>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 4>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 1>(v, idx, lane, desc);
  // Phase 7 (K=128): 6..0
  bitonic_substage<T, IdxT, 4, 128, 64>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 32>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 16>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 8>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 4>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 2>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 1>(v, idx, lane, desc);
}

// ====================== Block merge sort ======================
// SIMD bitonic handles mt = 2..32 (5 rounds equivalent) with zero
// threadgroup traffic. Cross-SIMD rounds (mt = 64..BN) keep the classic
// threadgroup-mem partition+merge pattern.

template <typename T, typename IdxT, short BN, short TN>
inline void block_merge_sort(
    threadgroup T* tv, threadgroup IdxT* ti,
    int size, uint lid, bool desc) {
  int base = lid * TN;
  thread T lv[TN]; thread IdxT li[TN];
  for (int i = 0; i < TN; ++i) { lv[i] = tv[base+i]; li[i] = ti[base+i]; }

  if constexpr (TN == 4 && BN >= 32) {
    // Hand-unrolled SIMD bitonic sort: sorts 128 items within each SIMD.
    simd_bitonic_sort4<T, IdxT>(lv, li, lid & 31u, desc);
  } else {
    // Fallback: classic thread_sort + threadgroup rounds up to mt=32.
    if (base < size) thread_sort<T, IdxT, TN>(lv, li, desc);
    for (int mt = 2; mt <= 32 && mt <= BN; mt *= 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < TN; ++i) { tv[base+i] = lv[i]; ti[base+i] = li[i]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int grp = lid / mt, lane = lid % mt;
      int sz = TN * mt, st = sz * grp;
      int hsz = sz / 2;
      int diag = TN * lane;
      int p = merge_partition(tv + st, tv + st + hsz, hsz, hsz, diag, desc);
      merge_step<T, IdxT, TN>(
          tv + st + p, tv + st + hsz + diag - p,
          ti + st + p, ti + st + hsz + diag - p,
          hsz - p, hsz - diag + p, lv, li, desc);
    }
  }

  // First cross-SIMD round (mt=64) skips the leading barrier: SIMD bitonic
  // (or thread_sort + mt<=32 rounds) leaves sorted items in registers only,
  // and each thread's tv[base..base+TN) slots were last read by itself at
  // line 235 — no other thread's write or read touches those slots.
  if (BN >= 64) {
    constexpr int mt_first = 64;
    for (int i = 0; i < TN; ++i) { tv[base+i] = lv[i]; ti[base+i] = li[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int grp = lid / mt_first, lane = lid % mt_first;
    int sz = TN * mt_first, st = sz * grp;
    int hsz = sz / 2;
    int diag = TN * lane;
    int p = merge_partition(tv + st, tv + st + hsz, hsz, hsz, diag, desc);
    merge_step<T, IdxT, TN>(
        tv + st + p, tv + st + hsz + diag - p,
        ti + st + p, ti + st + hsz + diag - p,
        hsz - p, hsz - diag + p, lv, li, desc);
  }
  for (int mt = 128; mt <= BN; mt *= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < TN; ++i) { tv[base+i] = lv[i]; ti[base+i] = li[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int grp = lid / mt, lane = lid % mt;
    int sz = TN * mt, st = sz * grp;
    int hsz = sz / 2;
    int diag = TN * lane;
    int p = merge_partition(tv + st, tv + st + hsz, hsz, hsz, diag, desc);
    merge_step<T, IdxT, TN>(
        tv + st + p, tv + st + hsz + diag - p,
        ti + st + p, ti + st + hsz + diag - p,
        hsz - p, hsz - diag + p, lv, li, desc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < TN; ++i) { tv[base+i] = lv[i]; ti[base+i] = li[i]; }
}

// ====================== Single-block sort ======================
// Outputs int64 indices directly since there's no intermediate buffer.
// Reads input with a (stride_sort, stride_seg) layout (so we can skip the
// caller-side values.copy_(self) for contiguous or last-dim-strided inputs)
// and writes output contiguously to out_vals/out_idx.

template <typename T, short BN, short TN>
kernel void sort_block(
    const device T* inp [[buffer(0)]],
    device T* out_vals [[buffer(1)]],
    device long* out_idx [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant long& stride_sort [[buffer(4)]],
    constant long& stride_seg [[buffer(5)]],
    constant bool& desc [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr int NPB = BN * TN;
  threadgroup T tgv[NPB];
  threadgroup uint tgi[NPB];

  T init = sort_init<T>(desc);
  long base_in = tid.y * stride_seg;
  long base_out = long(tid.y) * long(size);
  for (int i = lid.x; i < NPB; i += BN) {
    tgv[i] = i < size ? inp[base_in + i * stride_sort] : init;
    tgi[i] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort<T, uint, BN, TN>(tgv, tgi, size, lid.x, desc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = lid.x; i < size; i += BN) {
    out_vals[base_out + i] = tgv[i];
    out_idx[base_out + i] = long(tgi[i]);
  }
}

// ====================== Multi-block kernels ======================
// Intermediate buffers use either uint32 or uint16 indices depending on
// sort_size: sort_size ≤ 65536 fits in ushort, halving the per-row index
// bandwidth (~25% total BW reduction for merge path) and cutting per-TG
// tgmem from 16KB → 12KB (f32) which fits more TGs per core on some GPUs.

template <typename T, typename IdxT, short BN, short TN>
kernel void mb_sort_block(
    const device T* inp [[buffer(0)]],
    device T* dv [[buffer(1)]],
    device IdxT* di [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant long& stride_sort [[buffer(4)]],
    constant long& stride_seg [[buffer(5)]],
    constant bool& desc [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr int NPB = BN * TN;
  T init = sort_init<T>(desc);
  long seg = tid.y * stride_seg;
  int blk = tid.x * NPB;
  threadgroup T tgv[NPB]; threadgroup IdxT tgi[NPB];
  for (int i = lid.x; i < NPB; i += BN) {
    int g = blk + i;
    tgv[i] = g < size ? inp[seg + g * stride_sort] : init;
    tgi[i] = IdxT(g);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort<T, IdxT, BN, TN>(tgv, tgi, size, lid.x, desc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int row = tid.y * size;
  for (int i = lid.x; i < NPB; i += BN) {
    int g = blk + i;
    if (g < size) { dv[row+g] = tgv[i]; di[row+g] = tgi[i]; }
  }
}

// Merge kernel with inline per-block partition.  Each block used to read
// precomputed Ast/Aed from a partitions buffer written by a separate
// `mb_partition` dispatch; now we just redo the two binary searches locally
// (O(log(NPB*merge_tiles)) global reads, well inside L1). Saves one
// dispatch per merge round.
// Templated on InIdxT (ushort for small sort_size, uint otherwise) and
// OutIdxT (InIdxT for intermediate rounds, long for the final round writing
// directly into the caller's int64 output tensor).
template <typename T, typename InIdxT, typename OutIdxT, short BN, short TN>
kernel void mb_merge(
    const device T* vi [[buffer(0)]],
    const device InIdxT* ii [[buffer(1)]],
    device T* vo [[buffer(2)]],
    device OutIdxT* io [[buffer(3)]],
    constant int& size [[buffer(4)]],
    constant int& merge_tiles [[buffer(5)]],
    constant int& n_blocks [[buffer(6)]],
    constant bool& desc [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr int NPB = BN * TN;
  T init = sort_init<T>(desc);
  vi += tid.y * size; ii += tid.y * size;
  vo += tid.y * size; io += tid.y * size;

  int bi = tid.x, mg = bi / merge_tiles;
  int sst = NPB * merge_tiles * mg, ssz = NPB * merge_tiles;
  int smd = NPB * bi - sst;

  // Inline partition: at smaller BN we have threadgroup-memory headroom,
  // so thread 0 does the two binary searches and broadcasts. At BN=1024
  // we're at the 32KB threadgroup limit from tgv/tgi (NPB=4096 × 8 bytes
  // for 4-byte T), so we fall back to having every thread redo the
  // binary search — same global addresses across lanes means L1 absorbs
  // the reads and the SIMD-lockstep cost matches one thread's work.
  constexpr int kMaxTG = 32768;
  constexpr int kTgvTgiBytes = NPB * (int)sizeof(T) + NPB * (int)sizeof(uint);
  constexpr bool kUseBroadcast = kTgvTgiBytes + 2 * (int)sizeof(int) <= kMaxTG;
  int A0 = min(size, sst);
  int A1 = min(size, sst + ssz/2);
  int B0 = A1;
  int B1 = min(size, A1 + ssz/2);
  int ml = bi % merge_tiles;
  int asz0 = A1 - A0, bsz0 = B1 - B0;
  int diag_lo = min(asz0 + bsz0, NPB * ml);
  int diag_hi = min(asz0 + bsz0, NPB * (ml + 1));
  int Ast, Aed;
  if (kUseBroadcast) {
    threadgroup int bcast[2];
    if (lid.x == 0) {
      bcast[0] = A0 + merge_partition_global(vi + A0, vi + B0, asz0, bsz0, diag_lo, desc);
      bcast[1] = A0 + merge_partition_global(vi + A0, vi + B0, asz0, bsz0, diag_hi, desc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ast = bcast[0];
    Aed = bcast[1];
  } else {
    Ast = A0 + merge_partition_global(vi + A0, vi + B0, asz0, bsz0, diag_lo, desc);
    Aed = A0 + merge_partition_global(vi + A0, vi + B0, asz0, bsz0, diag_hi, desc);
  }

  int Bst = min(size, 2*sst + ssz/2 + smd - Ast);
  int Bed = min(size, 2*sst + ssz/2 + smd + NPB - Aed);
  int Asz = Aed - Ast, Bsz = Bed - Bst;

  // Direct global → tgv coalesced load. Skips the register-intermediate
  // load/write loop and its paired barrier (tgv has not been touched before
  // this point; bcast[] lives in a separately-allocated tgmem region).
  threadgroup T tgv[NPB]; threadgroup InIdxT tgi[NPB];
  for (int i = 0; i < TN; ++i) {
    int x = BN * i + lid.x;
    if (x < Asz + Bsz) {
      tgv[x] = (x < Asz) ? vi[Ast+x] : vi[Bst+x-Asz];
      tgi[x] = (x < Asz) ? ii[Ast+x] : ii[Bst+x-Asz];
    } else { tgv[x] = init; tgi[x] = InIdxT(0); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int md = min(Asz+Bsz, TN*int(lid.x));
  int ap = merge_partition(tgv, tgv+Asz, Asz, Bsz, md, desc);
  thread T lv[TN]; thread InIdxT li[TN];
  merge_step<T, InIdxT, TN>(tgv+ap, tgv+Asz+md-ap, tgi+ap, tgi+Asz+md-ap,
                             Asz-ap, Bsz-md+ap, lv, li, desc);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < TN; ++i) { tgv[lid.x*TN+i] = lv[i]; tgi[lid.x*TN+i] = li[i]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int base = tid.x * NPB;
  for (int i = lid.x; i < NPB; i += BN) {
    int g = base + i;
    if (g < size) { vo[g] = tgv[i]; io[g] = OutIdxT(tgi[i]); }
  }
}

#pragma clang diagnostic pop

// ====================== Radix Sort ======================
// 4-bit radix sort for large arrays. Fewer passes than merge sort
// at large sizes: 4 passes for 16-bit types, 8 for 32-bit.
// Uses SIMD-based stable ranking within each block.

template <typename T>
inline uint to_radix_key(T val, bool desc);

template <>
inline uint to_radix_key<float>(float val, bool desc) {
  uint bits = as_type<uint>(val);
  if (val != val) return desc ? 0u : 0xFFFFFFFFu;
  uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  uint result = bits ^ mask;
  return desc ? ~result : result;
}
template <>
inline uint to_radix_key<half>(half val, bool desc) {
  ushort bits = as_type<ushort>(val);
  if (val != val) return desc ? 0u : uint(0xFFFF);
  ushort mask = (bits & 0x8000u) ? ushort(0xFFFF) : ushort(0x8000);
  ushort result = bits ^ mask;
  return desc ? uint(ushort(~result)) : uint(result);
}
template <>
inline uint to_radix_key<bfloat>(bfloat val, bool desc) {
  ushort bits = as_type<ushort>(val);
  if (val != val) return desc ? 0u : uint(0xFFFF);
  ushort mask = (bits & 0x8000u) ? ushort(0xFFFF) : ushort(0x8000);
  ushort result = bits ^ mask;
  return desc ? uint(ushort(~result)) : uint(result);
}
template <>
inline uint to_radix_key<int>(int val, bool desc) {
  uint result = as_type<uint>(val) ^ 0x80000000u;
  return desc ? ~result : result;
}
template <>
inline uint to_radix_key<long>(long val, bool desc) {
  ulong r = as_type<ulong>(val) ^ 0x8000000000000000UL;
  // Truncate to uint for the current nibble (shift handles which bits matter)
  return desc ? uint(~r) : uint(r);
}
template <>
inline uint to_radix_key<short>(short val, bool desc) {
  ushort result = as_type<ushort>(val) ^ ushort(0x8000);
  return desc ? uint(ushort(~result)) : uint(result);
}
template <>
inline uint to_radix_key<char>(char val, bool desc) {
  uchar result = as_type<uchar>(val) ^ uchar(0x80);
  return desc ? uint(uchar(~result)) : uint(result);
}
template <>
inline uint to_radix_key<uchar>(uchar val, bool desc) {
  return desc ? uint(uchar(~val)) : uint(val);
}
template <>
inline uint to_radix_key<bool>(bool val, bool desc) {
  return desc ? uint(!val) : uint(val);
}

// Radix is parameterized by RBITS ∈ {4, 8}. 4-bit uses per-thread register
// histograms (16 counters); 8-bit (256 counters) would overflow register budget
// so it goes through threadgroup-atomic counting per item.
template <typename T, short RBN, short EPT, short RBITS>
kernel void radix_count(
    const device T* input [[buffer(0)]],
    device uint* histograms [[buffer(1)]],
    constant int& sort_size [[buffer(2)]],
    constant int& n_blocks [[buffer(3)]],
    constant int& shift [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  constexpr int RSIZE = 1 << RBITS;
  constexpr uint RMASK = RSIZE - 1;
  constexpr int NPB = RBN * EPT;
  constexpr int SIMD_W = 32;
  uint lid = lid3.x;
  uint lane = lid & (SIMD_W - 1);
  int row = tid.y;
  int block_idx = tid.x;
  int block_start = block_idx * NPB;
  int items_this_block = min(NPB, sort_size - block_start);
  const device T* rk = input + row * sort_size + block_start;

  threadgroup atomic_uint local_hist[RSIZE];
  for (uint d = lid; d < uint(RSIZE); d += RBN)
    atomic_store_explicit(&local_hist[d], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (RBITS <= 4) {
    // Per-thread register histogram + simd_sum to cut atomic contention.
    uint my_counts[RSIZE];
    _Pragma("clang loop unroll(full)") for (uint d = 0; d < uint(RSIZE); ++d) my_counts[d] = 0;
    _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
      int pos = i * RBN + int(lid);
      if (pos < items_this_block) {
        uint key = to_radix_key(rk[pos], desc);
        my_counts[(key >> shift) & RMASK]++;
      }
    }
    _Pragma("clang loop unroll(full)") for (uint d = 0; d < uint(RSIZE); ++d) {
      uint s = simd_sum(my_counts[d]);
      if (lane == 0 && s > 0)
        atomic_fetch_add_explicit(&local_hist[d], s, memory_order_relaxed);
    }
  } else {
    // 8-bit: direct per-item atomic add. With 256 bins contention is low
    // (avg 16 items/bin for NPB=4096) and per-thread register pressure drops.
    _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
      int pos = i * RBN + int(lid);
      if (pos < items_this_block) {
        uint key = to_radix_key(rk[pos], desc);
        uint d = (key >> shift) & RMASK;
        atomic_fetch_add_explicit(&local_hist[d], 1u, memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = lid; d < uint(RSIZE); d += RBN) {
    uint count = atomic_load_explicit(&local_hist[d], memory_order_relaxed);
    histograms[row * RSIZE * n_blocks + d * n_blocks + block_idx] = count;
  }
}

// Parallel exclusive scan over [n_entries] per row, using SCAN_BN threads.
// SCAN_BN=1024 = hardware max per TG: keeps each row's scan as a single TG
// (one scan dispatch launches 1 TG per row) while maxing parallel work per
// thread. For single-row configs with large n_entries (e.g. f32 [1,524288]
// has n_entries=65536), this cuts the serial per-thread chunk 4× vs SCAN_BN=256.
// MAX_SIMD_GROUPS = 32 exactly, matching SIMD_W so the cross-SIMD prefix can
// be done with a single simd_prefix_exclusive_sum on simd 0.
constant constexpr int SCAN_BN = 1024;

[[host_name("radix_scan")]]
kernel void radix_scan(
    device uint* histograms [[buffer(0)]],
    constant int& n_entries [[buffer(1)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  constexpr int SIMD_W = 32;
  constexpr int MAX_SIMD_GROUPS = SCAN_BN / SIMD_W;
  int row = tid.y;
  device uint* rh = histograms + row * n_entries;
  uint lid = lid3.x;
  uint simd_id = lid / SIMD_W;
  uint lane = lid % SIMD_W;

  // Each thread owns a contiguous chunk of entries.
  uint chunk = (uint(n_entries) + SCAN_BN - 1) / SCAN_BN;
  uint my_start = min(lid * chunk, uint(n_entries));
  uint my_end = min(my_start + chunk, uint(n_entries));

  // Phase 1: reduce per chunk.
  uint local_sum = 0;
  for (uint i = my_start; i < my_end; ++i)
    local_sum += rh[i];

  // Phase 2: threadgroup exclusive scan over local_sum[].
  threadgroup uint simd_totals[MAX_SIMD_GROUPS];
  uint my_prefix = simd_prefix_exclusive_sum(local_sum);
  uint simd_total = simd_sum(local_sum);
  if (lane == 0) simd_totals[simd_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0 && lane < uint(MAX_SIMD_GROUPS)) {
    uint t = simd_totals[lane];
    simd_totals[lane] = simd_prefix_exclusive_sum(t);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint my_base = simd_totals[simd_id] + my_prefix;

  // Phase 3: write exclusive prefix sums back over my chunk.
  uint running = my_base;
  for (uint i = my_start; i < my_end; ++i) {
    uint v = rh[i];
    rh[i] = running;
    running += v;
  }
}

// Block-local radix sort: load all NPB items, fully sort them by the current
// RBITS-bit digit via RBITS stable binary splits in threadgroup memory, then
// write coalesced to global. Same-digit items end up in one contiguous run per
// block, so consecutive threads writing consecutive stage positions produce
// consecutive global writes.
// Templated on OutIdxT so the last pass can write int64 indices directly into
// the public output tensor, skipping a uint32→int64 copy after the sort.
// FUSED_SCAN=true reads raw per-block counts and computes the scan in
// threadgroup memory (saving one dispatch per pass). Only usable when
// n_blocks ≤ kMaxFusedBlocks since we load the full n_entries into tgmem.
constexpr constant int kMaxFusedBlocks = 4;
template <typename T, typename InIdxT, typename OutIdxT, short RBN, short EPT,
          short RBITS, bool FUSED_SCAN = false>
kernel void radix_scatter(
    const device T* keys_in [[buffer(0)]],
    const device InIdxT* vals_in [[buffer(1)]],
    device T* keys_out [[buffer(2)]],
    device OutIdxT* vals_out [[buffer(3)]],
    const device uint* offsets [[buffer(4)]],  // raw counts if FUSED_SCAN, else prefix-summed
    constant int& sort_size [[buffer(5)]],
    constant int& n_blocks [[buffer(6)]],
    constant int& shift [[buffer(7)]],
    constant bool& desc [[buffer(8)]],
    constant bool& first_pass [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  constexpr int RSIZE = 1 << RBITS;
  constexpr uint RMASK = RSIZE - 1;
  constexpr int NPB = RBN * EPT;
  constexpr int SIMD_W = 32;
  constexpr int MAX_SIMD_GROUPS = RBN / SIMD_W;
  constexpr int FUSED_BUF_SIZE = FUSED_SCAN ? RSIZE * kMaxFusedBlocks : 1;
  uint lid = lid3.x;
  uint simd_id = lid / SIMD_W;
  uint lane = lid & (SIMD_W - 1);
  int row = tid.y;
  int block_idx = tid.x;
  int block_start = block_idx * NPB;
  int items_this_block = min(NPB, sort_size - block_start);

  threadgroup T stage_keys[NPB];
  threadgroup InIdxT stage_idxs[NPB];
  threadgroup uint simd_sum_buf[MAX_SIMD_GROUPS];
  threadgroup uint tg_total_zeros;
  threadgroup uint block_offsets[RSIZE];
  threadgroup uint digit_start[RSIZE];
  // (local_hist removed: boundary detection replaces atomic rebuild)
  threadgroup uint fused_buf[FUSED_BUF_SIZE];

  // 1) Strided (coalesced) load into stage.
  _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
    int pos = i * RBN + int(lid);
    if (pos < items_this_block) {
      stage_keys[pos] = keys_in[row * sort_size + block_start + pos];
      stage_idxs[pos] = first_pass ? InIdxT(block_start + pos)
                                   : vals_in[row * sort_size + block_start + pos];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) Read into registers in contiguous-per-thread layout for stable local sort.
  // No trailing barrier: the first threadgroup write in the binary split loop
  // is to simd_sum_buf (line ~740), not stage. Its paired barrier (line ~744)
  // also flushes these stage reads before any thread scatters to stage later.
  T local_keys[EPT];
  InIdxT local_idxs[EPT];
  uint my_active_start = min(uint(lid) * uint(EPT), uint(items_this_block));
  uint my_active_end = min((uint(lid) + 1) * uint(EPT), uint(items_this_block));
  uint my_active_count = my_active_end - my_active_start;
  _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
    if (uint(i) < my_active_count) {
      local_keys[i] = stage_keys[my_active_start + i];
      local_idxs[i] = stage_idxs[my_active_start + i];
    } else {
      local_keys[i] = sort_init<T>(desc);
      local_idxs[i] = InIdxT(0);
    }
  }

  // 3) RBITS binary split passes to fully sort the block by the RBITS-bit digit.
  for (int bit = 0; bit < RBITS; ++bit) {
    bool bits[EPT];
    uint my_zeros = 0;
    _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
      if (uint(i) < my_active_count) {
        uint d = (to_radix_key(local_keys[i], desc) >> shift) & RMASK;
        bits[i] = (d >> bit) & 1;
        if (!bits[i]) ++my_zeros;
      } else {
        bits[i] = false;
      }
    }

    uint simd_prefix_val = simd_prefix_exclusive_sum(my_zeros);
    uint simd_total = simd_sum(my_zeros);
    if (lane == 0) simd_sum_buf[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
      uint t = (lane < uint(MAX_SIMD_GROUPS)) ? simd_sum_buf[lane] : 0u;
      uint p = simd_prefix_exclusive_sum(t);
      if (lane < uint(MAX_SIMD_GROUPS)) simd_sum_buf[lane] = p;
      if (lane == uint(MAX_SIMD_GROUPS - 1)) tg_total_zeros = p + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_zero_prefix = simd_sum_buf[simd_id] + simd_prefix_val;
    uint total_zeros = tg_total_zeros;
    uint my_one_prefix = my_active_start - my_zero_prefix;

    uint next_zero = my_zero_prefix;
    uint next_one = total_zeros + my_one_prefix;

    _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
      if (uint(i) < my_active_count) {
        uint new_pos = bits[i] ? next_one++ : next_zero++;
        stage_keys[new_pos] = local_keys[i];
        stage_idxs[new_pos] = local_idxs[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Skip the reload on the last iteration: the histogram below reads directly
    // from stage_keys, and the final coalesced writeback also uses stage.
    if (bit < RBITS - 1) {
      _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
        if (uint(i) < my_active_count) {
          local_keys[i] = stage_keys[my_active_start + i];
          local_idxs[i] = stage_idxs[my_active_start + i];
        }
      }
    }
    // The scan's first barrier in the next iteration (or the atomic_store
    // barrier below, on the last iteration) already synchronizes these reads
    // before any thread starts writing to stage again — no explicit barrier
    // needed here.
  }

  // 4) Build per-digit start offsets via boundary detection. Stage is sorted
  // by digit after the binary splits; digit_start[d] is the position where
  // digit d first appears. For digits not present, digit_start[d] stays stale
  // (never read). This replaces a ~NPB-atomic histogram build + scan with one
  // tg read + compare per element.
  _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
    uint pos = my_active_start + uint(i);
    if (uint(i) < my_active_count) {
      uint d_here = (to_radix_key(stage_keys[pos], desc) >> shift) & RMASK;
      uint d_prev;
      if (pos == 0u) {
        d_prev = 0xFFFFFFFFu;  // sentinel: no predecessor
      } else {
        d_prev = (to_radix_key(stage_keys[pos - 1u], desc) >> shift) & RMASK;
      }
      if (d_here != d_prev) {
        digit_start[d_here] = pos;
      }
    }
  }
  if (FUSED_SCAN) {
    // Inline scan of raw per-block counts: load RSIZE*n_blocks entries,
    // exclusive prefix sum cooperatively, then index out our block's digit
    // offsets. Only valid when n_blocks ≤ kMaxFusedBlocks.
    int n_entries = RSIZE * n_blocks;
    for (int i = int(lid); i < n_entries; i += RBN) {
      fused_buf[i] = offsets[row * n_entries + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(n_entries) + RBN - 1) / RBN;
    uint my_start = min(lid * chunk, uint(n_entries));
    uint my_end = min(my_start + chunk, uint(n_entries));

    uint local_sum = 0;
    for (uint i = my_start; i < my_end; ++i) local_sum += fused_buf[i];

    uint my_prefix = simd_prefix_exclusive_sum(local_sum);
    uint simd_total = simd_sum(local_sum);
    if (lane == 0) simd_sum_buf[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane < uint(MAX_SIMD_GROUPS)) {
      uint t = simd_sum_buf[lane];
      simd_sum_buf[lane] = simd_prefix_exclusive_sum(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_base = simd_sum_buf[simd_id] + my_prefix;
    uint running = my_base;
    for (uint i = my_start; i < my_end; ++i) {
      uint v = fused_buf[i];
      fused_buf[i] = running;
      running += v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = lid; d < uint(RSIZE); d += RBN)
      block_offsets[d] = fused_buf[d * n_blocks + block_idx];
  } else {
    for (uint d = lid; d < uint(RSIZE); d += RBN)
      block_offsets[d] = offsets[row * RSIZE * n_blocks + d * n_blocks + block_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 5) Coalesced global write: strided read of stage (digit-sorted).
  _Pragma("clang loop unroll(full)") for (int i = 0; i < EPT; ++i) {
    int pos = i * RBN + int(lid);
    if (pos < items_this_block) {
      T k = stage_keys[pos];
      uint idx = stage_idxs[pos];
      uint d = (to_radix_key(k, desc) >> shift) & RMASK;
      uint local_rank = uint(pos) - digit_start[d];
      uint global_pos = block_offsets[d] + local_rank;
      keys_out[row * sort_size + global_pos] = k;
      vals_out[row * sort_size + global_pos] = OutIdxT(idx);
    }
  }
}

// ====================== Instantiation ======================

#define INSTANTIATE_SORT(T, BN, TN)                                           \
  template [[host_name("sort_block_" #T "_bn" #BN)]]                          \
  kernel void sort_block<T, BN, TN>(                                          \
      const device T*, device T*, device long*, constant int&,                \
      constant long&, constant long&, constant bool&, uint3, uint3);          \
  template [[host_name("mb_sort_block_" #T "_bn" #BN)]]                       \
  kernel void mb_sort_block<T, uint, BN, TN>(                                 \
      const device T*, device T*, device uint*, constant int&,                \
      constant long&, constant long&, constant bool&, uint3, uint3);          \
  template [[host_name("mb_merge_" #T "_bn" #BN)]]                            \
  kernel void mb_merge<T, uint, uint, BN, TN>(                                \
      const device T*, const device uint*,                                    \
      device T*, device uint*, constant int&, constant int&,                  \
      constant int&, constant bool&, uint3, uint3);                           \
  template [[host_name("mb_merge_final_" #T "_bn" #BN)]]                      \
  kernel void mb_merge<T, uint, long, BN, TN>(                                \
      const device T*, const device uint*,                                    \
      device T*, device long*, constant int&, constant int&,                  \
      constant int&, constant bool&, uint3, uint3);                           \
  /* uint16 index variants: used when sort_size <= 65536 (intermediate        \
     indices fit in ushort, halving index bandwidth). */                      \
  template [[host_name("mb_sort_block_" #T "_bn" #BN "_u16")]]                \
  kernel void mb_sort_block<T, ushort, BN, TN>(                               \
      const device T*, device T*, device ushort*, constant int&,              \
      constant long&, constant long&, constant bool&, uint3, uint3);          \
  template [[host_name("mb_merge_" #T "_bn" #BN "_u16")]]                     \
  kernel void mb_merge<T, ushort, ushort, BN, TN>(                            \
      const device T*, const device ushort*,                                  \
      device T*, device ushort*, constant int&, constant int&,                \
      constant int&, constant bool&, uint3, uint3);                           \
  template [[host_name("mb_merge_final_" #T "_bn" #BN "_u16")]]               \
  kernel void mb_merge<T, ushort, long, BN, TN>(                              \
      const device T*, const device ushort*,                                  \
      device T*, device long*, constant int&, constant int&,                  \
      constant int&, constant bool&, uint3, uint3);

#define INSTANTIATE_ALL_BN(T) \
  INSTANTIATE_SORT(T, 32, 4)  \
  INSTANTIATE_SORT(T, 64, 4)  \
  INSTANTIATE_SORT(T, 128, 4) \
  INSTANTIATE_SORT(T, 256, 4) \
  INSTANTIATE_SORT(T, 512, 4)

// BN=1024 for types ≤ 4 bytes (NPB=4096, threadgroup mem ≤ 32KB)
#define INSTANTIATE_ALL_BN_1024(T) \
  INSTANTIATE_ALL_BN(T)            \
  INSTANTIATE_SORT(T, 1024, 4)

INSTANTIATE_ALL_BN_1024(float);
INSTANTIATE_ALL_BN_1024(half);
INSTANTIATE_ALL_BN_1024(bfloat);
INSTANTIATE_ALL_BN_1024(int);
INSTANTIATE_ALL_BN(long);
INSTANTIATE_ALL_BN_1024(short);
INSTANTIATE_ALL_BN_1024(char);
INSTANTIATE_ALL_BN_1024(uchar);
INSTANTIATE_ALL_BN_1024(bool);

// Radix sort instantiations.
// EPT is tuned so block-local stage_keys + stage_idxs fit in threadgroup
// memory (≤ 32KB): 4-byte types use EPT=4 (NPB=2048 → 16KB), 1-2 byte types
// use EPT=8 (NPB=4096 → 20-24KB).
// RBITS: 4-bit radix (4 passes for 16-bit, 8 passes for 32-bit), or 8-bit
// radix (2 passes for 16-bit). For 16-bit types we pick 8-bit to halve global
// memory traffic. 32-bit 8-bit radix needs 4 passes and block-local sort via
// 8 binary passes per radix pass — higher per-block compute, so we keep 4-bit.
#define INSTANTIATE_RADIX(T, RBN, EPT, RBITS)                                      \
  template [[host_name("radix_count_" #T "_" #RBITS "bit")]]                       \
  kernel void radix_count<T, RBN, EPT, RBITS>(                                     \
      const device T*, device uint*, constant int&, constant int&,                 \
      constant int&, constant bool&, uint3, uint3);                                \
  template [[host_name("radix_scatter_" #T "_" #RBITS "bit")]]                     \
  kernel void radix_scatter<T, uint, uint, RBN, EPT, RBITS, false>(                \
      const device T*, const device uint*, device T*, device uint*,                \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_final_" #T "_" #RBITS "bit")]]               \
  kernel void radix_scatter<T, uint, long, RBN, EPT, RBITS, false>(                \
      const device T*, const device uint*, device T*, device long*,                \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_fused_" #T "_" #RBITS "bit")]]               \
  kernel void radix_scatter<T, uint, uint, RBN, EPT, RBITS, true>(                 \
      const device T*, const device uint*, device T*, device uint*,                \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_fused_final_" #T "_" #RBITS "bit")]]         \
  kernel void radix_scatter<T, uint, long, RBN, EPT, RBITS, true>(                 \
      const device T*, const device uint*, device T*, device long*,                \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  /* u16 variants: used when sort_size ≤ 65536 — intermediate indices fit in     \
     ushort, halving the index buffer bandwidth. Only scatter needs u16; the     \
     count kernel reads only keys, and the final pass outputs int64. */           \
  template [[host_name("radix_scatter_" #T "_" #RBITS "bit_u16")]]                 \
  kernel void radix_scatter<T, ushort, ushort, RBN, EPT, RBITS, false>(            \
      const device T*, const device ushort*, device T*, device ushort*,            \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_final_" #T "_" #RBITS "bit_u16")]]           \
  kernel void radix_scatter<T, ushort, long, RBN, EPT, RBITS, false>(              \
      const device T*, const device ushort*, device T*, device long*,              \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_fused_" #T "_" #RBITS "bit_u16")]]           \
  kernel void radix_scatter<T, ushort, ushort, RBN, EPT, RBITS, true>(             \
      const device T*, const device ushort*, device T*, device ushort*,            \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_fused_final_" #T "_" #RBITS "bit_u16")]]     \
  kernel void radix_scatter<T, ushort, long, RBN, EPT, RBITS, true>(               \
      const device T*, const device ushort*, device T*, device long*,              \
      const device uint*, constant int&, constant int&, constant int&,             \
      constant bool&, constant bool&, uint3, uint3);

INSTANTIATE_RADIX(char, 512, 8, 4);
INSTANTIATE_RADIX(uchar, 512, 8, 4);
INSTANTIATE_RADIX(bool, 512, 8, 4);
INSTANTIATE_RADIX(half, 1024, 4, 8);
INSTANTIATE_RADIX(bfloat, 1024, 4, 8);
INSTANTIATE_RADIX(short, 1024, 4, 8);
INSTANTIATE_RADIX(float, 512, 4, 8);
INSTANTIATE_RADIX(int, 512, 4, 8);

// Alternative smaller-TG instantiation for 2-byte types with BN-suffixed names.
// NPB=512*4=2048 (half of default) — halves tgmem so more concurrent TGs per
// GPU core. Used when n_rows >= 32 (enough TGs for occupancy gains).
#define INSTANTIATE_RADIX_BN512(T)                                                \
  template [[host_name("radix_count_" #T "_8bit_bn512")]]                         \
  kernel void radix_count<T, 512, 4, 8>(                                          \
      const device T*, device uint*, constant int&, constant int&,                \
      constant int&, constant bool&, uint3, uint3);                               \
  template [[host_name("radix_scatter_" #T "_8bit_bn512")]]                       \
  kernel void radix_scatter<T, uint, uint, 512, 4, 8, false>(                     \
      const device T*, const device uint*, device T*, device uint*,               \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_final_" #T "_8bit_bn512")]]                 \
  kernel void radix_scatter<T, uint, long, 512, 4, 8, false>(                     \
      const device T*, const device uint*, device T*, device long*,               \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_fused_" #T "_8bit_bn512")]]                 \
  kernel void radix_scatter<T, uint, uint, 512, 4, 8, true>(                      \
      const device T*, const device uint*, device T*, device uint*,               \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_fused_final_" #T "_8bit_bn512")]]           \
  kernel void radix_scatter<T, uint, long, 512, 4, 8, true>(                      \
      const device T*, const device uint*, device T*, device long*,               \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  /* u16 variants for bn512. */                                                   \
  template [[host_name("radix_scatter_" #T "_8bit_bn512_u16")]]                   \
  kernel void radix_scatter<T, ushort, ushort, 512, 4, 8, false>(                 \
      const device T*, const device ushort*, device T*, device ushort*,           \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_final_" #T "_8bit_bn512_u16")]]             \
  kernel void radix_scatter<T, ushort, long, 512, 4, 8, false>(                   \
      const device T*, const device ushort*, device T*, device long*,             \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_fused_" #T "_8bit_bn512_u16")]]             \
  kernel void radix_scatter<T, ushort, ushort, 512, 4, 8, true>(                  \
      const device T*, const device ushort*, device T*, device ushort*,           \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);                              \
  template [[host_name("radix_scatter_fused_final_" #T "_8bit_bn512_u16")]]       \
  kernel void radix_scatter<T, ushort, long, 512, 4, 8, true>(                    \
      const device T*, const device ushort*, device T*, device long*,             \
      const device uint*, constant int&, constant int&, constant int&,            \
      constant bool&, constant bool&, uint3, uint3);

INSTANTIATE_RADIX_BN512(half);
INSTANTIATE_RADIX_BN512(bfloat);
INSTANTIATE_RADIX_BN512(short);

