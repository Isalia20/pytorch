// CooperativeTensors.h - NAX cooperative tensor primitives for PyTorch MPS
//
// Provides fragment and tile abstractions over Apple's MetalPerformancePrimitives
// cooperative tensor API (Metal 4.0+, macOS 26.2+, Apple Silicon gen 17+).
//
// A cooperative tensor fragment represents a 16x16 matrix distributed across
// a 32-thread SIMD group. Each thread holds 8 elements (2 rows x 4 cols).
// The MMA (matrix multiply-accumulate) operation uses mpp::tensor_ops::matmul2d
// to perform hardware-accelerated 16x32x16 tile multiplications.
//
// Adapted from MLX's steel/gemm/nax.h (Copyright 2025 Apple Inc.)

#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace pytorch_nax {

namespace mpp = MetalPerformancePrimitives;

// Reduction / binary operation functors for softmax
struct MaxOp {
  template <typename T>
  static T apply(T a, T b) { return a > b ? a : b; }
};

struct SumOp {
  template <typename T>
  static T apply(T a, T b) { return a + b; }
};

struct MulOp {
  template <typename T>
  static T apply(T a, T b) { return a * b; }
};

struct ExpSubOp {
  template <typename T>
  static T apply(T a, T b) { return fast::exp2(a - b); }
};

// Base fragment: 16x16 tile, 8 elements per thread across 32-lane SIMD group
struct NAXFrag {
  static constexpr short kFragRows = 16;
  static constexpr short kFragCols = 16;
  static constexpr short kElemsPerFrag = (kFragRows * kFragCols) / 32; // 8
  static constexpr short kElemRows = 2;
  static constexpr short kElemCols = 4;
  static constexpr short kElemRowsJump = 8;

  template <typename U>
  using frag_t = vec<U, kElemsPerFrag>;

  // Thread's (col, row) coordinate within the 16x16 fragment.
  // Derived from SIMD lane ID using the hardware's element mapping.
  static short2 get_coord() {
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short row = ((qid & 4) | ((lane >> 1) & 3));
    const short col = ((qid & 2) | (lane & 1)) * 4;
    return short2(col, row);
  }

  // Load a 16x16 fragment from device memory (row-major, stride = ld)
  template <typename T, typename U>
  static void load(thread frag_t<T>& dst, const device U* src,
                   const int ld, const short off_r = 0, const short off_c = 0) {
    const short2 sc = get_coord();
    src += sc.y * ld + sc.x;

    for (short i = 0; i < kElemRows; i++) {
      const short r = off_r + i * kElemRowsJump;
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[r * ld + off_c + j]);
      }
    }
  }

  // Load with row bounds checking (out-of-bounds rows get zero)
  template <typename T, typename U>
  static void load_safe(thread frag_t<T>& dst, const device U* src,
                        const int ld, const short lim_rows, const short lim_cols,
                        const short off_r = 0, const short off_c = 0) {
    const short2 sc = get_coord();
    src += sc.y * ld + sc.x;
    const short lr = lim_rows - sc.y;
    const short lc = lim_cols - sc.x;

    for (short i = 0; i < kElemRows; i++) {
      const short r = off_r + i * kElemRowsJump;
      for (short j = 0; j < kElemCols; j++) {
        if (r < lr && (off_c + j) < lc)
          dst[i * kElemCols + j] = static_cast<T>(src[r * ld + off_c + j]);
        else
          dst[i * kElemCols + j] = T(0);
      }
    }
  }

  // Store a 16x16 fragment to device memory
  template <typename T, typename U>
  static void store(const thread frag_t<T>& src, device U* dst,
                    const int ld, const short off_r = 0, const short off_c = 0) {
    const short2 sc = get_coord();
    dst += sc.y * ld + sc.x;

    for (short i = 0; i < kElemRows; i++) {
      const short r = off_r + i * kElemRowsJump;
      for (short j = 0; j < kElemCols; j++) {
        dst[r * ld + off_c + j] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  // Store with row bounds checking
  template <typename T, typename U>
  static void store_rows(const thread frag_t<T>& src, device U* dst,
                         const int ld, const short lim_rows,
                         const short off_r = 0, const short off_c = 0) {
    const short2 sc = get_coord();
    dst += sc.y * ld + sc.x;
    const short lr = lim_rows - sc.y;

    for (short i = 0; i < kElemRows; i++) {
      const short r = off_r + i * kElemRowsJump;
      if (r < lr) {
        for (short j = 0; j < kElemCols; j++) {
          dst[r * ld + off_c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  // Row-wise reduction across the SIMD group.
  // Reduces each of the 2 per-thread rows using shuffle_xor for
  // cross-lane communication, then accumulates into reduced_vals.
  template <typename Op, typename T>
  static void row_reduce(const thread frag_t<T>& vals, thread T* reduced) {
    for (short i = 0; i < kElemRows; i++) {
      T local = Op::apply(
          Op::apply(vals[i * kElemCols + 0], vals[i * kElemCols + 1]),
          Op::apply(vals[i * kElemCols + 2], vals[i * kElemCols + 3]));

      T pair = simd_shuffle_xor(local, ushort(1));
      pair = Op::apply(local, pair);

      T full = simd_shuffle_xor(pair, ushort(8));
      full = Op::apply(pair, full);

      reduced[i] = Op::apply(reduced[i], full);
    }
  }

  // Apply a binary op element-wise: vals[row] = Op(vals[row], row_vals[row])
  template <typename Op, typename T>
  static void row_bin_op(thread frag_t<T>& vals, thread T* row_vals) {
    for (short i = 0; i < kElemRows; i++) {
      for (short j = 0; j < kElemCols; j++) {
        vals[i * kElemCols + j] =
            Op::apply(vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  // Matrix multiply-accumulate: C[16x32] += A[16x16] * B[16x32]
  // Uses mpp::tensor_ops for hardware-accelerated cooperative MMA.
  // A is one 16x16 fragment, B and C are two 16x16 fragments (forming 16x32).
  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  static void mma(
      thread frag_t<CType>& C0, thread frag_t<CType>& C1,
      const thread frag_t<AType>& A,
      const thread frag_t<BType>& B0, const thread frag_t<BType>& B1) {

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), CType>();

    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
      ct_b[i] = B0[i];
      ct_b[kElemsPerFrag + i] = B1[i];
      ct_c[i] = C0[i];
      ct_c[kElemsPerFrag + i] = C1[i];
    }

    op.run(ct_a, ct_b, ct_c);

    for (short i = 0; i < kElemsPerFrag; i++) {
      C0[i] = ct_c[i];
      C1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

// Tile: a grid of kTileRows x kTileCols fragments (each 16x16).
// E.g. NAXTile<float, 1, 4> = one row of four fragments = 16x64 tile.
template <typename T, short kTileRows_, short kTileCols_>
struct NAXTile {
  using Frag = NAXFrag;
  using elem_type = T;
  using frag_type = typename Frag::frag_t<T>;

  static constexpr short kFragRows = Frag::kFragRows;   // 16
  static constexpr short kFragCols = Frag::kFragCols;   // 16
  static constexpr short kElemsPerFrag = Frag::kElemsPerFrag; // 8
  static constexpr short kTileRows = kTileRows_;
  static constexpr short kTileCols = kTileCols_;
  static constexpr short kRows = kTileRows * kFragRows;
  static constexpr short kCols = kTileCols * kFragCols;
  static constexpr short kNumFrags = kTileRows * kTileCols;
  static constexpr short kElemsPerTile = kNumFrags * kElemsPerFrag;
  static constexpr short kRowsPerThread = kTileRows * Frag::kElemRows;

  frag_type val_frags[kNumFrags];

  void clear() {
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  thread frag_type& frag_at(short i, short j) {
    return val_frags[i * kTileCols + j];
  }

  const thread frag_type& frag_at(short i, short j) const {
    return val_frags[i * kTileCols + j];
  }

  thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  // Load entire tile from device memory (row-major)
  template <typename U>
  void load(const device U* src, const int ld) {
    for (short ir = 0; ir < kTileRows; ir++) {
      for (short ic = 0; ic < kTileCols; ic++) {
        Frag::load(frag_at(ir, ic), src, ld,
                   ir * kFragRows, ic * kFragCols);
      }
    }
  }

  // Load with bounds checking
  template <typename U>
  void load_safe(const device U* src, const int ld,
                 const short lim_rows, const short lim_cols) {
    for (short ir = 0; ir < kTileRows; ir++) {
      for (short ic = 0; ic < kTileCols; ic++) {
        Frag::load_safe(frag_at(ir, ic), src, ld,
                        lim_rows, lim_cols,
                        ir * kFragRows, ic * kFragCols);
      }
    }
  }

  // Store entire tile to device memory
  template <typename U>
  void store(device U* dst, const int ld) const {
    for (short ir = 0; ir < kTileRows; ir++) {
      for (short ic = 0; ic < kTileCols; ic++) {
        Frag::store(frag_at(ir, ic), dst, ld,
                    ir * kFragRows, ic * kFragCols);
      }
    }
  }

  // Store with row bounds checking
  template <typename U>
  void store_rows(device U* dst, const int ld, const short n_rows) const {
    for (short ir = 0; ir < kTileRows; ir++) {
      for (short ic = 0; ic < kTileCols; ic++) {
        Frag::store_rows(frag_at(ir, ic), dst, ld, n_rows,
                         ir * kFragRows, ic * kFragCols);
      }
    }
  }

  // Row-wise reduction across all fragments
  template <typename Op>
  void row_reduce(thread vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        Frag::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * Frag::kElemRows]);
      }
    }
  }

  // Row-wise binary operation
  template <typename Op>
  void row_bin_op(thread vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        Frag::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * Frag::kElemRows]);
      }
    }
  }
};

// Tile-level MMA: C += A * B^T (or A * B) using cooperative tensor MMA.
// Iterates over fragments, dispatching 16x32x16 MMA operations.
// Requires: C.cols == B.cols (or B.rows if transposed), etc.
template <class CTile, class ATile, class BTile,
          bool transpose_a = false, bool transpose_b = false>
void tile_mma(thread CTile& C, thread ATile& A, thread BTile& B) {
  constexpr short TM = CTile::kTileRows;
  constexpr short TN = CTile::kTileCols;
  constexpr short TK = transpose_a ? ATile::kTileRows : ATile::kTileCols;

  // N-dimension must be even for the 16x32x16 MMA layout
  static_assert(TN % 2 == 0 || TN == 1, "tile_mma: TN must be even or 1");

  if constexpr (TN >= 2) {
    for (short mm = 0; mm < TM; ++mm) {
      for (short nn = 0; nn < TN; nn += 2) {
        for (short kk = 0; kk < TK; ++kk) {
          auto& a_frag = transpose_a ? A.frag_at(kk, mm) : A.frag_at(mm, kk);
          auto& b0 = transpose_b ? B.frag_at(nn, kk) : B.frag_at(kk, nn);
          auto& b1 = transpose_b ? B.frag_at(nn + 1, kk) : B.frag_at(kk, nn + 1);

          NAXFrag::template mma<
              typename CTile::elem_type,
              typename ATile::elem_type,
              typename BTile::elem_type,
              transpose_a, transpose_b>(
              C.frag_at(mm, nn), C.frag_at(mm, nn + 1),
              a_frag, b0, b1);
        }
      }
    }
  }
}

} // namespace pytorch_nax
