// CooperativeAttention.metal - NAX cooperative tensor attention kernel
//
// Implements scaled dot-product attention using Apple's cooperative tensor
// hardware (NAX / MetalPerformancePrimitives). Each SIMD group processes a
// 16-row block of queries against all K/V tiles, using hardware-accelerated
// MMA for QK^T and score*V, and cooperative reductions for online softmax.
//
// Algorithm (per SIMD group):
//   For each KV tile:
//     S[16xBK] = Q[16xBD] @ K[BKxBD]^T    (cooperative tensor MMA)
//     S *= scale
//     online softmax update on S
//     O[16xBD] += softmax(S) @ V[BKxBD]    (cooperative tensor MMA)
//   O /= sum_exp
//
// Requires: Metal 4.0, macOS 26.2+, Apple Silicon gen 17+

#include <metal_stdlib>
#include <metal_simdgroup>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ---- Constants ----

constant constexpr short kFragSize = 16;
constant constexpr short kElems = 8;      // (16*16)/32 elements per thread
constant constexpr short kElemRows = 2;
constant constexpr short kElemCols = 4;
constant constexpr short kRowJump = 8;

// ---- Reduction functors ----

struct MaxOp {
  template <typename T> static T apply(T a, T b) { return a > b ? a : b; }
};
struct SumOp {
  template <typename T> static T apply(T a, T b) { return a + b; }
};
struct MulOp {
  template <typename T> static T apply(T a, T b) { return a * b; }
};
struct ExpSubOp {
  template <typename T> static T apply(T a, T b) { return fast::exp2(a - b); }
};

// ---- Fragment helpers (free functions, no struct-level constexpr) ----

static inline short2 frag_get_coord() {
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = lane >> 2;
  return short2(((qid & 2) | (lane & 1)) * 4,
                (qid & 4) | ((lane >> 1) & 3));
}

template <typename T, typename U>
static void frag_load(thread vec<T, kElems>& dst, const device U* src,
                      int ld, short off_r = 0, short off_c = 0) {
  const short2 sc = frag_get_coord();
  src += sc.y * ld + sc.x;
  for (short i = 0; i < kElemRows; i++) {
    short r = off_r + i * kRowJump;
    for (short j = 0; j < kElemCols; j++)
      dst[i * kElemCols + j] = static_cast<T>(src[r * ld + off_c + j]);
  }
}

template <typename T, typename U>
static void frag_load_safe(thread vec<T, kElems>& dst, const device U* src,
                           int ld, short lim_r, short lim_c,
                           short off_r = 0, short off_c = 0) {
  const short2 sc = frag_get_coord();
  src += sc.y * ld + sc.x;
  short lr = lim_r - sc.y;
  short lc = lim_c - sc.x;
  for (short i = 0; i < kElemRows; i++) {
    short r = off_r + i * kRowJump;
    for (short j = 0; j < kElemCols; j++) {
      if (r < lr && (off_c + j) < lc)
        dst[i * kElemCols + j] = static_cast<T>(src[r * ld + off_c + j]);
      else
        dst[i * kElemCols + j] = T(0);
    }
  }
}

template <typename T, typename U>
static void frag_store(const thread vec<T, kElems>& src, device U* dst,
                       int ld, short off_r = 0, short off_c = 0) {
  const short2 sc = frag_get_coord();
  dst += sc.y * ld + sc.x;
  for (short i = 0; i < kElemRows; i++) {
    short r = off_r + i * kRowJump;
    for (short j = 0; j < kElemCols; j++)
      dst[r * ld + off_c + j] = static_cast<U>(src[i * kElemCols + j]);
  }
}

template <typename T, typename U>
static void frag_store_rows(const thread vec<T, kElems>& src, device U* dst,
                            int ld, short lim_r,
                            short off_r = 0, short off_c = 0) {
  const short2 sc = frag_get_coord();
  dst += sc.y * ld + sc.x;
  short lr = lim_r - sc.y;
  for (short i = 0; i < kElemRows; i++) {
    short r = off_r + i * kRowJump;
    if (r < lr) {
      for (short j = 0; j < kElemCols; j++)
        dst[r * ld + off_c + j] = static_cast<U>(src[i * kElemCols + j]);
    }
  }
}

template <typename Op, typename T>
static void frag_row_reduce(const thread vec<T, kElems>& v, thread T* out) {
  for (short i = 0; i < kElemRows; i++) {
    T local = Op::apply(
        Op::apply(v[i * kElemCols], v[i * kElemCols + 1]),
        Op::apply(v[i * kElemCols + 2], v[i * kElemCols + 3]));
    T p = simd_shuffle_xor(local, ushort(1));
    p = Op::apply(local, p);
    T f = simd_shuffle_xor(p, ushort(8));
    f = Op::apply(p, f);
    out[i] = Op::apply(out[i], f);
  }
}

template <typename Op, typename T>
static void frag_row_bin_op(thread vec<T, kElems>& v, thread T* row_v) {
  for (short i = 0; i < kElemRows; i++)
    for (short j = 0; j < kElemCols; j++)
      v[i * kElemCols + j] = Op::apply(v[i * kElemCols + j], row_v[i]);
}

// C[16x32] += A[16x16] * B[16x32] via cooperative tensor MMA
template <typename CT, typename AT, typename BT,
          bool ta = false, bool tb = false>
static void frag_mma(thread vec<CT, kElems>& C0, thread vec<CT, kElems>& C1,
                     const thread vec<AT, kElems>& A,
                     const thread vec<BT, kElems>& B0,
                     const thread vec<BT, kElems>& B1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, ta, tb, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), CT>();
  for (short i = 0; i < kElems; i++) {
    ct_a[i] = A[i];
    ct_b[i] = B0[i];
    ct_b[kElems + i] = B1[i];
    ct_c[i] = C0[i];
    ct_c[kElems + i] = C1[i];
  }
  op.run(ct_a, ct_b, ct_c);
  for (short i = 0; i < kElems; i++) {
    C0[i] = ct_c[i];
    C1[i] = ct_c[kElems + i];
  }
}

// ---- Tile: grid of TR x TC fragments ----
// Uses raw arrays + free functions instead of struct-level constexpr/typedefs

template <typename T, short TR, short TC>
struct Tile {
  vec<T, kElems> frags[TR * TC];

  void clear() {
    for (short i = 0; i < TR * TC; i++) frags[i] = vec<T, kElems>(0);
  }

  thread vec<T, kElems>& at(short r, short c) { return frags[r * TC + c]; }
  const thread vec<T, kElems>& at(short r, short c) const { return frags[r * TC + c]; }

  thread T* elems() { return reinterpret_cast<thread T*>(frags); }
  const thread T* elems() const { return reinterpret_cast<const thread T*>(frags); }

  template <typename U>
  void load(const device U* src, int ld) {
    for (short ir = 0; ir < TR; ir++)
      for (short ic = 0; ic < TC; ic++)
        frag_load(at(ir, ic), src, ld, ir * 16, ic * 16);
  }

  template <typename U>
  void load_safe(const device U* src, int ld, short lim_r, short lim_c) {
    for (short ir = 0; ir < TR; ir++)
      for (short ic = 0; ic < TC; ic++)
        frag_load_safe(at(ir, ic), src, ld, lim_r, lim_c, ir * 16, ic * 16);
  }

  template <typename U>
  void store(device U* dst, int ld) const {
    for (short ir = 0; ir < TR; ir++)
      for (short ic = 0; ic < TC; ic++)
        frag_store(at(ir, ic), dst, ld, ir * 16, ic * 16);
  }

  template <typename U>
  void store_rows(device U* dst, int ld, short n_rows) const {
    for (short ir = 0; ir < TR; ir++)
      for (short ic = 0; ic < TC; ic++)
        frag_store_rows(at(ir, ic), dst, ld, n_rows, ir * 16, ic * 16);
  }

  template <typename Op>
  void row_reduce(thread vec<T, TR * kElemRows>& v) const {
    auto vp = (thread T*)(&v);
    for (short i = 0; i < TR; i++)
      for (short j = 0; j < TC; j++)
        frag_row_reduce<Op>(at(i, j), &vp[i * kElemRows]);
  }

  template <typename Op>
  void row_bin_op(thread vec<T, TR * kElemRows>& v) {
    auto vp = (thread T*)(&v);
    for (short i = 0; i < TR; i++)
      for (short j = 0; j < TC; j++)
        frag_row_bin_op<Op>(at(i, j), &vp[i * kElemRows]);
  }
};

// ---- Attention kernel ----

template <typename T, int BK, int BD, int WM>
kernel void cooperative_attention(
    const device T* Q      [[buffer(0)]],
    const device T* K      [[buffer(1)]],
    const device T* V      [[buffer(2)]],
    device T* O            [[buffer(3)]],
    const constant uint& qL          [[buffer(4)]],
    const constant uint& kL          [[buffer(5)]],
    const constant uint& gqa_factor  [[buffer(6)]],
    const constant float& scale      [[buffer(7)]],
    const constant uint& NK          [[buffer(8)]],
    const constant uint3& Q_strides  [[buffer(9)]],
    const constant uint3& K_strides  [[buffer(10)]],
    const constant uint3& V_strides  [[buffer(11)]],
    const constant uint3& O_strides  [[buffer(12)]],
    uint3 group_pos   [[threadgroup_position_in_grid]],
    uint simd_gid      [[simdgroup_index_in_threadgroup]],
    uint simd_lid      [[thread_index_in_simdgroup]]) {

  constexpr short kU = 16;
  constexpr short TK = BK / kU;
  constexpr short TD = BD / kU;
  constexpr short TQ = 1;
  constexpr short kRowsPT = TQ * kElemRows;

  const float scale2 = scale * M_LOG2E_F;

  const uint batch = group_pos.z;
  const uint head = group_pos.y;
  const uint q_tile = group_pos.x * WM + simd_gid;

  const short q_start = q_tile * kU;
  if (q_start >= (short)qL) return;

  const uint kv_head = head / gqa_factor;

  const device T* Q_ptr = Q + batch * Q_strides.x + head * Q_strides.y
                            + q_start * Q_strides.z;
  const device T* K_ptr = K + batch * K_strides.x + kv_head * K_strides.y;
  const device T* V_ptr = V + batch * V_strides.x + kv_head * V_strides.y;
  device T* O_ptr = O + batch * O_strides.x + head * O_strides.y
                      + q_start * O_strides.z;

  const int Q_stride = Q_strides.z;
  const int K_stride = K_strides.z;
  const int V_stride = V_strides.z;

  const bool align_Q = (q_start + kU <= (short)qL);
  const short lim_rows_q = (short)qL - q_start;

  Tile<float, TQ, TD> Otile;
  Otile.clear();

  vec<float, kRowsPT> max_score(-INFINITY);
  vec<float, kRowsPT> sum_score(0.0f);

  for (uint kb = 0; kb < NK; kb++) {
    const uint k_base = kb * BK;
    const bool is_last_k = (k_base + BK > kL);
    const short k_rem = (short)kL - (short)k_base;

    Tile<float, TQ, TK> Stile;
    Stile.clear();

    // Phase 1: S = Q @ K^T via cooperative tensor MMA
    for (short id = 0; id < TD; id++) {
      vec<T, kElems> Q_frag;
      if (align_Q) {
        frag_load(Q_frag, Q_ptr, Q_stride, short(0), short(id * kU));
      } else {
        frag_load_safe(Q_frag, Q_ptr, Q_stride, lim_rows_q, short(BD),
                       short(0), short(id * kU));
      }

      if constexpr (TK >= 2) {
        for (short ik = 0; ik < TK; ik += 2) {
          vec<T, kElems> K_frag0, K_frag1;
          const device T* K_base = K_ptr + (k_base + ik * kU) * K_stride + id * kU;

          if (!is_last_k) {
            frag_load(K_frag0, K_base, K_stride);
            frag_load(K_frag1, K_base, K_stride, short(kU), short(0));
          } else {
            frag_load_safe(K_frag0, K_base, K_stride, k_rem, short(kU));
            short rem1 = k_rem - kU;
            if (rem1 > 0)
              frag_load_safe(K_frag1, K_base, K_stride, k_rem, short(kU),
                             short(kU), short(0));
            else
              K_frag1 = vec<T, kElems>(0);
          }

          frag_mma<float, T, T, false, true>(
              Stile.at(0, ik), Stile.at(0, ik + 1),
              Q_frag, K_frag0, K_frag1);
        }
      }
    }

    // Phase 2: Scale and mask
    for (short i = 0; i < TQ * TK * kElems; i++) {
      Stile.elems()[i] *= scale2;
    }

    if (is_last_k) {
      const short2 sc = frag_get_coord();
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.at(0, ik);
        for (short ii = 0; ii < kElemRows; ii++) {
          for (short jj = 0; jj < kElemCols; jj++) {
            short col = sc.x + jj + ik * kU;
            short actual_col = (short)k_base + col;
            if (actual_col >= (short)kL)
              fg[ii * kElemCols + jj] = -INFINITY;
          }
        }
      }
    }

    // Phase 3: Online softmax
    vec<float, kRowsPT> new_max(-INFINITY);
    for (short i = 0; i < kRowsPT; i++) new_max[i] = max_score[i];

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    vec<float, kRowsPT> factor;
    for (short i = 0; i < kRowsPT; i++) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    Otile.template row_bin_op<MulOp>(factor);
    for (short i = 0; i < kRowsPT; i++)
      sum_score[i] = sum_score[i] * factor[i];
    Stile.template row_reduce<SumOp>(sum_score);

    // Phase 4: O += softmax(S) @ V via cooperative tensor MMA
    for (short id = 0; id < TD; id += 2) {
      for (short ik = 0; ik < TK; ik++) {
        vec<T, kElems> V_frag0, V_frag1;
        const device T* V_base = V_ptr + (k_base + ik * kU) * V_stride + id * kU;

        if (!is_last_k) {
          frag_load(V_frag0, V_base, V_stride);
          frag_load(V_frag1, V_base, V_stride, short(0), short(kU));
        } else {
          short v_rem = k_rem - ik * kU;
          if (v_rem > 0) {
            frag_load_safe(V_frag0, V_base, V_stride, v_rem, short(BD));
            frag_load_safe(V_frag1, V_base, V_stride, v_rem, short(BD),
                           short(0), short(kU));
          } else {
            V_frag0 = vec<T, kElems>(0);
            V_frag1 = vec<T, kElems>(0);
          }
        }

        frag_mma<float, float, T, false, false>(
            Otile.at(0, id), Otile.at(0, id + 1),
            Stile.at(0, ik), V_frag0, V_frag1);
      }
    }
  }

  // Phase 5: Normalize and write output
  vec<float, kRowsPT> rcp;
  for (short i = 0; i < kRowsPT; i++)
    rcp[i] = 1.0f / sum_score[i];
  Otile.template row_bin_op<MulOp>(rcp);

  if (align_Q) {
    Otile.store(O_ptr, (int)O_strides.z);
  } else {
    Otile.store_rows(O_ptr, (int)O_strides.z, lim_rows_q);
  }
}

// ---- Template instantiations ----

#define INSTANTIATE_COOP_ATTN(DTYPE, bk, bd, wm)                              \
  template [[host_name("cooperative_attention_" #DTYPE                         \
                       "_bk" #bk "_bd" #bd "_wm" #wm)]]                       \
  kernel void cooperative_attention<DTYPE, bk, bd, wm>(                        \
      const device DTYPE* Q [[buffer(0)]],                                     \
      const device DTYPE* K [[buffer(1)]],                                     \
      const device DTYPE* V [[buffer(2)]],                                     \
      device DTYPE* O [[buffer(3)]],                                           \
      const constant uint& qL [[buffer(4)]],                                   \
      const constant uint& kL [[buffer(5)]],                                   \
      const constant uint& gqa_factor [[buffer(6)]],                           \
      const constant float& scale [[buffer(7)]],                               \
      const constant uint& NK [[buffer(8)]],                                   \
      const constant uint3& Q_strides [[buffer(9)]],                           \
      const constant uint3& K_strides [[buffer(10)]],                          \
      const constant uint3& V_strides [[buffer(11)]],                          \
      const constant uint3& O_strides [[buffer(12)]],                          \
      uint3 group_pos [[threadgroup_position_in_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

INSTANTIATE_COOP_ATTN(half, 32, 64, 4)
INSTANTIATE_COOP_ATTN(half, 32, 128, 4)
INSTANTIATE_COOP_ATTN(bfloat, 32, 64, 4)
INSTANTIATE_COOP_ATTN(bfloat, 32, 128, 4)
INSTANTIATE_COOP_ATTN(float, 32, 64, 4)
INSTANTIATE_COOP_ATTN(float, 32, 128, 4)
