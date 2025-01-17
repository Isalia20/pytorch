#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
template <typename T>
T dot_product(constant T* v1, constant T* v2, ulong2 strides, uint32_t size) {
  T rc = T(0.0);
  for (uint32_t i = 0; i < size; ++i) {
    rc += v1[i * strides.x] * v2[i * strides.y];
  }
  return rc;
}

template <typename T>
kernel void naive_matmul(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint thread_index [[thread_position_in_grid]]) {
  uint y = thread_index / sizes.x;
  uint x = thread_index % sizes.x;
  if (x >= sizes.x || y >= sizes.z) {
    return;
  }
  auto rc = dot_product(
      mat1Data + x * strides[0].x,
      mat2Data + y * strides[1].y,
      ulong2(strides[0].y, strides[1].x),
      sizes.y);
  outputData[x * strides[2].x + y * strides[2].y] = rc;
}

inline float blockReduceSum(
    threadgroup float* sharedScratch,
    float val,
    uint linear_tid) {
  float simd_result = simd_sum(val);
  // each warp's first index should write the result to consecutive
  // ids in sharedScratch buffer
  if (linear_tid % 32 == 0) {
    sharedScratch[linear_tid / 32] = simd_result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // final reduction across first warp
  if (linear_tid < 8) { // 256/32 = 8 simdgroups
    float sum = sharedScratch[linear_tid];
    sum = simd_sum(sum);
    sharedScratch[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return sharedScratch[0];
}

// ============================================================
// 2) factorDiagonalBlock kernel
//    2D threadgroup: (32, 8, 1)
// ============================================================
kernel void factorDiagonalBlock(
    device float* A [[buffer(0)]],
    device int* success [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  // Compute a linear thread ID from (tid.x, tid.y)
  uint tx = tid.x;
  uint ty = tid.y;
  uint linear_tid = ty * tpg.x + tx; // = ty*32 + tx
  uint group_size = tpg.x * tpg.y; // = 256

  const uint actSize = min(N - k * NB, NB);
  const uint batch_offset = bid.x * N * N;
  const uint row0 = k * NB;
  const uint col0 = k * NB;

  threadgroup float tile[32][33];
  threadgroup float reduceScratch[8]; // 256 / 32(warp_size)
  const uint tileSize = actSize * actSize;

  for (uint i = linear_tid; i < tileSize; i += group_size) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kk = 0; kk < actSize; kk++) {
    float diagElt = 0.0f;
    if (kk > 0) {
      float partialSum = 0.0f;
      for (uint i = linear_tid; i < kk; i += group_size) {
        float val = tile[kk][i];
        partialSum = fma(val, val, partialSum);
      }
      diagElt = blockReduceSum(reduceScratch, partialSum, linear_tid);
    }

    if (linear_tid == 0) {
      float diagVal = tile[kk][kk] - diagElt;
      if (diagVal <= 0.0f) {
        success[bid.x] = 0; // matrix is not positive definite
        return;
      }
      tile[kk][kk] = sqrt(diagVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = tile[kk][kk];

    for (uint j = kk + 1 + linear_tid; j < actSize; j += group_size) {
      float partialSum = 0.0f;
      for (uint i = 0; i < kk; i++) {
        partialSum = fma(tile[j][i], tile[kk][i], partialSum);
      }
      float val = tile[j][kk];
      val -= partialSum;
      val /= pivot;
      tile[j][kk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = linear_tid; i < tileSize; i += group_size) {
    uint r = i / actSize;
    uint c = i % actSize;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = tile[r][c];
  }
}

kernel void applyTRSM(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint tx = tid.x;
  uint ty = tid.y;
  uint linear_tid = ty * tpg.x + tx;
  uint group_size = tpg.x * tpg.y; // = 32*8 = 256
  uint b = tgid.x;
  uint idxJ = tgid.y;

  const uint actSize_k = min(int64_t(N - k * NB), int64_t(NB));
  const uint batch_offset = b * N * N;
  const uint j = (k + 1) + idxJ;

  if (actSize_k == 0 || j >= (N + NB - 1) / NB) {
    return;
  }
  if (j == k) {
    return;
  }

  uint row0 = j * NB;
  uint col0 = k * NB;
  uint actSize_j = min((int)(N - row0), (int)NB);

  if (actSize_j == 0) {
    return;
  }

  threadgroup float diag[32 * 32];
  threadgroup float target[32 * 32];

  for (uint i = linear_tid; i < actSize_k * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    diag[i] = A[batch_offset + (k * NB + r) * N + (k * NB + c)];
  }
  // ---------------------------------------------------
  // 2) Load the target block (actSize_j x actSize_k)
  // ---------------------------------------------------
  for (uint i = linear_tid; i < actSize_j * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    target[i] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint col = 0; col < actSize_k; col++) {
    float diag_val = diag[col * actSize_k + col];
    if (fabs(diag_val) < 1e-6f) {
      diag_val = (diag_val < 0.0f) ? -1e-6f : 1e-6f;
    }

    for (uint row = linear_tid; row < actSize_j; row += group_size) {
      float sum = target[row * actSize_k + col];

      for (uint p = 0; p < col; p++) { //
        sum = fma(target[row * actSize_k + p], -diag[col * actSize_k + p], sum);
      }
      target[row * actSize_k + col] = sum / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = linear_tid; i < actSize_j * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = target[i];
  }
}

kernel void applySYRK(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  // Flatten 2D portion of tid
  uint tx = tid.x;
  uint ty = tid.y;

  // Batch index and pairID from threadgroup grid
  uint b = tgid.x;
  uint pairID = tgid.y;

  // Decompose pairID into (jRel, hRel) using the inverse triangular mapping
  uint jRel = (uint)((-1.0 + sqrt(1.0 + 8.0 * float(pairID))) / 2.0);
  uint hRel = pairID - ((jRel * (jRel + 1)) >> 1);

  // Starting block indices
  // NOTE: If your block decomposition uses a different formula for j/h, change
  // here:
  uint startJ = (k + 1);
  uint j = startJ + jRel;
  uint h = startJ + hRel;

  uint row0 = j * NB; // top-left row of the j-th block
  uint col0 = h * NB; // top-left column of the h-th block

  // Actual block sizes (to handle boundary conditions)
  const uint actSize_k = min(int64_t(N - k * NB), int64_t(NB));
  const uint actSize_j = min((uint)(N - row0), NB);
  const uint actSize_h = min((uint)(N - col0), NB);

  const uint batch_offset = b * N * N;

  // If there is no valid work in this block region, exit early
  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0) {
    return;
  }

  // --------------------------------------------------------------------------------
  //    SYRK update, i.e. C = C - A * A^T, using simdgroup_matrix instructions.
  //    We'll process the whole matrix of C in 8×8 chunks.
  //
  //    NOTE: This assumes actSize_j, actSize_h, actSize_k are multiples of 8.
  // --------------------------------------------------------------------------------

  // Warp ID and lane ID within the warp
  uint warp_id = sgitg; // [0..7] (assuming 8 warps of 32 threads each)

  simdgroup_matrix<float, 8, 8> negative_identity =
      simdgroup_matrix<float, 8, 8>(-1.0);
  simdgroup_matrix<float, 8, 8> identity = simdgroup_matrix<float, 8, 8>(1.0);
  simdgroup_matrix<float, 8, 8> Prod;
  simdgroup_matrix<float, 8, 8> Afrag;
  simdgroup_matrix<float, 8, 8> Bfrag;

// Each warp handles 2 sub-blocks because we have 16 sub-blocks and 8 warps.
#pragma unroll 2
  for (uint sb = warp_id; sb < 16; sb += 8) {
    // sub-block (sb) => (sb_x, sb_y), each in multiples of 8
    uint sb_x = (sb % 4) * 8;
    uint sb_y = (sb / 4) * 8;

    // If j == h, skip upper triangular sub-blocks to match typical SYRK usage
    if (j == h && sb_y < sb_x) {
      continue;
    }

    // Load Cfrag (8×8) from global memory
    simdgroup_matrix<float, 8, 8> Cfrag;
    simdgroup_load(
        Cfrag, &A[batch_offset + (row0 + sb_y) * N + col0 + sb_x], N);

    // For each chunk of size 8 in the k-dimension
    for (uint kk = 0; kk < actSize_k; kk += 8) {
      simdgroup_load(
          Afrag, &A[batch_offset + (row0 + sb_y) * N + k * NB + kk], N);

      simdgroup_load(
          Bfrag,
          &A[batch_offset + (col0 + sb_x) * N + k * NB + kk],
          N,
          0,
          true);

      // Multiply Afrag × Bfrag
      simdgroup_multiply(Prod, Afrag, Bfrag);

      // Make it negative, then subtract from Cfrag => Cfrag -= Afrag × Bfrag
      simdgroup_multiply(Prod, Prod, negative_identity);
      simdgroup_multiply_accumulate(Cfrag, Cfrag, identity, Prod);
    }

    // Store updated Cfrag back into global memory
    simdgroup_store(
        Cfrag,
        &A[batch_offset + (row0 + sb_y) * N + col0 + sb_x],
        N); //
  }
}

#define INSTANTIATE_NAIVE_MM(DTYPE)                          \
  template [[host_name("naive_matmul_" #DTYPE)]] kernel void \
  naive_matmul<DTYPE>(                                       \
      constant DTYPE * mat1Data [[buffer(0)]],               \
      constant DTYPE * mat2Data [[buffer(1)]],               \
      device DTYPE * outputData [[buffer(2)]],               \
      constant array<ulong2, 3> & strides [[buffer(3)]],     \
      constant uint3 & sizes [[buffer(4)]],                  \
      uint thread_index [[thread_position_in_grid]])

INSTANTIATE_NAIVE_MM(float);
INSTANTIATE_NAIVE_MM(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_NAIVE_MM(bfloat);
#endif
