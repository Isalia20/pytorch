#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;


inline void reduceMax(
    threadgroup float* attn_scores,
    threadgroup float row_max[8][256],
    uint seq_len_k,
    uint num_threads,
    uint linear_tid) {
  // Calculate which row this thread belongs to (0-3)
  uint row_idx = linear_tid / (num_threads / 8);
  
  // calculate max for each thread initially
  float max_val = -INFINITY;
  
  // Determine this thread's starting position within its assigned row
  uint row_start = row_idx * seq_len_k;
  uint threads_per_row = num_threads / 8;
  uint thread_idx_in_row = linear_tid % threads_per_row;
  
  // Each thread processes elements with stride equal to threads_per_row
  for (uint i = thread_idx_in_row; i < seq_len_k; i += threads_per_row) {
    uint offset = row_start + i;
    max_val = fmax(max_val, attn_scores[offset]);
  }
  
  // Store thread's max value in shared memory for reduction
  row_max[row_idx][thread_idx_in_row] = max_val;
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Perform reduction within each row
  // Only the first thread in each row participates
  if (thread_idx_in_row == 0) {
    float row_max_val = row_max[row_idx][0];
    for (uint i = 1; i < threads_per_row; i++) {
      row_max_val = fmax(row_max_val, row_max[row_idx][i]);
    }
    // Store the final max value for this row
    row_max[row_idx][0] = row_max_val;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void reduceSum(
    threadgroup float* attn_scores,
    threadgroup float row_sum[8][256],
    uint seq_len_k,
    uint num_threads,
    uint linear_tid) {
  // Calculate which row this thread belongs to (0-7)
  uint row_idx = linear_tid / (num_threads / 8);
  
  // calculate sum for each thread initially
  float sum_val = 0.0f;
  
  // Determine this thread's starting position within its assigned row
  uint row_start = row_idx * seq_len_k;
  uint threads_per_row = num_threads / 8;
  uint thread_idx_in_row = linear_tid % threads_per_row;
  
  // Each thread processes elements with stride equal to threads_per_row
  for (uint i = thread_idx_in_row; i < seq_len_k; i += threads_per_row) {
    uint offset = row_start + i;
    sum_val += attn_scores[offset];
  }
  
  // Store thread's sum value in shared memory for reduction
  row_sum[row_idx][thread_idx_in_row] = sum_val;
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Perform reduction within each row
  // Only the first thread in each row participates
  if (thread_idx_in_row == 0) {
    float row_sum_val = row_sum[row_idx][0];
    for (uint i = 1; i < threads_per_row; i++) {
      row_sum_val += row_sum[row_idx][i];
    }
    // Store the final sum value for this row
    row_sum[row_idx][0] = row_sum_val;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
}


kernel void attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& seq_len_q [[buffer(6)]],
    constant uint& seq_len_k [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant bool& is_causal [[buffer(10)]],
    constant uint& blocks_per_simdgroup [[buffer(11)]],
    threadgroup char* shared_memory_q  [[threadgroup(0)]],
    threadgroup char* shared_memory_attn  [[threadgroup(1)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tid_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Calculate batch and head indices.
    const uint batch_idx = tgid.z / num_heads;
    const uint head_idx = tgid.z % num_heads;
    const uint row_start = tgid.y * 8;
    const uint col_start = sgid * 8;
    const uint batch_head_offset = batch_idx * num_heads * seq_len_q * head_dim 
                                 + head_idx * seq_len_q * head_dim;
    const uint out_batch_head_offset = batch_idx * num_heads * seq_len_q * seq_len_k 
                                     + head_idx * seq_len_q * seq_len_k;
    uint linear_tid = (sgid * 32) + tid_in_simdgroup;

    // Load entire row of Q into shared memory.
    const uint q_shared_mem_size = 8 * head_dim;
    threadgroup float* shared_Q = (threadgroup float*)(shared_memory_q);
    // shared_attn will be used later to temporarily store the dot-products and exponentials.
    threadgroup float* shared_attn = (threadgroup float*)(shared_memory_attn);
    
    // Each thread cooperatively loads query data into shared memory.
    const uint q_base_idx = batch_head_offset + row_start * head_dim;
    for (uint i = tid_in_simdgroup; i < q_shared_mem_size; i += 32) {
        uint local_row = i / head_dim;
        uint local_col = i % head_dim;
        if (local_row < 8 && (row_start + local_row) < seq_len_q) {
            shared_Q[local_row * head_dim + local_col] =
                query[q_base_idx + local_row * head_dim + local_col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Declare simdgroup matrix tiles for the multiply‑accumulate.
    simdgroup_matrix<float, 8, 8> a_tile;
    simdgroup_matrix<float, 8, 8> b_tile;
    simdgroup_matrix<float, 8, 8> result_tile;
    const uint out_row_offset = row_start * seq_len_k;
    
    // We will compute softmax “online” across all tiles.
    // Initialize our running global maximum (for a row) to -INFINITY,
    // and our running sum to 0.
    float local_max = -INFINITY;
    float local_max_lower = -INFINITY;
    float local_sum = 0.0f;
    float local_sum_lower = 0.0f;
    float exp_val = 0.0f;
    float exp_val_lower = 0.0f;
    
    // Process each tile (block) for the softmax row.
    for (uint block_idx = 0; block_idx < blocks_per_simdgroup; block_idx += 1) {
        // Compute the starting column for this block.
        const uint block_col_start = col_start + block_idx * 256; // 256=tile block stride
        const uint block_col_offset = block_col_start * head_dim;
        result_tile = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        
        // Compute dot-product for this tile.
        for (uint k = 0; k < head_dim; k += 8) {
            simdgroup_load(a_tile, &shared_Q[k], head_dim, 0, false);
            simdgroup_load(b_tile, &key[batch_head_offset + block_col_offset + k], head_dim, 0, true);
            simdgroup_multiply_accumulate(result_tile, a_tile, b_tile, result_tile);
        }
        // Store the computed tile into shared memory.
        simdgroup_store(result_tile, &shared_attn[block_col_start], seq_len_k, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each thread processes one element in the tile.
    uint row_idx_simd = tid_in_simdgroup / 8;
    uint col_idx_simd = tid_in_simdgroup % 8;
    threadgroup float rowMax[8][256];
    threadgroup float rowSum[8][256];
    // code for global max calculation per row
    reduceMax(shared_attn, rowMax, seq_len_k, tptg.x * tptg.y, linear_tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total_elems = 8 * seq_len_k;
    for (uint idx = linear_tid; idx < total_elems; idx += tptg.x * tptg.y) {
        uint row = idx / seq_len_k;
        shared_attn[idx] = exp(shared_attn[idx] - rowMax[row][0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Compute the sum of the exponentials per row.
    reduceSum(shared_attn, rowSum, seq_len_k, tptg.x * tptg.y, linear_tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 4: Normalize: divide each exponentiated value by the row sum.
    // For this example, the code is written to have each thread output two elements:
    // one from the “upper” half (rows 0-3) and one from the “lower” half (rows 4-7).
    for (uint block_idx = 0; block_idx < blocks_per_simdgroup; block_idx++) {
        uint block_col_start = col_start + block_idx * 256;
        // Compute the index for the upper half.
        uint index_upper = row_idx_simd * seq_len_k + col_idx_simd + block_col_start;
        // Compute the index for the lower half.
        uint index_lower = (row_idx_simd + 4) * seq_len_k + col_idx_simd + block_col_start;
        
        // Divide the exponentiated score by the corresponding row’s sum.
        float softmax_upper = shared_attn[index_upper] / rowSum[row_idx_simd][0];
        float softmax_lower = shared_attn[index_lower] / rowSum[row_idx_simd + 4][0];
        
        // Write back the normalized (softmax) values to the output buffer.
        output[out_batch_head_offset + out_row_offset + block_col_start +
               row_idx_simd * seq_len_k + col_idx_simd] = softmax_upper;
        output[out_batch_head_offset + out_row_offset + block_col_start +
               (row_idx_simd + 4) * seq_len_k + col_idx_simd] = softmax_lower;
    }
    // ===== End of Softmax =====
}