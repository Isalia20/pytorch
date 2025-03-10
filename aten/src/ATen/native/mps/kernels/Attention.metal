#include <metal_stdlib>
using namespace metal;

kernel void attention(
    const device float* Q         [[ buffer(0) ]],
    const device float* K         [[ buffer(1) ]],
    const device float* V         [[ buffer(2) ]],
    
    // Constant tensor dimensions and tiling parameters
    constant int &N               [[ buffer(3) ]],
    constant int &d               [[ buffer(4) ]],
    constant int &Tc              [[ buffer(5) ]],
    constant int &Tr              [[ buffer(6) ]],
    constant int &Bc              [[ buffer(7) ]],
    constant int &Br              [[ buffer(8) ]],
    constant float &softmax_scale [[ buffer(9) ]],
    
    // Global accumulation buffers for intermediate values
    device float* l               [[ buffer(10) ]],
    device float* m               [[ buffer(11) ]],
    device float* O               [[ buffer(12) ]],
    
    // Number of heads (nh) for the Q, K, V tensors.
    constant int &nh              [[ buffer(13) ]],
    
    // Built-ins for thread indices:
    uint tid_x                    [[ thread_index_in_threadgroup ]],
    uint2 group_id                [[ threadgroup_position_in_grid ]],
    
    // Dynamic threadgroup (shared) memory.
    threadgroup float* sram       [[ threadgroup(0) ]]
) {
    // Thread id and block (group) id conversion (Metal returns unsigned ints)
    int tx = int(tid_x);
    int bx = int(group_id.x);
    int by = int(group_id.y);

    // Compute the offsets for this (batch, head) pair.
    // In the original CUDA code, gridDim.y corresponds to the number of heads.
    // Thus, qkv_offset locates the (batch, head) slice in Q, K, and V.
    int qkv_offset = (bx * nh * N * d) + (by * N * d);
    int lm_offset  = (bx * nh * N) + (by * N);
    
    // Each tile for Q, K, and V is of size: Bc * d.
    int tile_size = Bc * d;
    
    // Partition the threadgroup (shared) memory.
    // sram was allocated on the host with size: (3 * Bc * d + (Bc * Br)) * sizeof(float)
    // We allocate contiguous regions for Qi, Kj, Vj, and S.
    threadgroup float* Qi = sram;
    threadgroup float* Kj = sram + tile_size;
    threadgroup float* Vj = sram + (tile_size * 2);
    threadgroup float* S  = sram + (tile_size * 3);
    
    // Loop over tiles for Keys/Values.
    for (int j = 0; j < Tc; j++) {
        // Each thread loads its portion of Kj and Vj from global memory to shared memory.
        for (int x = 0; x < d; x++) {
            int index = tx * d + x;
            Kj[index] = K[qkv_offset + (j * tile_size) + index];
            Vj[index] = V[qkv_offset + (j * tile_size) + index];
        }
        // Ensure all threads have loaded Kj and Vj.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Loop over tiles for Queries.
        for (int i = 0; i < Tr; i++)  {
            // Each thread loads its portion of Qi into shared memory.
            for (int x = 0; x < d; x++) {
                int index = tx * d + x;
                Qi[index] = Q[qkv_offset + (i * tile_size) + index];
            }
            
            // Load the current accumulated values for this row.
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];
            
            // Compute S = Qi * Kj^T and determine the row’s maximum
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                // Dot-product between Qi (current thread’s row) and each Kj row.
                for (int x = 0; x < d; x++) {
                    int qi_index = tx * d + x;
                    int kj_index = y * d + x;
                    sum += Qi[qi_index] * Kj[kj_index];
                }
                sum *= softmax_scale;
                
                // Store the result in S for further processing.
                int s_index = tx * Bc + y;
                S[s_index] = sum;
                if (sum > row_m) {
                    row_m = sum;
                }
            }
            
            // Compute the softmax denominator. S becomes the exponentiated probabilities.
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                int s_index = tx * Bc + y;
                S[s_index] = exp(S[s_index] - row_m);
                row_l += S[s_index];
            }
            
            // Combine the current iteration’s data with the previous accumulations.
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (exp(row_m_prev - row_m_new) * row_l_prev) +
                              (exp(row_m - row_m_new) * row_l);
            
            // Compute the weighted sum by multiplying the softmax probabilities with Vj.
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    int s_index = tx * Bc + y;
                    int vj_index = y * d + x;
                    pv += S[s_index] * Vj[vj_index];
                }
                int o_index = qkv_offset + (i * tile_size) + (tx * d) + x;
                // Combine with the previous output, applying the appropriate normalization.
                O[o_index] = (1.0f / row_l_new) *
                             ((row_l_prev * exp(row_m_prev - row_m_new) * O[o_index]) +
                              (exp(row_m - row_m_new) * pv));
            }
            // Store updated accumulation values.
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        // Synchronize before loading the next tile of Kj and Vj.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}