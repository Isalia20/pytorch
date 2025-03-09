#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;


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
    uint3 tgid [[threadgroup_position_in_grid]]) {
    // Calculate batch and head indices
    const uint batch_idx = tgid.z / num_heads;
    const uint head_idx = tgid.z % num_heads;
    const uint row_start = tgid.y * 8;
    const uint col_start = tgid.x * 8;
    
    // Initialize the result matrix for this tile
    simdgroup_matrix<float, 8, 8> a_tile;
    simdgroup_matrix<float, 8, 8> b_tile;
    // simdgroup_matrix<float, 8, 8> scale_mat = simdgroup_matrix<float, 8, 8>(1.0f / scale);
    simdgroup_matrix<float, 8, 8> result_tile = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    
    // Iterate through all 8x8 tiles needed to compute this output tile
    for (uint k = 0; k < head_dim; k += 8) {
        // Load the A tile (8x8 block)
        simdgroup_load(a_tile, 
                       &query[row_start * head_dim + k],  // Pointer to start of tile
                       head_dim,  // Leading dimension (stride between rows)
                       0,  // No matrix offset
                       false);  // Not transposed
        
        // Load the B tile (8x8 block)
        // B tile is column major since we did a transpose before this kernel
        simdgroup_load(b_tile, 
                       &key[col_start * head_dim + k],  // Pointer to start of tile
                       head_dim,  // Leading dimension
                       0,  // No matrix offset
                       true);  // Transpose B for computing A*B
        // Multiply tiles and accumulate
        simdgroup_multiply_accumulate(result_tile, a_tile, b_tile, result_tile);
    }
    // scale the matrix // TODO uncomment this once done testing
    // simdgroup_multiply(result_tile, result_tile, scale_mat);

    // Store the result tile to the output matrix
    simdgroup_store(result_tile, 
                    &output[row_start * seq_len_k + col_start],  // Pointer to start of output tile
                    seq_len_k,  // Leading dimension
                    0);  // No matrix offset
}
