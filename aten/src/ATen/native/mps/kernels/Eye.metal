#include <metal_stdlib>
using namespace metal;

// Single-pass: writes both 0s and 1s in one dispatch (better for small tensors)
template <typename T>
kernel void eye(
    device T* output [[buffer(0)]],
    constant long& stride0 [[buffer(1)]],
    constant long& stride1 [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  output[pos.y * stride0 + pos.x * stride1] =
      (pos.x == pos.y) ? static_cast<T>(1) : static_cast<T>(0);
}

// Diagonal-only: writes 1s to pre-zeroed tensor (better for large tensors)
template <typename T>
kernel void eye_diag(
    device T* output [[buffer(0)]],
    constant long& diag_stride [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index * diag_stride] = static_cast<T>(1);
}

#define REGISTER_EYE_OP(DTYPE)                                            \
  template [[host_name("eye_" #DTYPE)]] kernel void eye<DTYPE>(           \
      device DTYPE * output [[buffer(0)]],                                \
      constant long& stride0 [[buffer(1)]],                               \
      constant long& stride1 [[buffer(2)]],                               \
      uint2 pos [[thread_position_in_grid]]);                             \
  template [[host_name("eye_diag_" #DTYPE)]] kernel void eye_diag<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                \
      constant long& diag_stride [[buffer(1)]],                           \
      uint index [[thread_position_in_grid]]);

REGISTER_EYE_OP(float);
REGISTER_EYE_OP(half);
REGISTER_EYE_OP(bfloat);
REGISTER_EYE_OP(long);
REGISTER_EYE_OP(int);
REGISTER_EYE_OP(short);
REGISTER_EYE_OP(char);
REGISTER_EYE_OP(uchar);
REGISTER_EYE_OP(bool);
