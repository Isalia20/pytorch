#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void fill_scalar(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = value;
}

template <typename T>
kernel void fill_scalar_vec4(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  uint base = index * 4;
  if (base + 4 <= numel) {
    output[base] = value;
    output[base + 1] = value;
    output[base + 2] = value;
    output[base + 3] = value;
  } else {
    for (uint i = base; i < numel; i++)
      output[i] = value;
  }
}

template <typename T>
kernel void fill_scalar_2d(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant long& stride0 [[buffer(2)]],
    constant long& stride1 [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  output[pos.x * stride0 + pos.y * stride1] = value;
}

template <typename T>
kernel void fill_scalar_2d_vec4(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant long& stride_slow [[buffer(2)]],
    constant uint& fast_size [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  long row_start = pos.y * stride_slow;
  uint base = pos.x * 4;
  if (base + 4 <= fast_size) {
    output[row_start + base] = value;
    output[row_start + base + 1] = value;
    output[row_start + base + 2] = value;
    output[row_start + base + 3] = value;
  } else {
    for (uint i = base; i < fast_size; i++)
      output[row_start + i] = value;
  }
}

template <typename T>
kernel void fill_scalar_strided(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant long* strides [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant int& ndim [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  long offset = 0;
  uint remaining = index;
  for (int i = ndim - 1; i >= 0; --i) {
    offset += (remaining % (uint)sizes[i]) * strides[i];
    remaining /= (uint)sizes[i];
  }
  output[offset] = value;
}

#define REGISTER_FILL_OP(DTYPE)                                     \
  template [[host_name("fill_scalar_" #DTYPE)]] kernel void         \
  fill_scalar<DTYPE>(                                               \
      device DTYPE * output [[buffer(0)]],                          \
      constant DTYPE & value [[buffer(1)]],                         \
      uint index [[thread_position_in_grid]]);                      \
  template [[host_name("fill_scalar_vec4_" #DTYPE)]] kernel void    \
  fill_scalar_vec4<DTYPE>(                                          \
      device DTYPE * output [[buffer(0)]],                          \
      constant DTYPE & value [[buffer(1)]],                         \
      constant uint & numel [[buffer(2)]],                          \
      uint index [[thread_position_in_grid]]);                      \
  template [[host_name("fill_scalar_2d_" #DTYPE)]] kernel void      \
  fill_scalar_2d<DTYPE>(                                            \
      device DTYPE * output [[buffer(0)]],                          \
      constant DTYPE & value [[buffer(1)]],                         \
      constant long& stride0 [[buffer(2)]],                         \
      constant long& stride1 [[buffer(3)]],                         \
      uint2 pos [[thread_position_in_grid]]);                       \
  template [[host_name("fill_scalar_2d_vec4_" #DTYPE)]] kernel void \
  fill_scalar_2d_vec4<DTYPE>(                                       \
      device DTYPE * output [[buffer(0)]],                          \
      constant DTYPE & value [[buffer(1)]],                         \
      constant long& stride_slow [[buffer(2)]],                     \
      constant uint& fast_size [[buffer(3)]],                       \
      uint2 pos [[thread_position_in_grid]]);                       \
  template [[host_name("fill_scalar_strided_" #DTYPE)]] kernel void \
  fill_scalar_strided<DTYPE>(                                       \
      device DTYPE * output [[buffer(0)]],                          \
      constant DTYPE & value [[buffer(1)]],                         \
      constant long* strides [[buffer(2)]],                         \
      constant long* sizes [[buffer(3)]],                           \
      constant int& ndim [[buffer(4)]],                             \
      uint index [[thread_position_in_grid]]);

REGISTER_FILL_OP(float);
REGISTER_FILL_OP(half);
REGISTER_FILL_OP(bfloat);
REGISTER_FILL_OP(long);
REGISTER_FILL_OP(int);
REGISTER_FILL_OP(short);
REGISTER_FILL_OP(char);
REGISTER_FILL_OP(uchar);
REGISTER_FILL_OP(bool);
