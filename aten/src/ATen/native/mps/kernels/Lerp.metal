#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void lerp_tensor_kernel(
    device const T* self [[buffer(0)]],
    device const T* end [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  T s = self[tid];
  out[tid] = s + mul(weight[tid], end[tid] - s);
}

#define INSTANTIATE_LERP(DTYPE)                                    \
  template [[host_name("lerp_tensor_kernel_" #DTYPE)]] kernel void \
  lerp_tensor_kernel<DTYPE>(                                       \
      device const DTYPE* self [[buffer(0)]],                      \
      device const DTYPE* end [[buffer(1)]],                       \
      device const DTYPE* weight [[buffer(2)]],                    \
      device DTYPE* out [[buffer(3)]],                             \
      uint tid [[thread_position_in_grid]]);

INSTANTIATE_LERP(float);
INSTANTIATE_LERP(half);
INSTANTIATE_LERP(bfloat);
INSTANTIATE_LERP(float2);
