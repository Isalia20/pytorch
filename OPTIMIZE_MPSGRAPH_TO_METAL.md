# Migrating MPS Ops from MPSGraph to Metal Kernels

Summary of optimizations done migrating PyTorch MPS operations from Apple's MPSGraph API to direct Metal compute kernels.

## Why Metal over MPSGraph?

MPSGraph adds per-op overhead: graph construction, caching, placeholder creation, feed dictionary setup. For simple element-wise ops, this overhead dominates at small-to-medium tensor sizes. Metal kernels dispatch directly to the GPU with minimal CPU-side work.

Typical speedups observed:
- Small tensors (16x16 to 256x256): **2-5x faster** (overhead-dominated regime)
- Medium tensors (1024x1024): **1.2-1.8x faster**
- Large tensors (4096+): **1.0-1.2x faster** for contiguous, **1.5-3x faster** for non-contiguous

## Ops Migrated

### 1. fill_ / zero_ (ConstantOps.mm)

**Kernels:** `fill_scalar`, `fill_scalar_vec4`, `fill_scalar_2d`, `fill_scalar_2d_vec4`, `fill_scalar_strided`

Key optimizations:
- **2D grid dispatch** for non-contiguous 2D tensors eliminates expensive integer division/modulo in stride offset computation. Thread.x maps to the smallest-stride dimension for memory coalescing.
- **Vec4 kernels** (4 elements/thread) for 2-byte types (fp16, bf16) reduce per-thread scheduling overhead.
- **tgY=1 threadgroup sizing** for 2D dispatch prevents cache thrashing when stride_slow is large (e.g., `tensor[::2]` with large row gaps).
- Contiguous zero fill still uses `fillBuffer` (hardware memset) via the existing fast path.

### 2. eye (Eye.mm)

**Kernels:** `eye` (2D single-pass), `eye_diag` (diagonal-only for large tensors)

- Small tensors: single 2D dispatch writes both 0s and 1s.
- Large tensors: `zero_()` + diagonal-only kernel. The memset is faster than branching writes at scale.
- Complex type support (float2/half2) with correct identity value `(1, 0)` not `(1, 1)`.

### 3. relu (Activation.mm)

**Kernel:** `relu_functor` registered via `REGISTER_UNARY_OP`

Used the existing `exec_unary_kernel` infrastructure from `c10/metal/indexing.h` which handles contiguous, strided, and dtype dispatch automatically via TensorIterator. The implementation is ~10 lines replacing ~80 lines of MPSGraph boilerplate per function.

## Infrastructure Improvements

### Vec4 Unary Kernel (c10/metal/indexing.h)

Added `unary_dense_vec4` template to the generic unary kernel infrastructure. Processes 4 elements per thread for 2-byte types, automatically selected by `exec_unary_kernel` when:
- Tensor is contiguous
- Element size <= 2 bytes (fp16, bf16)
- No alpha scalar parameter

This benefits ALL ops registered with `REGISTER_UNARY_OP` (relu, hardsigmoid, hardswish, special math ops, etc.) without any per-op changes.

## Patterns and Lessons

### Dispatch patterns by tensor layout

| Layout | Strategy | Why |
|--------|----------|-----|
| Contiguous, >= 4 byte elements | 1D dispatch, 1 element/thread | Already bandwidth-saturated |
| Contiguous, <= 2 byte elements | 1D dispatch, vec4 (4 elem/thread) | Reduces thread overhead for small elements |
| Non-contiguous 2D | 2D grid dispatch | Avoids division/modulo for offset computation |
| Non-contiguous 2D, stride_fast=1, <= 2 byte | 2D grid + vec4 along fast dim | Combines both optimizations |
| Non-contiguous nD (n>2) | General strided with div/mod loop | Fallback |

### Using the exec_unary_kernel infrastructure

For simple element-wise ops, the functor + `exec_unary_kernel` pattern is the cleanest approach:

```metal
// In ActivationKernel.metal
struct my_op_functor {
  template <typename T>
  inline T operator()(const T x) {
    return /* your op */;
  }
};
REGISTER_UNARY_OP(my_op, float, float);
REGISTER_UNARY_OP(my_op, half, half);
REGISTER_UNARY_OP(my_op, bfloat, bfloat);
```

```c++
// Host side
auto iter = at::TensorIteratorConfig()
    .add_output(output)
    .add_input(input)
    .build();
lib.exec_unary_kernel(iter, "my_op");
```

This handles contiguous/strided dispatch, dtype selection, vec4 optimization, and 32-bit indexing decomposition for large tensors automatically.

### Threadgroup sizing matters

For 2D dispatch, using `tgY=1` (all threads in the fast dimension) prevents cache thrashing when rows are far apart in memory. With `tgY=32`, each threadgroup touches 32 rows potentially megabytes apart, causing L1/L2 cache pressure.

### When NOT to migrate

- Ops with complex broadcasting between multiple inputs (addcmul, addcdiv) -- MPSGraph handles broadcasting automatically
- Ops that are already compute-bound at typical sizes -- the MPSGraph overhead is amortized
- Reduction ops (sum, softmax) -- need parallel reduction algorithms, not just element-wise kernels
