# Metal Kernel Migration Tracker

## Completed

| Op | Files Changed | PR/Branch | Status |
|----|---------------|-----------|--------|
| `eye` | Eye.metal, Eye.mm | eye-metal | Done |
| `fill_`, `zero_` | ConstantOps.metal, ConstantOps.mm | eye-metal | Done |
| `lerp` | (previously migrated) | | Done |
| `relu`, `relu_` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `threshold` (forward) | ActivationKernel.metal, Activation.h, Activation.mm | eye-metal | Done |
| `threshold_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `sigmoid_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `tanh_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `silu`, `silu_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `gelu`, `gelu_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `mish`, `mish_backward` | ActivationKernel.metal, Activation.mm | eye-metal | Done |
| `softplus`, `softplus_backward` | ActivationKernel.metal, Activation.h, Activation.mm | eye-metal | Done (bf16 fwd 0.78x at 4096 — no native bf16 transcendentals) |

## Infrastructure Improvements

| Change | File | Description |
|--------|------|-------------|
| Vec4 unary kernel | c10/metal/indexing.h | 4 elem/thread for fp16/bf16, auto-applies to all `REGISTER_UNARY_OP` ops |
| Vec4 unary dispatch | OperationUtils.mm | `exec_unary_kernel` selects vec4 when element_size <= 2 |

## Remaining Candidates (Tier 1)

| Op | File | Lines | Notes |
|----|------|-------|-------|
| PixelShuffle | PixelShuffle.mm | 94 | Reshape/transpose, may not benefit much |
| addcmul, addcdiv | PointwiseOps.mm | 117 | Broadcasting + type promotion, complex |
| SummaryOps | SummaryOps.mm | 121 | Reduction (bincount) |

## Remaining Candidates (Tier 2 - High Impact)

| Op | File | Lines | Notes |
|----|------|-------|-------|
| softmax, log_softmax | SoftMax.mm | 195 | Parallel reduction, critical for transformers |
| threshold_backward | Activation.mm | ~45 | Done — direct dispatch bypassing exec_binary_kernel overhead |
| gelu, silu, sigmoid + backwards | Activation.mm | ~600 | High impact activations, use functor pattern |
| sort | Sort.mm | 188 | GPU sorting algorithm |
| weight_norm | WeightNorm.mm | 175 | L2 norm + normalization |

## Remaining Candidates (Tier 3 - Large)

| Op | File | Lines | Notes |
|----|------|-------|-------|
| Loss functions | LossOps.mm | 1293 | mse, cross_entropy, nll_loss |
| Activations (rest) | Activation.mm | 1412 | log_softmax, prelu, mish, softplus, glu |
| Unary math | UnaryOps.mm | 406 | sin, cos, exp, log, etc. |
| Binary ops | BinaryOps.mm | 295 | add, sub, mul, div |
| Copy | Copy.mm | 344 | dtype casting, cross-device |
| Range factories | RangeFactories.mm | 256 | arange, linspace |
