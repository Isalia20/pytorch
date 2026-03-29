#include <ATen/native/mps/kernels/Activation.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : x;
  }
};

struct softshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    if (x > lambda) {
      return x - lambda;
    } else if (x < -lambda) {
      return x + lambda;
    } else {
      return T(0);
    }
  }
};

struct shrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : grad_output;
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, bfloat, bfloat);

REGISTER_UNARY_ALPHA_OP(softshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(softshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(softshrink, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(shrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(shrink_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(shrink_backward, bfloat, bfloat, bfloat);

struct relu_functor {
  template <typename T>
  inline T operator()(const T x) {
    return x > T(0) ? x : T(0);
  }
};

REGISTER_UNARY_OP(relu, float, float);
REGISTER_UNARY_OP(relu, half, half);
REGISTER_UNARY_OP(relu, bfloat, bfloat);

struct threshold_functor {
  template <typename T>
  inline T operator()(const T x, ThresholdParams<T> params) {
    return x > params.threshold ? x : params.value;
  }
};

#define REGISTER_THRESHOLD_OP(T)                  \
  typedef ThresholdParams<T> ThresholdParams_##T; \
  REGISTER_UNARY_ALPHA_OP(threshold, T, ThresholdParams_##T, T);

REGISTER_THRESHOLD_OP(float);
REGISTER_THRESHOLD_OP(half);
REGISTER_THRESHOLD_OP(bfloat);

struct threshold_backward_functor {
  template <typename T>
  inline T operator()(const T self, const T grad_output, const T threshold_val) {
    return self > threshold_val ? grad_output : T(0);
  }
};

REGISTER_BINARY_ALPHA_OP(threshold_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(threshold_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(threshold_backward, bfloat, bfloat, bfloat);

struct sigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T output) {
    return static_cast<T>(
        float(grad_output) * float(output) * (1.0f - float(output)));
  }
};

REGISTER_BINARY_OP(sigmoid_backward, float, float);
REGISTER_BINARY_OP(sigmoid_backward, half, half);
REGISTER_BINARY_OP(sigmoid_backward, bfloat, bfloat);

struct tanh_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T output) {
    return static_cast<T>(
        float(grad_output) * (1.0f - float(output) * float(output)));
  }
};

REGISTER_BINARY_OP(tanh_backward, float, float);
REGISTER_BINARY_OP(tanh_backward, half, half);
REGISTER_BINARY_OP(tanh_backward, bfloat, bfloat);

struct hardsigmoid_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(min(max(x + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardsigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr auto one_sixth = 1.0f / 6.0f;
    return static_cast<T>(
        abs(float(self)) < 3.0f ? float(grad_output) * one_sixth : 0.0f);
  }
};

REGISTER_UNARY_OP(hardsigmoid, float, float);
REGISTER_UNARY_OP(hardsigmoid, half, half);
REGISTER_UNARY_OP(hardsigmoid, bfloat, bfloat);

REGISTER_BINARY_OP(hardsigmoid_backward, float, float);
REGISTER_BINARY_OP(hardsigmoid_backward, half, half);
REGISTER_BINARY_OP(hardsigmoid_backward, bfloat, bfloat);

struct hardswish_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(float(x) * min(max(float(x) + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardswish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr T zero(0);
    constexpr T three(3);
    constexpr T neg_three(-3);

    if (self <= neg_three) {
      return zero;
    } else if (self >= three) {
      return grad_output;
    } else {
      return static_cast<T>(float(grad_output) * (float(self) / 3.0f + 0.5f));
    }
  }
};

REGISTER_UNARY_OP(hardswish, float, float);
REGISTER_UNARY_OP(hardswish, half, half);
REGISTER_UNARY_OP(hardswish, bfloat, bfloat);

REGISTER_BINARY_OP(hardswish_backward, float, float);
REGISTER_BINARY_OP(hardswish_backward, half, half);
REGISTER_BINARY_OP(hardswish_backward, bfloat, bfloat);

struct elu_functor {
  template <typename T>
  inline T operator()(const T self_, const ELUParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto self = static_cast<op_T>(self_);
    auto neg_res = alpha * (::metal::precise::exp(self * input_scale) - 1);
    return static_cast<T>(scale * (self < 0 ? neg_res : self));
  }
};

struct elu_backward_functor {
  template <typename T>
  inline T operator()(
      const T grad_output_,
      const T self_,
      ELUBackwardParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto grad_output = static_cast<op_T>(grad_output_);
    auto self = static_cast<op_T>(self_);

    if (params.is_result) {
      auto neg_coef = input_scale * (self + alpha * scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    } else {
      auto neg_coef = input_scale * alpha * scale *
          ::metal::precise::exp(self * input_scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    }
  }
};

#define REGISTER_ELU_OP(T)            \
  typedef ELUParams<T> ELUParams_##T; \
  REGISTER_UNARY_ALPHA_OP(elu, T, ELUParams_##T, T);

REGISTER_ELU_OP(float);
REGISTER_ELU_OP(half);
REGISTER_ELU_OP(bfloat);

#define REGISTER_ELU_BACKWARD_OP(T)                   \
  typedef ELUBackwardParams<T> ELUBackwardParams_##T; \
  REGISTER_BINARY_ALPHA_OP(elu_backward, T, ELUBackwardParams_##T, T);

REGISTER_ELU_BACKWARD_OP(float);
REGISTER_ELU_BACKWARD_OP(half);
REGISTER_ELU_BACKWARD_OP(bfloat);

struct silu_functor {
  template <typename T>
  inline T operator()(const T x) {
    float xf = float(x);
    return static_cast<T>(xf / (1.0f + ::metal::precise::exp(-xf)));
  }
};

REGISTER_UNARY_OP(silu, float, float);
REGISTER_UNARY_OP(silu, half, half);
REGISTER_UNARY_OP(silu, bfloat, bfloat);

struct silu_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    float sf = float(self);
    float sig = 1.0f / (1.0f + ::metal::precise::exp(-sf));
    return static_cast<T>(float(grad_output) * sig * (1.0f + sf - sf * sig));
  }
};

REGISTER_BINARY_OP(silu_backward, float, float);
REGISTER_BINARY_OP(silu_backward, half, half);
REGISTER_BINARY_OP(silu_backward, bfloat, bfloat);

// gelu: x * 0.5 * (1 + erf(x / sqrt(2)))
struct gelu_functor {
  template <typename T>
  inline T operator()(const T x) {
    constexpr float SQRT1_2 = 0.707106781186547524400844362104849039f;
    float xf = float(x);
    return static_cast<T>(xf * 0.5f * (1.0f + c10::metal::erf(xf * SQRT1_2)));
  }
};

REGISTER_UNARY_OP(gelu, float, float);
REGISTER_UNARY_OP(gelu, half, half);
REGISTER_UNARY_OP(gelu, bfloat, bfloat);

// gelu_backward (erf mode): grad * (cdf + x * pdf)
// where cdf = 0.5*(1+erf(x/sqrt(2))), pdf = exp(-x^2/2)/sqrt(2*pi)
struct gelu_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr float SQRT1_2 = 0.707106781186547524400844362104849039f;
    constexpr float kBeta = 0.7978845608028654f * 0.5f; // M_2_SQRTPI * M_SQRT1_2 * 0.5
    float xf = float(self);
    float cdf = 0.5f * (1.0f + c10::metal::erf(xf * SQRT1_2));
    float pdf = kBeta * ::metal::precise::exp(-0.5f * xf * xf);
    return static_cast<T>(float(grad_output) * (cdf + xf * pdf));
  }
};

REGISTER_BINARY_OP(gelu_backward, float, float);
REGISTER_BINARY_OP(gelu_backward, half, half);
REGISTER_BINARY_OP(gelu_backward, bfloat, bfloat);

// gelu tanh: 0.5*x*(1+tanh(y)) where y = beta*(x + kappa*x^3)
// Fused: 0.5*x*(1+tanh(y)) = x*e2/(e2+1) where e2 = exp(2y)
struct gelu_tanh_functor {
  template <typename T>
  inline T operator()(const T x) {
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float xf = float(x);
    float y = kBeta * (xf + kKappa * xf * xf * xf);
    float e2 = ::metal::exp(2.0f * y);
    return static_cast<T>(xf * e2 / (e2 + 1.0f));
  }
};

REGISTER_UNARY_OP(gelu_tanh, float, float);
REGISTER_UNARY_OP(gelu_tanh, half, half);
REGISTER_UNARY_OP(gelu_tanh, bfloat, bfloat);

// gelu_tanh backward: grad * (e2*(e2+1 + 2*x*y') / (e2+1)^2)
// where y = beta*(x+kappa*x^3), y' = beta*(1+3*kappa*x^2), e2 = exp(2y)
struct gelu_tanh_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float xf = float(self);
    float x_sq = xf * xf;
    float y = kBeta * (xf + kKappa * xf * x_sq);
    float dy = kBeta * (1.0f + 3.0f * kKappa * x_sq);
    float e2 = ::metal::exp(2.0f * y);
    float e2p1 = e2 + 1.0f;
    float w = e2 * (e2p1 + 2.0f * xf * dy) / (e2p1 * e2p1);
    return static_cast<T>(float(grad_output) * w);
  }
};

REGISTER_BINARY_OP(gelu_tanh_backward, float, float);
REGISTER_BINARY_OP(gelu_tanh_backward, half, half);
REGISTER_BINARY_OP(gelu_tanh_backward, bfloat, bfloat);

// mish: x * tanh(log(1+exp(x)))
// Fused: let e=exp(x), then tanh(log(1+e)) = e(2+e)/(2+2e+e^2)
// So mish(x) = x * e * (2+e) / (2+2e+e^2)  — one exp instead of three transcendentals
struct mish_functor {
  template <typename T>
  inline T operator()(const T x) {
    float xf = float(x);
    float e = ::metal::exp(xf);
    float d = 2.0f + e * (2.0f + e);
    return static_cast<T>(xf * e * (2.0f + e) / d);
  }
};

REGISTER_UNARY_OP(mish, float, float);
REGISTER_UNARY_OP(mish, half, half);
REGISTER_UNARY_OP(mish, bfloat, bfloat);

// mish'(x) = tanh(sp) + x * sigmoid(x) * sech^2(sp)
// Fused with e=exp(x), d=2+2e+e^2:
//   = e*((2+e)*d + 4*x*(1+e)) / d^2
struct mish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    float xf = float(self);
    float e = ::metal::exp(xf);
    float d = 2.0f + e * (2.0f + e);
    float w = e * ((2.0f + e) * d + 4.0f * xf * (1.0f + e)) / (d * d);
    return static_cast<T>(float(grad_output) * w);
  }
};

REGISTER_BINARY_OP(mish_backward, float, float);
REGISTER_BINARY_OP(mish_backward, half, half);
REGISTER_BINARY_OP(mish_backward, bfloat, bfloat);

// softplus: log(1 + exp(x*beta)) / beta, linear when x*beta > threshold
struct softplus_functor {
  template <typename T>
  inline T operator()(const T x, const SoftplusParams<T> p) {
    T bx = x * p.beta;
    if (float(bx) > float(p.threshold))
      return x;
    return static_cast<T>(::metal::log(T(1) + ::metal::exp(bx))) / p.beta;
  }
};

#define REGISTER_SOFTPLUS_OP(T)                       \
  typedef SoftplusParams<T> SoftplusParams_##T;       \
  REGISTER_UNARY_ALPHA_OP(softplus, T, SoftplusParams_##T, T);

REGISTER_SOFTPLUS_OP(float);
REGISTER_SOFTPLUS_OP(half);
REGISTER_SOFTPLUS_OP(bfloat);

// softplus_backward: grad * exp(x*beta) / (1 + exp(x*beta)) when x*beta <= threshold, else grad
struct softplus_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self, const SoftplusParams<T> p) {
    float bx = float(self) * float(p.beta);
    if (bx > float(p.threshold))
      return grad_output;
    float e = ::metal::exp(bx);
    return static_cast<T>(float(grad_output) * e / (1.0f + e));
  }
};

#define REGISTER_SOFTPLUS_BACKWARD_OP(T)              \
  REGISTER_BINARY_ALPHA_OP(softplus_backward, T, SoftplusParams_##T, T);

REGISTER_SOFTPLUS_BACKWARD_OP(float);
REGISTER_SOFTPLUS_BACKWARD_OP(half);
REGISTER_SOFTPLUS_BACKWARD_OP(bfloat);

struct leaky_relu_functor {
  template <typename T>
  inline T operator()(const T x, const T negative_slope) {
    return float(x) > 0.0f ? x
                           : static_cast<T>(float(x) * float(negative_slope));
  }
};

struct leaky_relu_backward_functor {
  template <typename T>
  inline T operator()(
      const T self,
      const T grad_output,
      const T negative_slope) {
    return float(self) > 0.0f
        ? grad_output
        : static_cast<T>(float(grad_output) * float(negative_slope));
  }
};

REGISTER_UNARY_ALPHA_OP(leaky_relu, float, float, float);
REGISTER_UNARY_ALPHA_OP(leaky_relu, half, half, half);
REGISTER_UNARY_ALPHA_OP(leaky_relu, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, bfloat, bfloat, bfloat);
