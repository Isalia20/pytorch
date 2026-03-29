//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Activation.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/glu_native.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/sigmoid_backward_native.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/tanh_backward_native.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>
#endif

using namespace at::mps;

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ActivationKernel_metallib.h>
#endif

Tensor relu_mps(const Tensor& self) {
  Tensor output = at::empty_like(self);
  if (output.numel() == 0)
    return output;
  auto iter = at::TensorIteratorConfig()
      .add_output(output)
      .add_input(self)
      .build();
  lib.exec_unary_kernel(iter, "relu");
  return output;
}

Tensor& relu_mps_(Tensor& self) {
  if (self.numel() == 0)
    return self;
  auto iter = at::TensorIteratorConfig()
      .add_output(self)
      .add_input(self)
      .set_check_mem_overlap(false)
      .build();
  lib.exec_unary_kernel(iter, "relu");
  return self;
}

TORCH_IMPL_FUNC(log_softmax_mps_out)
(const Tensor& self, const int64_t dim, const bool half_to_float, const Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(self.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(self.scalar_type()),
                              "log_softmax for complex is not supported for MPS");
  TORCH_CHECK_NOT_IMPLEMENTED(self.scalar_type() != kBool, "log_softmax for bool is not supported for MPS");
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "log_softmax_mps_out" + getTensorsStringKey({self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* maximumsTensor = [mpsGraph reductionMaximumWithTensor:inputTensor axis:dim name:nil];
      MPSGraphTensor* inputTensorSubMax = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:maximumsTensor
                                                                            name:nil];
      MPSGraphTensor* exponentTensor = [mpsGraph exponentWithTensor:inputTensorSubMax name:nil];

      MPSGraphTensor* exponentTensorReduced = [mpsGraph reductionSumWithTensor:exponentTensor axis:dim name:nil];

      MPSGraphTensor* logSumExpTensor = [mpsGraph logarithmWithTensor:exponentTensorReduced name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensorSubMax
                                                            secondaryTensor:logSumExpTensor
                                                                       name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(log_softmax_backward_mps_out)
(const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, const Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(grad_output.scalar_type()),
                              "log_softmax for complex is not supported for MPS");
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kBool, "log_softmax for bool is not supported for MPS");
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  if (output.numel() == 0) {
    return;
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "log_softmax_backward_mps_out:" + getMPSTypeString(grad_output) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_output));
      MPSGraphTensor* outputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(output));

      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:outputTensor name:nil];
      MPSGraphTensor* sumTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor axis:dim name:nil];
      MPSGraphTensor* multiplicationTensor = [mpsGraph multiplicationWithPrimaryTensor:expTensor
                                                                       secondaryTensor:sumTensor
                                                                                  name:nil];
      MPSGraphTensor* resultTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                            secondaryTensor:multiplicationTensor
                                                                       name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->gradInputTensor_ = resultTensor;
    });

    Placeholder gradPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder resultPlaceholder = Placeholder(cachedGraph->gradInputTensor_, out);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradPlaceholder, outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, resultPlaceholder);
  }
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_mps(const Tensor& self, Tensor& output, Tensor& buffer) {
  TORCH_CHECK_NOT_IMPLEMENTED(self.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return std::forward_as_tuple(output, buffer);
  }

  output.resize_as_(self);

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor output_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    std::string key = "log_sigmoid_forward_out:" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* minTensor = [mpsGraph minimumWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
      MPSGraphTensor* absInputTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
      MPSGraphTensor* negAbsInputTensor = [mpsGraph negativeWithTensor:absInputTensor name:nil];
      MPSGraphTensor* expNegAbsInputTensor = [mpsGraph exponentWithTensor:negAbsInputTensor name:nil];
      MPSGraphTensor* outputTensor = at::native::mps::log1p(mpsGraph, expNegAbsInputTensor);
      outputTensor = [mpsGraph subtractionWithPrimaryTensor:minTensor secondaryTensor:outputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, executeGatherOp ? output_ : output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (executeGatherOp) {
    output.copy_(output_);
  }
  return std::forward_as_tuple(output, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_mps(const Tensor& self) {
  auto output = at::empty_like(self);
  auto buffer = at::empty({0}, self.options());
  log_sigmoid_forward_out_mps(self, output, buffer);
  return std::make_tuple(output, buffer);
}

Tensor& log_sigmoid_backward_mps_out(const Tensor& grad_output,
                                     const Tensor& self,
                                     const Tensor& buffer,
                                     Tensor& grad_input) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  if (self.numel() == 0) {
    return grad_input;
  }

  grad_input.resize_as_(self);

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor grad_input_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    std::string key = "log_sigmoid_backward_out:" + getTensorsStringKey({self, grad_output});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* negOneTensor = [mpsGraph constantWithScalar:-1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* inputNegPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                    secondaryTensor:zeroTensor
                                                                               name:nil];
      MPSGraphTensor* maxDerivativeTensor = [mpsGraph selectWithPredicateTensor:inputNegPredicateTensor
                                                            truePredicateTensor:oneTensor
                                                           falsePredicateTensor:zeroTensor
                                                                           name:nil];
      MPSGraphTensor* signTensor = [mpsGraph selectWithPredicateTensor:inputNegPredicateTensor
                                                   truePredicateTensor:oneTensor
                                                  falsePredicateTensor:negOneTensor
                                                                  name:nil];
      MPSGraphTensor* absInputTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
      MPSGraphTensor* negAbsInputTensor = [mpsGraph negativeWithTensor:absInputTensor name:nil];
      MPSGraphTensor* expNegAbsInputTensor = [mpsGraph exponentWithTensor:negAbsInputTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:expNegAbsInputTensor
                                                         secondaryTensor:oneTensor
                                                                    name:nil];
      outputTensor = [mpsGraph divisionWithPrimaryTensor:expNegAbsInputTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:signTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph subtractionWithPrimaryTensor:maxDerivativeTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor secondaryTensor:outputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->gradInputTensor_, executeGatherOp ? grad_input_ : grad_input, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (executeGatherOp) {
    grad_input.copy_(grad_input_);
  }
  return grad_input;
}

Tensor log_sigmoid_backward_mps(const Tensor& grad_output, const Tensor& self, const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  log_sigmoid_backward_mps_out(grad_output, self, buffer, grad_input);
  return grad_input;
}

TORCH_IMPL_FUNC(sigmoid_backward_out_mps)(const Tensor& grad_output, const Tensor& output, const Tensor& grad_input) {
  if (this->numel() == 0)
    return;
  lib.exec_binary_kernel(*this, "sigmoid_backward");
}

TORCH_IMPL_FUNC(tanh_backward_out_mps)(const Tensor& grad_output, const Tensor& output, const Tensor& grad_input) {
  if (this->numel() == 0)
    return;
  lib.exec_binary_kernel(*this, "tanh_backward");
}

TORCH_IMPL_FUNC(threshold_out_mps)
(const Tensor& self, const Scalar& threshold, const Scalar& value, const Tensor& result) {
  using namespace mps;
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, this->common_dtype(), "threshold_mps", [&]() {
    ThresholdParams<scalar_t> params{threshold.to<scalar_t>(), value.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        *this,
        "threshold",
        params,
        fmt::format("ThresholdParams_{}", scalarToMetalTypeString(this->common_dtype())));
  });
}

TORCH_IMPL_FUNC(threshold_backward_out_mps)
(const Tensor& grad, const Tensor& self, const Scalar& threshold, const Tensor& gradInput) {
  if (this->numel() == 0)
    return;
  lib.exec_binary_kernel(*this, "threshold_backward", threshold);
}

static MPSGraphTensor* normcdf(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // (1.0f + erf(x*SQRT1_2)) * 0.5f;
  auto dataType = [inputTensor dataType];
  const float SQRT1_2 = 0.707106781186547524400844362104849039f;
  MPSGraphTensor* sqrt1_2 = [mpsGraph constantWithScalar:SQRT1_2 shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];

  MPSGraphTensor* erfTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:sqrt1_2 name:nil];
  erfTensor = [mpsGraph erfWithTensor:erfTensor name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:onef name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:halff name:nil];

  return erfTensor;
}

static MPSGraphTensor* tanh(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // 0.5 * x * (1 + text{Tanh}(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  auto dataType = [inputTensor dataType];
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715f;
  MPSGraphTensor* betaf = [mpsGraph constantWithScalar:kBeta shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* kappaf = [mpsGraph constantWithScalar:kKappa shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* erfTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                        secondaryTensor:inputTensor
                                                                   name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:inputTensor name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:kappaf name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:inputTensor name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:betaf name:nil];
  erfTensor = [mpsGraph tanhWithTensor:erfTensor name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:onef name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:halff name:nil];

  return erfTensor;
}

TORCH_IMPL_FUNC(gelu_out_mps)(const Tensor& self, std::string_view approximate, const Tensor& output) {
  if (this->numel() == 0)
    return;
  auto kernel = get_gelutype_enum(approximate) == GeluType::Tanh ? "gelu_tanh" : "gelu";
  lib.exec_unary_kernel(*this, kernel);
}

TORCH_IMPL_FUNC(gelu_backward_out_mps)
(const Tensor& grad, const Tensor& self, std::string_view approximate, const Tensor& grad_input) {
  if (this->numel() == 0)
    return;
  auto kernel = get_gelutype_enum(approximate) == GeluType::Tanh ? "gelu_tanh_backward" : "gelu_backward";
  lib.exec_binary_kernel(*this, kernel);
}

TORCH_IMPL_FUNC(glu_out_mps)(const Tensor& self, const int64_t dim, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(output.is_mps());

  // Empty output
  if (output.numel() == 0)
    return;

  TORCH_CHECK_NOT_IMPLEMENTED(self.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", nIn);

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "glu_out_mps" + getTensorsStringKey({self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
      NSArray<MPSGraphTensor*>* outputTensorsArray = [mpsGraph splitTensor:inputTensor
                                                                 numSplits:2
                                                                      axis:wrap_dim
                                                                      name:nil];
      MPSGraphTensor* firstHalf = outputTensorsArray[0];
      MPSGraphTensor* secondHalf = [mpsGraph sigmoidWithTensor:outputTensorsArray[1] name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph multiplicationWithPrimaryTensor:firstHalf
                                                               secondaryTensor:secondHalf
                                                                          name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

Tensor& glu_backward_mps_out(const Tensor& grad_output, const Tensor& self, const int64_t dim, Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  // Empty output
  if (grad_input.numel() == 0)
    return grad_input;

  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", nIn);

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "glu_backward_mps_out" + getTensorsStringKey({grad_output, self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_output), getMPSShape(grad_output));
      NSArray<MPSGraphTensor*>* inputTensorsArray = [mpsGraph splitTensor:inputTensor
                                                                numSplits:2
                                                                     axis:wrap_dim
                                                                     name:nil];

      // first half
      MPSGraphTensor* sigmoidOutputTensor = [mpsGraph sigmoidWithTensor:inputTensorsArray[1] name:nil];
      MPSGraphTensor* firstHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:sigmoidOutputTensor
                                                                        secondaryTensor:gradOutputTensor
                                                                                   name:nil];

      // second half
      MPSGraphTensor* one_val = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* secondHalfOutputTensor = [mpsGraph subtractionWithPrimaryTensor:one_val
                                                                      secondaryTensor:sigmoidOutputTensor
                                                                                 name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:sigmoidOutputTensor
                                                                    name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:inputTensorsArray[0]
                                                                    name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:gradOutputTensor
                                                                    name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph concatTensor:firstHalfOutputTensor
                                                 withTensor:secondHalfOutputTensor
                                                  dimension:wrap_dim
                                                       name:nil];
      newCachedGraph->gradInputTensor_ = outputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
    });

    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
  return grad_input;
}

Tensor glu_backward_mps(const Tensor& grad_output, const Tensor& self, const int64_t dim) {
  Tensor grad_input = at::empty(self.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  grad_input = glu_backward_mps_out(grad_output, self, dim, grad_input);
  return grad_input;
}

TORCH_IMPL_FUNC(softplus_out_mps)
(const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& result) {
  if (this->numel() == 0)
    return;
  using namespace mps;
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, this->common_dtype(), "softplus_mps", [&]() {
    SoftplusParams<scalar_t> params{beta.to<scalar_t>(), threshold.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        *this, "softplus", params,
        fmt::format("SoftplusParams_{}", scalarToMetalTypeString(this->common_dtype())));
  });
}

TORCH_IMPL_FUNC(softplus_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& grad_input) {
  if (this->numel() == 0)
    return;
  using namespace mps;
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, this->common_dtype(), "softplus_backward_mps", [&]() {
    SoftplusParams<scalar_t> params{beta.to<scalar_t>(), threshold.to<scalar_t>()};
    lib.exec_binary_kernel_with_params(
        *this, "softplus_backward", params,
        fmt::format("SoftplusParams_{}", scalarToMetalTypeString(this->common_dtype())));
  });
}

TORCH_IMPL_FUNC(mish_out_mps)
(const Tensor& self, const Tensor& result) {
  if (this->numel() == 0)
    return;
  lib.exec_unary_kernel(*this, "mish");
}

Tensor mish_backward_mps(const Tensor& grad_output, const Tensor& self) {
  Tensor grad_input = at::empty_like(self, self.suggest_memory_format());
  if (grad_input.numel() == 0)
    return grad_input;
  auto iter = at::TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(grad_output)
      .add_const_input(self)
      .build();
  lib.exec_binary_kernel(iter, "mish_backward");
  return grad_input;
}

Tensor prelu_mps(const Tensor& self, const Tensor& weight_) {
  using namespace mps;

  Tensor result = at::empty_like(self, self.suggest_memory_format());
  TORCH_INTERNAL_ASSERT(weight_.defined());

  if (result.numel() == 0) {
    return result;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "prelu_mps:" + getTensorsStringKey({self, weight_});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* reluTensor = [mpsGraph reLUWithTensor:inputTensor name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                            secondaryTensor:zeroTensor
                                                                       name:nil];
      MPSGraphTensor* weightedTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                       truePredicateTensor:inputTensor
                                                      falsePredicateTensor:zeroTensor
                                                                      name:nil];
      weightedTensor = [mpsGraph multiplicationWithPrimaryTensor:weightedTensor secondaryTensor:weightTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:reluTensor
                                                         secondaryTensor:weightedTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, weightPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return result;
}

std::tuple<Tensor, Tensor> prelu_backward_mps(const Tensor& grad_output, const Tensor& self, const Tensor& weight_) {
  using namespace mps;

  Tensor grad_input = at::empty_like(self, self.suggest_memory_format());
  Tensor weight_grad = at::empty_like(self, at::MemoryFormat::Contiguous);
  if (grad_output.numel() == 0) {
    return std::tuple<Tensor, Tensor>{grad_input, weight_grad};
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* weightedGradTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "prelu_backward_mps:" + getTensorsStringKey({grad_output, self, weight_});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* weightedGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:weightTensor
                                                                           secondaryTensor:gradOutputTensor
                                                                                      name:nil];
      MPSGraphTensor* inputGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                        secondaryTensor:gradOutputTensor
                                                                                   name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:zeroTensor
                                                                          name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:gradOutputTensor
                                                    falsePredicateTensor:weightedGradOutputTensor
                                                                    name:nil];
      MPSGraphTensor* weightedGradTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                           truePredicateTensor:zeroTensor
                                                          falsePredicateTensor:inputGradOutputTensor
                                                                          name:nil];
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->weightedGradTensor_ = weightedGradTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->outputTensor_, grad_input);
    Placeholder weightedGradPlaceholder = Placeholder(cachedGraph->weightedGradTensor_, weight_grad);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder, weightPlaceholder);
    auto results = dictionaryFromPlaceholders(gradInputPlaceholder, weightedGradPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
  return std::tuple<Tensor, Tensor>{grad_input, weight_grad};
}

TORCH_IMPL_FUNC(silu_out_mps)(const Tensor& self, const Tensor& result) {
  if (this->numel() == 0)
    return;
  lib.exec_unary_kernel(*this, "silu");
}

TORCH_IMPL_FUNC(silu_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Tensor& grad_input) {
  if (this->numel() == 0)
    return;
  lib.exec_binary_kernel(*this, "silu_backward");
}

// -------------------------------------------------
// Hardtanh backward

Tensor hardtanh_backward_mps(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max) {
  Tensor grad_input =
      at::empty(grad_output.sizes(), grad_output.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  grad_input = hardtanh_backward_out_mps(grad_output, self, min, max, grad_input);
  return grad_input;
}

// Hardtanh backward
Tensor& hardtanh_backward_out_mps(const Tensor& grad_output,
                                  const Tensor& self,
                                  const Scalar& min,
                                  const Scalar& max,
                                  Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(grad_output.is_mps());

  // Empty output
  if (grad_input.numel() == 0)
    return grad_input;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "hardtanh_backward_out_mps:" + getTensorsStringKey({grad_output}) + ":" +
        std::to_string(min.to<double>()) + ":" + std::to_string(max.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // TODO: Compute gradient
      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* minTensor = [mpsGraph constantWithScalar:min.to<double>()
                                                         shape:@[ @1 ]
                                                      dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* maxTensor = [mpsGraph constantWithScalar:max.to<double>()
                                                         shape:@[ @1 ]
                                                      dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* greaterThanMaxPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                             secondaryTensor:maxTensor
                                                                                        name:nil];
      MPSGraphTensor* lesserThanMinPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                         secondaryTensor:minTensor
                                                                                    name:nil];
      MPSGraphTensor* greaterThanMaxGradTensor = [mpsGraph selectWithPredicateTensor:greaterThanMaxPredicateTensor
                                                                 truePredicateTensor:zeroTensor
                                                                falsePredicateTensor:unitTensor
                                                                                name:nil];
      MPSGraphTensor* lesserThanMinGradTensor = [mpsGraph selectWithPredicateTensor:lesserThanMinPredicateTensor
                                                                truePredicateTensor:zeroTensor
                                                               falsePredicateTensor:unitTensor
                                                                               name:nil];
      MPSGraphTensor* gradTensor = [mpsGraph multiplicationWithPrimaryTensor:greaterThanMaxGradTensor
                                                             secondaryTensor:lesserThanMinGradTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    auto results = dictionaryFromPlaceholders(gradInputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return grad_input;
}

} // namespace at::native
