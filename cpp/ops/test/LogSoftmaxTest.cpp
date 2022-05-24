/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "Dnnl/LogSoftmax.h"
#include "Dnnl/Utils.h"
#include "TestUtils.h"

using namespace ops;

TEST(LogSoftmaxTest, ForwardSanityAxis1) {
  std::vector<int32_t> shape = {2, 3};
  std::vector<float> input;
  std::vector<float> res;
  append_incrementing(input, product(shape));
  append_zeros(res, product(shape));

  log_softmax(shape, input.data(), res.data(), 1);

  std::vector<float> expected = {-2.40761, -1.40761, -0.40761,
                                 -2.40761, -1.40761, -0.40761};
  vector_expect_near(res, expected, 1e-5);
}

TEST(LogSoftmaxTest, BackwardSanityAxis1) {
  std::vector<int32_t> shape = {2, 3};
  std::vector<float> input;
  std::vector<float> fwd_res;
  std::vector<float> seed;
  std::vector<float> grad;
  append_ones(seed, product(shape));
  append_incrementing(input, product(shape));
  append_zeros(grad, product(shape));
  append_zeros(fwd_res, product(shape));

  int32_t axis = 1;
  log_softmax(shape, input.data(), fwd_res.data(), axis);
  log_softmax_grad(shape, grad.data(), seed.data(), fwd_res.data(), axis);

  std::vector<float> expected = {0.729908, 0.265814, -0.995723,
                                 0.729908, 0.265814, -0.995723};
  vector_expect_near(grad, expected, 1e-5);
}
