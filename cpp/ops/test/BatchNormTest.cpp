/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "Dnnl/BatchNorm.h"
#include "Dnnl/Utils.h"
#include "TestUtils.h"

using namespace ops;

// Runs a sanity check on a single-pixel, multi-channel image.
TEST(BatchNormTest, SanityCheck) {
  int32_t C = 3;
  std::vector<int32_t> input_shape = {1, 1, 1, C}; // NHWC
  std::vector<float> input;
  append_incrementing(input, product(input_shape));
  std::vector<int32_t> scale_shift_shape = {
      2, C}; // 2 because scale (1xC) and shift (1xC) are combined
  std::vector<float> scale_shift;
  append_random(scale_shift, product(scale_shift_shape));

  std::vector<float> res;
  append_zeros(res, product(input_shape));
  std::vector<float> mean;
  append_zeros(mean, C);
  std::vector<float> variance;
  append_zeros(variance, C);

  std::vector<float> expected_mean = {1, 2, 3};
  std::vector<float> expected_variance = {0, 0, 0};
  std::vector<float> expected;
  for (size_t i = 0; i < C; i++) {
    expected.push_back(scale_shift[i] * (input[i] - expected_mean[i]) +
                       scale_shift[i + C]);
  }

  batch_norm(input_shape, res.data(), mean.data(), variance.data(),
             input.data(), scale_shift.data());

  EXPECT_EQ(mean, expected_mean);
  EXPECT_EQ(variance, expected_variance);
  EXPECT_EQ(res, expected);
}

// Sanity check that mean and variance work. Does not test scale or shift.
TEST(BatchNormTest, MeanAndVariance) {
  int32_t C = 1;
  std::vector<int32_t> input_shape = {1, 4, 4, C}; // NHWC
  std::vector<float> input;
  append_incrementing(input, product(input_shape));
  std::vector<int32_t> scale_shift_shape = {
      2, C}; // 2 because scale (1xC) and shift (1xC) are combined
  std::vector<float> scale_shift;
  append_ones(scale_shift, C);  // no-op scale
  append_zeros(scale_shift, C); // no-op shift

  std::vector<float> res;
  append_zeros(res, product(input_shape));
  std::vector<float> mean;
  append_zeros(mean, C);
  std::vector<float> variance;
  append_zeros(variance, C);

  std::vector<float> expected_mean = {8.5};
  std::vector<float> expected_variance = {21.25};
  std::vector<float> expected;
  for (size_t i = 0; i < input.size(); i++) {
    expected.push_back((input[i] - expected_mean[0]) /
                           std::sqrt(expected_variance[0]) +
                       1.e-10f);
  }

  batch_norm(input_shape, res.data(), mean.data(), variance.data(),
             input.data(), scale_shift.data());

  EXPECT_EQ(mean, expected_mean);
  EXPECT_EQ(variance, expected_variance);
  vector_expect_near(res, expected);
}

TEST(BatchNormTest, Grad) {
  std::vector<int32_t> input_shape = {1, 2, 2, 1}; // NHWC
  std::vector<float> input = {1, 2, 3, 10};
  std::vector<int32_t> scale_shift_shape = {
      2, 1}; // 2 because scale (1xC) and shift (1xC) are combined
  std::vector<float> scale_shift = {1, 0};
  std::vector<float> seed;
  append_incrementing(seed, product(input_shape));

  // result buffers for forward
  std::vector<float> res = {0, 0, 0, 0};
  std::vector<float> mean = {0};
  std::vector<float> variance = {0};

  // result buffers for backward
  std::vector<float> input_grad = {0, 0, 0, 0};
  std::vector<float> scale_shift_grad = {0, 0};

  batch_norm(input_shape, res.data(), mean.data(), variance.data(),
             input.data(), scale_shift.data());
  batch_norm_grad(input_shape, input_grad.data(), scale_shift_grad.data(),
                  seed.data(), input.data(), scale_shift.data(), mean.data(),
                  variance.data());

  std::vector<float> expected_input_grad = {-0.1867, 0.0170, 0.2206, -0.0509};
  std::vector<float> expected_scale_shift_grad = {3.9598, 10};
  vector_expect_near(input_grad, expected_input_grad, 1e-4f);
  vector_expect_near(scale_shift_grad, expected_scale_shift_grad, 1e-5f);
}
