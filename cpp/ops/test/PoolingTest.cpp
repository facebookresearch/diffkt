/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "Dnnl/Pooling.h"
#include "TestUtils.h"

using namespace ops;

TEST(MaxPoolTest, DoesSingleImageSingleChannelMaxPool) {
  int32_t img_size = 4;
  int32_t pool_size = 2;
  int32_t res_size = 2;

  std::vector<float> img = {2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2};
  std::vector<float> res;
  std::vector<uint8_t> workspace;
  append_zeros(res, res_size * res_size);
  append_zeros(workspace, res_size * res_size);

  max_pool({1, res_size, res_size, 1}, {1, img_size, img_size, 1}, res.data(),
           workspace.data(), img.data(), pool_size, pool_size);

  std::vector<float> expected = {2, 2, 2, 2};
  std::vector<uint8_t> expected_workspace = {0, 1, 2, 3};
  EXPECT_EQ(res, expected);
  EXPECT_EQ(workspace, expected_workspace);
}

TEST(MaxPoolTest, DoesSingleImageSingleChannelMaxPoolGrad) {
  int32_t img_grad_size = 4;
  int32_t pool_size = 2;
  int32_t seed_size = 2;

  std::vector<uint8_t> workspace = {0, 1, 2, 3};
  std::vector<float> img_grad;
  append_zeros(img_grad, img_grad_size * img_grad_size);
  std::vector<float> seed;
  // Non uniform seed to help verify that the indices are correct
  append_incrementing(seed, seed_size * seed_size);

  max_pool_grad({1, img_grad_size, img_grad_size, 1},
                {1, seed_size, seed_size, 1}, img_grad.data(), workspace.data(),
                seed.data(), pool_size, pool_size);

  std::vector<float> expected = {1, 0, 0, 2, 0, 0, 0, 0,
                                 0, 0, 0, 0, 3, 0, 0, 4};

  EXPECT_EQ(img_grad, expected);
}

TEST(AvgPoolTest, DoesSingleImageSingleChannelAvgPool) {
  int32_t img_size = 4;
  int32_t pool_size = 2;
  int32_t res_size = 2;

  std::vector<float> img;
  std::vector<float> res;
  append_zeros(res, res_size * res_size);
  append_incrementing(img, img_size * img_size);

  avg_pool({1, res_size, res_size, 1}, {1, img_size, img_size, 1}, res.data(),
           img.data(), pool_size, pool_size);

  std::vector<float> expected = {3.5, 5.5, 11.5, 13.5};
  EXPECT_EQ(res, expected);
}

TEST(AvgPoolTest, DoesSingleImageSingleChannelAvgPoolGrad) {
  int32_t img_grad_size = 4;
  int32_t pool_size = 2;
  int32_t seed_size = 2;

  std::vector<float> img_grad;
  append_zeros(img_grad, img_grad_size * img_grad_size);
  std::vector<float> seed;
  // Non uniform seed to help verify that the indices are correct
  append_incrementing(seed, seed_size * seed_size);

  avg_pool_grad({1, img_grad_size, img_grad_size, 1},
                {1, seed_size, seed_size, 1}, img_grad.data(), seed.data(),
                pool_size, pool_size);

  std::vector<float> expected = {0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5,
                                 0.75, 0.75, 1,   1,   0.75, 0.75, 1,   1};

  EXPECT_EQ(img_grad, expected);
}
