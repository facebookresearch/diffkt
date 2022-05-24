/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "Dnnl/Conv.h"
#include "TestUtils.h"

using namespace ops;

TEST(ConvTest, DoesSingleImageSingleChannelConv) {
  int32_t res_size = 5;
  int32_t img_size = 5;
  int32_t wei_size = 3;

  std::vector<float> res;
  std::vector<float> img;
  std::vector<float> wei;

  append_zeros(res, res_size * res_size);
  append_incrementing(wei, wei_size * wei_size);
  append_incrementing(img, img_size * img_size);

  conv({1, res_size, res_size, 1}, // res shape; NHWC
       {1, img_size, img_size, 1}, // image shape; NHWC
       {1, wei_size, wei_size, 1}, // weights (filter) shape; OHWI
       res.data(), img.data(), wei.data(),
       1, // hstride
       1, // wstride
       Padding{1, 1, 1, 1});

  std::vector<float> expected = {128, 202, 241, 280, 184, 276, 411, 456, 501,
                                 318, 441, 636, 681, 726, 453, 606, 861, 906,
                                 951, 588, 320, 436, 457, 478, 280};

  EXPECT_EQ(res, expected);
}

TEST(ConvGradTest, DoesSingleImageSingleChannelGradImage) {
  int32_t img_grad_size = 5;
  int32_t seed_size = 5;
  int32_t weights_size = 3;

  std::vector<float> img_grad;
  std::vector<float> seed;
  std::vector<float> weights;

  append_zeros(img_grad, img_grad_size * img_grad_size);
  append_ones(seed, seed_size * seed_size);
  append_incrementing(weights, weights_size * weights_size);

  conv_grad_image({1, img_grad_size, img_grad_size, 1}, // img_grad shape; NHWC
                  {1, seed_size, seed_size, 1},         // seed shape; NHWC
                  {1, weights_size, weights_size, 1},   // weights shape; OHWI
                  img_grad.data(), seed.data(), weights.data(),
                  1, // hstride
                  1, // wstride
                  Padding{1, 1, 1, 1});

  std::vector<float> expected = {12, 21, 21, 21, 16, 27, 45, 45, 45,
                                 33, 27, 45, 45, 45, 33, 27, 45, 45,
                                 45, 33, 24, 39, 39, 39, 28};

  EXPECT_EQ(img_grad, expected);
}

TEST(ConvGradTest, DoesSingleImageSingleChannelGradWeights) {
  int32_t weights_grad_size = 3;
  int32_t seed_size = 5;
  int32_t img_size = 5;

  std::vector<float> weights_grad;
  std::vector<float> seed;
  std::vector<float> img;

  append_zeros(weights_grad, weights_grad_size * weights_grad_size);
  append_ones(seed, seed_size * seed_size);
  append_incrementing(img, img_size * img_size);

  conv_grad_filter(
      {1, weights_grad_size, weights_grad_size, 1}, // weights_grad shape; OHWI
      {1, seed_size, seed_size, 1},                 // seed shape; NHWC
      {1, img_size, img_size, 1},                   // img shape; NHWC
      weights_grad.data(), seed.data(), img.data(),
      1, // hstride
      1, // wstride
      Padding{1, 1, 1, 1});

  std::vector<float> expected = {160, 210, 176, 250, 325, 270, 240, 310, 256};

  EXPECT_EQ(weights_grad, expected);
}
