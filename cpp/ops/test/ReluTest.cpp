/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "Dnnl/Relu.h"
#include "Dnnl/Utils.h"
#include "TestUtils.h"

using namespace ops;

TEST(ReluTest, DoesRelu) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<float> data;
  std::vector<float> res;
  append_random(data, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (auto idata : data) {
    expected.push_back(std::max(idata, 0.0f));
  }

  relu(shape, res.data(), data.data());
  EXPECT_EQ(res, expected);
}

TEST(ReluTest, DoesReluGrad) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<float> seed;
  std::vector<float> data;
  std::vector<float> res;
  append_random(seed, product(shape));
  append_random(data, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = 0; i < product(shape); i++) {
    expected.push_back(data[i] > 0.0f ? seed[i] : 0.0f);
  }

  relu_grad(shape, res.data(), seed.data(), data.data());
  EXPECT_EQ(res, expected);
}
