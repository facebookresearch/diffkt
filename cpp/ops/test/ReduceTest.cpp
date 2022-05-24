/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"
#include "dnnl.hpp"

#include "Dnnl/Utils.h"
#include "Dnnl/Reduce.h"
#include "TestUtils.h"

using namespace dnnl;
using namespace ops;

TEST(ReduceTest, Sum) {
  std::vector<int32_t> src_dims = {2, 3, 2};
  std::vector<int32_t> dst_dims = {1, 3, 1};
  auto src = std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto dst = std::vector<float>{0, 0, 0};

  reduce_sum(dst_dims, dst.data(), src_dims, src.data());

  std::vector<float> expected = {18, 26, 34};
  vector_expect_near(dst, expected);
}
