/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"
#include <iostream>

#include "Dnnl/ArithmeticDnnl.h"
#include "Dnnl/Utils.h"
#include "TestUtils.h"

using namespace ops;

TEST(AddTest, DoesAdd) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<int32_t> contig_strides = {6, 2, 1};
  int32_t zero_offset = 0;
  std::vector<float> lhs;
  std::vector<float> rhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape));
  append_ones(rhs, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = 0; i < lhs.size(); i++) {
    expected.push_back(lhs[i] + rhs[i]);
  }

  add(shape, contig_strides, contig_strides, zero_offset, zero_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(AddTest, DoesOffsetAdd) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<int32_t> contig_strides = {6, 2, 1};
  int32_t lhs_offset = 2;
  int32_t rhs_offset = 3;
  std::vector<float> lhs;
  std::vector<float> rhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape) + lhs_offset);
  append_ones(rhs, product(shape) + rhs_offset);
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = lhs_offset; i < lhs.size(); i++) {
    expected.push_back(lhs[i] + rhs[i - lhs_offset + rhs_offset]);
  }

  add(shape, contig_strides, contig_strides, lhs_offset, rhs_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(AddTest, DoesStridedAdd) {
  std::vector<int32_t> shape = {2, 3, 4};
  std::vector<float> lhs;
  std::vector<int32_t> lhs_strides = {3, 1, 0}; // broadcast {2,3} along last dim
  std::vector<float> rhs;
  std::vector<int32_t> rhs_strides = {0, 4, 1}; // broadcast {3,4} along first dim
  int32_t zero_offset = 0;
  std::vector<float> res;
  append_incrementing(lhs, 2 * 3);
  append_incrementing(rhs, 3 * 4);
  append_zeros(res, product(shape));

  std::vector<float> expected = {
    2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18 };

  add(shape, lhs_strides, rhs_strides, zero_offset, zero_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(SubtractTest, DoesSubtract) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<int32_t> contig_strides = {6, 2, 1};
  int32_t zero_offset = 0;
  std::vector<float> lhs;
  std::vector<float> rhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape));
  append_ones(rhs, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = 0; i < lhs.size(); i++) {
    expected.push_back(lhs[i] - rhs[i]);
  }

  sub(shape, contig_strides, contig_strides, zero_offset, zero_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(MultiplyTest, DoesMultiplyByScalar) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = 0; i < lhs.size(); i++) {
    expected.push_back(lhs[i] * 3.0f);
  }

  mul(shape, res.data(), lhs.data(), 3.0f);
  EXPECT_EQ(res, expected);
}

TEST(LinearTest, DoesLinear) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<int32_t> strides = {6, 2, 1};
  int32_t zero_offset = 0;
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape));
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = 0; i < lhs.size(); i++) {
    expected.push_back(lhs[i] * 3.0f + 1.0f);
  }

  linear(shape, strides, zero_offset, res.data(), lhs.data(), 3.0f, 1.0f);
  EXPECT_EQ(res, expected);
}

TEST(LinearTest, DoesOffsetLinear) {
  std::vector<int32_t> shape = {2, 3, 2};
  std::vector<int32_t> strides = {6, 2, 1};
  int32_t offset = 3;
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(shape) + offset);
  append_zeros(res, product(shape));

  std::vector<float> expected;
  for (size_t i = offset; i < lhs.size(); i++) {
    expected.push_back(lhs[i] * 3.0f + 1.0f);
  }

  linear(shape, strides, offset, res.data(), lhs.data(), 3.0f, 1.0f);
  EXPECT_EQ(res, expected);
}

TEST(MatmulTest, 4dMatmul) {
  std::vector<int32_t> lshape = {1, 2, 3, 4};
  std::vector<int32_t> lstrides = {24, 12, 4, 1};
  std::vector<int32_t> rshape = {1, 2, 4, 5};
  // Strides for a contig tensor of shape 1,4,2,5 transposed to 1,2,4,5
  std::vector<int32_t> rstrides = {40, 5, 10, 1};
  int32_t zero_offset = 0;
  std::vector<float> rhs;
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(lshape));
  append_incrementing(rhs, product(rshape));
  append_zeros(res, 2*3*5);

  std::vector<float> expected = {
    210.0, 220.0, 230.0, 240.0, 250.0, 466.0, 492.0, 518.0, 544.0, 570.0,
    722.0, 764.0, 806.0, 848.0, 890.0, 1268.0, 1326.0, 1384.0, 1442.0,
    1500.0, 1604.0, 1678.0, 1752.0, 1826.0, 1900.0, 1940.0, 2030.0, 2120.0,
    2210.0, 2300.0
  };

  mmul(lshape, lstrides, zero_offset, rshape, rstrides, zero_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(MatmulTest, ContigMatmulTransposed) {
  std::vector<int32_t> lshape = {2, 3};
  std::vector<int32_t> lstrides = {3, 1};
  std::vector<int32_t> rshape = {3, 4};
  std::vector<int32_t> rstrides = {1, 3};
  int32_t zero_offset = 0;
  std::vector<float> rhs;
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(lshape));
  append_incrementing(rhs, product(rshape));
  append_zeros(res, 2*4);

  std::vector<float> expected = { 14.0, 32.0, 50.0, 68.0, 32.0, 77.0, 122.0, 167.0 };

  // TODO: manually transpose lshape or just update expected when we have confidence
  mmul(lshape, lstrides, zero_offset, rshape, rstrides, zero_offset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);
}

TEST(MatmulTest, OffsetMatmul) {
  std::vector<int32_t> lshape = {2, 3};
  std::vector<int32_t> lstrides = {3, 1};
  int32_t loffset = 2;
  std::vector<int32_t> rshape = {3, 4};
  std::vector<int32_t> rstrides = {4, 1};
  int32_t roffset = 3;
  std::vector<float> rhs;
  std::vector<float> lhs;
  std::vector<float> res;
  append_incrementing(lhs, product(lshape) + loffset);
  append_incrementing(rhs, product(rshape) + roffset);
  append_zeros(res, 2*4);

  std::vector<float> expected = { 104.0, 116.0, 128.0, 140.0, 176.0, 197.0, 218.0, 239.0 };

  mmul(lshape, lstrides, loffset, rshape, rstrides, roffset, res.data(), lhs.data(), rhs.data());
  EXPECT_EQ(res, expected);


}
