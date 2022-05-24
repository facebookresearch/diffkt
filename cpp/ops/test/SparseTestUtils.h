/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef TEST_SPARSETESTUTILS_H_
#define TEST_SPARSETESTUTILS_H_

#include "gtest/gtest.h" // EXPECT_?
#include <utility> // pair
#include <algorithm> // sort
#include "Sparse/MemUtils.h"
#include "Sparse/SpMat.h"
#include "Sparse/SparseFloatTensor.h"

namespace ops {
  //https://stackoverflow.com/a/48211107/15994042
#define EXPECT_FLOATS_NEARLY_EQ(expected, actual, thresh) \
  EXPECT_EQ(expected.size(), actual.size()) << "Array sizes differ.";\
  for (size_t idx = 0; idx < std::min(expected.size(), actual.size()); ++idx) \
  { \
    EXPECT_NEAR(expected[idx], actual[idx], thresh) << "at index: " << idx;\
  }

  std::vector<DataType> toDenseData(SpMat & sm);
  /** A utility function to compare SpMatMap with a given CSR with vectors*/
  void compareCSR(SpMatMap & sm, DimensionType r, DimensionType c, std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner,
      std::vector<DataType> & value, bool colIdOrderedTheSame = true, DataType epsilon = 1e-6);
#ifdef EIGEN
  void compareCSR(SpMat & sm, DimensionType r, DimensionType c, std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner,
      std::vector<DataType> & value, bool colIdOrderedTheSame = true, DataType epsilon = 1e-6);
#endif // EIGEN

  /** A utility function to compare two SparseFloatTensors */
  void compareSparseFloatTensor(SparseFloatTensor & x, SparseFloatTensor & y);

  /** A utility function to generate SpMat from shape, outer, inner, and
   * values */
  SpMat genSpMat(std::vector<DimensionType> & shape,
      std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner, std::vector<DataType> & values);
}

#endif // TEST_SPARSETESTUTILS_H_
