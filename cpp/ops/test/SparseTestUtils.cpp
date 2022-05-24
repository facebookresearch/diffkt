/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SparseTestUtils.h"

namespace ops {
  std::vector<DataType> toDenseData(SpMat & sm) {
    std::vector<DataType> dm(sm.rows()*sm.cols(), 0);

    const OrdinalType * rowsStart, * rowsEnd;
    rowPointers(sm, rowsStart, rowsEnd);
    for (DimensionType i=0; i<sm.rows(); i++) {
      for (OrdinalType j=rowsStart[i]; j<rowsEnd[i]; j++) {
        DimensionType r = i, c = sm.innerIndexPtr()[j];
        dm[i*sm.cols() + c] = sm.valuePtr()[j];
      }
    }

    return dm;
  }

  void compareCSR(SpMatMap & sm, DimensionType r, DimensionType c, std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner,
      std::vector<DataType> & value, bool colIdOrderedTheSame, DataType epsilon)
  {
    EXPECT_EQ(sm.rows(), r);
    EXPECT_EQ(sm.cols(), c);
    EXPECT_EQ(sm.nonZeros(), outer[r]);

    // quit here if size check fails, to avoid reading below causes seg-fault
    if (sm.rows() != r || sm.cols() != c || sm.nonZeros() != outer[r])
      return;

    const OrdinalType * rowsStart, * rowsEnd;
    rowPointers(sm, rowsStart, rowsEnd);
    // then compare elements in each row one by one
    for (DimensionType i=0; i<r; i++) {
      EXPECT_EQ(rowsEnd[i] - rowsStart[i], outer[i+1] - outer[i]);

      if (colIdOrderedTheSame) { // expect the ordering of column indices to be the same,
        for (OrdinalType j=rowsStart[i], k=outer[i]; k<outer[i+1]; j++, k++) {
          EXPECT_EQ(sm.innerIndexPtr()[j], inner[k]);
          EXPECT_NEAR(sm.valuePtr()[j], value[k], epsilon);
        }
      } else { // sort the column indices then compare
        // create pairs of inner-value for the expected and the results
        std::vector<std::pair<DimensionType, DataType>> exp;
        std::vector<std::pair<DimensionType, DataType>> res;
        for (OrdinalType j=rowsStart[i], k=outer[i]; k<outer[i+1]; j++, k++) {
          exp.emplace_back(inner[k], value[k]);
          res.emplace_back(sm.innerIndexPtr()[j], sm.valuePtr()[j]);
        }
        // sort the pairs
        std::sort(exp.begin(), exp.end());
        std::sort(res.begin(), res.end());
        // compare the sorted results
        for (OrdinalType j=0; j<outer[i+1] - outer[i]; j++) {
          EXPECT_EQ(exp[j].first, res[j].first);
          EXPECT_NEAR(exp[j].second, res[j].second, epsilon);
        }
      }
    }
  }
  #ifdef EIGEN
  void compareCSR(SpMat & sm, DimensionType r, DimensionType c, std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner,
      std::vector<DataType> & value, bool colIdOrderedTheSame, DataType epsilon) {
    SpMatMap m(sm.rows(), sm.cols(), sm.nonZeros(), sm.outerIndexPtr(), sm.innerIndexPtr(), sm.valuePtr());
    compareCSR(m, r, c, outer, inner, value, colIdOrderedTheSame, epsilon);
  }
  #endif // EIGEN

  void compareSparseFloatTensor(SparseFloatTensor & x, SparseFloatTensor & y)
  {
    EXPECT_EQ(x.shape().vector(), y.shape().vector());
    EXPECT_FLOATS_NEARLY_EQ(x.values().vector(), y.values().vector(), 1e-6);
    EXPECT_EQ(x.dims().size(), y.dims().size());
    for (size_t i=0; i<x.dims().size(); i++) {
      EXPECT_EQ(x.dims()[i].outer().vector(), y.dims()[i].outer().vector());
      EXPECT_EQ(x.dims()[i].inner().vector(), y.dims()[i].inner().vector());
    }
  }

  SpMat genSpMat(std::vector<DimensionType> & shape,
      std::vector<OrdinalType> & outer, std::vector<DimensionType> & inner, std::vector<DataType> & values)
  {
    EXPECT_EQ(shape.size(), 2);
    EXPECT_GT(shape[0], 0);
    EXPECT_GT(shape[1], 0);
    EXPECT_EQ(inner.size(), values.size());
    EXPECT_EQ(inner.size(), outer.back());

    #ifdef EIGEN
    // generate eigen sparse matrix by creating the triplet-list
    typedef Eigen::Triplet<DataType> T;
    std::vector<T> tripletList;
    tripletList.reserve(values.size());
    for(size_t i=0; i<outer.size()-1; i++) {
      EXPECT_LE(outer[i], outer[i+1]);
      for (size_t j=outer[i]; j<outer[i+1]; j++) {
        tripletList.emplace_back(T(i,inner[j],values[j]));
      }
    }
    SpMat m(shape[0], shape[1]);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    return m;
    #else
    // generate sparse matrix with given CSR data
    Array<OrdinalType> outer_ (outer);
    Array<DimensionType> inner_(inner);
    Array<DataType> values_(values);
    SpMat mm(shape[0], shape[1], outer_, inner_, values_);
    return mm;
    #endif
  }
}
