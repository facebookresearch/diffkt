/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_SPMAT_H_
#define OPS_SPMAT_H_

#include "MemUtils.h"

/** define SpMat and SpMatMap
 *
 * SpMatMap has pointers pointing to data that will be included in a CSR
 * matrix. SpMat contains both the pointers and the ability of controlling
 * the data, mainly freeing the data. */
#ifdef EIGEN

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<ops::DataType, Eigen::RowMajor> SpMat;
typedef Eigen::Map<SpMat> SpMatMap;

#elif defined(MKL)

#include "MKLCSR.h"

typedef ops::MKLCSR SpMat;
typedef ops::MKLCSRMap SpMatMap;

#else

#include "CSR.h"

typedef ops::CSR SpMat;
typedef ops::CSRMap SpMatMap;

#endif // EIGEN

/** define row pointer functions
 *
 * This function is used to unify the interface between MKL and the
 * others.
 * MKL supports 4-array CSR format:
 *     rows_start, rows_end, col_index, values.
 * While others, like Eigen supports 3-array CSR format:
 *     rows_ptr, col_index, values.
 * This functions takes the sparse matrix and returns the:
 *     rows_start, and rows_end.
 * So all sparse matrix is used in the 4-array way. */
namespace ops {
  /** get the arrays of starting and ending pointers for rows */
  void rowPointers(const SpMatMap & m, const OrdinalType * & startPtr, const OrdinalType * & endPtr);
  #ifdef EIGEN
  void rowPointers(const SpMat & m, const OrdinalType * & startPtr, const OrdinalType * & endPtr);
  #endif
} // namespace ops

#endif // OPS_SPMAT_H_
