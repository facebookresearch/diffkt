/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_SPARSEARITHMETICUTILS_H_
#define OPS_SPARSEARITHMETICUTILS_H_

#ifndef EIGEN // both MKL and OMP would use these functions

#include "SpMat.h"
#include "COO.h"

namespace ops {
  typedef DataType OP(DataType, DataType);

  DataType times(DataType x, DataType y);
  DataType add(DataType x, DataType y);
  DataType sub(DataType x, DataType y);

  /**
   * In parallel, for each row, do a set intersection on the non-zeros, for the row on the
   * left and the right.
   *
   * It's currently used to implement `times` operation between sparse
   * matrices. */
  SpMat rowIntersection(SpMatMap & left, SpMatMap & right, OP * op);
  /**
   * In parallel, for each row, do a set union on the non-zeros, for the row on the
   * left and the right.
   *
   * It's currently used to implement `sub` and `add` operations between sparse
   * matrices. */
  SpMat rowUnion(SpMatMap & left, SpMatMap & right, OP * op);

  /** This function is used to check whether the non-zeros in a COO are
   * sorted, and it's used for optimizing the performance for coo to csr
   * conversion */
  bool sorted(COO & coo);
}

#endif // not defined EIGEN

#endif // OPS_SPARSEARITHMETICUTILS_H_
