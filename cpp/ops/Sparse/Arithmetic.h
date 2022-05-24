/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_SPARSEARITHMETIC_H_
#define OPS_SPARSEARITHMETIC_H_

#include "COO.h"
#include "SpMat.h"

#ifdef EIGEN

namespace ops {
  Eigen::SparseMatrix<DataType> inverse(Eigen::SparseMatrix<DataType> A);
  SpMat matdiv(SpMatMap & left, SpMatMap & right);
} // namespace ops

#endif // EIGEN

namespace ops {
  SpMat add(SpMatMap & left, SpMatMap & right);
  SpMat times(SpMatMap & left, SpMatMap & right);
  SpMat sub(SpMatMap & left, SpMatMap & right);
  SpMat matmul(SpMatMap & left, SpMatMap & right);
  SpMat cooTocsr(COO & coo);
  SpMat transpose(SpMatMap & tensor);
} // namespace ops


#endif // OPS_SPARSEARITHMETIC_H_
