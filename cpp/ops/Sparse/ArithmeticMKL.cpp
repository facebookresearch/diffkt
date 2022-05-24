/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef MKL

#include "Arithmetic.h"
#include "ArithmeticUtils.h"
#include "DebugUtils.h"

namespace ops {

  SpMat add(SpMatMap & left, SpMatMap & right) {
    sparse_matrix_t res;
    sparse_status_t status;
    // _s_ and _d_ in the function name means the values type are
    // single and double precision float respectively
    if (std::is_same<DataType, float>::value)
      status = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, left.get(), 1, right.get(), &res);
    else {
      Require((std::is_same<DataType, double>::value), "add operation only supports the data type to be float or double");
      status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, left.get(), 1, right.get(), &res);
    }
    Require(status == SPARSE_STATUS_SUCCESS, "Failed to compute sparse add");
    return SpMat(res);
  }

  /** NOTE: MKL doesn't support sparse times operation, thus we use our own parallel
   * version */
  SpMat times(SpMatMap & left, SpMatMap & right) {
    return rowIntersection(left, right, times);
  }

  SpMat sub(SpMatMap & left, SpMatMap & right) {
    sparse_matrix_t res;
    sparse_status_t status;
    // _s_ and _d_ in the function name means the values type are
    // single and double precision float respectively
    if (std::is_same<DataType, float>::value)
      status = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, right.get(), -1, left.get(), &res);
    else {
      Require((std::is_same<DataType, double>::value), "sub operation only supports the data type to be float or double");
      status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, right.get(), -1, left.get(), &res);
    }
    Require(status == SPARSE_STATUS_SUCCESS, "Failed to compute sparse sub");
    return SpMat(res);
  }

  SpMat matmul(SpMatMap & left, SpMatMap & right) {
    sparse_matrix_t res;
    sparse_status_t status;
    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, left.get(), right.get(), &res);
    Require(status == SPARSE_STATUS_SUCCESS, "Failed to compute sparse matmul");
    return SpMat(res);
  }

  SpMat cooTocsr(COO & coo) {
    if (coo.nonZeros() == 0) {
      return SpMat(coo.rows(), coo.cols());
    } else {
      // FIXME: valgrind reports memory lost, need to fix it later
      sparse_matrix_t csr;
      sparse_matrix_t mklcoo;
      sparse_status_t status;
      if (std::is_same<DataType, float>::value)
        status = mkl_sparse_s_create_coo(&mklcoo, SPARSE_INDEX_BASE_ZERO, coo.rows(), coo.cols(), coo.nonZeros(),
            coo.row_index().data(), coo.col_index().data(), (float *)coo.values().data());
      else {
        Require((std::is_same<DataType, double>::value), "creating coo operation only supports the data type to be float or double");
        status = mkl_sparse_d_create_coo(&mklcoo, SPARSE_INDEX_BASE_ZERO, coo.rows(), coo.cols(), coo.nonZeros(),
            coo.row_index().data(), coo.col_index().data(), (double *)coo.values().data());
      }
      Require(status == SPARSE_STATUS_SUCCESS, "Failed to create a COO");
      status = mkl_sparse_convert_csr(mklcoo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
      Require(status == SPARSE_STATUS_SUCCESS, "Failed to generate a CSR from COO");
      status = mkl_sparse_destroy(mklcoo);
      Require(status == SPARSE_STATUS_SUCCESS, "Failed to destroy sparse_matrix_t after converting from COO to CSR");

      return SpMat(csr);
    }
  }

  SpMat transpose(SpMatMap & tensor) {
    sparse_matrix_t res;
    sparse_status_t status;
    // NOTE: MKL fails when tensor is empty
    if (tensor.nonZeros() == 0) {
      return SpMat(tensor.cols(), tensor.rows());
    } else {
      status = mkl_sparse_convert_csr(tensor.get(), SPARSE_OPERATION_TRANSPOSE, &res);
      Require(status == SPARSE_STATUS_SUCCESS, "Failed to compute transpose");
      return SpMat(res);
    }
  }

} // namespace ops

#endif // MKL
