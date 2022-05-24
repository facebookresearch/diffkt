/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef EIGEN

#include "Arithmetic.h"
#include "DebugUtils.h"

namespace ops {

  SpMat add(SpMatMap & left, SpMatMap & right) {
    return left + right;
  }

  SpMat times(SpMatMap & left, SpMatMap & right) {
    return left.cwiseProduct(right);
  }

  SpMat sub(SpMatMap & left, SpMatMap & right) {
    return left - right;
  }

  SpMat matmul(SpMatMap & left, SpMatMap & right) {
    return left * right;
  }

  /** Inverse is achieved by solving: A*A_inv = I, where A is the input matrix,
   * I is an identity matrix.
   *
   * Note that:
   * there are multiple solvers provide by Eigen however, some
   * of them appears not working for float type.
   * The solver SparseLU, at the time of implementation, doesn't support
   * row-major order. Thus, column-major order is used.
   *
   * Implementation reference : https://stackoverflow.com/a/25929852/15994042 */
  Eigen::SparseMatrix<DataType> inverse(Eigen::SparseMatrix<DataType> A) {
    Eigen::SparseLU<Eigen::SparseMatrix<DataType>> solver;
    solver.compute(A);
    Require(solver.info()==Eigen::Success, "Matrix Inverse: solver failed");

    Eigen::SparseMatrix<DataType> I(A.rows(), A.rows());
    I.setIdentity();

    Eigen::SparseMatrix<DataType> A_inv = solver.solve(I);
    Require(solver.info()==Eigen::Success, "Matrix Inverse: solver failed");

    return A_inv;
  }

  SpMat matdiv(SpMatMap & left, SpMatMap & right) {

    // Convert both left and right to a column-major order
    // Remove the map
    Eigen::SparseMatrix<DataType> right_colMajor(right);
    Eigen::SparseMatrix<DataType> left_colMajor(left);
    auto right_colMajor_inv = inverse(right_colMajor);

    return left_colMajor * right_colMajor_inv;
  }

  SpMat cooTocsr(COO & coo) {
    typedef Eigen::Triplet<DataType> T;
    std::vector<T> tripletList;
    size_t numEntries = coo.row_index().size();
    tripletList.reserve(numEntries);
    for(int i = 0; i < numEntries; i++)
    {
      tripletList.push_back(T(coo.row_index()[i], coo.col_index()[i], coo.values()[i]));
    }
    SpMat csr(coo.rows(), coo.cols());
    csr.setFromTriplets(tripletList.begin(), tripletList.end());
    return csr;
  }

  SpMat transpose(SpMatMap & tensor) {
    SpMat t = tensor;
    return t.transpose();
  }

} // namespace ops

#endif // EIGEN
