/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef MKL

#include "Sparse/MKLCSR.h"

namespace ops {

  MKLCSRMap::MKLCSRMap(DimensionType r, DimensionType c, OrdinalType nnz,
      OrdinalType * outer, DimensionType * inner, DataType * values) {
    if (r <= 0 || c <= 0 || outer == NULL) {
      valid_ = false;
    } else if (nnz == 0) {
      gencsr(r, c, outer, outer + 1);
    } else {
      gencsr(r, c, outer, outer + 1, inner, values);
    }
  }

  MKLCSRMap::~MKLCSRMap() {
    if (valid_)
      status_ = mkl_sparse_destroy(csr_);
  }

  void MKLCSRMap::exportdata() {
    Require(valid_, "MKL's data export function requires valid mtx to success.");
    sparse_index_base_t indexing_; // zero-indexed or one-indexed
    if (std::is_same<DataType, float>::value)
      status_ = mkl_sparse_s_export_csr(csr_, &indexing_, &rows_, &cols_, &rows_start_, &rows_end_, &col_index_, (float **)&values_);
    else {
      Require((std::is_same<DataType, double>::value), "csr data export operation only supports the data type to be float or double");
      status_ = mkl_sparse_d_export_csr(csr_, &indexing_, &rows_, &cols_, &rows_start_, &rows_end_, &col_index_, (double **)&values_);
    }
    Require(status_ == SPARSE_STATUS_SUCCESS, "Failed to export data from MKL sparse matrix");

    // Update the indices to zeor-based if not
    if (indexing_ != SPARSE_INDEX_BASE_ZERO) {
      if (rows_start_ + 1 == rows_end_) {
        for(size_t i=0; i<=rows_; i++) rows_start_[i]--;
      } else {
        for(size_t i=0; i<rows_; i++) {
          rows_start_[i]--;
          rows_end_[i]--;
        }
      }
      for(size_t i=0; i<rows_end_[rows_ - 1]; i++) col_index_[i]--;
    }
    nnz_ = rows_end_[rows_-1];
  }

  void MKLCSRMap::gencsr(DimensionType r, DimensionType c, OrdinalType * rows_start, OrdinalType * rows_end,
      DimensionType * col_index, DataType * values) {
    if (std::is_same<DataType, float>::value)
      status_ = mkl_sparse_s_create_csr(&csr_, SPARSE_INDEX_BASE_ZERO,
          r, c, rows_start, rows_end, col_index, (float*)values);
    else {
      Require((std::is_same<DataType, double>::value), "csr creation operation only supports the data type to be float or double");
      status_ = mkl_sparse_d_create_csr(&csr_, SPARSE_INDEX_BASE_ZERO,
          r, c, rows_start, rows_end, col_index, (double*)values);
    }
    Require(status_ == SPARSE_STATUS_SUCCESS, "Failed to create a MKL sparse matrix");
    valid_ = true;

    exportdata();
  }

  void MKLCSRMap::gencsr(DimensionType r, DimensionType c, OrdinalType * rows_start, OrdinalType * rows_end) {
    // NOTE: having the last two parameters as rows_start, is a patch as MKL doesn't support inner or values to be
    // NULL even when creating an empty matrix.
    gencsr(r, c, rows_start, rows_end, (DimensionType *)rows_start, (DataType *)rows_start);
  }

  MKLCSR::MKLCSR(DimensionType r, DimensionType c) {
    if (r <= 0 || c <= 0)
      valid_ = false;
    else {
      outer_data_.resize(r + 1);
      outer_data_.assign(0);
      gencsr(r, c, outer_data_.data(), outer_data_.data() + 1);
    }
  }

  MKLCSR::MKLCSR(DimensionType r, DimensionType c, Array<OrdinalType> & outer,
      Array<DimensionType> & inner, Array<DataType> & values)
    : outer_data_(std::move(outer)), inner_data_(std::move(inner)), values_data_(std::move(values)) {
      if (r <= 0 || c <= 0 || outer_data_.size() == 0)
        valid_ = false;
      else {
        Require(outer_data_.size() == r + 1, "the size of outer array should be : the number of rows + 1");
        Require(inner_data_.size() == values_data_.size(), "the size of inner and value array should be the same");
        Require(outer_data_[0] == 0, "the first element of outer array should be zero");
        Require(outer_data_[r] == inner_data_.size(), "the last element of outer array should be the number of nonzeros");
        #ifdef DEBUG
        for (size_t i=0; i<r; i++)
          Require(outer_data_[i] <= outer_data_[i+1], "the outer array should be in the ascending order.");
        for (size_t i=0; i<inner_data_.size(); i++)
          Require(inner_data_[i] < c, "elements in the inner array should be less than the number of columns.");
        #endif

        if (inner_data_.size() == 0)
          gencsr(r, c, outer_data_.data(), outer_data_.data() + 1);
        else
          gencsr(r, c, outer_data_.data(), outer_data_.data() + 1,
              inner_data_.data(), values_data_.data());
      }
    }
} // namespace ops

#endif // MKL
