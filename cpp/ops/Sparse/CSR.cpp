/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(MKL) and !defined(EIGEN)

#include "Sparse/CSR.h"

namespace ops {

  CSRMap::CSRMap(DimensionType r, DimensionType c, OrdinalType nnz,
      OrdinalType * outer, DimensionType * inner, DataType * values) {
    if (r <= 0 || c <= 0 || outer == NULL) {
      valid_ = false;
    } else if (nnz == 0) {
      assign(r, c, nnz, outer, outer + 1);
    } else {
      assign(r, c, nnz, outer, outer + 1, inner, values);
    }
  }

  void CSRMap::assign(DimensionType r, DimensionType c, OrdinalType nnz, OrdinalType * rows_start, OrdinalType * rows_end,
      DimensionType * col_index, DataType * values) {
    rows_ = r;
    cols_ = c;
    nnz_ = nnz;
    rows_start_ = rows_start;
    rows_end_ = rows_end;
    col_index_ = col_index;
    values_ = values;
    valid_ = true;
  }

  void CSRMap::assign(DimensionType r, DimensionType c, OrdinalType nnz, OrdinalType * rows_start, OrdinalType * rows_end) {
    assign(r, c, nnz, rows_start, rows_end, NULL, NULL);
  }

  void CSRMap::assign(CSRMap& other) {
    assign(other.rows_, other.cols_, other.nnz_, other.rows_start_, other.rows_end_, other.col_index_, other.values_);
  }

  CSR::CSR(DimensionType r, DimensionType c) {
    if (r <= 0 || c <= 0)
      valid_ = false;
    else {
      outer_data_.resize(r + 1);
      outer_data_.assign(0);
      assign(r, c, 0, outer_data_.data(), outer_data_.data() + 1);
    }
  }

  CSR::CSR(DimensionType r, DimensionType c, Array<OrdinalType> & outer,
      Array<DimensionType> & inner, Array<DataType> & values)
    : outer_data_(std::move(outer)), inner_data_(std::move(inner)), values_data_(std::move(values)) {
      if (r <= 0 || c <= 0 || outer_data_.size() == 0) {
        assign(0, 0, 0, NULL, NULL, NULL, NULL);
        valid_ = false;
      } else {
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
          assign(r, c, 0, outer_data_.data(), outer_data_.data() + 1);
        else
          assign(r, c, inner_data_.size(), outer_data_.data(), outer_data_.data() + 1,
              inner_data_.data(), values_data_.data());
      }
    }
} // namespace ops

#endif // not defined MKL and not defined EIGEN
