/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_COO_H_
#define OPS_COO_H_

#include "MemUtils.h"

namespace ops {

  /** a class holding data for a sparse matrix with COO format */
  class COO {
    public:
      /** Empty constructor */
      COO() : rows_(0), cols_(0), row_index_(), col_index_(), values_() {}
      /** Construct from data and move the data to the object */
      COO(DimensionType rows, DimensionType cols, Array<DimensionType> & row_index,
          Array<DimensionType> & col_index, Array<DataType> & values) :
        rows_(rows), cols_(cols), row_index_(std::move(row_index)),
        col_index_(std::move(col_index)), values_(std::move(values))
      {
         Require(row_index_.size() == col_index_.size(), "the number of elements in row data, should be the same as in col data");
         Require(row_index_.size() == values_.size(), "the number of elements in row data, should be the same as in value data");
      }

      /** copy constructor */
      COO(const COO& other) = delete;
      /** copy assignment */
      COO& operator=(const COO& other) = delete;

      /** move constructor */
      COO(COO&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), row_index_(std::move(other.row_index_)),
        col_index_(std::move(other.col_index_)), values_(std::move(other.values_)) {
          other.rows_ = 0;
          other.cols_ = 0;
      }

      /** move assignment */
      COO& operator=(COO&& other) noexcept
      {
        row_index_ = std::move(other.row_index_);
        col_index_ = std::move(other.col_index_);
        values_ = std::move(other.values_);

        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = 0;
        other.cols_ = 0;

        return *this;
      }

      /** get rows_ */
      DimensionType rows() { return rows_; }
      /** get cols_ */
      DimensionType cols() { return cols_; }
      /** get the number of non-zeros */
      OrdinalType nonZeros() { return row_index_.size(); }
      /** get row_index_ */
      Array<DimensionType> & row_index() { return row_index_; }
      /** get col_index_ */
      Array<DimensionType> & col_index() { return col_index_; }
      /** get values_ */
      Array<DataType> & values() { return values_; }

    private:
      DimensionType rows_, cols_;
      Array<DimensionType> row_index_, col_index_;
      Array<DataType> values_;
  };

} // namespace ops
#endif // OPS_COO_H_
