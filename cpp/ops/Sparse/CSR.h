/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_CSR_H_
#define OPS_CSR_H_

#if !defined(MKL) and !defined(EIGEN)

#include "MemUtils.h"

namespace ops {

  /**
   * Acts similar like Eigen's Map, it only contains pointers to the data
   * for the sparse matrix, and will not be in charge of freeing the data
   * the pointers pointing to.
   * It also shouldn't do any changes to the data, thus all data access
   * are const. */
  class CSRMap {
    public:
      /** Empty constructor */
      CSRMap() : valid_(false) { assign(0, 0, 0, NULL, NULL, NULL, NULL); }
      /** 3-array CSR variant pointers for constructor */
      CSRMap(DimensionType r, DimensionType c, OrdinalType nnz,
        OrdinalType * outer, DimensionType * inner, DataType * values);

      /** De-constructor: destroy the sparse matrix created */
      ~CSRMap() {}
      /** copy constructor */
      CSRMap(const CSRMap& other) = delete;
      /** copy assignment */
      CSRMap& operator=(const CSRMap& other) = delete;

      /** move constructor */
      CSRMap(CSRMap&& other) noexcept
      {
        assign(other);
        valid_ = other.valid_;
        other.valid_ = false;
      }

      /** move assignment */
      CSRMap& operator=(CSRMap&& other) noexcept
      {
        assign(other);
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
      }

      /** get the number of rows */
      const DimensionType rows() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return rows_; }
      /** get the number of cols */
      const DimensionType cols() const {  Require(valid_ == true, "the matrix needs to be valid to access the data"); return cols_; }
      /** get the number of non-zeros */
      const OrdinalType nonZeros() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return nnz_; }
      /** get the pointer to the array of starting pointers for
       * each row in the inner/value array */
      const OrdinalType * rowStartPtr() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return rows_start_; }
      /** get the pointer to the array of ending pointers for
       * each row in the inner/value array */
      const OrdinalType * rowEndPtr() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return rows_end_; }
      /** get the pointer to the value array */
      const DataType * valuePtr() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return values_; }
      /** get the pointer to the array of column indices */
      const DimensionType * innerIndexPtr() const { Require(valid_ == true, "the matrix needs to be valid to access the data"); return col_index_; }
      /** get the valid_ */
      const bool valid() const { return valid_; }

    protected:
      /** assign the values to the variables */
      void assign(DimensionType r, DimensionType c, OrdinalType nnz, OrdinalType * rows_start, OrdinalType * rows_end,
          DimensionType * col_index, DataType * values);
      /** generate an empty matrix */
      void assign(DimensionType r, DimensionType c, OrdinalType nnz, OrdinalType * rows_start, OrdinalType * rows_end);

      void assign(CSRMap& other);

      /** variables storing info on the sparse matrix */
      DimensionType rows_, cols_;
      OrdinalType nnz_;
      OrdinalType *rows_start_, *rows_end_;
      DimensionType *col_index_;
      DataType * values_;

      /** label whether the sparse matrix is valid or not. */
      bool valid_;
  };

  /**
   * This class is based on CSRMap, and the difference is that it
   * also contains all data for the sparse matrix and will take care
   * of the data associated.
   *
   * The data is stored in the 3 Array objects, in which case, de-constructor
   * of those arrays will be able to free the memory. */
  class CSR : public CSRMap {
    public:
      CSR() : CSRMap() {}
      /** initialized with 3 Array objects (outer, inner, values), and in
       * this case, the memory will be managed by the Array objects. */
      CSR(DimensionType r, DimensionType c, Array<OrdinalType> & outer,
          Array<DimensionType> & inner, Array<DataType> & values);
      /** create an empty matrix with r rows and c cols */
      CSR(DimensionType r, DimensionType c);
      /** move constructor */
      CSR(CSR&& other) noexcept
        : CSRMap(std::move(other)) {
          outer_data_ = std::move(other.outer_data_);
          inner_data_ = std::move(other.inner_data_);
          values_data_ = std::move(other.values_data_);
        }
      /** move assignment */
      CSR& operator=(CSR&& other) noexcept
      {
        CSRMap::operator=(std::move(other));
        outer_data_ = std::move(other.outer_data_);
        inner_data_ = std::move(other.inner_data_);
        values_data_ = std::move(other.values_data_);
        return *this;
      }

    private:
      /** Arrays that keep data that is allocated for it
       * Since the usage here only considers the 3-array format, so
       * the data is stored in the 3-array format as shown below */
      Array<OrdinalType> outer_data_;
      Array<DimensionType> inner_data_;
      Array<DataType> values_data_;
  };
} // namespace ops

#endif

#endif // OPS_CSR_H_
