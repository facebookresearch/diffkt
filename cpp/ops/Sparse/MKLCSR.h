/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_MKLCSR_H_
#define OPS_MKLCSR_H_

#ifdef MKL

#include "MemUtils.h"
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

namespace ops {

  /**
   * Acts similar like Eigen's Map, it only contains pointers to the data
   * for the MKL sparse matrix, and will not be in charge of freeing the data
   * the pointers pointing to.
   * It also shouldn't do any changes to the data, thus all data access
   * are const. */
  class MKLCSRMap {
    public:
      /** Empty constructor */
      MKLCSRMap() : valid_(false) {}
      /** 3-array CSR variant pointers for constructor */
      MKLCSRMap(DimensionType r, DimensionType c, OrdinalType nnz,
        OrdinalType * outer, DimensionType * inner, DataType * values);

      /** De-constructor: destroy the MKL sparse matrix created */
      ~MKLCSRMap();
      /** copy constructor */
      MKLCSRMap(const MKLCSRMap& other) = delete;
      /** copy assignment */
      MKLCSRMap& operator=(const MKLCSRMap& other) = delete;

      /** move constructor */
      MKLCSRMap(MKLCSRMap&& other) noexcept
        : csr_(std::move(other.csr_)), valid_(other.valid_) {
          if (other.valid_ == true) {
            exportdata();
          }
          other.valid_ = false;
      }

      /** move assignment */
      MKLCSRMap& operator=(MKLCSRMap&& other) noexcept
      {
        csr_ = std::move(other.csr_);
        valid_ = other.valid_;
        if (other.valid_ == true) {
          exportdata();
        }
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
       * each row in the inner/value array
       *
       * MKL uses rows-start and rows-end arrays to store the starting and
       * ending pointers for each row separately. Thus data for all rows
       * is not necessary stored continuously. */
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

      /** get the const sparse_matrix_t object. As a memory map for a CSR,
       * the data inside should not be changed */
      const sparse_matrix_t & get() { return csr_; }

    protected:
      /** MKL needs to export the data explicitly to be able to access the
       * data inside of a sparse_matrix_t. This function exports the size
       * information and the non-zeros of the CSR format to the variables,
       * including rows_, cols_, nnz_, rows_start_, rows_end_, col_index_,
       * values_ */
      void exportdata();
      /** generate an MKL sparse matrix */
      void gencsr(DimensionType r, DimensionType c, OrdinalType * rows_start, OrdinalType * rows_end,
          DimensionType * col_index, DataType * values);
      /** generate an empty matrix
       *
       * MKL turns to fail when nnz == 0, such as when inner or
       * values are NULL. Thus we dealing with it separately. */
      void gencsr(DimensionType r, DimensionType c, OrdinalType * rows_start, OrdinalType * rows_end);

      /** MKL sparse matrix */
      sparse_matrix_t csr_;
      /** used to store and check the running status of all MKL calls */
      sparse_status_t status_;

      /** exported value or pointers from the MKL sparse matrix */
      DimensionType rows_, cols_;
      OrdinalType nnz_;
      OrdinalType *rows_start_, *rows_end_;
      DimensionType *col_index_;
      DataType * values_;

      /** label whether the `sparse_matrix_t csr_` is valid. That it,
       * whether it's created successfully with the MKL create csr function. */
      bool valid_;
  };

  /**
   * This class is based on MKLCSRMap, and the difference is that it
   * also contains all data for the sparse_matrix_t and will take care
   * of the data associated to the sparse_matrix_t.
   *
   * The data can be:
   *     controlled by MKL, in which case, mkl_sparse_destroy will be
   *     able to free the memory.
   *
   *     stored in the 3 Array objects, in which case, de-constructor
   *      of those arrays will be able to free the memory. */
  class MKLCSR : public MKLCSRMap {
    public:
      MKLCSR() : MKLCSRMap() {}
      /** initialized with a given sparse_matrix_t (assuming to be valid),
       * and in this case, the memory will be managed by MKL. */
      MKLCSR(sparse_matrix_t & csr) {
        csr_ = std::move(csr);
        valid_ = true;
        exportdata();
      }
      /** initialized with 3 Array objects (outer, inner, values), and in
       * this case, the memory will be managed by the Array objects. */
      MKLCSR(DimensionType r, DimensionType c, Array<OrdinalType> & outer,
          Array<DimensionType> & inner, Array<DataType> & values);
      /** create an empty matrix with r rows and c cols */
      MKLCSR(DimensionType r, DimensionType c);
      /** move constructor */
      MKLCSR(MKLCSR&& other) noexcept
        : MKLCSRMap(std::move(other)) {
          outer_data_ = std::move(other.outer_data_);
          inner_data_ = std::move(other.inner_data_);
          values_data_ = std::move(other.values_data_);
        }
      /** move assignment */
      MKLCSR& operator=(MKLCSR&& other) noexcept
      {
        MKLCSRMap::operator=(std::move(other));
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

#endif // MKL

#endif // OPS_MKLCSR_H_
