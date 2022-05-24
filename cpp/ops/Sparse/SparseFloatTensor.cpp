/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SparseFloatTensor.h"

namespace ops {

SparseFloatTensor::SparseFloatTensor(const std::vector<SpMat> & sparse2Ds, bool squeeze_batch)
{
  size_t batch_size = sparse2Ds.size();
  if (batch_size == 0)
    return;

  if (batch_size == 1 && squeeze_batch == true) { // If squeeze_batch is true and there is only one SpMat, construct a 2D tensor
    // compute shape_
    shape_ = {(DimensionType)sparse2Ds[0].rows(), (DimensionType)sparse2Ds[0].cols()};

    const OrdinalType * rowsStart, * rowsEnd;
    rowPointers(sparse2Ds[0], rowsStart, rowsEnd);
    if (rowsStart + 1 == rowsEnd) {
      // compute values_
      values_.assign(sparse2Ds[0].valuePtr(), sparse2Ds[0].nonZeros());
      // compute dims_
      dims_.resize(1);
      dims_[0].inner().assign(sparse2Ds[0].innerIndexPtr(), sparse2Ds[0].nonZeros());
      dims_[0].outer().assign(rowsStart, shape_[0] + 1);
    } else { // rowsStart and rowsEnd are independent and values, inners need to be reordered
      // compute dims_
      dims_.resize(1);
      dims_[0].inner().resize(sparse2Ds[0].nonZeros());
      dims_[0].outer().resize(shape_[0] + 1);
      // compute dim 0 outer
      dims_[0].outer()[0] = 0;
      // TODO: add a parallel prefix sum function
      for (size_t i=1; i<shape_[0] + 1; i++)
        dims_[0].outer()[i] = dims_[0].outer()[i-1] + (rowsEnd[i-1] - rowsStart[i-1]);
      // compute dim 0 inner
      #pragma omp parallel for
      for (size_t i=0; i<shape_[0]; i++) {
        for(size_t j=rowsStart[i], k=dims_[0].outer()[i]; j<rowsEnd[i]; j++, k++)
          dims_[0].inner()[k] = sparse2Ds[0].innerIndexPtr()[j];
      }
      // compute value_
      values_.resize(sparse2Ds[0].nonZeros());
      #pragma omp parallel for
      for (size_t i=0; i<shape_[0]; i++) {
        for(size_t j=rowsStart[i], k=dims_[0].outer()[i]; j<rowsEnd[i]; j++, k++)
          values_[k] = sparse2Ds[0].valuePtr()[j];
      }
    }
  } else { // Otherwise, always construct a 3D tensor
    for (size_t i=0; i<batch_size; i++) {
      Require(sparse2Ds[i].rows() == sparse2Ds[0].rows(), "The number of rows in each 2D tensor needs to be consistent to construct 3D tensor");
      Require(sparse2Ds[i].cols() == sparse2Ds[0].cols(), "The number of cols in each 2D tensor needs to be consistent to construct 3D tensor");
    }
    // compute shape_
    shape_ = {(DimensionType)batch_size, (DimensionType)sparse2Ds[0].rows(), (DimensionType)sparse2Ds[0].cols()};

    // compute dims_
    dims_.resize(2);
    // compute dim 0 outer
    dims_[0].outer().resize(batch_size + 1);
    dims_[0].outer()[0] = 0;

    // compute the number of non-empty rows in each 2D matrix
    #pragma omp parallel for
    for (size_t i=0; i<batch_size; i++) {
      dims_[0].outer()[i+1] = 0;
      const OrdinalType * rowsStart, * rowsEnd;
      rowPointers(sparse2Ds[i], rowsStart, rowsEnd);
      for (size_t j=0; j<sparse2Ds[i].rows(); j++) {
        if (rowsEnd[j] > rowsStart[j]) dims_[0].outer()[i+1]++;
      }
    }
    // prefix sum on the number of non-empty rows in each 2D matrix
    // to generate dim 0 outer
    for (size_t i=0; i<batch_size; i++)
      dims_[0].outer()[i+1] += dims_[0].outer()[i];

    // compute dim 0 inner, and dim 1 outer
    OrdinalType totalNonEmptyRows = dims_[0].outer()[batch_size];
    dims_[0].inner().resize(totalNonEmptyRows);
    dims_[1].outer().resize(totalNonEmptyRows+1);
    OrdinalType nnz = 0;
    Array<OrdinalType> nnzPrefixSum(batch_size);
    for (size_t i=0; i<batch_size; i++) {
      nnzPrefixSum[i] = nnz;
      nnz += sparse2Ds[i].nonZeros();
    }
    #pragma omp parallel for
    for (size_t i=0; i<batch_size; i++) {
      OrdinalType pos = dims_[0].outer()[i];
      OrdinalType offset = nnzPrefixSum[i];
      const OrdinalType * rowsStart, * rowsEnd;
      rowPointers(sparse2Ds[i], rowsStart, rowsEnd);
      for (size_t j=0; j<sparse2Ds[i].rows(); j++) {
        if (rowsEnd[j] > rowsStart[j]) {
          dims_[0].inner()[pos] = j;
          dims_[1].outer()[pos] = offset;
          pos++;
          offset += rowsEnd[j] - rowsStart[j];
        }
      }
    }
    dims_[1].outer()[totalNonEmptyRows] = nnz;

    // compute values_ and dim 1 inner
    values_.resize(nnz);
    dims_[1].inner().resize(nnz);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<batch_size; i++) {
      const OrdinalType * rowsStart, * rowsEnd;
      rowPointers(sparse2Ds[i], rowsStart, rowsEnd);
      if (rowsStart + 1 == rowsEnd) { // copy the whole array directly
        for (size_t j=0; j<sparse2Ds[i].nonZeros(); j++) {
          values_[nnzPrefixSum[i] + j] = sparse2Ds[i].valuePtr()[j];
          dims_[1].inner()[nnzPrefixSum[i] + j] = sparse2Ds[i].innerIndexPtr()[j];
        }
      } else { // copy row by row
        size_t offset = nnzPrefixSum[i];
        for (size_t r=0; r<sparse2Ds[i].rows(); r++)
          for (size_t j=rowsStart[r]; j<rowsEnd[r]; j++, offset++) {
            values_[offset] = sparse2Ds[i].valuePtr()[j];
            dims_[1].inner()[offset] = sparse2Ds[i].innerIndexPtr()[j];
          }
      }
    }
  }
}

std::vector<MemWrapper<SpMatMap>> SparseFloatTensor::toSparse2Ds()
{
  std::vector<MemWrapper<SpMatMap>> sparse2Ds;
  if (shape_.size() == 2) {
    SpMatMap map (shape_[0], shape_[1], values_.size(),
          dims_[0].outer().data(), dims_[0].inner().data(), values_.data());
    std::vector<void*> empty_mem {};
    sparse2Ds.emplace_back(std::move(map), empty_mem);
  } else {
    Require(shape_.size() == 3, "toSparse2Ds can only support 2D or 3D tensor transforming");
    sparse2Ds.reserve(shape_[0]);
    for (DimensionType batchId = 0; batchId < shape_[0]; batchId++)
      sparse2Ds.emplace_back(SpMatMap
          (0, 0, 0, NULL, NULL, NULL), std::vector<void*>());

    #pragma omp parallel for schedule(dynamic)
    for (DimensionType batchId = 0; batchId < shape_[0]; batchId++) {
      DimensionType * rowIds = dims_[0].inner().data() + dims_[0].outer()[batchId];
      OrdinalType * outerOffsetted = dims_[1].outer().data() + dims_[0].outer()[batchId];
      size_t numOfRows = dims_[0].outer()[batchId+1] - dims_[0].outer()[batchId];

      std::vector<void*> mem_to_free;
      OrdinalType nnz = outerOffsetted[numOfRows] - outerOffsetted[0];
      OrdinalType * outer;
      Calloc(outer, (shape_[1]+1), sizeof(OrdinalType));
      Require(outer != NULL, "Unable to allocate memory for outer");
      DimensionType * inner = NULL;
      DataType * values = NULL;
      mem_to_free.emplace_back((void*)outer);

      bool rowIdsSorted = std::is_sorted(rowIds, rowIds+numOfRows);

      // set outers
      if (numOfRows == shape_[1] && rowIdsSorted) {
        // if every row is represented in dims_[0].outer() (which might mean non-empty)
        // and row ids are sorted, and for outer, we just need to minus the
        // starting offset
        for (size_t i=0; i<=shape_[1]; i++) outer[i] = outerOffsetted[i] - outerOffsetted[0];
      } else {
        // count the non-zeros in each row and then do
        for (size_t i=0; i<numOfRows; i++) outer[rowIds[i] + 1] = outerOffsetted[i+1] - outerOffsetted[i];
        // prefix sum
        for (size_t i=0; i<shape_[1]; i++) outer[i + 1] += outer[i];
      }

      // set inner and value
      if (rowIdsSorted) {
        // if row id is sorted in ascending order, we can use the inner and
        // values_ directly
        inner = dims_[1].inner().data() + outerOffsetted[0];
        values = values_.data() + outerOffsetted[0];
      } else {
        Malloc(inner, nnz * sizeof(DimensionType));
        mem_to_free.emplace_back((void*)inner);
        Malloc(values, nnz * sizeof(DataType));
        mem_to_free.emplace_back((void*)values);
        for (size_t i=0; i<numOfRows; i++) {
          OrdinalType start = outerOffsetted[i];
          OrdinalType end = outerOffsetted[i+1];
          DimensionType rowId = rowIds[i];
          for (size_t j=0; j<end-start; j++) {
            inner[outer[rowId] + j] = dims_[1].inner()[start + j];
            values[outer[rowId] + j] = values_[start + j];
          }
        }
      }

      SpMatMap map(shape_[1], shape_[2], nnz,
          outer, inner, values);
      sparse2Ds[batchId] = std::move(MemWrapper<SpMatMap>(std::move(map), mem_to_free));
    }
  }

  return sparse2Ds;
}

} // namespace ops
