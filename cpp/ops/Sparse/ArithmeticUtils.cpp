/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef EIGEN

#include "ArithmeticUtils.h"
#include "DebugUtils.h"

#include <unordered_map>
#include <set>

namespace ops {

  DataType times(DataType x, DataType y) { return x*y; }
  DataType add(DataType x, DataType y) { return x+y; }
  DataType sub(DataType x, DataType y) { return x-y; }

  /** This function is used to check whether the non-zeros are the same
   * (also ordered the same) in the left and right side. If so a lot
   * computations can be simplified and optimized, such as times operation */
  bool orderedTheSame(SpMatMap & left, SpMatMap & right) {
    // check the number of non-zeros
    if (left.nonZeros() != right.nonZeros()) return false;
    // check the size for each row
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      // `break` is not allowed for a OpenMP parallel for loop.
      // However, to reduce memory access, we use count to check whether
      // the memory accessing is necessary or not.
      if (count == 0) {
        OrdinalType left_row_size = left.rowEndPtr()[i] - left.rowStartPtr()[i];
        OrdinalType right_row_size = right.rowEndPtr()[i] - right.rowStartPtr()[i];
        if (left_row_size != right_row_size) count++;
      }
    }
    if (count != 0) return false;

    // check the ordering of non-zeros for each row
    count = 0;
    #pragma omp parallel for reduction(+:count) schedule(dynamic)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      if (count == 0) {
        for (OrdinalType j=left.rowStartPtr()[i], k=right.rowStartPtr()[i];
            j<left.rowEndPtr()[i]; ++j, ++k) {
          if (left.innerIndexPtr()[j] != right.innerIndexPtr()[k]) {
            count++;
            // as this loop is executed sequentially for a thread, `break` is allowed here
            break;
          }
        }
      }
    }
    return count == 0;
  }

  /** This function is used to check whether the non-zeros in a row are sorted.
   * If so a lot computations can be simplified and optimized, such as times operation */
  bool sorted(const DimensionType * data, OrdinalType start, OrdinalType end) {
    for (OrdinalType pos = start + 1; pos < end; pos++) {
      if (*(data + pos) <= *(data + pos - 1)) return false;
    }
    return true;
  }

  bool sorted(COO & coo) {
    size_t unsorted = 0;
    #pragma omp parallel for reduction(+:unsorted)
    for (OrdinalType i=1; i<coo.nonZeros(); ++i) {
      if (unsorted == 0)
        if (coo.row_index()[i-1] > coo.row_index()[i]) unsorted++;
    }
    return unsorted == 0;
  }

  // rowIntersection when either of the side is empty or both sides have same
  // non-zeros, in these cases, computations can be simplified.
  SpMat rowIntersectionTrival(SpMatMap & left, SpMatMap & right, OP * op) {
    // If one of the matrices is empty, then directly return an empty
    // matrix
    if (left.nonZeros() == 0 || right.nonZeros() == 0)
      return SpMat(left.rows(), left.cols());

    // Code below assuming the non-zeros in both side are the same.
    Array<OrdinalType> outer(left.rows()+1);
    Array<DimensionType> inner(left.nonZeros());
    Array<DataType> values(left.nonZeros());
    // If for both side, the CSR is in 3-array format, then
    // parallelization can simplified: more balanced for loop, and less
    // computation and memory access
    if (left.rowEndPtr() == left.rowStartPtr() + 1 && right.rowEndPtr() == right.rowStartPtr() + 1) {
      #pragma omp parallel for
      for (OrdinalType i=0; i<=left.rows(); ++i) outer[i] = left.rowStartPtr()[i];
      #pragma omp parallel for
      for (OrdinalType j=0; j<left.nonZeros(); ++j) {
        inner[j] = left.innerIndexPtr()[j];
        values[j] = op(left.valuePtr()[j], right.valuePtr()[j]);
      }
    } else {
      outer[0] = 0;
      #pragma omp parallel for
      for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] = left.rowEndPtr()[i] - left.rowStartPtr()[i];
      // prefix sum to get the outer
      for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] += outer[i];
      #pragma omp parallel for
      for (OrdinalType i=0; i<left.rows(); ++i) {
        OrdinalType j=left.rowStartPtr()[i];
        OrdinalType k=right.rowStartPtr()[i];
        OrdinalType w=outer[i];
        for (; j<left.rowEndPtr()[i]; ++j, ++k, ++w) {
          inner[w] = left.innerIndexPtr()[j];
          values[w] = op(left.valuePtr()[j], right.valuePtr()[k]);
        }
      }
    }
    return SpMat(left.rows(), left.cols(), outer, inner, values);
  }

  // TODO: add prune option
  SpMat rowIntersection(SpMatMap & left, SpMatMap & right, OP * op) {
    Require(left.valid() && right.valid(), "matrices in both side should be valid.");
    Require(left.rows() == right.rows(), "the number of rows on both side should be the same.");
    Require(left.cols() == right.cols(), "the number of cols on both side should be the same.");

    if (left.nonZeros() == 0 || right.nonZeros() == 0 || orderedTheSame(left, right))
      rowIntersectionTrival(left, right, op);

    // compute the number of non-zeros for each row in the resulting
    // matrix
    Array<OrdinalType> outer(left.rows()+1);
    bool * sortedrow;
    Malloc(sortedrow, left.rows());
    outer[0] = 0;
    #pragma omp parallel for schedule(dynamic)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      OrdinalType left_row_size = left.rowEndPtr()[i] - left.rowStartPtr()[i];
      OrdinalType right_row_size = right.rowEndPtr()[i] - right.rowStartPtr()[i];
      if (left_row_size == 0 || right_row_size == 0) {
        outer[i+1] = 0;
        continue;
      }

      // find size of the intersection of the two rows
      OrdinalType count = 0;
      sortedrow[i] = sorted(left.innerIndexPtr(), left.rowStartPtr()[i], left.rowEndPtr()[i]) &&
        sorted(right.innerIndexPtr(), right.rowStartPtr()[i], right.rowEndPtr()[i]);
      if (sortedrow[i]) {
        // find the size by merging two sorted rows
        OrdinalType j=left.rowStartPtr()[i];
        OrdinalType k=right.rowStartPtr()[i];
        while(j < left.rowEndPtr()[i] && k < right.rowEndPtr()[i]) {
          if (left.innerIndexPtr()[j] < right.innerIndexPtr()[k]) {
            ++j;
          } else if (left.innerIndexPtr()[j] > right.innerIndexPtr()[k]) {
            ++k;
          } else {
            ++j, ++k, ++count;
          }
        }
      } else {
        // use set to store the existing column indices
        std::set<DimensionType> s;
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j)
          s.insert(left.innerIndexPtr()[j]);
        for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
          if (s.find(right.innerIndexPtr()[j]) != s.end()) count++;
        }
      }
      outer[i+1] = count;
    }
    // prefix sum to get the outer
    for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] += outer[i];

    // compute the inner and values
    Array<DimensionType> inner(outer[left.rows()]);
    Array<DataType> values(outer[left.rows()]);
    #pragma omp parallel for schedule(dynamic)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      OrdinalType left_row_size = left.rowEndPtr()[i] - left.rowStartPtr()[i];
      OrdinalType right_row_size = right.rowEndPtr()[i] - right.rowStartPtr()[i];
      if (left_row_size == 0 || right_row_size == 0) {
        continue;
      }
      if (outer[i] == outer[i+1]) continue;

      OrdinalType count = outer[i];
      if (sortedrow[i]) {
        // find the size by merging two sorted rows
        OrdinalType j=left.rowStartPtr()[i];
        OrdinalType k=right.rowStartPtr()[i];
        while(j < left.rowEndPtr()[i] && k < right.rowEndPtr()[i]) {
          if (left.innerIndexPtr()[j] < right.innerIndexPtr()[k]) {
            ++j;
          } else if (left.innerIndexPtr()[j] > right.innerIndexPtr()[k]) {
            ++k;
          } else {
            inner[count] = left.innerIndexPtr()[j];
            values[count] = op(left.valuePtr()[j], right.valuePtr()[k]);
            ++j, ++k, ++count;
          }
        }
      } else {
        // find the intersection of the two rows
        // use map to store the existing column indices and their values
        std::unordered_map<DimensionType, DataType> m;
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j)
          m.insert({left.innerIndexPtr()[j], left.valuePtr()[j]});
        for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
          auto it = m.find(right.innerIndexPtr()[j]);
          if (it != m.end()) {
            inner[count] = it -> first;
            values[count] = op(it -> second, right.valuePtr()[j]);
            count++;
          }
        }
      }
    }
    FREE(sortedrow);

    return SpMat(left.rows(), left.cols(), outer, inner, values);
  }

  // rowUnion when either of the side is empty or both sides have same
  // non-zeros
  // - If one of the matrices is empty, then directly return the another
  // matrix
  // - If the non-zeros in both side are ordered the same, computations
  // can be simplified.
  SpMat rowUnionTrival(SpMatMap & left, SpMatMap & right, OP * op) {
      SpMatMap & nonempty = (left.nonZeros() == 0) ? right : left;
      Array<OrdinalType> outer(nonempty.rows()+1);
      Array<DimensionType> inner(nonempty.nonZeros());
      Array<DataType> values(nonempty.nonZeros());
      // If for both side, the CSR is in 3-array format, then
      // parallelization can simplified: more balanced for loop, and less
      // computation and memory access
      if (left.rowEndPtr() == left.rowStartPtr() + 1 && right.rowEndPtr() == right.rowStartPtr() + 1) {
        #pragma omp parallel for
        for (OrdinalType i=0; i<=left.rows(); ++i) outer[i] = nonempty.rowStartPtr()[i];
        #pragma omp parallel for
        for (OrdinalType j=0; j<nonempty.nonZeros(); ++j) {
          inner[j] = nonempty.innerIndexPtr()[j];
          if (left.nonZeros() == 0)
            values[j] = op(0, right.valuePtr()[j]);
          else if (right.nonZeros() == 0)
            values[j] = op(left.valuePtr()[j], 0);
          else
            values[j] = op(left.valuePtr()[j], right.valuePtr()[j]);
        }
      } else { // the non-zeros couldn't be read continuously, thus coarser parallelism.
        outer[0] = 0;
        #pragma omp parallel for
        for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] = nonempty.rowEndPtr()[i] - nonempty.rowStartPtr()[i];
        // prefix sum to get the outer
        for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] += outer[i];
        #pragma omp parallel for
        for (OrdinalType i=0; i<left.rows(); ++i) {
          OrdinalType j=left.rowStartPtr()[i];
          OrdinalType k=right.rowStartPtr()[i];
          OrdinalType w=outer[i];
          for (; w<outer[i+1]; ++j, ++k, ++w) {
            inner[w] = nonempty.innerIndexPtr()[j];
            if (left.nonZeros() == 0)
              values[w] = op(0, right.valuePtr()[k]);
            else if (right.nonZeros() == 0)
              values[w] = op(left.valuePtr()[j], 0);
            else
              values[w] = op(left.valuePtr()[j], right.valuePtr()[k]);
          }
        }
      }
      return SpMat(left.rows(), left.cols(), outer, inner, values);
  }

  // TODO: add prune option
  SpMat rowUnion(SpMatMap & left, SpMatMap & right, OP * op) {
    Require(left.valid() && right.valid(), "matrices in both side should be valid.");
    Require(left.rows() == right.rows(), "the number of rows on both side should be the same.");
    Require(left.cols() == right.cols(), "the number of cols on both side should be the same.");

    if (left.nonZeros() == 0 || right.nonZeros() == 0 || orderedTheSame(left, right)) {
      return rowUnionTrival(left, right, op);
    }

    // compute the number of non-zeros for each row in the resulting
    // matrix
    Array<OrdinalType> outer(left.rows()+1);
    bool * sortedrow;
    Malloc(sortedrow, left.rows());
    outer[0] = 0;
    #pragma omp parallel for schedule(dynamic)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      OrdinalType left_row_size = left.rowEndPtr()[i] - left.rowStartPtr()[i];
      OrdinalType right_row_size = right.rowEndPtr()[i] - right.rowStartPtr()[i];
      OrdinalType count = left_row_size + right_row_size;
      if (left_row_size == 0 || right_row_size == 0) {
        outer[i+1] = count;
        continue;
      }

      // find size of the intersection of the two rows
      sortedrow[i] = sorted(left.innerIndexPtr(), left.rowStartPtr()[i], left.rowEndPtr()[i]) &&
        sorted(right.innerIndexPtr(), right.rowStartPtr()[i], right.rowEndPtr()[i]);
      if (sortedrow[i]) {
        // find the size by merging two sorted rows
        OrdinalType j=left.rowStartPtr()[i];
        OrdinalType k=right.rowStartPtr()[i];
        while(j < left.rowEndPtr()[i] && k < right.rowEndPtr()[i]) {
          if (left.innerIndexPtr()[j] < right.innerIndexPtr()[k]) {
            ++j;
          } else if (left.innerIndexPtr()[j] > right.innerIndexPtr()[k]) {
            ++k;
          } else {
            ++j, ++k, --count;
          }
        }
      } else {
        // use set to store the existing column indices
        std::set<DimensionType> s;
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j)
          s.insert(left.innerIndexPtr()[j]);
        for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
          if (s.find(right.innerIndexPtr()[j]) != s.end()) --count;
        }
      }
      outer[i+1] = count;
    }
    // prefix sum to get the outer
    for (OrdinalType i=0; i<left.rows(); ++i) outer[i+1] += outer[i];

    // compute the inner and values
    Array<DimensionType> inner(outer.back());
    Array<DataType> values(outer.back());
    #pragma omp parallel for schedule(dynamic)
    for (OrdinalType i=0; i<left.rows(); ++i) {
      if (outer[i] == outer[i+1]) continue;

      OrdinalType left_row_size = left.rowEndPtr()[i] - left.rowStartPtr()[i];
      OrdinalType right_row_size = right.rowEndPtr()[i] - right.rowStartPtr()[i];
      OrdinalType count = outer[i];
      if (left_row_size == 0 || right_row_size == 0) {
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j) {
          inner[count] = left.innerIndexPtr()[j];
          values[count] = op(left.valuePtr()[j], 0);
          count++;
        }
        for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
          inner[count] = right.innerIndexPtr()[j];
          values[count] = op(0, right.valuePtr()[j]);
          count++;
        }
        continue;
      }

      if (sortedrow[i]) {
        // find the size by merging two sorted rows
        OrdinalType j=left.rowStartPtr()[i];
        OrdinalType k=right.rowStartPtr()[i];
        while(j < left.rowEndPtr()[i] && k < right.rowEndPtr()[i]) {
          if (left.innerIndexPtr()[j] < right.innerIndexPtr()[k]) {
            inner[count] = left.innerIndexPtr()[j];
            values[count] = op(left.valuePtr()[j], 0);
            ++j;
          } else if (left.innerIndexPtr()[j] > right.innerIndexPtr()[k]) {
            inner[count] = right.innerIndexPtr()[k];
            values[count] = op(0, right.valuePtr()[k]);
            ++k;
          } else {
            inner[count] = left.innerIndexPtr()[j];
            values[count] = op(left.valuePtr()[j], right.valuePtr()[k]);
            ++j, ++k;
          }
          ++count;
        }
        while(j < left.rowEndPtr()[i]) {
            inner[count] = left.innerIndexPtr()[j];
            values[count] = op(left.valuePtr()[j], 0);
            ++j, ++count;
        }
        while(k < right.rowEndPtr()[i]) {
            inner[count] = right.innerIndexPtr()[k];
            values[count] = op(0, right.valuePtr()[k]);
            ++k, ++count;
        }
      } else {
        // find the intersection of the two rows
        // use map to store the existing column indices and their values
        std::unordered_map<DimensionType, DataType> m;
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j)
          m.insert({left.innerIndexPtr()[j], left.valuePtr()[j]});
        for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
          auto it = m.find(right.innerIndexPtr()[j]);
          if (it != m.end()) {
            inner[count] = it -> first;
            values[count] = op(it -> second, right.valuePtr()[j]);
            count++;
            m.erase(it);
          } else {
            inner[count] = right.innerIndexPtr()[j];
            values[count] = op(0, right.valuePtr()[j]);
            count++;
          }
        }
        for (auto &e : m) {
            inner[count] = e.first;
            values[count] = op(e.second, 0);
            count++;
        }
      }
    }
    FREE(sortedrow);

    return SpMat(left.rows(), left.cols(), outer, inner, values);
  }
}
#endif // not defined EIGEN
