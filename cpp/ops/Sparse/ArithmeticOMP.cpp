/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(EIGEN) and !defined(MKL)

#include "Arithmetic.h"
#include "ArithmeticUtils.h"
#include "DebugUtils.h"

#include <omp.h>
#include <limits>

namespace ops {

  SpMat add(SpMatMap & left, SpMatMap & right) {
    return rowUnion(left, right, add);
  }

  SpMat times(SpMatMap & left, SpMatMap & right) {
    return rowIntersection(left, right, times);
  }

  SpMat sub(SpMatMap & left, SpMatMap & right) {
    return rowUnion(left, right, sub);
  }

  template<typename T>
  void prefixsum(T * data, size_t size) {
    for (size_t i=1; i<size; i++) data[i] += data[i-1];
  }

  /**
   * This function computes the compressed matrix of m
   *
   * compressedOuter, compressedInner, and compressedValues together form
   * a CSR representation of the compressed matrix.
   * In the compressed format, each value with data type of uint32_t,
   * represents continuous 32 elements in the binary format of m.
   *
   * More details about matrix compression section 3.2 of the paper:
   *   https://doi.org/10.1016/j.parco.2018.06.009
   *
   * This function assumes the non-zeros in m are sorted within each row.
   * If computeOuter == true, this function computes compressedOuter only;
   * otherwise, computes compressedInner, compressedValues. */
  void compute_compressed(const SpMatMap & m, Array<OrdinalType> & compressedOuter,
      Array<DimensionType> & compressedInner, Array<uint32_t> & compressedValues,
      bool computeOuter) {
    if (computeOuter) {
      compressedOuter.resize(m.rows()+1);
      compressedOuter[0]=0;
    } else {
      compressedInner.resize(compressedOuter.back());
      compressedValues.resize(compressedOuter.back());
    }
    #pragma omp parallel for
    for (DimensionType i=0; i<m.rows(); ++i) {
      DimensionType count = (computeOuter) ? 0 : compressedOuter[i];

      OrdinalType start = m.rowStartPtr()[i];
      OrdinalType end = m.rowEndPtr()[i];
      DimensionType preVcompressed;
      uint32_t value;
      if (start != end) {
        DimensionType v = m.innerIndexPtr()[start];
        preVcompressed = v >> 5;
        if (!computeOuter) {
          compressedInner[count] = preVcompressed;
          value = 1 << (v & 31);
        }
        count++;
      }

      for (OrdinalType j=start+1; j<end; ++j) {
        DimensionType v = m.innerIndexPtr()[j];

        if ((v >> 5) != preVcompressed) {
          // if the compressed value is not the same as the previous one,
          // increase the size of count, and update the
          // preVcompressed.
          preVcompressed = v >> 5;
          if (!computeOuter) {
            // update compressedValues, compressedInner and value
            compressedValues[count - 1] = value;
            value = (1 << (v & 31));
            compressedInner[count] = preVcompressed;
          }
          ++count;
        } else {
          if (!computeOuter) {
            // to compute compressedValues, we need to record the all non-zeros
            // update the value
            value |= (1 << (v & 31));
          }
        }
      }

      if (computeOuter)
        compressedOuter[i+1] = count;
      else if (start != end) {
        // updating compressedValues is behind of updating compressedInner
        // so this extra step is needed.
        compressedValues[count - 1] = value;
      }
    }

    if (computeOuter)
      prefixsum(compressedOuter.data(), compressedOuter.size());
  }

  void compute_compressed(const SpMatMap & m, Array<OrdinalType> & compressedOuter) {
    Array<DimensionType> inner;
    Array<uint32_t> values;
    compute_compressed(m, compressedOuter, inner, values, true);
  }

  /**
   * This function is used to collect information for matrix
   * multiplication.
   *
   * It collects information about the right input matrix:
   *   sortedRight: whether the non-zeros are sorted within each row.
   *   compressedOuter : prefix sum of the number of nonzeros in each row
   *   after compression.
   *
   * It collects information about the result matrix:
   *   vMin: the minimum column index among all non-zeros, for each row.
   *   vRange: the difference between the minimum and maximum column index
   *   among all non-zeros, for each row.
   *   maxInsRange: the maximum on all values in the vRange, across all
   *   rows.
   *   maxIns: the maximum number of multiplications needed in a row when
   *   using Gustavson's algorithm
   *   (https://dl.acm.org/doi/abs/10.1145/355791.355796).
   *   This value is efficiently computed without any multiplications, but
   *   can be thought of in the following way:
   *   First, create a binary matrix from the left and right input
   *   matrices.  Second, multiply them together into a new result matrix
   *   D. Third, compute the maximum sum of values in any row of D. This
   *   value is maxIns.
   *   totalIns: similar with maxIns. Instead of finding the maximum, the
   *   sum is computed.
   *   maxInsCompressed: similar with maxIns, but it's on the compressed
   *   right matrix.
   *   totalInsCompressed: similar with totalIns, but it's on the
   *   compressed right matrix. */
  void matmul_analysis(const SpMatMap & left, const SpMatMap & right,
      bool & sortedRight, Array<OrdinalType> & compressedOuter,
      Array<DimensionType> & vMin, Array<DimensionType> & vRange,
      OrdinalType & maxInsRange, OrdinalType & maxIns, OrdinalType & totalIns,
      OrdinalType & maxInsCompressed, OrdinalType & totalInsCompressed) {
    // Compute sortedRight
    // Compute the min and max column indices for each row on the right
    // matrix, which will be used to compute vMin, vRange and maxInsRange
    uint16_t unsorted = 0;
    Array<DimensionType> rightMin(right.rows());
    Array<DimensionType> rightMax(right.rows());
    #pragma omp parallel for reduction(+:unsorted)
    for (DimensionType i=0; i<right.rows(); ++i) {
      DimensionType min = right.cols();
      DimensionType max = 0;
      DimensionType preV = 0;
      for (OrdinalType j=right.rowStartPtr()[i]; j<right.rowEndPtr()[i]; ++j) {
        DimensionType v = right.innerIndexPtr()[j];
        // if current thread hasn't found a unsorted pair of non-zeros
        if (unsorted == 0) {
          if (preV > v) // if a pair of non-zeros (preV, v) is not sorted
            unsorted = 1;
          else // if it's sorted
            preV = v;
        }
        min = (v < min) ? v : min;
        max = (v > max) ? v : max;
      }
      rightMin[i] = min;
      rightMax[i] = max;
    }
    sortedRight = (unsorted == 0);

    if (sortedRight)
      compute_compressed(right, compressedOuter);

    // compute vMin, vRange, maxInsRange, maxIns, totalIns,
    // maxInsCompressed, totalInsCompressed
    vMin.resize(left.rows());
    vRange.resize(left.rows());
    maxInsRange = 0;
    maxIns = 0;
    totalIns = 0;
    maxInsCompressed = 0;
    totalInsCompressed = 0;
    #pragma omp parallel for reduction(max:maxInsRange) reduction(max:maxIns) reduction(+:totalIns) reduction(max:maxInsCompressed) reduction(+:totalInsCompressed)
    for (DimensionType i=0; i<left.rows(); ++i) {
      DimensionType min = right.cols();
      DimensionType max = 0;
      OrdinalType ins = 0;
      OrdinalType insCompressed = 0;
      for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j) {
        DimensionType v = left.innerIndexPtr()[j];
        min = (rightMin[v] < min) ? rightMin[v] : min;
        max = (rightMax[v] > max) ? rightMax[v] : max;
        ins += right.rowEndPtr()[v] - right.rowStartPtr()[v];
        if (sortedRight)
          insCompressed += compressedOuter[v+1] - compressedOuter[v];
      }
      vMin[i] = (max >= min) ? min : 0;
      vRange[i] = (max >= min) ? max - min + 1 : 0;
      maxInsRange = (vRange[i] > maxInsRange) ? vRange[i] : maxInsRange;
      maxIns = (ins > maxIns) ? ins: maxIns;
      totalIns += ins;
      if (sortedRight) {
        maxInsCompressed = (insCompressed > maxInsCompressed) ? insCompressed: maxInsCompressed;
        totalInsCompressed += insCompressed;
      }
    }
  }

  /** This function generates the chunk size for a task to be dynamically scheduled among threads */
  OrdinalType get_chunk_size(OrdinalType numOfTasks, OrdinalType tasksPerThread) {
    OrdinalType chunk_size;
    #pragma omp parallel
    {
      chunk_size = numOfTasks / omp_get_num_threads() / tasksPerThread;
    }
    return (chunk_size == 0) ? 1 : chunk_size;
  }

  /**
   * The accumulation implementation optimized for dense insertions.
   * That is, a large number of insertions happen in a small range of
   * values, such as for bandwidth matrix multiplication.
   * In the symbolic phase (if symbolic is true), this computes the outer
   * array. Otherwise, it assumes that outer has been computed and will
   * compute and fill inner and values.
   * Note that this approach will prune out zero results directly. */
  template<typename T>
  void accumulate_denseInsertion(const SpMatMap & left, const SpMatMap & right,
      Array<OrdinalType> & outer, Array<DimensionType> & inner, Array<DataType> & values,
      const Array<DimensionType> & vMin, const Array<DimensionType> & vRange, const OrdinalType maxInsRange,
      const OrdinalType chunk_size, const bool symbolic) {
    Array<DimensionType> rowSizes;
    if (symbolic) {
      outer.resize(left.rows() + 1);
      outer[0] = 0;
    } else {
      inner.resize(outer.back());
      values.resize(outer.back());
      // used to compute the sizes for each row in the numeric, and check the
      // consistency between symbolic and numeric, and prune the zero
      // values if necessary
      rowSizes.resize(left.rows()+1);
    }
    OrdinalType nonzeros = 0;
    T initialvalue = 0;
    #pragma omp parallel
    {
      // use vector instead of Array for table to avoid adding parallelism
      // inside of parallel region. Array also doesn't support `bool`
      std::vector<T> table(maxInsRange, initialvalue);
      #pragma omp for schedule(dynamic, chunk_size) reduction(+:nonzeros)
      for (DimensionType i=0; i<left.rows(); ++i) {
        DimensionType rowMin = vMin[i];
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j) {
          DimensionType rowRight = left.innerIndexPtr()[j];
          for (OrdinalType k=right.rowStartPtr()[rowRight]; k<right.rowEndPtr()[rowRight]; ++k) {
            DimensionType colRight = right.innerIndexPtr()[k];
            table[colRight-rowMin] = symbolic ? 1 : (T)((DataType)table[colRight-rowMin] + left.valuePtr()[j] * right.valuePtr()[k]);
          }
        }
        DimensionType count = 0;
        for (OrdinalType j=0; j<vRange[i]; j++) {
          if (table[j] != initialvalue) {
            if (!symbolic) {
              inner[count+outer[i]] = j + rowMin;
              values[count+outer[i]] = table[j];
            }
            ++count;
            table[j] = initialvalue;
          }
        }
        if (symbolic)
          outer[i+1] = count;
        else {
          rowSizes[i+1] = count;
          nonzeros += count;
        }
      }
    }
    if (symbolic)
      prefixsum(outer.data(), outer.size());
    else {
      Require(nonzeros <= outer.back(), "nonzeros computed from numeric phase should be smaller or equal to the one computed from symbolic.");
      // Since this approach prunes out zero values directly, then it is
      // possible that the size information from symbolic and numeric are
      // inconsistent. Therefore, in that case, we need to further
      // compress the memory afterwards.
      if (nonzeros < outer.back()) {
        // compress the data if the total number of nonzero is inconsistent
        // between symbolic and numeric.
        prefixsum(rowSizes.data(), rowSizes.size());
        Require(rowSizes.back() == nonzeros, "The last element in rowSizes should be the same as the number of non-zeros");
        Array<DimensionType> innerPruned(nonzeros);
        Array<DataType> valuesPruned(nonzeros);
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (DimensionType i=0; i<left.rows(); ++i) {
          OrdinalType k = outer[i];
          for (OrdinalType j=rowSizes[i]; j<rowSizes[i+1]; ++j, ++k) {
            innerPruned[j] = inner[k];
            valuesPruned[j] = values[k];
          }
        }
        outer = std::move(rowSizes);
        inner = std::move(innerPruned);
        values = std::move(valuesPruned);
      }
    }
  }

  /** This function counts the number of 1s in the binary form of x */
  template<typename T>
  uint8_t bitcounts(T x) {
    uint8_t b = sizeof(T)*8;
    uint8_t c = 0;
    for(uint8_t i=0; i<b && x!=0; i++) {
      c += (x & 1);
      x = x>>1;
    }
    return c;
  }

  /**
   * The accumulation is implemented with compression.
   * This only applies to symbolic phase to compute outer.
   * This function assumes compressedOuter is computed already. */
  void accumulate_compress(const SpMatMap & left, const SpMatMap & right,
      Array<OrdinalType> & compressedOuter,
      Array<OrdinalType> & outer,
      const Array<DimensionType> & vMin, const Array<DimensionType> & vRange, const OrdinalType maxInsRange,
      const OrdinalType maxInsCompressed, const OrdinalType chunk_size,
      bool denseInsertion) {
    outer.resize(left.rows() + 1);
    outer[0] = 0;

    Array<DimensionType> compressedInner;
    Array<uint32_t> compressedValues;
    compute_compressed(right, compressedOuter, compressedInner, compressedValues, false);

    uint32_t initialvalue = 0;
    OrdinalType tableSize = ((maxInsRange + 31) >> 5) + 1;
    #pragma omp parallel
    {
      std::vector<uint32_t> table(tableSize, initialvalue);
      Array<DimensionType> colIndices;
      if (!denseInsertion)
        colIndices.resize(std::min(maxInsCompressed, tableSize));
      #pragma omp for schedule(dynamic, chunk_size)
      for (DimensionType i=0; i<left.rows(); ++i) {
        DimensionType rowMin = vMin[i] >> 5;
        DimensionType countIns = 0;
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j) {
          DimensionType rowRight = left.innerIndexPtr()[j];
          for (OrdinalType k=compressedOuter[rowRight]; k<compressedOuter[rowRight+1]; ++k) {
            DimensionType colRight = compressedInner[k];
            if (!denseInsertion)
              if (table[colRight-rowMin] == initialvalue)
                colIndices[countIns++] = colRight;
            table[colRight-rowMin] |= compressedValues[k];
          }
        }
        DimensionType count = 0;
        if (denseInsertion) {
          DimensionType range = ((vRange[i] + 31) >> 5) + 1;
          for (OrdinalType j=0; j<range; j++) {
            if (table[j] != initialvalue) {
              count += bitcounts(table[j]);
              table[j] = initialvalue;
            }
          }
        } else {
          for (OrdinalType j=0; j<countIns; ++j) {
            count += bitcounts(table[colIndices[j]-rowMin]);
            table[colIndices[j]-rowMin] = initialvalue;
          }
        }
        outer[i+1] = count;
      }
    }
    prefixsum(outer.data(), outer.size());
  }

  /**
   * The accumulator for matrix multiplication for general matrix cases.
   * If symbolic is true, this computes outer;
   * Otherwise, it assumes outer is computed and computes inner and
   * values.
   * Note that this approach will not prune out zero results directly. */
  template<typename T>
  void accumulate(const SpMatMap & left, const SpMatMap & right,
      Array<OrdinalType> & outer, Array<DimensionType> & inner, Array<DataType> & values,
      const Array<DimensionType> & vMin, const OrdinalType maxInsRange, const OrdinalType maxIns,
      const OrdinalType chunk_size, const bool symbolic) {
    if (symbolic) {
      outer.resize(left.rows() + 1);
      outer[0] = 0;
    } else {
      inner.resize(outer.back());
      values.resize(outer.back());
    }
    T initial_value = symbolic ? 0 : std::numeric_limits<T>::max();
    #pragma omp parallel
    {
      // Similar to above, we use a vector here instead of Array.
      // `table` is used for storing whether a column index appeared
      // already. `colIndices` is used for storing the column indices that
      // appeared.
      std::vector<T> table(maxInsRange, initial_value);
      Array<DimensionType> colIndices(std::min(maxIns, maxInsRange));
      #pragma omp for schedule(dynamic, chunk_size)
      for (DimensionType i=0; i<left.rows(); ++i) {
        DimensionType count = 0;
        DimensionType rowMin = vMin[i];
        for (OrdinalType j=left.rowStartPtr()[i]; j<left.rowEndPtr()[i]; ++j) {
          DimensionType rowRight = left.innerIndexPtr()[j];
          for (OrdinalType k=right.rowStartPtr()[rowRight]; k<right.rowEndPtr()[rowRight]; ++k) {
            DimensionType colRight = right.innerIndexPtr()[k];
            if (table[colRight-rowMin] == initial_value) {
              colIndices[count++] = colRight;
              table[colRight-rowMin] = symbolic ? 1 : (T)(left.valuePtr()[j] * right.valuePtr()[k]);
            } else if (!symbolic) {
              table[colRight-rowMin] = (T)((DataType)table[colRight-rowMin] + left.valuePtr()[j] * right.valuePtr()[k]);
            }
          }
        }
        for (OrdinalType j=0; j<count; ++j) {
          if (!symbolic) {
            inner[j+outer[i]] = colIndices[j];
            values[j+outer[i]] = table[colIndices[j]-rowMin];
          }
          table[colIndices[j]-rowMin] = initial_value;
        }
        if (symbolic)
          outer[i+1] = count;
      }
    }
    if (symbolic)
      prefixsum(outer.data(), outer.size());
  }

  /** This function selects between dense insertion case and the general
   * case */
  template<typename T>
  void accumulate(const SpMatMap & left, const SpMatMap & right,
      Array<OrdinalType> & outer, Array<DimensionType> & inner, Array<DataType> & values,
      const Array<DimensionType> & vMin, const Array<DimensionType> & vRange,
      const OrdinalType maxInsRange, const OrdinalType maxIns,
      const OrdinalType chunk_size, const bool symbolic, const bool denseInsertion) {
    if (denseInsertion) // bandwidth-like computation
        accumulate_denseInsertion<T>(left, right, outer, inner, values, vMin, vRange, maxInsRange, chunk_size, symbolic);
    else // general case
        accumulate<T>(left, right, outer, inner, values, vMin, maxInsRange, maxIns, chunk_size, symbolic);
  }

  SpMat matmul(SpMatMap & left, SpMatMap & right) {
    Require(left.cols() == right.rows(), "In matmul operation, the number of columns on the left\
        should be equal to the number of rows on the right.");

    // analysis: generate size information during insertion
    // information about the right input matrix
    bool sortedRight;
    Array<DimensionType> compressedOuter;
    // information about the result matrix
    Array<DimensionType> vMin, vRange;
    OrdinalType maxInsRange, maxIns, totalIns, maxInsCompressed, totalInsCompressed;
    matmul_analysis(left, right, sortedRight, compressedOuter,
        vMin, vRange, maxInsRange, maxIns, totalIns, maxInsCompressed, totalInsCompressed);

    if (maxInsRange == 0 || maxIns == 0 || totalIns == 0) return SpMat(left.rows(), right.cols());

    Array<OrdinalType> outer;
    Array<DimensionType> inner;
    Array<DataType> values;
    // the number of continues rows as a task to schedule dynamically
    OrdinalType chunk_size  = get_chunk_size(left.rows(), 30);

    // When the number of insertion can be largely reduced with
    // compression, then use compression for the symbolic phase
    bool usecompression = sortedRight && totalInsCompressed < (totalIns >> 1);
    // Within maxInsRange, each value in average is inserted enough times,
    // then we call this dense insertion, and according optimizations can
    // be applied
    bool denseInsertion = maxInsRange*left.rows()*4 < totalIns;

    // symbolic: generate outer
    if (usecompression) {
      accumulate_compress(left, right, compressedOuter, outer, vMin, vRange, maxInsRange, maxInsCompressed, chunk_size, denseInsertion);
    } else {
      accumulate<bool>(left, right, outer, inner, values, vMin, vRange, maxInsRange, maxIns, chunk_size, true, denseInsertion);
    }
    // numeric: generate inner, values
    accumulate<DataType>(left, right, outer, inner, values, vMin, vRange, maxInsRange, maxIns, chunk_size, false, denseInsertion);
    return SpMat(left.rows(), right.cols(), outer, inner, values);
  }


  OrdinalType * genOuter(const DimensionType * row_indices, OrdinalType nonzeros, DimensionType num_row) {
    OrdinalType * outerptr;
    size_t num_threads;
    #pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }

    Calloc(outerptr, (num_row+1)*num_threads, sizeof(OrdinalType));
    // generate the outer array
    #pragma omp parallel for
    for (OrdinalType i=0; i<nonzeros; ++i) {
      outerptr[omp_get_thread_num()*(num_row+1) + row_indices[i]+1]++;
    }
    #pragma omp parallel for
    for (OrdinalType i=1; i<num_row+1; ++i) {
      size_t offset = i+num_row+1;
      for (OrdinalType j=1; j<omp_get_num_threads(); ++j, offset+=num_row+1)
        outerptr[i] += outerptr[offset];
    }
    prefixsum(outerptr, num_row + 1);
    return outerptr;
  }

  SpMat cooTocsr(COO & coo) {
    if (coo.nonZeros() == 0) return SpMat(coo.rows(), coo.cols());

    Array<OrdinalType> outer(coo.rows()+1);
    Array<DimensionType> inner(coo.nonZeros());
    Array<DataType> values(coo.nonZeros());
    if (sorted(coo)) {
      inner.assign(coo.col_index().data(), coo.nonZeros());
      values.assign(coo.values().data(), coo.nonZeros());
      #pragma omp parallel for
      for (OrdinalType i=0; i<=coo.row_index()[0]; ++i) outer[i] = 0;
      #pragma omp parallel for
      for (OrdinalType i=coo.row_index()[coo.nonZeros()-1]+1; i<=coo.rows(); ++i) outer[i] = coo.nonZeros();
      #pragma omp parallel for
      for (OrdinalType i=1; i<coo.nonZeros(); ++i) {
        if (coo.row_index()[i] != coo.row_index()[i-1])
          for (OrdinalType j=coo.row_index()[i-1]+1; j<=coo.row_index()[i]; ++j) outer[j] = i;
      }
    } else {
      // generate outer
      OrdinalType * outerptr = genOuter(coo.row_index().data(), coo.nonZeros(), coo.rows());
      outer.assign(outerptr, coo.rows() + 1);

      // generate the value and inner index
      #pragma omp parallel for
      for (OrdinalType i=0; i<coo.nonZeros(); ++i) {
        DimensionType r = coo.row_index()[i];
        OrdinalType pos;
        #pragma omp atomic capture
        pos = outerptr[r]++;

        inner[pos] = coo.col_index()[i];
        values[pos] = coo.values()[i];
      }
      FREE(outerptr);
    }
    return SpMat(coo.rows(), coo.cols(), outer, inner, values);
  }

  SpMat transpose(SpMatMap & tensor) {
    Array<OrdinalType> outer(tensor.cols()+1);
    Array<DimensionType> inner(tensor.nonZeros());
    Array<DataType> values(tensor.nonZeros());

    // generate outer
    OrdinalType * outerptr = genOuter(tensor.innerIndexPtr(), tensor.nonZeros(), tensor.cols());
    outer.assign(outerptr, tensor.cols() + 1);

    // generate the value and inner index
    #pragma omp parallel for
    for (OrdinalType i=0; i<tensor.rows(); ++i) {
      for (OrdinalType j=tensor.rowStartPtr()[i]; j<tensor.rowEndPtr()[i]; ++j) {
        DimensionType r = tensor.innerIndexPtr()[j];
        OrdinalType pos;
        #pragma omp atomic capture
        pos = outerptr[r]++;

        inner[pos] = i;
        values[pos] = tensor.valuePtr()[j];
      }
    }

    FREE(outerptr);
    return SpMat(tensor.cols(), tensor.rows(), outer, inner, values);
  }

} // namespace ops

#endif // not defined EIGEN and not defined MKL
