/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "Sparse/Arithmetic.h"
#include "Sparse/SparseFloatTensor.h"
#include <iostream>

#include <omp.h>

using namespace ops;

MemWrapper<SpMatMap> genBandCSR(DimensionType rows, DimensionType cols, DimensionType bandwidth)
{
  EXPECT_GT(rows, 0);
  EXPECT_GT(cols, 0);
  EXPECT_GT(bandwidth, 0);
  EXPECT_EQ(rows, cols);
  EXPECT_EQ(bandwidth%2, 1);

  DimensionType halfsize = bandwidth / 2;
  DimensionType nnz = rows * bandwidth - (halfsize + 1) * halfsize;

  OrdinalType * outer;
  DimensionType * inner;
  DataType * values;
  Malloc(outer, (rows + 1) * sizeof(OrdinalType));
  Malloc(inner, nnz * sizeof(DimensionType));
  Malloc(values, nnz * sizeof(DataType));

  OrdinalType offset = 0;
  for (DimensionType i=0; i<rows; ++i) {
    outer[i] = offset;

    DimensionType s = std::max((DimensionType)0, i-halfsize);
    DimensionType e = std::min(cols, i+halfsize+1);
    for (DimensionType j=s; j<e; ++j, ++offset) {
      inner[offset] = j;
      values[offset] = 1;
    }
  }
  EXPECT_EQ(nnz, offset);
  outer[rows] = nnz;

  return MemWrapper<SpMatMap>(SpMatMap(rows, cols, nnz, outer, inner, values), std::vector<void *>{ values, outer, inner });
}

COO genBandCOO(DimensionType rows, DimensionType cols, DimensionType bandwidth)
{
  EXPECT_GT(rows, 0);
  EXPECT_GT(cols, 0);
  EXPECT_GT(bandwidth, 0);
  EXPECT_EQ(rows, cols);
  EXPECT_EQ(bandwidth%2, 1);

  DimensionType halfsize = bandwidth / 2;
  DimensionType nnz = rows * bandwidth - (halfsize + 1) * halfsize;
  Array<DimensionType> row_index(nnz);
  Array<DimensionType> col_index(nnz);
  Array<DataType> values(nnz);

  OrdinalType offset = 0;
  for (DimensionType i=0; i<rows; ++i) {

    DimensionType s = std::max((DimensionType)0, i-halfsize);
    DimensionType e = std::min(cols, i+halfsize+1);
    for (DimensionType j=s; j<e; ++j, ++offset) {
      row_index[offset] = i;
      col_index[offset] = j;
      values[offset] = 1;
    }
  }
  EXPECT_EQ(nnz, offset);
  return COO(rows, cols, row_index, col_index, values);
}

size_t widthstart = 1000;
size_t widthend = 1024000;
size_t bandsize = 101;
size_t runs = 10;

template<class T>
void perfTestUnary(T op, const char * opname) {
  for (size_t x=widthstart; x<=widthend; x*=2) {
    for (size_t it=0; it<=runs; it++) {
      auto A = genBandCSR(x, x, bandsize);
      double timebegin, timeend;
      timebegin = omp_get_wtime();
      auto B = op(A.get());
      timeend = omp_get_wtime();
      printf("PerfTest: %s %ld th run with matrix width %ld band-size %ld took %f seconds\n",
          opname, it, x, bandsize, timeend - timebegin);
    }
  }
}

template<class T>
void perfTestBinary(T op, const char * opname) {
  for (size_t x=widthstart; x<=widthend; x*=2) {
    for (size_t it=0; it<=runs; it++) {
      auto A = genBandCSR(x, x, bandsize);
      auto B = genBandCSR(x, x, bandsize);
      double timebegin, timeend;
      timebegin = omp_get_wtime();
      auto C = op(A.get(), B.get());
      timeend = omp_get_wtime();
      printf("PerfTest: %s %ld th run with matrix width %ld band-size %ld took %f seconds\n",
          opname, it, x, bandsize, timeend - timebegin);
    }
  }
}

TEST(OnBandmatrices, matmul) {
  perfTestBinary(matmul, "matmul");
}

TEST(OnBandmatrices, times) {
  perfTestBinary(times, "times");
}

TEST(OnBandmatrices, sub) {
  perfTestBinary(sub, "sub");
}

TEST(OnBandmatrices, add) {
  perfTestBinary(add, "add");
}

TEST(OnBandmatrices, transpose) {
  perfTestUnary(transpose, "transpose");
}

TEST(OnBandmatrices, cooTocsr) {
  for (size_t x=widthstart; x<=widthend; x*=2) {
    auto A = genBandCOO(x, x, bandsize);
    for (size_t it=0; it<=runs; it++) {
      double timebegin, timeend;
      timebegin = omp_get_wtime();
      auto B = cooTocsr(A);
      timeend = omp_get_wtime();
      printf("PerfTest: cooTocsr %ld th run with matrix width %ld band-size %ld took %f seconds\n",
          it, x, bandsize, timeend - timebegin);
    }
  }
}
