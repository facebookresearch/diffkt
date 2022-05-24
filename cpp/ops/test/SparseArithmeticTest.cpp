/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "Sparse/Arithmetic.h"
#include "SparseTestUtils.cpp"
#include <iostream>

using namespace ops;

TEST(AddTest, DoesAdd) {
  std::vector<DimensionType> shape = {5, 5};
  std::vector<DataType> values = {22, 7, 3, 5, 14, 1, 17, 8};
  std::vector<DataType> valuesE = {44, 14, 6, 10, 28, 2, 34, 16};
  std::vector<DimensionType> inner = {1, 2, 0, 2, 4, 2, 1, 4};
  std::vector<OrdinalType> outer = {0, 2, 4, 5, 6, 8};
  SpMatMap left(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMatMap right(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMat t = add(left, right);

  compareCSR(t, shape[0], shape[1], outer, inner, valuesE);
}

#ifndef EIGEN // In this test, Eigen will assert fail, which can not be caught
TEST(TestAll, ShapeChecking) {
  std::vector<DimensionType> shape1 = {2, 2};
  std::vector<DimensionType> shape2 = {3, 3};
  std::vector<DataType> values1 = {1, 2};
  std::vector<DataType> values2 = {};
  std::vector<DimensionType> inner1 = {0, 1};
  std::vector<DimensionType> inner2 = {};
  std::vector<OrdinalType> outer1 = {0, 1, 2};
  std::vector<OrdinalType> outer2 = {0, 0, 0, 0};
  SpMatMap left(shape1[0], shape1[1], inner1.size(), outer1.data(), inner1.data(),
      values1.data());
  SpMatMap right(shape2[0], shape2[1], inner2.size(), outer2.data(), inner2.data(),
      values2.data());

  try {
    SpMat t = matmul(left, right);
    EXPECT_EQ(1, 0);
  } catch (...) {
    EXPECT_EQ(1, 1);
  }

  try {
    SpMat t = times(left, right);
    EXPECT_EQ(1, 0);
  } catch (...) {
    EXPECT_EQ(1, 1);
  }

  try {
    SpMat t = add(left, right);
    EXPECT_EQ(1, 0);
  } catch (...) {
    EXPECT_EQ(1, 1);
  }

  try {
    SpMat t = sub(left, right);
    EXPECT_EQ(1, 0);
  } catch (...) {
    EXPECT_EQ(1, 1);
  }
}
#endif

TEST(TestAll, BetweenEmpty) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values1 = {1, 2};
  std::vector<DataType> values2 = {};
  std::vector<DimensionType> inner1 = {0, 1};
  std::vector<DimensionType> inner2 = {};
  std::vector<OrdinalType> outer1 = {0, 1, 2};
  std::vector<OrdinalType> outer2 = {0, 0, 0};
  SpMatMap left(shape[0], shape[1], inner1.size(), outer1.data(), inner1.data(),
      values1.data());
  SpMatMap right(shape[0], shape[1], inner2.size(), outer2.data(), inner2.data(),
      values2.data());

  std::vector<DataType> valuesE = {};
  std::vector<DimensionType> innerE = {};
  std::vector<OrdinalType> outerE = {0, 0, 0};

  SpMat t = matmul(left, right);
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);

  t = std::move(matmul(right, left));
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);

  t = std::move(times(left, right));
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);

  t = std::move(times(right, left));
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);

  t = std::move(add(left, right));
  compareCSR(t, shape[0], shape[1], outer1, inner1, values1);

  t = std::move(add(right, left));
  compareCSR(t, shape[0], shape[1], outer1, inner1, values1);

  t = std::move(sub(left, right));
  compareCSR(t, shape[0], shape[1], outer1, inner1, values1);

  t = std::move(sub(right, left));
  valuesE = {-1, -2};
  compareCSR(t, shape[0], shape[1], outer1, inner1, valuesE);
}

TEST(MatmulTest, EmptyResults) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values1 = {1, 2};
  std::vector<DataType> values2 = {1, 2};
  std::vector<DimensionType> inner1 = {1, 1};
  std::vector<DimensionType> inner2 = {0, 1};
  std::vector<OrdinalType> outer1 = {0, 1, 2};
  std::vector<OrdinalType> outer2 = {0, 2, 2};
  SpMatMap left(shape[0], shape[1], inner1.size(), outer1.data(), inner1.data(),
      values1.data());
  SpMatMap right(shape[0], shape[1], inner2.size(), outer2.data(), inner2.data(),
      values2.data());

  std::vector<DataType> valuesE = {};
  std::vector<DimensionType> innerE = {};
  std::vector<OrdinalType> outerE = {0, 0, 0};

  SpMat t = matmul(left, right);
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);
}

TEST(MatmulTest, DoesMatmul) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values = {1, 2};
  std::vector<DimensionType> inner1 = {0, 1};
  std::vector<DimensionType> inner2 = {1, 1};
  std::vector<OrdinalType> outer = {0, 1, 2};
  SpMatMap left(shape[0], shape[1], inner1.size(), outer.data(), inner1.data(),
      values.data());
  SpMatMap right(shape[0], shape[1], inner2.size(), outer.data(), inner2.data(),
      values.data());
  SpMat t = matmul(left, right);
  std::vector<DataType> valuesE = {1, 4};
  std::vector<DimensionType> innerE = {1, 1};
  std::vector<OrdinalType> outerE = {0, 1, 2};

  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);
}

TEST(TestAll, SameIndices) {
  std::vector<DimensionType> shape = {5, 5};
  std::vector<DataType> values = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<DimensionType> inner = {1, 2, 0, 2, 4, 2, 1, 4};
  std::vector<OrdinalType> outer = {0, 2, 4, 5, 6, 8};

  std::vector<DataType> valuesTimes = {1, 4, 9, 16, 25, 36, 49, 64};
  std::vector<DataType> valuesAdd = {2, 4, 6, 8, 10, 12, 14, 16};
  // NOTE: both MKL and Eigen doesn't compress for this case, thus our OMP
  // does the same for now.
  std::vector<DataType> valuesSub = {0, 0, 0, 0, 0, 0, 0, 0};

  SpMatMap left(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMatMap right(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());

  SpMat t = times(left, right);
  compareCSR(t, shape[0], shape[1], outer, inner, valuesTimes);

  t = std::move(add(left, right));
  compareCSR(t, shape[0], shape[1], outer, inner, valuesAdd);

  t = std::move(sub(left, right));
  compareCSR(t, shape[0], shape[1], outer, inner, valuesSub);
}

TEST(TestAll, DifferentIndices) {
  std::vector<DimensionType> shape = {7, 3};
  std::vector<OrdinalType> outer1 = {0, 1, 2, 3, 3, 3, 6, 8};
  std::vector<OrdinalType> outer2 = {0, 1, 2, 2, 3, 3, 5, 7};
  std::vector<DimensionType> inner1 = {0, 1, 2, 0, 1, 2, 0, 1};
  std::vector<DimensionType> inner2 = {1, 1, 2, 0, 1, 1, 2};
  std::vector<DataType> values1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<DataType> values2 = {1, 2, 3, 4, 5, 6, 7};
  SpMatMap left(shape[0], shape[1], inner1.size(), outer1.data(), inner1.data(),
      values1.data());
  SpMatMap right(shape[0], shape[1], inner2.size(), outer2.data(), inner2.data(),
      values2.data());

  std::vector<DataType> valuesTimes = {4, 16, 25, 48};
  std::vector<DimensionType> innerTimes = {1, 0, 1, 1};
  std::vector<OrdinalType> outerTimes = {0, 0, 1, 1, 1, 1, 3, 4};
  SpMat t = times(left, right);
  compareCSR(t, shape[0], shape[1], outerTimes, innerTimes, valuesTimes);

  std::vector<DataType> valuesAdd = {1, 1, 4, 3, 3, 8, 10, 6, 7, 14, 7};
  std::vector<DimensionType> innerAdd = {0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2};
  std::vector<OrdinalType> outerAdd = {0, 2, 3, 4, 5, 5, 8, 11};
  t = std::move(add(left, right));
  compareCSR(t, shape[0], shape[1], outerAdd, innerAdd, valuesAdd);

  std::vector<DataType> valuesSub = {1, -1, 0, 3, -3, 0, 0 , 6, 7, 2, -7};
  t = std::move(sub(left, right));
  compareCSR(t, shape[0], shape[1], outerAdd, innerAdd, valuesSub);
}

#ifdef EIGEN
/**
*  Test cases generation for matrix division:
*  test cases are generated with scripts under the folder
*    ./python/utilities/testCasesGen
*  The scripts will generate random input with given size and sparsity
*  requirement, call numpy for the according operations and then print
*  the input(s) and output to the format needed.
*
*  NOTE:
*  For inverse operation and the second matrix in division, both numpy
*  and the solver used in Eigen only support square matrix
*  (Eigen has assertion for checking the shape, so we couldn't test the
*  non-square case; similarly, with matmul, when the dimension
*  of the column of the left matrix is not the same as the row of the
*  right matrix) thus, we are only testing with square matrix case.
*  The inverse result won't be always nearly the same among
*  different libraries. Such as, the input generated by r=9,c=9,s=7,z=0.3
*  Not all matrices can be inversed. With sparser matrix, it turns to be
*  have higher chance to generate a singular matrix.
*
**/

// TEST on divided by a identity matrix, expect result should be the same
// as first input matrix
TEST(MatdivTest, DoesMatdiv1) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values = {1, 2};
  std::vector<DimensionType> inner = {0, 1};
  std::vector<OrdinalType> outer = {0, 1, 2};
  std::vector<DataType> valuesI = {1, 1};
  SpMatMap left(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMatMap right(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      valuesI.data());
  SpMat t = matdiv(left, right);

  compareCSR(t, shape[0], shape[1], outer, inner, values);
}

// TEST on a matrix divided by a diagonal matrix
TEST(MatdivTest, DoesMatdiv2) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values1 = {1, 2};
  std::vector<DataType> values2 = {1, 2};
  std::vector<DimensionType> inner1 = {0, 1};
  std::vector<DimensionType> inner2 = {0, 1};
  std::vector<OrdinalType> outer = {0, 1, 2};
  SpMatMap left(shape[0], shape[1], inner1.size(), outer.data(), inner1.data(),
      values1.data());
  SpMatMap right(shape[0], shape[1], inner2.size(), outer.data(), inner2.data(),
      values2.data());
  SpMat t = matdiv(left, right);

  std::vector<DataType> valuesE = {1, 1};
  std::vector<DimensionType> innerE = {0, 1};
  std::vector<OrdinalType> outerE = {0, 1, 2};

  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);
}


//  TEST: with given inputs, A is non-square, B as square, compare output with Numpy's output
//  Inputs generated randomly div.py
//  with p = 5, k =9, q = 9, seed = 16, z = 0.1 0.3
TEST(MatdivTest, DoesMatdiv4) {
  //input A:
  std::vector<DimensionType> shapeA = {5, 9};
  std::vector<DataType> valuesA = {
    0.38877912391436853,0.7800433424910276
  };
  std::vector<DimensionType> innerA = {
    8,3
  };
  std::vector<OrdinalType> outerA = {
    0,1,1,1,2,
    2
  };
  //input B:
  std::vector<DimensionType> shapeB = {9, 9};
  std::vector<DataType> valuesB = {
    0.22329107915353885,0.36072883534777267,0.22308094169128878,0.6887261618213565,0.941010860245836,
    0.07799233938998706,0.2934872561021975,0.46426408804015173,0.21506214371433952,0.12087875153895689,
    0.917687007744725,0.04321572973662613,0.7524449169599731,0.44335096201326485,0.4679961473799611,
    0.3406742720441993,0.27149077328868065,0.21542897828136154,0.018399946630616282,0.7423517360658496,
    0.420103554589757,0.9720455259033695,0.6756664492813946,0.19916263217137087,0.39386732248283196,
    0.33918540311781975,0.4072471068550777
  };
  std::vector<DimensionType> innerB = {
    0,4,5,6,0,
    2,6,8,0,2,
    3,7,8,7,8,
    8,2,4,6,0,
    2,4,1,8,1,
    2,5
  };
  std::vector<OrdinalType> outerB = {
    0,4,8,13,15,
    16,19,22,24,27
  };
  //expected output:
  std::vector<DimensionType> shapeE = {5, 9};
  std::vector<DataType> valuesE = {
    1.1412048276540474,0.1073318494237487,-0.23509817372107958,0.8500102277878319,-0.08285492854382911,
    -1.4632403080983682,-0.2676059269963153,0.019476843308245978,0.03427288554210957,-0.058794008944152015
  };
  std::vector<DimensionType> innerE = {
    4,0,1,2,3,
    4,5,6,7,8
  };
  std::vector<OrdinalType> outerE = {
    0,1,1,1,10,
    10
  };

  SpMatMap left(shapeA[0], shapeA[1], innerA.size(), outerA.data(), innerA.data(),
      valuesA.data());
  SpMatMap right(shapeB[0], shapeB[1], innerB.size(), outerB.data(), innerB.data(),
      valuesB.data());
  SpMat t = matdiv(left, right);

  compareCSR(t, shapeE[0], shapeE[1], outerE, innerE, valuesE);
}

//  TEST: divided by a singular matrix input
//  Inputs generated randomly through matdiv.py
//  with p = 2, k =2, q = 2, seed = 0, z = 0.3 0.3
TEST(MatdivTest, DoesMatdiv5) {
  //input:
  //input A:
  std::vector<DimensionType> shapeA = {2, 2};
  std::vector<DataType> valuesA = {
    0.3834415188257777
  };
  std::vector<DimensionType> innerA = {
    1
  };
  std::vector<OrdinalType> outerA = {
    0,1,1
  };
  //input B:
  std::vector<DimensionType> shapeB = {2, 2};
  std::vector<DataType> valuesB = {
    0.5448831829968969
  };
  std::vector<DimensionType> innerB = {
    1
  };
  std::vector<OrdinalType> outerB = {
    0,0,1
  };

  //expected output: failing and throw error
  SpMatMap left(shapeA[0], shapeA[1], innerA.size(), outerA.data(), innerA.data(),
      valuesA.data());
  SpMatMap right(shapeB[0], shapeB[1], innerB.size(), outerB.data(), innerB.data(),
      valuesB.data());
  try {
    SpMat t = matdiv(left, right);
    EXPECT_EQ(1, 0);
  } catch (...) {
    EXPECT_EQ(1, 1);
  }
}

// TEST on inverse of a diagonal matrix
TEST(InvTest, DoesInv1) {
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values = {1, 2};
  std::vector<DimensionType> inner = {0, 1};
  std::vector<OrdinalType> outer = {0, 1, 2};
  SpMatMap A(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMat t = inverse(A);

  std::vector<DataType> valuesE = {1, 0.5};
  std::vector<DimensionType> innerE = {0, 1};
  std::vector<OrdinalType> outerE = {0, 1, 2};

  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);
}

//  TEST: with a given input, compare output with Numpy's output
//  Inputs generated randomly through inverse.py
//  row = 3, cols =3, seed = 0, z = 0.5
TEST(InvTest, DoesInv2) {
  //input:
  std::vector<DimensionType> shape = {3, 3};
  std::vector<DataType> values = {0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.9636627605010293};
  std::vector<DimensionType> inner = {1, 2, 0, 1, 2};
  std::vector<OrdinalType> outer = {0, 2, 4, 5};
  //expected output:
  std::vector<DimensionType> shapeE = {3, 3};
  std::vector<DataType> valuesE = {-1.087145513064705, 1.835255759775752, 0.6800008536132711, 1.3982310797938675, -0.8745823961762934, 1.0377074231654217};
  std::vector<DimensionType> innerE = {0, 1, 2, 0, 2, 2};
  std::vector<OrdinalType> outerE = {0, 3, 5, 6};

  SpMatMap A(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMat t = inverse(A);

  compareCSR(t, shapeE[0], shapeE[1], outerE, innerE, valuesE);
}

//  TEST: with a given input, compare output with Numpy's output
//  Inputs generated randomly through inverse.py
//  with row = 9, cols =9, seed = 16, z = 0.3
TEST(InvTest, DoesInv3) {
  //input:
  std::vector<DimensionType> shape = {9, 9};
  std::vector<DataType> values = {
    0.22329107915353885,0.36072883534777267,0.22308094169128878,0.6887261618213565,0.941010860245836,
    0.07799233938998706,0.2934872561021975,0.46426408804015173,0.21506214371433952,0.12087875153895689,
    0.917687007744725,0.04321572973662613,0.7524449169599731,0.44335096201326485,0.4679961473799611,
    0.3406742720441993,0.27149077328868065,0.21542897828136154,0.018399946630616282,0.7423517360658496,
    0.420103554589757,0.9720455259033695,0.6756664492813946,0.19916263217137087,0.39386732248283196,
    0.33918540311781975,0.4072471068550777
  };
  std::vector<DimensionType> inner = {
    0,4,5,6,0,
    2,6,8,0,2,
    3,7,8,7,8,
    8,2,4,6,0,
    2,4,1,8,1,
    2,5
  };
  std::vector<OrdinalType> outer = {
    0,4,8,13,15,
    16,19,22,24,27
  };
  //expected output:
  std::vector<DimensionType> shapeE = {9, 9};
  std::vector<DataType> valuesE = {
    -0.3405435125255594,0.863724670635094,-1.1134951087384828,-1.0299463187596036,0.35463786274130166,
    -0.10874133716654279,0.18654219068054698,-0.8652391590239443,1.4800202097699988,-0.4387307623387359,
    0.751399880405254,-0.9420918385813427,4.436918569481935,-0.8205150643229108,-0.14009419647725657,
    0.2403269905764879,0.13759728924932563,-0.30139116753474987,1.0896961508233232,-0.10621836509652943,
    -1.8758448773187228,-0.34306545857019916,0.024968924478026695,0.04393715537993172,-0.07537274628406727,
    0.44968616062447847,-0.9843708385120964,1.2575348835383482,-1.1309993157544012,1.1125356099250387,
    0.1435924415324881,-0.2463281150167808,0.36540731162765056,-0.6258211956521841,1.6214559165677855,
    -3.69539276831797,0.6833854143392304,-1.3147145350415341,2.25534958348537,1.208477624890337,
    0.4382550434665829,-0.8228400686359161,2.1232438531819775,-0.9190320373868115,0.38588746528113615,
    -0.661977266469136,2.2555494082137133,-3.098527009201964,2.935355211884798
  };
  std::vector<DimensionType> innerE = {
    0,1,4,5,6,
    7,8,4,7,0,
    1,4,5,6,7,
    8,0,1,2,3,
    4,5,6,7,8,
    0,1,4,5,6,
    7,8,0,1,4,
    5,6,7,8,0,
    1,4,5,6,7,
    8,3,4,4
  };
  std::vector<OrdinalType> outerE = {
    0,7,9,16,25,
    32,39,46,48,49
  };

  SpMatMap A(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMat t = inverse(A);

  compareCSR(t, shapeE[0], shapeE[1], outerE, innerE, valuesE);
}

//  TEST: inverse with a singular matrix input
//  Inputs generated randomly through inverse.py
//  with row = 2, cols = 2, seed = 0, z = 0.3
TEST(InvTest, DoesInv4) {
  //input:
  std::vector<DimensionType> shape = {2, 2};
  std::vector<DataType> values = {
      0.5448831829968969
  };
  std::vector<DimensionType> inner = {
      1
  };
  std::vector<OrdinalType> outer = {
      0,0,1
  };

  //expected output: failing and throw error
  SpMatMap A(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  try {
      SpMat t = inverse(A);
      EXPECT_EQ(1, 0);
  } catch (...) {
      EXPECT_EQ(1, 1);
  }
}

//  TEST: with a given input, multiply the input with the inversed and
//  compare that with the Identity matrix.
//  Inputs generated randomly through inverse.py
//  with row = 9, cols =9, seed = 7, z = 0.3
TEST(InvTest, DoesInv5) {
  //input:
  std::vector<DimensionType> shape = {9, 9};
  std::vector<DataType> values = {
    0.4384092314408935,0.7234651778309412,0.5011204636599379,0.07205113335976154,0.49988250082555996,
    0.6792299961209405,0.3809411331485384,0.9095935277196137,0.2133853535799155,0.9501295004136456,
    0.9091283748867313,0.13316944575925016,0.5234125806737658,0.4774011548515884,0.36589038578059296,
    0.6573994627797582,0.3703510829880352,0.4129918291138346,0.9064232691643387,0.42237404364314035,
    0.42645357268494233,0.4148859784394432,0.0014268805627581926,0.5243455967651972,0.30885268486379713,
    0.13687611879745087,0.34353652970435833,0.16550140046578743
  };
  std::vector<DimensionType> inner = {
    2,3,6,7,0,
    1,3,6,7,3,
    6,7,8,6,7,
    6,7,1,2,5,
    6,0,1,4,0,
    4,5,8
  };
  std::vector<OrdinalType> outer = {
    0,4,9,13,15,
    17,21,24,25,28
  };

  //the identity matrix
  std::vector<DataType> valuesDenseI = {
    1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1
  };

  SpMatMap A(shape[0], shape[1], inner.size(), outer.data(), inner.data(),
      values.data());
  SpMat t = inverse(A);

  SpMatMap A_inv(t.rows(), t.cols(), t.nonZeros(), t.outerIndexPtr(),
      t.innerIndexPtr(), t.valuePtr());
  SpMat Iout = matmul(A, A_inv);
  std::vector<DataType> valuesDenseIout = toDenseData(Iout);

  EXPECT_FLOATS_NEARLY_EQ(valuesDenseIout, valuesDenseI, 1e-6);
}
#endif //EIGEN

//  TEST: with given inputs, compare output with Numpy's output
//  Inputs generated randomly by testall.py
//  with ./testall.py -p 9 -z 0.3 -s 16
TEST(TestAll, 9by9) {
  //input A:
  DimensionType rowsA = 9, colsA = 9;
  std::vector<DataType> valuesA = {
    0.5678649545949946,0.6264006261919175,0.4956239121885867,0.1854445319921587,0.38877912391436853,
    0.5727042902682571,0.19178860193593794,0.4153338145563784,0.6365081754393465,0.005970365451990145,
    0.35634256423348254,0.22042364138234116,0.20979241153998585,0.8287415038352453,0.6984641963412034,
    0.05974170895963615,0.552216901543784,0.24070623413629832,0.025207974983498915,0.8846354387701332,
    0.13073848426080437,0.5411990852645557,0.27647797964142207,0.5011804871957674,0.04214445837531333,
    0.3155458416589393
  };
  std::vector<DimensionType> innerA = {
    3,5,6,7,8,
    0,3,5,6,7,
    2,3,6,7,1,
    2,6,7,0,4,
    5,6,4,5,7,
    8
  };
  std::vector<OrdinalType> outerA = {
    0,5,10,12,14,
    14,18,20,22,26
  };
  //input B:
  DimensionType rowsB = 9, colsB = 9;
  std::vector<DataType> valuesB = {
    0.22329107915353885,0.36072883534777267,0.22308094169128878,0.6887261618213565,0.941010860245836,
    0.07799233938998706,0.2934872561021975,0.46426408804015173,0.21506214371433952,0.12087875153895689,
    0.917687007744725,0.04321572973662613,0.7524449169599731,0.44335096201326485,0.4679961473799611,
    0.3406742720441993,0.27149077328868065,0.21542897828136154,0.018399946630616282,0.7423517360658496,
    0.420103554589757,0.9720455259033695,0.6756664492813946,0.19916263217137087,0.39386732248283196,
    0.33918540311781975,0.4072471068550777
  };
  std::vector<DimensionType> innerB = {
    0,4,5,6,0,
    2,6,8,0,2,
    3,7,8,7,8,
    8,2,4,6,0,
    2,4,1,8,1,
    2,5
  };
  std::vector<OrdinalType> outerB = {
    0,4,8,13,15,
    16,19,22,24,27
  };
  //expected add output:
  DimensionType rowsAdd = 9, colsAdd = 9;
  std::vector<DataType> valuesAdd = {
    0.22329107915353885,0.5678649545949946,0.36072883534777267,0.8494815678832063,1.1843500740099433,
    0.1854445319921587,0.38877912391436853,1.513715150514093,0.07799233938998706,0.19178860193593794,
    0.4153338145563784,0.929995431541544,0.005970365451990145,0.46426408804015173,0.21506214371433952,
    0.4772213157724394,1.138110649127066,0.04321572973662613,0.7524449169599731,0.20979241153998585,
    1.2720924658485102,0.4679961473799611,0.3406742720441993,0.6984641963412034,0.3312324822483168,
    0.21542897828136154,0.5706168481744003,0.24070623413629832,0.7675597110493485,0.420103554589757,
    1.8566809646735027,0.6756664492813946,0.13073848426080437,0.5411990852645557,0.19916263217137087,
    0.39386732248283196,0.33918540311781975,0.27647797964142207,0.908427594050845,0.04214445837531333,
    0.3155458416589393
  };
  std::vector<DimensionType> innerAdd = {
    0,3,4,5,6,
    7,8,0,2,3,
    5,6,7,8,0,
    2,3,7,8,6,
    7,8,8,1,2,
    4,6,7,0,2,
    4,1,5,6,8,
    1,2,4,5,7,
    8
  };
  std::vector<OrdinalType> outerAdd = {
    0,7,14,19,22,
    23,28,31,35,41
  };
  //expected sub output:
  DimensionType rowsSub = 9, colsSub = 9;
  std::vector<DataType> valuesSub = {
    -0.22329107915353885,0.5678649545949946,-0.36072883534777267,0.4033196845006287,-0.19310224963276978,
    0.1854445319921587,0.38877912391436853,-0.36830656997757893,-0.07799233938998706,0.19178860193593794,
    0.4153338145563784,0.34302091933714896,0.005970365451990145,-0.46426408804015173,-0.21506214371433952,
    0.23546381269452565,-0.6972633663623838,-0.04321572973662613,-0.7524449169599731,0.20979241153998585,
    0.3853905418219804,-0.4679961473799611,-0.3406742720441993,0.6984641963412034,-0.2117490643290445,
    -0.21542897828136154,0.5338169549131677,0.24070623413629832,-0.7171437610823507,-0.420103554589757,
    -0.08741008713323628,-0.6756664492813946,0.13073848426080437,0.5411990852645557,-0.19916263217137087,
    -0.39386732248283196,-0.33918540311781975,0.27647797964142207,0.09393338034068965,0.04214445837531333,
    0.3155458416589393
  };
  std::vector<DimensionType> innerSub = {
    0,3,4,5,6,
    7,8,0,2,3,
    5,6,7,8,0,
    2,3,7,8,6,
    7,8,8,1,2,
    4,6,7,0,2,
    4,1,5,6,8,
    1,2,4,5,7,
    8
  };
  std::vector<OrdinalType> outerSub = {
    0,7,14,19,22,
    23,28,31,35,41
  };
  //expected times output:
  DimensionType rowsTimes = 9, colsTimes = 9;
  std::vector<DataType> valuesTimes = {
    0.13973804156690592,0.34134915474853034,0.5389209568518136,0.18680703789630992,0.04307424428473392,
    0.202279911896357,0.3674233429856758,0.01621932276303892,0.010160761516929911,0.018713183991704924,
    0.8599059203120721,0.20410430342269456
  };
  std::vector<DimensionType> innerTimes = {
    5,6,0,6,2,
    3,7,2,6,0,
    4,5
  };
  std::vector<OrdinalType> outerTimes = {
    0,2,4,6,7,
    7,9,11,11,12
  };
  //expected matmul output:
  DimensionType rowsMatmul = 9, colsMatmul = 9;
  std::vector<DataType> valuesMatmul = {
    0.15832917341977834,0.2784260410431654,0.3679272716489455,0.011525738091315902,0.61671385326897,
    0.5101435615121522,0.30269223209590035,0.25176347391330983,0.0040339756258584904,0.3801586455143588,
    0.09094540051585127,0.0850296611714772,0.4020785477168084,0.127759412383684,0.9147808150794559,
    0.6003927080673659,0.37128556611236446,0.11312463740689836,0.32701094150350707,0.04307424428473392,
    0.07663579576071748,0.1650543392934877,0.5599528292684834,0.20392777500592166,0.0881345378139052,
    0.15573976092014968,0.16263712653876872,0.5367799684738567,0.002581781548403812,0.05482419013272473,
    0.41716387555087,0.20499034046980633,0.29368464308972775,1.0800497497939021,0.30137253412751597,
    0.01736139185767398,0.005623418797449388,0.009093243457273352,0.005628715937340883,0.4017600805033927,
    0.002405581132966467,0.5542350075407665,0.2628539516509206,0.12850513109575348,0.15275879232213357,
    0.009221694216708386,0.10796880029113916,0.24309442153121563,0.10258253571205661
  };
  std::vector<DimensionType> innerMatmul = {
    5,1,0,6,4,
    2,8,7,1,2,
    8,7,6,5,4,
    0,8,7,3,2,
    0,8,1,4,2,
    0,1,4,7,3,
    8,6,2,0,8,
    6,5,4,0,0,
    6,4,2,5,1,
    6,4,2,8
  };
  std::vector<OrdinalType> outerMatmul = {
    0,8,16,21,26,
    26,34,39,43,49
  };
  //expected matdiv output:
  DimensionType rowsMatdiv = 9, colsMatdiv = 9;
  std::vector<DataType> valuesMatdiv = {
    0.9059784554711483,-0.34595459132886885,0.6188002552096269,0.35796161731868736,0.1092351580394686,
    -1.457280770109116,-0.013242225158781595,-0.6073325820616003,1.0418590876432745,0.75233076177222,
    0.45588364749947674,0.2089913013013781,-0.006905017479728437,-0.8662673908474214,-0.8390102239192025,
    -0.09324695373668745,-0.3542748611619327,0.6077468828894173,-0.12600874931922013,0.20132202151985548,
    0.24019479356479773,-0.023413038816255987,-0.7491879800305157,1.5054432027338018,-0.28688070075817956,
    -0.04023673742610865,0.06902480087962026,0.25352943521785826,0.09194258243841574,1.8692674085377303,
    -2.740503535591017,0.4454404482464987,-0.1928059474058856,0.08095626192438211,-0.13887780711720785,
    0.6411312440866531,0.2869017551565289,0.5429248039594792,-1.8608417112071853,1.4375602396816316,
    -0.5565239962749926,1.2384652400949965,-0.35119748985584437,0.3892239016683549,-0.849036578749672,
    1.084390966622367,-1.0264849369787539,0.9931281298056186,0.12428581361220088,-0.21320822923324742,
    0.7017997832133478,0.15536431409884194,-0.2333336036369965,0.6659675818574476,-0.4080345247259106,
    0.036958157678230574,-0.06340050505591377,0.3074633356286053,-0.5858062323841562,0.09505890814792546,
    1.9559761584992423,-2.164755153800923,0.6500910326122288,-0.6192091230687915,1.0622329034792495
  };
  std::vector<DimensionType> innerMatdiv = {
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,4,5,6,
    7,8,0,1,3,
    4,5,6,7,8,
    0,1,3,4,5,
    6,7,8,0,1,
    4,5,6,7,8,
    0,1,4,5,6,
    7,8,0,1,3,
    4,5,6,7,8
  };
  std::vector<OrdinalType> outerMatdiv = {
    0,9,18,27,35,
    35,43,50,57,65
  };

  SpMatMap left(rowsA, colsA, innerA.size(), outerA.data(), innerA.data(),
      valuesA.data());
  SpMatMap right(rowsB, colsB, innerB.size() , outerB.data(), innerB.data(),
      valuesB.data());

  SpMat t = add(left, right);
  compareCSR(t, rowsAdd, colsAdd, outerAdd, innerAdd, valuesAdd);

  t = std::move(sub(left, right));
  compareCSR(t, rowsSub, colsSub, outerSub, innerSub, valuesSub);

  t = std::move(times(left, right));
  compareCSR(t, rowsTimes, colsTimes, outerTimes, innerTimes, valuesTimes);

  t = std::move(matmul(left, right));
  compareCSR(t, rowsMatmul, colsMatmul, outerMatmul, innerMatmul, valuesMatmul, false);

#ifdef EIGEN
  t = std::move(matdiv(left, right));
  compareCSR(t, rowsMatdiv, colsMatdiv, outerMatdiv, innerMatdiv, valuesMatdiv);
#endif
}

//  TEST: with given inputs, compare output with Numpy's output
//  Inputs generated randomly by testall.py
//  with ./testall.py -p 9 -z 0.9 -s 88
TEST(TestAll, 9by9Dense) {
  //input A:
  DimensionType rowsA = 9, colsA = 9;
  std::vector<DataType> valuesA = {
    0.6027633760716439,0.5448831829968969,0.6458941130666561,0.8917730007820798,0.9636627605010293,
    0.7917250380826646,0.5680445610939323,0.08712929970154071,0.832619845547938,0.8700121482468192,
    0.7991585642167236,0.46147936225293185,0.11827442586893322,0.6399210213275238,0.1433532874090464,
    0.5218483217500717,0.26455561210462697,0.7742336894342167,0.5684339488686485,0.6176354970758771,
    0.6120957227224214,0.6169339968747569,0.9437480785146242,0.6818202991034834,0.359507900573786,
    0.43703195379934145,0.6976311959272649,0.6667667154456677,0.1289262976548533,0.3637107709426226,
    0.10204481074802807,0.2088767560948347,0.16130951788499626,0.6531083254653984,0.2532916025397821,
    0.24442559200160274,0.6563295894652734,0.1381829513486138,0.1965823616800535,0.8209932298479351,
    0.09710127579306127,0.8379449074988039,0.9764594650133958,0.9767610881903371,0.7392635793983017,
    0.039187792254320675,0.2828069625764096,0.11872771895424405
  };
  std::vector<DimensionType> innerA = {
    2,3,5,7,8,
    1,3,6,8,1,
    3,4,6,7,8,
    1,3,4,6,8,
    0,1,2,3,4,
    5,6,8,2,4,
    8,0,1,2,3,
    5,8,0,1,3,
    4,5,7,0,2,
    3,4,7
  };
  std::vector<OrdinalType> outerA = {
    0,5,9,15,20,
    28,31,37,43,48
  };
  //input B:
  DimensionType rowsB = 9, colsB = 9;
  std::vector<DataType> valuesB = {
    0.7220555994703479,0.8663823259286292,0.9755215050028858,0.855803342392611,0.011714084185001972,
    0.3599780644783639,0.729990562424058,0.17162967726144052,0.5210366062041293,0.05433798833925363,
    0.19999652489640007,0.01852179446061397,0.22392468806038013,0.3453516806969027,0.9280812934655909,
    0.7044144019235328,0.03183892953130785,0.16469415649791275,0.5772285886041676,0.23789282137450862,
    0.9342139979247938,0.613965955965896,0.5356328030249583,0.7301220295167696,0.31194499547960186,
    0.3982210622160919,0.20984374897512215,0.18619300588033616,0.9443723899839336,0.7395507950492876,
    0.4904588086175671,0.22741462797332324,0.25435648177039294,0.05802916032387562,0.4344166255581208,
    0.3117958819941026,0.6963434888154595,0.1796036775596348,0.02467872839133123,0.06724963146324858,
    0.6793927734985673,0.4536968445560453,0.5365792111087222,0.8966712930403421,0.9903389473967044,
    0.21689698439847394,0.6630782031001008,0.26332237673715064,0.02065099946572868,0.32001715082246784,
    0.38346389417189797,0.5883171135536057,0.8310484552361904,0.6289818435911487,0.8726506554473953,
    0.7980468339125637,0.1856359443059522,0.9527916569719446,0.21550767711355845,0.7308558067701578,
    0.25394164259502583,0.025662718054531575,0.2074700754411094,0.42468546875150626,0.37416998033422555,
    0.4635754243648107,0.2776287062947319,0.5867843464581688,0.8638556059232314,0.5173791071541142
  };
  std::vector<DimensionType> innerB = {
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,4,5,6,
    7,8,0,2,3,
    4,5,6,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,5,6,7,
    8,0,1,2,3,
    4,5,6,7,0,
    1,2,3,4,5,
    7,8,0,2,4,
    5,8,0,1,2,
    3,4,5,6,8
  };
  std::vector<OrdinalType> outerB = {
    0,9,17,24,33,
    41,49,57,62,70
  };
  //expected add output:
  DimensionType rowsAdd = 9, colsAdd = 9;
  std::vector<DataType> valuesAdd = {
    0.7220555994703479,0.8663823259286292,1.5782848810745298,1.4006865253895078,0.011714084185001972,
    1.0058721775450201,0.729990562424058,1.0634026780435204,1.4846993667051587,0.05433798833925363,
    0.9917215629790647,0.01852179446061397,0.5680445610939323,0.22392468806038013,0.3453516806969027,
    1.0152105931671316,0.7044144019235328,0.8644587750792458,0.16469415649791275,0.8700121482468192,
    0.5772285886041676,1.0370513855912322,1.3956933601777255,0.613965955965896,0.6539072288938915,
    0.6399210213275238,0.873475316925816,0.31194499547960186,0.9200693839661636,0.20984374897512215,
    0.45074861798496313,1.7186060794181501,0.7395507950492876,1.0588927574862157,0.22741462797332324,
    0.87199197884627,0.670124883046297,1.0513506224328777,1.2555439605087266,1.3781637879189428,
    0.359507900573786,0.6166356313589763,0.7223099243185961,0.06724963146324858,1.346159488944235,
    0.4536968445560453,0.5365792111087222,1.0255975906951953,0.9903389473967044,0.5806077553410965,
    0.6630782031001008,0.26332237673715064,0.02065099946572868,0.10204481074802807,0.5288939069173025,
    0.5447734120568942,1.2414254390190043,1.0843400577759725,0.6289818435911487,1.1170762474489981,
    0.7980468339125637,0.8419655337712256,1.0909746083205585,0.1965823616800535,0.21550767711355845,
    0.8209932298479351,0.8279570825632191,1.0918865500938297,0.9764594650133958,0.025662718054531575,
    1.1842311636314466,0.42468546875150626,1.1134335597325271,0.5027632166191314,0.5604356688711415,
    0.5867843464581688,0.8638556059232314,0.11872771895424405,0.5173791071541142
  };
  std::vector<DimensionType> innerAdd = {
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,4,5,6,
    7,8,0,1,2,
    3,4,5,6,7,
    8,0,1,2,3,
    4,5,6,7,8,
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    7,8,0,1,2,
    3,4,5,7,8,
    0,1,2,3,4,
    5,6,7,8
  };
  std::vector<OrdinalType> outerAdd = {
    0,9,18,27,36,
    45,54,62,70,79
  };
  //expected sub output:
  DimensionType rowsSub = 9, colsSub = 9;
  std::vector<DataType> valuesSub = {
    -0.7220555994703479,-0.8663823259286292,-0.37275812893124194,-0.31092015939571416,-0.011714084185001972,
    0.2859160485882922,-0.729990562424058,0.7201433235206393,0.4426261542969,-0.05433798833925363,
    0.5917285131862645,-0.01852179446061397,0.5680445610939323,-0.22392468806038013,-0.3453516806969027,
    -0.8409519937640502,-0.7044144019235328,0.8007809160166302,-0.16469415649791275,0.8700121482468192,
    -0.5772285886041676,0.561265742842215,-0.4727346356718619,-0.613965955965896,-0.4173583771560251,
    0.6399210213275238,-0.5867687421077232,-0.31194499547960186,0.1236272595339798,-0.20984374897512215,
    0.07836260622429081,-0.17013870054971691,-0.7395507950492876,0.07797514025108143,-0.22741462797332324,
    0.3632790153054841,0.5540665623985458,0.18251737131663615,0.6319521965205216,-0.014523189711976081,
    0.359507900573786,0.25742827623970665,0.6729524675359336,-0.06724963146324858,-0.012626058052899625,
    -0.4536968445560453,-0.5365792111087222,-0.7677449953854888,-0.9903389473967044,0.14681378654414867,
    -0.6630782031001008,-0.26332237673715064,-0.02065099946572868,0.10204481074802807,-0.11114039472763315,
    -0.2221543762869017,0.06479121191179271,-0.5777568526964083,-0.6289818435911487,-0.6282250634457925,
    -0.7980468339125637,0.4706936451593212,-0.8146087056233308,0.1965823616800535,-0.21550767711355845,
    0.8209932298479351,-0.6337545309770966,0.5840032649037781,0.9764594650133958,-0.025662718054531575,
    0.7692910127492277,-0.42468546875150626,0.3650935990640761,-0.42438763211049,0.005178256281677673,
    -0.5867843464581688,-0.8638556059232314,0.11872771895424405,-0.5173791071541142
  };
  std::vector<DimensionType> innerSub = {
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,4,5,6,
    7,8,0,1,2,
    3,4,5,6,7,
    8,0,1,2,3,
    4,5,6,7,8,
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    7,8,0,1,2,
    3,4,5,7,8,
    0,1,2,3,4,
    5,6,7,8
  };
  std::vector<OrdinalType> outerSub = {
    0,9,18,27,36,
    45,54,62,70,79
  };
  //expected times output:
  DimensionType rowsTimes = 9, colsTimes = 9;
  std::vector<DataType> valuesTimes = {
    0.5880086357860305,0.46631284922226907,0.23250771267970438,0.1530547123146947,0.5021035742567589,
    0.15834225629000293,0.08086307316575703,0.026509724588769223,0.1901140855671178,0.4311204799700956,
    0.06335166225434434,0.10466539314099374,0.20781099300299843,0.04925840464027274,0.7311649196970698,
    0.27879343733989637,0.15709959205272792,0.035519400827417905,0.2680063851144161,0.2942567645207068,
    0.47478112582291976,0.07849254611343413,0.01721665080160855,0.4529964880831622,0.11560451002508133,
    0.07888776941069894,0.06684414435850854,0.06185637589517209,0.3842348048756321,0.21049759501498505,
    0.21329815306831626,0.12183836311632397,0.13165956318071934,0.07096703125814939,0.21278910621438324,
    0.20264869665478935,0.2766102389652717,0.01816649742421675,0.07851533115123126
  };
  std::vector<DimensionType> innerTimes = {
    2,3,5,7,8,
    1,6,8,3,4,
    6,8,1,3,4,
    6,8,0,1,2,
    3,5,6,8,2,
    4,0,1,2,3,
    5,8,0,4,5,
    0,2,3,4
  };
  std::vector<OrdinalType> outerTimes = {
    0,5,8,12,17,
    24,26,32,35,39
  };
  //expected matmul output:
  DimensionType rowsMatmul = 9, colsMatmul = 9;
  std::vector<DataType> valuesMatmul = {
    0.1372527653340163,0.9728108847275339,1.1001496865721578,1.5926464443059638,1.9932438152577387,
    2.137073023960996,1.3312309870587344,1.594184896292937,1.6118903695414364,0.5641566924612805,
    0.616647935859546,0.7564174234901662,1.7326509772681935,1.258123111743279,0.999694300071779,
    0.49666577805048284,0.7715620547532582,0.4208459763431296,0.6348927243311289,0.6570432534118456,
    0.9200122827834654,1.334621621973019,1.3241942153483226,1.5314019306410935,0.5888292385121475,
    0.7989495158881452,1.0006502569439064,1.3471077858701785,1.0349887751681162,0.933465133029328,
    1.1667262852677243,1.3733918257247346,0.8957017154837055,0.8721038667170857,1.0263341707039122,
    0.4658606140576681,1.919767134424647,1.2846311810585815,2.559238858521778,2.8714620571665037,
    2.3895736737316273,2.447307371006614,2.460252914747145,1.866569755368813,1.4243403686007405,
    0.02445941530510539,0.20133895407822178,0.3940302525983805,0.16618505534565312,0.2043584472023798,
    0.14877532076184824,0.3312437343031729,0.22600557027688123,0.06351048306548185,0.9948147124467186,
    0.2121280461316931,1.40757847034258,1.266405058470421,1.1231436278842608,0.9276107845225894,
    1.101644206790668,0.7239818544899368,0.5932259078733677,0.37811080648162837,0.3727320368578142,
    0.9090260048963678,1.5458255464983928,1.7163604629838027,1.1685859477215788,1.3027932287302026,
    0.9777780780329869,1.6827324908036918,1.253832459013496,0.19557173153275748,1.1351995134473827,
    0.9154196816294902,0.8258529574972676,1.2160081824502207,1.5015636138143509,0.9847099940871279,
    0.9687864710318123
  };
  std::vector<DimensionType> innerMatmul = {
    7,1,8,6,5,
    4,3,2,0,3,
    8,7,6,5,4,
    2,1,0,3,8,
    7,6,5,4,2,
    1,0,3,8,7,
    6,5,4,2,1,
    0,8,7,6,5,
    4,3,2,1,0,
    7,1,8,6,5,
    4,3,2,0,8,
    7,6,5,4,3,
    2,1,0,8,7,
    6,5,4,3,2,
    1,0,8,7,6,
    5,4,3,2,1,
    0
  };
  std::vector<OrdinalType> outerMatmul = {
    0,9,18,27,36,
    45,54,63,72,81
  };
  //expected matdiv output:
  DimensionType rowsMatdiv = 9, colsMatdiv = 9;
  std::vector<DataType> valuesMatdiv = {
    1.1631218999151725,-1.6218676004969415,0.3417316985256522,-1.7024520790811846,-1.7288548010919114,
    -2.6229361128788855,2.9975775422329294,-0.4577513043057535,2.363173401692074,0.154898223287194,
    0.12131770953094956,-0.38983727244238126,1.008231452581309,1.682397176515885,-0.13030651551886585,
    -0.56610623047162,-0.12854351368049194,-0.499424675761331,-1.0060875808627756,2.727590577272456,
    -0.12906128618582582,1.9025062055320332,3.7851180325523575,2.399629951065199,-2.312544563989259,
    0.23361013729902716,-3.7830125012221827,-0.7697176994721905,2.0184822272314458,0.5046392907749395,
    1.6592006317280483,2.950877886758039,1.6938230344906637,-2.381429952655239,0.24265460704396444,
    -2.7156289998160874,0.7548550892357544,-0.06462979912246614,0.42200986321629036,0.04233323493404506,
    0.014567242221067478,0.045587410252570404,-0.11976444754478599,0.018137890039780686,-0.060881595751156925,
    -0.107538324398346,0.4714378621700853,0.4467477728939888,0.4288807686812489,0.5702083732232877,
    0.5056651634000462,-0.5763488891876425,-0.00401219651438774,-1.1065469453032655,1.3408024470132518,
    -1.6081161134075086,0.36408777073040655,-0.6141026144951818,-1.4963572078298868,-1.6520708331287333,
    1.474927173754522,-0.45982997128322606,1.2638911382988707,-0.7494610149370259,0.1365211979413541,
    -0.7851480880651844,-0.8278409864118115,0.1463494788707118,-0.24443782220619348,1.494136377994539,
    0.387968735046168,1.5138223254674454,1.0737128980518043,-0.3652531719637204,0.4361094038685681,
    -0.6350625141093069,-1.0097501146705488,-0.4825269205062819,0.5188026435553031,0.5360289845910745,
    -0.24883633162750973
  };
  std::vector<DimensionType> innerMatdiv = {
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,4,5,6,
    7,8,0,1,2,
    3,4,5,6,7,
    8,0,1,2,3,
    4,5,6,7,8,
    0,1,2,3,4,
    5,6,7,8,0,
    1,2,3,4,5,
    6,7,8,0,1,
    2,3,4,5,6,
    7,8,0,1,2,
    3,4,5,6,7,
    8
  };
  std::vector<OrdinalType> outerMatdiv = {
    0,9,18,27,36,
    45,54,63,72,81
  };

  SpMatMap left(rowsA, colsA, innerA.size(), outerA.data(), innerA.data(),
      valuesA.data());
  SpMatMap right(rowsB, colsB, innerB.size() , outerB.data(), innerB.data(),
      valuesB.data());

  SpMat t = add(left, right);
  compareCSR(t, rowsAdd, colsAdd, outerAdd, innerAdd, valuesAdd);

  t = std::move(sub(left, right));
  compareCSR(t, rowsSub, colsSub, outerSub, innerSub, valuesSub);

  t = std::move(times(left, right));
  compareCSR(t, rowsTimes, colsTimes, outerTimes, innerTimes, valuesTimes);

  t = std::move(matmul(left, right));
  compareCSR(t, rowsMatmul, colsMatmul, outerMatmul, innerMatmul, valuesMatmul, false);

#ifdef EIGEN
  t = std::move(matdiv(left, right));
  compareCSR(t, rowsMatdiv, colsMatdiv, outerMatdiv, innerMatdiv, valuesMatdiv, true, 1e-4);
#endif
}

//  TEST: with given inputs, compare output with Numpy's output
//  Inputs generated randomly by testall.py
//  Note: matdiv test is excluded as it turns to be not invertible when
//  the matrix is really sparse.
//  with ./testall.py -p 43 -z 0.01 -s 4
TEST(TestAll, 43by43Sparse) {
  //input A:
  DimensionType rowsA = 43, colsA = 43;
  std::vector<DataType> valuesA = {
    0.06554234218850441,0.6103696958683686,0.6970228612065846,0.9095057177332495,0.41703501720955627,
    0.6826855142190257,0.5187110584773698,0.05644661549091945,0.9917959130956936,0.45151410108068846,
    0.8191325647022716,0.348824920575207,0.5249464237156384,0.2754619690068991,0.5174819930842142,
    0.0504396584190101,0.20651229977422791,0.5521632296735103,0.4016583600824969,0.8599312806059249,
    0.5598885187652017,0.01632661169945937,0.3570755130818438
  };
  std::vector<DimensionType> innerA = {
    7,27,42,34,33,
    42,24,18,0,23,
    18,28,6,16,8,
    25,31,13,36,18,
    14,33,7
  };
  std::vector<OrdinalType> outerA = {
    0,1,2,3,3,
    3,3,3,3,4,
    4,4,5,5,5,
    5,5,5,6,6,
    6,7,8,8,10,
    10,10,10,12,13,
    14,17,17,17,17,
    18,19,20,20,22,
    22,23,23,23
  };
  //input B:
  DimensionType rowsB = 43, colsB = 43;
  std::vector<DataType> valuesB = {
    0.5472322491757223,0.6977288245972708,0.9825118679501008,0.94109751393948,0.1943989049435213,
    0.387205883659379,0.25830878015809644,0.19604592789013897,0.06416658760401361,0.20689825522200944,
    0.39623665373638217,0.9305614253986958,0.842919424345656,0.3874165861974992,0.25949321594878283,
    0.22639770222115696,0.7076027014376451,0.003109327148096863,0.6241579904311372,0.6196648505696182,
    0.542561400998539,0.7821820064485294,0.6470209574230928,0.3345292456414355
  };
  std::vector<DimensionType> innerB = {
    1,4,12,11,16,
    38,41,36,8,25,
    32,23,5,23,33,
    14,16,18,4,33,
    5,0,14,11
  };
  std::vector<OrdinalType> outerB = {
    0,2,2,2,2,
    2,2,2,2,3,
    4,6,6,6,7,
    8,8,9,11,12,
    12,12,12,12,14,
    15,15,16,16,16,
    16,16,17,17,18,
    20,21,21,21,23,
    23,24,24,24
  };
  //expected add output:
  DimensionType rowsAdd = 43, colsAdd = 43;
  std::vector<DataType> valuesAdd = {
    0.5472322491757223,0.6977288245972708,0.06554234218850441,0.6103696958683686,0.6970228612065846,
    0.9825118679501008,0.9095057177332495,0.94109751393948,0.1943989049435213,0.387205883659379,
    0.41703501720955627,0.25830878015809644,0.19604592789013897,0.06416658760401361,0.20689825522200944,
    0.39623665373638217,0.6826855142190257,0.9305614253986958,0.5187110584773698,0.05644661549091945,
    0.9917959130956936,0.842919424345656,0.8389306872781876,0.25949321594878283,0.22639770222115696,
    0.8191325647022716,0.348824920575207,0.5249464237156384,0.2754619690068991,0.5174819930842142,
    0.0504396584190101,0.20651229977422791,0.7076027014376451,0.003109327148096863,0.6241579904311372,
    0.5521632296735103,0.6196648505696182,0.542561400998539,0.4016583600824969,0.8599312806059249,
    0.7821820064485294,1.2069094761882946,0.01632661169945937,0.3570755130818438,0.3345292456414355
  };
  std::vector<DimensionType> innerAdd = {
    1,4,7,27,42,
    12,34,11,16,38,
    33,41,36,8,25,
    32,42,23,24,18,
    0,5,23,33,14,
    18,28,6,16,8,
    25,31,16,18,4,
    13,33,5,36,18,
    0,14,33,7,11
  };
  std::vector<OrdinalType> outerAdd = {
    0,3,4,5,5,
    5,5,5,5,7,
    8,10,11,11,12,
    13,13,14,17,18,
    18,19,20,20,23,
    24,24,25,27,28,
    29,32,33,33,34,
    37,39,40,40,43,
    43,45,45,45
  };
  //expected sub output:
  DimensionType rowsSub = 43, colsSub = 43;
  std::vector<DataType> valuesSub = {
    -0.5472322491757223,-0.6977288245972708,0.06554234218850441,0.6103696958683686,0.6970228612065846,
    -0.9825118679501008,0.9095057177332495,-0.94109751393948,-0.1943989049435213,-0.387205883659379,
    0.41703501720955627,-0.25830878015809644,-0.19604592789013897,-0.06416658760401361,-0.20689825522200944,
    -0.39623665373638217,0.6826855142190257,-0.9305614253986958,0.5187110584773698,0.05644661549091945,
    0.9917959130956936,-0.842919424345656,0.06409751488318927,-0.25949321594878283,-0.22639770222115696,
    0.8191325647022716,0.348824920575207,0.5249464237156384,0.2754619690068991,0.5174819930842142,
    0.0504396584190101,0.20651229977422791,-0.7076027014376451,-0.003109327148096863,-0.6241579904311372,
    0.5521632296735103,-0.6196648505696182,-0.542561400998539,0.4016583600824969,0.8599312806059249,
    -0.7821820064485294,-0.08713243865789111,0.01632661169945937,0.3570755130818438,-0.3345292456414355
  };
  std::vector<DimensionType> innerSub = {
    1,4,7,27,42,
    12,34,11,16,38,
    33,41,36,8,25,
    32,42,23,24,18,
    0,5,23,33,14,
    18,28,6,16,8,
    25,31,16,18,4,
    13,33,5,36,18,
    0,14,33,7,11
  };
  std::vector<OrdinalType> outerSub = {
    0,3,4,5,5,
    5,5,5,5,7,
    8,10,11,11,12,
    13,13,14,17,18,
    18,19,20,20,23,
    24,24,25,27,28,
    29,32,33,33,34,
    37,39,40,40,43,
    43,45,45,45
  };
  //expected times output:
  DimensionType rowsTimes = 43, colsTimes = 43;
  std::vector<DataType> valuesTimes = {
    0.1749240516607129,0.36225960546165814
  };
  std::vector<DimensionType> innerTimes = {
    23,14
  };
  std::vector<OrdinalType> outerTimes = {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,2,
    2,2,2,2
  };
  //expected matmul output:
  DimensionType rowsMatmul = 43, colsMatmul = 43;
  std::vector<DataType> valuesMatmul = {
    0.5635887246713874,0.5676752610660141,0.0012966983007167158,0.13460200071248982,0.052527042970162106,
    0.1749240516607129,0.38059000616688027,0.6920045966846352,0.5427427082466456,0.7622531669998353,
    0.017675454565855276,0.14612866120034446,0.5084321996557124,0.14262861030511928,0.8002188782255754,
    5.076477699356488e-05,0.10976386417635946
  };
  std::vector<DimensionType> innerMatmul = {
    33,4,18,33,23,
    23,5,4,1,23,
    8,16,12,41,23,
    18,36
  };
  std::vector<OrdinalType> outerMatmul = {
    0,0,0,0,0,
    0,0,0,0,2,
    2,2,3,3,3,
    3,3,3,3,3,
    3,4,5,5,9,
    9,9,9,10,10,
    11,13,13,13,13,
    14,14,15,15,17,
    17,17,17,17
  };

  SpMatMap left(rowsA, colsA, innerA.size(), outerA.data(), innerA.data(),
      valuesA.data());
  SpMatMap right(rowsB, colsB, innerB.size() , outerB.data(), innerB.data(),
      valuesB.data());

  SpMat t = add(left, right);
  compareCSR(t, rowsAdd, colsAdd, outerAdd, innerAdd, valuesAdd);

  t = std::move(sub(left, right));
  compareCSR(t, rowsSub, colsSub, outerSub, innerSub, valuesSub);

  t = std::move(times(left, right));
  compareCSR(t, rowsTimes, colsTimes, outerTimes, innerTimes, valuesTimes);

  t = std::move(matmul(left, right));
  compareCSR(t, rowsMatmul, colsMatmul, outerMatmul, innerMatmul, valuesMatmul, false);
}


/** Test Array Class: constructor */
TEST(ArrayTest, ListConstructor) {
  Array<int32_t> a {0, 1, 2};
  EXPECT_EQ(a.size(), 3);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
}

/** Test DimData Class : move constructor */
TEST(ArrayTest, MoveConstructor) {
  Array<int32_t> a {0, 1, 2};

  Array<int32_t> b { std::move(a) };

  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(b.size(), 3);
  EXPECT_EQ(b[0], 0);
  EXPECT_EQ(b[1], 1);
  EXPECT_EQ(b[2], 2);
}

/** Test DimData Class : move assignment */
TEST(DimDataTest, MoveAssignment) {
  DimData dim1 {{0,1,2}, {0,1}};

  DimData dim2 {{5,5},{2,1}};

  Array<DimensionType> known_inner1 {0,1,2};
  Array<OrdinalType> known_outer1 {0,1};
  Array<DimensionType> known_inner2 {5,5,};
  Array<OrdinalType> known_outer2 {2,1};

  EXPECT_EQ(dim2.inner(), known_inner2);
  EXPECT_EQ(dim2.outer(), known_outer2);
  EXPECT_EQ(dim1.inner(), known_inner1);
  EXPECT_EQ(dim1.outer(), known_outer1);

  dim2 = std::move(dim1);

  Array<DimensionType> empty_inner {};
  Array<OrdinalType> empty_outer {};

  EXPECT_EQ(dim2.inner(), known_inner1);
  EXPECT_EQ(dim2.outer(), known_outer1);
  EXPECT_EQ(dim1.inner(), empty_inner);
  EXPECT_EQ(dim1.outer(), empty_outer);
}

/** Test DimData Class : move constructor */
TEST(DimDataTest, MoveConstructor) {
  DimData dim {{5,12,999,2345}, {0,0,4,4,9,9,1000}};

  DimData dim2 { std::move(dim) };

  Array<DimensionType> known_inner {5,12,999,2345};
  Array<OrdinalType> known_outer {0,0,04,4,9,9,1000};
  Array<DimensionType> empty_inner {};
  Array<OrdinalType> empty_outer {};

  EXPECT_EQ(dim2.inner(), known_inner);
  EXPECT_EQ(dim2.outer(), known_outer);
  EXPECT_EQ(dim.inner(), empty_inner);
  EXPECT_EQ(dim.outer(), empty_outer);
}

/** Test SparseFloatTensor Class : move constructor */
TEST(SparseFloatTensor, MoveConstructor) {
  SparseFloatTensor t1;
  t1.shape() = {400,31,56, 349};
  t1.values() = {7.1, 1.2, 38.2, 5.3, 49.134, 2.66};
  t1.dims().push_back({{10, 123, 56, 22}, {0, 1, 2, 4, 4}});
  t1.dims().push_back({{10, 12, 22, 49, 28}, {0, 1, 3, 4, 5}});

  SparseFloatTensor known_t;
  known_t.shape() = {400,31,56, 349};
  known_t.values() = {7.1, 1.2, 38.2, 5.3, 49.134, 2.66};
  known_t.dims().push_back({{10, 123, 56, 22}, {0, 1, 2, 4, 4}});
  known_t.dims().push_back({{10, 12, 22, 49, 28}, {0, 1, 3, 4, 5}});

  SparseFloatTensor t2 { std::move(t1) };

  SparseFloatTensor empty;
  compareSparseFloatTensor(t1, empty);

  compareSparseFloatTensor(t2, known_t);
}

/** Test SparseFloatTensor Class : move assignment */
TEST(SparseFloatTensor, MoveAssignment) {
  SparseFloatTensor t1;
  t1.shape() = {400,31,56, 349};
  t1.values() = {7.1, 1.2, 38.2, 5.3, 49.134, 2.66};
  t1.dims().push_back({{10, 123, 56, 22}, {0, 1, 2, 4, 4}});
  t1.dims().push_back({{10, 12, 22, 49, 28}, {0, 1, 3, 4, 5}});

  SparseFloatTensor t2;
  t2.shape() = {4169};
  t2.values() = {2.2134266};
  t2.dims().push_back({{1, 22}, {}});
  t2.dims().push_back({{12, 49, 28}, {45}});

  SparseFloatTensor known_t;
  known_t.shape() = {400,31,56, 349};
  known_t.values() = {7.1, 1.2, 38.2, 5.3, 49.134, 2.66};
  known_t.dims().push_back({{10, 123, 56, 22}, {0, 1, 2, 4, 4}});
  known_t.dims().push_back({{10, 12, 22, 49, 28}, {0, 1, 3, 4, 5}});

  t2 = std::move(t1);

  SparseFloatTensor empty;
  compareSparseFloatTensor(t1, empty);

  compareSparseFloatTensor(t2, known_t);
}
/** Test toSparse2Ds function which convert SparseFloatTensor to a vector
 * of sparse 2D matrices */
TEST(ConvToSparse2D, basic) {
  SparseFloatTensor t1;
  t1.shape() = {4,3,5};
  t1.values() = {1, 2, 3, 4, 5};
  t1.dims().push_back({{0, 1, 0, 1}, {0, 1, 2, 4, 4}});
  t1.dims().push_back({{0, 1, 2, 2, 4}, {0, 1, 3, 4, 5}});

  auto sparse2Ds = t1.toSparse2Ds();

  // expected outputs
  std::vector<std::vector<OrdinalType>> outers = {
    {0, 1, 1, 1},
    {0, 0, 2, 2},
    {0, 1, 2, 2},
    {0, 0, 0, 0}
  };
  std::vector<std::vector<DimensionType>> inners = {
    {0},
    {1, 2},
    {2, 4},
    {}
  };
  std::vector<std::vector<DataType>> values = {
    {1},
    {2, 3},
    {4, 5},
    {}
  };

  EXPECT_EQ(sparse2Ds.size(), 4);

  for(size_t i=0; i<sparse2Ds.size(); i++) {
    compareCSR(sparse2Ds[i].get(), t1.shape()[1], t1.shape()[2], outers[i], inners[i], values[i]);
  }
}

/** Test toSparse2Ds function which convert SparseFloatTensor to a vector
 * of sparse matrices
 *
 * With unordered row ids */
TEST(ConvToSparse2D, unorderedRowIds) {
  SparseFloatTensor t1;
  t1.shape() = {4,3,5};
  t1.values() = {1, 2, 3, 5, 4};
  t1.dims().push_back({{0, 1, 1, 0}, {0, 1, 2, 4, 4}});
  t1.dims().push_back({{0, 1, 2, 4, 2}, {0, 1, 3, 4, 5}});

  auto sparse2Ds = t1.toSparse2Ds();

  // expected outputs
  std::vector<std::vector<OrdinalType>> outers = {
    {0, 1, 1, 1},
    {0, 0, 2, 2},
    {0, 1, 2, 2},
    {0, 0, 0, 0}
  };
  std::vector<std::vector<DimensionType>> inners = {
    {0},
    {1, 2},
    {2, 4},
    {}
  };
  std::vector<std::vector<DataType>> values = {
    {1},
    {2, 3},
    {4, 5},
    {}
  };

  EXPECT_EQ(sparse2Ds.size(), 4);

  for(size_t i=0; i<sparse2Ds.size(); i++) {
    compareCSR(sparse2Ds[i].get(), t1.shape()[1], t1.shape()[2], outers[i], inners[i], values[i]);
  }
}

/** Test toSparse2Ds function which convert SparseFloatTensor to a vector
 * of sparse matrices
 *
 * With empty matrices */
TEST(ConvToSparse2D, emptyMatrices) {
  SparseFloatTensor t1;
  t1.shape() = {2,1,1};
  t1.values() = {};
  t1.dims().push_back({{}, {0, 0, 0}});
  t1.dims().push_back({{}, {0}});

  auto sparse2Ds = t1.toSparse2Ds();

  // expected outputs
  std::vector<std::vector<OrdinalType>> outers = {
    {0, 0},
    {0, 0}
  };
  std::vector<std::vector<DimensionType>> inners = {
    {},
    {}
  };
  std::vector<std::vector<DataType>> values = {
    {},
    {}
  };

  EXPECT_EQ(sparse2Ds.size(), t1.shape()[0]);

  for(size_t i=0; i<sparse2Ds.size(); i++) {
    compareCSR(sparse2Ds[i].get(), t1.shape()[1], t1.shape()[2], outers[i], inners[i], values[i]);
  }
}

/** Test constructing SparseFloatTensor from a list of sparse
 * matrices */
TEST(GenFromSparse2D, basic) {
  // inputs
  std::vector<std::vector<OrdinalType>> outers = {
    {0, 1, 1, 1},
    {0, 0, 2, 2},
    {0, 1, 2, 2},
    {0, 0, 0, 0}
  };
  std::vector<std::vector<DimensionType>> inners = {
    {0},
    {1, 2},
    {2, 4},
    {}
  };
  std::vector<std::vector<DataType>> values = {
    {1},
    {2, 3},
    {4, 5},
    {}
  };
  std::vector<DimensionType> shape = {3, 5};

  // expected outputs
  SparseFloatTensor tExp;
  tExp.shape() = {4,3,5};
  tExp.values() = {1, 2, 3, 4, 5};
  tExp.dims().push_back({{0, 1, 0, 1}, {0, 1, 2, 4, 4}});
  tExp.dims().push_back({{0, 1, 2, 2, 4}, {0, 1, 3, 4, 5}});

  // generate the list of SpMat
  std::vector<SpMat> sparse2Ds;
  for (size_t i=0; i<outers.size(); i++)
    sparse2Ds.emplace_back(std::move(genSpMat(shape, outers[i], inners[i], values[i])));

  SparseFloatTensor t(sparse2Ds);

  compareSparseFloatTensor(tExp, t);
}


/** Test constructing SparseFloatTensor from a list of sparse
 * matrices
 *
 * Convert from empty sparse matrices */
TEST(GenFromSparse2D, emptySpMat) {
  // inputs
  std::vector<std::vector<OrdinalType>> outers = {
    {0, 0},
    {0, 0}
  };
  std::vector<std::vector<DimensionType>> inners = {
    {},
    {}
  };
  std::vector<std::vector<DataType>> values = {
    {},
    {}
  };
  std::vector<DimensionType> shape = {1, 1};

  // expected outputs
  SparseFloatTensor tExp;
  tExp.shape() = {2,1,1};
  tExp.values() = {};
  tExp.dims().push_back({{}, {0, 0, 0}});
  tExp.dims().push_back({{}, {0}});

  // generate the list of SpMat
  std::vector<SpMat> sparse2Ds;
  for (size_t i=0; i<outers.size(); i++)
    sparse2Ds.emplace_back(std::move(genSpMat(shape, outers[i], inners[i], values[i])));

  SparseFloatTensor t(sparse2Ds);
  compareSparseFloatTensor(tExp, t);
}

/** Test constructing SparseFloatTensor from a list of sparse
 * matrices
 *
 * Convert from empty list of sparse matrices */
TEST(GenFromSparse2D, emptyList) {
  // expected outputs
  SparseFloatTensor tExp;

  // generate the list of SpMat
  std::vector<SpMat> sparse2Ds = {};

  SparseFloatTensor t(sparse2Ds);
  compareSparseFloatTensor(tExp, t);
}

/** Test the batched matmul/3D matmul */
TEST(ComputeBatchedMatmul, basic) {
  SparseFloatTensor t1;
  t1.shape() = {4,3,5};
  t1.values() = {1, 2, 3, 5, 4};
  t1.dims().push_back({{0, 1, 1, 0}, {0, 1, 2, 4, 4}});
  t1.dims().push_back({{0, 1, 2, 4, 2}, {0, 1, 3, 4, 5}});

  SparseFloatTensor t2;
  t2.shape() = {4,5,5};
  t2.values() = {1, 2, 3, 5, 4};
  t2.dims().push_back({{0, 1, 1, 0}, {0, 1, 2, 4, 4}});
  t2.dims().push_back({{0, 1, 2, 4, 2}, {0, 1, 3, 4, 5}});

  auto sparse2Dsleft = t1.toSparse2Ds();
  auto sparse2Dsright = t2.toSparse2Ds();

  std::vector<SpMat> sparse2Ds;
  for (size_t i=0; i<sparse2Dsleft.size(); i++)
    sparse2Ds.emplace_back(matmul(sparse2Dsleft[i].get(), sparse2Dsright[i].get()));

  SparseFloatTensor t(sparse2Ds, t1.shape().size());

  SparseFloatTensor tExp;
  tExp.shape() = {4,3,5};
  tExp.values() = {1, 4, 6};
  tExp.dims().push_back({{0, 1}, {0, 1, 2, 2, 2}});
  tExp.dims().push_back({{0, 1, 2}, {0, 1, 3}});

  compareSparseFloatTensor(tExp, t);
}

/** Test coo to csr */
TEST(CooToCSR, basic) {
  Array<DimensionType> shape =  {5, 5};
  Array<DimensionType> rowsIndex {0, 1, 2, 3, 4};
  Array<DimensionType> colsIndex {0, 1, 2, 3, 4};
  Array<DataType>      values    {0, 1, 2, 3, 4};
  COO coo(shape[0], shape[1], rowsIndex, colsIndex, values);

  // CSR Expected
  DimensionType rowsE = 5, colsE = 5;
  std::vector<OrdinalType>   outerE  {0, 1, 2, 3, 4, 5};
  std::vector<DimensionType> innerE  {0, 1, 2, 3, 4};
  std::vector<DataType>      valuesE {0, 1, 2, 3, 4};

  // result CSR
  SpMat t = cooTocsr(coo);
  compareCSR(t, shape[0], shape[1], outerE, innerE, valuesE);
}

/** Test coo to csr */
TEST(CooToCSR, empty) {
  Array<DimensionType> shape =  {5, 5};
  Array<DimensionType> rowsIndex {};
  Array<DimensionType> colsIndex {};
  Array<DataType>      values    {};
  COO coo(shape[0], shape[1], rowsIndex, colsIndex, values);

  // CSR Expected
  DimensionType rowsE = 5, colsE = 5;
  std::vector<OrdinalType>   outerE  {0, 0, 0, 0, 0, 0};
  std::vector<DimensionType> innerE  {};
  std::vector<DataType>      valuesE {};

  // result CSR
  SpMat t = cooTocsr(coo);
  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE);
}

/** Test generated with python/utilities/testCasesGen/cootocsr.py directly */
TEST(CooToCSR, random3by3) {
  //COO Input:
  Array<DimensionType> shape(2);
  shape[0] = 3, shape[1] = 3;
  Array<DimensionType> rowsIndex {0, 0, 1, 1, 2};
  Array<DimensionType> colsIndex {1, 2, 0, 1, 2};
  Array<DataType> values {0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.9636627605010293};
  //CSR Expected:
  DimensionType rowsE = 3, colsE = 3;
  std::vector<DataType> valuesE = {
    0.7151893663724195,0.6027633760716439,0.5448831829968969,0.4236547993389047,0.9636627605010293
  };
  std::vector<DimensionType> innerE = {
    1,2,0,1,2
  };
  std::vector<OrdinalType> outerE = {
    0,2,4,5
  };

  COO coo(shape[0], shape[1], rowsIndex, colsIndex, values);
  SpMat t = cooTocsr(coo);
  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

/** Test generated with:
 * python/utilities/testCasesGen/cootocsr.py -r 10 -c 5 -n 0.1 */
TEST(CooToCSR, random10by5) {
  //COO Input:
  Array<DimensionType> shape(2);
  shape[0] = 10, shape[1] = 5;
  Array<DimensionType> rowsIndex {0, 4, 4, 7};
  Array<DimensionType> colsIndex {2, 0, 2, 4};
  Array<DataType> values {0.6027633760716439, 0.978618342232764, 0.46147936225293185, 0.6818202991034834};
  //CSR Expected:
  DimensionType rowsE = 10, colsE = 5;
  std::vector<DataType> valuesE = {
    0.6027633760716439,0.978618342232764,0.46147936225293185,0.6818202991034834
  };
  std::vector<DimensionType> innerE = {
    2,0,2,4
  };
  std::vector<OrdinalType> outerE = {
    0,1,1,1,1,
    3,3,3,4,4,
    4
  };

  COO coo(shape[0], shape[1], rowsIndex, colsIndex, values);
  SpMat t = cooTocsr(coo);
  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

/** Test generated with:
 * python/utilities/testCasesGen/cootocsr.py -r 10 -c 5 -n 0.1 -p */
TEST(CooToCSR, random10by5with01Permuted) {
  //COO Input:
  Array<DimensionType> shape(2);
  shape[0] = 10, shape[1] = 5;
  Array<DimensionType> rowsIndex {0, 7, 4, 4};
  Array<DimensionType> colsIndex {2, 4, 2, 0};
  Array<DataType> values {0.6027633760716439, 0.6818202991034834, 0.46147936225293185, 0.978618342232764};
  //CSR Expected:
  DimensionType rowsE = 10, colsE = 5;
  std::vector<DataType> valuesE = {
    0.6027633760716439,0.46147936225293185,0.978618342232764,0.6818202991034834
  };
  std::vector<DimensionType> innerE = {
    2,2,0,4
  };
  std::vector<OrdinalType> outerE = {
    0,1,1,1,1,
    3,3,3,4,4,
    4
  };

  COO coo(shape[0], shape[1], rowsIndex, colsIndex, values);
  SpMat t = cooTocsr(coo);
  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

TEST(TestSpMat, createEmpty) {
  SpMat t(5, 4);

  // CSR Expected
  DimensionType rowsE = 5, colsE = 4;
  std::vector<OrdinalType>   outerE  {0, 0, 0, 0, 0, 0};
  std::vector<DimensionType> innerE  {};
  std::vector<DataType>      valuesE {};

  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE);
}

TEST(TransposeTest, DoesTranspose) {
  std::vector<DimensionType> shape = {5, 4};
  std::vector<DataType> values = {1, 2, 3};
  std::vector<DimensionType> inner = {0, 1, 1};
  std::vector<OrdinalType> outer = {0, 2, 3, 3, 3, 3};
  std::vector<DimensionType> expectedInner = {0, 0, 1};
  std::vector<OrdinalType> expectedOuter = {0, 1, 3, 3, 3};
  SpMatMap tensor(shape[0], shape[1], inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, shape[1], shape[0], expectedOuter, expectedInner, values, false);
}

TEST(TransposeTest, Empty) {
  std::vector<DimensionType> shape = {5, 4};
  std::vector<DataType> values = {};
  std::vector<DimensionType> inner = {};
  std::vector<OrdinalType> outer = {0, 0, 0, 0, 0, 0};
  SpMatMap tensor(shape[0], shape[1], inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  std::vector<OrdinalType> outerExpected = {0, 0, 0, 0, 0};
  compareCSR(t, shape[1], shape[0], outerExpected, inner, values, false);
}

TEST(TransposeTest, Symetric) {
  std::vector<DimensionType> shape = {4, 4};
  std::vector<DataType> values = {1, 2, 2};
  std::vector<DimensionType> inner = {0, 1, 0};
  std::vector<OrdinalType> outer = {0, 2, 3, 3, 3};
  SpMatMap tensor(shape[0], shape[1], inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, shape[1], shape[0], outer, inner, values, false);
}

/** Test generated with python/utilities/testCasesGen/transpose.py directly */
TEST(TransposeTest, random3by3) {
  //CSR Input:
  DimensionType rows = 3, cols = 3;
  std::vector<DataType> values = {
    0.7151893663724195,0.6027633760716439,0.5448831829968969,0.4236547993389047,0.9636627605010293
  };
  std::vector<DimensionType> inner = {
    1,2,0,1,2
  };
  std::vector<OrdinalType> outer = {
    0,2,4,5
  };
  //CSR Expected:
  DimensionType rowsE = 3, colsE = 3;
  std::vector<DataType> valuesE = {
    0.5448831829968969,0.7151893663724195,0.4236547993389047,0.6027633760716439,0.9636627605010293
  };
  std::vector<DimensionType> innerE = {
    1,0,1,0,2
  };
  std::vector<OrdinalType> outerE = {
    0,1,3,5
  };

  SpMatMap tensor(rows, cols, inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

/** Test generated with:
 * python/utilities/testCasesGen/transpose.py -r 5 -c 1 */
TEST(TransposeTest, random5by1) {
  //CSR Input:
  DimensionType rows = 5, cols = 1;
  std::vector<DataType> values = {
    0.5488135039273248,0.6027633760716439,0.5448831829968969
  };
  std::vector<DimensionType> inner = {
    0,0,0
  };
  std::vector<OrdinalType> outer = {
    0,1,1,2,3,
    3
  };
  //CSR Expected:
  DimensionType rowsE = 1, colsE = 5;
  std::vector<DataType> valuesE = {
    0.5488135039273248,0.6027633760716439,0.5448831829968969
  };
  std::vector<DimensionType> innerE = {
    0,2,3
  };
  std::vector<OrdinalType> outerE = {
    0,3
  };

  SpMatMap tensor(rows, cols, inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

/** Test generated with:
 * python/utilities/testCasesGen/transpose.py -r 1 -c 4 */
TEST(TransposeTest, random1by4) {
  //CSR Input:
  DimensionType rows = 1, cols = 4;
  std::vector<DataType> values = {
    0.7151893663724195,0.5448831829968969
  };
  std::vector<DimensionType> inner = {
    1,3
  };
  std::vector<OrdinalType> outer = {
    0,2
  };
  //CSR Expected:
  DimensionType rowsE = 4, colsE = 1;
  std::vector<DataType> valuesE = {
    0.7151893663724195,0.5448831829968969
  };
  std::vector<DimensionType> innerE = {
    0,0
  };
  std::vector<OrdinalType> outerE = {
    0,0,1,1,2
  };

  SpMatMap tensor(rows, cols, inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}

/** Test generated with:
 * python/utilities/testCasesGen/transpose.py -r 7 -c 11 -n 0.05 */
TEST(TransposeTest, random7by11) {
  //CSR Input:
  DimensionType rows = 7, cols = 11;
  std::vector<DataType> values = {
    0.1433532874090464,0.45615033221654855,0.6706378696181594,0.9767610881903371
  };
  std::vector<DimensionType> inner = {
    4,10,1,6
  };
  std::vector<OrdinalType> outer = {
    0,0,0,2,2,
    3,3,4
  };
  //CSR Expected:
  DimensionType rowsE = 11, colsE = 7;
  std::vector<DataType> valuesE = {
    0.6706378696181594,0.1433532874090464,0.9767610881903371,0.45615033221654855
  };
  std::vector<DimensionType> innerE = {
    4,2,6,2
  };
  std::vector<OrdinalType> outerE = {
    0,0,1,1,1,
    2,2,3,3,3,
    3,4
  };

  SpMatMap tensor(rows, cols, inner.size(), outer.data(), inner.data(), values.data());
  SpMat t = transpose(tensor);

  compareCSR(t, rowsE, colsE, outerE, innerE, valuesE, false);
}
