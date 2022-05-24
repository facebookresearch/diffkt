/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SparseOps.h"
#include "Sparse/Utils.h"

#include <assert.h>
#include <iostream>
#include <vector>
#include <exception>

#include "Sparse/Arithmetic.h"

#include <omp.h>

typedef SpMat (&binary_sparseops_function)(SpMatMap &, SpMatMap &);
typedef SpMat (&unary_sparseops_function)(SpMatMap &);

static const std::string ERROR_FQ_NAME = "java/lang/Error";

jobject unaryCall(JNIEnv *env, jobject operand,
                   unary_sparseops_function op) {

  jclass errorClass = env->FindClass(ERROR_FQ_NAME.c_str());
  Require(errorClass != NULL, "Unable to retrieve Java error");

  jobject res;
  try{
    ops::SparseFloatTensor tensor = ops::javaToCPPSparseTensor(env, operand);

    auto tensor2Ds = tensor.toSparse2Ds();

    std::vector<SpMat> resSparse2Ds(tensor2Ds.size());
    size_t exceptionCount = 0;
    #ifdef EIGEN
    #pragma omp parallel for schedule(dynamic) reduction(+:exceptionCount)
    for (size_t i=0; i<tensor2Ds.size(); i++) {
      try{
        resSparse2Ds[i] = std::move(op(tensor2Ds[i].get()));
      } catch (...) {
        exceptionCount++;
      }
    }
    Require(exceptionCount == 0, "error in computing unary matrix operation");
    #else // MKL: do it sequentially as the computation will be done in parallel already.
    for (size_t i=0; i<tensor2Ds.size(); i++)
      resSparse2Ds[i] = std::move(op(tensor2Ds[i].get()));
    #endif

    res = ops::cppToJavaSparseTensor(env, ops::SparseFloatTensor(resSparse2Ds, tensor.shape().size() == 2));

  } catch (...) {
    env->ThrowNew(errorClass, "error in computing unary matrix operation");
  }

  return res;
}

jobject binaryCall(JNIEnv *env, jobject left, jobject right,
                   binary_sparseops_function op) {

  jclass errorClass = env->FindClass(ERROR_FQ_NAME.c_str());
  Require(errorClass != NULL, "Unable to retrieve Java error");

  jobject res;
  try{
    ops::SparseFloatTensor leftTensor = ops::javaToCPPSparseTensor(env, left);
    ops::SparseFloatTensor rightTensor = ops::javaToCPPSparseTensor(env, right);

    // Shape requirements
    Require(leftTensor.shape().size() == rightTensor.shape().size(), "The number of dimensions for matrices in both side should be consistent.");
    Require(leftTensor.shape().size() <= 3, "The number of dimensions should not exceed the maximum supported: 3");
    if (leftTensor.shape().size() == 3)
      Require(leftTensor.shape()[0] == rightTensor.shape()[0], "For 3D batch operation, the number of batch in both side should be consistent");

    auto leftSparse2Ds = leftTensor.toSparse2Ds();
    auto rightSparse2Ds = rightTensor.toSparse2Ds();

    std::vector<SpMat> resSparse2Ds(leftSparse2Ds.size());
    size_t exceptionCount = 0;
    #ifdef EIGEN
    #pragma omp parallel for schedule(dynamic) reduction(+:exceptionCount)
    for (size_t i=0; i<leftSparse2Ds.size(); i++) {
      try{
        resSparse2Ds[i] = std::move(op(leftSparse2Ds[i].get(), rightSparse2Ds[i].get()));
      } catch (...) {
        exceptionCount++;
      }
    }
    Require(exceptionCount == 0, "error in computing binary matrix operation");
    #else // MKL: do it sequentially as the computation will be done in parallel already.
    for (size_t i=0; i<leftSparse2Ds.size(); i++)
      resSparse2Ds[i] = std::move(op(leftSparse2Ds[i].get(), rightSparse2Ds[i].get()));
    #endif

    res = ops::cppToJavaSparseTensor(env, ops::SparseFloatTensor(resSparse2Ds, leftTensor.shape().size() == 2));

  } catch (...) {
    env->ThrowNew(errorClass, "error in computing binary matrix operation");
  }

  return res;
}

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_add(JNIEnv *env,
                                                             jobject obj,
                                                             jobject left,
                                                             jobject right) {
  return binaryCall(env, left, right, ops::add);
}

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_times(JNIEnv *env,
                                                             jobject obj,
                                                             jobject left,
                                                             jobject right) {
  return binaryCall(env, left, right, ops::times);
}

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_sub(JNIEnv *env,
                                                             jobject obj,
                                                             jobject left,
                                                             jobject right) {
  return binaryCall(env, left, right, ops::sub);
}

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_matmul(JNIEnv *env,
                                                             jobject obj,
                                                             jobject left,
                                                             jobject right) {
  return binaryCall(env, left, right, ops::matmul);
}

#ifdef EIGEN
JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_matdiv(JNIEnv *env,
                                                             jobject obj,
                                                             jobject left,
                                                             jobject right) {
  return binaryCall(env, left, right, ops::matdiv);
}
#endif

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_transpose(JNIEnv *env,
                                                             jobject obj,
                                                             jobject operand) {
  return unaryCall(env, operand, ops::transpose);
}

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_convertToCoo(JNIEnv *env,
                                                             jobject obj,
                                                             jintArray shape,
                                                             jintArray rows,
                                                             jintArray cols,
                                                             jfloatArray values) {
  jclass errorClass = env->FindClass(ERROR_FQ_NAME.c_str());
  Require(errorClass != NULL, "Unable to retrieve Java error");

  jobject res;
  try{
    auto coo = ops::javaToCOO(env, shape, rows, cols, values);
    SpMat t = cooTocsr(coo);
    std::vector<SpMat> l;
    l.emplace_back(std::move(t));
    auto tensor = ops::SparseFloatTensor(l);
    res = ops::cppToJavaSparseTensor(env, tensor);
  } catch (...) {
    env->ThrowNew(errorClass, "error computing coo conversion operation");
  }
  return res;
}
