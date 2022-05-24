/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DnnlOps.h"

#include <assert.h>
#include <iostream>

#include "dnnl.hpp"

#include "Dnnl/ArithmeticDnnl.h"
#include "Dnnl/BatchNorm.h"
#include "Dnnl/Conv.h"
#include "Dnnl/LogSoftmax.h"
#include "Dnnl/Pooling.h"
#include "Dnnl/Reduce.h"
#include "Dnnl/Relu.h"

static const std::string OOM_ERROR_FQ_NAME = "java/lang/OutOfMemoryError";

// Check assumption that jint == int32_t and jfloat == float.
static_assert(sizeof(jint) == sizeof(int32_t),
              "Size of jint and int32_t do not match");
static_assert(sizeof(jfloat) == sizeof(float),
              "Size of jfloat and float do not match");

// Throw a Java OutOfMemoryError
void out_of_memory(JNIEnv *env) {
  jclass oomErrorClass = env->FindClass(OOM_ERROR_FQ_NAME.c_str());
  // We should not fail to find the OOM Error class, but if by chance we do,
  // raise the error raised from that.
  if (oomErrorClass == NULL)
    return;
  env->ThrowNew(oomErrorClass, "");
}

// Given an array of ints, return a vector of a copy of the ints.
// This can raise a Java OutOfMemoryError, so the caller should check if an
// exception has occurred after calling this.
std::vector<int32_t> get_ints(JNIEnv *env, jintArray ints_data) {
  int32_t size = env->GetArrayLength(ints_data);
  int32_t *ints = (jint *)env->GetPrimitiveArrayCritical(ints_data, 0);
  if (ints == nullptr) {
    out_of_memory(env);
    // Return dummy result. Caller is responsible for checking if an exception
    // occurred
    return {};
  }
  std::vector<int32_t> vints{ints, ints + size};
  env->ReleasePrimitiveArrayCritical(ints_data, ints, 0);
  return vints;
}

// Given an array of dims, return a vector of a copy of the dims.
// This can raise a Java OutOfMemoryError, so the caller should check if an
// exception has occurred after calling this.
std::vector<int32_t> get_shape(JNIEnv *env, jintArray dims_data) {
  int32_t rank = env->GetArrayLength(dims_data);
  int32_t *dims = (jint *)env->GetPrimitiveArrayCritical(dims_data, 0);
  if (dims == nullptr) {
    out_of_memory(env);
    // Return dummy result. Caller is responsible for checking if an exception
    // occurred
    return {};
  }
  std::vector<int32_t> vdims{dims, dims + rank};
  env->ReleasePrimitiveArrayCritical(dims_data, dims, 0);
  return vdims;
}

// Releases C++ float arrays via ReleasePrimitiveArrayCritical.
void release_arrays(JNIEnv *env, std::vector<float *> arrs,
                    std::vector<jfloatArray> jarrs) {
  for (size_t i = 0; i < arrs.size(); i++) {
    env->ReleasePrimitiveArrayCritical(jarrs[i], arrs[i], 0);
  }
}

// Gets a C++ float array for each jfloatArray via GetPrimitiveArrayCritical.
//
// Returns a vector of C++ float arrays in the same order they were passed in.
// If unsuccessful, sets an exception in env and returns an empty vector.
// After use, arrays must be released via release_arrays.
std::vector<float *> get_arrays(JNIEnv *env, std::vector<jfloatArray> jarrs) {
  std::vector<float *> res;
  for (auto &jarr : jarrs) {
    float *arr = (float *)env->GetPrimitiveArrayCritical(jarr, 0);
    if (arr == nullptr) {
      out_of_memory(env);
      release_arrays(env, res, jarrs);
      return {};
    }
    res.push_back(arr);
  }
  return res;
}

// std::function doesn't work here, but a function reference does.
// https://en.cppreference.com/w/cpp/language/overloaded_address
// https://stackoverflow.com/questions/30393285/stdfunction-fails-to-distinguish-overloaded-functions
// Args are:
// - shape
// - lhs strides
// - rhs strides
// - lhs offset
// - rhs offset
// - lhs data
// - rhs data
// - res data
typedef void (&binary_arithmetic_function)(std::vector<int32_t>,
  std::vector<int32_t>, std::vector<int32_t>, int32_t, int32_t, float *, float *, float *);


void binary_arithmetic_helper(
    JNIEnv *env, jintArray shape_data,
    jintArray lhs_strides_data, jintArray rhs_strides_data,
    jint lhs_offset, jint rhs_offset,
    jfloatArray res_buffer, jfloatArray lhs_buffer,
    jfloatArray rhs_buffer,
    binary_arithmetic_function op) {
  auto shape = get_ints(env, shape_data);
  auto lhs_strides = get_ints(env, lhs_strides_data);
  auto rhs_strides = get_ints(env, rhs_strides_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_buffer, lhs_buffer, rhs_buffer};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do binary arithmetic op
  op(shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, arrays[0], arrays[1], arrays[2]);

  release_arrays(env, arrays, jarrays);
}



JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_add(
    JNIEnv *env, jobject obj, jintArray shape_data, jintArray lhs_strides,
    jintArray rhs_strides, jint lhs_offset, jint rhs_offset, jfloatArray res_buffer,
    jfloatArray lhs_buffer, jfloatArray rhs_buffer) {
  binary_arithmetic_helper(env, shape_data, lhs_strides, rhs_strides, lhs_offset, rhs_offset, res_buffer,
                           lhs_buffer, rhs_buffer, ops::add);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_sub(
    JNIEnv *env, jobject obj, jintArray shape_data, jintArray lhs_strides,
    jintArray rhs_strides, jint lhs_offset, jint rhs_offset, jfloatArray res_buffer,
    jfloatArray lhs_buffer, jfloatArray rhs_buffer) {
  binary_arithmetic_helper(env, shape_data, lhs_strides, rhs_strides, lhs_offset, rhs_offset, res_buffer,
                           lhs_buffer, rhs_buffer, ops::sub);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_avgPool(
    JNIEnv *env, jobject obj,
    /* result */
    jintArray res_shape_data, jfloatArray res_data,
    /* image */
    jintArray img_shape_data, jfloatArray img_data,
    /* pool dims */
    jint pool_height, jint pool_width) {
  // Get shapes
  auto img_shape = get_ints(env, img_shape_data);
  auto res_shape = get_ints(env, res_shape_data);
  // Return if any exceptions occurred while getting shapes.
  if (env->ExceptionOccurred())
    return;

  // Get data
  float *res = (jfloat *)env->GetPrimitiveArrayCritical(res_data, 0);
  float *img = (jfloat *)env->GetPrimitiveArrayCritical(img_data, 0);

  if (res == nullptr || img == nullptr)
    return out_of_memory(env);

  // Do avgpool
  ops::avg_pool(res_shape, img_shape, res, img, pool_height, pool_width);

  env->ReleasePrimitiveArrayCritical(res_data, res, 0);
  env->ReleasePrimitiveArrayCritical(img_data, img, 0);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_avgPoolGrad(
    JNIEnv *env, jobject obj,
    /* result */
    jintArray res_shape_data, jfloatArray res_data,
    /* seed */
    jintArray seed_shape_data, jfloatArray seed_data,
    /* pool dims */
    jint pool_height, jint pool_width) {
  // Get shapes
  auto seed_shape = get_ints(env, seed_shape_data);
  auto res_shape = get_ints(env, res_shape_data);
  // Return if any exceptions occurred while getting shapes.
  if (env->ExceptionOccurred())
    return;

  // Get data
  float *res = (jfloat *)env->GetPrimitiveArrayCritical(res_data, 0);
  float *img = (jfloat *)env->GetPrimitiveArrayCritical(seed_data, 0);

  if (res == nullptr || img == nullptr)
    return out_of_memory(env);

  // Do avgpool grad
  ops::avg_pool_grad(res_shape, seed_shape, res, img, pool_height, pool_width);

  env->ReleasePrimitiveArrayCritical(res_data, res, 0);
  env->ReleasePrimitiveArrayCritical(seed_data, img, 0);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_batchNorm(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray result_data,
    jfloatArray mean_data, jfloatArray variance_data, jfloatArray input_data,
    jfloatArray scale_shift_data) {
  auto input_shape = get_ints(env, shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{result_data, mean_data, variance_data,
                                          input_data, scale_shift_data};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do batch norm
  ops::batch_norm(input_shape, arrays[0], arrays[1], arrays[2], arrays[3],
                  arrays[4]);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_batchNormGrad(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray input_grad_data,
    jfloatArray scale_shift_grad_data, jfloatArray seed_data,
    jfloatArray input_data, jfloatArray scale_shift_data, jfloatArray mean_data,
    jfloatArray variance_data) {
  auto input_shape = get_ints(env, shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{
      input_grad_data, scale_shift_grad_data, seed_data,
      input_data,      scale_shift_data,      mean_data,
      variance_data};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do batch norm grad
  ops::batch_norm_grad(input_shape, arrays[0], arrays[1], arrays[2], arrays[3],
                       arrays[4], arrays[5], arrays[6]);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_conv2d(
    JNIEnv *env, jobject obj,
    /* result */
    jintArray res_shape_data, jfloatArray res_data,
    /* image */
    jintArray img_shape_data, jfloatArray img_data,
    /* filter */
    jintArray fil_shape_data, jfloatArray fil_data,
    /* strides */
    jint hstride, jint wstride,
    /* padding */
    jint padding_left, jint padding_right, jint padding_top,
    jint padding_bottom) {

  auto img_shape = get_shape(env, img_shape_data);
  auto fil_shape = get_shape(env, fil_shape_data);
  auto res_shape = get_shape(env, res_shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_data, img_data, fil_data};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do conv
  ops::conv(res_shape, img_shape, fil_shape, arrays[0], arrays[1], arrays[2],
            hstride, wstride,
            {padding_left, padding_right, padding_top, padding_bottom});

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_conv2dGradImage(
    JNIEnv *env, jobject obj,
    /* result (image grad) */
    jintArray res_shape_data, jfloatArray res_data,
    /* seed */
    jintArray seed_shape_data, jfloatArray seed_data,
    /* filter */
    jintArray fil_shape_data, jfloatArray fil_data,
    /* strides */
    jint hstride, jint wstride,
    /* padding */
    jint padding_left, jint padding_right, jint padding_top,
    jint padding_bottom) {

  auto seed_shape = get_shape(env, seed_shape_data);
  auto fil_shape = get_shape(env, fil_shape_data);
  auto res_shape = get_shape(env, res_shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_data, seed_data, fil_data};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do conv grad w.r.t. image
  ops::conv_grad_image(
      res_shape, seed_shape, fil_shape, arrays[0], arrays[1], arrays[2],
      hstride, wstride,
      {padding_left, padding_right, padding_top, padding_bottom});

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_conv2dGradFilter(
    JNIEnv *env, jobject obj,
    /* result (filter grad) */
    jintArray res_shape_data, jfloatArray res_data,
    /* seed */
    jintArray seed_shape_data, jfloatArray seed_data,
    /* image */
    jintArray img_shape_data, jfloatArray img_data,
    /* strides */
    jint hstride, jint wstride,
    /* padding */
    jint padding_left, jint padding_right, jint padding_top,
    jint padding_bottom) {

  // Get shapes
  auto seed_shape = get_shape(env, seed_shape_data);
  auto img_shape = get_shape(env, img_shape_data);
  auto res_shape = get_shape(env, res_shape_data);

  // Return if any exceptions occurred while getting shapes.
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_data, seed_data, img_data};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do conv grad w.r.t. filter
  ops::conv_grad_filter(
      res_shape, seed_shape, img_shape, arrays[0], arrays[1], arrays[2],
      hstride, wstride,
      {padding_left, padding_right, padding_top, padding_bottom});

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_linear(
    JNIEnv *env, jobject obj, jintArray shape_data, jintArray stride_data, jint offset,
    jfloatArray result, jfloatArray input, jfloat scale, jfloat shift) {
  auto shape = get_ints(env, shape_data);
  auto strides = get_ints(env, stride_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{result, input};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  ops::linear(shape, strides, offset, arrays[0], arrays[1], scale, shift);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_logSoftmax(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray input,
    jfloatArray result, jint axis) {
  auto shape = get_ints(env, shape_data);
  std::vector<jfloatArray> jarrays = {input, result};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;
  ops::log_softmax(shape, arrays[0], arrays[1], axis);
  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_logSoftmaxGrad(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray grad,
    jfloatArray seed, jfloatArray fwd_result, jint axis) {
  auto shape = get_ints(env, shape_data);
  std::vector<jfloatArray> jarrays = {grad, seed, fwd_result};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;
  ops::log_softmax_grad(shape, arrays[0], arrays[1], arrays[2], axis);
  release_arrays(env, arrays, jarrays);
}


JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_maxPool(
    JNIEnv *env, jobject obj,
    /* result */
    jintArray res_shape_data, jfloatArray res_data,
    /* workspace result */
    jbyteArray workspace_data,
    /* image */
    jintArray img_shape_data, jfloatArray img_data,
    /* pool dims */
    jint pool_height, jint pool_width) {
  // Get shapes
  auto img_shape = get_ints(env, img_shape_data);
  auto res_shape = get_ints(env, res_shape_data);
  // Return if any exceptions occurred while getting shapes.
  if (env->ExceptionOccurred())
    return;

  // Get data
  float *res = (jfloat *)env->GetPrimitiveArrayCritical(res_data, 0);
  // workspace_data is a jbyteArray even though the returned data is actually
  // unsigned. This is for readability because there is no ubyte.
  uint8_t *workspace =
      (uint8_t *)env->GetPrimitiveArrayCritical(workspace_data, 0);
  float *img = (jfloat *)env->GetPrimitiveArrayCritical(img_data, 0);

  if (res == nullptr || workspace == nullptr || img == nullptr)
    return out_of_memory(env);

  // Do maxpool
  ops::max_pool(res_shape, img_shape, res, workspace, img, pool_height,
                pool_width);

  env->ReleasePrimitiveArrayCritical(res_data, res, 0);
  env->ReleasePrimitiveArrayCritical(workspace_data, workspace, 0);
  env->ReleasePrimitiveArrayCritical(img_data, img, 0);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_maxPoolGrad(
    JNIEnv *env, jobject obj,
    /* result */
    jintArray res_shape_data, jfloatArray res_data,
    /* workspace */
    jbyteArray workspace_data,
    /* seed */
    jintArray seed_shape_data, jfloatArray seed_data,
    /* pool dims */
    jint pool_height, jint pool_width) {
  // Get shapes
  auto seed_shape = get_ints(env, seed_shape_data);
  auto res_shape = get_ints(env, res_shape_data);
  // Return if any exceptions occurred while getting shapes.
  if (env->ExceptionOccurred())
    return;

  // Get data
  float *res = (jfloat *)env->GetPrimitiveArrayCritical(res_data, 0);
  // workspace_data comes in as a jbyteArray even though the data is actually
  // unsigned. This is for readability becaues there is no ubyte.
  uint8_t *workspace =
      (uint8_t *)env->GetPrimitiveArrayCritical(workspace_data, 0);
  float *img = (jfloat *)env->GetPrimitiveArrayCritical(seed_data, 0);

  if (res == nullptr || workspace == nullptr || img == nullptr)
    return out_of_memory(env);

  // Do maxpool grad
  ops::max_pool_grad(res_shape, seed_shape, res, workspace, img, pool_height,
                     pool_width);

  env->ReleasePrimitiveArrayCritical(res_data, res, 0);
  env->ReleasePrimitiveArrayCritical(workspace_data, workspace, 0);
  env->ReleasePrimitiveArrayCritical(seed_data, img, 0);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_mulScalar(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray res_buffer,
    jfloatArray lhs_buffer, jfloat rhs) {
  auto shape = get_ints(env, shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_buffer, lhs_buffer};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do tensor/scalar mul
  ops::mul(shape, arrays[0], arrays[1], rhs);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_reduceSum(
    JNIEnv *env, jobject obj,
    jintArray res_shape_data,
    jfloatArray res_buffer,
    jintArray input_shape_data,
    jfloatArray input_buffer) {
  auto res_shape = get_ints(env, res_shape_data);
  auto input_shape = get_ints(env, input_shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_buffer, input_buffer};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do reduce_sum
  ops::reduce_sum(res_shape, arrays[0], input_shape, arrays[1]);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_relu(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray res,
    jfloatArray input) {
  auto shape = get_ints(env, shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res, input};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do relu
  ops::relu(shape, arrays[0], arrays[1]);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_Dnnl_reluGrad(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray res,
    jfloatArray seed, jfloatArray input) {
  auto shape = get_ints(env, shape_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res, seed, input};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  // Do relu grad
  ops::relu_grad(shape, arrays[0], arrays[1], arrays[2]);

  release_arrays(env, arrays, jarrays);
}

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_matmul(JNIEnv *env, jobject obj,
    jintArray lhs_shape_data, jintArray lhs_stride_data, jint lhs_offset,
    jintArray rhs_shape_data, jintArray rhs_stride_data, jint rhs_offset,
    jfloatArray res_buffer, jfloatArray lhs_buffer, jfloatArray rhs_buffer) {
  auto lhs_shape = get_ints(env, lhs_shape_data);
  auto rhs_shape = get_ints(env, rhs_shape_data);
  auto lhs_strides = get_ints(env, lhs_stride_data);
  auto rhs_strides = get_ints(env, rhs_stride_data);
  if (env->ExceptionOccurred())
    return;

  auto jarrays = std::vector<jfloatArray>{res_buffer, lhs_buffer, rhs_buffer};
  auto arrays = get_arrays(env, jarrays);
  if (env->ExceptionOccurred())
    return;

  ops::mmul(lhs_shape, lhs_strides, lhs_offset, rhs_shape, rhs_strides, rhs_offset, arrays[0], arrays[1], arrays[2]);

  release_arrays(env, arrays, jarrays);
}
