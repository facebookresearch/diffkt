/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ops_helper.h"

#include "Math/math.h"
#include "Predicate/ifThenElse.h"
#include <random>

JNIEXPORT void JNICALL Java_org_diffkt_external_External_plus(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray b, jfloatArray res,
    jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto b_data = env->GetFloatArrayElements(b, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if ((a_data == NULL) || (b_data == NULL) | (res_data == NULL)) {
    return;
  }

  math::plus(a_data, b_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(b, b_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_minus(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray b, jfloatArray res,
    jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto b_data = env->GetFloatArrayElements(b, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if ((a_data == NULL) || (b_data == NULL) | (res_data == NULL)) {
    return;
  }

  math::minus(a_data, b_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(b, b_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_unaryMinus(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray res, jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::unaryMinus(a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_times(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray b, jfloatArray res,
    jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto b_data = env->GetFloatArrayElements(b, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if ((a_data == NULL) || (b_data == NULL) | (res_data == NULL)) {
    return;
  }

  math::times(a_data, b_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(b, b_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_exp(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray res, jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::exp(a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_log(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray res, jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::log(a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_lgamma__F(
    JNIEnv *env, jobject obj, jfloat f) {
  return math::lgamma(f);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_lgamma___3F_3FI(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray res, jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::lgamma(a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_digamma__F(
    JNIEnv *env, jobject obj, jfloat f) {
  return math::digamma(f);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_digamma___3F_3FI(
    JNIEnv *env, jobject obj, jfloatArray a, jfloatArray res, jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::digamma(a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_polygamma__IF(
    JNIEnv *env, jobject obj, jint n, jfloat f) {
  return math::polygamma(n, f);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_polygamma__I_3F_3FI(
    JNIEnv *env, jobject obj, jint n, jfloatArray a, jfloatArray res,
    jint size) {
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if (a_data == NULL || res_data == NULL) {
    return;
  }

  math::polygamma(n, a_data, res_data, size);

  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}

JNIEXPORT void JNICALL Java_org_diffkt_external_External_ifThenElse(
    JNIEnv *env, jobject obj, jfloatArray p, jfloatArray a, jfloatArray b,
    jfloatArray res, jint size) {
  auto p_data = env->GetFloatArrayElements(p, NULL);
  auto a_data = env->GetFloatArrayElements(a, NULL);
  auto b_data = env->GetFloatArrayElements(b, NULL);
  auto res_data = env->GetFloatArrayElements(res, NULL);
  if ((p_data == NULL) || (a_data == NULL) || (b_data == NULL) ||
      (res_data == NULL)) {
    return;
  }

  predicate::ifThenElse(p_data, a_data, b_data, res_data, size);

  env->ReleaseFloatArrayElements(p, p_data, 0);
  env->ReleaseFloatArrayElements(a, a_data, 0);
  env->ReleaseFloatArrayElements(b, b_data, 0);
  env->ReleaseFloatArrayElements(res, res_data, 0);
}
