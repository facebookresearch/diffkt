/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_HELP_H_
#define OPS_HELP_H_

#include <jni.h>

extern "C" {

// Math
JNIEXPORT void JNICALL Java_org_diffkt_external_External_plus(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_minus(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_unaryMinus(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jint);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_times(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jint);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_exp(JNIEnv *, jobject,
                                                             jfloatArray,
                                                             jfloatArray, jint);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_log(JNIEnv *, jobject,
                                                             jfloatArray,
                                                             jfloatArray, jint);

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_lgamma__F(JNIEnv *,
                                                                     jobject,
                                                                     jfloat);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_lgamma___3F_3FI(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jint);

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_digamma__F(JNIEnv *,
                                                                      jobject,
                                                                      jfloat);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_digamma___3F_3FI(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jint);

JNIEXPORT jfloat JNICALL Java_org_diffkt_external_External_polygamma__IF(
    JNIEnv *, jobject, jint, jfloat);

JNIEXPORT void JNICALL Java_org_diffkt_external_External_polygamma__I_3F_3FI(
    JNIEnv *, jobject, jint, jfloatArray, jfloatArray, jint);

// Predicate
JNIEXPORT void JNICALL Java_org_diffkt_external_External_ifThenElse(
    JNIEnv *, jobject, jfloatArray, jfloatArray, jfloatArray, jfloatArray,
    jint size);
}

#endif // OPS_HELP_H_
