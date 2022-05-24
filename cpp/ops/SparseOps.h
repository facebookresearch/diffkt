/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SPARSEOPS_H_
#define SPARSEOPS_H_

#include <jni.h>

extern "C" {

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_add(JNIEnv *, jobject,
                                                             jobject, jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_times(JNIEnv *, jobject,
                                                             jobject, jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_sub(JNIEnv *, jobject,
                                                             jobject, jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_matmul(JNIEnv *, jobject,
                                                             jobject, jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_matdiv(JNIEnv *, jobject,
                                                             jobject, jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_transpose(JNIEnv *, jobject,
                                                             jobject);

JNIEXPORT jobject JNICALL Java_org_diffkt_external_SparseOps_convertToCoo(JNIEnv *, jobject, jintArray,
                                                             jintArray, jintArray, jfloatArray);
}

#endif // SPARSEOPS_H_
