/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef DNNLOPS_H_
#define DNNLOPS_H_

#include <jni.h>

extern "C" {

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_add(JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* lhs strides */
    jintArray,
    /* rhs strides */
    jintArray,
    /* lhs offset */
    jint,
    /* rhs offset */
    jint,
    /* result */
    jfloatArray,
    /* left-hand side */
    jfloatArray,
    /* right-hand side */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_maxPool(JNIEnv *, jobject,
    /* result */
    jintArray, jfloatArray,
    /* workspace result */
    jbyteArray,
    /* image */
    jintArray, jfloatArray,
    /* pool dims */
    jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_maxPoolGrad(JNIEnv *, jobject,
    /* result */
    jintArray,
    jfloatArray,
    /* workspace */
    jbyteArray,
    /* seed */
    jintArray,
    jfloatArray,
    /* pool dims */
    jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_avgPool(JNIEnv *, jobject,
    /* result */
    jintArray, jfloatArray,
    /* image */
    jintArray, jfloatArray,
    /* pool dims */
    jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_avgPoolGrad(JNIEnv *, jobject,
    /* result */
    jintArray,
    jfloatArray,
    /* seed */
    jintArray,
    jfloatArray,
    /* pool dims */
    jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_batchNorm(JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* result */
    jfloatArray,
    /* mean (output) */
    jfloatArray,
    /* variance (output) */
    jfloatArray,
    /* input */
    jfloatArray,
    /* scale and shift */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_batchNormGrad(JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* input grad */
    jfloatArray,
    /* scale and shift grad */
    jfloatArray,
    /* seed */
    jfloatArray,
    /* input */
    jfloatArray,
    /* scale and shift */
    jfloatArray,
    /* mean */
    jfloatArray,
    /* variance */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_conv2d(JNIEnv *, jobject,
    /* result */
    jintArray, jfloatArray,
    /* image */
    jintArray, jfloatArray,
    /* filter */
    jintArray, jfloatArray,
    /* strides */
    jint, jint,
    /* padding */
    jint, jint, jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_conv2dGradImage(
    JNIEnv *, jobject,
    /* result (image grad) */
    jintArray, jfloatArray,
    /* seed */
    jintArray, jfloatArray,
    /* filter */
    jintArray, jfloatArray,
    /* strides */
    jint, jint,
    /* padding */
    jint, jint, jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_conv2dGradFilter(
    JNIEnv *, jobject,
    /* result (filter grad) */
    jintArray, jfloatArray,
    /* seed */
    jintArray, jfloatArray,
    /* image */
    jintArray, jfloatArray,
    /* strides */
    jint, jint,
    /* padding */
    jint, jint, jint, jint);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_linear(
    JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* strides */
    jintArray,
    /* offset */
    jint,
    /* result */
    jfloatArray,
    /* input */
    jfloatArray,
    /* scale */
    jfloat,
    /* shift */
    jfloat);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_logSoftmax(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray input,
    jfloatArray result, jint axis);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_logSoftmaxGrad(
    JNIEnv *env, jobject obj, jintArray shape_data, jfloatArray grad,
    jfloatArray seed, jfloatArray fwd_result, jint axis);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_mulScalar(
    JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* result */
    jfloatArray,
    /* left-hand side */
    jfloatArray,
    /* right-hand side */
    jfloat);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_reduceSum(JNIEnv *, jobject,
    /* result shape */
    jintArray,
    /* result */
    jfloatArray,
    /* input shape */
    jintArray,
    /* input */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_relu(
    JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* result */
    jfloatArray,
    /* input */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_reluGrad(
    JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* result */
    jfloatArray,
    /* seed */
    jfloatArray,
    /* input */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_matmul(JNIEnv *, jobject,
    /* lhs shape */
    jintArray,
    /* lhs strides */
    jintArray,
    /* lhs offset */
    jint,
    /* rhs shape */
    jintArray,
    /* rhs strides */
    jintArray,
    /* rhs offset */
    jint,
    /* result */
    jfloatArray,
    /* left-hand side */
    jfloatArray,
    /* right-hand side */
    jfloatArray);

JNIEXPORT void JNICALL
Java_org_diffkt_external_Dnnl_sub(JNIEnv *, jobject,
    /* shape */
    jintArray,
    /* lhs strides */
    jintArray,
    /* rhs strides */
    jintArray,
    /* lhs offset */
    jint,
    /* rhs offset */
    jint,
    /* result */
    jfloatArray,
    /* left-hand side */
    jfloatArray,
    /* right-hand side */
    jfloatArray);

} // extern "C"

#endif // DNNLOPS_H_
