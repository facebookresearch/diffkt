/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_JNI_H
#define OPS_JNI_H

#include <jni.h>

extern "C" {

// Tensor utils

JNIEXPORT void JNICALL Java_org_diffkt_external_Gpu_deleteHandle(JNIEnv *, jobject, jlong);

JNIEXPORT jintArray JNICALL Java_org_diffkt_external_Gpu_getShape(JNIEnv *, jobject, jlong);

JNIEXPORT jfloatArray JNICALL Java_org_diffkt_external_Gpu_getFloatData(JNIEnv *, jobject, jlong);

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_putFloatTensor(JNIEnv *, jobject, jintArray, jfloatArray);

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_zeros(JNIEnv *, jobject, jintArray);

// Misc utils

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_getAllocatedBytes(JNIEnv *, jobject);

// Ops

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_add(JNIEnv *, jobject, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_addGradLhs(JNIEnv *, jobject, jlong, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_addGradRhs(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_avgPool(JNIEnv *, jobject, jlong, jint, jint);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_avgPoolGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_batchNorm2d(JNIEnv *, jobject, jlong, jlong, jlong, jlong, jfloat);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_batchNorm2dGradInput(JNIEnv *, jobject, jlong, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_batchNorm2dGradScaleShift(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_broadcastTo(JNIEnv *, jobject, jlong, jintArray);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_broadcastToGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_conv2d(JNIEnv *, jobject, jlong, jlong, jintArray, jintArray);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_conv2dGradImages(JNIEnv *, jobject, jlong, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_conv2dGradFilters(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_div(JNIEnv *, jobject, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_logSoftmax(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_logSoftmaxGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_matmul(JNIEnv *, jobject, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_matmulGradLhs(JNIEnv *, jobject, jlong, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_matmulGradRhs(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_maxPool(JNIEnv *, jobject, jlong, jint, jint);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_maxPoolGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_nllLoss(JNIEnv *, jobject, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_nllLossGradX(JNIEnv *, jobject, jlong, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_nllLossGradLabels(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_relu(JNIEnv *, jobject, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_reluGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_reshape(JNIEnv *, jobject, jlong, jintArray);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_reshapeGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_sub(JNIEnv *, jobject, jlong, jlong);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_isub(JNIEnv *env, jobject, jlong jlhs, jlong jrhs);

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_sum(JNIEnv *, jobject, jlong, jintArray, jboolean);
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_sumGrad(JNIEnv *, jobject, jlong, jlong, jlong);

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_times(JNIEnv *, jobject, jlong, jlong);

} // extern "C"

#endif // OPS_JNI_H
