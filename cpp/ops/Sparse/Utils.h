/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_UTILS_H_
#define OPS_UTILS_H_

#include <jni.h>

#include "SparseFloatTensor.h"
#include "COO.h"
#include "DebugUtils.h"

namespace ops {

  /** Converte a java SparseFloatTensor objuect to a C++ SparseFloatTensor
   * object */
  SparseFloatTensor javaToCPPSparseTensor(JNIEnv *, jobject);

  /** Converte a C++ SparseFloatTensor object to a java SparseFloatTensor
   * object */
  jobject cppToJavaSparseTensor(JNIEnv *, const SparseFloatTensor &);

  /** Converte data in COO format from Java to a C++ COO */
  COO javaToCOO(JNIEnv *, jintArray shape,
                                 jintArray rows,
                                 jintArray cols,
                                 jfloatArray values);
} // namespace ops

#endif // OPS_UTILS_H_
