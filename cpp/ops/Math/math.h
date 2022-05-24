/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef MATH_OPS_H_
#define MATH_OPS_H_

namespace math {

void plus(float *a, float *b, float *res, int size);
void minus(float *a, float *b, float *res, int size);
void unaryMinus(float *a, float *res, int size);
void times(float *a, float *b, float *res, int size);
void exp(float *a, float *res, int size);
void log(float *a, float *res, int size);
void lgamma(float *a, float *res, int size);
void digamma(float *a, float *res, int size);
void polygamma(int n, float *a, float *res, int size);

// Scalar functions
float lgamma(float f);
float digamma(float f);
float polygamma(int n, float f);


} // namespace math

#endif // MATH_OPS_H_
