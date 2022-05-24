/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "math.h"
#include "util.h"
#include <iostream>
#include <cmath>

namespace math {

using namespace std;

void plus(float *a, float *b, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = a[i] + b[i];
}

void minus(float *a, float *b, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = a[i] - b[i];
}

void unaryMinus(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = -a[i];
}

void times(float *a, float *b, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = a[i] * b[i];
}

void exp(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = std::exp(a[i]);
}

void log(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = std::log(a[i]);
}

void lgamma(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = std::lgamma(a[i]);
}

void digamma(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = util::digamma(a[i]);
}

void trigamma(float *a, float *res, int size) {
  for (size_t i = 0; i < size; i++)
    res[i] = util::trigamma(a[i]);
}

void polygamma(int n, float *a, float *res, int size) {
  if (n == 0) {
    digamma(a, res, size);
  } else if (n == 1) {
    trigamma(a, res, size);
  } else {
    for (size_t i = 0; i < size; i ++)
      res[i] = util::polygamma(n, a[i]);
  }
}

float lgamma(float f) {
  return std::lgamma(f);
}

float digamma(float f) {
  return util::digamma(f);
}

float polygamma(int n, float f) {
  if (n == 0) {
    return util::digamma(f);
  } else if (n == 1) {
    return util::trigamma(f);
  } else {
    return util::polygamma(n, f);
  }
}

} // namespace math
