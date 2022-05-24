/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdint>

#ifndef MATH_UTIL_H_
#define MATH_UTIL_H_
namespace math { namespace util {

double digamma(double x);
double trigamma(double x);
double polygamma(int64_t n, double x);

}} // namespace math::util


#endif // MATH_UTIL_H_
