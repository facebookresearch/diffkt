/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ifThenElse.h"
#include <iostream>

namespace predicate {

using namespace std;

void ifThenElse(float *p, float *a, float *b, float *res, int size) {
  for (size_t i = 0; i < size; i++) {
    if (p[i] > 0.f) {
      res[i] = a[i];
    } else {
      res[i] = b[i];
    }
  }
}

} // namespace predicate
