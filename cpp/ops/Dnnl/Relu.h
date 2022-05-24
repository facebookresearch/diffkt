/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_RELU_H_
#define OPS_RELU_H_

#include <stdint.h>
#include <vector>

namespace ops {

// Relu (forward)
void relu(std::vector<int32_t> shape, float *res, float *data);

// Relu grad
// data, res, and seed should all be the same memory format.
void relu_grad(std::vector<int32_t> shape, float *res, float *seed,
               float *data);

} // namespace ops

#endif // OPS_RELU_H_
