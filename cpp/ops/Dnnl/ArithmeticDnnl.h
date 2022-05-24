/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_ARITHMETIC_H_
#define OPS_ARITHMETIC_H_

#include <stdint.h>
#include <vector>

namespace ops {

// Elementwise add
// Result will have major-to-minor memory format.
void add(std::vector<int32_t> shape, std::vector<int32_t> lhs_strides,
         std::vector<int32_t> rhs_strides, int32_t lhs_offset, int32_t rhs_offset, float *res, float *lhs, float *rhs);

// Elementwise subtract
// Result will have major-to-minor memory format.
void sub(std::vector<int32_t> shape, std::vector<int32_t> lhs_strides,
         std::vector<int32_t> rhs_strides, int32_t lhs_offset, int32_t rhs_offset,float *res, float *lhs, float *rhs);

// Elementwise multiply with a scalar
void mul(std::vector<int32_t> shape, float *res, float *lhs, float rhs);

// Matrix multiplication
// Lhs dims are {batches, M, K} and rhs dims are {batches, K, N},
// where batches is the same length and at each index is equal to
// the other argument or is 1.
void mmul(std::vector<int32_t> lhs_dims, std::vector<int32_t> lhs_strides, int32_t lhs_offset,
          std::vector<int32_t> rhs_dims, std::vector<int32_t> rhs_strides, int32_t rhs_offset,
          float *res, float *lhs, float *rhs);

// Linear transform: scale * input + shift
void linear(std::vector<int32_t> shape, std::vector<int32_t> strides, int32_t offset, float *res,
            float *data, float scale, float shift);

} // namespace ops

#endif // OPS_ARITHMETIC_H_
