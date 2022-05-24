/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_POOLING_H_
#define OPS_POOLING_H_

#include <stdint.h>
#include <vector>

namespace ops {

// AvgPool (forward)
// Populates res with the result of the avgpool and workspace with the index of
// each avg (as a linear offset relative to each pool window). Workspace is used
// for calculating the gradient. Workspace has the same shape as res.
void avg_pool(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
              float *res, float *img, int32_t pool_height, int32_t pool_width);

// AvgPool gradient
// Requires workspace in the format that AvgPool forward returns. Workspace has
// the same shape as seed.
void avg_pool_grad(std::vector<int32_t> res_shape,
                   std::vector<int32_t> seed_shape, float *res, float *seed,
                   int32_t pool_height, int32_t pool_width);

// MaxPool (forward)
// Populates res with the result of the maxpool and workspace with the index of
// each max (as a linear offset relative to each pool window). Workspace is used
// for calculating the gradient. Workspace has the same shape as res.
void max_pool(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
              float *res, uint8_t *workspace, float *img, int32_t pool_height,
              int32_t pool_width);

// MaxPool gradient
// Requires workspace in the format that MaxPool forward returns. Workspace has
// the same shape as seed.
void max_pool_grad(std::vector<int32_t> res_shape,
                   std::vector<int32_t> seed_shape, float *res,
                   uint8_t *workspace, float *seed, int32_t pool_height,
                   int32_t pool_width);

} // namespace ops

#endif // OPS_POOLING_H_
