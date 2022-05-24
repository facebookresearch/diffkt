/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_REDUCE_H_
#define OPS_REDUCE_H_

#include <stdint.h>
#include <vector>

namespace ops {

void reduce_sum(std::vector<int32_t> res_shape, float *res, std::vector<int32_t> input_shape, float *input);

} // namespace ops

#endif // OPS_REDUCE_H_
