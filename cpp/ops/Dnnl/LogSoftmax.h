/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_LOGSOFTMAX_H_
#define OPS_LOGSOFTMAX_H_

#include <stdint.h>
#include <vector>

namespace ops {

void log_softmax(std::vector<int32_t> shape, float *src_buffer,
                 float *dst_buffer, int axis);

void log_softmax_grad(std::vector<int32_t> shape, float *seed, float *fwd_res,
                      float *grad, int axis);

} // namespace ops

#endif // OPS_LOGSOFTMAX_H_
