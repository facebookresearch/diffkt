/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_BATCHNORM_H_
#define OPS_BATCHNORM_H_

#include <stdint.h>
#include <vector>

namespace ops {

// Batch Normalization (forward)
//
// Inputs: input (NHWC), scale and shift (2C)
// Outputs: result (NHWC), mean (C), variance (C)
void batch_norm(std::vector<int32_t> input_shape, float *res_buffer,
                float *mean_buffer, float *variance_buffer, float *input_buffer,
                float *scale_shift_buffer);

// Batch Normalization gradient
//
// Inputs: seed (NHWC), input (NHWC), mean (C), variance (C), scale and shift
// (2C)
// Outputs: input grad (NHWC), scale and shift grad (2C)
void batch_norm_grad(std::vector<int32_t> input_shape, float *input_grad_buffer,
                     float *scale_shift_grad_buffer, float *seed_buffer,
                     float *input_buffer, float *scale_shift_buffer,
                     float *mean_buffer, float *variance_buffer);

} // namespace ops

#endif // OPS_BATCHNORM_H_
