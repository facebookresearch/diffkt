/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_CONV_H_
#define OPS_CONV_H_

#include <stdint.h>
#include <vector>

namespace ops {

struct Padding {
  int left;
  int right;
  int top;
  int bottom;
};

void conv(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
          std::vector<int32_t> fil_shape, float *res, float *img, float *fil,
          int32_t hstride, int32_t wstride, Padding padding);

void conv_grad_image(std::vector<int32_t> res_shape,
                     std::vector<int32_t> seed_shape,
                     std::vector<int32_t> fil_shape, float *res, float *seed,
                     float *fil, int32_t hstride, int32_t wstride,
                     Padding padding);

void conv_grad_filter(std::vector<int32_t> res_shape,
                      std::vector<int32_t> seed_shape,
                      std::vector<int32_t> img_shape, float *res, float *seed,
                      float *img, int32_t hstride, int32_t wstride,
                      Padding padding);

} // namespace ops

#endif // OPS_CONV_H_
