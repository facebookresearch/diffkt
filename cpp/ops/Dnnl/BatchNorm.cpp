/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "BatchNorm.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// TODO make epsilon specifyable by user
const float EPSILON = 1.e-5f;

batch_normalization_forward::primitive_desc make_bnorm_pd(memory::desc src_md) {
  auto bnorm_d = batch_normalization_forward::desc(
      prop_kind::forward_training, src_md, EPSILON,
      normalization_flags::use_scale_shift);
  return batch_normalization_forward::primitive_desc(bnorm_d, ENG);
}

memory::desc get_nhwc_md(std::vector<int32_t> input_shape) {
  memory::dim N = input_shape[0], H = input_shape[1], W = input_shape[2],
              C = input_shape[3];
  return memory::desc({N, C, H, W}, dt::f32, tag::nhwc);
}

memory::desc get_nchw_md(std::vector<int32_t> input_shape) {
  memory::dim N = input_shape[0], H = input_shape[1], W = input_shape[2],
              C = input_shape[3];
  return memory::desc({N, C, H, W}, dt::f32, tag::nchw);
}

memory::desc get_nc_md(std::vector<int32_t> input_shape) {
  memory::dim C = input_shape[3];
  return memory::desc({2, C}, dt::f32, tag::nc);
}

memory::desc get_c_md(std::vector<int32_t> input_shape) {
  memory::dim C = input_shape[3];
  return memory::desc({C}, dt::f32, tag::a);
}

void batch_norm(std::vector<int32_t> input_shape, float *res_buffer,
                float *mean_buffer, float *variance_buffer, float *input_buffer,
                float *scale_shift_buffer) {
  assert(input_shape.size() == 4);

  auto nhwc_md = get_nhwc_md(input_shape);
  // Make user memories, and set the backing data to our buffers.
  auto user_src = memory(nhwc_md, ENG, input_buffer);
  auto user_dst = memory(nhwc_md, ENG, res_buffer);
  auto user_scale_shift =
      memory(get_nc_md(input_shape), ENG, scale_shift_buffer);
  auto user_mean = memory(get_c_md(input_shape), ENG, mean_buffer);
  auto user_variance = memory(get_c_md(input_shape), ENG, variance_buffer);

  auto bnorm_pd = make_bnorm_pd(nhwc_md);

  // Create and execute the primitive
  auto bnorm_prim = batch_normalization_forward(bnorm_pd);
  bnorm_prim.execute(S, {{DNNL_ARG_SRC, user_src},
                         {DNNL_ARG_MEAN, user_mean},
                         {DNNL_ARG_VARIANCE, user_variance},
                         {DNNL_ARG_SCALE_SHIFT, user_scale_shift},
                         {DNNL_ARG_DST, user_dst}});

  // Wait for all primitives in the stream to finish.
  S.wait();
}

void batch_norm_grad(std::vector<int32_t> input_shape, float *input_grad_buffer,
                     float *scale_shift_grad_buffer, float *seed_buffer,
                     float *input_buffer, float *scale_shift_buffer,
                     float *mean_buffer, float *variance_buffer) {
  assert(input_shape.size() == 4);

  // Make user memories, and set the backing data to our buffers.
  auto nhwc_md = get_nhwc_md(input_shape);
  auto nc_md = get_nc_md(input_shape);
  auto c_md = get_c_md(input_shape);
  // Outputs
  auto user_diff_src = memory(nhwc_md, ENG, input_grad_buffer);
  auto user_diff_scale_shift = memory(nc_md, ENG, scale_shift_grad_buffer);
  // Inputs
  auto user_diff_dst = memory(nhwc_md, ENG, seed_buffer);
  auto user_src = memory(nhwc_md, ENG, input_buffer);
  auto user_scale_shift = memory(nc_md, ENG, scale_shift_buffer);
  auto user_mean = memory(c_md, ENG, mean_buffer);
  auto user_variance = memory(c_md, ENG, variance_buffer);

  auto bnorm_bwd_d = batch_normalization_backward::desc(
      prop_kind::backward, nhwc_md, nhwc_md, EPSILON,
      normalization_flags::use_scale_shift);
  auto bnorm_pd = make_bnorm_pd(nhwc_md);
  auto bnorm_bwd_pd =
      batch_normalization_backward::primitive_desc(bnorm_bwd_d, ENG, bnorm_pd);

  // Create and execute the primitive
  auto bnorm_bwd_prim = batch_normalization_backward(bnorm_bwd_pd);
  bnorm_bwd_prim.execute(S, {{DNNL_ARG_DIFF_SRC, user_diff_src},
                             {DNNL_ARG_DIFF_SCALE_SHIFT, user_diff_scale_shift},
                             {DNNL_ARG_DIFF_DST, user_diff_dst},
                             {DNNL_ARG_SRC, user_src},
                             {DNNL_ARG_SCALE_SHIFT, user_scale_shift},
                             {DNNL_ARG_MEAN, user_mean},
                             {DNNL_ARG_VARIANCE, user_variance}});

  // Wait for all primitives in the stream to finish.
  S.wait();
}

} // namespace ops
