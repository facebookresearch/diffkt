/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "LogSoftmax.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

logsoftmax_forward::primitive_desc make_log_softmax_pd(memory::desc md,
                                                       int axis) {
  auto log_softmax_desc =
      logsoftmax_forward::desc(prop_kind::forward_training, md, axis);
  return logsoftmax_forward::primitive_desc(log_softmax_desc, ENG);
}

void log_softmax(std::vector<int32_t> shape, float *src_buffer,
                 float *dst_buffer, int axis) {
  memory::dims dims = {shape.begin(), shape.end()};
  auto md =
      memory::desc(dims, memory::data_type::f32, get_plain_tag(shape.size()));
  auto user_src = memory(md, ENG, src_buffer);
  auto user_dst = memory(md, ENG, dst_buffer);

  auto log_softmax_pd = make_log_softmax_pd(md, axis);

  // Check assumption that our dst is what log_softmax_pd expects
  assert(log_softmax_pd.dst_desc() == md);

  auto log_softmax = logsoftmax_forward(log_softmax_pd);
  log_softmax.execute(S, {{DNNL_ARG_SRC, user_src}, {DNNL_ARG_DST, user_dst}});
  S.wait();
}

// Computes grad based on the result of the forward op and seed/diff_dst
void log_softmax_grad(std::vector<int32_t> shape, float *grad, float *seed,
                      float *fwd_res, int axis) {
  memory::dims dims = {shape.begin(), shape.end()};
  auto md =
      memory::desc(dims, memory::data_type::f32, get_plain_tag(shape.size()));
  auto user_dst = memory(md, ENG, fwd_res);
  auto user_diff_src = memory(md, ENG, grad);
  auto user_diff_dst = memory(md, ENG, seed);

  auto log_softmax_pd = make_log_softmax_pd(md, axis);
  auto log_softmax_bwd_desc = logsoftmax_backward::desc(md, md, axis);
  auto log_softmax_bwd_pd = logsoftmax_backward::primitive_desc(
      log_softmax_bwd_desc, ENG, log_softmax_pd);

  auto log_softmax_bwd = logsoftmax_backward(log_softmax_bwd_pd);
  log_softmax_bwd.execute(S, {{DNNL_ARG_DST, user_dst},
                              {DNNL_ARG_DIFF_SRC, user_diff_src},
                              {DNNL_ARG_DIFF_DST, user_diff_dst}});
  S.wait();
}

} // namespace ops
