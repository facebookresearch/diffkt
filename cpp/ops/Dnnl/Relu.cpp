/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Relu.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

const float NEGATIVE_SLOPE = 0.0f;

eltwise_forward::primitive_desc make_relu_pd(memory::desc md) {
  auto relu_desc = eltwise_forward::desc(
      prop_kind::forward_training, algorithm::eltwise_relu, md, NEGATIVE_SLOPE);
  return eltwise_forward::primitive_desc(relu_desc, ENG);
}

void relu(std::vector<int32_t> shape, float *res, float *data) {
  auto *src_buffer = data;
  auto *dst_buffer = res;

  // Make user src and dst memories, and set the backing data to our buffers.
  // We set the format_tag to "a" for a single linear blob of memory with size
  // a.
  auto md = memory::desc({product(shape)}, memory::data_type::f32,
                         memory::format_tag::a);
  auto user_src = memory(md, ENG, src_buffer);
  auto user_dst = memory(md, ENG, dst_buffer);

  // Make the primitive descriptor
  // Since this is an elementwise operation, the memory format shouldn't matter.
  // So, unlike Conv, we directly give the op the user-format memory
  // descriptors.
  auto relu_pd = make_relu_pd(md);

  // Check assumption that our dst is what relu_pd expects
  assert(relu_pd.dst_desc() == md);

  // Do relu
  auto relu = eltwise_forward(relu_pd);
  relu.execute(S, {{DNNL_ARG_SRC, user_src}, {DNNL_ARG_DST, user_dst}});

  // Wait for all primitives in the stream to finish.
  S.wait();
}

void relu_grad(std::vector<int32_t> shape, float *res, float *seed,
               float *data) {
  auto *src_buffer = data;
  auto *diff_src_buffer = res;
  auto *diff_dst_buffer = seed;

  // Make user src and dst memories, and set the backing data to our buffers.
  // We set the format_tag to "a" for a single linear blob of memory with size
  // a. Assume the callee is using the same memory format for data, res, and
  // seed.
  auto md = memory::desc({product(shape)}, memory::data_type::f32,
                         memory::format_tag::a);
  auto user_src = memory(md, ENG, src_buffer);
  auto user_diff_src = memory(md, ENG, diff_src_buffer);
  auto user_diff_dst = memory(md, ENG, diff_dst_buffer);

  // Make the backward primitive descriptor
  auto relu_pd = make_relu_pd(md);
  auto relu_bwd_desc =
      eltwise_backward::desc(algorithm::eltwise_relu, md, md, NEGATIVE_SLOPE);
  auto relu_bwd_pd =
      eltwise_backward::primitive_desc(relu_bwd_desc, ENG, relu_pd);

  // Do relu backward
  auto relu_bwd = eltwise_backward(relu_bwd_pd);
  relu_bwd.execute(S, {{DNNL_ARG_SRC, user_src},
                       {DNNL_ARG_DIFF_SRC, user_diff_src},
                       {DNNL_ARG_DIFF_DST, user_diff_dst}});

  // Wait for all primitives in the stream to finish.
  S.wait();
}

} // namespace ops
