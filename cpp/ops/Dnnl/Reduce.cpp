/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Reduce.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

void reduce_sum(std::vector<int32_t> res_shape, float *res, std::vector<int32_t> input_shape, float *input) {
  auto *src_buffer = input;
  auto *dst_buffer = res;

  memory::dims src_dims = {input_shape.begin(), input_shape.end()};
  memory::dims dst_dims = {res_shape.begin(), res_shape.end()};

  auto src_md = memory::desc(src_dims, memory::data_type::f32, get_plain_tag(src_dims.size()));
  auto dst_md = memory::desc(dst_dims, memory::data_type::f32, get_plain_tag(dst_dims.size()));
  auto user_src = memory(src_md, ENG, src_buffer);
  auto user_dst = memory(dst_md, ENG, dst_buffer);

  auto reduction_d = reduction::desc(algorithm::reduction_sum, src_md, dst_md, 0.f, 0.f);
  auto reduction_pd = reduction::primitive_desc(reduction_d, ENG);
  auto reduction_p = reduction(reduction_pd);

  reduction_p.execute(S, {{DNNL_ARG_SRC, user_src}, {DNNL_ARG_DST, user_dst}});
  S.wait();
}

} // namespace ops
