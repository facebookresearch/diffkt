/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Pooling.h"

#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

void pooling_helper(algorithm alg, std::vector<int32_t> res_shape,
                    std::vector<int32_t> img_shape, float *res, float *img,
                    int32_t pool_height, int32_t pool_width,
                    unsigned char *workspace_buffer = NULL) {
  auto *src_buffer = img;
  auto *dst_buffer = res;
  auto &src_shape = img_shape;
  auto &dst_shape = res_shape;

  // Make all the dims
  // N and C are the same for src and dst
  const memory::dim N = src_shape[0], C = src_shape[3];
  const memory::dim IH = src_shape[1], IW = src_shape[2];
  const memory::dim OH = dst_shape[1], OW = dst_shape[2];
  memory::dims kernel = {pool_height, pool_width};
  memory::dims strides = {pool_height, pool_width};
  memory::dims padding = {0, 0};

  // Make user src and dst memories, and set the backing
  // data to our buffers
  auto user_src =
      memory({{N, C, IH, IW}, memory::data_type::f32, memory::format_tag::nhwc},
             ENG, src_buffer);
  auto user_dst =
      memory({{N, C, OH, OW}, memory::data_type::f32, memory::format_tag::nhwc},
             ENG, dst_buffer);

  // Create dst memory descriptor with format "any"
  // Note: src memory descriptor with format "any" does not appear to work,
  // nor does it show up in any official examples, so we skip this for src
  // and just use the user format.
  auto dst_md = memory::desc(user_dst.get_desc());
  dst_md.data.format_kind = dnnl_format_kind_any;

  // Make the primitive descriptor
  auto pool_d = pooling_forward::desc(prop_kind::forward_training, alg,
                                      user_src.get_desc(), dst_md, strides,
                                      kernel, padding, padding);
  auto pool_pd = pooling_forward::primitive_desc(pool_d, ENG);

  memory dst = user_dst;
  bool reorder_dst = false;
  // If result memory (dst) doesn't have the right format,
  // make one that does.
  if (pool_pd.dst_desc() != user_dst.get_desc()) {
    dst = memory(pool_pd.dst_desc(), ENG);
    reorder_dst = true;
  }

  std::unordered_map<int, memory> pooling_args;
  pooling_args.insert({DNNL_ARG_SRC, user_src});
  pooling_args.insert({DNNL_ARG_DST, user_dst});

  bool reorder_workspace = false;
  memory workspace;
  memory user_workspace;
  // Workspace is only used by the max pool algorithm
  if (alg == algorithm::pooling_max) {
    // The workspace saves the indices where maximum was found, and is used in
    // backward pooling to perform upsampling. It has the same shape as dst but
    // the memory format is u8 for indices.
    user_workspace = memory(
        {{N, C, OH, OW}, memory::data_type::u8, memory::format_tag::nhwc}, ENG,
        workspace_buffer);
    // If result memory (workspace) doesn't have the right format,
    // make one that does
    workspace = user_workspace;
    if (pool_pd.workspace_desc() != user_workspace.get_desc()) {
      workspace = memory(pool_pd.workspace_desc(), ENG);
      reorder_workspace = true;
    }
    pooling_args.insert({DNNL_ARG_WORKSPACE, workspace});
  }

  // Do pool
  auto pool = pooling_forward(pool_pd);
  pool.execute(S, pooling_args);

  // Conditionally reorder dst and workspace
  if (reorder_dst)
    reorder(dst, user_dst);
  if (reorder_workspace)
    reorder(workspace, user_workspace);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

void pooling_grad_helper(algorithm alg, std::vector<int32_t> res_shape,
                         std::vector<int32_t> seed_shape, float *res,
                         float *seed, int32_t pool_height, int32_t pool_width,
                         uint8_t *workspace_buffer = NULL) {
  auto *diff_src_buffer = res;
  auto *diff_dst_buffer = seed;
  auto &diff_src_shape = res_shape;
  auto &diff_dst_shape = seed_shape;

  // Make all the dims
  // N and C are the same for input and output
  const memory::dim N = diff_src_shape[0], C = diff_src_shape[3];
  const memory::dim IH = diff_src_shape[1], IW = diff_src_shape[2];
  const memory::dim OH = diff_dst_shape[1], OW = diff_dst_shape[2];
  memory::dims kernel = {pool_height, pool_width};
  memory::dims strides = {pool_height, pool_width};
  memory::dims padding = {0, 0};

  // Make user src and dst memories, and set the backing data to
  // our buffers
  auto user_diff_src =
      memory({{N, C, IH, IW}, memory::data_type::f32, memory::format_tag::nhwc},
             ENG, diff_src_buffer);
  auto user_diff_dst =
      memory({{N, C, OH, OW}, memory::data_type::f32, memory::format_tag::nhwc},
             ENG, diff_dst_buffer);

  // Create dst memory descriptor with format "any"
  // Note: src memory descriptor with format "any" does not appear to work,
  // nor does it show up in any official examples, so we skip this for src
  // and just use the user format.
  auto diff_dst_md = memory::desc(user_diff_dst.get_desc());
  diff_dst_md.data.format_kind = dnnl_format_kind_any;

  // Make the forward primitive descriptor
  auto pool_d = pooling_forward::desc(prop_kind::forward_training, alg,
                                      user_diff_src.get_desc(), diff_dst_md,
                                      strides, kernel, padding, padding);
  auto pool_pd = pooling_forward::primitive_desc(pool_d, ENG);
  // Make the backward primitive descriptor
  auto pool_bwd_d =
      pooling_backward::desc(alg, user_diff_src.get_desc(), diff_dst_md,
                             strides, kernel, padding, padding);
  auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_d, ENG, pool_pd);

  // Initialize pooling argumentsls
  std::unordered_map<int, memory> pooling_args;
  pooling_args.insert({DNNL_ARG_DIFF_SRC, user_diff_src});
  pooling_args.insert({DNNL_ARG_DIFF_DST, user_diff_dst});

  dnnl::memory user_workspace;
  dnnl::memory workspace;
  // Workspace is only used by max pool.
  if (alg == algorithm::pooling_max) {
    // Same shape as dst but with memory format u8
    user_workspace = memory(
        {{N, C, OH, OW}, memory::data_type::u8, memory::format_tag::nhwc}, ENG,
        workspace_buffer);
    // Ensure workspace the right memory format
    workspace = reorder_if_needed(user_workspace, pool_bwd_pd.workspace_desc());
    pooling_args.insert({DNNL_ARG_WORKSPACE, workspace});
  }

  // Ensure diff_dst has the right memory format
  memory diff_dst =
      reorder_if_needed(user_diff_dst, pool_bwd_pd.diff_dst_desc());

  // Do pool backward
  auto pool_bwd = pooling_backward(pool_bwd_pd);
  pool_bwd.execute(S, pooling_args);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

void avg_pool(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
              float *res, float *img, int32_t pool_height, int32_t pool_width) {
  pooling_helper(algorithm::pooling_avg, res_shape, img_shape, res, img,
                 pool_height, pool_width);
}

void avg_pool_grad(std::vector<int32_t> res_shape,
                   std::vector<int32_t> seed_shape, float *res, float *seed,
                   int32_t pool_height, int32_t pool_width) {
  pooling_grad_helper(algorithm::pooling_avg, res_shape, seed_shape, res, seed,
                      pool_height, pool_width);
}

void max_pool(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
              float *res, uint8_t *workspace, float *img, int32_t pool_height,
              int32_t pool_width) {
  pooling_helper(algorithm::pooling_max, res_shape, img_shape, res, img,
                 pool_height, pool_width, workspace);
}

void max_pool_grad(std::vector<int32_t> res_shape,
                   std::vector<int32_t> seed_shape, float *res,
                   uint8_t *workspace, float *seed, int32_t pool_height,
                   int32_t pool_width) {
  pooling_grad_helper(algorithm::pooling_max, res_shape, seed_shape, res, seed,
                      pool_height, pool_width, workspace);
}
} // namespace ops
