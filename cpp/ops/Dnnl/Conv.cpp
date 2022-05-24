/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Conv.h"

#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

const algorithm CONV_ALGORITHM = algorithm::convolution_direct;

// DNNL Conv2D (forward)
void conv(std::vector<int32_t> res_shape, std::vector<int32_t> img_shape,
          std::vector<int32_t> fil_shape, float *res, float *img, float *fil,
          int32_t hstride, int32_t wstride, Padding padding) {
  const memory::dim BATCH = img_shape[0];
  const memory::dim IC = img_shape[3], OC = fil_shape[0];
  const memory::dim IH = img_shape[1], FH = fil_shape[1], OH = res_shape[1];
  const memory::dim IW = img_shape[2], FW = fil_shape[2], OW = res_shape[2];

  // Create memory descriptors for user memory, and set the backing data
  // to our buffers
  auto user_src = memory(
      {{BATCH, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, img);
  auto user_wei = memory(
      {{OC, IC, FH, FW}, memory::data_type::f32, memory::format_tag::ohwi}, ENG,
      fil);
  auto user_dst = memory(
      {{BATCH, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, res);

  // Make a copy of the memory descriptors for the conv descriptor and
  // set data format to "any" to allow conv to pick the best format
  auto conv_src_md = memory::desc(user_src.get_desc());
  auto conv_wei_md = memory::desc(user_wei.get_desc());
  auto conv_dst_md = memory::desc(user_dst.get_desc());
  conv_src_md.data.format_kind = dnnl_format_kind_any;
  conv_wei_md.data.format_kind = dnnl_format_kind_any;
  conv_dst_md.data.format_kind = dnnl_format_kind_any;

  // Set strides and padding
  // https://intel.github.io/mkl-dnn/group__dnnl__api__convolution.html
  const memory::dims strides = {hstride, wstride};
  const memory::dims padding_low = {padding.top, padding.left};
  const memory::dims padding_high = {padding.bottom, padding.right};

  // Create the convolution descriptor and primitive descriptor.
  auto conv_d = convolution_forward::desc(
      prop_kind::forward_training, CONV_ALGORITHM, conv_src_md, conv_wei_md,
      conv_dst_md, strides, padding_low, padding_high);
  auto conv_pd = convolution_forward::primitive_desc(conv_d, ENG);

  // Conditinally reorder src and weights in case the user format does
  // not match the one convolution picked. This probably always happens.
  memory conv_src = reorder_if_needed(user_src, conv_pd.src_desc());
  memory conv_wei = reorder_if_needed(user_wei, conv_pd.weights_desc());

  // If dst memory doesn't have the right memory format, make memory that does.
  // If new memory is made, we have to do a reorder back into the usr_dst after
  // executing our op.
  memory conv_dst = user_dst;
  bool reorder_dst = false;
  if (conv_pd.dst_desc() != user_dst.get_desc()) {
    conv_dst = memory(conv_pd.dst_desc(), ENG);
    reorder_dst = true;
  }

  // Do convolution
  auto conv = convolution_forward(conv_pd);
  conv.execute(S, {{DNNL_ARG_SRC, conv_src},
                   {DNNL_ARG_WEIGHTS, conv_wei},
                   {DNNL_ARG_DST, conv_dst}});

  // Conditionally reorder result
  if (reorder_dst)
    reorder(conv_dst, user_dst);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

// Make convolution primitive_descriptor for convolution_backward
convolution_forward::primitive_desc
make_conv_pd_for_bwd(std::vector<int32_t> src_shape,
                     std::vector<int32_t> dst_shape,
                     std::vector<int32_t> wei_shape, memory::dims strides,
                     memory::dims padding_low, memory::dims padding_high) {
  const memory::dim BATCH = src_shape[0];
  const memory::dim IC = src_shape[3], OC = wei_shape[0];
  const memory::dim IH = src_shape[1], FH = wei_shape[1], OH = dst_shape[1];
  const memory::dim IW = src_shape[2], FW = wei_shape[2], OW = dst_shape[2];

  auto src_md = memory::desc({BATCH, IC, IH, IW}, memory::data_type::f32,
                             memory::format_tag::any);
  auto wei_md = memory::desc({OC, IC, FH, FW}, memory::data_type::f32,
                             memory::format_tag::any);
  auto dst_md = memory::desc({BATCH, OC, OH, OW}, memory::data_type::f32,
                             memory::format_tag::any);

  auto conv_d = convolution_forward::desc(
      prop_kind::forward_training, CONV_ALGORITHM, src_md, wei_md, dst_md,
      strides, padding_low, padding_high);
  return convolution_forward::primitive_desc(conv_d, ENG);
}

// Conv gradient w.r.t. image
void conv_grad_image(std::vector<int32_t> res_shape,
                     std::vector<int32_t> seed_shape,
                     std::vector<int32_t> fil_shape, float *res, float *seed,
                     float *fil, int32_t hstride, int32_t wstride,
                     Padding padding) {
  // This function calls DNNL's convolution_backward_data. "data" refers to
  // image (as opposed to filters). This op takes diff_dst (dst grad/seed) and
  // weights (filter), and returns diff_src (src/image grad).
  //
  // Note that we go from diff_dst to diff_src; this appears backwards because
  // we are walking backward up the chain of operations for reverse-mode AD
  // ("backprop"); thus we are starting with the grad of dst and using it to get
  // the grad of src.

  // Convert user-friendly named parameters to the DNNL convention.
  auto &diff_src_shape = res_shape;
  auto &diff_dst_shape = seed_shape;
  auto &wei_shape = fil_shape;
  auto user_diff_src_buffer = res;
  auto user_diff_dst_buffer = seed;
  auto user_wei_buffer = fil;

  // Here I (input) refers to the input to the forward operation (convolution)
  // and O (output) refers to its output.
  const memory::dim BATCH = diff_src_shape[0];
  const memory::dim IC = diff_src_shape[3], OC = wei_shape[0];
  const memory::dim IH = diff_src_shape[1], KH = wei_shape[1],
                    OH = diff_dst_shape[1];
  const memory::dim IW = diff_src_shape[2], KW = wei_shape[2],
                    OW = diff_dst_shape[2];

  // Create memory descriptors for user memory and set the backing
  // data to our buffers
  auto user_diff_src_m = memory(
      {{BATCH, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, user_diff_src_buffer);
  auto user_diff_dst_m = memory(
      {{BATCH, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, user_diff_dst_buffer);
  auto user_wei_m = memory(
      {{OC, IC, KH, KW}, memory::data_type::f32, memory::format_tag::ohwi}, ENG,
      user_wei_buffer);

  // Make memory descriptors, strides, and padding for the conv_backward_data
  // descriptor
  auto diff_dst_md = memory::desc(user_diff_dst_m.get_desc());
  auto wei_md = memory::desc(user_wei_m.get_desc());
  auto diff_src_md = memory::desc(user_diff_src_m.get_desc());
  diff_dst_md.data.format_kind = dnnl_format_kind_any;
  wei_md.data.format_kind = dnnl_format_kind_any;
  diff_src_md.data.format_kind = dnnl_format_kind_any;

  const memory::dims strides = {hstride, wstride};
  const memory::dims padding_low = {padding.top, padding.left};
  const memory::dims padding_high = {padding.bottom, padding.right};

  // Make the conv forward primitive descriptor for the conv_backward_data
  // primitive descriptor
  auto conv_pd = make_conv_pd_for_bwd(diff_src_shape, diff_dst_shape, wei_shape,
                                      strides, padding_low, padding_high);

  // Finally make the conv_backward_data descriptor and primitive descriptor
  auto conv_bwd_data_d = convolution_backward_data::desc(
      CONV_ALGORITHM, diff_src_md, wei_md, diff_dst_md, strides, padding_low,
      padding_high);
  auto conv_bwd_data_pd =
      convolution_backward_data::primitive_desc(conv_bwd_data_d, ENG, conv_pd);

  // Conditinally reorder seed and weights in case the user format does
  // not match the one the op picked.
  memory diff_dst_m =
      reorder_if_needed(user_diff_dst_m, conv_bwd_data_pd.diff_dst_desc());
  memory wei_m = reorder_if_needed(user_wei_m, conv_bwd_data_pd.weights_desc());

  // If dst memory doesn't have the right memory format, make memory that does.
  // If new memory is made, we have to do a reorder back into the usr_dst after
  // executing our op.
  memory diff_src_m = user_diff_src_m;
  bool reorder_dst = false;
  if (conv_bwd_data_pd.diff_src_desc() != user_diff_src_m.get_desc()) {
    diff_src_m = memory(conv_bwd_data_pd.diff_src_desc(), ENG);
    reorder_dst = true;
  }

  // Finally run the op
  auto conv_bwd_data = convolution_backward_data(conv_bwd_data_pd);
  conv_bwd_data.execute(S, {{DNNL_ARG_DIFF_DST, diff_dst_m},
                            {DNNL_ARG_DIFF_SRC, diff_src_m},
                            {DNNL_ARG_WEIGHTS, wei_m}});

  // Conditionally reorder result
  if (reorder_dst)
    reorder(diff_src_m, user_diff_src_m);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

// Conv grad w.r.t. filter
void conv_grad_filter(std::vector<int32_t> res_shape,
                      std::vector<int32_t> seed_shape,
                      std::vector<int32_t> img_shape, float *res, float *seed,
                      float *img, int32_t hstride, int32_t wstride,
                      Padding padding) {
  // This function calls DNNL's convolution_backward_weights. This op takes
  // diff_dst (seed) and src (image), and returns diff_weights (weights/filter
  // grad).

  // Convert user-friendly named parameters to the DNNL convention.
  auto &diff_weights_shape = res_shape;
  auto &diff_dst_shape = seed_shape;
  auto &src_shape = img_shape;
  auto user_diff_weights_buffer = res;
  auto user_diff_dst_buffer = seed;
  auto user_src_buffer = img;

  const memory::dim BATCH = src_shape[0];
  const memory::dim IC = src_shape[3], OC = diff_weights_shape[0];
  const memory::dim IH = src_shape[1], KH = diff_weights_shape[1],
                    OH = diff_dst_shape[1];
  const memory::dim IW = src_shape[2], KW = diff_weights_shape[2],
                    OW = diff_dst_shape[2];

  // Create memory descriptors for user memory and set the backing
  // data to our buffers.
  auto user_src_m = memory(
      {{BATCH, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, user_src_buffer);
  auto user_diff_dst_m = memory(
      {{BATCH, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nhwc},
      ENG, user_diff_dst_buffer);
  auto user_diff_weights_m = memory(
      {{OC, IC, KH, KW}, memory::data_type::f32, memory::format_tag::ohwi}, ENG,
      user_diff_weights_buffer);

  // Make memory descriptors, strides, and padding for the conv_backward_data
  // descriptor
  auto diff_dst_md = memory::desc(user_diff_dst_m.get_desc());
  auto src_md = memory::desc(user_src_m.get_desc());
  auto diff_weights_md = memory::desc(user_diff_weights_m.get_desc());
  diff_dst_md.data.format_kind = dnnl_format_kind_any;
  src_md.data.format_kind = dnnl_format_kind_any;
  diff_weights_md.data.format_kind = dnnl_format_kind_any;

  const memory::dims strides = {hstride, wstride};
  const memory::dims padding_low = {padding.top, padding.left};
  const memory::dims padding_high = {padding.bottom, padding.right};

  // Make the conv forward primitive descriptor for the conv_backward_weights
  // primitive descriptor
  auto conv_pd =
      make_conv_pd_for_bwd(src_shape, diff_dst_shape, diff_weights_shape,
                           strides, padding_low, padding_high);

  // Finally make the conv_backward_weights descriptor and primitive descriptor
  auto conv_bwd_weights_d = convolution_backward_weights::desc(
      CONV_ALGORITHM, src_md, diff_weights_md, diff_dst_md, strides,
      padding_low, padding_high);
  auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
      conv_bwd_weights_d, ENG, conv_pd);

  // Conditinally reorder seed and weights in case the user format does
  // not match the one the op picked.
  memory diff_dst_m =
      reorder_if_needed(user_diff_dst_m, conv_bwd_weights_pd.diff_dst_desc());
  memory src_m = reorder_if_needed(user_src_m, conv_bwd_weights_pd.src_desc());

  // If dst memory doesn't have the right memory format, make memory that does.
  // If new memory is made, we have to do a reorder back into the usr_dst after
  // executing our op.
  memory diff_weights_m = user_diff_weights_m;
  bool reorder_dst = false;
  if (conv_bwd_weights_pd.diff_weights_desc() !=
      user_diff_weights_m.get_desc()) {
    diff_weights_m = memory(conv_bwd_weights_pd.diff_weights_desc(), ENG);
    reorder_dst = true;
  }

  // Finally run the op
  auto conv_bwd_weights = convolution_backward_weights(conv_bwd_weights_pd);
  conv_bwd_weights.execute(S, {{DNNL_ARG_DIFF_DST, diff_dst_m},
                               {DNNL_ARG_SRC, src_m},
                               {DNNL_ARG_DIFF_WEIGHTS, diff_weights_m}});

  // Conditionally reorder result
  if (reorder_dst)
    reorder(diff_weights_m, user_diff_weights_m);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

} // namespace ops
