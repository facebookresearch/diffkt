/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ArithmeticDnnl.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include "dnnl.hpp"

#include "Utils.h"

namespace ops {

using namespace dnnl;

void binary_op(algorithm alg, std::vector<int32_t> shape, std::vector<int32_t> lhs_strides,
            std::vector<int32_t> rhs_strides, int32_t lhs_offset, int32_t rhs_offset, float *res, float *lhs, float *rhs) {
  auto src0_md = memory::desc(to_dims(shape), memory::data_type::f32,
                              to_dims(lhs_strides));
  auto src1_md = memory::desc({to_dims(shape)}, memory::data_type::f32,
                              to_dims(rhs_strides));
  auto dst_md = memory::desc(to_dims(shape), memory::data_type::f32,
                             get_plain_tag(shape.size()));
  auto src0 = memory(src0_md, ENG, lhs + lhs_offset);
  auto src1 = memory(src1_md, ENG, rhs + rhs_offset);
  auto dst = memory(dst_md, ENG, res);

  // Make args have the same format, and the same format as dst for simplicity for now
  src0 = reorder_if_needed(src0, dst_md);
  src1 = reorder_if_needed(src1, dst_md);

  auto desc = binary::desc(alg, src0.get_desc(), src1.get_desc(), dst.get_desc());
  auto pd = binary::primitive_desc(desc, ENG);

  auto op = binary(pd);
  op.execute(
      S, {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_SRC_1, src1}, {DNNL_ARG_DST, dst}});
  S.wait();
}

void add(std::vector<int32_t> shape, std::vector<int32_t> lhs_strides,
         std::vector<int32_t> rhs_strides, int32_t lhs_offset, int32_t rhs_offset, float *res, float *lhs, float *rhs) {
  binary_op(algorithm::binary_add, shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, res, lhs, rhs);
}

void sub(std::vector<int32_t> shape, std::vector<int32_t> lhs_strides,
         std::vector<int32_t> rhs_strides, int32_t lhs_offset, int32_t rhs_offset, float *res, float *lhs, float *rhs) {
  binary_op(algorithm::binary_sub, shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, res, lhs, rhs);
}

void mul(std::vector<int32_t> shape, float *res, float *lhs, float rhs) {
  // TODO: need to handle non-contig cases
  // We do not use DNNL here because it's much slower
  auto size = product(shape);
  for (int i = 0; i < size; i++) {
    res[i] = lhs[i] * rhs;
  }
}

// Lhs dims are {batches, M, K}, rhs dims are {batches, K, N}.
// Matmul is named mmul here to avoid conflict with the DNNL
// function name
void mmul(std::vector<int32_t> lhs_dims, std::vector<int32_t> lhs_strides, int32_t lhs_offset,
          std::vector<int32_t> rhs_dims, std::vector<int32_t> rhs_strides, int32_t rhs_offset,
          float *res, float *lhs, float *rhs) {
  auto rank = lhs_dims.size();
  assert(rank >= 2);
  assert(rhs_dims.size() == rank);
  assert(lhs_strides.size() == rank);
  assert(rhs_strides.size() == rank);
  assert(lhs_dims[rank - 1] == rhs_dims[rank - 2]);
  for (int i = 0; i < rank - 2; i++) { assert(lhs_dims[i] == rhs_dims[i]); }


  auto *src0_buffer = lhs + lhs_offset;
  auto *src1_buffer = rhs + rhs_offset;
  auto *dst_buffer = res;

  // Get the shaping dimensions, accounting for a transpose if needed
  auto M = lhs_dims[rank - 2];
  auto K = lhs_dims[rank - 1];
  auto N = rhs_dims[rank - 1];

  // Make the result dimensions
  memory::dims dst_dims = {M, N};
  // The destination dim for each batch dimension is the max dim for src and weights
  for (int i = 0; i < rank - 2; i++) {
    dst_dims.insert(dst_dims.begin() + i, std::max(lhs_dims[i], rhs_dims[i]));
  }

  // Make memory descriptors.
  auto lhs_md = memory::desc(to_dims(lhs_dims), memory::data_type::f32, to_dims(lhs_strides));
  auto rhs_md = memory::desc(to_dims(rhs_dims), memory::data_type::f32, to_dims(rhs_strides));
  auto dst_md =
      memory::desc(dst_dims, memory::data_type::f32, get_plain_tag(rank));

  // Make user src and dst memories, and set the backing data to our buffers.
  auto user_src0 = memory(lhs_md, ENG, src0_buffer);
  auto user_src1 = memory(rhs_md, ENG, src1_buffer);
  auto user_dst = memory(dst_md, ENG, dst_buffer);

  auto matmul_d = matmul::desc(lhs_md, rhs_md, dst_md);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(matmul_d, ENG);
  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, user_src0});
  matmul_args.insert({DNNL_ARG_WEIGHTS, user_src1});
  matmul_args.insert({DNNL_ARG_DST, user_dst});

  matmul_prim.execute(S, matmul_args);

  // Wait for all primitives in the stream to finish.
  S.wait();
}

void linear(std::vector<int32_t> shape, std::vector<int32_t> strides, int32_t offset, float *res,
            float *data, float scale, float shift) {
  auto md = memory::desc(to_dims(shape), memory::data_type::f32,
                         to_dims(strides));
  auto src = memory(md, ENG, data + offset);
  auto dst = memory(md, ENG, res);

  auto eltwise_d = eltwise_forward::desc(
    prop_kind::forward_training, algorithm::eltwise_linear, src.get_desc(), scale, shift);
  auto eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, ENG);

  auto op = eltwise_forward(eltwise_pd);
  op.execute(S, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  S.wait();
}

} // namespace ops
