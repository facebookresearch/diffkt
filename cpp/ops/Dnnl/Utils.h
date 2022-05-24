/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_UTILS_H_
#define OPS_UTILS_H_
#include <dnnl.hpp>
#include <jni.h>

namespace ops {
// DNNL engine and stream
// Extern because their value is set in the .cpp
// https://stackoverflow.com/a/18113888
extern dnnl::engine ENG;
extern dnnl::stream S;

// Returns the product of a numerical vector, or 1 if the vector is empty.
template <typename T> T product(std::vector<T> ns) {
  T res = 1;
  for (auto n : ns)
    res *= n;
  return res;
}

// Executes a reorder primitive on src and dst.
void reorder(dnnl::memory src, dnnl::memory dst);

// Reorders src if its memory format doesn't match that of dst_md. Returns a
// memory object with the right ordering. If reordering is needed, a new memory
// object is created from dst_md.
//
// Non-blocking; caller is responsible for calling S.wait() before accessing
// result data.
dnnl::memory reorder_if_needed(dnnl::memory src, dnnl::memory::desc dst_md);

// Reorders src if its memory format doesn't match that of dst. Returns a memory
// object with the right ordering.
//
// Non-blocking; caller is responsible for calling S.wait() before accessing
// result data.
dnnl::memory reorder_if_needed(dnnl::memory src, dnnl::memory dst);

// Returns a memory::dims object constructed from a vector of int32_t
dnnl::memory::dims to_dims(std::vector<int32_t> v);

// Returns a plain tag for the given rank.
//
// Examples:
//   get_plain_tag(1) -> format_tag::a
//   get_plain_tag(3) -> format_tag::abc
dnnl::memory::format_tag get_plain_tag(int rank);
} // namespace ops

#endif // OPS_UTILS_H_
