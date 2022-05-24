/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"
#include <array>
#include <iostream>
#include <stdint.h>
#include <vector>

#include "dnnl.hpp"
#include <omp.h>

namespace ops {
using dnnl::memory;

dnnl::engine ENG(dnnl::engine::kind::cpu, 0);
dnnl::stream S(ENG);

namespace thread_detail {

int get_value_for(std::string body, std::string value_desc) {
  size_t line_start = body.find("\n" + value_desc) + 1;
  size_t line_end = body.find("\n", line_start);
  size_t value_start =
      body.find_last_of(" ", line_end - 1, line_end - line_start + 1) + 1;
  auto value = body.substr(value_start, line_end - value_start);
  return std::stoi(value);
}

// https://stackoverflow.com/a/478960
std::string exec(std::string cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// Set the maximum OMP threads to the number of physical CPUs by default.
int dummy_for_setting_num_threads = [] {
  // Setting via `omp_set_num_threads` overrides the environment variable;
  // if the environment variable is set, don't set via the function.
  // Note that we must set via the function because setting the env var
  // within the program appears to have no effect.
  if (std::getenv("OMP_NUM_THREADS") != NULL)
    return 0;

  try {
    auto output = exec("lscpu");
    int num_cpus = get_value_for(output, "CPU(s):");
    int threads_per_cpu = get_value_for(output, "Thread(s) per core:");
    int num_physical_cpus = num_cpus / threads_per_cpu;
    omp_set_num_threads(num_physical_cpus);
  } catch (...) {
  }

  return 0;
}();

} // namespace thread_detail

void reorder(memory src, memory dst) {
  auto r_pd = dnnl::reorder::primitive_desc(src, dst);
  dnnl::reorder(r_pd).execute(S, src, dst);
}

memory reorder_if_needed(memory src, memory::desc dst_md) {
  if (dst_md != src.get_desc()) {
    memory dst = memory(dst_md, ENG);
    reorder(src, dst);
    return dst;
  } else
    return src;
}

memory reorder_if_needed(memory src, memory dst) {
  if (src.get_desc() != dst.get_desc()) {
    reorder(src, dst);
    return dst;
  } else
    return src;
}

memory::dims to_dims(std::vector<int32_t> v) {
  return memory::dims{v.begin(), v.end()};
}

memory::format_tag get_plain_tag(int rank) {
  switch (rank) {
  case 1:
    return memory::format_tag::a;
  case 2:
    return memory::format_tag::ab;
  case 3:
    return memory::format_tag::abc;
  case 4:
    return memory::format_tag::abcd;
  case 5:
    return memory::format_tag::abcde;
  case 6:
    return memory::format_tag::abcdef;
  case 7:
    return memory::format_tag::abcdefg;
  case 8:
    return memory::format_tag::abcdefgh;
  case 9:
    return memory::format_tag::abcdefghi;
  case 10:
    return memory::format_tag::abcdefghij;
  default:
    throw std::runtime_error("Rank greater than 10 not handled");
  }
}

} // namespace ops
