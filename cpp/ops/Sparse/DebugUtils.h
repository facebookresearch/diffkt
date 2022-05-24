/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_DEBUGUTILS_H_
#define OPS_DEBUGUTILS_H_

#include <fstream>
#include <iostream>

/** The reason for not using std::assert is that: assert will be compiled out
 * when NDEBUG is defined; however, here, we want to use Require for
 * runtime input correctness check which is needed even in the production
 * code.
 * In the production code, this will not print an error but raise a runtime
 * error.
 * Its name 'Require' is mapping to we use in Kotlin. */
#ifdef DEBUG
#define Require(c, str) do { if (!(c)) { ops::printF(str); throw std::runtime_error(str); } } while (0)
#else // DEBUG
#define Require(c, str) do { if (!(c)) { throw std::runtime_error(str); } } while (0)
#endif // DEBUG

namespace ops {

  #ifdef DEBUG
  inline void printF(std::string str, const char * filename="debugLog.txt") {
    std::ofstream f;
    f.open(filename, std::ofstream::out | std::ofstream::app);
    f << str;
    f << std::endl;
    f.close();
  }
  #endif // DEBUG

} // namespace ops

#endif // OPS_DEBUGUTILS_H_
