/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef TEST_TESTUTILS_H_
#define TEST_TESTUTILS_H_

#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace ops {

// Deterministically set the seed for the random engine.
// This is not const because calling `dist(random_eng)` modifies the state of
// the eng.
std::default_random_engine RANDOM_ENG(42);

template <typename T> void append_incrementing(std::vector<T> &v, size_t size) {
  v.reserve(size);
  for (size_t i = 1; i <= size; i++) {
    v.push_back(static_cast<T>(i));
  }
}

template <typename T> void append_zeros(std::vector<T> &v, size_t size) {
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    v.push_back(0);
  }
}

void append_ones(std::vector<float> &v, size_t size) {
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    v.push_back(1);
  }
}

void append_value(std::vector<float> &v, size_t size, float f) {
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    v.push_back(f);
  }
}

// Appends random floats uniformly distributed between -1 and 1
void append_random(std::vector<float> &v, size_t size) {
  std::uniform_real_distribution<float> dist(-1, 1);
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    v.push_back(dist(RANDOM_ENG));
  }
}

// TODO: make this spit out a nicer error message
void vector_expect_near(std::vector<float> a, std::vector<float> b,
                        float epsilon = 1.e-6f) {
  EXPECT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_NEAR(a[i], b[i], epsilon);
  }
}

} // namespace ops

#endif // TEST_TESTUTILS_H_
