/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Sparse/SpMat.h"

#ifdef EIGEN

namespace ops {

  void rowPointers(const SpMat & m, const OrdinalType * & startPtr, const OrdinalType * & endPtr) {
    startPtr = m.outerIndexPtr();
    endPtr = m.outerIndexPtr() + 1;
  }

  void rowPointers(const SpMatMap & m, const OrdinalType * & startPtr, const OrdinalType * & endPtr) {
    startPtr = m.outerIndexPtr();
    endPtr = m.outerIndexPtr() + 1;
  }
} // namespace ops

#else

namespace ops {

  void rowPointers(const SpMatMap & m, const OrdinalType * & startPtr, const OrdinalType * & endPtr) {
    startPtr = m.rowStartPtr();
    endPtr = m.rowEndPtr();
  }
}

#endif // EIGEN
