/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import kotlin.math.pow
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

// Tensor outer product

@SType("S1: Shape, S2: Shape")
@AllowUnreduced
infix fun @SType("S1") DTensor.outerProduct(right: @SType("S2") DTensor): @SType("concat(S1, S2)") DTensor {
    if (this.isScalar || right.isScalar)
        return this * right as DScalar

    val (operations, derivativeId) = commonKind(this, right)
    return operations.outerProduct(this, right, derivativeId)
}
