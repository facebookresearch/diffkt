/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

operator fun DScalar.div(right: Int): DScalar = this * (1F / right)

operator fun Float.div(right: DScalar): DScalar = FloatScalar(this) / right

operator fun DScalar.div(right: DScalar): DScalar {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.div(this, right, derivativeId) as DScalar
}

operator fun DScalar.div(right: Float): DScalar = this * (1F / right)

@SType("S: Shape")
operator fun @SType("S") DTensor.div(right: DScalar): @SType("S") DTensor = this * (1F / right)

@SType("S: Shape")
operator fun @SType("S") DTensor.div(right: Float): @SType("S") DTensor = this * (1F / right)

@SType("S1: Shape, S2: Shape")
@AllowUnreduced
operator fun @SType("S1") DTensor.div(that: @SType("S2") DTensor): @SType("broadcast(S1,S2)") DTensor {
    if (this is DScalar && that is DScalar) return this / that
    val (left, right) = Broadcasting.broadcastToCommonShape(this, that)
    assert(left.shape == right.shape)
    val (operations, derivativeId) = commonKind(left, right)
    return operations.div(left, right, derivativeId) as @SType("broadcast(S1,S2)") DTensor
}
