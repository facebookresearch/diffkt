/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

/**
 * Scalar (floating-point) subtraction.
 */
operator fun DScalar.minus(right: Float): DScalar = this - FloatScalar(right)

operator fun Float.minus(right: DScalar): DScalar = FloatScalar(this) - right

operator fun DScalar.minus(right: DScalar): DScalar {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.minus(this, right, derivativeId) as DScalar
}

operator fun DScalar.unaryMinus(): DScalar {
    return this.operations.unaryMinus(this) as DScalar
}

/**
 * Tensor subtraction.
 */
@SType("S1: Shape, S2: Shape")
@AllowUnreduced
operator fun @SType("S1") DTensor.minus(right: @SType("S2") DTensor): @SType("broadcast(S1,S2)") DTensor {
    return if (this.shape == right.shape) {
        val (operations, derivativeId) = commonKind(this, right)
        operations.minus(this, right, derivativeId)
    } else {
        val (broadcastedLeft, broadcastedRight) = Broadcasting.broadcastToCommonShape(this, right)
        val (operations, derivativeId) = commonKind(broadcastedLeft, broadcastedRight)
        operations.minus(broadcastedLeft, broadcastedRight, derivativeId)
    } as  @SType("broadcast(S1,S2)") DTensor
}

operator fun Float.minus(right: DTensor): DTensor = FloatScalar(this) - right

operator fun DTensor.minus(right: Float): DTensor = this - FloatScalar(right)

operator fun DTensor.unaryMinus(): DTensor {
    return this.operations.unaryMinus(this)
}
