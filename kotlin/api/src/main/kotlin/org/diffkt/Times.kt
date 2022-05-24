/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

// Multiplication

operator fun Int.times(right: DScalar): DScalar = this.toFloat() * right

operator fun DScalar.times(right: Float): DScalar {
    when (right) {
        0F -> return FloatScalar.ZERO
        1F -> return this
    }
    return this * FloatScalar(right)
}

operator fun Float.times(right: DScalar): DScalar {
    return right * this
}

operator fun DScalar.times(right: DScalar): DScalar {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.timesScalar(this, right, derivativeId) as DScalar
}

operator fun Float.times(right: DTensor) = right * this

operator fun DTensor.times(right: Float): DTensor {
    when (right) {
        0f -> return this.operations.zeroOfSameKind(this, this.shape)
        1f -> return this
    }
    return this * FloatScalar(right)
}

operator fun DScalar.times(right: DTensor) = right * this

operator fun DTensor.times(right: DScalar): DTensor {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.timesScalar(right, this, derivativeId)
}

@SType("S1: Shape, S2: Shape")
@AllowUnreduced
operator fun @SType("S1") DTensor.times(that: @SType("S2") DTensor): @SType("broadcast(S1, S2)") DTensor {
    val (operations, derivativeId) = commonKind(this, that)
    return when {
        that is DScalar -> operations.timesScalar(that, this, derivativeId)
        this is DScalar -> operations.timesScalar(this, that, derivativeId)
        else -> {
            val (left, right) = Broadcasting.broadcastToCommonShape(this, that)
            assert(left.shape == right.shape)
            operations.times(left, right, derivativeId)
        }
    } as @SType("broadcast(S1, S2)") DTensor
}
