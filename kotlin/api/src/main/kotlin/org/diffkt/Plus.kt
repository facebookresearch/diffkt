/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.reverse.ReverseScalar
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

// Addition - scalar

/**
 * The addition of a floating-point number to a [DScalar] simply updates the primal with the scaled value.
 */
operator fun DScalar.plus(right: Float): DScalar {
    if (right == 0F) return this
    return this + FloatScalar(right)
}

/**
 * Because scalar addition is commutative, we can implement this operation in terms of the same
 * operation with the operands swapped.
 */
operator fun Float.plus(right: DScalar) = right + this

/**
 * Scalar (floating-point) addition.
 */
operator fun DScalar.plus(right: DScalar): DScalar {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.plus(this, right, derivativeId) as DScalar
}

operator fun ReverseScalar.unaryPlus() = this

/**
 * Tensor addition.
 */
@SType("S1: Shape, S2: Shape")
@AllowUnreduced
operator fun @SType("S1") DTensor.plus(right: @SType("S2") DTensor): @SType("broadcast(S1,S2)") DTensor {
    return if (this.shape == right.shape) {
        val (operations, derivativeId) = commonKind(this, right)
        operations.plus(this, right, derivativeId)
    } else {
        val (broadcastedLeft, broadcastedRight) = Broadcasting.broadcastToCommonShape(this, right)
        val (operations, derivativeId) = commonKind(broadcastedLeft, broadcastedRight)
        operations.plus(broadcastedLeft, broadcastedRight, derivativeId)
    } as @SType("broadcast(S1,S2)") DTensor
}

@SType("S: Shape")
operator fun @SType("S") DTensor.plus(right: Float): @SType("S") DTensor {
    return this + FloatScalar(right)
}

@SType("S: Shape")
operator fun Float.plus(right: @SType("S") DTensor): @SType("S") DTensor = right + this
