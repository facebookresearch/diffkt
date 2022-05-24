/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.tracing.TracingScalar
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

fun ifThenElse(condition: FloatScalar, truePart: DScalar, falsePart: DScalar): DScalar =
    if (condition.value > 0f) truePart else falsePart

fun ifThenElse(condition: Float, truePart: Float, falsePart: Float): Float =
    if (condition > 0f) truePart else falsePart

/**
 * Test whether an input value [condition] is greater than zero (true), or less than or equal to zero (false).
 * Each element of the result is, depending on the value in [condition]:
 *   - If the input value is greater than zero: the corresponding value taken from [truePart]
 *   - Otherwise: the corresponding value taken from [falsePart]
 *
 * The three argument tensors should be the same shape (after broadcasting).
 */
@SType("A: Shape, B: Shape, C: Shape")
@AllowUnreduced
fun ifThenElse(
    condition: @SType("A") DTensor,
    truePart: @SType("B") DTensor,
    falsePart: @SType("C") DTensor
): @SType("broadcast(broadcast(A,B),C)") DTensor {
    val cond = condition.primal(NoDerivativeID)
    return when {
        cond is FloatScalar -> if (cond.value > 0f) truePart else falsePart
        cond is TracingScalar -> {
            // preserve the condition as a scalar when we can.
            val (broadcastedLeft, broadcastedRight) = Broadcasting.broadcastToCommonShape(truePart, falsePart)
            val (ops, did) = commonKind(broadcastedLeft, broadcastedRight)
            if (did != NoDerivativeID)
                ops.ifThenElse(cond, broadcastedLeft, broadcastedRight, did)
            else
                cond.operations.ifThenElse(cond, broadcastedLeft, broadcastedRight, did)
        }
        else -> {
            val (pb, broadcastedLeft, broadcastedRight) = Broadcasting.broadcastToCommonShape(cond, truePart, falsePart)
            val (ops, did) = commonKind(broadcastedLeft, broadcastedRight)
            if (did != NoDerivativeID)
                ops.ifThenElse(pb, broadcastedLeft, broadcastedRight, did)
            else
                pb.operations.ifThenElse(pb, broadcastedLeft, broadcastedRight, did)
        }
    } as @SType("broadcast(broadcast(A,B),C)") DTensor
}

/**
 * Test whether an input value [tested] is greater than zero, or less than or equal to zero.
 * Each element of the result is, depending on the value in [tested]:
 *   - If the input value is greater than zero: the corresponding value taken from [whenGreater]
 *   - Otherwise: the corresponding value taken from [whenLessThanOrEqual]
 *
 * The three argument tensors should be the same shape (after broadcasting).
 */
fun ifThenElse(condition: DScalar, truePart: DScalar, falsePart: DScalar): DScalar {
    val cond = condition.primal(NoDerivativeID)
    if (cond is FloatScalar)
        return if (cond.value > 0f) truePart else falsePart

    require(cond is TracingScalar)
    val (ops, did) = commonKind(truePart, falsePart)
    return if (did != NoDerivativeID)
        ops.ifThenElse(cond, truePart, falsePart, did) as DScalar
    else
        cond.operations.ifThenElse(cond, truePart, falsePart, did) as DScalar
}
