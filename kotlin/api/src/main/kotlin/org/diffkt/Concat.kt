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
 * Concatenate two tensors along the provided axis.
 *
 * The shape of the inputs must match except at the provided axis.
 * The shape of the result are the same as those of the inputs, except
 * at the provided axis, where it is the sum of the dimensions of the inputs
 * at the provided axis.
 *
 * Example:
 *   val x = tensorOf(listOf(3, 4, 5), { ... })
 *   val y = tensorOf(listOf(3, 6, 5), { ... })
 *   x.concat(y, axis = 1) // dimensions are (3, 10, 5)
 */
@SType("S1: Shape, S2: Shape, A: Dim")
@AllowUnreduced
fun @SType("S1") DTensor.concat(
    right: @SType("S2") DTensor,
    axis: @SType("A") Int
): @SType("concatOnAxis(S1, S2, A)") DTensor {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.concat(this, right, axis, derivativeId)
}

@SType("S1: Shape, S2: Shape")
@AllowUnreduced
fun @SType("S1") DTensor.concat(right: @SType("S2") DTensor): @SType("concatOnAxis(S1, S2, 0)") DTensor {
    val (operations, derivativeId) = commonKind(this, right)
    return operations.concat(this, right, 0, derivativeId)
}

fun FloatTensor.concat(right: FloatTensor, axis: Int = 0): FloatTensor =
    (this as DTensor).concat(right, axis) as FloatTensor

fun concat(slices: List<DTensor>, axis: Int = 0): DTensor {
    when (slices.size) {
        0 -> throw IllegalArgumentException("Cannot concat empty list of tensors")
        1 -> return slices.single()
        2 -> return slices[0].concat(slices[1], axis)
    }

    val sample = highestDerivativeID(slices)
    val derivativeID = sample.derivativeID
    return sample.operations.concat(slices, axis, derivativeID)
}
