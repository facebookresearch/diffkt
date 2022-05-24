/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Stack two tensors along the provided new axis.
 *
 * The shape of the inputs must match exactly.
 * The shape of the result is the same as those of the inputs, except with
 * a new axis inserted, where it is the of inputs stacked together.
 *
 * Example:
 *   val x = tensorOf(listOf(3, 6, 5), { ... })
 *   val y = tensorOf(listOf(3, 6, 5), { ... })
 *   x.stack(y, axis = 1) // dimensions are (3, 2, 6, 5)
 */

fun DTensor.stack(right: DTensor, axis: Int = 0): DTensor =
    this.unsqueeze(axis).concat(right.unsqueeze(axis), axis)

fun stack(slices: List<DTensor>, axis: Int = 0): DTensor {
    when (slices.size) {
        0 -> throw IllegalArgumentException("Cannot stack an empty list of tensors")
        1 -> return slices.single()
        2 -> return slices[0].stack(slices[1], axis)
    }
    val sample = highestDerivativeID(slices)
    val derivativeID = sample.derivativeID
    return sample.operations.concat(slices.map { it.unsqueeze(axis) }, axis, derivativeID)
}