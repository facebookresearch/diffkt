/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/** Flattens a tensor into a vector (a tensor with rank 1). */
fun DTensor.flatten(): DTensor = reshape(shape.product())

/**
 * Returns a tensor with dimensions [startDim, endDim] (inclusive range) flattened.
 *
 * If startDim is greater than endDim, returns the input tensor.
*/
fun DTensor.flatten(startDim: Int = 0, endDim: Int = rank - 1): DTensor {
    if (startDim >= endDim) return this
    val flattenedDim = (startDim..endDim).fold(1, { acc, nextDim -> acc * shape[nextDim] })
    val newDims = shape.take(startDim) + flattenedDim + shape.drop(endDim + 1)
    return reshape(newDims)
}
