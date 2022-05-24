/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun DTensor.sum(): DScalar = this.sum(keepDims = false) as DScalar

@JvmName("varargSum")
fun DTensor.sum(vararg axes: Int, keepDims: Boolean = false) = this.sum(axes, keepDims)

/**
 * Sum over given axes. If keepDims is true, the original rank of input is preserved. Otherwise,
 * the dimensions provided by axis are removed from the output shape.
 */
fun DTensor.sum(axes: IntArray = IntArray(rank) { it }, keepDims: Boolean = false): DTensor {
    // Handle cases that don't require any actual summing.
    // Note: DNNL doesn't like cases that don't require any actual summing.
    if (axes.isEmpty()) return this
    if (axes.all { shape[it] == 1}) {
        return if (keepDims) this
        else axes.reversed().fold(this) { acc, it -> acc.squeeze(it) }
    }

    return this.operations.sum(this, axes, keepDims)
}
