/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun DTensor.withChange(index: Int, axis: Int, replacementValue: DTensor): DTensor {
    val replacementValueShape = this.shape.remove(axis)
    val broadcastReplacementValue = replacementValue.broadcastTo(replacementValueShape)
    require(axis in 0 until rank)
    require(index >= 0 && index < shape[axis])
    require(replacementValue.shape == replacementValueShape)
    val replacement = broadcastReplacementValue.unsqueeze(axis)
    return when (index) {
        0 -> {
            if (shape[axis] == 1) {
                replacement
            } else {
                val right = this.slice(1, shape[axis], axis)
                replacement.concat(right, axis)
            }
        }
        shape[axis] - 1 -> {
            val left = this.slice(0, index, axis)
            left.concat(replacement, axis)
        }
        else -> {
            val left = this.slice(0, index, axis)
            val right = this.slice(index + 1, shape[axis], axis)
            left.concat(replacement, axis).concat(right, axis)
        }
    }
}

fun DTensor.withChange(indices: IntArray, replacementValue: DTensor): DTensor {
    val replacementValueShape = this.shape.drop(indices.size)
    val broadcastReplacementValue = replacementValue.broadcastTo(replacementValueShape)
    require(indices.size <= this.rank)
    require(indices.indices.all { indices[it] >= 0 && indices[it] < shape[it] })
    require(broadcastReplacementValue.shape == replacementValueShape)
    if (indices.isEmpty())
        return broadcastReplacementValue
    return this.withChange(
            indices[0],
            0,
            this.view(indices[0], 0).withChange(indices.drop(1).toIntArray(), broadcastReplacementValue)
    )
}

fun DTensor.withChange(index: IntRange, axis: Int, replacementValue: DTensor): DTensor {
    val replacementValueShape = this.shape.updated(axis, index.endInclusive - index.start + 1)
    val broadcastReplacementValue = replacementValue.broadcastTo(replacementValueShape)
    require(axis in 0 until rank)
    require(index.first >= 0 && index.last < shape[axis])
    require(index.step == 1) // nontrivial steps not yet supported
    require(broadcastReplacementValue.shape == replacementValueShape)
    return if (index.first == 0) {
        if (index.last < shape[axis] - 1) {
            val right = this.slice(index.last + 1, shape[axis], axis)
            broadcastReplacementValue.concat(right, axis)
        } else {
            broadcastReplacementValue
        }
    } else {
        if (index.last < shape[axis] - 1) {
            val left = this.slice(0, index.first, axis)
            val right = this.slice(index.last + 1, shape[axis], axis)
            left.concat(broadcastReplacementValue, axis).concat(right, axis)
        } else {
            val left = this.slice(0, index.first, axis)
            left.concat(broadcastReplacementValue, axis)
        }
    }
}
