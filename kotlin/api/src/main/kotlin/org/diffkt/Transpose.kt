/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Returns x transposed over axes. If no axes are provided, tensor is transposed over all axes (that is,
 * the order of the axes are reversed).
 *
 * Note: API is numpy transpose, aka torch permute.
 * https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
 *
 * Example:
 * >>> val t = FloatTensor(Shape(1, 2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
 * >>> t.transpose(0, 2, 1)
 * FloatTensor(Shape(1, 3, 2), floatArrayOf(1f, 3f, 5f, 2f, 4f, 6f))
 */
fun DTensor.transpose(axes: IntArray = this.allAxes.reversedArray()): DTensor {
    assert(axes.distinct().size == axes.size && axes.sorted() == (0 until this.shape.rank).toList()) {
        "transpose: axes $axes are not a permutation of ${0 until this.shape.rank}" }

    fun isIdentityPermutation(axes: IntArray): Boolean {
        for (i in axes.indices)
            if (axes[i] != i) return false
        return true
    }

    if (isIdentityPermutation(axes))
        return this

    return this.operations.transpose(this, axes)
}

/**
 * Given the shape A,B,D (where A, B and D are lists
 * of Ints, and D is possibly empty), this function converts a tensor of
 * shape A,B,D to a tensor of shape B,A,D, shuffling the data
 * so that the element at position i,j,k is at position j,i,k (where i, j
 * and k are lists of integers corresponding to the shape A, B, and D).
 */
fun DTensor.leftTranspose(a: Shape, b: Shape): DTensor {
    require((a + b).isPrefix(this.shape))
    if (a.isScalar) return this
    if (b.isScalar) return this
    val d = this.shape.drop(a.rank + b.rank)
    assert((a + b + d) == this.shape)
    val aRank = a.rank
    val bRank = b.rank
    val axes = IntArray(rank) { i ->
        when {
            i < bRank -> i + aRank
            i < aRank + bRank -> i - bRank
            else -> i
        }
    }

    return this.transpose(axes)
}

/**
 * Given the shape A,B,D (where A, B and D are lists
 * of Ints, and D is possibly empty), this function converts a tensor of
 * shape D,A,B to a tensor of shape D,B,A, shuffling the data
 * so that the element at position i,j,k is at position i,k,j (where i, j
 * and k are lists of integers corresponding to the shape D, B, and A).
 */
fun DTensor.rightTranspose(a: Shape, b: Shape): DTensor {
    if (a.isScalar) return this
    if (b.isScalar) return this
    val d = this.shape.dropLast(a.rank + b.rank)
    assert(this.shape == d + a + b)
    val aRank = a.rank
    val bRank = b.rank
    val dRank = d.rank
    val axes = IntArray(rank) { i ->
        when {
            i < dRank -> i
            i < dRank + bRank -> i + aRank
            else -> i - bRank
        }
    }

    return this.transpose(axes)
}
