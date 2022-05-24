/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.SparseOps
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType
import kotlin.math.max

/**
 * Matrix multiply of two tensors.  See https://pytorch.org/docs/stable/generated/torch.matmul.html for the specification.
 */
@SType("A: Shape, B: Shape")
@AllowUnreduced
fun @SType("A") DTensor.matmul(right: @SType("B") DTensor): @SType("matmul(A,B)") DTensor {
    val x = this
    val y = right
    // TODO: the sparse implementation should be moved into SparseFloatTensorOperations.
    if (x is SparseFloatTensor && y is SparseFloatTensor) {
        require(x.shape.rank == y.shape.rank) { "The number of dimensions in both side should be consistent." }
        when (x.shape.rank) {
            2 -> {
                require(x.shape[1] == y.shape[0]) { "The number of cols of left should be the same as the number of rows on the right." }
            }
            3 -> {
                require(x.shape[0] == y.shape[0] && x.shape[2] == y.shape[1]) {
                    "For 3D matmul, the number of batches on both side should be same;" +
                            "the number of cols of left should be the same as the number of rows on the right."
                }
            }
            else -> throw IllegalArgumentException ("The numbers of dimensions supported for Spare Matmul are 2 and 3.")
        }

        return SparseOps.matmul(x, y)
    }

    var newX = if (x.rank == 1) x.unsqueeze(0) else x
    var newY = if (y.rank == 1) y.unsqueeze(1) else y

    val (m, k1) = Pair(newX.shape[newX.rank - 2], newX.shape[newX.rank - 1])
    val (k2, n) = Pair(newY.shape[newY.rank - 2], newY.shape[newY.rank - 1])
    require(k1 == k2) {
        "Shape mismatch: Inner dimensions of ${x.shape} and ${y.shape} do not match"
    }
    val rank = max(newX.rank, newY.rank)

    // Make tensors match rank
    if (newX.rank > newY.rank) { val diff = rank - newY.rank; (0 until diff).forEach { newY = newY.unsqueeze(0) } }
    if (newY.rank > newX.rank) { val diff = rank - newX.rank; (0 until diff).forEach { newX = newX.unsqueeze(0) } }

    // Calculate and check compatibility for batch dims
    val batchDims = mutableListOf<Int>()
    (0 until rank - 2).forEach {
        batchDims.add(max(newX.shape[it], newY.shape[it]))
        require(newX.shape[it] == newY.shape[it] || newX.shape[it] == 1 || newY.shape[it] == 1) {
            "Shape mismatch: Batch dimension mismatch for ${x.shape} and ${y.shape}"
        }
    }

    val batchShape = Shape(batchDims)
    newX = newX.broadcastTo(batchShape + m + k1)
    newY = newY.broadcastTo(batchShape + k1 + n)
    val result = newX.matmul(newY, batchShape, Shape(m), Shape(k1), Shape(n))
    assert(result.shape == batchShape + m + n)
    // if x was one dimensional, remove m
    val r1 = if (x.rank == 1) result.squeeze(result.rank - 2) else result
    // if y was one dimensional, remove n
    val r2 = if (y.rank == 1) r1.squeeze(r1.rank - 1) else r1
    return r2
}

/**
 * Generalized matrix multiply of two tensors.
 *
 * Given lists of integers A,B,C, and D,
 *
 * Takes an input (left) of Shape(A,B,C)
 * and an input (right) of Shape(A,C,D)
 * and returns an output of Shape(A,B,D). -- note C is eliminated
 *
 * A is known as the list of batch dimensions.
 */
fun DTensor.matmul(right: DTensor, a: Shape, b: Shape, c: Shape, d: Shape): DTensor {
    val left = this
    assert(left.shape == a + b + c)
    assert(right.shape == a + c + d)
    if (c.isScalar)
        return this outerProduct right
    val (operations, derivativeId) = commonKind(this, right)
    return operations.matmul(this, right, a, b, c, d, derivativeId)
}
