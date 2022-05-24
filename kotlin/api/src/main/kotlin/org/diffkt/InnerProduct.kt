/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Takes a tensor of shape T<A,B>, a shape B, and a tensor of shape T<B,C> and
 * returns a tensor of shape T<A,C> which is the inner product of the two.
 */
fun DTensor.innerProduct(b: Shape, right: DTensor): DTensor {
    val left = this
    val a = left.shape.take(left.shape.rank - b.rank)
    val c = right.shape.drop(b.rank)

    return this.matmul(right, Shape(), a, b, c)
}
