/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.quadratic

import org.diffkt.*

/**
 * Demonstrates how to run gradient descent on a quadratic loss using reverse-mode automatic differentiation
 */

fun loss(x: DTensor): DScalar {
    return x.pow(2).sum()
}

fun main() {
    var x: DTensor = FloatTensor(
        Shape(2),
        floatArrayOf(1f, 2f)
    )

    for (i in 0 until 100) {
        val grad = reverseDerivative(x, ::loss)
        x -= grad * 0.1f
        println(x)
    }
}