/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.conjugateGradient.optim

import examples.conjugateGradient.DScalarListVectorSpace
import examples.conjugateGradient.scalarListOf
import examples.utils.conjugateGradient.NewtonCGOptimizer
import org.diffkt.*

/**
 * Minimizes the function
 * f(x0, x1) = (x0 - 3)^2 + (x1 - 5)^2
 */

fun main() {
    val vectorSpace = DScalarListVectorSpace

    val targetVector = scalarListOf(3f, 5f)
    println("target vector: ${targetVector}")

    fun f(v: List<DScalar>): DScalar {
        with (vectorSpace) {
            val d = v - targetVector
            return d.dot(d)
        }
    }

    val optimizer = NewtonCGOptimizer(vectorSpace)

    var x = scalarListOf(15f, 35f)
    println("")
    println("initial vector: ${x}")
    println("initial loss: ${f(x)}")

    with (vectorSpace) {
        x = optimizer.step(
            ::f, x,
            isAlmostZero = { v: List<DScalar> -> v.dot(v) < 1e-5f }
        )
    }

    println("")
    println("optimized vector: ${x}")
    println("optimized loss: ${f(x)}")
}