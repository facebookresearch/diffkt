/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.conjugateGradient.linsolve

import examples.conjugateGradient.DScalarListVectorSpace
import examples.conjugateGradient.scalarListOf
import examples.utils.conjugateGradient.ConjugateGradientLinearSolver
import org.diffkt.*

/**
 * Solves the linear system
 * 5 x0 + 2 x1 = 9
 * 2 x0 + 3 x1 = 8
 */

fun main() {
    val vectorSpace = DScalarListVectorSpace

    val b = scalarListOf(9f, 8f)
    println("target output: ${b}")

    fun f(v: List<DScalar>): List<DScalar> {
        val x0 = v[0]
        val x1 = v[1]
        return scalarListOf(
            5 * x0 + 2 * x1,
            2 * x0 + 3 * x1
        )
    }

    val linearSolver = ConjugateGradientLinearSolver(vectorSpace)

    var x = scalarListOf(15f, 35f)
    println("")
    println("initial vector: ${x}")
    println("initial output: ${f(x)}")

    with (vectorSpace) {
        x = linearSolver.solve(
            ::f, b, x,
            isAlmostZero = { v: List<DScalar> -> v.dot(v) < 1e-12f }
        )
    }

    println("")
    println("solution vector: ${x}")
    println("solution output: ${f(x)}")
}