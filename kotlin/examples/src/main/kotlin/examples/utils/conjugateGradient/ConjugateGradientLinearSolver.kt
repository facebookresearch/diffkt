/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.conjugateGradient

class ConjugateGradientLinearSolver<Vector, Scalar>(val vectorSpace: VectorSpace<Vector, Scalar>) {
    fun solve(f: (Vector) -> Vector, b: Vector, x0: Vector, maxIters: Int = 100, isAlmostZero: (Vector) -> Boolean): Vector {
        var x = x0
        with (vectorSpace) {
            var d = b - f(x)
            for (i in 0 until maxIters) {
                val r0 = if (i == 0) d else b - f(x)
                if (isAlmostZero(r0)) break

                val denom = d.dot(f(d))
                if (denom <= vectorSpace.zeroScalar) {
                    throw Exception("non-positive definiteness detected")
                }

                val r0q = r0.dot(r0)
                val alpha = r0q / denom
                x += d * alpha
                val r1 = b - f(x)
                val beta = r1.dot(r1) / r0q
                d = r1 + d * beta
            }
        }
        return x
    }
}