/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.dynamic

class GradientDescentWithLineSearch<Vector, Scalar>(
    val vectorSpace: VectorSpace<Vector, Scalar>,
    val maxIters: Int = 100,
    val maxLineSearchIters: Int,
    val backtrackingFactor: Scalar
) {
    interface VectorSpace<Vector, Scalar> {
        operator fun Vector.plus(b: Vector): Vector
        operator fun Vector.unaryMinus(): Vector
        operator fun Vector.times(b: Scalar): Vector
        fun Vector.isAlmostZero(): Boolean

        operator fun Scalar.compareTo(b: Scalar): Int

        @Suppress("INAPPLICABLE_JVM_NAME")
        @JvmName("ScalarTimesScalar")
        operator fun Scalar.times(b: Scalar): Scalar

        val scalarOne: Scalar
    }

    fun optimize(
        x0: Vector,
        f: (Vector) -> Scalar,
        primalAndReverseDerivative: (Vector, (Vector) -> Scalar) -> Pair<Scalar, Vector>
    ): Vector {
        var x = x0
        
        with(vectorSpace) {
            for (i in 0 until maxIters) {
                val (fAtX, grad) = primalAndReverseDerivative(x, f)

                if (grad.isAlmostZero()) break

                val descentDir = -grad

                val (stepSize, x1) = lineSearch(x, f, descentDir, fAtX)
                x = x1
            }
        }

        return x
    }

    fun lineSearch(
        x: Vector,
        f: (Vector) -> Scalar,
        descentDir: Vector,
        fAtX: Scalar
    ): Pair<Scalar, Vector> {
        with(vectorSpace) {
            var stepSize: Scalar = scalarOne
            var x1: Vector = x + descentDir * stepSize
            var fAtX1 = f(x1)
            var iters = 0

            while (fAtX1 >= fAtX && iters < maxLineSearchIters) {
                stepSize *= backtrackingFactor
                x1 = x + descentDir * stepSize
                fAtX1 = f(x1)
                iters++
            }

            return Pair(stepSize, x1)
        }
    }
}