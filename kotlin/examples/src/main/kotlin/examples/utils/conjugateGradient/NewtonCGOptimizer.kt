/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.conjugateGradient

class NewtonCGOptimizer<Vector, Tangent, Scalar>(
    val vectorSpace: DifferentiableVectorSpace<Vector, Tangent, Scalar>,
    val maxLinSolveIters: Int = 100,
) {
    val cgSolver = ConjugateGradientLinearSolver(vectorSpace.tangentVectorSpace)

    abstract class DifferentiableVectorSpace<Vector, Tangent, Scalar> :
        examples.utils.conjugateGradient.DifferentiableVectorSpace<Vector, Tangent, Scalar> {
        abstract fun grad(f: (Vector) -> Scalar, x: Vector): Tangent
        open fun grad(f: (Vector) -> Scalar): (Vector) -> Tangent = { grad(f, it) }

        abstract fun jvp(
            f: (Vector) -> Tangent,
            x: Vector,
            v: Tangent
        ): Tangent

        fun gvp(f: (Vector) -> Scalar, x: Vector): (Tangent) -> Scalar {
            return { v: Tangent ->
                with (tangentVectorSpace) {
                    grad(f, x).dot(v)
                }
            }
        }

        fun hvpReverseOverReverse(f: (Vector) -> Scalar, x: Vector): (Tangent) -> Tangent {
            return { v: Tangent ->
                grad({ d -> gvp(f, d)(v) }, x)
            }
        }

        fun hvpForwardOverReverse(f: (Vector) -> Scalar, x: Vector): (Tangent) -> Tangent {
            val grad: (Vector) -> Tangent = this.grad(f)
            return { v: Tangent ->
                jvp(grad, x, v)
            }
        }

        fun hvp(f: (Vector) -> Scalar, x: Vector): (Tangent) -> Tangent {
            // return hvpReverseOverReverse(f, x)
            return hvpForwardOverReverse(f, x)
        }
    }

    fun step(f: (Vector) -> Scalar, x: Vector, isAlmostZero: (Tangent) -> Boolean): Vector {
        with (vectorSpace) {
            val grad = grad(f, x)
            val H = hvp(f, x)
            val b = with (vectorSpace.tangentVectorSpace) { -grad }
            val dx0 = b
            val dx = cgSolver.solve(H, b, dx0, maxIters = maxLinSolveIters, isAlmostZero = isAlmostZero)
            return x + dx
        }
    }
}