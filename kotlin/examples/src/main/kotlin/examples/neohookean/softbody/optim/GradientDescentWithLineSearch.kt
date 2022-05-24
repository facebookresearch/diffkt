/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.optim

import org.diffkt.*

class GradientDescentWithLineSearch<Primal : Any, Tangent : Any>(
    val differentiableSpace: DifferentiableSpace<Primal, Tangent, DScalar>,
    val maxLineSearchIters: Int = 100,
    val backtrackingFactor: Float = 0.3f,
    val maxIters: Int = 100
) {
    fun optimize(
        x: Primal, f: (Primal) -> DScalar,
        isAlmostZero: (Tangent) -> Boolean
    ): Pair<Primal, GradientDescentWithLineSearch<Primal, Tangent>> {
        var optimizer = this
        var x1 = x

        with (differentiableSpace) {
            for (i in 0 until maxIters) {
                val (primal, grad) = primalAndGradient(x1, f, extractInputTangent = ::extractInputTangent)
                if (isAlmostZero(grad)) break
                val descentDir = with (differentiableSpace.tangentVectorSpace) { -grad }
                x1 = updateWithLineSearch(x1, f, descentDir, primal)
                optimizer = GradientDescentWithLineSearch(differentiableSpace, maxLineSearchIters, backtrackingFactor, maxIters)
            }
        }
        return Pair(x1, optimizer)
    }

    fun updateWithLineSearch(
        x: Primal,
        f: (Primal) -> DScalar,
        descentDir: Tangent,
        fAtX: DScalar
    ): Primal {
        with (differentiableSpace) {
            var stepSize: DScalar = FloatScalar(1f)
            var x1 = x + with (differentiableSpace.tangentVectorSpace) { descentDir * stepSize }
            var fAtX1 = f(x1)
            var iters = 0

            while (fAtX1 >= fAtX && iters < maxLineSearchIters) {
                stepSize *= backtrackingFactor
                x1 = x + with (differentiableSpace.tangentVectorSpace) { descentDir * stepSize }
                fAtX1 = f(x1)
                iters++
            }

            return x1
        }
    }
}