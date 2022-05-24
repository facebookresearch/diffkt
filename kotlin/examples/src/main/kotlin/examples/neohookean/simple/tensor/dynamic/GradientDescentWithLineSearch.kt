/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.dynamic

import org.diffkt.*

class GradientDescentWithLineSearch(
    eps: Float = 1e-3f,
    val maxIters: Int = 100,
    val maxLineSearchIters: Int = 100,
    val backtrackingFactor: Float = 0.3f
) {
    val eps2 = eps * eps

    fun optimize(x0: DTensor, f: (DTensor) -> DScalar): DTensor {
        // run an optimization loop until convergence

        var x = x0

        for (i in 0 until maxIters) {
            val (fAtX, grad) = primalAndGradient(x, f)

            // check if we reached convergence
            val q = (grad.pow(2).sum(1) as FloatTensor).max() as FloatScalar
            if (q < eps2) break

            val descentDir = -grad

            // find a good step size to guarantee that we reduce the loss
            val (stepSize, x1) = lineSearch(
                x, f, descentDir,
                fAtX as DScalar
            )
            x = x1
        }

        return x
    }

    fun lineSearch(
        x: DTensor,
        f: (DTensor) -> DScalar,
        descentDir: DTensor,
        fAtX: DScalar
    ): Pair<Float, DTensor> {
        var stepSize = 1.0f
        var x1 = x + descentDir * stepSize
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