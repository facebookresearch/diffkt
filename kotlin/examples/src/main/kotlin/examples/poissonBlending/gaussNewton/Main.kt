/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.poissonBlending.gaussNewton

import examples.utils.conjugateGradient.ConjugateGradientLinearSolver
import examples.poissonBlending.*
import examples.poissonBlending.DTensorVectorSpace.dot
import examples.utils.timing.warmupAndTime
import org.diffkt.*

/**
 * Returns a lambda that given v, computes J^T * J * v, used for the approximation of the Hessian vector product
 * for minimizing a sum of squared function values. (In other words, for minimizing sum(residualF(x).pow(2)).)
 *
 * The Jacobian matrix, J, is a matrix of partial derivatives of the input coefficients [x]
 * w.r.t. the output of [residualF] evaluated at [x].
 *
 */
fun JTJv(x: DTensor, residualF: (DTensor) -> DTensor): (DTensor) -> DTensor {
    return { v: DTensor ->
        vjp(x, jvp(x, v, residualF), residualF)
    }
}

/**
 * Uses the Gauss-Newton method, a modification of Newton's method.
 *
 * For each update, the gradient is scaled by an approximation of the Hessian.
 *
 * The gradient is 2 * J^T * r, where r is the residual values and J is the Jacobian of inputs to residuals.
 * The Hessian is approximated by 2 * J^T * J.
 * The update, dx, is (2 * J^T * J)^-1 * 2 * J^T * r => (J^T * J)^-1 * J^T * r.
 *
 * Gauss-Newton produces similar results much faster and with fewer iterations compared to gradient descent.
 */
fun run(iters: Int, maxLinSolveIters: Int): DTensor {
    val cg = ConjugateGradientLinearSolver(DTensorVectorSpace)

    var x = base
    repeat(iters) {
        val grad = vjp(x, { r -> r }, ::residuals) // J^T * r

        // Solves for dx in J^T * J * dx = - grad, such that dx = - (J^T * J)^-1 * grad
        val dx = cg.solve(JTJv(x, ::residuals),
            b = - grad,
            x0 = - grad,
            maxIters = maxLinSolveIters
        ) { it.dot(it) < 1e-5f }

        x += dx
    }
    return x
}


fun main(args: Array<String>) {
    val iters = args.getOrNull(0)?.toIntOrNull() ?: 5
    val maxLinSolveIters = args.getOrNull(1)?.toIntOrNull() ?: 100

    val isBenchmark = args.lastOrNull() == "benchmark"
    if (isBenchmark) {
        val time = warmupAndTime({ run(iters, maxLinSolveIters) })
        println("===========================================")
        println("POISSON IMAGE EDITING: GAUSS-NEWTON")
        println("iters: $iters  maxLinSolveIters: $maxLinSolveIters")
        println("time: $time ns")
        println("loss: ${loss(run(iters, maxLinSolveIters))}")
        return
    }

    val x = run(5, 100)
    println("final loss: ${loss(x)}")

    val outputFilePath = "$resourcesPath/result.png"
    saveTensorAsImage(x, outputFilePath)
}
