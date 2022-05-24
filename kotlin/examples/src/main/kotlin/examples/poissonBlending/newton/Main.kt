/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.poissonBlending.newton

import examples.utils.conjugateGradient.NewtonCGOptimizer
import examples.poissonBlending.*
import examples.poissonBlending.DTensorVectorSpace.dot
import examples.utils.timing.warmupAndTime
import org.diffkt.*

fun run(iters: Int, maxLinSolveIters: Int): DTensor {
    var x = base
    repeat(iters) {
        x = NewtonCGOptimizer(DTensorVectorSpace, maxLinSolveIters).step(
            ::loss,
            x,
            isAlmostZero = { it.dot(it) < 1e-5f}
        )
    }
    return x
}

/**
 * Uses Newton's method.
 *
 * Because the residual function is a linear equation, Newton's method and Gauss-Newton are numerically equivalent.
 *
 * However, Gauss-Newton is slightly more efficient because we avoid computing the sum of squares.
 *
 * Example benchmark command:
 * ./gradlew benchmark -Ppackage=poissonBlending.cg -Piters=1 -Pargs="1 100; 2 500" -Pfile=results.txt
 *
 */
fun main(args: Array<String>) {
    val iters = args.getOrNull(0)?.toIntOrNull() ?: 5
    val maxLinSolveIters = args.getOrNull(1)?.toIntOrNull() ?: 100

    val isBenchmark = args.lastOrNull() == "benchmark"
    if (isBenchmark) {
        val time = warmupAndTime({ run(iters, maxLinSolveIters) })
        println("===========================================")
        println("POISSON IMAGE EDITING: NEWTON")
        println("iters: $iters  maxLinSolveIters: $maxLinSolveIters")
        println("time: $time ns")
        println("loss: ${loss(run(iters, maxLinSolveIters))}")
        return
    }

    val x = run(5, 100)
    val outputFilePath = "$resourcesPath/result.png"
    saveTensorAsImage(x, outputFilePath)
}