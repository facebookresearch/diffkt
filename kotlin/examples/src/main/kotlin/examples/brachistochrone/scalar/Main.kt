/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.brachistochrone.scalar

import org.diffkt.*
import kotlin.system.measureNanoTime

fun main() {
    val numberSections = 100
    val xStart = 0f
    val xEnd = 4f
    val yStart = 1f
    val yEnd = 0f

    val x = linspace(xStart, xEnd, numberSections + 1).map { FloatScalar(it) }
    val y = linspace(yStart, yEnd, numberSections + 1).map { FloatScalar(it) }.toMutableList<DScalar>()

    /**
     * Computes the time taken to slide down the slope by summing the time it takes for each section.
     * This version operates in scalar mode by iterating over the sections.
     */
    fun computeTimeTaken(x: List<FloatScalar>, y: List<DScalar>): DScalar {
        val g = 9.8f

        var totalTimeAcc: DScalar = FloatScalar(0f)
        for (i in 1 until numberSections + 1) {
            // since energy is conserved, we can compute the velocity by knowing the current y location
            val prevVelocity = sqrt(2f * g * (yStart - y[i - 1]))
            val velocity = sqrt(2f * g * (yStart - y[i]))

            val deltaX = x[i] - x[i - 1]
            val deltaY = y[i] - y[i - 1]

            val dist = sqrt(deltaX.pow(2) + deltaY.pow(2))

            // the time taken on this section is approximately the Euclidean distance divided by the average velocity
            totalTimeAcc += dist / ((prevVelocity + velocity) / 2)
        }

        return totalTimeAcc
    }

    // the goal is to find the path that requires the least amount of time to slide down (no initial velocity)
    val iterations = 10000
    val elapsed = measureNanoTime {
        repeat(iterations) {
            val yDerivatives = reverseDerivative(y) { yy -> computeTimeTaken(x, yy) }

            for (i in 1 until numberSections) {
                y[i] = y[i] - 0.01f * yDerivatives[i]
            }
            y[0] = FloatScalar(yStart)
            y[numberSections] = FloatScalar(yEnd)
        }
    }

    println("brachistochrone_scalar compute = ${elapsed / 1E9} sec")
    println("brachistochrone_scalar rollingTime = ${computeTimeTaken(x, y)}")
}

private fun linspace(start: Float, end: Float, steps: Int): List<Float> {
    val sep = (end - start) / (steps - 1)
    return listOf(start) + (1 until steps - 1 ).map { start + sep * it } + end
}
