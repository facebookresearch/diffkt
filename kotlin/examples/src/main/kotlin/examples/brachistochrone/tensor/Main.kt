/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.brachistochrone.tensor

import org.diffkt.*
import kotlin.system.measureNanoTime

val numberSections = 100
val xStart = 0f
val xEnd = 4f
val yStart = 1f
val yEnd = 0f

val x = meld(linspace(xStart, xEnd, numberSections + 1))
val initialY = meld(linspace(yStart, yEnd, numberSections + 1))

private val prevX = x.slice(0, numberSections, 0)
private val nextX = x.slice(1, numberSections + 1, 0)
private val deltaX = nextX - prevX
private val deltaX2 = deltaX.pow(2)

/**
 * Computes the time taken to slide down the slope by summing the time it takes for each section.
 * This version operates in tensor mode by processing all of the sections at once.
 */
fun computeTimeTaken(y: DTensor): DScalar {
    val g = 9.8f

    val velocity = sqrt(2F * g * (yStart - y))

    val prevY = y.slice(0, numberSections, 0)
    val nextY = y.slice(1, numberSections + 1, 0)

    val prevVelocity = velocity.slice(0, numberSections, 0)
    val nextVelocity = velocity.slice(1, numberSections + 1, 0)

    val averageVelocity = (prevVelocity + nextVelocity) / 2F

    val deltaY = nextY - prevY
    val dist = sqrt(deltaX2 + deltaY.pow(2))

    val time = dist / averageVelocity

    val totalTime = time.sum()
    return totalTime
}

fun learn(y: DTensor, learningRate: Float): DTensor {
    val yDerivative = reverseDerivative(y) { yy -> computeTimeTaken(yy) }
    val y2 = y - learningRate * yDerivative
    val y3 = y2.withChange(0, 0, FloatScalar(yStart))
    val y4 = y3.withChange(numberSections, 0, FloatScalar(yEnd))
    return y4
}

fun main() {
    var y = initialY

    // the goal is to find the path that requires the least amount of time to slide down (no initial velocity)
    val iterations = 10000
    val elapsed = measureNanoTime {
        repeat(iterations) {
            y = learn(y, 0.01F)
        }
    }

    println("brachistochrone compute = ${elapsed / 1E9} sec")
    println("brachistochrone rollingTime = ${computeTimeTaken(y)}")
}

fun linspace(start: Float, end: Float, steps: Int): DTensor {
    val sep = (end - start) / (steps - 1)
    val l = listOf(start) + (1 until steps - 1 ).map { start + sep * it } + end
    return meld(l.map{ FloatScalar(it) })
}
