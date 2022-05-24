/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.freefall

import examples.utils.visualization.plotting.display
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggsize
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.lets_plot
import org.diffkt.*

/** Acceleration due to gravity: 9.8 m/s^2 **/
const val G = 9.8f
/** Mass of object **/
const val mass = 2f
/** Height of "ground" **/
const val H0 = 0f

/**
 * Computes the potential energy of an object at free fall with mass [mass] at height [h].
 */
fun energy(h: DScalar): DScalar {
    val dh = h - H0
    return mass * G * dh
}

fun buildPlot(data: Map<String, List<Number>>): Plot {
    var p = lets_plot(data)
    p += geom_line(color="dark_green") { x="time"; y="height" }
    p + ggsize(500, 250)
    return p
}

fun main() {
    val secondsPerIteration = 0.001f // seconds / iteration
    val pointsToPlot = mutableListOf<Pair<DScalar, Float>>()

    // Starting height
    var h: DScalar = FloatScalar(100f)
    var v: DScalar = FloatScalar(0f)

    // Update height.
    repeat(100000) { i ->
        val grad = reverseDerivative(h, ::energy)
        val timeElapsed = i * secondsPerIteration

        // The unit of measure of grad is g * m / s^2.
        // Update velocity (m/s).
        v -= secondsPerIteration * grad / mass

        // Update height (m).
        h += secondsPerIteration * v

        // Bounce back up (no drag, friction).
        if (h <= 0f) {
            h = FloatScalar(0f)
            v *= -1f
        }

        // Record for plot.
        pointsToPlot.add(h to timeElapsed)
    }

    val heights = pointsToPlot.map { (it.first as FloatScalar).value }
    val times = pointsToPlot.map { it.second }

    buildPlot(data = mapOf("height" to heights, "time" to times))
        .display(name = "Free Fall Height over Time")
}
