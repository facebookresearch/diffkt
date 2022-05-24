/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.springArea

import examples.utils.visualization.plotting.display
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggsize
import jetbrains.letsPlot.lets_plot
import org.diffkt.*
import kotlin.math.exp
import kotlin.math.pow

internal object SpringArea{
    const val steps = 1024

    const val N = 3
    const val mass = 1f
    const val springStiffness = 10f
    const val damping = 20f

    const val dt = 0.001f

    const val dampening = 0.99f

    const val targetArea = 0.1f

    const val interval = 256

    val edges = listOf(
        Pair(0, 1),
        Pair(1, 2),
        Pair(2, 0)
    )

    val initPositions = FloatTensor(Shape(N, 2),
        0.3f, 0.5f,
        0.3f, 0.4f,
        0.4f, 0.4f
    )

    // Helper tensor operations

    fun DTensor.norm() = (this.pow(2).sum() + 1e-8f).pow(0.5f)

    operator fun DTensor.compareTo(that: Float) = (this as DScalar).basePrimal().value.compareTo(that)

    fun springEnergy(a: DTensor, b: DTensor, springLength: DTensor): DScalar {
        val dist = a - b
        val length = dist.norm()
        val dl = length - springLength
        return 0.5f * (springStiffness * dl.pow(2)).sum()
    }

    fun applySpringForce(x: DTensor, springLengths: DTensor): DTensor {
        var forces: DTensor = FloatTensor.zeros(Shape(N, 2))
        edges.forEachIndexed { i, edge ->
            val (a, b) = edge
            val springLength = springLengths[i]
            val (forceA, forceB) = reverseDerivative(x[a], x[b]) { xA, xB ->
                - springEnergy(xA, xB, springLength)
            }
            forces = forces.withChange(a, 0, forces[a] + forceA)
            forces = forces.withChange(b, 0, forces[b] + forceB)
        }
        return forces
    }

    fun timeIntegrate(forces: DTensor, xInit: DTensor, vInit: DTensor): Pair<DTensor, DTensor> {
        val s = exp(-dt * damping)
        var x = xInit
        var v = vInit
        for (i in 0 until N) {
            val newV = (s * v[i] + dt * forces[i] / mass) * dampening
            val newX = x[i] + dt * newV
            x = x.withChange(i, 0, newX)
            v = v.withChange(i, 0, newV)
        }

        return x to v
    }

    fun computeLoss(x: DTensor): DScalar {
        val x01 = x[0] - x[1]
        val x02 = x[0] - x[2]
        val area = 0.5f * abs((x01[0] * x02[1] - x01[1] * x02[0]) as DScalar)
        return (area - targetArea).pow(2)
    }

    fun forward(springLengths: DTensor): DScalar {
        var x: DTensor = initPositions
        var v: DTensor = FloatTensor.zeros(Shape(N, 2))

        for (t in 0 until steps) {
            val forces = applySpringForce(x, springLengths)
            val (newX, newV) = timeIntegrate(forces, x, v)
            x = newX; v = newV

            if ((t + 1) % interval == 0) {
                Visualization.capture(x.basePrimal())
            }
        }
        return computeLoss(x)
    }
}

fun main() {
    var springLengths: DTensor = FloatTensor(Shape(3, 1), 0.1f, 0.1f, 0.1f * 2f.pow(0.5f))
    val learningRate = 5f

    val losses = mutableListOf<Float>()
    val iterations = 0 until 25
    for (i in iterations) {
        val (loss, grad) = primalAndReverseDerivative(springLengths, SpringArea::forward)
        springLengths -= grad * learningRate
        println("iter $i $loss")
        losses.add((loss as DScalar).basePrimal().value)
    }
    println(springLengths)

    val data = mapOf("loss" to losses, "iteration" to iterations)
    lets_plot(data)
        .plus(geom_line(color="blue") { x = "iteration"; y = "loss" })
        .plus(ggsize(500, 250))
        .display("Loss")

    // Comment out the plotting code above and uncomment the below line to visualize.
//    Visualization.visualize()
}
