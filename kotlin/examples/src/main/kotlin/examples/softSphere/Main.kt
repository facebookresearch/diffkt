/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.softSphere

import examples.utils.mapIndexed
import examples.utils.visualization.animate
import org.diffkt.*

const val N = 50
const val D = 2
val radii = floatArrayOf(.15f, .2f)
val radiusT = FloatTensor(Shape(N), FloatArray(N) { radii[it % radii.size] })
val l0 = FloatTensor(Shape(N, N), FloatArray(N * N) {
    val i = it % N
    val j = it / N
    if (i != j) {
        val r1 = radii[i % radii.size]
        val r2 = radii[j % radii.size]
        r1 + r2
    } else 0f
})
const val ground = -3f
const val leftSide = -3f
const val rightSide = 3f
val groundT = ground + radiusT
val mass = (10000f * 4f / 3f * Math.PI.toFloat() * radiusT.pow(3))
const val overlapCoefficient = 100000f // Large because molecules should never really overlap.

/**
 * [x] is a (N,D) tensor where N is the number of points and D is the dimensionality.
 *
 * Returns a (N,N) tensor of L2 distances between all points.
 */
fun distances(x: DTensor, eps: Float = 1e-5f): DTensor {
    val x2 = x.pow(2).sum(1, keepDims = true)
    val mm = x.matmul(x.transpose()) * -2f
    return (mm + x2 + x2.transpose() + eps).pow(.5f)
}

fun height(x: DTensor) = mass * 9.8f * (x.transpose()[1] - groundT)

fun overlap(x: DTensor) = relu(l0 - distances(x))

fun energy(x: DTensor): DScalar {
    val overlap = overlap(x)
    val overlapEnergy = overlapCoefficient * (overlap * overlap).sum()
    val heightEnergy = height(x).sum()
    return overlapEnergy + heightEnergy
}

fun initialPositionsTensor() = FloatTensor(Shape(N, D), FloatArray(N * D) { i ->
    if (i % 2 == 0)
        (i % 10) / 6f
    else
        i / 10 / 6f - 1.5f
})

/**
 * Runs the example taking into account acceleration (for a "real-time" simulation).
 *
 * We can obtain the acceleration through the following:
 * Derivative(energy) = - Force = - mass * acc =>
 * acc = - Derivative(energy) / mass
 *
 * For each time step, we update the velocity with the acceleration and the
 * position by the updated velocity.
 *
 */

fun main() {
    val initPositions = initialPositionsTensor()

    var vel: DTensor = FloatTensor.zeros(Shape(N, D))
    val dt = .0025f
    val dampen = .999f

    animate(
        name = "Molecular Dynamics",
        init = initPositions,
        update = { x ->
            val grad = reverseDerivative(x, ::energy)
            val acc = -grad / mass.unsqueeze(1)
            vel += acc * dt
            val pos = x + vel * dt

            // Adjust values based on the bounds.
            val (boundedPos, boundedVel) =
                (pos as FloatTensor to vel as FloatTensor).mapIndexed { i, fl, vel ->
                    val radius = radii[(i / 2) % radii.size]
                    if (i % D == 0) {
                        val xMin = leftSide + radius
                        val xMax = rightSide - radius
                        when {
                            (fl < xMin) -> xMin to -vel
                            (fl > xMax) -> xMax to -vel
                            else -> fl to vel
                        }
                    } else {
                        val yMin = ground + radius
                        if (fl < yMin) yMin to -vel else fl to vel
                    }
                }
            vel = boundedVel * dampen
            boundedPos
        },
        radii = List(N) { i -> radii[i % radii.size].toDouble() },
        interval = 10
    )
}
