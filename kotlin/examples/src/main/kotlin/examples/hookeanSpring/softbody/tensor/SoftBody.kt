/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody.tensor

import examples.hookeanSpring.softbody.Vertex
import examples.hookeanSpring.softbody.*
import examples.utils.mapIndexed
import examples.utils.timing.warmupAndTime
import examples.utils.visualization.animate
import org.diffkt.*

/**
 * Times the reverse derivative computation for the softbody example.
 * [s] is the number of vertices on the side of the square.
 */
fun benchmark(s: Int) {
    val N = s * s
    val edges = makeEdges(s)
    val initVertices = makeVertices(s).toPositionTensor()

    val system = System(
        gravity = Gravity(9.8f, ground),
        springs = Springs(edges, k, N)
    )
    val time = warmupAndTime( {
        reverseDerivative(
            initVertices,
            system::energy
        )
    } )
    println("SoftBody time: $time")
}

/**
 * Runs the softbody example with animation.
 * [s] is the number of vertices on the side of the square.
 */
fun run(s: Int) {
    val N = s * s
    val edges = makeEdges(s)
    val initVertices = makeVertices(s).toPositionTensor()

    val system = System(
        gravity = Gravity(9.8f, ground),
        springs = Springs(edges, k, N)
    )

    var vel: DTensor = FloatTensor.zeros(Shape(N, D))
    var pos: DTensor = initVertices

    fun update(dt: Float): DTensor {
        val grad = reverseDerivative(pos, system::energy)
        val acc = - grad / mass
        vel = (vel + acc * dt) * dampen
        pos += vel * dt

        val (boundedPos, boundedVel) =
            (pos as FloatTensor to vel as FloatTensor).mapIndexed { i, fl, vel ->
                if (i % D == 0) {
                    val xMin = leftSide
                    val xMax = rightSide
                    when {
                        (fl < xMin) -> xMin to - vel
                        (fl > xMax) -> xMax to - vel
                        else -> fl to vel
                    }
                } else {
                    val yMin = ground
                    if (fl < yMin) yMin to - vel else fl to vel
                }
            }
        vel = boundedVel
        return boundedPos
    }

    animate(
        name = "SoftBody",
        init = initVertices,
        update = { update(dt) }, // Function to update vertex positions, low time step for animation.
        edges = edges.map { it.left to it.right }, // SpringSystem (used for initial vertex positions and edges)
        interval = 30
    )
}

/**
 * Helper function to convert [this] (List<Vertex) to a FloatTensor representing the vertex positions.
 * The mass of each vertex is ignored.
 */
private fun List<Vertex>.toPositionTensor(): FloatTensor {
    return FloatTensor(Shape(this.size, 2),
        this.flatMap { listOf(it.pos.x.basePrimal().value, it.pos.y.basePrimal().value) }.toFloatArray()
    )
}
