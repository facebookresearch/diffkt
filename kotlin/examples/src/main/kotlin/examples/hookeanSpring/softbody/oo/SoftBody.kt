/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody.oo

import examples.hookeanSpring.softbody.*
import examples.utils.timing.warmupAndTime
import examples.utils.visualization.animate
import org.diffkt.FloatScalar
import org.diffkt.compareTo
import org.diffkt.unaryMinus

/**
 * Times the reverse derivative computation for the softbody example.
 * [s] is the number of vertices on the side of the square.
 * If [parallel] is true, the loops in the energy computation are run with `parallelStream`.
*/
fun benchmark(s: Int, parallel: Boolean) {
    val edges = makeEdges(s)
    val initVertices = makeVertices(s)

    val system = System(
        gravity = Gravity(9.8f, ground, parallel),
        springs = Springs(edges, k, parallel)
    )

    val time = warmupAndTime( { reverseDerivative(
        initVertices,
        system::energy
    ) } )
    println("SoftBody OO time: $time")
}

/**
 * Runs the softbody example with animation.
 * [s] is the number of vertices on the side of the square.
 * If [parallel] is true, the loops in the energy computation are run with `parallelStream`.
 */
fun run(s: Int, parallel: Boolean = false) {
    val N = s * s
    val edges = makeEdges(s)
    val initVertices = makeVertices(s)

    val system = System(
        gravity = Gravity(9.8f, ground, parallel),
        springs = Springs(edges, k, parallel)
    )

    var vertices: List<Vertex> = initVertices

    fun update(dt: Float): List<Vertex> {
        val grad = reverseDerivative(
            vertices,
            system::energy
        )

        val acc = grad.map { -it.pos / mass }

        vertices = vertices.mapIndexed { i, vertex ->
            val v = (vertex.vel + acc[i] * dt) * dampen
            val p = vertex.pos + v * dt
            bound(Vertex(p, v, vertices[i].mass))
        }

        return vertices
    }

    animate(
        "SoftBody OO",
        init = vertices,
        edges = edges,
        update = { update(dt) },
        interval = 30
    )
}

/**
 * Helper function to bound vertices by the sides and ground of the animation.
 */
private fun bound(vertex: Vertex): Vertex {
    val (pos, vel, mass) = vertex
    val xMin = leftSide
    val xMax = rightSide
    val (pointX, pointY) = pos

    val (newPointX, newVelX) = when {
        (pointX < xMin) -> xMin to -vel.x
        (pointX > xMax) -> xMax to -vel.x
        else -> (pointX as FloatScalar).value to vel.x
    }

    val yMin = ground

    val (newPointY, newVelY) =
        if (pointY < yMin) yMin to -vel.y else (pointY as FloatScalar).value to vel.y

    return Vertex(Vector2(newPointX, newPointY), Vector2(newVelX, newVelY), mass)
}
