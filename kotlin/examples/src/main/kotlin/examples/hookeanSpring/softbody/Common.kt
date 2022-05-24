/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody

internal const val ground = -3f
internal const val leftSide = -3f
internal const val rightSide = 3f

internal const val mass = 1f // Mass of each vertex
internal const val k = 600f // Spring stiffness factor

internal const val dt: Float = 0.00025f // Time step
internal const val dampen: Float = .9995f // Dampening factor

internal const val straight = .5f // Length of straight edge
internal val diagonal = kotlin.math.sqrt(straight * straight + straight * straight) // Length of diagonal edge

internal val D = 2 // Dimensionality

/**
 * Create vertices given [s], the number of vertices on the side of the square.
 */
fun makeVertices(s: Int): List<Vertex> {
    return List(s * s) { id ->
        Vertex(id % s * straight + -.75f, id / s * straight + 1f,
            mass
        )
    }
}

/**
 * Create edges given [s], the number of vertices on the side of the square.
 */
fun makeEdges(s: Int): List<Edge> {
    val N = s * s
    return (0 until N).flatMap { x ->
        listOf(
            Edge(x, x + 1, straight),
            Edge(x, x + s, straight),
            Edge(x + 1, x + s, diagonal),
            Edge(x, x + s + 1, diagonal)
        ).mapNotNull { (start, end, len) ->
            if (end >= N || (start % s == s - 1 && end % s == 0) || (start % s == 0 && end % s == s - 1))
                null else Edge(start, end, len)
        }
    }
}
