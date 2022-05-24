/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody.oo

import examples.hookeanSpring.softbody.Edge
import examples.hookeanSpring.softbody.Vertex
import org.diffkt.*

sealed class Gravity private constructor(
    val gravityConstant: Float,
    val ground: Float
) {
    abstract fun energy(vertices: List<Vertex>): DScalar

    private class GravitySeq(
        gravityConstant: Float,
        ground: Float): Gravity(gravityConstant, ground) {
        override fun energy(vertices: List<Vertex>): DScalar {
            return vertices.fold(FloatScalar(0f) as DScalar) { acc, vertex ->
                acc + (vertex.pos.y - ground) * gravityConstant * vertex.mass
            }
        }
    }

    private class GravityParallel(
        gravityConstant: Float,
        ground: Float): Gravity(gravityConstant, ground) {
        override fun energy(vertices: List<Vertex>): DScalar {
            return vertices.parallelStream().map { vertex ->
                 (vertex.pos.y - ground) * gravityConstant * vertex.mass
            }.reduce(DScalar::plus).get()
        }
    }

    companion object {
        operator fun invoke(
            gravityConstant: Float = 9.8f,
            ground: Float = 0f,
            parallel: Boolean = false
        ): Gravity =
            if (parallel)
                GravityParallel(gravityConstant, ground)
            else
                GravitySeq(gravityConstant, ground)
    }
}

sealed class Springs private constructor(val edges: List<Edge>, val k: Float) {
    abstract fun energy(vertices: List<Vertex>): DScalar

    private class SpringsSeq(edges: List<Edge>, k: Float): Springs(edges, k) {
        override fun energy(vertices: List<Vertex>): DScalar {
            return edges.fold(FloatScalar(0f) as DScalar) { acc, edge ->
                val left = vertices[edge.left]
                val right = vertices[edge.right]
                val dx = left.pos.x - right.pos.x
                val dy = left.pos.y - right.pos.y
                val q = dx * dx + dy * dy
                val l = (q + 1e-6f).pow(0.5f)
                val dl = l - edge.restLength
                acc + 0.5f * (k * (dl * dl))
            }
        }
    }

    private class SpringsParallel(edges: List<Edge>, k: Float): Springs(edges, k) {
        override fun energy(vertices: List<Vertex>): DScalar {
            return edges.parallelStream().map { edge ->
                val left = vertices[edge.left]
                val right = vertices[edge.right]
                val dx = left.pos.x - right.pos.x
                val dy = left.pos.y - right.pos.y
                val q = dx * dx + dy * dy
                val l = (q + 1e-6f).pow(0.5f)
                val dl = l - edge.restLength
                0.5f * (k * (dl * dl))
            }.reduce(DScalar::plus).get()
        }
    }

    companion object {
        operator fun invoke(edges: List<Edge>, k: Float, parallel: Boolean = false): Springs =
            if (parallel)
                SpringsParallel(edges, k)
            else
                SpringsSeq(edges, k)
    }
}

data class System(
    val gravity: Gravity,
    val springs: Springs
) {
    fun energy(vertices: List<Vertex>): DScalar {
        return gravity.energy(vertices) + springs.energy(vertices)
    }
}
