/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.sim

import org.diffkt.*

fun List<DScalar>.sum() = if (this.isEmpty()) FloatScalar(0f) else this.reduce(DScalar::plus)

class System(
    val triangles: List<NeohookeanTriangle> = listOf(),
    val lineColliders: List<LineCollider> = listOf(),
    val g: Float = 9.8f
) {
    fun energy(systemState0: SystemState, systemState: SystemState, dt: Float): DScalar {
        val triangleEnergy = triangles.map { it.energy(systemState.vertices) }.sum()
        val gravityEnergy = systemState.vertices.map { vertex ->
            g * vertex.pos.y * vertex.mass
        }.sum()
        val lineColliderEnergy = lineColliders.map { it.energy(systemState0, systemState, dt) }.sum()
        return triangleEnergy + lineColliderEnergy + gravityEnergy
    }

    fun makeBackwardEulerLoss(state0: SystemState, h: Float): (SystemState) -> DScalar {
        return { state ->
            require(state0.vertices.size == state.vertices.size) {
                "inconsistent number of vertices ${state0.vertices.size} != ${state.vertices.size}"
            }

            val inertia = state0.vertices.zip(state.vertices) { vertex0, vertex1 ->
                val y = vertex0.pos + vertex0.vel * h
                val d = (vertex1.pos - y) * vertex0.mass
                d.q()
            }.sum()

            0.5f * inertia + h * h * energy(state0, state, h)
        }
    }

    class Builder(
        val triangles: MutableList<NeohookeanTriangle> = mutableListOf(),
        val lineColliders: MutableList<LineCollider> = mutableListOf(),
        var g: Float = 9.8f
    ) {
        fun addTriangle(
            i1: Int, i2: Int, i3: Int,
            restShapeInv: Matrix2x2,
            mu: Float = 1f, lambda: Float = 1f,
            restArea: Float = 1f
        ) {
            val triangle = NeohookeanTriangle(i1, i2, i3, restShapeInv, mu, lambda, restArea)
            triangles.add(triangle)
        }

        fun addLineCollider(
            pos: Vector2,
            normal: Vector2, collisionK: Float = 500f,
            frictionEps: Float = 0.01f, frictionK: Float = 100f
        ) {
            val lineCollider = LineCollider(pos, normal, collisionK, frictionEps, frictionK)
            lineColliders.add(lineCollider)
        }

        fun build(): System {
            return System(triangles, lineColliders, g)
        }
    }

    companion object {
        fun build(init: Builder.() -> Unit): System {
            val builder = Builder()
            builder.init()
            return builder.build()
        }
    }
}