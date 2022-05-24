/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.dynamic

import org.diffkt.*

class System(val triangles: List<NeohookeanTriangle> = listOf()) {
    fun addTriangle(
        i1: Int, i2: Int, i3: Int,
        restShapeInv: Matrix2x2,
        mu: Float = 1f, lambda: Float = 1f,
        restArea: Float = 1f
    ): System {
        val triangle = NeohookeanTriangle(i1, i2, i3, restShapeInv, mu, lambda, restArea)
        return System(triangles + triangle)
    }

    fun potentialEnergy(vertices: List<Vertex>): DScalar {
        return triangles.map { triangle -> triangle.energy(vertices) }.reduce(DScalar::plus)
    }

    fun makeBackwardEulerLoss(s0: SystemState, h: Float): (SystemState) -> DScalar {
        fun loss(s: SystemState): DScalar {
            val inertia = s0.vertices.zip(s.vertices) {
                    v0, v1 ->
                val y = v0.pos + v0.vel * h
                val d = v1.pos - y
                d.q() * v0.mass
            }.reduce(DScalar::plus) * 0.5f
            val h2 = h * h
            return h2 * potentialEnergy(s.vertices) + inertia
        }
        return ::loss
    }
}