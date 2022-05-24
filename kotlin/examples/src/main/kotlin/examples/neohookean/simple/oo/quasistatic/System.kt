/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.quasistatic

import org.diffkt.*

fun List<DScalar>.sum() = this.reduce(DScalar::plus)

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

    fun energy(vertices: List<Vertex>): DScalar {
        return triangles.map { triangle -> triangle.energy(vertices) }.sum()
    }
}