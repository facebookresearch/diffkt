/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.sim

import org.diffkt.*

class NeohookeanTriangle(
    val i1: Int, val i2: Int, val i3: Int,
    val restShapeInv: Matrix2x2,
    val mu: Float = 1f,
    val lambda: Float = 1f,
    val restArea: Float = 1f
) {
    fun shape(vertices: List<Vertex>): Matrix2x2 {
        val a = vertices[i1].pos
        val b = vertices[i2].pos
        val c = vertices[i3].pos

        val abx = b.x - a.x
        val aby = b.y - a.y

        val acx = c.x - a.x
        val acy = c.y - a.y

        return Matrix2x2(
            abx, acx,
            aby, acy
        )
    }

    fun deformationGradient(vertices: List<Vertex>): Matrix2x2 {
        val shape = this.shape(vertices)
        return shape.mm(restShapeInv)
    }

    fun qlog(x: DScalar): DScalar {
        return -1.5f + 2f * x - 0.5f * x * x
    }

    fun energy(vertices: List<Vertex>): DScalar {
        val F = deformationGradient(vertices)
        val I1 = F.q()
        val J = F.det()
        val qlogJ = qlog(J)
        val energyDensityMu = mu * (0.5f * (I1 - 2f) - qlogJ)
        val energyDensityLambda = lambda * 0.5f * qlogJ * qlogJ
        return restArea * (energyDensityMu + energyDensityLambda)
    }
}