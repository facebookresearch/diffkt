/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody.tensor

import examples.hookeanSpring.softbody.Edge
import examples.hookeanSpring.softbody.mass
import org.diffkt.*

class Gravity(
    val gravityConstant: Float,
    val ground: Float
) {
    fun energy(vertices: DTensor): DScalar =
        (mass * gravityConstant * (vertices.view(index = 1, axis = 1) - ground)).sum()
}

class Springs(val edges: List<Edge>, val k: Float, private val numVertices: Int) {
    private val numSprings = edges.size
    private val incidence: DTensor = makeIncidence()
    private val l0 = FloatTensor(Shape(edges.size), edges.map { it.restLength }.toFloatArray())

    /** Creates a dense matrix (incidence) representing the connections between vertices. */
    private fun makeIncidence(): DTensor {
        val incidenceList = MutableList(numSprings * numVertices) { 0.0f }
        for (i in edges.indices) {
            incidenceList[i * numVertices + edges[i].left] = 1.0f
            incidenceList[i * numVertices + edges[i].right] = -1.0f
        }
        return FloatTensor(Shape(numSprings, numVertices), incidenceList.toFloatArray())
    }

    fun energy(vertices: DTensor): DScalar {
        val d = incidence.matmul(vertices)
        val q = (d * d).sum(1)
        val l = (q + 1e-6f).pow(0.5f)
        val dl = l - l0
        return 0.5f * (k * (dl * dl)).sum()
    }
}

data class System(
    val gravity: Gravity,
    val springs: Springs
) {
    fun energy(vertices: DTensor): DScalar {
        return gravity.energy(vertices) + springs.energy(vertices)
    }
}
