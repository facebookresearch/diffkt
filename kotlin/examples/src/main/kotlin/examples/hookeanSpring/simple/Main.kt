/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.simple

import examples.utils.visualization.animate
import org.diffkt.*

fun SpringSystem.update(x: DTensor, lr: Float): DTensor {
    val grad = reverseDerivative(x, ::energy)
    return x - grad * lr
}

fun main() {
    val springs = SpringSystem.DEFAULT

    animate(
        name = "Hookean Spring",
        init = springs.initVertices,
        update = { x -> springs.update(x, lr = 0.0005f) }, // Function to update vertex positions, low learning rate for animation.
        edges = springs.edges, // SpringSystem (used for initial vertex positions and edges)
        interval = 30
    )
}

class SpringSystem (
    dimensions: Int,
    vertices: List<Float>,
    val edges: List<Pair<Int, Int>>,
    restLengths: List<Float>,
    val k: Float = 1f
) {
    init {
        require(vertices.size % dimensions == 0)
        require(restLengths.size == edges.size)
    }

    private val numSprings = edges.size
    private val numVertices = vertices.size / dimensions

    val l0 = FloatTensor(Shape(restLengths.size), restLengths.toFloatArray())
    val incidence: DTensor = makeIncidence()

    /** Tensor representing initial vertices. */
    val initVertices: FloatTensor = FloatTensor(Shape(numVertices, dimensions), vertices.toFloatArray())

    /** Creates a dense matrix (incidence) representing the connections between vertices. */
    private fun makeIncidence(): DTensor {
        val incidenceList = MutableList(numSprings * numVertices) { 0.0f }
        for (i in edges.indices) {
            incidenceList[i * numVertices + edges[i].first] = 1.0f
            incidenceList[i * numVertices + edges[i].second] = -1.0f
        }
        return FloatTensor(Shape(numSprings, numVertices), incidenceList.toFloatArray())
    }

    /**
     * [x] is a Shape(numVertices,dimensions) Tensor that contains the position for each vertex.
     * Returns a scalar value representing the energy of the spring system.
     */
    fun energy(x: DTensor): DScalar {
        val d = incidence.matmul(x)
        val q = (d * d).sum(1)
        val l = (q + 1e-6f).pow(0.5f)
        val dl = l - l0
        return 0.5f * (k * (dl * dl)).sum()
    }

    companion object {
        val DEFAULT: SpringSystem =
            SpringSystem(
                dimensions = 2,
                vertices = listOf(
                    0f, 0f,
                    1f, 0f,
                    2f, 0.5f,
                    1f, 1f,
                    1.5f, 2f,
                    2f, -1f,
                    0f, -1f,
                    1f, -2f
                ),
                edges = listOf(
                    Pair(0, 1),
                    Pair(0, 3),
                    Pair(1, 3),
                    Pair(1, 2),
                    Pair(2, 3),
                    Pair(3, 4),
                    Pair(2, 4),
                    Pair(1, 5),
                    Pair(5, 6),
                    Pair(6, 0),
                    Pair(6, 1),
                    Pair(6, 5),
                    Pair(5, 7),
                    Pair(6, 7)
                ),
                restLengths = mutableListOf(
                    1f,
                    1.3f,
                    1.4f,
                    1.5f,
                    2.0f,
                    1.4f,
                    1.4f,
                    1.3f,
                    1f,
                    0.8f,
                    1f,
                    1f,
                    1f,
                    0.9f
                )
            )

    }
}