/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.quasistatic

import examples.utils.visualization.meshvis2d.mesh.Mesh
import org.diffkt.*
import examples.utils.visualization.meshvis2d.openrndr.Viewport
import examples.utils.visualization.meshvis2d.viewport.DragEvent
import examples.utils.visualization.meshvis2d.viewport.Scene
import examples.utils.visualization.meshvis2d.viewport.VertexDragEvent
import examples.utils.visualization.meshvis2d.math.Vector2 as vec2

fun makeInitialScene(neohookeanTriangles: NeohookeanTriangles, x: DTensor): Scene {
    val numVertices = x.shape[0]
    return Scene.build {
        addMesh(Mesh.build {
            makeDraggable()
            (0 until numVertices).forEach { i ->
                val pos = vec2(
                    x[i, 0].basePrimal().at(0),
                    x[i, 1].basePrimal().at(0)
                )
                addVertex(pos, 0.1f, 0.3f)
            }

            neohookeanTriangles.triangleIndices.forEach { triangle ->
                addTriangle(triangle.a, triangle.b, triangle.c)
            }
        })
    }
}

fun main() {
    val spaceDim = 2
    val numVertices = 4

    var x: DTensor = FloatTensor(
        Shape(numVertices, spaceDim),
        floatArrayOf(
            1.9f, 0f,
            1f, 0f,
            0f, 1f,
            1f, 1f
        )
    )

    val elementDim = 2
    val triangleIndices = listOf(
        IndexTriplet(0, 1, 2),
        IndexTriplet(3, 2, 1)
    )
    val numTriangles = triangleIndices.size
    val restShapeInv = FloatTensor(
        Shape(numTriangles, elementDim, elementDim),
        floatArrayOf(
            1f, 0f,
            0f, 1f,

            1f, 0f,
            0f, 1f
        ),
    )

    val triangles = NeohookeanTriangles(triangleIndices, restShapeInv, numVertices)

    val viewport = Viewport(500, 500)

    var simulationIsPaused = false

    fun update(scene: Scene, dragEvents: List<DragEvent>): Scene {
        dragEvents.forEach { event ->
            when (event) {
                is VertexDragEvent.VertexDragBegin ->
                    simulationIsPaused = true
                is VertexDragEvent.VertexDrag ->
                    x = x.withChange(event.vertex.id, 0, tensorOf(event.pos.x, event.pos.y))
                is VertexDragEvent.VertexDragEnd ->
                    simulationIsPaused = false
            }
        }

        if (!simulationIsPaused) {
            val grad = reverseDerivative(x, triangles::energy)
            x -= grad * 0.02f
        }

        val vertexPositions = (0 until numVertices).map { i ->
            vec2(
                x[i, 0].basePrimal().at(0),
                x[i, 1].basePrimal().at(0)
            )
        }
        return scene.withVertexPositions(vertexPositions)
    }

    viewport.lookAt(vec2(0f, 0f), 5f)
    viewport.show(makeInitialScene(triangles, x), ::update)
}