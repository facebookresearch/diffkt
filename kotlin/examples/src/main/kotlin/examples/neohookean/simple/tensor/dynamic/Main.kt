/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.dynamic

import org.diffkt.*
import examples.utils.visualization.meshvis2d.openrndr.Viewport
import examples.utils.visualization.meshvis2d.viewport.DragEvent
import examples.utils.visualization.meshvis2d.viewport.Scene
import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.viewport.VertexDragEvent
import examples.utils.visualization.meshvis2d.math.Vector2 as vec2

fun makeInitialScene(system: System, x: DTensor): Scene {
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
            system.triangles.triangleIndices.forEach { triangle ->
                addTriangle(triangle.a, triangle.b, triangle.c)
            }
        })
    }
}

fun main() {
    val spaceDim = 2
    val numVertices = 4
    val elementDim = 2

    val triangleIndices = listOf(
        IndexTriplet(0, 1, 2),
        IndexTriplet(3, 2, 1)
    )
    val numTriangles = triangleIndices.size

    val restShapeInv = FloatTensor(
        Shape(numTriangles, elementDim, elementDim),
        1f, 0f,
        0f, 1f,

        1f, 0f,
        0f, 1f
    )

    val system = System(
        NeohookeanTriangles(
            triangleIndices, restShapeInv, numVertices,
            mu = 100f, lambda = 10f
        )
    )

    val x: DTensor = FloatTensor(
        Shape(numVertices, spaceDim),
        floatArrayOf(
            1.9f, 0f,
            1f, 0f,
            0f, 1f,
            1f, 1f
        )
    )
    val v: DTensor = FloatTensor(
        Shape(numVertices, spaceDim),
        floatArrayOf(
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f
        )
    )
    val m: DTensor = FloatTensor.ones(Shape(numVertices))
    var state = Vertices(x, v, m)

    val optimizer = GradientDescentWithLineSearch()
    val h = 0.033f

    val viewport = Viewport(500, 500)

    var simulationIsPaused = false

    fun update(scene: Scene, dragEvents: List<DragEvent>): Scene {
        dragEvents.forEach { event ->
            when (event) {
                is VertexDragEvent.VertexDragBegin ->
                    simulationIsPaused = true
                is VertexDragEvent.VertexDrag ->
                    state = state.withX(state.x.withChange(event.vertex.id, 0, tensorOf(event.pos.x, event.pos.y)))
                is VertexDragEvent.VertexDragEnd ->
                    simulationIsPaused = false
            }
        }

        if (!simulationIsPaused) {
            val s0 = state
            val loss = system.makeBackwardEulerLoss(s0, h)
            val s1 = state.preBackwardEulerOptim(h)
            var x1 = s1.x
            x1 = optimizer.optimize(x1, loss)
            state = s0.postBackwardEulerOptim(x1, h)
        }

        val vertexPositions = (0 until numVertices).map { i ->
            vec2(
                state.x[i, 0].basePrimal().at(0),
                state.x[i, 1].basePrimal().at(0)
            )
        }
        return scene.withVertexPositions(vertexPositions)
    }

    viewport.lookAt(vec2(0f, 0f), 5f)
    viewport.show(makeInitialScene(system, x), ::update)
}