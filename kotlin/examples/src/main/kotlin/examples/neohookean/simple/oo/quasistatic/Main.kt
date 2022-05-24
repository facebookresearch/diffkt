/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.quasistatic

import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.openrndr.Viewport
import examples.utils.visualization.meshvis2d.viewport.DragEvent
import examples.utils.visualization.meshvis2d.viewport.Scene
import examples.utils.visualization.meshvis2d.viewport.VertexDragEvent
import org.diffkt.basePrimal
import examples.utils.visualization.meshvis2d.math.Vector2 as vec2

fun List<Vertex>.withVertexPos(i: Int, x: Float, y: Float) =
    this.mapIndexed { j, vertexJ -> if (i == j) Vertex(x, y) else vertexJ }

fun makeInitialScene(system: System, vertices: List<Vertex>): Scene {
    return Scene.build {
        addMesh(Mesh.build {
            makeDraggable()
            vertices.forEach { vertex ->
                addVertex(vec2(vertex.x.basePrimal().value, vertex.y.basePrimal().value), 0.1f, 0.3f)
            }
            system.triangles.forEach { triangle ->
                addTriangle(triangle.i1, triangle.i2, triangle.i3)
            }
        })
    }
}

fun main() {
    val system = System()
        .addTriangle(0, 1, 2, Matrix2x2.identity())
        .addTriangle(3, 2, 1, Matrix2x2.identity())

    var vertices = listOf(
        Vertex(1.9f, 0f),
        Vertex(1f, 0f),
        Vertex(0f, 1f),
        Vertex(1f, 1f)
    )

    val viewport = Viewport(500, 500)

    var simulationIsPaused = false

    fun update(scene: Scene, dragEvents: List<DragEvent>): Scene {
        dragEvents.forEach { event ->
            when (event) {
                is VertexDragEvent.VertexDragBegin ->
                    simulationIsPaused = true
                is VertexDragEvent.VertexDrag ->
                    vertices = vertices.withVertexPos(event.vertex.id, event.pos.x, event.pos.y)
                is VertexDragEvent.VertexDragEnd ->
                    simulationIsPaused = false
            }
        }

        if (!simulationIsPaused) {
            val (_, dVertices) = primalAndReverseDerivative(vertices, system::energy)
            vertices = vertices.zip(dVertices) { v, dv -> v.gradientDescent(dv, 0.02f) }
        }

        return scene.withVertexPositions(vertices.map { vertex -> vec2(vertex.x.basePrimal().value, vertex.y.basePrimal().value) })
    }

    viewport.lookAt(vec2(0f, 0f), 5f)
    viewport.show(makeInitialScene(system, vertices), ::update)
}