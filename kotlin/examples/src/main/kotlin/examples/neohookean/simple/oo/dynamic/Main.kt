/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.dynamic

import org.diffkt.*
import examples.utils.visualization.meshvis2d.math.Vector2 as vec2
import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.openrndr.Viewport
import examples.utils.visualization.meshvis2d.viewport.DragEvent
import examples.utils.visualization.meshvis2d.viewport.Scene
import examples.utils.visualization.meshvis2d.viewport.VertexDragEvent

fun List<Vertex>.withVertexPos(i: Int, x: Float, y: Float) =
    this.mapIndexed { j, vertexJ -> if (i == j) Vertex(x, y) else vertexJ }

fun makeInitialScene(system: System, systemState: SystemState): Scene {
    return Scene.build {
        addMesh(Mesh.build {
            makeDraggable()
            systemState.vertices.forEach { vertex ->
                addVertex(vec2(vertex.pos.x.basePrimal().value, vertex.pos.y.basePrimal().value), 0.1f, 0.3f)
            }
            system.triangles.forEach { triangle ->
                addTriangle(triangle.i1, triangle.i2, triangle.i3)
            }
        })
    }
}

fun main() {
    val system = System()
        .addTriangle(0, 1, 2, Matrix2x2.identity(), mu = 100f, lambda = 10f)
        .addTriangle(3, 2, 1, Matrix2x2.identity(), mu = 100f, lambda = 10f)

    var vertices = listOf(
        Vertex(1.9f, 0f),
        Vertex(1f, 0f),
        Vertex(0f, 1f),
        Vertex(1f, 1f)
    )

    var state = SystemState(vertices)

    val optimizer = GradientDescentWithLineSearch(
        SystemState.VectorSpace(1e-3f),
        100, 100, FloatScalar(0.3f)
    )
    val h = 0.033f

    val viewport = Viewport(500, 500)

    var simulationIsPaused = false

    fun update(scene: Scene, dragEvents: List<DragEvent>): Scene {
        dragEvents.forEach { event ->
            when (event) {
                is VertexDragEvent.VertexDragBegin ->
                    simulationIsPaused = true
                is VertexDragEvent.VertexDrag ->
                    state = SystemState(state.vertices.withVertexPos(event.vertex.id, event.pos.x, event.pos.y))
                is VertexDragEvent.VertexDragEnd ->
                    simulationIsPaused = false
            }
        }

        if (!simulationIsPaused) {
            val s0 = state
            val loss = system.makeBackwardEulerLoss(s0, h)
            val s1 = optimizer.optimize(
                s0.preBackwardEulerOptim(h),
                loss,
                ::primalAndReverseDerivative
            )
            state = s0.postBackwardEulerOptim(s1, h)
        }

        return scene.withVertexPositions(state.vertices.map { vec2(it.pos.x.basePrimal().value, it.pos.y.basePrimal().value) })
    }

    viewport.lookAt(vec2(0f, 0f), 5f)
    viewport.show(makeInitialScene(system, state), ::update)
}