/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody

import examples.neohookean.softbody.optim.*
import examples.neohookean.softbody.sim.*
import examples.utils.visualization.meshvis2d.math.Vector2 as vec2
import examples.utils.visualization.meshvis2d.openrndr.Viewport
import examples.utils.visualization.meshvis2d.viewport.DragEvent
import examples.utils.visualization.meshvis2d.viewport.VertexDragEvent
import examples.utils.visualization.meshvis2d.viewport.Scene
import org.diffkt.*

fun main() {
    // line colliders have infinite length, but we choose an arbitrary length just for visualization purposes
    val renderLineColliderLength = 15f
    val (system, initialSystemState) = makeSystem(renderLineColliderLength = renderLineColliderLength)

    var systemState = initialSystemState
    val h = 0.033f

    var optimizer = GradientDescentWithLineSearch(SystemState.DifferentiableSpace, maxIters = 20, maxLineSearchIters = 50)

    val width = 500
    val height = 500
    val viewport = Viewport(width, height, windowTitle = "sim")

    val (initialScene, deformableMeshId) = makeScene(system, systemState, renderLineColliderLength)

    fun update(scene: Scene, dragEvents: List<DragEvent>): Scene {
        dragEvents.forEach { event ->
            systemState = when (event) {
                is VertexDragEvent.VertexDragBegin -> systemState.fixVertex(event.vertex.id, event.vertex.pos.x, event.vertex.pos.y)
                is VertexDragEvent.VertexDrag -> systemState.fixVertex(event.vertex.id, event.pos.x, event.pos.y)
                is VertexDragEvent.VertexDragEnd -> systemState.freeVertex(event.vertex.id)
                else -> systemState
            }
        }

        optimizer.optimize(
            systemState.preBackwardEulerOptim(h),
            system.makeBackwardEulerLoss(systemState, h), systemState::tangentIsAlmostZero
        ).also {
            systemState = systemState.postBackwardEulerOptim(it.first, h)
            optimizer = it.second
        }

        val vertexPositions = systemState.vertices.map { vec2(it.pos.x.basePrimal().value, it.pos.y.basePrimal().value) }
        return scene.withMesh(deformableMeshId, scene.meshes[deformableMeshId].withVertexPositions(vertexPositions))
    }

    viewport.lookAt(vec2(0f, 7.0f), 18f)
    viewport.show(initialScene, ::update)
}