/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.viewport

import examples.utils.visualization.meshvis2d.math.Vector2
import examples.utils.visualization.meshvis2d.mesh.Vertex

interface DragEvent
data class LookAt(val worldCenter: Vector2) : DragEvent
interface VertexDragEvent : DragEvent {
    data class VertexDragBegin(val vertex: Vertex) : VertexDragEvent
    data class VertexDrag(val vertex: Vertex, val pos: Vector2) : VertexDragEvent
    data class VertexDragEnd(val vertex: Vertex) : VertexDragEvent
}

sealed interface DragState {
    fun cursorDown(ctx: TransitionContext, viewportPos: Vector2): TransitionResult
    fun cursorUp(ctx: TransitionContext, viewportPos: Vector2): TransitionResult
    fun cursorMove(ctx: TransitionContext, viewportPos: Vector2): TransitionResult

    data class TransitionContext(val viewport: Viewport, val scene: Scene)

    data class TransitionResult(val state: DragState, val events: List<DragEvent> = listOf()) {
        constructor(state: DragState, event: DragEvent): this(state, listOf(event))
    }
}

class ActiveDragInfo(
    val beginWorldCenter: Vector2,
    val beginWorldPos: Vector2,
    val beginViewportPos: Vector2
)

class BackgroundDragState(val activeDragInfo: ActiveDragInfo) : DragState {
    override fun cursorDown(ctx: DragState.TransitionContext, viewportPos: Vector2) = DragState.TransitionResult(this)

    override fun cursorUp(ctx: DragState.TransitionContext, viewportPos: Vector2) = DragState.TransitionResult(NoDragState)

    override fun cursorMove(ctx: DragState.TransitionContext, viewportPos: Vector2): DragState.TransitionResult {
        val worldToViewportScale = ctx.viewport.camera.inferWorldToViewportScale()
        val dViewport = viewportPos - activeDragInfo.beginViewportPos
        val dWorld = Vector2(
            -dViewport.x * (1f / worldToViewportScale),
            dViewport.y * (1f / worldToViewportScale)
        )
        val worldCenter = activeDragInfo.beginWorldCenter + dWorld
        return DragState.TransitionResult(this, LookAt(worldCenter))
    }
}

class VertexDragState(
    val activeDragInfo: ActiveDragInfo,
    val hitVertex: Vertex,
    val hitVertexPos: Vector2
): DragState {
    override fun cursorDown(ctx: DragState.TransitionContext, viewportPos: Vector2) =
        DragState.TransitionResult(this)

    override fun cursorUp(ctx: DragState.TransitionContext, viewportPos: Vector2) =
        DragState.TransitionResult(NoDragState, VertexDragEvent.VertexDragEnd(hitVertex))

    override fun cursorMove(ctx: DragState.TransitionContext, viewportPos: Vector2): DragState.TransitionResult {
        val camera = ctx.viewport.camera
        val worldPos = camera.viewportToWorld(viewportPos)
        val dWorld = worldPos - activeDragInfo.beginWorldPos
        return DragState.TransitionResult(
            this,
            // TODO should we use hitVertex or hitVertex.id ?
            VertexDragEvent.VertexDrag(hitVertex, hitVertexPos + dWorld)
        )
    }
}

object NoDragState : DragState {
    override fun cursorDown(ctx: DragState.TransitionContext, viewportPos: Vector2): DragState.TransitionResult {
        val cursorWorldPos = ctx.viewport.camera.viewportToWorld(viewportPos)
        val activeDragInfo = ActiveDragInfo(ctx.viewport.worldCenter, cursorWorldPos, viewportPos)
        val vertex: Vertex? = ctx.scene.hitTest(cursorWorldPos)
        return if (vertex != null) {
            DragState.TransitionResult(
                VertexDragState(activeDragInfo, vertex, cursorWorldPos),
                VertexDragEvent.VertexDragBegin(vertex)
            )
        } else {
            DragState.TransitionResult(BackgroundDragState(activeDragInfo))
        }
    }

    override fun cursorUp(ctx: DragState.TransitionContext, viewportPos: Vector2): DragState.TransitionResult =
        DragState.TransitionResult(this)

    override fun cursorMove(ctx: DragState.TransitionContext, viewportPos: Vector2): DragState.TransitionResult =
        DragState.TransitionResult(this)
}