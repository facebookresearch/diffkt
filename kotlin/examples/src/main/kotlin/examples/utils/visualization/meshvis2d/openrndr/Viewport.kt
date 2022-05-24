/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.openrndr

import examples.utils.visualization.meshvis2d.math.Vector2
import examples.utils.visualization.meshvis2d.viewport.*
import org.openrndr.MouseButton
import org.openrndr.MouseEvent
import org.openrndr.application
import org.openrndr.color.ColorRGBa

class Viewport(
    width: Int, height: Int,
    camera: Camera = Camera(),
    val scrollSensitivity: Float = 0.1f,
    val windowTitle: String = "window",
) : examples.utils.visualization.meshvis2d.viewport.Viewport(width, height, camera) {
    override fun show(
        initialScene: Scene,
        update: (Scene, List<DragEvent>) -> Scene,
    ) {
        val viewport = this
        val viewportWidth = this.width
        val viewportHeight = this.height

        var scene = initialScene
        var dragState: DragState = NoDragState
        val dragEvents: MutableList<DragEvent> = mutableListOf()

        application {
            configure {
                title = windowTitle
                width = viewportWidth
                height = viewportHeight
                windowResizable = false
            }

            program {
                mouse.buttonDown.listen { event: MouseEvent ->
                    if (event.button == MouseButton.LEFT) {
                        dragState.cursorDown(DragState.TransitionContext(viewport, scene), event.toVector2()).also {
                            dragState = it.state
                            dragEvents.addAll(it.events)
                        }
                    }
                }

                mouse.moved.listen { event: MouseEvent ->
                    dragState.cursorMove(DragState.TransitionContext(viewport, scene), event.toVector2()).also {
                        dragState = it.state
                        dragEvents.addAll(it.events)
                    }
                }

                mouse.buttonUp.listen { event: MouseEvent ->
                    if (event.button == MouseButton.LEFT) {
                        dragState.cursorUp(DragState.TransitionContext(viewport, scene), event.toVector2()).also {
                            dragState = it.state
                            dragEvents.addAll(it.events)
                        }
                    }
                }

                mouse.scrolled.listen { event: MouseEvent ->
                    val viewportPos = Vector2(event.position.x.toFloat(), event.position.y.toFloat())
                    val dScale = event.rotation.y.toFloat() * scrollSensitivity
                    zoom(viewportPos, dScale)
                }

                extend {
                    dragEvents.forEach { event ->
                        when (event) {
                            is LookAt -> lookAt(event.worldCenter)
                        }
                    }
                    scene = update(scene, dragEvents)
                    dragEvents.clear()
                    drawer.clear(ColorRGBa.WHITE)
                    scene.meshes.forEach { it.render(camera, drawer) }
                }
            }
        }
    }
}