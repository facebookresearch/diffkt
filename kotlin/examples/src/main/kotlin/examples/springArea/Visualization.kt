/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.springArea

import examples.springArea.SpringArea.edges
import org.diffkt.DTensor
import org.diffkt.FloatTensor
import org.diffkt.Shape
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.math.Vector2

internal object Visualization {
    private val frames: MutableList<DTensor> = mutableListOf()
    private var frameIndex = 0
    fun capture(x: DTensor) {
        frames.add(x)
    }
    private const val WINDOW_SIZE = 1000
    private const val CENTER = WINDOW_SIZE / 2 - 200.0
    private const val SCALING_FACTOR = 500.0
    private const val DEFAULT_RADIUS = 20.0

    /**
     * Scale coordinate to be approximately centered for visualization.
     *
     * This assumes that the vertices are somewhat centered around (0,0).
     */
    private fun DTensor.toVector(): Vector2 {
        require(this is FloatTensor)
        require(this.shape == Shape(2))
        return Vector2(
            (CENTER + this.at(0) * SCALING_FACTOR),
            (CENTER - this.at(1) * SCALING_FACTOR)
        )
    }

    /**
     * Animates 2D vertices and edges.
     */
    fun visualize() {
        application {
            configure {
                title = "Spring Area"
                width = WINDOW_SIZE
                height = WINDOW_SIZE
                windowResizable = true
            }

            program {
                extend {
                    // Update xTensor before displaying.
                    val vertices = frames[frameIndex] as FloatTensor
                    if (this.frameCount % 5 == 0 && frameIndex != frames.size - 1) frameIndex++

                    // Set up drawer.
                    drawer.clear(ColorRGBa.WHITE)
                    drawer.stroke = null

                    // Draw vertices.
                    drawer.fill = ColorRGBa.PINK
                    drawer.stroke = ColorRGBa.BLACK
                    val vertexCoords = (0 until 3).map { i -> vertices[i].toVector()}
                    vertexCoords.forEach { vec ->
                        drawer.circle(vec, DEFAULT_RADIUS)
                    }

                    // Draw edges.
                    drawer.stroke = ColorRGBa.BLACK
                    edges.forEach { edge ->
                        drawer.lineSegment(
                            vertexCoords[edge.first],
                            vertexCoords[edge.second]
                        )
                    }
                }
            }
        }
    }
}