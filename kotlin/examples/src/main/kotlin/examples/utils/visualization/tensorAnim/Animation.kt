/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization

import examples.hookeanSpring.softbody.Edge
import examples.hookeanSpring.softbody.Vertex
import org.diffkt.*
import org.openrndr.MouseButton
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.math.Vector2

private const val WINDOW_SIZE = 1000
private const val CENTER = WINDOW_SIZE / 2
private const val SCALING_FACTOR = 150.0
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

private fun Vector2.toTensor(): DTensor {
    return tensorOf(
        ((this.x - CENTER) / SCALING_FACTOR).toFloat(),
        ((- this.y + CENTER) / SCALING_FACTOR).toFloat()
    )
}

/**
 * Animates 2D vertices and edges.
 */
fun animate(
    name: String,
    init: FloatTensor, update: (DTensor) -> DTensor,
    edges: List<Pair<Int, Int>>? = null,
    radii: List<Double>? = null,
    interval: Int
) {
    require(init.shape[1] == 2) { "Animation is for 2D vertices only." }
    var vertices = init

    // Get the ith vertex position to draw.
    fun getVertexCoordinates(i: Int): Vector2 {
        return vertices[i].toVector()
    }

    application {
        configure {
            title = name
            width = WINDOW_SIZE
            height = WINDOW_SIZE
            windowResizable = true
        }

        program {
            var downPos: Vector2? = null
            mouse.buttonDown.listen { if (it.button == MouseButton.LEFT) downPos = it.position }

            var dragPos: Vector2? = null
            mouse.dragged.listen { if (it.button == MouseButton.LEFT) dragPos = it.position }

            var releasePos: Vector2? = null
            mouse.buttonUp.listen { if (it.button == MouseButton.LEFT) releasePos = it.position }

            var selectedVertexId: Int? = null

            extend {
                // Update xTensor before displaying.
                repeat(interval) { vertices = update(vertices) as FloatTensor }

                // If a vertex is moved, update vertices tensor to releasePos.
                releasePos?.let { newPos ->
                    selectedVertexId?.let { id ->
                        vertices = vertices.withChange(id, 0, newPos.toTensor()) as FloatTensor
                    }
                    selectedVertexId = null
                    dragPos = null
                    releasePos = null
                }

                val numVertices = vertices.shape[0]
                val vertexCoords = (0 until numVertices).map { i -> getVertexCoordinates(i) }.toMutableList()

                // If a vertex is clicked on, update selectedVertexId.
                downPos?.let { clicked ->
                    selectedVertexId = vertexCoords.mapIndexedNotNull { id, vec ->
                        val distance = vec.distanceTo(clicked)
                        if (distance > 20) null else id to distance
                    }.minByOrNull { (_, distance) -> distance }?.first

                    downPos = null
                }

                // If a vertex is dragged, update selectedVertexId's coordinates to dragPos.
                dragPos?.let {
                    selectedVertexId?.let{ id ->
                        vertexCoords[id] = dragPos!!
                    }
                }

                // Set up drawer.
                drawer.clear(ColorRGBa.WHITE)
                drawer.stroke = null

                // Draw vertices.
                drawer.fill = ColorRGBa.PINK
                drawer.stroke = ColorRGBa.BLACK
                vertexCoords.forEachIndexed { i, vec ->
                    drawer.circle(vec, radii?.get(i)?.times(SCALING_FACTOR) ?: DEFAULT_RADIUS)
                }

                // Draw edges.
                drawer.stroke = ColorRGBa.BLACK
                edges?.forEach { edge ->
                    drawer.lineSegment(
                        vertexCoords[edge.first],
                        vertexCoords[edge.second]
                    )
                }
            }
        }
    }
}

private fun Vertex.toVector(): Vector2 {
    val vX = (this.pos.x as FloatScalar).value.toDouble()
    val vY = (this.pos.y as FloatScalar).value.toDouble()

    return Vector2(
        (CENTER + vX * SCALING_FACTOR),
        (CENTER - vY * SCALING_FACTOR)
    )
}

private fun Vector2.toPoint(): Vertex {
    return Vertex(
        ((this.x - CENTER) / SCALING_FACTOR).toFloat(),
        ((- this.y + CENTER) / SCALING_FACTOR).toFloat()
    )
}

/**
 * Animates 2D vertices and edges. Expects List<Point> and List<Edge>.
 */
fun animate(
    name: String,
    init: List<Vertex>, update: () -> List<Vertex>,
    edges: List<Edge>? = null,
    radii: List<Double>? = null,
    interval: Int
) {
    var vertices = init

    // Get the ith vertex position to draw.
    fun getVertexCoordinates(i: Int): Vector2 {
        return vertices[i].toVector()
    }

    application {
        configure {
            title = name
            width = WINDOW_SIZE
            height = WINDOW_SIZE
            windowResizable = true
        }

        program {
            var downPos: Vector2? = null
            mouse.buttonDown.listen { if (it.button == MouseButton.LEFT) downPos = it.position }

            var dragPos: Vector2? = null
            mouse.dragged.listen { if (it.button == MouseButton.LEFT) dragPos = it.position }

            var releasePos: Vector2? = null
            mouse.buttonUp.listen { if (it.button == MouseButton.LEFT) releasePos = it.position }

            var selectedVertexId: Int? = null

            extend {
                // Update xTensor before displaying.
                repeat(interval) { vertices = update() }

                // If a vertex is moved, update vertices tensor to releasePos.
                releasePos?.let { newPos ->
                    selectedVertexId?.let { id ->
                        val mutableVertices = vertices.toMutableList()
                        mutableVertices[id] = newPos.toPoint()
                        vertices = mutableVertices
                    }
                    selectedVertexId = null
                    dragPos = null
                    releasePos = null
                }

                val numVertices = vertices.size
                val vertexCoords = (0 until numVertices).map { i -> getVertexCoordinates(i) }.toMutableList()

                // If a vertex is clicked on, update selectedVertexId.
                downPos?.let { clicked ->
                    selectedVertexId = vertexCoords.mapIndexedNotNull { id, vec ->
                        val distance = vec.distanceTo(clicked)
                        if (distance > 20) null else id to distance
                    }.minByOrNull { (_, distance) -> distance }?.first

                    downPos = null
                }

                // If a vertex is dragged, update selectedVertexId's coordinates to dragPos.
                dragPos?.let {
                    selectedVertexId?.let{ id ->
                        vertexCoords[id] = dragPos!!
                    }
                }

                // Set up drawer.
                drawer.clear(ColorRGBa.WHITE)
                drawer.stroke = null

                // Draw vertices.
                drawer.fill = ColorRGBa.PINK
                drawer.stroke = ColorRGBa.BLACK
                vertexCoords.forEachIndexed { i, vec ->
                    drawer.circle(vec, radii?.get(i)?.times(SCALING_FACTOR) ?: DEFAULT_RADIUS)
                }

                // Draw edges.
                drawer.stroke = ColorRGBa.BLACK
                edges?.forEach { edge ->
                    drawer.lineSegment(
                        vertexCoords[edge.left],
                        vertexCoords[edge.right]
                    )
                }
            }
        }
    }
}
