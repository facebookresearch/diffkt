/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.mesh

import examples.utils.visualization.meshvis2d.math.Vector2

class Mesh(
    val vertices: List<Vertex>,
    val lines: List<Line>,
    val triangles: List<Triangle>,
    val draggable: Boolean
) {
    fun withVertexPositions(positions: List<Vector2>): Mesh {
        return Mesh(
            vertices.zip(positions) { vertex, pos ->
                vertex.withPos(pos)
            },
            lines,
            triangles,
            draggable
        )
    }

    fun hitTest(p: Vector2): Vertex? {
        vertices.forEach { if (it.hitTest(p)) return it }
        return null
    }

    class Builder(
        val vertices: MutableList<Vertex> = mutableListOf(),
        val lines: MutableList<Line> = mutableListOf(),
        val triangles: MutableList<Triangle> = mutableListOf(),
        var draggable: Boolean = false
    ) {
        fun addVertex(pos: Pair<Float, Float>, radius: Float): Builder {
            return addVertex(Vector2(pos.first, pos.second), radius)
        }

        fun addVertex(pos: Vector2, renderRadius: Float, hitTestRadius: Float): Builder {
            val vertex = Vertex(vertices.size, pos, renderRadius, hitTestRadius)
            vertices.add(vertex)
            return this
        }

        fun addVertex(pos: Vector2, radius: Float = 0.5f): Builder = addVertex(pos, radius, radius)

        fun addTriangle(i1: Int, i2: Int, i3: Int): Builder {
            val triangle = Triangle(i1, i2, i3)
            triangles.add(triangle)
            return this
        }

        fun addLine(a: Int, b: Int): Builder {
            val line = Line(a, b)
            lines.add(line)
            return this
        }

        fun makeDraggable(): Builder {
            this.draggable = true
            return this
        }

        fun build() = Mesh(vertices, lines, triangles, draggable)
    }

    companion object {
        fun build(init: Builder.() -> Unit): Mesh {
            val builder = Builder()
            builder.init()
            return builder.build()
        }
    }
}