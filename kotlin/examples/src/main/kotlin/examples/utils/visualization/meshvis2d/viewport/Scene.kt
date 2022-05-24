/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.viewport

import examples.utils.visualization.meshvis2d.math.Vector2
import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.mesh.Vertex

class Scene(val meshes: List<Mesh>) {
    fun hitTest(p: Vector2): Vertex? {
        meshes.forEach { mesh ->
            if (mesh.draggable) {
                val vertex = mesh.hitTest(p)
                if (vertex != null) return vertex
            }
        }
        return null
    }

    fun withMesh(meshId: Int, newMesh: Mesh): Scene {
        return Scene(
            meshes.mapIndexed { i, mesh ->
                if (i == meshId) newMesh else mesh
            }
        )
    }

    fun withVertexPositions(vertexPositions: List<Vector2>): Scene {
        require(meshes.size == 1) {
            "expected exactly 1 mesh in the scene, found ${meshes.size}"
        }
        return Scene(listOf(meshes[0].withVertexPositions(vertexPositions)))
    }

    class Builder(val meshes: MutableList<Mesh> = mutableListOf()) {
        fun addMesh(mesh: Mesh): Builder {
            meshes.add(mesh)
            return this
        }

        fun build(): Scene = Scene(meshes)
    }

    companion object {
        fun build(init: Builder.() -> Unit): Scene {
            val builder = Builder()
            builder.init()
            return builder.build()
        }
    }
}