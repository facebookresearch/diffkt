/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody

import examples.neohookean.softbody.sim.Matrix2x2
import examples.neohookean.softbody.sim.System
import examples.neohookean.softbody.sim.SystemState
import examples.neohookean.softbody.sim.Vector2
import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.viewport.Scene
import org.diffkt.basePrimal

fun makeSystem(s: Int = 6, x0: Float = 0.0f, y0: Float = 0.5f, sideSize: Float = 0.4f, renderLineColliderLength: Float = 1f): Pair<System, SystemState> {
    val system = System.build {
        (0 until s - 1).forEach { col ->
            (0 until s - 1).forEach { row ->
                val offset = s * row
                addTriangle(
                    offset + col, offset + col + 1, offset + col + s,
                    mu = 100f, lambda = 50f,
                    restShapeInv = Matrix2x2.identity()
                )
                addTriangle(
                    offset + col + s + 1, offset + col + s, offset + col + 1,
                    mu = 100f, lambda = 50f,
                    restShapeInv = Matrix2x2.identity()
                )
            }
        }

        addLineCollider(
            Vector2(0f, 0f), Vector2(0f, 1f).normalize(),
            frictionK = 100f, frictionEps = 1e-2f, collisionK = 1300f
        )
        addLineCollider(
            Vector2(renderLineColliderLength * 0.5f, renderLineColliderLength * 0.5f), Vector2(-1f, 0f).normalize(),
            frictionK = 100f, frictionEps = 1e-2f, collisionK = 1300f
        )
        addLineCollider(
            Vector2(-renderLineColliderLength * 0.5f, renderLineColliderLength * 0.5f), Vector2(1f, 0f).normalize(),
            frictionK = 100f, frictionEps = 1e-2f, collisionK = 1300f
        )
    }

    val systemState = SystemState.build {
        (0 until s * s).forEach { id ->
            addVertex(Vector2(id % s * sideSize + x0, id / s * sideSize + sideSize + y0), 0.3f)
        }
    }

    return Pair(system, systemState)
}

fun makeScene(system: System, systemState: SystemState, renderLineColliderLength: Float): Pair<Scene, Int> {
    val scene = Scene.build {
        system.lineColliders.forEach {
            addMesh(Mesh.build {
                addVertex((it.pos + it.tangent * renderLineColliderLength * 0.5f).toFloatPair(), 0.1f)
                addVertex((it.pos - it.tangent * renderLineColliderLength * 0.5f).toFloatPair(), 0.1f)
                addLine(0, 1)
            })
        }

        addMesh(Mesh.build {
            makeDraggable()

            systemState.vertices.forEach { vertex ->
                addVertex(examples.utils.visualization.meshvis2d.math.Vector2(vertex.pos.x.basePrimal().value,
                    vertex.pos.y.basePrimal().value), 0.2f, 0.7f)
            }

            system.triangles.forEach {
                addTriangle(it.i1, it.i2, it.i3)
            }
        })
    }

    val deformableMeshId = scene.meshes.size - 1
    return Pair(scene, deformableMeshId)
}
