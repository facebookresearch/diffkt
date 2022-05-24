/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.sim

import examples.neohookean.softbody.optim.VectorSpace as IVectorSpace
import examples.neohookean.softbody.optim.DifferentiableSpace as IDifferentiableSpace
import org.diffkt.*

class SystemState(val vertices: List<Vertex> = listOf()) : Wrappable<SystemState> {
    override fun wrap(wrapper: Wrapper): SystemState {
        return SystemState(vertices.map(wrapper::wrap))
    }

    fun fixVertex(id: Int, x: Float, y: Float): SystemState = fixVertex(id, Vector2(x, y))

    fun fixVertex(id: Int, pos: Vector2): SystemState {
        return SystemState(vertices.map { if (it.id == id) it.fix(pos) else it })
    }

    fun freeVertex(id: Int): SystemState {
        return SystemState(vertices.map { if (it.id == id) it.free() else it })
    }

    fun preBackwardEulerOptim(h: Float): SystemState {
        return SystemState(vertices.map { it.preBackwardEulerOptim(h) })
    }

    fun postBackwardEulerOptim(optimizedState: SystemState, h: Float): SystemState {
        return SystemState(
            vertices.zip(optimizedState.vertices) { vertex0, optimizedVertex ->
                vertex0.postBackwardEulerOptim(optimizedVertex, h)
            }
        )
    }

    fun tangentIsAlmostZero(tangent: Tangent, qEps: Float = 1e-6f): Boolean {
        val maxQ = tangent.vertices.map { vertexTangent ->
            val vertexPosTangent = vertexTangent.pos
            vertexPosTangent.q()
        }.reduce { a, b -> if (a > b) a else b }
        return maxQ < qEps
    }

    class Tangent(val vertices: List<Vertex.Tangent>)

    object DifferentiableSpace : IDifferentiableSpace<SystemState, Tangent, DScalar> {
        override fun SystemState.plus(tangent: Tangent): SystemState {
            return SystemState(
                vertices.zip(tangent.vertices) { vertex, vertexTangent ->
                    with (Vertex.DifferentiableSpace) { vertex + vertexTangent }
                }
            )
        }

        override val tangentVectorSpace = object : IVectorSpace<Tangent, DScalar> {
            override fun Tangent.plus(b: Tangent): Tangent {
                return Tangent(
                    vertices.zip(b.vertices) { vertex, vertexTangent ->
                        with (Vertex.Tangent.VectorSpace) { vertex + vertexTangent }
                    }
                )
            }

            override fun Tangent.times(b: DScalar): Tangent {
                return Tangent(
                    vertices.map { vertex ->
                        with (Vertex.Tangent.VectorSpace) { vertex * b }
                    }
                )
            }

            override fun Tangent.unaryMinus(): Tangent {
                return Tangent(
                    vertices.map { vertex ->
                        with (Vertex.Tangent.VectorSpace) { -vertex }
                    }
                )
            }
        }

        override fun extractInputTangent(input: SystemState, extractTensorTangent: (DTensor) -> DTensor): Tangent {
            return Tangent(input.vertices.map { vertex ->
                with (Vertex.DifferentiableSpace) { extractInputTangent(vertex, extractTensorTangent) }
            })
        }
    }

    class Builder(val vertices: MutableList<Vertex> = mutableListOf()) {
        fun addVertex(pos: Vector2, mass: Float = 1f) {
            val vertex = Vertex(vertices.size, pos, Vector2(0f, 0f), mass, true)
            vertices.add(vertex)
        }

        fun build(): SystemState {
            return SystemState(vertices)
        }
    }

    companion object {
        fun build(init: Builder.() -> Unit): SystemState {
            val builder = Builder()
            builder.init()
            return builder.build()
        }
    }
}