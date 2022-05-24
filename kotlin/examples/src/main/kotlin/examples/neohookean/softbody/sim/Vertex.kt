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

class Vertex(val id: Int, val pos: Vector2, val vel: Vector2, val mass: Float, val isFree: Boolean) : Differentiable<Vertex> {
    override fun wrap(wrapper: Wrapper): Vertex {
        return if (isFree) Vertex(id, wrapper.wrap(pos), vel, mass, isFree) else this
    }

    fun free(): Vertex = if (isFree) this else Vertex(id, pos, vel, mass, true)

    fun fix(pos: Vector2): Vertex = Vertex(id, pos, vel, mass, false)

    fun preBackwardEulerOptim(h: Float): Vertex {
        return if (!isFree) this else Vertex(id, pos + vel * h, vel, mass, isFree)
    }

    fun postBackwardEulerOptim(optimizedVertex: Vertex, h: Float): Vertex {
        return if (!isFree) this else Vertex(id, optimizedVertex.pos, (optimizedVertex.pos - pos) / h, mass, isFree)
    }

    class Tangent(val pos: Vector2) {
        object VectorSpace : IVectorSpace<Tangent, DScalar> {
            override fun Tangent.plus(b: Tangent): Tangent {
                return Tangent(pos + b.pos)
            }

            override fun Tangent.times(b: DScalar): Tangent {
                return Tangent(pos * b)
            }

            override fun Tangent.unaryMinus(): Tangent {
                return Tangent(-pos)
            }
        }
    }

    object DifferentiableSpace : IDifferentiableSpace<Vertex, Tangent, DScalar> {
        override fun Vertex.plus(tangent: Tangent): Vertex {
            return Vertex(id, pos + tangent.pos, vel, mass, isFree)
        }

        override fun extractInputTangent(input: Vertex, extractTensorTangent: (DTensor) -> DTensor): Tangent {
            return if (input.isFree) {
                Tangent(
                    Vector2(
                        extractTensorTangent(input.pos.x) as DScalar,
                        extractTensorTangent(input.pos.y) as DScalar
                    )
                )
            } else {
                Tangent(Vector2(FloatScalar(0f), FloatScalar(0f)))
            }
        }

        override val tangentVectorSpace = Tangent.VectorSpace
    }
}