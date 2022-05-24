package examples.neohookean.simple.oo.dynamic

import org.diffkt.*
import org.diffkt.times as diffktTimes
import org.diffkt.compareTo as diffktCompareTo

class SystemState(val vertices: List<Vertex>): Differentiable<SystemState> {
    override fun wrap(wrapper: Wrapper): SystemState {
        return SystemState(vertices.map { vertex -> wrapper.wrap(vertex) })
    }

    fun preBackwardEulerOptim(h: Float): SystemState {
        return SystemState(
            this.vertices.map { v -> Vertex(v.pos + v.vel * h, v.vel, v.mass) }
        )
    }

    fun postBackwardEulerOptim(s1: SystemState, h: Float): SystemState {
        return SystemState(
            this.vertices.zip(s1.vertices) { v0, v1 -> Vertex(v1.pos, (v1.pos - v0.pos) / h, v0.mass) }
        )
    }

    class VectorSpace(eps: Float) : GradientDescentWithLineSearch.VectorSpace<SystemState, DScalar> {
        val eps2 = eps * eps

        override operator fun SystemState.plus(b: SystemState): SystemState {
            return SystemState(
                vertices.zip(b.vertices) {
                        va, vb ->
                    Vertex(
                        va.pos + vb.pos,
                        va.vel,
                        va.mass
                    )
                }
            )
        }

        override operator fun SystemState.unaryMinus(): SystemState {
            return SystemState(vertices.map { v -> Vertex(-v.pos, v.vel, v.mass) })
        }

        override operator fun SystemState.times(b: DScalar): SystemState {
            return SystemState(vertices.map { v -> Vertex(v.pos * b, v.vel, v.mass) })
        }

        override fun SystemState.isAlmostZero(): Boolean {
            val q = vertices.map { v -> v.pos.q() }.reduce { a, b -> if (a > b) a else b }
            return q < FloatScalar(eps2)
        }

        override fun DScalar.compareTo(b: DScalar): Int = this.diffktCompareTo(b)

        override fun DScalar.times(b: DScalar): DScalar = this.diffktTimes(b)

        override val scalarOne: DScalar = FloatScalar.ONE
    }
}

fun primalAndReverseDerivative(state: SystemState, f: (SystemState) -> DScalar): Pair<DScalar, SystemState> {
    return primalAndReverseDerivative(
        state,
        f,
        extractDerivative = {
                state, output, extract ->
            SystemState(
                state.vertices.map {
                        vertex ->
                    Vertex(
                        Vector2(
                            extract(vertex.pos.x, output) as DScalar,
                            extract(vertex.pos.y, output) as DScalar
                        ),
                        Vector2(
                            extract(vertex.vel.x, output) as DScalar,
                            extract(vertex.vel.y, output) as DScalar
                        ),
                        extract(vertex.mass, output) as DScalar
                    )
                }
            )
        }
    )
}