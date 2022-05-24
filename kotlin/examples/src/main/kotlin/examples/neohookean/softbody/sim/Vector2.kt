/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.sim

import org.diffkt.*

data class Vector2(val x: DScalar, val y: DScalar) : Differentiable<Vector2> {
    constructor(x: Float, y: Float): this(FloatScalar(x), FloatScalar(y))

    override fun wrap(wrapper: Wrapper): Vector2 {
        return Vector2(wrapper.wrap(x), wrapper.wrap(y))
    }

    operator fun unaryMinus(): Vector2 {
        return Vector2(-x, -y)
    }

    operator fun plus(b: Vector2): Vector2 {
        return Vector2(x + b.x, y + b.y)
    }

    operator fun minus(b: Vector2): Vector2 {
        return Vector2(x - b.x, y - b.y)
    }

    operator fun times(c: Float): Vector2 {
        return Vector2(c * x, c * y)
    }

    operator fun times(c: DScalar): Vector2 {
        return Vector2(c * x, c * y)
    }

    operator fun div(c: Float): Vector2 {
        return Vector2(x / c, y / c)
    }

    operator fun div(c: DScalar): Vector2 {
        return Vector2(x / c, y / c)
    }

    fun dot(b: Vector2): DScalar {
        return this.x * b.x + this.y * b.y
    }

    fun q(): DScalar {
        return x * x + y * y
    }

    fun norm(): DScalar {
        return sqrt(q())
    }

    fun normalize(): Vector2 {
        // TODO what if norm == 0?
        return this / norm()
    }

    fun toFloatPair(): Pair<Float, Float> {
        return Pair(x.basePrimal().value, y.basePrimal().value)
    }
}