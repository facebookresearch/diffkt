/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.vector2

import org.diffkt.*

/**
 * Demonstrates taking the derivative of a custom differentiable class, [Vector2].
 *
 * The example uses gradient descent to "recover" a vector [target].
 */

data class Vector2(val x: DScalar, val y: DScalar): Differentiable<Vector2> {
    // Secondary constructor
    constructor(x: Float, y: Float): this(FloatScalar(x), FloatScalar(y))

    override fun wrap(wrapper: Wrapper): Vector2 {
        return Vector2(wrapper.wrap(x), wrapper.wrap(y))
    }

    // Vector2 Operations
    operator fun minus(that: Vector2) = Vector2(this.x - that.x, this.y - that.y)

    operator fun times(that: Float) = Vector2(this.x * that, this.y * that)
}

val target = Vector2(1f, 2f)

fun loss(vector: Vector2): DScalar {
    return (target.x - vector.x).pow(2) + (target.y - vector.y).pow(2)
}

fun main() {
    var vector = Vector2(-1f, -2f)
    val lr = 0.1f
    for (i in 0 until 100) {
        val grad = primalAndGradient(vector, ::loss).second
        vector -= grad * lr
        println(vector)
    }
    println("target: $target")
    println("optimized: $vector")
}
