/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.softbody

import org.diffkt.*

data class Vertex(val pos: Vector2, val vel: Vector2, val mass: Float = 1f) : Differentiable<Vertex> {
    constructor(x: Float, y: Float, mass: Float = 1f): this(Vector2(x, y), Vector2(0f, 0f), mass)

    override fun wrap(wrapper: Wrapper): Vertex {
        return Vertex(wrapper.wrap(pos), wrapper.wrap(vel), mass)
    }
}

internal fun reverseDerivative(vertices: List<Vertex>, f: (List<Vertex>) -> DScalar) = primalAndGradient(vertices, f).second
