/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.dynamic

import org.diffkt.*

class Vertex(val pos: Vector2, val vel: Vector2, val mass: DScalar): Differentiable<Vertex> {
    constructor(x: Float, y: Float): this(Vector2(x, y), Vector2(0f, 0f), FloatScalar(1f)) {
    }

    override fun wrap(wrapper: Wrapper): Vertex {
        return Vertex(wrapper.wrap(pos), wrapper.wrap(vel), wrapper.wrap(mass))
    }
}