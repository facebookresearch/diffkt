/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.mesh

import examples.utils.visualization.meshvis2d.math.Vector2

class Vertex(val id: Int, val pos: Vector2, val renderRadius: Float, val hitTestRadius: Float) {
    val hitTestRadius2 get() = hitTestRadius * hitTestRadius

    fun withPos(pos: Vector2): Vertex = Vertex(id, pos, renderRadius, hitTestRadius)

    fun hitTest(p: Vector2): Boolean {
        val q = (pos - p).q()
        return q < hitTestRadius2
    }
}