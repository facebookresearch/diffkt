/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.math

import kotlin.math.sqrt

data class Vector2(val x: Float, val y: Float) {
    operator fun plus(b: Vector2): Vector2 = Vector2(x + b.x, y + b.y)

    operator fun minus(b: Vector2): Vector2 = Vector2(x - b.x, y - b.y)

    operator fun times(b: Float): Vector2 = Vector2(x * b, y * b)

    operator fun unaryMinus(): Vector2 = Vector2(-x, -y)

    fun q(): Float = x * x + y * y

    fun norm(): Float = sqrt(q())
}