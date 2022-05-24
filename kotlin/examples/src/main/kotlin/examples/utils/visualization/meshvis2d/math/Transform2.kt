/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.math

class Transform2(
    val translation: Vector2 = Vector2(0f, 0f),
    val linear: Matrix2x2 = Matrix2x2.identity()
) {
    fun apply(v: Vector2): Vector2 {
        return linear.apply(v) + translation
    }

    val inv: Transform2 by lazy {
        val linearInv = linear.inv()
        Transform2(-linearInv.apply(translation), linearInv)
    }

    companion object {
        fun identity() = Transform2(
            Vector2(0f, 0f),
            Matrix2x2.identity()
        )
    }
}