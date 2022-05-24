/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.math

class Matrix2x2(val m00: Float, val m01: Float, val m10: Float, val m11: Float) {
    fun q(): Float {
        // sum of squares of all elements
        return m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11
    }

    fun det(): Float {
        return m00 * m11 - m01 * m10
    }

    fun mm(b: Matrix2x2): Matrix2x2 {
        return Matrix2x2(
            this.m00 * b.m00 + this.m01 * b.m10, this.m00 * b.m01 + this.m01 * b.m11,
            this.m10 * b.m00 + this.m11 * b.m10, this.m10 * b.m01 + this.m11 * b.m11,
        )
    }

    fun apply(b: Vector2): Vector2 {
        return Vector2(
            this.m00 * b.x + this.m01 * b.y,
            this.m10 * b.x + this.m11 * b.y
        )
    }

    fun inv(): Matrix2x2 {
        val det = this.det()
        return Matrix2x2(
            m11 / det, -m01 / det,
            -m10 / det,  m00 / det
        )
    }

    companion object {
        fun identity(): Matrix2x2 {
            return Matrix2x2(
                1f, 0f,
                0f, 1f
            )
        }
    }
}