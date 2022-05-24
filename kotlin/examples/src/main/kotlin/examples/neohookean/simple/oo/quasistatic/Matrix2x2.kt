/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.quasistatic

import org.diffkt.*

class Matrix2x2(val m00: DScalar, val m01: DScalar, val m10: DScalar, val m11: DScalar) {
    constructor(m00: Float, m01: Float, m10: Float, m11: Float): this(
        FloatScalar(m00), FloatScalar(m01), FloatScalar(m10), FloatScalar(m11)
    )

    fun toTensor(): DTensor {
        return m00.stack(m01).stack(m10.stack(m11))
    }

    fun q(): DScalar {
        // sum of squares of all elements
        return m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11
    }

    fun det(): DScalar {
        return m00 * m11 - m01 * m10
    }

    fun mm(b: Matrix2x2): Matrix2x2 {
        return Matrix2x2(
            this.m00 * b.m00 + this.m01 * b.m10, this.m00 * b.m01 + this.m01 * b.m11,
            this.m10 * b.m00 + this.m11 * b.m10, this.m10 * b.m01 + this.m11 * b.m11,
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