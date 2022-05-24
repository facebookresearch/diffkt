/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.*

class TransposeTest : AnnotationSpec() {
    @Test
    fun transpose() {
        val t = FloatTensor(Shape(4, 3, 2), floats(24))
        val s = t.transpose()
        s shouldBeExactly FloatTensor(
            Shape(2, 3, 4), floatArrayOf(
                1f, 7f, 13f, 19f, 3f, 9f, 15f, 21f, 5f, 11f, 17f, 23f,
                2f, 8f, 14f, 20f, 4f, 10f, 16f, 22f, 6f, 12f, 18f, 24f
                ))
        t.transpose().transpose() shouldBeExactly t
    }

    @Test
    fun transposeDerivative() {
        val t = FloatTensor(Shape(2, 3), floats(6))
        fun f(t: DTensor): DTensor {
            val s = t.transpose()
            s shouldBeExactly FloatTensor(
                Shape(3, 2), floatArrayOf(
                    1F, 4F, 2F, 5F, 3F, 6F
            ))
            t.transpose().transpose() shouldBeExactly t
            return s
        }
        forwardDerivative(t, ::f) shouldBeExactly FloatTensor(
            Shape(3, 2, 2, 3), floatArrayOf(
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F
        ))

        reverseDerivative(t, ::f) shouldBeExactly FloatTensor(
            Shape(2, 3, 3, 2), floatArrayOf(
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F
        ))
    }

    @Test
    fun transposeAxes() {
        val t = FloatTensor(
            Shape(4, 3, 2), floatArrayOf(
                1f, 2f,
                3f, 4f,
                5f, 6f,

                7f, 8f,
                9f, 10f,
                11f, 12f,

                13f, 14f,
                15f, 16f,
                17f, 18f,

                19f, 20f,
                21f, 22f,
                23f, 24f))
        val s = t.transpose(intArrayOf(2, 0, 1))
        s shouldBeExactly FloatTensor(
            Shape(2, 4, 3), floatArrayOf(
                1f, 3f, 5f, 7f, 9f, 11f, 13f, 15f, 17f, 19f, 21f, 23f,
                2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f, 24f
        ))
    }

    // TODO https://github.com/facebookincubator/diffkt/issues/101: test transpose derivatives
}
