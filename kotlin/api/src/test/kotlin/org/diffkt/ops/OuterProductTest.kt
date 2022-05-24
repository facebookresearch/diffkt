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

class OuterProductTest : AnnotationSpec() {
    val t1 = FloatTensor(Shape(1, 2), floatArrayOf(1F, 2F))
    val t2 = FloatTensor(
        Shape(3, 4), floatArrayOf(
        3F, 4F, 5F, 6F,
        7F, 8F, 9F, 10F,
        11F, 12F, 13F, 14F))
    val d1F = tensorOf(
            3.0F, 0.0F, 4.0F, 0.0F, 5.0F, 0.0F, 6.0F, 0.0F,
            7.0F, 0.0F, 8.0F, 0.0F, 9.0F, 0.0F, 10.0F, 0.0F,
            11.0F, 0.0F, 12.0F, 0.0F, 13.0F, 0.0F, 14.0F, 0.0F,

            0.0F, 3.0F, 0.0F, 4.0F, 0.0F, 5.0F, 0.0F, 6.0F,
            0.0F, 7.0F, 0.0F, 8.0F, 0.0F, 9.0F, 0.0F, 10.0F,
            0.0F, 11.0F, 0.0F, 12.0F, 0.0F, 13.0F, 0.0F, 14.0F).reshape(Shape(1, 2, 3, 4, 1, 2))
    val d1R = d1F.leftTranspose(Shape(1, 2, 3, 4), Shape(1, 2))
    val d2F = tensorOf(
            1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,

            2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F).reshape(Shape(1, 2, 3, 4, 3, 4))
    val d2R = d2F.leftTranspose(Shape(1, 2, 3, 4), Shape(3, 4))

    fun checkResult(t1: DTensor, t2: DTensor): DTensor {
        val p = t1 outerProduct t2
        p shouldBeExactly FloatTensor(
            Shape(1, 2, 3, 4), floatArrayOf(
                3F, 4F, 5F, 6F,
                7F, 8F, 9F, 10F,
                11F, 12F, 13F, 14F,

                6F, 8F, 10F, 12F,
                14F, 16F, 18F, 20F,
                22F, 24F, 26F, 28F,
        ))
        return p
    }

    @Test
    fun OuterProductFloatFloat() {
        checkResult(t1, t2)
    }

    @Test
    fun OuterProductFloatForward() {
        forwardDerivative(t2) { t2 -> checkResult(t1, t2) } shouldBeExactly d2F
    }

    @Test
    fun OuterProductFloatReverse() {
        reverseDerivative(t2) { t2 -> checkResult(t1, t2) } shouldBeExactly d2R
    }

    @Test
    fun OuterProductForwardForward() {
        forwardDerivative(t1, t2) { t1, t2 -> checkResult(t1, t2) } shouldBeExactly Pair(d1F, d2F)
    }

    @Test
    fun OuterProductReverseReverse() {
        reverseDerivative(t1, t2) { t1, t2 -> checkResult(t1, t2) } shouldBeExactly Pair(d1R, d2R)
    }

    @Test
    fun OuterProductForwardFloat() {
        forwardDerivative(t1) { t1 -> checkResult(t1, t2) } shouldBeExactly d1F
    }

    @Test
    fun OuterProductReverseFloat() {
        reverseDerivative(t1) { t1 -> checkResult(t1, t2) } shouldBeExactly d1R
    }

    @Test
    fun OuterProduct01() {
        forwardDerivative(t1, t2) { x0, y0 ->
            reverseDerivative(x0, y0) { x1, y1 ->
                forwardDerivative(x1, y1) { x2, y2 ->
                    reverseDerivative(x2, y2) { x3, y3 ->
                        x3 outerProduct y3
                    }.first
                }.first
            }.first
        }.first
    }

    @Test
    fun OuterProduct02() {
        forwardDerivative(t1, t2) { x0, y0 ->
            reverseDerivative(x0, y0) { x1, y1 ->
                forwardDerivative(x1, y1) { x2, y2 ->
                    reverseDerivative(x2, y2) { _, y3 ->
                        t1 outerProduct y3
                    }.first
                }.first
            }.first
        }.first
    }

    @Test
    fun OuterProduct03() {
        forwardDerivative(t1, t2) { x0, y0 ->
            reverseDerivative(x0, y0) { x1, y1 ->
                forwardDerivative(x1, y1) { x2, y2 ->
                    reverseDerivative(x2, y2) { x3, _ ->
                        x3 outerProduct t2
                    }.first
                }.first
            }.first
        }.first
    }

    @Test
    fun OuterProduct04() {
        reverseDerivative(t1, t2) { x0, y0 ->
            forwardDerivative(x0, y0) { x1, y1 ->
                reverseDerivative(x1, y1) { x2, y2 ->
                    forwardDerivative(x2, y2) { x3, y3 ->
                        x3 outerProduct y3
                    }.first
                }.first
            }.first
        }.first
    }

    @Test
    fun OuterProduct05() {
        reverseDerivative(t1, t2) { x0, y0 ->
            forwardDerivative(x0, y0) { x1, y1 ->
                reverseDerivative(x1, y1) { x2, y2 ->
                    forwardDerivative(x2, y2) { _, y3 ->
                        t1 outerProduct y3
                    }.first
                }.first
            }.first
        }.first
    }

    @Test
    fun OuterProduct06() {
        reverseDerivative(t1, t2) { x0, y0 ->
            forwardDerivative(x0, y0) { x1, y1 ->
                reverseDerivative(x1, y1) { x2, y2 ->
                    forwardDerivative(x2, y2) { x3, _ ->
                        x3 outerProduct t2
                    }.first
                }.first
            }.first
        }.first
    }
}
