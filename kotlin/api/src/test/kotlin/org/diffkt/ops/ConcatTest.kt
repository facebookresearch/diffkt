/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import testutils.*

class ConcatTest : AnnotationSpec() {

    @Test fun axis0() {
        val xShape = Shape(2, 3)
        val yShape = Shape(5, 3)
        val x = FloatTensor(xShape) { it.toFloat() }
        val y = FloatTensor(yShape) { it.toFloat() }
        fun f(x: DTensor, y: DTensor): DTensor = x.concat(y, axis = 0)

        val outShape = Shape(7, 3)
        checkForwardDerivatives(x, y, ::f, outShape)
        checkReverseDerivatives(x, y, ::f, outShape)
        checkMixedDerivatives(x, y, ::f, outShape)
    }

    @Test fun axis2() {
        val xShape = Shape(2, 3, 5, 2)
        val yShape = Shape(2, 3, 1, 2)
        val x = FloatTensor(xShape) { it.toFloat() }
        val y = FloatTensor(yShape) { it.toFloat() }
        fun f(x: DTensor, y: DTensor): DTensor = x.concat(y, axis = 2)

        val outShape = Shape(2, 3, 6, 2)
        checkForwardDerivatives(x, y, ::f, outShape)
        checkReverseDerivatives(x, y, ::f, outShape)
        checkMixedDerivatives(x, y, ::f, outShape)
    }

    private fun checkForwardDerivatives(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor, outShape: Shape) {
        val der0 = f(x, y)
        val (dx1, dy1) = forwardDerivative(x, y, f)
        val (dx2, dx1y1) = forwardDerivative(x, y) { xx, yy -> forwardDerivative(xx, yy, f).first }
        val (dy1x1, dy2) = forwardDerivative(x, y) { xx, yy -> forwardDerivative(xx, yy, f).second }

        val xShape = x.shape
        val yShape = y.shape

        der0.shape shouldBe outShape
        dx1.shape shouldBe outShape + xShape
        dy1.shape shouldBe outShape + yShape
        dx2.shape shouldBe outShape + xShape + xShape
        dx1y1.shape shouldBe outShape + xShape + yShape
        dy2.shape shouldBe outShape + yShape + yShape
        dy1x1.shape shouldBe outShape + yShape + xShape
    }

    private fun checkReverseDerivatives(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor, outShape: Shape) {
        val der0 = f(x, y)
        val (dx1, dy1) = reverseDerivative(x, y, f)
        val (dx2, dx1y1) = reverseDerivative(x, y) { xx, yy -> reverseDerivative(xx, yy, f).first }
        val (dy1x1, dy2) = reverseDerivative(x, y) { xx, yy -> reverseDerivative(xx, yy, f).second }

        val xShape = x.shape
        val yShape = y.shape

        der0.shape shouldBe outShape
        dx1.shape shouldBe xShape + outShape
        dy1.shape shouldBe yShape + outShape
        dx2.shape shouldBe xShape + xShape + outShape
        dx1y1.shape shouldBe yShape + xShape + outShape
        dy2.shape shouldBe yShape + yShape + outShape
        dy1x1.shape shouldBe xShape + yShape + outShape
    }

    private fun checkMixedDerivatives(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor, outShape: Shape) {
        val xShape = x.shape
        val yShape = y.shape

        run {
            // Reverse, then forward
            val (dx2, dx1y1) = forwardDerivative(x, y) { xx, yy -> reverseDerivative(xx, yy, f).first }
            val (dy1x1, dy2) = forwardDerivative(x, y) { xx, yy -> reverseDerivative(xx, yy, f).second }

            dx2.shape shouldBe xShape + outShape + xShape
            dx1y1.shape shouldBe xShape + outShape + yShape
            dy2.shape shouldBe yShape + outShape + yShape
            dy1x1.shape shouldBe yShape + outShape + xShape
        }

        run {
            // Forward, then reverse
            val (dx2, dx1y1) = reverseDerivative(x, y) { xx, yy -> forwardDerivative(xx, yy, f).first }
            val (dy1x1, dy2) = reverseDerivative(x, y) { xx, yy -> forwardDerivative(xx, yy, f).second }

            dx2.shape shouldBe xShape + outShape + xShape
            dx1y1.shape shouldBe yShape + outShape + xShape
            dy2.shape shouldBe yShape + outShape + yShape
            dy1x1.shape shouldBe xShape + outShape + yShape
        }
    }

    @Test fun emptyList() {
        val e = shouldThrow<IllegalArgumentException> { concat(listOf<DTensor>(), 0) }
        e.message shouldBe "Cannot concat empty list of tensors"
    }

    @Test fun axis0Lists() {
        val shapes = listOf(Shape(2,3), Shape(5,3), Shape(3,3))
        val tensors: List<DTensor> = shapes.map { shape -> FloatTensor(shape) { it.toFloat() } }
        concat(tensors, 0) shouldBeExactly tensorOf(
                0f, 1f, 2f,
                3f, 4f, 5f,
                0f, 1f, 2f,
                3f, 4f, 5f,
                6f, 7f, 8f,
                9f, 10f, 11f,
                12f, 13f, 14f,
                0f, 1f, 2f,
                3f, 4f, 5f,
                6f, 7f, 8f
        ).reshape(Shape(10, 3))
    }

    @Test fun axis0ListsReverse() {
        val t1 = FloatTensor(Shape(2,3)) { it.toFloat() }
        val t2 = FloatTensor(Shape(4,3)) { it.toFloat() }
        val t3 = FloatTensor(Shape(1,3)) { it.toFloat() }

        val (dt1, dt2) = reverseDerivative(t1, t2) { x, y -> concat(listOf(x, t3, y), 0) }
        val (dt1Ref, dt2Ref) = reverseDerivative(t1, t2) { x, y -> x.concat(t3, 0).concat(y, 0)}

        dt1 shouldBeExactly dt1Ref
        dt1 shouldBeExactly tensorOf(
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F
        ).reshape(Shape(2, 3, 7, 3))

        dt2 shouldBeExactly dt2Ref
        dt2 shouldBeExactly tensorOf(
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F
        ).reshape(Shape(4, 3, 7, 3))
    }

}
