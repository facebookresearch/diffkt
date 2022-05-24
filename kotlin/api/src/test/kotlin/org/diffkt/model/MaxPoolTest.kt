/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.diffkt.tensor.op

import org.diffkt.*
import org.diffkt.model.*
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.data.forAll
import io.kotest.data.headers
import io.kotest.data.row
import io.kotest.data.table
import io.kotest.matchers.string.shouldContain
import testutils.*

object MaxPoolTestValues {
    object MaxPool {
        val tensor get() =
            FloatTensor(Shape(2, 6, 6, 3), floats(108) + floats(108))
        val expectedOut = FloatTensor(
            Shape(2, 3, 3, 3),
            floatArrayOf(
                22f, 23f, 24f, 28f, 29f, 30f, 34f, 35f, 36f,
                58f, 59f, 60f, 64f, 65f, 66f, 70f, 71f, 72f,
                94f, 95f, 96f, 100f, 101f, 102f, 106f, 107f, 108f,

                22f, 23f, 24f, 28f, 29f, 30f, 34f, 35f, 36f,
                58f, 59f, 60f, 64f, 65f, 66f, 70f, 71f, 72f,
                94f, 95f, 96f, 100f, 101f, 102f, 106f, 107f, 108f
            )
        )
        val expectedGrad = FloatTensor(
            Shape(2, 6, 6, 3),
            floatArrayOf(
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 22f, 23f, 24f, 0f, 0f, 0f, 28f, 29f, 30f, 0f, 0f, 0f, 34f, 35f, 36f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 58f, 59f, 60f, 0f, 0f, 0f, 64f, 65f, 66f, 0f, 0f, 0f, 70f, 71f, 72f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 94f, 95f, 96f, 0f, 0f, 0f, 100f, 101f, 102f, 0f, 0f, 0f, 106f, 107f, 108f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 22f, 23f, 24f, 0f, 0f, 0f, 28f, 29f, 30f, 0f, 0f, 0f, 34f, 35f, 36f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 58f, 59f, 60f, 0f, 0f, 0f, 64f, 65f, 66f, 0f, 0f, 0f, 70f, 71f, 72f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 94f, 95f, 96f, 0f, 0f, 0f, 100f, 101f, 102f, 0f, 0f, 0f, 106f, 107f, 108f
            )
        )
    }

    object MaxPool2 {
        val tensor get() = FloatTensor(
            Shape(2, 4, 4, 2),
            floatArrayOf(
                8f, 1f, 5f, 2f, 1f, 4f, 2f, 6f,
                1f, 4f, 2f, 3f, 4f, 0f, 3f, -1f,
                4f, 8f, 6f, 5f, 3f, 3f, 0f, 0f,
                0f, 1f, -1f, 2f, 5f, 5f, 0f, 0f,

                8f, 1f, 5f, 2f, 1f, 4f, 2f, 6f,
                1f, 4f, 2f, 3f, 4f, 0f, 3f, -1f,
                4f, 8f, 6f, 5f, 3f, 3f, 0f, 0f,
                0f, 1f, -1f, 2f, 5f, 5f, 0f, 0f
            ),
        )
        val expectedOut = FloatTensor(
            Shape(2, 2, 2, 2),
            floatArrayOf(
                8f, 4f, 4f, 6f,
                6f, 8f, 5f, 5f,
                8f, 4f, 4f, 6f,
                6f, 8f, 5f, 5f
            )
        )
        val expectedGrad = FloatTensor(
            Shape(2, 4, 4, 2),
            floatArrayOf(
                8f, 0f, 0f, 0f, 0f, 0f, 0f, 6f,
                0f, 4f, 0f, 0f, 4f, 0f, 0f, 0f,
                0f, 8f, 6f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 5f, 5f, 0f, 0f,

                8f, 0f, 0f, 0f, 0f, 0f, 0f, 6f,
                0f, 4f, 0f, 0f, 4f, 0f, 0f, 0f,
                0f, 8f, 6f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 5f, 5f, 0f, 0f
            )
        )
    }
}

class MaxPoolTest : AnnotationSpec() {
    @Test fun simple() {
        val x = FloatTensor(Shape(1, 4, 4, 1), floats(16))
        val y = maxPool(x, 2, 2)
        y shouldBeExactly FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(6F, 8F, 14F, 16F))
    }
    @Test fun highOrder() {
        val inputShape = Shape(1, 4, 4, 1)
        val x = FloatTensor(inputShape, floats(16))
        fun f(x: DTensor) = maxPool(x * x * x, 2, 2)
        val outputShape = Shape(1, 2, 2, 1)
        f(x) shouldBeExactly FloatTensor(outputShape, floatArrayOf(216F, 512F, 2744F, 4096F))
        val d1Shape = outputShape + inputShape
        val d1 = FloatTensor(d1Shape, floatArrayOf(
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 108.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 192.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 588.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 768.0F))
        forwardDerivative(x, ::f) shouldBeExactly d1
        reverseDerivative(x, ::f) shouldBeExactly d1.leftTranspose(outputShape, inputShape)
        val d2 = forwardDerivative(x) { xx -> forwardDerivative(xx, ::f) }
        d2.shape shouldBe outputShape + inputShape + inputShape
        forwardDerivative(x) { xx -> forwardDerivative(xx, ::f) } shouldBeExactly d2
        reverseDerivative(x) { xx -> forwardDerivative(xx, ::f) } shouldBeExactly d2.leftTranspose(outputShape + inputShape, inputShape)
        reverseDerivative(x) { xx -> reverseDerivative(xx, ::f) } shouldBeExactly d2.leftTranspose(outputShape, inputShape + inputShape).leftTranspose(inputShape, inputShape)
        forwardDerivative(x) { xx -> reverseDerivative(xx, ::f) } shouldBeExactly d2.leftTranspose(outputShape + inputShape, inputShape)
    }
    @Test fun maxPool() {
        forAll(table(
            headers("Tensor", "Expected Out", "Expected Grad"),
            row(
                MaxPoolTestValues.MaxPool.tensor,
                MaxPoolTestValues.MaxPool.expectedOut,
                MaxPoolTestValues.MaxPool.expectedGrad
            ),
            row(
                MaxPoolTestValues.MaxPool2.tensor,
                MaxPoolTestValues.MaxPool2.expectedOut,
                MaxPoolTestValues.MaxPool2.expectedGrad
            )
        )) { tensor, expectedOut, expectedGrad ->
            val (output, pullback) = primalAndPullback(tensor) { tensor_ -> maxPool(tensor_, 2, 2) }
            output shouldBeExactly expectedOut

            // note: non-unitary seed to verify correct reverse indexing in gradient
            // nothing special about output as gradient seed, it's just the right shape
            val grad = pullback(output)
            grad shouldBeExactly expectedGrad
        }
    }

    @Test fun nonDivisibleHeight() {
        val shape = Shape(1, 5, 4, 1)
        val tensor = FloatTensor(shape, floats(shape.product()))
        val e = shouldThrow<IllegalArgumentException> { maxPool(tensor, 2, 2) }
        e.message shouldContain "height"
    }

    @Test fun nonDivisibleWidth() {
        val shape = Shape(1, 4, 5, 1)
        val tensor = FloatTensor(shape, floats(shape.product()))
        val e = shouldThrow<IllegalArgumentException> { maxPool(tensor, 2, 2) }
        e.message shouldContain "width"
    }
}
