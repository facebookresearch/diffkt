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
import kotlin.math.*

class TensorTranscendentalTest : AnnotationSpec() {
    val v1 = FloatTensor(Shape(6), { i -> (i + 1) / 10F })

    val t1 = FloatTensor.ones(Shape(3, 2))
    val gamma = 0.577215665F

    @Test fun testLn() {
        val t1 = tensorOf(2F, 3F)
        val output = ln(t1)
        output shouldBeExactly tensorOf(ln(2F), ln(3F))
    }

    @Test fun testLnForwardDerivative() {
        val t1 = tensorOf(2F, 3F)
        val d1 = forwardDerivative(t1) { x: DTensor -> ln(x) }
        d1 shouldBeExactly tensorOf(0.5F, 0f, 0f, 1/3F).reshape(2,2)
    }

    @Test fun testLnReverseDerivative() {
        val t1 = tensorOf(2F, 3F)
        val d1 = reverseDerivative(t1) { x: DTensor -> ln(x) }
        d1 shouldBeExactly tensorOf(0.5F, 0f, 0f, 1/3F).reshape(2,2)
    }

    @Test fun invalidRange() {
        val t = tensorOf(2F, -1F)
        val resultData = (ln(t) as FloatTensor).elements.map { it.value }
        assert(resultData[0] == ln(2f))
        assert(resultData[1].isNaN())
    }

    @Test fun zeros() {
        val t1 = tensorOf(0F, -0F)
        ln(t1) shouldBeExactly tensorOf(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY)
    }

    @Test fun testLnGradientShapeForward() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = forwardDerivative(t1) { x: DTensor ->
            val result = ln(x.reshape(6))
            result
        }
        d1 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0.5f, 0f, 0f, 0f, 0f,
                0f, 0f, 1/3f, 0f, 0f, 0f,
                0f, 0f, 0f, 1/4f, 0f, 0f,
                0f, 0f, 0f, 0f, 0.2f, 0f,
                0f, 0f, 0f, 0f, 0f, 1/6f
        ).reshape(Shape(6, 2, 3))
    }

    @Test fun testLnGradientShapeReverse() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = reverseDerivative(t1) { x: DTensor ->
            val result = ln(x).reshape(6)
            result
        }
        d1 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0.5f, 0f, 0f, 0f, 0f,
                0f, 0f, 1/3f, 0f, 0f, 0f,
                0f, 0f, 0f, 1/4f, 0f, 0f,
                0f, 0f, 0f, 0f, 0.2f, 0f,
                0f, 0f, 0f, 0f, 0f, 1/6f
        ).reshape(Shape(2, 3, 6))
    }

    @Test fun testExp() {
        val t1 = tensorOf(2F, 3F)
        val output = exp(t1)
        output shouldBeExactly tensorOf(exp(2F), exp(3F))
    }

    @Test fun testExpForwardDerivative() {
        val t1 = tensorOf(2F, 3F)
        val f = { x: DTensor -> exp(x) }
        val d1 = forwardDerivative(t1, f)
        val d2 = forwardDerivative2(t1, f)
        d1 shouldBeExactly tensorOf(exp(2F), 0f, 0f, exp(3F)).reshape(2,2)
        d2 shouldBeExactly tensorOf(
                exp(2f), 0f, 0f, 0f,
                0f, 0f, 0f, exp(3f)
        ).reshape(2,2,2)
    }

    @Test fun testExpReverseDerivative() {
        val t1 = tensorOf(2F, 3F)
        val f = { x: DTensor -> exp(x) }
        val d1 = reverseDerivative(t1, f)
        val d2 = reverseDerivative2(t1, f)
        d1 shouldBeExactly tensorOf(exp(2F), 0f, 0f, exp(3F)).reshape(2,2)
        d2 shouldBeExactly tensorOf(
                exp(2f), 0f, 0f, 0f,
                0f, 0f, 0f, exp(3f)
        ).reshape(2,2,2)
    }

    @Test fun testExpMixedDerivative() {
        val t1 = tensorOf(2F, 3F)
        val f = { x: DTensor -> exp(x) }
        val forwardReverseDer = reverseDerivative(t1) { y: DTensor -> forwardDerivative(y, f) }
        val reverseForwardDer = forwardDerivative(t1) { y: DTensor -> reverseDerivative(y, f) }
        val expected = tensorOf(
                exp(2f), 0f, 0f, 0f,
                0f, 0f, 0f, exp(3f)
        ).reshape(2,2,2)
        forwardReverseDer shouldBeExactly expected
        reverseForwardDer shouldBeExactly expected
    }

    @Test fun testExpGradientShapeForward() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = forwardDerivative(t1) { x: DTensor ->
            val result = exp(x.reshape(6))
            result
        }
        d1 shouldBeExactly tensorOf(
                exp(1f), 0f, 0f, 0f, 0f, 0f,
                0f, exp(2f), 0f, 0f, 0f, 0f,
                0f, 0f, exp(3f), 0f, 0f, 0f,
                0f, 0f, 0f, exp(4f), 0f, 0f,
                0f, 0f, 0f, 0f, exp(5f), 0f,
                0f, 0f, 0f, 0f, 0f, exp(6f)
        ).reshape(Shape(6, 2, 3))
    }

    @Test fun testExpGradientShapeReverse() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = reverseDerivative(t1) { x: DTensor ->
            val result = exp(x).reshape(6)
            result
        }
        d1 shouldBeExactly tensorOf(
                exp(1f), 0f, 0f, 0f, 0f, 0f,
                0f, exp(2f), 0f, 0f, 0f, 0f,
                0f, 0f, exp(3f), 0f, 0f, 0f,
                0f, 0f, 0f, exp(4f), 0f, 0f,
                0f, 0f, 0f, 0f, exp(5f), 0f,
                0f, 0f, 0f, 0f, 0f, exp(6f)
        ).reshape(Shape(2, 3, 6))
    }

    @Test fun sinTensor() {
        sin(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> sin((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            if (x == y) 2F * cos(2F * (x + 1) / 10F) else 0F
        })
        forwardDerivative(v1) { x: DTensor -> sin(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> sin(2F*x) } shouldBeExactly deriv
    }

    @Test fun cosTensor() {
        cos(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> cos((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            if (x == y) - 2F * sin(2F * (x + 1) / 10F) else 0F
        })
        forwardDerivative(v1) { x: DTensor -> cos(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> cos(2F*x) } shouldBeExactly deriv
    }

    @Test fun atanTensor() {
        atan(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> atan((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            val k =   2F *  (x + 1) / 10F
            if (x == y) 2F / (1 + k.pow(2)) else 0F
        })
        forwardDerivative(v1) { x: DTensor -> atan(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> atan(2F*x) } shouldBeExactly deriv
    }

    @Test fun expTensor() {
        exp(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> exp((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            if (x == y) 2F * exp(2F * (x + 1) / 10F) else 0F
        })
        forwardDerivative(v1) { x: DTensor -> exp(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> exp(2F*x) } shouldBeExactly deriv
    }

    @Test fun lnTensor() {
        ln(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> ln((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            val k = (x + 1) / 10F
            if (x == y) 1F / k else 0F
        })
        forwardDerivative(v1) { x: DTensor -> ln(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> ln(2F*x) } shouldBeExactly deriv
    }

    @Test fun lgammaTensor() {
        lgamma(t1) shouldBeExactly FloatTensor.zeros(Shape(3, 2))

        forwardDerivative(t1) { x: DTensor -> lgamma(2F * x).sum()} shouldBeExactly FloatTensor.const(
            (2F - 2F * gamma),
            Shape(3, 2)
        )
        reverseDerivative(t1) { x: DTensor -> lgamma(2F * x).sum()} shouldBeExactly FloatTensor.const(
            (2F - 2F * gamma),
            Shape(3, 2)
        )
    }

    @Test fun digammaTensor() {
        digamma(t1) shouldBeExactly FloatTensor.const(-gamma, Shape(3, 2))

        forwardDerivative(t1) { x: DTensor -> digamma(2F * x).sum()} shouldBeExactly FloatTensor.const(
            (PI * PI).toFloat() / 3F - 2F,
            Shape(3, 2)
        )
        reverseDerivative(t1) { x: DTensor -> digamma(2F * x).sum()} shouldBeExactly FloatTensor.const(
            (PI * PI).toFloat() / 3F - 2F,
            Shape(3, 2)
        )
    }

    @Test fun polygammaTensor() {
        polygamma(1, t1) shouldBeExactly FloatTensor.const((PI * PI).toFloat() / 6F, Shape(3, 2))
        polygamma(2, t1) shouldBeExactly FloatTensor.const(-2.4041138F, Shape(3, 2))

        forwardDerivative(t1) { x: DTensor -> polygamma(1, 2F * x).sum()} shouldBeExactly FloatTensor.const(
            -0.8082276F,
            Shape(3, 2)
        )
        reverseDerivative(t1) { x: DTensor -> polygamma(1, 2F * x).sum()} shouldBeExactly FloatTensor.const(
            -0.8082276F,
            Shape(3, 2)
        )

        forwardDerivative(t1) { x: DTensor -> polygamma(2, 2F * x).sum()} shouldBeExactly FloatTensor.const(
            2F * PI.pow(
                4
            ).toFloat() / 15F - 12F, Shape(3, 2)
        )
        reverseDerivative(t1) { x: DTensor -> polygamma(2, 2F * x).sum()} shouldBeExactly FloatTensor.const(
            2F * PI.pow(
                4
            ).toFloat() / 15F - 12F, Shape(3, 2)
        )
    }

    private fun square(x: Float) = x * x
    private fun tanhDeriv(x: Float) = 1F - square(tanh(x))
    @Test fun tanhTensor() {
        tanh(v1) shouldBeExactly FloatTensor(Shape(6)) { i -> tanh((i + 1) / 10F) }

        val deriv = FloatTensor(Shape(6, 6), FloatArray(36) { i ->
            val x = i / 6
            val y = i % 6
            val v = (x + 1) / 10F
            if (x == y) 2F * tanhDeriv(2F * v) else 0F
        })
        forwardDerivative(v1) { x: DTensor -> tanh(2F*x) } shouldBeExactly deriv
        reverseDerivative(v1) { x: DTensor -> tanh(2F*x) } shouldBeExactly deriv
    }
}
