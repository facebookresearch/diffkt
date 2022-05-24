/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import org.diffkt.*
import kotlin.math.pow
import kotlin.test.assertTrue
import testutils.*

class PowerTest : AnnotationSpec() {

    @Test fun testPow() {
        val f = { x: DTensor -> x.pow(3f)}
        val t1 = tensorOf(2F, 3F)
        val t2 = f(t1)
        assertTrue(t2 is FloatTensor)
        t2 shouldBeExactly tensorOf(8F, 27F)
    }

    @Test fun testPowInversion() {
        val f = { x: DTensor -> x.pow(-1f)}
        val t1 = tensorOf(2F, 3F)
        val t2 = f(t1)
        assertTrue(t2 is FloatTensor)
        t2 shouldBeExactly tensorOf(0.5F, 1/3F)
    }

    @Test fun testForwardDerivativePow() {
        val t1 = tensorOf(2F, 3F)
        val d1 = forwardDerivative1(t1) { x: DTensor -> x.pow(3f)}
        val d2 = forwardDerivative2(t1) { x: DTensor -> x.pow(3f) }
        d1 shouldBeExactly tensorOf(12F, 0F, 0F, 27F).reshape(2,2)
        d2 shouldBeExactly tensorOf(12f, 0f, 0f, 0f, 0f, 0f, 0f, 18f).reshape(Shape(2, 2, 2))
    }

    @Test fun testReverseDerivativePow() {
        val t1 = tensorOf(2F, 3F)
        val d1 = reverseDerivative(t1) { x: DTensor -> x.pow(3f) }
        val d2 = reverseDerivative2(t1) { x: DTensor -> x.pow(3f) }
        d1 shouldBeExactly tensorOf(12F, 0F, 0F, 27F).reshape(2,2)
        d2 shouldBeExactly tensorOf(12f, 0f, 0f, 0f, 0f, 0f, 0f, 18f).reshape(Shape(2, 2, 2))
    }

    @Test fun testForwardDerivativePowNegativeBase() {
        val t1 = tensorOf(-2F, -3F)
        val d1 = forwardDerivative(t1) { x: DTensor -> x.pow(3)}
        val d2 = forwardDerivative2(t1) { x: DTensor -> x.pow(3)}
        d1 shouldBeExactly tensorOf(12F, 0F, 0F, 27F).reshape(2,2)
        d2 shouldBeExactly tensorOf(-12f, 0f, 0f, 0f, 0f, 0f, 0f, -18f).reshape(Shape(2, 2, 2))
    }

    @Test fun testReverseDerivativePowNegativeBase() {
        val t1 = tensorOf(-2F, -3F)
        val d1 = reverseDerivative(t1) { x: DTensor -> x.pow(3) }
        val d2 = reverseDerivative2(t1) { x: DTensor -> x.pow(3)}
        d1 shouldBeExactly tensorOf(12F, 0F, 0F, 27F).reshape(2,2)
        d2 shouldBeExactly tensorOf(-12f, 0f, 0f, 0f, 0f, 0f, 0f, -18f).reshape(Shape(2, 2, 2))
    }

    @Test fun testForwardDerivativePow3D() {
        val t1 = FloatTensor(Shape(2,3,2)) { it.toFloat() }
        val d1 = forwardDerivative(t1) { x: DTensor -> x.pow(3f)}
        d1 shouldBeExactly tensorOf(
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 3f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 12f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 27f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 48f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 75f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 108f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 147f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 192f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 243f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 300f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 363f
        ).reshape(2,3,2,2,3,2)
    }

    @Test fun testReverseDerivativePow3D() {
        val t1 = FloatTensor(Shape(2,3,2)) { it.toFloat() }
        val d1 = reverseDerivative(t1) { x: DTensor -> x.pow(3f)}
        d1 shouldBeExactly tensorOf(
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 3f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 12f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 27f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 48f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 75f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 108f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 147f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 192f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 243f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 300f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 363f
        ).reshape(2,3,2,2,3,2)
    }

    @Test fun testForwardDerivativePowInversion() {
        val t1 = tensorOf(2F, 3F)
        val d1 = forwardDerivative1(t1) { x: DTensor -> x.pow(-1f)}
        d1 shouldBeExactly tensorOf(-0.25F, 0F, 0F, -1F/9F).reshape(2,2)
    }

    @Test fun testReverseDerivativePowInversion() {
        val t1 = tensorOf(2F, 3F)
        val d1 = reverseDerivative(t1) { x: DTensor -> x.pow(-1f)}
        d1 shouldBeExactly tensorOf(-0.25F, 0F, 0F, -1F/9F).reshape(2,2)
    }

    @Test fun testForwardDerivativeRoot() {
        val t1 = tensorOf(4F, 16F)
        val d1 = forwardDerivative1(t1) { x: DTensor -> x.pow(1.5f)}
        d1 shouldBeExactly tensorOf(3F, 0F, 0F, 6F).reshape(2,2)
    }

    @Test fun testReverseDerivativeRoot() {
        val t1 = tensorOf(4F, 16F)
        val d1 = reverseDerivative(t1) { x: DTensor -> x.pow(1.5f)}
        d1 shouldBeExactly tensorOf(3F, 0F, 0F, 6F).reshape(2,2)
    }

    @Test fun testPowGradientShape01() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = forwardDerivative(t1) { x: DTensor ->
            val result = x.reshape(6).pow(2)
            result
        }
        d1 shouldBeExactly tensorOf(
            2f, 0f, 0f, 0f, 0f, 0f,
            0f, 4f, 0f, 0f, 0f, 0f,
            0f, 0f, 6f, 0f, 0f, 0f,
            0f, 0f, 0f, 8f, 0f, 0f,
            0f, 0f, 0f, 0f, 10f, 0f,
            0f, 0f, 0f, 0f, 0f, 12f
        ).reshape(Shape(6, 2, 3))
    }

    @Test fun testPowGradientShape02() {
        val t1 = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val d1 = reverseDerivative(t1) { x: DTensor ->
            val result = x.pow(2).reshape(6)
            result
        }
        d1 shouldBeExactly tensorOf(
            2f, 0f, 0f, 0f, 0f, 0f,
            0f, 4f, 0f, 0f, 0f, 0f,
            0f, 0f, 6f, 0f, 0f, 0f,
            0f, 0f, 0f, 8f, 0f, 0f,
            0f, 0f, 0f, 0f, 10f, 0f,
            0f, 0f, 0f, 0f, 0f, 12f
        ).reshape(Shape(2, 3, 6))
    }

    @Test fun divideByZero() {
        val t1 = tensorOf(2F, 0F)
        t1.pow(-0.2F) shouldBeExactly tensorOf(2F.pow(-0.2F), 0F.pow(-0.2F))
    }

    @Test fun noRealRoot() {
        val t1 = tensorOf(2F, -5F)
        t1.pow(3.2F) shouldBeExactly tensorOf(2F.pow(3.2F), (-5F).pow(3.2F))
    }

    @Test fun powDScalar01() {
        val baseValue = tensorOf(2F, 3F)
        val exponent = FloatScalar(2F)
        baseValue.pow(exponent) shouldBeExactly tensorOf(4F, 9F)
        forwardDerivative(exponent as DTensor) { EXP: DTensor -> baseValue.pow(EXP as DScalar) } shouldBeExactly tensorOf(2.7725887F, 9.88751F)
        reverseDerivative(exponent as DTensor) { EXP: DTensor -> baseValue.pow(EXP as DScalar) } shouldBeExactly tensorOf(2.7725887F, 9.88751F)
    }

    @Test fun powDScalar02() {
        val baseValue = FloatScalar(1.1f)
        val exponent = FloatScalar(2.2f)
        val result: DScalar = baseValue.pow(exponent)
        result.value shouldBeExactly 1.1f.pow(2.2f)
    }

    @Test fun powTwoTensors() {
        val base = tensorOf(2F, 3F)
        val exponent = tensorOf(3F, 2f)
        base.pow(exponent) shouldBeExactly tensorOf(8f, 9f)
    }

    @Test fun powTwoTensorsForward() {
        val base = tensorOf(2F, 3F)
        val exponent = tensorOf(3F, 2f)
        val (dBase, dExp) = forwardDerivative(base, exponent) { b, e -> b.pow(e) }
        dBase shouldBeExactly tensorOf( 12F, 0F, 0F, 6F).reshape(Shape(2, 2))
        dExp shouldBeExactly tensorOf( 5.5451775F, 0.0F, 0.0F, 9.88751F).reshape(Shape(2, 2))
    }

    @Test fun powTwoTensorsReverse() {
        val base = tensorOf(2F, 3F)
        val exponent = tensorOf(3F, 2f)
        val (dBase, dExp) = reverseDerivative(base, exponent) { b, e -> b.pow(e) }
        dBase shouldBeExactly tensorOf( 12F, 0F, 0F, 6F).reshape(Shape(2, 2))
        dExp shouldBeExactly tensorOf( 5.5451775F, 0.0F, 0.0F, 9.88751F).reshape(Shape(2, 2))
    }
}
