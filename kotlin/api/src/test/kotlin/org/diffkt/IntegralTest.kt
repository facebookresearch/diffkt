/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.Matcher
import io.kotest.matchers.MatcherResult
import io.kotest.matchers.shouldBe
import testutils.shouldBeExactly
import kotlin.math.PI
import kotlin.math.absoluteValue
import testutils.*

private infix fun Float.isNear(x: Float): Boolean {
    if (this == x) return true
    val absDiff = (this - x).absoluteValue
    if (absDiff < 0.0001) return true
    return absDiff / (this.absoluteValue + x.absoluteValue) < 0.0001
}
private fun near(x: Float) = object : Matcher<Float> {
    override fun test(value: Float) = MatcherResult(value isNear x, "$value should be near $x", "$value should not be near $x")
}

private infix fun Float.shouldBeNear(x: Float) = this shouldBe near(x)

class IntegralTest : AnnotationSpec() {
    private val functions: Array<(DScalar) -> DScalar> = arrayOf(
            { _: DScalar -> FloatScalar(1F) },
            { x: DScalar -> x },
            { x: DScalar -> sin(x) },
            { x: DScalar -> cos(3F * x) },
            { x: DScalar -> exp(x) },
            { x: DScalar -> 3F + x * (4F + x * (5F + x * 6F)) },
            { x: DScalar -> 2F / (x * x) },
            { x: DScalar -> ln(x) },
    )

    @Test
    fun integralTest0f() {
        for (i in functions) {
            val f = forwardDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = integral(a, b, f = f)
            val expected = i(b) - i(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest0r() {
        for (i in functions) {
            val f = reverseDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = integral(a, b, f = f)
            val expected = i(b) - i(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest1ff() {
        for (i in functions) {
            val f = forwardDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = forwardDerivative(a) { aa: DScalar -> integral(aa, b, f = f) }
            val expected = -f(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest1fr() {
        for (i in functions) {
            val f = forwardDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = reverseDerivative(a) { aa: DScalar -> integral(aa, b, f = f) }
            val expected = -f(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest1rr() {
        for (i in functions) {
            val f = reverseDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = reverseDerivative(a) { aa: DScalar -> integral(aa, b, f = f) }
            val expected = -f(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest1rf() {
        for (i in functions) {
            val f = reverseDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = forwardDerivative(a) { aa: DScalar -> integral(aa, b, f = f) }
            val expected = -f(a)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest2ff() {
        for (i in functions) {
            val f = forwardDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = forwardDerivative(b) { bb: DScalar -> integral(a, bb, f = f) }
            val expected = f(b)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest2fr() {
        for (i in functions) {
            val f = forwardDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = reverseDerivative(b) { bb: DScalar -> integral(a, bb, f = f) }
            val expected = f(b)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest2rf() {
        for (i in functions) {
            val f = reverseDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = forwardDerivative(b) { bb: DScalar -> integral(a, bb, f = f) }
            val expected = f(b)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest2rr() {
        for (i in functions) {
            val f = reverseDiff(i)
            val a = FloatScalar(1F)
            val b = FloatScalar(2F)
            val result = reverseDerivative(b) { bb: DScalar -> integral(a, bb, f = f) }
            val expected = f(b)
            result.value shouldBeNear expected.value
        }
    }

    @Test
    fun integralTest3() {
        val k = FloatScalar(1.2F)
        fun f(x: DScalar) = k * sin(k * x)
        fun i(x: DScalar) = -cos(k * x)
        val a = FloatScalar(1F)
        val b = FloatScalar(2F)
        integral(a, b, f = ::f).value shouldBeNear (i(b) - i(a)).value
    }

    /**
     * Tests the case when the function doesn't return a [FloatScalar].
     */
    @Test
    fun integralTest4() {
        fun f(x: DScalar, k: DScalar) = k * sin(k * x)
        fun i(x: DScalar, k: DScalar) = -cos(k * x)

        val a = FloatScalar(1.2F)
        val b = FloatScalar(2.3F)
        val k = FloatScalar(1.4F)
        val fd = forwardDerivative(k) { kk: DScalar -> integral(a, b) { x: DScalar -> f(x, kk) } }
        fd.value shouldBeNear (f(k, b) - f(k, a)).value
        val rd = reverseDerivative(k) { kk: DScalar -> integral(a, b) { x: DScalar -> f(x, kk) } }
        rd.value shouldBeNear (f(k, b) - f(k, a)).value
    }

    /**
     * Test the derivative of the integral of a discontinuous function.
     * See https://dl.acm.org/doi/pdf/10.1145/3450626.3459775
     */
    @Test
    fun derivativeIntegralDiscontinuous01() {
        val t = PI.toFloat()
        fun f(x: DScalar) = if (x < t) FloatScalar.ZERO else FloatScalar.ONE
        fun fi(l: DScalar) = integral(FloatScalar.ZERO, l, f = ::f)
        val b1 = FloatScalar(3f)
        val b2 = FloatScalar(4f)
        fun fid1(l: DScalar) = forwardDerivative(l, ::fi)
        fun fid2(l: DScalar) = reverseDerivative(l, ::fi)
        fid1(b1) shouldBeExactly f(b1)
        fid2(b1) shouldBeExactly f(b1)
        fid1(b2) shouldBeExactly f(b2)
        fid2(b2) shouldBeExactly f(b2)
    }
}
