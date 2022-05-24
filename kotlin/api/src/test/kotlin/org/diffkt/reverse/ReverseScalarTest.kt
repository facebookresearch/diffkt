/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.value
import kotlin.math.*
import kotlin.test.assertTrue

class ReverseScalarTest : AnnotationSpec() {
    private fun isClose(x: Float, y: Float) = (x == y) || (x - y).absoluteValue / (x.absoluteValue + y.absoluteValue) < 0.0002 || (x - y).absoluteValue < 0.0002
    private fun assertClose(expected: Float, actual: Float) {
        val fact = isClose(expected, actual)
        if (!fact) // this condition is present so that we can place a breakpoint on the following line to diagnose failures
            assertTrue(fact, "Expected: $expected, Actual: $actual")
    }

    fun reverseDerivative1(x: Float, f: (DScalar) -> DScalar): Float = reverseDerivative(FloatScalar(x), f).value
    fun reverseDerivative2(x: Float, f: (DScalar) -> DScalar): Float = reverseDerivative(2, FloatScalar(x), f).value
    fun reverseDerivative3(x: Float, f: (DScalar) -> DScalar): Float = reverseDerivative(3, FloatScalar(x), f).value
    fun reverseDerivative4(x: Float, f: (DScalar) -> DScalar): Float = reverseDerivative(4, FloatScalar(x), f).value

    @Test fun testMul1() {
        val f = { x: DScalar -> x * x }
        val x = 3F
        assertClose(x * x, primal(x, f))
        assertClose(2 * x, reverseDerivative1(x, f))
        assertClose( 2F, reverseDerivative2(x, f))
        assertClose( 0F, reverseDerivative3(x, f))
    }
    @Test fun testMul2() {
        val f = { x: DScalar -> x * x * x }
        val x = 5.1F
        assertClose(x * x * x, primal(x, f))
        assertClose(3 * x * x, reverseDerivative1(x, f))
        assertClose(  6 * x, reverseDerivative2(x, f))
        assertClose(   6F, reverseDerivative3(x, f))
        assertClose(   0F, reverseDerivative4(x, f))
    }

    @Test fun testReciprocal() {
        val f = { x: DScalar -> 1F / x }
        val x = 2.1F
        assertClose(1F / x, primal(x, f))
        assertClose(-1F / (x * x), reverseDerivative1(x, f))
        assertClose(2F / (x * x * x), reverseDerivative2(x, f))
        assertClose(-6F / (x * x * x * x), reverseDerivative3(x, f))
    }

    @Test fun testSin1() {
        val sin1 = { x: DScalar -> sin(x) } // primal
        val cos1 = reverseDiff(sin1) // first derivative
        val sin2 = reverseDiff(cos1) // second derivative
        val cos2 = reverseDiff(sin2) // third derivative

        val value = 1.2F
        val v = FloatScalar(value)
        assertClose(sin1(v).value, sin(value))
        assertClose(cos1(v).value, cos(value))
        assertClose(sin2(v).value, -sin(value))
        assertClose(cos2(v).value, -cos(value))
    }
    @Test fun testSin2() {
        val sin1 = { x: DScalar -> sin(2F * x) } // primal
        val cos1 = reverseDiff(sin1) // first derivative
        val sin2 = reverseDiff(cos1) // second derivative
        val cos2 = reverseDiff(sin2) // third derivative

        val value = 1.1F
        val v = FloatScalar(value)
        assertClose(sin1(v).value, sin(2 * value))
        assertClose(cos1(v).value, 2 * cos(2 * value))
        assertClose(sin2(v).value, -4 * sin(2 * value))
        assertClose(cos2(v).value, -8 * cos(2 * value))
    }

    @Test fun testExp() {
        val exp = { x: DScalar -> exp(x) }
        val value = 1.1F
        assertClose(exp(value), primal(value, exp))
        assertClose(exp(value), reverseDerivative1(value, exp))
        assertClose(exp(value), reverseDerivative2(value, exp))
        assertClose(exp(value), reverseDerivative3(value, exp))
        assertClose(exp(value), reverseDerivative4(value, exp))
    }

    @Test fun testLn() {
        val ln = { x: DScalar -> ln(x) }
        val lnp = { x: DScalar -> 1F / x }
        val lnpp = { x: DScalar -> - 1F / (x * x) }
        val lnppp = { x: DScalar -> 2F / (x * x * x) }
        val lnpppp = { x: DScalar -> - 6F / (x * x * x * x) }

        val value = 1.1F
        assertClose(ln(value), primal(value, ln))
        assertClose(primal(value, lnp), reverseDerivative1(value, ln))
        assertClose(primal(value, lnpp), reverseDerivative2(value, ln))
        assertClose(primal(value, lnppp), reverseDerivative3(value, ln))
        assertClose(primal(value, lnpppp), reverseDerivative4(value, ln))
    }

    @Test fun testTan() {
        val tan = { x: DScalar -> tan(x) }
        val value = 1.1F
        val dtan = { x: DScalar -> // derivative of tan(x)
            val y = cos(x)
            1F / (y * y)
        }
        assertClose(tan(value), primal(value, tan))
        assertClose(primal(value, dtan), reverseDerivative1(value, tan))
        assertClose(reverseDerivative1(value, dtan), reverseDerivative2(value, tan))
        assertClose(reverseDerivative2(value, dtan), reverseDerivative3(value, tan))
        assertClose(reverseDerivative3(value, dtan), reverseDerivative4(value, tan))
    }

    private val functions = arrayOf(
            Pair("1.1/x", { x: DScalar -> 1.1F / x }),
            Pair("3x", { x: DScalar -> 3F * x }),
            Pair("sin(x)", { x: DScalar -> sin(x) }),
            Pair("cos(x)", { x: DScalar -> cos(x) }),
            Pair("x*x", { x: DScalar -> x * x }),
            Pair("(x+5)/(x-5)", { x: DScalar -> (x + 5F) / (x - 5F) }),
            Pair("exp(x)", { x: DScalar -> exp(x) }),
            // Pair("ln(x+1.1)", { x: DScalar -> if (x.value <= -0.999F) zeroDFloat else ln(x + 1.1F) }), // protect against undefined values
            Pair("x+1.3", { x: DScalar -> x + 1.3F }),
            Pair("x", { x: DScalar -> x })
    )
    private val values = arrayOf(0.2F, 0.5F, 1.1F, 2.1F)

    @Test fun testChain() {
        fun testChain(f: (DScalar)-> DScalar, g: (DScalar)-> DScalar, xv: Float) {
            val fg = { x: DScalar -> f(g(x)) }

            // d/dx f(g(x)) = f'(g(x))g'(x)
            val fp = reverseDiff(f)
            val gp = reverseDiff(g)
            val x = FloatScalar(xv)
            val expected1 = (fp(g(x)) * gp(x))
            val actual1 = reverseDerivative1(xv, fg)
            assertClose(expected1.value, actual1)

            // d/dx d/dx f(g(x)) = f'(g(x))g''(x) + g'(x)f''(g(x))g'(x)
            val fpp = reverseDiff(fp)
            val gpp = reverseDiff(gp)
            val expected2 = fp(g(x)) * gpp(x) + gp(x) * fpp(g(x)) * gp(x)
            val actual2 = reverseDerivative(2, FloatScalar(xv), fg)
            assertClose(expected2.value, actual2.value)
        }

        for (f in functions) for (g in functions) for (xv in values)
            testChain(f.second, g.second, xv)
    }

    @Test fun testDivision() {
        fun testDiv(g: (DScalar)-> DScalar, h: (DScalar)-> DScalar, xv: Float) {
            val f = { x: DScalar -> g(x) / h(x) }

            // d/dx g(x)/h(x) = g'(x)/h(x) - g(x)h'(x)/h(x)h(x)
            val gp = reverseDiff(g)
            val hp = reverseDiff(h)
            val fp = { x: DScalar -> gp(x) / h(x) - g(x) * hp(x) / (h(x) * h(x)) }

            // d/dx d/dx g(x)/h(x) = (g''(x) - 2f'(x)h'(x) - f(x)h''(x)) / h(x)
            val gpp = reverseDiff(gp)
            val hpp = reverseDiff(hp)
            val fpp = { x: DScalar -> (gpp(x) - 2F * fp(x) * hp(x) - f(x) * hpp(x)) / h(x) }

            // Check the first derivative
            val x = FloatScalar(xv)
            val expected1 = fp(x)
            val actual1 = reverseDerivative1(xv, f)
            assertClose(expected1.value, actual1)

            // Check the second derivative
            val expected2 = fpp(x)
            val actual2 = reverseDerivative2(xv, f)
            assertClose(expected2.value, actual2)
        }

        for (f in functions) for (g in functions) for (xv in values)
            testDiv(f.second, g.second, xv)
    }
}
