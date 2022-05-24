/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import kotlin.math.*
import kotlin.random.Random
import kotlin.test.assertTrue
import testutils.*

class MixedScalarTest : AnnotationSpec() {
    private val random: Random = Random(1000)

    private fun isClose(x: Float, y: Float) = (x == y) || (x - y).absoluteValue / (x.absoluteValue + y.absoluteValue) < 0.0002 || (x - y).absoluteValue < 0.0002
    private fun assertClose(expected: Float, actual: Float) {
        val fact = isClose(expected, actual)
        if (!fact) // this condition is present so that we can place a breakpoint on the following line to diagnose failures
            assertTrue(fact, "Expected: $expected, Actual: $actual")
    }

    private fun mixedDerivative1(x: Float, f: (DScalar) -> DScalar): Float = mixedDerivative(FloatScalar(x), f).value
    private fun mixedDerivative2(x: Float, f: (DScalar) -> DScalar): Float = mixedDerivative(2, FloatScalar(x), f).value
    private fun mixedDerivative3(x: Float, f: (DScalar) -> DScalar): Float = mixedDerivative(3, FloatScalar(x), f).value
    private fun mixedDerivative4(x: Float, f: (DScalar) -> DScalar): Float = mixedDerivative(4, FloatScalar(x), f).value

    private fun mixedDerivative(n: Int, x: DScalar, f: (DScalar) -> DScalar): DScalar {
        if (n == 0) return f(x)
        return mixedDerivative(x) { y: DScalar -> mixedDerivative(n - 1, y, f) }
    }

    private fun mixedDerivative(x: DScalar, f: (DScalar) -> DScalar): DScalar {
        return if (random.nextBoolean()) forwardDerivative(x, f) else reverseDerivative(x, f)
    }

    private fun mixedDiff(f: (DScalar) -> DScalar): (DScalar) -> DScalar = { x: DScalar -> mixedDerivative(x, f) }

    @Test fun testMul1() {
        val f = { x: DScalar -> x * x }
        val x = 3F
        assertClose(x * x, primal(x, f))
        assertClose(2 * x, mixedDerivative1(x, f))
        assertClose( 2F, mixedDerivative2(x, f))
        assertClose( 0F, mixedDerivative3(x, f))
    }
    @Test fun testMul2() {
        val f = { x: DScalar -> x * x * x }
        val x = 5.1F
        assertClose(x * x * x, primal(x, f))
        assertClose(3 * x * x, mixedDerivative1(x, f))
        assertClose(  6 * x, mixedDerivative2(x, f))
        assertClose(   6F, mixedDerivative3(x, f))
        assertClose(   0F, mixedDerivative4(x, f))
    }

    @Test fun testReciprocal() {
        val f = { x: DScalar -> 1F / x }
        val x = 2.1F
        assertClose(1F / x, primal(x, f))
        assertClose(-1F / (x * x), mixedDerivative1(x, f))
        assertClose(2F / (x * x * x), mixedDerivative2(x, f))
        assertClose(-6F / (x * x * x * x), mixedDerivative3(x, f))
    }

    @Test fun testSin1() {
        val sin1 = { x: DScalar -> sin(x) } // primal
        val cos1 = mixedDiff(sin1) // first derivative
        val sin2 = mixedDiff(cos1) // second derivative
        val cos2 = mixedDiff(sin2) // third derivative

        val value = 1.2F
        val v = FloatScalar(value)
        assertClose(sin1(v).value, sin(value))
        assertClose(cos1(v).value, cos(value))
        assertClose(sin2(v).value, -sin(value))
        assertClose(cos2(v).value, -cos(value))
    }
    @Test fun testSin2() {
        val sin1 = { x: DScalar -> sin(2F*x) } // primal
        val cos1 = mixedDiff(sin1) // first derivative
        val sin2 = mixedDiff(cos1) // second derivative
        val cos2 = mixedDiff(sin2) // third derivative

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
        assertClose(exp(value), mixedDerivative1(value, exp))
        assertClose(exp(value), mixedDerivative2(value, exp))
        assertClose(exp(value), mixedDerivative3(value, exp))
        assertClose(exp(value), mixedDerivative4(value, exp))
    }

    @Test fun testLn() {
        val ln = { x: DScalar -> ln(x) }
        val lnp = { x: DScalar -> 1F / x }
        val lnpp = { x: DScalar -> - 1F / (x * x) }
        val lnppp = { x: DScalar -> 2F / (x * x * x) }
        val lnpppp = { x: DScalar -> - 6F / (x * x * x * x) }

        val value = 1.1F
        assertClose(ln(value), primal(value, ln))
        assertClose(primal(value, lnp), mixedDerivative1(value, ln))
        assertClose(primal(value, lnpp), mixedDerivative2(value, ln))
        assertClose(primal(value, lnppp), mixedDerivative3(value, ln))
        assertClose(primal(value, lnpppp), mixedDerivative4(value, ln))
    }

    @Test fun testTan() {
        val tan = { x: DScalar -> tan(x) }
        val value = 1.1F
        val dtan = { x: DScalar -> // derivative of tan(x)
            val y = cos(x)
            1F / (y * y)
        }
        assertClose(tan(value), primal(value, tan))
        assertClose(primal(value, dtan), mixedDerivative1(value, tan))
        assertClose(mixedDerivative1(value, dtan), mixedDerivative2(value, tan))
        assertClose(mixedDerivative2(value, dtan), mixedDerivative3(value, tan))
        assertClose(mixedDerivative3(value, dtan), mixedDerivative4(value, tan))
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
        fun testChain(f: (DScalar)->DScalar, g: (DScalar)->DScalar, xv: Float) {
            val fg = { x: DScalar -> f(g(x)) }

            // d/dx f(g(x)) = f'(g(x))g'(x)
            val fp = mixedDiff(f)
            val gp = mixedDiff(g)
            val x = FloatScalar(xv)
            val expected1 = (fp(g(x)) * gp(x))
            val actual1 = mixedDerivative1(xv, fg)
            assertClose(expected1.value, actual1)

            // d/dx d/dx f(g(x)) = f'(g(x))g''(x) + g'(x)f''(g(x))g'(x)
            val fpp = mixedDiff(fp)
            val gpp = mixedDiff(gp)
            val expected2 = fp(g(x)) * gpp(x) + gp(x) * fpp(g(x)) * gp(x)
            val actual2 = mixedDerivative(2, FloatScalar(xv), fg)
            assertClose(expected2.value, actual2.value)
        }

        for (f in functions) for (g in functions) for (xv in values)
            testChain(f.second, g.second, xv)
    }

    @Test fun testDivision() {
        fun testDiv(g: (DScalar)->DScalar, h: (DScalar)->DScalar, xv: Float) {
            val f = { x: DScalar -> g(x) / h(x) }

            // d/dx g(x)/h(x) = g'(x)/h(x) - g(x)h'(x)/h(x)h(x)
            val gp = mixedDiff(g)
            val hp = mixedDiff(h)
            val fp = { x: DScalar -> gp(x)/h(x) - g(x)*hp(x)/(h(x)*h(x)) }

            // d/dx d/dx g(x)/h(x) = (g''(x) - 2f'(x)h'(x) - f(x)h''(x)) / h(x)
            val gpp = mixedDiff(gp)
            val hpp = mixedDiff(hp)
            val fpp = { x: DScalar -> (gpp(x) - 2F*fp(x)*hp(x) - f(x)*hpp(x)) / h(x) }

            // Check the first derivative
            val x = FloatScalar(xv)
            val expected1 = fp(x)
            val actual1 = mixedDerivative1(xv, f)
            assertClose(expected1.value, actual1)

            // Check the second derivative
            val expected2 = fpp(x)
            val actual2 = mixedDerivative2(xv, f)
            assertClose(expected2.value, actual2)
        }

        for (f in functions) for (g in functions) for (xv in values)
            testDiv(f.second, g.second, xv)
    }
}
