/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.forward.ForwardScalar
import org.diffkt.reverse.ReverseDerivativeID
import org.diffkt.reverse.ReverseScalar
import kotlin.math.abs

/**
 * Compute the integral of the function [f] from [a] to [b] using [Romberg's method](https://en.wikipedia.org/wiki/Romberg%27s_method).
 *
 * Note: if the performance of this becomes important (for example, we want to minimize the number of function evaluations), then we might
 * consider moving to another adaptive method such as [Clenshaw–Curtis quadrature](https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature)
 *
 * @param a The start point of the integral
 * @param b The end point of the integral
 * @param n The maximum order of the integration.  This may require O(2^n) function evaluations,
 *          but can give a precise result for polynomials of order n.
 * @param eps Desired accuracy.  If an order *i* and order *i+1* results agree within a
 *            relative accuracy of [eps], then that result is returned rather than continuing
 *            to compute the integral using further samples.
 * @param f The function to be integrated
 */
fun integral(
        a: DScalar,
        b: DScalar,
        n: Int = 5,
        eps: Float = 1E-5F,
        f: (DScalar) -> DScalar
): DScalar {
    return when {
        a.derivativeID == b.derivativeID -> {
            when (a) {
                is FloatScalar -> rombergIntegralTryPrimitive(a, b as FloatScalar, f, n, eps)
                is ForwardScalar -> {
                    b as ForwardScalar
                    val primal = integral(a.primal, b.primal, n, eps, f)
                    val gradient = b.tangent * f(b.primal) - a.tangent * f(a.primal)
                    ForwardScalar(primal, a.derivativeID, gradient)
                }
                is ReverseScalar -> IntegralReverseScalar(integral(a.primal, (b as ReverseScalar).primal, n, eps, f), a, b, f, a.derivativeID)
                else -> throw IllegalArgumentException()
            }
        }
        a.derivativeID.sequence > b.derivativeID.sequence -> {
            when (a) {
                is ForwardScalar -> {
                    val primal = integral(a.primal, b, n, eps, f)
                    val gradient = - a.tangent * f(a.primal)
                    ForwardScalar(primal, a.derivativeID, gradient)
                }
                is ReverseScalar -> IntegralReverseScalar(integral(a.primal, b, n, eps, f), a, null, f, a.derivativeID)
                else -> throw IllegalArgumentException()
            }
        }
        else -> {
            when (b) {
                is ForwardScalar -> {
                    val primal = integral(a, b.primal, n, eps, f)
                    val gradient = b.tangent * f(b.primal)
                    ForwardScalar(primal, b.derivativeID, gradient)
                }
                is ReverseScalar -> IntegralReverseScalar(integral(a, b.primal, n, eps, f), null, b, f, b.derivativeID)
                else -> throw IllegalArgumentException()
            }
        }
    }
}

class IntegralReverseScalar(
    primal: DScalar,
    val a: ReverseScalar?,
    val b: ReverseScalar?,
    val f: (DScalar)->DScalar,
    derivativeID: ReverseDerivativeID
) : ReverseScalar(primal, derivativeID) {
    override fun backpropagate() {
        a?.pushback(- upstream * f(a.primal))
        b?.pushback(upstream * f(b.primal))
    }
}

/**
 * This is the core implementation of integration, when the bounds are FloatScalars.
 * We atttempt to perform the computation using [Float] values, but if the function
 * returns something other than a [FloatScalar] we fall back to a more general implementation.
 */
private fun rombergIntegralTryPrimitive(a: FloatScalar, b: FloatScalar, f: (DScalar)->DScalar, n: Int, eps: Float): DScalar {
    fun closeEnough(x: Float, y: Float): Boolean {
        val diff = y - x
        return abs(diff) < (eps * (abs(x) + abs(y)) / 2)
    }

    var new: Array<Float> = Array(n+1) { 0F }
    var old: Array<Float> = Array(n+1) { 0F }
    new[0] = 0F
    val fa = f(a)
    if (!(fa is FloatScalar)) return rombergIntegralFloatScalar(a, b, f, n, eps)
    val avalue = a.value
    val fb = f(b)
    if (!(fb is FloatScalar)) return rombergIntegralFloatScalar(a, b, f, n, eps)
    val bvalue = b.value
    new[1] = ((bvalue - avalue) * 0.5F) * (fa.value + fb.value)

    var m: Int = 1
    var t = (bvalue - avalue) / m
    var lastI = 1
    for (i in 2..n) {
        lastI = i
        if (i > 2 && closeEnough(new[i-1], new[i-2])) break
        m *= 2
        val temp = new
        new = old
        old = temp
        t *= 0.5F

        var sum = 0F
        for (j in 1 until m step 2) {
            val fx = f(a + j * t)
            if (!(fx is FloatScalar)) return rombergIntegralFloatScalar(a, b, f, n, eps)
            sum += fx.value
        }

        sum *= t
        new[1] = old[1] * 0.5F + sum

        var k = 1
        for (j in 2..i) {
            k *= 2
            new[j] = (k * new[j-1] - old[j-1]) / (k-1)
        }
    }

    return FloatScalar(new[lastI-1])
}

/**
 * This is the core implementation of integration, when the bounds are FloatScalars.
 * Note that we still have to do the computations using DScalars, as the values returned
 * by [f] might not be FloatScalars.
 */
private fun rombergIntegralFloatScalar(a: FloatScalar, b: FloatScalar, f: (DScalar)->DScalar, n: Int, eps: Float): DScalar {
    fun closeEnough(x: Float, y: Float): Boolean {
        val diff = y - x
        return abs(diff) < (eps * (abs(x) + abs(y)) / 2)
    }

    var new: Array<DScalar> = Array(n+1) { FloatScalar.ZERO }
    var old: Array<DScalar> = Array(n+1) { FloatScalar.ZERO }
    new[0] = FloatScalar.ZERO
    new[1] = ((b - a) * 0.5F) * (f(a) + f(b))

    var m: Int = 1
    var t = (b - a) / m
    var lastI = 1
    for (i in 2..n) {
        lastI = i
        @Suppress("DEPRECATION")
        if (i > 2 && closeEnough(new[i-1].basePrimal().value, new[i-2].basePrimal().value)) break
        m *= 2
        val temp = new
        new = old
        old = temp
        t *= 0.5F

        var sum: DScalar = FloatScalar.ZERO
        for (j in 1 until m step 2) {
            sum += f(a + j * t)
        }

        sum *= t
        new[1] = old[1] * 0.5F + sum

        var k = 1
        for (j in 2..i) {
            k *= 2
            new[j] = (k * new[j-1] - old[j-1]) / (k-1)
        }
    }

    return new[lastI-1]
}
