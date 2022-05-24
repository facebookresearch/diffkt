/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.*
import org.diffkt.forward.ForwardScalar
import kotlin.random.Random
import testutils.*

class CustomPullbackTest : AnnotationSpec() {
    @Test
    fun example0a() {
        /**
         * The function without compiler optimizations.
         */
        fun f(x: DScalar, y: DScalar): DScalar {
            return (x + 3F) * (y + 4F)
        }

        val x: DScalar = FloatScalar(10F)
        val y: DScalar = FloatScalar(100F)

        val pads = primalAndReverseDerivativeTransposed(x, y, ::f)
        val primal = pads.first
        val dx = pads.second.first
        val dy = pads.second.second
        primal.value shouldBeExactly 1352F
        dx.value shouldBeExactly 104F
        dy.value shouldBeExactly 13F
    }

    @Test
    fun example0b() {
        /**
         * A compiler-optimized implementation of the function.
         */
        fun f(x: DScalar, y: DScalar): DScalar {
            val did = if (x.derivativeID.sequence > y.derivativeID.sequence) x.derivativeID else y.derivativeID
            if (did is ReverseDerivativeID) {
                val xPrimal = x.primal(did)
                val yPrimal = y.primal(did)
                val t1 = xPrimal + 3F
                val t2 = yPrimal + 4F

                // custom pullback for f
                return object : ReverseScalar(t1 * t2, did) {
                    override fun backpropagate() {
                        val upstream = this.upstream
                        if (x.derivativeID == derivativeID) (x as ReverseScalar).pushback(upstream * t2)
                        if (y.derivativeID == derivativeID) (y as ReverseScalar).pushback(upstream * t1)
                    }
                }
            }
            return (x + 3F) * (y + 4F)
        }

        val x = FloatScalar(10F)
        val y = FloatScalar(100F)

        val pads = primalAndReverseDerivative(x, y, ::f)
        val primal = pads.first
        val dx = pads.second.first
        val dy = pads.second.second
        primal.value shouldBeExactly 1352F
        dx.value shouldBeExactly 104F
        dy.value shouldBeExactly 13F
    }

    /**
     * An example/test based on linReg
     */
    @Test
    fun example1a() {
        val n = 1000

        // We fix the random seed so that we can assert the exact behavior.
        val r = Random(1234567)

        val actualSlope = r.nextFloat()
        val actualIntercept = r.nextFloat()

        val x = FloatArray(n) { r.nextFloat() }
        val y = FloatArray(n) { actualIntercept + actualSlope * x[it] }

        var slopeGuess: DScalar = FloatScalar(r.nextFloat())
        var interceptGuess: DScalar = FloatScalar(r.nextFloat())
        val learnRate = 0.5F

        /**
         * The function to be optimized.
         */
        fun cost(slope: DScalar, intercept: DScalar): DScalar {
            var totalCost: DScalar = FloatScalar(0F)
            for (i in x.indices) {
                val yGuess = x[i] * slope + intercept
                val err = yGuess - y[i]
                totalCost += (err * err)
            }
            return totalCost / n.toFloat()
        }

        for (i in 0 until 200) {
            val pad = primalAndReverseDerivative(slopeGuess, interceptGuess, ::cost)
            val costValue = pad.first
            val slopeGradient = pad.second.first
            val interceptGradient = pad.second.second
            // println(" iteration $i cost ${costValue.value}")
            slopeGuess -= slopeGradient * learnRate
            interceptGuess -= interceptGradient * learnRate
            // The particular test, given the random seed above, converges.
            if (i > 100) costValue.value shouldBeLessThan 1E-7F
            if (i >= 170) costValue.value shouldBeLessThan 1E-11F
        }
    }

    /* LinReg with manual differentiation */
    @Test
    fun example1a_manual() {
        val n = 1000
        // We fix the random seed so that we can assert the exact behavior.
        val r = Random(1234567)

        val actualSlope = r.nextFloat()
        val actualIntercept = r.nextFloat()

        val x = FloatArray(n) { r.nextFloat() }
        val y = FloatArray(n) { actualIntercept + actualSlope * x[it] }

        var slopeGuess: Float = r.nextFloat()
        var interceptGuess: Float = r.nextFloat()

        val learnRate = 0.5F

        fun cost(slope: Float, intercept: Float): Float {
            var totalCost: Float = 0F
            for (i in x.indices) {
                val yGuess = x[i] * slope + intercept
                val err = yGuess - y[i]
                totalCost += (err * err)
            }
            return totalCost/n.toFloat()
        }

        fun cost_grad(slope: Float, intercept: Float): Pair<Float, Float>{
            var ds = 0F
            var di = 0F
            for (i in x.indices){
                val temp = 2F*(x[i]*slope+intercept-y[i])
                ds += temp*x[i]
                di += temp
            }
            return Pair(ds/n.toFloat(), di/n.toFloat())
        }

        for (i in 0 until 200) {
            val costValue = cost(slopeGuess, interceptGuess)
            val gradients = cost_grad(slopeGuess, interceptGuess)
            val slopeGradient = gradients.first
            val interceptGradient = gradients.second
            //println(" iteration $i cost ${costValue}")
            slopeGuess -= slopeGradient * learnRate
            interceptGuess -= interceptGradient * learnRate
            if (i > 100) costValue shouldBeLessThan 1E-5F
            // The particular test, given the random seed above, converges.
            if (i >= 194) costValue shouldBeExactly 1.2573054E-14F
        }
    }

    /**
     * General optimization of LinReg
     */
    @Test
    fun example1b() {
        val n = 1000
        // We fix the random seed so that we can assert the exact behavior.
        val r = Random(1234567)

        val actualSlope = r.nextFloat()
        val actualIntercept = r.nextFloat()

        val x = FloatArray(n) { r.nextFloat() }
        val y = FloatArray(n) { actualIntercept + actualSlope * x[it] }

        var slopeGuess: DScalar = FloatScalar(r.nextFloat())
        var interceptGuess: DScalar = FloatScalar(r.nextFloat())
        val learnRate = 0.5F


        /**
         * The function to be optimized.
         */
        fun cost_original(slope: DScalar, intercept: DScalar): DScalar {
            var totalCost: DScalar = FloatScalar(0F)
            for (i in x.indices) {
                val yGuess = x[i] * slope + intercept
                val err = yGuess - y[i]
                totalCost += (err * err)
            }
            return totalCost/n.toFloat()
        }

        fun cost_DFloat(slope: Float, intercept: Float): Float {
            var totalCost: Float = 0F
            for (i in x.indices) {
                val yGuess = x[i] * slope + intercept
                val err = yGuess - y[i]
                totalCost += (err * err)
            }
            return totalCost/n.toFloat()
        }

        fun cost_grad_Float(slope: Float, intercept: Float): Pair<Float, Float>{
            var ds = 0F
            var di = 0F
            for (i in x.indices){
                val temp = 2F*(x[i]*slope+intercept-y[i])
                ds += temp*x[i]
                di += temp
            }
            return Pair(ds/n.toFloat(),di/n.toFloat())
        }

        fun cost_grad_original(slope: DScalar, intercept: DScalar): Pair<DScalar, DScalar>{
            var ds: DScalar = FloatScalar.ZERO
            var di: DScalar = FloatScalar.ZERO
            for (i in x.indices){
                val temp = 2F*(x[i]*slope+intercept-y[i])
                ds += temp*x[i]
                di += temp
            }
            return Pair(ds/n.toFloat(),di/n.toFloat())
        }

        fun cost_grad(slope: DScalar, intercept: DScalar): Pair<DScalar, DScalar>{
            return when {
                (slope.derivativeID == intercept.derivativeID) -> {
                    when (slope){
                        is FloatScalar -> {
                            val rst = cost_grad_Float(slope.value, intercept.value)
                            Pair(FloatScalar(rst.first), FloatScalar(rst.second))
                        }
                        else -> cost_grad_original(slope, intercept)
                    }
                }
                else -> cost_grad_original(slope, intercept)
            }
        }

        fun cost(slope: DScalar, intercept: DScalar): DScalar {
            // must be a function, not a class or causes uninitialized this error
            fun costReverseScalar(
                slope0: ReverseScalar?,
                intercept0: ReverseScalar?,
                slopeValue: DScalar,
                interceptValue: DScalar,
                derivativeID: ReverseDerivativeID
            ): ReverseScalar {
                return object : ReverseScalar(cost(slopeValue, interceptValue), derivativeID) {
                    override fun backpropagate() {
                        val grads = cost_grad(slopeValue, interceptValue)
                        slope0?.pushback(upstream * grads.first)
                        intercept0?.pushback(upstream * grads.second)
                    }
                }
            }

            return when {
                slope.derivativeID == intercept.derivativeID -> {
                    when (slope) {
                        is FloatScalar -> FloatScalar(cost_DFloat(slope.value, intercept.value))
                        is ForwardScalar -> cost_original(slope, intercept)
                        is ReverseScalar -> {
                            intercept as ReverseScalar
                            costReverseScalar(slope, intercept, slope.primal, intercept.primal, slope.derivativeID)
                        }
                        else -> throw IllegalArgumentException()
                    }
                }
                slope.derivativeID.sequence > intercept.derivativeID.sequence -> {
                    when (slope) {
                        is ForwardScalar -> cost_original(slope, intercept)
                        is ReverseScalar -> {
                            costReverseScalar(slope, null, slope.primal, intercept, slope.derivativeID)
                        }
                        else -> throw IllegalArgumentException()
                    }
                }
                else -> {
                    when (intercept) {
                        is ForwardScalar -> cost_original(slope, intercept)
                        is ReverseScalar -> {
                            costReverseScalar(null, intercept, slope, intercept.primal, intercept.derivativeID)
                        }
                        else -> throw IllegalArgumentException()
                    }
                }
            }
        }

        for (i in 0 until 200) {
            val pad = primalAndReverseDerivative(slopeGuess, interceptGuess, ::cost)
            val costValue = pad.first
            val slopeGradient = pad.second.first
            val interceptGradient = pad.second.second
            // println(" iteration $i cost ${costValue.value}")
            slopeGuess -= slopeGradient * learnRate
            interceptGuess -= interceptGradient * learnRate
            if (i > 100) costValue.value shouldBeLessThan 1E-5F
            // The particular test, given the random seed above, converges.
            if (i >= 194) costValue.value shouldBeExactly 1.2573054E-14F
        }
    }
}
