/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.*
import org.diffkt.ops.nextRandom
import testutils.*
import testutils.floats
import kotlin.math.*
import kotlin.random.Random

class BatchNormTest : AnnotationSpec() {
    @Test
    fun `test the distribution of the output of batchNorm`() {
        fun testBatchNormWithSeed(seed: Int) {
            val random = Random(seed)
            val inputMean = random.nextDouble(from = -100.0, until = 100.0).toFloat()
            val inputStdev = random.nextDouble(from = 1.0, until = 100.0).toFloat()
            val bn = BatchNormTraining(1)

            // Since we are not training the scaling part of this BN node,
            // it should just normalize the input without scaling.
            val mean = 0F
            val stddev = 1F
            val variance = stddev.pow(2)

            // let the batch normalization gather data for a while.  It does this by computing running stats of the input
            for (i in 0..100) {
                val input = FloatTensor(Shape(1000, 1)) { nextRandom(random, inputMean, inputStdev) }
                val x = bn.invoke(input)
                val (computedMean, computedVariance) = x.stats()
                abs((computedMean - mean).value) shouldBeLessThan 0.1F
                abs((computedVariance - variance).value / variance) shouldBeLessThan 0.1F
            }

            // freeze the BN node and test its behavior
            val frozen: AffineTransform = bn.inferenceMode
            val input = FloatTensor(Shape(1000, 1)) { nextRandom(random, inputMean, inputStdev) }
            val x = frozen.invoke(input)
            val (computedMean, computedVariance) = x.stats()
            abs((computedMean - mean).value) shouldBeLessThan 0.1F
            abs((computedVariance - variance).value / variance) shouldBeLessThan 0.1F
        }
        for (i in 0..3) testBatchNormWithSeed(12345 + i)
    }

    @Test
    fun `2x2 image with three channels using V1`() {
        val values = BatchNormTestValues.ThreeChannels
        val epsilon = values.epsilon
        val x = values.x
        val channels = x.shape[3]
        val scaleShift = values.scaleShift

        val (runningMean, runningVariance, momentum) = noopRunningStats(channels)
        val (res, mean, variance) = batchNormTrainV1(x, scaleShift, runningMean, runningVariance, momentum)

        for (i in 0 until 3) {
            val chRes = res.slice(i, i + 1, 3)
            chRes.shouldBeNear(values.expectedCh, epsilon)
        }
        mean.shouldBeNear(values.expectedMeans, epsilon)
        variance.shouldBeNear(values.expectedVariances, epsilon)
    }

    @Test
    fun `gradients for 2x2 image using V1`() {
        fun f(x: DTensor, scaleShift: DTensor): DTensor {
            val channels = x.shape[3]
            val (runningMean, runningVariance, momentum) = noopRunningStats(channels)
            val (res, _, _) = batchNormTrainV1(x, scaleShift, runningMean, runningVariance, momentum)
            return res
        }

        val values = BatchNormTestValues.ImageGradients
        val epsilon = values.epsilon
        val x = values.x
        val scaleShift = values.scaleShift

        val xGrad = vjp(x, values.seed) { xx -> f(xx, scaleShift) }
        xGrad.shouldBeNear(values.expectedXGrad, epsilon)

        val scaleShiftGrad = vjp(scaleShift, values.seed) { scaleShiftD -> f(x, scaleShiftD) }
        scaleShiftGrad.shouldBeNear(values.expectedScaleShiftGrad, epsilon)
    }

    @Test
    fun `gradients for 2x2 image`() {
        fun f(x: DTensor, scaleShift: DTensor): DTensor {
            val channels = x.shape[3]
            val (runningMean, _, momentum) = noopRunningStats(channels)
            val (res, _) = batchNormTrainV2(x, scaleShift, 0F, runningSum = runningMean, runningSumOfSquares = runningMean, momentum)
            return res
        }

        val values = BatchNormTestValues.ImageGradients
        val epsilon = values.epsilon
        val x = values.x
        val scaleShift = values.scaleShift

        val xGrad = vjp(x, values.seed) { xx -> f(xx, scaleShift) }
        xGrad.shouldBeNear(values.expectedXGrad, epsilon)

        val scaleShiftGrad = vjp(scaleShift, values.seed) { scaleShiftD -> f(x, scaleShiftD) }
        scaleShiftGrad.shouldBeNear(values.expectedScaleShiftGrad, epsilon)
    }

    @Test
    fun `scale and shift`() {
        val values = BatchNormTestValues.ScaleAndShift
        val epsilon = values.epsilon
        val x = values.x
        val channels = x.shape[3]
        val (runningMean, runningVariance, momentum) = noopRunningStats(channels)

        val (unscaledRes, unscaledMean, unscaledVariance) = batchNormTrainV1(x, values.noopScaleShift, runningMean, runningVariance, momentum)
        val scaleTwoShiftTen = FloatTensor(Shape(2, 3), floatArrayOf(2f, 2f, 2f, 10f, 10f, 10f))
        val (res, mean, variance) = batchNormTrainV1(x, scaleTwoShiftTen, runningMean, runningVariance, momentum)

        res.shouldBeNear(unscaledRes * FloatScalar(2f) + FloatScalar(10f), epsilon)
        mean.shouldBeNear(unscaledMean, epsilon)
        variance.shouldBeNear(unscaledVariance, epsilon)
    }

    @Test
    fun `variance for normalization is 0`() {
        val values = BatchNormTestValues.ZeroVariance
        val epsilon = values.epsilon
        val x = FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(-1f, 0f, 1f, 2f))
        val mean = FloatTensor.zeros(Shape(1))
        val variance = FloatTensor.zeros(Shape(1))
        val scaleShift = values.scaleShift

        val res = batchNormInfer(x, scaleShift, mean, variance)
        res.shouldBeNear(values.expectedRes, epsilon)
    }
}

fun batchNormInfer(input: DTensor, scaleShift: DTensor, mean: FloatTensor, variance: FloatTensor): DTensor {
    return freezeBatchNorm(scaleShift, mean, variance).invoke(input)
}

/**
 * Returns a Triple of runningMean, runningVariance, and momentum that
 * when passed to batchNormTrain will just return the mean and variance
 * over the sample input.
 */
fun noopRunningStats(
    channels: Int,
): Triple<FloatTensor, FloatTensor, Float> {
    // Since runningMean and runningVariance are ignored, on account of
    // momentum being 1, their values don't matter
    return Triple(
        FloatTensor.zeros(Shape(channels)),
        FloatTensor.zeros(Shape(channels)),
        1f
    )
}

object BatchNormTestValues {

    private fun noopScaleShift(channels: Int): FloatTensor {
        return FloatTensor.ones(Shape(1, channels)).concat(FloatTensor.zeros(Shape(1, channels)))
    }

    object ThreeChannels {
        const val epsilon = 4e-5f
        val x = FloatTensor(
            Shape(1, 2, 2, 3),
            floatArrayOf(
                1f, 10f, 2f,
                2f, 11f, 4f,
                3f, 12f, 6f,
                4f, 13f, 8f
            )
        )
        val scaleShift = noopScaleShift(3)
        val expectedCh = FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(-1.3416355f, -0.44721183f, 0.44721183f, 1.3416395f))
        val expectedMeans = tensorOf(2.5f, 11.5f, 5.0f)
        val expectedVariances = tensorOf(1.6667f, 1.6667f, 6.6667f)
    }

    object ImageGradients {
        const val epsilon = 4e-5f
        private val inputShape = Shape(1, 2, 2, 1)
        val x get() = FloatTensor(inputShape, floatArrayOf(1f, 2f, 3f, 10f))
        val scaleShift = noopScaleShift(1)
        val seed = FloatTensor(inputShape, floats(4))
        val expectedXGrad = FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(-0.1867f, 0.0170f, 0.2206f, -0.0509f))
        val expectedScaleShiftGrad = FloatTensor(Shape(2, 1), floatArrayOf(3.9598f, 10f))
    }

    object ScaleAndShift {
        val epsilon = 1e-5f
        val x = FloatTensor.random(Random(123), Shape(1, 2, 2, 3))
        val noopScaleShift = noopScaleShift(3)
        val scaleTwoShiftTen = FloatTensor(Shape(2, 3), floatArrayOf(2f, 2f, 2f, 10f, 10f, 10f))
    }

    object ZeroVariance {
        val epsilon = 1e-5f
        val scaleShift = noopScaleShift(1)
        val expectedRes = FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(-316.22778f, 0.0f, 316.22778f, 632.45557f))
    }
}
