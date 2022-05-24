/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.*

class GammaTest: AnnotationSpec() {
    @Test fun `gamma with alpha = 4`() {
        val n = 100000
        val shape = Shape(n)
        val r = RandomKey()

        val alpha = FloatTensor.const(4f, shape)

        val gamma = r.gamma(shape, alpha)
        gamma.shape shouldBe shape

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 4f
        val expectedVariance = 4f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gammaWithRate with alpha = 4 and beta = 2`() {
        val n = 100000
        val shape = Shape(n)
        val r = RandomKey()

        val alpha = FloatTensor.const(4f, shape)
        val beta = FloatTensor.const(2f, shape)

        val gamma = r.gammaWithRate(shape, alpha, beta)
        gamma.shape shouldBe shape

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 2f
        val expectedVariance = 1f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gamma with alpha 0,5`() {
        val n = 100000
        val shape = Shape(n)
        val r = RandomKey()

        val alpha = FloatTensor.const(0.5f, shape)

        val gamma = r.gamma(shape, alpha)
        gamma.shape shouldBe shape

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 0.5f
        val expectedVariance = 0.5f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gamma with FloatScalars and Shape(n)`() {
        val n = 100000
        val shape = Shape(n)
        val r = RandomKey()

        val alpha =  FloatScalar(0.5f)
        val beta = FloatScalar(2f)

        val gamma = r.gammaWithRate(shape, alpha, beta)
        gamma.shape shouldBe shape

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 0.25f
        val expectedVariance = 0.125f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gamma with FloatScalars and Shape()`() {
        val n = 100000

        val alpha =  FloatScalar(10f)

        val distribution: MutableList<FloatScalar> = mutableListOf()
        val keys = RandomKey().split(n)
        for (i in 0 until n) {
            val r = keys[i]
            distribution.add(r.gamma(Shape(), alpha) as FloatScalar)
        }

        val gamma = tensorOf(distribution).reshape(Shape(n))

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 10f
        val expectedVariance = 10f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gammaWithScale with alpha = 4 and beta = 0,5`() {
        val n = 100000
        val shape = Shape(n)
        val r = RandomKey()

        val alpha = FloatTensor.const(4f, shape)
        val beta = FloatTensor.const(0.5f, shape)

        val gamma = r.gammaWithScale(shape, alpha, beta)
        gamma.shape shouldBe shape

        // Marsaglia and Tsang's Method generates a gamma distribution with the mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
        val expectedMean = 2f
        val expectedVariance = 1f
        stats(gamma) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }
}