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
import kotlin.math.nextDown

class RandomKeyDistributionsTest : AnnotationSpec() {
    @Test fun `gaussian test`() {
        val n = 100000
        val r = RandomKey()
        val gaussian = r.gaussian(Shape(n))

        // boxMuller method generates a standard normal distribution with the mean of 0 and variance of 1
        val expectedMean = 0f
        val expectedVariance = 1f
        stats(gaussian) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `gaussian test with mean 2 and std 3`() {
        val n = 100000
        val r = RandomKey()
        val gaussian = r.gaussian(Shape(n), FloatScalar(2f), FloatScalar(3f))

        val expectedMean = 2f
        // variance is the square of the standard deviation
        val expectedVariance = 9f
        stats(gaussian) shouldBeCloseTo Pair(expectedMean, expectedVariance)

    }

    @Test fun `gaussian test with mean -5 and std 0,5`() {
        val n = 100000
        val r = RandomKey()
        val gaussian = r.gaussian(Shape(n), FloatScalar(-5f), FloatScalar(0.5f))

        val expectedMean = -5f
        // variance is the square of the standard deviation
        val expectedVariance = 0.25f
        stats(gaussian) shouldBeCloseTo Pair(expectedMean, expectedVariance)

    }

    @Test fun `cauchy with loc 0 and scale 1`() {
        val n = 100000
        val r = RandomKey(1411587246503371L)
        val shape = Shape(n)

        val loc = FloatTensor.const(0f, shape)
        val scale = FloatTensor.const(1f, shape)
        val cauchy = r.cauchy(shape, loc, scale)

        // the mean and variance of a cauchy distribution is undefined
        // the output of the cdf should be a uniform distribution
        val uniform = atan((cauchy - loc) / scale) * (1f/ FloatScalar.PI) + 0.5f

        val numBuckets = 10
        val t = (uniform * numBuckets.toFloat()) as FloatTensor
        val buckets = IntArray(numBuckets)
        for (i in 0 until t.size) {
            val bucket = ((t[i] as FloatScalar).value * 1f.nextDown()).toInt()
            buckets[bucket]++
        }
        val expected = n.toFloat() / numBuckets
        for (element in buckets) {
            element.toFloat() shouldBeCloseTo expected
        }
    }

    @Test fun `cauchy with loc 10 and scale 5`() {
        val n = 100000
        val r = RandomKey(1411587246503371L)
        val shape = Shape(n)

        val loc = FloatTensor.const(10f, shape)
        val scale = FloatTensor.const(5f, shape)
        val cauchy = r.cauchy(shape, loc, scale)

        val uniform = atan((cauchy - loc) / scale) * (1f/ FloatScalar.PI) + 0.5f

        val numBuckets = 10
        val t = (uniform * numBuckets.toFloat()) as FloatTensor
        val buckets = IntArray(numBuckets)
        for (i in 0 until t.size) {
            val bucket = ((t[i] as FloatScalar).value * 1f.nextDown()).toInt()
            buckets[bucket]++
        }
        val expected = n.toFloat() / numBuckets
        for (element in buckets) {
            element.toFloat() shouldBeCloseTo expected
        }
    }

    @Test fun `chi square test with dof 1`() {
        val n = 100000
        val r = RandomKey()

        val dof = FloatScalar.ONE
        val chiSquare = r.chiSquare(Shape(n), dof)

        // chi square has the mean of dof and variance of 2 * dof
        val expectedMean = 1f
        val expectedVariance = 2f
        stats(chiSquare) shouldBeCloseTo Pair(expectedMean, expectedVariance)

    }

    @Test fun `chi square test with dof 10`() {
        val n = 100000
        val r = RandomKey()

        val dof = FloatScalar(10f)
        val chiSquare = r.chiSquare(Shape(n), dof)

        // chi square has the mean of dof and variance of 2 * dof
        val expectedMean = 10f
        val expectedVariance = 20f
        stats(chiSquare) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `chi square test with dof 100`() {
        val n = 100000
        val r = RandomKey()

        val dof =  FloatScalar(100f)
        val chiSquare = r.chiSquare(Shape(n), dof)

        // chi square has the mean of dof and variance of 2 * dof
        val expectedMean = 100f
        val expectedVariance = 200f
        stats(chiSquare) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }
}
