/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldNotBe
import org.diffkt.*
import testutils.*
import kotlin.math.nextDown
import kotlin.math.pow

class DiffktRandomTest : AnnotationSpec() {
    @Test fun `test that keys are not being reused`() {
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val gaussian = random.nextGaussian(Shape())
        val newKey = random.getRandomKey()
        newKey shouldNotBe randomKey

        val cauchy = random.nextCauchy(Shape())
        random.getRandomKey() shouldNotBe newKey
    }

    @Test fun `uniform distributionw with high and low`() {
        val n = 100000
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val lowValue = 10f
        val highValue = 20f
        val sample = random.nextUniform(Shape(n), FloatScalar(lowValue), FloatScalar(highValue))

        val expectedMean = (lowValue + highValue) / 2
        val expectedVariance = (highValue - lowValue).pow(2) / 12
        stats(sample) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `test that the random tensors are different`() {
        val num = 5
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val gaussian1 = random.nextGaussian(Shape(num))
        val gaussian2 = random.nextGaussian(Shape(num))

        for (i in 0 until num) {
            (gaussian1[i] as FloatScalar).value shouldNotBe (gaussian2[i] as FloatScalar).value
        }
    }

    @Test fun `test that getUniformFloatScalar works`() {
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val float1 = random.nextUniform()
        val float2 = random.nextUniform()
        val float3 = random.nextUniform(Shape()) as FloatScalar

        float1.value shouldNotBe float2.value
        float2.value shouldNotBe float3.value

        val n = 100000
        val distribution: MutableList<DScalar> = mutableListOf()
        for (i in 0 until n) {
            distribution.add(random.nextUniform())
        }

        val uniform = tensorOf(distribution).reshape(Shape(n))
        val expectedMean = 0.5f
        val expectedVariance = 0.083f
        stats(uniform) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `test that getGaussianFloatScalar works`() {
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val float1 = random.nextGaussian()
        val float2 = random.nextGaussian()
        val float3 = random.nextGaussian(Shape()) as FloatScalar

        float1.value shouldNotBe float2.value
        float2.value shouldNotBe float3.value

        val n = 100000
        val distribution: MutableList<DScalar> = mutableListOf()
        for (i in 0 until n) {
            distribution.add(random.nextGaussian())
        }

        val gaussian = tensorOf(distribution).reshape(Shape(n))
        val expectedMean = 0f
        val expectedVariance = 1f
        stats(gaussian) shouldBeCloseTo Pair(expectedMean, expectedVariance)
    }

    @Test fun `test that getCauchyFloatScalar works`() {
        val randomKey = RandomKey()
        val random = DiffktRandom(randomKey)

        val float1 = random.nextCauchy()
        val float2 = random.nextCauchy()
        val float3 = random.nextCauchy(Shape()) as FloatScalar

        float1.value shouldNotBe float2.value
        float2.value shouldNotBe float3.value

        val n = 100000
        val distribution: MutableList<DScalar> = mutableListOf()
        for (i in 0 until n) {
            distribution.add(random.nextCauchy())
        }

        val cauchy = tensorOf(distribution).reshape(Shape(n))

        val uniform = atan(cauchy) * (1f/ FloatScalar.PI) + 0.5f

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
}
