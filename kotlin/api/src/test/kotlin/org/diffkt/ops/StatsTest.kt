/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.*
import kotlin.math.abs
import kotlin.math.pow
import kotlin.random.Random
import testutils.*

class StatsTest : AnnotationSpec() {
    @Test
    fun `test DTensor stats`() {
        fun testStatsWithSeed(seed: Int) {
            val random = Random(seed)
            val mean = random.nextDouble(from = -100.0, until = 100.0).toFloat()
            val stdev = random.nextDouble(from = 1.0, until = 100.0).toFloat()
            val variance = stdev.pow(2)
            val x = FloatTensor(Shape(1000)) { nextRandom(random, mean, stdev) }
            val (computedMean, computedVariance) = x.stats()
            abs((computedMean - mean).value / mean) shouldBeLessThan 0.1F
            abs((computedVariance - variance).value / variance) shouldBeLessThan 0.1F
        }
        for (i in 0..3) testStatsWithSeed(12345 + i)
    }
}

/**
 * The standard deviation of random.nextFloat()
 */
private val stddevUniform = 1 / kotlin.math.sqrt(12F)

/**
 * Produce a random float with the given mean and standard deviation.
 * Any distribution would do, but here we implement a uniform distribution.
 */
fun nextRandom(random: Random, mean: Float, stddev: Float): Float {
    val x = (random.nextFloat() - 0.5F) / stddevUniform // mean 0, stddev 1
    return (x * stddev) + mean
}
