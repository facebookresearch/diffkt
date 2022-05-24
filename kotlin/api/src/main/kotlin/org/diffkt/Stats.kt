/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Compute the mean and variance of the data.
 */
fun DTensor.stats(): Pair<DScalar, DScalar> {
    val n = this.size.toFloat()
    val sum = this.sum()
    val sum2 = this.pow(2).sum()
    val mean = sum / n
    val variance = sum2 / n - mean.pow(2)
    return Pair(mean, variance)
}

/**
 * Internal utility for timing measurements.
 */
internal class TimingStats(val name: String) {
    private var lastStartTimeNanos: Long = 0L

    private var sum: Double = 0.0
    private var sum2: Double = 0.0
    private var n: Long = 0

    fun start() {
        lastStartTimeNanos = System.nanoTime()
    }

    fun end() {
        val endTimeNanos = System.nanoTime()
        val elapsed = (endTimeNanos - lastStartTimeNanos).toDouble()
        sum += elapsed
        sum2 += elapsed * elapsed
        n++
        report()
    }

    private val shouldReport: Boolean get() {
        return n < 10 ||
                n < 100 && n % 10 == 0L ||
                n < 1000 && n % 100 == 0L ||
                n < 10000 && n % 1000 == 0L ||
                n < 100000 && n % 10000 == 0L ||
                n % 100000 == 0L
    }

    private fun report() {
        if (shouldReport) println(report)
    }

    val report: String get() {
        val mean = sum / n
        val variance = sum2 / n - (mean * mean)
        val stddev = kotlin.math.sqrt(variance)
        return " stats $name: n=$n mean=${mean.toFloat() / 1e6f} ms; stddev=${stddev.toFloat() / 1e6f} ms"
    }
}

/**
 * Internal utility for frequency measurement, for example frequency of cache hits.
 */
internal class FrequencyStats(var name: String) {
    private var successCount: Long = 0
    private var failureCount: Long = 0

    private fun noteSuccess() { successCount++ }
    private fun noteFailure() { failureCount++ }
    fun note(success: Boolean) {
        if (success) noteSuccess() else noteFailure()
        report()
    }
    private val n get() = successCount + failureCount

    private val shouldReport: Boolean get() {
        return n < 10 ||
                n < 100 && n % 10 == 0L ||
                n < 1000 && n % 100 == 0L ||
                n < 10000 && n % 1000 == 0L ||
                n < 100000 && n % 10000 == 0L ||
                n % 100000 == 0L
    }

    private fun report() {
        if (shouldReport) println(report)
    }

    val report: String get() {
        val successRate = 100 * successCount / n.toFloat()
        return " stats $name: ${successRate}%; $successCount hits and  $failureCount misses"
    }
}
