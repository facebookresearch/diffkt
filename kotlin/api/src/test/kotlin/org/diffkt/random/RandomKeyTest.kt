/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeGreaterThan
import io.kotest.matchers.floats.shouldBeLessThan
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import org.diffkt.*
import kotlin.math.PI
import testutils.*

class RandomKeyTest : AnnotationSpec() {
    @Test fun `random numbers appear well distributed`() {
        val n = 100000
        val splits = 10
        for (r in RandomKey().split(splits)) {
            val avg = r.floats(n).sum() / n
            avg.value shouldBeGreaterThan 0.495f
            avg.value shouldBeLessThan 0.505f
        }
    }

    @Test fun `random generator is idempotent`() {
        val r = RandomKey().permitReuse()
        val n = 3
        val t1 = r.floats(n)
        r.split()
        r.floats(23)
        val t2 = r.floats(n)
        t1 shouldBe t2
    }

    @Test fun `random split produces distinct generators`() {
        val n = 3
        val r = RandomKey().permitReuse()
        val (r1, r2) = r.split(n).map { it.permitReuse() }
        r.floats(n) shouldNotBe r2.floats(n)
        r1.floats(n) shouldNotBe r.floats(n)
        r1.floats(n) shouldNotBe r2.floats(n)
    }

    @Test fun `random split is idempotent`() {
        val n = 3
        val r = RandomKey().permitReuse()
        val (r1, r2) = r.split(n).map { it.permitReuse() }
        r.floats(7)
        val (r3, r4) = r.split(n).map { it.permitReuse() }
        r1 shouldBe r3
        r1.hashCode() shouldBe r3.hashCode()
        r2 shouldBe r4
        r2.hashCode() shouldBe r4.hashCode()
        r1.floats(n) shouldBe r3.floats(n)
        r2.floats(n) shouldBe r4.floats(n)
        r1.floats(n) shouldNotBe r2.floats(n)
    }

    @Test fun `equality is value-based`() {
        val k1 = RandomKey(101).permitReuse()
        val k2 = RandomKey(101).permitReuse()
        val k3 = RandomKey(102)
        k1 shouldBe k1
        k1 shouldBe k2
        k1.hashCode() shouldBe k2.hashCode()
        k1 shouldNotBe k3
        k1.split(3)[1] shouldBe k2.split(3)[1]
        k1.split(3)[1].floats(3) shouldBe k2.split(3)[1].floats(3)
        k1.split(3)[1].floats(3) shouldNotBe k3.split(3)[1].floats(3)
    }

    /**
     * Tests that the nanosecond timer, which we use to select time-of-day-based seeds,
     * changes between construction of [RandomKey] instances.  This might not be guaranteed
     * on every platform, so if this test fails or is flaky, please feel free to disable
     * or delete it.
     */
    @Test fun `the timer is fast`() {
        val r1 = RandomKey.fromTimeOfDay()
        val r2 = RandomKey.fromTimeOfDay()
        r1 shouldNotBe r2
    }

    /**
     * Tests that the default constructor for [RandomKey] produces the same result every time.
     */
    @Test fun `The no-arg constructor is deterministic`() {
        val k1 = RandomKey()
        val k2 = RandomKey()
        k1 shouldBe k2
    }

    @Test fun `test that the generated numbers are nicely distributed`() {
        val nBuckets = 100
        val nSamples = 1000000
        val t = (RandomKey().floats(nSamples) * nBuckets.toFloat()) as FloatTensor
        val buckets = IntArray(nBuckets)
        for (i in 0 until t.size) {
            val bucket = (t[i] as FloatScalar).value.toInt()
            buckets[bucket]++
        }
        val expected = nSamples.toFloat() / nBuckets
        for (count in buckets) {
            count.toFloat() shouldBeCloseTo expected
        }
    }

    @Test fun `test that repeated splits produce distinct keys`() {
        val n1 = 10
        val n2 = 10
        // test that splits produce distinct keys
        val r0 = RandomKey()
        val s = HashSet<RandomKey>()
        for (r1 in r0.split(n1)) {
            for (r2 in r1.split(n2)) {
                s.add(r2)
            }
        }

        s.size shouldBe n1 * n2
    }

    @Test fun `test splitting across block boundaries produces distinct keys`() {
        val n = 210
        val m = 32 // 5 bits to represent m-1. fills up 1050 bits > 16 longs.
        var r = RandomKey()
        val indexRandom = ((RandomKey().floats(n) * m.toFloat()) as FloatTensor).asStrided().data
        val s = HashSet<RandomKey>()
        for (i in 0 until n) {
            val rs = r.split(m)
            s.addAll(rs)
            val index = indexRandom[i].toInt()
            r = rs[index]
        }
        s.size shouldBe n * m
    }

    @Test fun `test that splits consume the expected number of bits of the unconsumed array`() {
        val bitsPerLong = Long.SIZE_BITS
        val blockSize = bitsPerLong * Sha512Random.unconsumedSize
        for (bitsPerSplit in 0 until 12) {
            val splitSize = 1 shl bitsPerSplit
            val numSplits = 3 + 2 * blockSize * bitsPerSplit
            var r = RandomKey().permitReuse() as Sha512Random
            val indices = r.floats(Shape(numSplits)).normalize().data.map { (it * splitSize).toInt() }
            var endOfInputMarkerIndex = bitsPerLong
            for (i in 0 until numSplits) {
                // check that the end-of-input marker is where expected.
                val endOfInputMarkerWord = endOfInputMarkerIndex / bitsPerLong
                endOfInputMarkerWord shouldBe r.unconsumed.size - 1
                val endOfInputMarkerBit = bitsPerLong - endOfInputMarkerIndex % bitsPerLong - 1
                val endOfInputMarkerMask = 1L shl endOfInputMarkerBit
                val lastWord = r.unconsumed[endOfInputMarkerWord]
                lastWord and endOfInputMarkerMask shouldBe endOfInputMarkerMask // end of input marker should be present
                lastWord and (endOfInputMarkerMask - 1L) shouldBe 0L            // followed by zeroes

                // prepare for next split
                r = r.split(splitSize)[indices[i]]
                endOfInputMarkerIndex += bitsPerSplit
                if (endOfInputMarkerIndex >= blockSize) // when we exhaust a block we put all the data into a fresh block
                    endOfInputMarkerIndex = bitsPerSplit
            }
        }
    }

    @Test fun `test infinite gaussian stream`() {
        val r = RandomKey().permitReuse()
        val floats = r.floats(2)
        val u1 = (floats[0] as FloatScalar).value
        val u2 = (floats[1] as FloatScalar).value
        val epsilon = 1f / UInt.MAX_VALUE.toFloat()

        val gaussianTensor = (r as Sha512Random).gaussian(Shape(3))
        val gaussian1 = (gaussianTensor[0] as FloatScalar).value
        val gaussian2 = (gaussianTensor[1] as FloatScalar).value
        val gaussian3 = (gaussianTensor[2] as FloatScalar).value

        // Testing that the gaussian values are not the same
        gaussian1 shouldNotBe gaussian2
        gaussian2 shouldNotBe gaussian3
        // Testing that storing the extra gaussian value works properly
        gaussian1 shouldBe kotlin.math.sqrt(-2 * kotlin.math.ln(u1 + epsilon)) * kotlin.math.cos(2 * PI.toFloat() * u2)
        gaussian2 shouldBe kotlin.math.sqrt(-2 * kotlin.math.ln(u1 + epsilon)) * kotlin.math.sin(2 * PI.toFloat() * u2)
    }

    @Test fun `test protection from reuse of RandomKey instances`() {
        fun doOp(key: RandomKey, op: Int) {
            when (op) {
                1 -> key.permitReuse()
                2 -> key.split(2)
                3 -> key.floats(2)
                4 -> key.gaussian(Shape(2))
                5 -> key.gamma(FloatScalar.PI)
                else -> throw IllegalStateException()
            }
        }

        for (i in 1 .. 5) {
            for (j in 1 .. 5) {
                val key = RandomKey()
                doOp(key, i)
                shouldThrow<java.lang.IllegalStateException> { doOp(key, j) }
            }
        }

        for (i in 1 .. 5) {
            for (j in 1 .. 5) {
                val key = RandomKey().permitReuse()
                doOp(key, i)
                doOp(key, j)
            }
        }
    }
}
