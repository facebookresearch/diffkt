/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import org.diffkt.*
import kotlin.math.*

/**
 * A random number generator based on the SHA-512 specification (https://en.wikipedia.org/wiki/SHA-2)
 * adapted according to the construction in the paper "Splittable pseudorandom number
 * generators using cryptographic hashing" by Claessen and Palka
 * (https://publications.lib.chalmers.se/records/fulltext/183348/local_183348.pdf)
 * and “Parallel Random Numbers: As Easy as 1, 2, 3” by Salmon, Moraes, Dror, and Shaw
 * (https://www.cct.lsu.edu/~sjha/share/SC11/src/pdf/papers/tp15.pdf).
 */
internal class Sha512Random(
    // The state of the hash value being computed
    val hashState: LongArray,
    // The unconsumed input that has yet to be merged into the hash value
    val unconsumed: LongArray,
    val permitsReuse: Boolean = false
) : RandomKey {
    var hasBeenUsed = false

    constructor(a: Long) : this(
        initialState,
        longArrayOf(a, Long.MIN_VALUE)
        )

    override fun split(n: Int): List<Sha512Random> {
        if (hasBeenUsed && !permitsReuse)
            throw IllegalStateException("RandomKey instance has already been used.")
        hasBeenUsed = true
        // No need to split if n==1
        if (n == 1) return listOf(this)
        if (n == 0) return listOf()
        if (n < 0) throw IllegalArgumentException("n must be >= 0")

        // fixed message length includes the end of input marker
        val messageLength = Long.SIZE_BITS - (n - 1).toLong().countLeadingZeroBits() + 1
        var finalBitPosition = unconsumed.last().countTrailingZeroBits() + 1

        var overflow = messageLength > finalBitPosition
        val unconsumedFull = unconsumed.size == unconsumedSize && overflow

        // if block is full, finish it before constructing new states
        val state = if (unconsumedFull) finish(hashState, unconsumed) else hashState
        val oldUnconsumed = if (unconsumedFull) emptyUnconsumedInputArray else unconsumed
        if (unconsumedFull) {
            finalBitPosition = Long.SIZE_BITS
            overflow = false
        }

        val difference = abs(finalBitPosition - messageLength)
        val unsetFinalBitMask = 1L.shl(finalBitPosition-1).inv()

        return LazyList<Sha512Random>(n) { i: Int ->
            val message = i.toLong().shl(1) + 1
            val newUnconsumed = oldUnconsumed.copyOf(if (overflow) oldUnconsumed.size + 1 else oldUnconsumed.size)
            if (overflow) {
                newUnconsumed[newUnconsumed.lastIndex - 1] = (newUnconsumed[newUnconsumed.lastIndex - 1] and unsetFinalBitMask) or message.ushr(difference)
                newUnconsumed[newUnconsumed.lastIndex] = newUnconsumed[newUnconsumed.lastIndex] or message.shl(Long.SIZE_BITS - difference)
            } else {
                newUnconsumed[newUnconsumed.lastIndex] = (newUnconsumed[newUnconsumed.lastIndex] and unsetFinalBitMask) or message.shl(difference)
            }
            Sha512Random(state, newUnconsumed)
        }
    }

    override fun floats(shape: Shape): FloatTensor {
        if (hasBeenUsed && !permitsReuse)
            throw IllegalStateException("RandomKey instance has already been used.")
        hasBeenUsed = true
        val floats = Floats(this)
        val data = FloatArray(shape.product) { floats.nextFloat() }
        return FloatTensor(shape, data)
    }

    override fun permitReuse(): Sha512Random {
        if (hasBeenUsed && !permitsReuse)
            throw IllegalStateException("RandomKey instance has already been used.")
        hasBeenUsed = true
        return Sha512Random(hashState, unconsumed, permitsReuse = true)
    }

    override fun toString(): String {
        val b = StringBuilder()
        fun s(l: Long): String {
            return String.format("%64s", java.lang.Long.toBinaryString(l)).replace(" ", "0");
        }
        b.append("Sha512Random(state=[")
        hashState.joinTo(b, transform = { s(it) })
        b.append("], unconsumed=[")
        unconsumed.joinTo(b, transform = { s(it) })
        b.append("])")
        return b.toString()
    }

    override fun hashCode(): Int {
        return hashState[0].hashCode()
    }

    override fun equals(other: Any?): Boolean {
        // We don't consider [hasBeenUsed] part of the state for purposes of [equals].
        return this === other ||
                other is Sha512Random &&
                other.hashState.contentEquals(this.hashState) &&
                other.unconsumed.contentEquals(this.unconsumed)
    }

    /**
     * Produce an effectively infinite lazily-constructed stream of floats.
     */
    private class Floats(r: Sha512Random) : FloatIterator() {
        override fun hasNext(): Boolean = true

        val keyScheduleBuffer = LongArray(rounds)

        // if block is full, finish it before constructing new states
        val oldState = if (r.unconsumed.size < unconsumedSize) r.hashState else finish(r.hashState, r.unconsumed, keyScheduleBuffer = keyScheduleBuffer)
        val oldUnconsumed = if (r.unconsumed.size < unconsumedSize) r.unconsumed else emptyUnconsumedInputArray
        val newState = LongArray(stateSize)
        var nextGroupIndex = 0L
        var nextStateIndex = stateSize
        var left = true
        val newUnconsumed = LongArray(oldUnconsumed.size + 1).also {
            for (i in oldUnconsumed.indices) it[i] = oldUnconsumed[i]
        }

        fun refill() {
            newUnconsumed[oldUnconsumed.size] = nextGroupIndex.shl(1) + 1
            nextGroupIndex++
            finish(oldState, newUnconsumed,
                newStateBuffer = newState, keyScheduleBuffer = keyScheduleBuffer)
            nextStateIndex = 0
            left = true
        }

        override fun nextFloat(): Float {
            if (nextStateIndex >= stateSize) refill()
            return if (left) {
                left = false
                leftFloat(newState[nextStateIndex])
            } else {
                val result = rightFloat(newState[nextStateIndex])
                left = true
                nextStateIndex++
                result
            }
        }
    }

    @Suppress("NOTHING_TO_INLINE")
    companion object {
        /**
         * The size, in longs, of a state.
         */
        internal const val stateSize = 8

        /**
         * The size, in longs, of a block of input for SHA-512
         */
        internal const val unconsumedSize = 16

        /**
         * The number of rounds for SHA-512.  The specified full number of rounds is 80.
         * SHA-512 has resisted cryptanalysis with 58 rounds.  We might consider
         * fewer rounds still to trade off performance for cryptographic security, which is
         * probably not a requirement for users of this API.
         */
        internal const val rounds = 80

        init {
            require(rounds >= unconsumedSize) // ensure we consume all input
        }

        private val emptyUnconsumedInputArray = longArrayOf(Long.MIN_VALUE)

        /**
         * Incorporate the unconsumed input into the state, returning the new state.
         * The [newStateBuffer] parameter is used to hold the result.
         */
        private fun finish(
            state: LongArray,
            unconsumed: LongArray,
            newStateBuffer: LongArray = LongArray(stateSize),
            keyScheduleBuffer: LongArray = LongArray(rounds)
        ): LongArray {
            // Expand the message block
            val w: LongArray = expand(unconsumed, keyScheduleBuffer = keyScheduleBuffer)

            var a = state[0]
            var b = state[1]
            var c = state[2]
            var d = state[3]
            var e = state[4]
            var f = state[5]
            var g = state[6]
            var h = state[7]

            for (j in 0 until rounds) {
                val t1 = h + S1(e) + Ch(e, f, g) + k[j] + w[j]
                val t2 = S0(a) + Maj(a, b, c)
                h = g
                g = f
                f = e
                e = d + t1
                d = c
                c = b
                b = a
                a = t1 + t2
            }

            newStateBuffer[0] = state[0] + a
            newStateBuffer[1] = state[1] + b
            newStateBuffer[2] = state[2] + c
            newStateBuffer[3] = state[3] + d
            newStateBuffer[4] = state[4] + e
            newStateBuffer[5] = state[5] + f
            newStateBuffer[6] = state[6] + g
            newStateBuffer[7] = state[7] + h
            return newStateBuffer
        }

        /**
         * The SHA-512 function to expand an input block to the values to be used in each round.
         */
        private fun expand(
            u: LongArray,
            keyScheduleBuffer: LongArray
        ): LongArray {
            val w = keyScheduleBuffer
            for (j in 0 until unconsumedSize)
                w[j] = if (j < u.size) u[j] else if (j == u.size) endOfInputMarker else 0L
            for (j in unconsumedSize until rounds)
                w[j] = s1(w[j - 2]) + w[j - 7] + s0(w[j - 15]) + w[j - 16]
            return w
        }

        /**
         * The start of padding data, as specified (a one bit followed by zeros)
         */
        private const val endOfInputMarker = Long.MIN_VALUE

        /**
         * A constant used to scale [UInt] values to convert them to floating-point.
         * We have to calculate this carefully so that 1.0f is not a possible output
         * because the generated values are specified to be exclusive of 1.0f.
         */
        private val scale: Float = run {
            // The maximum output we want
            val maxOutput = 1.0f.nextDown()
            // The maximum input that should map to that value
            val maxInput = UInt.MAX_VALUE.toFloat()

            // what do we multiply maxInput by to get maxOutput?
            val value = maxOutput / maxInput

            // Check that the scale works.
            require(UInt.MIN_VALUE.toFloat() * value == 0f)
            require(UInt.MAX_VALUE.toFloat() * value == 1.0f.nextDown())
            value
        }

        /** Extract a float sample from the high-order bits of [x]. */
        internal inline fun leftFloat(x: Long) = x.shr(32).toUInt().toFloat() * scale

        /** Extract a float sample from the low-order bits of [x]. */
        internal inline fun rightFloat(x: Long) = x.toUInt().toFloat() * scale

        // SHA-512 initial state
        internal val initialState = listOf(
            0x6a09e667f3bcc908UL,
            0xbb67ae8584caa73bUL,
            0x3c6ef372fe94f82bUL,
            0xa54ff53a5f1d36f1UL,
            0x510e527fade682d1UL,
            0x9b05688c2b3e6c1fUL,
            0x1f83d9abfb41bd6bUL,
            0x5be0cd19137e2179UL)
                .map { it.toLong() }.toLongArray()

        // SHA-512 round constants
        internal val k = listOf(
            0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL, 0x3956c25bf348b538UL,
            0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL, 0xd807aa98a3030242UL, 0x12835b0145706fbeUL,
            0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL, 0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL,
            0xc19bf174cf692694UL, 0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
            0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL, 0x983e5152ee66dfabUL,
            0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL, 0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL,
            0x06ca6351e003826fUL, 0x142929670a0e6e70UL, 0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL,
            0x53380d139d95b3dfUL, 0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
            0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL, 0xd192e819d6ef5218UL,
            0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL, 0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL,
            0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL, 0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL,
            0x682e6ff3d6b2b8a3UL, 0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
            0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL, 0xca273eceea26619cUL,
            0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL, 0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL,
            0x113f9804bef90daeUL, 0x1b710b35131c471bUL, 0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL,
            0x431d67c49c100d4cUL, 0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL)
                .map { it.toLong() }.toLongArray()

        // Helper functions used in [finish]
        internal inline fun Ch(x: Long, y: Long, z: Long) = (x and y) xor (x.inv() and z)
        internal inline fun Maj(x: Long, y: Long, z: Long) = (x and y) xor (x and z) xor (y and z)
        internal inline fun S0(x: Long) = x.rotateRight(28) xor x.rotateRight(34) xor x.rotateRight(39)
        internal inline fun S1(x: Long) = x.rotateRight(14) xor x.rotateRight(18) xor x.rotateRight(41)
        // Helper functions used in [expand]
        internal inline fun s0(x: Long) = x.rotateRight(1) xor x.rotateRight(8) xor x.shr(7)
        internal inline fun s1(x: Long) = x.rotateRight(19) xor x.rotateRight(61) xor x.shr(6)
    }

    /**
     * Produce an effectively infinite lazily-constructed stream of gaussian floats with mean zero and variance one.
     * This is based on the boxMuller method
     * https://en.wikipedia.org/wiki/Box-Muller_transform
     */
    private class GaussianFloats(val floats: Floats) : FloatIterator() {
        private var hasStoredGaussian = false
        private var storedGaussian = 0f
        override fun hasNext(): Boolean = true

        override fun nextFloat(): Float {
            return if (hasStoredGaussian) {
                hasStoredGaussian = false
                storedGaussian
            } else {
                hasStoredGaussian = true

                val u1 = floats.nextFloat()
                val u2 = floats.nextFloat()
                // We take the log of u1 + epsilon to avoid infinity when u1 is zero
                val epsilon = 1f / UInt.MAX_VALUE.toFloat()
                val r = sqrt((-2f) * ln(u1 + epsilon))
                val theta = (2 * PI.toFloat()) * u2
                storedGaussian = r * sin(theta)
                r * cos(theta)
            }
        }
    }

    /**
     * Samples from a gamma distribution with mean [alpha] / [beta] and variance of [alpha] / ([beta] ** 2)
     * This is based on the Marsaglia and Tsang's Method
     * https://en.wikipedia.org/wiki/Gamma_distribution
     * https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
     * Helper function used in [gamma]
     */
    override fun gamma(alpha: FloatTensor): DTensor {
        if (hasBeenUsed && !permitsReuse)
            throw IllegalStateException("RandomKey instance has already been used.")
        hasBeenUsed = true
        val floats = Floats(this)
        val gaussianFloats = GaussianFloats(floats)
        return alpha.impureMap {
            require(it > 0)
            generateGammaFloat(it, floats, gaussianFloats)
        }
    }

    // Helper function used in [generateGammaTensor]
    private fun generateGammaFloat(alpha: Float, floats: Floats, gaussianFloats: GaussianFloats): Float {
        if (alpha >= 1) {
            val d = alpha - (1f / 3)
            val c = 1 / sqrt(9f * d)
            while (true) {
                // X ~ N(0, 1)
                val x = gaussianFloats.nextFloat()
                if (x <= -1 / c)
                    continue
                // U ~ U(0, 1)
                val u = floats.nextFloat()
                val v = (1 + c * x).pow(3)
                // We want x > -1 / c and ln(u) < (0.5 * x.pow(2)) + d * (1 - v + ln(v))
                // If not, generate x and u again
                if (ln(u) < (0.5 * x.pow(2)) + d * (1 - v + ln(v))) {
                    return d * v
                }
            }
        } else {
            return generateGammaFloat(alpha + 1f, floats, gaussianFloats) * floats.nextFloat().pow(1 / alpha)
        }
    }

    // Helper function to generate a Gaussian tensor
    override fun gaussian(shape: Shape): DTensor {
        if (hasBeenUsed && !permitsReuse)
            throw IllegalStateException("RandomKey instance has already been used.")
        hasBeenUsed = true
        val floats = Floats(this)
        val gaussianFloats = GaussianFloats(floats)
        val array = FloatArray(shape.product) {
            gaussianFloats.nextFloat()
        }

        return FloatTensor(shape, array)
    }
}