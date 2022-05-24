/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import org.diffkt.*

/**
 * A functional interface for generating random numbers.  See also the paper "Splittable
 * pseudorandom number generators using cryptographic hashing" by Claessen and Palka
 * (https://publications.lib.chalmers.se/records/fulltext/183348/local_183348.pdf).
 * A given [RandomKey] should be used either for the [split] operation, or [floats].  Once it is
 * used for one of these, the given [RandomKey] instance should be discarded, as it would only
 * generate the same values again.  By default, a [RandomKey] will throw an exception if you
 * try to use it for more than one operation.  To permit multiple uses of the same [RandomKey]
 * instance, for example because you want deterministic results, call [permitReuse] and use
 * its return value instead.
 */
interface RandomKey: Wrappable<RandomKey> {
    /**
     * Split this random key into a new set of distinct and statistically independent keys.
     */
    fun split(n: Int = 2): List<RandomKey>

    /**
     * Wrap random key
     */
    override fun wrap(wrapper: Wrapper): RandomKey = wrapper.wrapRandomKey(this)

    /**
     * Produce a fresh tensor of the given shape filled with uniformly distributed
     * float values between 0 (inclusive) and 1 (exclusive).  This method will
     * always produce the same sequence of values for a given [RandomKey] instance.
     */
    fun floats(shape: Shape): DTensor

    /**
     * Return a [RandomKey] just like this one, except that the newly returned instance
     * permits repeated calls to [split] and [floats].
     */
    fun permitReuse(): RandomKey

    /**
     * Produce a fresh vector of the given length filled with random float values
     * between 0 (inclusive) and 1 (exclusive).  This method will
     * always produce the same sequence of values for a given key.
     */
    fun floats(n: Int) = floats(Shape(n))

    /**
     * Produce a fresh tensor of the given shape filled with normally distributed
     * float values with a mean of 0 and standard deviation of 1.  This method will
     * always produce the same sequence of values for a given [RandomKey] instance.
     */
    fun gaussian(shape: Shape): DTensor

    /**
     * Produces a fresh tensor of samples from a gamma distribution,
     * each with shape parameter from the corresponding value in the
     * [alpha] parameter.  This method will
     * always produce the same sequence of values for a given [RandomKey] instance.
     */
    fun gamma(alpha: FloatTensor): DTensor

    companion object {
        /**
         * Creates a new random number generator seeded from the time of day.
         */
        fun fromTimeOfDay(): RandomKey = Sha512Random(System.nanoTime())

        /**
         * Creates a new random number generator with a fixed seed.
         */
        operator fun invoke(): RandomKey = Sha512Random(18270326L) // RIPLVB

        /**
         * Creates a new random number generator seeded from the given value
         */
        operator fun invoke(n: Long): RandomKey = Sha512Random(n)

        /**
         * Creates a new random number generator seeded from the given value
         */
        operator fun invoke(n: Int): RandomKey = Sha512Random(n.toLong())
    }
}
