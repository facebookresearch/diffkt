/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import org.diffkt.*

class DiffktRandom (val randomKey: RandomKey): Wrappable<DiffktRandom> {
    private val keyPoolSize = 1 shl 16
    private val randomPoolSize = 16

    @JvmName("getKey")
    fun getRandomKey(): RandomKey {
        return getKeyFromKeyPool()
    }

    override fun wrap(wrapper: Wrapper): DiffktRandom {
        require(!::keyPool.isInitialized) { "for now we only wrap unused DiffktRandom"}
        return DiffktRandom(wrapper.wrapRandomKey(randomKey))
    }

    /**
     * Grabs the next key from the pool
     * Refills pool if all the keys have already been used by using the last key
     */
    private var keyIndex = -1
    private lateinit var keyPool: List<RandomKey>
    private fun getKeyFromKeyPool(): RandomKey {
        keyIndex++
        if (!this::keyPool.isInitialized) {
            keyPool = randomKey.split(keyPoolSize)
        }
        if (keyIndex == keyPool.size - 1) {
            keyPool = keyPool[keyIndex].split(keyPoolSize)
            keyIndex = 0
        }
        return keyPool[keyIndex]
    }

    /**
     * Grabs the next uniform FloatScalar from the pool
     * Refills pool if all the uniform FloatScalars have been used
     */

    private var uniformIndex = -1
    private lateinit var uniformPool: DTensor
    fun nextUniform(): DScalar {
        uniformIndex++
        if (!this::uniformPool.isInitialized || uniformIndex == uniformPool.size) {
            uniformPool = getKeyFromKeyPool().floats(randomPoolSize)
            uniformIndex = 0
        }
        return uniformPool[uniformIndex] as DScalar
    }

    /**
     * Samples from a uniform distribution between 0 and 1
     */
    fun nextUniform(shape: Shape): DTensor {
        val uniformKey = getKeyFromKeyPool()
        return uniformKey.floats(shape)
    }

    /**
     * Samples from a uniform distribution between [low] and [high]
     */
    fun nextUniform(shape: Shape, low: DTensor, high: DTensor): DTensor {
        val uniformKey = getKeyFromKeyPool()
        val floats = uniformKey.floats(shape)
        return floats * high + (1f - floats) * low
    }

    /**
     * Grabs the next gaussian FloatScalar from the pool
     * Refills pool if all the gaussian FloatScalars have been used
     */
    private var gaussianIndex = -1
    private lateinit var gaussianPool: DTensor
    fun nextGaussian(): DScalar {
        gaussianIndex++
        if (!this::gaussianPool.isInitialized || gaussianIndex == gaussianPool.size) {
            gaussianPool = nextGaussian(Shape(randomPoolSize))
            gaussianIndex = 0
        }
        return gaussianPool[gaussianIndex] as DScalar
    }

    /**
     * Samples from a gaussian distribution with mean 0 and std 1 (standard deviation)
     */
    fun nextGaussian(shape: Shape): DTensor {
        val gaussianKey = getKeyFromKeyPool()
        return gaussianKey.gaussian(shape)
    }

    /**
     * Samples from a gaussian distribution with [mean] and [std] (standard deviation)
     */
    fun nextGaussian(shape: Shape, mean: DTensor, std: DTensor): DTensor {
        val gaussianKey = getKeyFromKeyPool()
        return gaussianKey.gaussian(shape, mean, std)
    }

    /**
     * Grabs the next cauchy FloatScalar from the pool
     * Refills pool if all the cauchy FloatScalars have been used
     */
    private var cauchyIndex = -1
    private lateinit var cauchyPool: DTensor
    fun nextCauchy(): DScalar {
        cauchyIndex++
        if (!this::cauchyPool.isInitialized || cauchyIndex == cauchyPool.size) {
            cauchyPool = nextCauchy(Shape(randomPoolSize))
            cauchyIndex = 0
        }
        return cauchyPool[cauchyIndex] as DScalar
    }

    /**
     * Samples from a cauchy distribution with loc 0 and scale 1
     */
    fun nextCauchy(shape: Shape): DTensor {
        val cauchyKey = getKeyFromKeyPool()
        return cauchyKey.cauchy(shape)
    }

    /**
     * Samples from a cauchy distribution with [loc] and [scale]
     */
    fun nextCauchy(shape: Shape, loc: DTensor, scale: DTensor): DTensor {
        val cauchyKey = getKeyFromKeyPool()
        return cauchyKey.cauchy(shape, loc, scale)
    }

    /**
    * Samples from a chi square distribution
    */
    fun nextChiSquare(shape: Shape, dof: DTensor): DTensor {
        val chiSquareKey = getKeyFromKeyPool()
        return chiSquareKey.chiSquare(shape, dof)
    }

    /**
     * Samples from a gamma distribution with a shape parameter [alpha] (also known as k) and a scale/ rate parameter of 1
     */
    fun nextGamma(shape: Shape, alpha: DTensor): DTensor {
        val gammaKey = getKeyFromKeyPool()
        return gammaKey.gamma(shape, alpha)
    }

    /**
     * Gamma distribution with a shape parameter [k], which is equal to alpha, and a scale parameter [θ].
     */
    fun nextGammaWithScale(shape: Shape, k: DTensor, theta: DTensor): DTensor {
        val gammaKey = getKeyFromKeyPool()
        return gammaKey.gammaWithScale(shape, k, theta)
    }

    /**
     * Gamma distribution with a shape parameter [alpha], which is equal to k, and a rate parameter [β] = 1/θ
     */
    fun nextGammaWithRate(shape: Shape, alpha: DTensor, beta: DTensor): DTensor {
        val gammaKey = getKeyFromKeyPool()
        return gammaKey.gammaWithRate(shape, alpha, beta)
    }

    /*
     * In both [hashCode] and [equals] we ignore the initialized state of keyPool and its possible values.
     *
     * This is because when the Tracing Jit wraps a DiffktRandom input, it stores an instance of DiffktRandom
     * with a wrapped RandomKey as part of the cache key. Successive calls to the resulting jitted function mutate
     * the cached DiffktRandom by either initializing the keyPool, increasing the keyIndex, or modifying the keyPool contents.
     * Therefore, equality check with an uninitialized DiffktRandom will fail causing the function cache to increase.
     * For now, we will just reject using an uninitialized DiffktRandom in [wrap]
     *
     * In addition, DiffKtRandom is not a value type so we should not normally use equals and hashCode on it.
     */
    override fun hashCode(): Int = combineHash("DiffktRandom", randomKey.hashCode())

    override fun equals(other: Any?): Boolean =
        this === other ||
                other is DiffktRandom &&
                randomKey == other.randomKey

    companion object {
        fun fromTimeOfDay() = DiffktRandom(RandomKey.fromTimeOfDay())

        operator fun invoke(): DiffktRandom {
            val randomKey = RandomKey()
            return DiffktRandom(randomKey)
        }
    }
}
