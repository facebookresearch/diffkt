/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.FloatTensor
import org.diffkt.Shape
import shapeTyping.annotations.AllowUnreduced
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.random.asJavaRandom

object Initializer {
    /**
     * Initializer producing normally distributed random values with provided mean and variance.
     */
    @AllowUnreduced
    fun gaussian(mean: Float = 0f, variance: Float = 1f): (Shape, Random) -> FloatTensor = { shape: Shape, random: Random ->
        val r = random.asJavaRandom()
        val stddev = sqrt(variance)
        FloatTensor(shape) { (r.nextGaussian() * stddev + mean).toFloat() }
    }

    fun uniform(min: Float = 0f, max: Float = 1f) = { shape: Shape, random: Random ->
        FloatTensor.random(random, shape, min, max)
    }

    object kaimingUniform {
        operator fun invoke(fanMode: FanMode, activationGainFactor: Float) = { shape: Shape, random: Random ->
            val fan = fanMode.fanFactor(shape)
            val gain = activationGainFactor
            val bound = sqrt(3f / fan.toFloat()) * gain
            FloatTensor.random(random, shape, -bound, bound)
        }

        /**
         * These values can be used for the activationGainFactor parameter to [kaimingUniform.invoke], depending on the activation function.
         */

        const val LinearTransformGainFactor = 1f
        const val ConvTransformGainFactor = 1f
        const val SigmoidTransformGainFactor = 1f
        const val TanhTransformGainFactor = 5f / 3
        val ReluTransformGainFactor = sqrt(2f)
        fun LeakyReluTransformGainFactor(slope: Float) = sqrt(2f / (1 + slope.pow(2)))
    }
}
