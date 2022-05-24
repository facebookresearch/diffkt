/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType
import kotlin.random.Random

class Dropout(val dropoutPercent: Float): Layer<Dropout>, LayerWithInferenceMode {
    init {
        require(dropoutPercent >= 0f && dropoutPercent < 1f) { "Probability must be >= 0f and < 1f; got $dropoutPercent" }
    }

    override fun invoke(vararg inputs: DTensor): DTensor {
        throw IllegalArgumentException("Should use invoke(vararg inputs: DTensor, random: Random)")
    }


    @SType("S: Shape")
    @AllowUnreduced
    operator fun invoke(input: @SType("S") DTensor, random: Random): @SType("S") DTensor {
        val yes = 1f / (1f - dropoutPercent)
        val dropout = FloatTensor(input.shape) { if (random.nextFloat() > dropoutPercent) yes else 0f }
        return input * dropout
    }

    override val inferenceMode: Layer<*>
        get() = Sequential()
}
