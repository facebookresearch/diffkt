/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Densely-connected layer
 *
 * @returns input matmul weights + bias
 *
 * Input shape: (N, numInputs)
 * Output shape: (N, numOutputs)
 */
class Dense private constructor(
    private val trainableW: TrainableTensor,
    private val trainableB: TrainableTensor,
    val activation: Activation,
    val bias: Boolean,
) : TrainableLayerSingleInput<Dense> {
    /** Full public constructor */
    constructor(
        numInputs: Int,
        numOutputs: Int,
        random: Random,
        bias: Boolean = true,
        activation: Activation = defaultActivation,
        weightInit: (Shape, Random)->FloatTensor = defaultInit(numInputs),
        biasInit: (Shape, Random)->FloatTensor = defaultInit(numInputs)
    ) : this(
        trainableW = TrainableTensor(weightInit(Shape(numInputs, numOutputs), random)),
        trainableB = TrainableTensor(
            if (bias) biasInit(Shape(numOutputs), random) else FloatScalar.ZERO
        ),
        activation = activation,
        bias = bias,
    )

    /** Convenience constructor where `bias` is always true. */
    constructor(
        numInputs: Int,
        numOutputs: Int,
        activation: Activation,
        random: Random,
        weightInit: (Shape, Random)->FloatTensor = defaultInit(numInputs),
        biasInit: (Shape, Random)->FloatTensor = defaultInit(numInputs)
    ) : this(numInputs, numOutputs, random, true, activation, weightInit, biasInit)

    val w: DTensor get() = trainableW.tensor
    val b: DTensor get() = trainableB.tensor

    override val trainables = if (bias) listOf(trainableW, trainableB) else listOf(trainableW)

    override fun withTrainables(trainables: List<Trainable<*>>): Dense {
        val t = trainables.toTypedArray()
        val newTrainableW = t[0] as TrainableTensor
        val newTrainableB = if (t.size > 1) t[1] as TrainableTensor else trainableB
        return Dense(trainableW = newTrainableW, trainableB = newTrainableB, activation = activation, bias = bias)
    }

    override fun invoke(input: DTensor): DTensor {
        require(input.rank >= 2) { "input rank to Dense must be at least 2, but was ${input.rank}" }
        return apply(input)
    }

    fun apply(input: DTensor): DTensor {
        return activation(input.matmul(w) + b)
    }

    override fun wrap(wrapper: Wrapper): Dense = Dense(
        trainableW = wrapper.wrap(trainableW),
        trainableB = wrapper.wrap(trainableB),
        activation = activation,
        bias = bias,
    )

    override fun hashCode(): Int = combineHash("Dense", trainableW, trainableB, activation, bias)
    override fun equals(other: Any?): Boolean = other is Dense &&
            other.trainableW == trainableW &&
            other.trainableB == trainableB &&
            other.activation == activation &&
            other.bias == bias

    companion object {
        // Default values for constructors
        private val defaultActivation = Activation.Identity
        private fun defaultInit(numInputs: Int): (Shape,Random)->FloatTensor {
            val v = sqrt(1f / numInputs)
            return Initializer.uniform(
                min = -v,
                max = v,
            )
        }
    }
}
