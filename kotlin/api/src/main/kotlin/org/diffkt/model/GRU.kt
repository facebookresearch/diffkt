/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import org.diffkt.model.RecurrentBase.RecurrentBase.AccType
import kotlin.random.Random

/**
 * Make a GRU you desire? See `invoke` in the companion object, or use the
 * GRUEncoder or GRUDecoder helpers.
 */
abstract class GRU internal constructor(
    numHidden: Int,
    initialHidden: DTensor? = null,
    override val accType: AccType = AccType.Fold,
) : RecurrentBase<GRU, DTensor> {
    override val initialState: DTensor = when (initialHidden) {
        null -> FloatTensor.zeros(Shape(1, numHidden))
        else -> {
            require(initialHidden.shape == Shape(1, numHidden)) {
                "Base hidden state value should be a vector of [numHidden] ($numHidden) rows." }
            initialHidden
        }
    }
    override val initialOutput = FloatTensor.zeros(Shape(1, numHidden))

    override fun hashCode(): Int = combineHash("GRU", initialState, initialOutput, accType)
    override fun equals(other: Any?): Boolean = other is GRU &&
            other.initialState == initialState &&
            other.initialOutput == initialOutput &&
            other.accType == accType

    override fun processForBatching(
        initialState: DTensor,
        initialOutput: DTensor,
        batchSize: Int
    ): Pair<DTensor, DTensor> {
        require(initialState.shape == this.initialState.shape) {
            "Expected initialHidden to have shape ${this.initialState.shape} but got ${initialState.shape}" }
        val newInitHidden = initialState.expand(initialState.shape.updated(0, batchSize))
        val newInitOutput = initialOutput.expand(initialOutput.shape.updated(0, batchSize))
        return Pair(newInitHidden, newInitOutput)
    }

    companion object {
        operator fun invoke(
            numInputs: Int,
            numHidden: Int,
            initialHidden: DTensor? = null,
            random: Random,
            acc: AccType = AccType.Fold,
            linearBeforeReset: Boolean = false
        ): GRU {
            return if (linearBeforeReset)
                LinearBeforeResetGRU(numInputs, numHidden, random, initialHidden, acc)
            else
                LinearAfterResetGru(numInputs, numHidden, random, initialHidden, acc)
        }
    }
}


/**
 * Linear-after-reset GRU
 *
 * In the computation of the candidate activation vector, the linear transform
 * is applied *after* the hidden state goes through the reset gate.
 *
 * \hat{h}_t = tanh(W_h x_t + U_h (r_t * h_{t-1}) + bias)
 */
class LinearAfterResetGru(
    val numInputs: Int,
    val numHidden: Int,
    val initialHidden: DTensor? = null,
    accType: AccType = AccType.Fold,
    private val xh2u: Dense,
    private val xh2r: Dense,
    private val xh2n: Dense,
): GRU(numHidden, initialHidden, accType) {

    constructor(numInputs: Int, numHidden: Int, random: Random, initialHidden: DTensor? = null, acc: AccType = AccType.Fold): this(
        numInputs,
        numHidden,
        initialHidden,
        acc,
        Dense(numInputs + numHidden, numHidden, random, activation = Activation.Sigmoid),
        Dense(numInputs + numHidden, numHidden, random, activation = Activation.Sigmoid),
        Dense(numInputs + numHidden, numHidden, random, activation = Activation.Tanh),
    )

    override val trainables: List<Trainable<*>> = listOf(xh2u, xh2r, xh2n)

    override fun withTrainables(trainables: List<Trainable<*>>): GRU {
        require(trainables.size == 3)
        return LinearAfterResetGru(
            numInputs,
            numHidden,
            initialHidden,
            accType,
            trainables[0] as Dense,
            trainables[1] as Dense,
            trainables[2] as Dense
        )
    }

    override fun cell(state: Pair<DTensor, DTensor>, x: DTensor): Pair<DTensor, DTensor> {
        val hidden = state.first
        val xh = x.concat(hidden, axis = 1)
        val update = xh2u(xh)
        val reset = xh2r(xh)
        val candidate = xh2n(x.concat(reset * hidden, axis = 1))
        val newHidden = (FloatTensor.ones(reset.shape) - update) * candidate + update * hidden
        // Hidden State is the same as the output TODO: Differentiate between Cell and Layer similar to pytorch?
        return Pair(newHidden, newHidden)
    }

    override fun hashCode(): Int = combineHash("LinearAfterResetGRU", numInputs, numHidden, super.hashCode())
    override fun equals(other: Any?): Boolean = other is LinearAfterResetGru &&
            other.numInputs == numInputs &&
            other.numHidden == numHidden &&
            other.accType == accType &&
            other.xh2u == xh2u &&
            other.xh2r == xh2r &&
            other.xh2n == xh2n &&
            super.equals(other)
}

/**
 * Linear-before-reset GRU
 *
 * In the computation of the candidate activation vector, the linear transform
 * is applied *before* the hidden state goes through the reset gate.
 *
 * \hat{h}_t = tanh(W_h x_t + b_1 + r_t * ( U_h h_{t-1} + b_2))
 *
 * As per version 1 of the the GRU paper (https://arxiv.org/abs/1406.1078v1),
 * TODO: Hookup to DNNL like in v1.
 */

class LinearBeforeResetGRU(
    val numInputs: Int,
    val numHidden: Int,
    val initialHidden: DTensor? = null,
    accType: AccType = AccType.Fold,
    private val xh2u: Dense,
    private val xh2r: Dense,
    private val x2n: Dense,
    private val h2n: Dense,
): GRU(numHidden, initialHidden, accType) {

    constructor(numInputs: Int, numHidden: Int, random: Random, initialHidden: DTensor? = null, acc: AccType = AccType.Fold): this(
        numInputs,
        numHidden,
        initialHidden,
        acc,
        Dense(numInputs + numHidden, numHidden, random, activation = Activation.Sigmoid),
        Dense(numInputs + numHidden, numHidden, random, activation = Activation.Sigmoid),
        Dense(numInputs, numHidden, random, activation = Activation.Identity),
        Dense(numHidden, numHidden, random, activation = Activation.Identity)
    )

    override val trainables: List<Trainable<*>> = listOf(xh2u, xh2r, x2n, h2n)

    override fun withTrainables(trainables: List<Trainable<*>>): GRU {
        require(trainables.size == 4)
        return LinearBeforeResetGRU(
            numInputs,
            numHidden,
            initialHidden,
            accType,
            trainables[0] as Dense,
            trainables[1] as Dense,
            trainables[2] as Dense,
            trainables[3] as Dense,
        )
    }

    override fun cell(state: Pair<DTensor, DTensor>, x: DTensor): Pair<DTensor, DTensor> {
        val hidden = state.first
        val xh = x.concat(hidden, axis = 1)
        val update = xh2u(xh)
        val reset = xh2r(xh)
        val candidate = tanh(x2n(x) + reset * h2n(hidden))
        val newHidden = (FloatTensor.ones(reset.shape) - update) * candidate + update * hidden
        return Pair(newHidden, newHidden)
    }

    override fun hashCode(): Int = combineHash("LinearBeforeResetGRU", numInputs, numHidden, super.hashCode())
    override fun equals(other: Any?): Boolean = other is LinearBeforeResetGRU &&
            other.numInputs == numInputs &&
            other.numHidden == numHidden &&
            other.accType == accType &&
            other.xh2r == xh2r &&
            other.xh2u == xh2u &&
            other.x2n == x2n &&
            other.h2n == h2n &&
            super.equals(other)
}