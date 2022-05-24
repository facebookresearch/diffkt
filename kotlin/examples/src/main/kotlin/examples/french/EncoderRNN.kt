/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.french

import org.diffkt.model.*
import org.diffkt.*
import kotlin.random.Random
import org.diffkt.model.RecurrentBase.RecurrentBase.AccType

class EncoderRNN private constructor(
    val hiddenSize: Int,
    val embedding: Embedding,
    val gru: LinearBeforeResetGRU
) : TrainableComponent<EncoderRNN> {
    constructor(inputSize: Int, hiddenSize: Int, random: Random) : this(
        hiddenSize,
        Embedding(inputSize, hiddenSize, random),
        LinearBeforeResetGRU(hiddenSize, hiddenSize, random, acc = AccType.AccMap)
        )

    override val trainables = listOf(embedding, gru)

    // returns encoder outputs
    // encoder outputs is (batch, seq_len, hidden_state)
    fun forward(input: IntTensor, hidden: DTensor): DTensor {
        val embed = embedding(input)
        return gru(embed.unsqueeze(0), hidden)
    }

    val initialHidden get() = FloatTensor.zeros(Shape(1, hiddenSize))

    override fun withTrainables(trainables: List<Trainable<*>>): EncoderRNN {
        require(trainables.size == 2)
        return EncoderRNN(hiddenSize, trainables[0] as Embedding, trainables[1] as LinearBeforeResetGRU)
    }
}
