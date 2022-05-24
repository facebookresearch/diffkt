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

class AttentionDecoderRNN(
    val embedding: Embedding,
    val attn: Dense,
    val attnCombine: Dense,
    val dropout: Layer<*>,
    val gru: LinearBeforeResetGRU,
    val out: Dense
) : TrainableLayer<AttentionDecoderRNN> {
    override fun hashCode(): Int = combineHash("AttentionDecoderRNN", embedding, attn, attnCombine, dropout, gru, out)
    override fun equals(other: Any?): Boolean = other is AttentionDecoderRNN &&
            other.embedding == embedding &&
            other.attn == attn &&
            other.attnCombine == attnCombine &&
            other.dropout == dropout &&
            other.gru == gru &&
            other.out == out

    constructor(hiddenSize: Int, outputSize: Int, random: Random, dropoutPct: Float = 0.1f, maxLength: Int = 10) : this(
        embedding = Embedding(outputSize, hiddenSize, random),
        attn = Dense(hiddenSize * 2, maxLength, random),
        attnCombine = Dense(hiddenSize * 2, hiddenSize, random),
        dropout = Dropout(dropoutPct),
        gru = LinearBeforeResetGRU(hiddenSize, hiddenSize, random, acc = RecurrentBase.RecurrentBase.AccType.AccMap),
        out = Dense(hiddenSize, outputSize, random)
    )

    override val trainables = listOf(embedding, attn, attnCombine, gru, out)

    override fun withTrainables(trainables: List<Trainable<*>>): AttentionDecoderRNN {
        require(trainables.size == 5)
        return AttentionDecoderRNN(
            embedding = trainables[0] as Embedding,
            attn = trainables[1] as Dense,
            attnCombine = trainables[2] as Dense,
            dropout = this.dropout,
            gru = trainables[3] as LinearBeforeResetGRU,
            out = trainables[4] as Dense)
    }

    private fun dropout(value: DTensor, random: Random): DTensor {
        return when (dropout) {
            is Dropout -> dropout.invoke(value, random)
            else -> dropout.invoke(value)
        }
    }

    /**
     * TODO: make batch compatible (gets rid of unneeded squeeze and unsqueeze)
     * Shapes:
     *  - input: Shape()
     *  - hidden: Shape(1, hidden) //
     *  - output: 3 Tensors of shape (1, outputSize) (1, hiddenSize) (1, maxLength)
     */
    fun forward(
        input: IntTensor,
        hidden: DTensor,
        encoderOutput: DTensor,
        random: Random
    ): Triple<DTensor, DTensor, DTensor> {
        val embedded = dropout(embedding(input), random).unsqueeze(0)
        val lastDim = embedded.shape.rank - 1
        val attnWeights = attn(embedded.concat(hidden, lastDim)).softmax(lastDim)
        val attnApplied = attnWeights.matmul(encoderOutput.squeeze(0))
        var output = embedded.concat(attnApplied, lastDim)
        output = attnCombine(output).relu()
        val gruOutput = gru(output.unsqueeze(0), hidden).squeeze(0)
        output = out(gruOutput).logSoftmax(lastDim)
        return Triple(output, gruOutput, attnWeights)
    }

    override fun invoke(vararg inputs: DTensor): DTensor {
        TODO("AttentionDecoderRNN requires the use of forward(...)")
    }
}
