/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.french

import org.diffkt.*
import org.diffkt.data.loaders.french.EngFrenchPair
import org.diffkt.data.loaders.french.Vocabulary
import org.diffkt.model.*
import kotlin.random.Random


// Basic Seq2Seq with Attention. No batching.
class French2English constructor(
    val englishVocab: Vocabulary,
    val frenchVocab: Vocabulary,
    val encoder: EncoderRNN,
    val decoder: AttentionDecoderRNN,
    val random: Random
): TrainableComponent<French2English> {

    constructor(vocabs: Pair<Vocabulary, Vocabulary>, random: Random): this(
        frenchVocab = vocabs.second,
        englishVocab = vocabs.first,
        encoder = EncoderRNN(vocabs.second.word2Idx.size, HIDDEN_SIZE, random),
        decoder = AttentionDecoderRNN(HIDDEN_SIZE, vocabs.first.word2Idx.size, random),
        random = random
    )

    constructor(englishVocab: Vocabulary, frenchVocab: Vocabulary, random: Random): this(
        frenchVocab = frenchVocab,
        englishVocab = englishVocab,
        encoder = EncoderRNN(frenchVocab.word2Idx.size, HIDDEN_SIZE, random),
        decoder = AttentionDecoderRNN(HIDDEN_SIZE, englishVocab.word2Idx.size, random),
        random = random
    )

    companion object {
        const val HIDDEN_SIZE = 256
    }

    override val trainables: List<Trainable<*>>
        get() = listOf(encoder, decoder)

    override fun withTrainables(trainables: List<Trainable<*>>): French2English {
        require(trainables.size == 2)
        val newEncoder = trainables[0] as EncoderRNN
        val newDecoder = trainables[1] as AttentionDecoderRNN
        return French2English(englishVocab, frenchVocab, newEncoder, newDecoder, random)
    }


    private fun oneHot(vocab: Vocabulary, label: Int): FloatTensor {
        require(0 <= label && label < vocab.size) {"Label $label not within expected range"}
        val data = FloatArray(vocab.size)
        data[label] = 1f
        return FloatTensor(Shape(vocab.size), data)
    }
    /*
     * Runs the seq2seq model on a single french-english pair
     * Returns the loss
     */
    fun lossStep(french: IntTensor, english: IntTensor): DScalar {
        val outputLength = english.size
        var loss:DScalar = FloatScalar(0f)
        // Encoder
        val encoderHidden: DTensor = encoder.initialHidden
        var encoderOutputs = encoder.forward(french, encoderHidden)
        // Pad encoder outputs to a minimum of length 10 for use in attention
        if (encoderOutputs.shape[1] < 10) {
            encoderOutputs = encoderOutputs.concat(FloatTensor.zeros(Shape(
                encoderOutputs.shape[0],
                10 - encoderOutputs.shape[1],
                HIDDEN_SIZE
            )), axis = 1)
        }
        // Decoder, always teacher forcing TODO: Make this randomly teacher force
        var decoderInput = intScalarOf(0)
        var decoderHidden = encoderOutputs.slice(french.size -1, french.size, 1).squeeze(0)
        for (di in 0 until outputLength) {
            val decoderResult = decoder.forward(decoderInput, decoderHidden, encoderOutputs, random)
            val l = nllLossFromOneHot(decoderResult.first, oneHot(englishVocab, english.at(di)))
            loss += l
            decoderHidden = decoderResult.second
            decoderInput = english[di]
        }
        // Normalize loss against sentence length
        loss /= outputLength
        return loss
    }

    fun trainStep(french: IntTensor, english: IntTensor, optimizer: Optimizer<French2English>): Pair<DScalar, French2English> {
        val (loss, tangent) = primalAndReverseDerivative(
            x = this,
            f = {model: French2English ->
                model.lossStep(french, english)
            },
            extractDerivative = { model: French2English,
                                  loss: DScalar,
                                  extractor: (input: DTensor, output: DTensor) -> DTensor ->
                model.extractTangent(loss, extractor)}
        )
        val trainedModel: French2English = optimizer.train(this, tangent)
        return Pair(loss, trainedModel)
    }

    /*
     * training function loosely ported from v1, unsure why we don't do epochs
     * TODO: Add trainUntil functionality
     */
    fun train(
        sentences: List<EngFrenchPair>,
        iterations: Int,
        printEvery: Int? = 1000,
        trainUntil: Int? = null,
        optimizer: Optimizer<French2English>,
    ): French2English {
        val sentencePairs = (0 until iterations).map {
            val sentence = sentences.random()
            Pair(frenchVocab.sentenceToTensor(sentence.second), englishVocab.sentenceToTensor(sentence.first))
        }
        var totalLoss = 0f
        val optimizedModel = (0 until iterations).fold(this) {model: French2English, i: Int ->
            val sentencePair = sentencePairs[i]
            val (loss, trainedModel) = trainStep(sentencePair.first, sentencePair.second, optimizer)
            printEvery?.let {
                totalLoss += (loss as FloatScalar).value
                if (i % it == 0) println("Iter $i loss: $loss")
            }

            trainedModel
        }
        return optimizedModel
    }
}