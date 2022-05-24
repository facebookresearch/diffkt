/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import java.lang.IllegalArgumentException
import kotlin.random.Random

/**
 * A trainable embedding table with size vocabSize x embeddingSize
 *
 * @param numEmbeddings the size of the vocabulary/number of embedding vectors
 * @param embeddingSize the size of each embedding vector
 *
 * Accepts a tensor of Shape(*) and returns a tensor of Shape(*, embeddingSize)
 */
class Embedding(
    val trainableWeights: TrainableTensor
) : TrainableLayer<Embedding> {

    constructor(numEmbeddings: Int,
                embeddingSize: Int,
                random: Random,
                initializer: (Shape, Random)->FloatTensor = Initializer.gaussian(),
    ) : this(TrainableTensor(initializer(Shape(numEmbeddings, embeddingSize), random)))

    override fun invoke(vararg inputs: DTensor): DTensor {
        throw IllegalArgumentException("Embedding must be called with an IntTensor")
    }

    override val trainables: List<Trainable<*>>
        get() = listOf(trainableWeights)

    override fun withTrainables(trainables: List<Trainable<*>>): Embedding {
        require(trainables.size == 1)
        return Embedding(trainables[0] as TrainableTensor)
    }

    operator fun invoke(input: IntTensor) : DTensor {
        return embedding(trainableWeights.tensor, input)
    }

    override fun hashCode(): Int = combineHash("Embedding", trainableWeights)
    override fun equals(other: Any?): Boolean = other is Embedding &&
            other.trainableWeights == trainableWeights
}
