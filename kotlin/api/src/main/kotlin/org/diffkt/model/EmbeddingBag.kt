/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import org.diffkt.model.Initializer.gaussian
import java.lang.IllegalArgumentException
import kotlin.random.Random

/**
 * A trainable embedding table with size vocabSize x embeddingSize
 *
 * @param numEmbeddings the size of the vocabulary/number of embedding vectors
 * @param embeddingSize the size of each embedding vector
 * @param reduction the reduction mode used to reduce each bag
 *
 * This layer is equivalent to Embedding followed by a reduction (e.g. sum)
 * across each bag of embeddings along the 0th axis.
 */
class EmbeddingBag(
        val trainableWeights: TrainableTensor,
        val reduction: Reduction,
) : TrainableLayer<EmbeddingBag> {

    override val trainables = listOf(trainableWeights)

    constructor(numEmbeddings: Int,
                embeddingSize: Int,
                reduction: Reduction,
                random: Random,
                initializer: (Shape, Random)->FloatTensor = Initializer.gaussian(),
    ) : this(TrainableTensor(initializer(Shape(numEmbeddings, embeddingSize), random)), reduction)

    /**
     * @param indices the indices into the embedding table
     * @param bagOffsets linear offsets into indices indicating the start of each bag.
     * @return FloatTensor of shape (numBags, embeddingSize)
     */
    operator fun invoke(indices: IntTensor, bagOffsets: IntTensor): DTensor {
        // TODO: Make this more efficient by doing the reduction in-place
        //   https://github.com/facebookincubator/diffkt/issues/276
        // Get the embeddings
        val embeddings = embedding(trainableWeights.tensor, indices.flatten())
        // Reduce the embeddings into bags
        val flatOffsets = bagOffsets.flatten()
        val lastOffset = indices.size
        val reducedBags = mutableListOf<DTensor>()
        for ((i, startOffset) in flatOffsets.dataIterator.withIndex()) {
            val endOffset = if (i + 1 == flatOffsets.size) lastOffset else flatOffsets.at(i + 1)
            reducedBags += reduction.reduce(embeddings.slice(startOffset, endOffset))
        }
        return concat(reducedBags)
    }

    override fun invoke(vararg inputs: DTensor): DTensor {
        throw IllegalArgumentException("Embedding must be called with an IntTensor")
    }

    override fun withTrainables(trainables: List<Trainable<*>>): EmbeddingBag {
        require(trainables.size == 1)
        return EmbeddingBag(trainables[0] as TrainableTensor, reduction)
    }

    override fun hashCode(): Int = combineHash("EmbeddingBag", trainableWeights, reduction)
    override fun equals(other: Any?): Boolean = other is EmbeddingBag &&
            other.trainableWeights == trainableWeights &&
            other.reduction == reduction

    companion object {
        sealed class Reduction {
            abstract fun reduce(x: DTensor): DTensor

            object Sum : Reduction() {
                override fun reduce(x: DTensor) = x.sum(0, keepDims = true)
            }
        }
    }
}
