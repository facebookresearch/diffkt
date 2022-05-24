/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.mnist

import org.diffkt.*
import org.diffkt.model.SGDOptimizer
import kotlin.math.min
import kotlin.random.Random
import kotlin.system.measureNanoTime

/**
 * Returns a normalized copy of `this`.
 *
 * Normalization is: (this - mean) / std
 */
fun DTensor.normalize(
    mean: Float,
    std: Float
): DTensor {
    return (this - mean) / std
}

fun main() {
    val initialWeights = MnistModel(Random(1234567))
    val (rawFeatures, allLabels) = loadTrainingData()

    // Manually set what we know to be the mean and std of the train dataset,
    // which makes data loading much faster.
    val normalizedFeatures = rawFeatures.normalize(mean = 0.13113596f, std = 0.2885157f)

    // Add a single "channel" dimension (the images are greyscale)
    val channelDim = 3
    val allFeatures = normalizedFeatures.unsqueeze(channelDim)

    // process a small subset of the data per batch.
    val samples = allFeatures.shape.first
    val batchSize = 300
    val alpha = SGDOptimizer<MnistModel>(0.035f, weightDecay = true, momentum = true)
    val epochs = 3
    val batchesPerEpoch = (samples + batchSize - 1) / batchSize // permit a partial batch at the end

    var trainedModel: MnistModel? = null
    val elapsed = measureNanoTime {
        trainedModel = (0 until epochs).fold(initial = initialWeights) { weights0: MnistModel, epoch: Int ->
            (0 until batchesPerEpoch).fold(initial = weights0) { weights1: MnistModel, batch: Int ->
                val start = (batch * batchSize) % allFeatures.shape.first
                // use a partial batch at the end
                val end = min(start + batchSize, samples)
                val features = allFeatures.slice(start, end)
                val labels = allLabels.slice(start, end)
                val (loss, newWeights) = learn(features, labels, alpha, weights1)
                if (batch % 20 == 0 || batch == (batchesPerEpoch - 1))
                    println("examples.mnist epoch $epoch batch $batch: loss = $loss")
                newWeights
            }
        }
    }

    trainedModel as MnistModel
    println("examples.mnist compute = ${elapsed / 1E9f} sec")
}
