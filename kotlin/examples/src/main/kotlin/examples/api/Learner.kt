/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.api

import org.diffkt.*
import org.diffkt.data.Data
import org.diffkt.model.*
import org.diffkt.tracing.jit

class Learner<T : Model<T>>(
    val batchedData: Iterable<Data>,
    val lossFunc: (predictions: DTensor, labels: DTensor) -> DScalar,
    val optimizer: Optimizer<T> = AdamOptimizer(),
    val useJit: Boolean = false,
) {
    var totalTime = 0L

    /**
     * Trains the given model on the data set, for [epochs] epochs processing the data of the [dataIterator] in
     * batches of size [batchSize], but with a maximum number of total batches processed [maxIters].  Returns
     * the trained model.
     */
    fun train(
        model: T,
        epochs: Int,
        printProgress: Boolean = false,
        maxIters: Int? = null,
        printProgressFrequently: Boolean = false,
        device: Device = Device.CPU
    ): T {
        var totalIters = 0

        // The model training step function, which could possibly be optimized.
        fun modelTrainStep(model2: T, batch: Data): Pair<DScalar, T> {
            val (loss, tangent) = primalAndReverseDerivative(
                x = model2,
                f = { model3: T ->
                    val output = model3.predict(batch.features)
                    val loss = lossFunc(output, batch.labels)
                    loss
                },
                extractDerivative = { model3: T,
                                      loss: DScalar,
                                      extractor: (input: DTensor, output: DTensor) -> DTensor ->
                    model3.extractTangent(loss, extractor)
                }
            )

            val trainedModel: T = optimizer.train(model2, tangent)
            return Pair(loss, trainedModel)
        }
        fun trainingFunction(p: Pair<T, Data>): Pair<DScalar, T> = modelTrainStep(p.first, p.second)
        val jittedTrainingFunction = if (useJit) jit(::trainingFunction) else ::trainingFunction

        val optimizedModel = (0 until epochs).fold(model) { model1: T, e: Int ->
            var lossTotal: DScalar = FloatScalar.ZERO
            val trainedModel = batchedData.fold(model1) { model2: T, batch: Data ->
                val batchOnDevice = batch.to(device)
                val t1 = System.nanoTime()
                val (loss, trainedModel) = jittedTrainingFunction(Pair(model2, batchOnDevice))
                val t2 = System.nanoTime()
                totalTime += t2 - t1
                if (printProgress) lossTotal += loss
                totalIters++
                if (printProgressFrequently) println("Iter $totalIters Batch Loss: $loss")
                if (maxIters != null && totalIters >= maxIters) return trainedModel
                trainedModel
            }
            if (printProgress) println("Epoch $e Cumulative Loss: $lossTotal")
            trainedModel
        }
        return optimizedModel
    }

    private fun e(n: Long) = n / 1e9f

    fun dumpTimes() {
        println("running time:  ${e(totalTime)} sec")
    }
}
