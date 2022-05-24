/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.mnist

import org.diffkt.*
import org.diffkt.model.*
import org.diffkt.data.loaders.mnist.*

// returns the features and labels
fun loadTrainingData(): Pair<DTensor, DTensor> {
    val loader = MNISTDataLoader()
    val examples = loader.getTrainExamples()
    val features = examples.features
    val labels = examples.labels
    return Pair(features, labels)
}

fun predict(features: DTensor, model: MnistModel): DTensor {
    return model.predict(features)
}

fun loss(features: DTensor, labels: DTensor, weights: MnistModel): DScalar {
    val prediction = predict(features, weights)
    return crossEntropyLossFromOneHot(prediction, labels)
}

fun learn(features: DTensor, labels: DTensor, rate: Optimizer<MnistModel>, weights: MnistModel) : Pair<DScalar, MnistModel> {
    val pad = primalAndReverseDerivative(
        x = weights,
        f = { model: MnistModel -> loss(features, labels, model) },
        extractDerivative = { model: MnistModel,
                              loss: DScalar,
                              extractor: (input: DTensor, output: DTensor) -> DTensor ->
            model.extractTangent(loss, extractor)
        })

    val loss: DScalar = pad.first
    val tangent: Trainable.Tangent = pad.second
    return Pair(loss, rate.train(weights, tangent))
}
